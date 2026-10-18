"""Extract index-relevant features from a SQL query.

Parses the query with pglast (PostgreSQL's own parser) and reports, per
table, which columns or expressions are filtered by equality or range, which
ones are sorted, grouped or joined on, and which predicates compare against
a literal (candidates for partial indexes).

Used by the candidate generator (src/candidates.py), and later by the
rule-based baseline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from pglast import ast, parse_sql
from pglast.enums import A_Expr_Kind, BoolExprType, SortByDir
from pglast.stream import RawStream
from pglast.visitors import Skip, Visitor

PLACEHOLDER_RE = re.compile(r"%\((\w+)\)s")

EQ_OPS = {"="}
RANGE_OPS = {"<", "<=", ">", ">="}
NEQ_OPS = {"<>", "!="}
VECTOR_OPS = {"<->", "<#>", "<=>", "<+>"}


@dataclass(frozen=True)
class Key:
    """An indexable key: a plain column or an expression over one table."""

    table: str
    expr: str  # column name, or deparsed expression such as "lower(email)"
    is_column: bool


@dataclass(frozen=True)
class Predicate:
    key: Key
    op: str
    kind: str  # "eq" | "range" | "neq"
    literal: str | None  # deparsed constant when compared with a literal, else None (parameter)

    def sql(self) -> str:
        """The predicate as SQL (only meaningful when ``literal`` is set)."""
        return f"{self.key.expr} {self.op} {self.literal}"


@dataclass
class QueryFeatures:
    tables: dict[str, str] = field(default_factory=dict)  # alias -> table
    predicates: list[Predicate] = field(default_factory=list)
    joins: list[tuple[Key, Key]] = field(default_factory=list)
    order_by: list[tuple[Key, bool]] = field(default_factory=list)  # (key, descending)
    group_by: list[Key] = field(default_factory=list)
    outputs: dict[str, set[str]] = field(default_factory=dict)  # table -> referenced columns
    vector_sort: bool = False  # ORDER BY embedding <-> ... (needs an ANN index, not B-tree)

    def table_names(self) -> list[str]:
        return list(dict.fromkeys(self.tables.values()))


def to_positional(sql: str) -> str:
    """Replace psycopg ``%(name)s`` placeholders with ``$n`` so pglast can parse them."""
    numbers: dict[str, int] = {}
    return PLACEHOLDER_RE.sub(lambda m: f"${numbers.setdefault(m.group(1), len(numbers) + 1)}", sql)


class _Refs(Visitor):
    """Collect ColumnRefs and ParamRefs, without descending into subqueries."""

    def __init__(self):
        super().__init__()
        self.columns: list[tuple[str, ...]] = []
        self.params = 0
        self.subqueries = 0

    def visit_ColumnRef(self, ancestors, node):
        self.columns.append(tuple(f.sval for f in node.fields if isinstance(f, ast.String)))

    def visit_ParamRef(self, ancestors, node):
        self.params += 1

    def visit_SubLink(self, ancestors, node):
        self.subqueries += 1
        return Skip


def _refs(node) -> _Refs:
    v = _Refs()
    v(node)
    return v


class _Extractor:
    def __init__(self, columns_by_table: dict[str, set[str]] | None):
        self.columns_by_table = columns_by_table or {}
        self.f = QueryFeatures()

    # -- resolution -----------------------------------------------------

    def resolve(self, ref: tuple[str, ...]) -> tuple[str, str] | None:
        """(table, column) for a ColumnRef, or None if it isn't a base-table column."""
        if len(ref) == 2:
            table = self.f.tables.get(ref[0])
            return (table, ref[1]) if table else None
        if len(ref) != 1:
            return None
        column = ref[0]
        tables = self.f.table_names()
        if len(tables) == 1:
            known = self.columns_by_table.get(tables[0])
            return (tables[0], column) if known is None or column in known else None
        owners = [t for t in tables if column in self.columns_by_table.get(t, ())]
        return (owners[0], column) if len(owners) == 1 else None

    def key(self, node) -> Key | None:
        """Key for an expression whose columns all come from one table."""
        refs = _refs(node)
        if refs.params or refs.subqueries or not refs.columns:
            return None
        resolved = [self.resolve(r) for r in refs.columns]
        if None in resolved or len({t for t, _ in resolved}) != 1:
            return None
        table = resolved[0][0]
        if isinstance(node, ast.ColumnRef):
            return Key(table, resolved[0][1], True)
        text = RawStream()(node)
        for alias in self.f.tables:
            text = re.sub(rf"\b{re.escape(alias)}\.", "", text)
        return Key(table, text, False)

    # -- clauses ----------------------------------------------------------

    def add_from(self, item) -> list:
        """Register tables; return join conditions found in JOIN ... ON."""
        if isinstance(item, ast.RangeVar):
            alias = item.alias.aliasname if item.alias else item.relname
            self.f.tables[alias] = item.relname
            return []
        if isinstance(item, ast.JoinExpr):
            quals = self.add_from(item.larg) + self.add_from(item.rarg)
            return quals + ([item.quals] if item.quals is not None else [])
        return []

    def add_condition(self, node) -> None:
        if isinstance(node, ast.BoolExpr):
            if node.boolop == BoolExprType.AND_EXPR:
                for arg in node.args:
                    self.add_condition(arg)
            return  # OR / NOT: not handled by simple candidates
        if not isinstance(node, ast.A_Expr):
            return
        op = node.name[-1].sval if node.name else ""
        if node.kind == A_Expr_Kind.AEXPR_BETWEEN:
            key = self.key(node.lexpr)
            if key:
                self.f.predicates.append(Predicate(key, "BETWEEN", "range", None))
            return
        if node.kind != A_Expr_Kind.AEXPR_OP:
            return

        left, right = self.key(node.lexpr), self.key(node.rexpr)
        if left and right:
            if op in EQ_OPS and left.table != right.table:
                self.f.joins.append((left, right))
            return
        key, other = (left, node.rexpr) if left else (right, node.lexpr)
        if key is None:
            return
        if left is None:  # constant on the left: flip the operator
            op = {"<": ">", ">": "<", "<=": ">=", ">=": "<="}.get(op, op)
        kind = "eq" if op in EQ_OPS else "range" if op in RANGE_OPS else "neq" if op in NEQ_OPS else None
        if kind is None:
            return
        other_refs = _refs(other)
        if other_refs.columns or other_refs.subqueries:
            return
        literal = RawStream()(other) if not other_refs.params else None
        self.f.predicates.append(Predicate(key, op, kind, literal))

    def run(self, stmt: ast.SelectStmt) -> QueryFeatures:
        join_quals = []
        for item in stmt.fromClause or ():
            join_quals += self.add_from(item)
        for qual in join_quals:
            self.add_condition(qual)
        if stmt.whereClause is not None:
            self.add_condition(stmt.whereClause)

        targets = [t.val for t in stmt.targetList or ()]
        for target in targets:
            for ref in _refs(target).columns:
                resolved = self.resolve(ref)
                if resolved:
                    self.f.outputs.setdefault(resolved[0], set()).add(resolved[1])

        def positional(node):
            # GROUP BY 1 / ORDER BY 1 refer to the select list.
            if isinstance(node, ast.A_Const) and isinstance(node.val, ast.Integer):
                i = node.val.ival - 1
                return targets[i] if 0 <= i < len(targets) else None
            return node

        for node in stmt.groupClause or ():
            key = self.key(positional(node)) if positional(node) is not None else None
            if key:
                self.f.group_by.append(key)
        for sort in stmt.sortClause or ():
            node = positional(sort.node)
            if isinstance(node, ast.A_Expr) and node.name and node.name[-1].sval in VECTOR_OPS:
                self.f.vector_sort = True
                continue
            key = self.key(node) if node is not None else None
            if key:
                self.f.order_by.append((key, sort.sortby_dir == SortByDir.SORTBY_DESC))
        return self.f


def extract(sql: str, columns_by_table: dict[str, set[str]] | None = None) -> QueryFeatures:
    """Features of a single SELECT. ``columns_by_table`` resolves unqualified
    columns in multi-table queries; without it they are skipped."""
    stmts = parse_sql(to_positional(sql))
    if len(stmts) != 1 or not isinstance(stmts[0].stmt, ast.SelectStmt):
        raise ValueError("expected exactly one SELECT statement")
    return _Extractor(columns_by_table).run(stmts[0].stmt)
