"""Enumerate plausible B-tree index candidates for a query.

Classic index-design rules, applied mechanically to QueryFeatures:
equality columns first, then one range or sort column; optional partial
predicate (``WHERE status = 'pending'``) and covering ``INCLUDE`` columns.
The candidates are scored later with HypoPG (src/hypo.py); this module only
proposes them.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .query_features import Key, QueryFeatures

MAX_INCLUDE_COLUMNS = 3


@dataclass(frozen=True)
class Candidate:
    table: str
    keys: tuple[str, ...]  # rendered key items: "customer_id", "created_at DESC", "(lower(email))"
    include: tuple[str, ...] = ()
    where: str | None = None

    @property
    def name(self) -> str:
        """Deterministic index name: readable prefix + hash of the full definition."""
        parts = [self.table] + [re.sub(r"\W+", "_", k.replace(" DESC", " desc")).strip("_") for k in self.keys]
        digest = hashlib.sha1(self.ddl().encode()).hexdigest()[:6]
        return f"{'_'.join(parts)[:48]}_{digest}"

    def ddl(self, concurrently: bool = False, named: bool = False) -> str:
        """CREATE INDEX statement. Unnamed by default (what HypoPG is given);
        ``concurrently=True, named=True`` is the production-safe form used in labels."""
        target = f"{self.name} " if named else ""
        sql = f"CREATE INDEX {'CONCURRENTLY ' if concurrently else ''}{target}ON {self.table} ({', '.join(self.keys)})"
        if self.include:
            sql += f" INCLUDE ({', '.join(self.include)})"
        if self.where:
            sql += f" WHERE {self.where}"
        return sql


def _render(key: Key, descending: bool = False) -> str:
    item = key.expr if key.is_column else f"({key.expr})"
    return item + (" DESC" if descending else "")


def _unique(items):
    return list(dict.fromkeys(items))


def generate(features: QueryFeatures, primary_keys: dict[str, tuple[str, ...]] | None = None) -> list[Candidate]:
    """All candidates for the query, deduplicated, in a stable order."""
    primary_keys = primary_keys or {}
    out: list[Candidate] = []

    for table in features.table_names():
        preds = [p for p in features.predicates if p.key.table == table]
        eq = _unique(p.key for p in preds if p.kind == "eq" and p.literal is None)
        rng = _unique(p.key for p in preds if p.kind == "range")
        literal_preds = _unique(p for p in preds if p.literal is not None)
        sort = [(k, d) for k, d in features.order_by if k.table == table]
        group = [k for k in features.group_by if k.table == table]
        join = _unique(k for a, b in features.joins for k in (a, b) if k.table == table)

        eq_items = [_render(k) for k in eq]
        structures: list[list[str]] = []
        # Single-key indexes.
        for k in _unique(eq + rng + [k for k, _ in sort] + group + join + [p.key for p in literal_preds]):
            structures.append([_render(k)])
        # Composites: equality columns first, then one range / sort / group column.
        if len(eq_items) > 1:
            structures += [eq_items, eq_items[::-1]]
        for k in rng:
            structures.append(eq_items + [_render(k)])
        if sort:
            structures.append(eq_items + [_render(k, d) for k, d in sort])
        if group:
            structures.append(eq_items + [_render(k) for k in group])

        structures = [s for s in _unique(tuple(s) for s in structures)
                      if s and s != tuple(primary_keys.get(table, ()))]
        for keys in structures:
            out.append(Candidate(table, keys))

        # Partial indexes: move a literal predicate into WHERE.
        for pred in literal_preds:
            rendered = _render(pred.key)
            for keys in structures:
                if rendered not in keys and len(keys) <= len(eq_items) + 1:
                    out.append(Candidate(table, keys, where=pred.sql()))

        # Covering indexes: add the remaining referenced columns so the
        # query can be answered by an index-only scan.
        if all(p.key.is_column for p in preds):
            referenced = set(features.outputs.get(table, ())) | {
                k.expr for k in eq + rng + [k for k, _ in sort] + group + join if k.is_column}
            for keys in structures:
                if len(keys) < 2 and len(structures) > 1:
                    continue
                bare = {item.removesuffix(" DESC") for item in keys}
                include = tuple(sorted(referenced - bare))
                if 0 < len(include) <= MAX_INCLUDE_COLUMNS:
                    out.append(Candidate(table, keys, include=include))

    return _unique(out)
