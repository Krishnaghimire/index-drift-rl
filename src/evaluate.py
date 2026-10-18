"""Score index recommendations against the benchmark, with HypoPG.

Every answer - from a rule, an untrained LLM or a trained one - goes through
the same checks and the same cost model the labels were built with:

1. Is it a single, parseable ``CREATE INDEX`` (B-tree) on a table the query uses?
2. Is it production-safe (``CONCURRENTLY``)?
3. How much does it reduce the planner's cost, compared with the label?

This is also the core of the Phase 4 RL reward.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

from pglast import ast, parse_sql
from pglast.stream import RawStream

from .build_dataset import TIE_TOLERANCE
from .prompts import parse_answer


@dataclass
class IndexCheck:
    ok: bool
    reason: str | None = None
    table: str | None = None
    concurrently: bool = False
    unique: bool = False
    canonical: str | None = None  # unnamed, non-concurrent DDL (what HypoPG is given)


def check_index(ddl: str, allowed_tables) -> IndexCheck:
    try:
        stmts = parse_sql(ddl)
    except Exception as e:  # pglast raises ParseError, but be defensive about odd input
        return IndexCheck(False, f"parse error: {e}")
    if len(stmts) != 1 or not isinstance(stmts[0].stmt, ast.IndexStmt):
        return IndexCheck(False, "not a single CREATE INDEX statement")
    stmt = stmts[0].stmt
    table = stmt.relation.relname
    if table not in allowed_tables:
        return IndexCheck(False, f"table {table!r} is not used by the query", table=table)
    if (stmt.accessMethod or "btree") != "btree":
        return IndexCheck(False, f"access method {stmt.accessMethod!r} is not B-tree", table=table)
    concurrently, unique = bool(stmt.concurrent), bool(stmt.unique)
    stmt.idxname, stmt.concurrent, stmt.if_not_exists = None, False, False
    return IndexCheck(True, None, table, concurrently, unique, RawStream()(stmt))


def canonical(ddl: str) -> str | None:
    """Normalised form for comparing two index definitions (ignores name, CONCURRENTLY, spacing)."""
    check = check_index(ddl, allowed_tables=_ANY)
    return check.canonical if check.ok else None


class _AnyTable:
    def __contains__(self, item):
        return True


_ANY = _AnyTable()


def score_single(example: dict, answer: str, hypo) -> dict:
    """Score one model answer for a ``single`` example."""
    kind, ddl = parse_answer(answer)
    label = example["label"]
    sql, params = example["sql"], example["params"]
    baseline = hypo.cost(sql, params)
    label_reduction = 0.0
    if label:
        # Re-cost the label now rather than trusting the stored number: table
        # statistics change whenever ANALYZE runs, and both sides must be
        # judged against the same statistics.
        label_cost = hypo.cost(sql, params, [canonical(label["ddl"])])
        label_reduction = round(1 - label_cost / baseline, 4) if label_cost is not None else label["reduction"]
    result = {"id": example["id"], "template": example["template"], "answer_kind": kind, "ddl": ddl,
              "valid": kind != "invalid", "reason": None if kind != "invalid" else "no CREATE INDEX or NO INDEX found",
              "concurrently": None, "reduction": 0.0, "label_reduction": label_reduction,
              "size_bytes": None, "label_size_bytes": label["size_bytes"] if label else None,
              "exact_match": False}

    if kind == "none":
        result["exact_match"] = label is None
    elif kind == "index":
        check = check_index(ddl, set(example["tables"]))
        result["concurrently"] = check.concurrently
        if not check.ok:
            result.update(valid=False, reason=check.reason)
        else:
            cost = hypo.cost(sql, params, [check.canonical])
            if cost is None:
                result.update(valid=False, reason="rejected by PostgreSQL (unknown column or invalid expression)")
            else:
                result["reduction"] = round(1 - cost / baseline, 4)
                result["size_bytes"] = hypo.size(check.canonical)
                result["exact_match"] = label is not None and canonical(label["ddl"]) == check.canonical

    if label is None:
        result["solved"] = kind == "none"
    else:
        result["solved"] = result["valid"] and result["reduction"] >= label_reduction - TIE_TOLERANCE
    # Share of the label's improvement that the answer captured, in [0, 1].
    result["capture"] = (min(max(result["reduction"], 0.0) / label_reduction, 1.0)
                         if label_reduction > 0 else float(result["solved"]))
    return result


def summarize(results: list[dict]) -> dict:
    n = len(results)
    index_answers = [r for r in results if r["answer_kind"] == "index"]
    size_ratios = [r["size_bytes"] / r["label_size_bytes"] for r in results
                   if r["solved"] and r["size_bytes"] and r["label_size_bytes"]]
    return {
        "n": n,
        "valid": round(sum(r["valid"] for r in results) / n, 4),
        "concurrently": round(sum(bool(r["concurrently"]) for r in index_answers) / max(1, len(index_answers)), 4),
        "solved": round(sum(r["solved"] for r in results) / n, 4),
        "capture": round(statistics.mean(r["capture"] for r in results), 4),
        "exact_match": round(sum(r["exact_match"] for r in results) / n, 4),
        "mean_reduction": round(statistics.mean(r["reduction"] for r in results), 4),
        "median_size_vs_label": round(statistics.median(size_ratios), 2) if size_ratios else None,
        "by_template": {
            t: round(statistics.mean(r["capture"] for r in results if r["template"] == t), 3)
            for t in sorted({r["template"] for r in results})
        },
    }
