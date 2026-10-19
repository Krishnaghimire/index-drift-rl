"""Reward functions for RL training (GRPO) of the index advisor.

    env = IndexEnvironment()                       # connects to the sandbox Postgres
    env.single(completion_text, example)           # -> Reward(value in [-1, 1], components)
    env.window(completion_text, example)
    reward_fn = make_trl_reward(env, "single")     # plug into trl.GRPOTrainer(reward_funcs=[...])

Design (each rule closes a way the model could "game" a naive reward):

* Hard failures first: unparseable output or anything but CREATE INDEX -> -1;
  missing CONCURRENTLY or a UNIQUE index (HypoPG would happily simulate a
  unique index that could never be built on real data) -> -0.5; an index
  PostgreSQL rejects (unknown column, non-immutable expression) -> -0.75.
* Quality = share of the label's cost reduction captured, averaged over the
  example's own parameters AND ``alt_params`` freshly sampled parameter sets
  of the same query. An index tailored to the literal values in the prompt
  (``WHERE tenant_id = 1``) earns little.
* Size penalty per doubling over the label's size: indexing every column is
  not free.
* Optional latency table (``python -m src.environment calibrate``): real
  measurements of common answers. Indexes that made real queries slower
  (the planner's blind spot for the dominant tenant) are penalised.
* Small penalty for chatter around the SQL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from pglast import ast, parse_sql

from . import config, db, workload
from .dataset import DatasetSpec
from .evaluate import canonical, check_index
from .hypo import HypoSession
from .prompts import WINDOW_BUDGET, parse_answer

DESTRUCTIVE_RE = re.compile(r"\b(DROP\s+(TABLE|SCHEMA|DATABASE)|TRUNCATE|DELETE\s+FROM|UPDATE\s+\w+\s+SET|ALTER\s+TABLE)\b",
                            re.IGNORECASE)
STATEMENT_RE = re.compile(r"\b(?:CREATE\s+(?:UNIQUE\s+)?INDEX|DROP\s+INDEX)\b[^;]*", re.IGNORECASE | re.DOTALL)
NO_CHANGES_RE = re.compile(r"\bNO\s+CHANGES\b", re.IGNORECASE)


@dataclass
class RewardWeights:
    invalid: float = -1.0
    rejected: float = -0.75
    unsafe: float = -0.5
    right_none: float = 1.0
    wrong_none: float = 0.0
    needless_index: float = 0.0  # index answer when nothing helps
    size_per_doubling: float = 0.1
    max_size_penalty: float = 0.3
    regression: float = 0.5  # x share of queries that got slower for real
    verbosity: float = 0.1
    over_budget: float = -0.5
    bad_drop: float = 0.1  # per DROP of an index that does not exist
    alt_params: int = 3
    min_label_gain: float = 0.05  # parameter sets where the label barely helps are not informative


@dataclass
class Reward:
    value: float
    components: dict = field(default_factory=dict)


def _completion_text(completion) -> str:
    """TRL passes plain strings or chat messages ([{"role": "assistant", "content": ...}])."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[-1], dict):
        return completion[-1].get("content", "")
    return str(completion)


def _clip(x: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, x))


class IndexEnvironment:
    def __init__(self, dsn: str | None = None, latency_table: Path | dict | None = None,
                 weights: RewardWeights | None = None):
        self.conn = db.connect(dsn or config.pg_dsn(), prepare_threshold=None)
        meta = db.dataset_meta(self.conn)
        if meta is None:
            raise RuntimeError("Database is not seeded. Run: python -m src.seed load")
        self.spec = DatasetSpec.from_dict(meta)
        self.hypo = HypoSession(self.conn)
        self.tables = set(db.DRIFT_TABLES)
        self.w = weights or RewardWeights()
        if isinstance(latency_table, (str, Path)):
            latency_table = json.loads(Path(latency_table).read_text(encoding="utf-8"))
        self.latency = {(r["template"], r["canonical"]): r for r in (latency_table or {}).get("entries", [])}
        self._sizes: dict[str, int | None] = {}

    def close(self) -> None:
        self.conn.close()

    def size(self, ddl: str) -> int | None:
        if ddl not in self._sizes:
            self._sizes[ddl] = self.hypo.size(ddl)
        return self._sizes[ddl]

    # -- single ----------------------------------------------------------------

    def param_sets(self, example: dict) -> list[dict]:
        """The example's own parameters plus ``alt_params`` deterministic resamples."""
        params = {k: v for k, v in example["params"].items() if v is not None}  # HF structs pad missing keys
        template = workload.TEMPLATES.get(example["template"])
        if template is None or self.w.alt_params <= 0:
            return [params]
        seed = int(hashlib.sha1(example["id"].encode()).hexdigest()[:8], 16)
        sampler = workload.ParamSampler(self.spec, random.Random(seed))
        return [params] + [template.sample(sampler) for _ in range(self.w.alt_params)]

    def single(self, completion, example: dict) -> Reward:
        text = _completion_text(completion)
        if DESTRUCTIVE_RE.search(text):
            return Reward(self.w.invalid, {"reason": "destructive statement"})
        kind, ddl = parse_answer(text)
        label = example.get("label")
        if kind == "invalid":
            return Reward(self.w.invalid, {"reason": "no CREATE INDEX or NO INDEX"})
        if kind == "none":
            value = self.w.right_none if label is None else self.w.wrong_none
            return Reward(value, {"reason": "no index"})
        if len(STATEMENT_RE.findall(text)) > 1:
            return Reward(self.w.invalid, {"reason": "more than one statement"})

        check = check_index(ddl, set(example["tables"]))
        if not check.ok:
            value = self.w.invalid if "parse" in check.reason or "single" in check.reason else self.w.rejected
            return Reward(value, {"reason": check.reason})
        if not check.concurrently or check.unique:
            return Reward(self.w.unsafe, {"reason": "UNIQUE" if check.unique else "missing CONCURRENTLY"})
        if label is None:
            return Reward(self.w.needless_index, {"reason": "no index helps this query"})

        label_ddl = canonical(label["ddl"])
        sql = example["sql"]
        captures, reductions = [], []
        for params in self.param_sets(example):
            base = self.hypo.cost(sql, params)
            answer_cost = self.hypo.cost(sql, params, [check.canonical])
            label_cost = self.hypo.cost(sql, params, [label_ddl])
            if answer_cost is None:
                return Reward(self.w.rejected, {"reason": "rejected by PostgreSQL"})
            answer_gain = 1 - answer_cost / base
            label_gain = 1 - label_cost / base if label_cost is not None else 0.0
            reductions.append(answer_gain)
            if label_gain >= self.w.min_label_gain:
                captures.append(_clip(answer_gain / label_gain, 0.0, 1.0))
        quality = statistics.mean(captures) if captures else 1.0

        size_penalty = 0.0
        size, label_size = self.size(check.canonical), self.size(label_ddl)
        if size and label_size and size > label_size:
            size_penalty = min(self.w.max_size_penalty, self.w.size_per_doubling * math.log2(size / label_size))

        measured = self.latency.get((example["template"], check.canonical))
        regression = self.w.regression * measured["slower_share"] if measured else 0.0
        verbose = len(text.strip()) > len(ddl) + 40
        verbosity = self.w.verbosity if verbose else 0.0

        value = _clip(quality - size_penalty - regression - verbosity)
        return Reward(round(value, 4), {"quality": round(quality, 4), "size_penalty": round(size_penalty, 4),
                                        "regression": round(regression, 4), "verbosity": verbosity,
                                        "reductions": [round(r, 4) for r in reductions]})

    # -- windows -----------------------------------------------------------------

    def window_objective(self, example: dict, ddls) -> float:
        """Workload-weighted planner cost with ``ddls`` present (simulated templates only)."""
        total = 0.0
        for w in example["workload"]:
            if not w["simulated"]:
                continue
            template = workload.TEMPLATES[w["template"]]
            tables = {t for t in self.tables if re.search(rf"\b{t}\b", template.sql)}
            relevant = sorted(d for d in ddls if _table_of(d) in tables)
            cost = self.hypo.cost(w["example_query"], {}, relevant)
            if cost is None:
                raise ValueError(f"could not cost {w['template']} with {relevant}")
            total += w["count"] * cost
        return total

    def window(self, completion, example: dict, budget: int = WINDOW_BUDGET) -> Reward:
        text = _completion_text(completion)
        if DESTRUCTIVE_RE.search(text):
            return Reward(self.w.invalid, {"reason": "destructive statement"})

        existing = {}  # index name -> canonical DDL
        for d in example["existing_indexes"]:
            existing[_index_name(d)] = canonical(d)
        statements = STATEMENT_RE.findall(text)
        if not statements and not NO_CHANGES_RE.search(text):
            return Reward(self.w.invalid, {"reason": "no statements and no NO CHANGES"})

        final = dict(existing)
        penalty, unsafe = 0.0, False
        for statement in statements:
            try:
                stmt = parse_sql(statement)[0].stmt
            except Exception:
                return Reward(self.w.invalid, {"reason": f"parse error: {statement[:60]}"})
            if isinstance(stmt, ast.DropStmt):
                unsafe |= not stmt.concurrent
                for obj in stmt.objects or ():
                    name = obj[-1].sval if isinstance(obj, (list, tuple)) else getattr(obj, "sval", str(obj))
                    if final.pop(name, None) is None:
                        penalty += self.w.bad_drop
            elif isinstance(stmt, ast.IndexStmt):
                check = check_index(statement, self.tables)
                if not check.ok:
                    return Reward(self.w.rejected, {"reason": check.reason})
                unsafe |= not check.concurrently or check.unique
                final[stmt.idxname or f"unnamed_{len(final)}"] = check.canonical
        if unsafe:
            return Reward(self.w.unsafe, {"reason": "missing CONCURRENTLY or UNIQUE"})
        if len(set(final.values())) > budget:
            return Reward(self.w.over_budget, {"reason": f"more than {budget} indexes"})

        # Validate first: an index PostgreSQL rejects must never look "free".
        if any(self.size(d) is None for d in set(final.values())):
            return Reward(self.w.rejected, {"reason": "rejected by PostgreSQL"})
        label = [canonical(d) for d in example["label"]["indexes"]]
        before = self.window_objective(example, existing.values())
        after = self.window_objective(example, set(final.values()))
        target = self.window_objective(example, label)
        label_gain = before - target
        gain = before - after
        if label_gain > self.w.min_label_gain * before:
            quality = _clip(gain / label_gain)
        else:  # the existing set is already (nearly) right: reward not making it worse
            quality = 1.0 if gain >= -self.w.min_label_gain * before else _clip(gain / before)
        value = _clip(quality - penalty)
        return Reward(round(value, 4), {"quality": round(quality, 4), "bad_drops": round(penalty, 4),
                                        "objective": {"before": round(before, 1), "after": round(after, 1),
                                                      "label": round(target, 1)},
                                        "final_indexes": sorted(set(final.values()))})


def _index_name(ddl: str) -> str:
    return parse_sql(ddl)[0].stmt.idxname


def _table_of(canonical_ddl: str) -> str:
    return parse_sql(canonical_ddl)[0].stmt.relation.relname


def make_trl_reward(env: IndexEnvironment, task: str = "single"):
    """Adapter for ``trl.GRPOTrainer(reward_funcs=[...])``.

    TRL calls ``reward(prompts=..., completions=..., **dataset_columns)`` with
    one list entry per completion; the dataset columns rebuild each example.
    """
    score = env.single if task == "single" else env.window

    def reward(prompts=None, completions=None, **columns) -> list[float]:
        out = []
        for i, completion in enumerate(completions):
            example = {k: v[i] for k, v in columns.items() if isinstance(v, list) and len(v) == len(completions)}
            out.append(score(completion, example).value)
        return out

    reward.__name__ = f"index_reward_{task}"
    return reward


# -- latency calibration -----------------------------------------------------------

def calibrate(dsn: str, data_dir: Path, per_template: int, queries: int, out: Path) -> dict:
    """Build common answers for real and record how many real queries got slower."""
    examples = [json.loads(line) for line in (data_dir / "single" / "train.jsonl").open(encoding="utf-8")]
    examples += [json.loads(line) for line in (data_dir / "single" / "test_unseen_templates.jsonl").open(encoding="utf-8")]
    by_template = defaultdict(list)
    for ex in examples:
        by_template[ex["template"]].append(ex)

    conn = db.connect(dsn, prepare_threshold=None)
    if db.secondary_indexes(conn):
        raise SystemExit("Secondary indexes exist. Run: python -m src.seed drop-indexes")

    def median_ms(sql, params) -> float:
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            conn.execute(sql, params).fetchall()
            times.append((time.perf_counter() - t0) * 1000)
        return statistics.median(times)

    entries = []
    for template, exs in sorted(by_template.items()):
        common = Counter(canonical(c["ddl"]) for ex in exs for c in ex["candidates"][:3])
        sample = random.Random(template).sample(exs, min(queries, len(exs)))
        before = [median_ms(e["sql"], e["params"]) for e in sample]
        for ddl, _ in common.most_common(per_template):
            table = _table_of(ddl)
            conn.execute(ddl.replace("CREATE INDEX ON", "CREATE INDEX drift_calibrate_idx ON", 1))
            try:
                conn.execute(f"ANALYZE {table}")
                after = [median_ms(e["sql"], e["params"]) for e in sample]
            finally:
                conn.execute("DROP INDEX IF EXISTS drift_calibrate_idx")
                conn.execute(f"ANALYZE {table}")
            speedups = [b / max(a, 1e-6) for b, a in zip(before, after)]
            entry = {"template": template, "canonical": ddl, "n": len(sample),
                     "median_speedup": round(statistics.median(speedups), 2),
                     "slower_share": round(sum(s < 0.9 for s in speedups) / len(speedups), 2)}
            entries.append(entry)
            print(f"  {template:<24} x{entry['median_speedup']:<7} slower {entry['slower_share']:.0%}  {ddl}",
                  file=sys.stderr)
    conn.close()
    table = {"note": "Real latency of common answers vs no index; slower = >10% slower than before.",
             "entries": entries}
    out.write_text(json.dumps(table, indent=2), encoding="utf-8")
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.environment", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_cal = sub.add_parser("calibrate", help="measure real latency of common answers (builds real indexes)")
    p_cal.add_argument("--data", type=Path, default=Path("data/dataset"))
    p_cal.add_argument("--per-template", type=int, default=5)
    p_cal.add_argument("--queries", type=int, default=12)
    p_cal.add_argument("--out", type=Path, default=Path("data/dataset/latency_table.json"))
    p_cal.add_argument("--dsn", default=config.pg_dsn())
    args = parser.parse_args(argv)
    if args.command == "calibrate":
        calibrate(args.dsn, args.data, args.per_template, args.queries, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
