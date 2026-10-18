"""Build the index-advisor benchmark dataset.

    python -m src.build_dataset --out data/dataset            # ~5 min at scale 1.0
    python -m src.build_dataset --out data/dataset --validate # + real-index latency check

Two tasks are produced from drifting workloads (src/workload.py):

* ``single``  - one slow query (+ schema + EXPLAIN plan) -> the best single
  index. Candidates come from src/candidates.py and are scored with HypoPG;
  the label is the smallest index within TIE_TOLERANCE of the best cost
  reduction, or null when no index helps by at least MIN_REDUCTION.
* ``windows`` - a window of the query log (template mix + examples) and the
  indexes left over from the previous window -> the index set to have now,
  under a budget, chosen greedily by workload-weighted cost. This is the
  drift task: some old indexes should be dropped, new ones created.

Splits: workloads from ``--train-seeds`` go to train, ``--test-seeds`` to
test. In ``single``, HOLDOUT_TEMPLATES never appear in train; they form a
separate ``test_unseen_templates`` split that measures generalisation.

Vector-search templates are excluded from scoring: HypoPG cannot simulate
HNSW indexes (they are covered by the Phase 6 benchmark instead).
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from . import candidates as cands
from . import config, db, query_features, workload
from .dataset import DatasetSpec
from .hypo import HypoSession

HOLDOUT_TEMPLATES = ("events_by_source", "low_stock_products")
MIN_REDUCTION = 0.2  # below this, the label is "no index"
TIE_TOLERANCE = 0.02  # candidates this close to the best count as ties; the smallest wins
TOP_CANDIDATES = 5  # candidates kept per example (for analysis / reward shaping)
POOL_PER_TEMPLATE = 4  # candidates per template considered by the window optimiser
MIN_WINDOW_GAIN = 0.01  # stop adding indexes when the gain is below 1% of the no-index cost
CHURN_TOLERANCE = 0.02  # keep an existing index if it is within 2% of the best new one


@dataclass
class Instance:
    uid: str
    split: str
    seed: int
    scenario: str
    record: dict  # workload query record

    @property
    def template(self) -> str:
        return self.record["template"]

    @property
    def query(self) -> tuple[str, dict]:
        return self.record["sql"], self.record["params"]


def _split_for(split: str, template: str) -> str | None:
    if template in HOLDOUT_TEMPLATES:
        return "test_unseen_templates" if split == "test" else None
    return split


class Builder:
    def __init__(self, conn, args):
        self.conn = conn
        self.args = args
        meta = db.dataset_meta(conn)
        if meta is None:
            raise SystemExit("Database is not seeded. Run: python -m src.seed load")
        if db.secondary_indexes(conn):
            raise SystemExit("Secondary indexes exist; labels assume the baseline. "
                             "Run: python -m src.seed drop-indexes")
        self.spec = DatasetSpec.from_dict(meta)
        self.hypo = HypoSession(conn)
        columns = db.columns_by_table(conn)
        pks = db.primary_keys(conn)
        self.schema = {t: db.table_ddl(conn, t) for t in db.DRIFT_TABLES}

        self.templates = {name: t for name, t in workload.TEMPLATES.items() if t.kind != "vector"}
        self.features = {name: query_features.extract(t.sql, columns) for name, t in self.templates.items()}
        self.candidates: dict[str, list[cands.Candidate]] = {
            name: cands.generate(f, pks) for name, f in self.features.items()}
        # Plain DDL (what HypoPG sees) <-> named, CONCURRENTLY form (what labels show).
        self.by_ddl = {c.ddl(): c for cs in self.candidates.values() for c in cs}
        self.by_label = {c.ddl(concurrently=True, named=True): c.ddl() for c in self.by_ddl.values()}
        self._sizes: dict[str, int | None] = {}

    def size(self, ddl: str) -> int | None:
        if ddl not in self._sizes:
            self._sizes[ddl] = self.hypo.size(ddl)
        return self._sizes[ddl]

    # -- workloads ---------------------------------------------------------

    def workloads(self) -> dict[tuple[str, int, str], list[dict]]:
        logs = {}
        for split, seeds in (("train", self.args.train_seeds), ("test", self.args.test_seeds)):
            for seed in seeds:
                for scenario in self.args.scenarios:
                    records = workload.generate(scenario, self.args.n_queries, self.spec, seed=seed)
                    logs[(split, seed, scenario)] = [r for r in records if r["kind"] == "query"]
        return logs

    # -- task: single ------------------------------------------------------

    def sample_instances(self, logs) -> list[Instance]:
        instances = []
        for (split, seed, scenario), records in logs.items():
            rng = random.Random(f"{split}-{seed}-{scenario}")
            groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
            for r in records:
                if r["template"] in self.templates:
                    groups[(r["phase"], r["template"])].append(r)
            for (_phase, template), rs in sorted(groups.items()):
                final_split = _split_for(split, template)
                if final_split is None:
                    continue
                picked = rs if len(rs) <= self.args.per_group else rng.sample(rs, self.args.per_group)
                for r in sorted(picked, key=lambda r: r["seq"]):
                    instances.append(Instance(f"{split}-s{seed}-{scenario}-{r['seq']}",
                                              final_split, seed, scenario, r))
        return instances

    def score_instances(self, instances: list[Instance]) -> dict[str, list[tuple[str, float | None]]]:
        """{uid: [(ddl, cost), ...]} plus baseline costs under the "" key."""
        by_template: dict[str, list[Instance]] = defaultdict(list)
        for inst in instances:
            by_template[inst.template].append(inst)
        scores: dict[str, list[tuple[str, float | None]]] = defaultdict(list)
        for template, group in by_template.items():
            queries = [i.query for i in group]
            for ddl in [""] + [c.ddl() for c in self.candidates[template]]:
                costs = self.hypo.costs(queries, [ddl] if ddl else [])
                for inst, cost in zip(group, costs):
                    scores[inst.uid].append((ddl, cost))
            print(f"  scored {template:<24} {len(group):>4} queries x "
                  f"{len(self.candidates[template])} candidates", file=sys.stderr)
        return scores

    def single_example(self, inst: Instance, scored: list[tuple[str, float | None]]) -> dict:
        baseline = scored[0][1]
        ranked = []
        for ddl, cost in scored[1:]:
            if cost is None:
                continue
            ranked.append({"ddl": self.by_ddl[ddl].ddl(concurrently=True, named=True), "cost": round(cost, 2),
                           "reduction": round(1 - cost / baseline, 4), "size_bytes": self.size(ddl)})
        ranked.sort(key=lambda c: (-c["reduction"], c["size_bytes"] or 0))
        label = None
        if ranked and ranked[0]["reduction"] >= MIN_REDUCTION:
            ties = [c for c in ranked if c["reduction"] >= ranked[0]["reduction"] - TIE_TOLERANCE]
            label = min(ties, key=lambda c: (c["size_bytes"] or 0, -c["reduction"]))
        sql, params = inst.query
        tables = self.features[inst.template].table_names()
        return {
            "id": inst.uid,
            "split": inst.split,
            "seed": inst.seed,
            "scenario": inst.scenario,
            "phase": inst.record["phase"],
            "template": inst.template,
            "seq": inst.record["seq"],
            "query": inst.record["query"],
            "sql": sql,
            "params": params,
            "tables": tables,
            "schema": [self.schema[t] for t in tables],
            "plan": self.hypo.plan_text(sql, params),
            "baseline_cost": round(baseline, 2),
            "candidates": ranked[:TOP_CANDIDATES],
            "label": label,
        }

    # -- task: windows -----------------------------------------------------

    def template_pools(self, examples: list[dict]) -> dict[str, list[str]]:
        """Top candidates per template by mean cost reduction (plain DDL)."""
        reductions: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for ex in examples:
            for c in ex["candidates"]:
                reductions[ex["template"]][self.by_label[c["ddl"]]].append(c["reduction"])
        return {t: sorted(r, key=lambda d: -statistics.mean(r[d]))[:POOL_PER_TEMPLATE]
                for t, r in reductions.items()}

    def window_examples(self, logs, pools) -> list[dict]:
        out = []
        for (split, seed, scenario), records in logs.items():
            rng = random.Random(f"windows-{split}-{seed}-{scenario}")
            existing: list[str] = []
            for w, start in enumerate(range(0, len(records), self.args.window)):
                chunk = records[start:start + self.args.window]
                ex = self.window_example(chunk, existing, pools, rng)
                ex.update(id=f"{split}-s{seed}-{scenario}-w{w}", split=split, seed=seed,
                          scenario=scenario, window=w)
                out.append(ex)
                existing = [self.by_label[d] for d in ex["label"]["indexes"]]
            print(f"  windows {split} seed={seed} {scenario}", file=sys.stderr)
        return out

    def window_example(self, chunk, existing, pools, rng) -> dict:
        counts = Counter(r["template"] for r in chunk)
        examples = {}
        for r in chunk:
            examples.setdefault(r["template"], r["query"])
        samples = {}
        for template in counts:
            if template in self.templates:
                rs = [r for r in chunk if r["template"] == template]
                picked = rs if len(rs) <= self.args.samples else rng.sample(rs, self.args.samples)
                samples[template] = [(r["sql"], r["params"]) for r in picked]

        tables = {t: set(self.features[t].table_names()) for t in samples}

        def objective(ddls) -> float:
            total = 0.0
            for template, queries in samples.items():
                relevant = [d for d in ddls if self.by_ddl[d].table in tables[template]]
                costs = self.hypo.costs(queries, relevant)
                total += counts[template] * statistics.mean(costs)
            return total

        pool = sorted({d for t in samples for d in pools.get(t, [])} | set(existing))
        none_cost = objective([])
        chosen: list[str] = []
        current = none_cost
        for _ in range(self.args.budget):
            gains = [(current - objective(chosen + [d]), d) for d in pool if d not in chosen]
            if not gains:
                break
            best = max(g for g, _ in gains)
            if best < MIN_WINDOW_GAIN * none_cost:
                break
            near = [d for g, d in gains if g >= best * (1 - CHURN_TOLERANCE)]
            pick = next((d for d in near if d in existing), None) or min(near, key=lambda d: self.size(d) or 0)
            chosen.append(pick)
            current = objective(chosen)

        existing_cost = objective(existing)
        concurrent = lambda ds: [self.by_ddl[d].ddl(concurrently=True, named=True) for d in ds]  # noqa: E731
        return {
            "start_seq": chunk[0]["seq"],
            "end_seq": chunk[-1]["seq"],
            "phases": dict(Counter(r["phase"] for r in chunk)),
            "workload": [{"template": t, "count": n, "share": round(n / len(chunk), 4),
                          "example_query": examples[t], "simulated": t in samples}
                         for t, n in counts.most_common()],
            "schema": list(self.schema.values()),
            "existing_indexes": concurrent(existing),
            "label": {
                "indexes": concurrent(chosen),
                "create": concurrent([d for d in chosen if d not in existing]),
                "drop": [f"DROP INDEX CONCURRENTLY {self.by_ddl[d].name}" for d in existing if d not in chosen],
            },
            "objective": {"no_indexes": round(none_cost, 2), "existing": round(existing_cost, 2),
                          "label": round(current, 2)},
        }

    # -- validation with real indexes -----------------------------------------

    def validate(self, examples: list[dict]) -> list[dict]:
        """Build each template's most common label for real and measure latency."""
        import psycopg

        results = []
        by_template: dict[str, list[dict]] = defaultdict(list)
        for ex in examples:
            if ex["split"] == "train" or ex["template"] in HOLDOUT_TEMPLATES:
                by_template[ex["template"]].append(ex)
        for template, exs in sorted(by_template.items()):
            labelled = [e for e in exs if e["label"]]
            if not labelled:
                continue
            ddl = Counter(e["label"]["ddl"] for e in labelled).most_common(1)[0][0]
            sample = [e for e in labelled if e["label"]["ddl"] == ddl][:10]

            def median_ms(ex) -> float:
                times = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    self.conn.execute(ex["sql"], ex["params"]).fetchall()
                    times.append((time.perf_counter() - t0) * 1000)
                return statistics.median(times)

            before = [median_ms(e) for e in sample]
            candidate = self.by_ddl[self.by_label[ddl]]
            table = candidate.table
            t0 = time.perf_counter()
            self.conn.execute(ddl)
            build_s = time.perf_counter() - t0
            try:
                self.conn.execute(f"ANALYZE {table}")  # expression indexes need fresh statistics
                after = [median_ms(e) for e in sample]
            finally:
                try:
                    self.conn.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {candidate.name}")
                except psycopg.Error:
                    pass
            est = statistics.mean(e["label"]["reduction"] for e in sample)
            # Per-query speedups: the same index can help small tenants and hurt the biggest one.
            speedups = [b / max(a, 1e-6) for b, a in zip(before, after)]
            row = {"template": template, "ddl": ddl, "n_queries": len(sample),
                   "estimated_cost_reduction": round(est, 4),
                   "before_ms": round(statistics.median(before), 3), "after_ms": round(statistics.median(after), 3),
                   "median_speedup": round(statistics.median(speedups), 1),
                   "min_speedup": round(min(speedups), 2),
                   "slower_share": round(sum(s < 1 for s in speedups) / len(speedups), 2),
                   "slower_tenants": sorted({e["params"]["tenant_id"] for e, s in zip(sample, speedups)
                                             if s < 1 and "tenant_id" in e["params"]}),
                   "build_s": round(build_s, 2)}
            results.append(row)
            print(f"  validate {template:<24} est -{est:.0%}  median x{row['median_speedup']}  "
                  f"min x{row['min_speedup']}  slower {row['slower_share']:.0%}", file=sys.stderr)
        return results


def _write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return len(rows)


def build(args) -> dict:
    out: Path = args.out
    t_start = time.perf_counter()
    with db.connect(args.dsn, prepare_threshold=None) as conn:
        b = Builder(conn, args)
        print("generating workloads", file=sys.stderr)
        logs = b.workloads()

        print("task single: scoring candidates with HypoPG", file=sys.stderr)
        instances = b.sample_instances(logs)
        scores = b.score_instances(instances)
        single = [b.single_example(i, scores[i.uid]) for i in instances]

        print("task windows: greedy index selection", file=sys.stderr)
        pools = b.template_pools([e for e in single if e["split"] == "train" or e["template"] in HOLDOUT_TEMPLATES])
        windows = b.window_examples(logs, pools)

        validation = []
        if args.validate:
            print("validating labels with real indexes", file=sys.stderr)
            validation = b.validate(single)

        counts = {}
        for split in ("train", "test", "test_unseen_templates"):
            counts[f"single/{split}"] = _write_jsonl(out / "single" / f"{split}.jsonl",
                                                     [e for e in single if e["split"] == split])
        for split in ("train", "test"):
            counts[f"windows/{split}"] = _write_jsonl(out / "windows" / f"{split}.jsonl",
                                                      [e for e in windows if e["split"] == split])
        (out / "schema.sql").write_text("\n".join(b.schema.values()) + "\n", encoding="utf-8")

        stats = {
            "dataset": b.spec.to_dict(),
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "dsn"},
            "files": counts,
            "single_label_rate": round(sum(e["label"] is not None for e in single) / max(1, len(single)), 4),
            "single_by_template": {
                t: {"examples": n,
                    "labelled": sum(1 for e in single if e["template"] == t and e["label"]),
                    "top_label": (Counter(e["label"]["ddl"] for e in single if e["template"] == t and e["label"])
                                  .most_common(1) or [[None]])[0][0]}
                for t, n in sorted(Counter(e["template"] for e in single).items())},
            "windows_mean_indexes": round(statistics.mean(len(w["label"]["indexes"]) for w in windows), 2),
            "windows_with_drops": sum(1 for w in windows if w["label"]["drop"]),
            "validation": validation,
            "build_seconds": round(time.perf_counter() - t_start, 1),
        }
        (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        (out / "README.md").write_text(dataset_card(stats), encoding="utf-8")
    return stats


def dataset_card(stats: dict) -> str:
    files = stats["files"]
    templates = "\n".join(
        f"| `{t}` | {s['examples']} | {s['labelled']} | `{s['top_label']}` |"
        for t, s in stats["single_by_template"].items())
    validation = "\n".join(
        f"| `{v['template']}` | -{v['estimated_cost_reduction']:.0%} | {v['before_ms']} | {v['after_ms']} "
        f"| x{v['median_speedup']} | x{v['min_speedup']} | {v['slower_share']:.0%} "
        f"| {', '.join(map(str, v['slower_tenants'])) or '-'} |"
        for v in stats["validation"]) or "| (run with `--validate`) | | | | | | | |"
    return f"""---
license: mit
language: [en]
task_categories: [text-generation]
tags: [sql, postgresql, database, index-tuning, query-optimization, reinforcement-learning]
pretty_name: index-drift-rl benchmark
configs:
- config_name: single
  data_files:
  - split: train
    path: single/train.jsonl
  - split: test
    path: single/test.jsonl
  - split: test_unseen_templates
    path: single/test_unseen_templates.jsonl
- config_name: windows
  data_files:
  - split: train
    path: windows/train.jsonl
  - split: test
    path: windows/test.jsonl
---

# index-drift-rl benchmark

Synthetic, reproducible benchmark for **LLM-based PostgreSQL index advisors**
under **drifting workloads**. Generated by
[index-drift-rl](https://github.com/Krishnaghimire/index-drift-rl) (Phase 2).

## Tasks

**`single`**: given a slow query, its `EXPLAIN` plan and the schema of the
tables involved, propose the single best B-tree index (or none).

**`windows`**: given a window of a query log (template mix with example
queries) and the indexes left by the previous window, propose the index set
to have now, with at most {stats['args']['budget']} indexes. The workload drifts between windows
(OLTP -> analytics -> vector search -> a small tenant becoming the busiest), so
good answers create new indexes and drop stale ones.

## How labels are made

1. Candidates are enumerated with classic index-design rules (equality columns,
   then a range/sort column; partial `WHERE` predicates; covering `INCLUDE`).
2. Each candidate is created as a **hypothetical index** with
   [HypoPG](https://github.com/HypoPG/hypopg) and scored by the planner's
   estimated cost (`EXPLAIN`). No index is actually built.
3. `single`: the label is the **smallest** candidate within {TIE_TOLERANCE:.0%} of the
   best cost reduction, or `null` if nothing reduces cost by at least {MIN_REDUCTION:.0%}.
   `windows`: greedy selection of the set minimising workload-weighted cost,
   preferring to keep an existing index when it is within {CHURN_TOLERANCE:.0%} of the best.

## Files

| File | Rows |
|---|---|
{chr(10).join(f"| `{k}.jsonl` | {v} |" for k, v in files.items())}

`schema.sql` holds the table definitions (with row-count estimates).

`single` fields: `query` (parameters inlined), `sql` + `params` (parameterised),
`schema`, `plan`, `baseline_cost`, `candidates` (top {TOP_CANDIDATES} with cost reduction and
estimated size), `label` (`ddl`, `reduction`, `size_bytes`, or `null`), plus
`scenario` / `phase` / `template` metadata.

## Per-template statistics (`single`)

| Template | Examples | Labelled | Most common label |
|---|---|---|---|
{templates}

`events_by_source` and `low_stock_products` never appear in `train`: they form
the `test_unseen_templates` split.

## Validation with real indexes

Most common label per template, built for real; up to 10 queries each, median
of 3 runs per query. Speedups are per query.

| Template | Estimated cost change | Median before (ms) | Median after (ms) | Median speedup | Worst speedup | Queries slower | Tenants slower |
|---|---|---|---|---|---|---|---|
{validation}

Where queries got slower, it is typically for **tenant 1, which owns ~27% of
rows**. The planner estimates the index is cheaper for that tenant too, then
picks a plan that is slower in reality. One index serving every tenant is not
always right: this is exactly the skew an advisor has to reason about.

## Limitations

- **Synthetic, single schema.** 10 relational query templates over one
  multi-tenant commerce schema; models can memorise template -> index. The
  `test_unseen_templates` split is the honest generalisation check.
- **Planner estimates, not latency.** Labels follow the planner's cost model,
  which can be wrong: HypoPG has no statistics for expression indexes (e.g.
  `lower(email)`), so their benefit is underestimated, and for heavily skewed
  tenants an "improving" index can lead to a slower plan. The validation table
  shows the real effect.
- **No vector indexes.** HypoPG cannot simulate HNSW; vector-search queries
  appear in `windows` workload summaries (`simulated: false`) but do not
  affect labels.

## Loading

```python
from datasets import load_dataset
single = load_dataset("<user>/index-drift-rl-bench", "single")
windows = load_dataset("<user>/index-drift-rl-bench", "windows")
```

Dataset scale: `{stats['dataset']}`.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.build_dataset", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/dataset"))
    parser.add_argument("--dsn", default=config.pg_dsn())
    parser.add_argument("--scenarios", nargs="+", default=list(workload.SCENARIOS), choices=list(workload.SCENARIOS))
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--test-seeds", nargs="+", type=int, default=[101])
    parser.add_argument("-n", "--n-queries", type=int, default=2000, help="queries per workload log")
    parser.add_argument("--per-group", type=int, default=15,
                        help="max single examples per (log, phase, template)")
    parser.add_argument("--window", type=int, default=250, help="queries per window")
    parser.add_argument("--samples", type=int, default=3, help="sampled queries per template per window")
    parser.add_argument("--budget", type=int, default=3, help="max indexes per window")
    parser.add_argument("--validate", action="store_true", help="build labels for real and measure latency")
    args = parser.parse_args(argv)

    stats = build(args)
    print(json.dumps({k: stats[k] for k in ("files", "single_label_rate", "windows_mean_indexes",
                                            "windows_with_drops", "build_seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
