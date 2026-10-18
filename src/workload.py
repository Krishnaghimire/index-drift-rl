"""Drifting query-workload generator and runner.

    python -m src.workload list
    python -m src.workload generate --scenario oltp_to_analytics -n 2000 --out data/workloads/oltp_to_analytics.jsonl
    python -m src.workload run data/workloads/oltp_to_analytics.jsonl --out data/runs/baseline

A *scenario* is a sequence of phases. Each phase has a weighted mix of query
templates and, optionally, a "hot" tenant that receives an outsized share of
traffic. Phases can blend into each other gradually (``transition``), which
is the "drift" an index advisor has to keep up with.

``generate`` needs no database: it writes a JSONL log (one header line, then
one line per query). ``run`` replays a log against PostgreSQL, measures
latency, and writes per-query results plus a slow-query report with EXPLAIN
plans - the input for later phases.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Callable, Iterator

from . import config, dataset
from .dataset import DatasetSpec

PLACEHOLDER_RE = re.compile(r"%\((\w+)\)s")


# --------------------------------------------------------------------------
# Parameter sampling
# --------------------------------------------------------------------------

class ParamSampler:
    """Draws query parameters that hit real rows of the seeded dataset.

    Mirrors the distributions used in src/seed.py: tenants and customers are
    skewed, and tenant ownership comes from dataset.tenant_of().
    """

    MAX_REJECTIONS = 10_000

    def __init__(self, spec: DatasetSpec, rng: random.Random):
        self.spec = spec
        self.rng = rng
        self.hot_tenant: int | None = None
        self.hot_tenant_prob = 0.0

    def _hot(self) -> int | None:
        if self.hot_tenant is not None and self.rng.random() < self.hot_tenant_prob:
            return self.hot_tenant
        return None

    def _skewed_id(self, n: int, exponent: float) -> int:
        return 1 + int(n * self.rng.random() ** exponent)

    def _owned_id(self, n: int, exponent: float, salt: int) -> tuple[int, int]:
        """(row id, tenant), restricted to the hot tenant when it is active."""
        want = self._hot()
        for _ in range(self.MAX_REJECTIONS):
            row_id = self._skewed_id(n, exponent)
            tenant = dataset.tenant_of(row_id, self.spec.n_tenants, salt)
            if want is None or tenant == want:
                break
        return row_id, tenant

    def tenant(self) -> int:
        hot = self._hot()
        return hot if hot is not None else 1 + int(self.spec.n_tenants * self.rng.random() ** 3)

    def customer(self) -> tuple[int, int]:
        """(customer_id, tenant_id), activity-skewed like the seeded orders/events."""
        return self._owned_id(self.spec.n_customers, 2.0, dataset.CUSTOMER_SALT)

    def product(self) -> tuple[int, int]:
        """(product_id, tenant_id), popularity-skewed like the seeded order_items."""
        return self._owned_id(self.spec.n_products, 1.5, dataset.PRODUCT_SALT)

    def category(self) -> str:
        return self.rng.choice(dataset.CATEGORIES)

    def event_source(self) -> str:
        sources = dataset.EVENT_SOURCES
        return sources[int(len(sources) * self.rng.random() ** 2)]

    def price_band(self) -> tuple[float, float]:
        low = round(self.rng.uniform(5, 400), 2)
        return low, round(low + self.rng.uniform(10, 100), 2)

    def date_window(self, min_days: int, max_days: int) -> tuple[str, str]:
        days = self.rng.randint(min_days, max_days)
        span = (dataset.DATA_END - dataset.DATA_START).days - days
        start = dataset.DATA_START + timedelta(days=self.rng.randint(0, span))
        return start.isoformat(), (start + timedelta(days=days)).isoformat()


# --------------------------------------------------------------------------
# Query templates
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class QueryTemplate:
    name: str
    kind: str  # "oltp" | "analytics" | "vector"
    sql: str
    sample: Callable[[ParamSampler], dict]
    description: str


def _orders_by_customer(s: ParamSampler) -> dict:
    customer_id, tenant_id = s.customer()
    return {"tenant_id": tenant_id, "customer_id": customer_id}


def _customer_by_email(s: ParamSampler) -> dict:
    customer_id, _ = s.customer()
    return {"email": f"user{customer_id}@example.com"}


def _customer_only(s: ParamSampler) -> dict:
    return {"customer_id": s.customer()[0]}


def _tenant_only(s: ParamSampler) -> dict:
    return {"tenant_id": s.tenant()}


def _products_in_price_band(s: ParamSampler) -> dict:
    # Anchor on a real product so the (tenant, category) pair is populated.
    product_id, tenant_id = s.product()
    low, high = s.price_band()
    return {"tenant_id": tenant_id, "category": dataset.category_of(product_id),
            "min_price": low, "max_price": high}


def _tenant_window(min_days: int, max_days: int) -> Callable[[ParamSampler], dict]:
    def sample(s: ParamSampler) -> dict:
        start, end = s.date_window(min_days, max_days)
        return {"tenant_id": s.tenant(), "start": start, "end": end}
    return sample


def _product_only(s: ParamSampler) -> dict:
    return {"product_id": s.product()[0]}


def _events_by_source(s: ParamSampler) -> dict:
    start, _ = s.date_window(30, 180)
    return {"tenant_id": s.tenant(), "source": s.event_source(), "start": start}


def _similar_products_in_tenant(s: ParamSampler) -> dict:
    product_id, tenant_id = s.product()
    return {"tenant_id": tenant_id, "product_id": product_id}


_TEMPLATE_LIST = [
    QueryTemplate(
        "orders_by_customer", "oltp",
        "SELECT id, status, total, created_at FROM orders "
        "WHERE tenant_id = %(tenant_id)s AND customer_id = %(customer_id)s "
        "ORDER BY created_at DESC LIMIT 20",
        _orders_by_customer,
        "A customer's most recent orders (order history page).",
    ),
    QueryTemplate(
        "customer_by_email", "oltp",
        "SELECT id, tenant_id, country, is_active FROM customers "
        "WHERE lower(email) = lower(%(email)s)",
        _customer_by_email,
        "Case-insensitive login lookup.",
    ),
    QueryTemplate(
        "pending_orders", "oltp",
        "SELECT id, customer_id, total, created_at FROM orders "
        "WHERE tenant_id = %(tenant_id)s AND status = 'pending' "
        "ORDER BY created_at LIMIT 100",
        _tenant_only,
        "Fulfilment queue: a tenant's oldest pending orders (rare status).",
    ),
    QueryTemplate(
        "customer_recent_events", "oltp",
        "SELECT event_type, payload, created_at FROM events "
        "WHERE customer_id = %(customer_id)s ORDER BY created_at DESC LIMIT 50",
        _customer_only,
        "Activity feed for a customer.",
    ),
    QueryTemplate(
        "low_stock_products", "oltp",
        "SELECT id, sku, stock FROM products "
        "WHERE tenant_id = %(tenant_id)s AND stock < 5 ORDER BY stock",
        _tenant_only,
        "Restock alert: a tenant's nearly sold-out products.",
    ),
    QueryTemplate(
        "products_in_price_band", "oltp",
        "SELECT id, sku, price FROM products "
        "WHERE tenant_id = %(tenant_id)s AND category = %(category)s "
        # ::numeric matters: a float8 parameter would force a cast on the
        # column side and make any index on price unusable.
        "AND price BETWEEN %(min_price)s::numeric AND %(max_price)s::numeric ORDER BY price LIMIT 50",
        _products_in_price_band,
        "Catalogue browsing with a price filter.",
    ),
    QueryTemplate(
        "tenant_daily_revenue", "analytics",
        "SELECT date_trunc('day', created_at) AS day, count(*) AS orders, sum(total) AS revenue "
        "FROM orders WHERE tenant_id = %(tenant_id)s "
        "AND created_at >= %(start)s::timestamptz AND created_at < %(end)s::timestamptz "
        "AND status <> 'cancelled' GROUP BY 1 ORDER BY 1",
        _tenant_window(7, 90),
        "Revenue dashboard for a date range.",
    ),
    QueryTemplate(
        "top_products", "analytics",
        "SELECT oi.product_id, sum(oi.quantity) AS units FROM orders o "
        "JOIN order_items oi ON oi.order_id = o.id "
        "WHERE o.tenant_id = %(tenant_id)s "
        "AND o.created_at >= %(start)s::timestamptz AND o.created_at < %(end)s::timestamptz "
        "GROUP BY oi.product_id ORDER BY units DESC LIMIT 10",
        _tenant_window(7, 60),
        "Best sellers for a tenant over a date range (join + aggregate).",
    ),
    QueryTemplate(
        "product_sales", "analytics",
        "SELECT count(DISTINCT order_id) AS orders, sum(quantity) AS units "
        "FROM order_items WHERE product_id = %(product_id)s",
        _product_only,
        "Lifetime sales of one product (no index on product_id by default).",
    ),
    QueryTemplate(
        "events_by_source", "analytics",
        "SELECT event_type, count(*) FROM events "
        "WHERE tenant_id = %(tenant_id)s AND payload->>'source' = %(source)s "
        "AND created_at >= %(start)s::timestamptz GROUP BY event_type",
        _events_by_source,
        "Traffic breakdown by client source (JSONB expression filter).",
    ),
    QueryTemplate(
        "similar_products", "vector",
        "SELECT id, sku, category FROM products "
        "ORDER BY embedding <-> (SELECT embedding FROM products WHERE id = %(product_id)s) LIMIT 10",
        _product_only,
        "Global nearest-neighbour recommendation (HNSW candidate).",
    ),
    QueryTemplate(
        "similar_products_in_tenant", "vector",
        "SELECT id, sku FROM products WHERE tenant_id = %(tenant_id)s "
        "ORDER BY embedding <-> (SELECT embedding FROM products WHERE id = %(product_id)s) LIMIT 10",
        _similar_products_in_tenant,
        "Tenant-filtered nearest-neighbour search (filtered ANN is hard).",
    ),
]
TEMPLATES: dict[str, QueryTemplate] = {t.name: t for t in _TEMPLATE_LIST}


# --------------------------------------------------------------------------
# Drift scenarios
# --------------------------------------------------------------------------

@dataclass
class Phase:
    name: str
    share: float  # relative length of the phase
    mix: dict[str, float]  # template name -> weight
    transition: float = 0.0  # leading fraction of the phase spent blending in from the previous mix
    hot_tenant: int | None = None
    hot_tenant_prob: float = 0.0


OLTP_MIX = {
    "orders_by_customer": 30, "customer_by_email": 20, "customer_recent_events": 20,
    "products_in_price_band": 15, "pending_orders": 10, "low_stock_products": 5,
}
ANALYTICS_MIX = {
    "tenant_daily_revenue": 30, "top_products": 25, "events_by_source": 25,
    "product_sales": 10, "orders_by_customer": 10,
}
VECTOR_MIX = {
    "similar_products": 40, "similar_products_in_tenant": 30,
    "products_in_price_band": 15, "orders_by_customer": 15,
}

SCENARIOS: dict[str, list[Phase]] = {
    "oltp_to_analytics": [
        Phase("oltp", 0.4, OLTP_MIX),
        Phase("analytics", 0.6, ANALYTICS_MIX, transition=0.5),
    ],
    "vector_launch": [
        Phase("oltp", 0.5, OLTP_MIX),
        Phase("vector", 0.5, VECTOR_MIX, transition=0.3),
    ],
    # The smallest tenant suddenly becomes the busiest: selectivity of
    # tenant_id flips, so indexes/statistics tuned for tenant 1 stop fitting.
    "tenant_onboarding": [
        Phase("steady", 0.5, OLTP_MIX),
        Phase("new_big_tenant", 0.5, OLTP_MIX, hot_tenant=dataset.N_TENANTS, hot_tenant_prob=0.6),
    ],
    "full": [
        Phase("oltp", 0.25, OLTP_MIX),
        Phase("analytics", 0.25, ANALYTICS_MIX, transition=0.4),
        Phase("vector", 0.25, VECTOR_MIX, transition=0.4),
        Phase("new_big_tenant", 0.25, OLTP_MIX, transition=0.2,
              hot_tenant=dataset.N_TENANTS, hot_tenant_prob=0.6),
    ],
}


def phase_bounds(phases: list[Phase], n: int) -> list[tuple[int, int]]:
    total = sum(p.share for p in phases)
    starts, acc = [], 0.0
    for p in phases:
        starts.append(round(n * acc / total))
        acc += p.share
    return list(zip(starts, starts[1:] + [n]))


def _blend(a: dict[str, float], b: dict[str, float], t: float) -> dict[str, float]:
    # Ordered key union (not a set) so generation is reproducible across runs.
    keys = list(a) + [k for k in b if k not in a]
    return {k: (1 - t) * a.get(k, 0.0) + t * b.get(k, 0.0) for k in keys}


def phase_state(phases: list[Phase], k: int, progress: float) -> tuple[dict[str, float], int | None, float]:
    """(mix, hot_tenant, hot_tenant_prob) at ``progress`` in [0, 1) through phase ``k``."""
    cur = phases[k]
    if k == 0 or cur.transition <= 0 or progress >= cur.transition:
        return cur.mix, cur.hot_tenant, cur.hot_tenant_prob
    prev = phases[k - 1]
    t = progress / cur.transition
    mix = _blend(prev.mix, cur.mix, t)
    if cur.hot_tenant is not None:
        prev_prob = prev.hot_tenant_prob if prev.hot_tenant == cur.hot_tenant else 0.0
        return mix, cur.hot_tenant, (1 - t) * prev_prob + t * cur.hot_tenant_prob
    if prev.hot_tenant is not None:
        return mix, prev.hot_tenant, (1 - t) * prev.hot_tenant_prob
    return mix, None, 0.0


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def _literal(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def render_sql(sql: str, params: dict) -> str:
    """Inline parameters for a human/LLM-readable query text. Execution always
    uses server-side parameter binding, never this string."""
    return PLACEHOLDER_RE.sub(lambda m: _literal(params[m.group(1)]), sql)


def generate(scenario: str, n_queries: int, spec: DatasetSpec, seed: int = 0,
             qps: float = 20.0) -> Iterator[dict]:
    """Yield a header record, then ``n_queries`` query records."""
    phases = SCENARIOS[scenario]
    bounds = phase_bounds(phases, n_queries)
    rng = random.Random(seed)
    sampler = ParamSampler(spec, rng)

    yield {
        "kind": "header",
        "format": "index-drift-rl/workload/v1",
        "scenario": scenario,
        "n_queries": n_queries,
        "seed": seed,
        "dataset": spec.to_dict(),
        "phases": [
            {"name": p.name, "start": s, "end": e, "mix": p.mix, "transition": p.transition,
             "hot_tenant": p.hot_tenant, "hot_tenant_prob": p.hot_tenant_prob}
            for p, (s, e) in zip(phases, bounds)
        ],
    }

    clock = dataset.DATA_END  # synthetic wall clock for the log
    k = 0
    for seq in range(n_queries):
        while seq >= bounds[k][1]:
            k += 1
        start, end = bounds[k]
        mix, sampler.hot_tenant, sampler.hot_tenant_prob = phase_state(
            phases, k, (seq - start) / max(1, end - start))
        name = rng.choices(list(mix), weights=list(mix.values()))[0]
        template = TEMPLATES[name]
        params = template.sample(sampler)
        yield {
            "kind": "query",
            "seq": seq,
            "logged_at": (clock + timedelta(seconds=seq / qps)).isoformat(),
            "phase": phases[k].name,
            "template": name,
            "category": template.kind,
            "sql": template.sql,
            "params": params,
            "query": render_sql(template.sql, params),
        }


def read_log(path: Path) -> tuple[dict, list[dict]]:
    header, records = None, []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("kind") == "header":
                header = rec
            else:
                records.append(rec)
    if header is None:
        raise ValueError(f"{path}: missing header line")
    return header, records


# --------------------------------------------------------------------------
# Replay against PostgreSQL
# --------------------------------------------------------------------------

def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile."""
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, math.ceil(q / 100 * len(ordered)) - 1))
    return ordered[k]


def _explain(conn, rec: dict) -> dict | None:
    import psycopg

    try:
        row = conn.execute("EXPLAIN (FORMAT JSON) " + rec["sql"], rec["params"]).fetchone()
        return row[0][0]
    except psycopg.Error as e:
        return {"error": str(e).strip()}


def run(log_path: Path, out_dir: Path, dsn: str, slow_ms: float = 50.0,
        timeout_ms: int = 15_000, explain: bool = True, limit: int | None = None) -> list[dict]:
    import psycopg

    from . import db

    header, records = read_log(log_path)
    if limit:
        records = records[:limit]
    out_dir.mkdir(parents=True, exist_ok=True)

    latencies: dict[tuple[str, str], list[float]] = defaultdict(list)
    outcomes: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    first_slow: dict[tuple[str, str], dict] = {}

    # prepare_threshold=None: no server-side prepared statements, so every
    # execution gets a custom plan and measurements are comparable.
    with db.connect(dsn, prepare_threshold=None) as conn, \
            (out_dir / "results.jsonl").open("w", encoding="utf-8") as out:
        meta = db.dataset_meta(conn)
        if meta is None:
            raise SystemExit("Database is not seeded. Run: python -m src.seed load")
        if float(meta["scale"]) != float(header["dataset"]["scale"]):
            print(f"warning: log was generated for scale {header['dataset']['scale']} "
                  f"but the database holds scale {meta['scale']}", file=sys.stderr)
        conn.execute(f"SET statement_timeout = {int(timeout_ms)}")

        for i, rec in enumerate(records, start=1):
            key = (rec["phase"], rec["template"])
            status, rows, error = "ok", 0, None
            t0 = time.perf_counter()
            try:
                cur = conn.execute(rec["sql"], rec["params"])
                rows = len(cur.fetchall()) if cur.description else cur.rowcount
            except psycopg.errors.QueryCanceled:
                status = "timeout"
            except psycopg.Error as e:
                status, error = "error", str(e).strip()
            ms = (time.perf_counter() - t0) * 1000

            latencies[key].append(ms)
            outcomes[key][status] += 1
            if (ms >= slow_ms or status != "ok") and key not in first_slow:
                first_slow[key] = {"seq": rec["seq"], "latency_ms": round(ms, 3),
                                   "query": rec["query"], "sql": rec["sql"], "params": rec["params"],
                                   "plan": _explain(conn, rec) if explain else None}
            out.write(json.dumps({"seq": rec["seq"], "phase": rec["phase"], "template": rec["template"],
                                  "status": status, "latency_ms": round(ms, 3), "rows": rows,
                                  "error": error}) + "\n")
            if i % 500 == 0:
                print(f"  {i}/{len(records)} queries", file=sys.stderr)

    phase_order = {p["name"]: i for i, p in enumerate(header["phases"])}
    summary = []
    for (phase, template), values in latencies.items():
        p95 = percentile(values, 95)
        counts = outcomes[(phase, template)]
        summary.append({
            "phase": phase, "template": template, "count": len(values),
            "ok": counts["ok"], "timeouts": counts["timeout"], "errors": counts["error"],
            "p50_ms": round(percentile(values, 50), 3), "p95_ms": round(p95, 3),
            "max_ms": round(max(values), 3), "mean_ms": round(sum(values) / len(values), 3),
            "slow": p95 >= slow_ms or counts["ok"] < len(values),
        })
    summary.sort(key=lambda r: (phase_order.get(r["phase"], 99), -r["p95_ms"]))

    slow = [{**row, "example": first_slow.get((row["phase"], row["template"]))}
            for row in summary if row["slow"]]
    run_info = {"log": str(log_path), "scenario": header["scenario"], "dataset": meta,
                "slow_ms": slow_ms, "timeout_ms": timeout_ms}
    (out_dir / "summary.json").write_text(json.dumps({**run_info, "groups": summary}, indent=2), encoding="utf-8")
    (out_dir / "slow_queries.json").write_text(json.dumps({**run_info, "slow": slow}, indent=2), encoding="utf-8")
    return summary


def print_summary(summary: list[dict]) -> None:
    print(f"{'phase':<16}{'template':<28}{'n':>6}{'p50 ms':>10}{'p95 ms':>10}{'max ms':>10}  flags")
    for r in summary:
        flags = []
        if r["slow"]:
            flags.append("SLOW")
        if r["timeouts"]:
            flags.append(f"{r['timeouts']} timeouts")
        if r["errors"]:
            flags.append(f"{r['errors']} errors")
        print(f"{r['phase']:<16}{r['template']:<28}{r['count']:>6}{r['p50_ms']:>10.1f}"
              f"{r['p95_ms']:>10.1f}{r['max_ms']:>10.1f}  {' '.join(flags)}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.workload", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list scenarios and query templates")

    p_gen = sub.add_parser("generate", help="write a drifting workload log (no database needed)")
    p_gen.add_argument("--scenario", choices=sorted(SCENARIOS), default="full")
    p_gen.add_argument("-n", "--n-queries", type=int, default=2000)
    p_gen.add_argument("--scale", type=float, default=1.0, help="must match `src.seed load --scale`")
    p_gen.add_argument("--seed", type=int, default=0)
    p_gen.add_argument("--out", type=Path, required=True)

    p_run = sub.add_parser("run", help="replay a workload log against PostgreSQL")
    p_run.add_argument("log", type=Path)
    p_run.add_argument("--out", type=Path, required=True, help="output directory")
    p_run.add_argument("--dsn", default=config.pg_dsn())
    p_run.add_argument("--slow-ms", type=float, default=50.0, help="p95 threshold for the slow-query report")
    p_run.add_argument("--timeout-ms", type=int, default=15_000)
    p_run.add_argument("--limit", type=int, help="only replay the first N queries")
    p_run.add_argument("--no-explain", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "list":
        for name, phases in SCENARIOS.items():
            print(f"{name}: " + " -> ".join(p.name for p in phases))
        print()
        for t in TEMPLATES.values():
            print(f"[{t.kind:<9}] {t.name:<28} {t.description}")
    elif args.command == "generate":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        counts: dict[str, int] = defaultdict(int)
        with args.out.open("w", encoding="utf-8") as f:
            for rec in generate(args.scenario, args.n_queries, DatasetSpec(scale=args.scale), seed=args.seed):
                f.write(json.dumps(rec) + "\n")
                if rec["kind"] == "query":
                    counts[rec["phase"]] += 1
        print(f"wrote {args.n_queries} queries to {args.out} ({dict(counts)})")
    elif args.command == "run":
        summary = run(args.log, args.out, args.dsn, slow_ms=args.slow_ms, timeout_ms=args.timeout_ms,
                      explain=not args.no_explain, limit=args.limit)
        print_summary(summary)
        print(f"\nresults: {args.out / 'results.jsonl'}\nslow queries: {args.out / 'slow_queries.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
