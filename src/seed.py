"""Create and populate the synthetic dataset inside PostgreSQL.

    python -m src.seed load --scale 1.0     # recreate schema + load data (~1-3 min)
    python -m src.seed status               # row counts, sizes, indexes, extensions
    python -m src.seed drop-indexes         # back to the primary-keys-only baseline

Data is generated server-side with generate_series(), so loading is fast and
needs almost no client memory. The only client-side randomness is the
per-category embedding centroids, seeded for reproducibility.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

from . import config, dataset, db
from .dataset import DatasetSpec

SCHEMA_SQL = (Path(__file__).parent / "sql" / "schema.sql").read_text(encoding="utf-8")

# Embedding = category centroid + uniform noise, so products cluster by category.
EMBEDDING_NOISE = 0.6

# NOTE: queries run with parameters, so a literal '%' would be read as a
# placeholder. Use mod() instead of the % operator.

_RANDOM_TS = "%(start)s::timestamptz + random() * (%(end)s::timestamptz - %(start)s::timestamptz)"


def _status_case(column: str) -> str:
    """CASE expression mapping a uniform [0,1) column to ORDER_STATUSES."""
    branches, cumulative = [], 0.0
    for status, p in dataset.ORDER_STATUSES[:-1]:
        cumulative += p
        branches.append(f"WHEN {column} < {cumulative:.6f} THEN '{status}'")
    return f"CASE {' '.join(branches)} ELSE '{dataset.ORDER_STATUSES[-1][0]}' END"


SEED_STEPS: list[tuple[str, str]] = [
    ("tenants", """
INSERT INTO tenants (id, name, plan, region, created_at)
SELECT g,
       'tenant_' || g,
       CASE WHEN g <= 5 THEN 'enterprise' WHEN g <= 20 THEN 'pro' ELSE 'starter' END,
       (ARRAY['us-east', 'eu-west', 'ap-south'])[1 + mod(g, 3)],
       %(start)s::timestamptz + make_interval(days => g)
FROM generate_series(1, %(n_tenants)s::int) AS g
"""),
    ("customers", f"""
INSERT INTO customers (id, tenant_id, email, country, signup_at, is_active)
SELECT g,
       drift_tenant_of(g, %(n_tenants)s::int, {dataset.CUSTOMER_SALT}),
       'User' || g || '@Example.com',
       (%(countries)s::text[])[1 + floor(power(random(), 2) * %(n_countries)s::int)::int],
       {_RANDOM_TS},
       random() < 0.85
FROM generate_series(1, %(n_customers)s::bigint) AS g
"""),
    ("products", f"""
INSERT INTO products (id, tenant_id, sku, category, price, stock, created_at, embedding)
SELECT g,
       drift_tenant_of(g, %(n_tenants)s::int, {dataset.PRODUCT_SALT}),
       'SKU-' || lpad(g::text, 8, '0'),
       c.category,
       round((5 + random() * 495)::numeric, 2),
       floor(random() * 200)::int,
       {_RANDOM_TS},
       (SELECT array_agg(u.v + (random() - 0.5) * %(noise)s::float8 ORDER BY u.ord)
          FROM unnest(c.centroid) WITH ORDINALITY AS u(v, ord))::vector
FROM generate_series(1, %(n_products)s::bigint) AS g
JOIN drift_centroids c
  ON c.idx = 1 + floor(drift_unit_hash(g, {dataset.CATEGORY_SALT}) * %(n_categories)s::int)::int
"""),
    # Customer activity is skewed (low ids are heavy users); every order and
    # event belongs to its customer's tenant. MATERIALIZED pins each random()
    # draw so customer_id is consistent within a row.
    ("orders", f"""
WITH s AS MATERIALIZED (
    SELECT g AS id,
           1 + floor(power(random(), 2) * %(n_customers)s::bigint)::bigint AS customer_id,
           random() AS r,
           {_RANDOM_TS} AS created_at
    FROM generate_series(1, %(n_orders)s::bigint) AS g
)
INSERT INTO orders (id, tenant_id, customer_id, status, total, created_at)
SELECT id,
       drift_tenant_of(customer_id, %(n_tenants)s::int, {dataset.CUSTOMER_SALT}),
       customer_id,
       {_status_case('r')},
       round((5 + exp(random() * 6))::numeric, 2),
       created_at
FROM s
"""),
    ("order_items", """
INSERT INTO order_items (order_id, line_no, product_id, quantity, unit_price)
SELECT o.id,
       l.line_no,
       1 + floor(power(random(), 1.5) * %(n_products)s::bigint)::bigint,
       1 + floor(random() * 4)::int,
       round((5 + random() * 495)::numeric, 2)
FROM orders o
CROSS JOIN LATERAL generate_series(1, 1 + mod(o.id, 5)::int) AS l(line_no)
"""),
    ("events", f"""
WITH s AS MATERIALIZED (
    SELECT g AS id,
           1 + floor(power(random(), 2) * %(n_customers)s::bigint)::bigint AS customer_id,
           {_RANDOM_TS} AS created_at
    FROM generate_series(1, %(n_events)s::bigint) AS g
)
INSERT INTO events (id, tenant_id, customer_id, event_type, payload, created_at)
SELECT id,
       drift_tenant_of(customer_id, %(n_tenants)s::int, {dataset.CUSTOMER_SALT}),
       customer_id,
       (%(event_types)s::text[])[1 + floor(random() * %(n_event_types)s::int)::int],
       jsonb_build_object(
           'source', (%(sources)s::text[])[1 + floor(power(random(), 2) * %(n_sources)s::int)::int],
           'latency_ms', floor(random() * 2000)::int,
           'app_version', '1.' || floor(random() * 20)::int
       ),
       created_at
FROM s
"""),
]


def _params(spec: DatasetSpec) -> dict:
    return {
        "start": dataset.DATA_START,
        "end": dataset.DATA_END,
        "n_tenants": spec.n_tenants,
        "n_customers": spec.n_customers,
        "n_products": spec.n_products,
        "n_orders": spec.n_orders,
        "n_events": spec.n_events,
        "countries": list(dataset.COUNTRIES),
        "n_countries": len(dataset.COUNTRIES),
        "event_types": list(dataset.EVENT_TYPES),
        "n_event_types": len(dataset.EVENT_TYPES),
        "sources": list(dataset.EVENT_SOURCES),
        "n_sources": len(dataset.EVENT_SOURCES),
        "n_categories": len(dataset.CATEGORIES),
        "noise": EMBEDDING_NOISE,
    }


def _centroids(seed: int) -> list[tuple[int, str, list[float]]]:
    rng = random.Random(seed)
    return [
        (i, category, [rng.uniform(-1, 1) for _ in range(dataset.EMBEDDING_DIM)])
        for i, category in enumerate(dataset.CATEGORIES, start=1)
    ]


def load(dsn: str, spec: DatasetSpec, seed: int = 42) -> None:
    from psycopg.types.json import Jsonb

    print(f"Loading dataset: {spec.to_dict()}")
    started = time.perf_counter()
    with db.connect(dsn) as conn:
        conn.execute(SCHEMA_SQL)
        conn.execute("SET synchronous_commit = off")
        # setseed() takes a value in [-1, 1]; best-effort reproducibility.
        conn.execute("SELECT setseed(%s)", ((seed % 1000) / 1000.0,))

        conn.execute("CREATE TEMP TABLE drift_centroids (idx int, category text, centroid float8[])")
        with conn.cursor() as cur:
            cur.executemany("INSERT INTO drift_centroids VALUES (%s, %s, %s)", _centroids(seed))

        params = _params(spec)
        for name, sql in SEED_STEPS:
            t0 = time.perf_counter()
            # Each statement references only some of the params; psycopg
            # ignores unused keys in a mapping.
            conn.execute(sql, params)
            print(f"  {name:<12} {time.perf_counter() - t0:6.1f}s")

        conn.execute(
            "INSERT INTO drift_meta (key, value) VALUES ('dataset', %s), ('seed', %s)",
            (Jsonb(spec.to_dict()), Jsonb(seed)),
        )
        t0 = time.perf_counter()
        conn.execute("VACUUM ANALYZE")
        print(f"  {'vacuum':<12} {time.perf_counter() - t0:6.1f}s")
    print(f"Done in {time.perf_counter() - started:.1f}s")


def status(dsn: str) -> None:
    with db.connect(dsn) as conn:
        meta = db.dataset_meta(conn)
        print(f"Dataset: {meta if meta else 'not seeded (run: python -m src.seed load)'}")

        print("\nTables:")
        rows = conn.execute(
            """
            SELECT c.relname, c.reltuples::bigint, pg_size_pretty(pg_total_relation_size(c.oid))
            FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = 'public' AND c.relkind = 'r' AND c.relname = ANY(%s)
            ORDER BY c.relname
            """,
            (list(db.DRIFT_TABLES),),
        ).fetchall()
        for name, est_rows, size in rows:
            print(f"  {name:<12} ~{est_rows:>10,} rows  {size:>10}")

        print("\nSecondary indexes:")
        indexes = db.secondary_indexes(conn)
        for _table, _name, definition, size in indexes:
            print(f"  [{size}] {definition}")
        if not indexes:
            print("  (none - primary keys only)")

        print("\nExtensions:")
        for name, version in conn.execute("SELECT extname, extversion FROM pg_extension ORDER BY 1"):
            print(f"  {name} {version}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.seed", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dsn", default=config.pg_dsn(), help="PostgreSQL DSN (env DRIFT_PG_DSN)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_load = sub.add_parser("load", help="recreate schema and load synthetic data")
    p_load.add_argument("--scale", type=float, default=1.0,
                        help="1.0 ~ 1M orders / 2M events (~600 MB); use 0.1 for a quick start")
    p_load.add_argument("--seed", type=int, default=42)

    sub.add_parser("status", help="show tables, indexes and extensions")
    sub.add_parser("drop-indexes", help="drop all secondary indexes (and hypothetical ones)")

    args = parser.parse_args(argv)
    if args.command == "load":
        load(args.dsn, DatasetSpec(scale=args.scale), seed=args.seed)
    elif args.command == "status":
        status(args.dsn)
    elif args.command == "drop-indexes":
        with db.connect(args.dsn) as conn:
            dropped = db.drop_secondary_indexes(conn)
        print("\n".join(f"dropped {name}" for name in dropped) or "nothing to drop")
    return 0


if __name__ == "__main__":
    sys.exit(main())
