"""Checks against a running, seeded sandbox. Skipped automatically otherwise.

    docker compose -f docker/docker-compose.yml up -d --build
    python -m src.seed load --scale 0.1
    pytest tests/test_db_integration.py
"""
import random

import pytest

from src import config, dataset, db, workload
from src.dataset import DatasetSpec


@pytest.fixture(scope="module")
def pg():
    psycopg = pytest.importorskip("psycopg")
    try:
        conn = db.connect(config.pg_dsn(), connect_timeout=2)
    except psycopg.OperationalError:
        pytest.skip("PostgreSQL sandbox is not running")
    if db.dataset_meta(conn) is None:
        conn.close()
        pytest.skip("dataset not seeded (python -m src.seed load)")
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def spec(pg):
    return DatasetSpec.from_dict(db.dataset_meta(pg))


def test_extensions_installed(pg):
    assert db.has_extension(pg, "vector")
    assert db.has_extension(pg, "hypopg")


@pytest.mark.parametrize("salt", [dataset.CUSTOMER_SALT, dataset.PRODUCT_SALT])
def test_sql_tenant_hash_matches_python(pg, salt):
    ids = [1, 2, 3, 17, 999, 123_456, 987_654_321] + list(range(1000, 1200))
    rows = pg.execute("SELECT i, drift_tenant_of(i, %s, %s) FROM unnest(%s::bigint[]) AS i",
                      (dataset.N_TENANTS, salt, ids)).fetchall()
    assert all(t == dataset.tenant_of(i, dataset.N_TENANTS, salt) for i, t in rows)


def test_row_counts_match_spec(pg, spec):
    for table, expected in [("customers", spec.n_customers), ("products", spec.n_products),
                            ("orders", spec.n_orders), ("events", spec.n_events)]:
        assert pg.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == expected


def test_sql_category_matches_python(pg):
    rows = pg.execute("SELECT id, category FROM products WHERE id <= 500").fetchall()
    assert all(category == dataset.category_of(product_id) for product_id, category in rows)


def test_orders_belong_to_customer_tenant(pg):
    mismatched = pg.execute(
        "SELECT count(*) FROM orders o JOIN customers c ON c.id = o.customer_id "
        "WHERE o.tenant_id <> c.tenant_id").fetchone()[0]
    assert mismatched == 0


def test_hypothetical_index_lowers_cost_and_cleans_up(pg):
    from src.hypo import HypoSession

    hypo = HypoSession(pg)
    sql = "SELECT * FROM events WHERE customer_id = %(c)s"
    params = {"c": 42}
    before = hypo.cost(sql, params)
    after = hypo.cost(sql, params, ["CREATE INDEX ON events (customer_id)"])
    assert after < before / 10
    assert hypo.cost(sql, params, ["CREATE INDEX ON events (no_such_column)"]) is None
    assert pg.execute("SELECT count(*) FROM hypopg_list_indexes").fetchone()[0] == 0


@pytest.mark.parametrize("name", sorted(workload.TEMPLATES))
def test_template_runs_and_finds_rows(pg, spec, name):
    template = workload.TEMPLATES[name]
    sampler = workload.ParamSampler(spec, random.Random(0))
    found = 0
    for _ in range(5):
        params = template.sample(sampler)
        pg.execute("EXPLAIN " + template.sql, params)
        found += len(pg.execute(template.sql, params).fetchall())
    assert found > 0, f"{name} returned no rows for 5 sampled parameter sets"
