import pytest

from src import candidates as cands
from src import query_features as qf
from src import workload

COLUMNS = {
    "orders": {"id", "tenant_id", "customer_id", "status", "total", "created_at"},
    "order_items": {"order_id", "line_no", "product_id", "quantity", "unit_price"},
}


def ddls(sql, **kw):
    return [c.ddl() for c in cands.generate(qf.extract(sql, COLUMNS), **kw)]


def test_to_positional_numbers_names_consistently():
    assert qf.to_positional("a = %(x)s AND b = %(y)s OR c = %(x)s") == "a = $1 AND b = $2 OR c = $1"


def test_extract_equality_range_and_sort():
    f = qf.extract(workload.TEMPLATES["tenant_daily_revenue"].sql)
    kinds = {(p.key.expr, p.kind, p.literal) for p in f.predicates}
    assert ("tenant_id", "eq", None) in kinds
    assert ("created_at", "range", None) in kinds
    assert ("status", "neq", "'cancelled'") in kinds


def test_extract_expressions():
    f = qf.extract(workload.TEMPLATES["customer_by_email"].sql)
    assert [(p.key.expr, p.key.is_column) for p in f.predicates] == [("lower(email)", False)]
    f = qf.extract(workload.TEMPLATES["events_by_source"].sql)
    assert "payload ->> 'source'" in {p.key.expr for p in f.predicates}


def test_extract_join_resolves_aliases():
    f = qf.extract(workload.TEMPLATES["top_products"].sql, COLUMNS)
    assert f.tables == {"o": "orders", "oi": "order_items"}
    (left, right), = f.joins
    assert {(left.table, left.expr), (right.table, right.expr)} == {("order_items", "order_id"), ("orders", "id")}
    assert [(k.table, k.expr) for k in f.group_by] == [("order_items", "product_id")]


def test_extract_flags_vector_sort():
    assert qf.extract(workload.TEMPLATES["similar_products"].sql).vector_sort


def test_rejects_non_select():
    with pytest.raises(ValueError):
        qf.extract("DELETE FROM orders")


def test_candidates_put_equality_before_sort():
    out = ddls(workload.TEMPLATES["orders_by_customer"].sql)
    assert "CREATE INDEX ON orders (tenant_id, customer_id, created_at DESC)" in out


def test_candidates_include_partial_index():
    out = ddls(workload.TEMPLATES["pending_orders"].sql)
    assert "CREATE INDEX ON orders (tenant_id, created_at) WHERE status = 'pending'" in out


def test_candidates_wrap_expressions():
    out = ddls(workload.TEMPLATES["customer_by_email"].sql)
    assert out == ["CREATE INDEX ON customers ((lower(email)))"]


def test_candidates_skip_primary_key_duplicates():
    out = ddls(workload.TEMPLATES["top_products"].sql, primary_keys={"orders": ("id",)})
    assert "CREATE INDEX ON orders (id)" not in out
    assert "CREATE INDEX ON order_items (product_id)" in out


def test_candidates_are_unique_and_named_deterministically():
    for name, t in workload.TEMPLATES.items():
        cs = cands.generate(qf.extract(t.sql, COLUMNS))
        assert len({c.ddl() for c in cs}) == len(cs), name
        assert len({c.name for c in cs}) == len(cs), name
        assert all(len(c.name) <= 63 for c in cs)  # PostgreSQL identifier limit


def test_named_concurrent_ddl():
    c = cands.Candidate("orders", ("tenant_id",), where="status = 'pending'")
    assert c.ddl(concurrently=True, named=True) == (
        f"CREATE INDEX CONCURRENTLY {c.name} ON orders (tenant_id) WHERE status = 'pending'")
