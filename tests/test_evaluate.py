import pytest

from src import evaluate, prompts


@pytest.mark.parametrize("text, expected", [
    ("CREATE INDEX CONCURRENTLY i ON orders (tenant_id);", ("index", "CREATE INDEX CONCURRENTLY i ON orders (tenant_id)")),
    ("```sql\ncreate index concurrently on orders\n  (tenant_id, created_at DESC);\n```",
     ("index", "create index concurrently on orders (tenant_id, created_at DESC)")),
    ("NO INDEX", ("none", None)),
    ("No index is needed here.", ("none", None)),
    ("I would add an index on tenant_id.", ("invalid", None)),
])
def test_parse_answer(text, expected):
    assert prompts.parse_answer(text) == expected


def test_check_index_accepts_and_canonicalises():
    check = evaluate.check_index("CREATE INDEX CONCURRENTLY my_idx ON orders (tenant_id, created_at DESC)", {"orders"})
    assert check.ok and check.concurrently and check.table == "orders"
    assert check.canonical == "CREATE INDEX ON orders (tenant_id, created_at DESC)"


@pytest.mark.parametrize("ddl, reason", [
    ("CREATE INDEX ON customers (email)", "not used by the query"),
    ("CREATE INDEX ON orders USING hash (tenant_id)", "not B-tree"),
    ("DROP TABLE orders", "not a single CREATE INDEX"),
    ("CREATE INDEX ON orders (tenant_id", "parse error"),
])
def test_check_index_rejects(ddl, reason):
    check = evaluate.check_index(ddl, {"orders"})
    assert not check.ok and reason in check.reason


def test_canonical_ignores_name_concurrently_and_spacing():
    a = evaluate.canonical("CREATE INDEX CONCURRENTLY a ON orders (tenant_id) WHERE status = 'pending'")
    b = evaluate.canonical("create   index b on orders(tenant_id) where status='pending'")
    assert a == b


def test_single_prompt_contains_inputs():
    example = {"schema": ["CREATE TABLE orders (id bigint);"], "query": "SELECT 1 FROM orders",
               "plan": "Seq Scan on orders"}
    text = prompts.single_prompt(example)
    assert all(part in text for part in ("CREATE TABLE orders", "SELECT 1 FROM orders", "Seq Scan", "NO INDEX"))
