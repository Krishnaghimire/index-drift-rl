"""Reward behaviour against a running, seeded sandbox (skipped otherwise)."""
import pytest

from src import config, db, workload

PENDING_LABEL = "CREATE INDEX CONCURRENTLY lbl ON orders (tenant_id, created_at) WHERE status = 'pending'"


@pytest.fixture(scope="module")
def env():
    psycopg = pytest.importorskip("psycopg")
    try:
        conn = db.connect(config.pg_dsn(), connect_timeout=2)
    except psycopg.OperationalError:
        pytest.skip("PostgreSQL sandbox is not running")
    seeded = db.dataset_meta(conn) is not None
    conn.close()
    if not seeded:
        pytest.skip("dataset not seeded")
    from src.environment import IndexEnvironment

    environment = IndexEnvironment()
    yield environment
    environment.close()


@pytest.fixture()
def pending_example():
    return {"id": "test-pending", "template": "pending_orders", "tables": ["orders"],
            "sql": workload.TEMPLATES["pending_orders"].sql, "params": {"tenant_id": 2},
            "label": {"ddl": PENDING_LABEL, "reduction": 0.98, "size_bytes": 1}}


def score(env, example, answer):
    return env.single(answer, example).value


def test_label_scores_full_marks(env, pending_example):
    assert score(env, pending_example, PENDING_LABEL + ";") == pytest.approx(1.0, abs=0.01)


def test_ordering_matches_expert_judgement(env, pending_example):
    good = score(env, pending_example, PENDING_LABEL + ";")
    bloated = score(env, pending_example,
                    "CREATE INDEX CONCURRENTLY a ON orders (tenant_id, status, created_at) INCLUDE (customer_id, total);")
    tailored = score(env, pending_example,
                     "CREATE INDEX CONCURRENTLY a ON orders (created_at) WHERE status = 'pending' AND tenant_id = 2;")
    useless = score(env, pending_example, "CREATE INDEX CONCURRENTLY a ON orders (total);")
    assert good > bloated > tailored > useless


@pytest.mark.parametrize("answer, expected", [
    ("DROP TABLE orders;", -1.0),
    ("make it faster", -1.0),
    ("CREATE INDEX a ON orders (tenant_id);", -0.5),
    ("CREATE UNIQUE INDEX CONCURRENTLY a ON orders (tenant_id);", -0.5),
    ("CREATE INDEX CONCURRENTLY a ON orders (no_such_column);", -0.75),
    ("CREATE INDEX CONCURRENTLY a ON customers (email);", -0.75),  # table not in the query
    ("CREATE INDEX CONCURRENTLY a ON orders (tenant_id); CREATE INDEX CONCURRENTLY b ON orders (status);", -1.0),
    ("NO INDEX", 0.0),
])
def test_penalties(env, pending_example, answer, expected):
    assert score(env, pending_example, answer) == expected


def test_no_index_is_right_when_label_is_none(env, pending_example):
    pending_example["label"] = None
    assert score(env, pending_example, "NO INDEX") == 1.0


def test_trl_adapter_handles_chat_completions(env, pending_example):
    from src.environment import make_trl_reward

    reward = make_trl_reward(env, "single")
    columns = {k: [v, v] for k, v in pending_example.items()}
    completions = [[{"role": "assistant", "content": PENDING_LABEL + ";"}],
                   [{"role": "assistant", "content": "NO INDEX"}]]
    values = reward(prompts=["p", "p"], completions=completions, **columns)
    assert values[0] == pytest.approx(1.0, abs=0.01) and values[1] == 0.0
    assert reward.__name__ == "index_reward_single"


def test_window_reward(env):
    query = workload.render_sql(workload.TEMPLATES["customer_recent_events"].sql, {"customer_id": 7})
    example = {
        "existing_indexes": ["CREATE INDEX CONCURRENTLY old_idx ON orders (status)"],
        "workload": [{"template": "customer_recent_events", "count": 100, "share": 1.0,
                      "example_query": query, "simulated": True}],
        "label": {"indexes": ["CREATE INDEX CONCURRENTLY ev_idx ON events (customer_id)"]},
    }
    best = "DROP INDEX CONCURRENTLY old_idx;\nCREATE INDEX CONCURRENTLY ev_idx ON events (customer_id);"
    assert env.window(best, example).value == pytest.approx(1.0, abs=0.01)
    assert env.window("NO CHANGES", example).value == pytest.approx(0.0, abs=0.01)
    assert env.window("DROP INDEX CONCURRENTLY old_idx;", example).value == pytest.approx(0.0, abs=0.01)
    assert env.window(best.replace("CONCURRENTLY ", ""), example).value == -0.5
    too_many = best + "".join(f"\nCREATE INDEX CONCURRENTLY x{i} ON orders ({col});"
                              for i, col in enumerate(("total", "created_at", "customer_id")))
    assert env.window(too_many, example).value == -0.5
