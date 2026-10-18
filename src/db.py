"""Small PostgreSQL helpers shared by the CLIs and tests."""
from __future__ import annotations

from . import config

DRIFT_TABLES = ("tenants", "customers", "products", "orders", "order_items", "events")

# Secondary indexes = everything except indexes backing a PK/unique constraint.
_SECONDARY_INDEXES_SQL = """
SELECT t.relname,
       quote_ident(n.nspname) || '.' || quote_ident(ic.relname),
       pg_get_indexdef(i.indexrelid),
       pg_size_pretty(pg_relation_size(i.indexrelid))
FROM pg_index i
JOIN pg_class ic ON ic.oid = i.indexrelid
JOIN pg_class t ON t.oid = i.indrelid
JOIN pg_namespace n ON n.oid = t.relnamespace
WHERE n.nspname = 'public'
  AND NOT EXISTS (SELECT 1 FROM pg_constraint c WHERE c.conindid = i.indexrelid)
ORDER BY t.relname, ic.relname
"""


def connect(dsn: str | None = None, **kwargs):
    import psycopg

    return psycopg.connect(dsn or config.pg_dsn(), autocommit=True, **kwargs)


def secondary_indexes(conn) -> list[tuple[str, str, str, str]]:
    """(table, qualified index name, definition, size) for each secondary index."""
    return conn.execute(_SECONDARY_INDEXES_SQL).fetchall()


def drop_secondary_indexes(conn) -> list[str]:
    """Return the database to its primary-keys-only baseline."""
    dropped = []
    for _table, name, _definition, _size in secondary_indexes(conn):
        # name comes from quote_ident(), so it is safe to splice in.
        conn.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {name}")
        dropped.append(name)
    if has_extension(conn, "hypopg"):
        conn.execute("SELECT hypopg_reset()")
    return dropped


def has_extension(conn, name: str) -> bool:
    row = conn.execute("SELECT 1 FROM pg_extension WHERE extname = %s", (name,)).fetchone()
    return row is not None


def dataset_meta(conn) -> dict | None:
    """The DatasetSpec dict written by ``src.seed load``, or None if not seeded."""
    if conn.execute("SELECT to_regclass('public.drift_meta')").fetchone()[0] is None:
        return None
    row = conn.execute("SELECT value FROM drift_meta WHERE key = 'dataset'").fetchone()
    return row[0] if row else None
