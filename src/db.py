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


def columns_by_table(conn, tables=DRIFT_TABLES) -> dict[str, set[str]]:
    rows = conn.execute(
        "SELECT table_name, column_name FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = ANY(%s)", (list(tables),)).fetchall()
    out: dict[str, set[str]] = {}
    for table, column in rows:
        out.setdefault(table, set()).add(column)
    return out


def primary_keys(conn, tables=DRIFT_TABLES) -> dict[str, tuple[str, ...]]:
    rows = conn.execute(
        """
        SELECT t.relname, array_agg(a.attname ORDER BY k.ord)
        FROM pg_index i
        JOIN pg_class t ON t.oid = i.indrelid
        CROSS JOIN LATERAL unnest(i.indkey) WITH ORDINALITY AS k(attnum, ord)
        JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = k.attnum
        WHERE i.indisprimary AND t.relname = ANY(%s)
        GROUP BY t.relname
        """, (list(tables),)).fetchall()
    return {table: tuple(cols) for table, cols in rows}


def table_ddl(conn, table: str) -> str:
    """Compact CREATE TABLE text with a row-count estimate, for prompts."""
    columns = conn.execute(
        "SELECT a.attname, format_type(a.atttypid, a.atttypmod), a.attnotnull "
        "FROM pg_attribute a WHERE a.attrelid = %s::regclass AND a.attnum > 0 AND NOT a.attisdropped "
        "ORDER BY a.attnum", (table,)).fetchall()
    parts = [f"{name} {typ}{' NOT NULL' if notnull else ''}" for name, typ, notnull in columns]
    parts += [r[0] for r in conn.execute(
        "SELECT pg_get_constraintdef(oid) FROM pg_constraint WHERE conrelid = %s::regclass AND contype = 'p'",
        (table,)).fetchall()]
    rows = conn.execute("SELECT reltuples::bigint FROM pg_class WHERE oid = %s::regclass", (table,)).fetchone()[0]
    return f"CREATE TABLE {table} ({', '.join(parts)});  -- ~{rows:,} rows"


def dataset_meta(conn) -> dict | None:
    """The DatasetSpec dict written by ``src.seed load``, or None if not seeded."""
    if conn.execute("SELECT to_regclass('public.drift_meta')").fetchone()[0] is None:
        return None
    row = conn.execute("SELECT value FROM drift_meta WHERE key = 'dataset'").fetchone()
    return row[0] if row else None
