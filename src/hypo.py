"""Cost queries under hypothetical indexes with HypoPG.

HypoPG adds fake index entries that only the planner sees. Plain EXPLAIN
(not EXPLAIN ANALYZE) then shows the plan and estimated cost as if the index
existed. That takes milliseconds, versus seconds to build a real index,
which is what makes scoring thousands of candidates (and later, the RL
reward) affordable.
"""
from __future__ import annotations

import json
from contextlib import contextmanager


class HypoSession:
    def __init__(self, conn):
        from . import db

        if not db.has_extension(conn, "hypopg"):
            raise RuntimeError("the hypopg extension is not installed in this database")
        self.conn = conn
        self.reset()
        self._cost_cache: dict[tuple, float | None] = {}

    def reset(self) -> None:
        self.conn.execute("SELECT hypopg_reset()")

    def explain_json(self, sql: str, params: dict) -> dict:
        return self.conn.execute("EXPLAIN (FORMAT JSON) " + sql, params).fetchone()[0][0]

    def plan_text(self, sql: str, params: dict) -> str:
        rows = self.conn.execute("EXPLAIN " + sql, params).fetchall()
        return "\n".join(r[0] for r in rows)

    def _create(self, ddl: str) -> int:
        """Create one hypothetical index; returns its size estimate in bytes."""
        oid = self.conn.execute("SELECT indexrelid FROM hypopg_create_index(%s)", (ddl,)).fetchone()[0]
        return self.conn.execute("SELECT hypopg_relation_size(%s)", (oid,)).fetchone()[0]

    @contextmanager
    def indexes(self, ddls):
        """Make ``ddls`` hypothetically exist; yields {ddl: size_bytes}.
        Raises psycopg.Error if any definition is invalid."""
        try:
            yield {ddl: self._create(ddl) for ddl in ddls}
        finally:
            self.reset()

    def costs(self, queries: list[tuple[str, dict]], ddls=()) -> list[float | None]:
        """Planner total cost of each (sql, params) with ``ddls`` present.
        None if a definition is invalid. Creates the indexes once per call; cached."""
        import psycopg

        ddl_key = frozenset(ddls)
        keys = [(sql, json.dumps(params, sort_keys=True), ddl_key) for sql, params in queries]
        missing = [(k, q) for k, q in zip(keys, queries) if k not in self._cost_cache]
        if missing:
            try:
                with self.indexes(sorted(ddls)):
                    for key, (sql, params) in missing:
                        self._cost_cache[key] = self.explain_json(sql, params)["Plan"]["Total Cost"]
            except psycopg.Error:
                for key, _ in missing:
                    self._cost_cache.setdefault(key, None)
        return [self._cost_cache[k] for k in keys]

    def cost(self, sql: str, params: dict, ddls=()) -> float | None:
        return self.costs([(sql, params)], ddls)[0]

    def size(self, ddl: str) -> int | None:
        import psycopg

        try:
            with self.indexes([ddl]) as sizes:
                return sizes[ddl]
        except psycopg.Error:
            return None
