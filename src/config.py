"""Connection settings, overridable through environment variables."""
from __future__ import annotations

import os


def pg_dsn() -> str:
    return os.environ.get("DRIFT_PG_DSN", "postgresql://drift:drift@localhost:5432/drift")


def es_url() -> str:
    return os.environ.get("DRIFT_ES_URL", "http://localhost:9200")
