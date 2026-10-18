"""Prompt format shared by the LLM baselines (Phase 3) and training (Phase 5).

Keeping one definition matters: a model must be evaluated with exactly the
prompt it was trained on.
"""
from __future__ import annotations

import re

SYSTEM = (
    "You are a PostgreSQL performance engineer. You recommend B-tree indexes that make "
    "slow queries fast without hurting production: always use CREATE INDEX CONCURRENTLY, "
    "prefer the smallest index that does the job, and recommend nothing when no index would help."
)

INSTRUCTIONS = (
    "Recommend the single best index for this query. Reply with exactly one SQL statement of the "
    "form `CREATE INDEX CONCURRENTLY <name> ON <table> (...);` or the words `NO INDEX` if no index "
    "would help. Do not explain."
)


def single_prompt(example: dict) -> str:
    """User message for a ``single`` task example."""
    schema = "\n".join(example["schema"])
    return (
        f"Schema (only primary keys are indexed):\n{schema}\n\n"
        f"Slow query:\n{example['query']}\n\n"
        f"Current plan (EXPLAIN):\n{example['plan']}\n\n"
        f"{INSTRUCTIONS}"
    )


_CREATE_RE = re.compile(r"CREATE\s+(?:UNIQUE\s+)?INDEX\b[^;]*", re.IGNORECASE | re.DOTALL)
_NONE_RE = re.compile(r"\bNO\s+INDEX\b", re.IGNORECASE)


def parse_answer(text: str) -> tuple[str, str | None]:
    """("index", ddl) | ("none", None) | ("invalid", None) from a model reply."""
    match = _CREATE_RE.search(text)
    if match:
        ddl = re.sub(r"\s+", " ", match.group(0).replace("```", "")).strip()
        return "index", ddl
    if _NONE_RE.search(text):
        return "none", None
    return "invalid", None
