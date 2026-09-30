"""Describe the live database schema for the LLM prompt.

The description is built by introspecting the database with SQLAlchemy's
inspector (not from hand-written docs), so it always matches what actually
exists. For low-cardinality text columns we also list the distinct values,
which lets the model write correct filters such as `status = 'completed'`.

Nothing here assumes a particular set of tables: the same code describes the
bundled sample database and any uploaded one. All introspection for one call
runs on a single connection, so an optional time budget covers all of it.
"""

from __future__ import annotations

import re

from sqlalchemy import Connection, Engine, inspect, text

from database.connection import time_limit

# Text columns with at most this many distinct values get them listed.
MAX_DISTINCT_VALUES = 12
# Long listed values are shortened so one odd column cannot flood the prompt.
MAX_VALUE_CHARS = 60

_DATE_TEXT = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATETIME_TEXT = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}")


def get_table_names(engine: Engine, timeout_seconds: float | None = None) -> list[str]:
    """User tables (SQLAlchemy already excludes SQLite's internal sqlite_* tables)."""
    with engine.connect() as conn, time_limit(conn, timeout_seconds):
        return sorted(inspect(conn).get_table_names())


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _count_rows(conn: Connection, table: str) -> int:
    return conn.execute(text(f"SELECT COUNT(*) FROM {_quote(table)}")).scalar_one()


def get_table_row_counts(engine: Engine, timeout_seconds: float | None = None) -> dict[str, int]:
    """Return ``{table: row_count}`` for every user table (sorted by name).

    Discovery and every COUNT(*) share the one ``timeout_seconds`` budget.
    """
    with engine.connect() as conn, time_limit(conn, timeout_seconds):
        return {t: _count_rows(conn, t) for t in sorted(inspect(conn).get_table_names())}


def _format_value(value: object) -> str:
    if isinstance(value, str) and len(value) > MAX_VALUE_CHARS:
        value = value[: MAX_VALUE_CHARS - 3] + "..."
    return repr(value)


def _date_storage_note(lo: object, hi: object) -> str:
    """Describe how dates are stored, judging by the actual min/max values."""
    if isinstance(lo, str) and isinstance(hi, str):
        if _DATE_TEXT.match(lo) and _DATE_TEXT.match(hi):
            return " (stored as 'YYYY-MM-DD' text)"
        if _DATETIME_TEXT.match(lo) and _DATETIME_TEXT.match(hi):
            return " (stored as 'YYYY-MM-DD HH:MM:SS' text)"
    if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
        return " (stored as numbers, not text)"
    return ""


def get_schema_description(engine: Engine, timeout_seconds: float | None = None) -> str:
    """Schema text for the prompt. Reflection, row counts, distinct-value
    sampling and date ranges all share the one ``timeout_seconds`` budget."""
    lines: list[str] = []

    with engine.connect() as conn, time_limit(conn, timeout_seconds):
        inspector = inspect(conn)
        for table in sorted(inspector.get_table_names()):
            pk_cols = set(inspector.get_pk_constraint(table).get("constrained_columns") or [])
            columns = inspector.get_columns(table)
            lines.append(f"Table {table} ({_count_rows(conn, table)} rows)")

            for col in columns:
                name = col["name"]
                parts = [f"  - {name} {col['type']}"]
                if name in pk_cols:
                    parts.append("PRIMARY KEY")
                if not col.get("nullable", True):
                    parts.append("NOT NULL")
                if str(col["type"]).upper().startswith(("VARCHAR", "TEXT", "CHAR")):
                    values = conn.execute(
                        text(
                            f"SELECT DISTINCT {_quote(name)} FROM {_quote(table)} "
                            f"ORDER BY 1 LIMIT {MAX_DISTINCT_VALUES + 1}"
                        )
                    ).scalars().all()
                    if len(values) <= MAX_DISTINCT_VALUES:
                        parts.append("values: " + ", ".join(_format_value(v) for v in values))
                lines.append(" ".join(parts))

            for fk in inspector.get_foreign_keys(table):
                for local, remote in zip(fk["constrained_columns"], fk["referred_columns"]):
                    lines.append(f"  FOREIGN KEY {local} -> {fk['referred_table']}.{remote}")

            date_cols = [c["name"] for c in columns
                         if str(c["type"]).upper().startswith(("DATE", "TIMESTAMP"))]
            for name in date_cols:
                lo, hi = conn.execute(
                    text(f"SELECT MIN({_quote(name)}), MAX({_quote(name)}) FROM {_quote(table)}")
                ).one()
                lines.append(f"  {name} range: {lo} to {hi}{_date_storage_note(lo, hi)}")
            lines.append("")

    return "\n".join(lines).strip()
