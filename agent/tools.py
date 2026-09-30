"""Safe execution of validated SQL against SQLite.

Defense-in-depth layers applied at execution time (independent of the
validator, so a validator bug alone cannot cause damage):

1. The engine opens the file read-only (``mode=ro``) with ``query_only=ON``.
2. An SQLite *authorizer* callback denies every action except reading data,
   SELECT, and calling functions. SQLite consults it while compiling the
   statement, so writes, ATTACH, PRAGMA, etc. are refused before running.
3. A *progress handler* aborts queries that run longer than the timeout.
4. At most ``max_rows`` rows are fetched; extra rows are reported as truncated.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Engine

logger = logging.getLogger(__name__)

# Opcodes SQLite runs between progress-handler calls (lower = finer timeout).
PROGRESS_HANDLER_STEPS = 10_000

# SQLITE_RECURSIVE is needed for WITH RECURSIVE (e.g. generating a month series).
_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
}
_DENIED_FUNCTIONS = {"load_extension", "readfile", "writefile", "edit"}


class QueryExecutionError(Exception):
    """A query failed. ``str(error)`` is safe to show to users and the LLM."""

    def __init__(self, message: str, kind: str = "database_error") -> None:
        super().__init__(message)
        self.kind = kind


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool
    elapsed_ms: float = 0.0  # wall time of execute + fetch, measured here

    def as_dict(self) -> dict[str, Any]:
        return {"columns": self.columns, "rows": self.rows, "truncated": self.truncated,
                "elapsed_ms": self.elapsed_ms}


def _authorizer(action: int, arg1: str | None, arg2: str | None, _db: str | None,
                _trigger: str | None) -> int:
    if action not in _ALLOWED_ACTIONS:
        return sqlite3.SQLITE_DENY
    # For SQLITE_FUNCTION, arg2 holds the function name.
    if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in _DENIED_FUNCTIONS:
        return sqlite3.SQLITE_DENY
    # Block reads of SQLite's internal catalog tables.
    if action == sqlite3.SQLITE_READ and (arg1 or "").lower().startswith("sqlite_"):
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def _sanitize_error(exc: Exception) -> str:
    """Keep the SQLite message (useful for self-correction) but nothing else."""
    message = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
    return message[:300]


def execute_readonly_query(engine: Engine, sql: str, max_rows: int,
                           timeout_seconds: float) -> QueryResult:
    """Execute a single read-only query and return at most ``max_rows`` rows."""
    deadline = time.monotonic() + timeout_seconds
    timed_out = False

    def progress_handler() -> int:
        nonlocal timed_out
        if time.monotonic() > deadline:
            timed_out = True
            return 1  # non-zero aborts the running statement
        return 0

    # Work at the DB-API level so we can install the sqlite3 hooks; the
    # SQLAlchemy engine still owns connection creation and pragmas.
    raw_conn = engine.raw_connection()
    dbapi_conn: sqlite3.Connection = raw_conn.driver_connection  # type: ignore[assignment]
    try:
        dbapi_conn.set_authorizer(_authorizer)
        dbapi_conn.set_progress_handler(progress_handler, PROGRESS_HANDLER_STEPS)
        cursor = dbapi_conn.cursor()
        started = time.perf_counter()
        try:
            # sqlite3's execute() refuses multiple statements on its own as well.
            cursor.execute(sql)
            columns = [d[0] for d in cursor.description or []]
            fetched = cursor.fetchmany(max_rows + 1)
            elapsed_ms = (time.perf_counter() - started) * 1000
        finally:
            cursor.close()
    except (sqlite3.Error, sqlite3.Warning) as exc:
        if timed_out:
            raise QueryExecutionError(
                f"Query exceeded the {timeout_seconds:g}s time limit and was stopped.",
                kind="timeout",
            ) from exc
        if "not authorized" in str(exc).lower():
            raise QueryExecutionError(
                "The database refused this operation (read-only access only).",
                kind="not_authorized",
            ) from exc
        logger.info("SQL execution error: %s", exc)
        raise QueryExecutionError(_sanitize_error(exc)) from exc
    finally:
        try:
            dbapi_conn.set_progress_handler(None, 0)
            dbapi_conn.set_authorizer(None)
        except Exception:  # connection may already be broken; it is discarded anyway
            pass
        raw_conn.close()

    truncated = len(fetched) > max_rows
    rows = [list(r) for r in fetched[:max_rows]]
    return QueryResult(columns=columns, rows=rows, truncated=truncated, elapsed_ms=elapsed_ms)
