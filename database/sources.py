"""Databases the app can query: the bundled sample and user-uploaded SQLite files.

Connection lifecycle of an upload (``open_uploaded_database``):

1. Cheap checks on the raw bytes: extension, size, SQLite magic header.
2. The bytes are written to a *private copy* under ``upload_dir`` with a random
   name (the user's filename is never used as a path), and the copy is then
   made read-only on disk. Streamlit's in-memory upload object can disappear
   between reruns; this copy cannot.
3. The copy is opened read-only + hardened + ``immutable=1``, integrity-checked
   and inspected (table discovery, row counts, schema text) under one overall
   time budget.
4. Only if every step succeeds is a ``ConnectedDatabase`` returned. On any
   failure the engine is disposed and the copy deleted, so a failed attempt
   leaves nothing behind and never touches the sample database.

Why ``immutable=1`` is safe here: QueryPilot writes the copy exactly once,
before any connection is opened, then sets the file read-only. Nothing in the
app writes to it afterwards (all connections are ``mode=ro`` + ``query_only``),
and it is only ever deleted, never rewritten, when the database is closed.

Cleanup: callers must call ``ConnectedDatabase.close()`` explicitly when
switching, disconnecting, or abandoning a database. A ``weakref.finalize``
fallback (garbage collection, interpreter exit) and the stale-copy sweep only
cover sessions that vanish without closing (e.g. a closed browser tab).

``PRAGMA quick_check`` only proves the file is structurally intact, not that
its contents are trustworthy. Uploaded schemas and data are still treated as
hostile by every later layer (validator, read-only connection, authorizer,
timeouts, row cap).
"""

from __future__ import annotations

import logging
import re
import stat
import time
import uuid
import weakref
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import Engine

from database.connection import (
    SQLITE_HEADER,
    DatabaseConnectionError,
    InvalidDatabaseError,
    describe_sqlite_failure,
    dispose_engine,
    get_engine,
    get_uploaded_engine,
    verify_connection,
)
from database.schema import get_schema_description, get_table_names, get_table_row_counts

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".db", ".sqlite", ".sqlite3")
# Limits that keep inspection fast and the schema prompt a sensible size.
MAX_TABLES = 100
MAX_SCHEMA_CHARS = 100_000
# Total budget for integrity check + discovery + counts + schema sampling.
INSPECTION_TIMEOUT_SECONDS = 15.0
# Private copies left behind by sessions that vanished are removed after this long.
STALE_UPLOAD_SECONDS = 24 * 3600
_STORED_SUFFIX = ".sqlite"


@dataclass(frozen=True)
class TableSummary:
    name: str
    row_count: int


def _delete_copy(path: Path) -> None:
    """Delete a private copy (it is read-only on disk, so allow writing first)."""
    try:
        if path.exists():
            path.chmod(stat.S_IREAD | stat.S_IWRITE)
            path.unlink()
    except OSError:  # e.g. still open on Windows; the stale sweep retries later
        logger.warning("Could not delete uploaded database copy %s", path.name)


def _release(engine: Engine | None, stored_path: Path | None) -> None:
    dispose_engine(engine)
    if stored_path is not None:
        _delete_copy(stored_path)


@dataclass(eq=False)
class ConnectedDatabase:
    """One queryable database: engine, schema text, and table summary.

    ``schema`` and ``tables`` were produced from ``engine`` itself, so the
    prompt, the validator's allowlist, and the executor always refer to the
    same database. ``connection_id`` is unique per connection and is used to
    tag results so they are never shown for a different database.
    """

    display_name: str
    engine: Engine
    schema: str
    tables: list[TableSummary]
    is_sample: bool = False
    stored_path: Path | None = None  # private copy of an upload, deleted on close
    connection_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    _finalizer: weakref.finalize = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Fallback only; close() is the primary cleanup path. The finalizer
        # must not reference ``self``, or it would never fire.
        self._finalizer = weakref.finalize(self, _release, self.engine, self.stored_path)

    @property
    def is_open(self) -> bool:
        return self._finalizer.alive

    @property
    def table_names(self) -> list[str]:
        return [t.name for t in self.tables]

    def close(self) -> None:
        """Dispose the engine and delete the private copy (idempotent)."""
        self._finalizer()


def sanitize_display_name(filename: str | None) -> str:
    """User-supplied filename -> short label that is safe to render.

    Only the base name is kept (no client paths), and characters that could
    act as markdown or control sequences are replaced.
    """
    name = re.split(r"[\\/]", filename or "")[-1]
    name = re.sub(r"[^\w.\- ()]", "_", name).strip(" .")
    if len(name) > 80:
        name = name[:77] + "..."
    return name or "uploaded database"


def _check_upload_bytes(data: bytes, filename: str | None, max_bytes: int) -> None:
    if not (filename or "").lower().endswith(SUPPORTED_EXTENSIONS):
        raise InvalidDatabaseError(
            "Unsupported file type. Upload a SQLite database ending in "
            + ", ".join(SUPPORTED_EXTENSIONS) + "."
        )
    if not data:
        raise InvalidDatabaseError("The uploaded file is empty.")
    if len(data) > max_bytes:
        raise InvalidDatabaseError(
            f"The file is too large (limit {max_bytes / 1_048_576:.0f} MB)."
        )
    if not data.startswith(SQLITE_HEADER):
        raise InvalidDatabaseError(
            "The file is not a SQLite database (or it is encrypted). "
            "Renaming another file type to .db does not make it a database."
        )


def cleanup_stale_uploads(upload_dir: Path, max_age_seconds: float = STALE_UPLOAD_SECONDS) -> None:
    """Best-effort removal of private copies orphaned by vanished sessions."""
    if not upload_dir.is_dir():
        return
    cutoff = time.time() - max_age_seconds
    for path in upload_dir.glob(f"*{_STORED_SUFFIX}"):
        try:
            if path.stat().st_mtime < cutoff:
                _delete_copy(path)
        except OSError:
            pass


def _inspect(engine: Engine, timeout_seconds: float) -> tuple[list[TableSummary], str]:
    """Validate and describe ``engine``'s database, or raise a user-safe error.

    Every step runs under what is left of one overall deadline, and the table
    count is checked before any per-table query runs.
    """
    deadline = time.monotonic() + timeout_seconds

    def remaining() -> float:
        return max(deadline - time.monotonic(), 0.001)

    verify_connection(engine, remaining())
    names = get_table_names(engine, remaining())
    if not names:
        raise InvalidDatabaseError(
            "The database has no tables, so there is nothing to ask questions about."
        )
    if len(names) > MAX_TABLES:
        raise InvalidDatabaseError(
            f"The database has {len(names)} tables; at most {MAX_TABLES} are supported."
        )
    counts = get_table_row_counts(engine, remaining())
    if not any(counts.values()):
        raise InvalidDatabaseError(
            "The database has tables but they contain no rows, so there is no data to analyse."
        )
    schema = get_schema_description(engine, remaining())
    if len(schema) > MAX_SCHEMA_CHARS:
        raise InvalidDatabaseError(
            "The database schema is too large to describe to the language model."
        )
    return [TableSummary(name, n) for name, n in counts.items()], schema


def open_uploaded_database(data: bytes, filename: str | None, upload_dir: Path, *,
                           max_bytes: int,
                           timeout_seconds: float = INSPECTION_TIMEOUT_SECONDS,
                           ) -> ConnectedDatabase:
    """Store, validate and inspect an uploaded SQLite file (see module docstring).

    Raises ``DatabaseConnectionError`` with a user-safe message on any failure,
    after releasing everything this attempt created.
    """
    _check_upload_bytes(data, filename, max_bytes)

    upload_dir.mkdir(parents=True, exist_ok=True)
    cleanup_stale_uploads(upload_dir)
    stored_path = upload_dir / f"{uuid.uuid4().hex}{_STORED_SUFFIX}"
    try:
        with stored_path.open("xb") as fh:  # "x": never overwrite an existing file
            fh.write(data)
        stored_path.chmod(stat.S_IREAD)  # the copy must not change while connected
    except OSError as exc:
        _delete_copy(stored_path)
        logger.exception("Could not store uploaded database")
        raise DatabaseConnectionError("The file could not be stored on the server.") from exc

    engine: Engine | None = None
    try:
        engine = get_uploaded_engine(stored_path)
        tables, schema = _inspect(engine, timeout_seconds)
    except BaseException as exc:
        # Never leave a half-connected database behind.
        _release(engine, stored_path)
        if isinstance(exc, DatabaseConnectionError) or not isinstance(exc, Exception):
            raise
        logger.exception("Unexpected failure while connecting an uploaded database")
        raise InvalidDatabaseError(describe_sqlite_failure(exc)) from exc
    return ConnectedDatabase(
        display_name=sanitize_display_name(filename),
        engine=engine,
        schema=schema,
        tables=tables,
        stored_path=stored_path,
    )


def open_sample_database(db_path: Path, timeout_seconds: float | None = None) -> ConnectedDatabase:
    """The bundled sample database (read-only, not immutable: init_db may rebuild it)."""
    engine = get_engine(db_path, read_only=True)
    try:
        counts = get_table_row_counts(engine, timeout_seconds)
        schema = get_schema_description(engine, timeout_seconds)
    except BaseException:
        dispose_engine(engine)
        raise
    return ConnectedDatabase(
        display_name="Sample sales database",
        engine=engine,
        schema=schema,
        tables=[TableSummary(name, n) for name, n in counts.items()],
        is_sample=True,
    )
