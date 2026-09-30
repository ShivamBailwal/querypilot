"""SQLAlchemy engine factories for the SQLite database.

Two kinds of engines exist:

* **Read-only** (used by the agent): the file is opened with the SQLite URI
  flag ``mode=ro`` so the OS-level handle cannot write, and ``PRAGMA query_only``
  is switched on as a second guard. This is one layer of the defense-in-depth
  strategy; SQL validation and the execution-time authorizer are the others.
  Because uploaded files are attacker-controlled, read-only connections also
  apply SQLite's recommended settings for untrusted database files
  (https://sqlite.org/security.html): ``trusted_schema=OFF``, defensive mode,
  ``cell_size_check=ON`` and no memory-mapped I/O.
* **Writable** (used only by ``database/init_db.py``) for creating and seeding.

Both enable ``PRAGMA foreign_keys=ON`` because SQLite leaves FK enforcement off
by default on every new connection.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Connection, Engine, create_engine, event
from sqlalchemy.exc import DBAPIError
from sqlalchemy.pool import NullPool

logger = logging.getLogger(__name__)

# Every SQLite database file starts with this 16-byte magic string.
SQLITE_HEADER = b"SQLite format 3\x00"
SQLITE_HEADER_SIZE = 100

# Opcodes SQLite runs between progress-handler calls in ``time_limit``.
_PROGRESS_STEPS = 10_000


class DatabaseNotFoundError(FileNotFoundError):
    """Raised when the agent is pointed at a database file that does not exist."""


class DatabaseConnectionError(Exception):
    """A database could not be connected. ``str(error)`` is safe to show users:
    it never contains file paths, SQL, or tracebacks."""


class InvalidDatabaseError(DatabaseConnectionError):
    """The file is not a usable SQLite database (wrong type, corrupted, empty...)."""


class DatabaseTimeoutError(DatabaseConnectionError):
    """Validating or inspecting the database exceeded its time budget."""


# SQLite silently ignores PRAGMAs it does not know, so hardening options are
# gated on the version that introduced them (and verified after being set).
_TRUSTED_SCHEMA_MIN_VERSION = (3, 31, 0)   # PRAGMA trusted_schema
_DEFENSIVE_MIN_VERSION = (3, 26, 0)        # SQLITE_DBCONFIG_DEFENSIVE


def untrusted_file_hardening_supported() -> bool:
    """Whether this Python/SQLite build supports every untrusted-file setting."""
    return (
        sqlite3.sqlite_version_info >= _TRUSTED_SCHEMA_MIN_VERSION
        and sqlite3.sqlite_version_info >= _DEFENSIVE_MIN_VERSION
        and hasattr(sqlite3.Connection, "setconfig")  # Python 3.12+
        and hasattr(sqlite3, "SQLITE_DBCONFIG_DEFENSIVE")
    )


def _enable_pragmas(dbapi_connection: sqlite3.Connection, read_only: bool,
                    untrusted: bool = False) -> None:
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys = ON")
        if read_only:
            cursor.execute("PRAGMA query_only = ON")
            # Detect malformed b-tree cells early instead of reading garbage.
            cursor.execute("PRAGMA cell_size_check = ON")
            cursor.execute("PRAGMA mmap_size = 0")
            if sqlite3.sqlite_version_info >= _TRUSTED_SCHEMA_MIN_VERSION:
                # Don't run functions/virtual tables with side effects from
                # inside the file's own schema (views, triggers, defaults, CHECKs).
                cursor.execute("PRAGMA trusted_schema = OFF")
            if untrusted_file_hardening_supported():
                dbapi_connection.setconfig(sqlite3.SQLITE_DBCONFIG_DEFENSIVE, True)
            if untrusted:
                # Refuse to hand out a connection whose guards did not apply.
                query_only = cursor.execute("PRAGMA query_only").fetchone()[0]
                trusted = cursor.execute("PRAGMA trusted_schema").fetchone()[0]
                defensive = dbapi_connection.getconfig(sqlite3.SQLITE_DBCONFIG_DEFENSIVE)
                if query_only != 1 or trusted != 0 or not defensive:
                    raise sqlite3.OperationalError("read-only hardening could not be applied")
    finally:
        cursor.close()


def get_engine(db_path: Path | str, read_only: bool = True, *,
               immutable: bool = False, untrusted: bool = False) -> Engine:
    """Create an engine for ``db_path``.

    A read-only engine requires the file to exist already (``mode=ro`` cannot
    create files), which also prevents silently querying an empty database.

    ``immutable=True`` adds the ``immutable=1`` URI flag: SQLite then skips
    locking and change detection and never creates ``-wal``/``-shm`` side files.
    This is only correct if *nothing* modifies the file while it is open; if
    the file did change, SQLite could return wrong results or report
    corruption. Use it only for private upload copies that QueryPilot wrote
    once and then made read-only (see ``database/sources.py``), never for the
    sample database, which ``init_db`` may rebuild while the app runs.

    ``untrusted=True`` makes every connection verify that ``query_only``,
    ``trusted_schema=OFF`` and defensive mode really took effect.
    """
    if untrusted and not read_only:
        raise ValueError("Untrusted databases can only be opened read-only.")
    path = Path(db_path).resolve()

    if read_only:
        if not path.exists():
            raise DatabaseNotFoundError(
                "Database file not found. Run `python -m database.init_db` first."
            )
        uri = f"{path.as_uri()}?mode=ro"
        if immutable:
            uri += "&immutable=1"

        def creator() -> sqlite3.Connection:
            # check_same_thread=False: Streamlit may serve requests on different
            # threads; NullPool below means every checkout gets a fresh
            # connection, so connections are never actually shared.
            return sqlite3.connect(uri, uri=True, check_same_thread=False)

        engine = create_engine("sqlite://", creator=creator, poolclass=NullPool)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(f"sqlite:///{path.as_posix()}", poolclass=NullPool)

    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_connection, _connection_record) -> None:  # type: ignore[no-untyped-def]
        _enable_pragmas(dbapi_connection, read_only, untrusted)

    return engine


# --------------------------------------------------------- uploaded databases
def validate_sqlite_file(db_path: Path | str) -> Path:
    """Cheap checks before SQLite touches the file; returns the resolved path.

    The extension is never trusted: the file must start with the SQLite magic
    header. Deeper corruption is caught later by ``verify_connection``.
    """
    path = Path(db_path).resolve()
    if not path.is_file():
        raise DatabaseNotFoundError("Database file not found.")
    size = path.stat().st_size
    if size == 0:
        raise InvalidDatabaseError("The file is empty.")
    with path.open("rb") as fh:
        header = fh.read(SQLITE_HEADER_SIZE)
    if not header.startswith(SQLITE_HEADER):
        raise InvalidDatabaseError(
            "The file is not a SQLite database (or it is encrypted)."
        )
    if size < SQLITE_HEADER_SIZE:
        raise InvalidDatabaseError("The database file is truncated or corrupted.")
    return path


def get_uploaded_engine(db_path: Path | str) -> Engine:
    """Read-only, hardened, immutable engine for a *private copy* of an upload.

    Never creates, initialises, or modifies the file. The caller guarantees
    the copy is not modified while connected (``immutable=1`` relies on it).
    """
    if not untrusted_file_hardening_supported():
        raise DatabaseConnectionError(
            "This server's SQLite/Python version is too old to open uploaded databases "
            f"safely (needs SQLite {'.'.join(map(str, _TRUSTED_SCHEMA_MIN_VERSION))}+ "
            "and Python 3.12+)."
        )
    return get_engine(validate_sqlite_file(db_path), read_only=True,
                      immutable=True, untrusted=True)


@contextmanager
def time_limit(connection: Connection, seconds: float | None) -> Iterator[None]:
    """Abort statements on ``connection`` that run past ``seconds`` in total.

    Raises ``DatabaseTimeoutError`` if the budget was hit. ``None`` disables it.
    """
    if not seconds:
        yield
        return
    dbapi_conn: sqlite3.Connection = connection.connection.driver_connection  # type: ignore[assignment]
    # perf_counter: monotonic() only ticks every ~16 ms on Windows.
    deadline = time.perf_counter() + seconds
    timed_out = False

    def handler() -> int:
        nonlocal timed_out
        if time.perf_counter() > deadline:
            timed_out = True
            return 1  # non-zero aborts the running statement
        return 0

    dbapi_conn.set_progress_handler(handler, _PROGRESS_STEPS)
    try:
        yield
    except Exception as exc:
        if timed_out:
            raise DatabaseTimeoutError(
                f"Inspecting the database took longer than {seconds:g}s. "
                "The file may be too large or complex."
            ) from exc
        raise
    finally:
        try:
            dbapi_conn.set_progress_handler(None, 0)
        except Exception:  # connection may already be broken; it is discarded anyway
            pass


def describe_sqlite_failure(exc: BaseException) -> str:
    """Map a low-level SQLite failure to a short, user-safe message."""
    text = str(exc).lower()
    if "not a database" in text or "encrypted" in text:
        return "The file is not a valid SQLite database (or it is encrypted)."
    if "malformed" in text or "corrupt" in text:
        return "The database file is corrupted."
    if "unable to open" in text:
        return "The database file could not be opened."
    return "The database file could not be read. It may be corrupted or unsupported."


def verify_connection(engine: Engine, timeout_seconds: float | None = 10.0) -> None:
    """Open a connection and run ``PRAGMA quick_check``; raise if unusable.

    Uses the engine's own (read-only) connections, so nothing is written.
    """
    try:
        with engine.connect() as conn, time_limit(conn, timeout_seconds):
            conn.exec_driver_sql("SELECT 1").scalar_one()
            problems = conn.exec_driver_sql("PRAGMA quick_check").scalars().all()
    except DatabaseConnectionError:
        raise
    except (DBAPIError, sqlite3.Error) as exc:
        logger.info("Database verification failed: %s", exc)
        raise InvalidDatabaseError(describe_sqlite_failure(exc)) from exc
    if problems != ["ok"]:
        logger.info("quick_check reported problems: %s", problems[:5])
        raise InvalidDatabaseError("The database file is corrupted (integrity check failed).")


def dispose_engine(engine: Engine | None) -> None:
    """Release an engine's connections; safe to call more than once."""
    if engine is None:
        return
    try:
        engine.dispose()
    except Exception:  # disposal must never break a disconnect/switch
        logger.exception("Error while disposing database engine")
