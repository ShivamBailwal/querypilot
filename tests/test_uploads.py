"""Uploaded SQLite databases: validation, lifecycle, schema, agent integration
and SQL safety. Offline: temporary databases and a fake LLM only."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

import database.connection as connection
import database.sources as sources
from agent import prompts
from agent.graph import build_graph, run_question
from agent.tools import QueryExecutionError, execute_readonly_query
from database.connection import (
    SQLITE_HEADER,
    DatabaseConnectionError,
    DatabaseNotFoundError,
    DatabaseTimeoutError,
    InvalidDatabaseError,
    dispose_engine,
    get_engine,
    get_uploaded_engine,
    time_limit,
    validate_sqlite_file,
    verify_connection,
)
from database.schema import get_schema_description, get_table_names
from database.sources import (
    ConnectedDatabase,
    cleanup_stale_uploads,
    open_sample_database,
    open_uploaded_database,
    sanitize_display_name,
)
from tests.conftest import RecordingLLMService, make_sqlite_db, sql

MAX_BYTES = 5 * 1024 * 1024
SAMPLE_TABLES = {"customers", "products", "orders", "order_items"}


@pytest.fixture()
def upload_dir(tmp_path: Path) -> Path:
    return tmp_path / "uploads"


@pytest.fixture()
def library(library_db_bytes: bytes, upload_dir: Path):
    db = open_uploaded_database(library_db_bytes, "library.db", upload_dir, max_bytes=MAX_BYTES)
    yield db
    db.close()


def _graph_for(db: ConnectedDatabase, llm, **kwargs):
    """Mirror of app.make_graph: engine, schema, allowlist and notes from one database."""
    return build_graph(llm, db.engine, database_schema=db.schema,
                       allowed_tables=db.table_names,
                       schema_notes=prompts.SAMPLE_DATABASE_NOTES if db.is_sample else "",
                       **kwargs)


def _stored_files(upload_dir: Path) -> list[Path]:
    return sorted(upload_dir.glob("*")) if upload_dir.exists() else []


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------ valid uploads
def test_valid_upload_connects(library: ConnectedDatabase, upload_dir: Path,
                               library_db_bytes: bytes) -> None:
    assert library.is_open and not library.is_sample
    assert library.display_name == "library.db"
    assert {t.name: t.row_count for t in library.tables} == {"authors": 3, "books": 5}
    # A private, randomly named copy inside the upload dir, byte-identical to the upload.
    assert library.stored_path is not None and library.stored_path.parent == upload_dir
    assert library.stored_path.name != "library.db"
    assert library.stored_path.read_bytes() == library_db_bytes
    assert not os.access(library.stored_path, os.W_OK)  # read-only on disk while connected


@pytest.mark.parametrize("name", ["data.db", "DATA.SQLITE", "x.sqlite3"])
def test_supported_extensions(name: str, library_db_bytes: bytes, upload_dir: Path) -> None:
    db = open_uploaded_database(library_db_bytes, name, upload_dir, max_bytes=MAX_BYTES)
    db.close()


def test_uploaded_engine_applies_read_only_hardening(library: ConnectedDatabase) -> None:
    with library.engine.connect() as conn:
        assert conn.exec_driver_sql("PRAGMA query_only").scalar() == 1
        assert conn.exec_driver_sql("PRAGMA trusted_schema").scalar() == 0
        assert conn.exec_driver_sql("PRAGMA cell_size_check").scalar() == 1
        dbapi = conn.connection.driver_connection
        assert dbapi.getconfig(sqlite3.SQLITE_DBCONFIG_DEFENSIVE)


def test_upload_refused_when_sqlite_cannot_be_hardened(monkeypatch, library_db_bytes,
                                                      upload_dir) -> None:
    monkeypatch.setattr(connection, "untrusted_file_hardening_supported", lambda: False)
    with pytest.raises(DatabaseConnectionError, match="too old"):
        open_uploaded_database(library_db_bytes, "library.db", upload_dir, max_bytes=MAX_BYTES)
    assert _stored_files(upload_dir) == []


# ---------------------------------------------------------- invalid uploads
def _header_plus_garbage(_: bytes) -> bytes:
    return SQLITE_HEADER + bytes(range(256)) * 48


def _truncated(valid: bytes) -> bytes:
    return valid[:4096]  # only page 1 survives; tables point at missing pages


def _garbage_page(valid: bytes) -> bytes:
    return valid[:4096] + bytes([0xAB]) * 4096 + valid[8192:]


@pytest.mark.parametrize(
    ("make_bytes", "filename", "message"),
    [
        (lambda _: b"", "empty.db", "empty"),
        (lambda _: b"id,name\n1,Ada\n" * 20, "people.db", "not a SQLite database"),
        (lambda _: b"%PDF-1.7 " + b"x" * 500, "report.sqlite", "not a SQLite database"),
        (lambda v: v, "library.csv", "Unsupported file type"),
        (lambda v: v, "library", "Unsupported file type"),
        (_header_plus_garbage, "garbage.db", ""),
        (_truncated, "truncated.db", "corrupt"),
        (_garbage_page, "damaged.db", "corrupt"),
    ],
    ids=["empty", "csv-renamed", "pdf-renamed", "wrong-ext", "no-ext", "garbage",
         "truncated", "damaged-page"],
)
def test_invalid_uploads_are_rejected_and_leave_nothing_behind(
        make_bytes, filename: str, message: str, library_db_bytes: bytes,
        upload_dir: Path) -> None:
    with pytest.raises(InvalidDatabaseError) as info:
        open_uploaded_database(make_bytes(library_db_bytes), filename, upload_dir,
                               max_bytes=MAX_BYTES)
    assert message.lower() in str(info.value).lower()
    assert str(upload_dir) not in str(info.value)  # no server paths in user messages
    assert "Traceback" not in str(info.value)
    assert _stored_files(upload_dir) == []


def test_too_large_upload_is_rejected(library_db_bytes: bytes, upload_dir: Path) -> None:
    with pytest.raises(InvalidDatabaseError, match="too large"):
        open_uploaded_database(library_db_bytes, "big.db", upload_dir,
                               max_bytes=len(library_db_bytes) - 1)


def test_database_without_tables_is_rejected(tmp_path: Path, upload_dir: Path) -> None:
    data = make_sqlite_db(tmp_path / "e.db", "CREATE TABLE t (x); DROP TABLE t;")
    assert data.startswith(SQLITE_HEADER)  # a real, valid, but empty SQLite file
    with pytest.raises(InvalidDatabaseError, match="no tables"):
        open_uploaded_database(data, "empty.db", upload_dir, max_bytes=MAX_BYTES)
    assert _stored_files(upload_dir) == []


def test_database_with_only_empty_tables_is_rejected(tmp_path: Path, upload_dir: Path) -> None:
    data = make_sqlite_db(tmp_path / "e.db", "CREATE TABLE a (x); CREATE TABLE b (y);")
    with pytest.raises(InvalidDatabaseError, match="no rows"):
        open_uploaded_database(data, "empty.db", upload_dir, max_bytes=MAX_BYTES)


def test_too_many_tables_is_rejected_before_inspecting_them(monkeypatch, library_db_bytes,
                                                           upload_dir) -> None:
    monkeypatch.setattr(sources, "MAX_TABLES", 1)
    counted = []
    monkeypatch.setattr(sources, "get_table_row_counts", lambda *a: counted.append(a) or {})
    with pytest.raises(InvalidDatabaseError, match="at most 1"):
        open_uploaded_database(library_db_bytes, "library.db", upload_dir, max_bytes=MAX_BYTES)
    assert counted == []
    assert _stored_files(upload_dir) == []


def test_nonexistent_path_does_not_create_a_database(tmp_path: Path) -> None:
    missing = tmp_path / "nope.db"
    with pytest.raises(DatabaseNotFoundError):
        get_uploaded_engine(missing)
    with pytest.raises(DatabaseNotFoundError):
        validate_sqlite_file(missing)
    with pytest.raises(DatabaseNotFoundError):
        get_engine(missing, read_only=True, immutable=True)
    assert not missing.exists()


def test_verify_connection_rejects_non_database(tmp_path: Path) -> None:
    path = tmp_path / "fake.db"
    path.write_bytes(_header_plus_garbage(b""))
    engine = get_engine(path, read_only=True)
    try:
        with pytest.raises(InvalidDatabaseError):
            verify_connection(engine)
    finally:
        dispose_engine(engine)


def test_untrusted_engine_cannot_be_writable(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        get_engine(tmp_path / "x.db", read_only=False, untrusted=True)


# ----------------------------------------------------- inspection time budget
@pytest.fixture()
def big_db_bytes(tmp_path: Path) -> bytes:
    path = tmp_path / "big_src.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, label TEXT)")
    conn.executemany("INSERT INTO events (label) VALUES (?)",
                     ((f"label-{i % 5000}",) for i in range(200_000)))
    conn.commit()
    conn.close()
    return path.read_bytes()


class _JumpingClock:
    """Each reading is 1000s after the previous one, so any budget below that
    is exceeded at the first progress-handler check. Makes timeouts deterministic."""

    def __init__(self) -> None:
        self.now = 0.0

    def perf_counter(self) -> float:
        self.now += 1000.0
        return self.now


@pytest.fixture()
def jumping_clock(monkeypatch) -> None:
    monkeypatch.setattr(connection, "time", _JumpingClock())


@pytest.fixture()
def big_engine(big_db_bytes: bytes, tmp_path: Path):
    path = tmp_path / "big.db"
    path.write_bytes(big_db_bytes)
    engine = get_uploaded_engine(path)
    yield engine
    dispose_engine(engine)


@pytest.mark.parametrize(
    "step",
    [
        lambda e: get_schema_description(e, timeout_seconds=1),   # reflection + DISTINCT sampling
        lambda e: verify_connection(e, timeout_seconds=1),         # quick_check
    ],
    ids=["schema", "quick_check"],
)
def test_inspection_steps_respect_time_budget(big_engine, jumping_clock, step) -> None:
    with pytest.raises(DatabaseTimeoutError):
        step(big_engine)


def test_table_discovery_respects_time_budget(tmp_path: Path, monkeypatch) -> None:
    # A hostile file with thousands of tables must not make discovery unbounded.
    script = "BEGIN;" + "".join(f"CREATE TABLE t{i} (x);" for i in range(3000)) + "COMMIT;"
    path = tmp_path / "many.db"
    make_sqlite_db(path, script)
    engine = get_uploaded_engine(path)
    try:
        assert len(get_table_names(engine, timeout_seconds=30)) == 3000
        monkeypatch.setattr(connection, "time", _JumpingClock())
        with pytest.raises(DatabaseTimeoutError):
            get_table_names(engine, timeout_seconds=1)
    finally:
        dispose_engine(engine)


def test_time_limit_stops_runaway_statement_with_real_clock(library: ConnectedDatabase) -> None:
    endless = "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n) SELECT COUNT(*) FROM n"
    start = time.monotonic()
    with library.engine.connect() as conn, pytest.raises(DatabaseTimeoutError):
        with time_limit(conn, 0.3):
            conn.exec_driver_sql(endless).fetchall()
    assert time.monotonic() - start < 5


def test_without_budget_pressure_inspection_completes(big_engine) -> None:
    assert "Table events (200000 rows)" in get_schema_description(big_engine, timeout_seconds=30)


def test_inspection_timeout_fails_without_partial_connection(big_db_bytes: bytes,
                                                             upload_dir: Path,
                                                             monkeypatch) -> None:
    monkeypatch.setattr(connection, "time", _JumpingClock())
    with pytest.raises(DatabaseTimeoutError) as info:
        open_uploaded_database(big_db_bytes, "big.db", upload_dir, max_bytes=50 * 2**20)
    assert "took longer" in str(info.value)
    assert _stored_files(upload_dir) == []
    monkeypatch.undo()
    # The same file connects fine with a normal budget.
    db = open_uploaded_database(big_db_bytes, "big.db", upload_dir, max_bytes=50 * 2**20)
    assert db.tables[0].row_count == 200_000
    db.close()


# ------------------------------------------------------------- lifecycle
def test_close_disposes_engine_and_deletes_private_copy(library: ConnectedDatabase) -> None:
    stored = library.stored_path
    assert stored is not None and stored.exists()
    library.close()
    assert not library.is_open
    assert not stored.exists()
    library.close()  # idempotent


def test_dispose_engine_is_safe_to_repeat(library: ConnectedDatabase) -> None:
    dispose_engine(library.engine)
    dispose_engine(library.engine)
    dispose_engine(None)


def test_garbage_collection_is_only_a_fallback(library_db_bytes: bytes, upload_dir: Path) -> None:
    import gc

    db = open_uploaded_database(library_db_bytes, "library.db", upload_dir, max_bytes=MAX_BYTES)
    stored = db.stored_path
    del db
    gc.collect()
    assert stored is not None and not stored.exists()


def test_closing_sample_never_deletes_the_sample_file(seeded_db_path: Path) -> None:
    sample = open_sample_database(seeded_db_path)
    assert sample.is_sample and sample.stored_path is None
    assert {t.name for t in sample.tables} == SAMPLE_TABLES
    sample.close()
    assert seeded_db_path.exists()


def test_stale_cleanup_removes_only_old_copies(library_db_bytes: bytes, upload_dir: Path) -> None:
    fresh = open_uploaded_database(library_db_bytes, "a.db", upload_dir, max_bytes=MAX_BYTES)
    old = open_uploaded_database(library_db_bytes, "b.db", upload_dir, max_bytes=MAX_BYTES)
    assert fresh.stored_path and old.stored_path
    week_ago = time.time() - 7 * 24 * 3600
    os.utime(old.stored_path, (week_ago, week_ago))
    cleanup_stale_uploads(upload_dir)
    assert fresh.stored_path.exists() and not old.stored_path.exists()
    fresh.close()
    old.close()


def test_each_connection_has_a_unique_id(library_db_bytes: bytes, upload_dir: Path) -> None:
    a = open_uploaded_database(library_db_bytes, "a.db", upload_dir, max_bytes=MAX_BYTES)
    b = open_uploaded_database(library_db_bytes, "a.db", upload_dir, max_bytes=MAX_BYTES)
    assert a.connection_id != b.connection_id
    assert a.stored_path != b.stored_path
    a.close()
    b.close()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("C:\\Users\\me\\Secret\\shop.db", "shop.db"),
     ("../../etc/passwd.db", "passwd.db"),
     ("**bold** [link] `code` <b>.db", "__bold__ _link_ _code_ _b_.db"),
     ("see http://evil.example/x.db", "x.db"),
     ("", "uploaded database")],
)
def test_display_names_are_sanitized(raw: str, expected: str) -> None:
    assert sanitize_display_name(raw) == expected


# ---------------------------------------------------- uploads stay unmodified
def test_uploaded_database_is_never_modified(library: ConnectedDatabase, upload_dir: Path) -> None:
    stored = library.stored_path
    assert stored is not None
    before = _sha256(stored)

    llm = RecordingLLMService([sql("SELECT COUNT(*) AS n FROM books")])
    assert run_question(_graph_for(library, llm), "How many books?")["status"] == "success"
    for attack in ["DELETE FROM books", "UPDATE books SET copies_sold = 0",
                   "DROP TABLE authors", "CREATE TABLE x (y)"]:
        with pytest.raises(QueryExecutionError):
            execute_readonly_query(library.engine, attack, max_rows=5, timeout_seconds=2)
    with pytest.raises(DBAPIError):  # even a raw engine connection cannot write
        with library.engine.begin() as conn:
            conn.execute(text("DELETE FROM books"))

    assert _sha256(stored) == before
    # immutable=1: no journal / WAL / shared-memory side files are created.
    assert [p.name for p in _stored_files(upload_dir)] == [stored.name]


# ------------------------------------------------------------------ schema
def test_schema_describes_the_custom_database(library: ConnectedDatabase) -> None:
    schema = library.schema
    assert "Table authors (3 rows)" in schema and "Table books (5 rows)" in schema
    assert "full_name TEXT NOT NULL" in schema
    assert "FOREIGN KEY author_id -> authors.author_id" in schema
    assert "values: 'SE', 'UK', 'US'" in schema
    assert "published_on range: 2020-11-30 to 2024-05-21 (stored as 'YYYY-MM-DD' text)" in schema
    assert not any(f"Table {t} " in schema for t in SAMPLE_TABLES)


def test_date_format_is_detected_not_assumed(tmp_path: Path, upload_dir: Path) -> None:
    data = make_sqlite_db(tmp_path / "s.db", """
        CREATE TABLE readings (id INTEGER PRIMARY KEY, taken_at DATETIME, epoch_day DATE);
        INSERT INTO readings VALUES (1, '2024-01-01 08:00:00', 1704067200),
                                    (2, '2024-01-02 09:30:00', 1704153600);
    """)
    db = open_uploaded_database(data, "sensors.db", upload_dir, max_bytes=MAX_BYTES)
    try:
        assert "taken_at range: 2024-01-01 08:00:00 to 2024-01-02 09:30:00 " \
               "(stored as 'YYYY-MM-DD HH:MM:SS' text)" in db.schema
        assert "epoch_day range: 1704067200 to 1704153600 (stored as numbers, not text)" in db.schema
        assert "'YYYY-MM-DD' text)" not in db.schema
    finally:
        db.close()


def test_long_values_are_shortened_in_schema(tmp_path: Path, upload_dir: Path) -> None:
    data = make_sqlite_db(tmp_path / "s.db", f"""
        CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT);
        INSERT INTO notes VALUES (1, '{"x" * 5000}');
    """)
    db = open_uploaded_database(data, "notes.db", upload_dir, max_bytes=MAX_BYTES)
    try:
        assert "x" * 100 not in db.schema and "xxx..." in db.schema
    finally:
        db.close()


# --------------------------------------------------------- agent integration
def test_agent_uses_custom_schema_and_queries_custom_database(library: ConnectedDatabase,
                                                              ro_engine) -> None:
    llm = RecordingLLMService([sql(
        "SELECT a.full_name, SUM(b.copies_sold) AS total_sold FROM books b "
        "JOIN authors a ON a.author_id = b.author_id GROUP BY a.full_name ORDER BY 2 DESC"
    )], explanation="Bo Novelist sold the most.")
    state = run_question(_graph_for(library, llm), "Which author sold the most copies?")

    assert state["status"] == "success"
    assert state["query_results"]["rows"] == [["Bo Novelist", 7700], ["Ada Writer", 2000],
                                              ["Cy Poet", 150]]
    # The LLM saw exactly this database's schema, with no sample business rules.
    assert llm.schemas == [library.schema]
    assert "Business definitions" not in llm.schemas[0]
    assert "order_items" not in llm.schemas[0]
    # And the sample database was not what got queried.
    with ro_engine.connect() as conn:
        assert "books" not in {r[0] for r in conn.exec_driver_sql(
            "SELECT name FROM sqlite_master WHERE type='table'")}


def test_sample_tables_are_unknown_on_a_custom_database(library: ConnectedDatabase) -> None:
    llm = RecordingLLMService([sql("SELECT COUNT(*) FROM orders"),
                               sql("SELECT * FROM customers")])
    state = run_question(_graph_for(library, llm, max_retries=1), "How many orders?")
    assert state["status"] == "failed"
    assert state.get("query_results") is None  # never executed
    assert "Unknown or disallowed table: orders" in llm.correct_calls[0][1]


def test_unfamiliar_table_and_column_names_work(tmp_path: Path, upload_dir: Path) -> None:
    data = make_sqlite_db(tmp_path / "w.db", """
        CREATE TABLE "Sales Data" ("Region Name" TEXT, "Unit Price ($)" REAL, qty INTEGER);
        INSERT INTO "Sales Data" VALUES ('North', 2.5, 4), ('South', 10.0, 1), ('North', 1.0, 3);
    """)
    db = open_uploaded_database(data, "weird names.sqlite", upload_dir, max_bytes=MAX_BYTES)
    try:
        assert 'Table Sales Data (3 rows)' in db.schema
        llm = RecordingLLMService([sql(
            'SELECT "Region Name" AS region, ROUND(SUM("Unit Price ($)" * qty), 2) AS sales '
            'FROM "Sales Data" GROUP BY 1 ORDER BY 2 DESC'
        )])
        state = run_question(_graph_for(db, llm), "Sales by region?")
        assert state["status"] == "success", state.get("final_answer")
        assert state["query_results"]["rows"] == [["North", 13.0], ["South", 10.0]]
    finally:
        db.close()


def test_graphs_are_bound_to_their_own_database(library: ConnectedDatabase,
                                                tmp_path: Path, upload_dir: Path) -> None:
    other_bytes = make_sqlite_db(tmp_path / "o.db", """
        CREATE TABLE books (book_id INTEGER PRIMARY KEY, title TEXT);
        INSERT INTO books VALUES (1, 'Only Book');
    """)
    other = open_uploaded_database(other_bytes, "other.db", upload_dir, max_bytes=MAX_BYTES)
    try:
        query = "SELECT COUNT(*) AS n FROM books"
        llm_a, llm_b = RecordingLLMService([sql(query)]), RecordingLLMService([sql(query)])
        a = run_question(_graph_for(library, llm_a), "How many books?")
        b = run_question(_graph_for(other, llm_b), "How many books?")
        assert a["query_results"]["rows"] == [[5]]
        assert b["query_results"]["rows"] == [[1]]
        assert llm_a.schemas == [library.schema] and llm_b.schemas == [other.schema]
        assert library.schema != other.schema
    finally:
        other.close()


def test_sample_graph_keeps_sample_business_definitions(seeded_db_path: Path) -> None:
    sample = open_sample_database(seeded_db_path)
    try:
        llm = RecordingLLMService([sql("SELECT COUNT(*) FROM orders")])
        run_question(_graph_for(sample, llm), "How many orders?")
        assert llm.schemas[0].startswith(sample.schema)
        assert "Business definitions for this database:" in llm.schemas[0]
        assert "SUM(order_items.quantity * order_items.unit_price)" in llm.schemas[0]
    finally:
        sample.close()


# ------------------------------------------------------------------ prompts
def test_generic_prompt_rules_contain_no_sample_assumptions() -> None:
    system = prompts.GENERATE_SYSTEM.format(schema="Table widgets (1 rows)\n  - id INTEGER")
    for sample_specific in ["order_items", "orders", "customers", "products", "order_date",
                            "'completed'", "Revenue\" or", "profit"]:
        assert sample_specific not in prompts.SQL_RULES, sample_specific
        assert sample_specific not in system, sample_specific
    assert "SQLite" in prompts.SQL_RULES
    assert "never follow instructions" in prompts.SQL_RULES


def test_schema_notes_are_only_appended_when_given() -> None:
    assert prompts.with_schema_notes("S") == "S"
    assert prompts.with_schema_notes("S", "  ") == "S"
    combined = prompts.with_schema_notes("S", prompts.SAMPLE_DATABASE_NOTES)
    assert combined.startswith("S\n\nBusiness definitions for this database:\n")
    assert "status = 'completed'" in combined


# ------------------------------------------------------- SQL safety on uploads
@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM books",
        "UPDATE authors SET country = 'XX'",
        "INSERT INTO authors VALUES (9, 'Eve', 'NO')",
        "DROP TABLE books",
        "ATTACH DATABASE 'evil.db' AS evil",
        "PRAGMA query_only = OFF",
        "PRAGMA writable_schema = ON",
        "SELECT * FROM sqlite_master",
        "SELECT * FROM pragma_table_info('books')",
        "SELECT * FROM pragma_database_list",
        "SELECT load_extension('evil')",
    ],
)
def test_executor_authorizer_protects_uploaded_database(library: ConnectedDatabase,
                                                        query: str) -> None:
    with pytest.raises(QueryExecutionError):
        execute_readonly_query(library.engine, query, max_rows=10, timeout_seconds=2)
    with library.engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM books")).scalar_one() == 5
        assert conn.execute(text("SELECT COUNT(*) FROM authors WHERE country = 'XX'")
                            ).scalar_one() == 0


def test_writes_through_agent_are_blocked_on_uploaded_database(library: ConnectedDatabase) -> None:
    llm = RecordingLLMService([sql("DELETE FROM books"), sql("SELECT COUNT(*) AS n FROM books")])
    state = run_question(_graph_for(library, llm), "Remove all books")
    assert "Validation failed" in llm.correct_calls[0][1]
    assert state["query_results"]["rows"] == [[5]]


def test_pragma_function_through_agent_is_rejected_before_execution(
        library: ConnectedDatabase) -> None:
    llm = RecordingLLMService([sql("SELECT name FROM pragma_table_info('books')")])
    state = run_question(_graph_for(library, llm, max_retries=0), "List columns")
    assert state["status"] == "failed"
    assert "Function not allowed: pragma_table_info" in state["final_answer"]


def test_timeout_applies_to_uploaded_database(library: ConnectedDatabase) -> None:
    heavy = ("WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n) "
             "SELECT COUNT(*) FROM n")
    with pytest.raises(QueryExecutionError) as info:
        execute_readonly_query(library.engine, heavy, max_rows=10, timeout_seconds=0.3)
    assert info.value.kind == "timeout"


def test_row_cap_applies_to_uploaded_database(library: ConnectedDatabase) -> None:
    llm = RecordingLLMService([sql("SELECT * FROM books")])
    state = run_question(_graph_for(library, llm, max_rows=2), "All books")
    assert len(state["query_results"]["rows"]) == 2
    assert state["query_results"]["truncated"] is True


def test_execution_errors_on_uploads_are_sanitized(library: ConnectedDatabase) -> None:
    with pytest.raises(QueryExecutionError) as info:
        execute_readonly_query(library.engine, "SELECT nope FROM books", max_rows=5,
                               timeout_seconds=2)
    assert "no such column" in str(info.value)
    assert library.stored_path is not None
    assert str(library.stored_path.parent) not in str(info.value)
