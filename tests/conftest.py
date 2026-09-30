"""Shared fixtures. Unit tests use a temporary database and a fake LLM."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.llm import SQLDraft  # noqa: E402
from agent.state import QueryResults  # noqa: E402
from database.connection import get_engine  # noqa: E402
from database.init_db import init_database  # noqa: E402


@pytest.fixture(scope="session")
def seeded_db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A freshly built sample database shared by read-only tests."""
    path = tmp_path_factory.mktemp("db") / "test.db"
    init_database(path, reset=True)
    return path


@pytest.fixture()
def ro_engine(seeded_db_path: Path) -> Iterator[Engine]:
    engine = get_engine(seeded_db_path, read_only=True)
    yield engine
    engine.dispose()


class FakeLLMService:
    """Scripted stand-in for GeminiLLMService.

    ``drafts`` are returned in order by generate_sql/correct_sql calls. An
    Exception instance in the list is raised instead of returned.
    """

    def __init__(self, drafts: list[SQLDraft | Exception],
                 explanation: str | Exception = "Fake explanation.") -> None:
        self.drafts = list(drafts)
        self.explanation = explanation
        self.generate_calls = 0
        self.correct_calls: list[tuple[str, str]] = []  # (failed_sql, error)
        self.explained: list[QueryResults] = []

    def _next(self) -> SQLDraft:
        if not self.drafts:
            raise AssertionError("FakeLLMService ran out of scripted drafts")
        item = self.drafts.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def generate_sql(self, question: str, schema: str) -> SQLDraft:
        self.generate_calls += 1
        assert "orders" in schema  # the real schema is passed in
        return self._next()

    def correct_sql(self, question: str, schema: str, sql: str, error: str) -> SQLDraft:
        self.correct_calls.append((sql, error))
        return self._next()

    def explain_results(self, question: str, sql: str, results: QueryResults) -> str:
        self.explained.append(results)
        if isinstance(self.explanation, Exception):
            raise self.explanation
        return self.explanation

    # Scripted reply for the on-demand query narrative; tests may reassign it.
    query_explanation: str | Exception = "Fake query explanation."

    def explain_query(self, question: str, sql: str, breakdown: str) -> str:
        if not hasattr(self, "query_explain_calls"):
            self.query_explain_calls: list[tuple[str, str, str]] = []
        self.query_explain_calls.append((question, sql, breakdown))
        if isinstance(self.query_explanation, Exception):
            raise self.query_explanation
        return self.query_explanation


def sql(query: str) -> SQLDraft:
    return SQLDraft(answerable=True, sql=query, reason="test")


def unanswerable(reason: str) -> SQLDraft:
    return SQLDraft(answerable=False, sql=None, reason=reason)


class RecordingLLMService(FakeLLMService):
    """FakeLLMService for arbitrary databases: records every schema it is given
    instead of asserting that it describes the sample database."""

    def __init__(self, drafts: list[SQLDraft | Exception],
                 explanation: str | Exception = "Fake explanation.") -> None:
        super().__init__(drafts, explanation)
        self.schemas: list[str] = []

    def generate_sql(self, question: str, schema: str) -> SQLDraft:
        self.generate_calls += 1
        self.schemas.append(schema)
        return self._next()

    def correct_sql(self, question: str, schema: str, sql: str, error: str) -> SQLDraft:
        self.schemas.append(schema)
        return super().correct_sql(question, schema, sql, error)


def make_sqlite_db(path: Path, script: str) -> bytes:
    """Build a SQLite file from an SQL script and return its bytes."""
    import sqlite3

    conn = sqlite3.connect(path)
    try:
        conn.executescript(script)
        conn.commit()
    finally:
        conn.close()
    return path.read_bytes()


# A small database that shares nothing with the sample schema.
LIBRARY_SQL = """
CREATE TABLE authors (
    author_id INTEGER PRIMARY KEY,
    full_name TEXT NOT NULL,
    country TEXT NOT NULL
);
CREATE TABLE books (
    book_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    author_id INTEGER NOT NULL REFERENCES authors(author_id),
    published_on DATE NOT NULL,
    copies_sold INTEGER NOT NULL
);
INSERT INTO authors VALUES (1, 'Ada Writer', 'UK'), (2, 'Bo Novelist', 'SE'),
                           (3, 'Cy Poet', 'US');
INSERT INTO books VALUES
    (1, 'First Light', 1, '2021-03-01', 1200),
    (2, 'Second Wind', 1, '2022-07-15', 800),
    (3, 'Cold Harbour', 2, '2020-11-30', 5400),
    (4, 'Quiet Verses', 3, '2023-01-09', 150),
    (5, 'Late Bloom', 2, '2024-05-21', 2300);
"""


@pytest.fixture()
def library_db_bytes(tmp_path: Path) -> bytes:
    return make_sqlite_db(tmp_path / "library_src.db", LIBRARY_SQL)
