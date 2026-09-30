"""Headless UI test of app.py via Streamlit's AppTest (fake LLM, temp database)."""

from __future__ import annotations

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import agent.llm
from tests.conftest import FakeLLMService, sql, unanswerable

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture()
def app_env(monkeypatch, seeded_db_path):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("QUERYPILOT_DB_PATH", str(seeded_db_path))
    st.cache_resource.clear()
    yield monkeypatch
    st.cache_resource.clear()


def _use_fake_llm(monkeypatch, fake: FakeLLMService) -> None:
    monkeypatch.setattr(agent.llm, "GeminiLLMService", lambda **_: fake)


def _ask(at: AppTest) -> None:
    next(b for b in at.button if b.label == "Ask").click().run()


def test_app_answers_question_and_shows_sql_and_table(app_env) -> None:
    _use_fake_llm(app_env, FakeLLMService(
        [sql("SELECT city, COUNT(*) AS customers FROM customers GROUP BY city ORDER BY 2 DESC")],
        explanation="Mumbai has the most customers.",
    ))
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not at.exception

    at.text_area(key="question").input("Which city has the most customers?")
    _ask(at)

    assert not at.exception
    assert at.success[0].value == "Mumbai has the most customers."
    assert "GROUP BY city" in at.code[0].value
    assert len(at.dataframe) == 1 and len(at.dataframe[0].value) == 8


def test_app_shows_cannot_answer_warning(app_env) -> None:
    _use_fake_llm(app_env, FakeLLMService([unanswerable("No profit data is stored.")]))
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    at.text_area(key="question").input("What is our profit?")
    _ask(at)

    assert not at.exception
    assert "No profit data is stored." in at.warning[0].value
    assert len(at.code) == 0 and len(at.dataframe) == 0


def test_app_shows_setup_help_when_database_missing(app_env, tmp_path) -> None:
    app_env.setenv("QUERYPILOT_DB_PATH", str(tmp_path / "missing.db"))
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not at.exception
    assert any("database.init_db" in e.value for e in at.error)


# ------------------------------------------------------- uploaded databases
from tests.conftest import RecordingLLMService, make_sqlite_db  # noqa: E402

BOOKS_SQL = "SELECT title, copies_sold FROM books ORDER BY copies_sold DESC"


@pytest.fixture()
def upload_env(app_env, tmp_path):
    upload_dir = tmp_path / "uploads"
    app_env.setenv("QUERYPILOT_UPLOAD_DIR", str(upload_dir))
    return upload_dir


def _sidebar_button(at: AppTest, label: str):
    return next(b for b in at.sidebar.button if b.label == label)


def _connect(at: AppTest, name: str, data: bytes) -> None:
    at.sidebar.file_uploader[0].upload(name, data)
    at.run()
    _sidebar_button(at, "Connect database").click().run()


def _sidebar_text(at: AppTest) -> str:
    return "\n".join(t.value for t in at.sidebar.text)


def _sidebar_markdown(at: AppTest) -> str:
    return "\n".join(m.value for m in at.sidebar.markdown)


def test_app_starts_on_sample_database_with_summary(app_env, upload_env) -> None:
    _use_fake_llm(app_env, FakeLLMService([]))
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not at.exception
    assert "Sample sales database" in _sidebar_markdown(at)
    assert "orders  ·  " in _sidebar_text(at)
    assert _sidebar_button(at, "Connect database").disabled
    assert not any(b.label == "Disconnect" for b in at.sidebar.button)
    assert any(b.label == EXAMPLE for b in at.sidebar.button)


EXAMPLE = "Which city has the most customers?"


def test_app_connect_ask_switch_and_disconnect(app_env, upload_env, library_db_bytes,
                                              tmp_path) -> None:
    fake = RecordingLLMService(
        [sql("SELECT COUNT(*) AS n FROM customers"), sql(BOOKS_SQL)],
        explanation="Cold Harbour sold the most.",
    )
    _use_fake_llm(app_env, fake)
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()

    # 1. A sample answer is on screen.
    at.text_area(key="question").input("How many customers?")
    _ask(at)
    assert at.success and "COUNT(*)" in at.code[0].value

    # 2. Connect an upload: status, name, tables; old result cleared.
    _connect(at, "library.db", library_db_bytes)
    assert not at.exception
    assert "Connected" in _sidebar_markdown(at)
    assert "library.db" in _sidebar_text(at)
    assert "books  ·  5 rows" in _sidebar_text(at)
    assert not at.success and not at.code and not at.dataframe  # sample result is gone
    assert not any(b.label == EXAMPLE for b in at.sidebar.button)  # sample examples hidden
    copies = list(upload_env.glob("*"))
    assert len(copies) == 1

    # 3. Ask: the LLM gets the upload's schema and rows come from the upload.
    at.text_area(key="question").input("Best-selling books?")
    _ask(at)
    assert not at.exception
    assert at.success[0].value == "Cold Harbour sold the most."
    assert at.dataframe[0].value.iloc[0].tolist() == ["Cold Harbour", 5400]
    assert "Table books (5 rows)" in fake.schemas[-1]
    assert "Business definitions" not in fake.schemas[-1]

    # 4. Switch to a second upload: the first copy is closed and deleted.
    other = make_sqlite_db(tmp_path / "pets.db",
                           "CREATE TABLE pets (name TEXT); INSERT INTO pets VALUES ('Rex');")
    _connect(at, "pets.sqlite", other)
    assert "pets.sqlite" in _sidebar_text(at) and "books" not in _sidebar_text(at)
    assert not at.success and not at.dataframe
    assert not copies[0].exists() and len(list(upload_env.glob("*"))) == 1

    # 5. Disconnect: back to the sample, upload deleted, examples restored.
    _sidebar_button(at, "Disconnect").click().run()
    assert not at.exception
    assert "Sample sales database" in _sidebar_markdown(at)
    assert list(upload_env.glob("*")) == []
    assert any(b.label == EXAMPLE for b in at.sidebar.button)


def test_app_invalid_upload_shows_error_and_keeps_current_database(
        app_env, upload_env, library_db_bytes) -> None:
    _use_fake_llm(app_env, RecordingLLMService([]))
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()

    _connect(at, "notes.db", b"this is just text, renamed to .db" * 10)
    assert not at.exception
    assert "not a SQLite database" in at.sidebar.error[0].value
    assert "Sample sales database" in _sidebar_markdown(at)
    assert list(upload_env.glob("*")) == []

    # With an upload connected, a failed second upload keeps the first one.
    _connect(at, "library.db", library_db_bytes)
    at.sidebar.file_uploader[0].upload("broken.sqlite", b"SQLite format 3\x00" + b"\x00" * 50)
    at.run()
    _sidebar_button(at, "Connect database").click().run()
    assert at.sidebar.error and "Could not connect" in at.sidebar.error[0].value
    assert "library.db" in _sidebar_text(at)
    assert len(list(upload_env.glob("*"))) == 1
    for e in at.sidebar.error:  # no server paths leak into the UI
        assert str(upload_env) not in e.value


def test_app_uploads_are_isolated_per_session(app_env, upload_env, library_db_bytes) -> None:
    fake = RecordingLLMService([sql(BOOKS_SQL), sql("SELECT COUNT(*) AS n FROM customers")])
    _use_fake_llm(app_env, fake)
    alice = AppTest.from_file(APP_PATH, default_timeout=30).run()
    _connect(alice, "library.db", library_db_bytes)
    alice.text_area(key="question").input("Best-selling books?")
    _ask(alice)
    assert alice.dataframe

    # A second browser session (same server process, same cache) still sees the sample.
    bob = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not bob.exception
    assert "Sample sales database" in _sidebar_markdown(bob)
    assert "library.db" not in _sidebar_text(bob)
    assert not bob.success and not bob.dataframe  # Alice's result is not visible
    bob.text_area(key="question").input("How many customers?")
    _ask(bob)
    assert "Table customers" in fake.schemas[-1] and "Table books" not in fake.schemas[-1]

    # Alice's session is unaffected by Bob's.
    assert "library.db" in _sidebar_text(alice.run())


# ---------------------------------------------------- analytics dashboard
import json  # noqa: E402

import agent.graph  # noqa: E402
import analytics.charts  # noqa: E402
import analytics.insights  # noqa: E402

TOP5_SQL = (
    "SELECT p.product_name, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue "
    "FROM order_items oi JOIN orders o ON o.order_id = oi.order_id "
    "JOIN products p ON p.product_id = oi.product_id WHERE o.status = 'completed' "
    "GROUP BY p.product_name ORDER BY revenue DESC LIMIT 5"
)


def _charts(at: AppTest) -> list[dict]:
    return [json.loads(el.proto.spec) for el in at.get("vega_lite_chart")]


def _chart_rows(at: AppTest) -> list[dict]:
    """Rows Streamlit actually ships to the browser for the first chart (it moves
    inline Vega-Lite datasets into Arrow-encoded proto fields)."""
    import pyarrow as pa

    dataset = at.get("vega_lite_chart")[0].proto.datasets[0]
    return pa.ipc.open_stream(dataset.data.data).read_all().to_pylist()


def _metrics(at: AppTest) -> dict[str, tuple[str, str]]:
    return {m.label: (m.value, m.delta) for m in at.metric}


def _ask_question(at: AppTest, question: str) -> None:
    at.text_area(key="question").input(question)
    _ask(at)


def _raise(exc: Exception):
    def raiser(*_args, **_kwargs):
        raise exc
    return raiser


def test_dashboard_for_top_products_question(app_env, ro_engine) -> None:
    _use_fake_llm(app_env, FakeLLMService([sql(TOP5_SQL)],
                                          explanation="Laptops lead revenue."))
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    _ask_question(at, "What were the top five products by revenue?")
    assert not at.exception

    # Ground truth straight from the database, independent of the app.
    with ro_engine.connect() as conn:
        expected = [tuple(r) for r in conn.exec_driver_sql(TOP5_SQL)]

    assert at.success[0].value == "Laptops lead revenue."            # answer kept
    metrics = _metrics(at)                                           # KPI cards
    assert metrics["Highest revenue"] == (expected[0][0], f"{expected[0][1]:,.2f}")
    assert metrics["Combined revenue"][0] == f"{sum(r[1] for r in expected):,.2f}"
    assert metrics["Rows returned"][0] == "5"
    charts = _charts(at)                                             # ranked bar chart
    assert len(charts) == 1
    assert "bar" in [layer["mark"]["type"] for layer in charts[0]["layer"]]
    table = at.dataframe[0].value                                    # underlying table
    assert [tuple(r) for r in table.itertuples(index=False)] == expected
    assert at.code[0].value == TOP5_SQL                              # exact SQL, expandable
    assert any(t.label == "SQL" for t in at.tabs)              # SQL moved to its own tab
    assert any(e.label == "Technical details" for e in at.expander)


def test_chart_failure_falls_back_to_table(app_env, monkeypatch) -> None:
    _use_fake_llm(app_env, FakeLLMService([sql(TOP5_SQL)]))
    monkeypatch.setattr(analytics.charts, "build_chart",
                        _raise(RuntimeError("renderer exploded at C:\\secret\\path")))
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    _ask_question(at, "Top products?")
    assert not at.exception
    assert _charts(at) == []
    assert any("chart couldn't be drawn" in i.value for i in at.info)
    assert all("secret" not in i.value for i in at.info)            # no internals leaked
    assert len(at.dataframe) == 1 and len(at.dataframe[0].value) == 5


def test_analysis_failure_still_shows_answer_and_table(app_env, monkeypatch) -> None:
    _use_fake_llm(app_env, FakeLLMService([sql(TOP5_SQL)], explanation="ok"))
    monkeypatch.setattr(analytics.insights, "analyze_results", _raise(ValueError("boom")))
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    _ask_question(at, "Top products?")
    assert not at.exception
    assert at.success[0].value == "ok"
    assert len(at.dataframe) == 1 and not at.metric and _charts(at) == []


def test_validation_failure_shows_no_dashboard(app_env) -> None:
    bad = sql("DELETE FROM orders")
    _use_fake_llm(app_env, FakeLLMService([bad, bad, bad]))
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    _ask_question(at, "Delete everything")
    assert not at.exception
    assert at.error and not at.metric and not at.dataframe and _charts(at) == []
    labels = [e.label for e in at.expander]
    assert "Last attempted SQL" in labels and "Error details" in labels
    details = next(e for e in at.expander if e.label == "Technical details")
    assert "rejected, so the query was not executed" in details.markdown[0].value


def test_execution_failure_shows_error_without_dashboard(app_env) -> None:
    broken = sql("SELECT nope FROM customers")
    _use_fake_llm(app_env, FakeLLMService([broken, broken, broken]))
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    _ask_question(at, "Anything")
    assert not at.exception
    assert "no such column" in at.error[0].value
    assert not at.metric and not at.dataframe and _charts(at) == []


def test_empty_result_shows_message_without_kpis_or_chart(app_env) -> None:
    _use_fake_llm(app_env, FakeLLMService(
        [sql("SELECT * FROM customers WHERE city = 'Atlantis'")]))
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    _ask_question(at, "Customers in Atlantis?")
    assert not at.exception
    assert not at.metric and _charts(at) == []
    assert any(m.value == "The query returned no rows." for m in at.markdown)


def test_truncated_results_are_flagged(app_env) -> None:
    app_env.setenv("QUERYPILOT_MAX_ROWS", "3")
    _use_fake_llm(app_env, FakeLLMService([sql("SELECT customer_name, city FROM customers")]))
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    _ask_question(at, "List customers")
    assert not at.exception
    assert len(at.dataframe[0].value) == 3                    # the row limit is not bypassed
    assert _metrics(at)["Rows returned"][0] == "3+"
    assert any(":orange-badge[Truncated]" in m.value for m in at.markdown)


def test_failed_run_clears_the_previous_result(app_env, monkeypatch) -> None:
    _use_fake_llm(app_env, FakeLLMService([sql(TOP5_SQL)]))
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    _ask_question(at, "Top products?")
    assert at.metric and _charts(at)

    monkeypatch.setattr(agent.graph, "run_question", _raise(RuntimeError("crash")))
    _ask_question(at, "Something else")
    assert not at.metric and not at.dataframe and _charts(at) == []
    at.run()  # and the stale result does not come back on the next rerun
    assert not at.metric and not at.dataframe and _charts(at) == []


def test_dashboard_follows_the_active_database(app_env, upload_env, library_db_bytes,
                                               tmp_path) -> None:
    fake = RecordingLLMService([
        sql(TOP5_SQL),
        sql("SELECT title, copies_sold FROM books ORDER BY copies_sold DESC"),
    ])
    _use_fake_llm(app_env, fake)
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    _ask_question(at, "Top products?")
    assert at.metric and _charts(at)

    _connect(at, "library.db", library_db_bytes)     # switching hides every visualization
    assert not at.metric and _charts(at) == [] and not at.dataframe

    _ask_question(at, "Best-selling books?")         # new visuals come from the upload only
    assert _metrics(at)["Highest copies sold"] == ("Cold Harbour", "5,400")
    assert [(d["label"], d["value"]) for d in _chart_rows(at)] == [
        ("Cold Harbour", 5400), ("Late Bloom", 2300), ("First Light", 1200),
        ("Second Wind", 800), ("Quiet Verses", 150)]

    other = make_sqlite_db(tmp_path / "pets.db",
                           "CREATE TABLE pets (name TEXT); INSERT INTO pets VALUES ('Rex');")
    _connect(at, "pets.sqlite", other)               # a second upload clears them again
    assert not at.metric and _charts(at) == [] and not at.dataframe

    _sidebar_button(at, "Disconnect").click().run()  # and so does returning to the sample
    assert not at.metric and _charts(at) == []


def test_data_derived_text_cannot_inject_markdown(app_env, upload_env, tmp_path) -> None:
    data = make_sqlite_db(tmp_path / "evil.db", """
        CREATE TABLE items (label TEXT, amount REAL);
        INSERT INTO items VALUES ('[click me](http://evil.example)', 9.0),
                                 (':red[alarm] $x^2$', 5.0), ('plain', 1.0);
    """)
    _use_fake_llm(app_env, RecordingLLMService([sql(
        "SELECT label, amount FROM items ORDER BY amount DESC")]))
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    _connect(at, "evil.db", data)
    _ask_question(at, "Items by amount")
    assert not at.exception
    value = _metrics(at)["Highest amount"][0]
    assert value == r"\[click me\](http\://evil.example)"   # rendered literally, not as a link
    assert at.dataframe[0].value.iloc[1, 0] == ":red[alarm] $x^2$"  # the table keeps raw values


# ------------------------------------------------------ query explanation
def _how_tab(at: AppTest):
    return next(t for t in at.tabs if t.label == "How it was calculated")


def _how_text(at: AppTest) -> str:
    tab = _how_tab(at)
    return "\n".join([m.value for m in tab.markdown] + [c.value for c in tab.caption])


def test_explanation_tab_breaks_down_the_executed_sql(app_env) -> None:
    _use_fake_llm(app_env, FakeLLMService([sql(TOP5_SQL)]))
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    _ask_question(at, "What were the top five products by revenue?")
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Chart & data", "How it was calculated", "SQL"]

    text = _how_text(at)
    # Tables, joins, filter, metric formula and ranking, all parsed from the SQL that ran.
    for expected in [r"order\_items (as oi)", "orders (as o)", "products (as p)",
                     r"Inner join with orders, matching rows where orders.order\_id equals",
                     "orders.status is 'completed'", "`o.status = 'completed'`",
                     r"revenue = sum of order\_items.quantity × order\_items.unit\_price, "
                     "rounded to 2 decimals",
                     "Sorted by revenue, descending (highest first)",
                     "Only the top 5 rows are kept"]:
        assert expected in text, expected
    # Checks reflect what the pipeline actually recorded.
    assert "Passed SQL validation" in text and "Returned 5 rows" in text
    assert "Business definitions applied" in text                # sample database only
    # The breakdown adds no code blocks or tables: the SQL tab holds the only code block.
    assert len(at.code) == 1 and at.code[0].value == TOP5_SQL
    assert len(at.dataframe) == 1


def test_explanation_on_upload_has_no_sample_definitions(app_env, upload_env,
                                                         library_db_bytes) -> None:
    _use_fake_llm(app_env, RecordingLLMService([sql(BOOKS_SQL)]))
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    _connect(at, "library.db", library_db_bytes)
    _ask_question(at, "Best-selling books?")
    text = _how_text(at)
    assert "Sorted by copies\\_sold, descending (highest first)" in text
    assert "Business definitions applied" not in text
    assert "order\\_items" not in text


def test_ai_explanation_is_on_demand_cached_and_cleared(app_env, upload_env,
                                                         library_db_bytes) -> None:
    fake = FakeLLMService([sql(TOP5_SQL), sql("SELECT COUNT(*) AS n FROM customers")])
    fake.query_explanation = "Revenue was summed per product from completed orders."
    _use_fake_llm(app_env, fake)
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    _ask_question(at, "Top products?")
    assert not hasattr(fake, "query_explain_calls")              # no call until asked

    at.button(key="explain_query").click().run()
    assert not at.exception
    assert len(fake.query_explain_calls) == 1
    question, sent_sql, breakdown = fake.query_explain_calls[0]
    assert question == "Top products?" and sent_sql == TOP5_SQL
    assert "Filters applied:\n- orders.status is 'completed'" in breakdown  # grounded
    assert "Revenue was summed per product from completed orders." in _how_text(at)

    at.run()                                                     # survives reruns, no new call
    assert "Revenue was summed per product" in _how_text(at)
    assert len(fake.query_explain_calls) == 1

    _ask_question(at, "How many customers?")                     # a new question clears it
    assert "Revenue was summed per product" not in _how_text(at)
    assert at.button(key="explain_query")

    at.button(key="explain_query").click().run()
    assert len(fake.query_explain_calls) == 2
    _connect(at, "library.db", library_db_bytes)                 # switching databases clears it
    assert not at.tabs


def test_ai_explanation_errors_are_mapped_to_safe_messages(app_env) -> None:
    fake = FakeLLMService([sql(TOP5_SQL)])
    fake.query_explanation = RuntimeError("429 RESOURCE_EXHAUSTED key=AIza-secret")
    _use_fake_llm(app_env, fake)
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    _ask_question(at, "Top products?")
    at.button(key="explain_query").click().run()
    assert not at.exception
    errors = [e.value for e in _how_tab(at).error]
    assert any("rate limit" in e for e in errors)
    assert all("AIza-secret" not in e for e in errors)
    assert at.button(key="explain_query")                         # can retry later


def test_ai_explanation_text_is_rendered_literally(app_env) -> None:
    fake = FakeLLMService([sql(TOP5_SQL)])
    fake.query_explanation = "See [this](http://evil.example) :red[now]"
    _use_fake_llm(app_env, fake)
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    _ask_question(at, "Top products?")
    at.button(key="explain_query").click().run()
    assert r"See \[this\](http\://evil.example) \:red\[now\]" in _how_text(at)


def test_identifiers_in_the_breakdown_cannot_inject_markdown(app_env, upload_env,
                                                             tmp_path) -> None:
    data = make_sqlite_db(tmp_path / "odd.db", """
        CREATE TABLE "t*" (label TEXT, "**bold**" REAL);
        INSERT INTO "t*" VALUES ('a', 1.0), ('b', 2.0);
    """)
    _use_fake_llm(app_env, RecordingLLMService([sql(
        'SELECT label, SUM("**bold**") AS "**bold**" FROM "t*" GROUP BY label')]))
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    _connect(at, "odd.db", data)
    _ask_question(at, "Totals")
    assert not at.exception
    text = _how_text(at)
    assert r"\*\*bold\*\* = sum of \*\*bold\*\*" in text
    assert "- **bold** =" not in text


def test_failed_and_unanswerable_results_have_no_tabs(app_env) -> None:
    bad = sql("DELETE FROM orders")
    _use_fake_llm(app_env, FakeLLMService([bad, bad, bad, unanswerable("No profit data.")]))
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    _ask_question(at, "Delete everything")
    assert not at.tabs and any(e.label == "Last attempted SQL" for e in at.expander)
    _ask_question(at, "What is our profit?")
    assert not at.tabs and at.warning
