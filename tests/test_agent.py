"""LangGraph workflow and executor tests with a mocked LLM (no API calls)."""

from __future__ import annotations

import pytest
from sqlalchemy import text

from agent.graph import NO_ROWS_ANSWER, build_graph, run_question
from agent.llm import describe_llm_error
from agent.tools import QueryExecutionError, execute_readonly_query
from tests.conftest import FakeLLMService, sql, unanswerable

REVENUE_SQL = """
SELECT ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_revenue
FROM orders o JOIN order_items oi ON oi.order_id = o.order_id
WHERE o.status = 'completed'
"""


def _graph(llm, engine, **kwargs):
    return build_graph(llm, engine, **kwargs)


# --------------------------------------------------------------- happy paths
def test_successful_question_returns_real_database_results(ro_engine) -> None:
    llm = FakeLLMService([sql(REVENUE_SQL)], explanation="Total revenue is X.")
    state = run_question(_graph(llm, ro_engine), "What is the total revenue from completed orders?")

    # Compute the expected value independently, straight from the database.
    with ro_engine.connect() as conn:
        expected = conn.execute(text(
            "SELECT ROUND(SUM(quantity * unit_price), 2) FROM order_items WHERE order_id IN "
            "(SELECT order_id FROM orders WHERE status = 'completed')"
        )).scalar_one()

    assert state["status"] == "success"
    assert state["query_results"]["columns"] == ["total_revenue"]
    assert state["query_results"]["rows"] == [[expected]]
    assert expected > 0
    assert state["generated_sql"].startswith("SELECT")
    assert state["final_answer"] == "Total revenue is X."
    assert state["retry_count"] == 0
    assert llm.explained[0]["rows"] == [[expected]]  # the LLM only explains real rows


def test_never_ordered_products_query(ro_engine) -> None:
    llm = FakeLLMService([sql(
        "SELECT p.product_name FROM products p LEFT JOIN order_items oi "
        "ON oi.product_id = p.product_id WHERE oi.order_item_id IS NULL ORDER BY 1"
    )])
    state = run_question(_graph(llm, ro_engine), "Which products have never been ordered?")
    assert state["status"] == "success"
    assert [r[0] for r in state["query_results"]["rows"]] == [
        "Espresso Machine Pro", "Rowing Machine", "Smart Ring"
    ]


def test_empty_result_is_reported_without_inventing_an_answer(ro_engine) -> None:
    llm = FakeLLMService([sql("SELECT * FROM customers WHERE city = 'Atlantis'")])
    state = run_question(_graph(llm, ro_engine), "Customers in Atlantis?")
    assert state["status"] == "success"
    assert state["query_results"]["rows"] == []
    assert state["final_answer"] == NO_ROWS_ANSWER
    assert llm.explained == []  # LLM not asked to explain nothing


def test_explanation_failure_falls_back_to_deterministic_summary(ro_engine) -> None:
    llm = FakeLLMService([sql("SELECT COUNT(*) AS n FROM customers")],
                         explanation=RuntimeError("LLM down"))
    state = run_question(_graph(llm, ro_engine), "How many customers?")
    assert state["status"] == "success"
    assert state["final_answer"].startswith("Result: n = ")


# ---------------------------------------------------------- retries/failures
def test_execution_error_is_corrected_on_retry(ro_engine) -> None:
    llm = FakeLLMService([
        sql("SELECT customer_nme FROM customers"),  # typo -> SQLite error
        sql("SELECT customer_name FROM customers LIMIT 3"),
    ])
    state = run_question(_graph(llm, ro_engine), "List some customers")
    assert state["status"] == "success"
    assert state["retry_count"] == 1
    assert len(state["query_results"]["rows"]) == 3
    failed_sql, error = llm.correct_calls[0]
    assert "customer_nme" in failed_sql
    assert "no such column" in error


def test_validation_failure_is_corrected_and_unsafe_sql_never_runs(ro_engine) -> None:
    llm = FakeLLMService([
        sql("DELETE FROM orders"),
        sql("SELECT COUNT(*) AS n FROM orders"),
    ])
    state = run_question(_graph(llm, ro_engine), "How many orders?")
    assert state["status"] == "success"
    assert state["retry_count"] == 1
    assert "Validation failed" in llm.correct_calls[0][1]
    with ro_engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM orders")).scalar_one() == \
            state["query_results"]["rows"][0][0] > 0


def test_retry_limit_is_enforced_and_fails_gracefully(ro_engine) -> None:
    bad = sql("SELECT nope FROM customers")
    llm = FakeLLMService([bad, bad, bad, bad, bad])
    state = run_question(_graph(llm, ro_engine, max_retries=2), "Anything")
    assert state["status"] == "failed"
    assert state["retry_count"] == 2
    assert len(llm.correct_calls) == 2       # exactly max_retries corrections
    assert len(llm.drafts) == 2              # remaining drafts were never requested
    assert "after 3 attempt(s)" in state["final_answer"]
    assert "no such column" in state["final_answer"]
    assert "Traceback" not in state["final_answer"]


def test_zero_retries_fails_immediately(ro_engine) -> None:
    llm = FakeLLMService([sql("DROP TABLE customers")])
    state = run_question(_graph(llm, ro_engine, max_retries=0), "drop it")
    assert state["status"] == "failed"
    assert llm.correct_calls == []
    assert state.get("query_results") is None


def test_llm_exception_fails_gracefully(ro_engine) -> None:
    llm = FakeLLMService([RuntimeError("secret-key-123 network error")])
    state = run_question(_graph(llm, ro_engine), "Total revenue?")
    assert state["status"] == "failed"
    assert "secret-key-123" not in state["final_answer"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("429 RESOURCE_EXHAUSTED quota key=AIza-secret", "rate limit"),
        ("403 PERMISSION_DENIED API key not valid AIza-secret", "GEMINI_API_KEY"),
        ("404 NOT_FOUND model x", "GEMINI_MODEL"),
        ("503 UNAVAILABLE This model is currently experiencing high demand", "overloaded"),
        ("boom AIza-secret", "could not be reached"),
    ],
)
def test_llm_errors_are_mapped_to_safe_messages(raw: str, expected: str) -> None:
    message = describe_llm_error(RuntimeError(raw))
    assert expected in message
    assert "AIza-secret" not in message


# ------------------------------------------------------------- unanswerable
def test_unanswerable_question_is_not_invented(ro_engine) -> None:
    llm = FakeLLMService([unanswerable("The database has no cost or profit data.")])
    state = run_question(_graph(llm, ro_engine), "What was our profit margin?")
    assert state["status"] == "cannot_answer"
    assert state["generated_sql"] is None
    assert state.get("query_results") is None
    assert "no cost or profit data" in state["final_answer"]
    assert llm.explained == []


def test_empty_question_is_rejected_without_llm_call(ro_engine) -> None:
    llm = FakeLLMService([])
    state = run_question(_graph(llm, ro_engine), "   ")
    assert state["status"] == "failed"
    assert llm.generate_calls == 0


# ------------------------------------- executor defense in depth (no validator)
@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM orders",
        "UPDATE products SET unit_price = 1",
        "DROP TABLE customers",
        "ATTACH DATABASE 'evil.db' AS evil",
        "PRAGMA query_only = OFF",
        "SELECT * FROM sqlite_master",
    ],
)
def test_executor_blocks_unsafe_sql_even_without_validator(ro_engine, query: str) -> None:
    with pytest.raises(QueryExecutionError):
        execute_readonly_query(ro_engine, query, max_rows=10, timeout_seconds=2)
    with ro_engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM orders")).scalar_one() > 0
        assert conn.execute(text("SELECT COUNT(*) FROM products WHERE unit_price = 1")).scalar_one() == 0


def test_executor_rejects_multiple_statements(ro_engine) -> None:
    with pytest.raises(QueryExecutionError):
        execute_readonly_query(ro_engine, "SELECT 1; SELECT 2", max_rows=10, timeout_seconds=2)


def test_executor_times_out_long_queries(ro_engine) -> None:
    heavy = ("WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n) "
             "SELECT COUNT(*) FROM n")
    with pytest.raises(QueryExecutionError) as info:
        execute_readonly_query(ro_engine, heavy, max_rows=10, timeout_seconds=0.3)
    assert info.value.kind == "timeout"


def test_executor_caps_rows_and_flags_truncation(ro_engine) -> None:
    result = execute_readonly_query(ro_engine, "SELECT * FROM order_items", max_rows=7,
                                    timeout_seconds=2)
    assert len(result.rows) == 7
    assert result.truncated is True
    small = execute_readonly_query(ro_engine, "SELECT * FROM products LIMIT 3", max_rows=7,
                                   timeout_seconds=2)
    assert small.truncated is False


def test_executor_error_message_is_sanitized(ro_engine, seeded_db_path) -> None:
    with pytest.raises(QueryExecutionError) as info:
        execute_readonly_query(ro_engine, "SELECT * FROM nonexistent", max_rows=5,
                               timeout_seconds=2)
    message = str(info.value)
    assert "no such table" in message
    assert str(seeded_db_path) not in message
