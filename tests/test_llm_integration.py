"""Live end-to-end tests against the real Gemini API.

Excluded from the default run (see pytest.ini). Run explicitly with:

    python -m pytest -m integration -v

Requires GEMINI_API_KEY (and network access). Each test makes 2+ API calls.
Free-tier keys allow only a few requests per minute, so tests are paced by
QUERYPILOT_INTEGRATION_DELAY seconds (default 30; set 0 on a paid key).
"""

from __future__ import annotations

import os
import time

import pytest
from sqlalchemy import text

from agent.graph import build_graph, run_question
from agent.llm import GeminiLLMService
from agent.prompts import SAMPLE_DATABASE_NOTES
from config import get_settings

pytestmark = pytest.mark.integration

settings = get_settings()
requires_key = pytest.mark.skipif(not settings.gemini_api_key, reason="GEMINI_API_KEY not set")


@pytest.fixture(autouse=True)
def _pace_requests():
    yield
    time.sleep(float(os.getenv("QUERYPILOT_INTEGRATION_DELAY", "30")))


@pytest.fixture(scope="module")
def live_graph(seeded_db_path):
    from database.connection import get_engine

    engine = get_engine(seeded_db_path, read_only=True)
    llm = GeminiLLMService(api_key=settings.gemini_api_key or "", model=settings.gemini_model)
    # Same configuration as the app's sample graph (sample business definitions included).
    yield build_graph(llm, engine, max_retries=settings.max_retries,
                      schema_notes=SAMPLE_DATABASE_NOTES), engine
    engine.dispose()


@requires_key
def test_live_total_revenue(live_graph) -> None:
    graph, engine = live_graph
    state = run_question(graph, "What is the total revenue from completed orders?")
    assert state["status"] == "success", state.get("final_answer")
    with engine.connect() as conn:
        expected = conn.execute(text(
            "SELECT SUM(oi.quantity * oi.unit_price) FROM order_items oi "
            "JOIN orders o ON o.order_id = oi.order_id WHERE o.status = 'completed'"
        )).scalar_one()
    value = state["query_results"]["rows"][0][0]
    assert value == pytest.approx(expected, abs=0.01)


@requires_key
def test_live_top_category(live_graph) -> None:
    graph, engine = live_graph
    state = run_question(graph, "Which product category generated the highest revenue?")
    assert state["status"] == "success", state.get("final_answer")
    with engine.connect() as conn:
        expected = conn.execute(text(
            "SELECT p.category FROM order_items oi JOIN orders o ON o.order_id = oi.order_id "
            "JOIN products p ON p.product_id = oi.product_id WHERE o.status = 'completed' "
            "GROUP BY p.category ORDER BY SUM(oi.quantity * oi.unit_price) DESC LIMIT 1"
        )).scalar_one()
    assert expected in [str(v) for v in state["query_results"]["rows"][0]]


@requires_key
def test_live_never_ordered_products(live_graph) -> None:
    graph, _ = live_graph
    state = run_question(graph, "Which products have never been ordered?")
    assert state["status"] == "success", state.get("final_answer")
    names = {str(v) for row in state["query_results"]["rows"] for v in row}
    assert {"Smart Ring", "Espresso Machine Pro", "Rowing Machine"} <= names


@requires_key
def test_live_unanswerable_question(live_graph) -> None:
    graph, _ = live_graph
    state = run_question(graph, "What was the profit margin after marketing costs last quarter?")
    assert state["status"] == "cannot_answer", state.get("final_answer")


@requires_key
def test_live_query_explanation() -> None:
    from analytics.explain import breakdown_to_text, explain_sql

    query = ("SELECT p.category, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue "
             "FROM order_items oi JOIN orders o ON o.order_id = oi.order_id "
             "JOIN products p ON p.product_id = oi.product_id WHERE o.status = 'completed' "
             "GROUP BY p.category ORDER BY revenue DESC")
    llm = GeminiLLMService(api_key=settings.gemini_api_key or "", model=settings.gemini_model)
    text = llm.explain_query("Which category earned the most?", query,
                             breakdown_to_text(explain_sql(query)))
    assert text and "completed" in text.lower() and "categor" in text.lower()
