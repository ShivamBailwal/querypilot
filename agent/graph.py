"""LangGraph workflow: question -> SQL -> validate -> execute -> answer.

    inspect_schema -> generate_sql -> validate_sql -> execute_sql -> format_answer -> END
                          |               |              |
                          |          (invalid)      (db error)
                          |               v              v
                          |          correct_sql <-------+   while retry_count < max_retries
                          |               |
                          |               +--> validate_sql
                          |
          (cannot answer) +--> END          (retries exhausted / LLM failure) -> handle_failure -> END

Only ``correct_sql`` increments ``retry_count`` and both routers check it
against ``max_retries``, so the loop always terminates.
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy import Engine

from agent.llm import LLMService, SQLDraft, describe_llm_error, fallback_summary
from agent.prompts import with_schema_notes
from agent.sql_validator import validate_sql
from agent.state import AgentState
from agent.tools import QueryExecutionError, execute_readonly_query
from database.schema import get_schema_description, get_table_names

logger = logging.getLogger(__name__)

NO_ROWS_ANSWER = "The query ran successfully but returned no matching records in the database."


def _apply_draft(draft: SQLDraft) -> AgentState:
    """Convert an LLM draft into a state update."""
    if not draft.answerable or not (draft.sql and draft.sql.strip()):
        reason = draft.reason.strip() or "The available data does not cover this question."
        return {
            "generated_sql": None,
            "status": "cannot_answer",
            "final_answer": f"I can't answer this from the database. {reason}",
            "error": None,
            "error_stage": None,
        }
    return {"generated_sql": draft.sql.strip(), "error": None, "error_stage": None}


def build_graph(llm: LLMService, engine: Engine, *, max_retries: int = 2,
                max_rows: int = 200, timeout_seconds: float = 5.0,
                database_schema: str | None = None,
                allowed_tables: set[str] | list[str] | None = None,
                schema_notes: str = "") -> CompiledStateGraph:
    """Wire the nodes together. Dependencies are injected for testability.

    A compiled graph is bound to exactly one database: ``engine`` is used for
    the table allowlist, the schema in the prompt, and execution. To switch
    databases, build a new graph for the new engine.

    ``database_schema`` and ``allowed_tables`` let a caller that already
    inspected ``engine`` (it must be that same engine) skip re-inspection;
    otherwise they are read from ``engine``. ``schema_notes`` are optional
    business definitions for this database appended to the schema in prompts.
    """
    max_retries = min(max(max_retries, 0), 5)  # keeps run_question's recursion_limit valid
    allowed_tables = set(allowed_tables if allowed_tables is not None
                         else get_table_names(engine))
    schema_cache: dict[str, str] = {}
    if database_schema is not None:
        schema_cache["schema"] = database_schema

    # ------------------------------------------------------------------ nodes
    def inspect_schema(state: AgentState) -> AgentState:
        if "schema" not in schema_cache:
            schema_cache["schema"] = get_schema_description(engine)
        return {"database_schema": with_schema_notes(schema_cache["schema"], schema_notes),
                "retry_count": 0,
                "status": "running", "error": None, "error_stage": None,
                "generated_sql": None, "query_results": None, "final_answer": None}

    def generate_sql(state: AgentState) -> AgentState:
        try:
            draft = llm.generate_sql(state["question"], state["database_schema"])
        except Exception as exc:
            logger.exception("LLM call failed in generate_sql")
            return {"error": describe_llm_error(exc), "error_stage": "llm"}
        logger.info("Generated SQL: %s", draft.sql)
        return _apply_draft(draft)

    def validate(state: AgentState) -> AgentState:
        result = validate_sql(state.get("generated_sql"), allowed_tables)
        if result.is_valid:
            return {"error": None, "error_stage": None}
        logger.info("SQL rejected by validator: %s", result.message)
        return {"error": f"Validation failed: {result.message}", "error_stage": "validation"}

    def execute_sql(state: AgentState) -> AgentState:
        try:
            result = execute_readonly_query(
                engine, state["generated_sql"] or "", max_rows, timeout_seconds
            )
        except QueryExecutionError as exc:
            return {"error": f"Execution failed: {exc}", "error_stage": "execution",
                    "query_results": None}
        return {"query_results": result.as_dict(), "error": None, "error_stage": None}

    def correct_sql(state: AgentState) -> AgentState:
        retry_count = state.get("retry_count", 0) + 1
        try:
            draft = llm.correct_sql(
                state["question"], state["database_schema"],
                state.get("generated_sql") or "", state.get("error") or "",
            )
        except Exception as exc:
            logger.exception("LLM call failed in correct_sql")
            return {"retry_count": retry_count, "error": describe_llm_error(exc),
                    "error_stage": "llm"}
        logger.info("Corrected SQL (attempt %d): %s", retry_count, draft.sql)
        return {"retry_count": retry_count, **_apply_draft(draft)}

    def format_answer(state: AgentState) -> AgentState:
        results = state["query_results"]
        assert results is not None
        if not results["rows"]:
            return {"final_answer": NO_ROWS_ANSWER, "status": "success"}
        try:
            answer = llm.explain_results(state["question"], state["generated_sql"] or "", results)
        except Exception:
            logger.exception("LLM call failed in format_answer; using fallback summary")
            answer = ""
        return {"final_answer": answer or fallback_summary(results), "status": "success"}

    def handle_failure(state: AgentState) -> AgentState:
        attempts = state.get("retry_count", 0) + 1
        if state.get("error_stage") == "llm":
            message = f"Sorry, I couldn't complete this request. {state.get('error')}"
        else:
            message = (
                f"Sorry, I couldn't produce a working SQL query for this question after "
                f"{attempts} attempt(s). Last error: {state.get('error')}"
            )
        return {"final_answer": message, "status": "failed"}

    # ---------------------------------------------------------------- routers
    def route_after_draft(state: AgentState) -> Literal["validate_sql", "handle_failure", "__end__"]:
        if state.get("status") == "cannot_answer":
            return END
        if state.get("error_stage") == "llm":
            return "handle_failure"
        return "validate_sql"

    def route_after_validation(
        state: AgentState,
    ) -> Literal["execute_sql", "correct_sql", "handle_failure"]:
        if state.get("error_stage") is None:
            return "execute_sql"
        return "correct_sql" if state.get("retry_count", 0) < max_retries else "handle_failure"

    def route_after_execution(
        state: AgentState,
    ) -> Literal["format_answer", "correct_sql", "handle_failure"]:
        if state.get("error_stage") is None:
            return "format_answer"
        return "correct_sql" if state.get("retry_count", 0) < max_retries else "handle_failure"

    # ------------------------------------------------------------------ graph
    builder = StateGraph(AgentState)
    builder.add_node("inspect_schema", inspect_schema)
    builder.add_node("generate_sql", generate_sql)
    builder.add_node("validate_sql", validate)
    builder.add_node("execute_sql", execute_sql)
    builder.add_node("correct_sql", correct_sql)
    builder.add_node("format_answer", format_answer)
    builder.add_node("handle_failure", handle_failure)

    builder.add_edge(START, "inspect_schema")
    builder.add_edge("inspect_schema", "generate_sql")
    builder.add_conditional_edges("generate_sql", route_after_draft)
    builder.add_conditional_edges("correct_sql", route_after_draft)
    builder.add_conditional_edges("validate_sql", route_after_validation)
    builder.add_conditional_edges("execute_sql", route_after_execution)
    builder.add_edge("format_answer", END)
    builder.add_edge("handle_failure", END)
    return builder.compile()


def run_question(graph: CompiledStateGraph, question: str) -> AgentState:
    """Run one question through the graph and return the final state."""
    question = (question or "").strip()
    if not question:
        return {"question": "", "status": "failed", "final_answer": "Please enter a question.",
                "error": "Empty question.", "retry_count": 0}
    # Recursion limit is a final safety net on top of max_retries (<= 5, so at
    # most 3 + 3*5 + 1 = 19 node executions).
    return graph.invoke({"question": question}, config={"recursion_limit": 30})
