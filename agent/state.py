"""Explicit state schema for the LangGraph workflow.

Each node receives the full state and returns only the keys it changes;
LangGraph merges the update into the state before the next node runs.
"""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict

ErrorStage = Literal["llm", "validation", "execution"]
Status = Literal["running", "success", "cannot_answer", "failed"]


class QueryResults(TypedDict):
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool
    elapsed_ms: NotRequired[float]  # measured by the executor around execute + fetch


class AgentState(TypedDict, total=False):
    question: str
    database_schema: str
    generated_sql: str | None
    query_results: QueryResults | None
    error: str | None              # user-safe message of the most recent failure
    error_stage: ErrorStage | None
    retry_count: int               # number of correction attempts used so far
    final_answer: str | None
    status: Status
