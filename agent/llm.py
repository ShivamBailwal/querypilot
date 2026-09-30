"""LLM access for QueryPilot (Gemini via langchain-google-genai).

The graph only depends on the small ``LLMService`` protocol below, so unit
tests can pass a scripted fake and never call the real API.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agent import prompts
from agent.state import QueryResults

logger = logging.getLogger(__name__)

# How many result rows are shown to the LLM when explaining (the UI shows all).
EXPLAIN_MAX_ROWS = 50


class SQLDraft(BaseModel):
    """Structured output the model must return for generate/correct steps."""

    answerable: bool = Field(
        description="False if the database schema does not contain the data needed."
    )
    sql: str | None = Field(
        default=None, description="A single read-only SQLite SELECT query, or null."
    )
    reason: str = Field(
        default="",
        description="One sentence: the approach taken, or which information is missing.",
    )


class LLMService(Protocol):
    def generate_sql(self, question: str, schema: str) -> SQLDraft: ...

    def correct_sql(self, question: str, schema: str, sql: str, error: str) -> SQLDraft: ...

    def explain_results(self, question: str, sql: str, results: QueryResults) -> str: ...

    def explain_query(self, question: str, sql: str, breakdown: str) -> str: ...


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return "NULL" if value is None else str(value)


def results_to_text(results: QueryResults, max_rows: int = EXPLAIN_MAX_ROWS) -> str:
    header = " | ".join(results["columns"])
    body = [" | ".join(_fmt(v) for v in row) for row in results["rows"][:max_rows]]
    return "\n".join([header, *body])


def fallback_summary(results: QueryResults) -> str:
    """Deterministic answer used if the LLM explanation step fails."""
    rows, cols = results["rows"], results["columns"]
    if len(rows) == 1 and len(cols) == 1:
        return f"Result: {cols[0]} = {_fmt(rows[0][0])}."
    note = " (truncated)" if results["truncated"] else ""
    return f"The query returned {len(rows)} row(s){note}. See the table below for details."


class GeminiLLMService:
    """LLMService implementation backed by Google Gemini."""

    def __init__(self, api_key: str, model: str, temperature: float = 0.0,
                 timeout: float = 60.0) -> None:
        # Imported lazily so tests and the DB tooling don't need the package loaded.
        from langchain_google_genai import ChatGoogleGenerativeAI

        self._chat = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=temperature,
            timeout=timeout,
            max_retries=2,
        )
        self._sql_chat = self._chat.with_structured_output(SQLDraft)

    def _draft(self, system: str, user: str) -> SQLDraft:
        result = self._sql_chat.invoke([SystemMessage(system), HumanMessage(user)])
        if not isinstance(result, SQLDraft):  # defensive: some versions return dicts
            result = SQLDraft.model_validate(result)
        return result

    def generate_sql(self, question: str, schema: str) -> SQLDraft:
        return self._draft(
            prompts.GENERATE_SYSTEM.format(schema=schema),
            prompts.GENERATE_USER.format(question=question),
        )

    def correct_sql(self, question: str, schema: str, sql: str, error: str) -> SQLDraft:
        return self._draft(
            prompts.CORRECT_SYSTEM.format(schema=schema),
            prompts.CORRECT_USER.format(question=question, sql=sql, error=error),
        )

    def explain_results(self, question: str, sql: str, results: QueryResults) -> str:
        shown = min(len(results["rows"]), EXPLAIN_MAX_ROWS)
        truncated_note = ""
        if results["truncated"] or shown < len(results["rows"]):
            truncated_note = f", only the first {shown} shown; more rows exist"
        response = self._chat.invoke(
            [
                SystemMessage(prompts.EXPLAIN_SYSTEM),
                HumanMessage(
                    prompts.EXPLAIN_USER.format(
                        question=question,
                        sql=sql,
                        row_count=len(results["rows"]),
                        truncated_note=truncated_note,
                        results=results_to_text(results),
                    )
                ),
            ]
        )
        return _message_text(response.content).strip()

    def explain_query(self, question: str, sql: str, breakdown: str) -> str:
        """Plain-English account of how ``sql`` computes the answer, grounded in
        the deterministic ``breakdown`` parsed from that same SQL."""
        response = self._chat.invoke(
            [
                SystemMessage(prompts.EXPLAIN_QUERY_SYSTEM),
                HumanMessage(prompts.EXPLAIN_QUERY_USER.format(
                    question=question, sql=sql, breakdown=breakdown)),
            ]
        )
        return _message_text(response.content).strip()


def describe_llm_error(exc: Exception) -> str:
    """Map an LLM exception to a short, user-safe message (never the raw text,
    which could include request details)."""
    text = f"{type(exc).__name__} {exc}".lower()
    if "ratelimit" in text or "resource_exhausted" in text or "429" in text:
        return ("The Gemini API rate limit or quota was reached. "
                "Please wait a minute and try again.")
    if any(k in text for k in ("permission_denied", "unauthenticated", "api key", "api_key",
                               "401", "403")):
        return "The Gemini API rejected the request. Check GEMINI_API_KEY in your .env file."
    if "unavailable" in text or "overloaded" in text or "503" in text:
        return ("The Gemini model is temporarily overloaded (high demand on Google's side). "
                "Please try again shortly, or set GEMINI_MODEL to another model.")
    if "not_found" in text or "notfound" in text or "404" in text:
        return "The configured Gemini model is not available. Check GEMINI_MODEL in your .env file."
    if "timeout" in text or "deadline" in text:
        return "The Gemini API timed out. Please try again."
    return "The language model could not be reached or returned an invalid response."


def _message_text(content: Any) -> str:
    """Chat content can be a string or a list of content blocks."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)
