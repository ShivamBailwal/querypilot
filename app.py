"""QueryPilot Streamlit UI.

Run with:  streamlit run app.py

Database lifecycle
------------------
* The sample database, its graph and the LLM client are ``st.cache_resource``
  singletons. They are read-only, so sharing them between sessions is safe.
* An uploaded database belongs to one browser session, so it lives in
  ``st.session_state``, never in ``st.cache_resource`` (which every user shares).
  Its graph is built from its own engine and schema when it connects and is
  stored in the same slot, so the graph can never point at another database.
* Every switch explicitly closes the previous upload (engine disposed,
  private copy deleted) and clears the last answer, SQL, error and results.
  Results are also tagged with the ``connection_id`` of the database that
  produced them and are only rendered while that same database is active.
"""

from __future__ import annotations

import logging
import re

import pandas as pd
import streamlit as st

from agent.graph import build_graph, run_question
from agent.llm import GeminiLLMService, describe_llm_error
from agent.prompts import SAMPLE_DATABASE_NOTES
from analytics.explain import SECTIONS, QueryBreakdown, breakdown_to_text, explain_sql
from analytics.insights import analyze_results
from config import Settings, get_settings
from database.connection import DatabaseConnectionError, DatabaseNotFoundError
from database.sources import (
    SUPPORTED_EXTENSIONS,
    ConnectedDatabase,
    open_sample_database,
    open_uploaded_database,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("querypilot.app")

EXAMPLE_QUESTIONS = [
    "What is the total revenue from completed orders?",
    "Which product category generated the highest revenue?",
    "Who are the top five customers by spending?",
    "What was the monthly revenue trend?",
    "Which city has the most customers?",
    "Which products have never been ordered?",
    "Compare revenue across the last three months available in the dataset.",
]

# Session-state keys.
CUSTOM_DB_KEY = "custom_db"            # (ConnectedDatabase, graph) of the uploaded database
LAST_STATE_KEY = "last_state"          # (connection_id, final agent state)
UPLOADER_COUNTER_KEY = "db_uploader_n"  # changing the uploader's key clears it
FLASH_KEY = "db_flash"                 # one-shot notice shown after a rerun
QUERY_EXPLANATION_KEY = "query_explanation"  # (connection_id, question, sql, AI narrative)

st.set_page_config(page_title="QueryPilot", page_icon="🧭", layout="wide")


@st.cache_resource(show_spinner=False)
def load_llm() -> GeminiLLMService:
    settings = get_settings()
    return GeminiLLMService(api_key=settings.gemini_api_key or "", model=settings.gemini_model)


def make_graph(db: ConnectedDatabase, settings: Settings):
    """The one LangGraph workflow, bound to ``db``'s engine and schema."""
    return build_graph(
        load_llm(), db.engine,
        max_retries=settings.max_retries,
        max_rows=settings.max_rows,
        timeout_seconds=settings.query_timeout_seconds,
        database_schema=db.schema,
        allowed_tables=db.table_names,
        schema_notes=SAMPLE_DATABASE_NOTES if db.is_sample else "",
    )


@st.cache_resource(show_spinner=False)
def load_sample():
    """Build the sample engine + graph once per Streamlit server process."""
    settings = get_settings()
    db = open_sample_database(settings.db_path)
    return db, make_graph(db, settings)


def active_database():
    """(ConnectedDatabase, graph) for this session: the upload, else the sample."""
    custom = st.session_state.get(CUSTOM_DB_KEY)
    if custom is not None:
        if custom[0].is_open:
            return custom
        st.session_state.pop(CUSTOM_DB_KEY, None)
    return load_sample()


def switch_database(new_custom) -> None:
    """Make ``new_custom`` (or the sample, if None) active and close the old upload."""
    old = st.session_state.pop(CUSTOM_DB_KEY, None)
    if new_custom is not None:
        st.session_state[CUSTOM_DB_KEY] = new_custom
    if old is not None:
        old[0].close()
    st.session_state.pop(LAST_STATE_KEY, None)
    st.session_state.pop(QUERY_EXPLANATION_KEY, None)
    st.session_state[UPLOADER_COUNTER_KEY] = st.session_state.get(UPLOADER_COUNTER_KEY, 0) + 1


def connect_upload(uploaded, settings: Settings) -> str | None:
    """Validate, inspect and activate an uploaded file. Returns an error message,
    or None on success. On failure the current database stays active."""
    try:
        new_db = open_uploaded_database(
            uploaded.getvalue(), uploaded.name, settings.upload_dir,
            max_bytes=settings.max_upload_mb * 1024 * 1024,
        )
    except DatabaseConnectionError as exc:
        return str(exc)
    except Exception:
        logger.exception("Unexpected error while connecting an uploaded database")
        return "An unexpected error occurred while opening the database."
    try:
        graph = make_graph(new_db, settings)
    except Exception:
        new_db.close()
        logger.exception("Failed to build the agent for an uploaded database")
        return "The database opened, but the agent could not be prepared for it."
    switch_database((new_db, graph))
    return None


def render_setup_problem() -> bool:
    """Show setup instructions and return True if the app cannot run."""
    settings = get_settings()
    problems = []
    if not settings.db_path.exists():
        problems.append("The sample database has not been created yet. Run:\n\n"
                        "```powershell\npython -m database.init_db\n```")
    if not settings.gemini_api_key:
        problems.append("`GEMINI_API_KEY` is not set. Copy `.env.example` to `.env` "
                        "and add your key, then restart the app.")
    for p in problems:
        st.error(p)
    return bool(problems)


def render_database_panel(db: ConnectedDatabase, settings: Settings) -> None:
    """Sidebar section: status, table summary, upload/connect, disconnect."""
    st.header("Database")
    flash = st.session_state.pop(FLASH_KEY, None)
    if flash:
        st.toast(flash, icon="✅")

    # User-controlled text (file and table names) is rendered with st.text,
    # which never interprets markdown.
    if db.is_sample:
        st.markdown(":blue[●] **Sample sales database** · default")
    else:
        st.markdown(":green[●] **Connected** · uploaded file")
        st.text(db.display_name)

    total_rows = sum(t.row_count for t in db.tables)
    with st.expander(f"{len(db.tables)} tables · {total_rows:,} rows", expanded=not db.is_sample):
        st.text("\n".join(f"{t.name}  ·  {t.row_count:,} rows" for t in db.tables))

    counter = st.session_state.get(UPLOADER_COUNTER_KEY, 0)
    uploaded = st.file_uploader(
        "Upload a SQLite database",
        type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
        key=f"db_upload_{counter}",
        max_upload_size=settings.max_upload_mb,
        help=f"{', '.join(SUPPORTED_EXTENSIONS)} up to {settings.max_upload_mb} MB. "
             "The file is opened read-only and is never modified.",
    )
    connect = st.button("Connect database", type="primary", width="stretch",
                        disabled=uploaded is None)
    disconnect = False
    if not db.is_sample:
        disconnect = st.button("Disconnect", width="stretch",
                               help="Close this database and return to the sample.")

    if connect and uploaded is not None:
        with st.spinner("Validating and inspecting the database…"):
            error = connect_upload(uploaded, settings)
        if error:
            st.error(f"Could not connect this database. {error}")
            st.caption("Still using the " + ("sample database." if db.is_sample
                                             else "previously connected database."))
            return
        st.session_state[FLASH_KEY] = "Database connected."
        st.rerun()
    if disconnect:
        switch_database(None)
        st.session_state[FLASH_KEY] = "Disconnected. Back to the sample database."
        st.rerun()


# ------------------------------------------------------------------ results
# Everything below renders the final agent state only. It never runs SQL:
# KPIs and charts are computed from the rows the protected executor returned.

_MD_SPECIAL = re.compile(r"([\\`*_\[\]<>:$~|#])")


def md(text: object) -> str:
    """Escape data-derived text (column names, labels from uploaded databases)
    before it reaches any widget that renders markdown, such as st.metric."""
    return _MD_SPECIAL.sub(r"\\\1", str(text))


def _display_frame(results: dict) -> pd.DataFrame:
    """Result rows as a DataFrame; duplicate headers are disambiguated for display."""
    seen: dict[str, int] = {}
    headers = []
    for name in results["columns"]:
        seen[name] = seen.get(name, 0) + 1
        headers.append(name if seen[name] == 1 else f"{name} ({seen[name]})")
    return pd.DataFrame(results["rows"], columns=headers)


def render_kpis(kpis) -> None:
    if not kpis:
        return
    for column, kpi in zip(st.columns(3), kpis):
        column.metric(
            md(kpi.label), md(kpi.value),
            delta=md(kpi.detail) if kpi.detail else None,
            delta_color="off", delta_arrow="off",  # context line, not a change indicator
            help=md(kpi.help) if kpi.help else None, border=True,
        )


def render_chart(insights, results: dict) -> None:
    """Draw the selected chart; any failure degrades to the table, never a crash."""
    spec = insights.chart
    if spec.kind == "table":
        return
    try:
        from analytics.charts import build_chart  # Altair is optional for the rest of the app
        chart = build_chart(spec, results["rows"], insights.profiles)
        if chart is None:
            return
        with st.container(border=True):
            st.markdown(f"**{md(spec.title)}**")
            st.altair_chart(chart, theme=None, width="stretch", alt=spec.title)
            for note in spec.notes:
                st.caption(md(note))
    except Exception:
        logger.exception("Chart rendering failed; falling back to the table")
        st.info("A chart couldn't be drawn for this result. All values are in the table below.")


def render_table(results: dict, insights) -> None:
    rows = results["rows"]
    with st.container(border=True):
        st.markdown("**Results**")
        meta = [f"{len(rows):,} row{'' if len(rows) == 1 else 's'}"]
        elapsed = results.get("elapsed_ms")
        if elapsed is not None:
            meta.append("executed in <1 ms" if elapsed < 1 else f"executed in {elapsed:,.0f} ms")
        st.caption(" · ".join(meta))
        if not rows:
            st.write("The query returned no rows.")
            return
        if results["truncated"]:
            st.markdown(f":orange-badge[Truncated] Showing the first {len(rows):,} rows; "
                        "the query returned more. Narrow the question to see the rest.")

        frame = _display_frame(results)
        config = {}
        if insights is not None:
            for profile, header in zip(insights.profiles, frame.columns):
                if profile.kind == "numeric":
                    config[header] = st.column_config.NumberColumn(format="localized")
                elif profile.kind == "identifier" or profile.granularity == "year":
                    config[header] = st.column_config.Column()  # no thousands separators
        st.dataframe(frame, width="stretch", hide_index=True, column_config=config or None,
                     placeholder="—", height="auto" if len(rows) <= 12 else 420)
        if insights is not None and insights.chart.kind == "table" and len(rows) > 1:
            st.caption(f"No chart: {md(insights.chart.reason)}")


def render_details(state: dict, insights, db: ConnectedDatabase) -> None:
    """Facts the agent actually recorded; no statuses are invented here."""
    status = {"success": "Answered", "cannot_answer": "Not answerable from this database"}
    lines = [f"- **Status:** {status.get(state.get('status'), 'Failed')}"]
    if state.get("generated_sql"):
        lines.append(f"- **Attempts:** {state.get('retry_count', 0) + 1}")
    if state.get("query_results") is not None:
        lines.append("- **SQL validation:** passed (read-only, allowed tables only)")
        results = state["query_results"]
        lines.append(f"- **Rows returned:** {len(results['rows']):,}"
                     + (" (truncated at the row limit)" if results["truncated"] else ""))
        if results.get("elapsed_ms") is not None:
            lines.append(f"- **Execution time:** {results['elapsed_ms']:,.1f} ms")
    elif state.get("error_stage") == "validation":
        lines.append("- **SQL validation:** rejected, so the query was not executed")
    if insights is not None and state.get("query_results") is not None:
        chart = insights.chart
        lines.append("- **Visualization:** " + (f"{chart.kind} chart" if chart.kind != "table"
                                                 else f"table only ({md(chart.reason)})"))
    lines.append(f"- **Database:** {md(db.display_name)}")
    st.markdown("\n".join(lines))


# ------------------------------------------------------- query explanation
def code_span(text: str) -> str:
    """Inline code that survives backticks in identifiers (markdown is not
    interpreted inside a code span, so no other escaping is needed)."""
    text = text.replace("\n", " ")
    fence = "`" * (max((len(run) for run in re.findall(r"`+", text)), default=0) + 1)
    pad = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{fence}{pad}{text}{pad}{fence}"


def _sample_definitions() -> list[str]:
    """SAMPLE_DATABASE_NOTES as one string per bullet."""
    items: list[list[str]] = []
    for line in SAMPLE_DATABASE_NOTES.splitlines():
        if line.startswith("- "):
            items.append([line[2:].strip()])
        elif line.strip() and items:
            items[-1].append(line.strip())
    return [" ".join(parts) for parts in items]


def _render_sections(breakdown: QueryBreakdown) -> None:
    number = 0
    for attr, title in SECTIONS:
        items = getattr(breakdown, attr)
        if not items:
            continue  # a section the query doesn't have is omitted, never invented
        number += 1
        st.markdown(f"**{number}. {title}**")
        st.markdown("\n".join(
            f"- {md(item.plain)}" + (f"  \n  {code_span(item.sql)}" if item.sql else "")
            for item in items))
    for note in breakdown.notes:
        st.caption(md(note))


def render_breakdown(breakdown: QueryBreakdown) -> None:
    if not breakdown.parsed:
        st.info("This query's structure couldn't be broken down automatically. "
                "The SQL tab shows exactly what ran.")
        return
    for number, (name, step) in enumerate(breakdown.steps, start=1):
        with st.container(border=True):
            st.markdown(f"**Step {number}** · intermediate result {code_span(name)}")
            if step.parsed:
                _render_sections(step)
            else:
                st.caption("This step couldn't be broken down; see the SQL tab.")
    if breakdown.steps:
        st.markdown("**Final step**")
    _render_sections(breakdown)


def render_checks(state: dict, db: ConnectedDatabase, settings: Settings) -> None:
    """What the pipeline verified for this result. Facts only, no claim of certainty."""
    results = state["query_results"]
    rows = len(results["rows"])
    elapsed = results.get("elapsed_ms")
    ok = ":material/check_circle:"
    lines = [
        f"{ok} Passed SQL validation: a single read-only `SELECT` that uses only tables "
        "in this database.",
        f"{ok} Ran on a read-only connection with the SQLite authorizer, a "
        f"{settings.query_timeout_seconds:g} s time limit and a {settings.max_rows:,}-row cap.",
        f"{ok} Returned {rows:,} row{'' if rows == 1 else 's'}"
        + (f" in {elapsed:,.0f} ms." if elapsed is not None else "."),
    ]
    if results["truncated"]:
        lines.append(":material/warning: The row cap was reached, so totals and charts cover "
                     "only the rows shown.")
    retries = state.get("retry_count", 0)
    if retries:
        lines.append(f":material/autorenew: The first attempt failed; the agent corrected the "
                     f"query {retries} time(s) before it passed these checks.")
    lines.append(f"{ok} KPI cards and the chart are computed directly from these rows; the "
                 "written answer was generated from them with instructions not to add numbers.")
    st.markdown("  \n".join(lines))
    if db.is_sample:
        st.markdown("**Business definitions applied** (sample database)")
        st.markdown("\n".join(f"- {md(item)}" for item in _sample_definitions()))
        st.caption("These definitions are given to the model with the schema. "
                   "Uploaded databases use generic SQLite rules only.")


def render_ai_explanation(state: dict, db: ConnectedDatabase,
                          breakdown: QueryBreakdown) -> None:
    """On-demand narrative: one Gemini call, only when asked, then reused."""
    sql = state.get("generated_sql") or ""
    question = state.get("question") or ""
    key = (db.connection_id, question, sql)
    stored = st.session_state.get(QUERY_EXPLANATION_KEY)
    st.markdown("**Plain-English summary**")
    if stored is None or stored[:3] != key:
        st.caption("Have the AI narrate this breakdown for a non-technical reader "
                   "(uses one Gemini request).")
        if not st.button("Explain in plain English", key="explain_query",
                         icon=":material/auto_awesome:"):
            return
        with st.spinner("Writing the explanation…"):
            try:
                text = load_llm().explain_query(question, sql, breakdown_to_text(breakdown))
            except Exception as exc:
                logger.exception("Query explanation failed")
                st.error(describe_llm_error(exc))  # mapped, never the raw error text
                return
        stored = (*key, text or "The model returned an empty explanation.")
        st.session_state[QUERY_EXPLANATION_KEY] = stored
    with st.container(border=True):
        st.markdown(md(stored[3]))
        st.caption("AI summary written from the breakdown above. Check it against the SQL tab.")


def render_explanation(state: dict, db: ConnectedDatabase, settings: Settings) -> None:
    try:
        breakdown = explain_sql(state.get("generated_sql"))
    except Exception:
        logger.exception("Query breakdown failed")
        breakdown = QueryBreakdown(parsed=False)
    render_breakdown(breakdown)
    st.divider()
    st.markdown("**Checks performed**")
    render_checks(state, db, settings)
    st.divider()
    render_ai_explanation(state, db, breakdown)


def render_result(state: dict, db: ConnectedDatabase, settings: Settings) -> None:
    status = state.get("status")
    answer = state.get("final_answer") or ""
    results = state.get("query_results")

    if state.get("question"):
        st.caption(f"Question: {md(state['question'])}")
    if status == "success":
        st.success(answer)
    elif status == "cannot_answer":
        st.warning(answer)
        st.caption("The agent decided the database doesn't contain the data needed, "
                   "so no SQL was run and no answer was guessed.")
    else:
        st.error(answer)

    retries = state.get("retry_count", 0)
    if retries:
        st.info(f"The first query attempt failed; the agent corrected it {retries} time(s).")

    insights = None
    if results is not None:
        try:
            insights = analyze_results(results, state.get("generated_sql"),
                                       state.get("question") or "")
        except Exception:
            logger.exception("Result analysis failed; showing the table only")
        if insights is not None:
            render_kpis(insights.kpis)

    sql = state.get("generated_sql")
    if results is not None and status == "success":
        tab_data, tab_how, tab_sql = st.tabs(["Chart & data", "How it was calculated", "SQL"])
        with tab_data:
            if insights is not None:
                render_chart(insights, results)
            render_table(results, insights)
        with tab_how:
            render_explanation(state, db, settings)
        with tab_sql:
            st.code(sql or "", language="sql", wrap_lines=True)
    else:
        if results is not None:
            render_table(results, insights)
        if sql:
            with st.expander("Last attempted SQL", icon=":material/code:"):
                st.code(sql, language="sql", wrap_lines=True)

    if status == "failed" and state.get("error"):
        with st.expander("Error details"):
            st.write(state["error"])

    with st.expander("Technical details", icon=":material/info:"):
        render_details(state, insights, db)


def main() -> None:
    st.title("🧭 QueryPilot")

    if render_setup_problem():
        return

    settings = get_settings()
    try:
        db, graph = active_database()
    except DatabaseNotFoundError as exc:
        st.error(str(exc))
        return
    except Exception:
        logger.exception("Failed to initialise QueryPilot")
        st.error("QueryPilot could not start. Check the server logs for details.")
        return

    if db.is_sample:
        st.caption("Ask questions about the sample sales database in plain English. "
                   "Answers are computed by running SQL against the real data.")
    else:
        st.caption("Ask questions about your uploaded database in plain English. "
                   "Answers are computed by running read-only SQL against its real data.")

    with st.sidebar:
        render_database_panel(db, settings)
        st.divider()
        if db.is_sample:
            st.header("Example questions")
            for q in EXAMPLE_QUESTIONS:
                if st.button(q, width="stretch"):
                    st.session_state["question"] = q
        else:
            st.caption("Ask about the tables listed above, e.g. totals, rankings, "
                       "trends over time, or records matching a condition.")
        st.divider()
        st.caption("Read-only access · single SELECT statements only · "
                   "results capped and time-limited.")

    with st.form("ask"):
        question = st.text_area(
            "Your question", key="question", height=80,
            placeholder=("e.g. Which city has the most customers?" if db.is_sample
                         else "e.g. Which records have the highest values?"),
        )
        submitted = st.form_submit_button("Ask", type="primary")

    if submitted:
        if not question.strip():
            st.warning("Please enter a question.")
            return
        # Drop the previous answer first, so a failed run can never leave an
        # older question's results on screen.
        st.session_state.pop(LAST_STATE_KEY, None)
        st.session_state.pop(QUERY_EXPLANATION_KEY, None)
        with st.spinner("Generating SQL, validating and querying the database…"):
            try:
                state = run_question(graph, question)
            except Exception:
                logger.exception("Unexpected agent failure")
                st.error("Something went wrong while answering. Please try again.")
                return
        st.session_state[LAST_STATE_KEY] = (db.connection_id, state)

    last = st.session_state.get(LAST_STATE_KEY)
    if last is not None and last[0] == db.connection_id:  # never show another database's result
        render_result(last[1], db, settings)
    st.caption(f"Model: {settings.gemini_model}")


main()
