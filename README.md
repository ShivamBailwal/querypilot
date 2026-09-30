# QueryPilot — Agentic AI Data Analyst (Phase 1)

QueryPilot turns natural-language questions into SQL, runs them against a real
SQLite sales database, and explains the actual results. The generated SQL and the
raw result table are always shown, so you can see how each answer was produced.

## Architecture

```
Streamlit UI (app.py)
      │ question
      ▼
LangGraph workflow (agent/graph.py)
  inspect_schema ─► generate_sql ─► validate_sql ─► execute_sql ─► format_answer ─► END
                        │                │               │
                        │           invalid          SQL error
                        │                ▼               ▼
                        │           correct_sql ◄────────┘   (while retry_count < MAX_RETRIES)
                        │                └──► validate_sql
                        └─ cannot answer ─► END      retries exhausted / LLM error ─► handle_failure ─► END
```

How the pieces work together:

| Component | Role |
|---|---|
| **SQLite** | Stores the sample data (`data/querypilot.db`). |
| **SQLAlchemy** | Creates the tables, inspects the live schema (tables, columns, PKs, FKs, distinct values) for the prompt, and provides the read-only engine used for execution. |
| **LLM (Gemini via `langchain-google-genai`)** | Gets the question + schema and returns a structured `SQLDraft` (`answerable`, `sql`, `reason`). It is also asked to fix failed SQL and to explain the returned rows. It never sees or invents data except the rows that were actually returned. |
| **LangGraph** | Orchestrates the steps as an explicit state machine (`agent/state.py`), with conditional edges for validation failure, execution errors, success, "cannot answer", and exhausted retries. |
| **sqlglot** | Parses the generated SQL into an AST for structural validation. |
| **Streamlit + Altair** | Shows the answer, KPI cards, an automatically chosen chart, the result table, the SQL, and any errors. |

### Project layout

```
app.py                  Streamlit UI
analytics/
  insights.py           KPI + chart selection from query results (pure functions)
  charts.py             Altair chart builders (validated palette, mark specs)
  explain.py            Plain-language breakdown of the executed SQL (pure, parsed not run)
.streamlit/config.toml  Theme
config.py               Settings from environment / .env
database/
  init_db.py            Table definitions + deterministic seed data + CLI
  connection.py         Read-only/writable engines, upload validation + hardening
  schema.py             Live schema description for the LLM prompt (time-bounded)
  sources.py            Sample + uploaded databases: ConnectedDatabase lifecycle
agent/
  state.py              AgentState TypedDict
  prompts.py            Prompt templates
  llm.py                LLMService protocol + Gemini implementation
  sql_validator.py      AST-based SQL validation
  tools.py              Sandboxed read-only query execution
  graph.py              LangGraph nodes, routers, run_question()
tests/
  test_database.py      Init, relationships, idempotency, FK/read-only enforcement
  test_sql_validator.py Allowed/forbidden SQL
  test_agent.py         Graph paths with a fake LLM + executor safety
  test_uploads.py       Uploaded databases: validation, cleanup, schema, agent, safety
  test_app.py           Headless UI tests (incl. connect/switch/disconnect, sessions, dashboard)
  test_analytics.py     KPI/chart selection, formatting, chart specs
  test_explain.py       Query breakdown: joins, filters, formulas, CTEs, fallbacks
  test_llm_integration.py  Live Gemini tests (opt-in)
```

## Sample database

`python -m database.init_db` creates `data/querypilot.db` (override with `QUERYPILOT_DB_PATH`).

| Table | Rows | Notes |
|---|---|---|
| `customers` | 50 | 8 cities, uneven distribution, signup dates 2024-01 → 2025-09 |
| `products` | 25 | 5 categories; 3 products are never ordered |
| `orders` | 420 | Jan–Dec 2025, festive-season spike; completed / cancelled / pending (pending only in the last ~45 days) |
| `order_items` | 782 | 1–4 products per order; `unit_price` is the price paid (≈15% have a 10% discount) |

* Primary keys, foreign keys, and CHECK constraints (status values, positive quantity/price) are defined.
* `PRAGMA foreign_keys=ON` is set on every connection (SQLite has it off by default).
* Data comes from a fixed random seed, so every build is identical.
* Running the script again does **not** duplicate data: seeding only happens when the tables are empty. Use `--reset` to drop and rebuild.
* "Revenue" in the prompt means `SUM(order_items.quantity * order_items.unit_price)` over completed orders.

## Analytics dashboard

Every successful answer is presented as a small dashboard, built only from the rows
the protected executor returned. The UI layer never runs extra SQL.

1. **Answer**: the agent's natural-language explanation, unchanged.
2. **KPI cards** (up to three), each derived directly from the result:
   * one-row results show their values (e.g. *Total revenue*), measures first;
   * rankings show the top (or, for ascending queries, lowest) item and, when it is
     safe, a combined total; time series show the latest and peak periods;
   * *Rows returned* is always available.
3. Three tabs:
   * **Chart & data**: a chart chosen automatically from the column types (bar, line,
     donut, or scatter), or no chart when one would be ambiguous or misleading; then the
     results table with the exact returned values (formatting never changes them), the row
     count, the execution time, and a *Truncated* badge when the row limit was reached.
   * **How it was calculated**: see *Query explanation* below.
   * **SQL**: the query exactly as executed.
4. Expandable **Technical details** (status, attempts, validation result, rows, timing,
   chart decision, database).

## Query explanation

The **How it was calculated** tab explains why the number on screen is what it is, in three parts:

1. **Breakdown of the executed SQL** (`analytics/explain.py`). The validated SQL is parsed with
   sqlglot (never re-executed) and every element becomes a plain sentence next to the SQL
   fragment it came from:
   * **Data read:** tables, with aliases resolved (`oi` → `order_items`).
   * **Joins:** how tables were combined, and what that means for which rows are kept
     (inner vs. left join).
   * **Filters:** WHERE and HAVING conditions (`=`, `IN`, `BETWEEN`, `LIKE`, `IS NULL`, `OR`
     groups, subqueries).
   * **Grouping:** one row per what.
   * **Columns calculated:** each output column's formula, e.g. *revenue = sum of
     order_items.quantity × order_items.unit_price, rounded to 2 decimals*, including
     `COUNT(DISTINCT)`, averages, month/year extraction, running totals.
   * **Sorting and limits.**
   * **Steps:** each CTE as its own step.

   Sentences are generated only from nodes of the query's syntax tree, so the explanation
   cannot mention a join, filter, or formula the query doesn't contain. Constructs without a
   rule are shown as their SQL fragment instead of being paraphrased, and queries the parser
   can't break down (e.g. `UNION`) say so and point to the SQL tab.
2. **Checks performed:** what the pipeline actually verified for this result. Validation
   passed, read-only execution with the authorizer, time limit and row cap, the row count and
   timing, whether the row cap was hit, and any self-corrections. On the sample database it
   also lists the business definitions the model was given (e.g. revenue counts completed
   orders only).
3. **Explain in plain English** (optional): one click asks Gemini for a 3–5 sentence narrative
   for non-technical readers. It is grounded only in the breakdown above, is told not to add
   facts or numbers, and is labelled as an AI summary. It costs a request only when clicked;
   the text is reused for that result and cleared when you ask a new question or switch
   databases. All text from the database or the model is escaped before display.

How results are interpreted (`analytics/insights.py`, pure functions):

* **Column types** come from the actual values: numbers, ISO dates/months/datetimes, years
  (only in a column named like *year*), text, and identifiers (`id`, `*_id`, `CustomerID`,
  codes). Identifiers are used as labels and never plotted as measures.
* **Column meaning** comes from parsing (never executing) the validated SQL: whether a
  column is a `SUM`/`COUNT` per group, a `COUNT(DISTINCT ...)`, an `AVG`, a window function,
  or a plain value. A combined total is only shown for per-group `SUM`/`COUNT` (groups are
  disjoint), never for distinct counts, averages, ratios, running totals, or values from a CTE
  or `SELECT *` whose meaning cannot be verified. Averages are only shown for plain row
  values from a non-`LIMIT` query. Truncated results get only the row count.
* **Formatting**: thousands separators, two decimals for money-like names, `%` only for
  columns named *percent/pct* already in percent units, compact `1.23M` in cards. No
  currency symbol is ever shown, because a database's currency is unknown.

Chart rules:

| Result shape | Chart |
|---|---|
| Date/month/year column + one measure | Line, in time order (up to 6 series when a text column splits it) |
| One label per row + a measure | Horizontal bar, ranked by value (or in the query's order when it sorts by the label or measure); top 15 shown, with a note |
| Same, and the question asks for a *share/breakdown/distribution*, the measure is a per-group `SUM`/`COUNT`, 2-6 non-negative slices, no `LIMIT` | Donut |
| Two measures, no labels | Scatter |
| One row, no rows, text only, repeated labels, too many series, ambiguous columns | Table only, with the reason shown |

Rows with missing values are left out of the chart (and the chart says so); they stay in
the table. If a chart fails to render, the table is still shown with a short notice.

Design: the theme (`.streamlit/config.toml`) uses navy surfaces with violet and cyan accents.
The six chart colors were checked with a palette validator (lightness band, chroma, color-blind
separation, and contrast) against both app surfaces, and pass with no warnings. Single-series
charts use one color, colors follow the series rather than its rank, and there is never a
second y-axis.

## Using your own SQLite database

The app starts on the sample database. To query your own data, use the
**Database** section in the sidebar:

1. Upload a `.db`, `.sqlite`, or `.sqlite3` file (default limit 50 MB, `QUERYPILOT_MAX_UPLOAD_MB`).
2. Click **Connect database**. The file is validated and inspected; the sidebar then shows
   the file name and every table with its row count. Sample example questions are hidden.
3. Ask questions as usual. The same LangGraph workflow, validator, and sandboxed executor
   are used, with the uploaded database's own schema.
4. Click **Disconnect** to return to the sample database.

How a connection is made (`database/sources.py`):

* The extension is checked, but the file must also start with the SQLite magic header;
  renamed CSV/PDF files, encrypted databases, empty files, and oversized files are rejected.
* The bytes are written to a **private copy** in `data/uploads/` with a random name (the
  uploaded filename is never used as a path), and the copy is set read-only on disk.
* The copy is opened with `mode=ro&immutable=1`, `query_only`, `trusted_schema=OFF`,
  defensive mode, and `cell_size_check=ON`. `immutable=1` is only used because QueryPilot
  wrote the copy once and nothing modifies it while it is connected. The sample database
  is never opened as immutable, because `init_db` may rebuild it.
* `PRAGMA quick_check`, table discovery, row counts, and schema sampling run under a single
  15-second budget. Databases with no tables, only empty tables, more than 100 tables, or a
  schema too large for the prompt are rejected with a clear message.
* If **any** step fails, the engine is disposed and the copy deleted. The previously active
  database (sample or upload) stays active, and an invalid upload is never shown as connected.
* `quick_check` proves the file is structurally intact, **not** that its contents are
  trustworthy. Table names, column names, and values are treated as hostile input by
  every safety layer below, and the prompt tells the model to treat them as data.

Lifecycle and isolation (`app.py`):

* The sample engine, graph, and LLM client are shared `st.cache_resource` singletons
  (read-only, so safe to share). Uploaded databases are stored **per browser session** in
  `st.session_state`, never in the shared cache, so one user's upload is invisible to others.
* Each database gets its own graph, built from its own engine, schema, table allowlist,
  and notes. The sample's business definitions (`SAMPLE_DATABASE_NOTES`, e.g. what
  "revenue" means) are only added for the sample database; uploads get generic SQLite rules.
* Switching or disconnecting explicitly calls `ConnectedDatabase.close()` (engine disposed,
  copy deleted) and clears the previous answer, SQL, errors, and results. Results are also
  tagged with the database's `connection_id` and are only rendered while it is active.
  Garbage collection and a 24-hour sweep of `data/uploads/` are fallbacks only, for
  sessions that vanish without disconnecting.

## SQL safety (defense in depth)

Generated SQL is treated as untrusted. A query must pass **every** layer:

1. **AST validation** (`agent/sql_validator.py`, sqlglot, SQLite dialect)
   - exactly one statement; the root must be `SELECT` or a set operation (`UNION`, etc.; CTEs allowed)
   - no write, DDL, or admin nodes anywhere in the tree (INSERT, UPDATE, DELETE, REPLACE, DROP, CREATE, ALTER, ATTACH, DETACH, PRAGMA, transactions, VACUUM/REINDEX…)
   - only known tables (allowlist built from the *connected* database's live schema; blocks `sqlite_master` and `db.table` references)
   - blocked functions (`load_extension`, `readfile`, `randomblob`, …) and all `pragma_*` table-valued functions (e.g. `pragma_table_info('t')`)
   - a keyword scan on SQL with comments and strings removed, as a backstop
2. **Read-only connection** (`database/connection.py`): SQLite URI `mode=ro`, plus `PRAGMA query_only=ON`, `trusted_schema=OFF` (SQLite 3.31+), defensive mode (Python 3.12+), `cell_size_check=ON`, and no memory-mapped I/O. For uploads, every connection verifies these settings took effect, and uploads are refused on a SQLite/Python build that cannot apply them.
3. **SQLite authorizer** (`agent/tools.py`): during statement compilation, only SELECT, READ, FUNCTION, and RECURSIVE actions are allowed. Everything else, including reads of `sqlite_*` tables, is denied, even if the validator were bypassed (covered by tests).
4. **Timeout**: a progress handler aborts queries that run longer than `QUERYPILOT_QUERY_TIMEOUT_SECONDS` (default 5s).
5. **Row cap**: at most `QUERYPILOT_MAX_ROWS` (default 200) rows are fetched; the UI notes when results are truncated.
6. **Error hygiene**: users and the LLM see only the one-line SQLite message (e.g. `no such column: x`) or a mapped LLM error (rate limit, bad key, unknown model). Stack traces stay in the server log, and API keys and file paths are never shown.

Queries that fail validation are **never executed**. They go to `correct_sql` until `QUERYPILOT_MAX_RETRIES` (default 2, max 5) is used up, then the run fails with an explanation.

## Setup (Windows PowerShell)

```powershell
# 1. Create and activate a virtual environment (Python 3.12 recommended)
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
# If activation is blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# 2. Install dependencies
python -m pip install -r requirements.txt

# 3. Configure your API key
Copy-Item .env.example .env
notepad .env        # set GEMINI_API_KEY=...

# 4. Create the sample database (safe to re-run; add --reset to rebuild)
python -m database.init_db

# 5. Run the unit tests (offline, no API calls)
python -m pytest -v

# 6. Optional: run the live Gemini tests (uses your API quota, ~4 minutes on free tier)
python -m pytest -m integration -v

# 7. Launch the app
streamlit run app.py
```

Then open http://localhost:8501.

### Tests

* **Unit tests** (default `pytest` run): use a temporary database and a scripted `FakeLLMService`. They cover DB init and relationships, idempotent seeding, FK and read-only enforcement, validator allow/deny cases, multi-statement rejection, execution errors, correction retries, retry exhaustion, unanswerable questions, executor sandboxing, timeouts, and row caps. Upload tests cover valid, invalid, corrupted, and empty files, cleanup after success and failure, inspection time budgets, schema and prompt correctness for unfamiliar databases, switching, per-session isolation, and `pragma_*` bypass attempts. Analytics tests cover KPI correctness (including the no-double-counting rules), chart selection for categorical, temporal, numeric, and unsupported results, nulls, negatives, large numbers, truncation, chart fallbacks, stale-result clearing across databases, and markdown injection from data. They make no network calls.
* **Integration tests** (`-m integration`): call the real Gemini API, check answers against SQL computed directly in the test, and are skipped automatically if `GEMINI_API_KEY` is not set. They are paced (`QUERYPILOT_INTEGRATION_DELAY`, default 30s) to fit free-tier rate limits.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | — (required) | Google AI Studio key |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Any Gemini chat model available to your key |
| `QUERYPILOT_DB_PATH` | `data/querypilot.db` | SQLite file |
| `QUERYPILOT_MAX_RETRIES` | `2` | SQL correction attempts (0–5) |
| `QUERYPILOT_MAX_ROWS` | `200` | Max rows returned |
| `QUERYPILOT_QUERY_TIMEOUT_SECONDS` | `5` | Per-query execution limit |
| `QUERYPILOT_UPLOAD_DIR` | `data/uploads` | Private copies of uploaded databases while connected |
| `QUERYPILOT_MAX_UPLOAD_MB` | `50` | Upload size limit (1-200) |

## Known limitations

* **Free-tier quota and availability**: a question uses 2 LLM calls (3–4 when a correction is needed). During development, the free tier allowed 5 requests per minute and **20 requests per day per model** for `gemini-3.8-flash`, and some models returned `503 UNAVAILABLE` (high demand) for long periods. Both cases are shown in the UI as clear messages, not answers. Quotas are counted per model, so switching `GEMINI_MODEL` (e.g. `gemini-3.5-flash`, `gemini-3.1-flash-lite`) is a practical workaround.
* **Model availability**: Google retires model names for new keys (e.g. `gemini-2.5-flash` returned 404 for the key used in development). If you get a "model not available" message, set `GEMINI_MODEL`.
* **Semantic correctness is not guaranteed**: validation proves a query is *safe*, not that it answers the question correctly. Always check the displayed SQL. Ambiguous questions (e.g. "best customers") may be interpreted differently from what you meant.
* **"Cannot answer" relies on the LLM**: the model decides whether the schema can answer a question. It may sometimes attempt a question it should decline. It cannot fabricate numbers, though, because answers are explained only from returned rows.
* Money is stored as SQLite `REAL` (floating point), and results are rounded for display.
* The sample data is synthetic and covers calendar year 2025 only.
* There is no conversation memory: each question is answered on its own.
* SQLite only. The timeout and authorizer use `sqlite3`-specific hooks.
* **Uploaded databases**:
  * Only tables are queryable. Views in an uploaded file are not listed in the schema or the allowlist.
  * The validator's keyword backstop also scans quoted identifiers, so a table or column whose name
    contains a standalone SQL keyword (e.g. a column named `"Update Date"`) is rejected. This is
    conservative on purpose.
  * The inspection budget is checked between SQLite steps. A single `COUNT(*)` is one SQLite step and
    cannot be interrupted midway; its cost is bounded by the upload size limit.
  * Schema text (names, sample values) from an uploaded file reaches the LLM. A malicious file could
    attempt prompt injection; any SQL it produces still has to pass every safety layer above.
  * An upload stays connected only for its browser session; reloading the page starts on the sample.
* **Analytics dashboard**:
  * Number formatting (money-style decimals, `%`) is inferred from column names. It never changes
    values or adds a currency symbol, but an unusual name can get plain formatting.
  * A chart shows one measure. When a result has several, the others are named under the
    chart and shown in the table (no dual axes by design).
  * Lines connect the periods that were returned. If the data has no row for a period, the line
    spans the gap rather than dropping to zero, because a missing row is not known to mean zero.
  * Chart colors are tuned and validated for the dark theme in `.streamlit/config.toml`. Switching
    Streamlit to its light theme keeps the dark-tuned chart colors.
