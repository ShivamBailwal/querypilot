"""Static validation of LLM-generated SQL.

Generated SQL is treated as untrusted input. The query is parsed into an AST
with sqlglot (SQLite dialect) and checked structurally, rather than relying
only on keyword matching, which is easy to fool with comments, casing or
string literals. A keyword scan still runs as an extra backstop.

This is only the *first* safety layer. Even a query that passes here is
executed on a read-only connection with an SQLite authorizer, a timeout and a
row cap (see ``agent/tools.py``).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

logger = logging.getLogger(__name__)

# Statement roots we accept: plain SELECTs (optionally with CTEs) and set ops.
ALLOWED_ROOT_TYPES: tuple[type[exp.Expression], ...] = (exp.Select, exp.SetOperation)

# Any of these nodes anywhere in the tree means the query is not read-only.
FORBIDDEN_NODE_TYPES: tuple[type[exp.Expression], ...] = (
    exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Drop, exp.Create,
    exp.Alter, exp.TruncateTable, exp.Attach, exp.Detach, exp.Pragma,
    exp.Command, exp.Transaction, exp.Commit, exp.Rollback, exp.Analyze,
    exp.Set, exp.Use, exp.Copy, exp.LoadData, exp.Grant,
)

# Functions that could reach outside the database or exhaust resources.
FORBIDDEN_FUNCTIONS = {
    "load_extension", "readfile", "writefile", "edit", "fts3_tokenizer",
    "randomblob", "zeroblob", "sqlite_offset",
}

# Backstop keyword scan (run on SQL with comments and string literals removed).
FORBIDDEN_KEYWORDS = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|"
    r"vacuum|reindex|truncate|grant|revoke|begin|commit|rollback|savepoint|"
    r"release|upsert|merge)\b",
    re.IGNORECASE,
)

MAX_SQL_LENGTH = 5000


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)

    @property
    def message(self) -> str:
        return "; ".join(self.errors)


def _strip_comments_and_strings(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    sql = re.sub(r"'(?:[^']|'')*'", "''", sql)
    return sql


def validate_sql(sql: str | None, allowed_tables: set[str] | list[str]) -> ValidationResult:
    """Return whether ``sql`` is a single, read-only query over known tables."""
    if sql is None or not sql.strip():
        return ValidationResult(False, ["No SQL query was produced."])
    if len(sql) > MAX_SQL_LENGTH:
        return ValidationResult(False, [f"Query is longer than {MAX_SQL_LENGTH} characters."])

    allowed = {t.lower() for t in allowed_tables}

    try:
        statements = [s for s in sqlglot.parse(sql, dialect="sqlite") if s is not None]
    except ParseError as exc:
        # Keep only the first line of the parser message; it is safe to show.
        return ValidationResult(False, [f"SQL could not be parsed: {str(exc).splitlines()[0]}"])
    except Exception:  # sqlglot can raise other errors on garbage input
        logger.exception("Unexpected SQL parser failure")
        return ValidationResult(False, ["SQL could not be parsed."])

    if len(statements) != 1:
        return ValidationResult(
            False, [f"Exactly one SQL statement is allowed (found {len(statements)})."]
        )

    tree = statements[0]
    errors: list[str] = []

    if not isinstance(tree, ALLOWED_ROOT_TYPES):
        errors.append(
            f"Only read-only SELECT queries are allowed (got {tree.key.upper()} statement)."
        )

    for node in tree.walk():
        if isinstance(node, FORBIDDEN_NODE_TYPES):
            errors.append(f"Forbidden operation in query: {node.key.upper()}.")
            break

    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    for table in tree.find_all(exp.Table):
        name = table.name.lower()
        if not name:
            continue
        if table.args.get("db") or table.args.get("catalog"):
            errors.append(f"Schema-qualified table references are not allowed: {table.sql()}.")
        elif name not in allowed and name not in cte_names:
            errors.append(f"Unknown or disallowed table: {table.name}.")

    for func in tree.find_all(exp.Func):
        func_name = (
            func.name if isinstance(func, exp.Anonymous) else func.sql_name()
        ).lower()
        # pragma_* are table-valued PRAGMA functions (e.g. FROM pragma_table_info('t')).
        if func_name in FORBIDDEN_FUNCTIONS or func_name.startswith("pragma_"):
            errors.append(f"Function not allowed: {func_name}.")

    keyword = FORBIDDEN_KEYWORDS.search(_strip_comments_and_strings(sql))
    if keyword:
        errors.append(f"Forbidden keyword in query: {keyword.group(0).upper()}.")

    # De-duplicate while preserving order.
    errors = list(dict.fromkeys(errors))
    return ValidationResult(not errors, errors)
