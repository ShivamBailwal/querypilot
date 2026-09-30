"""SQL validator: read-only single statements over known tables only."""

from __future__ import annotations

import pytest

from agent.sql_validator import validate_sql

TABLES = {"customers", "products", "orders", "order_items"}


@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM customers",
        "select count(*) from orders where status = 'completed'",
        "SELECT c.city, COUNT(*) AS n FROM customers c GROUP BY c.city ORDER BY n DESC LIMIT 5",
        """SELECT p.category, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
           FROM order_items oi JOIN orders o ON o.order_id = oi.order_id
           JOIN products p ON p.product_id = oi.product_id
           WHERE o.status = 'completed' GROUP BY p.category ORDER BY revenue DESC""",
        """WITH monthly AS (
               SELECT strftime('%Y-%m', o.order_date) AS month,
                      SUM(oi.quantity * oi.unit_price) AS revenue
               FROM orders o JOIN order_items oi ON oi.order_id = o.order_id
               WHERE o.status = 'completed' GROUP BY month)
           SELECT * FROM monthly ORDER BY month""",
        "SELECT product_name FROM products WHERE product_id NOT IN (SELECT product_id FROM order_items)",
        "SELECT city FROM customers UNION SELECT category FROM products",
        "SELECT 'please delete this' AS note FROM customers LIMIT 1",  # keyword inside a string
        "SELECT * FROM orders;",  # one trailing semicolon is fine
    ],
)
def test_valid_select_queries_pass(query: str) -> None:
    result = validate_sql(query, TABLES)
    assert result.is_valid, result.errors


@pytest.mark.parametrize(
    "query",
    [
        "INSERT INTO customers VALUES (100, 'Eve', 'Pune', '2025-01-01')",
        "UPDATE orders SET status = 'completed'",
        "DELETE FROM order_items",
        "DROP TABLE customers",
        "ALTER TABLE customers ADD COLUMN age INTEGER",
        "CREATE TABLE evil (id INTEGER)",
        "REPLACE INTO products VALUES (1, 'x', 'y', 1.0)",
        "ATTACH DATABASE 'other.db' AS other",
        "DETACH DATABASE other",
        "PRAGMA foreign_keys = OFF",
        "VACUUM",
        "REINDEX",
        "BEGIN TRANSACTION",
        "CREATE TEMP VIEW v AS SELECT * FROM customers",
        "WITH x AS (SELECT 1) DELETE FROM orders",
        "INSERT INTO orders SELECT * FROM orders",
    ],
)
def test_write_and_destructive_statements_are_rejected(query: str) -> None:
    result = validate_sql(query, TABLES)
    assert not result.is_valid
    assert result.errors


@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM customers; DROP TABLE customers",
        "SELECT 1; SELECT 2",
        "SELECT * FROM orders; -- harmless?\nDELETE FROM orders",
        "SELECT * FROM orders /* comment */ ; UPDATE orders SET status='x'",
    ],
)
def test_multiple_statements_are_rejected(query: str) -> None:
    result = validate_sql(query, TABLES)
    assert not result.is_valid


@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM sqlite_master",
        "SELECT * FROM employees",
        "SELECT * FROM main.customers",
        "SELECT load_extension('evil.dll')",
        "SELECT randomblob(1000000000)",
    ],
)
def test_unknown_tables_and_dangerous_functions_are_rejected(query: str) -> None:
    assert not validate_sql(query, TABLES).is_valid


@pytest.mark.parametrize(
    "query",
    ["", "   ", None, "SELEC * FRM customers", "SELECT * FROM (",
     pytest.param("x" * 6000, id="too-long")],
)
def test_empty_or_malformed_sql_is_rejected(query) -> None:
    result = validate_sql(query, TABLES)
    assert not result.is_valid
    assert result.message


@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM pragma_table_info('customers')",
        "SELECT name FROM pragma_table_info('orders') AS p",
        "SELECT * FROM PRAGMA_TABLE_INFO('customers')",
        "SELECT * FROM pragma_database_list()",
        "SELECT * FROM pragma_database_list",
        "SELECT * FROM pragma_index_list('orders')",
        "SELECT * FROM pragma_foreign_key_list('orders')",
        "SELECT * FROM pragma_compile_options",
        "SELECT * FROM customers WHERE city IN (SELECT name FROM pragma_table_info('customers'))",
        "SELECT c.* FROM customers c JOIN pragma_table_info('orders') p ON p.name = c.city",
        "WITH cols AS (SELECT name FROM pragma_table_info('orders')) SELECT * FROM cols",
        "SELECT (SELECT COUNT(*) FROM pragma_table_xinfo('customers')) AS n FROM customers",
    ],
)
def test_pragma_table_valued_functions_are_rejected(query: str) -> None:
    result = validate_sql(query, TABLES)
    assert not result.is_valid, query


def test_harmless_table_valued_functions_still_pass() -> None:
    # Guard against over-blocking: only pragma_* functions are targeted.
    assert validate_sql("SELECT value FROM json_each('[1, 2, 3]')", TABLES).is_valid


def test_allowlist_comes_from_the_connected_database() -> None:
    custom = {"books", "authors"}
    assert validate_sql("SELECT title FROM books", custom).is_valid
    assert not validate_sql("SELECT * FROM orders", custom).is_valid
    assert not validate_sql("SELECT title FROM books", TABLES).is_valid
