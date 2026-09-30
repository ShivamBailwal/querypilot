"""Deterministic query breakdown: every sentence must come from the SQL itself."""

from __future__ import annotations

import pytest

from analytics.explain import breakdown_to_text, explain_sql

TOP5_SQL = (
    "SELECT p.product_name, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue "
    "FROM order_items oi JOIN orders o ON o.order_id = oi.order_id "
    "JOIN products p ON p.product_id = oi.product_id WHERE o.status = 'completed' "
    "GROUP BY p.product_name ORDER BY revenue DESC LIMIT 5"
)


def _plain(items) -> list[str]:
    return [item.plain for item in items]


def test_top_products_query_is_fully_explained() -> None:
    b = explain_sql(TOP5_SQL)
    assert b.parsed
    assert _plain(b.tables) == ["order_items (as oi)", "orders (as o)", "products (as p)"]
    assert _plain(b.joins) == [
        "Inner join with orders, matching rows where orders.order_id equals "
        "order_items.order_id. Only rows with a match on both sides are kept",
        "Inner join with products, matching rows where products.product_id equals "
        "order_items.product_id. Only rows with a match on both sides are kept",
    ]
    assert _plain(b.filters) == ["orders.status is 'completed'"]
    assert _plain(b.grouping) == ["One row per product_name"]
    assert _plain(b.metrics) == [
        "product_name: taken directly from products",
        "revenue = sum of order_items.quantity × order_items.unit_price, rounded to 2 decimals",
    ]
    assert _plain(b.ordering) == ["Sorted by revenue, descending (highest first)",
                                  "Only the top 5 rows are kept"]
    # Each item also carries the SQL fragment it was derived from.
    assert b.filters[0].sql == "o.status = 'completed'"
    assert b.ordering[1].sql == "LIMIT 5"


def test_left_join_explains_which_rows_are_kept() -> None:
    b = explain_sql("SELECT c.city, COUNT(o.order_id) AS orders FROM customers c "
                    "LEFT JOIN orders o ON o.customer_id = c.customer_id GROUP BY c.city")
    assert "Every row from customers is kept, even without a match in orders" in b.joins[0].plain
    assert _plain(b.metrics)[1] == "orders = number of rows with a value in orders.order_id"


def test_join_using_and_cross_join() -> None:
    b = explain_sql("SELECT * FROM orders JOIN order_items USING (order_id) CROSS JOIN products")
    assert b.joins[0].plain.startswith("Inner join with order_items, matching rows on order_id")
    assert b.joins[1].plain == "Cross join with products: every combination of rows is formed"
    assert _plain(b.metrics) == ["all columns"]


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("COUNT(*)", "n = number of rows"),
     ("COUNT(DISTINCT customer_id)", "n = number of distinct customer_id"),
     ("AVG(total)", "n = average of total"),
     ("MIN(total)", "n = lowest of total"),
     ("MAX(total)", "n = highest of total"),
     ("SUM(total) / COUNT(*)", "n = sum of total ÷ number of rows"),
     ("ROUND(AVG(total), 1)", "n = average of total, rounded to 1 decimal"),
     ("CAST(total AS REAL)", "n = total converted to REAL"),
     ("COALESCE(total, 0)", "n = total (or 0 when missing)"),
     ("strftime('%Y-%m', order_date)", "n = month of order_date"),
     ("strftime('%Y', order_date)", "n = year of order_date"),
     ("SUM(total) OVER (ORDER BY order_date)",
      "n = running sum of total, computed per row ordered by order_date"),
     ("CASE WHEN total > 100 THEN 'big' ELSE 'small' END",
      "n = a conditional value (CASE WHEN total > 100 THEN 'big' ELSE 'small' END)")],
)
def test_metric_formulas(expression: str, expected: str) -> None:
    assert _plain(explain_sql(f"SELECT {expression} AS n FROM orders").metrics) == [expected]


@pytest.mark.parametrize(
    ("condition", "expected"),
    [("status = 'completed'", "status is 'completed'"),
     ("status <> 'cancelled'", "status is not 'cancelled'"),
     ("total > 100", "total is greater than 100"),
     ("total >= 100", "total is at least 100"),
     ("total < 5", "total is less than 5"),
     ("total <= 5", "total is at most 5"),
     ("status IN ('a', 'b')", "status is one of 'a', 'b'"),
     ("status NOT IN ('a')", "status is not one of 'a'"),
     ("order_date BETWEEN '2025-01-01' AND '2025-03-31'",
      "order_date is between '2025-01-01' and '2025-03-31' (inclusive)"),
     ("name LIKE 'A%'", "name matches the pattern 'A%'"),
     ("note IS NULL", "note is missing (NULL)"),
     ("note IS NOT NULL", "note is present (not NULL)"),
     ("(city = 'Pune' OR city = 'Goa')", "either city is 'Pune' or city is 'Goa'"),
     ("customer_id IN (SELECT customer_id FROM vip)",
      "customer_id appears in the result of a subquery")],
)
def test_filter_descriptions(condition: str, expected: str) -> None:
    assert _plain(explain_sql(f"SELECT * FROM orders WHERE {condition}").filters) == [expected]


def test_and_conditions_become_separate_filters() -> None:
    b = explain_sql("SELECT * FROM orders WHERE status = 'completed' AND total > 10 "
                    "AND (city = 'A' OR city = 'B')")
    assert len(b.filters) == 3


def test_having_ordinals_and_offset() -> None:
    b = explain_sql("SELECT city, COUNT(*) AS n FROM customers GROUP BY 1 "
                    "HAVING COUNT(*) > 2 ORDER BY 2 DESC, 1 LIMIT 3 OFFSET 6")
    assert _plain(b.grouping) == ["One row per city"]
    assert _plain(b.group_filters) == ["Keep only groups where number of rows is greater than 2"]
    assert _plain(b.ordering) == [
        "Sorted by n, descending (highest first)", "Sorted by city, ascending (lowest first)",
        "Only the top 3 rows are kept, after skipping the first 6"]


def test_ctes_are_explained_as_steps() -> None:
    b = explain_sql(
        "WITH monthly AS (SELECT strftime('%Y-%m', order_date) AS month, SUM(total) AS revenue "
        "FROM orders GROUP BY 1) SELECT month, revenue FROM monthly ORDER BY month")
    assert [name for name, _ in b.steps] == ["monthly"]
    step = b.steps[0][1]
    assert _plain(step.metrics) == ["month = month of order_date", "revenue = sum of total"]
    assert _plain(b.tables) == ["monthly (step 1)"]
    assert _plain(b.metrics) == ["month: taken directly from monthly (step 1)",
                                 "revenue: taken directly from monthly (step 1)"]


def test_subqueries_and_distinct_are_noted() -> None:
    b = explain_sql("SELECT DISTINCT city FROM (SELECT city FROM customers) s "
                    "WHERE city IN (SELECT city FROM vip)")
    assert any("subquery" in note for note in b.notes)
    assert any("DISTINCT" in note for note in b.notes)


@pytest.mark.parametrize("sql", ["SELECT a FROM t UNION SELECT b FROM u", "", None,
                                 "SELEC nonsense ((("])
def test_unsupported_queries_are_not_guessed(sql) -> None:
    b = explain_sql(sql)
    assert not b.parsed
    assert "could not be broken down" in breakdown_to_text(b)


def test_absent_clauses_produce_no_sections() -> None:
    b = explain_sql("SELECT name FROM customers")
    assert b.joins == () and b.filters == () and b.grouping == () and b.ordering == ()
    text = breakdown_to_text(b)
    assert "Filters" not in text and "combined" not in text


def test_nothing_is_described_that_the_sql_does_not_contain() -> None:
    b = explain_sql(TOP5_SQL)
    text = breakdown_to_text(b)
    for absent in ("customers", "cancelled", "pending", "average", "LEFT", "HAVING"):
        assert absent not in text, absent


def test_unknown_functions_fall_back_to_their_sql() -> None:
    b = explain_sql("SELECT upper(name) AS n FROM customers")
    assert _plain(b.metrics) == ["n = UPPER(name)"]


def test_text_form_is_deterministic_and_structured() -> None:
    text = breakdown_to_text(explain_sql(TOP5_SQL))
    assert text == breakdown_to_text(explain_sql(TOP5_SQL))
    assert text.splitlines()[0] == "Data read:"
    assert "Filters applied:\n- orders.status is 'completed'" in text


def test_long_fragments_are_shortened() -> None:
    values = ", ".join(f"'{i:04d}'" for i in range(200))
    b = explain_sql(f"SELECT * FROM orders WHERE code IN ({values})")
    assert len(b.filters[0].sql) <= 160 and b.filters[0].sql.endswith("…")
