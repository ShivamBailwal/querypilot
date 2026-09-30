"""KPI selection, chart selection, formatting and chart building.

Pure functions only: no Streamlit, no database, no LLM.
"""

from __future__ import annotations

import copy
import tomllib
from pathlib import Path

import pytest

from analytics.charts import PALETTE, SURFACE, build_chart
from analytics.insights import (
    MAX_BAR_CATEGORIES,
    analyze_results,
    analyze_sql,
    format_number,
    format_period,
    humanize,
    profile_column,
)

TOP5_SQL = (
    "SELECT p.product_name, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue "
    "FROM order_items oi JOIN products p ON p.product_id = oi.product_id "
    "GROUP BY p.product_name ORDER BY revenue DESC LIMIT 5"
)
TOP5 = {"columns": ["product_name", "revenue"],
        "rows": [["Laptop", 9000.5], ["Phone", 7000.0], ["Desk", 4000.25],
                 ["Chair", 3000.0], ["Lamp", 1200.0]],
        "truncated": False}


def _result(columns, rows, truncated=False):
    return {"columns": columns, "rows": rows, "truncated": truncated}


def _kpis(results, sql=None, question=""):
    return {k.label: k for k in analyze_results(results, sql, question).kpis}


# ---------------------------------------------------------------- KPIs
def test_top_products_kpis_come_from_the_rows() -> None:
    kpis = _kpis(TOP5, TOP5_SQL, "What were the top five products by revenue?")
    assert list(kpis) == ["Highest revenue", "Combined revenue", "Rows returned"]
    assert kpis["Highest revenue"].value == "Laptop"
    assert kpis["Highest revenue"].detail == "9,000.50"
    assert kpis["Combined revenue"].value == f"{sum(r[1] for r in TOP5['rows']):,.2f}"
    assert kpis["Combined revenue"].detail == "across the 5 rows shown"  # LIMIT is disclosed
    assert kpis["Rows returned"].value == "5"


def test_no_currency_symbol_is_ever_invented() -> None:
    for kpi in analyze_results(TOP5, TOP5_SQL).kpis:
        assert not any(sym in kpi.value + (kpi.detail or "") for sym in "$€£₹¥")


@pytest.mark.parametrize(
    ("sql", "why"),
    [
        ("SELECT city, COUNT(DISTINCT customer_id) AS customers FROM orders GROUP BY city",
         "distinct counts overlap across groups"),
        ("SELECT city, AVG(total) AS avg_total FROM orders GROUP BY city",
         "an average of averages is not the overall average"),
        ("SELECT city, SUM(total) OVER (ORDER BY city) AS running FROM orders",
         "running totals would double-count"),
        ("SELECT city, SUM(a) / COUNT(*) AS ratio FROM orders GROUP BY city",
         "ratios are not additive"),
        ("WITH c AS (SELECT city, SUM(total) AS total FROM orders GROUP BY city) "
         "SELECT city, total FROM c", "meaning is hidden inside a CTE"),
        ("SELECT city, total FROM (SELECT city, SUM(total) AS total FROM orders GROUP BY city)",
         "meaning is hidden inside a derived table"),
        ("SELECT * FROM city_totals", "SELECT * reveals nothing about meaning"),
        (None, "no SQL"),
    ],
)
def test_aggregates_are_never_summed_or_averaged_misleadingly(sql, why) -> None:
    results = _result(["city", "value"], [["Pune", 10.0], ["Delhi", 20.0], ["Goa", 5.0]])
    labels = set(_kpis(results, sql))
    assert not any(label.startswith(("Combined", "Average")) for label in labels), why
    assert "Highest value" in labels and "Rows returned" in labels


def test_additive_group_counts_can_be_combined() -> None:
    results = _result(["status", "orders"], [["completed", 300], ["pending", 40]])
    kpis = _kpis(results, "SELECT status, COUNT(*) AS orders FROM orders GROUP BY status")
    assert kpis["Combined orders"].value == "340"
    assert kpis["Combined orders"].detail == "across all 2 rows"


def test_row_level_values_get_an_average_but_not_for_a_limited_subset() -> None:
    rows = [["Desk", 100.0], ["Lamp", 20.0], ["Chair", 60.0]]
    results = _result(["product_name", "unit_price"], rows)
    plain = _kpis(results, "SELECT product_name, unit_price FROM products")
    assert plain["Average unit price"].value == "60.00"
    limited = _kpis(results, "SELECT product_name, unit_price FROM products "
                             "ORDER BY unit_price DESC LIMIT 3")
    assert not any(k.startswith("Average") for k in limited)


def test_ascending_rankings_report_the_lowest_value() -> None:
    results = _result(["product_name", "stock"], [["Lamp", 1], ["Desk", 4], ["Chair", 9]])
    kpis = _kpis(results, "SELECT product_name, stock FROM products ORDER BY stock ASC")
    assert kpis["Lowest stock"].value == "Lamp"


def test_ties_are_disclosed_and_nulls_ignored() -> None:
    results = _result(["city", "orders"], [["A", 5], ["B", 5], ["C", None]])
    kpis = _kpis(results, "SELECT city, COUNT(*) AS orders FROM t GROUP BY city")
    assert kpis["Highest orders"].value == "A"
    assert kpis["Highest orders"].detail == "5 (tied)"
    assert not any(k.startswith("Combined") for k in kpis)  # a null makes the total unknowable


def test_single_row_shows_its_values_measures_first() -> None:
    results = _result(["category", "total_revenue", "orders"],
                      [["Electronics", 1_234_567.891, 420]])
    kpis = analyze_results(results, "SELECT category, SUM(x) AS total_revenue, COUNT(*) AS "
                                    "orders FROM t GROUP BY category ORDER BY 2 DESC LIMIT 1").kpis
    assert [(k.label, k.value) for k in kpis] == [
        ("Total revenue", "1.23M"), ("Orders", "420"), ("Category", "Electronics")]
    assert "Exact: 1,234,567.89" in kpis[0].help


def test_single_value_result() -> None:
    kpis = analyze_results(_result(["total_revenue"], [[98765.4321]])).kpis
    assert [(k.label, k.value) for k in kpis] == [("Total revenue", "98,765.43")]


def test_null_single_value_is_shown_as_a_dash() -> None:
    kpis = analyze_results(_result(["total_revenue"], [[None]])).kpis
    assert kpis[0].value == "—"


def test_time_series_kpis_use_latest_and_peak_periods() -> None:
    results = _result(["month", "revenue"],
                      [["2025-03", 90.0], ["2025-01", 50.0], ["2025-02", 120.0]])
    kpis = _kpis(results, "SELECT month, SUM(x) AS revenue FROM t GROUP BY month")
    assert kpis["Revenue · latest"].value == "90.00"
    assert kpis["Revenue · latest"].detail == "Mar 2025"
    assert kpis["Highest revenue"].detail == "Feb 2025"


def test_truncated_results_get_only_the_row_count() -> None:
    results = _result(["product_name", "revenue"], [[f"P{i}", float(i)] for i in range(5)],
                      truncated=True)
    kpis = analyze_results(results, "SELECT product_name, revenue FROM t").kpis
    assert [(k.label, k.value) for k in kpis] == [("Rows returned", "5+")]
    assert "Row limit reached" in kpis[0].detail


def test_text_only_and_empty_results() -> None:
    assert [k.label for k in analyze_results(
        _result(["product_name"], [["A"], ["B"], ["C"]])).kpis] == ["Rows returned"]
    empty = analyze_results(_result(["a", "b"], []))
    assert empty.kpis == () and empty.chart.kind == "table"


def test_kpi_count_never_exceeds_three() -> None:
    results = _result(list("abcdef"), [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    assert len(analyze_results(results).kpis) == 3


# --------------------------------------------------------- chart selection
def test_ranked_bar_for_categories() -> None:
    chart = analyze_results(TOP5, TOP5_SQL, "top five products by revenue").chart
    assert chart.kind == "bar"
    assert chart.title == "Revenue by product name"
    assert (chart.x, chart.y) == (0, 1)
    assert chart.row_indexes == (0, 1, 2, 3, 4)


def test_bar_ranks_by_value_unless_query_orders_by_label() -> None:
    rows = [["Small", 5.0], ["Medium", 30.0], ["Large", 10.0]]
    unordered = analyze_results(_result(["size", "sales"], rows),
                                "SELECT size, SUM(x) AS sales FROM t GROUP BY size").chart
    assert unordered.row_indexes == (1, 2, 0)
    by_label = analyze_results(_result(["size", "sales"], rows),
                               "SELECT size, SUM(x) AS sales FROM t GROUP BY size "
                               "ORDER BY size_rank").chart  # unknown order key -> by value
    assert by_label.row_indexes == (1, 2, 0)
    kept = analyze_results(_result(["size", "sales"], rows),
                           "SELECT size, SUM(x) AS sales FROM t GROUP BY size ORDER BY size").chart
    assert kept.row_indexes == (0, 1, 2)


def test_many_categories_are_capped_with_a_note() -> None:
    rows = [[f"Product {i}", float(i)] for i in range(40)]
    chart = analyze_results(_result(["product_name", "revenue"], rows),
                            "SELECT product_name, SUM(x) AS revenue FROM t GROUP BY 1").chart
    assert chart.kind == "bar" and len(chart.row_indexes) == MAX_BAR_CATEGORIES
    assert rows[chart.row_indexes[0]][1] == 39.0
    assert f"top {MAX_BAR_CATEGORIES} of 40" in chart.notes[0]


def test_negative_values_still_chart_as_bars() -> None:
    rows = [["A", -5.0], ["B", 10.0], ["C", 2.5]]
    chart = analyze_results(_result(["region", "profit"], rows),
                            "SELECT region, SUM(p) AS profit FROM t GROUP BY region",
                            "share of profit by region").chart
    assert chart.kind == "bar"  # a pie with negative slices would be meaningless


def test_line_for_time_series_sorted_by_time() -> None:
    rows = [["2025-03", 3.0], ["2025-01", 1.0], ["2025-02", 2.0]]
    chart = analyze_results(_result(["month", "revenue"], rows)).chart
    assert chart.kind == "line" and chart.title == "Revenue by month"
    assert [rows[i][0] for i in chart.row_indexes] == ["2025-01", "2025-02", "2025-03"]


@pytest.mark.parametrize(
    ("name", "values", "kind"),
    [("order_date", ["2025-01-05", "2025-02-01"], "line"),
     ("created_at", ["2025-01-05 10:00:00", "2025-01-05T11:30:00"], "line"),
     ("year", [2023, 2024], "line"),
     ("year", ["2023", "2024"], "line"),
     ("code", ["2023", "2024"], "bar"),           # four digits alone are not a year
     ("month", ["2025-13", "2025-14"], "bar")],    # not real months -> plain text labels
)
def test_temporal_detection(name, values, kind) -> None:
    rows = [[v, float(i + 1)] for i, v in enumerate(values)]
    assert analyze_results(_result([name, "amount"], rows)).chart.kind == kind


def test_multi_series_line_and_its_limits() -> None:
    rows = [[m, s, float(i)] for i, (m, s) in enumerate(
        [(m, s) for m in ("2025-01", "2025-02") for s in ("a", "b")])]
    chart = analyze_results(_result(["month", "status", "orders"], rows)).chart
    assert chart.kind == "line" and chart.series == 1
    assert chart.title == "Orders by month, split by status"

    many = [[m, f"s{s}", 1.0] for m in ("2025-01", "2025-02") for s in range(7)]
    assert analyze_results(_result(["month", "status", "orders"], many)).chart.kind == "table"
    lonely = [["2025-01", "a", 1.0], ["2025-02", "b", 2.0]]
    assert analyze_results(_result(["month", "status", "orders"], lonely)).chart.kind == "table"


def test_repeated_periods_fall_back_to_table() -> None:
    rows = [["2025-01", 1.0], ["2025-01", 2.0], ["2025-02", 3.0]]
    chart = analyze_results(_result(["month", "amount"], rows)).chart
    assert chart.kind == "table" and "more than once" in chart.reason


def test_donut_only_for_explicit_part_of_whole_questions() -> None:
    rows = [["Books", 50.0], ["Toys", 30.0], ["Games", 20.0]]
    res = _result(["category", "revenue"], rows)
    sql = "SELECT category, SUM(r) AS revenue FROM t GROUP BY category"
    assert analyze_results(res, sql, "What share of revenue comes from each category?"
                           ).chart.kind == "donut"
    assert analyze_results(res, sql, "Revenue by category").chart.kind == "bar"
    assert analyze_results(res, sql + " ORDER BY revenue DESC LIMIT 3",  # top-N is not a whole
                           "share of revenue by category").chart.kind == "bar"
    avg_sql = "SELECT category, AVG(r) AS revenue FROM t GROUP BY category"
    assert analyze_results(res, avg_sql, "share of revenue").chart.kind == "bar"
    seven = _result(["category", "revenue"], [[f"c{i}", 1.0] for i in range(7)])
    assert analyze_results(seven, sql, "revenue breakdown by category").chart.kind == "bar"


def test_scatter_for_two_measures_without_labels() -> None:
    rows = [[1.0, 10.0], [2.0, 18.0], [3.0, 33.0], [4.0, 41.0]]
    chart = analyze_results(_result(["unit_price", "quantity"], rows)).chart
    assert chart.kind == "scatter" and chart.title == "Unit price vs quantity"


def test_identifiers_are_labels_never_measures() -> None:
    res = _result(["customer_id", "total_spent"], [[17, 900.0], [4, 1200.0], [9, 300.0]])
    chart = analyze_results(res, "SELECT customer_id, SUM(t) AS total_spent FROM o "
                                 "GROUP BY customer_id").chart
    assert chart.kind == "bar" and (chart.x, chart.y) == (0, 1)
    ids_only = analyze_results(_result(["order_id"], [[1], [2], [3]])).chart
    assert ids_only.kind == "table" and "no numeric values" in ids_only.reason
    profile = profile_column(0, "CustomerID", [1, 2, 3])
    assert profile.kind == "identifier"


@pytest.mark.parametrize(
    ("columns", "rows", "reason"),
    [(["revenue"], [[1.0], [2.0]], "no category or time column"),
     (["name", "city"], [["A", "X"], ["B", "Y"]], "no numeric values"),
     (["city", "amount"], [["X", 1.0], ["X", 2.0]], "Labels repeat"),
     (["a", "b"], [[1.0, 2.0], [3.0, 4.0]], "Too few points"),
     (["d1", "d2", "v"], [["2025-01-01", "2025-01-02", 1.0], ["2025-01-03", "2025-01-04", 2.0]],
      "ambiguous")],
)
def test_unsupported_shapes_fall_back_to_table(columns, rows, reason) -> None:
    chart = analyze_results(_result(columns, rows)).chart
    assert chart.kind == "table" and reason in chart.reason


def test_single_row_never_becomes_a_one_bar_chart() -> None:
    chart = analyze_results(_result(["category", "revenue"], [["Books", 5.0]])).chart
    assert chart.kind == "table" and "KPI cards" in chart.reason


def test_nulls_and_truncation_are_disclosed_in_chart_notes() -> None:
    rows = [["A", 1.0], ["B", None], [None, 3.0], ["D", 4.0]]
    chart = analyze_results(_result(["city", "amount"], rows, truncated=True)).chart
    assert chart.kind == "bar" and set(chart.row_indexes) == {0, 3}
    assert any("2 row(s) with missing values" in n for n in chart.notes)
    assert any("truncated" in n for n in chart.notes)


def test_other_measures_are_mentioned_not_dual_axised() -> None:
    res = _result(["product_name", "quantity", "revenue"],
                  [["A", 5, 50.0], ["B", 2, 80.0], ["C", 9, 20.0]])
    chart = analyze_results(res, "SELECT product_name, SUM(q) AS quantity, SUM(r) AS revenue "
                                 "FROM t GROUP BY 1 ORDER BY revenue DESC").chart
    assert chart.y == 2  # the measure the query ranks by
    assert chart.notes == ("Showing revenue; Quantity is in the table.",)


def test_analysis_never_modifies_the_results() -> None:
    results = copy.deepcopy(TOP5)
    insights = analyze_results(results, TOP5_SQL, "share of revenue")
    build_chart(insights.chart, results["rows"], insights.profiles)
    assert results == TOP5


# ------------------------------------------------------------------ SQL shape
def test_sql_shape_maps_order_by_forms() -> None:
    for order in ("revenue DESC", "2 DESC", "SUM(x) DESC"):
        shape = analyze_sql(f"SELECT c, SUM(x) AS revenue FROM t GROUP BY c ORDER BY {order}", 2)
        assert shape.order_by == (1,) and shape.order_desc == (True,), order
    assert not analyze_sql("SELECT a FROM t UNION SELECT b FROM u", 1).parsed
    assert not analyze_sql("SELECT a, b FROM t", 3).parsed  # column count mismatch


# ---------------------------------------------------------------- formatting
@pytest.mark.parametrize(
    ("value", "unit", "compact", "expected"),
    [(1234.5, "money", False, "1,234.50"),
     (-1234.5, "money", False, "-1,234.50"),
     (1234, "integer", False, "1,234"),
     (12.345, "decimal", False, "12.35"),
     (2.0, "decimal", False, "2"),
     (0.004567, "decimal", False, "0.00457"),
     (42.25, "percent", False, "42.2%"),
     (1_234_567, "money", True, "1.23M"),
     (999_999_999, "integer", True, "1B"),
     (-25_500_000_000, "money", True, "-25.5B"),
     (3.2e15, "integer", True, "3200T"),
     (999_999, "integer", True, "999,999"),
     (None, "money", False, "—"),
     (float("inf"), "decimal", False, "inf"),
     (20250101, "plain", False, "20250101")],
)
def test_format_number(value, unit, compact, expected) -> None:
    assert format_number(value, unit, compact=compact) == expected


def test_units_are_inferred_conservatively() -> None:
    assert profile_column(0, "revenue_share_pct", [12.5, 40.0]).unit == "percent"
    assert profile_column(0, "conversion_pct", [0.12, 0.4]).unit == "decimal"  # ambiguous scale
    assert profile_column(0, "total_revenue", [10.5, 3.0]).unit == "money"
    assert profile_column(0, "sales_count", [3, 9]).unit == "integer"
    assert profile_column(0, "score", [1.5, 2.25]).unit == "decimal"


def test_format_period_and_humanize() -> None:
    assert format_period("2025-03", "month") == "Mar 2025"
    assert format_period("2025-03-07", "day") == "7 Mar 2025"
    assert format_period(2024, "year") == "2024"
    assert humanize("total_revenue") == "Total revenue"
    assert humanize("Unit Price ($)") == "Unit Price ($)"


# ------------------------------------------------------------------ charts
def _built(results, sql=None, question=""):
    insights = analyze_results(results, sql, question)
    chart = build_chart(insights.chart, results["rows"], insights.profiles)
    assert chart is not None
    return chart.to_dict(validate=True)  # raises if the Vega-Lite spec is invalid


def _layers(spec):
    return spec.get("layer", [spec])


def _dataset(spec):
    return next(iter(spec["datasets"].values()))


def test_bar_chart_spec_uses_one_color_and_the_real_values() -> None:
    spec = _built(TOP5, TOP5_SQL)
    bar = next(layer for layer in _layers(spec) if layer["mark"]["type"] == "bar")
    assert bar["mark"]["color"] == PALETTE[0]
    assert bar["mark"]["cornerRadiusEnd"] == 4 and bar["mark"]["size"] <= 24
    assert [(d["label"], d["value"]) for d in _dataset(spec)] == [tuple(r) for r in TOP5["rows"]]
    assert any(layer["mark"]["type"] == "text" for layer in _layers(spec))  # values at tips


def test_line_chart_spec_has_series_colors_in_stable_order() -> None:
    rows = [["2025-01", "b", 1.0], ["2025-01", "a", 2.0], ["2025-02", "b", 3.0],
            ["2025-02", "a", 4.0]]
    spec = _built(_result(["month", "status", "orders"], rows))
    color = _layers(spec)[0]["encoding"]["color"]
    assert color["scale"]["domain"] == ["a", "b"]
    assert color["scale"]["range"] == list(PALETTE[:2])
    assert any(layer["mark"].get("strokeWidth") == 2 for layer in _layers(spec)
               if layer["mark"]["type"] == "line")


def test_donut_and_scatter_specs_build() -> None:
    donut = _built(_result(["category", "revenue"], [["A", 5.0], ["B", 3.0]]),
                   "SELECT category, SUM(r) AS revenue FROM t GROUP BY 1", "revenue share")
    assert donut["mark"]["type"] == "arc" and donut["mark"]["stroke"] == SURFACE
    scatter = _built(_result(["price", "qty"], [[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]]))
    assert scatter["mark"]["type"] == "circle"


def test_large_values_use_compact_axis_labels() -> None:
    res = _result(["region", "revenue"], [["N", 4.2e9], ["S", 1.1e9]])
    spec = _built(res, "SELECT region, SUM(r) AS revenue FROM t GROUP BY 1")
    axis = next(layer for layer in _layers(spec) if layer["mark"]["type"] == "bar")
    label_expr = (axis.get("encoding") or spec["encoding"])["x"]["axis"]["labelExpr"]
    assert "'G', 'B'" in label_expr


def test_table_only_spec_builds_no_chart() -> None:
    insights = analyze_results(_result(["name"], [["A"], ["B"]]))
    assert build_chart(insights.chart, [["A"], ["B"]], insights.profiles) is None


def test_chart_palette_matches_the_streamlit_theme() -> None:
    config = tomllib.loads((Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml")
                           .read_text(encoding="utf-8"))["theme"]
    assert tuple(config["chartCategoricalColors"]) == PALETTE
    assert config["backgroundColor"] == SURFACE  # the palette was validated on this surface
