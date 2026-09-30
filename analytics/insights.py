"""Turn a query result into KPI cards and a chart choice.

Everything here is pure: no Streamlit, no Altair, no database access. It reads
only the ``QueryResults`` that the protected executor already returned, plus
the validated SQL text, which is *parsed* (never executed) to learn what each
output column means: a SUM per group, a COUNT(DISTINCT ...), a running total
from a window function, or a plain row value. That is what keeps the KPIs
honest:

* A per-group SUM/COUNT may be combined across rows (groups are disjoint), but
  never a COUNT(DISTINCT ...) (entities can appear in several groups), an AVG
  (an average of averages is not the overall average), or a window function
  (running totals would double-count).
* Averages are only offered for plain row values from base tables, never for
  a LIMIT-ed subset.
* Truncated results get no derived KPIs at all: they would describe only the
  rows that happened to be returned.

When the result shape is ambiguous, the chart choice is ``table``: a table is
never misleading, a wrong chart is.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import sqlglot
from sqlglot import exp

Kind = Literal["numeric", "temporal", "text", "identifier", "empty", "other"]
Unit = Literal["integer", "money", "percent", "decimal", "plain"]
Granularity = Literal["year", "month", "day", "datetime"]
ChartKind = Literal["bar", "line", "donut", "scatter", "table"]

MAX_KPIS = 3
MAX_BAR_CATEGORIES = 15
MAX_BAR_VALUE_LABELS = 10
MAX_DONUT_SLICES = 6
MAX_SERIES = 6
MIN_SCATTER_POINTS = 3
MAX_LABEL_CHARS = 40

# ------------------------------------------------------------ SQL semantics
_AGGREGATES: dict[type[exp.Expression], str] = {
    exp.Sum: "sum", exp.Count: "count", exp.Avg: "avg", exp.Min: "min", exp.Max: "max",
}
# Wrappers that keep an aggregate's meaning: ROUND(SUM(x), 2), CAST(SUM(x) AS REAL)...
_TRANSPARENT = (exp.Round, exp.Cast, exp.Paren, exp.Coalesce)


@dataclass(frozen=True)
class OutputColumn:
    """What one result column is, according to the SQL that produced it."""

    aggregate: str | None = None  # sum | count | avg | min | max | derived | unknown | None
    distinct: bool = False        # COUNT(DISTINCT ...)
    windowed: bool = False        # computed by a window function (e.g. running total)

    @property
    def additive(self) -> bool:
        """Safe to add up across result rows (disjoint groups)."""
        return self.aggregate in ("sum", "count") and not self.distinct and not self.windowed


@dataclass(frozen=True)
class QueryShape:
    parsed: bool = False
    columns: tuple[OutputColumn, ...] = ()  # positional: same order as the result columns
    grouped: bool = False
    limited: bool = False
    order_by: tuple[int, ...] = ()          # result-column indexes, in ORDER BY order
    order_desc: tuple[bool, ...] = ()

    def column(self, index: int) -> OutputColumn:
        return self.columns[index] if self.parsed and index < len(self.columns) \
            else OutputColumn(aggregate="unknown")


def _describe_projection(node: exp.Expression, derived_source: bool) -> OutputColumn:
    inner = node.this if isinstance(node, exp.Alias) else node
    while isinstance(inner, _TRANSPARENT) and isinstance(inner.this, exp.Expression):
        inner = inner.this
    if inner.find(exp.Window):
        return OutputColumn(aggregate="derived", windowed=True)
    for cls, name in _AGGREGATES.items():
        if type(inner) is cls:
            return OutputColumn(aggregate=name, distinct=isinstance(inner.this, exp.Distinct))
    if inner.find(exp.AggFunc):
        return OutputColumn(aggregate="derived")  # e.g. SUM(a) / COUNT(*), a ratio
    # A plain column read from a CTE or derived table may itself be an aggregate.
    return OutputColumn(aggregate="unknown" if derived_source else None)


def analyze_sql(sql: str | None, column_count: int) -> QueryShape:
    """Parse (never execute) the validated SQL to learn what the columns mean.

    Anything unusual (set operations, ``SELECT *``, unparsable text) yields an
    unparsed shape, which disables every KPI that depends on column meaning.
    """
    if not sql:
        return QueryShape()
    try:
        tree = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return QueryShape()
    if not isinstance(tree, exp.Select):
        return QueryShape()
    projections = list(tree.expressions)
    if len(projections) != column_count or any(
        isinstance(p, exp.Star) or (isinstance(p, exp.Column) and isinstance(p.this, exp.Star))
        for p in projections
    ):
        return QueryShape()

    # sqlglot >= 30 stores these as "from_" / "with_"; older versions without the underscore.
    sources = [tree.args.get("from_") or tree.args.get("from"), *(tree.args.get("joins") or [])]
    derived_source = bool(tree.args.get("with_") or tree.args.get("with")) or any(
        s is not None and s.find(exp.Subquery) is not None for s in sources
    )
    columns = tuple(_describe_projection(p, derived_source) for p in projections)

    names = [p.alias_or_name.lower() for p in projections]
    inners = [p.this if isinstance(p, exp.Alias) else p for p in projections]
    order_idx: list[int] = []
    order_desc: list[bool] = []
    order = tree.args.get("order")
    for ordered in (order.expressions if order else []):
        key = ordered.this
        idx: int | None = None
        if isinstance(key, exp.Literal) and not key.is_string and key.this.isdigit():
            idx = int(key.this) - 1 if 0 < int(key.this) <= len(projections) else None
        elif isinstance(key, exp.Column) and not key.table and key.name.lower() in names:
            idx = names.index(key.name.lower())
        else:
            idx = next((i for i, inner in enumerate(inners) if inner == key), None)
        if idx is not None:
            order_idx.append(idx)
            order_desc.append(bool(ordered.args.get("desc")))

    return QueryShape(
        parsed=True,
        columns=columns,
        grouped=tree.args.get("group") is not None,
        limited=tree.args.get("limit") is not None,
        order_by=tuple(order_idx),
        order_desc=tuple(order_desc),
    )


# ---------------------------------------------------------- column profiles
_IDENTIFIER = re.compile(
    r"(?:^|[_\s])(?:id|uuid|guid|code|zip|zipcode|postcode|phone|number)$|^id(?:[_\s]|$)",
    re.IGNORECASE,
)
_CAMEL_ID = re.compile(r"[a-z](?:ID|Id)$")
_YEAR_NAME = re.compile(r"(?:^|_)(?:year|yr)(?:_|$)", re.IGNORECASE)
_PERCENT_NAME = re.compile(r"percent|pct", re.IGNORECASE)
_COUNT_NAME = re.compile(
    r"(?:^|_)(?:count|cnt|num|qty|quantity|units|n)(?:_|$)|^(?:number|total)_of_", re.IGNORECASE
)
# Only changes number formatting (two decimals); no currency symbol is ever shown,
# because the currency of an arbitrary database is unknown.
_MONEY_NAME = re.compile(
    r"revenue|sales|price|amount|spend|spent|cost|income|profit|paid|payment|fee|salary|"
    r"budget|earning|gmv",
    re.IGNORECASE,
)

_ISO_DATETIME = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?$")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ISO_MONTH = re.compile(r"^\d{4}-\d{2}$")
_ISO_YEAR = re.compile(r"^\d{4}$")


@dataclass(frozen=True)
class ColumnProfile:
    index: int
    name: str
    title: str
    kind: Kind
    unit: Unit
    non_null: int
    nulls: int
    distinct: int
    minimum: float | None = None
    maximum: float | None = None
    granularity: Granularity | None = None
    source: OutputColumn = OutputColumn()

    @property
    def is_measure(self) -> bool:
        return self.kind == "numeric" and self.non_null > 0

    @property
    def has_negative(self) -> bool:
        return self.minimum is not None and self.minimum < 0


def humanize(name: str) -> str:
    """'total_revenue' -> 'Total revenue'; names with spaces or capitals are kept."""
    if not name:
        return "Value"
    if " " in name or any(c.isupper() for c in name):
        return name
    words = re.sub(r"_+", " ", name).strip()
    return (words[:1].upper() + words[1:]) or name


def _lower_first(title: str) -> str:
    if len(title) > 1 and title[0].isupper() and not title[1].isupper():
        return title[0].lower() + title[1:]
    return title


def is_identifier_name(name: str) -> bool:
    return bool(_IDENTIFIER.search(name) or _CAMEL_ID.search(name))


def parse_temporal(value: Any, granularity: Granularity) -> datetime:
    if granularity == "year":
        return datetime(int(value), 1, 1)
    text = str(value)
    if granularity == "month":
        return datetime.fromisoformat(text + "-01")
    return datetime.fromisoformat(text)


def _temporal_granularity(values: list[Any], name: str) -> Granularity | None:
    if all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        if _YEAR_NAME.search(name) and all(1800 <= v <= 2200 for v in values):
            return "year"
        return None
    if not all(isinstance(v, str) for v in values):
        return None
    for pattern, granularity in ((_ISO_DATE, "day"), (_ISO_MONTH, "month"),
                                 (_ISO_DATETIME, "datetime"), (_ISO_YEAR, "year")):
        if all(pattern.match(v) for v in values):
            if granularity == "year" and not _YEAR_NAME.search(name):
                return None  # four digits alone are not enough evidence of a year
            try:
                for v in values:
                    parse_temporal(v, granularity)  # type: ignore[arg-type]
            except ValueError:
                return None
            return granularity  # type: ignore[return-value]
    return None


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def profile_column(index: int, name: str, values: Sequence[Any],
                   source: OutputColumn = OutputColumn()) -> ColumnProfile:
    present = [v for v in values if v is not None]
    nulls = len(values) - len(present)
    try:
        distinct = len(set(present))
    except TypeError:  # unhashable values cannot come from sqlite3, but be safe
        distinct = len(present)
    base = dict(index=index, name=name, title=humanize(name), non_null=len(present),
                nulls=nulls, distinct=distinct, source=source)

    if not present:
        return ColumnProfile(kind="empty", unit="plain", **base)
    granularity = _temporal_granularity(present, name)
    if granularity:
        return ColumnProfile(kind="temporal", unit="plain", granularity=granularity, **base)
    if all(_is_number(v) for v in present):
        if is_identifier_name(name):
            return ColumnProfile(kind="identifier", unit="plain", **base)
        finite = [float(v) for v in present if math.isfinite(v)]
        lo, hi = (min(finite), max(finite)) if finite else (None, None)
        all_int = all(isinstance(v, int) or float(v).is_integer() for v in present)
        if _PERCENT_NAME.search(name) and hi is not None and max(abs(lo), abs(hi)) > 1:
            unit: Unit = "percent"  # values are already in percent units (e.g. 42.5)
        elif source.aggregate == "count" or (all_int and _COUNT_NAME.search(name)):
            unit = "integer"
        elif _MONEY_NAME.search(name) and not _COUNT_NAME.search(name):
            unit = "money"
        elif all_int:
            unit = "integer"
        else:
            unit = "decimal"
        return ColumnProfile(kind="numeric", unit=unit, minimum=lo, maximum=hi, **base)
    if all(isinstance(v, str) for v in present):
        kind: Kind = "identifier" if is_identifier_name(name) else "text"
        return ColumnProfile(kind=kind, unit="plain", **base)
    return ColumnProfile(kind="other", unit="plain", **base)


# --------------------------------------------------------------- formatting
def format_number(value: Any, unit: Unit, *, compact: bool = False) -> str:
    """Readable text for one number. Never adds a currency symbol."""
    if value is None:
        return "—"
    if not _is_number(value):
        return str(value)
    v = float(value)
    if not math.isfinite(v):
        return str(value)
    suffix = "%" if unit == "percent" else ""
    if compact and abs(v) >= 1_000_000:
        return _compact(v) + suffix
    if unit == "money":
        return f"{v:,.2f}"
    if unit == "percent":
        return f"{v:,.1f}%"
    if unit in ("integer", "plain") and v.is_integer():
        return f"{int(v):,}" if unit == "integer" else str(int(v))
    if v != 0 and abs(v) < 0.01:
        return f"{v:.3g}"
    return f"{v:,.2f}".rstrip("0").rstrip(".")


def _compact(v: float) -> str:
    """1234567 -> '1.23M'; rounding never produces '1000M' (it becomes '1B')."""
    units = ((1e6, "M"), (1e9, "B"), (1e12, "T"))
    for i, (size, letter) in enumerate(units):
        scaled = v / size
        digits = 2 if abs(scaled) < 10 else 1 if abs(scaled) < 100 else 0
        text = f"{scaled:.{digits}f}"
        if abs(float(text)) < 1000 or i == len(units) - 1:
            return (text.rstrip("0").rstrip(".") if "." in text else text) + letter
    raise AssertionError("unreachable")


def format_period(value: Any, granularity: Granularity | None) -> str:
    if granularity is None or value is None:
        return "—" if value is None else str(value)
    try:
        moment = parse_temporal(value, granularity)
    except (TypeError, ValueError):
        return str(value)
    if granularity == "year":
        return str(moment.year)
    if granularity == "month":
        return moment.strftime("%b %Y")
    if granularity == "day":
        return f"{moment.day} {moment.strftime('%b %Y')}"
    return str(value)


def shorten(text: Any, limit: int = MAX_LABEL_CHARS) -> str:
    text = "—" if text is None else str(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


# --------------------------------------------------------------- KPIs/charts
@dataclass(frozen=True)
class Kpi:
    label: str
    value: str
    detail: str | None = None  # small context line under the value
    help: str = ""             # how the number was derived


@dataclass(frozen=True)
class ChartSpec:
    kind: ChartKind
    title: str = ""
    x: int | None = None             # result-column indexes
    y: int | None = None
    series: int | None = None
    row_indexes: tuple[int, ...] = ()  # result rows to plot, in display order
    notes: tuple[str, ...] = ()
    reason: str = ""                   # why only the table is shown (kind == "table")


@dataclass(frozen=True)
class ResultInsights:
    profiles: tuple[ColumnProfile, ...]
    shape: QueryShape
    kpis: tuple[Kpi, ...]
    chart: ChartSpec


_PART_OF_WHOLE = re.compile(
    r"\b(share|shares|proportion|proportions|percentage|percent|breakdown|distribution|"
    r"split|composition|mix|part of)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _Roles:
    measures: list[ColumnProfile]
    temporal: list[ColumnProfile]
    texts: list[ColumnProfile]
    identifiers: list[ColumnProfile]
    primary: ColumnProfile | None


def _roles(profiles: Sequence[ColumnProfile], shape: QueryShape) -> _Roles:
    measures = [p for p in profiles if p.is_measure]
    primary = None
    for idx in shape.order_by:  # the measure the query ranks by is the one that matters
        primary = next((m for m in measures if m.index == idx), None)
        if primary:
            break
    if primary is None and measures:
        primary = measures[0]
    return _Roles(measures=measures,
                  temporal=[p for p in profiles if p.kind == "temporal"],
                  texts=[p for p in profiles if p.kind == "text"],
                  identifiers=[p for p in profiles if p.kind == "identifier"],
                  primary=primary)


def _unique_label(rows: Sequence[Sequence[Any]], roles: _Roles) -> ColumnProfile | None:
    """A text (else identifier) column whose values name each row exactly once.

    Null labels don't disqualify a column; those rows are left out of the chart
    (and disclosed in its notes), but they stay in the table.
    """
    for candidate in [*roles.texts, *roles.identifiers]:
        values = [str(r[candidate.index]) for r in rows if r[candidate.index] is not None]
        if len(values) >= 2 and len(set(values)) == len(values):
            return candidate
    return None


def _value_kpi(profile: ColumnProfile, value: Any) -> Kpi:
    if profile.kind == "numeric":
        shown = format_number(value, profile.unit, compact=True)
        full = format_number(value, profile.unit)
        hint = "Value returned by the query." + (f" Exact: {full}." if shown != full else "")
        return Kpi(profile.title, shown, help=hint)
    if profile.kind == "temporal":
        return Kpi(profile.title, format_period(value, profile.granularity),
                   help="Value returned by the query.")
    return Kpi(profile.title, shorten(value), help="Value returned by the query.")


def select_kpis(profiles: Sequence[ColumnProfile], rows: Sequence[Sequence[Any]],
                shape: QueryShape, truncated: bool) -> list[Kpi]:
    """Up to ``MAX_KPIS`` cards, each derived directly from the returned rows."""
    n = len(rows)
    if n == 0:
        return []
    if n == 1:
        # A single row *is* the answer: show its values (measures first).
        ordered = sorted(profiles, key=lambda p: (p.kind != "numeric", p.kind == "identifier",
                                                  p.index))
        return [_value_kpi(p, rows[0][p.index]) for p in ordered[:MAX_KPIS]]

    roles = _roles(profiles, shape)
    cards: list[Kpi] = []
    m = roles.primary
    if not truncated and m is not None:
        present = [(i, r[m.index]) for i, r in enumerate(rows) if _is_number(r[m.index])]
        values = [v for _, v in present]
        ascending = bool(shape.order_by) and shape.order_by[0] == m.index \
            and not shape.order_desc[0]
        pick = min if ascending else max
        word = "Lowest" if ascending else "Highest"
        best = pick(values) if values else None
        best_rows = [i for i, v in present if v == best]
        series_free_time = (len(roles.temporal) == 1 and not roles.texts
                            and roles.temporal[0].distinct == n)
        label = _unique_label(rows, roles)

        if best is not None and series_free_time:
            t = roles.temporal[0]
            latest = max(range(n), key=lambda i: parse_temporal(rows[i][t.index], t.granularity))
            cards.append(Kpi(
                f"{m.title} · latest", format_number(rows[latest][m.index], m.unit, compact=True),
                detail=format_period(rows[latest][t.index], t.granularity),
                help=f"{m.title} in the most recent {_lower_first(t.title)} returned.",
            ))
            cards.append(Kpi(
                f"{word} {_lower_first(m.title)}", format_number(best, m.unit, compact=True),
                detail=format_period(rows[best_rows[0]][t.index], t.granularity)
                + (" (tied)" if len(best_rows) > 1 else ""),
                help=f"{word} value of {_lower_first(m.title)} across the returned rows.",
            ))
        elif best is not None and label is not None:
            cards.append(Kpi(
                f"{word} {_lower_first(m.title)}", shorten(rows[best_rows[0]][label.index]),
                detail=format_number(best, m.unit) + (" (tied)" if len(best_rows) > 1 else ""),
                help=f"The {_lower_first(label.title)} with the {word.lower()} "
                     f"{_lower_first(m.title)} in the result.",
            ))
            if shape.parsed and shape.grouped and m.source.additive and len(values) == n:
                scope = f"across the {n} rows shown" if shape.limited else f"across all {n} rows"
                cards.append(Kpi(
                    f"Combined {_lower_first(m.title)}",
                    format_number(sum(values), m.unit, compact=True), detail=scope,
                    help=f"Sum of {_lower_first(m.title)} over the returned groups. Each group "
                         "is counted once, so nothing is double-counted.",
                ))
            elif _average_allowed(m, shape) and values:
                cards.append(_average_kpi(m, values))
        elif m is not None and not roles.texts and not roles.temporal \
                and _average_allowed(m, shape) and values:
            cards.append(_average_kpi(m, values))

    cards.append(Kpi(
        "Rows returned", f"{n:,}+" if truncated else f"{n:,}",
        detail="Row limit reached; more rows exist" if truncated else None,
        help="Number of rows the query returned"
             + (" before hitting the row limit." if truncated else "."),
    ))
    return cards[:MAX_KPIS]


def _average_allowed(m: ColumnProfile, shape: QueryShape) -> bool:
    # Plain row values from base tables only: never averages of aggregates,
    # never a LIMIT-ed (top-N) subset.
    return shape.parsed and m.source.aggregate is None and not shape.limited


def _average_kpi(m: ColumnProfile, values: list[Any]) -> Kpi:
    return Kpi(f"Average {_lower_first(m.title)}",
               format_number(sum(values) / len(values), m.unit, compact=True),
               detail=f"across {len(values)} rows",
               help=f"Mean of {_lower_first(m.title)} over the returned rows.")


def _table(reason: str) -> ChartSpec:
    return ChartSpec(kind="table", reason=reason)


def _plottable(rows: Sequence[Sequence[Any]], *cols: int) -> tuple[list[int], int]:
    keep = [i for i, r in enumerate(rows)
            if all(r[c] is not None for c in cols)
            and all(not _is_number(r[c]) or math.isfinite(r[c]) for c in cols)]
    return keep, len(rows) - len(keep)


def _common_notes(dropped: int, truncated: bool, n: int) -> list[str]:
    notes = []
    if dropped:
        notes.append(f"{dropped} row(s) with missing values are not plotted.")
    if truncated:
        notes.append(f"Based on the first {n} rows returned; the result was truncated.")
    return notes


def choose_chart(profiles: Sequence[ColumnProfile], rows: Sequence[Sequence[Any]],
                 shape: QueryShape, truncated: bool, question: str = "") -> ChartSpec:
    """Pick a chart that faithfully shows the returned rows, or ``table``."""
    n = len(rows)
    if n == 0:
        return _table("The query returned no rows.")
    if n == 1:
        return _table("A single row is shown as KPI cards rather than a one-bar chart.")
    roles = _roles(profiles, shape)
    m = roles.primary
    if m is None:
        return _table("The result has no numeric values to chart.")
    others = [p.title for p in roles.measures if p is not m]
    other_note = ([f"Showing {_lower_first(m.title)}; "
                   f"{', '.join(others)} {'is' if len(others) == 1 else 'are'} in the table."]
                  if others else [])

    # ---- trend over time
    if roles.temporal:
        if len(roles.temporal) > 1 or len(roles.texts) > 1:
            return _table("Several date or text columns: the right chart is ambiguous.")
        t = roles.temporal[0]
        series = roles.texts[0] if roles.texts else None
        key_cols = [t.index] + ([series.index] if series else [])
        keep, dropped = _plottable(rows, *key_cols, m.index)
        keys = [tuple(rows[i][c] for c in key_cols) for i in keep]
        if len(set(keys)) != len(keys):
            return _table("Some periods appear more than once; charting them would "
                          "require aggregating the results.")
        if len({rows[i][t.index] for i in keep}) < 2:
            return _table("Fewer than two periods to draw a trend.")
        if series:
            per_series: dict[Any, int] = {}
            for i in keep:
                per_series[rows[i][series.index]] = per_series.get(rows[i][series.index], 0) + 1
            if len(per_series) > MAX_SERIES:
                return _table(f"More than {MAX_SERIES} series: too many lines to read reliably.")
            if min(per_series.values()) < 2:
                return _table("Some series have a single period, so there is no line to draw.")
        ordered = sorted(keep, key=lambda i: parse_temporal(rows[i][t.index], t.granularity))
        split = f", split by {_lower_first(series.title)}" if series else ""
        return ChartSpec(
            kind="line", title=f"{m.title} by {_lower_first(t.title)}{split}",
            x=t.index, y=m.index, series=series.index if series else None,
            row_indexes=tuple(ordered),
            notes=tuple(other_note + _common_notes(dropped, truncated, n)),
        )

    # ---- one value per category
    label = _unique_label(rows, roles)
    if label is not None:
        keep, dropped = _plottable(rows, label.index, m.index)
        if len(keep) < 2:
            return _table("Fewer than two values to compare.")
        values = [rows[i][m.index] for i in keep]
        part_of_whole = (
            bool(_PART_OF_WHOLE.search(question or "")) and shape.parsed and m.source.additive
            and not shape.limited and not truncated and not dropped
            and 2 <= len(keep) <= MAX_DONUT_SLICES and min(values) >= 0 and sum(values) > 0
        )
        by_value = sorted(keep, key=lambda i: rows[i][m.index], reverse=True)
        if part_of_whole:
            return ChartSpec(kind="donut", title=f"Share of {_lower_first(m.title)} by "
                                                 f"{_lower_first(label.title)}",
                             x=label.index, y=m.index, row_indexes=tuple(by_value),
                             notes=tuple(other_note))
        # Keep the query's own order when it sorts by the label (e.g. size bands) or
        # by this measure (e.g. "lowest five"); otherwise rank by value, highest first.
        query_ordered = bool(shape.order_by) and shape.order_by[0] in (label.index, m.index)
        ordered = keep if query_ordered else by_value
        notes = other_note + _common_notes(dropped, truncated, n)
        if len(ordered) > MAX_BAR_CATEGORIES:
            ranked_desc = not query_ordered or (shape.order_by[0] == m.index
                                                and shape.order_desc[0])
            how = "top" if ranked_desc else "first"
            notes.insert(0, f"Showing the {how} {MAX_BAR_CATEGORIES} of {len(ordered)} "
                            f"{_lower_first(label.title)} values; all rows are in the table.")
            ordered = ordered[:MAX_BAR_CATEGORIES]
        return ChartSpec(kind="bar", title=f"{m.title} by {_lower_first(label.title)}",
                         x=label.index, y=m.index, row_indexes=tuple(ordered),
                         notes=tuple(notes))

    if roles.texts:
        return _table("Labels repeat across rows; charting them would require "
                      "aggregating the results.")

    # ---- relationship between two measures
    if len(roles.measures) >= 2:
        x = next(p for p in roles.measures if p is not m)
        keep, dropped = _plottable(rows, x.index, m.index)
        if len(keep) < MIN_SCATTER_POINTS:
            return _table("Too few points to show a relationship.")
        extra = [p.title for p in roles.measures if p not in (m, x)]
        notes = ([f"{', '.join(extra)} {'is' if len(extra) == 1 else 'are'} in the table."]
                 if extra else []) + _common_notes(dropped, truncated, n)
        return ChartSpec(kind="scatter", title=f"{m.title} vs {_lower_first(x.title)}",
                         x=x.index, y=m.index, row_indexes=tuple(keep), notes=tuple(notes))

    return _table("There is no category or time column to plot the values against.")


def analyze_results(results: dict[str, Any] | None, sql: str | None = None,
                    question: str = "") -> ResultInsights:
    """Profile ``results`` (as returned by the executor) and pick KPIs + chart."""
    columns = list((results or {}).get("columns") or [])
    rows = list((results or {}).get("rows") or [])
    truncated = bool((results or {}).get("truncated"))
    shape = analyze_sql(sql, len(columns))
    profiles = tuple(
        profile_column(i, name, [r[i] for r in rows], shape.column(i))
        for i, name in enumerate(columns)
    )
    return ResultInsights(
        profiles=profiles,
        shape=shape,
        kpis=tuple(select_kpis(profiles, rows, shape, truncated)),
        chart=choose_chart(profiles, rows, shape, truncated, question),
    )
