"""Altair charts for a ``ChartSpec`` chosen by ``analytics.insights``.

The chart is built only from the rows the protected executor returned (the
spec names which rows and columns to plot). Values are copied into a small
chart-only DataFrame; the query result itself is never modified.

Visual spec (one system for every chart):
* Categorical palette below, validated with the dataviz palette checker
  against the app's dark surface (#0e131f) and card surface (#151c2c): every
  check passes with no warnings, including the donut's last-to-first pair.
  Single-series charts always use slot 1; colors follow the entity, never rank.
* Bars <= 24px with 4px rounded data-ends; 2px lines; 8px markers with a 2px
  surface ring; solid hairline gridlines; tooltips on every mark.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import altair as alt
import pandas as pd

from analytics.insights import (
    MAX_BAR_VALUE_LABELS,
    ChartSpec,
    ColumnProfile,
    Granularity,
    format_period,
    parse_temporal,
)

# Validated categorical order (violet, cyan, orange, magenta, blue, yellow).
PALETTE = ("#8568db", "#139cb6", "#cf6f19", "#c74c8e", "#3a7bcb", "#b18e15")
SURFACE = "#0e131f"          # must match backgroundColor in .streamlit/config.toml
INK_SECONDARY = "#b4bccb"
INK_MUTED = "#8a93a6"
GRID = "#1e2636"
BASELINE = "#2c3548"
FONT = '"Source Sans", "Source Sans Pro", system-ui, -apple-system, "Segoe UI", sans-serif'

_TIME_FORMATS: dict[Granularity, str] = {
    "year": "%Y", "month": "%b %Y", "day": "%d %b %Y", "datetime": "%d %b %Y %H:%M",
}


def _number_formats(profile: ColumnProfile) -> tuple[dict[str, Any], str]:
    """(axis kwargs, tooltip d3-format) for a measure column."""
    peak = max(abs(profile.minimum or 0), abs(profile.maximum or 0))
    if profile.unit == "percent":
        axis = {"labelExpr": "format(datum.value, ',.1~f') + '%'"}
        return axis, ",.2f"
    tooltip = ",.2f" if profile.unit in ("money", "decimal") else ",.0f"
    if peak >= 1_000_000:
        # d3's SI prefix uses G for billions; business readers expect B.
        return {"labelExpr": "replace(format(datum.value, '.3~s'), 'G', 'B')"}, tooltip
    if profile.unit == "integer":
        return {"format": ",.0f"}, tooltip
    return {"format": ",.2~f"}, tooltip


def _frame(rows: Sequence[Sequence[Any]], spec: ChartSpec, **columns: int) -> pd.DataFrame:
    return pd.DataFrame(
        [{key: rows[i][col] for key, col in columns.items()} for i in spec.row_indexes]
    )


def _bar(spec: ChartSpec, rows: Sequence[Sequence[Any]],
         label: ColumnProfile, measure: ColumnProfile) -> alt.TopLevelMixin:
    df = _frame(rows, spec, label=label.index, value=measure.index)
    df["label"] = df["label"].astype(str)
    axis_fmt, tip_fmt = _number_formats(measure)
    order = list(df["label"])
    values = [float(v) for v in df["value"]]
    show_values = len(df) <= MAX_BAR_VALUE_LABELS and min(values) >= 0
    # Leave room at the bar tips for the value labels so they are never clipped.
    domain_max = max(values) * (1.18 if show_values else 1.0)
    scale = alt.Scale(domain=[min(0.0, min(values)), domain_max], nice=False) \
        if show_values and domain_max > 0 else alt.Scale()

    base = alt.Chart(df).encode(
        y=alt.Y("label:N", sort=order, title=label.title,
                axis=alt.Axis(labelLimit=220, ticks=False, domain=False, labelPadding=8)),
        x=alt.X("value:Q", title=measure.title, scale=scale,
                axis=alt.Axis(tickCount=5, domain=False, ticks=False, **axis_fmt)),
        tooltip=[alt.Tooltip("label:N", title=label.title),
                 alt.Tooltip("value:Q", title=measure.title, format=tip_fmt)],
    )
    chart = base.mark_bar(color=PALETTE[0], size=18, cornerRadiusEnd=4)
    if show_values:
        chart = chart + base.mark_text(align="left", dx=6, color=INK_SECONDARY,
                                       fontSize=12).encode(
            text=alt.Text("value:Q", format=tip_fmt))
    return chart.properties(height=max(140, 34 * len(df) + 50))


def _line(spec: ChartSpec, rows: Sequence[Sequence[Any]], time: ColumnProfile,
          measure: ColumnProfile, series: ColumnProfile | None) -> alt.TopLevelMixin:
    cols = {"t": time.index, "value": measure.index}
    if series is not None:
        cols["series"] = series.index
    df = _frame(rows, spec, **cols)
    granularity = time.granularity or "day"
    axis_fmt, tip_fmt = _number_formats(measure)

    if granularity in ("month", "year"):
        # Whole periods are discrete, evenly spaced steps: a point axis with
        # pre-formatted labels avoids the edge-label collisions and the dropped
        # last label of a continuous time axis. Rows arrive in time order.
        df["t"] = [format_period(v, granularity) for v in df["t"]]
        periods = list(dict.fromkeys(df["t"]))
        x = alt.X("t:O", sort=periods, title=time.title,
                  scale=alt.Scale(padding=0.3),
                  axis=alt.Axis(grid=False, labelOverlap="greedy",
                                labelAngle=0 if len(periods) <= 12 else -40))
        time_tip = alt.Tooltip("t:N", title=time.title)
    else:
        df["t"] = [parse_temporal(v, granularity) for v in df["t"]]
        time_fmt = _TIME_FORMATS[granularity]
        x = alt.X("t:T", title=time.title, scale=alt.Scale(padding=14),  # off the y-axis
                  axis=alt.Axis(format=time_fmt, labelAngle=0, grid=False, tickCount=8,
                                labelFlush=False, labelOverlap="greedy"))
        time_tip = alt.Tooltip("t:T", title=time.title, format=time_fmt)

    tooltip = [time_tip, alt.Tooltip("value:Q", title=measure.title, format=tip_fmt)]
    if series is not None:
        names = sorted(df["series"].astype(str).unique())  # stable: color follows the entity
        df["series"] = df["series"].astype(str)
        color: Any = alt.Color("series:N", title=series.title,
                               scale=alt.Scale(domain=names, range=list(PALETTE[:len(names)])),
                               legend=alt.Legend(orient="top", direction="horizontal"))
        tooltip.insert(0, alt.Tooltip("series:N", title=series.title))
    else:
        color = alt.value(PALETTE[0])

    base = alt.Chart(df).encode(
        x=x,
        y=alt.Y("value:Q", title=measure.title, axis=alt.Axis(tickCount=5, domain=False,
                                                              ticks=False, **axis_fmt)),
        color=color,
    )
    layers = [base.mark_line(strokeWidth=2, strokeCap="round", strokeJoin="round")]
    if len(df) <= 60:
        layers.append(base.mark_point(filled=True, size=64, opacity=1, stroke=SURFACE,
                                      strokeWidth=2))
    # Invisible, larger hit targets so hovering doesn't require pixel precision.
    layers.append(base.mark_point(size=500, opacity=0).encode(tooltip=tooltip))
    return alt.layer(*layers).properties(height=320)


def _donut(spec: ChartSpec, rows: Sequence[Sequence[Any]],
           label: ColumnProfile, measure: ColumnProfile) -> alt.TopLevelMixin:
    df = _frame(rows, spec, label=label.index, value=measure.index)
    df["label"] = df["label"].astype(str)
    df["share"] = df["value"] / float(df["value"].sum())
    _, tip_fmt = _number_formats(measure)
    order = list(df["label"])  # spec order: largest first
    return alt.Chart(df).mark_arc(innerRadius=70, outerRadius=120, stroke=SURFACE,
                                  strokeWidth=2).encode(
        theta=alt.Theta("value:Q", stack=True),
        order=alt.Order("value:Q", sort="descending"),
        color=alt.Color("label:N", title=label.title,
                        scale=alt.Scale(domain=order, range=list(PALETTE[:len(order)])),
                        sort=order, legend=alt.Legend(orient="right")),
        tooltip=[alt.Tooltip("label:N", title=label.title),
                 alt.Tooltip("value:Q", title=measure.title, format=tip_fmt),
                 alt.Tooltip("share:Q", title="Share of total", format=".1%")],
    ).properties(height=280)


def _scatter(spec: ChartSpec, rows: Sequence[Sequence[Any]],
             x: ColumnProfile, y: ColumnProfile) -> alt.TopLevelMixin:
    df = _frame(rows, spec, x=x.index, y=y.index)
    x_axis, x_tip = _number_formats(x)
    y_axis, y_tip = _number_formats(y)
    return alt.Chart(df).mark_circle(size=72, color=PALETTE[0], opacity=0.9, stroke=SURFACE,
                                     strokeWidth=2).encode(
        # padding keeps extreme points (and their surface ring) off the plot edges
        x=alt.X("x:Q", title=x.title, scale=alt.Scale(zero=False, padding=12),
                axis=alt.Axis(tickCount=6, **x_axis)),
        y=alt.Y("y:Q", title=y.title, scale=alt.Scale(zero=False, padding=12),
                axis=alt.Axis(tickCount=5, domain=False, ticks=False, **y_axis)),
        tooltip=[alt.Tooltip("x:Q", title=x.title, format=x_tip),
                 alt.Tooltip("y:Q", title=y.title, format=y_tip)],
    ).properties(height=320)


def _style(chart: alt.TopLevelMixin) -> alt.TopLevelMixin:
    return (
        chart.configure(background="transparent", font=FONT, padding={"left": 4, "right": 12,
                                                                       "top": 8, "bottom": 4})
        .configure_view(stroke=None)
        .configure_axis(labelColor=INK_MUTED, titleColor=INK_SECONDARY, gridColor=GRID,
                        gridWidth=1, domainColor=BASELINE, tickColor=BASELINE,
                        labelFontSize=12, titleFontSize=12, titleFontWeight="normal",
                        titlePadding=10)
        .configure_legend(labelColor=INK_SECONDARY, titleColor=INK_MUTED, labelFontSize=12,
                          titleFontSize=12, titleFontWeight="normal", symbolType="circle",
                          symbolSize=90, labelLimit=220)
    )


def build_chart(spec: ChartSpec, rows: Sequence[Sequence[Any]],
                profiles: Sequence[ColumnProfile]) -> alt.TopLevelMixin | None:
    """Altair chart for ``spec``, or None when the spec is table-only."""
    if spec.kind == "table" or spec.x is None or spec.y is None:
        return None
    px, py = profiles[spec.x], profiles[spec.y]
    if spec.kind == "bar":
        chart = _bar(spec, rows, px, py)
    elif spec.kind == "line":
        series = profiles[spec.series] if spec.series is not None else None
        chart = _line(spec, rows, px, py, series)
    elif spec.kind == "donut":
        chart = _donut(spec, rows, px, py)
    elif spec.kind == "scatter":
        chart = _scatter(spec, rows, px, py)
    else:
        raise ValueError(f"Unknown chart kind: {spec.kind}")
    return _style(chart)
