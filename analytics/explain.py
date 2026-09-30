"""Explain an executed SQL query in plain language: what was read, joined,
filtered, grouped, calculated and sorted.

Pure functions only: the validated SQL is *parsed* with sqlglot, never
executed. Every sentence is derived from a node of the query's syntax tree,
so the explanation cannot describe a join, filter or formula the query does
not contain. Constructs without a dedicated rule are shown as their SQL
fragment rather than paraphrased, and anything the parser cannot handle
(set operations, unparsable text) yields ``parsed=False`` so the UI can say
so instead of guessing.

SQL fragments are regenerated from the syntax tree in the SQLite dialect, so
spacing and letter case may differ from the executed text; the meaning is
identical. The exact executed SQL is always shown separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sqlglot
from sqlglot import exp

MAX_FRAGMENT_CHARS = 160

_OPERATORS: dict[type[exp.Expression], str] = {
    exp.Mul: "×", exp.Div: "÷", exp.Add: "+", exp.Sub: "−",
}
_COMPARISONS: dict[type[exp.Expression], str] = {
    exp.EQ: "is", exp.NEQ: "is not", exp.GT: "is greater than", exp.GTE: "is at least",
    exp.LT: "is less than", exp.LTE: "is at most",
}
_TIME_PARTS = {"%Y-%m": "month", "%Y": "year", "%Y-%m-%d": "day", "%m": "month number",
               "%W": "week number", "%H": "hour"}


@dataclass(frozen=True)
class Item:
    """One explained element: a plain-language sentence plus its SQL fragment."""

    plain: str
    sql: str = ""


@dataclass(frozen=True)
class QueryBreakdown:
    parsed: bool
    tables: tuple[Item, ...] = ()
    joins: tuple[Item, ...] = ()
    filters: tuple[Item, ...] = ()
    grouping: tuple[Item, ...] = ()
    group_filters: tuple[Item, ...] = ()
    metrics: tuple[Item, ...] = ()
    ordering: tuple[Item, ...] = ()
    steps: tuple[tuple[str, QueryBreakdown], ...] = ()
    notes: tuple[str, ...] = ()


def _fragment(node: exp.Expression) -> str:
    text = node.sql(dialect="sqlite")
    return text if len(text) <= MAX_FRAGMENT_CHARS else text[: MAX_FRAGMENT_CHARS - 1] + "…"


class _Describer:
    """Turns expressions into phrases, resolving table aliases to table names."""

    def __init__(self, aliases: dict[str, str], outputs: list[tuple[str, exp.Expression]]):
        self.aliases = aliases      # alias (lower) -> display name of its source
        self.outputs = outputs      # (output name, expression) per SELECT column

    # ---------------------------------------------------------- values
    def column(self, col: exp.Column) -> str:
        if isinstance(col.this, exp.Star):
            source = self.aliases.get(col.table.lower(), col.table) if col.table else ""
            return f"all columns of {source}" if source else "all columns"
        if col.table:
            return f"{self.aliases.get(col.table.lower(), col.table)}.{col.name}"
        return col.name

    def expr(self, node: Any) -> str:
        if node is None:
            return ""
        if isinstance(node, exp.Alias):
            return self.expr(node.this)
        if isinstance(node, exp.Column):
            return self.column(node)
        if isinstance(node, exp.Star):
            return "all columns"
        if isinstance(node, exp.Literal):
            return f"'{node.this}'" if node.is_string else str(node.this)
        if isinstance(node, exp.Null):
            return "NULL"
        if isinstance(node, exp.Boolean):
            return "true" if node.this else "false"
        if isinstance(node, exp.Paren):
            inner = self.expr(node.this)
            return f"({inner})" if isinstance(node.this, exp.Binary) else inner
        if isinstance(node, exp.Neg):
            return f"−{self.expr(node.this)}"
        for cls, symbol in _OPERATORS.items():
            if type(node) is cls:
                return f"{self.expr(node.left)} {symbol} {self.expr(node.right)}"
        if isinstance(node, exp.Window):
            return self.window(node)
        if isinstance(node, exp.AggFunc):
            return self.aggregate(node)
        if isinstance(node, exp.Round):
            decimals = node.args.get("decimals")
            places = f"{decimals.this} decimal{'s' if str(decimals.this) != '1' else ''}" \
                if isinstance(decimals, exp.Literal) else "a whole number"
            return f"{self.expr(node.this)}, rounded to {places}"
        if isinstance(node, exp.Cast):
            return f"{self.expr(node.this)} converted to {node.to.sql(dialect='sqlite')}"
        if isinstance(node, exp.Coalesce):
            fallbacks = [self.expr(e) for e in node.expressions]
            return f"{self.expr(node.this)} (or {', then '.join(fallbacks)} when missing)"
        if isinstance(node, exp.TimeToStr):
            return self.time_part(node)
        if isinstance(node, (exp.TsOrDsToTimestamp, exp.TsOrDsToDate)):
            return self.expr(node.this)
        if isinstance(node, exp.Case):
            return f"a conditional value ({_fragment(node)})"
        if isinstance(node, exp.Subquery):
            return "the result of a subquery"
        return _fragment(node)  # no rule: show the exact fragment rather than paraphrase

    def aggregate(self, node: exp.AggFunc) -> str:
        inner = node.this
        if isinstance(node, exp.Count):
            if inner is None or isinstance(inner, exp.Star):
                return "number of rows"
            if isinstance(inner, exp.Distinct):
                return "number of distinct " + ", ".join(self.expr(e) for e in inner.expressions)
            return f"number of rows with a value in {self.expr(inner)}"
        if isinstance(inner, exp.Distinct):
            inner_text = "distinct " + ", ".join(self.expr(e) for e in inner.expressions)
        else:
            inner_text = self.expr(inner)
        name = {exp.Sum: "sum", exp.Avg: "average", exp.Min: "lowest", exp.Max: "highest"}
        for cls, word in name.items():
            if type(node) is cls:
                return f"{word} of {inner_text}"
        return _fragment(node)

    def window(self, node: exp.Window) -> str:
        parts = []
        partition = node.args.get("partition_by")
        if partition:
            parts.append("within each " + ", ".join(self.expr(p) for p in partition))
        order = node.args.get("order")
        if order:
            parts.append("ordered by " + ", ".join(self.expr(o.this) for o in order.expressions))
        base = self.expr(node.this)
        if isinstance(node.this, exp.Sum) and order:
            base = "running " + base
        return f"{base}, computed per row" + (f" {' '.join(parts)}" if parts else "")

    def time_part(self, node: exp.TimeToStr) -> str:
        fmt = node.args.get("format")
        fmt_text = fmt.this if isinstance(fmt, exp.Literal) else None
        part = _TIME_PARTS.get(fmt_text or "")
        target = self.expr(node.this)
        return f"{part} of {target}" if part else f"{target} formatted as '{fmt_text}'"

    # ---------------------------------------------------------- conditions
    def condition(self, node: Any) -> str:
        if isinstance(node, exp.Paren):
            return self.condition(node.this)
        if isinstance(node, exp.And):
            return f"{self.condition(node.left)} and {self.condition(node.right)}"
        if isinstance(node, exp.Or):
            return f"either {self.condition(node.left)} or {self.condition(node.right)}"
        for cls, words in _COMPARISONS.items():
            if type(node) is cls:
                if cls in (exp.EQ, exp.NEQ) and isinstance(node.right, exp.Column):
                    words = "equals" if cls is exp.EQ else "does not equal"  # column = column
                return f"{self.expr(node.left)} {words} {self.expr(node.right)}"
        if isinstance(node, exp.In):
            if node.args.get("query") is not None:
                return f"{self.expr(node.this)} appears in the result of a subquery"
            values = ", ".join(self.expr(e) for e in node.expressions)
            return f"{self.expr(node.this)} is one of {values}"
        if isinstance(node, exp.Between):
            return (f"{self.expr(node.this)} is between {self.expr(node.args.get('low'))} "
                    f"and {self.expr(node.args.get('high'))} (inclusive)")
        if isinstance(node, (exp.Like, exp.ILike)):
            return f"{self.expr(node.this)} matches the pattern {self.expr(node.expression)}"
        if isinstance(node, exp.Is) and isinstance(node.expression, exp.Null):
            return f"{self.expr(node.this)} is missing (NULL)"
        if isinstance(node, exp.Not):
            inner = node.this
            if isinstance(inner, exp.Is) and isinstance(inner.expression, exp.Null):
                return f"{self.expr(inner.this)} is present (not NULL)"
            if isinstance(inner, exp.In):
                return self.condition(inner).replace(" is one of ", " is not one of ", 1) \
                    .replace(" appears in ", " does not appear in ", 1)
            return f"not ({self.condition(inner)})"
        if isinstance(node, exp.Exists):
            return "a matching row exists in a subquery"
        return _fragment(node)

    def output_name(self, node: exp.Expression) -> str:
        """Name of a GROUP BY / ORDER BY key: ordinals and aliases map to output columns."""
        if isinstance(node, exp.Literal) and not node.is_string and str(node.this).isdigit():
            idx = int(node.this) - 1
            if 0 <= idx < len(self.outputs):
                return self.outputs[idx][0]
        if isinstance(node, exp.Column) and not node.table:
            for name, _ in self.outputs:
                if name.lower() == node.name.lower():
                    return name
        for name, expression in self.outputs:
            if expression == node:
                return name
        return self.expr(node)


def _source_name(node: exp.Expression, cte_names: dict[str, int]) -> tuple[str, str]:
    """(alias key, display name) of a FROM/JOIN source."""
    if isinstance(node, exp.Table):
        name = node.name
        alias = node.alias_or_name
        if name.lower() in cte_names:
            return alias.lower(), f"{name} (step {cte_names[name.lower()]})"
        return alias.lower(), name
    if isinstance(node, exp.Subquery):
        alias = node.alias_or_name or "subquery"
        return alias.lower(), f"subquery {alias}"
    return "", _fragment(node)


def _flatten_and(node: Any) -> list[exp.Expression]:
    while isinstance(node, exp.Paren) and isinstance(node.this, exp.And):
        node = node.this
    if isinstance(node, exp.And):
        return _flatten_and(node.left) + _flatten_and(node.right)
    return [node]


def _explain_select(tree: exp.Select, cte_names: dict[str, int]) -> QueryBreakdown:
    sources = [tree.args.get("from_") or tree.args.get("from")]
    joins = list(tree.args.get("joins") or [])
    aliases: dict[str, str] = {}
    table_items: list[Item] = []
    notes: list[str] = []
    base_display = ""

    from_node = sources[0].this if sources[0] is not None else None
    for node in [from_node, *(j.this for j in joins)]:
        if node is None:
            continue
        key, display = _source_name(node, cte_names)
        if key:
            aliases[key] = display
        if not base_display:
            base_display = display
        alias_note = f" (as {node.alias})" if getattr(node, "alias", "") and \
            node.alias.lower() != getattr(node, "name", "").lower() else ""
        table_items.append(Item(display + alias_note, _fragment(node)))
        if isinstance(node, exp.Subquery):
            notes.append(f"Reads from a subquery ({display}); its details are in the SQL tab.")

    outputs = [(p.alias_or_name or _fragment(p), p.this if isinstance(p, exp.Alias) else p)
               for p in tree.expressions]
    d = _Describer(aliases, outputs)

    join_items = []
    for j in joins:
        _, target = _source_name(j.this, cte_names)
        side = (j.args.get("side") or "").upper()
        kind = (j.args.get("kind") or "").upper()
        on, using = j.args.get("on"), j.args.get("using")
        if on is not None:
            match = f", matching rows where {d.condition(on)}"
        elif using:
            match = ", matching rows on " + ", ".join(i.name for i in using)
        else:
            match = ""
        if kind == "CROSS" or (not on and not using and not side):
            plain = f"Cross join with {target}: every combination of rows is formed"
        elif side == "LEFT":
            plain = (f"Left join with {target}{match}. Every row from {base_display} is kept, "
                     f"even without a match in {target}")
        elif side == "RIGHT":
            plain = (f"Right join with {target}{match}. Every row from {target} is kept, "
                     "even without a match")
        elif side == "FULL":
            plain = f"Full join with {target}{match}. Rows from both sides are kept"
        else:
            plain = f"Inner join with {target}{match}. Only rows with a match on both sides are kept"
        join_items.append(Item(plain, _fragment(j)))

    where = tree.args.get("where")
    # Sentences start with identifiers, which are never re-cased.
    filters = tuple(Item(d.condition(c), _fragment(c))
                    for c in (_flatten_and(where.this) if where else []))
    if any(c.find(exp.Subquery) for c in (_flatten_and(where.this) if where else [])):
        notes.append("A filter uses a subquery; its details are in the SQL tab.")

    group = tree.args.get("group")
    grouping: tuple[Item, ...] = ()
    if group and group.expressions:
        keys = [d.output_name(g) for g in group.expressions]
        grouping = (Item("One row per " + " and ".join(keys), _fragment(group)),)

    having = tree.args.get("having")
    group_filters = tuple(
        Item("Keep only groups where " + d.condition(c), _fragment(c))
        for c in (_flatten_and(having.this) if having else []))

    metrics = []
    only_source = base_display if len(table_items) == 1 else ""
    for (name, expression), projection in zip(outputs, tree.expressions):
        if isinstance(expression, exp.Column) and not isinstance(expression.this, exp.Star) \
                and expression.name.lower() == name.lower():
            source = aliases.get(expression.table.lower(), expression.table) \
                if expression.table else only_source
            plain = f"{name}: taken directly from {source or 'the data'}"
        elif isinstance(expression, exp.Star) or (isinstance(expression, exp.Column)
                                                  and isinstance(expression.this, exp.Star)):
            plain = d.expr(expression)
        else:
            plain = f"{name} = {d.expr(expression)}"
        metrics.append(Item(plain, _fragment(projection)))
    if tree.args.get("distinct"):
        notes.append("Duplicate rows are removed (SELECT DISTINCT).")

    ordering = []
    order = tree.args.get("order")
    for o in (order.expressions if order else []):
        direction = "descending (highest first)" if o.args.get("desc") \
            else "ascending (lowest first)"
        ordering.append(Item(f"Sorted by {d.output_name(o.this)}, {direction}", _fragment(o)))
    limit = tree.args.get("limit")
    offset = tree.args.get("offset")
    if limit is not None:
        count = d.expr(limit.expression)
        skip = f", after skipping the first {d.expr(offset.expression)}" if offset else ""
        kept = "Only the first" if not ordering else "Only the top"
        ordering.append(Item(f"{kept} {count} rows are kept{skip}", _fragment(limit)))

    return QueryBreakdown(
        parsed=True, tables=tuple(table_items), joins=tuple(join_items), filters=filters,
        grouping=grouping, group_filters=group_filters, metrics=tuple(metrics),
        ordering=tuple(ordering), notes=tuple(notes),
    )


def explain_sql(sql: str | None) -> QueryBreakdown:
    """Break the executed SQL down into explained steps (see module docstring)."""
    if not sql or not sql.strip():
        return QueryBreakdown(parsed=False)
    try:
        tree = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return QueryBreakdown(parsed=False)
    if not isinstance(tree, exp.Select):
        return QueryBreakdown(parsed=False)  # e.g. UNION: shown as SQL only

    with_ = tree.args.get("with_") or tree.args.get("with")
    ctes = list(with_.expressions) if with_ else []
    cte_names: dict[str, int] = {}
    steps: list[tuple[str, QueryBreakdown]] = []
    for number, cte in enumerate(ctes, start=1):
        name = cte.alias_or_name
        body = cte.this
        step = _explain_select(body, cte_names) if isinstance(body, exp.Select) \
            else QueryBreakdown(parsed=False)
        steps.append((name, step))
        cte_names[name.lower()] = number

    main = _explain_select(tree, cte_names)
    return QueryBreakdown(
        parsed=True, tables=main.tables, joins=main.joins, filters=main.filters,
        grouping=main.grouping, group_filters=main.group_filters, metrics=main.metrics,
        ordering=main.ordering, steps=tuple(steps), notes=main.notes,
    )


SECTIONS = (
    ("tables", "Data read"),
    ("joins", "How tables were combined"),
    ("filters", "Filters applied"),
    ("grouping", "Grouping"),
    ("group_filters", "Group filters"),
    ("metrics", "Columns calculated"),
    ("ordering", "Sorting and limits"),
)


def breakdown_to_text(breakdown: QueryBreakdown, indent: str = "") -> str:
    """Deterministic plain-text form (the grounding given to the LLM narrative)."""
    if not breakdown.parsed:
        return f"{indent}(The query structure could not be broken down automatically.)"
    lines: list[str] = []
    for number, (name, step) in enumerate(breakdown.steps, start=1):
        lines.append(f"{indent}Step {number}: intermediate result '{name}'")
        lines.append(breakdown_to_text(step, indent + "  "))
    if breakdown.steps:
        lines.append(f"{indent}Final step:")
    for attr, title in SECTIONS:
        items = getattr(breakdown, attr)
        if items:
            lines.append(f"{indent}{title}:")
            lines.extend(f"{indent}- {item.plain}" for item in items)
    lines.extend(f"{indent}Note: {note}" for note in breakdown.notes)
    return "\n".join(lines)
