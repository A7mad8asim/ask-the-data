"""SQL sandbox: decide whether a model-written query may run.

The query is parsed into a syntax tree with sqlglot and checked structurally, not by
keyword matching, so a rename or a comment cannot hide a forbidden statement.

Rules
1. Exactly one read-only query (SELECT / WITH ... SELECT / set operations).
2. Only tables from the data mart (plus the query's own CTEs); no table functions,
   no schema-qualified names, no file-reading or system functions.
3. Privacy: results must be aggregated. Identifier columns (patient_id, visit_id, ...)
   may only appear inside COUNT(...), never as output columns or group keys, and a
   query that returns raw rows from a patient-level table is rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import sqlglot
from sqlglot import exp

from .schema import IDENTIFIER_COLUMNS, PATIENT_TABLES, TABLES

ALLOWED_TABLES = frozenset(TABLES)

BLOCKED_FUNCTIONS = frozenset(
    {
        "glob", "query", "query_table", "getenv", "current_setting", "getvariable", "setvariable",
        "load_extension", "install_extension", "checkpoint", "force_checkpoint", "enable_profiling",
        "disable_profiling", "which_secret", "read_text", "read_blob",
    }
)
BLOCKED_FUNCTION_PREFIXES = (
    "read_", "duckdb_", "pragma_", "parquet_", "sniff_", "iceberg_", "delta_", "sqlite_",
    "postgres_", "mysql_", "http", "json_execute",
)
# Aggregates that pack individual values into one cell, which would undo the aggregation
ROW_LEVEL_AGGREGATES = (exp.ArrayAgg, exp.GroupConcat)
ROW_LEVEL_AGGREGATE_NAMES = frozenset({"list", "histogram", "string_agg", "listagg", "group_concat", "array_agg"})

_BLOCKED_STATEMENT_NAMES = [
    "Insert", "Update", "Delete", "Create", "Drop", "Alter", "Command", "Copy", "Attach", "Detach",
    "Pragma", "Install", "Set", "Use", "Into", "Merge", "TruncateTable", "Transaction", "Commit",
    "Rollback", "LoadData", "Describe", "Show", "Summarize", "Grant", "Kill", "Analyze", "Cache",
    "Uncache", "Refresh",
]
BLOCKED_STATEMENTS = tuple(getattr(exp, n) for n in _BLOCKED_STATEMENT_NAMES if hasattr(exp, n))


@dataclass(frozen=True)
class Verdict:
    ok: bool
    code: str = "ok"
    message: str = ""

    @property
    def is_privacy(self) -> bool:
        return self.code.startswith("privacy")


OK = Verdict(True)


def check_sql(sql: str) -> Verdict:
    """Return OK, or a Verdict whose message explains the rule that was broken.

    The message is written so it can be sent back to the model as repair feedback.
    """
    text = (sql or "").strip().rstrip(";").strip()
    if not text:
        return Verdict(False, "empty", "The query is empty.")
    try:
        statements = [s for s in sqlglot.parse(text, read="duckdb") if s is not None]
    except sqlglot.errors.ParseError as e:
        first = str(e).splitlines()[0] if str(e) else "syntax error"
        return Verdict(False, "parse_error", f"The SQL could not be parsed: {first}")
    except Exception as e:  # sqlglot can raise other errors on odd input
        return Verdict(False, "parse_error", f"The SQL could not be parsed: {e}")
    if len(statements) != 1:
        return Verdict(False, "multiple_statements", "Write exactly one SQL statement.")
    stmt = statements[0]
    if isinstance(stmt, BLOCKED_STATEMENTS) or not isinstance(stmt, exp.Query):
        return Verdict(False, "not_select", "Only read-only SELECT queries are allowed.")
    for node in stmt.find_all(*BLOCKED_STATEMENTS):
        return Verdict(False, "blocked_statement", f"{type(node).__name__.upper()} is not allowed; only SELECT queries can run.")

    ctes = cte_map(stmt)
    for table in stmt.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            return Verdict(False, "table_function", f"Table functions such as {table.this.sql()} are not allowed; query the tables directly.")
        if table.args.get("db") or table.args.get("catalog"):
            return Verdict(False, "qualified_table", f"Only the clinic tables can be queried, not {table.sql()}.")
        name = table.name.lower()
        if name not in ALLOWED_TABLES and name not in ctes:
            return Verdict(False, "unknown_table", f"Unknown table '{table.name}'. Available tables: {', '.join(sorted(ALLOWED_TABLES))}.")
    for func in stmt.find_all(exp.Func):
        name = function_name(func)
        if name in BLOCKED_FUNCTIONS or name.startswith(BLOCKED_FUNCTION_PREFIXES):
            return Verdict(False, "blocked_function", f"The function {name}() is not allowed.")

    return _check_privacy(stmt, ctes)


# ---------------------------------------------------------------------------- tree helpers


def function_name(func: exp.Func) -> str:
    return (func.name if isinstance(func, exp.Anonymous) else func.sql_name()).lower()


def cte_map(stmt: exp.Expression) -> dict[str, exp.Expression]:
    return {cte.alias_or_name.lower(): cte.this for cte in stmt.find_all(exp.CTE)}


def sources(select: exp.Select) -> list[exp.Expression]:
    """The tables and subqueries a SELECT reads from (FROM plus JOINs)."""
    out = []
    from_ = select.args.get("from") or select.args.get("from_")
    if from_ is not None:
        out.append(from_.this)
    for join in select.args.get("joins") or []:
        out.append(join.this)
    return out


def _path(node: exp.Expression, top: exp.Expression) -> Iterator[exp.Expression]:
    """Ancestors of `node`, from its parent up to and including `top` (none if node is top)."""
    if node is top:
        return
    p = node.parent
    while p is not None:
        yield p
        if p is top:
            return
        p = p.parent


def is_grouping_aggregate(node: exp.Expression, top: exp.Expression) -> bool:
    """An aggregate that collapses rows into groups (not a window function, not inside a subquery)."""
    return isinstance(node, exp.AggFunc) and not any(
        isinstance(a, (exp.Window, exp.Subquery, exp.Select)) for a in _path(node, top)
    )


def has_grouping_aggregate(expr: exp.Expression) -> bool:
    return any(is_grouping_aggregate(n, expr) for n in expr.find_all(exp.AggFunc))


# ---------------------------------------------------------------------------- privacy


def _identifier_names(stmt: exp.Expression) -> set[str]:
    """Identifier columns plus any alias that carries one (e.g. `patient_id AS pid`)."""
    names = set(IDENTIFIER_COLUMNS)
    while True:
        added = False
        for alias in stmt.find_all(exp.Alias):
            name = alias.alias.lower()
            if name and name not in names and _identifier_outside_count(alias.this, names):
                names.add(name)
                added = True
        for sub in [*stmt.find_all(exp.Subquery), *stmt.find_all(exp.CTE)]:
            table_alias = sub.args.get("alias")
            query = sub.this
            if not isinstance(table_alias, exp.TableAlias) or not table_alias.columns or not isinstance(query, exp.Select):
                continue
            for col_alias, projection in zip(table_alias.columns, query.expressions):
                name = col_alias.name.lower()
                if name not in names and _identifier_outside_count(projection, names):
                    names.add(name)
                    added = True
        if not added:
            return names


def _identifier_outside_count(expr: exp.Expression, names: set[str]) -> str | None:
    for col in expr.find_all(exp.Column):
        if col.name.lower() not in names:
            continue
        path = list(_path(col, expr))
        if any(isinstance(a, exp.Subquery) for a in path):
            continue  # nested subqueries are checked on their own
        if any(isinstance(a, exp.Star) for a in path):
            continue  # SELECT * EXCLUDE (patient_id): the column is removed, not shown
        if any(isinstance(a, exp.Count) for a in path):
            continue  # COUNT(patient_id) / COUNT(DISTINCT patient_id) is allowed
        return col.name
    return None


def _row_level_aggregate(expr: exp.Expression) -> str | None:
    for node in expr.find_all(exp.Func):
        if isinstance(node, ROW_LEVEL_AGGREGATES) or (
            isinstance(node, exp.Anonymous) and node.name.lower() in ROW_LEVEL_AGGREGATE_NAMES
        ):
            if not any(isinstance(a, exp.Subquery) for a in _path(node, expr)):
                return function_name(node)
    return None


def _check_privacy(stmt: exp.Query, ctes: dict[str, exp.Expression]) -> Verdict:
    names = _identifier_names(stmt)
    return _query_safe(stmt, ctes, names, depth=0)


def _query_safe(q: exp.Expression, ctes: dict, names: set[str], depth: int) -> Verdict:
    if depth > 25:
        return Verdict(False, "privacy_row_level", "The query is nested too deeply to check.")
    if isinstance(q, exp.Subquery):
        return _query_safe(q.this, ctes, names, depth + 1)
    if isinstance(q, exp.SetOperation):
        for side in (q.this, q.expression):
            verdict = _query_safe(side, ctes, names, depth + 1)
            if not verdict.ok:
                return verdict
        return OK
    if not isinstance(q, exp.Select):
        return OK

    for projection in q.expressions:
        ident = _identifier_outside_count(projection, names)
        if ident:
            return Verdict(
                False,
                "privacy_identifier",
                f"The query returns the identifier column '{ident}'. Answers must be aggregated: "
                "identifiers may only be used inside COUNT(DISTINCT ...), never shown.",
            )
        packed = _row_level_aggregate(projection)
        if packed:
            return Verdict(
                False,
                "privacy_row_level",
                f"{packed}() packs individual values into one cell; return counts, rates or averages instead.",
            )
        for sub in projection.find_all(exp.Subquery):
            verdict = _query_safe(sub.this, ctes, names, depth + 1)
            if not verdict.ok:
                return verdict

    group = q.args.get("group")
    if group is not None:
        for col in group.find_all(exp.Column):
            if col.name.lower() in names:
                return Verdict(
                    False,
                    "privacy_identifier",
                    f"Grouping by '{col.name}' produces one row per person or event. "
                    "Group by categories such as clinic, department, month or age band instead.",
                )

    if any(has_grouping_aggregate(p) for p in q.expressions):
        return OK

    # No aggregation here, so every row comes straight from the sources: they must be safe.
    for src in sources(q):
        if isinstance(src, exp.Table):
            name = src.name.lower()
            if name in ctes:
                verdict = _query_safe(ctes[name], ctes, names, depth + 1)
                if not verdict.ok:
                    return verdict
            elif name in PATIENT_TABLES:
                return Verdict(
                    False,
                    "privacy_row_level",
                    f"The query returns individual records from '{name}'. Aggregate the result, "
                    "for example with COUNT(*), AVG(...) and GROUP BY.",
                )
        elif isinstance(src, exp.Subquery):
            verdict = _query_safe(src.this, ctes, names, depth + 1)
            if not verdict.ok:
                return verdict
    return OK
