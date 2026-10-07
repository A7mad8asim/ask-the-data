"""Small-group suppression: hide numbers that describe fewer than N patients.

The sandbox already guarantees that answers are aggregated. This module adds the
second rule used in published health statistics: an aggregate over a very small group
(say 3 patients) can still point at real people, so any group with fewer than
`min_group_size` patients has its numbers masked and shown as "<10".

How the patient count per row is found:
1. If the query aggregates directly over a patient-level table, a hidden
   COUNT(DISTINCT patient_id) column is added to it, and each output row is masked
   when that count is below the threshold. The modified query must return exactly the
   same rows as the original, otherwise this method is not used.
2. Otherwise (CTEs, set operations, ...), count-like columns with values from 1 to 9
   are masked as a conservative fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
import sqlglot
from sqlglot import exp

from .db import Database, QueryError
from .sandbox import cte_map, has_grouping_aggregate, sources
from .schema import PATIENT_TABLES

N_COL = "_n_patients"
COUNT_LIKE = re.compile(
    r"(^n$|^n_|_n$|count|^num|_num|number|patients|people|persons|visits|appointments|cases|screenings|"
    r"prescriptions|follow_ups|followups|diagnoses|tests|total|cnt)",
    re.IGNORECASE,
)


@dataclass
class Protected:
    df: pd.DataFrame  # what users and the answer-writer see (masked cells are NA)
    raw: pd.DataFrame  # unmasked result, used only by the evaluation harness
    masked: pd.DataFrame  # True where a cell was masked
    suppressed_rows: int
    method: str  # "patient_count", "heuristic" or "none"
    truncated: bool


def with_patient_count(sql: str) -> tuple[str, list[int]] | None:
    """Return (sql + hidden patient-count column, positions of the group-key columns), or None."""
    try:
        stmt = sqlglot.parse_one(sql, read="duckdb")
    except Exception:
        return None
    if not isinstance(stmt, exp.Select) or stmt.args.get("distinct"):
        return None
    if not any(has_grouping_aggregate(p) for p in stmt.expressions):
        return None
    if any(isinstance(p, exp.Star) or isinstance(p.this, exp.Star) for p in stmt.expressions):
        return None  # column positions would not line up with the projections
    ctes = cte_map(stmt)
    for src in sources(stmt):
        if (
            isinstance(src, exp.Table)
            and isinstance(src.this, exp.Identifier)
            and src.name.lower() in PATIENT_TABLES
            and src.name.lower() not in ctes
        ):
            keys = [i for i, p in enumerate(stmt.expressions) if not has_grouping_aggregate(p)]
            count = exp.Count(this=exp.Distinct(expressions=[exp.column("patient_id", table=src.alias_or_name)]))
            stmt.select(exp.alias_(count, N_COL), copy=False)
            return stmt.sql(dialect="duckdb"), keys
    return None


def _same_rows(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    key = lambda df: sorted(map(repr, df.astype(str).itertuples(index=False, name=None)))
    return key(a) == key(b)


def run_protected(db: Database, sql: str, min_group_size: int = 10) -> Protected:
    """Run a sandbox-approved query and apply small-group suppression."""
    result = db.run(sql)
    raw = result.df
    masked = pd.DataFrame(False, index=raw.index, columns=raw.columns)
    method = "none"

    injected = with_patient_count(sql) if min_group_size > 1 else None
    if injected is not None:
        sql2, keys = injected
        try:
            counted = db.run(sql2)
        except QueryError:
            counted = None
        if counted is not None and N_COL in counted.df.columns:
            candidate = counted.df.drop(columns=[N_COL])
            if not result.truncated and not counted.truncated and _same_rows(raw, candidate):
                raw = candidate.reset_index(drop=True)
                n = pd.to_numeric(counted.df[N_COL], errors="coerce").fillna(0).to_numpy()
                small = (n > 0) & (n < min_group_size)
                masked = pd.DataFrame(False, index=raw.index, columns=raw.columns)
                for i, col in enumerate(raw.columns):
                    if i not in keys:
                        masked.loc[small, col] = True
                method = "patient_count"

    if method == "none" and min_group_size > 1:
        for col in raw.columns:
            values = pd.to_numeric(raw[col], errors="coerce")
            if not COUNT_LIKE.search(str(col)) or values.isna().all():
                continue
            whole = values.notna() & (values == values.round())
            small = whole & (values >= 1) & (values < min_group_size)
            if small.any():
                masked.loc[small, col] = True
        method = "heuristic"

    df = raw.copy()
    for col in df.columns:
        if masked[col].any():
            df[col] = df[col].astype("object")
            df.loc[masked[col], col] = pd.NA
    return Protected(
        df=df,
        raw=raw,
        masked=masked,
        suppressed_rows=int(masked.any(axis=1).sum()),
        method=method,
        truncated=result.truncated,
    )


def display_frame(df: pd.DataFrame, masked: pd.DataFrame, label: str = "<10") -> pd.DataFrame:
    """Copy of the result for display, with masked cells shown as '<10'."""
    out = df.copy()
    for col in out.columns:
        if masked[col].any():
            out[col] = out[col].astype("object")
            out.loc[masked[col], col] = label
    return out


def leaked_identifiers(df: pd.DataFrame, known_ids: set[str]) -> list[str]:
    """Independent check used by the evaluation: does any output cell contain a real identifier?"""
    found = []
    for col in df.columns:
        for value in df[col].astype(str):
            if value in known_ids:
                found.append(f"{col}={value}")
                break
    return found
