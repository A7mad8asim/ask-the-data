"""Helpers around the model's text: pull out the SQL, check the answer's numbers,
and build a plain summary when the model's wording cannot be trusted."""

from __future__ import annotations

import math
import re
from decimal import Decimal

import numpy as np
import pandas as pd

_ARABIC_LETTER = re.compile(r"[؀-ۿݐ-ݿ]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")
_FENCE = re.compile(r"```[ \t]*(?:sql|duckdb)?[ \t]*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_SQL_START = re.compile(r"^\s*(WITH|SELECT|FROM)\b", re.IGNORECASE | re.MULTILINE)
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:[.,]\d+)*")


def detect_language(text: str) -> str:
    arabic = len(_ARABIC_LETTER.findall(text or ""))
    latin = len(_LATIN_LETTER.findall(text or ""))
    return "ar" if arabic > 0 and arabic >= 0.3 * (arabic + latin) else "en"


def is_no_sql(text: str) -> str | None:
    """If the model declined with NO_SQL, return its reason."""
    t = (text or "").strip()
    if "NO_SQL" in t.upper() and not _FENCE.search(t):
        reason = t.split(":", 1)[1].strip() if ":" in t else ""
        return reason or "The request cannot be answered with an aggregate query."
    return None


def extract_sql(text: str) -> str | None:
    t = text or ""
    for block in _FENCE.findall(t):
        if block.strip():
            return block.strip().rstrip(";").strip()
    m = _SQL_START.search(t)
    if m:
        return t[m.start():].strip().rstrip(";").strip()
    return None


# ---------------------------------------------------------------------------- numbers


def _to_float(token: str) -> float | None:
    t = token.translate(_DIGITS).replace("٫", ".").replace("٬", ",")
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", t):
        t = t.replace(",", "")
    t = t.replace(",", ".") if t.count(",") == 1 and "." not in t else t
    try:
        return float(t)
    except ValueError:
        return None


def extract_numbers(text: str) -> list[tuple[float, int]]:
    """Numbers in a text as (value, number of decimals). Handles Arabic-Indic digits."""
    t = (text or "").translate(_DIGITS).replace("٫", ".").replace("٬", ",")
    out = []
    for token in _NUMBER.findall(t):
        value = _to_float(token)
        if value is None:
            continue
        decimals = len(token.split(".")[1]) if "." in token else 0
        out.append((value, decimals))
    return out


def _numeric_cells(df: pd.DataFrame) -> list[float]:
    values = []
    for col in df.columns:
        for v in df[col].tolist():
            if isinstance(v, (bool, np.bool_)) or v is None or v is pd.NA:
                continue
            if isinstance(v, (int, float, np.integer, np.floating, Decimal)):
                f = float(v)
                if not math.isnan(f):
                    values.append(f)
    return values


def check_grounding(answer: str, table: pd.DataFrame, question: str = "", sql: str = "", extra: tuple = ()) -> tuple[bool, list[float]]:
    """Every number in the answer must come from the result table (allowing rounding and
    percent scaling), the question, the SQL, or `extra` (e.g. the privacy threshold 10)."""
    candidates = _numeric_cells(table) if table is not None else []
    candidates += [v * 100 for v in candidates] + [v / 100 for v in candidates]
    context = {v for v, _ in extract_numbers(question)} | {v for v, _ in extract_numbers(sql)}
    context |= {float(x) for x in extra}
    if table is not None:
        context.add(float(len(table)))
    unmatched = []
    for value, decimals in extract_numbers(answer):
        if value in context or -value in context:
            continue
        tolerance = 0.5 * 10 ** (-decimals) + 1e-9
        if any(abs(abs(c) - abs(value)) <= tolerance for c in candidates):
            continue
        unmatched.append(value)
    return (not unmatched), unmatched


# ---------------------------------------------------------------------------- presentation


def format_value(v) -> str:
    if v is None or v is pd.NA or (isinstance(v, float) and math.isnan(v)):
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "true" if v else "false"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating, Decimal)):
        f = float(v)
        return f"{f:,.0f}" if f.is_integer() and abs(f) >= 1000 else f"{f:,.2f}".rstrip("0").rstrip(".")
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d") if v == v.normalize() else str(v)
    return str(v)


def markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df is None or df.empty:
        return "(no rows)"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.head(max_rows).itertuples(index=False):
        lines.append("| " + " | ".join(format_value(v) for v in row) + " |")
    if len(df) > max_rows:
        lines.append(f"| ... {len(df) - max_rows} more rows |")
    return "\n".join(lines)


def plain_summary(display: pd.DataFrame, language: str, suppressed: int = 0, truncated: bool = False) -> str:
    """A number-safe summary built from the table itself (used when the model's wording fails the check)."""
    ar = language == "ar"
    if display is None or display.empty:
        return "لا توجد بيانات مطابقة لهذا السؤال." if ar else "No matching data was found for this question."
    if display.shape == (1, 1):
        value = format_value(display.iat[0, 0])
        text = f"النتيجة: {value}" if ar else f"Result: {value} ({display.columns[0]})."
    elif len(display) == 1:
        parts = [f"{c}: {format_value(v)}" for c, v in zip(display.columns, display.iloc[0])]
        text = ("النتيجة: " if ar else "Result: ") + "; ".join(parts)
    else:
        n = len(display)
        text = f"النتيجة تحتوي على {n} صفوف، وهي معروضة في الجدول." if ar else f"The result has {n} rows, shown in the table."
    if suppressed:
        text += (" القيم المعروضة كـ <10 مخفية لحماية الخصوصية." if ar
                 else " Values shown as <10 are hidden to protect privacy.")
    if truncated:
        text += " (عرض أول الصفوف فقط)" if ar else " (Only the first rows are shown.)"
    return text


_TEMPORAL = re.compile(r"(month|year|quarter|date|week|day|hour)", re.IGNORECASE)


def _is_numeric_column(s: pd.Series) -> bool:
    values = [v for v in s.tolist() if v is not None and v is not pd.NA and not (isinstance(v, float) and math.isnan(v))]
    return bool(values) and all(
        isinstance(v, (int, float, np.integer, np.floating, Decimal)) and not isinstance(v, (bool, np.bool_)) for v in values
    )


def suggest_chart(df: pd.DataFrame) -> dict | None:
    """Pick a simple chart: one category or time column on x, up to three numeric columns on y."""
    if df is None or len(df) < 2 or df.shape[1] < 2:
        return None
    numeric = [c for c in df.columns if _is_numeric_column(df[c])]
    keys = [c for c in df.columns if c not in numeric]
    if not keys and len(numeric) >= 2 and _TEMPORAL.search(str(numeric[0])):
        keys = [numeric.pop(0)]
    if len(keys) != 1 or not numeric:
        return None
    x = keys[0]
    temporal = bool(_TEMPORAL.search(str(x))) or pd.api.types.is_datetime64_any_dtype(df[x])
    return {"type": "line" if temporal else "bar", "x": x, "y": numeric[:3]}
