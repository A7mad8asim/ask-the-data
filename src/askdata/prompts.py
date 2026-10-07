"""Prompts for writing SQL and for wording the answer.

The SQL system prompt is identical for every question (rules + schema + glossary),
so it can be cached by the provider. Everything that varies per question (retrieved
examples and the question itself) goes in the user message.
"""

from __future__ import annotations

from .resources import load_glossary
from .schema import schema_prompt

SQL_RULES = """You are Ask-the-Data. You write DuckDB SQL that answers questions about a clinic network's data.

Rules:
1. Write exactly one read-only query: SELECT, optionally starting with WITH. Use the DuckDB SQL dialect.
2. Use only the tables and columns in the schema below.
3. Answers must be aggregated. Never output patient_id or any other identifier column (*_id of patients, appointments, visits, diagnoses, lab results, prescriptions, screenings or follow-ups). Identifiers may appear only inside COUNT(DISTINCT ...), in joins and in filters.
4. If the question asks for individual patients or records, answer with an aggregate instead, for example how many patients match. If it asks to change data, read files, or see system settings, reply exactly: NO_SQL: <short reason>
5. Show percentages on a 0-100 scale. Round percentages and averages to 1 decimal place.
6. Give every output column a short English snake_case alias. Show clinics by clinic_name, never by clinic_id.
7. Reply with the SQL in a single ```sql code block and nothing else."""


def sql_system_prompt(include_glossary: bool = True) -> str:
    parts = [SQL_RULES, "## Schema\n\n" + schema_prompt()]
    if include_glossary:
        parts.append(load_glossary())
    return "\n\n".join(parts)


def sql_user_prompt(question: str, examples: list[dict]) -> str:
    lines = []
    if examples:
        lines.append("Examples of questions with correct SQL:\n")
        for ex in examples:
            lines.append(f"Q: {ex['question']}")
            if ex.get("note"):
                lines.append(f"Note: {ex['note']}")
            lines.append(f"SQL: {ex['sql']}\n")
    lines.append("Question: " + " ".join(question.split()))
    return "\n".join(lines)


def repair_prompt(error: str) -> str:
    return (
        f"That query cannot be used: {error}\n"
        "Write a corrected query for the same question. Reply with the SQL in a single ```sql code block."
    )


ANSWER_SYSTEM = """You write short answers for clinic managers, based only on a query result.

Rules:
- Reply in {language}.
- Use only numbers that appear in the result table. Do not calculate new numbers such as totals, differences or averages. You may round a number to 1 decimal place.
- A cell shown as "<10" describes fewer than 10 patients and is hidden for privacy. Say "fewer than 10" (in Arabic: أقل من 10). Never guess the hidden value.
- Write 1 to 3 plain sentences: no preamble, no tables, no SQL, no markdown.
- This is operational data for managers, not medical advice."""


def answer_system_prompt(language: str) -> str:
    return ANSWER_SYSTEM.format(language="Arabic (Modern Standard Arabic)" if language == "ar" else "English")


def answer_user_prompt(question: str, sql: str, table_markdown: str, n_rows: int, truncated: bool, suppressed: int) -> str:
    notes = []
    if truncated:
        notes.append(f"Only the first {n_rows} rows are shown.")
    if suppressed:
        notes.append(f"{suppressed} row(s) contain values hidden as <10 for privacy.")
    note = ("\nNotes: " + " ".join(notes)) if notes else ""
    return f"Question: {question}\n\nSQL used:\n{sql}\n\nResult ({n_rows} rows):\n{table_markdown}{note}"
