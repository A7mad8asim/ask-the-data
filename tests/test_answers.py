import pandas as pd

from askdata.answers import (
    check_grounding,
    detect_language,
    extract_numbers,
    extract_sql,
    is_no_sql,
    plain_summary,
    suggest_chart,
)


def test_extract_sql_variants():
    assert extract_sql("```sql\nSELECT 1;\n```") == "SELECT 1"
    assert extract_sql("Here you go:\n```\nSELECT 2\n```") == "SELECT 2"
    assert extract_sql("SELECT 3 FROM patients;") == "SELECT 3 FROM patients"
    assert extract_sql("I think\nWITH t AS (SELECT 1) SELECT * FROM t") == "WITH t AS (SELECT 1) SELECT * FROM t"
    assert extract_sql("no query here") is None


def test_no_sql_reply():
    assert is_no_sql("NO_SQL: deleting data is not allowed") == "deleting data is not allowed"
    assert is_no_sql("```sql\nSELECT 1\n```") is None


def test_language_detection():
    assert detect_language("كم عدد المرضى؟") == "ar"
    assert detect_language("كم عدد مرضى HbA1c في 2025؟") == "ar"
    assert detect_language("How many patients?") == "en"


def test_numbers_in_arabic_and_english():
    assert extract_numbers("12.5% and 1,500 visits") == [(12.5, 1), (1500.0, 0)]
    assert extract_numbers("النسبة ١٢٫٣٪") == [(12.3, 1)]
    assert extract_numbers("clinic C07 and code E11.9") == []


def test_grounding_accepts_numbers_from_the_table():
    table = pd.DataFrame({"clinic": ["A", "B"], "rate": [12.3456, 0.5], "visits": [1500, 20]})
    assert check_grounding("Clinic A had 12.3% and 1,500 visits.", table)[0]
    assert check_grounding("العيادة A: ١٢٫٣٥٪", table)[0]
    assert check_grounding("Clinic B is at 0.5.", table)[0]


def test_grounding_rejects_invented_numbers():
    table = pd.DataFrame({"rate": [12.3456]})
    ok, unmatched = check_grounding("The rate rose by 4.2 points to 12.3%.", table)
    assert not ok and unmatched == [4.2]


def test_grounding_allows_question_numbers_and_threshold():
    table = pd.DataFrame({"n": [pd.NA, 40]}, dtype="object")
    assert check_grounding("In 2025, fewer than 10 patients were in one group and 40 in the other.", table,
                           question="... in 2025?", extra=(10,))[0]


def test_fraction_shown_as_percentage():
    table = pd.DataFrame({"share": [0.1234]})
    assert check_grounding("About 12.3% of tests.", table)[0]


def test_chart_suggestions():
    months = pd.DataFrame({"month": [1, 2, 3], "visits": [10, 12, 9]})
    assert suggest_chart(months) == {"type": "line", "x": "month", "y": ["visits"]}
    clinics = pd.DataFrame({"clinic_name": ["A", "B"], "rate": [1.0, 2.0]})
    assert suggest_chart(clinics)["type"] == "bar"
    assert suggest_chart(pd.DataFrame({"n": [5]})) is None


def test_plain_summary():
    assert plain_summary(pd.DataFrame({"n": [5]}), "en") == "Result: 5 (n)."
    assert "صفوف" in plain_summary(pd.DataFrame({"a": [1, 2], "b": [3, 4]}), "ar")
    assert "No matching" in plain_summary(pd.DataFrame({"n": []}), "en")
