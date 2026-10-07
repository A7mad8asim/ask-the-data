import pytest

from askdata.sandbox import check_sql

ALLOWED = [
    "SELECT COUNT(*) FROM patients",
    "SELECT age_band, COUNT(*) AS n FROM patients GROUP BY age_band",
    "SELECT department, COUNT(DISTINCT patient_id) FROM visits GROUP BY ALL",
    "WITH per_patient AS (SELECT patient_id, COUNT(*) AS n FROM visits GROUP BY patient_id) SELECT AVG(n) FROM per_patient",
    "SELECT COUNT(*) FROM visits WHERE patient_id IN (SELECT patient_id FROM diagnoses WHERE icd10_code = 'I10')",
    "SELECT clinic_name, opens_at FROM clinics",
    "SELECT * FROM (SELECT clinic_id, COUNT(*) AS n FROM visits GROUP BY clinic_id) t",
    "FROM patients SELECT COUNT(*)",
    "SELECT month, COUNT(*) FROM calendar GROUP BY month",
    "SELECT 100.0 * COUNT(DISTINCT v.patient_id) / (SELECT COUNT(*) FROM patients) FROM visits v",
    "SELECT COUNT(*) FILTER (WHERE sex = 'Female') AS women FROM patients",
    "SELECT department, MEDIAN(wait_minutes) FROM visits GROUP BY department",
    "SELECT 100.0 * COUNT(t.patient_id) / COUNT(*) FROM (SELECT DISTINCT patient_id FROM lab_results) d "
    "LEFT JOIN (SELECT DISTINCT patient_id FROM prescriptions) t ON t.patient_id = d.patient_id",
    "SELECT 1",
    "select count(*) from patients;",
]

BLOCKED = [
    ("DROP TABLE patients", "not_select"),
    ("DELETE FROM appointments WHERE status = 'cancelled'", "not_select"),
    ("INSERT INTO clinics VALUES ('C09')", "not_select"),
    ("UPDATE lab_results SET value = 6.5", "not_select"),
    ("CREATE TABLE x AS SELECT 1", "not_select"),
    ("ATTACH 'x.db' AS x", "not_select"),
    ("COPY patients TO 'patients.csv'", "not_select"),
    ("INSTALL httpfs", "not_select"),
    ("LOAD httpfs", "not_select"),
    ("PRAGMA show_tables", "not_select"),
    ("SET threads = 1", "not_select"),
    ("DESCRIBE patients", "not_select"),
    ("SELECT 1; DROP TABLE patients", "multiple_statements"),
    ("SELECT * FROM read_csv('C:/Windows/win.ini')", "table_function"),
    ("SELECT * FROM read_csv_auto('x.csv')", "table_function"),
    ("SELECT * FROM glob('*')", "table_function"),
    ("SELECT * FROM duckdb_settings()", "table_function"),
    ("SELECT COUNT(*) FROM range(10)", "table_function"),
    ("SELECT * FROM 'patients.parquet'", "unknown_table"),
    ("SELECT * FROM secret_table", "unknown_table"),
    ("SELECT * FROM information_schema.tables", "qualified_table"),
    ("SELECT COUNT(*) FROM main.patients", "qualified_table"),
    ("SELECT getenv('HOME')", "blocked_function"),
    ("SELECT current_setting('threads')", "blocked_function"),
    ("SELECT patient_id FROM patients", "privacy_identifier"),
    ("SELECT * FROM patients", "privacy_row_level"),
    ("SELECT * EXCLUDE (patient_id) FROM patients", "privacy_row_level"),
    ("SELECT age_band, sex FROM patients LIMIT 10", "privacy_row_level"),
    ("SELECT DISTINCT department FROM visits", "privacy_row_level"),
    ("SELECT patient_id, COUNT(*) FROM visits GROUP BY patient_id", "privacy_identifier"),
    ("SELECT COUNT(*) FROM visits GROUP BY patient_id", "privacy_identifier"),
    ("SELECT MAX(patient_id) FROM patients", "privacy_identifier"),
    ("SELECT arg_max(patient_id, registration_date) FROM patients", "privacy_identifier"),
    ("SELECT list(age_band) FROM patients", "privacy_row_level"),
    ("SELECT string_agg(department, ',') FROM visits", "privacy_row_level"),
    ("SELECT string_agg(visit_id, ',') FROM visits", "privacy_identifier"),
    ("SELECT age_band, COUNT(*) OVER (PARTITION BY age_band) FROM patients", "privacy_row_level"),
    ("SELECT (SELECT patient_id FROM patients LIMIT 1) AS x", "privacy_identifier"),
    ("SELECT (SELECT age_band FROM patients LIMIT 1) AS x", "privacy_row_level"),
    ("WITH t AS (SELECT patient_id AS pid FROM patients) SELECT pid, COUNT(*) FROM t GROUP BY pid", "privacy_identifier"),
    ("WITH t(pid) AS (SELECT patient_id FROM patients) SELECT pid FROM t", "privacy_identifier"),
    ("SELECT clinic_id FROM clinics UNION ALL SELECT patient_id FROM patients", "privacy_identifier"),
    ("SELECT v.* FROM visits v", "privacy_row_level"),
    ("SELECT x.* FROM (SELECT * FROM lab_results) x", "privacy_row_level"),
    ("SELECT 'P' || substr(patient_id, 2) AS p FROM patients", "privacy_identifier"),
    ("SELECT age_band FROM patients USING SAMPLE 10", "privacy_row_level"),
    ("", "empty"),
]


@pytest.mark.parametrize("sql", ALLOWED)
def test_allowed(sql):
    verdict = check_sql(sql)
    assert verdict.ok, verdict.message


@pytest.mark.parametrize("sql,code", BLOCKED)
def test_blocked(sql, code):
    verdict = check_sql(sql)
    assert not verdict.ok
    assert verdict.code == code, verdict.message
    assert verdict.message  # every rejection explains itself, so the model can repair the query


def test_garbage_is_rejected():
    assert not check_sql("this is not sql at all").ok
