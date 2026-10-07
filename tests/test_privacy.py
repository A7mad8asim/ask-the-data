import pandas as pd

from askdata.privacy import N_COL, display_frame, leaked_identifiers, run_protected, with_patient_count


def test_patient_count_is_added_to_simple_aggregates():
    sql, keys = with_patient_count("SELECT age_band, COUNT(*) AS n FROM patients GROUP BY age_band")
    assert N_COL in sql and "COUNT(DISTINCT patients.patient_id)" in sql
    assert keys == [0]


def test_patient_count_uses_the_table_alias():
    sql, keys = with_patient_count(
        "SELECT c.clinic_name, AVG(v.wait_minutes) FROM visits v JOIN clinics c ON c.clinic_id = v.clinic_id GROUP BY 1"
    )
    assert "COUNT(DISTINCT v.patient_id)" in sql and keys == [0]


def test_patient_count_not_used_where_it_cannot_line_up():
    assert with_patient_count("WITH t AS (SELECT 1 AS x) SELECT x FROM t") is None
    assert with_patient_count("SELECT clinic_name FROM clinics") is None
    assert with_patient_count("SELECT COUNT(*) FROM clinics") is None  # not a patient-level table


def test_small_groups_are_masked_but_keys_stay(db):
    sql = (
        "SELECT nationality_group, age_band, municipality, COUNT(*) AS patients, "
        "AVG(year(registration_date)) AS avg_year FROM patients GROUP BY ALL"
    )
    result = run_protected(db, sql, min_group_size=10)
    assert result.method == "patient_count"
    assert result.suppressed_rows > 0
    small = result.raw["patients"] < 10
    assert result.masked.loc[small, "patients"].all() and result.masked.loc[small, "avg_year"].all()
    assert not result.masked["nationality_group"].any()  # group keys are never masked
    assert result.df.loc[small, "patients"].isna().all()
    assert (result.raw.loc[small, "patients"] >= 1).all()  # the evaluation still sees the real values
    assert not result.masked.loc[~small].any().any()


def test_heuristic_masks_small_counts_when_count_cannot_be_added(db):
    sql = (
        "WITH t AS (SELECT age_band, COUNT(*) AS n_patients FROM patients "
        "WHERE nationality_group = 'Western' AND municipality = 'Al Khor' GROUP BY age_band) SELECT * FROM t"
    )
    result = run_protected(db, sql, min_group_size=10)
    assert result.method == "heuristic"
    small = (result.raw["n_patients"] >= 1) & (result.raw["n_patients"] < 10)
    assert result.masked.loc[small, "n_patients"].all()


def test_zero_is_not_masked(db):
    result = run_protected(db, "SELECT COUNT(*) AS n FROM patients WHERE age_band = 'none'", 10)
    assert result.df.iat[0, 0] == 0 and result.suppressed_rows == 0


def test_display_and_leak_scan():
    df = pd.DataFrame({"group": ["a", "b"], "n": [5, 50]})
    masked = pd.DataFrame({"group": [False, False], "n": [True, False]})
    shown = display_frame(df, masked)
    assert shown["n"].tolist() == ["<10", 50]
    assert leaked_identifiers(pd.DataFrame({"x": ["P123", "ok"]}), {"P123"}) == ["x=P123"]
    assert leaked_identifiers(shown, {"P123"}) == []
