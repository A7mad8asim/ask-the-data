"""Checks on the gold set, the adversarial set, the few-shot examples and the generator."""

from collections import Counter

import pandas as pd
import pytest

from askdata.generator import generate
from askdata.resources import load_adversarial, load_examples, load_gold
from askdata.sandbox import check_sql
from askdata.schema import TABLES

GOLD = load_gold()
EXAMPLES = load_examples()


def squash(sql: str) -> str:
    return " ".join(sql.split()).casefold()


def test_gold_set_shape():
    assert len(GOLD) == 100
    assert len({g["id"] for g in GOLD}) == 100
    langs = Counter(g["lang"] for g in GOLD)
    assert langs == {"en": 50, "ar": 50}
    pairs = Counter(g["pair"] for g in GOLD)
    assert set(pairs.values()) == {2}  # every question exists in English and Arabic
    assert sum(g["dialect"] for g in GOLD) >= 15
    assert Counter(g["difficulty"] for g in GOLD) == {"easy": 30, "medium": 40, "hard": 30}


@pytest.mark.parametrize("item", GOLD, ids=lambda g: g["id"])
def test_gold_sql_is_allowed_and_runs(item, db):
    verdict = check_sql(item["sql"])
    assert verdict.ok, verdict.message
    assert len(db.run(item["sql"]).df) >= 1


def test_adversarial_set_shape():
    adv = load_adversarial()
    assert len(adv) == 20 and len({a["id"] for a in adv}) == 20


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda e: e["id"])
def test_examples_are_valid(example, db):
    verdict = check_sql(example["sql"])
    assert verdict.ok, verdict.message
    db.run(example["sql"])


def test_examples_do_not_leak_the_gold_set():
    gold_questions = {squash(g["question"]) for g in GOLD}
    gold_sql = {squash(g["sql"]) for g in GOLD}
    for e in EXAMPLES:
        assert squash(e["question"]) not in gold_questions, e["id"]
        assert squash(e["sql"]) not in gold_sql, e["id"]


def test_tables_match_the_schema(db):
    for name, table in TABLES.items():
        empty = db.run(f"SELECT * FROM {name} LIMIT 0").df
        assert list(empty.columns) == table.column_names, name


def test_generator_is_deterministic(tmp_path):
    a = generate(800, seed=7, data_dir=tmp_path / "a", verbose=False)
    b = generate(800, seed=7, data_dir=tmp_path / "b", verbose=False)
    assert a == b
    pa = pd.read_parquet(tmp_path / "a" / "parquet" / "lab_results.parquet")
    pb = pd.read_parquet(tmp_path / "b" / "parquet" / "lab_results.parquet")
    pd.testing.assert_frame_equal(pa, pb)


def test_built_in_patterns(db):
    """The documented patterns must actually be in the data."""
    rate = (
        "100.0 * SUM(CASE WHEN status = 'no_show' THEN 1 ELSE 0 END) / "
        "SUM(CASE WHEN status IN ('attended', 'no_show') THEN 1 ELSE 0 END)"
    )
    ramadan = db.scalar(f"SELECT {rate} FROM appointments a JOIN calendar c ON c.date = a.appointment_date WHERE c.is_ramadan")
    other = db.scalar(f"SELECT {rate} FROM appointments a JOIN calendar c ON c.date = a.appointment_date WHERE NOT c.is_ramadan")
    assert ramadan > other + 3

    waits = db.run("SELECT clinic_id, AVG(wait_minutes) AS w FROM visits GROUP BY 1 ORDER BY w DESC").df
    assert waits.iloc[0]["clinic_id"] == "C07"  # the Industrial Area Clinic has the longest waits

    completion = db.run(
        "SELECT p.nationality_group AS g, AVG(CASE WHEN f.completed_date IS NOT NULL THEN 1 ELSE 0 END) AS c "
        "FROM follow_ups f JOIN patients p USING (patient_id) GROUP BY 1"
    ).df.set_index("g")["c"]
    assert completion["South Asian"] < completion["Qatari"]

    summer = db.scalar("SELECT COUNT(*) FROM visits WHERE month(visit_date) IN (7, 8)") / 2
    autumn = db.scalar("SELECT COUNT(*) FROM visits WHERE month(visit_date) IN (10, 11)") / 2
    assert summer < autumn

    null_units = db.scalar("SELECT AVG(CASE WHEN unit IS NULL THEN 1 ELSE 0 END) FROM lab_results")
    assert 0.002 < null_units < 0.03
