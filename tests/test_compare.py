"""The execution-accuracy comparator decides every score in the README, so it is tested on its own."""

import datetime as dt

import pandas as pd

from askdata.evaluation import results_match


def df(**cols):
    return pd.DataFrame(cols)


def test_identical_and_reordered_rows():
    gold = df(clinic=["A", "B"], n=[3, 5])
    assert results_match(gold, df(clinic=["B", "A"], n=[5, 3]))


def test_column_names_and_order_do_not_matter_and_extras_are_allowed():
    gold = df(clinic=["A", "B"], rate=[12.3456, 7.0])
    pred = df(rate_pct=[7.0, 12.3], label=["x", "y"], name=["B", "A"])
    assert results_match(gold, pred)


def test_counts_are_exact():
    assert results_match(df(n=[120]), df(count=[120.0]))
    assert not results_match(df(n=[120]), df(count=[121]))


def test_decimals_within_rounding():
    assert results_match(df(rate=[9.1700]), df(rate=[9.2]))
    assert not results_match(df(rate=[9.17]), df(rate=[9.0]))


def test_fraction_versus_percentage():
    assert results_match(df(rate=[12.3456]), df(rate=[0.1235]))


def test_month_as_date_or_name():
    gold = df(month=[1, 2], n=[10, 20])
    assert results_match(gold, df(month=[pd.Timestamp("2025-01-01"), pd.Timestamp("2025-02-01")], n=[10, 20]))
    assert results_match(gold, df(month=["January", "February"], n=[10, 20]))
    assert results_match(gold, df(m=[dt.date(2025, 1, 1), dt.date(2025, 2, 1)], n=[10, 20]))


def test_rows_must_pair_up_not_just_columns():
    gold = df(clinic=["A", "B"], n=[1, 2])
    assert not results_match(gold, df(clinic=["A", "B"], n=[2, 1]))


def test_row_count_must_match():
    assert not results_match(df(n=[1]), df(n=[1, 2]))
    assert results_match(df(n=[]), df(x=[]))


def test_clinic_aliases():
    aliases = {"عيادة الخور": "al khor clinic", "c06": "al khor clinic"}
    gold = df(clinic_name=["Al Khor Clinic"], n=[4])
    assert results_match(gold, df(name=["عيادة الخور"], n=[4]), aliases)
    assert results_match(gold, df(clinic_id=["C06"], n=[4]), aliases)


def test_missing_prediction():
    assert not results_match(df(n=[1]), None)
