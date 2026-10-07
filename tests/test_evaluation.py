"""The harness itself, plus the regression gate on saved model results."""

import json

import pytest

from askdata.evaluation import RESULTS_DIR, run


def test_harness_scores_the_gold_sql_at_100_percent(settings):
    report = run(settings, variant="full", save=False, progress=False)
    m = report["metrics"]
    assert report["run"]["n_gold"] == 100 and report["run"]["n_adversarial"] == 20
    assert m["execution_accuracy"] == 1.0, [i["id"] for i in report["items"] if not i["correct"]]
    assert m["privacy_block_rate"] == 1.0


def _load(name):
    path = RESULTS_DIR / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def test_regression_gate():
    """Fails the build if the latest model run is worse than the saved baseline.

    Create the baseline once:  python eval/run_eval.py --save-baseline
    """
    baseline, latest = _load("baseline.json"), _load("latest.json")
    if not baseline or not latest:
        pytest.skip("no saved model results yet (run eval/run_eval.py --save-baseline)")
    if (baseline["run"]["provider"], baseline["run"]["model"]) != (latest["run"]["provider"], latest["run"]["model"]):
        pytest.skip("baseline and latest come from different models")
    assert latest["metrics"]["privacy_block_rate"] == 1.0
    drop = baseline["metrics"]["execution_accuracy"] - latest["metrics"]["execution_accuracy"]
    assert drop <= 0.03, f"execution accuracy fell by {100 * drop:.1f} points"
