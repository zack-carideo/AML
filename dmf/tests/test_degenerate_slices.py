"""Discrimination metrics on a target with a single class.

A holdout slice can clear ``metrics.min_slice_n`` on row count and still hold
zero fraud cases. sklearn's ``average_precision_score`` answers that with a
warning and a literal 0.0, which the slice table then presents as a segment
where the model scored terribly. The right value is null: nothing about
ranking is measurable there.

The row itself must survive. The slice table exists to expose flag-rate
disparity for fair-treatment review, and a segment with no fraud that is still
flagged heavily is exactly the finding it is for.
"""

from __future__ import annotations

import tempfile
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from conftest import make_cfg
from dmf.metrics import average_precision, evaluate_predictions, roc_auc
from dmf.research import ModelSelectionHarness


def _no_positive_class_warnings(records) -> list:
    return [w for w in records if "No positive class" in str(w.message)]


# --------------------------------------------------------------------------
# the metric functions
# --------------------------------------------------------------------------
def test_wrappers_match_sklearn_when_both_classes_are_present():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.2, 500)
    s = rng.random(500)
    assert average_precision(y, s) == pytest.approx(average_precision_score(y, s))
    assert roc_auc(y, s) == pytest.approx(roc_auc_score(y, s))


@pytest.mark.parametrize("y", [np.zeros(50, int), np.ones(50, int)])
def test_wrappers_are_nan_and_silent_on_a_single_class_target(y):
    s = np.linspace(0, 1, 50)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert np.isnan(average_precision(y, s))
        assert np.isnan(roc_auc(y, s))
    assert not _no_positive_class_warnings(rec)


def test_report_records_null_not_zero_for_a_zero_positive_target():
    y = np.zeros(80, int)
    s = np.linspace(0.01, 0.99, 80)
    m = make_cfg().metrics
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = evaluate_predictions(y, s, m)
    assert not _no_positive_class_warnings(rec)
    assert out["average_precision"] is None
    assert out["roc_auc"] is None
    assert out["ks_statistic"] is None
    # calibration-type metrics are still defined on a single-class target
    assert isinstance(out["brier_score"], float)
    assert out["prevalence"] == 0.0


# --------------------------------------------------------------------------
# the slice table
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sliced(raw):
    cfg = make_cfg()
    cfg.run.output_dir = tempfile.mkdtemp()
    # merchant_id has ~220 levels, so a low floor guarantees thin, fraud-free
    # levels that still pass the row-count gate
    cfg.metrics.slice_columns = ["channel", "merchant_id"]
    cfg.metrics.min_slice_n = 5
    X = raw.drop(columns=["is_fraudulent_dispute"]).copy()
    y = raw["is_fraudulent_dispute"].to_numpy()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = ModelSelectionHarness(cfg).run(X, y)
    return res, rec


def test_run_emits_no_positive_class_warning(sliced):
    _, rec = sliced
    assert not _no_positive_class_warnings(rec)


def test_zero_positive_levels_keep_their_row_with_null_discrimination(sliced):
    res, _ = sliced
    s = res.holdout_slices
    empty = s[s["prevalence"] == 0]
    assert len(empty) > 0, "the fixture should produce at least one fraud-free level"

    # the row is still there, and what is measurable on it is reported
    assert empty["n"].ge(5).all()
    assert empty["flag_rate_at_top_pct"].between(0, 1).all()
    assert empty["mean_score"].notna().all()
    # what is not measurable is null, never a number that reads as a score
    assert empty["average_precision"].isna().all()
    assert empty["roc_auc"].isna().all()

    # a level with positives is unaffected
    full = s[s["prevalence"] > 0]
    assert full["average_precision"].notna().any()
