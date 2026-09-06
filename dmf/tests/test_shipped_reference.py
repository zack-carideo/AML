"""What ships beside the model must describe the model that ships.

Two properties, each of which a two-variable model with a discrete score scale
violated before:

1. The stored decision threshold is the exact quantile, not a rounded copy.
   ``ProductionScorer`` applies ``score >= threshold``; rounding the value up by
   one unit in the sixth decimal drops every row tied at the quantile, so the
   shipped cut stops flagging the share of volume ``decision_flag_rate`` reports.
2. With ``run.refit_on_full_data`` the production pipeline is a different fit
   from the one the holdout validated. The monitoring reference and the
   capacity (``top_pct``) threshold are therefore re-derived from the shipped
   pipeline's own scores on the holdout rows, and every artifact says which fit
   each number came from. An ``fpr`` threshold cannot honestly be re-derived on
   rows the refit trained on, so it keeps the train-only value and says so.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import make_cfg
from dmf import ProductionScorer
from dmf.config import MetricsConfig
from dmf.metrics import (population_stability_index, psi_from_reference_quantiles,
                         reference_quantiles)
from dmf.research import ModelSelectionHarness, implied_thresholds


# --------------------------------------------------------------------------
# 1. full-precision thresholds
# --------------------------------------------------------------------------
def test_threshold_is_the_exact_quantile_and_ties_flag_as_reported():
    # 1000 scores: 900 below v, 60 tied exactly at v, 40 above. The 95th
    # percentile lands inside the tie block, so the threshold IS v -- and v is
    # chosen so that rounding to six decimals moves it UP past the ties.
    v = 0.8220566
    assert round(v, 6) > v
    s = np.r_[np.linspace(0.05, 0.80, 900), np.full(60, v), np.linspace(0.90, 0.99, 40)]
    y = (np.random.default_rng(0).random(1000) < 0.1).astype(int)

    out = implied_thresholds(y, s, MetricsConfig(lift_top_pct=0.05))
    thr = out["decision_threshold"]
    assert thr == v                                              # not rounded
    assert out["implied_threshold_top_pct"] == v

    flagged_at_stored = float((s >= thr).mean())
    assert flagged_at_stored == pytest.approx(0.10)              # the 60 ties + 40 above
    assert flagged_at_stored == pytest.approx(out["decision_flag_rate"], abs=1e-6)
    # and the bug this guards against: the rounded value would have dropped the ties
    assert float((s >= round(thr, 6)).mean()) == pytest.approx(0.04)


# --------------------------------------------------------------------------
# 2. the shipped pipeline's reference and threshold
# --------------------------------------------------------------------------
def _run(raw, out_dir, **over):
    cfg = make_cfg()
    cfg.run.output_dir = str(out_dir)
    cfg.run.save_fitted_model = True
    cfg.data.id_column = "dispute_id"
    for k, v in over.items():
        section, key = k.split("__")
        setattr(getattr(cfg, section), key, v)
    cfg.validate()
    X = raw.drop(columns=["is_fraudulent_dispute"]).copy()
    y = raw["is_fraudulent_dispute"].to_numpy()
    res = ModelSelectionHarness(cfg).run(X, y)
    return cfg, res, Path(cfg.run.output_dir) / cfg.run.name


def _holdout_frame(raw, res) -> pd.DataFrame:
    ids = res.predictions.loc[res.predictions["stage"] == "holdout", "row_id"]
    return raw.set_index("dispute_id").loc[ids].drop(columns=["is_fraudulent_dispute"])


@pytest.fixture(scope="module")
def refit_run(raw, tmp_path_factory):
    return _run(raw, tmp_path_factory.mktemp("refit"), run__refit_on_full_data=True)


def test_reference_and_top_pct_cut_describe_the_shipped_pipeline(raw, refit_run):
    cfg, res, out = refit_run
    hm = res.holdout_metrics
    X_ho = _holdout_frame(raw, res)
    proba_ship = res.fitted_model.predict_proba(X_ho)[:, 1]
    proba_train_only = res.predictions.loc[res.predictions["stage"] == "holdout", "y_score"].to_numpy()

    assert hm["reference_score_source"] == "production_refit"
    assert hm["decision_threshold_source"] == "production_refit"
    np.testing.assert_allclose(hm["reference_score_quantiles"], reference_quantiles(proba_ship), rtol=0, atol=1e-12)
    # every stored quantile is a score the shipped model actually produced
    assert set(hm["reference_score_quantiles"]) <= set(proba_ship.tolist())

    top = cfg.metrics.lift_top_pct
    assert hm["decision_threshold"] == pytest.approx(float(np.quantile(proba_ship, 1 - top)), abs=1e-12)
    assert hm["decision_threshold_train_only"] == pytest.approx(float(np.quantile(proba_train_only, 1 - top)), abs=1e-12)
    # the refit genuinely moved the scores, so the two are different numbers
    assert hm["decision_threshold"] != hm["decision_threshold_train_only"]
    # and the shipped cut flags the configured share of the shipped model's volume
    assert hm["decision_flag_rate_shipped"] == pytest.approx(top, abs=0.02)

    # the bundle carries the same numbers and the provenance
    scorer = ProductionScorer.from_joblib(out / "model.joblib")
    assert scorer.threshold == pytest.approx(hm["decision_threshold"], abs=0)
    assert scorer.metadata["decision_threshold_source"] == "production_refit"
    assert scorer.metadata["reference_score_source"] == "production_refit"
    scored = scorer.score(X_ho, return_report=False)
    assert float((scored["decision"] == "flag_fraud").mean()) == pytest.approx(top, abs=0.02)

    # the bundle reference is self-consistent with what the shipped model emits:
    # the shipped model's own holdout scores read as stable against it
    assert psi_from_reference_quantiles(scorer.metadata["reference_score_quantiles"], proba_ship) < 0.02


def test_validated_holdout_numbers_are_untouched_by_the_refit(refit_run):
    """The holdout metrics remain the train-only fit's out-of-sample evidence."""
    cfg, res, out = refit_run
    hm = res.holdout_metrics
    ho = res.predictions[res.predictions["stage"] == "holdout"]
    from sklearn.metrics import average_precision_score
    assert hm["average_precision"] == pytest.approx(average_precision_score(ho["y_true"], ho["y_score"]), abs=1e-6)
    # the operating stats at the *validated* cut are still computed on the train-only scores
    s = ho["y_score"].to_numpy()
    assert hm["decision_flag_rate"] == pytest.approx(float((s >= hm["decision_threshold_train_only"]).mean()), abs=1e-6)
    # provenance is written everywhere a reader might look
    import json
    meta = json.loads((out / "predictions_meta.json").read_text())
    assert meta["decision_threshold"] == pytest.approx(hm["decision_threshold"], abs=0)
    assert meta["decision_threshold_source"] == "production_refit"
    step = res.report.get("production_refit")
    assert step["decision_threshold_source"] == "production_refit"
    assert step["decision_threshold_shift"] == pytest.approx(
        hm["decision_threshold"] - hm["decision_threshold_train_only"], abs=1e-6)


def test_fpr_policy_keeps_the_train_only_cut_after_a_refit(raw, tmp_path):
    cfg, res, out = _run(raw, tmp_path, run__refit_on_full_data=True,
                         metrics__decision_threshold_policy="fpr")
    hm = res.holdout_metrics
    assert hm["reference_score_source"] == "production_refit"      # reference always re-derived
    assert hm["decision_threshold_source"] == "train_only_fit"      # the label-dependent cut is not
    assert hm["decision_threshold"] == hm["decision_threshold_train_only"] == hm["implied_threshold_at_fpr"]
    assert ProductionScorer.from_joblib(out / "model.joblib").metadata["decision_threshold_source"] == "train_only_fit"


def test_without_a_refit_everything_is_the_train_only_fit(raw, tmp_path):
    cfg, res, out = _run(raw, tmp_path)                             # make_cfg: refit_on_full_data=False
    hm = res.holdout_metrics
    s = res.predictions.loc[res.predictions["stage"] == "holdout", "y_score"].to_numpy()
    assert hm["reference_score_source"] == hm["decision_threshold_source"] == "train_only_fit"
    assert hm["decision_threshold"] == hm["decision_threshold_train_only"]
    np.testing.assert_allclose(hm["reference_score_quantiles"], reference_quantiles(s), rtol=0, atol=1e-12)
    assert hm["decision_flag_rate_shipped"] == pytest.approx(hm["decision_flag_rate"], abs=1e-6)


# --------------------------------------------------------------------------
# 3. PSI from the bundle alone
# --------------------------------------------------------------------------
def test_psi_from_quantiles_matches_the_raw_array_version_on_continuous_scores():
    rng = np.random.default_rng(1)
    ref = rng.beta(2, 8, 40_000)
    same = rng.beta(2, 8, 20_000)
    shifted = rng.beta(3, 6, 20_000)
    q = reference_quantiles(ref)                       # the grid the bundle stores
    for n_bins in (10, 20):
        for actual in (same, shifted):
            assert psi_from_reference_quantiles(q, actual, n_bins=n_bins) == pytest.approx(
                population_stability_index(ref, actual, n_bins=n_bins), abs=5e-3)
    assert psi_from_reference_quantiles(q, same) < 0.02
    assert psi_from_reference_quantiles(q, shifted) > 0.1


def test_psi_from_quantiles_handles_a_discrete_score_scale():
    # a reference with heavy ties: most grid levels collapse onto six values, and
    # a linearly interpolated grid would have put edges between them
    rng = np.random.default_rng(2)
    pool = [0.1, 0.3, 0.3, 0.3, 0.5, 0.9]
    ref = rng.choice(pool, size=10_000)
    q = reference_quantiles(ref)
    assert set(q) <= set(pool)
    assert len(np.unique(q)) < len(q)
    psi_same = psi_from_reference_quantiles(q, rng.choice(pool, size=3_000))
    psi_moved = psi_from_reference_quantiles(q, rng.choice([0.1, 0.5, 0.9, 0.9], size=3_000))
    assert np.isfinite(psi_same) and psi_same < 0.02
    assert psi_moved > 0.1
    # and it agrees with the raw-array computation on the same discrete data
    assert psi_same == pytest.approx(population_stability_index(ref, rng.choice(pool, size=3_000)), abs=0.02)
