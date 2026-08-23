"""Regressions for every defect found in the production-readiness review.

Each test names the failure it prevents from coming back.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV

from dmf import Config, DisputeFeaturePipeline, ProductionScorer
from dmf.research import ModelSelectionHarness
from dmf import build_model_pipeline, decile_table
from dmf.research.selection import _paired_test
from dmf.research.zoo import build_estimator, config_for_model


# --------------------------------------------------------------------------
# critical: leakage
# --------------------------------------------------------------------------
def test_target_encoder_cross_fitting_survives_fit_transform():
    """fit_transform must return the out-of-fold encoding, not re-transform.

    Re-running transform() after fit substitutes the full-data per-level mean,
    and the estimator is then trained on a column that has seen its own target.
    """
    rng = np.random.default_rng(0)
    n = 3000
    X = pd.DataFrame({"merchant_id": rng.integers(0, 400, n).astype(str)})
    y = rng.binomial(1, 0.2, n)                       # pure noise target
    cfg = Config.from_dict({
        "data": {"target": "t"},
        "columns": {"categorical": ["merchant_id"], "auto_infer": False},
        "preprocessing": {"categorical": {"encoder": "target",
                                          "rare_level": {"enabled": False}}},
    })
    fp = DisputeFeaturePipeline(config=cfg)
    fitted = fp.fit_transform(X, y).iloc[:, 0].to_numpy()
    assert abs(np.corrcoef(fitted, y)[0, 1]) < 0.05, "fit_transform returned a leaked encoding"


def test_grid_is_honest_on_pure_noise():
    """Variable ordering must run inside the CV loop.

    The flat construction (rank once on the training partition, cross-validate
    the ranked subsets on the same partition) reported ~0.62 AP on this exact
    dataset while the truth is 0.50, with an overfit gap of ~0. It was removed
    outright; this test pins the honest behaviour of the only remaining path.
    """
    rng = np.random.default_rng(0)
    n, p = 900, 40
    X = pd.DataFrame(rng.normal(0, 1, (n, p)), columns=[f"v{i}" for i in range(p)])
    X["t"] = rng.binomial(1, 0.5, n)                  # true AP = 0.50

    cfg = Config.from_dict({
        "run": {"name": "noise", "output_dir": "/tmp/dmf_bias", "n_jobs": 1,
                "verbose": 0, "save_fitted_model": False, "refit_on_full_data": False},
        "data": {"target": "t"}, "columns": {"auto_infer": True},
        "split": {"holdout_size": 0.25, "cv": {"n_splits": 3}},
        "selection": {"k_min": 1, "k_max": 4, "top_n": 2},
        "metrics": {"primary": "average_precision", "secondary": ["roc_auc"]},
        "models": {"lr": {"estimator": "sklearn.linear_model.LogisticRegression",
                          "requires_scaling": True, "params": {"max_iter": 400}}},
    })
    res = ModelSelectionHarness(cfg).run(X.copy())

    best = res.leaderboard["cv_average_precision_mean"].max()
    assert best < 0.56, f"{best - 0.5:.3f} of selection bias on pure noise"
    # the overfit-gap diagnostic must register the (real) overfitting
    assert res.leaderboard["overfit_gap"].max() > 0.01
    assert res.report.get("selection_stability") is not None


def test_sample_frac_keeps_rows_and_labels_aligned(tmp_path):
    """A non-integer index used to pair a random row sample with the first n labels."""
    rng = np.random.default_rng(0)
    n = 1200
    a = rng.normal(0, 1, n)
    df = pd.DataFrame({"a": a, "b": rng.normal(0, 1, n), "t": (a > 0).astype(int)},
                      index=[f"d{i}" for i in range(n)])
    cfg = Config.from_dict({
        "run": {"name": "frac", "output_dir": str(tmp_path), "n_jobs": 1, "verbose": 0,
                "save_fitted_model": False, "refit_on_full_data": False},
        "data": {"target": "t", "sample_frac": 0.5},
        "columns": {"auto_infer": True},
        "split": {"holdout_size": 0.25, "cv": {"n_splits": 3}},
        "selection": {"k_min": 1, "k_max": 2, "top_n": 1},
        "metrics": {"primary": "average_precision", "secondary": ["roc_auc"]},
        "models": {"lr": {"estimator": "sklearn.linear_model.LogisticRegression",
                          "requires_scaling": True, "params": {"max_iter": 400}}},
    })
    res = ModelSelectionHarness(cfg).run(df)
    # perfectly separable data: a scrambled target collapses this to ~0.5
    assert res.holdout_metrics["average_precision"] > 0.95


# --------------------------------------------------------------------------
# critical: guard state
# --------------------------------------------------------------------------
def _fitted_scorer(cfg, X, y, features):
    spec = cfg.models["logistic"]
    pipe = build_model_pipeline(config_for_model(cfg, spec), features,
                                build_estimator(spec, 0, 1, y))
    return pipe.fit(X, y)


def test_guard_is_not_bypassed_by_a_calibration_wrapper(cfg, Xy):
    """predict_proba runs on clones; the report must not come from a stale prototype."""
    X, y = Xy
    features = ["dispute_amount", "channel"]
    pipe = _fitted_scorer(cfg, X, y, features)
    calibrated = CalibratedClassifierCV(estimator=pipe, cv=3, method="sigmoid").fit(X, y)

    bad = X.head(200).copy()
    bad["dispute_amount"] = 1e9
    bad["channel"] = "brand_new_channel"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scorer = ProductionScorer(calibrated, features=features, threshold=0.5)
        scored, report = scorer.score(bad)

    assert report["verdict"] == "review_recommended"
    assert (scored["data_quality"] == "guarded").all()
    assert (scored["action"] == "manual_review").all()


def test_concurrent_scoring_does_not_cross_contaminate(cfg, Xy):
    """Two threads sharing one loaded model must not swap each other's reports."""
    from concurrent.futures import ThreadPoolExecutor

    X, y = Xy
    features = ["dispute_amount", "channel"]
    scorer = ProductionScorer(_fitted_scorer(cfg, X, y, features), features=features,
                              threshold=0.5)
    clean = X.head(200).copy()
    dirty = clean.copy()
    dirty["channel"] = "never_seen_level"

    def work(i):
        frame = clean if i % 2 == 0 else dirty
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scored, _ = scorer.score(frame)
        return i % 2, int((scored["data_quality"] == "guarded").sum())

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(work, range(24)))

    for kind, n_guarded in results:
        assert n_guarded == (0 if kind == 0 else 200), f"batch {kind} reported {n_guarded} guarded"


def test_missing_column_forces_manual_review_on_every_row(cfg, Xy):
    """A batch scored on imputed training defaults must never be auto-actioned."""
    X, y = Xy
    features = ["dispute_amount", "prior_disputes_12m", "channel"]
    scorer = ProductionScorer(_fitted_scorer(cfg, X, y, features), features=features,
                              threshold=0.5)
    batch = X.head(300).drop(columns=["prior_disputes_12m"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scored, report = scorer.score(batch)

    assert report["missing_columns"] == ["prior_disputes_12m"]
    assert (scored["n_newly_missing"] == 1).all()
    assert (scored["action"] == "manual_review").all()
    assert not (scored["action"] == "auto_action").any()


# --------------------------------------------------------------------------
# major: schema, decisions, statistics
# --------------------------------------------------------------------------
def test_transform_refuses_array_and_renamed_input(cfg, Xy):
    X, y = Xy
    fp = DisputeFeaturePipeline(config=cfg, features=["dispute_amount", "channel"]).fit(X, y)
    with pytest.raises(TypeError, match="requires a DataFrame"):
        fp.transform(X.to_numpy())
    with pytest.raises(KeyError, match="schema mismatch"):
        fp.transform(X.rename(columns={c: f"{c}_v2" for c in X.columns}))


def test_relative_cut_is_refused_on_a_small_batch(cfg, Xy):
    X, y = Xy
    features = ["dispute_amount", "channel"]
    scorer = ProductionScorer(_fitted_scorer(cfg, X, y, features), features=features,
                              top_pct=0.05)
    with pytest.raises(ValueError, match="at least 50 records"):
        scorer.score(X.head(10))
    scored, _ = scorer.score(X.head(200))               # a real batch is fine
    assert len(scored) == 200


def test_marginal_gain_direction_is_correct_for_loss_metrics(tmp_path):
    """With log_loss as primary, a decrease is an improvement."""
    rng = np.random.default_rng(0)
    n = 1500
    a, b = rng.normal(0, 1, n), rng.normal(0, 1, n)
    y = rng.binomial(1, 1 / (1 + np.exp(-(1.2 * a + 1.0 * b))))
    df = pd.DataFrame({"a": a, "b": b, "c": rng.normal(0, 1, n), "t": y})
    cfg = Config.from_dict({
        "run": {"name": "loss", "output_dir": str(tmp_path), "n_jobs": 1, "verbose": 0,
                "save_fitted_model": False, "refit_on_full_data": False},
        "data": {"target": "t"}, "columns": {"auto_infer": True},
        "split": {"holdout_size": 0.25, "cv": {"n_splits": 4}},
        "selection": {"k_min": 1, "k_max": 3, "top_n": 1, "marginal_gain_test": "paired_t"},
        "metrics": {"primary": "log_loss", "secondary": ["roc_auc"]},
        "models": {"lr": {"estimator": "sklearn.linear_model.LogisticRegression",
                          "requires_scaling": True, "params": {"max_iter": 400}}},
    })
    res = ModelSelectionHarness(cfg).run(df)
    step = res.marginal_gains[res.marginal_gains["to_k"] == 2].iloc[0]
    assert step["log_loss_to"] < step["log_loss_from"]      # loss fell
    assert step["mean_delta"] > 0                          # ...so the gain is positive
    assert step["verdict"] == "improves"
    assert res.selected["cost_of_parsimony"] >= 0


def test_paired_test_edge_cases():
    # wilcoxon cannot reach p<0.05 at 5 folds; say so instead of "not_significant"
    out = _paired_test(np.array([0.10, 0.11, 0.09, 0.12, 0.10]), "wilcoxon", 5)
    assert out["verdict"] == "underpowered" and "attainable" in out["note"]
    # the same deltas are clearly significant under the corrected paired t
    assert _paired_test(np.array([0.10, 0.11, 0.09, 0.12, 0.10]), "paired_t", 5)["verdict"] \
        == "improves"
    # every fold failing is not a tie
    assert _paired_test(np.array([np.nan] * 5), "paired_t", 5)["verdict"] == "all_folds_failed"
    assert Config().selection.marginal_gain_test == "paired_t"


def test_decile_table_handles_a_tiny_holdout():
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0.1, 0.9, 0.2, 0.8, 0.7])
    dt = decile_table(y, p, n_bins=10)
    assert len(dt) == len(y) and dt["n"].sum() == len(y)


def test_duplicate_column_names_raise_a_named_error(cfg):
    df = pd.DataFrame(np.zeros((10, 3)), columns=["a", "b", "a"])
    from dmf import infer_roles
    with pytest.raises(ValueError, match="Duplicate column names"):
        infer_roles(df, cfg)


# --------------------------------------------------------------------------
# artifacts
# --------------------------------------------------------------------------
def test_final_spec_pins_the_winning_specification(raw, tmp_path):
    cfg = Config.from_dict({
        "run": {"name": "pin", "output_dir": str(tmp_path), "n_jobs": 1, "verbose": 0,
                "save_fitted_model": True, "refit_on_full_data": False},
        "data": {"target": "is_fraudulent_dispute"},
        "columns": {"auto_infer": True, "drop": ["dispute_id", "merchant_id"]},
        "split": {"holdout_size": 0.25, "cv": {"n_splits": 3}},
        "selection": {"k_min": 1, "k_max": 3, "top_n": 2},
        "metrics": {"primary": "average_precision", "secondary": ["roc_auc"]},
        "models": {"lr": {"estimator": "sklearn.linear_model.LogisticRegression",
                          "requires_scaling": True, "params": {"max_iter": 400}}},
    })
    res = ModelSelectionHarness(cfg).run(raw.copy())
    spec = Config.from_yaml(tmp_path / "pin" / "final_spec.yaml")
    k = res.selected["k"]
    assert spec.columns.auto_infer is False
    assert spec.declared_features == res.selected_features
    assert spec.selection.k_min == spec.selection.k_max == k
    assert spec.tuning.enabled is False

    import joblib
    bundle = joblib.load(tmp_path / "pin" / "model.joblib")
    assert bundle["dmf_version"] and "decision_threshold" in bundle and "lineage" in bundle
