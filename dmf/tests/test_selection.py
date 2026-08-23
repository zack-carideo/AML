"""Selection harness: grid shape, ordering strategies, statistics, artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dmf import Config
from dmf.research import ModelSelectionHarness
from dmf.metrics import METRIC_REGISTRY, decile_table, evaluate_predictions, orient
from dmf.research.ordering import rank_variables
from dmf.research.selection import _paired_test
from dmf.research.zoo import build_estimator, config_for_model


@pytest.fixture(scope="module")
def result(tmp_path_factory, request):
    """Run the harness once and share the result across the assertions below."""
    from conftest import make_cfg

    out = tmp_path_factory.mktemp("artifacts")
    cfg = make_cfg()
    raw = request.getfixturevalue("raw")
    cfg.run.output_dir = str(out)
    cfg.run.save_fitted_model = True
    return ModelSelectionHarness(cfg).run(raw.copy()), cfg, out


def test_grid_covers_every_model_and_k(result):
    res, cfg, _ = result
    lb = res.leaderboard
    ks = list(range(cfg.selection.k_min, cfg.selection.k_max + 1))
    assert set(lb["model"]) == set(cfg.enabled_models)
    assert len(lb) == len(cfg.enabled_models) * len(ks)
    for model in cfg.enabled_models:
        assert sorted(lb.loc[lb["model"] == model, "k"]) == ks
    # nested subsets: the k-variable spec must contain the (k-1)-variable spec
    for model in cfg.enabled_models:
        sub = lb[lb["model"] == model].sort_values("k")
        prev = None
        for _, row in sub.iterrows():
            feats = row["features"].split(",")
            assert len(feats) == row["k"]
            if prev is not None:
                assert set(prev) < set(feats)
            prev = feats


def test_metrics_are_oriented_to_natural_units(result):
    res, cfg, _ = result
    lb = res.leaderboard
    assert (lb["cv_average_precision_mean"].between(0, 1)).all()
    assert (lb["cv_roc_auc_mean"].between(0, 1)).all()
    assert (lb["cv_brier_score_mean"] >= 0).all()          # un-negated loss
    assert (lb["cv_average_precision_se"] >= 0).all()


def test_top_specs_and_one_se_selection(result):
    res, cfg, _ = result
    assert len(res.top_specs) == cfg.selection.top_n
    scores = [s["cv_average_precision_mean"] for s in res.top_specs]
    assert scores == sorted(scores, reverse=True)

    best = res.leaderboard["cv_average_precision_mean"].max()
    sel = res.selected
    assert sel["rule"] == "one_standard_error"
    assert sel["cv_average_precision_mean"] <= best + 1e-12
    assert sel["cost_of_parsimony"] >= 0
    assert len(sel["features"]) == sel["k"]
    # the 1-SE choice must be within one SE of the outright best
    best_row = res.leaderboard.sort_values("cv_average_precision_mean", ascending=False).iloc[0]
    assert sel["cv_average_precision_mean"] >= best - best_row["cv_average_precision_se"] - 1e-12


def test_marginal_gains_are_paired_and_labelled(result):
    res, cfg, _ = result
    g = res.marginal_gains
    n_models = len(cfg.enabled_models)
    assert len(g) == n_models * (cfg.selection.k_max - cfg.selection.k_min)
    assert (g["to_k"] == g["from_k"] + 1).all()
    assert g["added_variable"].notna().all()
    assert set(g["verdict"]) <= {"improves", "degrades", "not_significant", "no_change",
                                 "insufficient_folds"}
    recomputed = g["average_precision_to"] - g["average_precision_from"]
    np.testing.assert_allclose(recomputed, g["mean_delta"], atol=1e-6)


def test_holdout_is_never_touched_during_selection(result):
    res, cfg, _ = result
    n_total = res.report.get("data_load")["n_rows"]
    split = res.report.get("holdout_split")
    assert split["n_train"] + split["n_holdout"] == n_total
    assert split["stratified"] is True
    assert split["prevalence_abs_diff"] < 0.01          # stratification worked
    assert res.holdout_metrics["average_precision"] is not None


def test_artifacts_written(result):
    res, cfg, out = result
    d = Path(out) / cfg.run.name
    for name in ["leaderboard.csv", "marginal_gains.csv", "best_per_model.csv",
                 "orderings.json", "top_specs.json", "selected_spec.json",
                 "holdout_metrics.json", "run_report.json", "final_spec.yaml",
                 "feature_pipeline_report.json", "model.joblib", "holdout_slices.csv"]:
        assert (d / name).exists(), name
    spec = json.loads((d / "selected_spec.json").read_text())
    assert spec["model"] == res.selected_model
    reloaded = Config.from_yaml(d / "final_spec.yaml")
    assert reloaded.run.name.endswith("__final")


def test_holdout_slice_report(result):
    """Per-segment holdout performance and flag rates, for fairness review."""
    res, cfg, _ = result
    slices = res.holdout_slices
    assert slices is not None and len(slices)
    assert set(slices["slice_column"]) == {"channel", "merchant_category"}
    assert (slices["n"] >= cfg.metrics.min_slice_n).all()
    assert slices["flag_rate_at_top_pct"].between(0, 1).all()
    step = res.report.get("holdout_slices")
    assert step["max_flag_rate_disparity"] >= 1.0


def test_leaderboard_se_uses_nadeau_bengio_correction(result):
    """The 1-SE band and the paired test must share one variance model."""
    from dmf.research.selection import _nb_factor

    res, cfg, _ = result
    n = cfg.split.cv.n_splits
    # at 5 folds the corrected SE is 1.5x the naive one; at 3 folds sqrt(1+3/2)
    assert _nb_factor(5, 5) * np.sqrt(5) == pytest.approx(1.5)
    lb = res.leaderboard
    expected = lb["cv_average_precision_std"] * _nb_factor(n, n)
    np.testing.assert_allclose(lb["cv_average_precision_se"], expected, atol=1e-5)


def test_fitted_model_scores_new_records(result):
    res, cfg, _ = result
    assert res.fitted_model is not None
    from conftest import generate_disputes

    new = generate_disputes(n=120, seed=555).drop(columns=["dispute_id", "is_fraudulent_dispute"])
    p = res.fitted_model.predict_proba(new)[:, 1]
    assert p.shape == (120,) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()


@pytest.mark.parametrize("strategy", ["importance", "rfe"])
def test_every_ordering_strategy_returns_a_full_ranking(cfg, Xy, strategy):
    X, y = Xy
    c = cfg.copy()
    c.selection.ordering_strategy = strategy
    spec = c.models["logistic"]
    order, report = rank_variables(X, y, config_for_model(c, spec),
                                   build_estimator(spec, 0, 1, y), c.declared_features)
    assert sorted(order) == sorted(c.declared_features)     # a permutation, nothing dropped
    assert report["ordering"] == order
    assert report["n_candidates"] == len(c.declared_features)


def test_ordering_prefers_signal_over_planted_noise(cfg, Xy):
    X, y = Xy
    spec = cfg.models["trees"]
    order, _ = rank_variables(X, y, config_for_model(cfg, spec),
                              build_estimator(spec, 0, 1, y), cfg.declared_features)
    assert order.index("prior_disputes_12m") < order.index("noise_gaussian")


def test_nadeau_bengio_correction_is_conservative():
    d = np.array([0.01, 0.012, 0.009, 0.011, 0.010])
    corrected = _paired_test(d, "paired_t", n_splits=5)
    naive_t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    assert abs(corrected["statistic"]) < abs(naive_t)      # variance inflated, t shrunk
    assert 0.0 <= corrected["p_value"] <= 1.0

    assert _paired_test(np.zeros(5), "wilcoxon", 5)["verdict"] == "no_change"
    assert _paired_test(np.array([1.0]), "paired_t", 5)["verdict"] == "insufficient_folds"


def test_metric_orientation_and_evaluation():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.1, 500)
    p = np.clip(0.1 + 0.6 * y + rng.normal(0, 0.15, 500), 0.001, 0.999)

    assert orient("brier_score", -0.2) == pytest.approx(0.2)
    assert orient("roc_auc", 0.8) == pytest.approx(0.8)

    cfg_metrics = Config().metrics
    out = evaluate_predictions(y, p, cfg_metrics)
    assert set(out) >= {"average_precision", "roc_auc", "ks_statistic", "prevalence",
                        "calibration_ratio", "calibration_error"}
    assert 0 <= out["average_precision"] <= 1 and out["roc_auc"] > 0.9

    dt = decile_table(y, p)
    assert len(dt) == 10
    assert dt["lift"].iloc[0] > dt["lift"].iloc[-1]
    assert dt["cumulative_capture"].iloc[-1] == pytest.approx(1.0, abs=1e-6)
    assert all(name in METRIC_REGISTRY for name in cfg_metrics.secondary)


def test_sweep_compares_configs_on_holdout(raw, tmp_path):
    """Two configs, same seed/split -> comparable; ranked on holdout primary."""
    from conftest import make_cfg
    from dmf.research import check_comparability, run_sweep

    a, b = make_cfg(), make_cfg()
    a.run.name, b.run.name = "onehot", "woe"
    b.preprocessing.categorical.encoder = "woe"
    for c in (a, b):
        c.run.output_dir = str(tmp_path)

    comparison, results = run_sweep([a, b], X=raw.copy())
    assert list(comparison["run"]) and set(results) == {"onehot", "woe"}
    assert comparison["comparable"].all()
    assert comparison["config_sha256"].nunique() == 2       # configs really differed
    assert comparison["data_sha256"].nunique() == 1         # ...on identical data
    # ranked on holdout, best first (same primary everywhere)
    hp = comparison["holdout_primary"].to_numpy()
    assert (hp[:-1] >= hp[1:]).all()
    assert (tmp_path / "sweep_comparison.csv").exists()

    # differing seeds must be flagged as not like-for-like
    b.run.random_state = 99
    assert "run.random_state" in check_comparability([a, b])
    with pytest.warns(RuntimeWarning, match="NOT like-for-like"):
        cmp2, _ = run_sweep([a, b], X=raw.copy())
    assert not cmp2["comparable"].any()


def test_functional_api_returns_objects(raw, tmp_path):
    """train()/score() must return objects, mirroring CLI flags as kwargs."""
    from conftest import make_cfg
    from dmf.research import SelectionResult, train
    from dmf.research.api import score

    cfg = make_cfg()
    cfg.run.save_fitted_model = True
    res = train(cfg, X=raw.copy(), output_dir=str(tmp_path), k_max=2, quiet=True)
    assert isinstance(res, SelectionResult)
    assert res.leaderboard["k"].max() == 2               # override applied

    with pytest.raises(TypeError, match="Unknown train override"):
        train(cfg, X=raw.copy(), bogus=1)

    out = tmp_path / "scored.csv"
    scored, report = score(tmp_path / "test_run" / "model.joblib",
                           raw.drop(columns=["is_fraudulent_dispute"]).head(100),
                           out=out, id_column="dispute_id", threshold=0.5)
    assert {"fraud_probability", "decision", "action"} <= set(scored.columns)
    assert scored.columns[0] == "dispute_id" and isinstance(report, dict)
    assert out.exists()
