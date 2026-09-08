"""Coverage-maximizing undersampling.

The load-bearing claim is that sampling touches *only* rows a model is fitted
on. Every reported metric -- CV, holdout, slice -- must still be measured on the
full population, or the feature is a leak wearing a config flag. The tests here
name the failure each one prevents.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression

from conftest import make_cfg
from dmf import Config, executive_report
from dmf.research import ModelSelectionHarness, load_predictions
from dmf.research.sampling import UndersampledSplit, coverage_sample, coverage_score


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _blobs(seed: int = 0, sizes=(9000, 900, 100)):
    """Three separated blobs plus an unbalanced categorical.

    The 100-row blob is the point: uniform random undersampling represents it in
    proportion to its 1% share, while a coverage sampler must keep it visible.
    ``y`` is independent of the blob, so nothing here rewards picking one.
    """
    rng = np.random.default_rng(seed)
    centres = np.array([[0, 0, 0, 0], [8, 8, 0, 0], [-8, 0, 8, 0]], dtype=float)
    parts, tags = [], []
    for i, (c, n) in enumerate(zip(centres, sizes)):
        parts.append(rng.normal(c, 1.0, size=(n, 4)))
        tags += [f"blob{i}"] * n
    X = pd.DataFrame(np.vstack(parts), columns=[f"v{i}" for i in range(4)])
    X["seg"] = rng.choice(["a", "b", "c"], size=len(X), p=[0.8, 0.15, 0.05])
    X["blob"] = tags
    y = rng.binomial(1, 0.08, len(X))
    return X, y


def _cfg(**sampling):
    base = {
        "run": {"random_state": 0, "verbose": 0},
        "columns": {"auto_infer": True, "drop": ["blob"]},
        "split": {"cv": {"n_splits": 3}},
        "sampling": {"enabled": True, "purpose": "rebalance",
                     "negative_positive_ratio": 3.0, "min_rows_after": 50,
                     "stratify_columns": ["seg"], **sampling},
    }
    return Config.from_dict(base)


def _sampling_cfg(**sampling):
    """`make_cfg` with a sampling block; imbalance removed so the guard is quiet."""
    cfg = make_cfg()
    for spec in cfg.models.values():
        spec.imbalance = None
    d = cfg.to_dict()
    d["sampling"] = {"enabled": True, "purpose": "rebalance", "negative_positive_ratio": 4.0,
                     "min_rows_after": 50, **sampling}
    return Config.from_dict(d)


# --------------------------------------------------------------------------
# config grammar
# --------------------------------------------------------------------------
def test_sampling_rejects_imbalance_double_correction():
    """Dropping negatives AND reweighting corrects the prior twice."""
    d = make_cfg().to_dict()                      # both models declare imbalance: balanced
    d["sampling"] = {"enabled": True, "purpose": "rebalance", "negative_positive_ratio": 3.0}
    with pytest.raises(ValueError, match="acknowledge_imbalance_double_correction"):
        Config.from_dict(d)


def test_sampling_acknowledged_double_correction_validates():
    d = make_cfg().to_dict()
    d["sampling"] = {"enabled": True, "purpose": "rebalance", "negative_positive_ratio": 3.0,
                     "acknowledge_imbalance_double_correction": True}
    assert Config.from_dict(d).sampling.shifts_prior


def test_compute_sampling_of_both_classes_does_not_trip_the_guard():
    """The guard keys off a prior *shift*, not off sampling being enabled.

    Sampling both classes proportionally for compute reasons leaves P(y=1)
    where it was, so an imbalance policy is not a double correction.
    """
    d = make_cfg().to_dict()
    d["sampling"] = {"enabled": True, "purpose": "compute", "max_fraction": 0.5,
                     "sample_classes": "both"}
    cfg = Config.from_dict(d)
    assert cfg.sampling.enabled and not cfg.sampling.shifts_prior


@pytest.mark.parametrize("block, match", [
    ({"purpose": "rebalance"}, "negative_positive_ratio is required"),
    ({"purpose": "compute"}, "exactly one of sampling.max_rows"),
    ({"purpose": "compute", "max_rows": 100, "negative_positive_ratio": 2.0},
     "does not apply when sampling.purpose='compute'"),
    ({"purpose": "compute", "max_rows": 100, "max_fraction": 0.5}, "exactly one of sampling.max_rows"),
    ({"purpose": "rebalance", "negative_positive_ratio": 2.0, "max_rows": 100}, "do not apply"),
    ({"purpose": "rebalance", "negative_positive_ratio": -1.0}, "must be > 0"),
    ({"purpose": "compute", "max_fraction": 1.5}, "must be in \\(0, 1\\]"),
    ({"purpose": "compute", "max_rows": 3}, "too few rows"),
    ({"purpose": "compute", "max_fraction": 0.5, "clusters_per_stratum": 0},
     "must be 'auto' or an integer"),
])
def test_sampling_budget_grammar(block, match):
    d = make_cfg().to_dict()
    for spec in d["models"].values():
        spec["imbalance"] = None
    d["sampling"] = {"enabled": True, **block}
    with pytest.raises(ValueError, match=match):
        Config.from_dict(d)


def test_disabled_sampling_never_validates_its_own_block():
    """A stale block must not block a run that does not use it."""
    d = make_cfg().to_dict()
    d["sampling"] = {"enabled": False, "purpose": "rebalance", "max_rows": 1}
    assert Config.from_dict(d).sampling.enabled is False


# --------------------------------------------------------------------------
# sampler properties
# --------------------------------------------------------------------------
def test_every_positive_and_every_categorical_level_survives():
    """The per-stratum floor is the coverage guarantee; a rare level must live."""
    X, y = _blobs(seed=1)
    X.loc[X.index[:30], "seg"] = "rare"           # ~0.3% of rows
    res = coverage_sample(X, y, _cfg(), seed=5)
    d = res.diagnostics
    assert d["n_positive_out"] == d["n_positive_in"], "positives must be kept verbatim"
    assert set(X.iloc[res.indices]["seg"]) == set(X["seg"]), "a level was sampled away"
    assert d["coverage_retained_categorical"] == 1.0


def test_stratified_cluster_beats_random_on_coverage():
    """The technique must earn its complexity against uniform random.

    Asserted on the TAIL, never the mean: in high dimensions no allocation rule
    moves *average* nearest-neighbour distance by more than ~1%, so a mean-based
    assertion would be false. Measured over five seeds on this fixture:

      sparse-blob share  2x random, 5/5 seeds
      p95 nn-distance    ~7% better, 5/5 seeds
      max nn-distance    ~18% better on average, but only 3/5 seeds -- it is a
                         single-point statistic, so it is asserted in aggregate
                         rather than per seed.
    """
    seeds = (0, 1, 2, 3, 4)
    s_max, r_max = [], []
    for seed in seeds:
        X, y = _blobs(seed=seed)
        s_res = coverage_sample(X, y, _cfg(), seed=seed)
        r_res = coverage_sample(X, y, _cfg(strategy="random"), seed=seed)
        strat, rand = s_res.diagnostics, r_res.diagnostics
        assert strat["n_rows_out"] == rand["n_rows_out"], "arms must share a budget"

        assert strat["p95_nn_distance"] <= rand["p95_nn_distance"], (
            f"seed {seed}: p95 coverage gap {strat['p95_nn_distance']} not better "
            f"than random's {rand['p95_nn_distance']}"
        )
        # the mechanism: the 1%-of-population blob is over-represented
        s_share = (X.iloc[s_res.indices]["blob"] == "blob2").mean()
        r_share = (X.iloc[r_res.indices]["blob"] == "blob2").mean()
        assert s_share >= r_share, f"seed {seed}: sparse blob {s_share} vs random {r_share}"
        s_max.append(strat["max_nn_distance"])
        r_max.append(rand["max_nn_distance"])

    assert np.mean(s_max) < np.mean(r_max), (
        f"worst-gap coverage {np.mean(s_max):.3f} not better than random {np.mean(r_max):.3f}"
    )


@pytest.mark.parametrize("allocation", ["sqrt", "proportional", "equal"])
def test_allocation_rules_all_respect_the_budget(allocation):
    X, y = _blobs(seed=3)
    res = coverage_sample(X, y, _cfg(cluster_allocation=allocation), seed=1)
    d = res.diagnostics
    n_pos = int(y.sum())
    assert d["n_positive_out"] == n_pos
    assert len(res) == pytest.approx(n_pos * 4, rel=0.05)     # positives + 3x negatives


def test_sampler_is_deterministic():
    X, y = _blobs(seed=2)
    a = coverage_sample(X, y, _cfg(), seed=11).indices
    b = coverage_sample(X, y, _cfg(), seed=11).indices
    c = coverage_sample(X, y, _cfg(), seed=12).indices
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c), "the seed must reach k-means"
    assert len(a) == len(c)


def test_sampler_handles_nulls_without_dropping_rows():
    """Missingness must not become an implicit selection criterion."""
    X, y = _blobs(seed=4)
    rng = np.random.default_rng(0)
    missing = rng.random(len(X)) < 0.3
    X.loc[missing, "v0"] = np.nan
    X["all_null"] = np.nan

    res = coverage_sample(X, y, _cfg(), seed=1)
    kept_missing_rate = missing[res.indices].mean()
    assert res.diagnostics["applied"]
    assert abs(kept_missing_rate - missing.mean()) < 0.10, (
        f"NaN rows kept at {kept_missing_rate:.3f} against a population rate of "
        f"{missing.mean():.3f} -- missingness is steering the sample"
    )


def test_sampler_is_a_no_op_when_the_budget_is_not_binding():
    X, y = _blobs(seed=5, sizes=(300, 60, 40))
    cfg = _cfg(negative_positive_ratio=1000.0)
    res = coverage_sample(X, y, cfg, seed=1)
    assert len(res) == len(X)
    assert res.diagnostics.get("budget_not_binding") is True


def test_coverage_score_is_zero_when_nothing_was_dropped():
    Z = np.random.default_rng(0).normal(size=(50, 3))
    out = coverage_score(Z, np.arange(50), rng=np.random.default_rng(0))
    assert out["max_nn_distance"] == 0.0


# --------------------------------------------------------------------------
# the safety claim
# --------------------------------------------------------------------------
def test_no_validation_or_holdout_row_is_ever_dropped(raw, tmp_path):
    """Sampling must reach fitting halves only -- proved from stored artifacts.

    If a validation fold or the holdout were ever sampled, every metric in the
    run would be measured on a population the model was optimised for.
    """
    cfg = _sampling_cfg()
    cfg.run.output_dir = str(tmp_path)
    cfg.run.save_predictions = "all"
    res = ModelSelectionHarness(cfg).run(raw.copy())

    preds, _ = load_predictions(tmp_path / cfg.run.name)
    rep = res.report
    split = rep.get("holdout_split")
    n_train, n_holdout = split["n_train"], split["n_holdout"]

    # the partition invariant still holds -- sampling added keys, redefined none
    assert n_train + n_holdout == rep.get("data_load")["n_rows"]

    # every training row appears in exactly one validation fold
    folds = res.fold_assignments
    assert len(folds) == n_train
    assert folds["row_id"].nunique() == n_train

    train_ids = set(folds["row_id"])
    cv = preds[preds["stage"] == "cv"]
    one_cell = cv[(cv["model"] == cv["model"].iloc[0]) & (cv["k"] == cv["k"].iloc[0])]
    assert set(one_cell["row_id"]) == train_ids, "a validation row was sampled away"

    assert len(preds[preds["stage"] == "holdout"]) == n_holdout, "the holdout was sampled"

    # and the fitting halves demonstrably were sampled
    cv_train_ids = set(preds[preds["stage"] == "cv_train"]["row_id"])
    assert cv_train_ids <= train_ids
    samp = rep.get("training_sampling")
    assert samp["sampling_rate"] < 1.0 and samp["holdout_untouched"] is True


def test_tuning_inner_test_folds_are_byte_identical():
    """A search owns its inner split; only the train side may shrink."""
    X, y = _blobs(seed=6, sizes=(1200, 300, 100))
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    wrapped = UndersampledSplit(inner, X, y, _cfg(), seed=2)

    got, expect = list(wrapped.split(X, y)), list(inner.split(X, y))
    assert wrapped.get_n_splits(X, y) == inner.get_n_splits(X, y)
    for (tr_w, te_w), (tr_p, te_p) in zip(got, expect):
        np.testing.assert_array_equal(te_w, te_p)          # scoring half untouched
        assert set(tr_w) < set(tr_p)                       # fitting half strictly smaller

    # memoized: a second pass yields the same folds, not a fresh k-means
    for (a, _), (b, _) in zip(got, list(wrapped.split(X, y))):
        np.testing.assert_array_equal(a, b)

    search = GridSearchCV(LogisticRegression(max_iter=500), {"C": [0.1, 1.0]},
                          cv=wrapped, scoring="average_precision")
    search.fit(X.drop(columns=["seg", "blob"]), y)
    assert search.n_splits_ == 3


# --------------------------------------------------------------------------
# reporting and the no-op guarantee
# --------------------------------------------------------------------------
def test_sampling_disabled_is_bit_identical(raw, tmp_path):
    """The feature must be inert when off, which is what makes it safe to ship."""
    def run(block, name):
        cfg = make_cfg()
        d = cfg.to_dict()
        d["run"]["name"] = name
        d["run"]["output_dir"] = str(tmp_path)
        if block is not None:
            d["sampling"] = block
        return ModelSelectionHarness(Config.from_dict(d)).run(raw.copy())

    absent = run(None, "absent")
    disabled = run({"enabled": False}, "disabled")
    # fit_seconds is wall-clock and never reproduces; everything the leaderboard
    # actually asserts about a model must
    drop = ["fit_seconds"]
    pd.testing.assert_frame_equal(absent.leaderboard.drop(columns=drop),
                                  disabled.leaderboard.drop(columns=drop))
    assert absent.report.get("training_sampling") is None


def test_sampling_report_and_risk_flags(raw, tmp_path):
    cfg = make_cfg()
    d = cfg.to_dict()
    d["run"]["output_dir"] = str(tmp_path)
    d["sampling"] = {"enabled": True, "purpose": "rebalance", "negative_positive_ratio": 3.0,
                     "min_rows_after": 50, "acknowledge_imbalance_double_correction": True}

    with pytest.warns(RuntimeWarning, match="not probabilities"):
        res = ModelSelectionHarness(Config.from_dict(d)).run(raw.copy())

    samp = res.report.get("training_sampling")
    assert samp is not None
    assert samp["prior_shifted"] is True
    assert samp["uncalibrated_prior_shift"] is True
    assert samp["double_corrected_imbalance"] is True      # acknowledged, so reported not raised
    assert samp["holdout_untouched"] is True
    assert samp["train_prevalence_fitted"] > samp["train_prevalence_available"]
    assert samp["overfit_gap_basis"] == "sampled_fit_half"
    assert samp["coverage_retained_categorical"] is not None


def test_scale_pos_weight_is_recomputed_from_the_sampled_rows(raw, tmp_path):
    """A frozen neg/pos from the unsampled partition would over-correct."""
    def basis(sampling):
        d = make_cfg().to_dict()
        d["run"]["output_dir"] = str(tmp_path)
        d["run"]["name"] = "spw" + str(bool(sampling))
        d["models"] = {"xgb": {"estimator": "sklearn.ensemble.HistGradientBoostingClassifier",
                               "family": "tree", "requires_scaling": False,
                               "imbalance": "scale_pos_weight",
                               "params": {"max_iter": 30}}}
        if sampling:
            d["sampling"] = sampling
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ModelSelectionHarness(Config.from_dict(d)).run(raw.copy())
        return res.report.get("estimator_zoo")["scale_pos_weight_basis"]

    assert basis(None) == "full_training_partition"
    assert basis({"enabled": True, "purpose": "rebalance", "negative_positive_ratio": 3.0,
                  "min_rows_after": 50, "acknowledge_imbalance_double_correction": True,
                  "acknowledge_uncalibrated": True}) == "sampled_rows"


def test_executive_report_surfaces_sampling(raw, tmp_path):
    """A new report step does not appear in the HTML on its own."""
    d = _sampling_cfg().to_dict()
    d["run"]["output_dir"] = str(tmp_path)
    d["sampling"]["acknowledge_uncalibrated"] = True
    res = ModelSelectionHarness(Config.from_dict(d)).run(raw.copy())
    assert res.report.get("training_sampling")["prior_shifted"] is True

    html = executive_report(tmp_path / d["run"]["name"]).read_text(encoding="utf-8")
    assert "Training sample rate" in html
    assert "Training rows were undersampled" in html


def test_sampled_model_ranks_but_decalibrates(raw, tmp_path):
    """Sampling preserves ranking and degrades calibration -- the stated trade.

    The signal is deliberately moderate. A near-separable one drives every score
    to 0 or 1, where the mean predicted probability tracks prevalence again and
    the prior shift becomes invisible -- the effect lives in the middle of the
    score range.
    """
    df = raw.drop(columns=["dispute_id"]).copy()
    rng = np.random.default_rng(0)
    df["signal"] = df["is_fraudulent_dispute"] * 1.6 + rng.normal(0, 1.0, len(df))

    d = {
        "run": {"random_state": 0, "verbose": 0, "output_dir": str(tmp_path),
                "save_fitted_model": False, "refit_on_full_data": False},
        "data": {"target": "is_fraudulent_dispute"},
        "columns": {"numeric": ["signal"], "categorical": [], "auto_infer": False},
        "split": {"holdout_size": 0.3, "cv": {"n_splits": 3}},
        "selection": {"k_min": 1, "k_max": 1, "top_n": 1},
        "sampling": {"enabled": True, "purpose": "rebalance", "negative_positive_ratio": 1.0,
                     "min_rows_after": 50, "acknowledge_uncalibrated": True},
        "models": {"logistic": {"estimator": "sklearn.linear_model.LogisticRegression",
                                "requires_scaling": True, "params": {"max_iter": 500}}},
    }
    res = ModelSelectionHarness(Config.from_dict(d)).run(df)
    hm = res.holdout_metrics
    assert hm["average_precision"] > 3 * hm["prevalence"], "ranking should survive sampling"
    assert hm["calibration_ratio"] > 1.5, "a 1:1 sample should inflate scores"
