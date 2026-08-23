"""Hardening regressions: column typing, split strategies, lineage, monitoring.

Every assertion here corresponds to a failure found by ``examples/edge_case_audit.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dmf import (
    Config,
    DisputeFeaturePipeline,
    ProductionScorer,
    build_model_pipeline,
    infer_roles,
    parse_kind,
    population_stability_index,
    psi_band,
    run_lineage,
    to_numeric_lenient,
)
from dmf.research import ModelSelectionHarness
from dmf.research.selection import _binarize_target
from dmf.research.zoo import build_estimator, config_for_model


# --------------------------------------------------------------------------
# value parsing
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "values,expected_kind",
    [
        (["$1,234.50", "$2.00", "$18.00"], "numeric"),
        (["1 234", "2 000", "3 500"], "numeric"),
        (["12.5%", "3.0%", "88.1%"], "numeric"),
        (["2026-01-05", "2026-02-06", "2026-03-07"], "datetime"),
        (["grocery", "fuel", "travel"], None),
    ],
)
def test_parse_kind_recognises_disguised_types(values, expected_kind):
    kind, _ = parse_kind(pd.Series(values * 5))
    assert kind == expected_kind


def test_sentinels_do_not_disqualify_a_numeric_column():
    s = pd.Series(["512.1"] * 97 + ["N/A", "N/A", "unknown"])
    kind, rate = parse_kind(s)
    assert kind == "numeric" and rate == pytest.approx(0.97)
    assert to_numeric_lenient(s).isna().sum() == 3


def test_non_finite_values_become_missing():
    out = to_numeric_lenient(pd.Series([1.0, np.inf, -np.inf, 4.0]))
    assert out.isna().sum() == 2
    assert np.isfinite(out.dropna().to_numpy()).all()


def test_datetime_conversion_is_idempotent():
    dt = pd.Series(pd.to_datetime(["2026-01-01", "2026-03-01"]))
    once = to_numeric_lenient(dt)
    twice = to_numeric_lenient(once, "datetime")          # already epoch days
    pd.testing.assert_series_equal(once, twice)
    assert once.iloc[1] - once.iloc[0] == pytest.approx(59.0)


# --------------------------------------------------------------------------
# role inference
# --------------------------------------------------------------------------
def _roles_frame(n=200):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "amount": [f"${v:,.2f}" for v in rng.uniform(5, 900, n)],
            "booked": pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 90, n), unit="D"),
            "channel": rng.choice(["ecom", "atm", "pos"], n),
            "tier": rng.integers(1, 4, n),                     # integer-coded category
            "all_null": np.nan,
            "constant": "ONE",
            "case_ref": [f"CR-{i}" for i in range(n)],         # identifier
            "target": rng.binomial(1, 0.2, n),
        }
    )


def test_auto_inference_types_and_drops_correctly():
    cfg = Config.from_dict({"data": {"target": "target"}, "columns": {"auto_infer": True}})
    roles = infer_roles(_roles_frame(), cfg)
    assert "amount" in roles.numeric and "amount" in roles.recovered
    assert "booked" in roles.numeric and roles.kinds["booked"] == "datetime"
    assert "channel" in roles.categorical
    assert "tier" in roles.categorical                          # few integer levels
    assert roles.dropped == {"all_null": "all_missing", "constant": "constant",
                             "case_ref": "identifier_like_cardinality"}
    assert "target" not in roles.all


def test_explicitly_requested_columns_are_never_dropped():
    """A requested specification must come back the width it was requested."""
    cfg = Config.from_dict({"data": {"target": "target"}, "columns": {"auto_infer": True}})
    wanted = ["constant", "case_ref", "channel"]
    roles = infer_roles(_roles_frame(), cfg, features=wanted)
    assert sorted(roles.all) == sorted(wanted)
    assert roles.dropped == {}


def test_dirty_columns_survive_the_full_pipeline():
    df = _roles_frame()
    y = df.pop("target").to_numpy()
    cfg = Config.from_dict({"data": {"target": "target"}, "columns": {"auto_infer": True}})
    Xt = DisputeFeaturePipeline(config=cfg).fit_transform(df, y)
    assert Xt.isna().sum().sum() == 0
    assert np.isfinite(Xt.to_numpy(dtype=float)).all()
    assert any(c.endswith("amount") for c in Xt.columns)        # not one-hot exploded


# --------------------------------------------------------------------------
# target handling
# --------------------------------------------------------------------------
def test_string_labels_and_auto_positive_class():
    y = pd.Series(["GENUINE"] * 90 + ["FRAUD"] * 10, name="t")
    arr, label = _binarize_target(y, "auto")
    assert label == "FRAUD" and arr.sum() == 10

    arr, label = _binarize_target(y, "FRAUD")
    assert arr.sum() == 10


def test_unmatched_positive_label_fails_loudly():
    y = pd.Series(["GENUINE"] * 90 + ["FRAUD"] * 10, name="t")
    with pytest.raises(ValueError, match="does not occur"):
        _binarize_target(y, 1)
    with pytest.raises(ValueError, match="single class"):
        _binarize_target(pd.Series([0] * 20, name="t"), 1)
    with pytest.raises(ValueError, match="binary outcome"):
        _binarize_target(pd.Series([0, 1, 2] * 10, name="t"), 1)


# --------------------------------------------------------------------------
# split strategies
# --------------------------------------------------------------------------
def _split_frame(raw: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = raw.drop(columns=["dispute_id"]).copy()
    df["customer_id"] = rng.integers(0, 120, len(df))
    df["booked_at"] = pd.Timestamp("2026-01-01") + pd.to_timedelta(
        rng.integers(0, 150, len(df)), unit="D"
    )
    return df


def _split_cfg(strategy: str, tmp_path, **kw) -> Config:
    return Config.from_dict(
        {
            "run": {"name": f"split_{strategy}", "output_dir": str(tmp_path), "n_jobs": 1,
                    "verbose": 0, "save_fitted_model": False, "refit_on_full_data": False},
            "data": {"target": "is_fraudulent_dispute", "positive_label": "auto"},
            "columns": {"auto_infer": True, "drop": ["merchant_id"]},
            "split": {"holdout_size": 0.25, "strategy": strategy, "cv": {"n_splits": 3}, **kw},
            "selection": {"k_min": 1, "k_max": 2, "top_n": 1},
            "metrics": {"primary": "average_precision", "secondary": ["roc_auc"]},
            "models": {"logistic": {"estimator": "sklearn.linear_model.LogisticRegression",
                                    "requires_scaling": True, "imbalance": "balanced",
                                    "params": {"C": 1.0, "max_iter": 800}}},
        }
    )


def test_group_split_keeps_customers_on_one_side(raw, tmp_path):
    df = _split_frame(raw)
    res = ModelSelectionHarness(_split_cfg("group", tmp_path, group_column="customer_id")).run(df)
    sp = res.report.get("holdout_split")
    assert sp["strategy"] == "group"
    assert sp["n_groups_leaked"] == 0
    assert "customer_id" not in res.report.get("candidate_variables")["numeric"]


def test_time_split_trains_on_the_past_only(raw, tmp_path):
    df = _split_frame(raw)
    res = ModelSelectionHarness(_split_cfg("time", tmp_path, time_column="booked_at")).run(df)
    sp = res.report.get("holdout_split")
    assert sp["strategy"] == "time"
    assert sp["train_period"][1] <= sp["holdout_period"][0]
    assert "booked_at" not in res.report.get("candidate_variables")["numeric"]


def test_split_strategy_requires_its_key():
    with pytest.raises(ValueError, match="group_column"):
        Config.from_dict({"split": {"strategy": "group"}})
    with pytest.raises(ValueError, match="time_column"):
        Config.from_dict({"split": {"strategy": "time"}})


def test_tuning_inner_folds_differ_from_outer(raw, tmp_path):
    """Hyper-parameters must never be selected on the folds they are scored on."""
    df = _split_frame(raw)
    for strategy, kw in [("time", {"time_column": "booked_at"}),
                         ("group", {"group_column": "customer_id"})]:
        cfg = _split_cfg(strategy, tmp_path, **kw)
        cfg.tuning.cv_splits = 4                    # != split.cv.n_splits (3)
        h = ModelSelectionHarness(cfg)
        X = df.drop(columns=["is_fraudulent_dispute"]).head(600)
        y = df["is_fraudulent_dispute"].head(600).to_numpy()
        h.groups_tr_ = X["customer_id"].to_numpy() if strategy == "group" else None
        outer = [tuple(va) for _, va in h._cv().split(X, y, h.groups_tr_)]
        inner = [tuple(va) for _, va in
                 h._cv(n_splits=cfg.tuning.cv_splits, seed_offset=1).split(X, y, h.groups_tr_)]
        assert not set(outer) & set(inner), f"{strategy}: an inner fold reproduces an outer fold"
    # deterministic time splits with equal counts would coincide -> refused
    with pytest.raises(ValueError, match="tuning.cv_splits"):
        Config.from_dict({"split": {"strategy": "time", "time_column": "t",
                                    "cv": {"n_splits": 3}},
                          "tuning": {"enabled": True, "cv_splits": 3}})


# --------------------------------------------------------------------------
# provenance and monitoring
# --------------------------------------------------------------------------
def test_lineage_is_stable_and_discriminating(cfg, Xy):
    X, _ = Xy
    a = run_lineage(cfg.to_dict(), X)
    b = run_lineage(cfg.to_dict(), X)
    assert a == b                                       # deterministic
    other = cfg.copy()
    other.selection.k_max = 99
    assert run_lineage(other.to_dict(), X)["config_sha256"] != a["config_sha256"]
    assert run_lineage(cfg.to_dict(), X.head(50))["data_sha256"] != a["data_sha256"]
    assert "scikit-learn" in a["packages"]


def test_psi_detects_a_shift_and_ignores_noise():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    assert population_stability_index(ref, rng.normal(0, 1, 5000)) < 0.10
    shifted = population_stability_index(ref, rng.normal(0.6, 1.3, 5000))
    assert shifted > 0.25 and psi_band(shifted) == "investigate"
    assert psi_band(0.05) == "stable" and psi_band(0.15) == "watch"


# --------------------------------------------------------------------------
# post-hoc wrappers
# --------------------------------------------------------------------------
def test_scorer_reaches_the_guard_through_a_post_hoc_wrapper(cfg, Xy):
    """Threshold tuning / calibration nest the pipeline; the scorer must cope."""
    from sklearn.model_selection import TunedThresholdClassifierCV

    X, y = Xy
    spec = cfg.models["logistic"]
    pipe = build_model_pipeline(config_for_model(cfg, spec), ["prior_disputes_12m", "channel"],
                                build_estimator(spec, 0, 1, y))
    tuned = TunedThresholdClassifierCV(estimator=pipe, scoring="f1", cv=3).fit(X, y)

    scorer = ProductionScorer(tuned, features=["prior_disputes_12m", "channel"],
                              threshold=tuned.best_threshold_)
    scored, report = scorer.score(X.head(40))
    assert len(scored) == 40
    assert report["decision_cut"] == pytest.approx(tuned.best_threshold_)
    assert not scorer.training_envelope().empty
