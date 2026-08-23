"""The guard must absorb inference-time surprises without failing or extrapolating."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from dmf import DisputeFeaturePipeline, ProductionScorer, build_model_pipeline
from dmf.research.zoo import build_estimator, config_for_model


def _fitted(cfg, X, y, **overrides):
    c = cfg.copy()
    for k, v in overrides.items():
        setattr(c.preprocessing.inference_guard, k, v)
    return DisputeFeaturePipeline(config=c).fit(X, y)


def test_unseen_category_does_not_raise_and_is_flagged(cfg, Xy):
    X, y = Xy
    fp = _fitted(cfg, X, y)
    new = X.head(20).copy()
    new.loc[new.index[:5], "channel"] = "crypto_offramp_never_seen"

    Xt, report, _ = fp.transform_with_quality(new)
    assert len(Xt) == 20 and Xt.isna().sum().sum() == 0
    assert report["n_rows_unseen_category"] == 5
    assert report["by_column"]["channel"]["issue"] == "unseen_category_level"
    assert "__UNSEEN__" in report["by_column"]["channel"]["action"]


def test_out_of_range_numeric_is_clipped_to_the_training_envelope(cfg, Xy):
    X, y = Xy
    fp = _fitted(cfg, X, y)
    envelope = fp.guard.numeric_bounds_["dispute_amount"]

    new = X.head(10).copy()
    new.loc[new.index[0], "dispute_amount"] = 10_000_000.0
    new.loc[new.index[1], "dispute_amount"] = -10_000_000.0

    Xt, report, _ = fp.transform_with_quality(new)
    assert report["by_column"]["dispute_amount"]["action"] == "clipped_to_envelope"
    assert report["n_rows_out_of_range"] == 2
    # the guarded value must sit exactly on the training boundary, not beyond it
    guarded = fp.pipeline_.named_steps["guard"].transform(new)["dispute_amount"]
    assert guarded.iloc[0] == pytest.approx(envelope["train_max"])
    assert guarded.iloc[1] == pytest.approx(envelope["train_min"])
    assert np.isfinite(Xt.to_numpy(dtype=float)).all()


def test_text_in_a_numeric_column_becomes_missing_not_an_exception(cfg, Xy):
    X, y = Xy
    fp = _fitted(cfg, X, y)
    new = X.head(10).copy()
    new["internal_risk_score"] = new["internal_risk_score"].astype(object)
    new.loc[new.index[:3], "internal_risk_score"] = "N/A"

    Xt, report, _ = fp.transform_with_quality(new)
    assert report["n_rows_coerced"] == 3
    assert Xt.isna().sum().sum() == 0        # the fitted imputer supplied the default


def test_absent_column_is_filled_and_escalated_at_batch_level(cfg, Xy):
    X, y = Xy
    fp = _fitted(cfg, X, y)
    new = X.head(15).drop(columns=["prior_disputes_12m"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Xt, report, _ = fp.transform_with_quality(new)

    assert report["missing_columns"] == ["prior_disputes_12m"]
    assert report["batch_safe"] is False
    assert report["verdict"] == "review_recommended"
    # batch-level, so it must not swamp the per-row signal
    assert report["n_rows_flagged"] == 0
    assert Xt.isna().sum().sum() == 0


def test_clean_batch_is_reported_as_safe(cfg, Xy):
    X, y = Xy
    fp = _fitted(cfg, X, y)
    _, report, flags = fp.transform_with_quality(X.head(100))
    assert report["batch_safe"] is True
    assert report["verdict"] == "ok"
    assert report["n_rows_flagged"] == 0


def test_error_policies_raise_when_explicitly_requested(cfg, Xy):
    X, y = Xy
    new = X.head(5).copy()
    new.loc[new.index[0], "channel"] = "brand_new_level"

    strict = _fitted(cfg, X, y, unseen_category_policy="error")
    with pytest.raises(ValueError, match="unseen level"):
        strict.transform(new)

    hot = X.head(5).copy()
    hot.loc[hot.index[0], "dispute_amount"] = 1e9
    strict_num = _fitted(cfg, X, y, numeric_policy="error")
    with pytest.raises(ValueError, match="training envelope"):
        strict_num.transform(hot)


def test_nan_policy_routes_through_the_imputer(cfg, Xy):
    X, y = Xy
    fp = _fitted(cfg, X, y, numeric_policy="nan")
    new = X.head(5).copy()
    new.loc[new.index[0], "dispute_amount"] = 1e9
    Xt, report, _ = fp.transform_with_quality(new)
    assert report["by_column"]["dispute_amount"]["action"] == "set_missing_then_imputed"
    assert Xt.isna().sum().sum() == 0


def test_tolerance_widens_the_envelope(cfg, Xy):
    X, y = Xy
    tight = _fitted(cfg, X, y, numeric_tolerance=0.0)
    loose = _fitted(cfg, X, y, numeric_tolerance=0.5)
    a = tight.guard.numeric_bounds_["dispute_amount"]
    b = loose.guard.numeric_bounds_["dispute_amount"]
    assert b["guard_max"] > a["guard_max"] and b["guard_min"] < a["guard_min"]


def test_scores_stay_finite_and_bounded_on_a_fully_drifted_batch(cfg, raw, drifted):
    X = raw.drop(columns=["dispute_id"]).copy()
    y = X.pop("is_fraudulent_dispute").to_numpy()
    spec = cfg.models["logistic"]
    pipe = build_model_pipeline(config_for_model(cfg, spec), cfg.declared_features,
                                build_estimator(spec, 0, 1, y))
    pipe.fit(X, y)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scorer = ProductionScorer(pipe, top_pct=0.05)
        scored, report = scorer.score(drifted)

    assert len(scored) == len(drifted)
    p = scored["fraud_probability"].to_numpy()
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    assert report["n_rows_flagged"] > 0
    assert (scored.loc[scored["data_quality"] == "guarded", "action"] == "manual_review").all()
    assert not scorer.explain_guard(drifted).empty


def test_scorer_roundtrips_through_joblib(tmp_path, cfg, Xy):
    import joblib

    X, y = Xy
    spec = cfg.models["logistic"]
    pipe = build_model_pipeline(config_for_model(cfg, spec), ["prior_disputes_12m", "channel"],
                                build_estimator(spec, 0, 1, y))
    pipe.fit(X, y)
    path = tmp_path / "model.joblib"
    joblib.dump({"pipeline": pipe, "features": ["prior_disputes_12m", "channel"],
                 "model": "logistic", "config": cfg.to_dict()}, path)

    scorer = ProductionScorer.from_joblib(path)
    assert scorer.features == ["prior_disputes_12m", "channel"]
    assert scorer.metadata["model"] == "logistic"

    scored, report = scorer.score(X.head(50))
    assert len(scored) == 50
    assert {"fraud_probability", "decision", "action", "data_quality"} <= set(scored.columns)
    assert report["schema"]["schema_ok"] is True

    # a single record has no batch to take a quantile of, so a relative cut is
    # refused rather than silently flagging everything
    with pytest.raises(ValueError, match="absolute threshold"):
        scorer.score_one(X.head(1).iloc[0].to_dict())

    fixed = ProductionScorer.from_joblib(path, threshold=0.8)
    one = fixed.score_one(X.head(1).iloc[0].to_dict())
    assert 0.0 <= one["fraud_probability"] <= 1.0
    assert one["decision"] in {"flag_fraud", "pass"}
