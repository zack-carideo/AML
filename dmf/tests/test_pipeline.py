"""Core feature-pipeline contract: sklearn compliance, reporting, no leakage."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score

from dmf import DisputeFeaturePipeline, build_model_pipeline
from dmf.research.zoo import build_estimator, config_for_model


def test_fit_transform_shapes_and_names(cfg, Xy):
    X, y = Xy
    fp = DisputeFeaturePipeline(config=cfg).fit(X, y)
    Xt = fp.transform(X)
    assert isinstance(Xt, pd.DataFrame)
    assert len(Xt) == len(X)
    assert list(Xt.columns) == list(fp.get_feature_names_out())
    assert Xt.isna().sum().sum() == 0
    assert Xt.shape[1] >= len(cfg.declared_features)


def test_feature_subsetting_is_respected(cfg, Xy):
    X, y = Xy
    subset = ["prior_disputes_12m", "channel"]
    fp = DisputeFeaturePipeline(config=cfg, features=subset).fit(X, y)
    assert set(fp.input_features_) == set(subset)
    assert set(fp.feature_source_map_.values()) <= set(subset)


def test_source_map_covers_every_output_column(cfg, Xy):
    X, y = Xy
    fp = DisputeFeaturePipeline(config=cfg).fit(X, y)
    assert set(fp.feature_source_map_) == set(fp.feature_names_out_)
    assert set(fp.feature_source_map_.values()) <= set(fp.input_features_)


def test_clone_and_pickle_roundtrip(cfg, Xy):
    X, y = Xy
    fp = DisputeFeaturePipeline(config=cfg).fit(X, y)
    expected = fp.transform(X)

    restored = pickle.loads(pickle.dumps(fp))
    pd.testing.assert_frame_equal(restored.transform(X), expected)

    fresh = clone(fp)                       # unfitted, same hyper-parameters
    assert not hasattr(fresh, "pipeline_")
    pd.testing.assert_frame_equal(fresh.fit(X, y).transform(X), expected)


def test_transform_is_deterministic_and_row_order_invariant(cfg, Xy):
    X, y = Xy
    fp = DisputeFeaturePipeline(config=cfg).fit(X, y)
    a = fp.transform(X)
    b = fp.transform(X.iloc[::-1]).iloc[::-1]
    pd.testing.assert_frame_equal(a, b)


def test_fit_report_is_quantitative_and_complete(cfg, Xy):
    X, y = Xy
    fp = DisputeFeaturePipeline(config=cfg).fit(X, y)
    steps = {s["step"] for s in fp.fit_report_["steps"]}
    for expected in {"input", "target", "role_resolution", "select", "guard", "output", "expansion"}:
        assert expected in steps
    out = fp.report_.get("output")
    assert out["n_rows"] == len(X) and out["n_columns"] == len(fp.feature_names_out_)
    exp = fp.report_.get("expansion")
    assert exp["n_output_columns"] >= exp["n_input_variables"]
    assert isinstance(fp.summary(), pd.DataFrame) and len(fp.summary()) == len(fp.report_)


@pytest.mark.parametrize("encoder", ["onehot", "ordinal", "woe", "target"])
def test_all_categorical_encoders_fit_and_transform(cfg, Xy, encoder):
    X, y = Xy
    c = cfg.copy()
    c.preprocessing.categorical.encoder = encoder
    fp = DisputeFeaturePipeline(config=c).fit(X, y)
    Xt = fp.transform(X)
    assert Xt.shape[0] == len(X) and Xt.isna().sum().sum() == 0
    assert np.isfinite(Xt.to_numpy(dtype=float)).all()
    if encoder == "woe":
        iv = fp.information_value()
        assert iv is not None and (iv >= 0).all()


def test_supervised_encoders_do_not_leak_across_folds(cfg, Xy):
    """A pure-noise target must not be learnable, even with WOE/target encoding.

    If the encoders were fitted on the full sample rather than inside each fold,
    the high-cardinality merchant id would memorise the shuffled target and the
    cross-validated AUC would sit well above 0.5.
    """
    X, _ = Xy
    rng = np.random.default_rng(0)
    y_noise = rng.binomial(1, 0.15, len(X))

    for encoder in ("woe", "target"):
        c = cfg.copy()
        c.preprocessing.categorical.encoder = encoder
        c.preprocessing.categorical.rare_level.enabled = False
        spec = c.models["logistic"]
        pipe = build_model_pipeline(config_for_model(c, spec), c.declared_features,
                                    build_estimator(spec, 0, 1, y_noise))
        auc = cross_val_score(
            pipe, X, y_noise, scoring="roc_auc",
            cv=StratifiedKFold(3, shuffle=True, random_state=0),
        ).mean()
        assert 0.40 < auc < 0.60, f"{encoder} encoder leaks: cv AUC={auc:.3f}"


def test_scaler_disabled_for_models_that_do_not_need_it(cfg):
    linear = config_for_model(cfg, cfg.models["logistic"])
    trees = config_for_model(cfg, cfg.models["trees"])
    assert linear.preprocessing.numeric.scaler == "standard"
    assert trees.preprocessing.numeric.scaler == "none"
    # the override must not mutate the source config
    assert cfg.preprocessing.numeric.scaler == "standard"


def test_missing_values_are_imputed_and_indicated(cfg, Xy):
    X, y = Xy
    X = X.copy()
    X.loc[X.index[:50], "dispute_amount"] = np.nan
    fp = DisputeFeaturePipeline(config=cfg).fit(X, y)
    Xt = fp.transform(X)
    assert Xt.isna().sum().sum() == 0
    assert any("missingindicator" in c for c in Xt.columns)
