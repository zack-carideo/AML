"""Configuration: round-tripping, validation, and typo detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from dmf import Config


def test_defaults_are_valid():
    cfg = Config()
    cfg.validate()
    assert cfg.metrics.primary == "average_precision"
    assert cfg.selection.ordering_strategy == "importance"
    assert cfg.preprocessing.inference_guard.enabled is True
    assert cfg.tuning.enabled is False          # tuning must be opt-in


def test_yaml_roundtrip(tmp_path, cfg):
    path = tmp_path / "cfg.yaml"
    cfg.to_yaml(path)
    reloaded = Config.from_yaml(path)
    assert reloaded.to_dict() == cfg.to_dict()


def test_copy_is_deep(cfg):
    other = cfg.copy()
    other.preprocessing.numeric.scaler = "robust"
    other.models["logistic"].params["C"] = 99
    assert cfg.preprocessing.numeric.scaler == "standard"
    assert cfg.models["logistic"].params["C"] == 1.0


def test_unknown_key_raises():
    with pytest.raises(ValueError, match="Unknown configuration key"):
        Config.from_dict({"selection": {"k_maxx": 5}})
    with pytest.raises(ValueError, match="Unknown configuration key"):
        Config.from_dict({"models": {"m": {"estimatorr": "x"}}})


@pytest.mark.parametrize(
    "patch,match",
    [
        ({"split": {"holdout_size": 1.5}}, "holdout_size"),
        ({"split": {"cv": {"n_splits": 1}}}, "n_splits"),
        ({"selection": {"k_min": 0}}, "k_min"),
        ({"selection": {"k_min": 5, "k_max": 2}}, "k_max"),
        ({"selection": {"ordering_strategy": "magic"}}, "ordering_strategy"),
        ({"preprocessing": {"categorical": {"encoder": "hashing"}}}, "encoder"),
        ({"preprocessing": {"inference_guard": {"numeric_policy": "yolo"}}}, "numeric_policy"),
        ({"preprocessing": {"numeric": {"winsorize": {"lower_quantile": 0.9,
                                                      "upper_quantile": 0.1}}}}, "winsorize"),
    ],
)
def test_validation_rejects_bad_values(patch, match):
    with pytest.raises(ValueError, match=match):
        Config.from_dict(patch)


def test_ordering_reference_model_must_exist():
    with pytest.raises(ValueError, match="ordering_reference_model"):
        Config.from_dict({
            "selection": {"ordering_reference_model": "nope"},
            "models": {"logistic": {"estimator": "sklearn.linear_model.LogisticRegression"}},
        })


def test_shipped_config_is_loadable():
    path = Path(__file__).resolve().parents[1] / "configs" / "dispute_fraud.yaml"
    cfg = Config.from_yaml(path)
    assert cfg.data.target == "is_fraudulent_dispute"
    assert "logistic_l2" in cfg.models and "xgboost" in cfg.models
    assert cfg.models["random_forest"].enabled is False
    assert len(cfg.declared_features) == 33
