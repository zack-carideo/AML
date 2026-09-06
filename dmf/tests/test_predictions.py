"""The prediction store: row-level (id, y_true, y_score) for post-run evaluation.

The property under test throughout: every aggregate the harness reports must be
recomputable from the stored rows alone -- no refit, no raw data.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from conftest import make_cfg
from dmf import Config
from dmf.inference import ProductionScorer
from dmf.research import (
    ModelSelectionHarness,
    compute_metrics,
    load_fold_assignments,
    load_predictions,
    operating_point_table,
    threshold_at_fpr,
)


@pytest.fixture(scope="module")
def full_run(raw, tmp_path_factory):
    """One harness run at the most verbose capture level, shared by the tests."""
    cfg = make_cfg()
    cfg.run.output_dir = str(tmp_path_factory.mktemp("pred_store"))
    cfg.run.save_predictions = "all"
    cfg.run.save_fitted_model = True
    cfg.data.id_column = "dispute_id"
    X = raw.drop(columns=["is_fraudulent_dispute"]).copy()
    y = raw["is_fraudulent_dispute"].to_numpy()
    res = ModelSelectionHarness(cfg).run(X, y)
    out = Path(cfg.run.output_dir) / cfg.run.name
    return cfg, res, out


# --------------------------------------------------------------------------
# artifacts and schema
# --------------------------------------------------------------------------
def test_store_files_stages_and_meta(full_run, raw):
    cfg, res, out = full_run
    preds, meta = load_predictions(out)

    assert set(preds["stage"].unique()) == {"cv", "cv_train", "holdout"}
    assert list(preds.columns) == ["row_id", "y_true", "y_score", "stage",
                                   "model", "k", "fold", "repeat"]
    assert preds["y_score"].between(0, 1).all()
    assert set(preds["y_true"].unique()) <= {0, 1}

    folds = load_fold_assignments(out)
    assert set(folds["fold"].unique()) == {0, 1, 2}
    # fold membership is a partition of the training rows
    assert folds["row_id"].is_unique
    assert len(folds) + int((preds["stage"] == "holdout").sum() / 1) <= len(raw)

    assert meta["row_id_source"] == "data.id_column:dispute_id"
    assert meta["positive_label"] in {"1", "True"}
    assert meta["save_predictions"] == "all"
    assert meta["decision_threshold"] is not None
    assert meta["rows_per_stage"]["holdout"] == int((preds["stage"] == "holdout").sum())

    # the in-memory result carries the same store
    assert res.predictions is not None and len(res.predictions) == len(preds)
    assert res.fold_assignments is not None and len(res.fold_assignments) == len(folds)


def test_row_ids_are_real_ids_and_never_features(full_run, raw):
    cfg, res, out = full_run
    preds, _ = load_predictions(out)
    assert set(preds["row_id"]).issubset(set(raw["dispute_id"]))
    # the identifier must never leak into the candidate variables
    assert not res.leaderboard["features"].str.contains("dispute_id").any()


# --------------------------------------------------------------------------
# reproducibility: stored rows regenerate reported aggregates
# --------------------------------------------------------------------------
def test_holdout_metrics_reproduce_from_store(full_run):
    cfg, res, out = full_run
    preds, meta = load_predictions(out)
    ho = preds[preds["stage"] == "holdout"]

    ap = average_precision_score(ho["y_true"], ho["y_score"])
    assert res.holdout_metrics["average_precision"] == pytest.approx(ap, abs=1e-6)

    table = compute_metrics(ho, meta, by=["stage"])
    assert table.loc[0, "average_precision"] == pytest.approx(
        res.holdout_metrics["average_precision"], abs=1e-6)
    assert table.loc[0, "roc_auc"] == pytest.approx(res.holdout_metrics["roc_auc"], abs=1e-6)


def test_cv_leaderboard_reproduces_from_store(full_run):
    cfg, res, out = full_run
    preds, _ = load_predictions(out)
    cv = preds[preds["stage"] == "cv"]
    for _, row in res.leaderboard.iterrows():
        sub = cv[(cv["model"] == row["model"]) & (cv["k"] == row["k"])]
        per_fold = sub.groupby(["repeat", "fold"]).apply(
            lambda g: average_precision_score(g["y_true"], g["y_score"]),
            include_groups=False,
        )
        assert len(per_fold) == cfg.split.cv.n_splits
        assert float(per_fold.mean()) == pytest.approx(
            row["cv_average_precision_mean"], abs=2e-6)


# --------------------------------------------------------------------------
# thresholds
# --------------------------------------------------------------------------
def test_decision_threshold_derivation(full_run):
    cfg, res, out = full_run
    preds, _ = load_predictions(out)
    ho = preds[preds["stage"] == "holdout"]

    expected = float(np.quantile(ho["y_score"], 1 - cfg.metrics.lift_top_pct))
    assert res.holdout_metrics["decision_threshold_policy"] == "top_pct"
    assert res.holdout_metrics["decision_threshold"] == pytest.approx(expected, abs=1e-6)
    assert res.holdout_metrics["implied_threshold_at_fpr"] is not None
    # the flag rate at the derived cut is the configured review budget
    assert res.holdout_metrics["decision_flag_rate"] == pytest.approx(
        cfg.metrics.lift_top_pct, abs=0.02)


def test_production_scorer_picks_up_bundle_threshold(full_run):
    cfg, res, out = full_run
    scorer = ProductionScorer.from_joblib(out / "model.joblib")
    assert scorer.threshold == pytest.approx(
        res.holdout_metrics["decision_threshold"], abs=1e-9)


def test_threshold_helpers(full_run):
    cfg, res, out = full_run
    preds, _ = load_predictions(out)
    ho = preds[preds["stage"] == "holdout"]

    taf = threshold_at_fpr(ho, max_fpr=0.05, by=["model"])
    assert len(taf) == 1
    assert 0.0 <= taf.loc[0, "threshold"] <= 1.0
    assert 0.0 <= taf.loc[0, "recall"] <= 1.0

    opt = operating_point_table(ho, by=["model"]).sort_values("threshold")
    assert (opt["tp"] + opt["fn"] == int(ho["y_true"].sum())).all()
    assert (opt["flag_rate"].diff().dropna() <= 1e-12).all()   # higher cut -> fewer flags
    assert (opt["fpr"].diff().dropna() <= 1e-12).all()

    # per-fold implied thresholds from the CV stage: one row per fold
    cv = preds[(preds["stage"] == "cv") & (preds["model"] == res.selected_model)
               & (preds["k"] == res.selected["k"])]
    per_fold = threshold_at_fpr(cv, max_fpr=0.05, by=["fold"])
    assert len(per_fold) == cfg.split.cv.n_splits


# --------------------------------------------------------------------------
# capture levels
# --------------------------------------------------------------------------
def _slim_cfg(tmp_path, level=None) -> Config:
    cfg = make_cfg()
    cfg.run.output_dir = str(tmp_path)
    cfg.models.pop("trees")
    cfg.selection.k_max = 2
    cfg.selection.top_n = 1
    if level is not None:
        cfg.run.save_predictions = level
    return cfg


def test_default_level_stores_holdout_only(raw, tmp_path):
    cfg = _slim_cfg(tmp_path)
    assert cfg.run.save_predictions == "holdout"      # the default
    X = raw.drop(columns=["is_fraudulent_dispute", "dispute_id"]).copy()
    y = raw["is_fraudulent_dispute"].to_numpy()
    res = ModelSelectionHarness(cfg).run(X, y)
    preds, meta = load_predictions(Path(cfg.run.output_dir) / cfg.run.name)
    assert set(preds["stage"].unique()) == {"holdout"}
    assert meta["row_id_source"] == "dataframe_index"
    # fold membership is still recorded at this level
    assert len(load_fold_assignments(Path(cfg.run.output_dir) / cfg.run.name))
    # and the bundle threshold is derived regardless of capture level
    assert res.holdout_metrics["decision_threshold"] is not None


def test_none_level_stores_nothing(raw, tmp_path):
    cfg = _slim_cfg(tmp_path, level="none")
    X = raw.drop(columns=["is_fraudulent_dispute", "dispute_id"]).copy()
    y = raw["is_fraudulent_dispute"].to_numpy()
    res = ModelSelectionHarness(cfg).run(X, y)
    assert res.predictions is None
    with pytest.raises(FileNotFoundError):
        load_predictions(Path(cfg.run.output_dir) / cfg.run.name)


def test_config_validation():
    with pytest.raises(ValueError, match="save_predictions"):
        Config.from_dict({"run": {"save_predictions": "sometimes"}})
    with pytest.raises(ValueError, match="decision_threshold_policy"):
        Config.from_dict({"metrics": {"decision_threshold_policy": "vibes"}})
    with pytest.raises(KeyError, match="id_column"):
        cfg = make_cfg()
        cfg.data.id_column = "no_such_column"
        ModelSelectionHarness(cfg).run(
            pd.DataFrame({"a": [1, 2, 3, 4], "t": [0, 1, 0, 1]}).drop(columns=[]),
            np.array([0, 1, 0, 1]),
        )
