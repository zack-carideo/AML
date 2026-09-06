"""Budget-dependent metrics swept across a list of operating points.

``recall_at_fpr`` and ``lift_at_top_pct`` are curves read at a chosen budget.
The config accepts a list of budgets, and each one becomes its own reported
metric named ``metric@budget`` -- through the CV grid, the leaderboard, the
holdout report and the prediction store alike.

The properties under test: a scalar config is byte-for-byte what it always
was, a list produces one correctly-valued column per budget, and everything
that can only take one number (the shipped decision threshold, the slice-table
cut, the leaderboard's ranking column) still resolves to exactly one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import make_cfg
from dmf import Config
from dmf.metrics import (
    evaluate_predictions,
    lift_at_top_pct,
    make_scorers,
    metric_names,
    operating_point,
    recall_at_fpr,
    resolve_metrics,
    score_vector,
    split_metric_name,
)
from dmf.research import ModelSelectionHarness, compute_metrics, load_predictions

FPRS = [0.01, 0.02, 0.05]
PCTS = [0.02, 0.05, 0.10]


@pytest.fixture()
def scores():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.1, 2000)
    s = np.clip(0.35 * y + rng.beta(2, 8, 2000), 0, 1)
    return y, s


def _metrics(**over) -> Config:
    cfg = make_cfg()
    for key, value in over.items():
        setattr(cfg.metrics, key, value)
    cfg.validate()
    return cfg.metrics


# --------------------------------------------------------------------------
# naming and resolution
# --------------------------------------------------------------------------
def test_scalar_config_keeps_the_bare_metric_name():
    m = _metrics(secondary=["roc_auc", "recall_at_fpr", "lift_at_top_pct"], recall_at_fpr=0.01)
    names = metric_names(m)
    assert "recall_at_fpr" in names and "lift_at_top_pct" in names
    assert not any("@" in n for n in names)
    assert names[0] == m.primary            # primary always leads


def test_list_config_expands_to_one_metric_per_budget():
    m = _metrics(secondary=["roc_auc", "recall_at_fpr", "lift_at_top_pct"],
                 recall_at_fpr=FPRS, lift_top_pct=PCTS)
    names = metric_names(m)
    assert [n for n in names if n.startswith("recall_at_fpr")] == [f"recall_at_fpr@{v:g}" for v in FPRS]
    assert [n for n in names if n.startswith("lift_at_top_pct")] == [f"lift_at_top_pct@{v:g}" for v in PCTS]
    # unparametrised metrics are untouched, and nothing is duplicated
    assert "roc_auc" in names and len(names) == len(set(names))


def test_explicit_operating_point_in_the_name_wins_over_the_config():
    m = _metrics(secondary=["recall_at_fpr@0.005"], recall_at_fpr=FPRS)
    names = metric_names(m)
    assert "recall_at_fpr@0.005" in names
    # the bare name was never requested, so the configured list does not expand
    assert [n for n in names if n.startswith("recall_at_fpr")] == ["recall_at_fpr@0.005"]
    assert split_metric_name("recall_at_fpr@0.005") == ("recall_at_fpr", 0.005)
    assert split_metric_name("roc_auc") == ("roc_auc", None)


def test_operating_point_helper_takes_the_first_listed_value():
    assert operating_point(_metrics(recall_at_fpr=FPRS), "recall_at_fpr") == FPRS[0]
    assert operating_point(_metrics(recall_at_fpr=0.03), "recall_at_fpr") == 0.03


# --------------------------------------------------------------------------
# values
# --------------------------------------------------------------------------
def test_each_column_is_the_metric_at_its_own_budget(scores):
    y, s = scores
    m = _metrics(secondary=["roc_auc", "recall_at_fpr", "lift_at_top_pct"],
                 recall_at_fpr=FPRS, lift_top_pct=PCTS)
    out = evaluate_predictions(y, s, m)

    for v in FPRS:
        assert out[f"recall_at_fpr@{v:g}"] == pytest.approx(recall_at_fpr(y, s, v), abs=1e-6)
    for v in PCTS:
        assert out[f"lift_at_top_pct@{v:g}"] == pytest.approx(lift_at_top_pct(y, s, v), abs=1e-6)

    # the budgets are genuinely different points on one curve: a looser
    # false-positive budget can only ever detect at least as much
    got = [out[f"recall_at_fpr@{v:g}"] for v in FPRS]
    assert got == sorted(got), f"recall must be non-decreasing in the FPR budget, got {got}"
    assert len(set(got)) > 1, "the budgets should not all collapse to one value"


def test_score_vector_and_scorers_agree_with_the_report(scores):
    y, s = scores
    m = _metrics(secondary=["roc_auc", "recall_at_fpr"], recall_at_fpr=FPRS)
    names = metric_names(m)
    # every surface is built from the same resolved list, so the key sets match
    assert list(score_vector(y, s, m)) == names
    assert list(make_scorers(m)) == names
    assert [k for k in evaluate_predictions(y, s, m) if k in names] == names
    # all of these are gains, so signed and natural units coincide
    assert score_vector(y, s, m, signed=True) == pytest.approx(score_vector(y, s, m))


# --------------------------------------------------------------------------
# config validation
# --------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [[], [0.0], [1.5], [-0.01], [0.01, 0.01], ["0.01"], 0, 1.2])
def test_invalid_operating_points_are_rejected(bad):
    with pytest.raises(ValueError, match="recall_at_fpr"):
        Config.from_dict({"metrics": {"recall_at_fpr": bad}})


def test_ambiguous_primary_is_a_config_error():
    with pytest.raises(ValueError, match="no single column to rank"):
        Config.from_dict({"metrics": {"primary": "recall_at_fpr", "recall_at_fpr": FPRS}})
    # naming the point resolves it
    cfg = Config.from_dict({"metrics": {"primary": "recall_at_fpr@0.02", "recall_at_fpr": FPRS}})
    assert metric_names(cfg.metrics)[0] == "recall_at_fpr@0.02"


def test_unknown_and_over_specified_metric_names_are_rejected():
    with pytest.raises(ValueError, match="Unknown metric"):
        Config.from_dict({"metrics": {"primary": "not_a_metric"}})
    with pytest.raises(ValueError, match="no operating point"):
        Config.from_dict({"metrics": {"secondary": ["roc_auc@0.1"]}})


# --------------------------------------------------------------------------
# end to end through the harness
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def swept(raw, tmp_path_factory):
    cfg = make_cfg()
    cfg.run.output_dir = str(tmp_path_factory.mktemp("op_points"))
    cfg.run.save_predictions = "holdout"
    cfg.metrics.secondary = ["roc_auc", "recall_at_fpr", "lift_at_top_pct"]
    cfg.metrics.recall_at_fpr = FPRS
    cfg.metrics.lift_top_pct = PCTS
    cfg.validate()
    X = raw.drop(columns=["is_fraudulent_dispute"]).copy()
    y = raw["is_fraudulent_dispute"].to_numpy()
    res = ModelSelectionHarness(cfg).run(X, y)
    return cfg, res, Path(cfg.run.output_dir) / cfg.run.name


def test_leaderboard_and_holdout_carry_every_budget(swept):
    cfg, res, _ = swept
    for v in FPRS:
        name = f"recall_at_fpr@{v:g}"
        assert res.holdout_metrics[name] is not None
        for suffix in ("mean", "std", "se"):
            assert f"cv_{name}_{suffix}" in res.leaderboard.columns
    for v in PCTS:
        assert res.holdout_metrics[f"lift_at_top_pct@{v:g}"] is not None
    # ranking still happens on the single primary column
    assert cfg.metrics.primary == "average_precision"
    assert res.leaderboard["cv_average_precision_mean"].is_monotonic_decreasing


def test_holdout_values_recompute_from_the_prediction_store(swept):
    cfg, res, out = swept
    preds, meta = load_predictions(out)
    ho = preds[preds["stage"] == "holdout"]
    table = compute_metrics(ho, meta, by=["stage"])

    for v in FPRS:
        name = f"recall_at_fpr@{v:g}"
        direct = recall_at_fpr(ho["y_true"], ho["y_score"], v)
        assert res.holdout_metrics[name] == pytest.approx(direct, abs=1e-6)
        # the store round-trips the list through predictions_meta.json
        assert table.loc[0, name] == pytest.approx(direct, abs=1e-6)


def test_one_threshold_is_shipped_and_it_names_its_budget(swept):
    cfg, res, out = swept
    preds, meta = load_predictions(out)
    s = preds.loc[preds["stage"] == "holdout", "y_score"].to_numpy(dtype=float)

    # the shipped cut uses the first listed budget, and says so
    assert res.holdout_metrics["decision_operating_point"] == PCTS[0]
    assert res.holdout_metrics["decision_threshold"] == pytest.approx(
        float(np.quantile(s, 1 - PCTS[0])), abs=1e-6)
    assert res.holdout_metrics["decision_flag_rate"] == pytest.approx(PCTS[0], abs=0.02)
    assert meta["decision_threshold"] == pytest.approx(
        res.holdout_metrics["decision_threshold"], abs=1e-9)
    # still exactly one threshold, not one per budget
    assert isinstance(res.holdout_metrics["decision_threshold"], float)


def test_slice_table_uses_a_single_cut(swept):
    cfg, res, _ = swept
    assert res.holdout_slices is not None and len(res.holdout_slices)
    assert res.holdout_slices["flag_rate_at_top_pct"].between(0, 1).all()
