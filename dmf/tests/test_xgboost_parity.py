"""Independent XGBoost parity check for the selection harness.

The harness wraps a lot of machinery around a fit: role inference, a
winsorize/impute/encode pipeline refit inside every fold, nested variable
re-ranking, a one-SE selection rule, a derived decision threshold. Every one
of those layers is a place where the reported numbers could drift away from
what a plain model on the same variables would show.

This module builds that plain model **outside** the framework -- raw
``xgboost.XGBClassifier`` on the harness's selected variables, with the least
preprocessing XGBoost tolerates (native NaN handling, integer codes for
categoricals, nothing else) -- and compares:

1. the holdout / CV partitions the harness used (must be reproducible exactly
   from ``sklearn`` primitives and the config seed);
2. the harness's own holdout scores (must be reproducible exactly by fitting
   its feature pipeline + estimator by hand -- no leakage, no row shuffling);
3. the selected variables (must agree with a plain gain ranking);
4. the accuracy metrics on the holdout and in CV (plain model within a
   tolerance -- and never *worse* than the harness by more than that
   tolerance, which is the signature of leakage);
5. the rank ordering of holdout scores (Spearman, decile capture, top-decile
   overlap);
6. the selected cutoff: its derivation from stored scores, and the operating
   point it implies against the same operating point on the plain model.

Tolerances are deliberately tight enough that a leaked column, a mis-aligned
row id, or a threshold computed on the wrong partition would fail.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split

from dmf import Config
from dmf.metrics import decile_table, ks_statistic, lift_at_top_pct, recall_at_fpr
from dmf.pipeline import DisputeFeaturePipeline
from dmf.research import ModelSelectionHarness, load_fold_assignments, load_predictions

xgb = pytest.importorskip("xgboost")

# --------------------------------------------------------------------------
# experiment definition
# --------------------------------------------------------------------------
SEED = 2026
N_ROWS = 5000
PREVALENCE = 0.12
HOLDOUT = 0.25
N_SPLITS = 3
TOP_PCT = 0.05
FPR_POINT = 0.01

NUMERIC = [
    "dispute_amount", "amount_to_daily_spend_ratio", "days_txn_to_dispute",
    "cardholder_tenure_months", "prior_disputes_12m", "prior_dispute_upheld_ratio",
    "night_txn_share_30d", "card_present_share_30d", "failed_logins_7d",
    "device_changes_90d", "geo_distance_km", "internal_risk_score",
    "noise_gaussian", "noise_uniform",
]
CATEGORICAL = ["merchant_category", "channel", "dispute_reason_code", "claim_channel"]
#: variables the synthetic generator actually wires into the latent risk
SIGNAL = {
    "amount_to_daily_spend_ratio", "days_txn_to_dispute", "cardholder_tenure_months",
    "prior_disputes_12m", "prior_dispute_upheld_ratio", "night_txn_share_30d",
    "card_present_share_30d", "failed_logins_7d", "device_changes_90d",
    "geo_distance_km", "internal_risk_score", "merchant_category", "channel",
    "dispute_reason_code", "claim_channel",
}
NOISE = {"noise_gaussian", "noise_uniform"}

XGB_PARAMS = dict(
    n_estimators=150, max_depth=3, learning_rate=0.08, subsample=0.9,
    colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.5,
    tree_method="hist", n_jobs=1,
)

# tolerances for "plain model vs harness" on the *same* rows. Two different
# preprocessings of the same variables can legitimately differ by this much;
# leakage or a row mix-up moves AP by 0.1+ on this data.
TOL_AP = 0.05
TOL_AUC = 0.03
TOL_RATE = 0.10          # precision / recall at a fixed operating point
MIN_SPEARMAN = 0.90
MIN_TOP_OVERLAP = 0.60   # Jaccard of the two models' top-decile row sets


def _config(out_dir: Path) -> Config:
    return Config.from_dict({
        "run": {"name": "xgb_parity", "output_dir": str(out_dir), "random_state": SEED,
                "n_jobs": 1, "verbose": 0, "save_fitted_model": True,
                "refit_on_full_data": False, "save_predictions": "cv"},
        "data": {"target": "is_fraudulent_dispute", "id_column": "dispute_id"},
        "columns": {"numeric": NUMERIC, "categorical": CATEGORICAL, "auto_infer": False},
        "split": {"holdout_size": HOLDOUT, "stratify": True,
                  "cv": {"n_splits": N_SPLITS, "shuffle": True}},
        "metrics": {"primary": "average_precision",
                    "secondary": ["roc_auc", "ks_statistic", "recall_at_fpr", "lift_at_top_pct"],
                    "recall_at_fpr": FPR_POINT, "lift_top_pct": TOP_PCT,
                    "decision_threshold_policy": "top_pct"},
        "selection": {"k_min": 1, "k_max": 8, "top_n": 1, "ordering_strategy": "importance",
                      "one_se_rule": True},
        "models": {
            "xgboost": {
                "estimator": "xgboost.XGBClassifier", "family": "tree", "tag": "challenger",
                "requires_scaling": False, "imbalance": "scale_pos_weight",
                "preprocessing_overrides": {"categorical.encoder": "ordinal"},
                "params": dict(XGB_PARAMS),
            },
        },
    })


# --------------------------------------------------------------------------
# the plain model: nothing from dmf touches the design matrix
# --------------------------------------------------------------------------
def plain_design(train: pd.DataFrame, other: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Minimal XGBoost-ready matrices: floats with NaN, categoricals as codes.

    Category codes are fixed on the training partition; anything unseen (or
    missing) becomes -1. No winsorizing, no imputation, no rare-level
    collapsing -- deliberately the least the model can be given.
    """
    def enc(df: pd.DataFrame, cats: Dict[str, pd.Index]) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for c in features:
            if c in cats:
                out[c] = pd.Categorical(df[c].astype(object), categories=cats[c]).codes.astype(float)
            else:
                out[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
        return out

    cats = {c: pd.Index(train[c].dropna().astype(object).unique()) for c in features if c in CATEGORICAL}
    return enc(train, cats), enc(other, cats)


def plain_xgb(y_train: np.ndarray, seed: int = SEED) -> "xgb.XGBClassifier":
    pos = float(y_train.sum())
    return xgb.XGBClassifier(**XGB_PARAMS, random_state=seed, verbosity=0,
                             scale_pos_weight=(len(y_train) - pos) / pos)


def _cut_stats(y: np.ndarray, s: np.ndarray, thr: float) -> Dict[str, float]:
    flag = s >= thr
    tp = int(((y == 1) & flag).sum())
    return {"threshold": float(thr), "flag_rate": float(flag.mean()),
            "precision": tp / max(int(flag.sum()), 1), "recall": tp / max(int(y.sum()), 1),
            "n_flagged": int(flag.sum())}


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a, b = set(a), set(b)
    return len(a & b) / max(len(a | b), 1)


# --------------------------------------------------------------------------
# fixtures: one harness run + one plain fit, shared by every assertion
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    from generate_synthetic_disputes import generate_disputes
    return generate_disputes(n=N_ROWS, seed=SEED, prevalence=PREVALENCE)


@pytest.fixture(scope="module")
def harness(data, tmp_path_factory):
    out = tmp_path_factory.mktemp("xgb_parity")
    cfg = _config(out)
    X = data.drop(columns=["is_fraudulent_dispute"]).copy()
    y = data["is_fraudulent_dispute"].to_numpy()
    res = ModelSelectionHarness(cfg).run(X, y)
    run_dir = out / cfg.run.name
    preds, meta = load_predictions(run_dir)
    return {"cfg": cfg, "res": res, "run_dir": run_dir, "preds": preds, "meta": meta}


@pytest.fixture(scope="module")
def split(data):
    """The holdout partition rebuilt from sklearn primitives and the seed only."""
    y = data["is_fraudulent_dispute"].to_numpy()
    tr, ho = train_test_split(np.arange(len(data)), test_size=HOLDOUT, random_state=SEED,
                              stratify=y, shuffle=True)
    return {"tr": tr, "ho": ho, "y": y,
            "ids_tr": data["dispute_id"].to_numpy()[tr], "ids_ho": data["dispute_id"].to_numpy()[ho]}


@pytest.fixture(scope="module")
def plain(data, split, harness):
    """Plain XGBoost on the harness's selected variables, holdout-scored."""
    feats = harness["res"].selected_features
    X_tr, X_ho = plain_design(data.iloc[split["tr"]], data.iloc[split["ho"]], feats)
    y_tr, y_ho = split["y"][split["tr"]], split["y"][split["ho"]]
    model = plain_xgb(y_tr).fit(X_tr, y_tr)
    return {"features": feats, "score_ho": model.predict_proba(X_ho)[:, 1], "y_ho": y_ho,
            "X_tr": X_tr, "y_tr": y_tr}


@pytest.fixture(scope="module")
def harness_holdout(harness, split) -> Tuple[np.ndarray, np.ndarray]:
    """Harness holdout scores aligned to the rebuilt holdout order."""
    ho = harness["preds"][harness["preds"]["stage"] == "holdout"].set_index("row_id")
    ho = ho.loc[split["ids_ho"]]
    return ho["y_true"].to_numpy(), ho["y_score"].to_numpy(dtype=float)


# --------------------------------------------------------------------------
# 1. partitions
# --------------------------------------------------------------------------
def test_holdout_partition_is_the_plain_stratified_split(harness, split):
    ho_ids = harness["preds"].loc[harness["preds"]["stage"] == "holdout", "row_id"].to_numpy()
    assert len(ho_ids) == len(split["ids_ho"])
    assert set(ho_ids) == set(split["ids_ho"])
    # and the labels stored beside the scores are the real labels for those rows
    y_true, _ = harness_holdout_from(harness, split)
    assert np.array_equal(y_true, split["y"][split["ho"]])


def harness_holdout_from(harness, split):
    ho = harness["preds"][harness["preds"]["stage"] == "holdout"].set_index("row_id").loc[split["ids_ho"]]
    return ho["y_true"].to_numpy(), ho["y_score"].to_numpy(dtype=float)


def test_cv_folds_are_the_plain_stratified_kfold(harness, split, data):
    folds = load_fold_assignments(harness["run_dir"])
    assert folds["row_id"].is_unique
    assert set(folds["row_id"]) == set(split["ids_tr"]), "CV folds must partition the training rows only"

    X_tr = data.iloc[split["tr"]]
    y_tr = split["y"][split["tr"]]
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    expected = {}
    for f, (_, va) in enumerate(skf.split(X_tr, y_tr)):
        expected[f] = set(split["ids_tr"][va])
    got = folds.groupby("fold")["row_id"].apply(set).to_dict()
    assert got == expected


# --------------------------------------------------------------------------
# 2. the harness's own numbers reproduce by hand
# --------------------------------------------------------------------------
def test_configured_policy_actually_reaches_the_estimator(harness, split):
    """The estimator the harness fitted must carry what the config declared.

    This is what first exposed a silent drop: ``XGBClassifier.__init__`` is
    ``(self, *, objective, **kwargs)``, so a signature probe on the subclass
    alone reports no ``scale_pos_weight`` / ``random_state`` / ``n_jobs`` and
    the imbalance policy and seed never reached XGBoost -- the scores were
    ~4x lower than a by-hand fit of the same specification.
    """
    est = harness["res"].fitted_model.named_steps["model"]
    assert isinstance(est, xgb.XGBClassifier)
    y_tr = split["y"][split["tr"]]
    pos = float(y_tr.sum())
    p = est.get_params()
    assert p["scale_pos_weight"] == pytest.approx((len(y_tr) - pos) / pos, abs=1e-5)
    assert p["random_state"] == SEED
    assert p["n_jobs"] == 1
    for k, v in XGB_PARAMS.items():
        assert p[k] == v, f"{k}: configured {v}, estimator has {p[k]}"


def test_harness_holdout_scores_reproduce_from_a_manual_fit(harness, split, data, harness_holdout):
    """Fit the harness's feature pipeline + estimator by hand on the training
    partition only. Identical scores prove nothing about the holdout leaked
    into the fit and no rows were re-ordered between fit and store."""
    cfg, res = harness["cfg"], harness["res"]
    feats = res.selected_features
    X_tr, X_ho = data.iloc[split["tr"]], data.iloc[split["ho"]]
    y_tr = split["y"][split["tr"]]

    # same per-model config view the harness used (scaler off, ordinal encoder)
    model_cfg = cfg.copy()
    model_cfg.preprocessing.numeric.scaler = "none"
    model_cfg.preprocessing.categorical.encoder = "ordinal"
    model_cfg.preprocessing.inference_guard.warn = False
    fp = DisputeFeaturePipeline(config=model_cfg, features=feats)
    Xt_tr = fp.fit_transform(X_tr, y_tr)
    Xt_ho = fp.transform(X_ho)

    pos = float(y_tr.sum())
    est = xgb.XGBClassifier(**XGB_PARAMS, random_state=SEED, verbosity=0,
                            scale_pos_weight=round((len(y_tr) - pos) / pos, 6))
    est.fit(Xt_tr, y_tr)
    manual = est.predict_proba(Xt_ho)[:, 1]

    _, stored = harness_holdout
    np.testing.assert_allclose(stored, manual, atol=1e-6)
    # the reported holdout AP is that vector's AP, nothing else
    assert res.holdout_metrics["average_precision"] == pytest.approx(
        average_precision_score(split["y"][split["ho"]], manual), abs=1e-6)


def test_cv_leaderboard_cell_reproduces_from_stored_scores(harness):
    res, preds = harness["res"], harness["preds"]
    k = res.selected["k"]
    cv = preds[(preds["stage"] == "cv") & (preds["k"] == k)]
    per_fold = [average_precision_score(g["y_true"], g["y_score"]) for _, g in cv.groupby("fold")]
    assert len(per_fold) == N_SPLITS
    assert res.selected["cv_average_precision_mean"] == pytest.approx(np.mean(per_fold), abs=2e-6)


# --------------------------------------------------------------------------
# 3. selected variables
# --------------------------------------------------------------------------
def test_selected_variables_agree_with_a_plain_gain_ranking(harness, split, data):
    res = harness["res"]
    sel = res.selected_features
    k = len(sel)
    assert k == res.selected["k"] >= 1

    # plain XGBoost on every candidate, train partition only, gain importance
    cands = NUMERIC + CATEGORICAL
    X_tr, _ = plain_design(data.iloc[split["tr"]], data.iloc[split["ho"]], cands)
    y_tr = split["y"][split["tr"]]
    model = plain_xgb(y_tr).fit(X_tr, y_tr)
    gain = pd.Series(model.get_booster().get_score(importance_type="gain")).reindex(cands).fillna(0.0)
    plain_top = list(gain.sort_values(ascending=False).index[:k])

    overlap = _jaccard(np.array(sel), np.array(plain_top))
    print(f"\nselected (k={k}): {sel}\nplain gain top-{k}: {plain_top}\njaccard={overlap:.2f}")
    assert overlap >= 0.5, f"harness picked {sel}, plain gain ranking says {plain_top}"
    # the parsimonious choice must be built from real signal
    assert not (set(sel) & NOISE), f"noise variables selected: {set(sel) & NOISE}"
    assert set(sel) <= SIGNAL


# --------------------------------------------------------------------------
# 4. accuracy metrics
# --------------------------------------------------------------------------
def test_holdout_metrics_match_plain_model(harness, plain, harness_holdout):
    y, s_h = harness_holdout
    s_p = plain["score_ho"]
    assert np.array_equal(y, plain["y_ho"])

    rows = []
    for name, fn in [("average_precision", average_precision_score),
                     ("roc_auc", roc_auc_score),
                     ("ks_statistic", ks_statistic),
                     ("recall_at_fpr", lambda a, b: recall_at_fpr(a, b, FPR_POINT)),
                     ("lift_at_top_pct", lambda a, b: lift_at_top_pct(a, b, TOP_PCT))]:
        h, p = float(fn(y, s_h)), float(fn(y, s_p))
        rows.append({"metric": name, "harness": h, "plain": p,
                     "reported": harness["res"].holdout_metrics.get(name), "delta": h - p})
    table = pd.DataFrame(rows)
    print("\n" + table.round(4).to_string(index=False))

    m = table.set_index("metric")
    # the reported numbers are the stored-score numbers
    for name in m.index:
        assert m.loc[name, "reported"] == pytest.approx(m.loc[name, "harness"], abs=1e-5)
    assert abs(m.loc["average_precision", "delta"]) <= TOL_AP
    assert abs(m.loc["roc_auc", "delta"]) <= TOL_AUC
    assert abs(m.loc["ks_statistic", "delta"]) <= TOL_AUC * 2
    assert abs(m.loc["recall_at_fpr", "delta"]) <= TOL_RATE
    # leakage shows up as the harness beating a plain fit of the same model
    # on the same variables by more than preprocessing can explain
    assert m.loc["average_precision", "delta"] <= TOL_AP


def test_cv_estimate_matches_plain_cross_validation(harness, plain, split, data):
    """Plain XGBoost, same fixed variables, same folds -> mean AP within tolerance
    of the harness's leaderboard cell for the selected specification."""
    res = harness["res"]
    X_tr, y_tr = plain["X_tr"], plain["y_tr"]
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    aps = []
    for tr, va in skf.split(X_tr, y_tr):
        m = plain_xgb(y_tr[tr]).fit(X_tr.iloc[tr], y_tr[tr])
        aps.append(average_precision_score(y_tr[va], m.predict_proba(X_tr.iloc[va])[:, 1]))
    plain_cv = float(np.mean(aps))
    harness_cv = float(res.selected["cv_average_precision_mean"])
    print(f"\ncv AP  harness(nested)={harness_cv:.4f}  plain(fixed vars)={plain_cv:.4f}")
    # the nested estimate re-ranks inside each fold, so it is allowed to sit a
    # little *below* a fixed-variable CV; it must not sit above it
    assert harness_cv <= plain_cv + TOL_AP
    assert abs(harness_cv - plain_cv) <= TOL_AP
    # and the holdout must corroborate the CV estimate
    assert abs(res.holdout_metrics["average_precision"] - harness_cv) <= TOL_AP


# --------------------------------------------------------------------------
# 5. rank ordering
# --------------------------------------------------------------------------
def test_rank_ordering_matches_plain_model(harness, plain, harness_holdout):
    y, s_h = harness_holdout
    s_p = plain["score_ho"]

    rho = float(stats.spearmanr(s_h, s_p).statistic)
    n_top = int(np.ceil(0.10 * len(y)))
    top_h = np.argsort(-s_h, kind="mergesort")[:n_top]
    top_p = np.argsort(-s_p, kind="mergesort")[:n_top]
    top_overlap = _jaccard(top_h, top_p)

    dec_h = decile_table(y, s_h)
    dec_p = decile_table(y, s_p)
    cmp = pd.DataFrame({"band": dec_h["band"],
                        "lift_harness": dec_h["lift"], "lift_plain": dec_p["lift"],
                        "capture_harness": dec_h["cumulative_capture"],
                        "capture_plain": dec_p["cumulative_capture"]})
    print(f"\nspearman={rho:.4f}  top-decile jaccard={top_overlap:.2f}\n" + cmp.to_string(index=False))

    assert rho >= MIN_SPEARMAN
    assert top_overlap >= MIN_TOP_OVERLAP
    # cumulative capture curves track each other band by band
    assert (np.abs(cmp["capture_harness"] - cmp["capture_plain"]) <= 0.06).all()
    # both orderings are genuinely rank-ordering: lift falls from top to bottom
    assert dec_h["lift"].iloc[0] > dec_h["lift"].iloc[-1]
    assert dec_h["lift"].iloc[0] >= 0.8 * dec_p["lift"].iloc[0]
    # the reported decile table is the stored-score decile table
    pd.testing.assert_frame_equal(harness["res"].holdout_deciles.reset_index(drop=True), dec_h)


# --------------------------------------------------------------------------
# 6. the selected cutoff
# --------------------------------------------------------------------------
def test_cutoff_derivation_and_operating_point(harness, plain, harness_holdout):
    res, meta = harness["res"], harness["meta"]
    y, s_h = harness_holdout
    s_p = plain["score_ho"]

    # derivation: the cut is the top-5% quantile of the *holdout* scores of the
    # train-only fit, and the same number reaches the bundle and the sidecar
    thr_h = float(res.holdout_metrics["decision_threshold"])
    assert thr_h == pytest.approx(float(np.quantile(s_h, 1 - TOP_PCT)), abs=1e-6)
    assert meta["decision_threshold"] == pytest.approx(thr_h, abs=1e-9)
    from dmf.inference import ProductionScorer
    assert ProductionScorer.from_joblib(harness["run_dir"] / "model.joblib").threshold == pytest.approx(thr_h, abs=1e-9)

    # the fpr-policy alternative is also derived from the same vector
    fpr, _, thr = roc_curve(y, s_h)
    thr = np.where(np.isfinite(thr), thr, s_h.max())
    assert res.holdout_metrics["implied_threshold_at_fpr"] == pytest.approx(
        float(np.interp(FPR_POINT, fpr, thr)), abs=1e-6)

    # operating point: apply the same *policy* to the plain model and compare
    # what the analyst queue would look like
    thr_p = float(np.quantile(s_p, 1 - TOP_PCT))
    op_h, op_p = _cut_stats(y, s_h, thr_h), _cut_stats(y, s_p, thr_p)
    flagged_h = np.flatnonzero(s_h >= thr_h)
    flagged_p = np.flatnonzero(s_p >= thr_p)
    table = pd.DataFrame([{"model": "harness", **op_h}, {"model": "plain", **op_p}])
    print("\n" + table.to_string(index=False) + f"\nflagged-set jaccard={_jaccard(flagged_h, flagged_p):.2f}")

    assert res.holdout_metrics["decision_flag_rate"] == pytest.approx(op_h["flag_rate"], abs=1e-6)
    assert res.holdout_metrics["decision_precision"] == pytest.approx(op_h["precision"], abs=1e-6)
    assert res.holdout_metrics["decision_recall"] == pytest.approx(op_h["recall"], abs=1e-6)
    # the cut flags the configured share of volume (within one row's worth)
    for op in (op_h, op_p):
        assert abs(op["flag_rate"] - TOP_PCT) <= 1.5 / len(y)
    assert abs(op_h["precision"] - op_p["precision"]) <= TOL_RATE
    assert abs(op_h["recall"] - op_p["recall"]) <= TOL_RATE
    assert _jaccard(flagged_h, flagged_p) >= 0.5
