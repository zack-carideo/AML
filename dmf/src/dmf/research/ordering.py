"""
Variable ordering strategies.

The harness evaluates *nested* subsets -- the top-1 variable, the top-2, ...,
the top-K -- so that "how much does the k-th variable buy me?" is a well-posed
question with a paired, fold-level answer. That requires an ordering, and how
you get the ordering materially changes the answer. Two are provided, both
selectable from the config:

``importance``
    Fit once on the full variable set, read off model-native importance
    (|standardised coefficient| for linear models, split gain for trees, or
    permutation importance), aggregate encoded columns back to source
    variables. Cost: one fit per model. Fast, and the ordering is stable, but
    it is a *marginal* ranking -- it ignores redundancy between variables.

``rfe``
    Recursive feature elimination on the encoded design matrix, eliminating
    the weakest column and refitting until one remains; the elimination order
    is the ranking. Cost: O(p / step) fits. Accounts for redundancy as the set
    shrinks, which importance ranking does not.

Both return a ranked list of *source* variables plus a quantitative
report, so the ordering step is auditable rather than a black box.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance

from ..config import Config
from ..metrics import make_scorers
from ..pipeline import DisputeFeaturePipeline, build_model_pipeline


# --------------------------------------------------------------------------
# encoded -> source aggregation
# --------------------------------------------------------------------------
def aggregate_to_source(
    values: np.ndarray,
    feature_names: List[str],
    source_map: Dict[str, str],
    how: str = "sum",
) -> pd.Series:
    """Collapse per-encoded-column statistics onto their source variables."""
    src = [source_map.get(n, n) for n in feature_names]
    s = pd.Series(np.asarray(values, dtype=float), index=src)
    grouped = s.groupby(level=0)
    return {"sum": grouped.sum(), "max": grouped.max(), "mean": grouped.mean(),
            "min": grouped.min()}[how]


def _raw_importance(estimator: Any, n_features: int) -> Optional[np.ndarray]:
    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        return np.abs(coef).reshape(-1, n_features).mean(axis=0)
    if hasattr(estimator, "feature_importances_"):
        return np.asarray(estimator.feature_importances_, dtype=float)
    return None


# --------------------------------------------------------------------------
# strategy 1: model-native / permutation importance
# --------------------------------------------------------------------------
def importance_ordering(
    X: pd.DataFrame,
    y: np.ndarray,
    cfg: Config,
    estimator: Any,
    candidates: List[str],
) -> Tuple[List[str], Dict[str, Any]]:
    method = cfg.selection.importance.method
    pipe = build_model_pipeline(cfg, candidates, estimator)
    pipe.fit(X, y)
    feats: DisputeFeaturePipeline = pipe.named_steps["features"]
    model = pipe.named_steps["model"]
    names = list(feats.feature_names_out_)

    used = method
    icfg = cfg.selection.importance
    raw = None if method == "permutation" else _raw_importance(model, len(names))
    if raw is not None and method in {"auto", "coef", "gain"}:
        is_coef = hasattr(model, "coef_")
        if is_coef and icfg.scale_by_std:
            # |beta_j| * sd(x_j): the effect on the log-odds of a one-SD move in
            # the encoded column. Puts a rare one-hot dummy (sd = sqrt(p(1-p)))
            # on the same footing as a standardised numeric (sd = 1), which raw
            # |beta| does not.
            Xt = feats.transform(X.loc[:, candidates])
            sd = np.nan_to_num(np.asarray(Xt, dtype=float).std(axis=0), nan=0.0)
            raw = raw * sd
        scores = aggregate_to_source(raw, names, feats.feature_source_map_, how=icfg.aggregate)
        used = ("coef_x_sd" if (is_coef and icfg.scale_by_std) else "coef") if is_coef else "gain"
    else:
        # permutation importance on the *raw* columns: no aggregation needed and
        # it is model-agnostic, at the cost of n_repeats extra scoring passes.
        scorer = make_scorers(cfg.metrics)[cfg.metrics.primary]
        r = permutation_importance(
            pipe,
            X.loc[:, candidates],
            y,
            scoring=scorer,
            n_repeats=cfg.selection.importance.permutation_repeats,
            random_state=cfg.run.random_state,
            n_jobs=cfg.run.n_jobs,
        )
        scores = pd.Series(r.importances_mean, index=candidates)
        used = "permutation"

    scores = scores.reindex(candidates).fillna(0.0)
    total = float(np.abs(scores).sum())
    normalized = (np.abs(scores) / total) if total > 0 else scores * 0.0
    ordered = list(normalized.sort_values(ascending=False, kind="mergesort").index)

    report = {
        "strategy": "importance",
        "method": used,
        "n_candidates": len(candidates),
        "n_encoded_columns": len(names),
        "importance_share": {k: round(float(v), 6) for k, v in normalized.sort_values(ascending=False).items()},
        "cumulative_share_top5": round(float(normalized.sort_values(ascending=False).head(5).sum()), 6),
        "n_zero_importance": int((normalized == 0).sum()),
        "ordering": ordered,
    }
    return ordered, report


# --------------------------------------------------------------------------
# strategy 2: recursive feature elimination
# --------------------------------------------------------------------------
def rfe_ordering(
    X: pd.DataFrame,
    y: np.ndarray,
    cfg: Config,
    estimator: Any,
    candidates: List[str],
) -> Tuple[List[str], Dict[str, Any]]:
    feats = DisputeFeaturePipeline(config=cfg, features=candidates)
    Xt = feats.fit_transform(X, y)
    names = list(feats.feature_names_out_)

    est = clone(estimator)
    selector = RFE(estimator=est, n_features_to_select=1, step=cfg.selection.rfe.step)
    try:
        # RFE needs a per-column importance signal, which is only observable
        # after a fit -- so probe by fitting rather than by inspecting the
        # unfitted class, and degrade gracefully instead of aborting a sweep.
        selector.fit(Xt, y)
    except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
        ordered, rep = importance_ordering(X, y, cfg, estimator, candidates)
        rep["strategy"] = "rfe->importance_fallback"
        rep["fallback_reason"] = f"RFE unusable with {type(est).__name__}: {exc}"
        return ordered, rep

    ranking = np.asarray(selector.ranking_, dtype=float)   # 1 = last survivor = strongest
    how = "min" if cfg.selection.rfe.aggregate == "min_rank" else "mean"
    src_rank = aggregate_to_source(ranking, names, feats.feature_source_map_, how=how)

    final_imp = _raw_importance(selector.estimator_, 1)
    tiebreak = pd.Series(0.0, index=src_rank.index)
    if final_imp is not None and len(final_imp):
        surviving = [n for n, keep in zip(names, selector.support_) if keep]
        if len(surviving) == len(final_imp):
            agg = aggregate_to_source(np.abs(final_imp), surviving, feats.feature_source_map_, how="sum")
            tiebreak = tiebreak.add(agg, fill_value=0.0).reindex(src_rank.index).fillna(0.0)

    frame = pd.DataFrame({"rank": src_rank, "tiebreak": tiebreak}).reindex(candidates)
    frame["rank"] = frame["rank"].fillna(frame["rank"].max() + 1)
    frame["tiebreak"] = frame["tiebreak"].fillna(0.0)
    ordered = list(frame.sort_values(["rank", "tiebreak"], ascending=[True, False], kind="mergesort").index)

    report = {
        "strategy": "rfe",
        "step": cfg.selection.rfe.step,
        "aggregate": cfg.selection.rfe.aggregate,
        "n_candidates": len(candidates),
        "n_encoded_columns": len(names),
        "n_elimination_rounds": int(ranking.max()),
        "source_rank": {k: round(float(v), 4) for k, v in frame["rank"].sort_values().items()},
        "ordering": ordered,
    }
    return ordered, report


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------
def rank_variables(
    X: pd.DataFrame,
    y: np.ndarray,
    cfg: Config,
    estimator: Any,
    candidates: List[str],
) -> Tuple[List[str], Dict[str, Any]]:
    """Produce the nested-subset ordering using the configured strategy."""
    strategy = cfg.selection.ordering_strategy
    if strategy == "importance":
        return importance_ordering(X, y, cfg, estimator, candidates)
    if strategy == "rfe":
        return rfe_ordering(X, y, cfg, estimator, candidates)
    raise ValueError(f"Unknown ordering strategy '{strategy}'.")  # pragma: no cover


__all__ = ["rank_variables", "importance_ordering", "rfe_ordering", "aggregate_to_source"]
