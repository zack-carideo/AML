"""
Metric registry for imbalanced binary classification.

Primary metric is **average precision (PR-AUC)**: with a low-prevalence
fraudulent-dispute target, ROC-AUC is dominated by the enormous true-negative
mass and moves very little between materially different models, whereas AP
tracks precision across the whole recall range and is the quantity an
investigations queue actually experiences.

Everything here is expressed both as sklearn scorers (for ``cross_validate``)
and as a direct evaluation function (for holdout reporting), so the CV grid and
the final holdout report are computed by the same code.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    make_scorer,
    roc_auc_score,
    roc_curve,
)


# --------------------------------------------------------------------------
# metric functions (y_true, y_score) -> float
# --------------------------------------------------------------------------
def ks_statistic(y_true, y_score) -> float:
    """Kolmogorov-Smirnov separation: max(TPR - FPR) over all thresholds."""
    y_true = np.asarray(y_true).ravel()
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, np.asarray(y_score).ravel())
    return float(np.max(tpr - fpr))


def recall_at_fpr(y_true, y_score, max_fpr: float = 0.01) -> float:
    """Detection rate at a fixed false-positive rate (analyst-capacity proxy)."""
    y_true = np.asarray(y_true).ravel()
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, np.asarray(y_score).ravel())
    return float(np.interp(max_fpr, fpr, tpr))


def lift_at_top_pct(y_true, y_score, top_pct: float = 0.05) -> float:
    """Precision within the top-scoring ``top_pct`` of volume, over prevalence.

    A lift of 6.0 at 5% means the review queue built from the model's top 5% of
    disputes contains 6x the fraud density of a random 5%.
    """
    y_true = np.asarray(y_true).ravel().astype(float)
    y_score = np.asarray(y_score).ravel()
    n = len(y_true)
    k = max(int(np.ceil(top_pct * n)), 1)
    prevalence = y_true.mean()
    if prevalence <= 0:
        return float("nan")
    idx = np.argsort(-y_score, kind="mergesort")[:k]
    return float(y_true[idx].mean() / prevalence)


def expected_calibration_error(y_true, y_score, n_bins: int = 10) -> float:
    """Bin-weighted |observed - predicted| across score bins."""
    y_true = np.asarray(y_true).ravel().astype(float)
    y_score = np.asarray(y_score).ravel()
    edges = np.quantile(y_score, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return float(abs(y_true.mean() - y_score.mean()))
    bins = np.clip(np.digitize(y_score, edges[1:-1], right=True), 0, len(edges) - 2)
    total = 0.0
    for b in np.unique(bins):
        m = bins == b
        total += m.mean() * abs(y_true[m].mean() - y_score[m].mean())
    return float(total)


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------
class MetricDef:
    def __init__(self, fn: Callable[..., float], greater_is_better: bool, needs: Optional[str] = None):
        self.fn = fn
        self.greater_is_better = greater_is_better
        self.needs = needs  # config attribute supplying the operating point


METRIC_REGISTRY: Dict[str, MetricDef] = {
    "average_precision": MetricDef(average_precision_score, True),
    "roc_auc": MetricDef(roc_auc_score, True),
    "ks_statistic": MetricDef(ks_statistic, True),
    "recall_at_fpr": MetricDef(recall_at_fpr, True, needs="recall_at_fpr"),
    "lift_at_top_pct": MetricDef(lift_at_top_pct, True, needs="lift_top_pct"),
    "brier_score": MetricDef(brier_score_loss, False),
    "log_loss": MetricDef(log_loss, False),
    "calibration_error": MetricDef(expected_calibration_error, False),
}


def _kwargs_for(name: str, metrics_cfg: Any) -> Dict[str, Any]:
    spec = METRIC_REGISTRY[name]
    if spec.needs is None:
        return {}
    value = getattr(metrics_cfg, spec.needs)
    key = {"recall_at_fpr": "max_fpr", "lift_top_pct": "top_pct"}[spec.needs]
    return {key: value}


def make_scorers(metrics_cfg: Any) -> Dict[str, Any]:
    """Build the ``scoring`` dict for ``cross_validate``.

    All scorers consume ``predict_proba[:, 1]``. Loss-type metrics are wrapped
    with ``greater_is_better=False``, so sklearn returns them negated; the
    harness un-negates them for display via :func:`orient`.
    """
    names: List[str] = [metrics_cfg.primary] + [m for m in metrics_cfg.secondary if m != metrics_cfg.primary]
    scorers: Dict[str, Any] = {}
    for name in names:
        if name not in METRIC_REGISTRY:
            raise ValueError(f"Unknown metric '{name}'. Available: {sorted(METRIC_REGISTRY)}")
        spec = METRIC_REGISTRY[name]
        scorers[name] = make_scorer(
            spec.fn,
            response_method="predict_proba",
            greater_is_better=spec.greater_is_better,
            **_kwargs_for(name, metrics_cfg),
        )
    return scorers


def score_vector(y_true, y_score, metrics_cfg: Any, signed: bool = False) -> Dict[str, float]:
    """Every configured metric computed from one probability vector.

    ``signed=True`` reproduces the sklearn scorer convention (loss-type metrics
    negated), so values are interchangeable with what ``make_scorers`` produces
    inside ``cross_validate`` -- the harness computes ``predict_proba`` once and
    derives all metrics from it instead of re-predicting once per scorer.
    A metric that cannot be computed on a fold is missing, not zero.
    """
    names = [metrics_cfg.primary] + [m for m in metrics_cfg.secondary if m != metrics_cfg.primary]
    out: Dict[str, float] = {}
    for name in names:
        spec = METRIC_REGISTRY[name]
        try:
            v = float(spec.fn(y_true, y_score, **_kwargs_for(name, metrics_cfg)))
        except Exception:
            v = float("nan")
        if signed and not spec.greater_is_better and np.isfinite(v):
            v = -v
        out[name] = v
    return out


def orient(name: str, value: float) -> float:
    """Convert a signed sklearn CV score back to its natural units."""
    if name in METRIC_REGISTRY and not METRIC_REGISTRY[name].greater_is_better:
        return -value
    return value


def greater_is_better(name: str) -> bool:
    return METRIC_REGISTRY[name].greater_is_better if name in METRIC_REGISTRY else True


# --------------------------------------------------------------------------
# direct evaluation (holdout reporting)
# --------------------------------------------------------------------------
def evaluate_predictions(y_true, y_score, metrics_cfg: Any) -> Dict[str, float]:
    """All configured metrics in natural units, plus calibration diagnostics."""
    names = [metrics_cfg.primary] + [m for m in metrics_cfg.secondary if m != metrics_cfg.primary]
    out: Dict[str, float] = {}
    for name in names:
        spec = METRIC_REGISTRY[name]
        try:
            out[name] = float(spec.fn(y_true, y_score, **_kwargs_for(name, metrics_cfg)))
        except Exception:
            out[name] = float("nan")
    y_true = np.asarray(y_true).ravel().astype(float)
    y_score = np.asarray(y_score).ravel()
    out["prevalence"] = float(y_true.mean())
    out["mean_predicted"] = float(y_score.mean())
    out["calibration_ratio"] = float(y_score.mean() / y_true.mean()) if y_true.mean() else float("nan")
    out["calibration_error"] = expected_calibration_error(y_true, y_score)
    return {k: (None if isinstance(v, float) and not np.isfinite(v) else round(float(v), 6))
            for k, v in out.items()}


def decile_table(y_true, y_score, n_bins: int = 10) -> pd.DataFrame:
    """Score-band gains table -- the standard artifact for a review queue."""
    y_true = np.asarray(y_true).ravel().astype(float)
    y_score = np.asarray(y_score).ravel()
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted, s_sorted = y_true[order], y_score[order]
    # never ask for more bands than there are records, or np.array_split yields
    # empty bands and every per-band statistic raises
    n_bins = max(min(n_bins, len(y_sorted)), 1)
    bands = [b for b in np.array_split(np.arange(len(y_sorted)), n_bins) if len(b)]
    prevalence = y_true.mean()
    rows = []
    cum_pos = 0
    total_pos = y_true.sum()
    for i, idx in enumerate(bands, start=1):
        pos = float(y_sorted[idx].sum())
        cum_pos += pos
        rows.append(
            {
                "band": i,
                "n": int(len(idx)),
                "score_min": round(float(s_sorted[idx].min()), 6),
                "score_max": round(float(s_sorted[idx].max()), 6),
                "n_positive": int(pos),
                "positive_rate": round(pos / len(idx), 6),
                "lift": round((pos / len(idx)) / prevalence, 4) if prevalence else None,
                "cumulative_capture": round(cum_pos / total_pos, 6) if total_pos else None,
            }
        )
    return pd.DataFrame(rows)


def population_stability_index(
    expected: np.ndarray, actual: np.ndarray, n_bins: int = 10, epsilon: float = 1e-6
) -> float:
    """PSI between a reference and a current score (or feature) distribution.

    .. math:: PSI = \\sum_b (a_b - e_b)\\,\\ln(a_b / e_b)

    Bins are the reference distribution's quantiles, so the reference is
    uniform by construction and all movement is attributable to the current
    batch. Conventional reading: < 0.10 stable, 0.10-0.25 watch, > 0.25
    investigate. Pair it with the guard's ``training_envelope`` -- the guard
    catches values outside the support, PSI catches a *shift within* it.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if expected.size == 0 or actual.size == 0:
        return float("nan")

    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, bins=edges)[0] / expected.size
    a = np.histogram(actual, bins=edges)[0] / actual.size
    e, a = np.clip(e, epsilon, None), np.clip(a, epsilon, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_band(psi: float) -> str:
    if not np.isfinite(psi):
        return "unknown"
    return "stable" if psi < 0.10 else ("watch" if psi < 0.25 else "investigate")


__all__ = [
    "METRIC_REGISTRY", "make_scorers", "orient", "greater_is_better",
    "evaluate_predictions", "score_vector", "decile_table",
    "population_stability_index", "psi_band",
    "ks_statistic", "recall_at_fpr", "lift_at_top_pct", "expected_calibration_error",
]
