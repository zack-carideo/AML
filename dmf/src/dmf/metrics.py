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

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

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


# --------------------------------------------------------------------------
# operating points: one reported metric per configured value
# --------------------------------------------------------------------------
# ``recall_at_fpr`` and ``lift_at_top_pct`` are not single numbers -- they are
# curves read at a chosen budget, and the choice of budget is exactly the kind
# of decision a review wants to see varied rather than asserted. So the config
# accepts either a scalar or a list of budgets, and a list produces one
# reported metric per value, named ``metric@value``.
#
# A scalar is left un-suffixed, so every existing config, artifact and column
# name is unchanged. The suffix appears only where a list asked for it.

#: separates a metric from the operating point it is evaluated at
OPERATING_POINT_SEP = "@"

#: config attribute -> the metric-function keyword it supplies
_NEEDS_KWARG = {"recall_at_fpr": "max_fpr", "lift_top_pct": "top_pct"}


@dataclass(frozen=True)
class ResolvedMetric:
    """One column of the metric report: its name and how to compute it."""

    name: str                                  # reported name, e.g. recall_at_fpr@0.005
    base: str                                  # registry key, e.g. recall_at_fpr
    kwargs: Dict[str, Any] = field(default_factory=dict)

    @property
    def spec(self) -> MetricDef:
        return METRIC_REGISTRY[self.base]

    @property
    def greater_is_better(self) -> bool:
        return self.spec.greater_is_better

    def compute(self, y_true, y_score) -> float:
        return float(self.spec.fn(y_true, y_score, **self.kwargs))


def format_operating_point(value: float) -> str:
    """Render an operating point for use inside a metric name."""
    return f"{float(value):g}"


def split_metric_name(name: str) -> Tuple[str, Optional[float]]:
    """``'recall_at_fpr@0.01'`` -> ``('recall_at_fpr', 0.01)``; bare names -> ``(name, None)``."""
    base, sep, point = str(name).partition(OPERATING_POINT_SEP)
    if not sep:
        return base, None
    try:
        return base, float(point)
    except ValueError:
        raise ValueError(
            f"Metric '{name}' has a non-numeric operating point '{point}'. "
            f"Expected e.g. 'recall_at_fpr{OPERATING_POINT_SEP}0.01'."
        ) from None


def operating_point(metrics_cfg: Any, attr: str) -> float:
    """The single value to use where only one operating point is possible.

    Threshold derivation and the slice-table cut need one number even when the
    metric report spans several. The convention is the **first** configured
    value, so the list is ordered by intent: lead with the budget the model is
    actually going to run at, and put the sensitivities behind it.
    """
    value = getattr(metrics_cfg, attr)
    values = list(value) if isinstance(value, (list, tuple)) else [value]
    if not values:
        raise ValueError(f"metrics.{attr} is an empty list; supply at least one value.")
    return float(values[0])


def resolve_metric(name: str, metrics_cfg: Any) -> List[ResolvedMetric]:
    """Expand one configured metric name into the columns it produces.

    A metric with no operating point yields one column under its own name. A
    name written with an explicit point (``recall_at_fpr@0.005``) yields one
    column at that point, whatever the config lists. A bare parametrised name
    yields one column per configured value when the config holds a list, and a
    single un-suffixed column when it holds a scalar.
    """
    base, explicit = split_metric_name(name)
    if base not in METRIC_REGISTRY:
        raise ValueError(f"Unknown metric '{base}'. Available: {sorted(METRIC_REGISTRY)}")
    spec = METRIC_REGISTRY[base]

    if spec.needs is None:
        if explicit is not None:
            raise ValueError(
                f"Metric '{base}' has no operating point, but '{name}' supplies one. "
                f"Parametrised metrics: {sorted(k for k, v in METRIC_REGISTRY.items() if v.needs)}."
            )
        return [ResolvedMetric(base, base)]

    kwarg = _NEEDS_KWARG[spec.needs]
    if explicit is not None:
        return [ResolvedMetric(f"{base}{OPERATING_POINT_SEP}{format_operating_point(explicit)}",
                               base, {kwarg: explicit})]

    configured = getattr(metrics_cfg, spec.needs)
    if not isinstance(configured, (list, tuple)):
        return [ResolvedMetric(base, base, {kwarg: float(configured)})]
    if not len(configured):
        raise ValueError(f"metrics.{spec.needs} is an empty list; supply at least one value.")
    return [
        ResolvedMetric(f"{base}{OPERATING_POINT_SEP}{format_operating_point(v)}",
                       base, {kwarg: float(v)})
        for v in configured
    ]


def resolve_metrics(metrics_cfg: Any) -> List[ResolvedMetric]:
    """Every metric column the config implies, primary first, deduplicated.

    This is the single source of truth for *which* metrics exist and *what they
    are called*. The CV scorers, the fold-level score vector, the holdout
    report and the leaderboard columns are all built from this one list, so
    they cannot disagree about the set or the order.
    """
    primary = resolve_metric(metrics_cfg.primary, metrics_cfg)
    if len(primary) > 1:
        raise ValueError(
            f"metrics.primary='{metrics_cfg.primary}' expands to {len(primary)} operating "
            f"points {[m.name for m in primary]}, so there is no single column to rank the "
            f"leaderboard on. Name the point to select on, e.g. primary: '{primary[0].name}'."
        )

    out: List[ResolvedMetric] = []
    seen: set = set()
    for name in [metrics_cfg.primary, *metrics_cfg.secondary]:
        for m in resolve_metric(name, metrics_cfg):
            if m.name not in seen:
                seen.add(m.name)
                out.append(m)
    return out


def metric_names(metrics_cfg: Any) -> List[str]:
    """The reported metric names, primary first."""
    return [m.name for m in resolve_metrics(metrics_cfg)]


def make_scorers(metrics_cfg: Any) -> Dict[str, Any]:
    """Build the ``scoring`` dict for ``cross_validate``.

    All scorers consume ``predict_proba[:, 1]``. Loss-type metrics are wrapped
    with ``greater_is_better=False``, so sklearn returns them negated; the
    harness un-negates them for display via :func:`orient`.
    """
    return {
        m.name: make_scorer(
            m.spec.fn,
            response_method="predict_proba",
            greater_is_better=m.greater_is_better,
            **m.kwargs,
        )
        for m in resolve_metrics(metrics_cfg)
    }


def score_vector(y_true, y_score, metrics_cfg: Any, signed: bool = False) -> Dict[str, float]:
    """Every configured metric computed from one probability vector.

    ``signed=True`` reproduces the sklearn scorer convention (loss-type metrics
    negated), so values are interchangeable with what ``make_scorers`` produces
    inside ``cross_validate`` -- the harness computes ``predict_proba`` once and
    derives all metrics from it instead of re-predicting once per scorer.
    A metric that cannot be computed on a fold is missing, not zero.
    """
    out: Dict[str, float] = {}
    for m in resolve_metrics(metrics_cfg):
        try:
            v = m.compute(y_true, y_score)
        except Exception:
            v = float("nan")
        if signed and not m.greater_is_better and np.isfinite(v):
            v = -v
        out[m.name] = v
    return out


def orient(name: str, value: float) -> float:
    """Convert a signed sklearn CV score back to its natural units."""
    return -value if not greater_is_better(name) else value


def greater_is_better(name: str) -> bool:
    """Direction of a reported metric, with or without an operating-point suffix."""
    base, _ = split_metric_name(name)
    return METRIC_REGISTRY[base].greater_is_better if base in METRIC_REGISTRY else True


# --------------------------------------------------------------------------
# direct evaluation (holdout reporting)
# --------------------------------------------------------------------------
def evaluate_predictions(y_true, y_score, metrics_cfg: Any) -> Dict[str, float]:
    """All configured metrics in natural units, plus calibration diagnostics."""
    out: Dict[str, float] = {}
    for m in resolve_metrics(metrics_cfg):
        try:
            out[m.name] = m.compute(y_true, y_score)
        except Exception:
            out[m.name] = float("nan")
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
    "MetricDef", "ResolvedMetric", "OPERATING_POINT_SEP",
    "resolve_metric", "resolve_metrics", "metric_names",
    "split_metric_name", "format_operating_point", "operating_point",
    "evaluate_predictions", "score_vector", "decile_table",
    "population_stability_index", "psi_band",
    "ks_statistic", "recall_at_fpr", "lift_at_top_pct", "expected_calibration_error",
]
