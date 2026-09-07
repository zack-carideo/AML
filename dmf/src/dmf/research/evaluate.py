"""
Row-level prediction store and post-run evaluation.

Everything the harness scores is also *storable*: for every fit/predict the
selection loop performs, the store keeps ``(row_id, y_true, y_score)`` plus the
provenance that makes the row analysable -- which stage, which model, which
variable count, which fold. From that table, any metric -- present in
:mod:`dmf.metrics` today or added later -- can be recomputed after the run
without refitting a single model, and so can everything a metric registry
cannot anticipate: calibration curves, threshold sweeps, slice analyses,
per-fold gains tables.

Two deliberate schema choices:

* **Scores, not labels.** A predicted label is ``score >= threshold`` for some
  threshold, and during selection there is no threshold -- every selection
  metric is threshold-free or fixes an operating point (an FPR, a review
  budget) and lets the threshold fall out. Storing raw scores keeps every
  operating point available post hoc; storing labels would bake one in.
* **``y_true`` travels with the score.** Without it, every downstream analysis
  has to re-join the raw data and re-run target binarisation to reproduce what
  the harness saw.

The sidecar ``predictions_meta.json`` records what the numbers *mean*: the
resolved positive label, the score semantics, the metric operating points in
force, and the derived decision threshold -- so a reader six months from now
does not have to guess.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from ..metrics import as_metrics_config, evaluate_predictions, operating_point
from ..reporting import json_safe

#: columns of the long-format prediction table, in storage order
PREDICTION_COLUMNS = ["row_id", "y_true", "y_score", "stage", "model", "k", "fold", "repeat"]

#: capture levels, cumulative: each level stores everything below it
_LEVELS = {"none": 0, "holdout": 1, "cv": 2, "all": 3}
#: minimum level at which each stage is captured
_STAGE_LEVEL = {"holdout": 1, "cv": 2, "cv_train": 3}


class PredictionLog:
    """Accumulates row-level predictions during a harness run.

    The log is written to by the harness and is append-only; it never reads
    fitted state back off an estimator. ``level`` is ``run.save_predictions``
    and gates what :meth:`add` actually keeps, so call sites can log
    unconditionally and stay readable.
    """

    def __init__(self, level: str = "holdout"):
        if level not in _LEVELS:
            raise ValueError(f"save_predictions='{level}' is invalid; expected one of {sorted(_LEVELS)}.")
        self.level = level
        self._chunks: List[Dict[str, Any]] = []
        self._folds: Dict[Tuple[int, int], np.ndarray] = {}

    @property
    def enabled(self) -> bool:
        return self.level != "none"

    def wants(self, stage: str) -> bool:
        return _LEVELS[self.level] >= _STAGE_LEVEL[stage]

    def add(
        self,
        stage: str,
        model: str,
        k: int,
        fold: Optional[int],
        repeat: Optional[int],
        row_ids: Sequence[Any],
        y_true: Sequence[Any],
        y_score: Sequence[float],
    ) -> None:
        if stage not in _STAGE_LEVEL:
            raise ValueError(f"Unknown prediction stage '{stage}'; expected one of {sorted(_STAGE_LEVEL)}.")
        if not self.wants(stage):
            return
        row_ids = np.asarray(row_ids)
        y_true = np.asarray(y_true).ravel().astype(np.int8)
        y_score = np.asarray(y_score, dtype=float).ravel()
        if not (len(row_ids) == len(y_true) == len(y_score)):
            raise ValueError(
                f"Misaligned prediction chunk: {len(row_ids)} ids, {len(y_true)} targets, "
                f"{len(y_score)} scores."
            )
        self._chunks.append(
            {"stage": stage, "model": str(model), "k": int(k),
             "fold": None if fold is None else int(fold),
             "repeat": None if repeat is None else int(repeat),
             "row_id": row_ids, "y_true": y_true, "y_score": y_score}
        )

    def add_fold(self, fold: int, repeat: int, row_ids: Sequence[Any]) -> None:
        """Record which rows form the validation partition of a CV fold.

        Idempotent per (fold, repeat): the grid visits every fold once per
        model x k cell, but membership is a property of the fold alone.
        """
        if not self.enabled:
            return
        self._folds.setdefault((int(fold), int(repeat)), np.asarray(row_ids))

    def to_frame(self) -> pd.DataFrame:
        frames = []
        for c in self._chunks:
            n = len(c["row_id"])
            df = pd.DataFrame(
                {"row_id": c["row_id"], "y_true": c["y_true"], "y_score": c["y_score"]}
            )
            df["stage"] = np.repeat(c["stage"], n)
            df["model"] = np.repeat(c["model"], n)
            df["k"] = np.repeat(np.int64(c["k"]), n)
            df["fold"] = pd.array([c["fold"]] * n, dtype="Int64")
            df["repeat"] = pd.array([c["repeat"]] * n, dtype="Int64")
            frames.append(df)
        if not frames:
            return pd.DataFrame(columns=PREDICTION_COLUMNS)
        return pd.concat(frames, ignore_index=True)[PREDICTION_COLUMNS]

    def fold_frame(self) -> pd.DataFrame:
        rows = [
            pd.DataFrame({"row_id": ids,
                          "fold": np.repeat(np.int64(fold), len(ids)),
                          "repeat": np.repeat(np.int64(repeat), len(ids))})
            for (fold, repeat), ids in sorted(self._folds.items())
        ]
        if not rows:
            return pd.DataFrame(columns=["row_id", "fold", "repeat"])
        return pd.concat(rows, ignore_index=True)

    def stage_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for c in self._chunks:
            out[c["stage"]] = out.get(c["stage"], 0) + len(c["row_id"])
        return out


# --------------------------------------------------------------------------
# artifact IO
# --------------------------------------------------------------------------
def read_table(path: Union[str, Path], **kwargs: Any) -> pd.DataFrame:
    """CSV or Parquet, decided by the file extension."""
    path = Path(path)
    reader = pd.read_parquet if path.suffix == ".parquet" else pd.read_csv
    return reader(path, **kwargs)


def write_table(df: pd.DataFrame, stem: Path) -> str:
    """Parquet with a CSV fallback, returning the file name actually written."""
    try:
        df.to_parquet(stem.with_suffix(".parquet"), index=False)
        return stem.with_suffix(".parquet").name
    except Exception:                     # no parquet engine installed
        df.to_csv(stem.with_suffix(".csv"), index=False)
        return stem.with_suffix(".csv").name


def _read_stored(run_dir: Union[str, Path], stem: str) -> Optional[pd.DataFrame]:
    """Whichever of ``<stem>.parquet`` / ``<stem>.csv`` a run wrote, or None."""
    for ext in (".parquet", ".csv"):
        path = Path(run_dir) / f"{stem}{ext}"
        if path.exists():
            return read_table(path)
    return None


def write_prediction_artifacts(
    out_dir: Union[str, Path], log: PredictionLog, meta: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """Write ``predictions``, ``fold_assignments`` and the meta sidecar.

    Returns ``{artifact: file name}`` for whatever was written; empty when the
    log is disabled.
    """
    if not log.enabled:
        return {}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}

    preds = log.to_frame()
    if len(preds):
        written["predictions"] = write_table(preds, out / "predictions")
    folds = log.fold_frame()
    if len(folds):
        written["fold_assignments"] = write_table(folds, out / "fold_assignments")

    meta = dict(meta or {})
    meta["save_predictions"] = log.level
    meta["n_rows"] = int(len(preds))
    meta["rows_per_stage"] = log.stage_counts()
    meta["files"] = written
    # full precision: the sidecar carries the decision threshold, and a rounded
    # copy of a cut is a different cut on a discrete score scale
    (out / "predictions_meta.json").write_text(json.dumps(json_safe(meta, ndigits=None), indent=2))
    written["meta"] = "predictions_meta.json"
    return written


def load_predictions(run_dir: Union[str, Path]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load a run's prediction store: ``(predictions, meta)``.

    ``run_dir`` is the per-run artifact directory (the one holding
    ``run_report.json``). Raises with a pointer at ``run.save_predictions``
    when the run did not store predictions.
    """
    preds = _read_stored(run_dir, "predictions")
    if preds is None:
        raise FileNotFoundError(
            f"No predictions.parquet/csv under '{run_dir}'. The run was executed with "
            f"run.save_predictions='none', or predates the prediction store."
        )
    meta_path = Path(run_dir) / "predictions_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return preds, meta


def load_fold_assignments(run_dir: Union[str, Path]) -> pd.DataFrame:
    folds = _read_stored(run_dir, "fold_assignments")
    if folds is None:
        raise FileNotFoundError(f"No fold_assignments.parquet/csv under '{run_dir}'.")
    return folds


# --------------------------------------------------------------------------
# post-hoc evaluation
# --------------------------------------------------------------------------
def _threshold_at_fpr(y: np.ndarray, s: np.ndarray, max_fpr: float) -> Tuple[Optional[float], Optional[float]]:
    """``(score cut, recall)`` at a false-positive budget; ``(None, None)`` on a single-class target."""
    if len(np.unique(y)) < 2:
        return None, None
    fpr, tpr, thr = roc_curve(y, s)
    thr = np.where(np.isfinite(thr), thr, float(np.max(s)))   # sklearn's leading +inf
    return float(np.interp(max_fpr, fpr, thr)), float(np.interp(max_fpr, fpr, tpr))


def _confusion(y: np.ndarray, s: np.ndarray, threshold: float) -> Dict[str, Any]:
    """Counts and rates of ``s >= threshold`` against ``y``."""
    flag = s >= threshold
    n, n_pos, n_flag = len(y), int(y.sum()), int(flag.sum())
    tp = int(((y == 1) & flag).sum())
    fp = n_flag - tp
    precision = tp / n_flag if n_flag else None
    recall = tp / n_pos if n_pos else None
    return {
        "n": n, "n_flagged": n_flag,
        "flag_rate": round(n_flag / n, 6) if n else None,
        "tp": tp, "fp": fp, "fn": n_pos - tp, "tn": (n - n_pos) - fp,
        "precision": round(precision, 6) if precision is not None else None,
        "recall": round(recall, 6) if recall is not None else None,
        "fpr": round(fp / (n - n_pos), 6) if n - n_pos else None,
        "f1": round(2 * precision * recall / (precision + recall), 6) if precision and recall else None,
    }


def _grouped(preds: pd.DataFrame, by: Iterable[str]):
    by = [b for b in by if b in preds.columns]
    if not by:
        yield [], (), preds
        return
    for key, g in preds.groupby(by, dropna=False, observed=True, sort=True):
        yield by, key if isinstance(key, tuple) else (key,), g


def compute_metrics(
    preds: pd.DataFrame,
    metrics_cfg: Any = None,
    by: Sequence[str] = ("stage", "model", "k"),
) -> pd.DataFrame:
    """All configured metrics, recomputed per group from stored predictions.

    Uses the same :func:`dmf.metrics.evaluate_predictions` the harness used, so
    for the groups the leaderboard reports, the recomputed numbers reproduce it.
    ``metrics_cfg`` may be a ``MetricsConfig``, the ``meta`` dict returned by
    :func:`load_predictions` (operating points travel with the store), or None
    for framework defaults. ``by`` controls granularity, e.g. add ``"fold"``
    for per-fold rows.
    """
    cfg = as_metrics_config(metrics_cfg)
    rows = []
    for keys, key, g in _grouped(preds, by):
        row: Dict[str, Any] = dict(zip(keys, key))
        row["n"] = int(len(g))
        row["n_positive"] = int(g["y_true"].sum())
        row.update(evaluate_predictions(g["y_true"], g["y_score"], cfg))
        rows.append(row)
    return pd.DataFrame(rows)


def threshold_at_fpr(
    preds: pd.DataFrame,
    max_fpr: float = 0.01,
    by: Sequence[str] = ("stage", "model", "k", "fold"),
) -> pd.DataFrame:
    """The score cut that achieves a given false-positive rate, per group.

    This is the number ``recall_at_fpr`` computes internally and discards: the
    implied operating threshold. The spread of this threshold across folds is
    the stability evidence to look at before hardcoding a production
    ``decision_threshold``.
    """
    rows = []
    for keys, key, g in _grouped(preds, by):
        thr, recall = _threshold_at_fpr(g["y_true"].to_numpy(), g["y_score"].to_numpy(dtype=float), max_fpr)
        rows.append({
            **dict(zip(keys, key)),
            "n": int(len(g)), "n_positive": int(g["y_true"].sum()), "max_fpr": float(max_fpr),
            "threshold": None if thr is None else round(thr, 6),
            "recall": None if recall is None else round(recall, 6),
        })
    return pd.DataFrame(rows)


def operating_point_table(
    preds: pd.DataFrame,
    thresholds: Optional[Sequence[float]] = None,
    by: Sequence[str] = ("stage", "model", "k"),
) -> pd.DataFrame:
    """Confusion-matrix economics at candidate absolute thresholds, per group.

    The artifact to hand to whoever signs off a deployment threshold: for each
    cut -- flag rate, precision, recall, FPR, F1 and the raw counts. Default
    thresholds are the stored score quantiles at 20/10/5/2/1% flag rates, so
    they always land inside the score distribution. Thresholds are shared
    across groups, which is what makes rows comparable.
    """
    if thresholds is None:
        s_all = preds["y_score"].to_numpy(dtype=float)
        thresholds = np.unique(np.round(np.quantile(s_all, [0.80, 0.90, 0.95, 0.98, 0.99]), 6))
    rows = []
    for keys, key, g in _grouped(preds, by):
        y = g["y_true"].to_numpy()
        s = g["y_score"].to_numpy(dtype=float)
        for t in thresholds:
            rows.append({**dict(zip(keys, key)), "threshold": round(float(t), 6), **_confusion(y, s, t)})
    return pd.DataFrame(rows)


def implied_thresholds(y_true, y_score, metrics_cfg: Any) -> Dict[str, Any]:
    """Derive the champion's decision threshold from its holdout scores.

    Both candidate operating points are always reported; which one becomes
    ``decision_threshold`` is ``metrics.decision_threshold_policy``. The
    threshold ships in the model bundle, so :class:`dmf.inference.ProductionScorer`
    applies a stable absolute cut instead of a batch-relative quantile.
    """
    cfg = as_metrics_config(metrics_cfg)
    y = np.asarray(y_true).ravel().astype(int)
    s = np.asarray(y_score, dtype=float).ravel()

    # a threshold is one number, so when the config lists several budgets the
    # first one is the operating point; the rest are reported sensitivities
    top_pct = operating_point(cfg, "lift_top_pct")
    fpr_point = operating_point(cfg, "recall_at_fpr")

    thr_top = float(np.quantile(s, 1 - top_pct)) if top_pct else None
    thr_fpr, _ = _threshold_at_fpr(y, s, fpr_point)

    policy = getattr(cfg, "decision_threshold_policy", "top_pct")
    decision = {"top_pct": thr_top, "fpr": thr_fpr, "none": None}[policy]
    if decision is not None and not np.isfinite(decision):
        decision = None

    # Thresholds are stored at full precision, never rounded. The scorer applies
    # ``score >= threshold``, and on a model with a discrete score scale many
    # rows tie exactly at the quantile; rounding the stored value up by one unit
    # in the sixth decimal silently drops every one of them, so the shipped cut
    # no longer flags the share of volume the reported decision_flag_rate says.
    out: Dict[str, Any] = {
        "implied_threshold_top_pct": thr_top,
        "implied_threshold_at_fpr": thr_fpr,
        "decision_threshold_policy": policy,
        # the budget the shipped threshold was derived at, so a reader never has
        # to work out which entry of a list the number came from
        "decision_operating_point": top_pct if policy == "top_pct" else (
            fpr_point if policy == "fpr" else None),
        "decision_threshold": decision,
    }
    if decision is not None:
        c = _confusion(y, s, decision)
        out.update(decision_flag_rate=c["flag_rate"], decision_precision=c["precision"],
                   decision_recall=c["recall"])
    return out


__all__ = [
    "PredictionLog", "PREDICTION_COLUMNS", "read_table", "write_table",
    "write_prediction_artifacts", "load_predictions", "load_fold_assignments",
    "compute_metrics", "threshold_at_fpr", "operating_point_table", "implied_thresholds",
]
