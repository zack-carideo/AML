"""
Production scoring wrapper.

Loading a pickled pipeline and calling ``predict_proba`` is the easy part. What
this module adds is the discipline around it: every scored batch comes back
with the probability *and* the evidence about whether the input was inside the
support the model was trained on.

The policy it implements is deliberately conservative. A record whose score
depended on a clipped numeric value or an out-of-vocabulary category still gets
a score -- refusing to score is its own operational failure -- but it is
labelled ``manual_review`` rather than ``auto_action``, so a dispute is never
auto-declined on the strength of an extrapolation the model never learned.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .pipeline import DisputeFeaturePipeline


class ProductionScorer:
    """Fitted-pipeline wrapper that returns scores with a data-quality verdict.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        A fitted ``features -> model`` pipeline (see
        :func:`dmf.pipeline.build_model_pipeline`).
    features : list[str], optional
        Recorded for the schema contract; defaults to the pipeline's own list.
    review_on_flag : bool
        Route any row that tripped a guard to manual review.
    threshold : float, optional
        Absolute probability cut for the ``flag_fraud`` decision.
    top_pct : float, optional
        Alternative relative cut -- flag the top share of the batch by score.
        Ignored when ``threshold`` is set.
    """

    def __init__(
        self,
        pipeline: Any,
        features: Optional[List[str]] = None,
        review_on_flag: bool = True,
        threshold: Optional[float] = None,
        top_pct: Optional[float] = 0.05,
    ):
        self.pipeline = pipeline
        self.review_on_flag = review_on_flag
        self.threshold = threshold
        self.top_pct = top_pct
        self.features = features or list(self._feature_step().input_features_)
        self.metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    @classmethod
    def from_joblib(cls, path: str | Path, **kwargs: Any) -> "ProductionScorer":
        """Load a bundle written by :class:`dmf.selection.ModelSelectionHarness`."""
        import joblib

        bundle = joblib.load(path)
        if not (isinstance(bundle, dict) and "pipeline" in bundle):
            return cls(bundle, **kwargs)
        # a threshold stored with the model wins over the constructor default,
        # so production does not silently fall back to a batch-relative cut
        kwargs.setdefault("threshold", bundle.get("decision_threshold"))
        scorer = cls(bundle["pipeline"], features=bundle.get("features"), **kwargs)
        scorer.metadata = {k: v for k, v in bundle.items() if k != "pipeline"}
        _warn_on_version_skew(bundle.get("dmf_version"))
        return scorer

    def _feature_step(self) -> DisputeFeaturePipeline:
        """Locate the fitted feature pipeline, however it has been wrapped.

        Post-hoc wrappers are a normal part of deployment -- a threshold tuned
        with ``TunedThresholdClassifierCV``, probabilities recalibrated with
        ``CalibratedClassifierCV``. Both nest the pipeline one level down, so
        the lookup walks the usual attributes rather than insisting the object
        it was handed is the pipeline itself.
        """
        step = _find_feature_step(self.pipeline)
        if step is None:
            raise ValueError(
                "No fitted DisputeFeaturePipeline found in the supplied estimator; "
                "expected a dmf production pipeline, optionally wrapped by a "
                "calibrator or threshold tuner."
            )
        return step

    # ------------------------------------------------------------------
    def check_schema(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Compare an inference frame against the training schema contract."""
        present = set(X.columns)
        required = list(self.features)
        missing = [c for c in required if c not in present]
        extra = [c for c in present if c not in required]
        return {
            "n_required": len(required),
            "n_present": len([c for c in required if c in present]),
            "missing_columns": missing,
            "extra_columns_ignored": extra[:50],
            "n_extra_ignored": len(extra),
            "schema_ok": not missing,
        }

    def training_envelope(self) -> pd.DataFrame:
        return self._feature_step().training_envelope()

    # ------------------------------------------------------------------
    def score(self, X: pd.DataFrame, return_report: bool = True):
        """Score a batch.

        Returns a frame with one row per input record::

            fraud_probability   model output
            score_rank          1 = highest risk in this batch
            data_quality        'ok' | 'guarded'
            n_out_of_range      numeric cells clipped/nulled by the guard
            n_unseen_category   categorical cells mapped out of vocabulary
            n_coerced           non-numeric junk found in a numeric column
            n_newly_missing     required columns absent from the input
            decision            'flag_fraud' | 'pass'
            action              'auto_action' | 'manual_review'
        """
        schema = self.check_schema(X)
        feats = self._feature_step()

        proba = self.pipeline.predict_proba(X)[:, 1]
        # A second, independent transform purely to obtain the quality report.
        # It costs one pass and buys correctness: the report is returned rather
        # than scraped off estimator state, so it is right under concurrency and
        # right when a calibrator or threshold tuner ran predict_proba on clones.
        _, report, flags = feats.transform_with_quality(X)
        flags = flags.set_axis(X.index)

        out = pd.DataFrame({"fraud_probability": proba}, index=X.index)
        out["score_rank"] = out["fraud_probability"].rank(ascending=False, method="first").astype(int)

        # A column the feed stopped sending affects every record identically, so
        # the guard records it once at batch level. The *decision* still has to
        # apply per row, or a batch scored on imputed training defaults would be
        # auto-actioned while a JSON field nobody reads says otherwise.
        n_missing = int(report.get("n_missing_columns", 0))
        flags = flags.assign(n_newly_missing=n_missing)
        batch_unsafe = report.get("batch_safe") is False

        out = out.join(flags)
        guarded = (flags.sum(axis=1) > 0) | batch_unsafe
        out["data_quality"] = np.where(guarded, "guarded", "ok")

        cut = self._decision_cut(proba)
        out["decision"] = np.where(out["fraud_probability"] >= cut, "flag_fraud", "pass")
        out["action"] = np.where(
            guarded & self.review_on_flag, "manual_review",
            np.where(out["decision"] == "flag_fraud", "manual_review", "auto_action"),
        )

        report.update(
            schema=schema,
            decision_cut=round(float(cut), 6),
            n_flagged=int((out["decision"] == "flag_fraud").sum()),
            flag_rate=round(float((out["decision"] == "flag_fraud").mean()), 6) if len(out) else 0.0,
            n_manual_review=int((out["action"] == "manual_review").sum()),
            mean_score=round(float(np.mean(proba)), 6) if len(proba) else None,
            score_p99=round(float(np.quantile(proba, 0.99)), 6) if len(proba) else None,
        )
        if not schema["schema_ok"]:
            report["verdict"] = "review_recommended"
            report["schema_warning"] = (
                f"{len(schema['missing_columns'])} required column(s) absent; the fitted imputer "
                f"supplied training defaults for them."
            )
        return (out, report) if return_report else out

    #: below this batch size a quantile of the batch is not a meaningful cut
    MIN_BATCH_FOR_RELATIVE_CUT = 50

    def _decision_cut(self, proba: np.ndarray) -> float:
        """The probability at or above which a dispute is flagged.

        An absolute ``threshold`` is stable: the same dispute gets the same
        decision whichever file it arrives in. A ``top_pct`` cut is a quantile
        of *this batch*, which is fine for a review queue sized to analyst
        capacity and meaningless for a handful of records -- for a single record
        it is that record's own probability, so everything flags. Small batches
        therefore require an explicit threshold rather than silently degrading.
        """
        if self.threshold is not None:
            return float(self.threshold)
        if not self.top_pct:
            return 0.5
        if len(proba) < self.MIN_BATCH_FOR_RELATIVE_CUT:
            raise ValueError(
                f"A relative cut (top_pct={self.top_pct}) needs at least "
                f"{self.MIN_BATCH_FOR_RELATIVE_CUT} records to be meaningful; this batch has "
                f"{len(proba)}. Pass an absolute threshold, e.g. "
                f"ProductionScorer(..., threshold=0.8), for small batches and real-time scoring."
            )
        return float(np.quantile(proba, 1.0 - self.top_pct))

    # ------------------------------------------------------------------
    def score_one(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Single-record path for a real-time endpoint.

        Requires an absolute ``threshold`` -- a batch-relative cut on one record
        is that record's own probability, which would flag everything.
        ``score`` enforces that, so this is pure delegation.
        """
        out, report = self.score(pd.DataFrame([record]))
        return {**out.iloc[0].to_dict(),
                "batch_safe": report.get("batch_safe"),
                "guard_detail": report.get("by_column", {})}

    def explain_guard(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-column breakdown of what the guard had to intervene on.

        Deliberately independent of the decision logic, so it works on a batch
        of any size and on a scorer with no threshold configured.
        """
        _, report, _ = self._feature_step().transform_with_quality(X)
        rows = []
        for col, d in (report.get("by_column") or {}).items():
            rows.append({"variable": col, **{k: v for k, v in d.items() if not isinstance(v, list)},
                         "novel_levels": ", ".join(map(str, d.get("novel_levels", [])[:5]))})
        return pd.DataFrame(rows)


def _find_feature_step(obj: Any, depth: int = 0) -> Optional[DisputeFeaturePipeline]:
    """Depth-first search for the fitted feature pipeline inside an estimator."""
    if depth > 4 or obj is None:
        return None
    # only a *fitted* pipeline is useful: CalibratedClassifierCV leaves its
    # prototype estimator unfitted and does the work in internal clones, so the
    # clones are searched before the prototype.
    if isinstance(obj, DisputeFeaturePipeline):
        return obj if hasattr(obj, "pipeline_") else None
    for cal in getattr(obj, "calibrated_classifiers_", []) or []:
        found = _find_feature_step(getattr(cal, "estimator", None), depth + 1)
        if found is not None:
            return found
    steps = getattr(obj, "named_steps", None)
    if steps is not None and "features" in steps:
        found = _find_feature_step(steps["features"], depth + 1)
        if found is not None:
            return found
    for attr in ("estimator_", "best_estimator_", "base_estimator_", "estimator"):
        found = _find_feature_step(getattr(obj, attr, None), depth + 1)
        if found is not None:
            return found
    return None


def _warn_on_version_skew(saved: Optional[str]) -> None:
    """A pickle is only valid against the code that defined its classes."""
    from . import __version__

    if saved and saved != __version__:
        import warnings

        warnings.warn(
            f"Model was saved by dmf {saved} but dmf {__version__} is installed. "
            f"Custom transformers are pickled by reference; verify scores against a "
            f"known batch before trusting this model.",
            RuntimeWarning,
            stacklevel=3,
        )


def load_scorer(path: str | Path, **kwargs: Any) -> ProductionScorer:
    return ProductionScorer.from_joblib(path, **kwargs)


__all__ = ["ProductionScorer", "load_scorer"]
