"""
dmf -- the production core of the dispute-model framework.

This package contains only what the deployed scoring path needs, and it is the
whole of what a production team maintains:

* :class:`dmf.pipeline.DisputeFeaturePipeline` -- a fitted, picklable sklearn
  transformer; the same instance serves training and inference.
* :class:`dmf.inference.ProductionScorer` -- loads a persisted model and scores
  new records, returning probability plus a data-quality verdict.
* :class:`dmf.config.Config` -- the typed YAML configuration. It also carries
  the experiment-time sections (selection, tuning) so that one YAML file drives
  both halves and a shipped model can store the exact config that produced it;
  the core itself never reads those sections.
* column typing, guarded transformers, metrics, and step reporting.

Everything experiment-time -- the model x k grid search, variable ordering, the
estimator zoo, the CLI -- lives in :mod:`dmf.research`, which imports this core
and is never imported by it. The dependency points one way so a change to the
research half can never alter what production executes.

Quick start::

    from dmf import ProductionScorer
    scorer = ProductionScorer.from_joblib("artifacts/run/model.joblib")
    scored, report = scorer.score(new_disputes)

    from dmf.research import ModelSelectionHarness    # experiment side
    result = ModelSelectionHarness(Config.from_yaml("config.yaml")).run()
"""

from .config import Config, ModelSpec
from .inference import ProductionScorer, load_scorer
from .metrics import (
    METRIC_REGISTRY,
    decile_table,
    evaluate_predictions,
    make_scorers,
    metric_names,
    operating_point,
    population_stability_index,
    psi_band,
    resolve_metrics,
)
from .pipeline import DisputeFeaturePipeline, build_model_pipeline
from .reporting import StepReport, run_lineage
from .transformers import (
    ROW_FLAG_COLUMNS,
    FrameSelector,
    InferenceGuard,
    NumericCoercer,
    QuantileWinsorizer,
    RareCategoryCollapser,
    Roles,
    WOEEncoder,
    infer_roles,
    parse_kind,
    to_numeric_lenient,
)

__version__ = "0.2.0"

_MOVED_TO_RESEARCH = {
    "ModelSelectionHarness", "SelectionResult", "rank_variables",
    "importance_ordering", "rfe_ordering", "build_estimator", "build_zoo",
    "config_for_model",
}


def __getattr__(name: str):
    """Point users of the pre-split layout at the new home (PEP 562)."""
    if name in _MOVED_TO_RESEARCH:
        raise ImportError(
            f"'{name}' moved to the experiment-time subpackage in dmf 0.2: "
            f"use 'from dmf.research import {name}'. The core 'dmf' package now "
            f"contains only the deployed scoring path."
        )
    raise AttributeError(f"module 'dmf' has no attribute '{name}'")


__all__ = [
    "Config",
    "ModelSpec",
    "DisputeFeaturePipeline",
    "build_model_pipeline",
    "ProductionScorer",
    "load_scorer",
    "StepReport",
    "run_lineage",
    "METRIC_REGISTRY",
    "make_scorers",
    "metric_names",
    "operating_point",
    "resolve_metrics",
    "evaluate_predictions",
    "decile_table",
    "population_stability_index",
    "psi_band",
    "ROW_FLAG_COLUMNS",
    "Roles",
    "infer_roles",
    "parse_kind",
    "to_numeric_lenient",
    "NumericCoercer",
    "QuantileWinsorizer",
    "RareCategoryCollapser",
    "WOEEncoder",
    "FrameSelector",
    "InferenceGuard",
    "__version__",
]
