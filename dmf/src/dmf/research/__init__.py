"""
dmf.research -- the experiment-time half of the framework.

Everything here runs on an analyst's machine during model development and is
never part of the deployed scoring path: the model x variable-count grid search
(:class:`ModelSelectionHarness`), the variable-ordering strategies, the
YAML-declared estimator zoo, and the command-line entry points.

The dependency rule is one-way: this package imports the ``dmf`` core; the core
never imports this package. Its only durable output is a winning specification
-- a variable list plus a model -- which the core materialises for production
via :func:`dmf.pipeline.build_model_pipeline`.
"""

from .api import score, train
from .ordering import aggregate_to_source, importance_ordering, rank_variables, rfe_ordering
from .selection import ModelSelectionHarness, SelectionResult
from .sweep import check_comparability, run_sweep
from .zoo import build_estimator, build_zoo, config_for_model, import_object

__all__ = [
    "ModelSelectionHarness",
    "SelectionResult",
    "train",
    "score",
    "run_sweep",
    "check_comparability",
    "rank_variables",
    "importance_ordering",
    "rfe_ordering",
    "aggregate_to_source",
    "build_estimator",
    "build_zoo",
    "config_for_model",
    "import_object",
]
