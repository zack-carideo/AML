"""
Functional equivalents of the CLI entry points.

Each ``dmf <subcommand>`` maps to one function here that **returns objects**
(never just prints): ``train`` -> :class:`SelectionResult`, ``score`` ->
``(scored DataFrame, report dict)``, ``sweep`` -> ``(comparison DataFrame,
{name: SelectionResult})``. The CLI commands are shells over these same
functions, so the two surfaces cannot drift: a flag on the CLI is a keyword
argument here, resolved through the same :func:`apply_overrides`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd

from ..config import Config, set_dotted
from ..inference import ProductionScorer
from .evaluate import read_table
from .selection import ModelSelectionHarness, SelectionResult
from .sweep import run_sweep  # noqa: F401  (re-exported: sweep's functional form)

#: train() keyword (and CLI train flag) -> the config key it assigns
OVERRIDES = {
    "output_dir": "run.output_dir",
    "name": "run.name",
    "ordering": "selection.ordering_strategy",
    "k_max": "selection.k_max",
    "top_n": "selection.top_n",
    "cv_splits": "split.cv.n_splits",
    "metric": "metrics.primary",
    "n_jobs": "run.n_jobs",
    "seed": "run.random_state",
}
#: keywords whose value needs interpreting rather than assigning
_FLAGS = {"data", "distinct_models", "tune", "no_tune", "quiet", "no_sampling"}
#: every keyword train() and the CLI accept
TRAIN_OVERRIDES = set(OVERRIDES) | _FLAGS


def apply_overrides(cfg: Config, **overrides: Any) -> Config:
    """Apply CLI-style overrides to ``cfg`` in place and re-validate it.

    ``None`` and ``False`` mean "not given"; adding a train flag is one row in
    :data:`OVERRIDES` (or one branch below for a flag with semantics) plus its
    ``add_argument`` in the CLI.
    """
    unknown = set(overrides) - TRAIN_OVERRIDES
    if unknown:
        raise TypeError(f"Unknown train override(s) {sorted(unknown)}; valid: {sorted(TRAIN_OVERRIDES)}")
    for key, dotted in OVERRIDES.items():
        if overrides.get(key) is not None:
            set_dotted(cfg, dotted, overrides[key])
    if data := overrides.get("data"):
        cfg.data.path = str(data)
        cfg.data.format = "parquet" if str(data).endswith(".parquet") else "csv"
    if overrides.get("distinct_models"):
        cfg.selection.top_n_distinct_models = True
    if overrides.get("tune"):
        cfg.tuning.enabled = True
    if overrides.get("no_tune"):
        cfg.tuning.enabled = False
    if overrides.get("no_sampling"):
        cfg.sampling.enabled = False
    if overrides.get("quiet"):
        cfg.run.verbose = 0
    cfg.validate()
    return cfg


def train(
    config: Union[str, Path, Config],
    X: Optional[pd.DataFrame] = None,
    y: Optional[Any] = None,
    **overrides: Any,
) -> SelectionResult:
    """Functional ``dmf train``: run the harness, return the SelectionResult.

    ``overrides`` take the same names as the CLI flags (``k_max=4``,
    ``ordering="rfe"``, ``quiet=True``, ...). ``X``/``y`` optionally supply
    in-memory data instead of ``data.path``. All run artifacts are still
    written to disk as usual.
    """
    cfg = apply_overrides(Config.load(config).copy(), **overrides)
    return ModelSelectionHarness(cfg).run(X, y)


def score(
    model: Union[str, Path, ProductionScorer],
    data: Union[str, Path, pd.DataFrame],
    out: Optional[Union[str, Path]] = None,
    id_column: Optional[str] = None,
    threshold: Optional[float] = None,
    top_pct: Optional[float] = 0.05,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Functional ``dmf score``: load, score, optionally write; return objects.

    ``model`` is a ``model.joblib`` path or an already-built ProductionScorer
    (in which case ``threshold`` / ``top_pct`` are ignored -- the scorer
    carries its own); ``data`` a CSV/Parquet path or a DataFrame. Returns the
    scored frame (probability, guard flags, decision, action) and the batch
    quality report. ``out`` additionally writes the frame to CSV.
    """
    scorer = model if isinstance(model, ProductionScorer) else ProductionScorer.from_joblib(
        model, threshold=threshold, top_pct=top_pct
    )
    frame = data if isinstance(data, pd.DataFrame) else read_table(data)

    scored, report = scorer.score(frame)
    if id_column and id_column in frame.columns:
        scored.insert(0, id_column, frame[id_column].to_numpy())
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(out, index=False)
    return scored, report


__all__ = ["train", "score", "run_sweep", "apply_overrides", "OVERRIDES", "TRAIN_OVERRIDES"]
