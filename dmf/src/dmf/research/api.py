"""
Functional equivalents of the CLI entry points.

Each `dmf <subcommand>` maps to one function here that **returns objects**
(never just prints): ``train`` -> :class:`SelectionResult`, ``score`` ->
``(scored DataFrame, report dict)``, ``sweep`` -> ``(comparison DataFrame,
{name: SelectionResult})``. The CLI commands are shells over these same
functions, so the two surfaces cannot drift: a flag on the CLI is a keyword
argument here, resolved through the same override map.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd

from ..config import Config
from ..inference import ProductionScorer
from .selection import ModelSelectionHarness, SelectionResult
from .sweep import run_sweep  # noqa: F401  (re-exported: sweep's functional form)

#: keyword arguments train() accepts, mirroring the CLI's train flags exactly
_TRAIN_KWARGS = {"data", "output_dir", "name", "ordering", "k_max", "top_n",
                 "distinct_models", "cv_splits", "metric", "tune", "no_tune",
                 "n_jobs", "seed", "quiet"}


def _as_config(config: Union[str, Path, Config]) -> Config:
    return config if isinstance(config, Config) else Config.from_yaml(config)


def train(
    config: Union[str, Path, Config],
    X: Optional[pd.DataFrame] = None,
    y: Optional[Any] = None,
    **overrides: Any,
) -> SelectionResult:
    """Functional ``dmf train``: run the harness, return the SelectionResult.

    ``overrides`` take the same names as the CLI flags (``k_max=4``,
    ``ordering="rfe"``, ``quiet=True``, ...) and are applied through the same
    map the CLI uses. ``X``/``y`` optionally supply in-memory data instead of
    ``data.path``. All run artifacts are still written to disk as usual.
    """
    from .cli import apply_overrides   # local import: keep api importable without argparse cost

    unknown = set(overrides) - _TRAIN_KWARGS
    if unknown:
        raise TypeError(f"Unknown train override(s) {sorted(unknown)}; valid: {sorted(_TRAIN_KWARGS)}")
    args = Namespace(**{k: overrides.get(k, False if k in ("distinct_models", "tune", "no_tune", "quiet")
                                          else None) for k in _TRAIN_KWARGS})
    cfg = apply_overrides(_as_config(config).copy(), args)
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

    ``model`` is a ``model.joblib`` path or an already-built ProductionScorer;
    ``data`` a CSV/Parquet path or a DataFrame. Returns the scored frame
    (probability, guard flags, decision, action) and the batch quality report.
    ``out`` additionally writes the frame to CSV, but the return value is
    always the objects themselves.
    """
    scorer = model if isinstance(model, ProductionScorer) else ProductionScorer.from_joblib(
        model, threshold=threshold, top_pct=top_pct
    )
    if not isinstance(data, pd.DataFrame):
        reader = pd.read_parquet if str(data).endswith(".parquet") else pd.read_csv
        data = reader(data)

    scored, report = scorer.score(data)
    if id_column and id_column in data.columns:
        scored.insert(0, id_column, data[id_column].to_numpy())
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(out, index=False)
    return scored, report


__all__ = ["train", "score", "run_sweep"]
