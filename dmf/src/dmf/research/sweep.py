"""
Run several configurations over the same data and compare them honestly.

A sweep is only as honest as its comparability: two runs are on the same
footing only if they share the seed, the split design, and the data -- otherwise
their holdout rows differ and the comparison is noise dressed as a table. The
sweep therefore checks those settings up front and stamps the result with what
matched and what did not, rather than assuming.

Comparison is done on **holdout** metrics, not leaderboards: each run's
leaderboard already picked its own winner, so choosing a config by leaderboard
re-introduces selection bias one level up. And the holdout itself erodes as a
referee if dozens of configs are swept against it -- for large sweeps, keep a
second untouched partition for the final call.
"""

from __future__ import annotations

import functools
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from ..config import Config
from .selection import ModelSelectionHarness, SelectionResult

#: settings that must agree for two runs to share holdout rows and CV folds
COMPARABILITY_KEYS = [
    "run.random_state",
    "split.strategy",
    "split.holdout_size",
    "split.cv.n_splits",
    "split.cv.n_repeats",
    "data.path",
]


def _get(cfg: Config, dotted: str) -> Any:
    return functools.reduce(getattr, dotted.split("."), cfg)


def check_comparability(configs: List[Config]) -> Dict[str, List[Any]]:
    """Settings that differ across configs and would break like-for-like comparison."""
    return {
        key: values
        for key in COMPARABILITY_KEYS
        if len(set(map(str, (values := [_get(c, key) for c in configs])))) > 1
    }


def run_sweep(
    configs: List[Union[str, Path, Config]],
    X: Optional[pd.DataFrame] = None,
    y: Optional[Any] = None,
    output_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, SelectionResult]]:
    """Run each config through the harness and return (comparison, results).

    ``X``/``y`` optionally supply one shared in-memory dataset; otherwise each
    config loads its own ``data.path``. Every run's full artifact set is still
    written under its own ``run.name``; the sweep adds one comparison table
    across them, keyed to holdout performance.
    """
    cfgs = [c if isinstance(c, Config) else Config.from_yaml(c) for c in configs]
    for i, cfg in enumerate(cfgs):                    # unique artifact dirs
        if output_dir:
            cfg.run.output_dir = output_dir
        if sum(c.run.name == cfg.run.name for c in cfgs) > 1:
            cfg.run.name = f"{cfg.run.name}_{i}"

    mismatched = check_comparability(cfgs)
    if mismatched:
        warnings.warn(
            f"Sweep runs are NOT like-for-like: {sorted(mismatched)} differ across "
            f"configs, so holdout rows and/or CV folds differ. The comparison table "
            f"is stamped comparable=False.",
            RuntimeWarning,
            stacklevel=2,
        )

    rows, results = [], {}
    for cfg in cfgs:
        res = ModelSelectionHarness(cfg).run(None if X is None else X.copy(), y)
        results[cfg.run.name] = res
        primary = cfg.metrics.primary
        lineage = res.report.get("lineage") or {}
        slices = res.report.get("holdout_slices") or {}
        rows.append({
            "run": cfg.run.name,
            "config_sha256": lineage.get("config_sha256"),
            "data_sha256": lineage.get("data_sha256"),
            "comparable": not mismatched,
            "primary_metric": primary,
            "selected_model": res.selected_model,
            "k": res.selected["k"],
            "cv_primary_mean": res.selected[f"cv_{primary}_mean"],
            "cv_primary_se": res.selected[f"cv_{primary}_se"],
            "holdout_primary": res.holdout_metrics.get(primary),
            "holdout_roc_auc": res.holdout_metrics.get("roc_auc"),
            "holdout_ks": res.holdout_metrics.get("ks_statistic"),
            "calibration_error": res.holdout_metrics.get("calibration_error"),
            "max_flag_rate_disparity": slices.get("max_flag_rate_disparity"),
            "features": ",".join(res.selected_features),
        })

    comparison = pd.DataFrame(rows)
    # rank on holdout only when every run optimised the same primary metric
    if comparison["primary_metric"].nunique() == 1:
        comparison = comparison.sort_values(
            "holdout_primary", ascending=False, kind="mergesort"
        ).reset_index(drop=True)

    out = Path(cfgs[0].run.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(out / "sweep_comparison.csv", index=False)
    return comparison, results


__all__ = ["run_sweep", "check_comparability", "COMPARABILITY_KEYS"]
