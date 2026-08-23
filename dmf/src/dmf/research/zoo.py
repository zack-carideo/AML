"""
Estimator zoo: YAML-declared model specifications -> instantiated estimators.

The zoo is deliberately thin. It resolves a dotted import path, injects the
run's ``random_state`` / ``n_jobs`` where the estimator accepts them, and
applies the declared class-imbalance policy. Everything else about a model --
its hyper-parameters, its family, whether it needs scaled inputs -- is data in
the config file, so adding a challenger architecture never requires a code
change.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..config import Config, ModelSpec


def import_object(dotted: str) -> Any:
    """Resolve ``package.module.ClassName`` to the class object."""
    if "." not in dotted:
        raise ValueError(f"'{dotted}' is not a dotted import path.")
    module_path, _, attr = dotted.rpartition(".")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            f"Could not import '{module_path}' for estimator '{dotted}'. "
            f"Install the dependency or disable the model in the config."
        ) from exc
    if not hasattr(module, attr):
        raise AttributeError(f"'{module_path}' has no attribute '{attr}'.")
    return getattr(module, attr)


def _accepts(cls: Any, param: str) -> bool:
    try:
        return param in inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):  # pragma: no cover
        return False


def _imbalance_kwargs(cls: Any, spec: ModelSpec, y: Optional[np.ndarray]) -> Dict[str, Any]:
    """Translate the declared imbalance policy into estimator-specific kwargs."""
    if not spec.imbalance:
        return {}
    policy = str(spec.imbalance).lower()
    if policy not in {"balanced", "scale_pos_weight", "auto"}:
        raise ValueError(f"models.{spec.estimator}: unsupported imbalance policy '{spec.imbalance}'.")

    if _accepts(cls, "class_weight") and policy in {"balanced", "auto"}:
        return {"class_weight": "balanced"}
    if _accepts(cls, "scale_pos_weight"):
        if y is None:
            return {}
        y = np.asarray(y).ravel()
        pos = float((y == 1).sum())
        neg = float(len(y) - pos)
        return {"scale_pos_weight": round(neg / pos, 6)} if pos else {}
    if _accepts(cls, "is_unbalance"):  # LightGBM alternative
        return {"is_unbalance": True}
    return {}


def build_estimator(
    spec: ModelSpec,
    run_random_state: int = 42,
    n_jobs: int = 1,
    y: Optional[np.ndarray] = None,
) -> Any:
    """Instantiate one estimator from its declared specification."""
    cls = import_object(spec.estimator)
    params: Dict[str, Any] = dict(spec.params)

    if _accepts(cls, "random_state") and "random_state" not in params:
        params["random_state"] = run_random_state
    # sklearn is removing n_jobs from the linear models (it has no effect there
    # since 1.8), so never inject it for that family.
    if (
        _accepts(cls, "n_jobs")
        and "n_jobs" not in params
        and not getattr(cls, "__module__", "").startswith("sklearn.linear_model")
    ):
        params["n_jobs"] = n_jobs
    if _accepts(cls, "verbosity") and "verbosity" not in params:
        params["verbosity"] = 0
    if "LGBM" in cls.__name__:
        params.setdefault("verbose", -1)   # LightGBM reads this out of **kwargs

    for k, v in _imbalance_kwargs(cls, spec, y).items():
        params.setdefault(k, v)

    return cls(**params)


def build_zoo(
    cfg: Config, y: Optional[np.ndarray] = None
) -> Dict[str, Tuple[ModelSpec, Any]]:
    """Instantiate every enabled model. Unimportable models are skipped loudly."""
    zoo: Dict[str, Tuple[ModelSpec, Any]] = {}
    for name, spec in cfg.enabled_models.items():
        try:
            est = build_estimator(spec, cfg.run.random_state, cfg.run.n_jobs, y)
        except ImportError as exc:
            print(f"[zoo] skipping '{name}': {exc}")
            continue
        zoo[name] = (spec, est)
    if not zoo:
        raise RuntimeError("No estimators could be instantiated from the configured zoo.")
    return zoo


def config_for_model(cfg: Config, spec: ModelSpec) -> Config:
    """Per-model preprocessing view of the config.

    Tree ensembles do not need centred/scaled inputs, and forcing a scaler on
    them wastes fit time and obscures split thresholds; linear models do. Rather
    than branch inside the pipeline, each model declares what it needs and gets
    its own preprocessing configuration.
    """
    out = cfg.copy()
    if not spec.requires_scaling:
        out.preprocessing.numeric.scaler = "none"
    for dotted, value in (spec.preprocessing_overrides or {}).items():
        _set_dotted(out.preprocessing, dotted, value)
    out.validate()
    return out


def _set_dotted(root: Any, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    obj = root
    for p in parts[:-1]:
        if not hasattr(obj, p):
            raise ValueError(f"preprocessing_overrides: no such section '{p}' in '{dotted}'.")
        obj = getattr(obj, p)
    leaf = parts[-1]
    if not hasattr(obj, leaf):
        raise ValueError(f"preprocessing_overrides: no such key '{leaf}' in '{dotted}'.")
    setattr(obj, leaf, value)


__all__ = ["import_object", "build_estimator", "build_zoo", "config_for_model"]
