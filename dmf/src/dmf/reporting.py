"""
Quantitative step reporting.

Every stage of the feature pipeline and every stage of the selection harness
emits a machine-readable summary dict. Two rules keep these useful:

1. Numbers, not prose. Row/column counts, missing rates, cardinality, moments,
   information value, score deltas -- things you can diff between two runs or
   monitor in production.
2. Reports are attached to the *fitted* object (``fit_report_``), so a pickled
   model carries the provenance of its own training statistics with it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _num(x: Any, ndigits: Optional[int] = 6) -> Any:
    """Coerce numpy scalars to json-serialisable python types.

    Floats are rounded to ``ndigits`` for readable reports; pass ``None`` to
    keep full precision. Anything applied as a cut -- a decision threshold, a
    reference quantile -- must be written unrounded, or ties at the value are
    silently dropped by the ``>=`` the scorer applies.
    """
    if x is None:
        return None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return None if not np.isfinite(v) else (round(v, ndigits) if ndigits is not None else v)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, float):
        return None if not np.isfinite(x) else (round(x, ndigits) if ndigits is not None else x)
    return x


def json_safe(obj: Any, ndigits: Optional[int] = 6) -> Any:
    """Recursively coerce a structure into something json.dumps can handle."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v, ndigits) for v in obj]
    if isinstance(obj, (np.ndarray,)):
        return [json_safe(v, ndigits) for v in obj.tolist()]
    if isinstance(obj, pd.Series):
        return {str(k): json_safe(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, pd.DataFrame):
        return json_safe(obj.to_dict(orient="records"), ndigits)
    return _num(obj, ndigits)


# --------------------------------------------------------------------------
# frame / matrix profiling
# --------------------------------------------------------------------------
def summarize_matrix(X: Any, name: str = "matrix", max_named: int = 40) -> Dict[str, Any]:
    """Shape / sparsity summary of a design matrix or DataFrame.

    Deliberately minimal: this profiles the matrix *after* imputation and
    scaling, where missingness is zero and moments are near-constant by
    construction; the informative per-column statistics live in the raw-input
    profile (:func:`summarize_frame`) and the per-step transformer reports.
    """
    df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.atleast_2d(np.asarray(X, dtype=float)))
    arr = df.to_numpy(dtype=float, na_value=np.nan) if df.shape[1] else np.empty((len(df), 0))
    with np.errstate(invalid="ignore"):
        col_std = np.nanstd(arr, axis=0) if arr.size else np.empty(0)
    columns = [str(c) for c in df.columns]
    return {
        "name": name,
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "missing_rate": _num(1.0 - np.isfinite(arr).mean()) if arr.size else 0.0,
        "n_constant_columns": int(np.sum(np.nan_to_num(col_std) == 0.0)),
        ("columns" if len(columns) <= max_named else "columns_head"): columns[:max_named],
    }


def summarize_frame(df: pd.DataFrame, name: str = "frame") -> Dict[str, Any]:
    """Column-role-aware profile of a raw input frame."""
    n_rows = len(df)
    per_col = {}
    for col in df.columns:
        s = df[col]
        entry: Dict[str, Any] = {
            "dtype": str(s.dtype),
            "missing_rate": _num(s.isna().mean()) if n_rows else 0.0,
            "n_unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s):
            entry.update(
                mean=_num(s.mean()),
                std=_num(s.std()),
                p01=_num(s.quantile(0.01)) if n_rows else None,
                p50=_num(s.median()) if n_rows else None,
                p99=_num(s.quantile(0.99)) if n_rows else None,
                skew=_num(s.skew()) if n_rows > 2 else None,
            )
        else:
            vc = s.astype("object").value_counts(normalize=True, dropna=True)
            entry["top_level"] = str(vc.index[0]) if len(vc) else None
            entry["top_level_share"] = _num(vc.iloc[0]) if len(vc) else None
        per_col[str(col)] = entry
    return {
        "name": name,
        "n_rows": int(n_rows),
        "n_columns": int(df.shape[1]),
        "total_missing_rate": _num(df.isna().to_numpy().mean()) if df.size else 0.0,
        "duplicate_row_rate": _num(df.duplicated().mean()) if n_rows else 0.0,
        "columns": per_col,
    }


def summarize_target(y: Any, name: str = "target") -> Dict[str, Any]:
    s = pd.Series(np.asarray(y).ravel())
    vc = s.value_counts(dropna=False)
    pos = int((s == 1).sum())
    n = int(len(s))
    return {
        "name": name,
        "n_rows": n,
        "n_positive": pos,
        "n_negative": n - pos,
        "prevalence": _num(pos / n) if n else None,
        "imbalance_ratio": _num((n - pos) / pos) if pos else None,
        "class_counts": {str(k): int(v) for k, v in vc.items()},
    }


# --------------------------------------------------------------------------
# report container
# --------------------------------------------------------------------------
@dataclass
class StepReport:
    """Ordered collection of per-step quantitative summaries."""

    run: str = "run"
    steps: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, step: str, **payload: Any) -> Dict[str, Any]:
        entry = {"step": step, **json_safe(payload)}
        self.steps.append(entry)
        return entry

    def get(self, step: str) -> Optional[Dict[str, Any]]:
        return next((e for e in self.steps if e.get("step") == step), None)

    def to_dict(self) -> Dict[str, Any]:
        return {"run": self.run, "steps": self.steps}

    def to_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        text = json.dumps(json_safe(self.to_dict()), indent=indent)
        if path:
            with open(path, "w") as fh:
                fh.write(text)
        return text

    def to_frame(self) -> pd.DataFrame:
        """Flat one-row-per-step view for quick eyeballing / logging."""
        rows = []
        for entry in self.steps:
            row = {"step": entry.get("step")}
            for k, v in entry.items():
                if k == "step":
                    continue
                if isinstance(v, (int, float, str, bool)) or v is None:
                    row[k] = v
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, (int, float, str, bool)) or vv is None:
                            row[f"{k}.{kk}"] = vv
            rows.append(row)
        return pd.DataFrame(rows)

    def render(self, max_width: int = 100) -> str:
        """Human-readable log block; one line of numbers per step."""
        lines = [f"=== step report: {self.run} ==="]
        for entry in self.steps:
            head = f"[{entry.get('step')}]"
            bits = []
            for k, v in entry.items():
                if k == "step":
                    continue
                if isinstance(v, (int, float, bool)) or v is None:
                    bits.append(f"{k}={v}")
                elif isinstance(v, str) and len(v) < 40:
                    bits.append(f"{k}={v}")
            line = f"{head} " + "  ".join(bits)
            lines.append(line if len(line) <= max_width else line[: max_width - 3] + "...")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.steps)


def run_lineage(config_dict: Dict[str, Any], df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Provenance a model-risk reviewer will ask for six months from now.

    Library versions, a hash of the exact configuration, and a fingerprint of
    the training frame -- enough to answer "was this the same code, the same
    settings and the same data?" without keeping a copy of the data.
    """
    import hashlib
    import platform
    from importlib import metadata

    versions = {}
    for pkg in ("numpy", "pandas", "scikit-learn", "scipy", "xgboost", "lightgbm"):
        try:
            versions[pkg] = metadata.version(pkg)
        except Exception:
            continue

    out: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "config_sha256": hashlib.sha256(
            json.dumps(json_safe(config_dict), sort_keys=True).encode()
        ).hexdigest()[:16],
    }
    if df is not None:
        try:
            digest = hashlib.sha256(
                pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()
            ).hexdigest()[:16]
        except Exception:  # pragma: no cover - exotic dtypes
            digest = None
        out.update(
            data_rows=int(len(df)),
            data_columns=int(df.shape[1]),
            data_sha256=digest,
            column_names_sha256=hashlib.sha256(
                ",".join(map(str, df.columns)).encode()
            ).hexdigest()[:16],
        )
    return out


__all__ = [
    "StepReport", "summarize_frame", "summarize_matrix", "summarize_target",
    "json_safe", "run_lineage",
]
