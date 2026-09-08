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

import hashlib
import html
import json
import platform
from dataclasses import dataclass, field
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd


def _num(x: Any, ndigits: Optional[int] = 6) -> Any:
    """Coerce numpy scalars to json-serialisable python types.

    Floats are rounded to ``ndigits`` for readable reports; pass ``None`` to
    keep full precision. Anything applied as a cut -- a decision threshold, a
    reference quantile -- must be written unrounded, or ties at the value are
    silently dropped by the ``>=`` the scorer applies.
    """
    if isinstance(x, (float, np.floating)):
        x = float(x)
        if not np.isfinite(x):
            return None
        return round(x, ndigits) if ndigits is not None else x
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
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


# --------------------------------------------------------------------------
# executive HTML report
# --------------------------------------------------------------------------
# One self-contained page assembled from a run's artifact directory. It reads
# only files -- report_manifest.json, run_report.json, holdout_metrics.json and
# the CSV tables -- so it stays inside the production core and imports nothing
# from dmf.research.
#
# Charts are emitted as inline SVG built from the same numbers as the tables,
# rather than embedded matplotlib PNGs. That keeps the core free of a plotting
# dependency, keeps the file readable at any zoom, and means the page can be
# regenerated from the artifacts alone long after the notebook session is gone.

#: artifact -> heading used when a table is rendered without an explicit caption
_TABLE_CAPTIONS = {
    "leaderboard.csv": "Leaderboard — every (model, k) cell scored by cross-validation",
    "marginal_gains.csv": "Marginal value of the k-th variable (paired, fold-level)",
    "best_per_model.csv": "Best variant per architecture",
    "holdout_deciles.csv": "Holdout gains table by score band",
    "holdout_slices.csv": "Holdout performance and flag rate per segment level",
}

_REPORT_CSS = """
:root{--ink:#14161a;--muted:#5b6472;--line:#e2e6ec;--bg:#ffffff;--panel:#f7f9fc;
      --accent:#1f4fa8;--warn:#a8541f;--bad:#a81f3c;--good:#1f7a4d;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
.wrap{max-width:1080px;margin:0 auto;padding:40px 28px 72px;}
h1{font-size:25px;line-height:1.25;margin:0 0 6px;letter-spacing:-.01em;}
h2{font-size:17px;margin:38px 0 12px;padding-bottom:7px;border-bottom:2px solid var(--line);}
h3{font-size:13px;margin:20px 0 8px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;}
p{margin:0 0 12px;} .sub{color:var(--muted);margin:0 0 22px;}
.meta{display:flex;flex-wrap:wrap;gap:6px 22px;font-size:12px;color:var(--muted);
      padding:12px 0 0;border-top:1px solid var(--line);}
.meta b{color:var(--ink);font-weight:600;}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:22px 0 8px;}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 14px;}
.kpi .k{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;}
.kpi .v{font-size:21px;font-weight:600;margin-top:3px;font-variant-numeric:tabular-nums;}
.kpi .n{font-size:11px;color:var(--muted);margin-top:2px;}
.card{border:1px solid var(--line);border-radius:8px;padding:2px 18px 16px;margin:0 0 20px;}
.kv{display:grid;grid-template-columns:minmax(190px,auto) 1fr;gap:5px 18px;margin:8px 0 4px;font-size:13px;}
.kv dt{color:var(--muted);} .kv dd{margin:0;font-variant-numeric:tabular-nums;}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin:6px 0 4px;}
th,td{text-align:right;padding:5px 9px;border-bottom:1px solid var(--line);
      font-variant-numeric:tabular-nums;white-space:nowrap;}
th{text-align:right;color:var(--muted);font-weight:600;font-size:11px;
   text-transform:uppercase;letter-spacing:.04em;border-bottom:2px solid var(--line);}
th:first-child,td:first-child{text-align:left;} tbody tr:hover{background:var(--panel);}
.scroll{overflow-x:auto;} caption{caption-side:top;text-align:left;color:var(--muted);
        font-size:12px;padding:8px 0 4px;}
.tag{display:inline-block;padding:1px 8px;border-radius:99px;font-size:11px;font-weight:600;}
.tag.good{background:#e6f4ec;color:var(--good);} .tag.warn{background:#fbf0e5;color:var(--warn);}
.tag.bad{background:#fbe9ed;color:var(--bad);} .tag.flat{background:var(--panel);color:var(--muted);}
.prov{font-size:11.5px;color:var(--muted);margin:10px 0 0;}
.prov code{background:var(--panel);padding:1px 5px;border-radius:4px;font-size:11px;}
.note{background:var(--panel);border-left:3px solid var(--accent);padding:10px 14px;
      margin:14px 0;font-size:12.5px;color:var(--muted);}
svg{display:block;max-width:100%;height:auto;margin:8px 0 4px;}
@media print{body{font-size:11.5px}.wrap{padding:0}h2{page-break-after:avoid}
  .card,table,svg{page-break-inside:avoid}.kpi{break-inside:avoid}}
"""


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _fmt(value: Any, digits: int = 4) -> str:
    """Render one scalar for display: floats to ``digits``, ints with separators."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        v = float(value)
        return f"{v:,.{digits}f}".rstrip("0").rstrip(".") if abs(v) < 1e6 else f"{v:,.0f}"
    return str(value)


def _render_value(value: Any, digits: int = 4) -> str:
    """Headline values are scalars, lists, or nested dicts; render each as HTML."""
    if isinstance(value, dict):
        if value and all(isinstance(v, dict) for v in value.values()):
            return _table(pd.DataFrame(value).T.reset_index(names=""), digits=digits)
        return "<dl class='kv'>" + "".join(
            f"<dt>{_esc(k)}</dt><dd>{_render_value(v, digits)}</dd>" for k, v in value.items()
        ) + "</dl>"
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(v, dict) for v in value):
            return _table(pd.DataFrame(list(value)), digits=digits)
        return _esc(", ".join(_fmt(v, digits) for v in value)) or "—"
    return _esc(_fmt(value, digits))


def _table(df: Optional[pd.DataFrame], max_rows: Optional[int] = None,
           caption: Optional[str] = None, digits: int = 4) -> str:
    """A DataFrame as an HTML table, numerically formatted and optionally truncated."""
    if df is None or not len(df):
        return ""
    shown, hidden = (df.head(max_rows), len(df) - max_rows) if max_rows and len(df) > max_rows else (df, 0)
    head = "".join(f"<th>{_esc(c)}</th>" for c in shown.columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{_esc(_fmt(v, digits))}</td>" for v in row) + "</tr>"
        for row in shown.itertuples(index=False, name=None)
    )
    cap = f"<caption>{_esc(caption)}</caption>" if caption else ""
    more = f"<p class='prov'>… {hidden} further row(s) in the source artifact.</p>" if hidden else ""
    return f"<div class='scroll'><table>{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>{more}"


def _svg_bars(pairs: Sequence, title: str = "", digits: int = 3,
              baseline: Optional[float] = None, baseline_label: str = "") -> str:
    """Horizontal bar chart as inline SVG -- no plotting library, no binary payload.

    ``pairs`` is ``[(label, value), ...]``. ``baseline`` draws a reference rule,
    e.g. lift = 1.0 (random) or the population flag rate.
    """
    pairs = [(str(k), float(v)) for k, v in pairs if v is not None and np.isfinite(float(v))]
    if not pairs:
        return ""
    row_h, pad_l, pad_r, top = 22, 150, 62, 26 if title else 6
    width, height = 720, top + row_h * len(pairs) + 10
    span = max(max(v for _, v in pairs), baseline or 0) or 1.0
    plot_w = width - pad_l - pad_r

    parts = [f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{_esc(title or 'chart')}'>"]
    if title:
        parts.append(f"<text x='0' y='14' font-size='12' font-weight='600' fill='#14161a'>{_esc(title)}</text>")
    for i, (label, value) in enumerate(pairs):
        y = top + i * row_h
        w = max(plot_w * value / span, 1.0)
        parts.append(
            f"<text x='{pad_l - 8}' y='{y + 14}' font-size='11.5' text-anchor='end' fill='#5b6472'>{_esc(label[:34])}</text>"
            f"<rect x='{pad_l}' y='{y + 4}' width='{w:.1f}' height='14' rx='2.5' fill='#1f4fa8' opacity='0.82'/>"
            f"<text x='{pad_l + w + 7:.1f}' y='{y + 15}' font-size='11' fill='#14161a'>{_esc(_fmt(value, digits))}</text>"
        )
    if baseline is not None and np.isfinite(baseline) and span:
        x = pad_l + plot_w * float(baseline) / span
        parts.append(
            f"<line x1='{x:.1f}' y1='{top}' x2='{x:.1f}' y2='{height - 12}' "
            f"stroke='#a81f3c' stroke-width='1.2' stroke-dasharray='3 3'/>"
            f"<text x='{x + 4:.1f}' y='{height - 2}' font-size='10' fill='#a81f3c'>{_esc(baseline_label)}</text>"
        )
    return "".join(parts) + "</svg>"


def _kpi(label: str, value: Any, note: str = "", digits: int = 4) -> str:
    note_html = f"<div class='n'>{_esc(note)}</div>" if note else ""
    return (f"<div class='kpi'><div class='k'>{_esc(label)}</div>"
            f"<div class='v'>{_esc(_fmt(value, digits))}</div>{note_html}</div>")


def _psi_tag(psi: Any) -> str:
    """PSI banded the way psi_band does, as a coloured chip."""
    if psi is None or not np.isfinite(float(psi)):
        return "<span class='tag flat'>unknown</span>"
    psi = float(psi)
    cls, word = (("good", "stable") if psi < 0.10 else
                 ("warn", "watch") if psi < 0.25 else ("bad", "investigate"))
    return f"<span class='tag {cls}'>{word} · {_fmt(psi, 3)}</span>"


def _read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except (OSError, ValueError):
        return None


def executive_report(
    run_dir: Union[str, Path],
    out_path: Optional[Union[str, Path]] = None,
    title: str = "Model Development Report",
    subtitle: str = "",
    max_leaderboard_rows: int = 12,
) -> Path:
    """Render one run's artifacts into a single self-contained executive HTML page.

    Reads whatever the run wrote and skips what it did not, so this works on any
    artifact directory :class:`~dmf.research.selection.ModelSelectionHarness`
    produced:

    * ``report_manifest.json`` -- section spine and headline numbers, when the
      research walkthrough wrote one. Without it the page still renders from the
      run report and the metric artifacts, just without the narrative sections.
    * ``run_report.json`` -- lineage, split design, selection rule, timings.
    * ``holdout_metrics.json`` -- the headline out-of-sample numbers and the
      shipped decision threshold.
    * ``leaderboard.csv``, ``marginal_gains.csv``, ``holdout_deciles.csv``,
      ``holdout_slices.csv`` -- the tables, rendered inline.

    The output has no external CSS, JS, fonts or images: charts are inline SVG
    generated from the same numbers shown in the tables, so the file can be
    emailed, printed, or attached to a model-risk submission and still render.

    Parameters
    ----------
    run_dir : path
        A run's artifact directory (the one holding ``run_report.json``).
    out_path : path, optional
        Where to write. Defaults to ``<run_dir>/executive_report.html``.
    title, subtitle : str
        Page heading and the line beneath it.
    max_leaderboard_rows : int
        Truncate the leaderboard table; the full grid stays in the CSV.

    Returns
    -------
    Path
        The file written.
    """
    run = Path(run_dir)
    if not run.is_dir():
        raise NotADirectoryError(f"No artifact directory at '{run}'.")
    out = Path(out_path) if out_path else run / "executive_report.html"

    manifest = _read_json(run / "report_manifest.json") or {}
    report = _read_json(run / "run_report.json") or {}
    holdout = _read_json(run / "holdout_metrics.json") or {}
    steps = {s.get("step"): s for s in report.get("steps", [])}
    meta = manifest.get("run", {})
    selected = meta.get("selected", {}) or steps.get("selection", {})

    body: List[str] = []

    # ---------------- header ----------------
    lineage = steps.get("lineage", {})
    facts = [
        ("run", meta.get("name") or report.get("run")),
        ("model", selected.get("model") or selected.get("selected_model")),
        ("variables", selected.get("k") or selected.get("selected_k")),
        ("selection rule", selected.get("rule")),
        ("dmf", meta.get("dmf_version")),
        ("config sha256", meta.get("config_sha256") or lineage.get("config_sha256")),
        ("data sha256", meta.get("data_sha256") or lineage.get("data_sha256")),
        ("generated", datetime.now().strftime("%Y-%m-%d %H:%M")),
    ]
    body.append(f"<h1>{_esc(title)}</h1>")
    if subtitle:
        body.append(f"<p class='sub'>{_esc(subtitle)}</p>")
    body.append("<div class='meta'>" + "".join(
        f"<span>{_esc(k)} <b>{_esc(_fmt(v))}</b></span>" for k, v in facts if v not in (None, "")
    ) + "</div>")

    if features := (selected.get("features") or []):
        body.append(f"<p class='prov'>Specification: <code>{_esc(', '.join(map(str, features)))}</code></p>")

    # ---------------- at a glance ----------------
    confirm = steps.get("holdout_confirmation", {})
    slices_step = steps.get("holdout_slices", {})
    sampling = steps.get("training_sampling", {})
    primary = next((k for k in ("average_precision", "roc_auc") if k in holdout), None)
    kpis = [
        _kpi("Holdout AP", holdout.get("average_precision"), "primary metric"),
        _kpi("Holdout ROC-AUC", holdout.get("roc_auc")),
        _kpi("KS", holdout.get("ks_statistic")),
        _kpi("CV − holdout", confirm.get("cv_minus_holdout"),
             "within 1 SE" if confirm.get("within_one_se") else "outside 1 SE"),
        _kpi("Decision threshold", holdout.get("decision_threshold"),
             f"{holdout.get('decision_threshold_policy', '')} @ "
             f"{_fmt(holdout.get('decision_operating_point'))}", digits=6),
        _kpi("Flag rate (shipped)", holdout.get("decision_flag_rate_shipped") or holdout.get("decision_flag_rate")),
        _kpi("Precision at cut", holdout.get("decision_precision")),
        _kpi("Recall at cut", holdout.get("decision_recall")),
    ]
    if sampling.get("enabled"):
        kpis += [
            _kpi("Training sample rate", sampling.get("sampling_rate"),
                 f"{_fmt(sampling.get('n_train_fitted'))} of "
                 f"{_fmt(sampling.get('n_train_available'))} rows"),
            _kpi("Prevalence shift", sampling.get("prevalence_ratio"), "fitted / source"),
        ]
    body.append("<div class='kpis'>" + "".join(kpis) + "</div>")

    cal = holdout.get("calibration_ratio")
    if cal is not None and np.isfinite(float(cal)) and abs(float(cal) - 1.0) > 0.25:
        body.append(
            f"<div class='note'><b>Scores are not calibrated probabilities.</b> Mean predicted "
            f"{_fmt(holdout.get('mean_predicted'))} against prevalence {_fmt(holdout.get('prevalence'))} "
            f"(ratio {_fmt(cal, 2)}, ECE {_fmt(holdout.get('calibration_error'))}). Ranking and the "
            f"capacity cut are unaffected; any use that multiplies score by exposure needs a "
            f"post-hoc calibrator first.</div>"
        )

    # placed next to the calibration note on purpose: when sampling is on, it is
    # a large part of why that note is there
    if sampling.get("prior_shifted"):
        body.append(
            f"<div class='note'><b>Training rows were undersampled.</b> The model was fitted "
            f"on {_fmt(sampling.get('n_train_fitted'))} of "
            f"{_fmt(sampling.get('n_train_available'))} available rows "
            f"(rate {_fmt(sampling.get('sampling_rate'))}), moving the training prevalence "
            f"from {_fmt(sampling.get('train_prevalence_available'))} to "
            f"{_fmt(sampling.get('train_prevalence_fitted'))} — a log-odds shift of "
            f"{_fmt(sampling.get('prior_shift_logit'))}. Validation folds and the holdout "
            f"kept every row, so the metrics above remain out-of-sample measurements on the "
            f"full population; the predicted <i>scores</i>, however, are raised throughout "
            f"and are not probabilities on the source population. A calibrator for them must "
            f"be fitted on unsampled rows."
            + (f" Coverage retained: {_fmt(sampling.get('coverage_retained_categorical'))} of "
               f"categorical levels, {_fmt(sampling.get('coverage_retained_numeric'))} of "
               f"numeric range." if sampling.get("coverage_retained_categorical") is not None else "")
            + "</div>"
        )

    # ---------------- narrative sections from the manifest ----------------
    tables = {name: _read_csv(run / name) for name in _TABLE_CAPTIONS}
    for section in manifest.get("sections", []):
        name = str(section.get("section", "Section"))
        body.append(f"<h2>{_esc(name)}</h2><div class='card'>")
        body.append(_render_value(section.get("headline", {})))

        lowered = name.lower()
        if "holdout gains" in lowered or "parity" in lowered:
            body.append(_deciles_chart(tables.get("holdout_deciles.csv")))
            body.append(_slices_chart(tables.get("holdout_slices.csv")))
        elif "out-of-sample" in lowered or "training" in lowered:
            body.append(_table(tables.get("leaderboard.csv"), max_leaderboard_rows,
                               _TABLE_CAPTIONS["leaderboard.csv"]))
        elif "monitoring" in lowered:
            body.append(_psi_summary(section.get("headline", {})))

        if artifacts := section.get("artifacts"):
            body.append("<p class='prov'>Source: " + " · ".join(
                f"<code>{_esc(a)}</code>" for a in artifacts) + "</p>")
        body.append("</div>")

    # ---------------- fallback + always-on detail ----------------
    if not manifest.get("sections"):
        body.append("<h2>Leaderboard</h2><div class='card'>")
        body.append(_table(tables.get("leaderboard.csv"), max_leaderboard_rows,
                           _TABLE_CAPTIONS["leaderboard.csv"]))
        body.append("</div>")
        body.append("<h2>Holdout</h2><div class='card'>")
        body.append(_deciles_chart(tables.get("holdout_deciles.csv")))
        body.append(_slices_chart(tables.get("holdout_slices.csv")))
        body.append("</div>")

    body.append("<h2>Supporting tables</h2>")
    for fname in ("marginal_gains.csv", "holdout_slices.csv", "holdout_deciles.csv"):
        if (frame := tables.get(fname)) is not None and len(frame):
            body.append(f"<div class='card'>{_table(frame, 25, _TABLE_CAPTIONS[fname])}</div>")

    # ---------------- provenance ----------------
    body.append("<h2>Provenance</h2><div class='card'>")
    packages = lineage.get("packages", {})
    body.append(_render_value({
        "python": lineage.get("python"),
        "platform": lineage.get("platform"),
        "packages": packages,
        "rows / columns": f"{_fmt(lineage.get('data_rows'))} / {_fmt(lineage.get('data_columns'))}",
        "config sha256": lineage.get("config_sha256"),
        "data sha256": lineage.get("data_sha256"),
        "artifact directory": str(run),
    }))
    body.append("<p class='prov'>Every number on this page was read from the artifacts in that "
                "directory; none was transcribed by hand.</p></div>")

    page = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{_esc(title)}</title><style>{_REPORT_CSS}</style></head>"
        f"<body><div class='wrap'>{''.join(body)}</div></body></html>"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out


def _deciles_chart(deciles: Optional[pd.DataFrame]) -> str:
    """Lift by score band, against the lift=1 random baseline."""
    if deciles is None or not len(deciles) or "lift" not in deciles:
        return ""
    pairs = [(f"band {int(r.band)}", r.lift) for r in deciles.itertuples()]
    return _svg_bars(pairs, "Lift by score band (holdout)", baseline=1.0, baseline_label="random = 1.0")


def _slices_chart(slices: Optional[pd.DataFrame]) -> str:
    """Flag rate per segment level -- the fair-treatment parity view."""
    if slices is None or not len(slices) or "flag_rate_at_top_pct" not in slices:
        return ""
    pairs = [(f"{r.slice_column} · {r.level}", r.flag_rate_at_top_pct) for r in slices.itertuples()]
    overall = float(slices["flag_rate_at_top_pct"].mean())
    return _svg_bars(pairs, "Flag rate at the review budget, by segment level",
                     baseline=overall, baseline_label="mean")


def _psi_summary(headline: Dict[str, Any]) -> str:
    """Monitoring batches rendered with their PSI band as a coloured chip."""
    batches = headline.get("batches")
    if not isinstance(batches, dict) or not batches:
        return ""
    rows = "".join(
        f"<tr><td>{_esc(name)}</td><td>{_esc(_fmt(v.get('n')))}</td>"
        f"<td>{_esc(_fmt(v.get('flag_rate')))}</td><td>{_psi_tag(v.get('psi'))}</td></tr>"
        for name, v in batches.items() if isinstance(v, dict)
    )
    return ("<div class='scroll'><table><caption>Score-distribution stability against the "
            "bundled reference</caption><thead><tr><th>batch</th><th>n</th><th>flag rate</th>"
            f"<th>PSI</th></tr></thead><tbody>{rows}</tbody></table></div>")


__all__ = [
    "StepReport", "summarize_frame", "summarize_matrix", "summarize_target",
    "json_safe", "run_lineage", "executive_report",
]
