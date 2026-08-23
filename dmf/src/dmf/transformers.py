"""
Column typing, lenient value parsing, and the leakage-safe transformers.

This module is the whole of "how a raw column becomes model input": one place
decides what each column *is* (:func:`infer_roles`), how its values are read
(:func:`parse_kind` / :func:`to_numeric_lenient`), and how it is transformed
(the estimator classes below). Keeping typing and transformation together means
the candidate list a selection harness searches over and the roles a fitted
pipeline uses can never disagree.

The parsing rules exist because of things real feed data does:

* amounts arrive as ``"$1,234.50"`` -- typed naively they become a
  1500-level categorical and a strong predictor is silently destroyed;
* dates arrive as ISO strings or ``datetime64`` -- as categories they are
  noise, as epoch-days they are a usable recency signal;
* a ratio divides by zero and yields ``inf``, which no scaler will accept;
* a column is 100% null, or constant, or is an identifier with one level per
  row -- all of which consume budget and none of which can carry signal.

Every transformer:

* follows the sklearn estimator contract (``__init__`` only assigns params,
  everything learned ends in a trailing-underscore attribute),
* implements ``get_feature_names_out`` so downstream attribution maps encoded
  columns back to source variables,
* exposes a ``report_`` dict with the quantitative summary of what the fit
  learned, which the pipeline harvests into its ``fit_report_``.

Because every learned statistic comes from ``fit`` only, placing these inside
an sklearn ``Pipeline`` guarantees they are re-estimated inside each CV fold
and never see validation rows -- including the supervised WOE encoder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# currency symbols, thousands separators, percent signs, ordinary and nbsp space
_NUMERIC_JUNK = re.compile(r"[,$£€%\s ]")

#: per-row guard counters, in the order they appear in scoring output
ROW_FLAG_COLUMNS = ["n_out_of_range", "n_unseen_category", "n_coerced", "n_newly_missing"]


def ensure_frame(X: Any, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """The one place array-like input becomes a DataFrame."""
    if isinstance(X, pd.DataFrame):
        return X
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if columns is None:
        columns = [f"x{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=list(columns))


# ==========================================================================
# value parsing
# ==========================================================================
def parse_kind(s: pd.Series, threshold: float = 0.95) -> Tuple[Optional[str], float]:
    """Classify how a column's *values* should be read.

    Returns ``("numeric" | "datetime" | None, parse_rate)``. A column qualifies
    when at least ``threshold`` of its non-null values parse cleanly, so a
    mostly-numeric column with a few ``"N/A"`` sentinels still counts as numeric.
    """
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
        return "numeric", 1.0
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime", 1.0

    non_null = s.dropna()
    if non_null.empty:
        return "numeric", 1.0                      # all-null: nothing to contradict it

    text = non_null.astype(str)
    as_num = pd.to_numeric(text.str.replace(_NUMERIC_JUNK, "", regex=True), errors="coerce")
    rate = float(as_num.notna().mean())
    if rate >= threshold:
        return "numeric", rate

    # only try dates when the text actually looks like one -- guessing on free
    # text is slow and produces nonsense matches
    if text.str.contains(r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}", regex=True).mean() >= threshold:
        as_dt = pd.to_datetime(text, errors="coerce", format="mixed")
        if float(as_dt.notna().mean()) >= threshold:
            return "datetime", float(as_dt.notna().mean())
    return None, rate


def to_numeric_lenient(s: pd.Series, kind: Optional[str] = None) -> pd.Series:
    """Coerce a column to finite floats, whatever shape it arrived in.

    Handles native numerics, ``datetime64`` and date strings (as epoch days),
    and text-formatted numbers. Anything unparseable -- and ``+/-inf``, which is
    a number no estimator can consume -- becomes ``NaN`` for the fitted imputer
    to fill.
    """
    if kind is None:
        kind, _ = parse_kind(s)

    # numeric dtype first, which makes this function idempotent: a datetime
    # column already converted to epoch days must not be re-parsed as a date.
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        out = s.astype("float64")
    elif kind == "datetime" or pd.api.types.is_datetime64_any_dtype(s):
        dt = s if pd.api.types.is_datetime64_any_dtype(s) else pd.to_datetime(
            s, errors="coerce", format="mixed"
        )
        if getattr(dt.dtype, "tz", None) is not None:
            dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
        # subtract-and-divide rather than a raw int64 view: pandas stores
        # datetimes at ns, us or s resolution depending on version and source,
        # and only this form is unit-agnostic.
        out = (dt - pd.Timestamp("1970-01-01")) / pd.Timedelta(days=1)
        out = pd.Series(np.asarray(out, dtype="float64"), index=s.index)
    else:
        out = pd.to_numeric(
            s.astype(str).str.replace(_NUMERIC_JUNK, "", regex=True), errors="coerce"
        ).astype("float64")

    return out.replace([np.inf, -np.inf], np.nan)


def frame_to_numeric(df: pd.DataFrame, kinds: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    kinds = kinds or {}
    return pd.DataFrame(
        {c: to_numeric_lenient(df[c], kinds.get(str(c))) for c in df.columns}, index=df.index
    )


# ==========================================================================
# role assignment
# ==========================================================================
@dataclass
class Roles:
    numeric: List[str] = field(default_factory=list)
    categorical: List[str] = field(default_factory=list)
    passthrough: List[str] = field(default_factory=list)
    kinds: Dict[str, str] = field(default_factory=dict)        # numeric col -> numeric|datetime
    dropped: Dict[str, str] = field(default_factory=dict)      # column -> reason
    recovered: List[str] = field(default_factory=list)         # text columns parsed as numeric

    @property
    def all(self) -> List[str]:
        return self.numeric + self.categorical + self.passthrough

    def report(self) -> Dict[str, Any]:
        by_reason: Dict[str, List[str]] = {}
        for col, reason in self.dropped.items():
            by_reason.setdefault(reason, []).append(col)
        return {
            "n_numeric": len(self.numeric),
            "n_categorical": len(self.categorical),
            "n_passthrough": len(self.passthrough),
            "n_dropped": len(self.dropped),
            "numeric": self.numeric,
            "categorical": self.categorical,
            "passthrough": self.passthrough,
            "parsed_as_datetime": sorted(c for c, k in self.kinds.items() if k == "datetime"),
            "recovered_from_text": sorted(self.recovered),
            "dropped_by_reason": by_reason,
        }


def infer_roles(
    df: pd.DataFrame,
    cfg: Any,
    features: Optional[Sequence[str]] = None,
) -> Roles:
    """Assign every usable column a role.

    ``features`` restricts the result to an explicit variable specification. A
    column named there is always honoured -- quality drops (constant,
    identifier-like, all-null) only apply to columns the framework chose for
    itself, so a requested specification can never come back a different width
    than requested.
    """
    cols_cfg = cfg.columns
    target = getattr(cfg.data, "target", None)
    excluded = set(cols_cfg.drop) | ({target} if target else set())
    requested = None if features is None else [f for f in features]

    declared = {c: "numeric" for c in cols_cfg.numeric}
    declared.update({c: "categorical" for c in cols_cfg.categorical})
    declared.update({c: "passthrough" for c in cols_cfg.passthrough})

    duplicated = df.columns[df.columns.duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(
            f"Duplicate column names in the input frame: {duplicated}. "
            f"A join that duplicates a key produces these; de-duplicate upstream, "
            f"since a name can only map to one role."
        )

    roles = Roles()
    n = max(len(df), 1)

    for col in df.columns:
        col = str(col)
        if col in excluded:
            continue
        if requested is not None and col not in requested:
            continue

        role = declared.get(col)
        if role is None and not cols_cfg.auto_infer:
            continue

        s = df[col]
        explicit = requested is not None and col in requested

        if role == "passthrough":
            roles.passthrough.append(col)
            continue

        kind, rate = parse_kind(s, cols_cfg.numeric_parse_threshold)

        # ---- quality gates (self-chosen columns only) ----
        if not explicit:
            if s.notna().sum() == 0:
                roles.dropped[col] = "all_missing"
                continue
            if cols_cfg.drop_constant and s.nunique(dropna=True) <= 1:
                roles.dropped[col] = "constant"
                continue
            looks_categorical = role == "categorical" or (role is None and kind is None)
            if looks_categorical and s.nunique(dropna=True) / n > cols_cfg.max_categorical_cardinality_ratio:
                roles.dropped[col] = "identifier_like_cardinality"
                continue

        # ---- role ----
        if role == "numeric" or (role is None and kind in {"numeric", "datetime"}
                                 and not _small_integer_code(s, cols_cfg)):
            roles.numeric.append(col)
            roles.kinds[col] = kind or "numeric"
            if kind == "numeric" and not pd.api.types.is_numeric_dtype(s) \
                    and not pd.api.types.is_datetime64_any_dtype(s):
                roles.recovered.append(col)
        else:
            roles.categorical.append(col)

    if requested is not None:                      # preserve the requested order
        order = {c: i for i, c in enumerate(requested)}
        for lst in (roles.numeric, roles.categorical, roles.passthrough):
            lst.sort(key=lambda c: order.get(c, 0))
    return roles


def _small_integer_code(s: pd.Series, cols_cfg: Any) -> bool:
    """Integer-coded columns with very few levels behave like categories."""
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return False
    vals = s.dropna()
    if vals.empty or vals.nunique() > min(10, cols_cfg.auto_infer_max_cardinality):
        return False
    arr = vals.to_numpy(dtype=float)
    return bool(np.all(np.isfinite(arr)) and np.all(np.mod(arr, 1) == 0))


# ==========================================================================
# shared estimator plumbing
# ==========================================================================
class _NativeFrameOutput:
    """Marker for transformers that always emit a correctly-named DataFrame.

    They opt out of sklearn's ``set_output`` auto-wrapping (which renames
    columns from ``get_feature_names_out`` and therefore cannot cope with a
    transform whose width depends on the incoming batch), but still need a
    no-op ``set_output`` so an enclosing ``Pipeline.set_output`` succeeds.
    """

    def set_output(self, *, transform=None):
        return self


class _NamedMixin:
    """Shared feature-name bookkeeping."""

    feature_names_in_: np.ndarray

    def _record_names(self, X: Any) -> pd.DataFrame:
        df = ensure_frame(X)
        self.feature_names_in_ = np.asarray([str(c) for c in df.columns], dtype=object)
        self.n_features_in_ = df.shape[1]
        return df

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        if input_features is not None:
            return np.asarray([str(c) for c in input_features], dtype=object)
        return np.asarray(self.feature_names_in_, dtype=object)


# ==========================================================================
# numeric transformers
# ==========================================================================
class NumericCoercer(_NamedMixin, BaseEstimator, TransformerMixin):
    """First stage of the numeric branch: make every column a finite float.

    Currency-formatted text, sentinel strings, dates and ``+/-inf`` all become
    either a real number or ``NaN`` here, so every transformer downstream can
    assume clean float input. The parse rate learned at fit time is reported,
    which is how a feed that starts sending ``"1.234,50"`` instead of
    ``"1,234.50"`` becomes visible rather than silently becoming all-missing.
    """

    def __init__(self, kinds: Optional[Dict[str, str]] = None):
        self.kinds = kinds

    def fit(self, X, y=None):
        df = self._record_names(X)
        kinds = self.kinds or {}
        out = frame_to_numeric(df, kinds)
        self.parse_rate_ = {str(c): float(out[c].notna().mean()) for c in df.columns}
        self.report_ = {
            "transformer": "NumericCoercer",
            "n_columns": int(df.shape[1]),
            "mean_parse_rate": round(float(np.mean(list(self.parse_rate_.values()))), 6)
            if self.parse_rate_ else None,
            "n_columns_parsed_as_datetime": sum(1 for k in kinds.values() if k == "datetime"),
            "columns_below_90pct_parsed": {
                c: round(r, 4) for c, r in self.parse_rate_.items() if r < 0.9
            },
            "n_non_finite_replaced": int(
                np.isinf(df.select_dtypes("number").to_numpy(dtype=float, na_value=np.nan)).sum()
            ) if df.select_dtypes("number").shape[1] else 0,
        }
        return self

    def transform(self, X):
        check_is_fitted(self, "parse_rate_")
        return frame_to_numeric(ensure_frame(X, self.feature_names_in_), self.kinds or {})


class QuantileWinsorizer(_NamedMixin, BaseEstimator, TransformerMixin):
    """Clip each numeric column to train-set quantiles.

    Outlier control that is *learned* rather than hard-coded: the clipping
    bounds become part of the fitted artifact, so a production record with an
    extreme amount is clipped to exactly the same bound the model was trained
    under. ``report_['clipped_rate']`` quantifies how much of the training data
    each bound actually touched.
    """

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        df = frame_to_numeric(self._record_names(X))
        self.lower_bounds_ = np.array(df.quantile(self.lower_quantile).to_numpy(dtype=float), copy=True)
        self.upper_bounds_ = np.array(df.quantile(self.upper_quantile).to_numpy(dtype=float), copy=True)
        # degenerate columns: keep bounds non-inverted
        bad = ~np.isfinite(self.lower_bounds_) | ~np.isfinite(self.upper_bounds_)
        self.lower_bounds_[bad] = -np.inf
        self.upper_bounds_[bad] = np.inf

        arr = df.to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            clipped = (arr < self.lower_bounds_) | (arr > self.upper_bounds_)
        n_valid = np.isfinite(arr).sum()
        self.report_ = {
            "transformer": "QuantileWinsorizer",
            "n_columns": int(df.shape[1]),
            "lower_quantile": self.lower_quantile,
            "upper_quantile": self.upper_quantile,
            "clipped_rate": float(clipped.sum() / n_valid) if n_valid else 0.0,
            "clipped_rate_by_column": {
                str(c): float(clipped[:, i].mean()) for i, c in enumerate(df.columns)
            },
        }
        return self

    def transform(self, X):
        check_is_fitted(self, "lower_bounds_")
        df = frame_to_numeric(ensure_frame(X, self.feature_names_in_))
        out = np.clip(df.to_numpy(dtype=float), self.lower_bounds_, self.upper_bounds_)
        return pd.DataFrame(out, columns=list(df.columns), index=df.index)


# ==========================================================================
# categorical transformers
# ==========================================================================
class RareCategoryCollapser(_NamedMixin, BaseEstimator, TransformerMixin):
    """Fold train-set-infrequent levels into a single ``other_label``.

    High-cardinality fields (merchant, MCC, device fingerprint) otherwise
    generate one-hot columns supported by a handful of rows, which inflates
    variance and makes the model brittle to unseen levels at inference time.
    """

    def __init__(self, min_frequency: float = 0.01, other_label: str = "__RARE__"):
        self.min_frequency = min_frequency
        self.other_label = other_label

    def fit(self, X, y=None):
        df = self._record_names(X).astype("object")
        self.keep_levels_: Dict[str, set] = {}
        detail = {}
        for col in df.columns:
            freq = df[col].value_counts(normalize=True, dropna=True)
            keep = set(freq[freq >= self.min_frequency].index)
            if not keep and len(freq):           # never collapse everything
                keep = {freq.index[0]}
            self.keep_levels_[str(col)] = keep
            detail[str(col)] = {
                "n_levels_in": int(len(freq)),
                "n_levels_kept": int(len(keep)),
                "collapsed_mass": float(freq[~freq.index.isin(list(keep))].sum()),
            }
        self.report_ = {
            "transformer": "RareCategoryCollapser",
            "min_frequency": self.min_frequency,
            "n_columns": int(df.shape[1]),
            "n_levels_in": int(sum(d["n_levels_in"] for d in detail.values())),
            "n_levels_kept": int(sum(d["n_levels_kept"] for d in detail.values())),
            "by_column": detail,
        }
        return self

    def transform(self, X):
        check_is_fitted(self, "keep_levels_")
        df = ensure_frame(X, self.feature_names_in_).astype("object").copy()
        for col in df.columns:
            keep = self.keep_levels_.get(str(col), set())
            df.loc[~df[col].isin(list(keep)), col] = self.other_label
        return df


class WOEEncoder(_NamedMixin, BaseEstimator, TransformerMixin):
    """Weight-of-Evidence encoding with Information Value reporting.

    For level :math:`l` of variable :math:`v`, with Laplace smoothing ``s`` over
    ``L`` observed levels:

    .. math::
        p_l = \\frac{n^{+}_l + s}{N^{+} + sL}, \\quad
        q_l = \\frac{n^{-}_l + s}{N^{-} + sL}, \\quad
        \\mathrm{WOE}_l = \\ln(p_l / q_l)

    and :math:`IV_v = \\sum_l (p_l - q_l)\\,\\mathrm{WOE}_l`.

    Sign convention: **positive WOE means elevated positive-class (fraud) rate**,
    which is the opposite of the classic good/bad credit convention but reads
    more naturally for a fraud target. IV is convention-invariant.

    Unseen levels at transform time map to 0 (the neutral log-odds shift).
    """

    def __init__(self, smoothing: float = 0.5, clip: float = 4.0):
        self.smoothing = smoothing
        self.clip = clip

    def fit(self, X, y):
        df = self._record_names(X).astype("object")
        y_arr = np.asarray(y).ravel().astype(float)
        if df.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        n_pos = float(y_arr.sum())
        n_neg = float(len(y_arr) - n_pos)
        if n_pos == 0 or n_neg == 0:
            raise ValueError("WOEEncoder requires both classes present in the fit sample.")

        s = float(self.smoothing)
        self.woe_maps_: Dict[str, Dict[Any, float]] = {}
        self.iv_: Dict[str, float] = {}
        for col in df.columns:
            grouped = pd.DataFrame({"lvl": df[col], "y": y_arr}).groupby("lvl", dropna=False)["y"]
            pos = grouped.sum()
            neg = grouped.count() - pos
            L = float(len(pos))
            p = (pos + s) / (n_pos + s * L)
            q = (neg + s) / (n_neg + s * L)
            woe = np.log(p / q).clip(-abs(self.clip), abs(self.clip))
            self.woe_maps_[str(col)] = {k: float(v) for k, v in woe.items()}
            self.iv_[str(col)] = float(((p - q) * woe).sum())

        self.report_ = {
            "transformer": "WOEEncoder",
            "n_columns": int(df.shape[1]),
            "smoothing": s,
            "clip": self.clip,
            "prevalence": float(n_pos / (n_pos + n_neg)),
            "information_value": {k: round(v, 6) for k, v in self.iv_.items()},
            "iv_strength": {k: _iv_band(v) for k, v in self.iv_.items()},
            "mean_iv": float(np.mean(list(self.iv_.values()))) if self.iv_ else None,
        }
        return self

    def transform(self, X):
        check_is_fitted(self, "woe_maps_")
        df = ensure_frame(X, self.feature_names_in_).astype("object")
        out = {
            f"{col}_woe": df[col].map(self.woe_maps_.get(str(col), {})).astype(float)
            .fillna(0.0).to_numpy()
            for col in df.columns
        }
        return pd.DataFrame(out, index=df.index)

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        base = input_features if input_features is not None else self.feature_names_in_
        return np.asarray([f"{c}_woe" for c in base], dtype=object)


def _iv_band(iv: float) -> str:
    """Conventional Siddiqi bands for Information Value."""
    a = abs(iv)
    if a < 0.02:
        return "not_predictive"
    if a < 0.1:
        return "weak"
    if a < 0.3:
        return "medium"
    if a < 0.5:
        return "strong"
    return "suspiciously_strong"


# ==========================================================================
# structural transformers
# ==========================================================================
class FrameSelector(_NativeFrameOutput, _NamedMixin, BaseEstimator, TransformerMixin,
                    auto_wrap_output_keys=None):
    """Select and order a subset of DataFrame columns.

    Used as the first stage of the core pipeline so that a fitted artifact
    hard-codes exactly which raw variables the final specification consumes and
    raises loudly at inference time if one is missing.
    """

    def __init__(self, columns: Optional[List[str]] = None, raise_on_missing: bool = True):
        self.columns = columns
        self.raise_on_missing = raise_on_missing

    def fit(self, X, y=None):
        df = ensure_frame(X)
        cols = list(self.columns) if self.columns is not None else [str(c) for c in df.columns]
        missing = [c for c in cols if c not in df.columns]
        if missing and self.raise_on_missing:
            raise KeyError(f"FrameSelector: columns absent from input: {missing}")
        self.columns_ = [c for c in cols if c in df.columns]
        self.feature_names_in_ = np.asarray([str(c) for c in df.columns], dtype=object)
        self.n_features_in_ = df.shape[1]
        self.report_ = {
            "transformer": "FrameSelector",
            "n_columns_in": int(df.shape[1]),
            "n_columns_selected": int(len(self.columns_)),
            "dropped": [c for c in df.columns if c not in self.columns_][:50],
        }
        return self

    def transform(self, X):
        check_is_fitted(self, "columns_")
        df = ensure_frame(X)
        missing = [c for c in self.columns_ if c not in df.columns]
        if missing and self.raise_on_missing:
            raise KeyError(f"FrameSelector: columns absent at transform time: {missing}")
        # Otherwise pass through what is present; the InferenceGuard downstream
        # re-materialises absent columns as missing and flags the batch.
        return df.loc[:, [c for c in self.columns_ if c in df.columns]]

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> np.ndarray:
        check_is_fitted(self, "columns_")
        return np.asarray(self.columns_, dtype=object)


class InferenceGuard(_NativeFrameOutput, _NamedMixin, BaseEstimator, TransformerMixin,
                     auto_wrap_output_keys=None):
    """Bound inference-time inputs to the support the model was trained on.

    Three failure modes kill a scoring job or, worse, silently corrupt a score:

    * a categorical level that did not exist in training (new merchant, new MCC,
      a renamed channel code),
    * a numeric value far outside the training range (a $2.4m debit "amount"
      from an upstream unit change, a negative tenure from a backfill bug),
    * a column that is missing, or that arrives as text when it was numeric.

    This transformer sits immediately after column selection and *before* any
    imputation or encoding, and applies an explicit, configurable policy to each
    case. The default posture is risk-averse:

    ``numeric_policy='clip'``
        Values are clipped to the training envelope, optionally widened by
        ``numeric_tolerance`` * range. The model is therefore never asked to
        extrapolate beyond its own training support -- it returns the score it
        would give at the boundary of what it has actually seen.
    ``unseen_category_policy='sentinel'``
        Unseen levels are rewritten to ``unseen_label``, which the rare-level
        collapser and every downstream encoder already treat as an out-of-
        vocabulary bucket (one-hot: infrequent/all-zero, ordinal: -1, WOE: 0
        i.e. a neutral log-odds shift, target encoding: the prior mean).
    ``missing_column_policy='fill'``
        An absent required column is materialised as all-missing so the fitted
        imputer supplies the training-time default, rather than raising.

    Nothing is silent, and nothing is stashed on the instance: every
    intervention is counted into the report and per-row flags **returned** by
    :meth:`transform_with_report`. Reading results back off mutable estimator
    state was the root of a scoring bug under concurrency and under calibration
    wrappers, so that surface no longer exists. Records whose score rests on
    guarded input should be routed to manual review, not auto-actioned --
    :class:`dmf.inference.ProductionScorer` does exactly that.
    """

    def __init__(
        self,
        numeric_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        passthrough_columns: Optional[List[str]] = None,
        numeric_kinds: Optional[Dict[str, str]] = None,
        numeric_policy: str = "clip",
        numeric_tolerance: float = 0.0,
        coerce_numeric: bool = True,
        unseen_category_policy: str = "sentinel",
        unseen_label: str = "__UNSEEN__",
        missing_column_policy: str = "fill",
        max_guarded_rate: float = 0.05,
        warn: bool = True,
    ):
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.passthrough_columns = passthrough_columns
        self.numeric_kinds = numeric_kinds
        self.numeric_policy = numeric_policy
        self.numeric_tolerance = numeric_tolerance
        self.coerce_numeric = coerce_numeric
        self.unseen_category_policy = unseen_category_policy
        self.unseen_label = unseen_label
        self.missing_column_policy = missing_column_policy
        self.max_guarded_rate = max_guarded_rate
        self.warn = warn

    # -------------------------------------------------- fit
    def fit(self, X, y=None):
        df = self._record_names(X)
        self.numeric_columns_ = [c for c in (self.numeric_columns or []) if c in df.columns]
        self.categorical_columns_ = [c for c in (self.categorical_columns or []) if c in df.columns]
        self.passthrough_columns_ = [c for c in (self.passthrough_columns or []) if c in df.columns]

        kinds = self.numeric_kinds or {}
        self.numeric_bounds_: Dict[str, Dict[str, float]] = {}
        for col in self.numeric_columns_:
            s = to_numeric_lenient(df[col], kinds.get(str(col)))
            lo, hi = float(s.min()), float(s.max())
            if not np.isfinite(lo) or not np.isfinite(hi):
                lo, hi = -np.inf, np.inf
            span = hi - lo
            pad = float(self.numeric_tolerance) * (span if np.isfinite(span) and span > 0 else 0.0)
            self.numeric_bounds_[str(col)] = {
                "train_min": lo, "train_max": hi,
                "guard_min": lo - pad, "guard_max": hi + pad,
                "train_median": float(s.median()) if s.notna().any() else 0.0,
            }

        self.seen_levels_: Dict[str, set] = {
            str(col): set(df[col].dropna().astype(str).unique().tolist())
            for col in self.categorical_columns_
        }

        self.report_ = {
            "transformer": "InferenceGuard",
            "numeric_policy": self.numeric_policy,
            "unseen_category_policy": self.unseen_category_policy,
            "numeric_tolerance": self.numeric_tolerance,
            "n_numeric_guarded": len(self.numeric_columns_),
            "n_categorical_guarded": len(self.categorical_columns_),
            "n_levels_learned": int(sum(len(v) for v in self.seen_levels_.values())),
            "levels_per_column": {k: len(v) for k, v in self.seen_levels_.items()},
            "numeric_envelope": {
                k: [round(v["guard_min"], 6), round(v["guard_max"], 6)]
                for k, v in self.numeric_bounds_.items()
            },
        }
        return self

    # -------------------------------------------------- transform
    def transform(self, X):
        return self.transform_with_report(X)[0]

    def transform_with_report(self, X) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
        """Return ``(guarded_frame, batch_report, per_row_flags)``.

        Orchestrates the four independent checks below; each mutates only the
        local ``df`` / ``flags`` / ``detail`` it is handed, so the estimator
        itself stays read-only during transform -- safe under threads and under
        wrappers that clone it.
        """
        check_is_fitted(self, "numeric_bounds_")
        df = ensure_frame(X).copy()
        required = self.numeric_columns_ + self.categorical_columns_ + self.passthrough_columns_
        flags = pd.DataFrame(0, index=df.index, columns=ROW_FLAG_COLUMNS)
        detail: Dict[str, Dict[str, Any]] = {}

        missing_cols = self._materialize_missing_columns(df, required, detail)
        for col in self.numeric_columns_:
            df[col] = self._guard_numeric(df[col], str(col), flags, detail)
        for col in self.categorical_columns_:
            df[col] = self._guard_categorical(df[col], str(col), flags, detail)

        report = self._batch_report(flags, detail, missing_cols, len(required))
        if self.warn and not report["batch_safe"]:
            import warnings

            warnings.warn(
                f"InferenceGuard: {report['escalation_reason']}; "
                f"{report['n_rows_flagged']} of {len(df)} rows flagged. "
                f"Columns: {sorted(detail)[:8]}",
                RuntimeWarning,
                stacklevel=2,
            )
        return df.loc[:, [c for c in required if c in df.columns]], report, flags

    # -------------------------------------------------- the four checks
    def _materialize_missing_columns(self, df, required, detail) -> List[str]:
        """Absent required columns become all-missing for the fitted imputer.

        Column absence is a *batch-level* schema fault, not a per-row one: it
        affects every record identically and is handled deterministically by
        imputation. Flagging every row here would drown the row-level signal,
        so it is escalated in the batch report instead (and the scorer then
        applies it to every row's action).
        """
        missing = [c for c in required if c not in df.columns]
        if missing and self.missing_column_policy == "error":
            raise KeyError(f"InferenceGuard: required columns missing at inference: {missing}")
        for c in missing:
            df[c] = np.nan
            detail[str(c)] = {"issue": "column_absent", "action": "filled_with_training_default",
                              "scope": "batch", "rate": 1.0}
        return missing

    def _guard_numeric(self, raw: pd.Series, col: str, flags, detail) -> pd.Series:
        """Coerce to float and confine to the training envelope."""
        bounds = self.numeric_bounds_[col]
        if self.coerce_numeric:
            s = to_numeric_lenient(raw, (self.numeric_kinds or {}).get(col))
            coerced = s.isna() & raw.notna()
            if coerced.any():
                flags.loc[coerced.to_numpy(), "n_coerced"] += 1
                detail.setdefault(col, {}).update(
                    coerced_to_missing=int(coerced.sum()),
                    coerced_rate=round(float(coerced.mean()), 6),
                )
        else:
            s = pd.to_numeric(raw, errors="coerce").astype(float)

        below = s < bounds["guard_min"]
        above = s > bounds["guard_max"]
        oor = (below | above).fillna(False)
        if oor.any():
            flags.loc[oor.to_numpy(), "n_out_of_range"] += 1
            d = detail.setdefault(col, {})
            d.update(
                issue="out_of_training_range",
                n_below=int(below.sum()),
                n_above=int(above.sum()),
                rate=round(float(oor.mean()), 6),
                observed_min=round(float(np.nanmin(s.to_numpy(dtype=float))), 6) if s.notna().any() else None,
                observed_max=round(float(np.nanmax(s.to_numpy(dtype=float))), 6) if s.notna().any() else None,
                envelope=[round(bounds["guard_min"], 6), round(bounds["guard_max"], 6)],
            )
            if self.numeric_policy == "error":
                raise ValueError(
                    f"InferenceGuard: column '{col}' has {int(oor.sum())} value(s) outside the "
                    f"training envelope [{bounds['guard_min']}, {bounds['guard_max']}]."
                )
            if self.numeric_policy == "clip":
                s = s.clip(lower=bounds["guard_min"], upper=bounds["guard_max"])
                d["action"] = "clipped_to_envelope"
            elif self.numeric_policy == "nan":
                s = s.mask(oor)
                d["action"] = "set_missing_then_imputed"
            else:
                d["action"] = "passthrough"
        return s.astype(float)

    def _guard_categorical(self, raw: pd.Series, col: str, flags, detail) -> pd.Series:
        """Map levels the training data never contained out of vocabulary."""
        seen = self.seen_levels_.get(col, set())
        s = raw.astype("object")
        as_str = s.where(s.isna(), s.astype(str))
        unseen = (~as_str.isin(list(seen))) & as_str.notna()
        if unseen.any():
            flags.loc[unseen.to_numpy(), "n_unseen_category"] += 1
            novel = sorted({str(v) for v in as_str[unseen].unique()})[:20]
            d = detail.setdefault(col, {})
            d.update(
                issue="unseen_category_level",
                n_unseen_rows=int(unseen.sum()),
                rate=round(float(unseen.mean()), 6),
                n_novel_levels=int(as_str[unseen].nunique()),
                novel_levels=novel,
                n_levels_seen_in_training=len(seen),
            )
            if self.unseen_category_policy == "error":
                raise ValueError(f"InferenceGuard: column '{col}' contains unseen level(s) {novel[:5]}.")
            if self.unseen_category_policy == "sentinel":
                as_str = as_str.mask(unseen, self.unseen_label)
                d["action"] = f"mapped_to_{self.unseen_label}"
            else:
                as_str = as_str.mask(unseen)
                d["action"] = "set_missing_then_imputed"
        return as_str

    def _batch_report(self, flags, detail, missing_cols, n_required: int) -> Dict[str, Any]:
        """Aggregate the row flags into the batch-level data-quality verdict."""
        n = len(flags)
        total_cells = max(n * max(n_required, 1), 1)
        guarded_cells = int(flags.to_numpy().sum())
        any_flag = (flags.sum(axis=1) > 0).to_numpy()
        report: Dict[str, Any] = {
            "n_rows": int(n),
            "n_columns_checked": n_required,
            "guarded_cells": guarded_cells,
            "guarded_cell_rate": round(guarded_cells / total_cells, 6),
            "n_rows_flagged": int(any_flag.sum()),
            "rows_flagged_rate": round(float(any_flag.mean()), 6) if n else 0.0,
            "n_rows_out_of_range": int((flags["n_out_of_range"] > 0).sum()),
            "n_rows_unseen_category": int((flags["n_unseen_category"] > 0).sum()),
            "n_rows_coerced": int((flags["n_coerced"] > 0).sum()),
            "missing_columns": missing_cols,
            "n_missing_columns": len(missing_cols),
            "by_column": detail,
        }
        report["batch_safe"] = bool(
            report["guarded_cell_rate"] <= self.max_guarded_rate and not missing_cols
        )
        report["verdict"] = "ok" if report["batch_safe"] else "review_recommended"
        if missing_cols:
            report["escalation_reason"] = (
                f"{len(missing_cols)} required column(s) absent from the batch: {missing_cols}"
            )
        elif not report["batch_safe"]:
            report["escalation_reason"] = (
                f"guarded cell rate {report['guarded_cell_rate']:.4f} exceeds "
                f"max_guarded_rate {self.max_guarded_rate}"
            )
        return report

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> np.ndarray:
        check_is_fitted(self, "numeric_bounds_")
        cols = self.numeric_columns_ + self.categorical_columns_ + self.passthrough_columns_
        return np.asarray(cols, dtype=object)


__all__ = [
    "ROW_FLAG_COLUMNS", "ensure_frame", "parse_kind", "to_numeric_lenient",
    "frame_to_numeric", "Roles", "infer_roles",
    "NumericCoercer", "QuantileWinsorizer", "RareCategoryCollapser", "WOEEncoder",
    "FrameSelector", "InferenceGuard",
]
