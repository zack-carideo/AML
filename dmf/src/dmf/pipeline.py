"""
The core artifact: ``DisputeFeaturePipeline``.

This is the object that is fitted once on the final training population and
then pickled and shipped. The *same* fitted instance serves training and
production inference, which is what makes train/serve skew structurally
impossible rather than merely discouraged.

Structure::

    DisputeFeaturePipeline
      └── sklearn.Pipeline
            ├── select        FrameSelector           (locks the variable list)
            └── column        ColumnTransformer
                  ├── num     winsorize -> impute(+indicator) -> scale -> var-threshold
                  ├── cat     impute -> collapse-rare -> encode(onehot|ordinal|woe|target)
                  └── pass    passthrough

Everything learned lives in the fitted sub-estimators, so the transformer is
re-estimated from scratch inside every cross-validation fold when it is nested
in a ``cross_validate`` call -- including the supervised encoders.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)
from sklearn.utils.validation import check_is_fitted

from .config import Config
from .reporting import StepReport, summarize_frame, summarize_matrix, summarize_target
from .transformers import (
    ROW_FLAG_COLUMNS,
    FrameSelector,
    InferenceGuard,
    NumericCoercer,
    QuantileWinsorizer,
    RareCategoryCollapser,
    Roles,
    WOEEncoder,
    ensure_frame,
    infer_roles,
)

NUM = "num"
CAT = "cat"
PASS = "pass"


class DisputeFeaturePipeline(BaseEstimator, TransformerMixin):
    """Config-driven, leakage-safe feature transformer.

    Parameters
    ----------
    config : Config | dict
        Full framework configuration. Only ``columns`` and ``preprocessing``
        are consumed here.
    features : list[str], optional
        The variable specification. ``None`` means "every variable declared in
        ``config.columns``". The harness passes the top-k subset here; the
        final production build passes the winning subset.
    verbose : int
        0 silences the fit log.

    Attributes
    ----------
    pipeline_ : sklearn.pipeline.Pipeline
        The fitted underlying pipeline.
    feature_names_out_ : list[str]
        Names of the transformed design-matrix columns.
    feature_source_map_ : dict[str, str]
        Encoded column -> source variable. Used to aggregate model importances
        back onto raw variables for selection and for analyst-facing reasons.
    fit_report_ : dict
        Quantitative summary of every pipeline step (see ``reporting``).
    """

    def __init__(
        self,
        config: Any = None,
        features: Optional[List[str]] = None,
        verbose: int = 0,
    ):
        self.config = config
        self.features = features
        self.verbose = verbose

    # ------------------------------------------------------------------
    # role resolution
    # ------------------------------------------------------------------
    def _resolve_config(self) -> Config:
        cfg = self.config
        if cfg is None:
            return Config()
        if isinstance(cfg, Config):
            return cfg
        if isinstance(cfg, dict):
            return Config.from_dict(cfg)
        raise TypeError("config must be a Config, a dict, or None.")

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _numeric_branch(self, cfg: Config, kinds: Dict[str, str]) -> Pipeline:
        p = cfg.preprocessing.numeric
        steps: List[Tuple[str, Any]] = [("to_numeric", NumericCoercer(kinds=kinds))]
        if p.winsorize.enabled:
            steps.append(
                ("winsorize", QuantileWinsorizer(p.winsorize.lower_quantile, p.winsorize.upper_quantile))
            )
        imputer_kwargs: Dict[str, Any] = {"strategy": p.imputer, "add_indicator": p.add_missing_indicator}
        if p.imputer == "constant":
            imputer_kwargs["fill_value"] = p.imputer_fill_value
        steps.append(("impute", SimpleImputer(**imputer_kwargs)))
        scaler = {
            "standard": StandardScaler(),
            "robust": RobustScaler(),
            "minmax": MinMaxScaler(),
            "none": None,
        }[p.scaler]
        if scaler is not None:
            steps.append(("scale", scaler))
        if p.variance_threshold is not None:
            steps.append(("variance", _GuardedVarianceThreshold(threshold=p.variance_threshold)))
        pipe = Pipeline(steps)
        pipe.set_output(transform="pandas")
        return pipe

    def _categorical_branch(self, cfg: Config) -> Pipeline:
        p = cfg.preprocessing.categorical
        steps: List[Tuple[str, Any]] = []
        imputer_kwargs: Dict[str, Any] = {"strategy": p.imputer}
        if p.imputer == "constant":
            imputer_kwargs["fill_value"] = p.imputer_fill_value
        steps.append(("impute", SimpleImputer(**imputer_kwargs)))
        if p.rare_level.enabled:
            steps.append(
                ("collapse_rare", RareCategoryCollapser(p.rare_level.min_frequency, p.rare_level.other_label))
            )
        if p.encoder == "onehot":
            enc: Any = OneHotEncoder(
                handle_unknown=p.onehot.handle_unknown,
                max_categories=p.onehot.max_categories,
                drop="first" if p.onehot.drop_first else None,
                sparse_output=False,
                min_frequency=None,
            )
        elif p.encoder == "ordinal":
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                 encoded_missing_value=-2)
        elif p.encoder == "woe":
            enc = WOEEncoder(smoothing=p.woe.smoothing, clip=p.woe.clip)
        elif p.encoder == "target":
            enc = TargetEncoder(cv=p.target.cv, smooth=p.target.smooth, target_type="binary")
        else:  # pragma: no cover - validated upstream
            raise ValueError(f"Unsupported categorical encoder '{p.encoder}'.")
        steps.append(("encode", enc))
        pipe = Pipeline(steps)
        pipe.set_output(transform="pandas")
        return pipe

    def _build(self, cfg: Config, roles: Roles) -> Pipeline:
        num, cat, pas = roles.numeric, roles.categorical, roles.passthrough
        branches: List[Tuple[str, Any, List[str]]] = []
        if num:
            branches.append((NUM, self._numeric_branch(cfg, roles.kinds), num))
        if cat:
            branches.append((CAT, self._categorical_branch(cfg), cat))
        if pas:
            branches.append((PASS, "passthrough", pas))
        if not branches:
            raise ValueError("No usable feature columns after role resolution.")

        ct = ColumnTransformer(
            transformers=branches,
            remainder="drop",
            verbose_feature_names_out=True,
        )
        ct.set_output(transform="pandas")

        steps: List[Tuple[str, Any]] = [
            ("select", FrameSelector(columns=num + cat + pas,
                                     raise_on_missing=cfg.preprocessing.inference_guard.missing_column_policy
                                     == "error")),
        ]
        g = cfg.preprocessing.inference_guard
        if g.enabled:
            steps.append(
                (
                    "guard",
                    InferenceGuard(
                        numeric_columns=num,
                        numeric_kinds=roles.kinds,
                        categorical_columns=cat,
                        passthrough_columns=pas,
                        numeric_policy=g.numeric_policy,
                        numeric_tolerance=g.numeric_tolerance,
                        coerce_numeric=g.coerce_numeric,
                        unseen_category_policy=g.unseen_category_policy,
                        unseen_label=g.unseen_label,
                        missing_column_policy=g.missing_column_policy,
                        max_guarded_rate=g.max_guarded_rate,
                        warn=g.warn,
                    ),
                )
            )
        steps.append(("column", ct))
        pipe = Pipeline(steps)
        pipe.set_output(transform="pandas")
        return pipe

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------
    def fit(self, X, y=None):
        cfg = self._resolve_config()
        df = ensure_frame(X)
        roles = infer_roles(df, cfg, self.features)
        num, cat, pas = roles.numeric, roles.categorical, roles.passthrough
        if not roles.all:
            raise ValueError(
                "No usable feature columns after role resolution. "
                f"Dropped: {roles.dropped or 'nothing'}; requested: {self.features}"
            )

        report = StepReport(run=f"feature_pipeline[{len(roles.all)} vars]")
        report.add("input", **summarize_frame(df.loc[:, [c for c in roles.all]], "raw_features"))
        if y is not None:
            report.add("target", **summarize_target(y))
        report.add(
            "role_resolution",
            **roles.report(),
            requested_features=list(self.features) if self.features is not None else None,
        )

        self.roles_ = roles
        self.pipeline_ = self._build(cfg, roles)
        Xt = self.pipeline_.fit_transform(df, y)

        self.numeric_features_ = num
        self.categorical_features_ = cat
        self.passthrough_features_ = pas
        self.input_features_ = num + cat + pas
        self.feature_names_in_ = np.asarray(self.input_features_, dtype=object)
        self.n_features_in_ = len(self.input_features_)
        self.feature_names_out_ = [str(c) for c in _feature_names(self.pipeline_, Xt)]
        self.feature_source_map_ = self._build_source_map()

        for entry in self._step_summaries(cfg):
            report.add(**entry)

        report.add("output", **summarize_matrix(Xt, "design_matrix"))
        report.add(
            "expansion",
            n_input_variables=len(self.input_features_),
            n_output_columns=len(self.feature_names_out_),
            expansion_ratio=round(len(self.feature_names_out_) / max(len(self.input_features_), 1), 4),
            columns_per_source={
                src: int(cnt)
                for src, cnt in pd.Series(list(self.feature_source_map_.values())).value_counts().items()
            },
        )

        self.report_ = report
        self.fit_report_ = report.to_dict()
        self._fit_output_ = Xt
        if self.verbose:
            print(report.render())
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Return the matrix produced during fitting, not a second transform.

        For ``TargetEncoder`` these differ: the fit-time matrix is the
        cross-fitted (out-of-fold) encoding the estimator must be trained on,
        while ``transform`` yields the full-data per-level mean. Handing the
        latter to the model is a leak that costs real out-of-sample AUC.
        """
        self.fit(X, y)
        out, self._fit_output_ = self._fit_output_, None
        return out

    def transform(self, X):
        check_is_fitted(self, "pipeline_")
        Xt = self.pipeline_.transform(self._validated(X))
        if isinstance(Xt, pd.DataFrame):
            return Xt
        return pd.DataFrame(np.asarray(Xt), columns=self.feature_names_out_)

    def _validated(self, X) -> pd.DataFrame:
        """Refuse input that cannot be matched to the training schema.

        Without this, an ndarray or a renamed frame produces synthetic column
        names, every required column reads as absent, the guard fills them all
        with training defaults, and the model returns one constant probability
        for every record -- with no exception anywhere. Loud failure is the
        only acceptable behaviour for a scoring job.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"{type(self).__name__}.transform requires a DataFrame with the training "
                f"column names; got {type(X).__name__}. The pipeline selects columns by "
                f"name, so positional array input cannot be matched to the schema."
            )
        present = [c for c in self.input_features_ if c in X.columns]
        if not present:
            raise KeyError(
                f"None of the {len(self.input_features_)} expected columns are present. "
                f"Expected e.g. {self.input_features_[:5]}; received e.g. "
                f"{[str(c) for c in X.columns[:5]]}. This is a schema mismatch, not missing data."
            )
        missing = [c for c in self.input_features_ if c not in X.columns]
        if missing and self._resolve_config().preprocessing.inference_guard.missing_column_policy == "error":
            raise KeyError(f"Required columns absent at transform time: {missing}")
        return X

    # NOTE: fit_transform is deliberately NOT overridden. TransformerMixin's
    # implementation returns the matrix produced *during* fitting, which for a
    # cross-fitted supervised encoder (sklearn's TargetEncoder) is the
    # out-of-fold encoding. Re-running transform() after fit would substitute
    # the full-data per-level mean and train the estimator on a leaked column.

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> np.ndarray:
        check_is_fitted(self, "feature_names_out_")
        return np.asarray(self.feature_names_out_, dtype=object)

    # ------------------------------------------------------------------
    # introspection helpers
    # ------------------------------------------------------------------
    def _build_source_map(self) -> Dict[str, str]:
        """Map every encoded output column back to the raw variable it came from."""
        sources = {NUM: self.numeric_features_, CAT: self.categorical_features_, PASS: self.passthrough_features_}
        out: Dict[str, str] = {}
        for name in self.feature_names_out_:
            branch, _, inner = name.partition("__")
            candidates = sources.get(branch, [])
            if not candidates:
                candidates = self.input_features_
            match = _longest_source_match(inner or name, candidates)
            out[name] = match if match is not None else (inner or name)
        return out

    def _step_summaries(self, cfg: Config) -> List[Dict[str, Any]]:
        """Harvest a quantitative summary from each fitted sub-transformer."""
        entries: List[Dict[str, Any]] = []
        for top_name in ("select", "guard"):
            est = self.pipeline_.named_steps.get(top_name)
            if est is not None and hasattr(est, "report_"):
                entries.append({"step": top_name, **dict(est.report_)})
        ct: ColumnTransformer = self.pipeline_.named_steps["column"]
        for branch_name, branch, cols in ct.transformers_:
            if branch == "passthrough" or branch == "drop":
                if branch_name == PASS and cols:
                    entries.append({"step": f"{PASS}.passthrough", "n_columns": len(cols), "columns": list(cols)})
                continue
            if not hasattr(branch, "named_steps"):
                continue
            for step_name, est in branch.named_steps.items():
                summary = _summarize_estimator(est, cols)
                if summary is not None:
                    entries.append({"step": f"{branch_name}.{step_name}", **summary})
        return entries

    # ------------------------------------------------------------------
    # inference-time safety
    # ------------------------------------------------------------------
    @property
    def guard(self) -> Optional[InferenceGuard]:
        """The fitted :class:`InferenceGuard`, or ``None`` if it is disabled."""
        check_is_fitted(self, "pipeline_")
        return self.pipeline_.named_steps.get("guard")

    def transform_with_quality(self, X) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
        """Design matrix, batch data-quality report, and per-row guard flags.

        The report and flags are *returned*, never read back off the estimator.
        Reading mutable state after a call is unsafe the moment two requests
        share a loaded model, and wrong the moment a wrapper such as
        ``CalibratedClassifierCV`` runs the transform on internal clones.
        """
        check_is_fitted(self, "pipeline_")
        df = self._validated(X)
        steps = self.pipeline_.named_steps
        guard = steps.get("guard")
        if guard is None:
            empty = pd.DataFrame(0, index=df.index, columns=ROW_FLAG_COLUMNS)
            return self.transform(df), {
                "verdict": "not_checked", "batch_safe": None,
                "note": "inference_guard is disabled in the configuration",
            }, empty
        guarded, report, flags = guard.transform_with_report(steps["select"].transform(df))
        Xt = steps["column"].transform(guarded)
        if not isinstance(Xt, pd.DataFrame):
            Xt = pd.DataFrame(np.asarray(Xt), columns=self.feature_names_out_, index=df.index)
        return Xt, report, flags.set_axis(df.index)

    def training_envelope(self) -> pd.DataFrame:
        """The numeric support and categorical vocabulary the model was fit on."""
        guard = self.guard
        if guard is None:
            return pd.DataFrame()
        rows = [
            {"variable": k, "type": "numeric", "train_min": v["train_min"], "train_max": v["train_max"],
             "guard_min": v["guard_min"], "guard_max": v["guard_max"], "n_levels": None}
            for k, v in guard.numeric_bounds_.items()
        ] + [
            {"variable": k, "type": "categorical", "train_min": None, "train_max": None,
             "guard_min": None, "guard_max": None, "n_levels": len(v)}
            for k, v in guard.seen_levels_.items()
        ]
        return pd.DataFrame(rows)

    def information_value(self) -> Optional[pd.Series]:
        """IV per categorical variable, when the WOE encoder is in use."""
        check_is_fitted(self, "pipeline_")
        enc = next((b.named_steps.get("encode")
                    for n, b, _ in self.pipeline_.named_steps["column"].transformers_
                    if n == CAT and hasattr(b, "named_steps")), None)
        return pd.Series(enc.iv_).sort_values(ascending=False) if isinstance(enc, WOEEncoder) else None

    def summary(self) -> pd.DataFrame:
        check_is_fitted(self, "report_")
        return self.report_.to_frame()


# --------------------------------------------------------------------------
# production artifact assembly
# --------------------------------------------------------------------------
def build_model_pipeline(config: Any, features: Optional[List[str]], estimator: Any) -> Pipeline:
    """Assemble the deployable ``features -> model`` pipeline.

    This exact object is what ``cross_validate`` scores during selection and
    what is refit and pickled for production, so the thing measured and the
    thing deployed are the same code path.
    """
    return Pipeline(
        [
            ("features", DisputeFeaturePipeline(config=config, features=features)),
            ("model", clone(estimator)),
        ]
    )


# --------------------------------------------------------------------------
# internals
# --------------------------------------------------------------------------
class _GuardedVarianceThreshold(VarianceThreshold):
    """VarianceThreshold that never returns an empty matrix.

    Fold-level constant columns are common with rare categories; sklearn's
    version raises in that case, which would abort a whole CV run.
    """

    def fit(self, X, y=None):
        try:
            super().fit(X, y)
        except ValueError:
            arr = np.asarray(X, dtype=float)
            self.variances_ = np.nanvar(arr, axis=0)
            self.n_features_in_ = arr.shape[1]
            if hasattr(X, "columns"):
                self.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)
        support = self._get_support_mask_raw()
        if not support.any():
            support[int(np.argmax(self.variances_))] = True
        self._forced_support = support
        return self

    def _get_support_mask_raw(self) -> np.ndarray:
        return np.asarray(self.variances_ > self.threshold)

    def _get_support_mask(self) -> np.ndarray:
        if getattr(self, "_forced_support", None) is not None:
            return self._forced_support
        return self._get_support_mask_raw()


def _feature_names(pipe: Pipeline, Xt: Any) -> Sequence[str]:
    if isinstance(Xt, pd.DataFrame):
        return [str(c) for c in Xt.columns]
    try:
        return [str(c) for c in pipe.get_feature_names_out()]
    except Exception:  # pragma: no cover
        return [f"f{i}" for i in range(np.asarray(Xt).shape[1])]


def _longest_source_match(encoded: str, candidates: Sequence[str]) -> Optional[str]:
    """Attribute an encoded column name to the longest matching source column."""
    stem = encoded
    if stem.startswith("missingindicator_"):
        stem = stem[len("missingindicator_"):]
    best: Optional[str] = None
    for c in candidates:
        c = str(c)
        if stem == c or stem.startswith(f"{c}_"):
            if best is None or len(c) > len(best):
                best = c
    return best


def _summarize_estimator(est: Any, cols: Sequence[str]) -> Optional[Dict[str, Any]]:
    """Numeric fit summary of a fitted transformer, by reflection.

    dmf transformers publish a curated ``report_`` and are returned verbatim.
    Third-party estimators are summarised from their learned trailing-underscore
    attributes rather than a per-class registry of sklearn internals -- those
    attribute names change between sklearn minor versions, and a reflection
    summary survives upgrades a hand-written one silently would not.
    """
    if hasattr(est, "report_"):
        return dict(est.report_)

    out: Dict[str, Any] = {"transformer": type(est).__name__, "n_columns": len(cols)}
    for name, value in sorted(vars(est).items()):
        if not name.endswith("_") or name.startswith("_") or name.endswith("__"):
            continue
        if isinstance(value, (bool, int, str)):
            out[name] = value
        elif isinstance(value, (float, np.floating)):
            out[name] = round(float(value), 6)
        elif isinstance(value, np.ndarray):
            entry: Dict[str, Any] = {"n": int(value.size)}
            if value.size and np.issubdtype(value.dtype, np.number):
                with np.errstate(all="ignore"):
                    entry["mean"] = round(float(np.nanmean(value.astype(float))), 6)
            out[name] = entry
        elif isinstance(value, (list, tuple)) and value and all(
            isinstance(v, np.ndarray) for v in value
        ):
            # e.g. OneHotEncoder.categories_: one array of levels per column
            out[name] = {"n_arrays": len(value),
                         "total_length": int(sum(len(v) for v in value))}
    return out


__all__ = ["DisputeFeaturePipeline", "build_model_pipeline"]
