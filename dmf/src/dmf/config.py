"""
Typed configuration for the dispute-model framework.

Everything the framework needs -- data location, column roles, preprocessing
choices, cross-validation design, the estimator zoo, the variable-selection
sweep and (optional) hyper-parameter tuning -- is declared in a single YAML
file and materialised into the dataclasses below.

Design notes
------------
* Config objects are *plain data*. They carry no fitted state, so they can be
  serialised alongside a fitted model to fully reproduce a specification.
* Unknown keys raise, rather than silently doing nothing. A typo in a YAML key
  is one of the most common silent-failure modes in ML config plumbing.
* ``Config.to_dict()`` round-trips, so the exact configuration that produced a
  champion model can be written back out next to the model artifact.
"""

from __future__ import annotations

import copy
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import yaml

T = TypeVar("T")


# --------------------------------------------------------------------------
# generic dataclass <-> dict plumbing
# --------------------------------------------------------------------------
def _dataclass_type(annotation: Any) -> Optional[type]:
    """The dataclass a field nests, if any -- unwrapping Optional[X]."""
    if is_dataclass(annotation):
        return annotation  # type: ignore[return-value]
    for arg in typing.get_args(annotation):
        if is_dataclass(arg):
            return arg
    return None


def _from_dict(cls: Type[T], data: Optional[Dict[str, Any]], path: str = "") -> T:
    """Recursively build a (possibly nested) dataclass from a mapping.

    Field annotations are resolved with ``typing.get_type_hints`` rather than a
    hand-maintained registry, so adding a new config section never requires
    registering it anywhere.
    """
    data = {} if data is None else dict(data)
    if not is_dataclass(cls):
        return data  # type: ignore[return-value]

    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        where = path or cls.__name__
        raise ValueError(
            f"Unknown configuration key(s) {sorted(unknown)} under '{where}'. "
            f"Valid keys: {sorted(known)}"
        )

    hints = typing.get_type_hints(cls)
    kwargs: Dict[str, Any] = {}
    for name in known:
        if name not in data:
            continue
        value = data[name]
        nested = _dataclass_type(hints.get(name))
        if nested is not None and isinstance(value, dict):
            kwargs[name] = _from_dict(nested, value, f"{path}.{name}" if path else name)
        else:
            kwargs[name] = value
    return cls(**kwargs)  # type: ignore[return-value]


def _to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------
@dataclass
class RunConfig:
    name: str = "dmf_run"
    output_dir: str = "./artifacts"
    random_state: int = 42
    n_jobs: int = 1
    verbose: int = 1
    save_fitted_model: bool = True
    # after the holdout has been scored, refit the winning specification on
    # train + holdout so the shipped model uses all available data
    refit_on_full_data: bool = True
    # row-level prediction store written under the run's artifact directory,
    # so any metric can be recomputed post-run without refitting anything.
    # Levels are cumulative:
    #   none    -- store nothing
    #   holdout -- the champion's holdout predictions (cheap; the default)
    #   cv      -- + every validation-fold prediction of every model x k cell
    #   all     -- + the training-side predictions of each CV fold (largest;
    #              enables row-level overfit diagnostics)
    save_predictions: str = "holdout"


@dataclass
class DataConfig:
    path: Optional[str] = None
    format: str = "csv"                    # csv | parquet
    target: str = "target"
    # column that uniquely identifies a record (dispute id, claim id, ...).
    # Used to key the prediction store; never used as a model feature. When
    # unset, the DataFrame index is recorded instead.
    id_column: Optional[str] = None
    positive_label: Any = 1
    sample_frac: Optional[float] = None
    read_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ColumnsConfig:
    numeric: List[str] = field(default_factory=list)
    categorical: List[str] = field(default_factory=list)
    passthrough: List[str] = field(default_factory=list)
    drop: List[str] = field(default_factory=list)
    # when True, any column not explicitly assigned a role is typed by dtype
    auto_infer: bool = True
    # an integer-coded column with <= this many distinct values is categorical
    auto_infer_max_cardinality: int = 25
    # share of non-null values that must parse for a text column to be read as
    # numeric or as a date (recovers "$1,234.50" and ISO timestamps)
    numeric_parse_threshold: float = 0.95
    # drop self-chosen columns that carry no information. Columns named
    # explicitly in the role lists are always honoured.
    drop_constant: bool = True
    # a categorical whose distinct-value share exceeds this is an identifier
    max_categorical_cardinality_ratio: float = 0.5


@dataclass
class WinsorizeConfig:
    enabled: bool = True
    lower_quantile: float = 0.01
    upper_quantile: float = 0.99


@dataclass
class NumericPreprocessing:
    imputer: str = "median"                # median | mean | constant
    imputer_fill_value: float = 0.0
    add_missing_indicator: bool = True
    winsorize: WinsorizeConfig = field(default_factory=WinsorizeConfig)
    scaler: str = "standard"               # standard | robust | minmax | none
    # None disables the step. A float drops columns whose train variance is <=
    # the threshold; guarded so it can never empty the design matrix.
    variance_threshold: Optional[float] = None


@dataclass
class RareLevelConfig:
    enabled: bool = True
    min_frequency: float = 0.01
    other_label: str = "__RARE__"


@dataclass
class OneHotConfig:
    handle_unknown: str = "infrequent_if_exist"
    max_categories: Optional[int] = None
    drop_first: bool = False


@dataclass
class WOEConfig:
    smoothing: float = 0.5                 # Laplace / Haldane-Anscombe style
    clip: float = 4.0                      # bound |WOE| for stability
    report_iv: bool = True


@dataclass
class TargetEncodingConfig:
    cv: int = 5                            # internal cross-fitting folds
    smooth: Any = "auto"


@dataclass
class CategoricalPreprocessing:
    imputer: str = "constant"              # constant | most_frequent
    imputer_fill_value: str = "__MISSING__"
    rare_level: RareLevelConfig = field(default_factory=RareLevelConfig)
    encoder: str = "onehot"                # onehot | ordinal | woe | target
    onehot: OneHotConfig = field(default_factory=OneHotConfig)
    woe: WOEConfig = field(default_factory=WOEConfig)
    target: TargetEncodingConfig = field(default_factory=TargetEncodingConfig)


@dataclass
class InferenceGuardConfig:
    """Policy for inputs at inference time that the training data never showed.

    The default posture is deliberately risk-averse: never extrapolate, never
    crash, always flag. A record that trips a guard still gets a score, but it
    is marked so an operational queue can route it to manual review instead of
    trusting a score built on unsupported input.
    """

    enabled: bool = True
    # numeric values outside the learned training envelope
    numeric_policy: str = "clip"           # clip | nan | passthrough | error
    # allowed expansion of the train [min, max] range, as a fraction of range
    numeric_tolerance: float = 0.0
    # non-numeric junk in a numeric column
    coerce_numeric: bool = True
    # categorical levels never seen during fit
    unseen_category_policy: str = "sentinel"   # sentinel | nan | error
    unseen_label: str = "__UNSEEN__"
    # a required column absent from the inference frame
    missing_column_policy: str = "fill"    # fill | error
    # batch-level share of guarded cells above which the report is marked unsafe
    max_guarded_rate: float = 0.05
    # per-row flags are always computed; this only controls stderr chatter
    warn: bool = True


@dataclass
class PreprocessingConfig:
    numeric: NumericPreprocessing = field(default_factory=NumericPreprocessing)
    categorical: CategoricalPreprocessing = field(default_factory=CategoricalPreprocessing)
    inference_guard: InferenceGuardConfig = field(default_factory=InferenceGuardConfig)


@dataclass
class CVConfig:
    n_splits: int = 5
    n_repeats: int = 1                     # >1 -> RepeatedStratifiedKFold
    shuffle: bool = True


@dataclass
class SplitConfig:
    holdout_size: float = 0.2
    stratify: bool = True
    # random -- i.i.d. rows.
    # group  -- keep every row sharing a key (customer, card, account) on one
    #           side of every split. Without this, a customer with six disputes
    #           lands on both sides and the holdout is optimistic.
    # time   -- the holdout is the most recent slice, and CV folds run forward
    #           only. The honest test for a fraud model that will score the
    #           future, and the only one that exposes concept drift.
    # group_time -- both at once: each group's *earliest* timestamp decides its
    #           side, so the holdout is the most recent share of groups whose
    #           activity lies entirely in the newest window, and any group with
    #           earlier transactions stays whole in training. CV within
    #           training is group-intact (StratifiedGroupKFold).
    strategy: str = "random"               # random | group | time | group_time
    group_column: Optional[str] = None     # required for strategy: group, group_time
    time_column: Optional[str] = None      # required for strategy: time, group_time
    cv: CVConfig = field(default_factory=CVConfig)


@dataclass
class MetricsConfig:
    primary: str = "average_precision"
    secondary: List[str] = field(
        default_factory=lambda: [
            "roc_auc",
            "ks_statistic",
            "brier_score",
            "recall_at_fpr",
            "lift_at_top_pct",
            "log_loss",
        ]
    )
    # Operating points for the two budget-dependent metrics. Either a scalar,
    # or a list to report the metric at several budgets at once -- a list of N
    # values produces N reported metrics named ``recall_at_fpr@<value>``, while
    # a scalar keeps the plain un-suffixed name. Where only one point is
    # possible (threshold derivation, the slice-table cut) the *first* value is
    # used, so list the budget you intend to operate at first.
    recall_at_fpr: Union[float, List[float]] = 0.01     # false-positive budget(s)
    lift_top_pct: Union[float, List[float]] = 0.05      # review budget(s)
    compute_train_scores: bool = True      # enables the overfit-gap diagnostic
    # how the champion's production decision threshold is derived from its
    # holdout score distribution (saved into the model bundle so the
    # ProductionScorer picks it up):
    #   top_pct -- the score cut that flags the top lift_top_pct of holdout
    #              volume (capacity-based; mirrors the scorer's top_pct fallback
    #              but as a stable absolute number)
    #   fpr     -- the score cut that achieves recall_at_fpr false-positive
    #              rate on the holdout
    #   none    -- do not derive one; the bundle ships decision_threshold=null
    decision_threshold_policy: str = "top_pct"
    # columns to break the holdout report down by (claim channel, reason code,
    # segment, ...). They need only exist in the data -- they do not have to be
    # model inputs -- so flag-rate parity can be checked on attributes the
    # model is deliberately not allowed to use. Levels thinner than
    # min_slice_n are skipped rather than reported on noise.
    slice_columns: List[str] = field(default_factory=list)
    min_slice_n: int = 50


@dataclass
class ImportanceOrderingConfig:
    method: str = "auto"                   # auto | coef | gain | permutation
    permutation_repeats: int = 5
    # multiply each coefficient by the SD of its encoded column before
    # aggregating to the source variable. Without this, summing |coef| across
    # the levels of a high-cardinality categorical mechanically outranks a
    # strong single numeric variable.
    scale_by_std: bool = True
    # how per-level contributions roll up to the source variable
    aggregate: str = "sum"                 # sum | max | mean


@dataclass
class RFEOrderingConfig:
    step: Any = 1                          # int or float share, as in sklearn RFE
    aggregate: str = "min_rank"            # min_rank | mean_rank  (encoded -> source)


@dataclass
class SelectionConfig:
    # Variable ordering is always re-ranked *inside* every outer CV fold, so no
    # validation row ever helped choose the features it is used to score.
    # Ranking once on the whole training partition and then cross-validating on
    # that same partition is feature selection outside the CV loop: it inflated
    # the leaderboard by ~0.12 AP on pure noise while the overfit-gap
    # diagnostic read zero. That construction was removed rather than left
    # behind a flag.
    ordering_strategy: str = "importance"  # importance | rfe
    ordering_reference_model: str = "per_model"   # 'per_model' or a model key
    importance: ImportanceOrderingConfig = field(default_factory=ImportanceOrderingConfig)
    rfe: RFEOrderingConfig = field(default_factory=RFEOrderingConfig)
    k_min: int = 1
    k_max: Optional[int] = None            # None -> all available variables
    k_step: int = 1
    top_n: int = 3
    # True -> the top-N list holds the best variant of N *different* models,
    # which is usually what a champion/challenger review wants. False -> the N
    # best (model, k) cells outright, which can all come from one model.
    top_n_distinct_models: bool = False
    one_se_rule: bool = True
    # paired_t applies the Nadeau-Bengio correction and works at any fold count.
    # wilcoxon is distribution-free but its smallest attainable two-sided p is
    # 2 / 2**n_folds, so at 5 folds nothing can ever reach 0.05.
    marginal_gain_test: str = "paired_t"   # paired_t | wilcoxon | none


@dataclass
class ModelSpec:
    estimator: str = ""                    # dotted import path
    params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    family: str = "other"                  # linear | tree | other  (documentation)
    requires_scaling: bool = False
    handles_categorical_natively: bool = False
    # 'balanced' -> class_weight='balanced' or scale_pos_weight=neg/pos
    imbalance: Optional[str] = None
    tag: str = ""                          # e.g. 'champion' / 'challenger'
    # dotted overrides applied to preprocessing for this model only,
    # e.g. {"numeric.scaler": "none", "categorical.encoder": "ordinal"}
    preprocessing_overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TuningConfig:
    enabled: bool = False
    strategy: str = "random"               # random | grid
    n_iter: int = 25
    cv_splits: int = 3
    apply_to: str = "top_n"                # top_n | all
    search_spaces: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    columns: ColumnsConfig = field(default_factory=ColumnsConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    models: Dict[str, ModelSpec] = field(default_factory=dict)
    tuning: TuningConfig = field(default_factory=TuningConfig)

    # ---------------- construction ----------------
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        data = copy.deepcopy(dict(data or {}))
        models_raw = data.pop("models", {}) or {}
        cfg = _from_dict(cls, data)
        cfg.models = {
            name: _from_dict(ModelSpec, spec, f"models.{name}")
            for name, spec in models_raw.items()
        }
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r") as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    def to_dict(self) -> Dict[str, Any]:
        return _to_dict(self)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False, default_flow_style=False)

    def copy(self) -> "Config":
        return Config.from_dict(self.to_dict())

    # ---------------- validation ----------------
    def validate(self) -> None:
        p = self.preprocessing
        g = p.inference_guard
        for value, allowed, where in [
            (p.numeric.imputer, {"median", "mean", "constant"}, "preprocessing.numeric.imputer"),
            (p.numeric.scaler, {"standard", "robust", "minmax", "none"}, "preprocessing.numeric.scaler"),
            (p.categorical.imputer, {"constant", "most_frequent"}, "preprocessing.categorical.imputer"),
            (p.categorical.encoder, {"onehot", "ordinal", "woe", "target"}, "preprocessing.categorical.encoder"),
            (g.numeric_policy, {"clip", "nan", "passthrough", "error"}, "preprocessing.inference_guard.numeric_policy"),
            (g.unseen_category_policy, {"sentinel", "nan", "error"}, "preprocessing.inference_guard.unseen_category_policy"),
            (g.missing_column_policy, {"fill", "error"}, "preprocessing.inference_guard.missing_column_policy"),
            (self.selection.ordering_strategy, {"importance", "rfe"}, "selection.ordering_strategy"),
            (self.selection.marginal_gain_test, {"wilcoxon", "paired_t", "none"}, "selection.marginal_gain_test"),
            (self.tuning.strategy, {"random", "grid"}, "tuning.strategy"),
            (self.tuning.apply_to, {"top_n", "all"}, "tuning.apply_to"),
            (self.split.strategy, {"random", "group", "time", "group_time"}, "split.strategy"),
            (self.run.save_predictions, {"none", "holdout", "cv", "all"}, "run.save_predictions"),
            (self.metrics.decision_threshold_policy, {"top_pct", "fpr", "none"},
             "metrics.decision_threshold_policy"),
        ]:
            _check_in(value, allowed, where)
        if g.numeric_tolerance < 0:
            raise ValueError("preprocessing.inference_guard.numeric_tolerance must be >= 0.")
        for attr in ("recall_at_fpr", "lift_top_pct"):
            _check_operating_points(getattr(self.metrics, attr), f"metrics.{attr}")
        # resolving the metric set here turns an unknown metric name, or a
        # primary that spans several operating points, into a config error
        # rather than a failure an hour into a sweep. Imported locally so the
        # config module stays free of a dependency on the metric registry.
        from .metrics import resolve_metrics

        resolve_metrics(self.metrics)
        if not 0.0 < self.columns.numeric_parse_threshold <= 1.0:
            raise ValueError("columns.numeric_parse_threshold must be in (0, 1].")
        if not 0.0 < self.columns.max_categorical_cardinality_ratio <= 1.0:
            raise ValueError("columns.max_categorical_cardinality_ratio must be in (0, 1].")
        # TimeSeriesSplit is deterministic, so equal fold counts would make the
        # tuning loop's inner folds identical to the outer ones -- and tuned
        # leaderboard rows would be scored on the folds that chose their
        # hyper-parameters. Shuffled strategies avoid this with a shifted seed.
        if (self.tuning.enabled and self.split.strategy == "time"
                and self.tuning.cv_splits == self.split.cv.n_splits):
            raise ValueError(
                "tuning.cv_splits must differ from split.cv.n_splits when "
                "split.strategy='time', or inner and outer folds coincide."
            )
        if self.split.strategy in ("group", "group_time") and not self.split.group_column:
            raise ValueError(f"split.strategy='{self.split.strategy}' requires split.group_column.")
        if self.split.strategy in ("time", "group_time") and not self.split.time_column:
            raise ValueError(f"split.strategy='{self.split.strategy}' requires split.time_column.")
        if not 0.0 < self.split.holdout_size < 1.0:
            raise ValueError("split.holdout_size must be in (0, 1).")
        if self.split.cv.n_splits < 2:
            raise ValueError("split.cv.n_splits must be >= 2.")
        if self.selection.k_min < 1:
            raise ValueError("selection.k_min must be >= 1.")
        if self.selection.k_max is not None and self.selection.k_max < self.selection.k_min:
            raise ValueError("selection.k_max must be >= selection.k_min.")
        if self.selection.top_n < 1:
            raise ValueError("selection.top_n must be >= 1.")

        w = p.numeric.winsorize
        if w.enabled and not 0.0 <= w.lower_quantile < w.upper_quantile <= 1.0:
            raise ValueError("winsorize quantiles must satisfy 0 <= lower < upper <= 1.")

        ref = self.selection.ordering_reference_model
        if ref != "per_model" and ref not in self.models:
            raise ValueError(
                f"selection.ordering_reference_model='{ref}' is not a key in models "
                f"({sorted(self.models)})."
            )
        if self.models and not any(m.enabled for m in self.models.values()):
            raise ValueError("No enabled models in the estimator zoo.")

    # ---------------- convenience ----------------
    @property
    def enabled_models(self) -> Dict[str, ModelSpec]:
        return {k: v for k, v in self.models.items() if v.enabled}

    @property
    def declared_features(self) -> List[str]:
        return list(self.columns.numeric) + list(self.columns.categorical) + list(self.columns.passthrough)


def _check_in(value: Any, allowed: set, where: str) -> None:
    if value not in allowed:
        raise ValueError(f"{where}='{value}' is invalid; expected one of {sorted(allowed)}.")


def _check_operating_points(value: Any, where: str) -> None:
    """A metric budget: one number in (0, 1], or a non-empty list of distinct ones."""
    values = list(value) if isinstance(value, (list, tuple)) else [value]
    if not values:
        raise ValueError(f"{where} is an empty list; supply at least one value.")
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0.0 < float(v) <= 1.0:
            raise ValueError(
                f"{where} must be a number in (0, 1] (or a list of them); got {v!r}."
            )
    if len({float(v) for v in values}) != len(values):
        raise ValueError(f"{where} contains duplicate operating points: {values}.")


__all__ = [
    "Config", "RunConfig", "DataConfig", "ColumnsConfig", "PreprocessingConfig",
    "InferenceGuardConfig",
    "NumericPreprocessing", "CategoricalPreprocessing", "SplitConfig", "CVConfig",
    "MetricsConfig", "SelectionConfig", "ModelSpec", "TuningConfig",
]
