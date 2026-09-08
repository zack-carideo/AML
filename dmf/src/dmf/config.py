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
import functools
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
    """The dataclass a field holds: directly, inside Optional[X], or as the
    value type of Dict[str, X]."""
    if is_dataclass(annotation):
        return annotation  # type: ignore[return-value]
    return next((arg for arg in typing.get_args(annotation) if is_dataclass(arg)), None)


def _from_dict(cls: Type[T], data: Optional[Dict[str, Any]], path: str = "") -> T:
    """Recursively build a (possibly nested) dataclass from a mapping.

    Field annotations are resolved with ``typing.get_type_hints`` rather than a
    hand-maintained registry, so adding a new config section -- or a new
    ``Dict[str, SomeSection]`` field like ``models`` -- never requires
    registering it anywhere.
    """
    data = dict(data or {})
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"Unknown configuration key(s) {sorted(unknown)} under '{path or cls.__name__}'. "
            f"Valid keys: {sorted(known)}"
        )

    hints = typing.get_type_hints(cls)
    kwargs: Dict[str, Any] = {}
    for name, value in data.items():
        hint = hints.get(name)
        nested = _dataclass_type(hint)
        where = f"{path}.{name}" if path else name
        if nested is not None and typing.get_origin(hint) is dict:
            kwargs[name] = {k: _from_dict(nested, v, f"{where}.{k}") for k, v in (value or {}).items()}
        elif nested is not None and isinstance(value, dict):
            kwargs[name] = _from_dict(nested, value, where)
        else:
            kwargs[name] = value
    return cls(**kwargs)  # type: ignore[return-value]


def get_dotted(obj: Any, dotted: str) -> Any:
    """``get_dotted(cfg, "split.cv.n_splits")`` -> ``cfg.split.cv.n_splits``."""
    return functools.reduce(getattr, dotted.split("."), obj)


def set_dotted(obj: Any, dotted: str, value: Any) -> None:
    """Assign through a dotted path, refusing to invent keys that do not exist."""
    *parents, leaf = dotted.split(".")
    target = obj
    for part in parents + [leaf]:
        if not hasattr(target, part):
            raise AttributeError(f"'{dotted}': no key '{part}' in {type(target).__name__}.")
        if part != leaf:
            target = getattr(target, part)
    setattr(target, leaf, value)


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
class SamplingConfig:
    """Coverage-maximizing undersampling of the rows a model is *fit* on.

    Two uses, declared by ``purpose``: rebalancing a low-prevalence target, and
    shrinking a large training population so a wide grid stays feasible.

    Three properties make it safe to switch on:

    * It is **experiment-time only**. Nothing here enters the shipped pipeline;
      the sampler chooses rows, it does not transform them.
    * It applies **only to fitting halves**. Every validation fold, the holdout,
      and the inner scoring folds of a hyper-parameter search keep every row.
      Because ``search.fit`` owns its inner split, the only correct way to reach
      it is a CV wrapper that shrinks each fold's *train* index -- resampling the
      training partition beforehand would corrupt every inner validation fold.
    * It **shifts the class prior**, and the framework does not correct for it.
      The usual global logit offset ``logit(p) - log(rate)`` is valid only for
      *uniform* undersampling; this sampler deliberately draws denser from sparse
      clusters, so the effective rate varies across the feature space and no
      scalar offset restores calibration. Scores stay usable for ranking and for
      the capacity cut; anything that multiplies score by exposure needs a
      calibrator fit on unsampled rows. The run report says so, loudly.

    Two double-correction traps are refused at config load: an active
    ``models[*].imbalance`` policy alongside a prior-shifting sample, and a
    sample that would leave too few positives for stratified CV.
    """

    enabled: bool = False
    purpose: str = "compute"                 # compute | rebalance | both
    strategy: str = "stratified_cluster"     # stratified_cluster | random
    apply_to: str = "all"                    # all | cv_only
    # Budget. The grammar depends on `purpose` and is enforced in validate():
    # rebalance needs the ratio and forbids the caps; compute needs exactly one
    # cap and forbids the ratio; both needs the ratio and exactly one cap.
    negative_positive_ratio: Optional[float] = None   # rebalance | both
    max_rows: Optional[int] = None                    # compute | both
    max_fraction: Optional[float] = None              # compute | both
    # 'negative' keeps every positive row verbatim -- the usual choice. 'both'
    # samples each class independently, which preserves the prior and is the
    # only combination that does NOT trip the calibration and imbalance guards.
    sample_classes: str = "negative"         # negative | both
    # Strata are formed from these columns; [] resolves to the inferred
    # categoricals. Name them explicitly on real data -- inheriting every
    # categorical picks up identifier-like columns and produces one stratum per
    # customer, which defeats the point.
    stratify_columns: List[str] = field(default_factory=list)
    max_strata: int = 50
    min_stratum_rows: int = 30
    # k-means runs on these RAW numeric columns; [] resolves to the inferred
    # numerics. Raw, not encoded: the fold's encoders are fit on the rows this
    # sampler is still choosing, so encoding first would be circular.
    cluster_columns: List[str] = field(default_factory=list)
    clusters_per_stratum: Any = "auto"       # "auto" | int
    max_clusters_per_stratum: int = 20
    min_cluster_rows: int = 10
    # How a stratum's budget is divided across its clusters. Measured on a
    # 9k-row, ~25-dimension sample against naive random undersampling: sqrt
    # closes ~37% of the worst coverage gap for ~0.009 AP; proportional closes
    # ~31% for ~0.003 but is otherwise statistically close to random (drawing in
    # proportion to cluster size reproduces the original density); equal closes
    # ~50% for ~0.024 AP. Spread and generalization trade off -- pick knowingly.
    cluster_allocation: str = "sqrt"         # sqrt | proportional | equal
    # 'spread' takes rows evenly along the distance-to-centroid ranking, so a
    # cluster contributes centre, mid-shell and boundary rows.
    within_cluster: str = "spread"           # spread | random
    kmeans_max_rows: int = 50_000            # above this -> MiniBatchKMeans
    kmeans_n_init: int = 3
    min_rows_after: int = 500                # never sample below this
    acknowledge_imbalance_double_correction: bool = False
    acknowledge_uncalibrated: bool = False

    @property
    def shifts_prior(self) -> bool:
        """True when the sample changes P(y=1) relative to the source population.

        Sampling both classes proportionally for compute reasons leaves the
        prior intact; every other combination moves it. Both safety guards key
        off this rather than off ``enabled``.
        """
        return self.enabled and not (self.purpose == "compute"
                                     and self.sample_classes == "both")


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
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    models: Dict[str, ModelSpec] = field(default_factory=dict)
    tuning: TuningConfig = field(default_factory=TuningConfig)

    # ---------------- construction ----------------
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        cfg = _from_dict(cls, copy.deepcopy(dict(data or {})))
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r") as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    @classmethod
    def load(cls, source: Union["Config", Dict[str, Any], str, Path]) -> "Config":
        """A Config from whatever a caller has: a Config, a mapping, or a YAML path."""
        if isinstance(source, Config):
            return source
        if isinstance(source, dict):
            return cls.from_dict(source)
        return cls.from_yaml(source)

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
            (self.sampling.purpose, {"compute", "rebalance", "both"}, "sampling.purpose"),
            (self.sampling.strategy, {"stratified_cluster", "random"}, "sampling.strategy"),
            (self.sampling.apply_to, {"all", "cv_only"}, "sampling.apply_to"),
            (self.sampling.sample_classes, {"negative", "both"}, "sampling.sample_classes"),
            (self.sampling.cluster_allocation, {"sqrt", "proportional", "equal"},
             "sampling.cluster_allocation"),
            (self.sampling.within_cluster, {"spread", "random"}, "sampling.within_cluster"),
        ]:
            _check_in(value, allowed, where)
        self._validate_sampling()
        if g.numeric_tolerance < 0:
            raise ValueError("preprocessing.inference_guard.numeric_tolerance must be >= 0.")
        # an unknown metric name, a bad operating point, or a primary that spans
        # several operating points is a config error here rather than a failure
        # an hour into a sweep. Imported locally so the config module stays free
        # of a module-level dependency on the metric registry.
        from .metrics import validate_metrics

        validate_metrics(self.metrics)
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

    def _validate_sampling(self) -> None:
        """Budget grammar, ranges, and the two double-correction guards.

        Everything is skipped when sampling is disabled, so a stale block in a
        config never blocks a run that does not use it.
        """
        s = self.sampling
        if not s.enabled:
            return

        caps = [s.max_rows, s.max_fraction]
        n_caps = sum(c is not None for c in caps)
        if s.purpose in ("compute", "both") and n_caps != 1:
            raise ValueError(
                f"exactly one of sampling.max_rows or sampling.max_fraction must be set "
                f"when sampling.purpose='{s.purpose}'; got {n_caps}."
            )
        if s.purpose == "rebalance" and n_caps:
            raise ValueError(
                "sampling.max_rows / sampling.max_fraction do not apply when "
                "sampling.purpose='rebalance'; the budget is negative_positive_ratio."
            )
        if s.purpose in ("rebalance", "both") and s.negative_positive_ratio is None:
            raise ValueError(
                f"sampling.negative_positive_ratio is required when "
                f"sampling.purpose='{s.purpose}'."
            )
        if s.purpose == "compute" and s.negative_positive_ratio is not None:
            raise ValueError(
                "sampling.negative_positive_ratio does not apply when "
                "sampling.purpose='compute'; use max_rows or max_fraction."
            )

        if s.negative_positive_ratio is not None and s.negative_positive_ratio <= 0:
            raise ValueError("sampling.negative_positive_ratio must be > 0.")
        if s.max_fraction is not None and not 0.0 < s.max_fraction <= 1.0:
            raise ValueError("sampling.max_fraction must be in (0, 1].")
        if s.max_rows is not None:
            if s.max_rows < 1:
                raise ValueError("sampling.max_rows must be >= 1.")
            if s.max_rows < 2 * self.split.cv.n_splits:
                raise ValueError(
                    f"sampling.max_rows={s.max_rows} leaves too few rows for "
                    f"{self.split.cv.n_splits}-fold stratified CV; must be >= "
                    f"2 * split.cv.n_splits."
                )
        for name, value in (
            ("max_strata", s.max_strata), ("min_stratum_rows", s.min_stratum_rows),
            ("min_cluster_rows", s.min_cluster_rows),
            ("max_clusters_per_stratum", s.max_clusters_per_stratum),
            ("kmeans_n_init", s.kmeans_n_init), ("min_rows_after", s.min_rows_after),
            ("kmeans_max_rows", s.kmeans_max_rows),
        ):
            if value < 1:
                raise ValueError(f"sampling.{name} must be >= 1.")
        if s.clusters_per_stratum != "auto" and not (
            isinstance(s.clusters_per_stratum, int)
            and not isinstance(s.clusters_per_stratum, bool)
            and s.clusters_per_stratum >= 1
        ):
            raise ValueError(
                f"sampling.clusters_per_stratum must be 'auto' or an integer >= 1; "
                f"got {s.clusters_per_stratum!r}."
            )

        # The prior would be corrected twice: once by dropping negatives, once by
        # the estimator's own reweighting. Keyed off `imbalance` being truthy and
        # never off its value -- zoo._imbalance_kwargs routes 'balanced' to
        # scale_pos_weight whenever the class has no class_weight parameter, so
        # the declared value does not tell you which branch fires.
        if s.shifts_prior:
            offenders = sorted(n for n, m in self.enabled_models.items() if m.imbalance)
            if offenders and not s.acknowledge_imbalance_double_correction:
                raise ValueError(
                    f"sampling shifts the class prior and models {offenders} also declare "
                    f"an 'imbalance' policy; the prior would be corrected twice. Note that "
                    f"imbalance='balanced' on an estimator without class_weight (XGBoost) "
                    f"becomes scale_pos_weight, which is computed once from the unsampled "
                    f"training labels and cloned into every fold. Remove models.*.imbalance, "
                    f"or set sampling.acknowledge_imbalance_double_correction: true to have "
                    f"it recomputed from the sampled rows and both values reported."
                )

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


__all__ = [
    "Config", "RunConfig", "DataConfig", "ColumnsConfig", "PreprocessingConfig",
    "InferenceGuardConfig",
    "NumericPreprocessing", "CategoricalPreprocessing", "SplitConfig", "CVConfig",
    "MetricsConfig", "SamplingConfig", "SelectionConfig", "ModelSpec", "TuningConfig",
    "get_dotted", "set_dotted",
]
