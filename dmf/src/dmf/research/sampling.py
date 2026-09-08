"""
Coverage-maximizing undersampling of the rows a model is fitted on.

The sampler answers one question: given a budget of rows, which rows preserve
the most of what the population looks like? Uniform random undersampling
answers "a random subset", which reproduces the original density and therefore
leaves sparse corners of the feature space unrepresented. This module instead
stratifies on declared categoricals, clusters the raw numerics inside each
stratum, and guarantees every stratum and every cluster contributes at least
one row.

Three things are deliberate and should not be "improved" without measurement:

* **Raw numerics, not the encoded matrix.** The fold's imputer, scaler and
  encoders are fitted on the rows this sampler is still choosing, so encoding
  first would be circular. The sampler therefore carries its own throwaway
  clip/impute/standardize, local to each stratum and discarded immediately.
* **Positional indices, never frames.** The harness keeps ``X`` (pandas),
  ``y`` (numpy) and the row-id array positionally parallel; returning indices
  is what lets all three be sliced identically. Returning a frame would break
  ``PredictionLog.add``'s alignment check.
* **No prior correction.** Undersampling raises every predicted probability.
  The usual global logit offset ``logit(p) - log(rate)`` undoes that only for
  *uniform* undersampling; this sampler draws denser from sparse clusters, so
  the effective rate varies across the feature space and no scalar offset
  restores calibration. The diagnostics report the shift; correcting it is a
  post-hoc calibrator's job, fitted on unsampled rows.

Measured against naive random undersampling on 9k rows in ~25 dimensions, the
gain is concentrated in the tail: average nearest-neighbour distance barely
moves (<1%), while the *worst* uncovered gap closes by 31-50% depending on
``cluster_allocation``. Coverage assertions should target p95/max, not the mean.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.model_selection import BaseCrossValidator
from sklearn.neighbors import NearestNeighbors

from ..config import Config
from ..transformers import infer_roles, to_numeric_lenient

#: joins stratify column values into one key; a control character so it cannot
#: collide with a real level
_KEY_SEP = "\x1f"
#: strata below ``min_stratum_rows`` are merged into this one
_SMALL = "__SMALL__"
#: used when no stratify column survives
_ALL = "__ALL__"


@dataclass
class SampleResult:
    """Rows to fit on, plus everything a report needs to explain the choice."""

    indices: np.ndarray                                  # positional, ascending
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.indices)


def fold_seed(cfg: Config, base: int, index: int = 0) -> int:
    """A seed unique to (run, call site, fold), so no two populations share one."""
    return int(cfg.run.random_state) + int(base) + int(index)


# --------------------------------------------------------------------------
# column resolution
# --------------------------------------------------------------------------
def resolve_sampling_columns(
    X: pd.DataFrame, cfg: Config, roles: Any = None
) -> Tuple[List[str], List[str]]:
    """``(stratify_columns, cluster_columns)`` actually usable on this frame.

    Unset lists fall back to the inferred roles. The group and time keys are
    always removed: a customer id would produce one stratum per customer, and a
    timestamp clusters into noise.
    """
    s = cfg.sampling
    if roles is None:
        roles = infer_roles(X, cfg)
    excluded = {c for c in (cfg.split.group_column, cfg.split.time_column) if c}
    strata = [c for c in (s.stratify_columns or roles.categorical)
              if c in X.columns and c not in excluded]
    clusters = [c for c in (s.cluster_columns or roles.numeric)
                if c in X.columns and c not in excluded]
    return strata, clusters


# --------------------------------------------------------------------------
# apportionment
# --------------------------------------------------------------------------
def _apportion(sizes: np.ndarray, weights: np.ndarray, budget: int, floor: int = 1) -> np.ndarray:
    """Split ``budget`` across groups by ``weights``, with a floor and a cap.

    Largest-remainder (Hamilton) apportionment. Every non-empty group receives
    at least ``floor`` before the remainder is distributed -- that floor is the
    coverage guarantee, and it is why a stratum or cluster holding 0.1% of the
    population is never zeroed out. No group exceeds its own size; overflow is
    redistributed until the budget is exactly consumed.

    When the floors alone exceed the budget, the highest-weight groups get one
    row each -- the budget is simply too small to represent everything.
    """
    sizes = np.asarray(sizes, dtype=np.int64)
    weights = np.asarray(weights, dtype=float)
    n = len(sizes)
    alloc = np.zeros(n, dtype=np.int64)
    budget = int(min(max(budget, 0), sizes.sum()))
    if budget <= 0 or n == 0:
        return alloc

    nonempty = sizes > 0
    base = np.where(nonempty, np.minimum(floor, sizes), 0)
    if base.sum() > budget:                       # cannot floor everyone
        rank = np.where(nonempty, weights, -np.inf)
        order = np.lexsort((np.arange(n), -rank))
        alloc[order[:budget]] = 1
        return np.minimum(alloc, sizes)

    alloc = base.copy()
    for _ in range(64):                           # bounded; converges in 1-2 passes
        remaining = budget - int(alloc.sum())
        if remaining <= 0:
            break
        headroom = sizes - alloc
        active = headroom > 0
        if not active.any():
            break
        w = np.where(active, weights, 0.0)
        if w.sum() <= 0:                          # degenerate weights -> fill by headroom
            w = np.where(active, headroom.astype(float), 0.0)
            if w.sum() <= 0:
                break
        share = remaining * w / w.sum()
        alloc = alloc + np.minimum(np.floor(share).astype(np.int64), headroom)
        remaining = budget - int(alloc.sum())
        if remaining <= 0:
            break
        # hand out the leftover rows by descending fractional remainder
        headroom = sizes - alloc
        rem = np.where(headroom > 0, share - np.floor(share), -np.inf)
        order = np.lexsort((np.arange(n), -rem))
        for i in order[: int(remaining)]:
            if sizes[i] - alloc[i] > 0:
                alloc[i] += 1
    return np.minimum(alloc, sizes)


# --------------------------------------------------------------------------
# the throwaway numeric space
# --------------------------------------------------------------------------
def _local_space(frame: pd.DataFrame, cols: Sequence[str], cfg: Config) -> Tuple[np.ndarray, int]:
    """Clip, impute and standardize raw numerics -- locally, then discard.

    Returns ``(matrix, n_degenerate_columns)``. Clipping first matters: a single
    1e9 dispute amount otherwise collapses k-means into one giant cluster plus a
    singleton, which destroys the coverage guarantee this module exists for.
    """
    w = cfg.preprocessing.numeric.winsorize
    lo_q, hi_q = (w.lower_quantile, w.upper_quantile) if w.enabled else (0.01, 0.99)
    out = np.empty((len(frame), len(cols)), dtype=float)
    degenerate = 0
    for j, col in enumerate(cols):
        v = to_numeric_lenient(frame[col]).to_numpy(dtype=float)
        if not np.isfinite(v).any():
            out[:, j] = 0.0
            degenerate += 1
            continue
        lo, hi = np.nanquantile(v, [lo_q, hi_q])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            v = np.clip(v, lo, hi)
        med = np.nanmedian(v)
        v = np.where(np.isfinite(v), v, med if np.isfinite(med) else 0.0)
        sd = v.std()
        out[:, j] = (v - v.mean()) / (sd if sd > 0 else 1.0)
    return out, degenerate


def _n_clusters(cfg: Config, allocation: int, n_rows: int, n_distinct: int) -> int:
    """How many clusters a stratum gets.

    Scales with the *budget*, not the population: spreading a 100-row allocation
    over 40 clusters gives 2-3 rows each, which is noise. ``sqrt(alloc / 2)``
    gives ~7 clusters at a 100-row budget and ~22 at 1000, capped so we never
    ask for more clusters than there are rows to fill them.
    """
    s = cfg.sampling
    ceiling = min(s.max_clusters_per_stratum, max(n_rows // s.min_cluster_rows, 1), n_distinct)
    if s.clusters_per_stratum != "auto":
        return int(max(min(int(s.clusters_per_stratum), ceiling), 1))
    return int(max(min(int(np.sqrt(max(allocation, 1) / 2.0)), ceiling), 1))


def _cluster_labels(Z: np.ndarray, k: int, cfg: Config, seed: int) -> np.ndarray:
    """k-means labels, or a single cluster if clustering is impossible."""
    if k <= 1 or len(Z) <= k:
        return np.zeros(len(Z), dtype=int)
    Est = MiniBatchKMeans if len(Z) > cfg.sampling.kmeans_max_rows else KMeans
    return Est(n_clusters=k, random_state=seed, n_init=cfg.sampling.kmeans_n_init).fit_predict(Z)


def _pick_within_cluster(
    Z: np.ndarray, members: np.ndarray, take: int, mode: str, rng: np.random.Generator
) -> np.ndarray:
    """Choose ``take`` rows from one cluster.

    ``spread`` walks the distance-to-centroid ranking at even intervals, so the
    cluster contributes its centre, its mid-shell and its boundary rather than
    ``take`` rows from wherever the RNG landed. It is deterministic given the
    clustering, which leaves k-means init as the only source of randomness.
    """
    if take >= len(members):
        return members
    if mode == "random":
        return rng.choice(members, size=take, replace=False)
    block = Z[members]
    d = np.linalg.norm(block - block.mean(axis=0), axis=1)
    order = members[np.lexsort((np.arange(len(members)), d))]
    return order[np.floor(np.linspace(0, len(order) - 1, take)).astype(int)]


# --------------------------------------------------------------------------
# coverage diagnostics
# --------------------------------------------------------------------------
def coverage_score(
    Z: np.ndarray, kept: np.ndarray, *, rng: np.random.Generator, max_probe: int = 2000
) -> Dict[str, float]:
    """How well ``kept`` covers the rows it was drawn from.

    For a sample of the *dropped* rows, the distance to the nearest kept row.
    Lower is better. The tail statistics are the informative ones: in high
    dimensions the mean is nearly invariant to the sampling rule, while p95 and
    max measure the regions a sample failed to represent at all.
    """
    empty = {"mean_nn_distance": None, "p95_nn_distance": None, "max_nn_distance": None}
    if len(Z) == 0 or len(kept) == 0:
        return empty
    dropped = np.setdiff1d(np.arange(len(Z)), kept, assume_unique=False)
    if len(dropped) == 0:
        return {"mean_nn_distance": 0.0, "p95_nn_distance": 0.0, "max_nn_distance": 0.0}
    if len(dropped) > max_probe:
        dropped = rng.choice(dropped, size=max_probe, replace=False)
    nn = NearestNeighbors(n_neighbors=1).fit(Z[kept])
    d = nn.kneighbors(Z[dropped], return_distance=True)[0].ravel()
    return {
        "mean_nn_distance": round(float(d.mean()), 6),
        "p95_nn_distance": round(float(np.quantile(d, 0.95)), 6),
        "max_nn_distance": round(float(d.max()), 6),
    }


def _categorical_coverage(frame: pd.DataFrame, cols: Sequence[str], kept: np.ndarray) -> Optional[float]:
    """Share of (column, level) pairs in the pool that survive into the sample."""
    if not cols or not len(kept):
        return None
    total = present = 0
    sub = frame.iloc[kept]
    for c in cols:
        levels = set(frame[c].astype(str).unique())
        total += len(levels)
        present += len(levels & set(sub[c].astype(str).unique()))
    return round(present / total, 6) if total else None


def _numeric_coverage(Z: np.ndarray, kept: np.ndarray) -> Optional[float]:
    """Mean over columns of sampled range / population range."""
    if not len(kept) or Z.shape[1] == 0:
        return None
    full = Z.max(axis=0) - Z.min(axis=0)
    sub = Z[kept].max(axis=0) - Z[kept].min(axis=0)
    ok = full > 0
    return round(float((sub[ok] / full[ok]).mean()), 6) if ok.any() else None


# --------------------------------------------------------------------------
# the sampler
# --------------------------------------------------------------------------
def _budget_for(cfg: Config, n_total: int, n_pos: int, n_pool: int) -> int:
    """Rows to keep from the sampled class, before flooring and capping."""
    s = cfg.sampling
    targets = []
    if s.negative_positive_ratio is not None:
        targets.append(int(round(s.negative_positive_ratio * max(n_pos, 1))))
    if s.max_rows is not None or s.max_fraction is not None:
        cap = s.max_rows if s.max_rows is not None else int(round(s.max_fraction * n_total))
        targets.append(int(cap) - (n_total - n_pool))    # the untouched class keeps its rows
    target = min(targets) if targets else n_pool
    return int(max(min(target, n_pool), 1))


def _sample_pool(
    X: pd.DataFrame, pool: np.ndarray, budget: int, cfg: Config, seed: int,
    strata_cols: List[str], cluster_cols: List[str], diag: Dict[str, Any],
) -> np.ndarray:
    """Stratify, cluster, allocate, and pick -- returns positions into ``X``."""
    s = cfg.sampling
    rng = np.random.default_rng(seed)
    frame = X.iloc[pool]

    # ---- strata, added greedily while the stratum count stays under budget.
    # A column that would blow the budget is *skipped*, not a stopping point:
    # an identifier-like column early in the list (merchant_id, ~150 levels)
    # would otherwise silently disable stratification altogether.
    #
    # An explicit stratify_columns list is honoured in the order written -- the
    # user's priority. An inherited list is tried lowest-cardinality first,
    # which fits the most columns under max_strata; inference order would spend
    # the whole budget on the first wide column it happened to return.
    candidates = list(strata_cols)
    if not cfg.sampling.stratify_columns:
        candidates.sort(key=lambda c: (frame[c].nunique(dropna=False), str(c)))
    used: List[str] = []
    keys = pd.Series([_ALL] * len(pool), index=frame.index)
    for col in candidates:
        vals = frame[col].astype(str).fillna(cfg.preprocessing.categorical.imputer_fill_value)
        trial = keys + _KEY_SEP + vals if used else vals
        if trial.nunique() > s.max_strata:
            continue
        keys, used = trial, used + [col]
    diag["stratify_columns_used"] = used
    diag["stratify_columns_skipped_cardinality"] = [c for c in candidates if c not in used]

    key_arr = keys.to_numpy()
    counts = pd.Series(key_arr).value_counts()
    small = set(counts.index[counts < s.min_stratum_rows])
    if small:
        key_arr = np.where(np.isin(key_arr, list(small)), _SMALL, key_arr)
    diag["n_strata_collapsed"] = len(small)

    stratum_keys = sorted(pd.unique(key_arr))
    groups = [np.flatnonzero(key_arr == k) for k in stratum_keys]
    sizes = np.array([len(g) for g in groups])
    diag["n_strata"] = len(groups)

    # ---- budget across strata: proportional, floor 1 per non-empty stratum
    per_stratum = _apportion(sizes, sizes.astype(float), budget, floor=1)

    picked: List[np.ndarray] = []
    n_clusters_total = n_singletons = fallbacks = degenerate = 0
    for i, (g, alloc) in enumerate(zip(groups, per_stratum)):
        if alloc <= 0:
            continue
        if alloc >= len(g):                     # budget covers the stratum outright
            picked.append(pool[g])
            continue
        if not cluster_cols:
            # no numeric space to cluster in: uniform draw within the stratum.
            # This is both the `strategy: random` baseline (one stratum, so a
            # plain uniform undersample) and the graceful degradation when a
            # frame has no numeric columns.
            picked.append(pool[rng.choice(g, size=int(alloc), replace=False)])
            continue
        Z, deg = _local_space(frame.iloc[g], cluster_cols, cfg)
        degenerate = max(degenerate, deg)
        k = _n_clusters(cfg, int(alloc), len(g), int(len(np.unique(Z, axis=0))))
        try:
            labels = _cluster_labels(Z, k, cfg, seed + i)
        except Exception:                       # a sampler must never abort a sweep
            labels, fallbacks = np.zeros(len(g), dtype=int), fallbacks + 1

        uniq = np.unique(labels)
        c_sizes = np.array([(labels == c).sum() for c in uniq])
        weights = {"sqrt": np.sqrt(c_sizes.astype(float)),
                   "proportional": c_sizes.astype(float),
                   "equal": np.ones(len(uniq))}[s.cluster_allocation]
        per_cluster = _apportion(c_sizes, weights, int(alloc), floor=1)
        n_clusters_total += len(uniq)
        n_singletons += int((per_cluster == 1).sum())

        for c, take in zip(uniq, per_cluster):
            if take <= 0:
                continue
            members = np.flatnonzero(labels == c)
            chosen = _pick_within_cluster(Z, members, int(take), s.within_cluster, rng)
            picked.append(pool[g[chosen]])

    diag["n_clusters_total"] = diag.get("n_clusters_total", 0) + n_clusters_total
    diag["n_singleton_clusters"] = diag.get("n_singleton_clusters", 0) + n_singletons
    diag["n_strata_cluster_fallback"] = diag.get("n_strata_cluster_fallback", 0) + fallbacks
    diag["cluster_columns_degenerate"] = max(diag.get("cluster_columns_degenerate", 0), degenerate)
    return np.sort(np.concatenate(picked)) if picked else np.array([], dtype=int)


def coverage_sample(
    X: pd.DataFrame, y: np.ndarray, cfg: Config, *, seed: int, roles: Any = None
) -> SampleResult:
    """Choose the rows a model should be fitted on.

    Returns positional indices into ``X`` (ascending, so the caller's row order
    and any time ordering are preserved) plus diagnostics describing what was
    kept and how far the class prior moved.

    Never call this on a population a metric will be computed on.
    """
    t0 = time.time()
    y = np.asarray(y).ravel()
    n = len(X)
    everything = np.arange(n)
    s = cfg.sampling
    diag: Dict[str, Any] = {
        "applied": False, "strategy": s.strategy, "purpose": s.purpose, "seed": int(seed),
        "n_rows_in": int(n), "n_positive_in": int((y == 1).sum()),
    }

    def _finish(idx: np.ndarray, reason: Optional[str] = None) -> SampleResult:
        idx = np.asarray(idx, dtype=int)
        diag["degraded_reason"] = reason
        diag["n_rows_out"] = int(len(idx))
        diag["n_positive_out"] = int((y[idx] == 1).sum()) if len(idx) else 0
        diag["sampling_rate"] = round(len(idx) / n, 6) if n else None
        p_in = diag["n_positive_in"] / n if n else 0.0
        p_out = diag["n_positive_out"] / len(idx) if len(idx) else 0.0
        diag["prevalence_in"] = round(p_in, 6)
        diag["prevalence_out"] = round(p_out, 6)
        diag["prevalence_ratio"] = round(p_out / p_in, 6) if p_in else None
        diag["prior_shift_logit"] = (
            round(float(np.log(p_out / (1 - p_out)) - np.log(p_in / (1 - p_in))), 6)
            if 0 < p_in < 1 and 0 < p_out < 1 else None
        )
        diag["sampling_seconds"] = round(time.time() - t0, 3)
        return SampleResult(indices=idx, diagnostics=diag)

    if not s.enabled:
        return _finish(everything)
    if n <= s.min_rows_after:
        return _finish(everything, "below_min_rows_after")

    # Columns are resolved once and used for two different things: the sampling
    # logic, and the coverage diagnostics. The `random` strategy switches the
    # former off but must keep the latter on -- it is the baseline arm that
    # coverage is compared against, and a baseline with no measurements is
    # useless.
    strata_cols, cluster_cols = resolve_sampling_columns(X, cfg, roles)
    diag["cluster_columns_used"] = cluster_cols
    if s.strategy == "random":
        sample_strata, sample_clusters = [], []
    else:
        sample_strata, sample_clusters = strata_cols, cluster_cols
        if not cluster_cols:
            diag["degraded_reason"] = "no_numeric_columns"

    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    pools = [neg] if s.sample_classes == "negative" else [neg, pos]
    keep: List[np.ndarray] = [pos] if s.sample_classes == "negative" else []

    for pool in pools:
        if len(pool) == 0:
            continue
        budget = _budget_for(cfg, n, len(pos), len(pool))
        if budget >= len(pool):
            diag["budget_not_binding"] = True
            keep.append(pool)
            continue
        keep.append(_sample_pool(X, pool, budget, cfg, seed, sample_strata, sample_clusters, diag))

    idx = np.sort(np.concatenate([k for k in keep if len(k)])) if keep else everything
    if len(idx) < s.min_rows_after:                    # never sample below the floor
        return _finish(everything, "would_fall_below_min_rows_after")

    n_pos_out = int((y[idx] == 1).sum())
    if n_pos_out < 2 * cfg.split.cv.n_splits:
        raise ValueError(
            f"sampling would leave {n_pos_out} positives for "
            f"{cfg.split.cv.n_splits}-fold stratified CV; raise "
            f"sampling.negative_positive_ratio or lower split.cv.n_splits."
        )

    diag["applied"] = True
    # measured on the resolved columns, not the sampling ones, so every strategy
    # -- including the random baseline -- reports comparable numbers
    if cluster_cols:
        Z, _ = _local_space(X, cluster_cols, cfg)
        diag.update(coverage_score(Z, idx, rng=np.random.default_rng(seed)))
        diag["coverage_retained_numeric"] = _numeric_coverage(Z, idx)
    if strata_cols:
        diag["coverage_retained_categorical"] = _categorical_coverage(X, strata_cols, idx)
    return _finish(idx)


# --------------------------------------------------------------------------
# tuning: the only correct insertion point inside a search
# --------------------------------------------------------------------------
class UndersampledSplit(BaseCrossValidator):
    """Wrap a splitter and undersample **only the train index** of each fold.

    ``GridSearchCV`` / ``RandomizedSearchCV`` own their inner split: they call
    ``cv.split(X, y, groups)`` themselves and score on the half they produce.
    Undersampling ``X_tr`` before ``search.fit`` would therefore shrink every
    inner *validation* fold too, and the score guiding the search would be
    computed on a resampled population. Replacing the train index of each
    yielded pair is the only place a resampler can sit inside a search without
    touching what it is scored on.

    The split list is memoized because ``BaseSearchCV`` may iterate the CV more
    than once -- without that, k-means would re-run per candidate and inner
    folds could differ between candidates.
    """

    def __init__(self, inner, X: pd.DataFrame, y: np.ndarray, cfg: Config, seed: int):
        self.inner = inner
        self.X = X
        self.y = np.asarray(y).ravel()
        self.cfg = cfg
        self.seed = int(seed)
        self._splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
        self.fold_diagnostics_: List[Dict[str, Any]] = []

    def _build(self, X, y, groups):
        if self._splits is not None:
            return self._splits
        splits, diags = [], []
        for f, (tr, te) in enumerate(self.inner.split(X, y, groups)):
            res = coverage_sample(self.X.iloc[tr], self.y[tr], self.cfg, seed=self.seed + f)
            diags.append(res.diagnostics)
            # res.indices is positional WITHIN the fold's train subframe, so it
            # must be mapped back through tr. te is yielded untouched.
            splits.append((tr[res.indices], te))
        self._splits, self.fold_diagnostics_ = splits, diags
        return splits

    def split(self, X, y=None, groups=None):
        if len(X) != len(self.X):
            raise ValueError(
                f"UndersampledSplit was built over {len(self.X)} rows but split() received "
                f"{len(X)}; it must wrap the same population it was constructed with."
            )
        yield from self._build(X, y, groups)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.inner.get_n_splits(X, y, groups)

    @property
    def mean_sampling_rate(self) -> Optional[float]:
        rates = [d.get("sampling_rate") for d in self.fold_diagnostics_ if d.get("sampling_rate")]
        return round(float(np.mean(rates)), 6) if rates else None


__all__ = [
    "SampleResult", "coverage_sample", "resolve_sampling_columns", "coverage_score",
    "fold_seed", "UndersampledSplit",
]
