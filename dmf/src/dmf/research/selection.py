"""
``ModelSelectionHarness`` -- the experiment wrapper around the core pipeline.

What it answers, in order:

1. **Which variables, in what order?**  One of three ordering strategies
   (see :mod:`dmf.ordering`), estimated on the training partition only.
2. **Which model architecture, at which variable count?**  A full
   ``model x k`` grid, each cell scored by stratified cross-validation with the
   feature pipeline refit inside every fold.
3. **Is the k-th variable actually earning its place?**  Paired, fold-level
   deltas between the k and k-1 specifications, with a Nadeau-Bengio corrected
   paired t-test or a Wilcoxon signed-rank test.
4. **Which three specifications should go forward?**  Ranked by mean
   out-of-sample primary metric, with a one-standard-error parsimony rule to
   pick the final champion.
5. **Does it hold up?**  A single confirmatory evaluation on the untouched
   stratified holdout, including calibration and a gains table.

The harness never touches the holdout during selection. Nothing here is part
of the deployed artifact -- its only durable output is a variable list plus a
model specification, which are handed back to :func:`dmf.pipeline.build_model_pipeline`.
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import (
    GridSearchCV,
    GroupShuffleSplit,
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)

from ..config import Config, ModelSpec
from ..metrics import (
    decile_table,
    evaluate_predictions,
    greater_is_better,
    make_scorers,
    metric_names,
    operating_point,
    orient,
    reference_quantiles,
    score_vector,
)
from ..pipeline import DisputeFeaturePipeline, build_model_pipeline
from ..reporting import StepReport, json_safe, run_lineage, summarize_frame, summarize_target
from ..transformers import infer_roles
from .evaluate import PredictionLog, implied_thresholds, write_prediction_artifacts
from .ordering import rank_variables
from .zoo import build_estimator, config_for_model


@dataclass
class SelectionResult:
    """Everything the harness learned, in analysable form."""

    leaderboard: pd.DataFrame
    marginal_gains: pd.DataFrame
    orderings: Dict[str, Any]
    top_specs: List[Dict[str, Any]]
    best_per_model: pd.DataFrame
    selected: Dict[str, Any]
    holdout_metrics: Dict[str, Any] = field(default_factory=dict)
    holdout_deciles: Optional[pd.DataFrame] = None
    holdout_slices: Optional[pd.DataFrame] = None
    fitted_model: Any = None
    feature_report: Dict[str, Any] = field(default_factory=dict)
    report: Optional[StepReport] = None
    config: Optional[Config] = None
    # row-level (row_id, y_true, y_score) for every stage run.save_predictions
    # captured, plus the CV fold-membership map -- see dmf.research.evaluate
    predictions: Optional[pd.DataFrame] = None
    fold_assignments: Optional[pd.DataFrame] = None

    @property
    def selected_features(self) -> List[str]:
        return list(self.selected.get("features", []))

    @property
    def selected_model(self) -> str:
        return str(self.selected.get("model", ""))

    def summary(self) -> str:
        return self.report.render() if self.report else ""


class ModelSelectionHarness:
    """Config-driven variable-selection and model-specification search."""

    def __init__(self, config: Config):
        self.config = config
        self.report = StepReport(run=config.run.name)
        self.groups_tr_: Optional[np.ndarray] = None
        self._orderings_cache: Dict[str, Any] = {}
        self.predictions_ = PredictionLog(config.run.save_predictions)
        self.row_ids_: Optional[np.ndarray] = None
        self.row_ids_tr_: Optional[np.ndarray] = None
        self.row_ids_ho_: Optional[np.ndarray] = None
        self.positive_label_: Any = None
        self._row_id_source: str = "dataframe_index"

    # ------------------------------------------------------------------
    # public entry point
    # ------------------------------------------------------------------
    def run(self, X: Optional[pd.DataFrame] = None, y: Optional[Any] = None) -> SelectionResult:
        cfg = self.config
        t0 = time.time()

        X, y = self._load(X, y)
        X_tr, X_ho, y_tr, y_ho = self._split(X, y)
        candidates = self._candidate_variables(X_tr)

        zoo = self._instantiate_zoo(y_tr)
        orderings = self._compute_orderings(X_tr, y_tr, zoo, candidates)
        leaderboard, fold_scores = self._evaluate_grid(X_tr, y_tr, zoo, orderings, candidates)
        gains = self._marginal_gains(leaderboard, fold_scores)

        if cfg.tuning.enabled:
            leaderboard, fold_scores, zoo = self._tune(X_tr, y_tr, zoo, orderings, leaderboard, fold_scores)
            gains = self._marginal_gains(leaderboard, fold_scores)

        top_specs, best_per_model, selected = self._select(leaderboard, orderings)
        fitted, holdout_metrics, deciles, slices, feat_report = self._confirm(
            X_tr, y_tr, X_ho, y_ho, X, y, zoo, selected
        )

        self.report.add("wall_clock", total_seconds=round(time.time() - t0, 2))

        result = SelectionResult(
            leaderboard=leaderboard,
            marginal_gains=gains,
            orderings=orderings,
            top_specs=top_specs,
            best_per_model=best_per_model,
            selected=selected,
            holdout_metrics=holdout_metrics,
            holdout_deciles=deciles,
            holdout_slices=slices,
            fitted_model=fitted,
            feature_report=feat_report,
            report=self.report,
            config=cfg,
            predictions=self.predictions_.to_frame() if self.predictions_.enabled else None,
            fold_assignments=self.predictions_.fold_frame() if self.predictions_.enabled else None,
        )
        self._write_artifacts(result)
        if cfg.run.verbose:
            print(self.report.render())
        return result

    # ------------------------------------------------------------------
    # step 1: data
    # ------------------------------------------------------------------
    def _load(self, X, y) -> Tuple[pd.DataFrame, np.ndarray]:
        cfg = self.config
        if X is None:
            if not cfg.data.path:
                raise ValueError("No dataframe supplied and data.path is unset in the config.")
            reader = {"csv": pd.read_csv, "parquet": pd.read_parquet}[cfg.data.format]
            frame = reader(cfg.data.path, **cfg.data.read_kwargs)
        else:
            frame = X.copy()

        if y is None:
            if cfg.data.target not in frame.columns:
                raise KeyError(f"Target column '{cfg.data.target}' not found in the data.")
            y_ser = frame[cfg.data.target]
            frame = frame.drop(columns=[cfg.data.target])
        else:
            y_ser = pd.Series(np.asarray(y).ravel(), index=frame.index)

        y_arr, positive_label = _binarize_target(y_ser, cfg.data.positive_label)
        self.positive_label_ = positive_label

        # resolve the row identity that keys the prediction store, before
        # sampling so the positional subset below keeps ids aligned
        id_col = cfg.data.id_column
        if id_col:
            if id_col not in frame.columns:
                raise KeyError(
                    f"data.id_column='{id_col}' is not present in the data. "
                    f"Unset it to fall back to the DataFrame index."
                )
            row_ids = frame[id_col].to_numpy()
            frame = frame.drop(columns=[id_col])   # an identifier is never a feature
            self._row_id_source = f"data.id_column:{id_col}"
        else:
            row_ids = frame.index.to_numpy()
            self._row_id_source = "dataframe_index"

        if cfg.data.sample_frac:
            # sample positions, not labels: frame.index holds labels while y_arr
            # is positional, and pairing the two by index silently scrambles the
            # target on any non-contiguous or non-integer index.
            keep = np.sort(
                np.random.default_rng(cfg.run.random_state).choice(
                    len(frame), size=max(int(round(len(frame) * cfg.data.sample_frac)), 1),
                    replace=False,
                )
            )
            frame, y_arr, row_ids = frame.iloc[keep], y_arr[keep], row_ids[keep]

        self.row_ids_ = row_ids
        drop = [c for c in cfg.columns.drop if c in frame.columns]
        if drop:
            frame = frame.drop(columns=drop)

        n_pos = int(y_arr.sum())
        n_splits = cfg.split.cv.n_splits
        if n_pos < 2 * n_splits:
            raise ValueError(
                f"Only {n_pos} positives for {n_splits}-fold stratified CV; "
                f"reduce split.cv.n_splits or supply more positive cases."
            )

        self.report.add("lineage", **run_lineage(cfg.to_dict(), frame))
        load = summarize_frame(frame, "features")
        self.report.add("data_load", **load)
        self.report.add(
            "data_target",
            **summarize_target(y_arr),
            positive_label=str(positive_label),
            # a fold with a handful of positives makes every fold statistic
            # unstable, which the leaderboard cannot show on its own
            expected_positives_per_validation_fold=round(
                n_pos * (1 - cfg.split.holdout_size) / n_splits, 1
            ),
            thin_positive_folds=bool(n_pos * (1 - cfg.split.holdout_size) / n_splits < 10),
        )
        self.report.add(
            "data_quality",
            n_dropped_columns=len(drop),
            dropped_columns=drop,
            duplicate_row_rate=load["duplicate_row_rate"],
            duplicate_rows_material=bool((load["duplicate_row_rate"] or 0) > 0.01),
            total_missing_rate=load["total_missing_rate"],
        )
        return frame, y_arr

    def _split(self, X: pd.DataFrame, y: np.ndarray):
        cfg = self.config
        strategy = cfg.split.strategy
        groups = self._series(X, cfg.split.group_column)

        if strategy == "time":
            order = np.argsort(self._series(X, cfg.split.time_column).to_numpy(), kind="mergesort")
            cut = int(round(len(order) * (1 - cfg.split.holdout_size)))
            tr_idx, ho_idx = order[:cut], order[cut:]
        elif strategy == "group":
            splitter = GroupShuffleSplit(n_splits=1, test_size=cfg.split.holdout_size,
                                         random_state=cfg.run.random_state)
            tr_idx, ho_idx = next(splitter.split(X, y, groups=groups))
        elif strategy == "group_time":
            # A group's side is decided by its *earliest* timestamp: the most
            # recent holdout_size share of groups (all activity inside the
            # newest window) becomes the holdout; any group with transactions
            # before that window stays whole in training. No group straddles.
            t = self._series(X, cfg.split.time_column)
            first_seen = t.groupby(groups.to_numpy()).min().sort_values(kind="mergesort")
            n_ho = max(1, int(round(len(first_seen) * cfg.split.holdout_size)))
            ho_groups = set(first_seen.index[-n_ho:])
            mask = groups.isin(ho_groups).to_numpy()
            tr_idx, ho_idx = np.flatnonzero(~mask), np.flatnonzero(mask)
        else:
            tr_idx, ho_idx = train_test_split(
                np.arange(len(X)),
                test_size=cfg.split.holdout_size,
                random_state=cfg.run.random_state,
                stratify=y if cfg.split.stratify else None,
                shuffle=True,
            )

        X_tr, X_ho = X.iloc[tr_idx], X.iloc[ho_idx]
        y_tr, y_ho = y[tr_idx], y[ho_idx]
        self.groups_tr_ = None if groups is None else groups.iloc[tr_idx].to_numpy()
        # _split is also exercised directly (tests, notebooks) without _load
        # having resolved row identity first; fall back to the frame's index
        if self.row_ids_ is None:
            self.row_ids_ = X.index.to_numpy()
        self.row_ids_tr_, self.row_ids_ho_ = self.row_ids_[tr_idx], self.row_ids_[ho_idx]

        extra: Dict[str, Any] = {}
        if strategy in ("group", "group_time"):
            g_tr, g_ho = set(groups.iloc[tr_idx]), set(groups.iloc[ho_idx])
            extra.update(group_column=cfg.split.group_column,
                         n_groups_train=len(g_tr), n_groups_holdout=len(g_ho),
                         n_groups_leaked=len(g_tr & g_ho))
        if strategy in ("time", "group_time"):
            t = self._series(X, cfg.split.time_column)
            extra.update(time_column=cfg.split.time_column,
                         train_period=[str(t.iloc[tr_idx].min()), str(t.iloc[tr_idx].max())],
                         holdout_period=[str(t.iloc[ho_idx].min()), str(t.iloc[ho_idx].max())])
        if strategy == "group_time":
            # spanners go to training, so the holdout skews toward short-tenure
            # groups; surface that skew rather than leaving it implicit
            extra.update(median_rows_per_group_train=float(np.median(np.bincount(
                             pd.factorize(groups.iloc[tr_idx])[0]))),
                         median_rows_per_group_holdout=float(np.median(np.bincount(
                             pd.factorize(groups.iloc[ho_idx])[0]))))

        self.report.add(
            "holdout_split",
            strategy=strategy,
            holdout_size=cfg.split.holdout_size,
            stratified=cfg.split.stratify and strategy == "random",
            **extra,
            n_train=len(X_tr),
            n_holdout=len(X_ho),
            train_prevalence=round(float(y_tr.mean()), 6),
            holdout_prevalence=round(float(y_ho.mean()), 6),
            prevalence_abs_diff=round(abs(float(y_tr.mean()) - float(y_ho.mean())), 6),
            n_train_positive=int(y_tr.sum()),
            n_holdout_positive=int(y_ho.sum()),
        )
        return X_tr, X_ho, y_tr, y_ho

    def _candidate_variables(self, X: pd.DataFrame) -> List[str]:
        """Type every column once, on the training partition, and drop the dead ones."""
        cfg = self.config
        scoped = cfg.copy()
        scoped.columns.drop = list(cfg.columns.drop) + [
            c for c in (cfg.split.group_column, cfg.split.time_column) if c
        ]
        roles = infer_roles(X, scoped)
        candidates = roles.all
        if not candidates:
            raise ValueError(
                f"No candidate variables survived role resolution. Dropped: {roles.dropped}"
            )
        self.report.add(
            "candidate_variables",
            n_candidates=len(candidates),
            source="config.columns" if cfg.declared_features else "auto_inferred",
            **roles.report(),
        )
        return candidates

    def _series(self, X: pd.DataFrame, column: Optional[str]) -> Optional[pd.Series]:
        if not column:
            return None
        if column not in X.columns:
            raise KeyError(f"split column '{column}' is not present in the data.")
        return X[column]

    def _cv(self, n_splits: Optional[int] = None, seed_offset: int = 0):
        """The resampling scheme implied by ``split.strategy``.

        ``group`` -> StratifiedGroupKFold, so a customer never appears in both
        the fitting and the scoring half of a fold.
        ``time``  -> TimeSeriesSplit on time-ordered rows: every fold trains on
        the past and scores the future, which is how the model will be used.
        ``group_time`` -> StratifiedGroupKFold as well: the temporal guarantee
        already lives in the holdout assignment, and within training the
        binding requirement is that every group stays whole in each fold.

        ``n_splits`` / ``seed_offset`` exist for the *inner* tuning loop, which
        must not reproduce the outer folds: hyper-parameters selected on the
        same folds they are later scored against make tuned leaderboard rows
        incomparable with untuned ones. A different fold count and a shifted
        seed keep inner and outer resampling distinct under every strategy.
        """
        cfg = self.config
        cv_cfg = cfg.split.cv
        n = n_splits or cv_cfg.n_splits
        seed = cfg.run.random_state + seed_offset
        if cfg.split.strategy in ("group", "group_time"):
            return StratifiedGroupKFold(n_splits=n, shuffle=cv_cfg.shuffle,
                                        random_state=seed if cv_cfg.shuffle else None)
        if cfg.split.strategy == "time":
            return TimeSeriesSplit(n_splits=n)
        if cv_cfg.n_repeats > 1 and n_splits is None:
            return RepeatedStratifiedKFold(
                n_splits=n, n_repeats=cv_cfg.n_repeats, random_state=seed,
            )
        return StratifiedKFold(
            n_splits=n, shuffle=cv_cfg.shuffle,
            random_state=seed if cv_cfg.shuffle else None,
        )

    # ------------------------------------------------------------------
    # step 2: zoo + orderings
    # ------------------------------------------------------------------
    def _instantiate_zoo(self, y_tr: np.ndarray) -> Dict[str, Dict[str, Any]]:
        cfg = self.config
        zoo: Dict[str, Dict[str, Any]] = {}
        skipped = {}
        for name, spec in cfg.enabled_models.items():
            try:
                est = build_estimator(spec, cfg.run.random_state, cfg.run.n_jobs, y_tr)
            except ImportError as exc:
                skipped[name] = str(exc).split(".")[0]
                continue
            model_cfg = config_for_model(cfg, spec)
            # A validation fold routinely contains category levels the training
            # folds did not, so guard warnings during the sweep are noise. The
            # production refit below restores the configured setting.
            model_cfg.preprocessing.inference_guard.warn = False
            zoo[name] = {"spec": spec, "estimator": est, "config": model_cfg}
        if not zoo:
            raise RuntimeError("No estimators could be instantiated from the configured zoo.")
        self.report.add(
            "estimator_zoo",
            n_models=len(zoo),
            models=list(zoo),
            tags={k: v["spec"].tag for k, v in zoo.items()},
            n_skipped=len(skipped),
            skipped=skipped,
        )
        return zoo

    def _compute_orderings(self, X_tr, y_tr, zoo, candidates) -> Dict[str, Any]:
        cfg = self.config
        ref = cfg.selection.ordering_reference_model
        orderings: Dict[str, Any] = {}

        if ref != "per_model":
            entry = zoo.get(ref)
            if entry is None:
                raise ValueError(f"ordering_reference_model '{ref}' is not among instantiable models.")
            ordered, rep = rank_variables(
                X_tr, y_tr, entry["config"], entry["estimator"], candidates
            )
            for name in zoo:
                orderings[name] = {"ordering": ordered, "report": rep, "source": f"shared:{ref}"}
            self.report.add(
                "variable_ordering",
                mode="shared",
                reference_model=ref,
                strategy=rep.get("strategy"),
                n_candidates=len(candidates),
                top10=ordered[:10],
            )
        else:
            for name, entry in zoo.items():
                ordered, rep = rank_variables(
                    X_tr, y_tr, entry["config"], entry["estimator"], candidates
                )
                orderings[name] = {"ordering": ordered, "report": rep, "source": "per_model"}
            first = next(iter(orderings.values()))
            agreement = _ordering_agreement({k: v["ordering"] for k, v in orderings.items()})
            self.report.add(
                "variable_ordering",
                mode="per_model",
                strategy=first["report"].get("strategy"),
                n_candidates=len(candidates),
                mean_rank_correlation=agreement["mean_spearman"],
                top1_agreement=agreement["top1_agreement"],
                top5_jaccard=agreement["top5_jaccard"],
                per_model_top5={k: v["ordering"][:5] for k, v in orderings.items()},
            )
        self._orderings_cache = orderings
        return orderings

    # ------------------------------------------------------------------
    # step 3: the model x k grid
    # ------------------------------------------------------------------
    def _k_values(self, n_candidates: int) -> List[int]:
        s = self.config.selection
        k_max = min(s.k_max or n_candidates, n_candidates)
        return list(range(s.k_min, k_max + 1, s.k_step))

    def _evaluate_grid(self, X_tr, y_tr, zoo, orderings, candidates):
        cfg = self.config
        primary = cfg.metrics.primary
        cv = self._cv()
        t0 = time.time()

        rows, fold_scores = self._grid_nested(X_tr, y_tr, zoo, candidates, cv)

        leaderboard = pd.DataFrame(rows)
        leaderboard = leaderboard.sort_values(
            f"cv_{primary}_mean", ascending=not greater_is_better(primary), kind="mergesort"
        ).reset_index(drop=True)
        leaderboard.insert(0, "rank", np.arange(1, len(leaderboard) + 1))

        best = leaderboard.iloc[0]
        self.report.add(
            "cv_grid",
            n_variants=len(leaderboard),
            n_models=len(zoo),
            k_values=self._k_values(len(next(iter(orderings.values()))["ordering"])),
            cv_scheme=f"{type(cv).__name__}({cfg.split.cv.n_splits}x{cfg.split.cv.n_repeats})",
            primary_metric=primary,
            estimate="nested: variables re-ranked inside each fold",
            best_variant=f"{best['model']}@k={int(best['k'])}",
            best_cv_score=float(best[f"cv_{primary}_mean"]),
            worst_cv_score=float(leaderboard.iloc[-1][f"cv_{primary}_mean"]),
            spread=round(float(best[f"cv_{primary}_mean"] - leaderboard.iloc[-1][f"cv_{primary}_mean"]), 6),
            median_overfit_gap=round(float(leaderboard["overfit_gap"].median()), 6)
            if "overfit_gap" in leaderboard else None,
            n_failed_cells=int(leaderboard[f"cv_{primary}_mean"].isna().sum()),
            elapsed_seconds=round(time.time() - t0, 2),
        )
        return leaderboard, fold_scores

    def _grid_nested(self, X_tr, y_tr, zoo, candidates, cv):
        """Re-rank the variables inside every fold, then score the k-subsets.

        This measures the *procedure* -- "rank the variables, keep the top k,
        fit" -- rather than one fixed variable list chosen with the help of the
        validation rows. It is the only construction under which the leaderboard
        number is an honest out-of-sample estimate.

        A side effect worth having: because each fold picks its own subset, the
        frequency with which a variable survives into the top k across folds is
        a direct measure of how stable the selection is. That is reported.
        """
        cfg = self.config
        primary = cfg.metrics.primary
        ks = self._k_values(len(candidates))
        k_cap = max(ks)

        cells: Dict[Tuple[str, int], Dict[str, List[float]]] = {}
        stability: Dict[str, Dict[str, int]] = {m: {} for m in zoo}
        n_folds = 0

        for f, (tr_idx, va_idx) in enumerate(cv.split(X_tr, y_tr, self.groups_tr_)):
            n_folds += 1
            repeat, fold = divmod(f, cfg.split.cv.n_splits)
            Xf, yf = X_tr.iloc[tr_idx], y_tr[tr_idx]
            Xv, yv = X_tr.iloc[va_idx], y_tr[va_idx]
            ids_f, ids_v = self.row_ids_tr_[tr_idx], self.row_ids_tr_[va_idx]
            self.predictions_.add_fold(fold, repeat, ids_v)

            for model_name, entry in zoo.items():
                fold_order, _ = rank_variables(
                    Xf, yf, entry["config"], entry["estimator"], candidates
                )
                for var in fold_order[:k_cap]:
                    stability[model_name][var] = stability[model_name].get(var, 0) + 1

                for k in ks:
                    cell = cells.setdefault((model_name, k), {"fit_time": []})
                    pipe = build_model_pipeline(entry["config"], fold_order[:k], entry["estimator"])
                    self._score_cell(cell, pipe, Xf, yf, Xv, yv, ids_f, ids_v,
                                     model_name, k, fold, repeat)

        # the reported variable list is the one the full training partition
        # chooses; the *score* beside it is the nested estimate above
        full_orderings = {m: e["ordering"] for m, e in
                          ((m, self._orderings_cache[m]) for m in zoo)}
        rows, fold_scores = [], {}
        for (model_name, k), cell in cells.items():
            cvres = {key: np.asarray(vals, dtype=float) for key, vals in cell.items()}
            features = full_orderings[model_name][:k]
            rows.append(self._row_from_cv(model_name, zoo[model_name]["spec"], k, features,
                                          cvres, primary))
            fold_scores[(model_name, k)] = np.asarray(
                [orient(primary, v) for v in cvres[f"test_{primary}"]], dtype=float
            )

        self.report.add(
            "selection_stability",
            n_folds=n_folds,
            k_cap=k_cap,
            note="share of folds in which a variable survived into the top-k ranking",
            always_selected={
                m: sorted(v for v, c in counts.items() if c == n_folds)
                for m, counts in stability.items()
            },
            selection_frequency={
                m: {v: round(c / n_folds, 3) for v, c in
                    sorted(counts.items(), key=lambda kv: -kv[1])[:15]}
                for m, counts in stability.items()
            },
        )
        return rows, fold_scores

    def _score_cell(self, cell, pipe, Xf, yf, Xv, yv, ids_f, ids_v,
                    model_name, k, fold, repeat) -> None:
        """Fit one grid cell on one fold and score it from a single prediction.

        ``predict_proba`` runs once per partition and every configured metric
        is derived from that vector (in the sklearn signed-scorer convention,
        so downstream ``orient`` calls are unchanged). The same vector feeds
        the prediction store, which is what makes every leaderboard number
        recomputable from the stored rows, bit for bit.
        """
        cfg = self.config
        names = metric_names(cfg.metrics)
        t = time.time()
        try:
            pipe.fit(Xf, yf)
        except Exception:      # a failed cell must not abort the sweep
            for name in names:
                cell.setdefault(f"test_{name}", []).append(np.nan)
                if cfg.metrics.compute_train_scores:
                    cell.setdefault(f"train_{name}", []).append(np.nan)
            cell["fit_time"].append(np.nan)
            return
        cell["fit_time"].append(time.time() - t)

        proba_v = _safe_proba(pipe, Xv)
        test = score_vector(yv, proba_v, cfg.metrics, signed=True) if proba_v is not None else {}
        for name in names:
            cell.setdefault(f"test_{name}", []).append(test.get(name, np.nan))
        if proba_v is not None:
            self.predictions_.add("cv", model_name, k, fold, repeat, ids_v, yv, proba_v)

        if cfg.metrics.compute_train_scores or self.predictions_.wants("cv_train"):
            proba_f = _safe_proba(pipe, Xf)
            if cfg.metrics.compute_train_scores:
                train = (score_vector(yf, proba_f, cfg.metrics, signed=True)
                         if proba_f is not None else {})
                for name in names:
                    cell.setdefault(f"train_{name}", []).append(train.get(name, np.nan))
            if proba_f is not None:
                self.predictions_.add("cv_train", model_name, k, fold, repeat, ids_f, yf, proba_f)

    def _row_from_cv(self, model_name, spec: ModelSpec, k, features, cvres, primary) -> Dict[str, Any]:
        cfg = self.config
        row: Dict[str, Any] = {
            "model": model_name,
            "family": spec.family,
            "tag": spec.tag,
            "k": int(k),
            "features": ",".join(features),
            "fit_seconds": round(float(np.nanmean(cvres["fit_time"])), 4),
        }
        for name in metric_names(cfg.metrics):
            test = np.asarray([orient(name, v) for v in cvres[f"test_{name}"]], dtype=float)
            row[f"cv_{name}_mean"] = round(float(np.nanmean(test)), 6)
            row[f"cv_{name}_std"] = round(float(np.nanstd(test, ddof=1)), 6) if len(test) > 1 else 0.0
            # Nadeau-Bengio corrected SE, not std/sqrt(n): folds share training
            # data, so the naive SE is ~33% too narrow at 5 folds and the 1-SE
            # parsimony band built from it under-selects. Same variance model
            # as _paired_test, so the band and the test can never disagree.
            row[f"cv_{name}_se"] = (
                round(float(np.nanstd(test, ddof=1) * _nb_factor(len(test), cfg.split.cv.n_splits)), 6)
                if len(test) > 1 else 0.0
            )
        if cfg.metrics.compute_train_scores:
            train = np.asarray([orient(primary, v) for v in cvres[f"train_{primary}"]], dtype=float)
            row[f"train_{primary}_mean"] = round(float(np.nanmean(train)), 6)
            row["overfit_gap"] = round(float(np.nanmean(train) - row[f"cv_{primary}_mean"]), 6)
        return row

    # ------------------------------------------------------------------
    # step 4: marginal value of the k-th variable
    # ------------------------------------------------------------------
    def _marginal_gains(self, leaderboard: pd.DataFrame, fold_scores) -> pd.DataFrame:
        cfg = self.config
        primary = cfg.metrics.primary
        n_splits = cfg.split.cv.n_splits
        rows = []
        for model_name in leaderboard["model"].unique():
            ks = sorted(leaderboard.loc[leaderboard["model"] == model_name, "k"].astype(int).tolist())
            for prev_k, k in zip(ks, ks[1:]):
                a = fold_scores.get((model_name, prev_k))
                b = fold_scores.get((model_name, k))
                if a is None or b is None:
                    continue
                # deltas are expressed as "improvement", so for a loss-type
                # primary (log_loss, brier) a decrease is the gain
                d = (b - a) if greater_is_better(primary) else (a - b)
                stat = _paired_test(d, cfg.selection.marginal_gain_test, n_splits)
                added = _added_variable(leaderboard, model_name, prev_k, k)
                rows.append(
                    {
                        "model": model_name,
                        "from_k": prev_k,
                        "to_k": k,
                        "added_variable": added,
                        f"{primary}_from": round(float(np.nanmean(a)), 6),
                        f"{primary}_to": round(float(np.nanmean(b)), 6),
                        "mean_delta": round(float(np.nanmean(d)), 6),
                        "relative_delta_pct": round(float(100 * np.nanmean(d) / abs(np.nanmean(a))), 4)
                        if np.nanmean(a) else None,
                        "sd_delta": round(float(np.nanstd(d, ddof=1)), 6) if len(d) > 1 else 0.0,
                        "n_folds_improved": int(np.nansum(d > 0)),
                        "n_folds": int(len(d)),
                        "test": stat["test"],
                        "statistic": stat["statistic"],
                        "p_value": stat["p_value"],
                        "verdict": stat["verdict"],
                    }
                )
        gains = pd.DataFrame(rows)
        if len(gains):
            sig = gains[gains["verdict"] == "improves"]
            self.report.add(
                "marginal_gains",
                n_comparisons=len(gains),
                test=cfg.selection.marginal_gain_test,
                n_significant_improvements=int(len(sig)),
                n_significant_degradations=int((gains["verdict"] == "degrades").sum()),
                largest_single_gain=round(float(gains["mean_delta"].max()), 6),
                median_gain=round(float(gains["mean_delta"].median()), 6),
                first_non_improving_k={
                    m: _first_non_improving(gains, m) for m in gains["model"].unique()
                },
                note="fold-level deltas are correlated; p-values use the Nadeau-Bengio "
                     "variance correction for paired_t and are indicative only for wilcoxon",
            )
        return gains

    # ------------------------------------------------------------------
    # step 5: selection
    # ------------------------------------------------------------------
    def _select(self, leaderboard: pd.DataFrame, orderings) -> Tuple[List[Dict], pd.DataFrame, Dict]:
        cfg = self.config
        primary = cfg.metrics.primary
        mean_col, se_col = f"cv_{primary}_mean", f"cv_{primary}_se"
        asc = not greater_is_better(primary)

        lb = leaderboard.dropna(subset=[mean_col]).copy()
        if lb.empty:
            raise RuntimeError("Every grid cell failed to score; check the estimator zoo and data.")

        ranked = lb.sort_values(mean_col, ascending=asc, kind="mergesort")
        if cfg.selection.top_n_distinct_models:
            ranked = ranked.drop_duplicates(subset="model", keep="first")
        top = ranked.head(cfg.selection.top_n)
        top_specs = [
            {
                "rank": i + 1,
                "model": r["model"],
                "family": r["family"],
                "tag": r["tag"],
                "k": int(r["k"]),
                "features": r["features"].split(","),
                f"cv_{primary}_mean": float(r[mean_col]),
                f"cv_{primary}_se": float(r[se_col]),
                "overfit_gap": float(r.get("overfit_gap", np.nan)) if "overfit_gap" in r else None,
            }
            for i, (_, r) in enumerate(top.iterrows())
        ]

        idx = lb.groupby("model")[mean_col].idxmax() if not asc else lb.groupby("model")[mean_col].idxmin()
        best_per_model = lb.loc[idx].sort_values(mean_col, ascending=asc).reset_index(drop=True)

        best_row = lb.sort_values(mean_col, ascending=asc, kind="mergesort").iloc[0]
        rule = "argmax"
        chosen = best_row
        if cfg.selection.one_se_rule:
            threshold = (
                best_row[mean_col] - best_row[se_col] if not asc else best_row[mean_col] + best_row[se_col]
            )
            within = lb[lb[mean_col] >= threshold] if not asc else lb[lb[mean_col] <= threshold]
            if len(within):
                chosen = within.sort_values(
                    ["k", mean_col], ascending=[True, asc], kind="mergesort"
                ).iloc[0]
                rule = "one_standard_error"

        selected = {
            "rule": rule,
            "model": chosen["model"],
            "family": chosen["family"],
            "tag": chosen["tag"],
            "k": int(chosen["k"]),
            "features": chosen["features"].split(","),
            f"cv_{primary}_mean": float(chosen[mean_col]),
            f"cv_{primary}_se": float(chosen[se_col]),
            "unconstrained_best": {
                "model": best_row["model"],
                "k": int(best_row["k"]),
                f"cv_{primary}_mean": float(best_row[mean_col]),
            },
            "cost_of_parsimony": round(float(best_row[mean_col] - chosen[mean_col]), 6),
            "ordering_strategy": cfg.selection.ordering_strategy,
        }

        self.report.add(
            "selection",
            rule=rule,
            top_n=cfg.selection.top_n,
            top_n_distinct_models=cfg.selection.top_n_distinct_models,
            selected_model=selected["model"],
            selected_k=selected["k"],
            selected_cv_score=selected[f"cv_{primary}_mean"],
            unconstrained_best_score=float(best_row[mean_col]),
            cost_of_parsimony=selected["cost_of_parsimony"],
            variables_saved=int(best_row["k"]) - selected["k"],
            top_variants=[f"{s['model']}@k={s['k']} ({s[f'cv_{primary}_mean']:.4f})" for s in top_specs],
            best_per_model={
                r["model"]: round(float(r[mean_col]), 6) for _, r in best_per_model.iterrows()
            },
        )
        return top_specs, best_per_model, selected

    # ------------------------------------------------------------------
    # step 6: optional hyper-parameter tuning
    # ------------------------------------------------------------------
    def _tune(self, X_tr, y_tr, zoo, orderings, leaderboard, fold_scores):
        cfg = self.config
        primary = cfg.metrics.primary
        scorers = make_scorers(cfg.metrics)
        mean_col = f"cv_{primary}_mean"
        asc = not greater_is_better(primary)

        pool = leaderboard.sort_values(mean_col, ascending=asc, kind="mergesort")
        targets = pool if cfg.tuning.apply_to == "all" else pool.head(cfg.selection.top_n)
        inner = self._cv(n_splits=cfg.tuning.cv_splits, seed_offset=1)
        outer = self._cv()
        new_rows, tuned_summary = [], []

        for _, row in targets.iterrows():
            model_name, k = row["model"], int(row["k"])
            space = cfg.tuning.search_spaces.get(model_name)
            if not space:
                continue
            entry = zoo[model_name]
            features = orderings[model_name]["ordering"][:k]
            base = build_model_pipeline(entry["config"], features, entry["estimator"])
            search_cls = RandomizedSearchCV if cfg.tuning.strategy == "random" else GridSearchCV
            kwargs = dict(estimator=base, cv=inner, scoring=scorers[primary],
                          n_jobs=cfg.run.n_jobs, refit=True, error_score=np.nan)
            if cfg.tuning.strategy == "random":
                search = search_cls(param_distributions=space, n_iter=cfg.tuning.n_iter,
                                    random_state=cfg.run.random_state, **kwargs)
            else:
                search = search_cls(param_grid=space, **kwargs)
            search.fit(X_tr, y_tr, groups=self.groups_tr_)

            tuned_est = search.best_estimator_.named_steps["model"]
            tuned_name = f"{model_name}__tuned"
            # scored through the same fold loop as the grid, so tuned rows are
            # computed identically to their baselines and their fold-level
            # predictions land in the store like any other cv cell
            cvres = self._outer_cv_scores(
                lambda: build_model_pipeline(entry["config"], features, tuned_est),
                X_tr, y_tr, outer, tuned_name, k,
            )
            spec_copy = ModelSpec(**{**entry["spec"].__dict__, "tag": f"{entry['spec'].tag}+tuned"})
            new_rows.append(self._row_from_cv(tuned_name, spec_copy, k, features, cvres, primary))
            fold_scores[(tuned_name, k)] = np.asarray(
                [orient(primary, v) for v in cvres[f"test_{primary}"]], dtype=float
            )
            zoo[tuned_name] = {"spec": spec_copy, "estimator": tuned_est, "config": entry["config"]}
            orderings[tuned_name] = orderings[model_name]
            tuned_summary.append({
                "model": model_name, "k": k,
                "baseline": round(float(row[mean_col]), 6),
                "tuned": round(float(new_rows[-1][mean_col]), 6),
                "delta": round(float(new_rows[-1][mean_col] - row[mean_col]), 6),
                "best_params": {kk: str(vv) for kk, vv in search.best_params_.items()},
            })

        if new_rows:
            leaderboard = pd.concat([leaderboard, pd.DataFrame(new_rows)], ignore_index=True)
            leaderboard = leaderboard.drop(columns=["rank"], errors="ignore")
            leaderboard = leaderboard.sort_values(mean_col, ascending=asc,
                                                  kind="mergesort").reset_index(drop=True)
            leaderboard.insert(0, "rank", np.arange(1, len(leaderboard) + 1))

        self.report.add(
            "hyperparameter_tuning",
            enabled=True,
            strategy=cfg.tuning.strategy,
            n_iter=cfg.tuning.n_iter,
            n_specs_tuned=len(tuned_summary),
            mean_improvement=round(float(np.mean([t["delta"] for t in tuned_summary])), 6)
            if tuned_summary else None,
            detail=tuned_summary,
        )
        return leaderboard, fold_scores, zoo

    def _outer_cv_scores(self, make_pipe, X_tr, y_tr, cv, model_name, k) -> Dict[str, np.ndarray]:
        """``cross_validate`` substitute for a fixed specification.

        Runs the same fold iteration and the same scoring path as the grid
        (:meth:`_score_cell`), returning arrays shaped like ``cross_validate``'s
        output so :meth:`_row_from_cv` consumes either interchangeably.
        """
        cell: Dict[str, List[float]] = {"fit_time": []}
        for f, (tr_idx, va_idx) in enumerate(cv.split(X_tr, y_tr, self.groups_tr_)):
            repeat, fold = divmod(f, self.config.split.cv.n_splits)
            self._score_cell(cell, make_pipe(),
                             X_tr.iloc[tr_idx], y_tr[tr_idx],
                             X_tr.iloc[va_idx], y_tr[va_idx],
                             self.row_ids_tr_[tr_idx], self.row_ids_tr_[va_idx],
                             model_name, k, fold, repeat)
        return {key: np.asarray(vals, dtype=float) for key, vals in cell.items()}

    # ------------------------------------------------------------------
    # step 7: holdout confirmation + production refit
    # ------------------------------------------------------------------
    def _confirm(self, X_tr, y_tr, X_ho, y_ho, X_all, y_all, zoo, selected):
        cfg = self.config
        entry = zoo[selected["model"]]
        features = selected["features"]

        pipe = build_model_pipeline(entry["config"], features, entry["estimator"])
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_ho)[:, 1]
        holdout = evaluate_predictions(y_ho, proba, cfg.metrics)
        deciles = decile_table(y_ho, proba)
        # the operating thresholds implied by the champion's holdout scores;
        # decision_threshold ships in the model bundle so ProductionScorer
        # applies a stable absolute cut instead of a batch-relative quantile.
        # (Computed on the train-only fit's holdout scores: the full-data refit
        # below saw the holdout labels, so its own holdout scores are tainted.)
        holdout.update(implied_thresholds(y_ho, proba, cfg.metrics))
        self.predictions_.add("holdout", selected["model"], selected["k"], None, None,
                              self.row_ids_ho_, y_ho, proba)

        primary = cfg.metrics.primary
        self.report.add(
            "holdout_confirmation",
            model=selected["model"],
            k=selected["k"],
            n_holdout=len(X_ho),
            cv_estimate=selected[f"cv_{primary}_mean"],
            holdout_score=holdout.get(primary),
            cv_minus_holdout=round(float(selected[f"cv_{primary}_mean"] - (holdout.get(primary) or 0)), 6),
            within_one_se=bool(
                abs(selected[f"cv_{primary}_mean"] - (holdout.get(primary) or 0))
                <= max(selected[f"cv_{primary}_se"], 1e-12)
            ),
            top_decile_lift=float(deciles.iloc[0]["lift"]) if len(deciles) else None,
            top_decile_capture=float(deciles.iloc[0]["cumulative_capture"]) if len(deciles) else None,
            calibration_ratio=holdout.get("calibration_ratio"),
            calibration_error=holdout.get("calibration_error"),
            **{f"holdout_{k}": v for k, v in holdout.items() if k != primary},
        )

        prod_cfg = entry["config"].copy()
        prod_cfg.preprocessing.inference_guard.warn = cfg.preprocessing.inference_guard.warn
        if cfg.run.refit_on_full_data:
            final = build_model_pipeline(prod_cfg, features, entry["estimator"])
            final.fit(X_all, y_all)
            refit_info: Dict[str, Any] = dict(
                refit_on="train+holdout",
                n_rows=len(X_all),
                n_variables=len(features),
                n_design_columns=len(final.named_steps["features"].feature_names_out_),
            )
        else:
            final = pipe
            refit_info = dict(refit_on="train_only", n_rows=len(X_tr))

        # Everything that ships beside the model must describe the model that
        # ships. The holdout numbers above are the *validated* evidence and stay
        # as measured on the train-only fit; the monitoring reference and the
        # capacity cut are re-derived from the production pipeline's own scores
        # on the same holdout rows, because with refit_on_full_data that is a
        # different fit -- and on a discrete score scale the two can disagree
        # badly (PSI > 3 between them has been observed on a two-variable model).
        holdout.update(self._shipped_reference(final, X_ho, proba, holdout))
        self.report.add(
            "production_refit",
            **refit_info,
            reference_score_source=holdout["reference_score_source"],
            decision_threshold_source=holdout["decision_threshold_source"],
            decision_threshold=holdout["decision_threshold"],
            decision_threshold_train_only=holdout["decision_threshold_train_only"],
            decision_threshold_shift=(
                round(float(holdout["decision_threshold"] - holdout["decision_threshold_train_only"]), 6)
                if holdout["decision_threshold"] is not None
                and holdout["decision_threshold_train_only"] is not None else None
            ),
            decision_flag_rate_shipped=holdout["decision_flag_rate_shipped"],
        )

        feat: DisputeFeaturePipeline = final.named_steps["features"]
        return final, holdout, deciles, self._holdout_slices(X_ho, y_ho, proba), feat.fit_report_

    def _shipped_reference(self, final, X_ho, proba_train_only, holdout) -> Dict[str, Any]:
        """Monitoring reference and decision threshold for the pipeline that ships.

        ``reference_score_quantiles`` is always the shipped pipeline's score
        distribution on the holdout rows. The ``top_pct`` threshold is a
        capacity cut -- a quantile of scores -- so it is re-derived from the same
        scores. The ``fpr`` threshold depends on labels, and after a refit that
        saw the holdout labels the holdout is no longer out-of-sample for it, so
        it is left as derived on the train-only fit and labelled as such.
        """
        cfg = self.config
        refit = self.config.run.refit_on_full_data
        proba_ship = np.asarray(final.predict_proba(X_ho))[:, 1] if refit else proba_train_only
        source = "production_refit" if refit else "train_only_fit"

        out: Dict[str, Any] = {
            "reference_score_quantiles": reference_quantiles(proba_ship),
            "reference_score_source": source,
            "decision_threshold_train_only": holdout.get("decision_threshold"),
            "decision_threshold_source": "train_only_fit",
        }
        policy = holdout.get("decision_threshold_policy")
        if refit and policy == "top_pct":
            out["decision_threshold"] = float(np.quantile(proba_ship, 1 - operating_point(cfg.metrics, "lift_top_pct")))
            out["decision_threshold_source"] = source
        thr = out.get("decision_threshold", holdout.get("decision_threshold"))
        out["decision_flag_rate_shipped"] = (
            round(float((proba_ship >= thr).mean()), 6) if thr is not None and len(proba_ship) else None
        )
        return out

    def _holdout_slices(self, X_ho, y_ho, proba) -> Optional[pd.DataFrame]:
        """Holdout performance and flag rate per segment level.

        A strong global AP can hide a 3x flag-rate disparity inside one claim
        channel or customer segment; for a model whose output drives denial of
        a consumer's Reg E dispute claim, this per-slice table is the first
        artefact a model-risk or fair-treatment reviewer asks for. Slice
        columns need only exist in the holdout frame -- they do not have to be
        model inputs.
        """
        m = self.config.metrics
        cols = [c for c in m.slice_columns if c in X_ho.columns]
        if not cols:
            return None
        cut = float(np.quantile(proba, 1 - operating_point(m, "lift_top_pct")))
        rows = [
            {"slice_column": col, "level": level, "n": int(len(idx)),
             "prevalence": scores.get("prevalence"),
             m.primary: scores.get(m.primary),
             "roc_auc": scores.get("roc_auc"),
             "mean_score": scores.get("mean_predicted"),
             "flag_rate_at_top_pct": round(float((proba[idx] >= cut).mean()), 6)}
            for col in cols
            for level, idx in X_ho.groupby(X_ho[col].astype(str)).indices.items()
            if len(idx) >= m.min_slice_n
            for scores in [evaluate_predictions(y_ho[idx], proba[idx], m)]
        ]
        slices = pd.DataFrame(rows)
        if len(slices):
            by = slices.groupby("slice_column")["flag_rate_at_top_pct"]
            self.report.add(
                "holdout_slices",
                slice_columns=cols,
                n_slices=len(slices),
                min_slice_n=m.min_slice_n,
                n_levels_below_min_n={
                    c: int(X_ho[c].astype(str).nunique() - (slices["slice_column"] == c).sum())
                    for c in cols
                },
                # worst across slice columns of (max / min) level flag rate:
                # 1.0 = perfectly even, 3.0 = one level flagged 3x another
                max_flag_rate_disparity=round(float((by.max() / by.min().clip(lower=1e-9)).max()), 3),
            )
        return slices

    # ------------------------------------------------------------------
    # step 8: artifacts
    # ------------------------------------------------------------------
    def _write_artifacts(self, result: SelectionResult) -> None:
        cfg = self.config
        out = Path(cfg.run.output_dir) / cfg.run.name
        out.mkdir(parents=True, exist_ok=True)

        tables = {
            "leaderboard.csv": result.leaderboard,
            "marginal_gains.csv": result.marginal_gains,
            "best_per_model.csv": result.best_per_model,
            "holdout_deciles.csv": result.holdout_deciles,
            "holdout_slices.csv": result.holdout_slices,
        }
        for name, frame in tables.items():
            if frame is not None and len(frame):
                frame.to_csv(out / name, index=False)

        blobs = {
            "orderings.json": result.orderings,
            "top_specs.json": result.top_specs,
            "selected_spec.json": result.selected,
            "holdout_metrics.json": result.holdout_metrics,
            "feature_pipeline_report.json": result.feature_report,
        }
        for name, payload in blobs.items():
            # holdout_metrics.json carries the thresholds and the reference
            # quantiles the bundle applies; those must round-trip exactly
            ndigits = None if name == "holdout_metrics.json" else 6
            (out / name).write_text(json.dumps(json_safe(payload, ndigits=ndigits), indent=2))

        lineage = self.report.get("lineage") or {}
        written = write_prediction_artifacts(out, self.predictions_, meta={
            "run": cfg.run.name,
            "created_by": "dmf.research.selection.ModelSelectionHarness",
            "config_sha256": lineage.get("config_sha256"),
            "data_sha256": lineage.get("data_sha256"),
            "row_id_source": self._row_id_source,
            "positive_label": str(self.positive_label_),
            "score_semantics": "y_score is predict_proba[:, 1] for the positive label; "
                               "y_true is 1 where the raw target equals positive_label",
            "selected_model": result.selected_model,
            "selected_k": result.selected.get("k"),
            "cv": {"strategy": cfg.split.strategy, "n_splits": cfg.split.cv.n_splits,
                   "n_repeats": cfg.split.cv.n_repeats},
            "metrics": cfg.to_dict()["metrics"],
            "decision_threshold": result.holdout_metrics.get("decision_threshold"),
            "decision_threshold_policy": result.holdout_metrics.get("decision_threshold_policy"),
            "decision_threshold_source": result.holdout_metrics.get("decision_threshold_source"),
            "reference_score_source": result.holdout_metrics.get("reference_score_source"),
            "implied_threshold_top_pct": result.holdout_metrics.get("implied_threshold_top_pct"),
            "implied_threshold_at_fpr": result.holdout_metrics.get("implied_threshold_at_fpr"),
        })
        if written:
            self.report.add(
                "prediction_store",
                level=cfg.run.save_predictions,
                rows_per_stage=self.predictions_.stage_counts(),
                files=sorted(written.values()),
            )
        self.report.to_json(str(out / "run_report.json"))

        # a config that reproduces exactly the winning specification
        final_cfg = cfg.copy()
        final_cfg.run.name = f"{cfg.run.name}__final"
        final_cfg.models = {result.selected_model.replace("__tuned", ""):
                            cfg.models.get(result.selected_model.replace("__tuned", ""), ModelSpec())}
        roles = result.fitted_model.named_steps["features"].roles_
        final_cfg.columns.numeric = list(roles.numeric)
        final_cfg.columns.categorical = list(roles.categorical)
        final_cfg.columns.passthrough = list(roles.passthrough)
        final_cfg.columns.auto_infer = False        # the variable list is now explicit
        k = int(result.selected["k"])
        final_cfg.selection.k_min = final_cfg.selection.k_max = k
        final_cfg.selection.top_n = 1
        final_cfg.tuning.enabled = False
        final_cfg.to_yaml(out / "final_spec.yaml")

        if cfg.run.save_fitted_model and result.fitted_model is not None:
            try:
                import joblib

                from .. import __version__

                joblib.dump(
                    {"pipeline": result.fitted_model,
                     "features": result.selected_features,
                     "model": result.selected_model,
                     "config": cfg.to_dict(),
                     "dmf_version": __version__,
                     # derived from the champion's holdout score distribution
                     # per metrics.decision_threshold_policy; ProductionScorer
                     # picks it up on load. None only when the policy is 'none'
                     # (or the holdout was degenerate) -- the scorer then falls
                     # back to a batch-relative cut.
                     "decision_threshold": result.holdout_metrics.get("decision_threshold"),
                     "decision_threshold_policy": result.holdout_metrics.get(
                         "decision_threshold_policy"),
                     # which fit the threshold and the monitoring reference
                     # describe: the shipped refit where possible, otherwise
                     # the train-only fit (see _shipped_reference)
                     "decision_threshold_source": result.holdout_metrics.get(
                         "decision_threshold_source"),
                     "reference_score_source": result.holdout_metrics.get(
                         "reference_score_source"),
                     # reference distribution for PSI monitoring, so drift can be
                     # measured in production without reloading the training table
                     "reference_score_quantiles": result.holdout_metrics.get(
                         "reference_score_quantiles"),
                     "lineage": (self.report.get("lineage") or {})},
                    out / "model.joblib",
                )
            except Exception as exc:  # pragma: no cover
                print(f"[artifacts] could not persist model: {exc}")

        self.report.add("artifacts", output_dir=str(out),
                        files=sorted(p.name for p in out.iterdir()))


# --------------------------------------------------------------------------
# statistics helpers
# --------------------------------------------------------------------------
def _safe_proba(pipe, X) -> Optional[np.ndarray]:
    """Positive-class probabilities, or None when prediction itself fails --
    the caller records NaN scores and stores no rows for that fold."""
    try:
        return np.asarray(pipe.predict_proba(X))[:, 1]
    except Exception:
        return None


def _binarize_target(y: pd.Series, positive_label: Any) -> Tuple[np.ndarray, Any]:
    """Map an arbitrary two-class target onto {0, 1}.

    Silently producing an all-zero target is the worst possible failure here --
    the run continues and every metric is meaningless -- so a ``positive_label``
    that does not occur in the data is an error that names the labels it did
    find. ``positive_label: auto`` picks the minority class, which is the
    convention that is right for fraud targets essentially always.
    """
    observed = pd.unique(y.dropna())
    if len(observed) < 2:
        raise ValueError(f"Target has a single class: {list(observed)}. Nothing to model.")
    if len(observed) > 2:
        raise ValueError(
            f"Target has {len(observed)} distinct values {sorted(map(str, observed))[:8]}; "
            f"this framework models a binary outcome. Collapse it upstream."
        )

    if str(positive_label).lower() == "auto":
        counts = y.value_counts()
        positive_label = counts.idxmin()
    elif not (y == positive_label).any():
        raise ValueError(
            f"data.positive_label={positive_label!r} does not occur in column '{y.name}'. "
            f"Observed labels: {sorted(map(str, observed))}. "
            f"Set data.positive_label to one of them, or to 'auto' for the minority class."
        )
    return (y == positive_label).astype(int).to_numpy(), positive_label


def _nb_factor(n: int, n_splits: int) -> float:
    """Nadeau-Bengio SE multiplier for the std of n CV fold scores.

    ``sqrt(1/n + r/(1-r))`` with ``r = 1/n_splits``: the extra ``r/(1-r)`` term
    accounts for folds sharing training data. Used by both the leaderboard SE
    (hence the 1-SE selection band) and the paired marginal-gain test, so the
    two always apply the same variance model.
    """
    r = 1.0 / n_splits
    return float(np.sqrt(1.0 / n + r / (1.0 - r)))


def _paired_test(d: np.ndarray, test: str, n_splits: int) -> Dict[str, Any]:
    """Paired significance test on fold-level score differences.

    ``paired_t`` uses the Nadeau-Bengio correction, which inflates the variance
    by ``1/n + n_test/n_train`` to account for the fact that CV folds share
    training data and are therefore *not* independent. The naive paired t-test
    on CV folds is badly anti-conservative; this one is merely optimistic.
    """
    d = np.asarray(d, dtype=float)
    n_finite = int(np.isfinite(d).sum())
    d = d[np.isfinite(d)]
    if n_finite == 0:
        # every fold errored; np.allclose([], 0) is True, which would otherwise
        # report a total failure as a genuine zero gain
        return {"test": test, "statistic": None, "p_value": None, "verdict": "all_folds_failed"}
    if len(d) < 2 or np.allclose(d, 0):
        return {"test": test, "statistic": None, "p_value": None,
                "verdict": "no_change" if np.allclose(d, 0) else "insufficient_folds"}

    if test == "none":
        return {"test": "none", "statistic": None, "p_value": None,
                "verdict": "improves" if d.mean() > 0 else "degrades"}

    if test == "wilcoxon" and 2 ** len(d) * 0.05 < 2:
        return {"test": "wilcoxon", "statistic": None, "p_value": None,
                "verdict": "underpowered",
                "note": f"the smallest attainable two-sided p at {len(d)} folds is "
                        f"{2 / 2 ** len(d):.4f}; use paired_t or raise n_splits"}

    if test == "paired_t":
        n = len(d)
        denom = d.std(ddof=1) * _nb_factor(n, n_splits)
        t = float(d.mean() / denom) if denom > 0 else 0.0
        p = float(2 * (1 - stats.t.cdf(abs(t), df=n - 1)))
        stat, pval = round(t, 4), round(p, 6)
    else:
        try:
            res = stats.wilcoxon(d, zero_method="zsplit", alternative="two-sided")
            stat, pval = round(float(res.statistic), 4), round(float(res.pvalue), 6)
        except ValueError:
            return {"test": "wilcoxon", "statistic": None, "p_value": None, "verdict": "no_change"}

    if pval is not None and pval < 0.05:
        verdict = "improves" if d.mean() > 0 else "degrades"
    else:
        verdict = "not_significant"
    return {"test": test, "statistic": stat, "p_value": pval, "verdict": verdict}


def _added_variable(leaderboard: pd.DataFrame, model: str, prev_k: int, k: int) -> Optional[str]:
    try:
        prev = set(leaderboard.loc[
            (leaderboard["model"] == model) & (leaderboard["k"] == prev_k), "features"].iloc[0].split(","))
        cur = leaderboard.loc[
            (leaderboard["model"] == model) & (leaderboard["k"] == k), "features"].iloc[0].split(",")
        new = [c for c in cur if c not in prev]
        return ",".join(new) if new else None
    except (IndexError, KeyError):
        return None


def _first_non_improving(gains: pd.DataFrame, model: str) -> Optional[int]:
    sub = gains[gains["model"] == model].sort_values("to_k")
    for _, r in sub.iterrows():
        if r["verdict"] != "improves":
            return int(r["to_k"])
    return None


def _ordering_agreement(orderings: Dict[str, List[str]]) -> Dict[str, Any]:
    """How much the per-model variable orderings agree with each other."""
    names = list(orderings)
    if len(names) < 2:
        return {"mean_spearman": None, "top1_agreement": 1.0, "top5_jaccard": 1.0}
    base = orderings[names[0]]
    rank = {m: {v: i for i, v in enumerate(o)} for m, o in orderings.items()}
    pairs = list(itertools.combinations(names, 2))
    rhos = [float(stats.spearmanr([rank[a][v] for v in base],
                                  [rank[b][v] for v in base]).statistic) for a, b in pairs]
    jac = [len(set(orderings[a][:5]) & set(orderings[b][:5]))
           / max(len(set(orderings[a][:5]) | set(orderings[b][:5])), 1) for a, b in pairs]
    return {
        "mean_spearman": round(float(np.nanmean(rhos)), 4) if rhos else None,
        "top1_agreement": round(1.0 / len({o[0] for o in orderings.values()}), 4),
        "top5_jaccard": round(float(np.mean(jac)), 4) if jac else None,
    }


__all__ = ["ModelSelectionHarness", "SelectionResult"]
