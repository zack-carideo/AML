# dmf — variable selection and model specification framework

An sklearn framework for building a binary classifier on tabular data — developed here for
debit-card dispute fraud — and for carrying the selected specification into production
unchanged. Everything is driven by one YAML file; nothing about the dispute dataset is
hard-coded.

The package is split in two, physically:

- **`dmf` — the production core** (~3,100 lines, 7 modules). The only code the scoring
  path executes and the only code a production team maintains.
- **`dmf.research` — the experiment half** (~2,600 lines, 8 modules). Grid search,
  variable orderings, estimator zoo, post-run evaluation and CLI.

The dependency points one way: research imports the core, the core never imports research.
A change on the research side cannot alter what production executes, and the two halves can
be reviewed, versioned and released to different standards. Importing a moved name from
`dmf` raises an `ImportError` naming its new home.

---

## Install and run

```bash
pip install -e ".[boosting,dev]"

python examples/generate_synthetic_disputes.py       # writes data/disputes.csv
dmf train --config configs/dispute_fraud.yaml        # or: python -m dmf.research.cli train ...
python examples/run_demo.py                          # narrated end-to-end walkthrough
python examples/edge_case_audit.py                   # ten production failure modes
pytest -q                                            # 140 tests
```

`examples/research_walkthrough.ipynb` is the same material as a model-documentation spec
with executed outputs.

Comparing several configurations over the same data:

```bash
dmf sweep --configs a.yaml b.yaml c.yaml --output-dir artifacts
```

Scoring new records with a persisted model:

```bash
dmf score \
  --model artifacts/dispute_fraud_v1/model.joblib \
  --data data/disputes_next_month.csv \
  --out scored.csv --id-column dispute_id --guard-report guard.json
```

Every CLI subcommand has a functional equivalent in `dmf.research` that returns objects
rather than printed text; the CLI is a thin printing shell over these, so the two cannot
drift:

```python
train(config, X=None, **cli_flags_as_kwargs)   # -> SelectionResult
score(model, data, out=None, ...)              # -> (scored DataFrame, report dict)
run_sweep(configs)                             # -> (comparison DataFrame, {name: SelectionResult})
```

`run_sweep` writes one `sweep_comparison.csv` ranked on **holdout** performance, never on
leaderboards — each run's leaderboard already picked its own winner, so ranking configs by
leaderboard re-introduces selection bias one level up. Runs are only like-for-like if they
share seed, split design and data; the sweep checks exactly that and stamps every row
`comparable` true/false with a warning. Lineage hashes make it auditable: same
`data_sha256`, different `config_sha256`. Across large sweeps the holdout erodes as a
referee — keep a second untouched partition for the final call.

---

## The core pipeline

```
DisputeFeaturePipeline
  └── sklearn.Pipeline
        ├── select    FrameSelector      locks the variable list into the artifact
        ├── guard     InferenceGuard     bounds inputs to the training support
        └── column    ColumnTransformer
              ├── num  coerce → winsorize → impute(+indicator) → scale → variance
              ├── cat  impute → collapse-rare → encode(onehot|ordinal|woe|target)
              └── pass passthrough
```

Every learned statistic — imputation medians, winsorising bounds, category vocabularies,
WOE tables, scaler moments — lives in the fitted sub-estimators. Nesting the object in
`cross_validate` therefore re-estimates all of it inside every fold, including the two
*supervised* encoders. A test pins this: with a shuffled target and high-cardinality
merchant ids, cross-validated AUC stays at 0.5.

```python
from dmf import Config, DisputeFeaturePipeline

cfg = Config.from_yaml("configs/dispute_fraud.yaml")
fp = DisputeFeaturePipeline(config=cfg, features=["prior_disputes_12m", "channel"])
Xt = fp.fit_transform(train_df, y)

fp.fit_report_            # per-step quantitative summary
fp.feature_source_map_    # encoded column -> source variable
fp.training_envelope()    # numeric support and category vocabulary
fp.information_value()    # IV per categorical, when the WOE encoder is in use
```

### Categorical encoders

Config-selectable, and overridable per model via `preprocessing_overrides`:

| encoder | what it does | when |
|---|---|---|
| `onehot` | rare-collapsed dummies | linear champion, full transparency |
| `ordinal` | integer codes | trees — splits on codes fine, keeps the matrix narrow |
| `woe` | weight of evidence + Information Value | scorecard convention; monotone, one column per variable |
| `target` | sklearn `TargetEncoder`, internally cross-fitted | high-cardinality merchant/MCC/device |

WOE uses the convention **positive WOE = elevated fraud rate**, the reverse of the classic
good/bad credit sign. IV is convention-invariant and reported with Siddiqi bands when
`categorical.woe.report_iv` is set.

### Column typing and data hygiene

`columns.auto_infer` types each column and applies quality gates. Gates apply only to
self-chosen columns — a column you name explicitly is never dropped from under you.

| behaviour | control |
|---|---|
| all-null column dropped, reason `all_missing` | — |
| constant column dropped, reason `constant` | `columns.drop_constant` |
| one-level-per-row column dropped, reason `identifier_like_cardinality` | `columns.max_categorical_cardinality_ratio` |
| `"$1,234.50"` and similar parsed as numeric rather than becoming a 1500-level categorical | `columns.numeric_parse_threshold` (share of values that must parse) |
| ISO strings and `datetime64` parsed to epoch days; unit-agnostic and idempotent | — |
| `inf` / `-inf` become missing before any statistic is computed, so winsorising is not a no-op | — |
| target labelled `FRAUD`/`GENUINE` resolved to the minority class | `data.positive_label: auto` |
| duplicate row rate above 1% flagged `duplicate_rows_material` | — |
| `thin_positive_folds` flagged when expected positives per training fold fall below 10; fewer than `2 · n_splits` positives overall is a hard error | — |

Recovered and dropped columns are named in the run report, never dropped silently.

---

## The selection harness

```python
from dmf import Config
from dmf.research import ModelSelectionHarness

result = ModelSelectionHarness(Config.from_yaml("configs/dispute_fraud.yaml")).run()

result.leaderboard        # every (model, k) cell, all metrics, mean/std/SE, overfit gap
result.marginal_gains     # paired fold-level delta for each k-1 -> k step
result.best_per_model     # best variant per architecture
result.top_specs          # the top N specifications
result.selected           # the final choice and why
result.holdout_metrics    # confirmation on data selection never touched
result.fitted_model       # ready to pickle and ship
```

**1 — Variable ordering.** Two strategies, config-selectable, estimated inside each CV fold:

- `importance` — one fit per model, then model-native importance aggregated back to source
  variables (`selection.importance.aggregate`: sum | max | mean). With
  `scale_by_std`, a linear coefficient is scaled by the SD of its encoded column
  (`|β| · sd(x)`); without it, summing `|β|` across the levels of a high-cardinality
  categorical mechanically outranks a strong single numeric. Cheap; ignores redundancy.
- `rfe` — recursive elimination on the encoded matrix; the elimination order is the
  ranking. `O(p/step)` fits. Accounts for redundancy as the set shrinks. Falls back to
  `importance` with a recorded reason if the estimator exposes no importance signal.

With `ordering_reference_model: per_model` each architecture gets its own ordering, and the
report includes the Spearman agreement between them.

**2 — The grid.** Every (model × k) cell scored by cross-validation with the feature
pipeline refit inside each fold. Metrics in natural units (loss metrics un-negated), plus
mean, SD, standard error and the train−OOS overfit gap.

Variables are re-ranked inside every fold, so no validation row helped choose the features
it is scored against. Ranking once on the whole training partition and then
cross-validating on that same partition is feature selection outside the CV loop: on 1,200
rows of pure noise with a random target (true AP 0.50) that construction reports 0.618
against the nested construction's 0.502, and its overfit-gap diagnostic reads ≈0.000
because train and validation are contaminated identically. Only the nested construction
exists; a test pins its behaviour.

Because each fold picks its own subset, how often a variable survives into the top-k across
folds measures selection stability directly, reported as `selection_stability`.

**3 — Marginal value of the k-th variable.** Per-fold paired deltas between the k and k−1
specifications. The default `paired_t` applies the **Nadeau–Bengio** variance correction:
CV folds share training data and are not independent, so the naive paired t-test is badly
anti-conservative. `wilcoxon` is available as a distribution-free alternative, but its
smallest attainable two-sided p is `2 / 2**n_folds` — at 5 folds nothing can reach 0.05, so
the verdict is reported as `underpowered` rather than a misleading `not_significant`.
Deltas are oriented to the primary metric, so for a loss-type primary a *decrease* is the
gain. These are decision aids, not publishable p-values: the leaderboard's cells are all
scored on the same folds, and no multiplicity correction is applied.

**4 — Selection.** Top N by mean OOS primary metric (optionally one entry per architecture
via `top_n_distinct_models`), then a **one-standard-error rule** picks the champion: among
all specifications within one SE of the best, take the fewest variables. The SE is
Nadeau–Bengio corrected via the same `_nb_factor` the paired test uses, so the band and the
test cannot disagree. The report states the cost of that parsimony in metric units and how
many variables it saved.

Note that the rule treats variable count as the only complexity measure, so a heavily
overfit model with fewer variables can win over a simpler architecture with more. Read
`overfit_gap` on the selected cell before shipping.

**5 — Confirmation.** One evaluation on the untouched stratified holdout, with calibration
diagnostics, a decile gains table, and — when `metrics.slice_columns` is set — a
per-segment breakdown (`holdout_slices.csv`): performance, prevalence and flag rate per
level, with a `max_flag_rate_disparity` headline. Slice columns need only exist in the
data, not be model inputs, so parity can be checked on attributes the model is deliberately
not allowed to use. Levels thinner than `metrics.min_slice_n` (default 50) are skipped
rather than reported on noise. Then a refit on train+holdout for the shipped artifact, when
`run.refit_on_full_data` is set.

---

## Metrics

Primary is **average precision (PR-AUC)**: under 7% prevalence ROC-AUC is dominated by the
true-negative mass and barely separates materially different models, while AP tracks
precision across the recall range, which is what an investigations queue experiences.

The registry holds exactly eight metrics:

| metric | direction | operating point |
|---|---|---|
| `average_precision` | higher better | — |
| `roc_auc` | higher better | — |
| `ks_statistic` | higher better | — |
| `recall_at_fpr` | higher better | `metrics.recall_at_fpr` |
| `lift_at_top_pct` | higher better | `metrics.lift_top_pct` |
| `brier_score` | lower better | — |
| `log_loss` | lower better | — |
| `calibration_error` | lower better | — |

Unknown metric names raise at config load, not an hour into a sweep.

### Operating points

`recall_at_fpr` and `lift_at_top_pct` are curves read at a chosen budget, not single
numbers. `metrics.recall_at_fpr` and `metrics.lift_top_pct` therefore accept either a
scalar or a list:

```yaml
metrics:
  recall_at_fpr: [0.005, 0.01, 0.02]   # three reported columns
  lift_top_pct: 0.05                   # one reported column
```

A list of N values produces N reported metrics named `recall_at_fpr@0.005`,
`recall_at_fpr@0.01`, … A scalar keeps the plain un-suffixed name, so existing configs,
artifacts and column names are unchanged. A metric name may also carry an explicit point
(`recall_at_fpr@0.005` in `metrics.secondary`), which overrides the config list.

`resolve_metrics()` is the single source of truth for which metrics exist and what they are
called; the CV scorers, fold score vector, holdout report and leaderboard columns are all
built from it, so they cannot disagree about the set or the order.

Where only one operating point is possible — threshold derivation, the slice-table cut —
the **first** configured value is used. Order the list by intent: lead with the budget you
will operate at, put sensitivities behind it. A `primary` that expands to several operating
points is a config error, since there would be no single column to rank on.

### Degenerate targets

`average_precision` and `roc_auc` return NaN, reported as `null`, when the target has only
one class. sklearn disagrees with itself here — `roc_auc_score` raises while
`average_precision_score` warns and returns 0.0, which lands in a slice table as if a model
had scored terribly on a segment where nothing was measurable.

---

## Splitting

`split.strategy` decides how honest the estimate is:

- `random` — i.i.d. rows.
- `group` — `GroupShuffleSplit` + `StratifiedGroupKFold` on `group_column`. **Use this the
  moment one customer can raise more than one dispute.** Otherwise a cardholder with six
  disputes lands on both sides of the split and the holdout is optimistic. The report
  asserts zero group leakage.
- `time` — the holdout is the most recent slice; folds run forward only. The honest test
  for a model that will score next month, and the only one that exposes concept drift.
- `group_time` — both, requiring `group_column` *and* `time_column`. Each group's
  **earliest** timestamp decides its side: the most recent `holdout_size` share of groups
  becomes the holdout; any customer with earlier activity stays whole in training, so no
  customer straddles the boundary. CV within training is `StratifiedGroupKFold`. Because
  spanners go to training, the holdout skews toward short-tenure customers; the split
  report surfaces `median_rows_per_group_train` against
  `median_rows_per_group_holdout` alongside the zero-leakage assertion and the train and
  holdout time periods.

Group and time keys are automatically excluded from the candidate variables.

## Hyper-parameter tuning

Off by default: selection is about *which variables and which architecture*; tuning is a
later, separate question. When enabled it searches on an inner CV split, and the tuned
specification is re-scored on the same outer CV as everything else so the leaderboard stays
comparable. The inner CV uses its own fold count and a shifted seed under every split
strategy; the one undetectable case (`time` with equal inner and outer fold counts, where
deterministic splits coincide) is refused at config validation.

---

## The prediction store

`run.save_predictions` writes a long-format `predictions.parquet` under the run directory
holding `(row_id, y_true, y_score)` plus provenance (`stage`, `model`, `k`, `fold`,
`repeat`), alongside `fold_assignments.parquet` and a `predictions_meta.json` sidecar
recording what the numbers mean — resolved positive label, score semantics, metric
operating points, derived decision threshold. Without a parquet engine installed both
tables fall back to CSV; the loaders read either.

Levels are cumulative:

| level | stores |
|---|---|
| `none` | nothing |
| `holdout` *(default)* | the champion's holdout predictions |
| `cv` | + every validation-fold prediction of every model × k cell |
| `all` | + training-side fold predictions, for row-level overfit work |

Row count scales with models × k values × folds, so `cv` and `all` grow fast.
`data.id_column` names the record identifier and is never used as a feature; unset, the
DataFrame index is recorded.

Raw scores are stored rather than predicted labels: a label is `score >= threshold` for
some threshold, and selection has none — every metric either integrates over all thresholds
or fixes an operating point and lets the threshold fall out. Stored scores keep every
operating point available afterwards.

From the store, `dmf.research` recomputes anything post-run with no model and no raw data:

```python
from dmf.research import (load_predictions, load_fold_assignments,
                          compute_metrics, threshold_at_fpr, operating_point_table)

preds, meta = load_predictions("artifacts/my_run")
compute_metrics(preds, meta, by=["stage", "model", "k"])          # reproduces the leaderboard
compute_metrics(preds, meta, by=["model", "k", "fold"])           # per-fold breakdown
threshold_at_fpr(preds, max_fpr=0.01, by=["model", "k", "fold"])  # implied cut + stability
operating_point_table(preds[preds.stage == "holdout"])            # precision/recall/flag rate per cut
```

`compute_metrics` runs through the same registry as the harness, so recomputed numbers
reproduce the leaderboard and holdout report exactly, and any metric added to `dmf.metrics`
later works on old runs. The same rows feed calibration curves, slice analyses on any column
(join on `row_id`), and PSI between folds.

### Decision threshold and the shipped reference

The champion's decision threshold is derived from its holdout score distribution per
`metrics.decision_threshold_policy` — `top_pct` (the cut flagging the top `lift_top_pct` of
volume; default), `fpr` (the cut achieving `recall_at_fpr` false-positive rate), or `none`
— and ships in `model.joblib` as `decision_threshold`, which `ProductionScorer.from_joblib`
picks up as a stable absolute cut.

The value is stored at full precision. The scorer applies `score >= threshold`, and on a
discrete score scale rounding it up by one unit in the sixth decimal drops every row tied at
the quantile.

Under `run.refit_on_full_data` the pipeline that ships is not the fit the holdout was scored
on, and on a discrete score scale the two can disagree badly. What ships beside the model is
therefore re-derived from the production pipeline's own scores on the holdout rows:
`reference_score_quantiles` always, and the `top_pct` threshold (a quantile of scores) too.
An `fpr` threshold depends on labels the refit has now seen, so it keeps the train-only
value. `decision_threshold_source` and `reference_score_source` — in the bundle, the meta
sidecar and `holdout_metrics.json` — say which fit each number describes, and
`decision_threshold_train_only` keeps the validated value for traceability.

The holdout metrics themselves stay as measured on the train-only fit. The shipped refit has
no out-of-sample evidence of its own; if you need that, keep a second partition that
threshold derivation never touches.

---

## Production inference

The risk is not that scoring crashes — it is that it succeeds on input the model has no
business scoring. `InferenceGuard` sits before all encoding and applies an explicit,
configurable policy to every way production data can differ from training data:

| situation | default policy | flagged as |
|---|---|---|
| numeric outside the training range | clip to the training envelope, never extrapolate | `n_out_of_range` |
| category level never seen in training | rewrite to `__UNSEEN__` → rare bucket / −1 / WOE 0 / prior mean | `n_unseen_category` |
| text in a numeric column, `inf`, `1e308` | parse leniently, else missing → fitted imputer | `n_coerced` |
| a required column absent | materialise as missing → fitted imputer | batch-level escalation |

Alternative policies per case: `nan` (route through the imputer), `passthrough`, or `error`.
Every intervention is counted per row and per batch.

The report is **returned**, never scraped off the estimator afterwards. Reading mutable
state after a call is wrong the moment two requests share a loaded model, and wrong the
moment `CalibratedClassifierCV` runs the transform on internal clones. The guard keeps no
post-transform state.

```python
from dmf import ProductionScorer

scorer = ProductionScorer.from_joblib("artifacts/dispute_fraud_v1/model.joblib")
scored, report = scorer.score(new_disputes)

report["verdict"]                    # 'ok' | 'review_recommended'
report["escalation_reason"]          # why, when it is not ok
scorer.explain_guard(new_disputes)   # per-variable breakdown of interventions
```

`scored` carries `fraud_probability`, `score_rank`, the per-row guard counts, `data_quality`
(`ok` / `guarded`), `decision` (`flag_fraud` / `pass`), and `action` (`auto_action` /
`manual_review`).

A row whose score depended on guarded input is routed to `manual_review`, never
`auto_action` — a dispute is not auto-declined on the strength of an extrapolation the model
never learned. Refusing to score is its own operational failure, so the row still gets a
probability; it just does not get trusted. A batch-level fault, such as a column the feed
stopped sending, marks every row, because the fault applies to every score equally.

**Thresholds.** An absolute `threshold` is stable: the same dispute gets the same decision
whichever file it arrives in. `top_pct` is a quantile of *this batch* — right for a queue
sized to analyst capacity, meaningless for a handful of records — so batches under 50 rows
and `score_one` require an explicit threshold rather than silently degrading. Persist the
tuned value as `decision_threshold` in the bundle and `from_joblib` picks it up.

**Schema.** `transform` refuses array input and a frame with none of the training column
names. Without that check a renamed frame returns one constant probability for every record
with no exception anywhere.

**Version skew.** The bundle records `dmf_version`; loading it under a different version
warns, because custom transformers are pickled by reference and a renamed class is a silent
wrong answer. Bump `dmf.__version__` when transformer classes change, or the check cannot
fire.

---

## Post-hoc steps, deliberately outside the framework

Each is one call, and each has a right configuration that is a business decision rather than
a code constant.

**Probability calibration.** With `imbalance: balanced` or `scale_pos_weight` the model's
output is a ranking score, not a calibrated probability — the holdout report says so
directly via `calibration_ratio`, `calibration_error` and the decile table. Anything that
multiplies score by exposure needs calibration first. `CalibratedClassifierCV` (isotonic,
fit on out-of-fold scores) wraps `result.fitted_model` directly. Calibrate first, then tune
the threshold.

**Threshold tuning.** The bundle's capacity- or FPR-based `decision_threshold` is the
starting point. A cost-based refinement stays post-hoc: `result.fitted_model` is a plain
sklearn `Pipeline`, so `TunedThresholdClassifierCV` wraps it. Score it with a *hard-label*
objective — a cost function encoding what a missed fraudulent dispute costs against an
analyst review plus a false accusation — not with `average_precision`, which is
threshold-free and gives a flat surface. Carry the resulting float into
`ProductionScorer(threshold=...)`, which overrides the bundled cut.

`ProductionScorer` reaches the feature pipeline through either wrapper, so the guard and the
training envelope keep working when the model is nested.

**Monitoring.** `population_stability_index(reference, current)` with `psi_band()` (< 0.10
stable, 0.10–0.25 watch, > 0.25 investigate). The guard catches values outside the support;
PSI catches a shift within it. The bundle ships `reference_score_quantiles` — the shipped
pipeline's holdout score distribution at 101 levels, taken as observed scores with no
interpolation — and `psi_from_reference_quantiles(quantiles, scores)` computes PSI from
those alone, so it runs in production without the training table or the holdout.

Interpolated quantiles are deliberately avoided: on a discrete score scale an interpolated
bin edge sits at a value no row ever takes, so the bin between two adjacent observed scores
carries expected mass it can never hold, and an unchanged population reads as drift.

**Non-goals.** Sample weights (class imbalance is handled by `class_weight` /
`scale_pos_weight`; per-record economic weighting belongs at threshold selection). SHAP and
reason codes (`feature_source_map_` and per-variable importances are the right inputs for an
explanation layer built elsewhere). Scheduled retraining (the lineage block, PSI helper and
guard reports are the triggers a scheduler consumes; the scheduler is infrastructure).

---

## Layout

```
src/dmf/                      PRODUCTION CORE — what the scoring path executes
  config.py                   typed YAML configuration; unknown keys raise
  transformers.py             column typing, lenient parsing, all transformers
                              (NumericCoercer, Winsorizer, RareCollapser, WOE,
                               FrameSelector, InferenceGuard)
  pipeline.py                 DisputeFeaturePipeline, build_model_pipeline
  inference.py                ProductionScorer
  metrics.py                  metric registry, operating points, holdout evaluation,
                              gains table, PSI
  reporting.py                StepReport, frame/target profiling, run lineage

src/dmf/research/             EXPERIMENT SIDE — imports the core, never the reverse
  selection.py                ModelSelectionHarness, paired tests, artifacts
  ordering.py                 importance / rfe, fold-nested
  evaluate.py                 prediction store, post-run metric recomputation
  api.py                      functional CLI equivalents returning objects
  sweep.py                    multi-config runs, comparability check
  zoo.py                      YAML -> estimator, imbalance policy
  cli.py                      dmf train / dmf score / dmf sweep

configs/                      dispute_fraud.yaml, dispute_fraud_v2.yaml
examples/                     data generator, demo, edge-case audit, research notebook
tests/                        140 tests
```

`Config` lives in the core even though it carries the experiment sections (`selection`,
`tuning`): one YAML drives both halves, and a shipped model stores the exact config that
produced it. The core never reads those sections.

## Artifacts written per run

`leaderboard.csv`, `marginal_gains.csv`, `best_per_model.csv`, `holdout_deciles.csv`,
`holdout_slices.csv`, `orderings.json`, `top_specs.json`, `selected_spec.json`,
`holdout_metrics.json`, `feature_pipeline_report.json`, `run_report.json`, `final_spec.yaml`
(a config reproducing the winning specification exactly), `model.joblib`, and — per
`run.save_predictions` — `predictions.parquet`, `fold_assignments.parquet` and
`predictions_meta.json`.

`holdout_metrics.json` and `predictions_meta.json` are written unrounded; everything else is
rounded to six decimals for readability. Anything applied as a cut must round-trip exactly.

Every run records `run_lineage` — package versions, config SHA-256, data fingerprint — into
`run_report.json` and into the `model.joblib` bundle, so a deployed model ties back to the
exact code, configuration and data that produced it.
