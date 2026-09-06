# dmf — variable selection and model specification framework

A two-package sklearn framework for building a debit-card dispute fraud model, and for
carrying the selected specification into production unchanged.

The split is physical, not just conceptual: **`dmf` is the production core** — the only
code the scoring path executes and the only code a production team maintains (~1,700
lines of code across 7 modules). **`dmf.research` is the experiment half** — the grid
search, orderings, estimator zoo and CLI — which imports the core and is never imported
by it. The dependency points one way, so a change to the research half cannot alter what
production executes, and the two can be reviewed, versioned and released to different
standards.

**Core (deployed).** `DisputeFeaturePipeline` is a fitted, picklable sklearn transformer
that turns a raw dispute record into a design matrix. The same fitted instance serves
training and inference, which is what makes train/serve skew structurally impossible
rather than merely discouraged. It is guarded at the edges against inputs outside the
support it was trained on.

**Research (experiment-time, `dmf.research`).** `ModelSelectionHarness` searches over estimator
architectures and nested variable subsets, quantifies the marginal value of the k-th
variable with paired fold-level tests, and hands back a winning specification — a
variable list plus a model — which the core then materialises for production. Nothing
in this layer is deployed.

Everything is driven by one YAML file. Nothing about the dispute dataset is hard-coded;
point the config at another table and the same harness runs unchanged.

---

## Install and run

```bash
pip install -e ".[boosting,dev]"

python examples/generate_synthetic_disputes.py       # writes data/disputes.csv
python -m dmf.research.cli train --config configs/dispute_fraud.yaml
python examples/run_demo.py                          # full narrated walkthrough
pytest -q                                            # 86 tests
python examples/edge_case_audit.py                   # 10 production failure modes
```

Comparing several configurations over the same data:

```bash
python -m dmf.research.cli sweep --configs a.yaml b.yaml c.yaml --output-dir artifacts
```

Every CLI subcommand has a **functional equivalent in `dmf.research` that returns
objects, never printed text** — the CLI is a thin printing shell over these, so the two
cannot drift: `train(config, X=None, **cli_flags_as_kwargs)` → `SelectionResult`;
`score(model, data, out=None, ...)` → `(scored DataFrame, report dict)`;
`run_sweep(configs)` → `(comparison DataFrame, {name: SelectionResult})`.

`run_sweep` (also importable from `dmf.research`) runs each config through the harness
and writes one `sweep_comparison.csv` ranked on **holdout** performance — never on
leaderboards, since each run's leaderboard already picked its own winner and choosing a
config by leaderboard re-introduces selection bias one level up. Runs are only
like-for-like if they share the seed, split design, and data; the sweep checks exactly
that and stamps every row `comparable` true/false (with a warning) rather than assuming.
Lineage hashes make it auditable: same `data_sha256`, different `config_sha256`. For
large sweeps, remember the holdout erodes as a referee — keep a second untouched
partition for the final call.

Scoring new records with a persisted model:

```bash
python -m dmf.research.cli score \
  --model artifacts/dispute_fraud_v1/model.joblib \
  --data data/disputes_next_month.csv \
  --out scored.csv --id-column dispute_id --guard-report guard.json
```

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
*supervised* encoders. A test asserts this: with a shuffled target and high-cardinality
merchant ids, cross-validated AUC must stay at 0.5.

```python
from dmf import Config, DisputeFeaturePipeline

cfg = Config.from_yaml("configs/dispute_fraud.yaml")
fp = DisputeFeaturePipeline(config=cfg, features=["prior_disputes_12m", "channel"])
Xt = fp.fit_transform(train_df, y)

fp.fit_report_            # per-step quantitative summary
fp.feature_source_map_    # encoded column -> source variable
fp.training_envelope()    # the numeric support and category vocabulary
fp.information_value()    # IV per categorical, when the WOE encoder is in use
```

### Categorical encoders

All four are config-selectable, per model:

| encoder | what it does | when |
|---|---|---|
| `onehot` | rare-collapsed dummies | linear champion, full transparency |
| `ordinal` | integer codes | trees — splits on codes fine, keeps the matrix narrow |
| `woe` | weight of evidence + Information Value | credit/fraud scorecard convention; monotone, one column per variable |
| `target` | sklearn `TargetEncoder`, internally cross-fitted | high-cardinality merchant/MCC/device |

WOE uses the convention **positive WOE = elevated fraud rate** (the reverse of the
classic good/bad credit sign), which reads more naturally for this target. IV is
convention-invariant and reported with Siddiqi bands.

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

**1 — Variable ordering.** Two strategies, config-selectable, estimated inside each CV
fold (see step 2):

- `importance` — one fit per model, then model-native importance aggregated back to
  source variables. For linear models the coefficient is scaled by the SD of its encoded
  column (`|β| · sd(x)`), without which summing `|β|` across the levels of a
  high-cardinality categorical mechanically outranks a strong single numeric. Cheap;
  ignores redundancy.
- `rfe` — recursive elimination on the encoded matrix; the elimination order is the
  ranking. `O(p/step)` fits. Accounts for redundancy as the set shrinks. Falls back to
  `importance` with a recorded reason if the estimator exposes no importance signal.

A CV-scored greedy-forward strategy existed and was removed: under fold-nested ordering
its cost is `O(K·p·folds)` CV runs — effectively unrunnable on a real column set — and an
unexercised code path is pure carrying cost.

When `ordering_reference_model: per_model`, each architecture gets its own ordering and
the report includes the Spearman agreement between them.

**2 — The grid.** Every (model × k) cell scored by cross-validation, with the feature
pipeline refit inside each fold. Metrics in natural units (loss metrics un-negated),
mean, SD, standard error, and the train−OOS overfit gap.

**The variables are always re-ranked inside every fold**, so no validation row ever
helped choose the features it is scored against. Ranking once on the whole training
partition and then cross-validating on that same partition is feature selection outside
the CV loop: on 1,200 rows of pure noise with a random target (true AP = 0.50) that
construction reported 0.618 while the nested one reports 0.502 — and its own overfit-gap
diagnostic read ≈0.000, because train and validation were contaminated identically. The
flat construction was **removed rather than left behind a flag**: a config switch that
silently inflates the leaderboard is a liability, and a test pins the honest behaviour of
the only remaining path.

A useful side effect: because each fold picks its own subset, how often a variable
survives into the top-k across folds is a direct measure of selection stability, and is
reported as `selection_stability`.

**3 — Marginal value of the k-th variable.** Per-fold paired deltas between the k and
k−1 specifications. The default `paired_t` applies the **Nadeau–Bengio** variance
correction — CV folds share training data and are not independent, so the naive paired
t-test is badly anti-conservative; the corrected one is merely optimistic. `wilcoxon` is
available as a distribution-free alternative, but its smallest attainable two-sided p is
`2 / 2**n_folds` — at 5 folds nothing can ever reach 0.05, so the verdict is reported as
`underpowered` rather than as a misleading `not_significant`. Deltas are oriented to the
primary metric, so for a loss-type primary a *decrease* is the gain. Either way these are
decision aids, not publishable p-values.

**4 — Selection.** Top N by mean OOS primary metric (optionally one entry per
architecture via `top_n_distinct_models`), then a **one-standard-error rule** picks the
final champion: among all specifications within one SE of the best, take the fewest
variables. The SE is Nadeau–Bengio corrected (folds share training data, so the naive
`std/√n` is ~33% too narrow at 5 folds), and it is computed by the same `_nb_factor` the
paired test uses, so the band and the test can never disagree. The report states the cost of that parsimony in metric units and how many
variables it saved.

**5 — Confirmation.** One evaluation on the untouched stratified holdout, with
calibration diagnostics, a decile gains table, and — when `metrics.slice_columns` is set
— a per-segment breakdown (`holdout_slices.csv`): performance, prevalence and flag rate
per level of claim channel, reason code, segment, and any other column named. Then a
refit on train+holdout for the shipped artifact.

### Metrics

Primary is **average precision (PR-AUC)**: under 7% prevalence, ROC-AUC is dominated by
the true-negative mass and barely separates materially different models, while AP tracks
precision across the recall range — which is what an investigations queue experiences.
Also available: `roc_auc`, `gini`, `ks_statistic`, `recall_at_fpr` (detection at a fixed
false-positive budget), `lift_at_top_pct` and `precision_at_top_pct` (review-capacity
operating points), `brier_score`, `log_loss`, `calibration_error`.

### Splitting

`split.strategy` decides how honest the estimate is:

- `random` — i.i.d. rows.
- `group` — `GroupShuffleSplit` + `StratifiedGroupKFold` on `group_column`. **Use this
  the moment one customer can raise more than one dispute.** Without it a cardholder
  with six disputes lands on both sides of the split and the holdout is optimistic. The
  report asserts zero group leakage.
- `time` — the holdout is the most recent slice; folds run forward only. The honest test
  for a model that will score next month, and the only one that exposes concept drift.
- `group_time` — both at once (requires `group_column` *and* `time_column`). Each group's
  **earliest** timestamp decides its side: the most recent `holdout_size` share of groups
  — customers whose activity lies entirely in the newest window — becomes the holdout,
  and any customer with earlier transactions stays whole in training. No customer ever
  straddles the boundary. CV within training is `StratifiedGroupKFold`, so every fold
  keeps groups intact too. Because spanners go to training, the holdout skews toward
  short-tenure customers; the split report surfaces `median_rows_per_group_train` vs
  `median_rows_per_group_holdout` so that skew is visible, alongside the zero-leakage
  assertion and the train/holdout time periods.

Group and time keys are automatically excluded from the candidate variables.

### Hyper-parameter tuning

Off by default and deliberately so: selection is about *which variables and which
architecture*; tuning is a later, separate question. When enabled it searches on an inner
CV split and the tuned specification is then re-scored on the same outer CV as everything
else, so the leaderboard stays comparable.

### The prediction store

Every score the harness computes is also *storable*: `run.save_predictions` writes a
long-format `predictions.parquet` under the run directory holding
`(row_id, y_true, y_score)` plus provenance (`stage`, `model`, `k`, `fold`, `repeat`),
alongside `fold_assignments.parquet` (which rows formed each CV validation fold) and a
`predictions_meta.json` sidecar recording what the numbers mean — the resolved positive
label, the score semantics, the metric operating points, and the derived decision
threshold. Levels are cumulative: `none` | `holdout` (default) | `cv` (every validation
fold of every model × k cell) | `all` (adds the training-side fold predictions for
row-level overfit work). `data.id_column` names the record identifier; unset, the
DataFrame index is used.

Raw scores are stored instead of predicted labels on purpose: a label is
`score >= threshold` for some threshold, and selection has none — every metric either
integrates over all thresholds or fixes an operating point (an FPR, a review budget) and
lets the threshold fall out. Stored scores keep every operating point available forever.

From the store, `dmf.research` recomputes anything post-run, with no model and no raw
data:

```python
from dmf.research import (load_predictions, compute_metrics,
                          threshold_at_fpr, operating_point_table)

preds, meta = load_predictions("artifacts/my_run")
compute_metrics(preds, meta, by=["stage", "model", "k"])        # reproduces the leaderboard
compute_metrics(preds, meta, by=["model", "k", "fold"])         # per-fold breakdown
threshold_at_fpr(preds, max_fpr=0.01, by=["model", "k", "fold"])  # implied cut + stability
operating_point_table(preds[preds.stage == "holdout"])          # precision/recall/flag-rate per cut
```

`compute_metrics` runs through the same registry as the harness, so recomputed numbers
reproduce the leaderboard and holdout report exactly — and any metric added to
`dmf.metrics` later works on old runs. The same rows also feed calibration curves, slice
analyses on any column (join on `row_id`), and PSI between folds.

The champion's **decision threshold** is derived automatically from its holdout score
distribution per `metrics.decision_threshold_policy` — `top_pct` (the cut flagging the
top `lift_top_pct` of volume; default), `fpr` (the cut achieving `recall_at_fpr`
false-positive rate), or `none` — and ships in `model.joblib` as `decision_threshold`,
which `ProductionScorer.from_joblib` picks up as a stable absolute cut. The value is stored
at full precision: the scorer applies `score >= threshold`, and on a discrete score scale
rounding it up by one unit in the sixth decimal would drop every row tied at the quantile.

With `run.refit_on_full_data` the pipeline that ships is not the fit the holdout was
scored on, so what ships beside it is re-derived from the production pipeline's own scores
on the holdout rows: `reference_score_quantiles` always, and the `top_pct` threshold (a
quantile of scores) too. An `fpr` threshold depends on labels the refit has now seen, so it
keeps the train-only value. `decision_threshold_source` and `reference_score_source` in the
bundle, the meta sidecar and `holdout_metrics.json` say which fit each number describes,
and `decision_threshold_train_only` keeps the validated value for traceability.

---

## Production inference

The risk is not that scoring crashes — it is that it succeeds on input the model has no
business scoring. `InferenceGuard` sits before all encoding and applies an explicit,
configurable policy to every way production data can differ from training data:

| situation | default policy | flagged as |
|---|---|---|
| numeric outside the training range | clip to the training envelope (never extrapolate) | `n_out_of_range` |
| category level never seen in training | rewrite to `__UNSEEN__` → rare bucket / −1 / WOE 0 / prior mean | `n_unseen_category` |
| text in a numeric column, `inf`, `1e308` | parse leniently, else missing → fitted imputer | `n_coerced` |
| a required column absent | materialise as missing → fitted imputer | batch-level escalation |

Alternative policies per case: `nan` (route through the imputer), `passthrough`, or
`error` (raise). Nothing is silent — every intervention is counted per row and per batch.

The report is **returned**, never scraped off the estimator afterwards. That matters:
reading mutable state after a call is wrong the moment two requests share a loaded model,
and wrong the moment `CalibratedClassifierCV` runs the transform on internal clones. The
guard no longer keeps any post-transform state at all — the surface that caused that bug
class does not exist to misuse.

```python
from dmf import ProductionScorer

scorer = ProductionScorer.from_joblib("artifacts/dispute_fraud_v1/model.joblib")
scored, report = scorer.score(new_disputes)

report["verdict"]            # 'ok' | 'review_recommended'
report["escalation_reason"]  # why, when it is not ok
scorer.explain_guard(new_disputes)   # per-variable breakdown of interventions
```

`scored` carries `fraud_probability`, `score_rank`, the per-row guard counts,
`data_quality` (`ok` / `guarded`), `decision`, and `action`. **A row whose score depended
on guarded input is routed to `manual_review`, never `auto_action`** — a dispute is not
auto-declined on the strength of an extrapolation the model never learned. Refusing to
score is its own operational failure, so it still gets a probability; it just does not
get trusted. A batch-level fault (a column the feed stopped sending) marks *every* row,
because the fault applies to every score equally.

**Thresholds.** An absolute `threshold` is stable — the same dispute gets the same
decision whichever file it arrives in. `top_pct` is a quantile of *this batch*: right for
a queue sized to analyst capacity, meaningless for a handful of records, so batches under
50 rows and `score_one` require an explicit threshold rather than silently degrading.
Persist the tuned value as `decision_threshold` in the bundle and `from_joblib` picks it
up.

**Schema.** `transform` refuses array input and a frame with none of the training column
names. Without that check a renamed frame returns one constant probability for every
record with no exception anywhere.

---

## Post-hoc steps, deliberately not in the framework

Two things belong after model selection and are one call each. They are left out because
their right configuration is a business decision that should not be frozen in code.

**Threshold tuning.** The bundle already ships a capacity- or FPR-based
`decision_threshold` derived from the holdout (see *The prediction store*), which is the
right starting point. A *cost-based* refinement stays post-hoc: `result.fitted_model` is
a plain sklearn `Pipeline`, so `TunedThresholdClassifierCV` wraps it directly. Score it
with a *hard-label* objective — a cost function encoding what a missed fraudulent dispute
costs versus an analyst review plus a false accusation — not with `average_precision`,
which is threshold-free and would give a flat surface. Carry the resulting float into
`ProductionScorer(threshold=...)`, which overrides the bundled cut.

**Probability calibration.** `CalibratedClassifierCV` wraps the same pipeline. The
holdout report already tells you whether you need it (`calibration_ratio`,
`calibration_error`, and the decile table). Calibrate first, then tune the threshold.

`ProductionScorer` reaches the feature pipeline through either wrapper, so the guard and
the training envelope keep working when the model is nested.

**Monitoring.** `population_stability_index(reference, current)` with `psi_band()`
(< 0.10 stable, 0.10–0.25 watch, > 0.25 investigate) completes the picture: the guard
catches values outside the support, PSI catches a shift *within* it. The bundle ships
`reference_score_quantiles` — the shipped pipeline's score distribution on the holdout
rows — and `psi_from_reference_quantiles(bundle["reference_score_quantiles"], scores)`
computes PSI from those quantiles alone, so it runs in production without the training
table or the holdout.

**Version skew.** The bundle records `dmf_version`; loading it under a different version
warns, because custom transformers are pickled by reference and a renamed class is a
silent wrong answer.

---

## Hardening: what the edge-case audit found

`examples/edge_case_audit.py` builds ten datasets, each carrying one production failure
mode, and runs each end to end (ordering → grid → selection → holdout → refit → reload →
score). Two crashed and several were silently corrupted. All ten now pass.

| # | failure mode | what happened | fix |
|---|---|---|---|
| 1 | column 100% null | consumed a candidate slot | dropped at role inference, reported |
| 2 | constant numeric and categorical | dead columns reached the design matrix | dropped as `constant` |
| 3 | 0.5% prevalence, 20 positives | ran, but fold instability was invisible | `thin_positive_folds` flag; hard error below 2·n_splits positives |
| 4 | identifier with one level per row | one-hot explosion, collapsed to noise | dropped as `identifier_like_cardinality` |
| 5 | collinear pair + 20% duplicate rows | duplicates spanned the split | `duplicate_rows_material` flag; `split.strategy: group` added |
| 6 | `"$1,234.50"` amounts as text | **became a 1500-level categorical — a strong predictor silently destroyed** | lenient numeric parsing at the 95% parse-rate threshold; recovery reported |
| 7 | ISO-string and `datetime64` dates | modelled as high-cardinality categories | parsed to epoch days; unit-agnostic, idempotent |
| 8 | target labelled `FRAUD`/`GENUINE` | **crashed, or worse would have produced an all-zero target** | `positive_label: auto` (minority class) and an error naming the observed labels |
| 9 | `inf` / `-inf` from a divide-by-zero ratio | **crashed in the scaler** — the 99.5th percentile was itself `inf`, so winsorising was a no-op | non-finite values become missing before any statistic is computed |
| 10 | 140 rows, 55 variables, 14 positives | ran, selected k=1 correctly | no change needed |

Three findings did not come from a single case: guard warnings fired on every CV fold
(expected inside a validation fold, so suppressed during the sweep and restored for the
production artifact); run provenance was unrecorded (`run_lineage` now captures library
versions plus config and data hashes); and `ProductionScorer` could not see through a
post-hoc wrapper.

### Second pass: an adversarial code review

A separate review of the whole package, probing each claim rather than reading it, found
more — including four defects in the deployed core. Every one has a named regression test
in `tests/test_review_fixes.py`.

| severity | defect | measured impact | fix |
|---|---|---|---|
| critical | variable ordering ran **outside** the CV loop | +0.12 AP on pure noise (0.618 reported, 0.502 true), and `overfit_gap` read ≈0.000 because both ends were contaminated | fold-nested ordering, now the only construction (the flat path was deleted) |
| critical | `fit_transform` overridden to `fit().transform()`, discarding the cross-fitted matrix | target encoding trained on a leaked column; encoding/target correlation 0.459 instead of 0.017 | let `TransformerMixin` return the fit-time matrix |
| critical | guard report read off estimator state after `predict_proba` | under `CalibratedClassifierCV`, 1,331 of 1,500 fabricated records auto-actioned; under two threads, 7 of 40 garbage batches reported clean | report is returned from `transform_with_quality`, never scraped |
| critical | `sample_frac` paired a random row sample with the first *n* labels | perfectly separable data collapsed from AP 1.00 to 0.46, silently | sample positions, not index labels |
| major | a missing column left every row `data_quality='ok'` | 1,425 records scored on imputed medians were auto-actioned | batch verdict forces `manual_review` on every row |
| major | no schema validation in `transform` | ndarray or renamed frame → one constant probability, no exception | refuse array input and a frame with no matching columns |
| major | marginal-gain verdicts inverted for loss-type primaries | a genuine log-loss improvement reported as `degrades` | orient deltas by `greater_is_better` |
| major | `score_one` always returned `flag_fraud` | a batch-relative cut on one record is that record's own score | small batches require an absolute threshold |
| major | `wilcoxon` default could never reach p < 0.05 at 5 folds | `n_significant_improvements` structurally always 0 | default `paired_t`; report `underpowered` |
| minor | duplicate column names, `decile_table` on n < bins, all-failed folds reading as a tie, `final_spec.yaml` not pinning the spec | crashes and mislabels | named errors, band clamping, `all_folds_failed`, pinned spec |

Two gaps in production support came out of the same pass and are now closed: the bundle
ships `reference_score_quantiles` so PSI is computable without the training table, and
`dmf_version` so a pickle loaded against changed transformer classes warns.

All three previously-open gaps are now closed, at a net −1 lines of code (each addition
paid for by removing repetition elsewhere): the leaderboard SE — and therefore the 1-SE
parsimony band — uses the same Nadeau–Bengio corrected variance as the paired test, via
one shared `_nb_factor`; the holdout report breaks down by `metrics.slice_columns`
(performance, prevalence and flag rate per level, written to `holdout_slices.csv` with a
`max_flag_rate_disparity` headline — slice columns need only exist in the data, not be
model inputs, so parity can be checked on attributes the model is deliberately not
allowed to use); and the tuning loop's inner CV now uses its own fold count and a shifted
seed under every split strategy, with the one undetectable case (`time` with equal fold
counts, where deterministic splits would coincide) refused at config validation.

### Deliberate non-goals

- **Sample weights.** Class imbalance is handled by `class_weight` / `scale_pos_weight`.
  Per-record economic weighting is a cost-model decision, better applied at threshold
  selection.
- **SHAP / reason codes.** The framework exposes `feature_source_map_` and per-variable
  importances, which are the right inputs for an explanation layer — but the analyst-
  facing narrative belongs to the GenAI phase, not here.
- **Automated retraining.** The lineage block, PSI helper and guard reports are the
  triggers a scheduler would consume; the scheduler itself is infrastructure.

---

## Layout

```
src/dmf/                      PRODUCTION CORE — what the scoring path executes
  config.py                   typed YAML configuration; unknown keys raise
  transformers.py             column typing + lenient parsing + all transformers
                              (NumericCoercer, Winsorizer, RareCollapser, WOE,
                               FrameSelector, InferenceGuard)
  pipeline.py                 DisputeFeaturePipeline, build_model_pipeline
  inference.py                ProductionScorer
  metrics.py                  metric registry, holdout evaluation, gains table, PSI
  reporting.py                StepReport, frame/target profiling, run lineage

src/dmf/research/             EXPERIMENT SIDE — imports the core, never the reverse
  selection.py                ModelSelectionHarness, paired tests, artifacts
  ordering.py                 importance / rfe (fold-nested)
  api.py                      functional CLI equivalents (train/score) returning objects
  sweep.py                    multi-config runs, comparability check, holdout comparison
  zoo.py                      YAML -> estimator, imbalance policy
  cli.py                      `dmf train` / `dmf score` / `dmf sweep`

configs/dispute_fraud.yaml
examples/                     data generator, narrated demo, edge-case audit
tests/                        86 tests
```

`Config` lives in the core even though it carries the experiment sections
(`selection`, `tuning`): one YAML drives both halves, and a shipped model stores the
exact config that produced it. The core never *reads* those sections. Importing a moved
name from `dmf` raises an `ImportError` that names its new home in `dmf.research`.

## Artifacts written per run

`leaderboard.csv`, `marginal_gains.csv`, `best_per_model.csv`, `holdout_deciles.csv`,
`orderings.json`, `top_specs.json`, `selected_spec.json`, `holdout_metrics.json`,
`feature_pipeline_report.json`, `run_report.json`, `final_spec.yaml` (a config that
reproduces exactly the winning specification), `model.joblib`, and — per
`run.save_predictions` — the prediction store: `predictions.parquet`,
`fold_assignments.parquet`, `predictions_meta.json`.

---

## Provenance

Built in the Claude (claude.ai) project **"fraud disputes"** (owner: zjc1002@gmail.com),
August 2026. The project's doc `claude/modelling-framework.md` holds the living summary —
decisions taken, validation performed, known gaps, and the pointer to the next phase (the
GenAI analyst-explanation layer). To continue work with full context, open a session
inside that project; a new session there reads that doc and can pick up where this left
off.

The chain of custody at runtime is separate and automatic: every training run records
`run_lineage` (package versions, config SHA-256, data fingerprint) into `run_report.json`
and into the `model.joblib` bundle, so any deployed model ties back to the exact code,
configuration, and data that produced it — and this README ties the code back to the
project. dmf version at delivery: 0.2.0.
