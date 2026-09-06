---
name: mrm-review
description: Predictive-model MRM review. Audits code that predicts a future event from historical data (credit risk, fraud, forecasting, GenAI scoring) for statistical soundness and produces a validation-ready executive briefing for Model Risk Management at a top-10 US bank. Use when asked for an MRM review, model validation, SR 11-7 / SR 26-2 readiness, or a pre-submission audit of a modelling pipeline.
---

# Predictive Model MRM Review

Review code that produces a prediction of a future event from historical data and produce a validation-ready executive briefing for Model Risk Management (MRM) at a top-10 US bank.

The center of gravity is the **quantitative and statistical soundness of the code as written**. Architecture identification and governance tiering are scoping steps that tell you which statistical tests apply; they are not the deliverable. MRM validators fail submissions on estimation errors, leakage, and invalid test design far more often than on documentation gaps, so audit the math in the code path that actually executes.

Governing principle: **verify every quantitative claim against the executing code.** A metric printed in a notebook, a docstring describing a sampling scheme, or a README claiming out-of-time validation are assertions to be tested, not evidence. Trace the actual call path, and where the data is available, re-derive the numbers yourself.

## Step 1 — Scope the pipeline

Read entry points, configs, training and inference scripts, environment files, and any shipped artifacts (model bundles, run reports, leaderboards). Build an inventory of every prediction-generating component: architecture family (classical statistical — GLM, regression, ARIMA, survival; traditional ML — GBM/XGBoost, random forest, SVM; deep learning; GenAI/LLM — prompted, fine-tuned, RAG, agentic; hybrid/ensemble); target definition and prediction horizon; input features and sources; the business decision the output drives; autonomous vs. human-in-the-loop.

Do not stop at the headline model. Statistical assumptions hide in components nobody calls a model: imputation and outlier treatment, binning and WOE transforms, scaling fit on the wrong sample, target/dummy encoding, post-model calibration layers, score overrides and threshold rules, and any GenAI step producing features or labels consumed downstream. Each goes in the inventory.

Also record: static vs. periodic retrain vs. online learning; third-party or vendor components (pretrained models, foundation-model APIs, vendor scorecards); the data vintages available for testing; and whether the shipped artifact was produced by the *current* code (compare artifact lineage, library versions, and estimator params against what the code builds today).

## Step 2 — Audit statistical soundness (the core)

For each item: state what the code does, whether it is correct, and the consequence if not. Cite file and line. Where you can execute the code or inspect the data, run the check rather than reasoning abstractly.

**Temporal validity and leakage.** Highest-yield category. Confirm every feature is computable using only information available at the prediction timestamp. Look for: aggregations over the full panel rather than expanding/rolling windows; joins to reference tables carrying current-state values instead of as-of values; target-derived features; label windows overlapping feature windows; `groupby().transform()` or `shift()` with the wrong sign or missing sort; scalers, imputers, encoders, or feature selectors fit on the full dataset before splitting; SMOTE/resampling applied before rather than inside CV folds. Verify fit-on-train/transform-on-test discipline (e.g. a scikit-learn `Pipeline` inside the CV loop).

**Sample design and representativeness.** Survivorship bias, through-the-door selection (reject inference or explicit acknowledgment), undocumented exclusions, sample/class weights dropped at scoring time, unresolved censoring. Confirm segment sizes support the estimates and that effective sample size accounts for clustering (multiple observations per customer are not independent).

**Target definition.** Consistent across the development window and matching the business event; outcome window; handling of indeterminates; mid-sample definition drift.

**Validation design.** Resampling must match the data-generating process: time-ordered data needs time-series CV or a strict out-of-time holdout; random k-fold on a temporal panel is a defect. Tuning, feature selection, and threshold selection happen inside the resampling loop or on a separate tuning set; the reported test metric comes from data never touched during selection. Confirm an out-of-time sample exists and is genuinely later. Note repeated consultation of the same test set. Check whether the reported CV number describes the *specification that ships* or only the selection *procedure* (nested re-ranking can report a number the fixed variable list never achieves); and whether the holdout is consumed (threshold derivation, refit on all data) before the shipped model exists.

**Estimation correctness.** Regression/GLM: link and family, multicollinearity (VIF) and sign stability, robust/clustered SEs, influential observations. Time series: stationarity, residual autocorrelation, unit roots, seasonality. ML: regularization applied and tuned, early stopping on a validation set distinct from the test set, class-imbalance handling that distorts probabilities without recalibration. Check coefficient signs and magnitudes against economic intuition. Inspect the selection rule itself: a parsimony rule that counts only variables can pick a heavily overfit tree over a simpler linear model within the same error band.

**Metric computation and calibration.** Recompute headline metrics from predictions and labels. Match metric to use case (AP/AUC/KS/Gini plus PSI/CSI for ranking; precision/recall/FPR at the operating threshold for alerting; RMSE/MAPE plus interval coverage for forecasting). If outputs are consumed as probabilities, check calibration (Brier vs. the constant-prevalence baseline, ECE, decile reliability); a well-discriminating but miscalibrated model is unfit for any use that multiplies score by exposure. Point estimates without confidence or prediction intervals are a finding for anything feeding capital or reserves.

**Overfitting evidence.** Train vs. validation vs. out-of-time decay; complexity vs. effective sample size and events-per-variable; degrees of freedom consumed by feature selection.

**Reproducibility and numerical integrity.** Seeds set and end-to-end determinism; shared feature code for train and inference; pinned data vintages and *exact* dependency versions (lower-bound pins are a finding); artifact loads without version warnings. Silent numerical issues: division by zero, log of non-positives, integer division, float comparison on money, NaN propagation through aggregations, dtype truncation.

**Statistical inference hygiene.** Multiple-comparison exposure without correction, p-values from tests with unverified assumptions, conclusions from insignificant differences.

## Step 3 — Governance tiering and GenAI-specific checks

Tier each inventory component separately. Traditional statistical and ML models meeting the model definition get the full three-pillar treatment (conceptual soundness, outcomes analysis, ongoing monitoring), with Step 2 findings supplying most of the evidence. Learned preprocessing and threshold rules are validated as components of that model.

For GenAI components, first apply the model-definition threshold test from whatever internal or regulatory guidance is in force (check project docs for SR 26-2 or an internal framework; align terminology to it rather than inventing new terms; if none is found, default to SR 11-7 / OCC 2011-12 and flag that as an open item). Components below the model definition still need lightweight quantitative governance: observability metrics (RAGAS-style groundedness/relevance/faithfulness, token overlap), golden-path holdout tests re-run whenever prompts, parameters, or upstream systems change, and unsupervised stability checks (embedding drift, clustering, anomaly detection) where no ground truth exists. Where a GenAI step generates a feature or label consumed downstream, treat its output as a measured variable: error rate against a human-labeled sample, stability under re-runs at nonzero temperature, and propagation of that noise into downstream estimates. In hybrid pipelines, hold the downstream statistical component to full validation regardless of upstream governance.

State the tiering decision and its rationale explicitly.

## Step 4 — Execute what you can, scope what you cannot

Run every check the code and data support, in parallel where independent:

- Re-run the shipped configuration with the current code and compare against the shipped artifact (selection, leaderboard, holdout).
- Recompute headline metrics from stored or regenerated predictions; add bootstrap confidence intervals.
- Reconstruct the split from the seed and confirm partition counts match the run report.
- Plain CV of the fixed shipped specification vs. the harness's reported number.
- Leakage probe: train on shuffled labels; AP near prevalence and AUC near 0.5 expected.
- Benchmark against a simpler challenger (logistic regression, naive seasonal forecast) and a single-variable baseline; a paired bootstrap of the difference.
- Calibration table and a post-hoc calibrator fit on out-of-fold scores to show the remedy.
- Fresh or later-vintage sample scored by the shipped bundle: metrics, flag rate at the shipped threshold, PSI against the bundled reference.
- VIF among selected features; sensitivity to small input perturbations (share of the flagged set that changes).
- Segment parity of flag rates on attributes the model is barred from using.
- Run the test suite and record the result.

The absence of any challenger or benchmark is itself a finding. Check for ongoing monitoring (drift detection, performance decay, data-quality gates, recalibration triggers); absence is a gap. Anything not evaluable from the code and data at hand is an open item for MRM, not a guess.

## Step 5 — Produce the executive output

Write in prose paragraphs (not bullet-heavy), sized to one or two pages for an executive audience, with these sections:

**Model/Framework Overview** — plain-language description of what the pipeline does, the architectures identified, and the business decision it supports. State up front if the evidence base is synthetic or a stale artifact.

**Validation Criteria Identified** — the governance tiering decision and the specific quantitative criteria selected, with rationale.

**Validation Results** — what was tested and found, ranked by severity: defects that invalidate reported performance (leakage, invalid validation design, sample bias, non-reproducible artifact) first; then defects that degrade reliability (miscalibration, overfitting, missing uncertainty); then hygiene (reproducibility, monitoring, inference). Each finding carries file/line evidence, the quantitative consequence, and whether it is confirmed or suspected pending data. Include what could not be evaluated and why. Lead with a short list of what was verified clean.

**Recommended Remediation Steps** — concrete and prioritized: must-fix-before-submission (anything invalidating the performance evidence) vs. ongoing conditions. For each, what to change and what re-test demonstrates closure.

Save the deliverable as a markdown or docx file in the project's documentation folder; it goes to MRM and stakeholders outside the conversation. Keep scratch scripts in the scratchpad, not the repo.
