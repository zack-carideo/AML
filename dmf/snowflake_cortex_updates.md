# dmf on Snowflake — porting guide

Changes and updates to make the `dmf` framework run natively on Snowflake, with training
data sourced from Snowflake tables and code executed inside Snowflake compute (Snowpark /
Container Runtime, models served through the Model Registry, and the phase-2 explanation
layer built on Cortex LLM functions). This is a design document for developers — no code
in the package changes until each item is deliberately taken.

The guiding principle of the port is the same one the package split established: **the
`dmf` core stays environment-agnostic.** It consumes a pandas DataFrame and emits a
fitted sklearn pipeline; nothing in it should import Snowpark. Everything
Snowflake-specific lands in thin adapter layers — an IO adapter feeding the harness, a
registry adapter shipping the model, and a UDF wrapper serving it. That keeps the 84
existing tests valid unchanged and means the same core runs on a laptop, in CI, and in
Snowflake.

---

## 1. Where each half runs

| dmf component | Snowflake home | Why |
|---|---|---|
| `dmf.research` (harness, CV grid, ordering) | **Snowflake Notebook on Container Runtime**, or an **ML Job** (`snowflake.ml.jobs`) on a compute pool | Long-running, CPU-parallel, needs the full frame in memory; Container Runtime allows arbitrary pip packages and real `n_jobs` parallelism |
| `dmf` core — training-time fit | Same environment as the harness (it is called by it) | — |
| `dmf` core — production inference | **Model Registry** model version, served in a warehouse or on Snowpark Container Services; batch scoring via `model_version.run()` or a vectorized UDF | The fitted pipeline is a plain sklearn object — exactly what the registry natively logs |
| Artifacts (leaderboard, reports, slices) | Snowflake **tables** (preferred) or a named **stage** | Local container filesystem is ephemeral |
| Phase-2 GenAI explanation layer | **Cortex** (`SNOWFLAKE.CORTEX.COMPLETE` / AISQL) over the scorer's outputs | Row-level LLM calls next to the data, no data egress |

A stored-procedure-on-warehouse deployment of the *harness* is possible but not
recommended: warehouse Python is restricted to the Snowflake Anaconda channel, memory is
tighter (use a Snowpark-optimized warehouse if forced), and wall-clock for the nested CV
grid benefits from Container Runtime's unrestricted parallelism.

---

## 2. Data sourcing (`DataConfig` and the load path)

### 2.1 Add table/query sources to the config

`data.path` + `data.format` assume files. Add two mutually-exclusive source fields and an
IO adapter *outside* the core:

```yaml
data:
  # exactly one of: path | table | query
  table: PROD_DB.DISPUTES.DISPUTE_FEATURES_V1
  # query: SELECT * FROM ... WHERE claim_date >= '2026-01-01'
  order_by: dispute_id          # REQUIRED for table/query sources -- see 2.2
  target: is_fraudulent_dispute
```

The adapter is ~15 lines that the harness calls before `run()`:

```python
def load_from_snowflake(session, cfg) -> pd.DataFrame:
    df = session.table(cfg.data.table) if cfg.data.table else session.sql(cfg.data.query)
    return df.sort(cfg.data.order_by).to_pandas()
```

Keep `ModelSelectionHarness.run(X, y)` accepting a DataFrame as it does today — the
adapter feeds it. Do **not** teach `selection._load` to import Snowpark.

### 2.2 Row order is not deterministic in Snowflake — this breaks reproducibility silently

A `SELECT` without `ORDER BY` returns rows in whatever order micro-partitions are
scanned, which varies run to run. dmf's holdout split, CV folds, `sample_frac`, and the
`run_lineage` data fingerprint are all **positional**. Two "identical" runs on the same
table will produce different splits, different leaderboards, and different
`data_sha256` values, and nothing will error.

**Make `order_by` mandatory for table/query sources** (validate it in the adapter) and
sort on a unique key before `to_pandas()`. This is the single most important line in this
document.

### 2.3 Type-mapping gotchas at the `to_pandas()` boundary

Snowflake → pandas conversion produces shapes the core mostly already survives, but two
deserve explicit handling in the adapter:

- `NUMBER(p, 0)` wider than int64, and some `NUMBER(p, s)` columns, arrive as
  `object` dtype holding `decimal.Decimal`. `NumericCoercer` / `parse_kind` recover
  these via `pd.to_numeric`, but at the cost of the column being *inferred* rather than
  declared. Cleaner: cast in SQL (`col::FLOAT`) or in the adapter, so the declared
  `columns.numeric` roles match arriving dtypes.
- `TIMESTAMP_TZ` / `TIMESTAMP_LTZ` arrive tz-aware; `to_numeric_lenient` already
  normalises to UTC epoch-days, so date columns work — but confirm the `time` split
  column is `TIMESTAMP_NTZ` or consistently zoned, or the forward-only split boundary
  can shift with session timezone.
- `VARIANT`/`OBJECT` columns arrive as JSON strings → they will be typed as
  high-cardinality categoricals and dropped as `identifier_like_cardinality`. Flatten
  them in SQL before they reach the frame.

### 2.4 Push sampling and filtering into SQL

`data.sample_frac` currently samples in pandas after a full `to_pandas()`. On a
100M-row dispute table that is the memory bill. Prefer
`SAMPLE (10)` / `TABLESAMPLE BERNOULLI` or a `WHERE` window in the source query, and keep
`sample_frac` as a post-load convenience for small data. Same for the training window:
filter months in SQL, not pandas.

---

## 3. Execution environment and packaging

- **Dependencies.** `numpy`, `pandas`, `scikit-learn`, `scipy`, `pyyaml`, `joblib`,
  `xgboost`, `lightgbm` are all in the Snowflake Anaconda channel — warehouse execution
  is possible dependency-wise. On Container Runtime, plain `pip install` works. **Pin
  the sklearn version everywhere** (training env, registry `conda_dependencies`, any
  UDF): the model bundle already warns on `dmf_version` skew; sklearn skew is the other
  half of pickle risk.
- **Packaging dmf itself.** Two supported routes:
  1. Build the wheel (`python -m build`) and `pip install` it in the Container Runtime
     image / ML Job payload — preferred, versioned, matches `dmf_version` in the bundle.
  2. Zip `src/dmf` to a stage and pass via `imports=[...]` for warehouse
     sprocs/UDFs — works because dmf is pure-Python with no compiled extensions.
- **Parallelism.** `run.n_jobs` maps to real cores only on Container Runtime / ML Jobs.
  In a warehouse sproc, keep `n_jobs: 1` and let Snowflake scale the warehouse instead.
- **Python version.** Match the training Python (3.10/3.11) with the serving runtime
  Python; pickles are not guaranteed portable across minor versions.
- **No egress needed.** dmf makes no network calls, so no External Access Integrations
  are required — a genuine advantage for bank security review.

---

## 4. Artifact persistence (`_write_artifacts`)

The harness writes CSVs/JSON to a local directory that evaporates with the container.
Add a persistence adapter (again, outside the core) with two targets:

- **Tables (preferred).** `leaderboard`, `marginal_gains`, `holdout_slices`,
  `holdout_deciles` are naturally relational — write each to
  `ML_OPS.DMF_RUNS.<artifact>` with `run_name` and `run_ts` columns via
  `session.write_pandas`. This is what makes runs queryable ("show me every run's
  holdout AP by month") and lets the slice/fairness table feed dashboards directly.
- **Stage (fallback).** `session.file.put` the JSON blobs and `final_spec.yaml` to
  `@ML_OPS.DMF_RUNS.ARTIFACTS/<run_name>/` for byte-exact copies.

`model.joblib` should **stop being the production artifact** in Snowflake — see §5. Keep
writing it to the stage as the portable escape hatch.

---

## 5. Model Registry instead of `model.joblib`

The fitted `features → model` pipeline is a standard sklearn `Pipeline`, which the
Snowflake Model Registry logs natively:

```python
from snowflake.ml.registry import Registry

reg = Registry(session, database_name="ML_OPS", schema_name="MODELS")
mv = reg.log_model(
    result.fitted_model,
    model_name="dispute_fraud",
    version_name=cfg.run.name,                       # e.g. dispute_fraud_v1
    conda_dependencies=["scikit-learn==<pinned>", "pandas", "numpy"],
    sample_input_data=X_train.head(100),             # locks the input signature
    metrics=result.holdout_metrics,                  # AP, KS, calibration...
    comment=f"dmf {dmf.__version__} | config {lineage['config_sha256']}",
)
```

Three dmf-specific points:

1. **The bundle's sidecar fields need a home.** `decision_threshold`,
   `reference_score_quantiles`, `dmf_version`, and `lineage` currently live beside the
   pipeline in the joblib dict. In the registry, store them as **model version
   metadata/metrics** — they must travel with the version, not in a separate table
   someone forgets to join.
2. **`predict_proba` alone loses the guard.** Logging the bare pipeline gives you
   probabilities but silently discards the data-quality verdict, per-row flags, and
   decision/action columns — the part of dmf that exists for a bank. Wrap
   `ProductionScorer` in a `snowflake.ml.model.custom_model.CustomModel` whose
   `predict` returns the full `score()` frame (probability, flags, `data_quality`,
   `decision`, `action`), and log *that*. This is the registry-native equivalent of the
   scorer and should be the default deployment.
3. **Registry loads re-instantiate your classes**, so the environment serving the model
   must have the same `dmf` version installed (wheel in the model's
   `code_paths`/dependencies). The existing `_warn_on_version_skew` logic should be
   mirrored by pinning `dmf==x.y.z` in the logged model's requirements.

---

## 6. Inference patterns and guard semantics

- **Batch scoring:** `mv.run(session.table("NEW_DISPUTES"), function_name="predict")`
  or a scheduled task calling the same. This replaces `dmf score` CLI in production;
  keep the CLI for local/dev parity.
- **Partitioned execution changes what "batch" means.** Warehouse-served models and
  vectorized UDFs process data in **chunks the engine chooses**. Two dmf behaviours are
  batch-relative and must be revisited:
  - `top_pct` relative cut: a per-chunk quantile is meaningless. In Snowflake, **always
    deploy with an absolute `decision_threshold`** (the bundle field exists for exactly
    this); alternatively compute the day's cut in SQL over the full scored table and
    apply `decision` there.
  - The guard's `max_guarded_rate` / `batch_safe` verdict becomes per-chunk. Aggregate
    instead in SQL over the scored output (`AVG(n_out_of_range > 0)` etc.) and alert on
    the daily total; treat per-chunk verdicts as advisory.
- **Missing-column policy still works** (the signature check happens at the model
  boundary), but prefer failing at the SQL layer: the registry's locked input signature
  will refuse a mis-shaped relation before dmf's guard ever sees it — two fences are
  better than one.

---

## 7. Lineage additions (`run_lineage`)

Extend the lineage dict (via the adapter, passing extras into the report) with the
Snowflake-native provenance a model-risk reviewer will actually ask for:

- `query_id` of the extraction (`session.sql(...).collect()` → last query id) — replays
  the exact read.
- **Time Travel anchor**: run the extraction `AT (STATEMENT => '<query_id>')` or record
  the table version timestamp, so "the same data" is re-fetchable even after upstream
  loads.
- Replace/augment the pandas `data_sha256` with a pushed-down
  `SELECT HASH_AGG(*) FROM <source>` — order-independent, computed in-warehouse, and
  free compared to hashing 100M rows in pandas.
- `warehouse`, `role`, `database.schema`, `snowflake-ml-python` version.

---

## 8. Monitoring: dmf PSI vs ML Observability

Snowflake **ML Observability** (`CREATE MODEL MONITOR`) provides drift, volume, and
performance monitoring over a registry model's inference table. Recommended division of
labour:

- Let a **model monitor** own score drift, feature drift, and (once labels arrive)
  performance-over-time — it is the platform-native, dashboarded path.
- Keep dmf's `reference_score_quantiles` + `population_stability_index` as the
  **portable check** used in CI and pre-promotion gates, so the same number is
  computable outside Snowflake.
- The `holdout_slices` table maps directly onto segmented monitoring: persist it per run
  (§4) and point fairness dashboards at it; `metrics.slice_columns` should match the
  monitor's segmentation columns so the training-time and serving-time slice views line
  up.
- Route the guard's per-row flag counts into the scored table (they are already output
  columns) and alert via a simple task/`ALERT` on daily aggregates.

---

## 9. Cortex hook for the phase-2 explanation layer

The project's second goal — analyst-readable reasons and investigation steps — is where
Cortex proper enters. The scorer's outputs are already the right prompt inputs; no core
change is needed, only a consumer:

```sql
SELECT dispute_id,
       SNOWFLAKE.CORTEX.COMPLETE(
         'claude-sonnet',          -- or the org-approved model
         CONCAT('A dispute scored ', fraud_probability, ' for first-party fraud. ',
                'Top model drivers: ', top_drivers, '. Data-quality flags: ', guard_flags,
                '. Segment context: ', slice_context,
                '. Write 3 concrete investigation steps for the analyst.')
       ) AS investigation_steps
FROM scored_disputes WHERE decision = 'flag_fraud';
```

To feed `top_drivers`, add per-record contributions upstream (coefficient × value for
the WOE champion — trivially available from `feature_source_map_` — or SHAP for the
challengers) and emit them as a column from the CustomModel wrapper. Guard flags should
be surfaced verbatim: "the amount was outside anything seen in training" is itself an
investigation step. Keep prompts versioned in a table, and never let the LLM see columns
excluded for fairness reasons — the same `slice_columns`-not-features discipline applies.

---

## 10. Testing and CI

- The 84 tests need **no Snowflake** — that is the point of keeping the core
  environment-agnostic. Run them in CI as today.
- Add a thin integration test layer for the adapters using the **Snowpark local testing
  framework** (`session = Session.builder.config("local_testing", True)...`) for the IO
  adapter, plus one live smoke test per environment (extract → harness on a 5k sample →
  registry log → `mv.run` on 100 rows → guard columns present).
- Gate promotion in CI on: suite green, edge-case audit green, holdout AP within
  tolerance of the leaderboard estimate, `max_flag_rate_disparity` under a declared
  bound, and PSI vs the prior model's reference quantiles.

---

## 11. Prioritised change list

| P | Change | Touches | Core code change? |
|---|---|---|---|
| **P0** | Snowflake IO adapter with **mandatory `order_by`** on table/query sources | new `dmf_snowflake/io.py` (adapter pkg) + `DataConfig` fields | config fields only |
| **P0** | Deploy with absolute `decision_threshold`; never `top_pct` under partitioned inference | deployment config | none |
| **P0** | Registry logging via a `CustomModel` wrapper that preserves guard/decision outputs | new adapter module | none |
| **P1** | Artifacts → tables (`write_pandas`), stage fallback; `model.joblib` demoted to escape hatch | adapter around `_write_artifacts` | none if adapter post-processes the artifact dir; small hook if done inline |
| **P1** | Lineage extras: `query_id`, Time Travel anchor, `HASH_AGG(*)`, warehouse/role | adapter passes extras; `run_lineage` accepts `**extra` | ~2 lines |
| **P1** | Type-cast contract in SQL (`::FLOAT`, flatten VARIANT), tz-consistent time column | extraction SQL | none |
| **P1** | Batch-verdict aggregation in SQL; per-chunk guard verdicts advisory | scoring SQL/alerts | none |
| **P2** | ML Observability monitor wired to registry model; slices table → fairness dashboard | Snowflake objects | none |
| **P2** | Cortex COMPLETE explanation consumer + per-record driver column from the wrapper | phase-2 layer | none (wrapper-level) |
| **P2** | Sampling/window pushdown to SQL; retire pandas-side `sample_frac` for large tables | extraction SQL | none |

The pattern in the last column is deliberate: **almost nothing in `dmf` itself changes.**
The port is an adapter package (`dmf_snowflake/`, roughly 150–250 lines) plus SQL and
platform objects, which keeps the production core's maintenance surface exactly where the
package split put it.
