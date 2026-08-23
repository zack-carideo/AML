"""
Adversarial audit: ten datasets, each carrying one production failure mode.

Each case is run end-to-end through the harness (ordering -> CV grid ->
selection -> holdout -> production refit -> reload -> score a fresh batch).
The point is not that every case should score well -- several are designed to
be unlearnable -- but that none should *crash*, silently mis-type a column, or
produce a nonsensical artifact.

    python examples/edge_case_audit.py
"""

from __future__ import annotations

import sys
import traceback
import warnings
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore", category=FutureWarning)

from dmf import Config, ProductionScorer
from dmf.research import ModelSelectionHarness  # noqa: E402
from generate_synthetic_disputes import generate_disputes  # noqa: E402

N = 1500
SEED = 3


def _base(n: int = N, seed: int = SEED, prevalence: float = 0.12) -> pd.DataFrame:
    return generate_disputes(n=n, seed=seed, prevalence=prevalence).drop(
        columns=[
            "dispute_id", "noise_uniform", "noise_gamma", "spend_decile",
            "prior_disputes_lifetime", "distinct_merchants_30d", "days_since_address_change",
            "days_since_card_reissue", "txn_hour", "geo_distance_km", "region",
            "customer_segment", "claim_channel", "device_type", "cardholder_tenure_years",
            "card_present_share_30d", "night_txn_share_30d", "txn_count_30d",
            "avg_monthly_spend", "failed_logins_7d", "merchant_id",
        ]
    )


# --------------------------------------------------------------------------
# the ten cases
# --------------------------------------------------------------------------
def case_all_missing_column() -> Tuple[pd.DataFrame, str]:
    df = _base()
    df["kyc_refresh_score"] = np.nan          # feed never populated in this window
    return df, "a feature column that is 100% missing"


def case_constant_column() -> Tuple[pd.DataFrame, str]:
    df = _base()
    df["product_code"] = "DEBIT_STD"          # single-product portfolio
    df["fee_waiver_flag"] = 0
    return df, "zero-variance numeric and categorical columns"


def case_extreme_imbalance() -> Tuple[pd.DataFrame, str]:
    return _base(n=4000, prevalence=0.005), "prevalence 0.5% (20 positives)"


def case_unique_id_column() -> Tuple[pd.DataFrame, str]:
    df = _base()
    df["case_reference"] = [f"CR-{i:08d}" for i in range(len(df))]   # cardinality == n
    return df, "an ID-like categorical with one level per row"


def case_collinear_and_duplicate_rows() -> Tuple[pd.DataFrame, str]:
    df = _base()
    df["dispute_amount_cents"] = df["dispute_amount"] * 100          # perfectly collinear
    df = pd.concat([df, df.head(300)], ignore_index=True)            # 20% duplicate rows
    return df, "perfectly collinear feature pair and duplicated rows"


def case_dirty_numeric_strings() -> Tuple[pd.DataFrame, str]:
    df = _base()
    df["dispute_amount"] = df["dispute_amount"].map(lambda v: f"${v:,.2f}")
    df["internal_risk_score"] = df["internal_risk_score"].map(
        lambda v: "N/A" if not np.isfinite(v) else f"{v:.1f}"
    )
    df["chargeback_eligible"] = np.where(df["prior_disputes_12m"] > 0, "Y", "N")
    return df, "currency-formatted and sentinel-coded numerics arriving as text"


def case_datetime_columns() -> Tuple[pd.DataFrame, str]:
    df = _base()
    base = pd.Timestamp("2026-01-01")
    offs = pd.to_timedelta(np.random.default_rng(0).integers(0, 240, len(df)), unit="D")
    df["txn_date"] = (base + offs).astype(str)                       # ISO strings
    df["claim_ts"] = base + offs + pd.to_timedelta(3, unit="D")      # datetime64 dtype
    return df, "date columns as ISO strings and as datetime64"


def case_string_target_labels() -> Tuple[pd.DataFrame, str]:
    df = _base()
    df["is_fraudulent_dispute"] = np.where(df["is_fraudulent_dispute"] == 1, "FRAUD", "GENUINE")
    return df, "target labelled 'FRAUD'/'GENUINE' rather than 1/0"


def case_infinities() -> Tuple[pd.DataFrame, str]:
    df = _base()
    rng = np.random.default_rng(1)
    idx = rng.choice(df.index, 40, replace=False)
    df.loc[idx[:20], "amount_to_daily_spend_ratio"] = np.inf          # divide-by-zero ratio
    df.loc[idx[20:], "amount_to_daily_spend_ratio"] = -np.inf
    df.loc[rng.choice(df.index, 10, replace=False), "dispute_amount"] = 1e308
    return df, "inf / -inf and 1e308 from a divide-by-zero ratio"


def case_tiny_wide() -> Tuple[pd.DataFrame, str]:
    df = _base(n=140, prevalence=0.10)
    rng = np.random.default_rng(2)
    for i in range(45):
        df[f"derived_{i:02d}"] = rng.normal(0, 1, len(df))            # p > useful n
    return df, "140 rows, 55 candidate variables, 14 positives"


CASES: Dict[str, Callable[[], Tuple[pd.DataFrame, str]]] = {
    "01_all_missing_column": case_all_missing_column,
    "02_constant_columns": case_constant_column,
    "03_extreme_imbalance": case_extreme_imbalance,
    "04_unique_id_column": case_unique_id_column,
    "05_collinear_duplicates": case_collinear_and_duplicate_rows,
    "06_dirty_numeric_strings": case_dirty_numeric_strings,
    "07_datetime_columns": case_datetime_columns,
    "08_string_target_labels": case_string_target_labels,
    "09_infinities": case_infinities,
    "10_tiny_wide": case_tiny_wide,
}


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------
def audit_config(name: str, out_dir: Path, positive_label=1) -> Config:
    """A deliberately generic config: nothing dataset-specific, auto-inferred roles."""
    return Config.from_dict(
        {
            "run": {"name": name, "output_dir": str(out_dir), "random_state": 0,
                    "n_jobs": 1, "verbose": 0, "save_fitted_model": True,
                    "refit_on_full_data": True},
            "data": {"target": "is_fraudulent_dispute", "positive_label": positive_label},
            "columns": {"auto_infer": True},
            "split": {"holdout_size": 0.25, "cv": {"n_splits": 3}},
            "selection": {"k_min": 1, "k_max": 4, "top_n": 3, "ordering_strategy": "importance"},
            "metrics": {"primary": "average_precision",
                        "secondary": ["roc_auc", "ks_statistic", "brier_score"]},
            "models": {
                "logistic": {
                    "estimator": "sklearn.linear_model.LogisticRegression",
                    "family": "linear", "tag": "champion", "requires_scaling": True,
                    "imbalance": "balanced", "params": {"C": 1.0, "max_iter": 1000},
                },
                "gbm": {
                    "estimator": "sklearn.ensemble.HistGradientBoostingClassifier",
                    "family": "tree", "tag": "challenger", "requires_scaling": False,
                    "imbalance": "balanced",
                    "params": {"max_iter": 60, "learning_rate": 0.1, "early_stopping": False},
                },
            },
        }
    )


def run_case(name: str, builder, out_dir: Path) -> Dict[str, object]:
    df, description = builder()
    # 'auto' = minority class, which is right for a fraud target whatever the
    # labels happen to be called in the source system.
    cfg = audit_config(name, out_dir, positive_label="auto")
    row: Dict[str, object] = {"case": name, "issue": description,
                              "n_rows": len(df), "n_cols": df.shape[1] - 1}
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = ModelSelectionHarness(cfg).run(df.copy())

            model_path = out_dir / name / "model.joblib"
            scorer = ProductionScorer.from_joblib(model_path)
            fresh = df.drop(columns=["is_fraudulent_dispute"]).sample(
                min(200, len(df)), random_state=9
            )
            scored, report = scorer.score(fresh)

        p = scored["fraud_probability"].to_numpy()
        cand = res.report.get("candidate_variables")
        row.update(
            status="ok",
            n_candidates=cand["n_candidates"],
            dropped=cand["dropped_by_reason"] or "",
            recovered=cand["recovered_from_text"] or "",
            as_datetime=cand["parsed_as_datetime"] or "",
            selected_model=res.selected_model,
            k=res.selected["k"],
            cv_ap=round(res.selected["cv_average_precision_mean"], 4),
            holdout_ap=res.holdout_metrics.get("average_precision"),
            design_cols=len(res.fitted_model.named_steps["features"].feature_names_out_),
            scores_finite=bool(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()),
            guard_verdict=report.get("verdict"),
            n_warnings=len({str(w.message)[:60] for w in caught}),
            note="",
        )
    except Exception as exc:  # noqa: BLE001 - the audit is about finding these
        row.update(status="FAIL", note=f"{type(exc).__name__}: {exc}",
                   traceback=traceback.format_exc().splitlines()[-4:])
    return row


def main() -> pd.DataFrame:
    out_dir = Path(__file__).resolve().parents[1] / "artifacts" / "edge_cases"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, builder in CASES.items():
        print(f"--- {name} ...", flush=True)
        row = run_case(name, builder, out_dir)
        print(f"    {row['status']}  {row.get('note', '')}", flush=True)
        rows.append(row)

    report = pd.DataFrame(rows)
    pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_colwidth", 60)
    cols = ["case", "status", "n_candidates", "dropped", "recovered", "as_datetime",
            "design_cols", "selected_model", "k", "cv_ap", "holdout_ap", "scores_finite",
            "guard_verdict", "note"]
    print("\n" + "=" * 120)
    print(report[[c for c in cols if c in report.columns]].to_string(index=False))
    report.to_csv(out_dir / "audit_summary.csv", index=False)
    return report


if __name__ == "__main__":
    main()
