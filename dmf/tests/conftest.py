import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from dmf import Config  # noqa: E402
from generate_synthetic_disputes import generate_disputes  # noqa: E402


@pytest.fixture(scope="session")
def raw() -> pd.DataFrame:
    return generate_disputes(n=1800, seed=11, prevalence=0.12)


@pytest.fixture(scope="session")
def drifted() -> pd.DataFrame:
    df = generate_disputes(n=400, seed=99, prevalence=0.12, drifted=True)
    return df.drop(columns=["is_fraudulent_dispute"])


@pytest.fixture()
def Xy(raw):
    df = raw.drop(columns=["dispute_id"]).copy()
    y = df.pop("is_fraudulent_dispute").to_numpy()
    return df, y


def make_cfg() -> Config:
    """Small, fast configuration exercising both a linear and a tree model."""
    return Config.from_dict(
        {
            "run": {"name": "test_run", "random_state": 0, "n_jobs": 1, "verbose": 0,
                    "save_fitted_model": False, "refit_on_full_data": False},
            "data": {"target": "is_fraudulent_dispute"},
            "columns": {
                "numeric": ["dispute_amount", "amount_to_daily_spend_ratio", "prior_disputes_12m",
                            "cardholder_tenure_months", "device_changes_90d", "internal_risk_score",
                            "noise_gaussian"],
                "categorical": ["merchant_category", "channel", "merchant_id"],
                "drop": ["dispute_id"],
                "auto_infer": False,
            },
            "split": {"holdout_size": 0.25, "cv": {"n_splits": 3}},
            "selection": {"k_min": 1, "k_max": 3, "top_n": 2, "ordering_strategy": "importance"},
            "metrics": {"primary": "average_precision",
                        "secondary": ["roc_auc", "ks_statistic", "brier_score"],
                        "slice_columns": ["channel", "merchant_category"],
                        "min_slice_n": 20},
            "models": {
                "logistic": {
                    "estimator": "sklearn.linear_model.LogisticRegression",
                    "family": "linear", "tag": "champion", "requires_scaling": True,
                    "imbalance": "balanced",
                    "params": {"C": 1.0, "max_iter": 1000},
                },
                "trees": {
                    "estimator": "sklearn.ensemble.RandomForestClassifier",
                    "family": "tree", "tag": "challenger", "requires_scaling": False,
                    "imbalance": "balanced",
                    "params": {"n_estimators": 40, "min_samples_leaf": 15, "n_jobs": 1},
                },
            },
        }
    )


@pytest.fixture()
def cfg() -> Config:
    return make_cfg()
