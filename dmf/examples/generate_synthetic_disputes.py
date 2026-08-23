"""
Synthetic debit-card dispute data.

Generated from an explicit latent logistic model so the ground truth is known:
some variables carry real signal, several are pure noise, two are strongly
collinear with signal-carrying variables, and one is a high-cardinality
merchant id that exists mainly to exercise rare-level collapsing. That mix is
what makes it a fair test of a *selection* framework rather than just a fitting
framework.

The ``drifted`` variant additionally produces records the training data could
not have seen -- new merchants, a channel code that did not exist, amounts an
order of magnitude beyond the training range, a numeric column arriving as
text, and a missing column -- to exercise the inference guard.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

MERCHANT_CATEGORIES = [
    "grocery", "fuel", "restaurant", "digital_goods", "gaming", "electronics",
    "travel", "subscription", "pharmacy", "apparel",
]
CHANNELS = ["card_present", "ecommerce", "mobile_wallet", "recurring", "atm"]
CLAIM_CHANNELS = ["mobile_app", "phone", "branch", "web_portal"]
DEVICE_TYPES = ["ios", "android", "desktop", "unknown"]
REGIONS = ["northeast", "southeast", "midwest", "southwest", "west"]
SEGMENTS = ["mass", "mass_affluent", "affluent", "student", "senior"]
REASON_CODES = [
    "10.4_other_fraud", "10.1_ec_liability", "13.1_not_received",
    "13.3_not_as_described", "13.6_credit_not_processed", "12.5_incorrect_amount",
]


def generate_disputes(
    n: int = 12_000,
    seed: int = 7,
    prevalence: float = 0.075,
    missing_rate: float = 0.03,
    drifted: bool = False,
) -> pd.DataFrame:
    """Return a dispute-level frame with a binary first-party-fraud target."""
    rng = np.random.default_rng(seed)

    # ---------------- behavioural / account variables ----------------
    tenure = rng.gamma(2.2, 22.0, n).clip(1, 400)
    prior_disputes_12m = rng.poisson(0.35 + 1.8 * (tenure < 12), n)
    prior_disputes_life = prior_disputes_12m + rng.poisson(0.5, n)
    upheld_ratio = np.clip(rng.beta(5, 3, n) - 0.25 * (prior_disputes_12m > 2), 0, 1)

    avg_monthly_spend = rng.lognormal(6.6, 0.75, n)
    txn_amount = rng.lognormal(3.5, 1.15, n).clip(1, 6_000)
    dispute_amount = txn_amount * rng.uniform(0.85, 1.0, n)
    amount_ratio = dispute_amount / (avg_monthly_spend / 30.0 + 1.0)

    days_txn_to_dispute = rng.gamma(2.0, 9.0, n).clip(0, 180)
    txn_count_30d = rng.poisson(np.clip(avg_monthly_spend / 90.0, 2, 120), n)
    distinct_merchants_30d = np.minimum(txn_count_30d, rng.poisson(9, n) + 1)
    night_share = np.clip(rng.beta(2, 8, n), 0, 1)
    card_present_share = np.clip(rng.beta(5, 3, n), 0, 1)

    days_since_address_change = rng.exponential(420, n).clip(0, 3_000)
    days_since_reissue = rng.exponential(300, n).clip(0, 2_000)
    failed_login_7d = rng.poisson(0.25, n)
    device_changes_90d = rng.poisson(0.4, n)
    geo_distance_km = rng.exponential(35, n).clip(0, 8_000)
    txn_hour = rng.integers(0, 24, n)
    internal_risk_score = np.clip(rng.normal(520, 95, n), 250, 900)

    # deliberately redundant with tenure / spend, to test redundancy handling
    tenure_years = tenure / 12.0 + rng.normal(0, 0.05, n)
    spend_decile = pd.qcut(avg_monthly_spend, 10, labels=False, duplicates="drop").astype(float)

    # pure noise
    noise_a = rng.normal(0, 1, n)
    noise_b = rng.uniform(0, 100, n)
    noise_c = rng.gamma(2, 2, n)

    # ---------------- categorical variables ----------------
    n_merchants = 220
    merchant_id = rng.choice([f"M{100000 + i}" for i in range(n_merchants)], n,
                             p=_zipf_weights(n_merchants, rng))
    merchant_category = rng.choice(MERCHANT_CATEGORIES, n)
    channel = rng.choice(CHANNELS, n, p=[0.30, 0.34, 0.18, 0.11, 0.07])
    claim_channel = rng.choice(CLAIM_CHANNELS, n, p=[0.52, 0.24, 0.09, 0.15])
    device_type = rng.choice(DEVICE_TYPES, n, p=[0.38, 0.40, 0.18, 0.04])
    region = rng.choice(REGIONS, n)
    segment = rng.choice(SEGMENTS, n, p=[0.44, 0.26, 0.10, 0.12, 0.08])
    reason_code = rng.choice(REASON_CODES, n, p=[0.30, 0.14, 0.22, 0.14, 0.11, 0.09])

    # ---------------- latent risk of a *fraudulent* dispute ----------------
    z = (
        0.62 * _z(prior_disputes_12m)
        + 0.48 * _z(amount_ratio)
        - 0.41 * _z(np.log1p(tenure))
        + 0.37 * _z(days_txn_to_dispute)
        + 0.33 * _z(device_changes_90d)
        - 0.30 * _z(internal_risk_score)
        + 0.26 * _z(night_share)
        - 0.24 * _z(card_present_share)
        + 0.21 * _z(failed_login_7d)
        - 0.18 * _z(upheld_ratio)
        + 0.15 * _z(np.log1p(geo_distance_km))
        + 0.55 * np.isin(merchant_category, ["digital_goods", "gaming"])
        + 0.42 * (channel == "ecommerce")
        + 0.35 * (reason_code == "13.1_not_received")
        + 0.28 * (claim_channel == "mobile_app")
        - 0.22 * (segment == "senior")
        # interaction the linear champion cannot represent but a tree can
        + 0.85 * ((amount_ratio > np.quantile(amount_ratio, 0.8)) & (prior_disputes_12m >= 2))
    )
    z = z + rng.normal(0, 0.9, n)
    # solve for the intercept that puts expected prevalence exactly on target
    z = z - _intercept_for_prevalence(z, prevalence)
    p = 1.0 / (1.0 + np.exp(-z))
    y = rng.binomial(1, p)

    df = pd.DataFrame(
        {
            "dispute_id": [f"D{2026_000000 + i}" for i in range(n)],
            "dispute_amount": np.round(dispute_amount, 2),
            "txn_amount": np.round(txn_amount, 2),
            "amount_to_daily_spend_ratio": np.round(amount_ratio, 4),
            "days_txn_to_dispute": np.round(days_txn_to_dispute, 1),
            "cardholder_tenure_months": np.round(tenure, 1),
            "cardholder_tenure_years": np.round(tenure_years, 3),
            "prior_disputes_12m": prior_disputes_12m,
            "prior_disputes_lifetime": prior_disputes_life,
            "prior_dispute_upheld_ratio": np.round(upheld_ratio, 4),
            "avg_monthly_spend": np.round(avg_monthly_spend, 2),
            "spend_decile": spend_decile,
            "txn_count_30d": txn_count_30d,
            "distinct_merchants_30d": distinct_merchants_30d,
            "night_txn_share_30d": np.round(night_share, 4),
            "card_present_share_30d": np.round(card_present_share, 4),
            "days_since_address_change": np.round(days_since_address_change, 1),
            "days_since_card_reissue": np.round(days_since_reissue, 1),
            "failed_logins_7d": failed_login_7d,
            "device_changes_90d": device_changes_90d,
            "geo_distance_km": np.round(geo_distance_km, 2),
            "txn_hour": txn_hour,
            "internal_risk_score": np.round(internal_risk_score, 1),
            "noise_gaussian": np.round(noise_a, 4),
            "noise_uniform": np.round(noise_b, 3),
            "noise_gamma": np.round(noise_c, 4),
            "merchant_id": merchant_id,
            "merchant_category": merchant_category,
            "channel": channel,
            "claim_channel": claim_channel,
            "device_type": device_type,
            "region": region,
            "customer_segment": segment,
            "dispute_reason_code": reason_code,
            "is_fraudulent_dispute": y,
        }
    )

    # ---------------- realistic missingness ----------------
    for col in ["internal_risk_score", "geo_distance_km", "device_type",
                "days_since_address_change", "prior_dispute_upheld_ratio"]:
        mask = rng.random(n) < missing_rate
        df.loc[mask, col] = np.nan

    if drifted:
        df = _apply_drift(df, rng)
    return df


def _apply_drift(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Inject exactly the inference-time surprises the guard is meant to absorb."""
    n = len(df)
    out = df.copy()

    # 1. merchants that did not exist during training
    novel = rng.random(n) < 0.12
    out.loc[novel, "merchant_id"] = [f"M9{rng.integers(100000, 999999)}" for _ in range(int(novel.sum()))]

    # 2. a channel code introduced after the model was built
    new_channel = rng.random(n) < 0.05
    out.loc[new_channel, "channel"] = "crypto_offramp"

    # 3. amounts an order of magnitude outside the training range
    blowout = rng.random(n) < 0.02
    out.loc[blowout, "dispute_amount"] = out.loc[blowout, "dispute_amount"] * rng.uniform(30, 90, int(blowout.sum()))

    # 4. an upstream bug turning a numeric column into text
    junk = rng.random(n) < 0.01
    out["internal_risk_score"] = out["internal_risk_score"].astype(object)
    out.loc[junk, "internal_risk_score"] = "N/A"

    # 5. a column the upstream feed stopped sending
    out = out.drop(columns=["days_since_card_reissue"])

    # 6. a negative value where the training data had none
    neg = rng.random(n) < 0.01
    out.loc[neg, "cardholder_tenure_months"] = -out.loc[neg, "cardholder_tenure_months"]
    return out


def _intercept_for_prevalence(z: np.ndarray, target: float, tol: float = 1e-6) -> float:
    """Bisect for the shift c such that mean(sigmoid(z - c)) == target."""
    lo, hi = float(z.min()) - 20.0, float(z.max()) + 20.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        rate = float(np.mean(1.0 / (1.0 + np.exp(-(z - mid)))))
        if abs(rate - target) < tol:
            return mid
        if rate > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _z(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = np.nanstd(x)
    return (x - np.nanmean(x)) / (sd if sd > 0 else 1.0)


def _zipf_weights(k: int, rng: np.random.Generator) -> np.ndarray:
    w = 1.0 / np.arange(1, k + 1) ** 1.1
    return w / w.sum()


def main(out_dir: str = "data", n: int = 12_000, seed: int = 7) -> Tuple[str, str]:
    from pathlib import Path

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    train = generate_disputes(n=n, seed=seed)
    drift = generate_disputes(n=max(n // 6, 500), seed=seed + 101, drifted=True)
    p1 = str(Path(out_dir) / "disputes.csv")
    p2 = str(Path(out_dir) / "disputes_next_month_drifted.csv")
    train.to_csv(p1, index=False)
    drift.drop(columns=["is_fraudulent_dispute"]).to_csv(p2, index=False)
    print(f"wrote {p1}  shape={train.shape}  prevalence={train['is_fraudulent_dispute'].mean():.4f}")
    print(f"wrote {p2}  shape={drift.shape}  (unseen levels + out-of-range values + missing column)")
    return p1, p2


if __name__ == "__main__":
    main()
