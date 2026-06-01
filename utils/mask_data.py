

Skip to content
Using Gmail with screen readers

3 of 42,058
data masker
Inbox

Zachary Carideo <zjc1002@gmail.com>
Attachments
2:33 PM (56 minutes ago)
to me

 One attachment
  •  Scanned by Gmail
"""
data_masker.py — Generalizable data masking utility for tabular data.

Guarantees:
    • Row-wise uniqueness   — distinct rows stay distinct after masking
                              (1:1 per-column value mapping)
    • Attribute relationships — every occurrence of value v in column c maps
                              to the same masked token, so functional
                              dependencies, equality joins, and value
                              co-occurrence patterns are preserved
    • Column dtypes (best effort) — strings → strings, numerics → numerics,
                              datetimes → datetimes, bools → bools

Each fitted instance carries a unique salt, so the same DataFrame fitted
twice produces two different (but internally consistent) masks. Pass
`salt=...` to get reproducible output.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Optional

import numpy as np
import pandas as pd


class DataMasker:
    """Deterministic, dtype-aware tabular data masker.

    Parameters
    ----------
    salt : str, optional
        Hex string used to key the HMAC. If omitted, a cryptographically
        random 128-bit salt is generated — this is what makes each fit
        produce a "unique mask for any dataset".
    preserve_dtypes : bool, default True
        If True, masked numerics stay numeric and masked datetimes stay
        datetime. If False, every column becomes a hashed string token.
    numeric_strategy : {"tokenize", "shift_scale"}, default "tokenize"
        "tokenize"   — map each unique value to a deterministic float
                       in [0, 1) via HMAC. Destroys order/correlation
                       but guarantees 1:1.
        "shift_scale" — apply a deterministic affine transform a*x + b
                       per column. Preserves order and correlations
                       (useful for EDA) but leaks ordering info.
    datetime_jitter_days : int, default 3650
        Half-range of the per-column datetime shift, in days.
    text_passthrough : bool, default True
        If True, auto-detect natural-text columns (long, high-cardinality
        strings — e.g. descriptions, memos, comments) and leave them
        unchanged. Hashing them destroys NLP value and gives little
        privacy back, since uniqueness already identifies the row.
    text_uniqueness_threshold : float, default 0.85
        Min ratio of unique non-null values for a column to qualify
        as natural text.
    text_min_avg_length : int, default 20
        Min average character length for a column to qualify.
    text_min_avg_words : float, default 3.0
        Min average word count — filters out UUIDs, hashes, and codes
        which can be highly unique but aren't natural language.
    passthrough_cols : list[str], optional
        Explicit list of columns to leave unchanged, overriding detection.
    force_mask_cols : list[str], optional
        Explicit list of columns to always mask, overriding detection.
    exclude_all_categoricals : bool, default False
        If True, every categorical-like column (pandas Categorical dtype
        or object/string dtype) is left unchanged. Useful when you want
        to mask only numerics, IDs, and datetimes while keeping segment
        labels, country codes, etc. readable for downstream analysis.
    """

    def __init__(
        self,
        salt: Optional[str] = None,
        preserve_dtypes: bool = True,
        numeric_strategy: str = "tokenize",
        datetime_jitter_days: int = 3650,
        text_passthrough: bool = True,
        text_uniqueness_threshold: float = 0.85,
        text_min_avg_length: int = 20,
        text_min_avg_words: float = 3.0,
        passthrough_cols: Optional[list] = None,
        force_mask_cols: Optional[list] = None,
        exclude_all_categoricals: bool = False,
    ):
        if numeric_strategy not in ("tokenize", "shift_scale"):
            raise ValueError("numeric_strategy must be 'tokenize' or 'shift_scale'")
        self.salt = salt or secrets.token_hex(16)
        self.preserve_dtypes = preserve_dtypes
        self.numeric_strategy = numeric_strategy
        self.datetime_jitter_days = datetime_jitter_days
        self.text_passthrough = text_passthrough
        self.text_uniqueness_threshold = text_uniqueness_threshold
        self.text_min_avg_length = text_min_avg_length
        self.text_min_avg_words = text_min_avg_words
        self.passthrough_cols = set(passthrough_cols or [])
        self.force_mask_cols = set(force_mask_cols or [])
        self.exclude_all_categoricals = exclude_all_categoricals
        self._column_maps: dict = {}
        self._column_kinds: dict = {}
        self._passthrough_reasons: dict = {}
        self._fitted = False

    # ---------- core hashing primitive ----------
    def _hmac(self, col: str, value) -> bytes:
        key = f"{self.salt}::{col}".encode()
        msg = repr(value).encode()  # repr distinguishes 1 from "1"
        return hmac.new(key, msg, hashlib.sha256).digest()

    def _hex_token(self, col: str, value, length: int = 16) -> str:
        return self._hmac(col, value).hex()[:length]

    # ---------- natural-text detector ----------
    def _is_categorical_like(self, series: pd.Series) -> bool:
        """True for pandas Categorical or non-numeric/non-datetime object dtypes."""
        if isinstance(series.dtype, pd.CategoricalDtype):
            return True
        return (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        )

    def _is_natural_text(self, series: pd.Series) -> bool:
        """Heuristic: high uniqueness + long, multi-word values."""
        s = series.dropna().astype(str)
        if len(s) == 0:
            return False
        uniqueness = s.nunique() / len(s)
        if uniqueness < self.text_uniqueness_threshold:
            return False
        avg_len = s.str.len().mean()
        if avg_len < self.text_min_avg_length:
            return False
        avg_words = s.str.split().str.len().mean()
        if avg_words < self.text_min_avg_words:
            return False
        return True

    # ---------- per-dtype mappers ----------
    def _build_string_map(self, col: str, series: pd.Series) -> dict:
        return {v: self._hex_token(col, v) for v in series.dropna().unique()}

    def _build_numeric_map(self, col: str, series: pd.Series) -> dict:
        uniq = series.dropna().unique()
        if self.numeric_strategy == "tokenize":
            return {
                v: int.from_bytes(self._hmac(col, v)[:8], "big") / 2**64
                for v in uniq
            }
        # shift_scale: derive (a, b) deterministically from column + salt
        a_raw = int.from_bytes(self._hmac(col, "__scale__")[:8], "big") / 2**64
        b_raw = int.from_bytes(self._hmac(col, "__shift__")[:8], "big") / 2**64
        a = 0.5 + a_raw  # in [0.5, 1.5) — avoids degenerate scale
        col_range = float(np.nanmax(uniq) - np.nanmin(uniq)) + 1e-9
        b = (b_raw - 0.5) * col_range
        return {v: a * float(v) + b for v in uniq}

    def _build_datetime_map(self, col: str, series: pd.Series) -> dict:
        shift_raw = int.from_bytes(self._hmac(col, "__dt_shift__")[:4], "big")
        shift_days = (shift_raw % (2 * self.datetime_jitter_days)) - self.datetime_jitter_days
        delta = pd.Timedelta(days=shift_days)
        return {v: pd.Timestamp(v) + delta for v in series.dropna().unique()}

    def _build_bool_map(self, col: str) -> dict:
        flip = self._hmac(col, "__bool_flip__")[0] & 1
        return {True: bool(True ^ flip), False: bool(False ^ flip)}

    # ---------- fit / transform ----------
    def fit(self, df: pd.DataFrame) -> "DataMasker":
        self._column_maps.clear()
        self._column_kinds.clear()
        self._passthrough_reasons.clear()
        for col in df.columns:
            s = df[col]

            # ---- precedence (highest first) ----
            # 1. explicit user passthrough list
            if col in self.passthrough_cols:
                self._column_kinds[col] = "passthrough"
                self._column_maps[col] = {}
                self._passthrough_reasons[col] = "user_specified"
                continue

            forced = col in self.force_mask_cols
            categorical_like = self._is_categorical_like(s)

            # 2. exclude_all_categoricals — but force_mask_cols overrides
            if self.exclude_all_categoricals and categorical_like and not forced:
                self._column_kinds[col] = "passthrough"
                self._column_maps[col] = {}
                self._passthrough_reasons[col] = "categorical_excluded"
                continue

            # 3. dtype-based dispatch (with natural-text auto-detect for strings)
            if pd.api.types.is_bool_dtype(s):
                kind, mapping = "bool", self._build_bool_map(col)
            elif pd.api.types.is_datetime64_any_dtype(s):
                kind, mapping = "datetime", self._build_datetime_map(col, s)
            elif pd.api.types.is_numeric_dtype(s):
                kind, mapping = "numeric", self._build_numeric_map(col, s)
            else:
                if (
                    self.text_passthrough
                    and not forced
                    and self._is_natural_text(s)
                ):
                    self._column_kinds[col] = "passthrough"
                    self._column_maps[col] = {}
                    self._passthrough_reasons[col] = "natural_text_detected"
                    continue
                kind, mapping = "string", self._build_string_map(col, s)

            self._column_kinds[col] = kind
            self._column_maps[col] = mapping
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("DataMasker is not fitted. Call fit() or fit_transform() first.")
        out = {}
        for col in df.columns:
            kind = self._column_kinds.get(col)
            if kind == "passthrough":
                out[col] = df[col].copy()
                continue
            mapping = self._column_maps.setdefault(col, {})
            s = df[col]

            def _map_value(v, col=col, kind=kind, mapping=mapping):
                if pd.isna(v):
                    return v
                if v in mapping:
                    return mapping[v]
                # unseen value (new column or value not in training set):
                # extend mapping deterministically so calls stay consistent.
                if kind == "numeric":
                    new = int.from_bytes(self._hmac(col, v)[:8], "big") / 2**64
                elif kind == "datetime":
                    sample = next(iter(mapping), None)
                    delta = mapping[sample] - pd.Timestamp(sample) if sample is not None else pd.Timedelta(0)
                    new = pd.Timestamp(v) + delta
                elif kind == "bool":
                    new = mapping.get(bool(v), v)
                else:
                    new = self._hex_token(col, v)
                mapping[v] = new
                return new

            masked = s.map(_map_value)
            if self.preserve_dtypes and kind == "numeric":
                masked = pd.to_numeric(masked, errors="coerce")
            elif self.preserve_dtypes and kind == "datetime":
                masked = pd.to_datetime(masked, errors="coerce")
            out[col] = masked
        return pd.DataFrame(out, index=df.index)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def report(self) -> pd.DataFrame:
        """Tabular summary of how each column was treated."""
        if not self._fitted:
            raise RuntimeError("Fit the masker first.")
        rows = [
            {
                "column": c,
                "treatment": self._column_kinds[c],
                "passthrough_reason": self._passthrough_reasons.get(c, ""),
            }
            for c in self._column_kinds
        ]
        return pd.DataFrame(rows)


# ---------- quick demo ----------
if __name__ == "__main__":
    df = pd.DataFrame({
        "customer_id":  ["C001", "C002", "C001", "C003", "C002"],
        "name":         ["Alice", "Bob", "Alice", "Carol", "Bob"],
        "segment":      pd.Categorical(["Retail", "Wholesale", "Retail",
                                        "DTC", "Wholesale"]),
        "balance_usd":  [1250.50, 7800.00, 1250.50, 320.10, 7800.00],
        "opened_on":    pd.to_datetime(["2021-04-12", "2019-11-30",
                                        "2021-04-12", "2022-08-05",
                                        "2019-11-30"]),
        "is_active":    [True, False, True, True, False],
        "support_note": [
            "Customer called about a missed wire transfer on Tuesday morning.",
            "Requested statement reissue for the previous billing cycle.",
            "Disputed an ATM withdrawal flagged by fraud monitoring.",
            "Asked to update mailing address after relocation overseas.",
            "Inquired about mortgage refinancing options and rate locks.",
        ],
    })

    print("=== default behaviour (mask everything except natural text) ===")
    masker = DataMasker(numeric_strategy="tokenize")
    masked = masker.fit_transform(df)
    print(masker.report(), "\n")

    print("=== exclude_all_categoricals=True (keep segment + IDs readable) ===")
    masker2 = DataMasker(
        numeric_strategy="tokenize",
        exclude_all_categoricals=True,
        force_mask_cols=["customer_id", "name"],   # still mask explicit PII
    )
    masked2 = masker2.fit_transform(df)
    print(masker2.report(), "\n")
    print(masked2, "\n")

    print("=== custom passthrough list ===")
    masker3 = DataMasker(passthrough_cols=["segment", "is_active"])
    masker3.fit_transform(df)
    print(masker3.report(), "\n")

    # ---- sanity checks on default run ----
    print("== invariants (default run) ==")
    print("row-uniqueness preserved :",
          df.duplicated().sum() == masked.duplicated().sum())
    print("relationship preserved   :",
          masked.loc[df["customer_id"] == "C001", "customer_id"].nunique() == 1)
    print("support_note untouched   :",
          (masked["support_note"] == df["support_note"]).all())
