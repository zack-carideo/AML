"""
data_masker.py — Generalizable data masking utility for tabular data.

Supports both Polars and Pandas DataFrames (default: Polars).

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
from typing import Optional, Union

import numpy as np

_VALID_ENGINES = ("polars", "pandas")


class DataMasker:
    """Deterministic, dtype-aware tabular data masker.

    Parameters
    ----------
    engine : {"polars", "pandas"}, default "polars"
        Backend engine for DataFrame operations.
    salt : str, optional
        Hex string used to key the HMAC. If omitted, a cryptographically
        random 128-bit salt is generated.
    preserve_dtypes : bool, default True
        If True, masked numerics stay numeric and masked datetimes stay
        datetime. If False, every column becomes a hashed string token.
    numeric_strategy : {"tokenize", "shift_scale"}, default "tokenize"
        "tokenize"   — map each unique value to a deterministic float
                       in [0, 1) via HMAC. Destroys order/correlation
                       but guarantees 1:1.
        "shift_scale" — apply a deterministic affine transform a*x + b
                       per column. Preserves order and correlations.
    datetime_jitter_days : int, default 3650
        Half-range of the per-column datetime shift, in days.
    text_passthrough : bool, default True
        If True, auto-detect natural-text columns and leave them unchanged.
    text_uniqueness_threshold : float, default 0.85
        Min ratio of unique non-null values for a column to qualify
        as natural text.
    text_min_avg_length : int, default 20
        Min average character length for a column to qualify.
    text_min_avg_words : float, default 3.0
        Min average word count.
    passthrough_cols : list[str], optional
        Explicit list of columns to leave unchanged.
    force_mask_cols : list[str], optional
        Explicit list of columns to always mask.
    exclude_all_categoricals : bool, default False
        If True, every categorical-like column is left unchanged.
    """

    def __init__(
        self,
        engine: str = "polars",
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
        if engine not in _VALID_ENGINES:
            raise ValueError(f"engine must be one of {_VALID_ENGINES}, got '{engine}'")
        if numeric_strategy not in ("tokenize", "shift_scale"):
            raise ValueError("numeric_strategy must be 'tokenize' or 'shift_scale'")
        self.engine = engine
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
        msg = repr(value).encode()
        return hmac.new(key, msg, hashlib.sha256).digest()

    def _hex_token(self, col: str, value, length: int = 16) -> str:
        return self._hmac(col, value).hex()[:length]

    # ========== PANDAS BACKEND ==========
    def _pd_is_categorical_like(self, series) -> bool:
        import pandas as pd
        if isinstance(series.dtype, pd.CategoricalDtype):
            return True
        return (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        )

    def _pd_is_natural_text(self, series) -> bool:
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

    def _pd_build_string_map(self, col: str, series) -> dict:
        return {v: self._hex_token(col, v) for v in series.dropna().unique()}

    def _pd_build_numeric_map(self, col: str, series) -> dict:
        uniq = series.dropna().unique()
        if self.numeric_strategy == "tokenize":
            return {
                v: int.from_bytes(self._hmac(col, v)[:8], "big") / 2**64
                for v in uniq
            }
        a_raw = int.from_bytes(self._hmac(col, "__scale__")[:8], "big") / 2**64
        b_raw = int.from_bytes(self._hmac(col, "__shift__")[:8], "big") / 2**64
        a = 0.5 + a_raw
        col_range = float(np.nanmax(uniq) - np.nanmin(uniq)) + 1e-9
        b = (b_raw - 0.5) * col_range
        return {v: a * float(v) + b for v in uniq}

    def _pd_build_datetime_map(self, col: str, series) -> dict:
        import pandas as pd
        shift_raw = int.from_bytes(self._hmac(col, "__dt_shift__")[:4], "big")
        shift_days = (shift_raw % (2 * self.datetime_jitter_days)) - self.datetime_jitter_days
        delta = pd.Timedelta(days=shift_days)
        return {v: pd.Timestamp(v) + delta for v in series.dropna().unique()}

    def _pd_build_bool_map(self, col: str) -> dict:
        flip = self._hmac(col, "__bool_flip__")[0] & 1
        return {True: bool(True ^ flip), False: bool(False ^ flip)}

    def _pd_fit(self, df) -> "DataMasker":
        import pandas as pd
        self._column_maps.clear()
        self._column_kinds.clear()
        self._passthrough_reasons.clear()
        for col in df.columns:
            s = df[col]
            if col in self.passthrough_cols:
                self._column_kinds[col] = "passthrough"
                self._column_maps[col] = {}
                self._passthrough_reasons[col] = "user_specified"
                continue

            forced = col in self.force_mask_cols
            categorical_like = self._pd_is_categorical_like(s)

            if self.exclude_all_categoricals and categorical_like and not forced:
                self._column_kinds[col] = "passthrough"
                self._column_maps[col] = {}
                self._passthrough_reasons[col] = "categorical_excluded"
                continue

            if pd.api.types.is_bool_dtype(s):
                kind, mapping = "bool", self._pd_build_bool_map(col)
            elif pd.api.types.is_datetime64_any_dtype(s):
                kind, mapping = "datetime", self._pd_build_datetime_map(col, s)
            elif pd.api.types.is_numeric_dtype(s):
                kind, mapping = "numeric", self._pd_build_numeric_map(col, s)
            else:
                if self.text_passthrough and not forced and self._pd_is_natural_text(s):
                    self._column_kinds[col] = "passthrough"
                    self._column_maps[col] = {}
                    self._passthrough_reasons[col] = "natural_text_detected"
                    continue
                kind, mapping = "string", self._pd_build_string_map(col, s)

            self._column_kinds[col] = kind
            self._column_maps[col] = mapping
        self._fitted = True
        return self

    def _pd_transform(self, df):
        import pandas as pd
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

    # ========== POLARS BACKEND ==========
    def _pl_is_categorical_like(self, dtype) -> bool:
        import polars as pl
        return dtype in (pl.Categorical, pl.Utf8, pl.String) or dtype == pl.Enum

    def _pl_is_natural_text(self, series) -> bool:
        s = series.drop_nulls().cast(str)
        if len(s) == 0:
            return False
        uniqueness = s.n_unique() / len(s)
        if uniqueness < self.text_uniqueness_threshold:
            return False
        avg_len = s.str.len_chars().mean()
        if avg_len < self.text_min_avg_length:
            return False
        avg_words = s.str.split(" ").list.len().mean()
        if avg_words < self.text_min_avg_words:
            return False
        return True

    def _pl_build_string_map(self, col: str, series) -> dict:
        return {v: self._hex_token(col, v) for v in series.drop_nulls().unique().to_list()}

    def _pl_build_numeric_map(self, col: str, series) -> dict:
        uniq = series.drop_nulls().unique().to_list()
        if self.numeric_strategy == "tokenize":
            return {
                v: int.from_bytes(self._hmac(col, v)[:8], "big") / 2**64
                for v in uniq
            }
        a_raw = int.from_bytes(self._hmac(col, "__scale__")[:8], "big") / 2**64
        b_raw = int.from_bytes(self._hmac(col, "__shift__")[:8], "big") / 2**64
        a = 0.5 + a_raw
        col_range = float(max(uniq) - min(uniq)) + 1e-9
        b = (b_raw - 0.5) * col_range
        return {v: a * float(v) + b for v in uniq}

    def _pl_build_datetime_map(self, col: str, series) -> dict:
        from datetime import timedelta
        shift_raw = int.from_bytes(self._hmac(col, "__dt_shift__")[:4], "big")
        shift_days = (shift_raw % (2 * self.datetime_jitter_days)) - self.datetime_jitter_days
        delta = timedelta(days=shift_days)
        return {v: v + delta for v in series.drop_nulls().unique().to_list()}

    def _pl_build_bool_map(self, col: str) -> dict:
        flip = self._hmac(col, "__bool_flip__")[0] & 1
        return {True: bool(True ^ flip), False: bool(False ^ flip)}

    def _pl_fit(self, df) -> "DataMasker":
        import polars as pl
        self._column_maps.clear()
        self._column_kinds.clear()
        self._passthrough_reasons.clear()
        for col in df.columns:
            s = df[col]
            dtype = s.dtype

            if col in self.passthrough_cols:
                self._column_kinds[col] = "passthrough"
                self._column_maps[col] = {}
                self._passthrough_reasons[col] = "user_specified"
                continue

            forced = col in self.force_mask_cols
            categorical_like = self._pl_is_categorical_like(dtype)

            if self.exclude_all_categoricals and categorical_like and not forced:
                self._column_kinds[col] = "passthrough"
                self._column_maps[col] = {}
                self._passthrough_reasons[col] = "categorical_excluded"
                continue

            if dtype == pl.Boolean:
                kind, mapping = "bool", self._pl_build_bool_map(col)
            elif dtype in (pl.Date, pl.Datetime) or dtype.is_temporal():
                kind, mapping = "datetime", self._pl_build_datetime_map(col, s)
            elif dtype.is_numeric():
                kind, mapping = "numeric", self._pl_build_numeric_map(col, s)
            else:
                if self.text_passthrough and not forced and self._pl_is_natural_text(s):
                    self._column_kinds[col] = "passthrough"
                    self._column_maps[col] = {}
                    self._passthrough_reasons[col] = "natural_text_detected"
                    continue
                kind, mapping = "string", self._pl_build_string_map(col, s)

            self._column_kinds[col] = kind
            self._column_maps[col] = mapping
        self._fitted = True
        return self

    def _pl_transform(self, df):
        import polars as pl
        from datetime import timedelta
        if not self._fitted:
            raise RuntimeError("DataMasker is not fitted. Call fit() or fit_transform() first.")
        out = {}
        for col in df.columns:
            kind = self._column_kinds.get(col)
            if kind == "passthrough":
                out[col] = df[col]
                continue
            mapping = self._column_maps.setdefault(col, {})
            s = df[col]

            def _map_value(v, col=col, kind=kind, mapping=mapping):
                if v is None:
                    return v
                if v in mapping:
                    return mapping[v]
                if kind == "numeric":
                    new = int.from_bytes(self._hmac(col, v)[:8], "big") / 2**64
                elif kind == "datetime":
                    sample = next(iter(mapping), None)
                    delta = (mapping[sample] - sample) if sample is not None else timedelta(0)
                    new = v + delta
                elif kind == "bool":
                    new = mapping.get(bool(v), v)
                else:
                    new = self._hex_token(col, v)
                mapping[v] = new
                return new

            mapped_values = [_map_value(v) for v in s.to_list()]

            if self.preserve_dtypes and kind == "numeric":
                out[col] = pl.Series(col, mapped_values, dtype=pl.Float64)
            elif self.preserve_dtypes and kind == "datetime":
                out[col] = pl.Series(col, mapped_values, dtype=s.dtype)
            elif kind == "bool":
                out[col] = pl.Series(col, mapped_values, dtype=pl.Boolean)
            else:
                out[col] = pl.Series(col, mapped_values, dtype=pl.String)
        return pl.DataFrame(out)

    # ---------- public API ----------
    def fit(self, df) -> "DataMasker":
        if self.engine == "polars":
            return self._pl_fit(df)
        return self._pd_fit(df)

    def transform(self, df):
        if self.engine == "polars":
            return self._pl_transform(df)
        return self._pd_transform(df)

    def fit_transform(self, df):
        return self.fit(df).transform(df)

    def report(self):
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
        if self.engine == "polars":
            import polars as pl
            return pl.DataFrame(rows)
        import pandas as pd
        return pd.DataFrame(rows)


# ---------- quick demo ----------
if __name__ == "__main__":
    import polars as pl
    import pandas as pd

    # --- Polars demo (default) ---
    print("=" * 60)
    print("POLARS ENGINE (default)")
    print("=" * 60)
    df_pl = pl.DataFrame({
        "customer_id":  ["C001", "C002", "C001", "C003", "C002"],
        "name":         ["Alice", "Bob", "Alice", "Carol", "Bob"],
        "segment":      ["Retail", "Wholesale", "Retail", "DTC", "Wholesale"],
        "balance_usd":  [1250.50, 7800.00, 1250.50, 320.10, 7800.00],
        "opened_on":    pl.Series(["2021-04-12", "2019-11-30",
                                   "2021-04-12", "2022-08-05",
                                   "2019-11-30"]).str.to_date(),
        "is_active":    [True, False, True, True, False],
        "support_note": [
            "Customer called about a missed wire transfer on Tuesday morning.",
            "Requested statement reissue for the previous billing cycle.",
            "Disputed an ATM withdrawal flagged by fraud monitoring.",
            "Asked to update mailing address after relocation overseas.",
            "Inquired about mortgage refinancing options and rate locks.",
        ],
    })

    print("\n=== default behaviour (mask everything except natural text) ===")
    masker = DataMasker(engine="polars", numeric_strategy="tokenize")
    masked = masker.fit_transform(df_pl)
    print(masker.report(), "\n")

    print("=== exclude_all_categoricals=True ===")
    masker2 = DataMasker(
        engine="polars",
        numeric_strategy="tokenize",
        exclude_all_categoricals=True,
        force_mask_cols=["customer_id", "name"],
    )
    masked2 = masker2.fit_transform(df_pl)
    print(masker2.report(), "\n")
    print(masked2, "\n")

    # --- Pandas demo ---
    print("=" * 60)
    print("PANDAS ENGINE")
    print("=" * 60)
    df_pd = pd.DataFrame({
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

    print("\n=== default behaviour (mask everything except natural text) ===")
    masker3 = DataMasker(engine="pandas", numeric_strategy="tokenize")
    masked3 = masker3.fit_transform(df_pd)
    print(masker3.report(), "\n")
    print(masked3, "\n")
