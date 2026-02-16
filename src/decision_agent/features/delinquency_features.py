"""
Delinquency Feature Engineering
=================================
Two complementary feature sets built from real schema fields:

  1. DLNQ_HIST string parser  (crcard_card_dly.DLNQ_HIST)
     Monthly DPD bucket history encoded as a string.
     Format: each character = one month, rightmost = most recent month.
     Bucket encoding (standard card system):
       '0' = current (0 DPD)
       '1' = 1–30 DPD
       '2' = 31–60 DPD
       '3' = 61–90 DPD
       '4' = 91–120 DPD
       '5' = 121–150 DPD
       '6' = 151–180 DPD
       '7' = 181+ DPD / pre-chargeoff
       'X' = charged off
       'W' = written off
       '-' or ' ' = no data / account not open
     Length varies (typically 12–36 months).

  2. Arrears progression features  (AMT_IN_ARRS_1_PRIOD … 9_PRIOD)
     Monthly arrears amounts for the last 9 periods.
     Period 1 = most recent, period 9 = oldest.
     Used for: trend direction, acceleration, stability.

Features produced (all numeric, pandas-native — no Spark dependency):
  From DLNQ_HIST:
    dlnq_max_bucket_12m       : worst bucket in last 12 months (0–7)
    dlnq_max_bucket_24m       : worst bucket in last 24 months
    dlnq_times_30plus_12m     : months at 1+ bucket (30+ DPD) in last 12m
    dlnq_times_60plus_12m     : months at 2+ bucket (60+ DPD) in last 12m
    dlnq_times_90plus_12m     : months at 3+ bucket (90+ DPD) in last 12m
    dlnq_current_streak       : consecutive non-zero months from most recent
    dlnq_cure_count_12m       : times dropped back to 0 from non-zero (cures)
    dlnq_worsening_count_12m  : times bucket increased month-on-month
    dlnq_improving_count_12m  : times bucket decreased month-on-month
    dlnq_months_since_worst   : months since worst-ever bucket in history
    dlnq_pct_delinquent_12m   : fraction of last 12m in any delinquency
    dlnq_ever_charged_off     : 1 if 'X' or 'W' appears in history
    dlnq_history_length       : number of valid (non-null) months in string

  From AMT_IN_ARRS_*_PRIOD:
    arrs_latest               : most recent arrears amount (period 1)
    arrs_max_9m               : maximum arrears in last 9 periods
    arrs_mean_9m              : average arrears over 9 periods
    arrs_trend_3m             : period_1 - period_3 (+ = worsening)
    arrs_trend_6m             : period_1 - period_6 (+ = worsening)
    arrs_acceleration         : (p1-p3) - (p4-p6)  (+ = accelerating)
    arrs_nonzero_count_9m     : number of periods with arrears > 0
    arrs_pct_nonzero_9m       : fraction of periods with arrears > 0
    arrs_volatility_9m        : std dev of arrears amounts
    arrs_recovered_flag       : 1 if latest=0 but had arrears in prior periods
    arrs_peak_to_now_ratio    : arrs_latest / arrs_max_9m (1=at peak, 0=recovered)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Bucket character → numeric DPD bucket
BUCKET_MAP: Dict[str, int] = {
    "0": 0, "1": 1, "2": 2, "3": 3,
    "4": 4, "5": 5, "6": 6, "7": 7,
    "X": 8,   # charged off — highest severity
    "W": 8,   # written off — same severity
}
NULL_CHARS = {"-", " ", "N", "n", ""}

# Internal arrears column names (post schema_mapper rename)
ARRS_COLS = [f"arrs_period_{i}" for i in range(1, 10)]  # period_1 = most recent


# ─────────────────────────────────────────────────────────────────────────────
# DLNQ_HIST PARSER
# ─────────────────────────────────────────────────────────────────────────────

class DlnqHistParser:
    """
    Parses DLNQ_HIST string into a list of integer buckets
    and extracts delinquency features.

    Convention: index 0 = most recent month.
    """

    def parse(self, hist: Optional[str]) -> List[int]:
        """
        Convert DLNQ_HIST string to list[int], most recent first.
        Returns [] if string is null or empty.
        """
        if hist is None or not isinstance(hist, str) or hist.strip() == "":
            return []

        buckets = []
        for ch in reversed(hist.strip()):   # reverse: rightmost = most recent
            if ch in NULL_CHARS:
                buckets.append(-1)           # -1 = missing / account not open
            else:
                buckets.append(BUCKET_MAP.get(ch.upper(), 0))

        return buckets

    def extract_features(self, hist: Optional[str]) -> Dict[str, float]:
        """
        Extract all DLNQ_HIST features for one account row.
        Returns dict of feature_name → float value.
        Gracefully returns zeros on missing/malformed history.
        """
        buckets = self.parse(hist)
        valid   = [b for b in buckets if b >= 0]   # drop -1 (no data months)

        feats: Dict[str, float] = {}

        # ── history length ────────────────────────────────────────────────────
        feats["dlnq_history_length"] = float(len(valid))

        if not valid:
            return _zero_dlnq_features(feats)

        # ── ever charged off ──────────────────────────────────────────────────
        feats["dlnq_ever_charged_off"] = float(any(b >= 8 for b in valid))

        # ── windowed features (12m = first 12 valid months) ───────────────────
        last12 = valid[:12]
        last24 = valid[:24]

        feats["dlnq_max_bucket_12m"]   = float(max(last12)) if last12 else 0.0
        feats["dlnq_max_bucket_24m"]   = float(max(last24)) if last24 else 0.0

        feats["dlnq_times_30plus_12m"] = float(sum(1 for b in last12 if b >= 1))
        feats["dlnq_times_60plus_12m"] = float(sum(1 for b in last12 if b >= 2))
        feats["dlnq_times_90plus_12m"] = float(sum(1 for b in last12 if b >= 3))

        feats["dlnq_pct_delinquent_12m"] = (
            feats["dlnq_times_30plus_12m"] / len(last12)
            if last12 else 0.0
        )

        # ── current streak (consecutive non-zero from most recent) ────────────
        streak = 0
        for b in valid:
            if b > 0:
                streak += 1
            else:
                break
        feats["dlnq_current_streak"] = float(streak)

        # ── cure count (non-zero → 0 transitions in last 12m) ────────────────
        cures = 0
        for i in range(1, len(last12)):
            if last12[i - 1] == 0 and last12[i] > 0:   # remember: index 0 = most recent
                cures += 1                               # so i-1 is more recent than i
        feats["dlnq_cure_count_12m"] = float(cures)

        # ── worsening / improving transitions in last 12m ─────────────────────
        worsening = 0
        improving = 0
        for i in range(1, len(last12)):
            delta = last12[i - 1] - last12[i]  # positive = more recent is worse
            if delta > 0:
                worsening += 1
            elif delta < 0:
                improving += 1
        feats["dlnq_worsening_count_12m"] = float(worsening)
        feats["dlnq_improving_count_12m"] = float(improving)

        # ── months since worst bucket in full history ─────────────────────────
        if valid:
            worst_val = max(valid)
            # find first occurrence from left (= most recent)
            months_since = next(
                (i for i, b in enumerate(valid) if b == worst_val), len(valid)
            )
            feats["dlnq_months_since_worst"] = float(months_since)
        else:
            feats["dlnq_months_since_worst"] = 0.0

        return feats


def _zero_dlnq_features(feats: Dict) -> Dict[str, float]:
    """Fill all DLNQ_HIST features with 0 for accounts with no history."""
    defaults = {
        "dlnq_ever_charged_off":   0.0,
        "dlnq_max_bucket_12m":     0.0,
        "dlnq_max_bucket_24m":     0.0,
        "dlnq_times_30plus_12m":   0.0,
        "dlnq_times_60plus_12m":   0.0,
        "dlnq_times_90plus_12m":   0.0,
        "dlnq_pct_delinquent_12m": 0.0,
        "dlnq_current_streak":     0.0,
        "dlnq_cure_count_12m":     0.0,
        "dlnq_worsening_count_12m":0.0,
        "dlnq_improving_count_12m":0.0,
        "dlnq_months_since_worst": 0.0,
    }
    feats.update(defaults)
    return feats


# ─────────────────────────────────────────────────────────────────────────────
# ARREARS PROGRESSION FEATURES
# ─────────────────────────────────────────────────────────────────────────────

class ArrearsFeatureBuilder:
    """
    Builds progression and trend features from AMT_IN_ARRS_*_PRIOD columns.
    Period 1 = most recent, period 9 = oldest.
    """

    def extract_features(self, account: pd.Series) -> Dict[str, float]:
        """
        Extract arrears features from a single account row.
        Uses internal names (post schema_mapper rename).
        """
        # Pull 9 periods, coerce to float, None → NaN
        periods: List[float] = []
        for col in ARRS_COLS:
            val = account.get(col)
            try:
                periods.append(float(val) if val is not None and not _is_nan(val) else np.nan)
            except (TypeError, ValueError):
                periods.append(np.nan)

        valid_periods = [p for p in periods if not np.isnan(p)]
        n_valid = len(valid_periods)

        feats: Dict[str, float] = {}

        if n_valid == 0:
            return _zero_arrs_features()

        p = [p if not np.isnan(p) else 0.0 for p in periods]  # fill NaN with 0 for indexing

        # ── basic stats ───────────────────────────────────────────────────────
        feats["arrs_latest"]     = p[0]
        feats["arrs_max_9m"]     = float(max(valid_periods))
        feats["arrs_mean_9m"]    = float(np.mean(valid_periods))
        feats["arrs_volatility_9m"] = float(np.std(valid_periods)) if n_valid > 1 else 0.0

        # ── non-zero counts ───────────────────────────────────────────────────
        feats["arrs_nonzero_count_9m"] = float(sum(1 for v in valid_periods if v > 0))
        feats["arrs_pct_nonzero_9m"]   = feats["arrs_nonzero_count_9m"] / n_valid

        # ── trend: period_1 - period_3  (positive = worsening recently) ───────
        # Both periods must be available for a meaningful trend
        if not np.isnan(periods[0]) and n_valid >= 3:
            p3 = next((periods[i] for i in range(2, 4) if not np.isnan(periods[i])), None)
            feats["arrs_trend_3m"] = float(p[0] - p3) if p3 is not None else 0.0
        else:
            feats["arrs_trend_3m"] = 0.0

        if not np.isnan(periods[0]) and n_valid >= 6:
            p6 = next((periods[i] for i in range(5, 7) if not np.isnan(periods[i])), None)
            feats["arrs_trend_6m"] = float(p[0] - p6) if p6 is not None else 0.0
        else:
            feats["arrs_trend_6m"] = 0.0

        # ── acceleration: recent trend vs older trend ─────────────────────────
        # (p1-p3) - (p4-p6): positive = trend is accelerating (getting worse faster)
        if n_valid >= 6:
            recent_trend = p[0] - p[2]                          # last 3m change
            older_trend  = p[3] - p[5]                          # 3m-6m change
            feats["arrs_acceleration"] = float(recent_trend - older_trend)
        else:
            feats["arrs_acceleration"] = 0.0

        # ── recovery flag: currently 0 but was in arrears ─────────────────────
        feats["arrs_recovered_flag"] = float(
            p[0] == 0.0 and any(v > 0 for v in valid_periods[1:])
        )

        # ── peak-to-now ratio: 1 = at peak, 0 = fully recovered ──────────────
        max_arrs = feats["arrs_max_9m"]
        feats["arrs_peak_to_now_ratio"] = (
            float(p[0] / max_arrs) if max_arrs > 0 else 0.0
        )

        return feats


def _zero_arrs_features() -> Dict[str, float]:
    return {
        "arrs_latest":           0.0,
        "arrs_max_9m":           0.0,
        "arrs_mean_9m":          0.0,
        "arrs_volatility_9m":    0.0,
        "arrs_nonzero_count_9m": 0.0,
        "arrs_pct_nonzero_9m":   0.0,
        "arrs_trend_3m":         0.0,
        "arrs_trend_6m":         0.0,
        "arrs_acceleration":     0.0,
        "arrs_recovered_flag":   0.0,
        "arrs_peak_to_now_ratio":0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED DELINQUENCY FEATURE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

class DelinquencyFeatureBuilder:
    """
    Combines DLNQ_HIST and AMT_IN_ARRS_* features into one feature dict.

    Usage (single row):
        builder = DelinquencyFeatureBuilder()
        features = builder.extract(account_row)

    Usage (DataFrame):
        features_df = builder.extract_batch(df)
    """

    def __init__(self):
        self._dlnq  = DlnqHistParser()
        self._arrs  = ArrearsFeatureBuilder()

    def extract(self, account: pd.Series) -> Dict[str, float]:
        """Extract all delinquency features for one account row."""
        hist = account.get("delinquency_history")     # internal name (post mapper)

        dlnq_feats = self._dlnq.extract_features(hist)
        arrs_feats = self._arrs.extract_features(account)

        # ── Cross-source consistency check ────────────────────────────────────
        # If DLNQ_HIST says bucket 3+ but arrears = 0, flag it
        # (may indicate data lag or partial data)
        consistency_flag = 0.0
        if dlnq_feats.get("dlnq_max_bucket_12m", 0) >= 3:
            if arrs_feats.get("arrs_latest", 0) == 0 and arrs_feats.get("arrs_max_9m", 0) == 0:
                consistency_flag = 1.0
                logger.debug(
                    "Account %s: DLNQ_HIST shows 90+ DPD but arrears amounts are zero — "
                    "possible data lag or different product scope",
                    account.get("account_id", "unknown"),
                )

        combined = {**dlnq_feats, **arrs_feats, "dlnq_arrs_inconsistency_flag": consistency_flag}
        return combined

    def extract_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract delinquency features for all rows in a DataFrame.
        Returns DataFrame with one feature column per feature.
        """
        records = []
        for _, row in df.iterrows():
            feats = self.extract(row)
            feats["account_id"] = row.get("account_id", "unknown")
            records.append(feats)

        feat_df = pd.DataFrame(records)

        # Put account_id first
        cols = ["account_id"] + [c for c in feat_df.columns if c != "account_id"]
        return feat_df[cols]

    @staticmethod
    def feature_names() -> List[str]:
        """Return the complete list of feature names produced."""
        dlnq = [
            "dlnq_history_length",
            "dlnq_ever_charged_off",
            "dlnq_max_bucket_12m",
            "dlnq_max_bucket_24m",
            "dlnq_times_30plus_12m",
            "dlnq_times_60plus_12m",
            "dlnq_times_90plus_12m",
            "dlnq_pct_delinquent_12m",
            "dlnq_current_streak",
            "dlnq_cure_count_12m",
            "dlnq_worsening_count_12m",
            "dlnq_improving_count_12m",
            "dlnq_months_since_worst",
        ]
        arrs = [
            "arrs_latest",
            "arrs_max_9m",
            "arrs_mean_9m",
            "arrs_volatility_9m",
            "arrs_nonzero_count_9m",
            "arrs_pct_nonzero_9m",
            "arrs_trend_3m",
            "arrs_trend_6m",
            "arrs_acceleration",
            "arrs_recovered_flag",
            "arrs_peak_to_now_ratio",
        ]
        return dlnq + arrs + ["dlnq_arrs_inconsistency_flag"]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _is_nan(val) -> bool:
    try:
        return np.isnan(val)
    except (TypeError, ValueError):
        return False


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    builder = DelinquencyFeatureBuilder()

    accounts = pd.DataFrame([
        {
            # ACC001: worsening trend — climbing buckets, still in arrears
            "account_id": "ACC001",
            "delinquency_history": "000000001123",  # 12 months, recently 1→2→3
            "arrs_period_1": 15000, "arrs_period_2": 12000, "arrs_period_3": 9000,
            "arrs_period_4":  6000, "arrs_period_5":  3000, "arrs_period_6": 0,
            "arrs_period_7":     0, "arrs_period_8":     0, "arrs_period_9": 0,
        },
        {
            # ACC002: recovered — was in arrears, now current
            "account_id": "ACC002",
            "delinquency_history": "000111100000",
            "arrs_period_1": 0, "arrs_period_2": 0,    "arrs_period_3": 0,
            "arrs_period_4": 0, "arrs_period_5": 8000, "arrs_period_6": 8000,
            "arrs_period_7": 5000, "arrs_period_8": 0, "arrs_period_9": 0,
        },
        {
            # ACC003: charged off — X in history, long streak
            "account_id": "ACC003",
            "delinquency_history": "000000123456X",
            "arrs_period_1": 50000, "arrs_period_2": 48000, "arrs_period_3": 45000,
            "arrs_period_4": 40000, "arrs_period_5": 35000, "arrs_period_6": 28000,
            "arrs_period_7": 20000, "arrs_period_8": 10000, "arrs_period_9": 0,
        },
        {
            # ACC004: minimal data — no history string, no arrears
            "account_id": "ACC004",
            "delinquency_history": None,
            "arrs_period_1": None, "arrs_period_2": None, "arrs_period_3": None,
            "arrs_period_4": None, "arrs_period_5": None, "arrs_period_6": None,
            "arrs_period_7": None, "arrs_period_8": None, "arrs_period_9": None,
        },
        {
            # ACC005: inconsistency — DLNQ_HIST shows 90+ DPD, arrears = 0
            "account_id": "ACC005",
            "delinquency_history": "000000003330",
            "arrs_period_1": 0, "arrs_period_2": 0, "arrs_period_3": 0,
            "arrs_period_4": 0, "arrs_period_5": 0, "arrs_period_6": 0,
            "arrs_period_7": 0, "arrs_period_8": 0, "arrs_period_9": 0,
        },
    ])

    feats_df = builder.extract_batch(accounts)

    print("── ALL FEATURES ──")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(feats_df.T.to_string())

    print(f"\nTotal features: {len(builder.feature_names())}")
    print("\n── KEY SIGNALS PER ACCOUNT ──")
    key = [
        "account_id",
        "dlnq_max_bucket_12m", "dlnq_current_streak", "dlnq_worsening_count_12m",
        "dlnq_cure_count_12m", "dlnq_ever_charged_off",
        "arrs_trend_3m", "arrs_trend_6m", "arrs_acceleration",
        "arrs_recovered_flag", "arrs_peak_to_now_ratio",
        "dlnq_arrs_inconsistency_flag",
    ]
    print(feats_df[key].to_string(index=False))
