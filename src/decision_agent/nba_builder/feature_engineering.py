"""
Feature Engineering Pipeline
==============================
Computes all NBA features from raw account data.

Features computed:
  - Delinquency state (bucket, delay, normalised DPD)
  - Payment history (rolling sums, rates, regularity)
  - Contact history (channel attempts, response rates, fatigue score)
  - Offer history (settlement acceptance, plan completion)
  - Capacity / balance features (utilisation, payment capacity)
  - Persona features (one-hot encoded)

Works with pandas DataFrames (no Spark dependency for unit tests).
On Databricks: wrap with .toPandas() / spark.createDataFrame().
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
FATIGUE_WEIGHTS = {
    "line":     0.2,
    "sms":      0.3,
    "voice_ivr":0.5,
    "call":     1.0,
    "email":    0.1,
}
FATIGUE_HALF_LIFE = 14   # days


class FeatureEngineer:
    """
    Computes NBA features from a raw accounts DataFrame.

    Expected input columns (minimum required):
        account_id, days_past_due, balance, business_date

    All other columns are optional — features are computed if source columns
    exist, else filled with sensible defaults.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.feature_cols: List[str] = []

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC
    # ──────────────────────────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features. Returns enriched DataFrame.
        Modifies a copy — original is unchanged.
        """
        out = df.copy()
        out = self._delinquency_features(out)
        out = self._payment_features(out)
        out = self._fatigue_features(out)
        out = self._offer_features(out)
        out = self._balance_features(out)
        out = self._contact_channel_features(out)
        out = self._persona_onehot(out)
        out = self._fill_defaults(out)
        self.feature_cols = self._get_feature_cols(out)
        return out

    def get_feature_columns(self) -> List[str]:
        """Return list of all computed feature column names."""
        return self.feature_cols

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: FEATURE GROUPS
    # ──────────────────────────────────────────────────────────────────────────

    def _delinquency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bucket, delay, normalised DPD."""
        if "days_past_due" in df.columns:
            dpd = df["days_past_due"].clip(lower=0)
            df["dpd_norm"]         = (dpd / 360.0).clip(upper=1.0)
            df["bucket"]           = df.get("bucket", (dpd // 30).clip(upper=5).astype(int))
            df["is_current"]       = (dpd == 0).astype(int)
            df["is_early_bucket"]  = (dpd.between(1, 60)).astype(int)
            df["is_late_bucket"]   = (dpd > 90).astype(int)
            df["log_dpd"]          = np.log1p(dpd)
        return df

    def _payment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Payment history ratios and windows."""
        # MAD coverage
        if "payment_sum_30d" in df.columns and "MAD" in df.columns:
            mad = df["MAD"].replace(0, np.nan)
            df["mad_coverage_30d"] = (df["payment_sum_30d"] / mad).clip(0, 3).fillna(0)

        # Payment regularity (if we have counts across windows)
        for w in [7, 14, 30]:
            col = f"payment_count_{w}d"
            if col in df.columns:
                df[f"has_payment_{w}d"] = (df[col] > 0).astype(int)

        # Days since last payment (normalised)
        if "days_since_last_payment" in df.columns:
            df["days_since_pay_norm"] = (
                df["days_since_last_payment"].clip(0, 180) / 180.0
            )

        # Payment amount as fraction of balance
        if "last_payment_amount" in df.columns and "balance" in df.columns:
            bal = df["balance"].replace(0, np.nan)
            df["last_pay_to_balance"] = (
                df["last_payment_amount"] / bal
            ).clip(0, 2).fillna(0)

        return df

    def _fatigue_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute fatigue score if raw contact counts exist.
        fatigue_score = weighted sum of contacts × decay factor.
        """
        if "fatigue_score" in df.columns:
            return df  # Already computed upstream

        score = pd.Series(np.zeros(len(df)), index=df.index)
        for channel, weight in FATIGUE_WEIGHTS.items():
            for days in [7, 14, 30]:
                col = f"{channel}_attempts_{days}d"
                if col in df.columns:
                    decay = 0.5 ** (days / FATIGUE_HALF_LIFE)
                    score += df[col].fillna(0) * weight * decay

        df["fatigue_score"] = score.clip(0, 1.0).round(4)

        # Derived fatigue flags
        df["high_fatigue"]    = (df["fatigue_score"] >= 0.70).astype(int)
        df["medium_fatigue"]  = (df["fatigue_score"].between(0.40, 0.70)).astype(int)
        df["fresh_account"]   = (df["fatigue_score"] < 0.20).astype(int)
        return df

    def _offer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Settlement / plan offer history."""
        if "settlement_acceptance_rate" in df.columns:
            df["high_settlement_accept"] = (
                df["settlement_acceptance_rate"] > 0.5
            ).astype(int)

        if "ptp_kept_rate_6m" in df.columns:
            df["reliable_ptp"] = (df["ptp_kept_rate_6m"] > 0.7).astype(int)

        if "payment_plan_completion_rate" in df.columns:
            df["completes_plans"] = (
                df["payment_plan_completion_rate"] > 0.6
            ).astype(int)

        return df

    def _balance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Balance-derived features."""
        if "balance" in df.columns:
            bal = df["balance"].clip(lower=0)
            df["log_balance"]    = np.log1p(bal)
            df["balance_tier"]   = pd.cut(
                bal,
                bins=[-1, 999, 4999, 9999, 49999, float("inf")],
                labels=[0, 1, 2, 3, 4],
            ).astype(float).fillna(0)

        if "balance" in df.columns and "credit_limit" in df.columns:
            lim = df["credit_limit"].replace(0, np.nan)
            df["utilization"] = (df["balance"] / lim).clip(0, 1).fillna(0.5)

        return df

    def _contact_channel_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Channel availability and response rates."""
        # Binary availability features (already in schema, just ensure present)
        for flag in ["mobile_present", "email_present", "line_optin", "sms_optin"]:
            if flag in df.columns:
                df[flag] = df[flag].fillna(False).astype(int)

        # Response rate differences (best channel signal)
        if "line_response_rate_90d" in df.columns and "sms_response_rate_90d" in df.columns:
            df["line_better_than_sms"] = (
                df["line_response_rate_90d"] > df["sms_response_rate_90d"]
            ).astype(int)

        # Days since last contact (normalised)
        for ch in ["line", "sms", "call", "email"]:
            col = f"days_since_last_{ch}"
            if col in df.columns:
                df[f"{col}_norm"] = (df[col].clip(0, 90) / 90.0)

        # Total contacts last 7 days (normalised)
        if "total_contacts_7d" in df.columns:
            df["contacts_7d_norm"] = (df["total_contacts_7d"].clip(0, 10) / 10.0)

        return df

    def _persona_onehot(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode persona segment."""
        personas = [
            "high_value_cooperative",
            "high_value_unresponsive",
            "medium_value_willing",
            "medium_value_struggling",
            "low_value_chronic",
            "promise_keeper",
            "standard",
        ]
        if "persona_segment" in df.columns or "persona" in df.columns:
            col = "persona_segment" if "persona_segment" in df.columns else "persona"
            for p in personas:
                df[f"persona_{p}"] = (df[col] == p).astype(int)
        return df

    def _fill_defaults(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill any remaining NaN numeric features with 0."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        return df

    def _get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        """Return feature columns (exclude meta columns)."""
        exclude = {
            "account_id", "customer_id", "business_date", "due_date",
            "last_payment_date", "last_settlement_offer_date", "ptp_date",
            "created_timestamp", "pipeline_version",
            # raw outcome cols excluded from features
            "outcome_pay_any_7d", "outcome_pay_any_14d", "outcome_pay_any_30d",
            "outcome_pay_amount_7d", "outcome_pay_amount_14d",
            "outcome_bucket_7d", "outcome_bucket_change_7d",
        }
        return [
            c for c in df.columns
            if c not in exclude
            and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.int8, bool]
        ]
