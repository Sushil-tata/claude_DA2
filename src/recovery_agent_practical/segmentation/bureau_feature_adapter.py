"""
Bureau Feature Adapter
======================
Maps bureau_stage_dynamics.py PySpark output into the Pandas feature schema
expected by PersonaBuilder.

This adapter is the ONLY authorised bridge between the PySpark feature factory
and the Pandas-based recovery pipeline. It must be deterministic and
reproducible: the same input Parquet always produces the same output.

Column mapping (bureau_stage_dynamics → PersonaBuilder / SEGMENTATION_FEATURES):

  bureau_stage_dynamics column           PersonaBuilder field
  ─────────────────────────────────────────────────────────────────────────────
  ref_no                               → account_id                  (key rename)
  dpd_shape_type                       → dpd_shape_type              (direct)
  dpd_slope_12m                        → dpd_slope_12m               (direct)
  npl_stickiness_index                 → stage_stickiness_score      (rename)
  cure_to_redefault_ratio              → re_default_rate_rolling_24m (rename)
  stage_reversal_rate                  → cure_rate_sm_to_lower       (rename)
  avg_payment_effort_ratio_npl         → payment_effort_ratio_npl    (direct)
  cure_count_lifetime  (*)             → cure_count_24m              (APPROX)
  re_default_count_lifetime (*)        → re_default_count_24m        (APPROX)
  TIME_SINCE_LAST_PAY_OVERALL          → months_dormant              (rename)
  months_observed_total                → bureau_months_on_book       (rename)
  dpd_time_above_90_24m / 24           → pct_months_npl_24m          (derived)
  max_stage_ever_reached (int 0–5) (**)→ worst_stage_ever            (decoded)
  ─────────────────────────────────────────────────────────────────────────────

(*) KNOWN GAP: bureau_stage_dynamics produces only LIFETIME cure/re-default
    counts. A 24-month windowed version does not exist in the feature factory.
    The adapter uses lifetime counts as proxies and sets `_is_approx=True` in
    data quality metadata. This degrades segmentation quality for accounts with
    long history — the impact is that chronic re-defaulters from years ago may
    be mis-classified. Acceptable for v1 integration; fix by adding a 24m-
    windowed `build_cure_redefault_dynamics()` variant in bureau_stage_dynamics.

(**) max_stage_ever_reached is an integer ordinal (0=CURRENT … 5=CO_DEEP).
     Decoded using ORDINAL_TO_STAGE_MAP.

CRITICAL FEATURES: if any of these are NULL for >CRITICAL_NULL_PCT of rows,
the adapter raises BureauFeatureAdapterError (fail-fast):
  dpd_shape_type, dpd_slope_12m, payment_effort_ratio_npl

SOFT FEATURES: these are allowed to be NULL (missing bureau sub-module output),
filled with documented defaults:
  stage_stickiness_score → 0.5 (no stickiness information = neutral)
  re_default_rate_rolling_24m → 0.0 (no re-default observed)
  cure_rate_sm_to_lower → 0.0 (no cure data)
  cure_count_24m → 0
  re_default_count_24m → 0
  months_dormant → 0
  pct_months_npl_24m → 0.0
  worst_stage_ever → "CURRENT" (conservative: assume no bad history)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Fail-fast if critical features exceed this NULL percentage
CRITICAL_NULL_PCT = 0.30  # 30% null threshold

# Ordinal to stage label (matches bureau_stage_dynamics.py STAGE_DEF)
ORDINAL_TO_STAGE_MAP: Dict[int, str] = {
    0: "CURRENT",
    1: "X",
    2: "SM",
    3: "NPL",
    4: "CO",
    5: "CO_DEEP",
}

# Critical features — high null rates trigger fail-fast
CRITICAL_FEATURES = ["dpd_shape_type", "dpd_slope_12m", "payment_effort_ratio_npl"]

# Soft features — allowed to be null, filled with defaults
SOFT_FEATURE_DEFAULTS: Dict[str, object] = {
    "stage_stickiness_score":     0.5,   # neutral — no stickiness info
    "re_default_rate_rolling_24m": 0.0,  # no re-default observed
    "cure_rate_sm_to_lower":      0.0,   # no cure data
    "cure_count_24m":             0,     # lifetime proxy; 0 = no cures observed
    "re_default_count_24m":       0,     # lifetime proxy; 0 = no re-defaults
    "months_dormant":             0,     # no dormancy data
    "pct_months_npl_24m":         0.0,   # no NPL history data
    "worst_stage_ever":           "CURRENT",  # conservative default
}

# Column rename map: bureau_stage_dynamics name → PersonaBuilder / SEGMENTATION name
COLUMN_RENAME_MAP: Dict[str, str] = {
    "ref_no":                        "account_id",
    "npl_stickiness_index":          "stage_stickiness_score",
    "cure_to_redefault_ratio":       "re_default_rate_rolling_24m",
    "stage_reversal_rate":           "cure_rate_sm_to_lower",
    "avg_payment_effort_ratio_npl":  "payment_effort_ratio_npl",
    "cure_count_lifetime":           "cure_count_24m",           # APPROX
    "re_default_count_lifetime":     "re_default_count_24m",     # APPROX
    "TIME_SINCE_LAST_PAY_OVERALL":   "months_dormant",
    "months_observed_total":         "bureau_months_on_book",
}

# Final output columns (subset of SEGMENTATION_FEATURES that this adapter produces)
# Pass-through from features_df: balance, bureau_total_outstanding,
# bureau_monthly_instalment, bureau_delinquent_other, bureau_new_loan_12m,
# bureau_secured_loan_flag, wrong_number_flag, dispute_flag, complaint_flag,
# lawyer_mentioned, legal_representation_flag, sms_opt_out, last_contact_outcome
ADAPTER_OUTPUT_COLUMNS: List[str] = [
    "account_id",
    "dpd_shape_type",
    "dpd_slope_12m",
    "stage_stickiness_score",
    "re_default_rate_rolling_24m",
    "cure_rate_sm_to_lower",
    "payment_effort_ratio_npl",
    "cure_count_24m",
    "re_default_count_24m",
    "months_dormant",
    "bureau_months_on_book",
    "pct_months_npl_24m",
    "worst_stage_ever",
]


# ─────────────────────────────────────────────────────────────────────────────
# ERROR CLASS
# ─────────────────────────────────────────────────────────────────────────────

class BureauFeatureAdapterError(Exception):
    """Raised when critical bureau features are missing or excessively null."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# DATA QUALITY REPORT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BureauAdapterQAReport:
    """Data quality summary from the adapter run."""
    observation_date: str
    n_accounts: int
    null_rates: Dict[str, float] = field(default_factory=dict)
    approx_columns: List[str] = field(default_factory=list)   # columns using APPROX proxy
    defaults_applied: Dict[str, int] = field(default_factory=dict)  # col → n rows filled
    critical_failures: List[str] = field(default_factory=list)
    is_valid: bool = True

    def to_dict(self) -> dict:
        return {
            "observation_date":   self.observation_date,
            "n_accounts":         self.n_accounts,
            "null_rates":         self.null_rates,
            "approx_columns":     self.approx_columns,
            "defaults_applied":   self.defaults_applied,
            "critical_failures":  self.critical_failures,
            "is_valid":           self.is_valid,
        }


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTER
# ─────────────────────────────────────────────────────────────────────────────

class BureauFeatureAdapter:
    """
    Maps bureau_stage_dynamics.py output into PersonaBuilder input schema.

    Usage:
        adapter = BureauFeatureAdapter()
        seg_features, qa_report = adapter.adapt(bureau_df, observation_date="2024-01-31")
        # seg_features is a Pandas DataFrame with ADAPTER_OUTPUT_COLUMNS
        # Merge with other segmentation features (balance, bureau flags) before
        # passing to PersonaBuilder.assign_batch()

    Input:
        bureau_df must be a Pandas DataFrame (collected from Spark) containing
        the merged output of bureau_stage_dynamics.py functions:
            build_dpd_profile_shape_features()   → dpd_shape_type, dpd_slope_12m
            build_stage_velocity_features()       → npl_stickiness_index, stage_reversal_rate
            build_cure_redefault_dynamics()       → cure_count_lifetime, cure_to_redefault_ratio
            build_payment_effort_dynamics()       → avg_payment_effort_ratio_npl
            build_vintage_seasoning_features()    → months_observed_total
            build_recovery_indicators()           → TIME_SINCE_LAST_PAY_OVERALL
            build_delinquency_severity_features() → max_stage_ever_reached (ordinal)

        ref_no is the join key (renamed to account_id in output).
        observation_date is used for audit trail only — PIT enforcement happens
        upstream in bureau_stage_dynamics.py.
    """

    def __init__(self, critical_null_threshold: float = CRITICAL_NULL_PCT):
        """
        Args:
            critical_null_threshold: Fraction of nulls that triggers fail-fast
                                     for CRITICAL_FEATURES. Default 0.30.
        """
        self.critical_null_threshold = critical_null_threshold

    def adapt(
        self,
        bureau_df: pd.DataFrame,
        observation_date: str,
    ) -> tuple:
        """
        Map bureau_stage_dynamics output to PersonaBuilder feature schema.

        Args:
            bureau_df        : Pandas DataFrame (collected from Spark).
                               Must have ref_no as the account key.
            observation_date : Snapshot date (YYYY-MM-DD). Used for QA report.

        Returns:
            (adapted_df, qa_report):
                adapted_df : Pandas DataFrame with ADAPTER_OUTPUT_COLUMNS schema.
                             Exactly one row per account_id.
                qa_report  : BureauAdapterQAReport with null rates, approx flags.

        Raises:
            BureauFeatureAdapterError: if critical features exceed null threshold.
        """
        if bureau_df.empty:
            raise BureauFeatureAdapterError("bureau_df is empty — nothing to adapt.")

        df = bureau_df.copy()
        qa = BureauAdapterQAReport(
            observation_date=observation_date,
            n_accounts=len(df),
        )

        # ── Step 1: Rename columns ────────────────────────────────────────────
        rename_map = {k: v for k, v in COLUMN_RENAME_MAP.items() if k in df.columns}
        df = df.rename(columns=rename_map)

        # Columns using APPROX proxies (lifetime → 24m window)
        approx = []
        if "cure_count_24m" in df.columns and "cure_count_lifetime" not in df.columns:
            approx.append("cure_count_24m (proxy: cure_count_lifetime)")
        if "re_default_count_24m" in df.columns and "re_default_count_lifetime" not in df.columns:
            approx.append("re_default_count_24m (proxy: re_default_count_lifetime)")
        qa.approx_columns = approx

        # ── Step 2: Derive pct_months_npl_24m ────────────────────────────────
        # Derived from dpd_time_above_90_24m (months DPD>90 in last 24m)
        # Approximation: NPL = DPD > 90, so pct = time_above_90 / 24
        if "pct_months_npl_24m" not in df.columns:
            if "dpd_time_above_90_24m" in df.columns:
                df["pct_months_npl_24m"] = (
                    df["dpd_time_above_90_24m"].fillna(0) / 24.0
                ).clip(0, 1)
            else:
                df["pct_months_npl_24m"] = np.nan

        # ── Step 3: Decode worst_stage_ever from ordinal ──────────────────────
        if "worst_stage_ever" not in df.columns:
            if "max_stage_ever_reached" in df.columns:
                df["worst_stage_ever"] = (
                    df["max_stage_ever_reached"]
                    .map(ORDINAL_TO_STAGE_MAP)
                )
            else:
                df["worst_stage_ever"] = np.nan

        # ── Step 4: Null rate audit (before filling) ──────────────────────────
        all_output_cols = ADAPTER_OUTPUT_COLUMNS + list(SOFT_FEATURE_DEFAULTS.keys())
        for col in set(all_output_cols):
            if col in df.columns:
                null_rate = df[col].isna().mean()
                qa.null_rates[col] = round(float(null_rate), 4)

        # ── Step 5: Fail-fast on critical features ────────────────────────────
        critical_failures = []
        for col in CRITICAL_FEATURES:
            if col not in df.columns:
                critical_failures.append(f"{col}: column missing entirely")
            else:
                null_rate = df[col].isna().mean()
                if null_rate > self.critical_null_threshold:
                    critical_failures.append(
                        f"{col}: {null_rate:.1%} null (threshold {self.critical_null_threshold:.0%})"
                    )

        if critical_failures:
            qa.critical_failures = critical_failures
            qa.is_valid = False
            raise BureauFeatureAdapterError(
                f"Critical bureau features failed data quality checks:\n"
                + "\n".join(f"  • {f}" for f in critical_failures)
            )

        # ── Step 6: Fill soft feature defaults ───────────────────────────────
        defaults_applied: Dict[str, int] = {}
        for col, default_val in SOFT_FEATURE_DEFAULTS.items():
            if col not in df.columns:
                df[col] = default_val
                defaults_applied[col] = len(df)
            else:
                null_count = df[col].isna().sum()
                if null_count > 0:
                    df[col] = df[col].fillna(default_val)
                    defaults_applied[col] = int(null_count)
        qa.defaults_applied = defaults_applied

        # ── Step 7: Validate dpd_shape_type values ────────────────────────────
        valid_shapes = {"CLIFF", "SLIDE", "OSCILLATOR", "RECOVERING", "STABLE"}
        if "dpd_shape_type" in df.columns:
            df["dpd_shape_type"] = df["dpd_shape_type"].str.upper()
            invalid_shapes = ~df["dpd_shape_type"].isin(valid_shapes) & df["dpd_shape_type"].notna()
            n_invalid = invalid_shapes.sum()
            if n_invalid > 0:
                logger.warning(
                    f"BureauFeatureAdapter: {n_invalid} rows have unexpected "
                    f"dpd_shape_type values. Setting to NaN."
                )
                df.loc[invalid_shapes, "dpd_shape_type"] = np.nan

        # ── Step 8: Clamp derived ratio columns ──────────────────────────────
        for col in ["re_default_rate_rolling_24m", "cure_rate_sm_to_lower",
                    "pct_months_npl_24m", "stage_stickiness_score"]:
            if col in df.columns:
                df[col] = df[col].clip(0.0, 1.0)

        for col in ["cure_count_24m", "re_default_count_24m", "months_dormant",
                    "bureau_months_on_book"]:
            if col in df.columns:
                df[col] = df[col].clip(lower=0)

        # ── Step 9: Select output columns only ───────────────────────────────
        available_output = [c for c in ADAPTER_OUTPUT_COLUMNS if c in df.columns]
        missing_output = [c for c in ADAPTER_OUTPUT_COLUMNS if c not in df.columns]
        if missing_output:
            logger.warning(
                f"BureauFeatureAdapter: output columns not available after adaptation: "
                f"{missing_output}. They will be absent from the output."
            )

        adapted = df[available_output].drop_duplicates(subset=["account_id"])

        logger.info(
            f"BureauFeatureAdapter: adapted {len(adapted)} accounts for {observation_date}. "
            f"Approx columns: {len(approx)}. Defaults applied: {len(defaults_applied)}."
        )
        return adapted, qa

    # ── BASIC DISTRIBUTION CHECKS ─────────────────────────────────────────────

    def check_distributions(self, adapted_df: pd.DataFrame) -> Dict[str, dict]:
        """
        Run basic distribution checks on adapted features.

        Returns a dict of {col: {"mean": ..., "p5": ..., "p95": ..., "pct_zero": ...}}
        for numeric columns and {"unique_values": ..., "pct_mode": ...} for string cols.
        """
        report = {}
        for col in ADAPTER_OUTPUT_COLUMNS:
            if col not in adapted_df.columns or col == "account_id":
                continue
            series = adapted_df[col].dropna()
            if series.dtype == object:
                vc = series.value_counts(normalize=True)
                report[col] = {
                    "unique_values": series.nunique(),
                    "pct_mode": round(float(vc.iloc[0]) if len(vc) > 0 else 0, 4),
                    "mode": vc.index[0] if len(vc) > 0 else None,
                }
            else:
                report[col] = {
                    "mean":     round(float(series.mean()), 4) if len(series) > 0 else None,
                    "p5":       round(float(np.percentile(series, 5)), 4) if len(series) > 0 else None,
                    "p95":      round(float(np.percentile(series, 95)), 4) if len(series) > 0 else None,
                    "pct_zero": round(float((series == 0).mean()), 4) if len(series) > 0 else None,
                }
        return report
