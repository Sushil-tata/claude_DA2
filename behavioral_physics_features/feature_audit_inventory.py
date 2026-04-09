"""
Feature Audit Inventory — CardX Recovery Modelling Platform
============================================================
Complete feature classification per the Step 1–7 audit framework.

Purpose:
    Single authoritative source of truth for feature metadata.
    Used by: data scientists (feature selection), MLflow logging,
             PSI monitoring, SHAP explainability, data contracts.

Classification schema per feature:
    name:        Feature name (matches column name in feature store)
    source:      "cardx" | "bureau" | "derived"
    category:    State | Transition | Exposure | Payment | Engagement |
                 CreditSeeking | PortfolioStructure | Physics
    time_window: "lifetime" | "at_co" | "3m" | "6m" | "12m" | "24m" | "48m"
    usage:       "segmentation" | "propensity" | "both"
    leakage_risk:"none" | "low" | "high"
    notes:       Free text — rationale for usage assignment

Usage:
    from behavioral_physics_features.feature_audit_inventory import (
        SEGMENTATION_FEATURES,
        PROPENSITY_FEATURES,
        get_features_by_source,
        get_features_by_category,
        check_leakage_risks,
        print_audit_report,
    )

Author: Behavioral Physics Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import textwrap


@dataclass
class FeatureMeta:
    name: str
    source: str          # cardx | bureau | derived
    category: str
    time_window: str
    usage: str           # segmentation | propensity | both
    leakage_risk: str    # none | low | high
    notes: str = ""


# ============================================================================
# A) CARDX INTERNAL FEATURES
# ============================================================================

CARDX_FEATURES: List[FeatureMeta] = [

    # ── State / DPD ──────────────────────────────────────────────────────────
    FeatureMeta("cardx_dpd_at_co",          "cardx", "State",      "at_co",    "both",         "none",
        "DPD value at charge-off date. Segmentation: structural starting point. Propensity: recency signal."),
    FeatureMeta("cardx_max_dpd_ever",        "cardx", "State",      "lifetime", "segmentation", "none",
        "Worst-ever CardX DPD. Captures peak stress, not recency. Stable trait — segmentation only."),
    FeatureMeta("months_in_s1_cardx_24m",    "cardx", "State",      "24m",      "segmentation", "none",
        "Months in 1-30 DPD bucket in last 24M. Structural delinquency exposure."),
    FeatureMeta("months_in_s2_cardx_24m",    "cardx", "State",      "24m",      "segmentation", "none",
        "Months in 31-90 DPD bucket. Chronic SM exposure."),
    FeatureMeta("months_in_s3_cardx_24m",    "cardx", "State",      "24m",      "segmentation", "none",
        "Months in 91-180 DPD. NPL structural flag."),

    # ── Transition ────────────────────────────────────────────────────────────
    FeatureMeta("cardx_cure_count_24m",      "cardx", "Transition", "24m",      "segmentation", "none",
        "Times CardX returned to current from delinquency. Recidivism indicator."),
    FeatureMeta("cardx_cure_durability_avg_days", "cardx", "Transition", "24m", "segmentation", "none",
        "Average days in current status between episodes. Short = unstable cures."),
    FeatureMeta("delinquency_episode_count_24m", "cardx", "Transition", "24m",  "segmentation", "none",
        "Distinct delinquency episodes. Chronic vs acute discriminator."),
    FeatureMeta("dpd_trajectory_slope_6m",   "cardx", "Transition", "6m",       "segmentation", "none",
        "Linear slope of CardX DPD over last 6M pre-CO. Captures deterioration speed."),
    FeatureMeta("dpd_trajectory_slope_12m",  "cardx", "Transition", "12m",      "segmentation", "none",
        "12M slope. More stable than 6M — captures structural trend not noise."),
    FeatureMeta("dpd_acceleration_6m",       "cardx", "Transition", "6m",       "segmentation", "none",
        "Second derivative of DPD. Worsening faster or decelerating?"),
    FeatureMeta("dpd_velocity_3m_vs_6m",     "cardx", "Transition", "6m",       "segmentation", "none",
        "Momentum: recent 3M rate vs prior 3M. Shock detector."),
    FeatureMeta("first_delinquency_months_ago","cardx","Transition", "lifetime", "segmentation", "none",
        "Age of problem. Fresh = shock; old = chronic."),

    # ── Exposure / Balance ────────────────────────────────────────────────────
    FeatureMeta("outstanding_balance_at_co", "cardx", "Exposure",   "at_co",    "both",         "none",
        "Balance at CO. Segmentation: exposure bucket. Propensity: economic priority."),
    FeatureMeta("log_balance_at_co",         "cardx", "Exposure",   "at_co",    "segmentation", "none",
        "log1p(balance_at_co). Heavy-tail safe. Item 3 addition."),
    FeatureMeta("balance_bucket",            "cardx", "Exposure",   "at_co",    "segmentation", "none",
        "Ordinal bucket: 0-5k/5-20k/20-50k/50-100k/100k+. Actionability threshold."),
    FeatureMeta("balance_volatility_12m",    "cardx", "Exposure",   "12m",      "segmentation", "none",
        "Std dev of balance in last 12M. High vol = erratic behaviour."),
    FeatureMeta("balance_at_co_to_limit_ratio","cardx","Exposure",  "at_co",    "segmentation", "none",
        "Utilisation at CO. Near 1.0 = fully drawn. Structural exposure."),
    FeatureMeta("utilization_trend_6m",      "cardx", "Exposure",   "6m",       "segmentation", "none",
        "Slope of utilisation over 6M pre-CO."),
    FeatureMeta("utilization_volatility_12m","cardx", "Exposure",   "12m",      "segmentation", "none",
        "Std dev of utilisation. Stable vs erratic credit use."),

    # ── Payment ───────────────────────────────────────────────────────────────
    FeatureMeta("avg_payment_ratio_12m",     "cardx", "Payment",    "12m",      "segmentation", "none",
        "Avg payment/minimum_due in last 12M. Core structural payment capacity."),
    FeatureMeta("min_payment_ratio_12m",     "cardx", "Payment",    "12m",      "segmentation", "none",
        "Worst payment coverage month. Floor of payment behaviour."),
    FeatureMeta("payment_ratio_trend_6m",    "cardx", "Payment",    "6m",       "segmentation", "none",
        "Slope of payment ratio over 6M. Improving or deteriorating?"),
    FeatureMeta("payment_velocity_3m",       "cardx", "Payment",    "3m",       "segmentation", "none",
        "Rate of change in payment amount over 3M."),
    FeatureMeta("payment_decay_ratio",       "cardx", "Payment",    "12m",      "segmentation", "none",
        "avg_payment_last_3M / avg_payment_6to12M. Collapsing payment effort."),
    FeatureMeta("last_payment_months_before_co","cardx","Payment",  "lifetime", "segmentation", "none",
        "Recency of last payment before CO. Null=never paid."),
    FeatureMeta("payment_count_last_12m_pre_co","cardx","Payment",  "12m",      "segmentation", "none",
        "Count of payments in final 12M. Zero = total disengagement."),
    FeatureMeta("payment_to_debt_ratio",     "derived","Payment",   "12m",      "segmentation", "none",
        "avg_payment_ratio_12m / (bureau_total_debt+1). Affordability proxy. Item 6."),
    FeatureMeta("PTP_kept_rate_12m",         "cardx", "Payment",    "12m",      "segmentation", "none",
        "% PTPs kept. Stable willingness-to-pay signal. Segmentation only — not propensity."),
    FeatureMeta("PTP_broken_rate_12m",       "cardx", "Payment",    "12m",      "segmentation", "none",
        "% PTPs broken. Strategic defaulter indicator."),

    # ── Engagement — PROPENSITY ONLY ─────────────────────────────────────────
    FeatureMeta("contact_success_rate_3m",   "cardx", "Engagement", "3m",       "propensity",   "low",
        "Right-party contact rate last 3M. Short-horizon — propensity only."),
    FeatureMeta("rpc_count_last_30d",        "cardx", "Engagement", "1m",       "propensity",   "low",
        "RPC count last 30 days. Tactical. Do NOT use in segmentation."),
    FeatureMeta("ptp_count_last_30d",        "cardx", "Engagement", "1m",       "propensity",   "low",
        "PTP count last 30 days. Recent promise — propensity only."),
    FeatureMeta("days_since_last_rpc",       "cardx", "Engagement", "1m",       "propensity",   "low",
        "Days since last right-party contact. Recency signal — propensity."),
    FeatureMeta("digital_open_rate_3m",      "cardx", "Engagement", "3m",       "propensity",   "low",
        "SMS/email open rate last 3M. Channel responsiveness — propensity only."),
    FeatureMeta("contact_attempts_last_30d", "cardx", "Engagement", "1m",       "propensity",   "high",
        "Treatment intensity (policy). NEVER use in segmentation. Propensity: recency of attempt."),
    FeatureMeta("last_payment_post_co_days", "cardx", "Engagement", "at_co",    "propensity",   "high",
        "Days since any payment post-CO. POST-CO — propensity only. EXCLUDE from segmentation."),
    FeatureMeta("paid_in_first_30d",         "cardx", "Engagement", "at_co",    "propensity",   "high",
        "Binary: payment in first 30d post-CO. REMOVED from segmentation (outcome proxy). Item 2."),
    FeatureMeta("paid_in_first_60d",         "cardx", "Engagement", "at_co",    "propensity",   "high",
        "Binary: payment in first 60d post-CO. REMOVED from segmentation (outcome proxy). Item 2."),
]


# ============================================================================
# B) BUREAU (NCB) FEATURES
# ============================================================================

BUREAU_FEATURES: List[FeatureMeta] = [

    # ── State / DPD ──────────────────────────────────────────────────────────
    FeatureMeta("bureau_max_dpd",            "bureau", "State",      "at_co",    "both",         "none",
        "Max DPD across all bureau tradelines at CO snapshot."),
    FeatureMeta("ncb_accounts_delinquent_count","bureau","State",    "at_co",    "segmentation", "none",
        "Count of delinquent bureau accounts. Systemic vs isolated."),
    FeatureMeta("ncb_total_delinquent_pct",  "bureau", "State",      "at_co",    "segmentation", "none",
        "Fraction of total credit exposure delinquent. Structural exposure."),
    FeatureMeta("pct_s0_48m",                "bureau", "State",      "48m",      "segmentation", "none",
        "% months in S0 over 48M. Long-horizon current status stability."),
    FeatureMeta("pct_s2_48m",                "bureau", "State",      "48m",      "segmentation", "none",
        "% months in SM/S2. Chronic SM exposure indicator."),
    FeatureMeta("pct_s3_48m",                "bureau", "State",      "48m",      "segmentation", "none",
        "% months in NPL/S3. Chronic NPL indicator — strongest structural signal."),

    # ── Transition ────────────────────────────────────────────────────────────
    FeatureMeta("cure_rate_s2",              "bureau", "Transition", "lifetime", "segmentation", "none",
        "Rate of curing from S2 (SM). Source: bureau_stage_dynamics Section 5."),
    FeatureMeta("cure_rate_s3",              "bureau", "Transition", "lifetime", "segmentation", "none",
        "Rate of curing from S3 (NPL). Very low in chronic defaulters."),
    FeatureMeta("roll_forward_rate_s1",      "bureau", "Transition", "lifetime", "segmentation", "none",
        "Rate rolling from X to SM. Leading indicator of chronic pattern."),
    FeatureMeta("roll_forward_rate_s2",      "bureau", "Transition", "lifetime", "segmentation", "none",
        "Rate rolling from SM to NPL."),
    FeatureMeta("episode_count_s2",          "bureau", "Transition", "lifetime", "segmentation", "none",
        "# episodes in SM. Chronic oscillators vs single-episode distress."),
    FeatureMeta("episode_count_s3",          "bureau", "Transition", "lifetime", "segmentation", "none",
        "# episodes in NPL. Severe chronic indicator."),
    FeatureMeta("avg_duration_s2",           "bureau", "Transition", "lifetime", "segmentation", "none",
        "Avg months per SM episode. Long = trapped."),
    FeatureMeta("dpd_diff_velocity_3m",      "bureau", "Transition", "3m",       "segmentation", "none",
        "Bureau DPD slope over 3M. Recent deterioration speed."),
    FeatureMeta("dpd_diff_velocity_6m",      "bureau", "Transition", "6m",       "segmentation", "none",
        "Bureau DPD slope over 6M. More stable trajectory signal."),
    FeatureMeta("dpd_accel_1m",              "bureau", "Transition", "1m",       "propensity",   "low",
        "Bureau DPD acceleration (2nd diff). Short-horizon — propensity context."),
    FeatureMeta("state_entropy_48m",         "bureau", "Transition", "48m",      "segmentation", "none",
        "Shannon entropy of DPD states over 48M. High = unpredictable, erratic behaviour."),
    FeatureMeta("months_since_transition",   "bureau", "Transition", "lifetime", "segmentation", "none",
        "Months since last DPD state change. High = stuck/inert."),

    # ── Within-Stage DPD Dynamics (NEW — bureau_stage_dynamics.py) ───────────
    FeatureMeta("avg_dpd_at_entry_sm",       "bureau", "Transition", "lifetime", "segmentation", "none",
        "Avg DPD when entering SM episodes. Low entry DPD = barely crossed boundary."),
    FeatureMeta("avg_dpd_slope_within_sm",   "bureau", "Transition", "lifetime", "segmentation", "none",
        "DPD slope while inside SM. Positive = heading to NPL."),
    FeatureMeta("retreat_ratio_sm",          "bureau", "Transition", "lifetime", "segmentation", "none",
        "# months DPD decreased within SM / total SM months. Self-cure attempt signal."),
    FeatureMeta("avg_boundary_proximity_sm", "bureau", "Transition", "lifetime", "segmentation", "none",
        "Distance to SM/NPL boundary as fraction of stage width. Near 0 = about to roll."),
    FeatureMeta("avg_dpd_slope_within_npl",  "bureau", "Transition", "lifetime", "segmentation", "none",
        "DPD slope while inside NPL. Key for CO proximity."),
    FeatureMeta("retreat_ratio_npl",         "bureau", "Transition", "lifetime", "segmentation", "none",
        "DPD retreats within NPL. Even small retreats = residual willingness."),

    # ── Exposure ─────────────────────────────────────────────────────────────
    FeatureMeta("bureau_total_debt",         "bureau", "Exposure",   "at_co",    "segmentation", "none",
        "Total bureau outstanding debt. Structural debt burden."),
    FeatureMeta("ncb_total_revolving_util",  "bureau", "Exposure",   "at_co",    "segmentation", "none",
        "Total revolving utilisation. Systemic utilisation stress."),
    FeatureMeta("tradeline_stability_index", "bureau", "Exposure",   "at_co",    "segmentation", "none",
        "Active/total tradelines ratio. Portfolio stability."),
    FeatureMeta("avg_balance_entry_sm",      "bureau", "Exposure",   "lifetime", "segmentation", "none",
        "Avg balance when entering SM. High = large exposure under stress."),
    FeatureMeta("avg_balance_slope_sm",      "bureau", "Exposure",   "lifetime", "segmentation", "none",
        "Balance trend within SM episodes. Rising = no payment effort."),
    FeatureMeta("avg_dip_magnitude_sm",      "bureau", "Exposure",   "lifetime", "segmentation", "none",
        "Avg balance dip size in SM. Proxy: payment amount made while in SM."),
    FeatureMeta("total_dip_count_sm",        "bureau", "Exposure",   "lifetime", "segmentation", "none",
        "Total balance dips in SM episodes. Payment attempt frequency."),
    FeatureMeta("avg_limit_change_sm",       "bureau", "Exposure",   "lifetime", "segmentation", "none",
        "Avg limit change during SM. Negative = lender reducing exposure."),
    FeatureMeta("avg_financed_entry_sm",     "bureau", "Exposure",   "lifetime", "segmentation", "none",
        "Amount financed at SM entry. High = recently took on new credit."),
    FeatureMeta("avg_util_entry_npl",        "bureau", "Exposure",   "lifetime", "segmentation", "none",
        "Utilisation when entering NPL. Near 1.0 = fully drawn at NPL entry."),

    # ── Payment ───────────────────────────────────────────────────────────────
    FeatureMeta("pct_full_pay",              "bureau", "Payment",    "lifetime", "segmentation", "none",
        "% months with full payment across bureau tradelines."),
    FeatureMeta("pct_min_pay",               "bureau", "Payment",    "lifetime", "segmentation", "none",
        "% months with minimum payment only."),
    FeatureMeta("consecutive_min_pay_streak","bureau", "Payment",    "at_co",    "propensity",   "low",
        "Current streak of minimum payments. Recent — propensity context."),
    FeatureMeta("zero_pay_streak",           "bureau", "Payment",    "at_co",    "propensity",   "high",
        "Current streak of zero payments. Very recent signal — propensity only."),
    FeatureMeta("secured_dip_count_12m",     "bureau", "Payment",    "12m",      "segmentation", "none",
        "Balance dips on secured tradelines (12M). Secured payment effort proxy."),
    FeatureMeta("unsecured_dip_count_12m",   "bureau", "Payment",    "12m",      "segmentation", "none",
        "Balance dips on unsecured tradelines (12M). Unsecured payment effort."),
    FeatureMeta("secured_payment_priority_score","bureau","Payment", "12m",      "segmentation", "none",
        "secured_dip / (secured+unsecured dips). Near 1 = pays secured first."),

    # ── Portfolio Structure ───────────────────────────────────────────────────
    FeatureMeta("ncb_other_accounts_current","bureau", "PortfolioStructure","at_co","segmentation","none",
        "Other bureau accounts still current. Strategic defaulter signal."),
    FeatureMeta("ncb_mortgage_current",      "bureau", "PortfolioStructure","at_co","segmentation","none",
        "Mortgage current at CO. Strongest strategic defaulter indicator."),
    FeatureMeta("ncb_secured_loan_current",  "bureau", "PortfolioStructure","at_co","segmentation","none",
        "Secured loan current at CO."),
    FeatureMeta("secured_to_total_ratio",    "bureau", "PortfolioStructure","at_co","segmentation","none",
        "Secured debt / total debt. High = protected assets, may prioritise secured."),
    FeatureMeta("pct_commercial_bank",       "bureau", "PortfolioStructure","at_co","segmentation","none",
        "% balance with Thai commercial banks. Lender mix."),
    FeatureMeta("pct_sfi",                   "bureau", "PortfolioStructure","at_co","segmentation","none",
        "% balance with SFIs (state banks). Often lower rates — priority payers."),
    FeatureMeta("pct_personal_loan",         "bureau", "PortfolioStructure","at_co","segmentation","none",
        "% balance with personal loan lenders."),
    FeatureMeta("lender_hhi",                "bureau", "PortfolioStructure","at_co","segmentation","none",
        "Herfindahl-Hirschman Index of lender concentration. Low = diversified."),
    FeatureMeta("num_lenders",               "bureau", "PortfolioStructure","at_co","segmentation","none",
        "Total number of lenders. Breadth of credit relationships."),
    FeatureMeta("avg_loans_change_sm",       "bureau", "PortfolioStructure","lifetime","segmentation","none",
        "Avg tradeline count change during SM episodes. Positive = opening new credit under stress."),
    FeatureMeta("total_loans_opened_s2",     "bureau", "PortfolioStructure","lifetime","segmentation","none",
        "New tradelines opened while in SM. Stress credit seeking."),

    # ── Credit Seeking / Enquiries ────────────────────────────────────────────
    FeatureMeta("enquiry_count_12m",         "bureau", "CreditSeeking","12m",    "both",         "none",
        "Total enquiries in 12M. High under stress = searching for liquidity."),
    FeatureMeta("enquiry_velocity_12m",      "bureau", "CreditSeeking","12m",    "propensity",   "none",
        "Enquiry rate per month. Recent urgency — propensity context."),
    FeatureMeta("pct_enq_personal_12m",      "bureau", "CreditSeeking","12m",    "segmentation", "none",
        "% enquiries for personal loans. Liquidity search pattern."),
    FeatureMeta("pct_enq_card_12m",          "bureau", "CreditSeeking","12m",    "segmentation", "none",
        "% enquiries for credit cards."),
    FeatureMeta("bureau_enquiry_velocity_3m","bureau", "CreditSeeking","3m",     "propensity",   "none",
        "Enquiry rate last 3M. Very recent — propensity context."),
    FeatureMeta("enquiry_momentum_12m",      "bureau", "CreditSeeking","12m",    "segmentation", "none",
        "count × velocity. Panic search indicator."),

    # ── Bureau Score ──────────────────────────────────────────────────────────
    FeatureMeta("ncb_score_at_co",           "bureau", "State",      "at_co",    "segmentation", "none",
        "NCB bureau score at CO. Structural creditworthiness snapshot."),
    FeatureMeta("ncb_score_trend_6m",        "bureau", "State",      "6m",       "segmentation", "none",
        "Score change 6M before CO. Improving or collapsing?"),

    # ── Cross-Lender Consistency (NEW — bureau_stage_dynamics.py) ────────────
    FeatureMeta("cardx_selective_default_months","bureau","State",   "lifetime", "segmentation", "none",
        "# months CardX stressed but bureau current. Selective default evidence."),
    FeatureMeta("selective_default_rate",    "bureau", "State",      "lifetime", "segmentation", "none",
        "Proportion of months with selective default pattern."),
    FeatureMeta("cardx_selective_default_flag","bureau","State",     "lifetime", "segmentation", "none",
        "Binary: selective default rate > 25%. Structural defaulter signal."),
    FeatureMeta("systemic_stress_flag",      "bureau", "State",      "lifetime", "segmentation", "none",
        "Binary: systemic stress rate > 25%. Sudden-shock / liquidity constrained."),
    FeatureMeta("cross_lender_divergence_score","bureau","State",    "lifetime", "segmentation", "none",
        "Mismatch ratio between CardX and bureau delinquency states."),

    # ── Pre-Existing Stress (NEW — bureau_stage_dynamics.py) ─────────────────
    FeatureMeta("pre_existing_stress_flag",  "bureau", "State",      "lifetime", "segmentation", "none",
        "Bureau was in S2+ BEFORE CardX first deteriorated. Systemic slide indicator."),
    FeatureMeta("bureau_stress_lead_months", "bureau", "State",      "lifetime", "segmentation", "none",
        "Months bureau stress preceded CardX. Positive = bureau led."),
    FeatureMeta("bureau_clean_before_cardx_flag","bureau","State",   "lifetime", "segmentation", "none",
        "Bureau was clean when CardX first hit S1. Supports isolated shock."),

    # ── Physics Features (ncb_feature_factory_v2.py Sections 10-11) ──────────
    FeatureMeta("credit_inertia_score",      "bureau", "Physics",    "lifetime", "segmentation", "none",
        "months_since_transition × dpd_ordinal × log(dpd). Resistance to change."),
    FeatureMeta("credit_momentum_3m",        "bureau", "Physics",    "3m",       "segmentation", "none",
        "dpd_velocity × max_dpd. Mass × velocity analogue."),
    FeatureMeta("dpd_zscore_12m",            "bureau", "Physics",    "12m",      "segmentation", "none",
        "DPD z-score vs 12M mean. Unusual current position."),
    FeatureMeta("dpd_log_decay_rate_1m",     "bureau", "Physics",    "1m",       "propensity",   "none",
        "Log-scale DPD decay rate. Recent — propensity context."),
    FeatureMeta("entropy_production_rate",   "bureau", "Physics",    "1m",       "propensity",   "low",
        "Rate of entropy change. Very recent — propensity context."),
    FeatureMeta("stress_tensor_magnitude",   "bureau", "Physics",    "3m",       "segmentation", "none",
        "Combined stress: sqrt(velocity² + accel² + entropy_rate²)."),
    FeatureMeta("critical_slowing_indicator","bureau", "Physics",    "12m",      "segmentation", "none",
        "autocorr × variance_ratio. Pre-tipping-point early warning."),
    FeatureMeta("phase_boundary_proximity",  "bureau", "Physics",    "at_co",    "both",         "none",
        "Inverse distance to nearest DPD stage boundary."),
    FeatureMeta("hysteresis_gap",            "bureau", "Physics",    "lifetime", "segmentation", "none",
        "Max historical state − current state. Scar from past stress."),
    FeatureMeta("num_stress_cycles_lifetime","bureau", "Physics",    "lifetime", "segmentation", "none",
        "S0→stress→S0 cycles. Chronic oscillators have high counts."),
    FeatureMeta("momentum_sign_flip_6m",     "bureau", "Physics",    "6m",       "segmentation", "none",
        "# DPD direction changes in 6M. High = erratic/oscillating."),
    FeatureMeta("variance_ratio_12m_3m",     "bureau", "Physics",    "12m",      "segmentation", "none",
        "Variance ratio: rising 3M variance vs 12M base. Critical slowing signal."),
]


# ============================================================================
# COMBINED REGISTRY
# ============================================================================

ALL_FEATURES: List[FeatureMeta] = CARDX_FEATURES + BUREAU_FEATURES


# ============================================================================
# DERIVED LISTS (used by trainers and validators)
# ============================================================================

SEGMENTATION_FEATURES: List[str] = [
    f.name for f in ALL_FEATURES
    if f.usage in ("segmentation", "both") and f.leakage_risk == "none"
]

PROPENSITY_FEATURES: List[str] = [
    f.name for f in ALL_FEATURES
    if f.usage in ("propensity", "both")
]

HIGH_LEAKAGE_FEATURES: List[str] = [
    f.name for f in ALL_FEATURES if f.leakage_risk == "high"
]

CARDX_ONLY_FEATURES: List[str] = [
    f.name for f in ALL_FEATURES if f.source == "cardx"
]

BUREAU_ONLY_FEATURES: List[str] = [
    f.name for f in ALL_FEATURES if f.source == "bureau"
]


# ============================================================================
# SIGNAL AVAILABILITY MATRIX (2×2 grid)
# ============================================================================

# Features available when CardX data present (even if bureau absent)
CARDX_PRESENT_FEATURES: List[str] = [
    f.name for f in ALL_FEATURES if f.source in ("cardx", "derived")
]

# Features available when Bureau present (even if CardX absent)
BUREAU_PRESENT_FEATURES: List[str] = [
    f.name for f in ALL_FEATURES if f.source == "bureau"
]

# Minimum feature set for LIMITED_SIGNAL (no CardX, no bureau)
# These must be derivable from basic account metadata alone
LIMITED_SIGNAL_FALLBACK_FEATURES: List[str] = [
    "cardx_max_dpd_ever",
    "outstanding_balance_at_co",
    "log_balance_at_co",
    "balance_bucket",
    "first_delinquency_months_ago",
]

SIGNAL_AVAILABILITY_GRID: Dict[str, List[str]] = {
    "FULL_SIGNAL (CardX + Bureau)":       SEGMENTATION_FEATURES,
    "CARDX_ONLY (no bureau)":             [f for f in SEGMENTATION_FEATURES if f in CARDX_PRESENT_FEATURES],
    "BUREAU_ONLY (no CardX)":             [f for f in SEGMENTATION_FEATURES if f in BUREAU_PRESENT_FEATURES],
    "LIMITED_SIGNAL (neither)":           LIMITED_SIGNAL_FALLBACK_FEATURES,
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_features_by_source(source: str) -> List[FeatureMeta]:
    return [f for f in ALL_FEATURES if f.source == source]


def get_features_by_category(category: str) -> List[FeatureMeta]:
    return [f for f in ALL_FEATURES if f.category == category]


def get_features_by_usage(usage: str) -> List[FeatureMeta]:
    return [f for f in ALL_FEATURES if f.usage in (usage, "both")]


def check_leakage_risks() -> List[FeatureMeta]:
    return [f for f in ALL_FEATURES if f.leakage_risk in ("low", "high")]


def print_audit_report():
    """Print structured audit report to stdout."""
    border = "=" * 80

    print(border)
    print("FEATURE AUDIT INVENTORY — CardX Recovery Modelling")
    print(border)

    # Counts
    print(f"\nTotal features catalogued : {len(ALL_FEATURES)}")
    print(f"  CardX internal          : {len([f for f in ALL_FEATURES if f.source == 'cardx'])}")
    print(f"  Bureau (NCB)            : {len([f for f in ALL_FEATURES if f.source == 'bureau'])}")
    print(f"  Derived                 : {len([f for f in ALL_FEATURES if f.source == 'derived'])}")
    print(f"\nSegmentation features     : {len(SEGMENTATION_FEATURES)}")
    print(f"Propensity features       : {len(PROPENSITY_FEATURES)}")
    print(f"High leakage risk         : {len(HIGH_LEAKAGE_FEATURES)}")

    # Category breakdown
    print(f"\n{'─'*40}")
    print("Features by category:")
    cats = {}
    for f in ALL_FEATURES:
        cats[f.category] = cats.get(f.category, 0) + 1
    for cat, cnt in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat:<25} {cnt:3d}")

    # Signal availability
    print(f"\n{'─'*40}")
    print("Signal availability matrix:")
    for segment, feats in SIGNAL_AVAILABILITY_GRID.items():
        print(f"  {segment:<45} {len(feats):3d} features")

    # Leakage risks
    risks = check_leakage_risks()
    if risks:
        print(f"\n{'─'*40}")
        print(f"⚠  Leakage risk features ({len(risks)}):")
        for f in risks:
            print(f"  [{f.leakage_risk.upper():4s}] {f.name:<45} usage={f.usage}")
            print(f"         → {textwrap.shorten(f.notes, 70)}")

    # Step 7 Gap Summary
    print(f"\n{'─'*40}")
    print("STEP 7 — Gap Assessment:")
    print("""
  Covered behavioural dimensions:
    ✓  Delinquency state (time-in-bucket, DPD slopes, entropy)
    ✓  Transition dynamics (roll-forward, cure, oscillation, episode count)
    ✓  Exposure (balance, limit, utilisation, log-transform, bucket)
    ✓  Within-stage dynamics (NEW: balance dip proxy, DPD position, retreat ratio)
    ✓  Payment behaviour (ratio, trend, PTP, streak, secured vs unsecured priority)
    ✓  Cross-lender consistency (NEW: selective default flag, systemic stress)
    ✓  Portfolio structure (lender mix, HHI, secured/unsecured split)
    ✓  Credit seeking (enquiry velocity, momentum, purpose mix)
    ✓  Physics families (inertia, momentum, critical slowing, hysteresis, entropy)
    ✓  Pre-existing stress (NEW: bureau stress lead months before CardX)

  Remaining gaps (recommended next build):
    △  Credit vintage across bureau tradelines (open date → tradeline age)
    △  Instalment vs revolving split dynamics within stages
    △  Amount financed trend per lender type (not just total)
    △  Rejection proxy validation (high enquiry + zero new accounts confirmed)
    △  CardX vintage (months since first CardX account opened)
    △  Post-CO propensity features: settlement offer response rate,
       broken promise count (propensity only, bounded 3–6M post-CO)

  Known leakage risks:
    ✗  paid_in_first_30d/60d — removed from segmentation (Item 2)
    ✗  contact_attempts — treatment intensity, never in segmentation
    ✗  zero_pay_streak — only in propensity (very recent window)
    """)

    print(border)


# ============================================================================
# TEMPORAL COVERAGE AUDIT (Step 3 output)
# ============================================================================

TEMPORAL_COVERAGE = {
    "pre_co_lifetime": [
        f.name for f in ALL_FEATURES
        if f.time_window in ("lifetime", "48m", "24m") and f.leakage_risk == "none"
    ],
    "pre_co_rolling_12m": [
        f.name for f in ALL_FEATURES
        if f.time_window in ("12m",) and f.leakage_risk == "none"
    ],
    "pre_co_rolling_6m": [
        f.name for f in ALL_FEATURES
        if f.time_window in ("6m",) and f.leakage_risk == "none"
    ],
    "pre_co_rolling_3m": [
        f.name for f in ALL_FEATURES
        if f.time_window in ("3m",) and f.leakage_risk == "none"
    ],
    "at_co_snapshot": [
        f.name for f in ALL_FEATURES
        if f.time_window == "at_co" and f.leakage_risk == "none"
    ],
    "post_co_bounded_propensity": [
        f.name for f in ALL_FEATURES
        if f.leakage_risk in ("low", "high") and f.usage in ("propensity", "both")
    ],
    "leakage_excluded": HIGH_LEAKAGE_FEATURES,
}


if __name__ == "__main__":
    print_audit_report()
