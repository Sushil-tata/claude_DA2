"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        BUREAU FEATURE COMPLETE — CardX Recovery Modelling Platform          ║
║        Single-File Consolidated Feature Factory                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

PURPOSE
-------
Single authoritative source for all bureau and CardX-derived features used
in the recovery modelling pipeline (segmentation + propensity). Consolidates:
    - ncb_feature_factory_v2.py        (DPD states, trajectory, physics)
    - bureau_stage_dynamics.py         (within-stage dynamics)
    - modules/repayment_dynamics.py    (regime-aware payment features)
    - modules/cardx_bureau_interactions.py (lead-lag, divergence)
    - decision_engine/bfe/vintage.py   (account maturity, cohort)
    - decision_engine/bfe/payment.py   (RFM framework)
    - decision_engine/bfe/interactions.py (cross-domain interactions)

DATA INPUTS (NCB Thailand — 4 tables + optional CardX internal)
---------------------------------------------------------------
    mnf_cra_rvw_id_dummy    — customer identity, ref_no → id_no mapping
    mnf_cra_rvw_s_account   — tradeline snapshots (balance, limit, status)
    mnf_cra_rvw_s_history   — monthly history (DPD, balance, payment strings)
    mnf_cra_rvw_s_enquiry   — credit enquiries
    [optional] cardx_monthly_df       — CardX internal monthly DPD + balance
    [optional] cardx_first_delq_df    — first CardX delinquency date per account

JOIN KEY: ref_no (from bridge table — do NOT use cust_id or id_no internally)
PIT KEY:  receive_dt (point-in-time anchor — bureau records filtered ≤ receive_dt)

═══════════════════════════════════════════════════════════════════════════════
FEATURE FAMILIES — COMPLETE MAP (~500 features total)
═══════════════════════════════════════════════════════════════════════════════

SECTION 5  │ DPD STATE BUILDER                  │ ~25 features
           │ What DPD bucket is the customer in each month?
           │ S0=Current(0), S1=X(1-30), S2=SM(31-90), S3=NPL(91-180), S4=CO(180+)
           │ Features: dpd_state, dpd_bucket_ordinal, pct_s0..s4 over 48M,
           │           bureau_max_dpd, dpd_trajectory_48m array

SECTION 6  │ STAGE EPISODE BUILDER              │ (structural, no direct output)
           │ Labels contiguous runs in same stage as episodes.
           │ Foundation for all within-stage computation in Sections 7-8.

SECTION 7  │ WITHIN-STAGE EXPOSURE DYNAMICS     │ ~60 features (12 × 5 stages)
           │ How does balance/limit/utilisation/amount-financed move while
           │ the customer is stuck inside each stage?
           │ KEY INSIGHT: Balance dip within a stage = payment proxy.
           │ A customer in S2 with falling balance made a partial payment
           │ even without curing. This is the "effort signal".
           │ Features per stage (S0–S4):
           │   avg_balance_slope_{stage}    — trend of balance within stage
           │   total_dip_count_{stage}      — # months balance fell (payment proxy)
           │   avg_dip_magnitude_{stage}    — avg size of balance dip (payment amount proxy)
           │   avg_util_entry_{stage}       — utilisation when entering stage
           │   avg_limit_change_{stage}     — lender limit response during stage
           │   avg_financed_entry_{stage}   — amount financed on stage entry

SECTION 8  │ WITHIN-STAGE LOAN COUNT DYNAMICS   │ ~30 features (6 × 5 stages)
           │ Did the customer open or close tradelines while stressed?
           │ Opening new credit during S2/S3 = desperation or strategic search.
           │ Features per stage: avg_loans_entry, avg_loans_change,
           │                     total_loans_opened, avg_loan_count_vol

SECTION 9  │ WITHIN-STAGE DPD COUNTER DYNAMICS  │ ~40 features (8 × 5 stages)
           │ WHERE within the stage is the customer heading?
           │ DPD=32 (just entered S2) vs DPD=88 (near S3 boundary) = different risk.
           │ Features per stage: avg_dpd_at_entry, avg_dpd_slope_within,
           │                     retreat_ratio, avg_boundary_proximity

SECTION 10 │ CROSS-STAGE TRANSITIONS            │ ~40 features (8 × 5 stages)
           │ Roll-forward rates, cure rates, oscillation counts per stage.
           │ Features: roll_forward_rate_{stage}, cure_rate_{stage},
           │           episode_count_{stage}, ever_in_stage_{stage}

SECTION 11 │ TRAJECTORY & PHYSICS               │ ~87 features
           │ Behavioral physics applied to credit dynamics:
           │   Velocity: rate of DPD change (slope over 3/6/12M)
           │   Acceleration: second derivative — worsening faster or slowing?
           │   Entropy: Shannon entropy of state distribution — unpredictability
           │   Inertia: resistance to state change (stuck-in-bad-state signal)
           │   Momentum: mass × velocity (directional credit force)
           │   Critical Slowing: autocorr × variance ratio (pre-tipping-point signal)
           │   Hysteresis: scar from past stress (max ever state vs current)
           │   Phase Boundary: proximity to next DPD stage boundary

SECTION 12 │ REGIME-DEPENDENT REPAYMENT         │ ~35 features
           │ Payment behaviour is DIFFERENT in stressed vs normal regimes.
           │ A customer who pays 80% in S0 but only 10% in S2 is different
           │ from one who pays 10% in both. The DELTA matters.
           │ NORMAL regime (S0/S1): payment effort, consistency, discipline score
           │ STRESSED regime (S2-S4): cure attempts, payment fatigue, chronicity
           │ Delta features: behaviour change magnitude between regimes

SECTION 13 │ VINTAGE & ACCOUNT MATURITY         │ ~16 features
           │ How old is the bureau relationship? Does delinquency happen early
           │ (origination defect) or late (external shock)?
           │ Features: months_on_book, lifecycle_stage (NEW/SEASONED/MATURE/AGED),
           │           early_delinquency_flag, origination_cohort,
           │           payment_ratio_first_6M vs recent_6M (evolution)
           │ GAP filled: this was the only confirmed missing dimension in the audit.

SECTION 14 │ LENDER ECOLOGY                     │ ~20 features
           │ Who does the customer borrow from and how concentrated are they?
           │ Features: pct_sfi, pct_commercial_bank, pct_personal_loan, pct_leasing,
           │           lender_hhi (concentration), num_lenders,
           │           lender_diversification_score, lender_network_density

SECTION 15 │ ENQUIRIES ENGINE                   │ ~12 features
           │ Credit seeking behaviour — the desperation signal.
           │ High enquiry + no new accounts = rejected (liquidity crisis).
           │ Features: enquiry_count_12m, enquiry_velocity, enquiry_momentum,
           │           pct_enq by purpose (auto/personal/card/mortgage)

SECTION 16 │ PAYMENT FEATURES — RFM + BUREAU    │ ~35 features
           │ RFM (Recency-Frequency-Monetary) framework applied to bureau payments.
           │ Features: rfm_recency_score (1-5), rfm_frequency_score (1-5),
           │           rfm_monetary_score (1-5), rfm_composite (0-15),
           │           rfm_segment (CHAMPIONS/AT_RISK/LOST/etc.),
           │           payment_elasticity (payment ratio response to balance change),
           │           payment_regime (CONSISTENT/ERRATIC/MINIMAL/COLLAPSED)

SECTION 17 │ CROSS-LENDER DYNAMICS              │ ~15 features
           │ CardX vs Bureau delinquency alignment and lead-lag timing.
           │ Selective default: CardX stressed but bureau current = CAN pay, won't.
           │ Lead-lag: did CardX deteriorate before or after bureau? (months)
           │ Cross-trigger: CardX delinquency cascades to bureau within N months.
           │ Utilisation spread: CardX util vs bureau util gap.

SECTION 18 │ DEBT PRIORITISATION                │ ~10 features
           │ Who does the customer pay first when under stress?
           │ Secured (mortgage/auto) paid before unsecured (cards) = strategic capacity.
           │ Features: secured_to_total_ratio, secured_payment_priority_score,
           │           secured_dip_count_12m vs unsecured_dip_count_12m

SECTION 19 │ PRE-EXISTING BUREAU STRESS         │ ~5 features
           │ Was bureau already stressed before CardX deteriorated?
           │ Positive lead (bureau stressed first) = systemic slide.
           │ Negative lead (CardX first) = isolated shock, CardX-specific.
           │ Features: pre_existing_stress_flag, bureau_stress_lead_months,
           │           bureau_clean_before_cardx_flag

SECTION 20 │ CROSS-DOMAIN INTERACTIONS          │ ~25 features
           │ Non-linear patterns invisible when looking at features independently.
           │ RFM × Delinquency: high-value customer behaving badly (anomaly)
           │ Vintage × Performance: early delinquency within cohort = defect
           │ Bureau × RFM: bureau stress incongruent with internal payment value
           │ Bureau × Payment: bureau payment history vs CardX payment delta

SECTION 22 │ BFE STATIC SNAPSHOT FEATURES       │ ~45 features
           │ Portfolio-level snapshot of the bureau tradeline portfolio.
           │ Aggregated at the ref_no grain from most-recent s_account snapshot.
           │ Sourced from: decision_engine/bfe/bureau.py (BFE v1.3)
           │ Account counts: total, active, closed, secured, unsecured, active_ratio
           │ Account mix: card/personal/home/auto/diversity
           │ Utilisation: total_credit_limit, total_owed, utilisation_ratio,
           │              maxed_out_accounts (>90%), high_util_accounts (>70%)
           │ Account age: oldest/avg/newest account age (months), opened 6m/12m
           │ Payment strings: on_time_payment_ratio from PAYMENTHISTORY1 char arrays
           │ Risk: restructured_debt, joint_accounts, collateralised_accounts

SECTION 23 │ EXTENDED VINTAGE & LIFECYCLE       │ ~10 features
           │ Fills the 6M exact early-delinquency window gap from Section 13.
           │ Sourced from: decision_engine/bfe/vintage.py (BFE v1.2)
           │ lifecycle_stage (NEW ≤6M / SEASONED ≤24M / MATURE ≤60M / AGED >60M)
           │ vintage_early_delinquency_flag — delinquent in FIRST 6M of account
           │ months_since_first_delinquency, months_since_first_cure
           │ origination_cohort (YYYY-MM), year, quarter, season
           │ payment_ratio_first_6M, payment_ratio_recent_6M, payment_evolution

SECTION 24 │ DELINQUENCY REGIME CLASSIFICATION  │ ~12 features
           │ What behavioural regime is the customer in right now?
           │ Sourced from: decision_engine/bfe/delinquency.py (BFE v1.0)
           │ Regimes: STABLE / DETERIORATING / VOLATILE / RECOVERING /
           │          SEVERELY_DELINQUENT / ELEVATED
           │ Features: delinquency_regime, regime_confidence,
           │           bucket_deterioration_streak, bucket_improvement_streak,
           │           ever_co_flag, ever_npl_flag, months_since_last_delinquency,
           │           time_to_npl_from_sm (acceleration measure)

SECTION 25 │ EXTENDED BFE INTERACTIONS          │ ~20 features
           │ Richer set of cross-signal interaction flags.
           │ Sourced from: decision_engine/bfe/interactions.py
           │ bureau_bad_internal_clean (selective default signal)
           │ rfm_high_value_delinquent, champions_at_risk, cant_lose_severely_delinq
           │ overleveraged_delinquent, erratic_payer_stable_dpd
           │ rapid_re_delinquency, serial_curer, mature_deteriorating
           │ payment/bureau divergence signals (5 variants)

SECTION 26 │ TDR / RESTRUCTURING DYNAMICS       │ ~22 features
           │ Debt restructuring (TDR) history, velocity, and post-TDR outcomes.
           │ Sourced from: behavioral_physics_features/modules/tdr_restructuring.py
           │ tdr_count_lifetime, tdr_count_12m, months_since_last_tdr
           │ tdr_velocity_12m, tdr_exhaustion_flag (3+ in 12M)
           │ tdr_adherence_rate (proxy), tdr_breach_count
           │ post_tdr_delinquency_flag, pre_tdr_dpd_avg, tdr_severity_score
           │ post_tdr_cure_months, tdr_relapse_flag, tdr_friction_score

SECTION 27 │ LEGAL ACTIONS & SETTLEMENT         │ ~18 features
           │ Legal filing, write-off, and settlement dynamics.
           │ Sourced from: behavioral_physics_features/modules/legal_actions.py
           │ has_legal_action_flag, num_legal_actions_12m
           │ months_since_first_legal, legal_status_current
           │ settlement_attempt_count, settlement_breach_count
           │ settlement_success_rate, post_settlement_cure_flag
           │ writeoff_flag, writeoff_amount, months_since_writeoff
           │ legal_friction_score (resistance to resolution)

SECTION 28 │ MAIN ASSEMBLY                      │ Orchestration only
           │ run_complete_bureau_features() — calls all sections, joins on ref_no

═══════════════════════════════════════════════════════════════════════════════
COLUMN CONTRACT (do not deviate — audit-enforced)
═══════════════════════════════════════════════════════════════════════════════
    Join key:     ref_no          (from bridge — sole key through pipeline)
    Time key:     asofdate        (bureau history month, from s_history)
    PIT anchor:   receive_dt      (from bridge — never add to schema maps)
    DPD column:   overduemonths   (× 30 = DPD proxy, from s_history)
    Balance:      amountowed      (from s_history)
    Limit:        creditlimit     (from s_account)
    Financed:     amountfinanced  (from s_account)
    Lender:       membershortname (lowercased on load — ONLY lender identifier)

EXCLUDED (never use):
    lender_id, lender_type_raw, MEMBERCODE — see THAI_LENDER_CLASSIFICATION.md
    XXX values — treat as null via _as_int_safe_col() before any DPD arithmetic
    receive_dt, dl_data_dt in schema maps — exist only from bridge join

═══════════════════════════════════════════════════════════════════════════════
USAGE
═══════════════════════════════════════════════════════════════════════════════
    from behavioral_physics_features.bureau_feature_complete import (
        run_complete_bureau_features,
        SEGMENTATION_FEATURES,
        PROPENSITY_FEATURES,
    )

    feature_df = run_complete_bureau_features(
        spark        = spark,
        schema_name  = "cdx_mdz_prd.cdx_persist_mnf_res_db",
        bridge_df    = bridge_df,               # ref_no, receive_dt, as_of_month
        cardx_monthly_df       = cardx_monthly, # optional
        cardx_first_delq_df    = cardx_delq,    # optional
    )

Author: Behavioral Physics Team
Version: 4.2.0 (Sections 34–41 added — full bureau_stage_dynamics structural parity)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# All hardcoded values live here. Change here only — nowhere else.
# ══════════════════════════════════════════════════════════════════════════════

CFG: Dict = {
    # ── Key columns ───────────────────────────────────────────────────────────
    "ref_col":    "ref_no",          # sole join key — from bridge table
    "id_col":     "id_no",           # national ID — bureau key
    "anchor_col": "receive_dt",      # PIT anchor — exists only from bridge join

    # ── DPD state bins (BOT Thailand classification) ─────────────────────────
    # S0 = Current, S1 = X bucket (1-30), S2 = SM (31-90),
    # S3 = NPL (91-180), S4 = CO (180+)
    "dpd_state_bins": [
        (0,   0,     "S0"),
        (1,   30,    "S1"),
        (31,  90,    "S2"),
        (91,  180,   "S3"),
        (181, 99999, "S4"),
    ],
    "dpd_bucket_ordinal": {"S0": 0, "S1": 1, "S2": 2, "S3": 3, "S4": 4},
    "dpd_stage_upper_boundary": {"S0": 30, "S1": 90, "S2": 90, "S3": 180, "S4": 999},
    "dpd_stage_width":          {"S0": 30, "S1": 29, "S2": 60, "S3": 89,  "S4": 820},

    # ── OVERDUEMONTHS → DPD conversion ───────────────────────────────────────
    "odm_to_dpd_multiplier": 30,

    # ── Lookback windows ──────────────────────────────────────────────────────
    "trajectory_window": 48,    # months for entropy, state trajectory array
    "enquiry_window":    12,    # months for enquiry velocity
    "windows_months":    [1, 3, 6, 12],

    # ── Bureau tables (NCB Thailand — exactly 4) ─────────────────────────────
    "bureau_tables": {
        "id_dummy": "mnf_cra_rvw_id_dummy",
        "account":  "mnf_cra_rvw_s_account",
        "history":  "mnf_cra_rvw_s_history",
        "enquiry":  "mnf_cra_rvw_s_enquiry",
    },

    # ── Lender taxonomy (Thai financial institution types) ────────────────────
    "lender_taxonomy": {
        "SFI":             ["GOVERNMENT SAVINGS BANK", "GSB", "BAAC", "GH BANK",
                            "GOVERNMENT HOUSING BANK", "SME BANK", "EXIM BANK"],
        "COMMERCIAL_BANK": ["BANGKOK BANK", "BBL", "KASIKORN", "KBANK", "SCB",
                            "SIAM COMMERCIAL", "KRUNG THAI", "KTB", "TMB",
                            "THANACHART", "TISCO", "KIATNAKIN", "CIMB", "UOB",
                            "STANDARD CHARTERED", "CITIBANK", "HSBC"],
        "PERSONAL_LOAN":   ["MUANG THAI", "EASY BUY", "KRUNGSRI CONSUMER", "AEON",
                            "NGERN TIDLOR", "PROMISE", "CAPITAL OK"],
        "LEASING":         ["TOYOTA", "ISUZU", "HONDA", "NISSAN",
                            "AYUDHYA CAPITAL", "SRISAWAD", "ORIX"],
        "FINTECH":         ["RABBIT", "GRAB", "SHOPEE", "LAZADA",
                            "DIGITALVENTURES", "KASIKORN VISION"],
        "CARDX":           ["CARDX", "CARDX_INTERNAL"],
    },

    # ── Missingness threshold ─────────────────────────────────────────────────
    "missingness_threshold": 0.60,  # accounts with >60% null features → Dormant fallback

    # ── Stability threshold (Item 4) ─────────────────────────────────────────
    "min_stability_rate": 0.70,     # ≥70% accounts must retain persona month-on-month

    # ── RFM score thresholds ──────────────────────────────────────────────────
    "rfm_recency_bins":   [1, 30, 60, 90, 180],  # days since last payment
    "rfm_frequency_bins": [1, 2,  4,  8,  12],   # payments in 12M
}

# Stage label mapping (S0-S4 → readable names)
STAGE_LABELS: Dict[str, str] = {
    "S0": "current",
    "S1": "x_bucket",   # X = 1-30 DPD (Thai regulatory 'X' classification)
    "S2": "sm",          # Special Mention = 31-90 DPD
    "S3": "npl",         # Non-Performing Loan = 91-180 DPD
    "S4": "co",          # Charge-Off = 180+ DPD
}

# ── Segmentation vs Propensity split (used by model trainers) ────────────────
# Segmentation: structural, long-horizon, no outcome proxies, no recency
# Propensity:   tactical, short-horizon, recent engagement signals
SEGMENTATION_FEATURES: List[str] = [
    # DPD State (Section 5)
    "bureau_max_dpd", "worst_dpd_ordinal", "state_entropy_48m",
    "pct_s0_48m", "pct_s1_48m", "pct_s2_48m", "pct_s3_48m", "pct_s4_48m",
    # Within-stage exposure (Section 7) — all 5 stages × 6 metrics
    *[f"{m}_{s}" for s in STAGE_LABELS.values()
      for m in ["avg_balance_slope", "total_dip_count", "avg_dip_magnitude",
                "avg_util_entry", "avg_limit_change", "avg_financed_entry"]],
    # Loan count dynamics (Section 8) — all 5 stages
    *[f"{m}_{s}" for s in STAGE_LABELS.values()
      for m in ["avg_loans_entry", "avg_loans_change",
                "total_loans_opened", "avg_loan_count_vol"]],
    # DPD counter dynamics (Section 9) — all 5 stages
    *[f"{m}_{s}" for s in STAGE_LABELS.values()
      for m in ["avg_dpd_at_entry", "avg_dpd_slope_within",
                "retreat_ratio", "avg_boundary_proximity"]],
    # Cross-stage transitions (Section 10)
    *[f"{m}_{s}" for s in STAGE_LABELS.values()
      for m in ["episode_count", "roll_forward_rate", "cure_rate", "ever_in_stage"]],
    # Trajectory & Physics (Section 11)
    # dpd_accel_1m and dpd_log_decay_rate_1m are in PROPENSITY_FEATURES (1M window)
    "dpd_diff_velocity_3m", "dpd_diff_velocity_6m", "dpd_diff_velocity_12m",
    "credit_inertia_score", "credit_momentum_3m",
    "dpd_zscore_12m", "stress_tensor_magnitude",
    "critical_slowing_indicator", "phase_boundary_proximity",
    "hysteresis_gap", "num_stress_cycles_lifetime", "momentum_sign_flip_6m",
    # Regime-dependent repayment (Section 12)
    "payment_effort_normal", "payment_consistency_normal", "payment_cv_normal",
    "payment_discipline_score_normal",
    "cure_attempt_count_stressed", "cure_success_rate_stressed",
    "payment_effort_stressed", "chronicity_index",
    "payment_effort_delta", "payment_consistency_delta",
    # Vintage (Section 13)
    "vintage_months_on_book", "vintage_lifecycle_stage",
    "vintage_early_delinquency_flag", "vintage_payment_evolution",
    "vintage_months_since_first_delinquency",
    # Lender ecology (Section 14)
    "lender_hhi", "num_lenders", "lender_diversification_score",
    "pct_sfi", "pct_commercial_bank", "pct_personal_loan",
    # Enquiries (Section 15)
    "enquiry_count_12m", "enquiry_momentum_12m",
    "pct_enq_personal_12m", "pct_enq_card_12m",
    # Cross-lender (Section 17)
    "cardx_selective_default_flag", "systemic_stress_flag",
    "cross_lender_divergence_score", "selective_default_rate",
    "cardx_leads_bureau_flag", "lead_lag_months",
    # Debt prioritisation (Section 18)
    "secured_to_total_ratio", "secured_payment_priority_score",
    # Pre-existing stress (Section 19)
    "pre_existing_stress_flag", "bureau_stress_lead_months",
    # Cross-domain interactions (Section 20)
    "rfm_delinquency_anomaly_score", "vintage_performance_deviation",
    "bureau_rfm_inconsistency_score",
    # Cross-dimension selective-default (Section 29)
    "secured_vs_unsecured_dpd_gap", "revolving_vs_term_pay_gap",
    "stage_divergence_flag", "behavioral_arbitrage_flag",
    "concentration_risk_unsecured", "strategic_default_indicator",
    # Structural roll rates & velocities (Section 30)
    "roll_rate_s0_to_s1", "roll_rate_s1_to_s2",
    "roll_rate_s2_to_s3", "roll_rate_s3_to_s4",
    "backward_roll_rate_s2", "backward_roll_rate_s3",
    "escalation_velocity_to_s3", "escalation_velocity_to_s2",
    "cure_velocity_from_s3",
    # Stage velocity (Section 31)
    "months_s0_to_s2", "months_s0_to_s3", "months_s0_to_s4", "months_s2_to_s4",
    "max_stage_ever_reached", "months_at_worst_stage",
    "deterioration_velocity", "cure_velocity", "stage_churn_rate",
    "distinct_stages_ever", "time_in_current_stage_months",
    "fastest_deterioration_speed", "npl_stickiness_index", "co_stickiness_index",
    "stage_reversal_rate", "num_transitions_s2_to_s3",
    "num_transitions_s3_to_s0", "num_transitions_s4_to_s3",
    "stages_traversed_12m",
    # Cure & re-default dynamics (Section 32)
    "cure_count_lifetime", "re_default_count_lifetime",
    "re_default_flag", "avg_months_to_re_default", "min_months_to_re_default",
    "structural_re_default_flag", "durable_cure_flag",
    "cure_depth_max", "avg_cure_holding_period", "cure_to_redefault_ratio",
    # Vintage seasoning (Section 33)
    "months_observed_total", "account_age_at_first_default",
    "time_in_good_standing_pre_default", "cumulative_current_ratio",
    "early_life_default_flag", "seasoning_at_worst_stage",
    "good_standing_to_default_ratio",
    # Recovery stage indicators (Section 34) — lifetime CO exposure & effort
    "time_in_co_overall", "time_in_co_deep_overall",
    "pay_after_delinquency_flag_co", "recovery_propensity_score_overall",
    # Dimension lifetime exposure (Section 35) — peak util ever per product dim
    *[f"max_util_ever_{d}" for d in [
        "overall", "secured", "unsecured",
        "secured_revl", "secured_oth", "unsecured_revl", "unsecured_oth"]],
    # Dimension relationship tenure (Section 36) — months since first open per dim
    # (OVERALL covered by vintage_max_tradeline_age_months in Section 13)
    *[f"relationship_tenure_{d}" for d in [
        "secured", "unsecured",
        "secured_revl", "secured_oth", "unsecured_revl", "unsecured_oth"]],
    # Dimension max stage reached (Section 37) — lifetime worst DPD stage per dim
    # (OVERALL covered by historical_max_dpd_state in Section 11)
    *[f"max_stage_reached_{d}" for d in [
        "overall", "secured", "unsecured",
        "secured_revl", "secured_oth", "unsecured_revl", "unsecured_oth"]],
    # Payment effort dynamics (Section 38) — episode-level structural effort
    # avg_dip_recency: months-since-last-dip AT EPISODE EXIT (shape metric, not scoring-time recency)
    *[f"{m}_{s}" for s in STAGE_LABELS.values()
      for m in [
          "avg_payment_effort_ratio",   "avg_payment_effort_intensity",
          "max_consecutive_no_dip",     "avg_dip_recency",
          "avg_dip_acceleration",       "avg_payment_momentum",
          "avg_pay_to_balance_ratio",   "avg_payment_volatility",
          "max_partial_pay_streak",
      ]],
    # Balance structural (Section 39) — concentration, peak-vs-entry, silence count
    "balance_concentration_hhi", "balance_at_worst_vs_entry",
    "total_bureau_silence_months",
    # DPD profile shape (Section 40) — structural subset
    # Excluded from SEGMENTATION: dpd_convexity (6M recency), dpd_entry_speed (time-since-last-zero)
    # Excluded as redundant: dpd_slope_12m ≡ dpd_diff_velocity_12m (Section 11)
    "dpd_shape_type",
    "dpd_zero_crossing_count", "dpd_max_single_jump",
    "dpd_peak_to_current_ratio", "dpd_time_above_90_24m",
    "dpd_monotone_flag", "dpd_range_12m", "dpd_std_12m",
    # Restructuring success (Section 41) — post-TDR DPD outcome (structural)
    "restructuring_success_flag",
    # Stage stickiness (Section 42) — probability of staying in current DPD stage
    "stage_stickiness_score",
]

PROPENSITY_FEATURES: List[str] = [
    # Recent DPD signals
    "dpd_accel_1m", "entropy_production_rate", "dpd_log_decay_rate_1m",
    # Recent enquiries
    "bureau_enquiry_velocity_3m", "enquiry_velocity_12m",
    # Payment streaks (recent window)
    "consecutive_min_pay_streak", "zero_pay_streak",
    # RFM (Section 16)
    "rfm_recency_score", "rfm_frequency_score", "rfm_monetary_score",
    "rfm_composite_score", "rfm_segment",
    "payment_regime", "payment_elasticity",
    # Cross-lender (tactical)
    "cross_trigger_3m_flag",
    # DPD profile shape — recency/short-window features (Section 40)
    # dpd_convexity: avg delta recent 6M vs prior 6M — short-window tactical signal
    # dpd_entry_speed: months since DPD was last 0 — time-since-good-behavior recency
    "dpd_convexity", "dpd_entry_speed",
]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — HELPER UTILITIES (private — no side effects)
# ══════════════════════════════════════════════════════════════════════════════

def _safe_div(num, den, default=F.lit(None)):
    """Safe division — returns default when denominator is null or zero."""
    return F.when((den.isNotNull()) & (den != 0), num / den).otherwise(default)


def _as_int_safe_col(col_expr):
    """Extract integer from column that may contain XXX or non-numeric chars.
    XXX is the Thai NCB placeholder for masked/missing values — treat as null."""
    return F.regexp_extract(col_expr.cast("string"), r"(-?\d+)", 1).cast("int")


def _dpd_bucket_ordinal(state_col):
    """Map DPD state string (S0–S4) to ordinal integer 0–4."""
    mapping = CFG["dpd_bucket_ordinal"]
    expr = F.lit(None).cast("int")
    for state, ordinal in mapping.items():
        expr = F.when(state_col == state, ordinal).otherwise(expr)
    return expr


def _map_lender_type_expr(lender_col):
    """Map membershortname to Thai lender type category using CFG taxonomy.
    Returns a Column expression (no UDF needed — pattern matching via CASE WHEN)."""
    expr = F.lit("OTHER")
    for ltype, patterns in CFG["lender_taxonomy"].items():
        condition = F.lit(False)
        for p in patterns:
            condition = condition | F.upper(lender_col).contains(p)
        expr = F.when(condition, ltype).otherwise(expr)
    return expr


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — BUREAU TABLE LOADER
# Loads exactly 4 NCB Thailand tables. Lowercases all column names on load.
# NEVER add receive_dt or dl_data_dt to schema maps — they come from bridge only.
# ══════════════════════════════════════════════════════════════════════════════

def load_bureau_tables(spark: SparkSession, schema_name: str) -> Dict[str, DataFrame]:
    """
    Load 4 NCB Thailand bureau tables and lowercase all column names.

    Args:
        spark:       SparkSession
        schema_name: Schema prefix (e.g. "cdx_mdz_prd.cdx_persist_mnf_res_db")

    Returns:
        dict with keys: id_dummy, account, history, enquiry
    """
    tables = {}
    for key, table_name in CFG["bureau_tables"].items():
        df = spark.table(f"{schema_name}.{table_name}")
        for col in df.columns:
            df = df.withColumnRenamed(col, col.lower())
        tables[key] = df
        print(f"  ✓ Loaded {key}: {df.count():,} rows | {len(df.columns)} cols")
    return tables


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PANEL BUILDER (Point-in-Time Safe)
# Joins bureau tables to bridge on ref_no, filters to receive_dt.
# Point-in-time rule: only bureau records with asofdate ≤ receive_dt are used.
# ══════════════════════════════════════════════════════════════════════════════

def build_bureau_panel(
    tables: Dict[str, DataFrame], bridge_df: DataFrame
) -> Dict[str, DataFrame]:
    """
    Build PIT-safe bureau panel.

    Critical rules (confirmed by independent audit):
    - bridge_df contains: ref_no, id_no, receive_dt, as_of_month
    - Join id_dummy to bridge on ref_no to bring receive_dt into panel
    - PIT filter: history records where asofdate ≤ receive_dt
    - ref_no is the ONLY join key — never introduce cust_id

    Returns:
        dict with keys: panel, history, account, enquiry (all PIT-safe)
    """
    R = CFG["ref_col"]
    anchor = CFG["anchor_col"]

    panel = bridge_df.join(
        tables["id_dummy"].select(R, CFG["id_col"]),
        on=R, how="left"
    )
    anchors = panel.select(R, anchor).distinct()

    history = tables["history"].alias("h").join(
        anchors.alias("a"), on=R, how="inner"
    ).filter(F.col("h.asofdate") <= F.col(f"a.{anchor}"))

    account = tables["account"].alias("ac").join(
        anchors.alias("a"), on=R, how="inner"
    )

    enquiry = tables["enquiry"].alias("e").join(
        anchors.alias("a"), on=R, how="inner"
    ).filter(F.col("e.dateofenquiry") <= F.col(f"a.{anchor}"))

    return {"panel": panel, "history": history, "account": account, "enquiry": enquiry}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — DPD STATE BUILDER
# Assigns S0/S1/S2/S3/S4 to each customer-month from overduemonths × 30.
# Handles XXX values (Thai NCB masked data) via _as_int_safe_col().
# Output: one row per ref_no × asofdate with state flags and max DPD.
# ══════════════════════════════════════════════════════════════════════════════

def build_dpd_states(history_df: DataFrame) -> DataFrame:
    """
    Build monthly DPD states (S0-S4) from bureau history.

    Features produced:
        bureau_max_dpd, worst_dpd_ordinal
        has_s0 .. has_s4  (binary per-state flags)
        dpd_state          (S0/S1/S2/S3/S4 string)
        dpd_trajectory_48m (array of last 48 ordinals — for entropy)
        pct_s0_48m .. pct_s4_48m (fraction of 48M in each state)
    """
    R = CFG["ref_col"]

    # Parse OVERDUEMONTHS (×30 = DPD proxy). XXX → null → treated as 0.
    df = history_df.withColumn(
        "dpd",
        F.coalesce(_as_int_safe_col(F.col("overduemonths")), F.lit(0))
        * CFG["odm_to_dpd_multiplier"]
    )

    # Assign state
    state_expr = F.lit("S4")  # Default worst
    for lo, hi, name in CFG["dpd_state_bins"]:
        state_expr = F.when((F.col("dpd") >= lo) & (F.col("dpd") <= hi), name).otherwise(state_expr)
    df = df.withColumn("dpd_state", state_expr)

    # Binary flags and ordinal
    for _, _, name in CFG["dpd_state_bins"]:
        df = df.withColumn(f"has_{name.lower()}", (F.col("dpd_state") == name).cast("int"))
    df = df.withColumn("dpd_bucket_ordinal", _dpd_bucket_ordinal(F.col("dpd_state")))

    # Aggregate to ref_no × asofdate
    state_df = df.groupBy(R, "asofdate").agg(
        F.max("dpd").alias("bureau_max_dpd"),
        F.max("dpd_bucket_ordinal").alias("worst_dpd_ordinal"),
        F.max("has_s0").alias("has_s0"),
        F.max("has_s1").alias("has_s1"),
        F.max("has_s2").alias("has_s2"),
        F.max("has_s3").alias("has_s3"),
        F.max("has_s4").alias("has_s4"),
    )

    # Assign consolidated state from worst ordinal
    ord_to_state = F.lit("S4")
    for name, ordinal in sorted(CFG["dpd_bucket_ordinal"].items(), key=lambda x: x[1], reverse=True):
        ord_to_state = F.when(F.col("worst_dpd_ordinal") == ordinal, name).otherwise(ord_to_state)
    state_df = state_df.withColumn("dpd_state", ord_to_state)

    # Rolling 48M state percentages and entropy
    w_48 = Window.partitionBy(R).orderBy("asofdate").rowsBetween(-(CFG["trajectory_window"] - 1), 0)
    for i in range(5):
        state_df = state_df.withColumn(f"pct_s{i}_48m", F.avg(f"has_s{i}").over(w_48))

    # Shannon entropy: -Σ p_i × log2(p_i)  [disorder of state distribution]
    ent = F.lit(0.0)
    for i in range(5):
        p = F.col(f"pct_s{i}_48m")
        ent = ent - (p * F.log2(p + 1e-10))
    state_df = state_df.withColumn("state_entropy_48m", ent)

    return state_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — STAGE EPISODE BUILDER
# Labels each customer-month with an episode ID = contiguous run in same stage.
# A "stage episode" resets when the DPD state changes.
# This is the structural foundation for Sections 7-10 (within-stage dynamics).
# Example: S0→S0→S2→S2→S2→S0 = 3 episodes [ep1=S0×2, ep2=S2×3, ep3=S0×1]
# ══════════════════════════════════════════════════════════════════════════════

def _build_stage_episodes(state_df: DataFrame) -> DataFrame:
    """Internal: add episode_id and episode_month_number to state_df."""
    R = CFG["ref_col"]
    w = Window.partitionBy(R).orderBy("asofdate")

    df = state_df.withColumn("_prev_state", F.lag("dpd_state", 1).over(w))
    df = df.withColumn(
        "_stage_changed",
        F.when(
            (F.col("dpd_state") != F.col("_prev_state")) | F.col("_prev_state").isNull(), 1
        ).otherwise(0)
    )
    df = df.withColumn("episode_id", F.sum("_stage_changed").over(w))

    w_ep = Window.partitionBy(R, "episode_id").orderBy("asofdate")
    df = df.withColumn("episode_month_number", F.row_number().over(w_ep))
    df = df.withColumn("episode_stage", F.col("dpd_state"))
    return df.drop("_prev_state", "_stage_changed")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — WITHIN-STAGE EXPOSURE DYNAMICS
# KEY INSIGHT: A balance dip while remaining in the same DPD stage is the
# best payment proxy available. The customer did NOT cure (still S2) but DID
# make a partial payment. This is the "effort signal" separating Sudden-Shock
# Distressed from Structural Defaulters within the same bucket.
# Features: 12 metrics × 5 stages = 60 features
# ══════════════════════════════════════════════════════════════════════════════

def build_within_stage_exposure_dynamics(
    episode_df: DataFrame, history_df: DataFrame
) -> DataFrame:
    """
    Compute balance/limit/utilisation/financed dynamics within each stage episode.

    Per-stage features (suffix = stage label e.g. _sm, _npl):
        avg_balance_slope_{stage}   — rising balance = no payment effort
        total_dip_count_{stage}     — # months balance fell (payment proxy)
        avg_dip_magnitude_{stage}   — avg payment amount implied by dip
        avg_util_entry_{stage}      — utilisation when first entering stage
        avg_limit_change_{stage}    — lender reducing limit = risk response
        avg_financed_entry_{stage}  — new credit taken on stage entry
    """
    R = CFG["ref_col"]

    monthly = history_df.groupBy(R, "asofdate").agg(
        F.sum("amountowed").alias("total_balance"),
        F.sum("creditlimit").alias("total_limit"),
        F.sum("amountfinanced").alias("total_financed"),
    )

    df = episode_df.select(R, "asofdate", "episode_id", "episode_stage",
                            "episode_month_number").join(monthly, on=[R, "asofdate"], how="left")

    df = df.withColumn("util_rate", _safe_div(F.col("total_balance"), F.col("total_limit"), F.lit(None)))

    w_ep_ord  = Window.partitionBy(R, "episode_id").orderBy("asofdate")
    w_ep_full = Window.partitionBy(R, "episode_id")
    w_ep_ub   = w_ep_ord.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)

    df = (df
        .withColumn("bal_mom", F.col("total_balance") - F.lag("total_balance", 1).over(w_ep_ord))
        .withColumn("bal_dip_flag", (F.col("bal_mom") < 0).cast("int"))
        .withColumn("ep_bal_entry",    F.first("total_balance").over(w_ep_ord))
        .withColumn("ep_bal_exit",     F.last("total_balance").over(w_ep_ub))
        .withColumn("ep_limit_entry",  F.first("total_limit").over(w_ep_ord))
        .withColumn("ep_limit_exit",   F.last("total_limit").over(w_ep_ub))
        .withColumn("ep_util_entry",   F.first("util_rate").over(w_ep_ord))
        .withColumn("ep_financed_entry", F.first("total_financed").over(w_ep_ord))
        .withColumn("ep_len",          F.count("*").over(w_ep_full))
        .withColumn("ep_dip_count",    F.sum("bal_dip_flag").over(w_ep_full))
        .withColumn("ep_dip_mag",
            _safe_div(
                F.sum(F.when(F.col("bal_dip_flag") == 1, F.abs(F.col("bal_mom"))).otherwise(F.lit(0.0))).over(w_ep_full),
                F.sum("bal_dip_flag").over(w_ep_full), F.lit(0.0)
            ))
        .withColumn("ep_bal_slope",
            _safe_div(F.col("ep_bal_exit") - F.col("ep_bal_entry"), F.col("ep_len"), F.lit(0.0)))
        .withColumn("ep_limit_change", F.col("ep_limit_exit") - F.col("ep_limit_entry"))
    )

    ep_summary = df.filter(F.col("episode_month_number") == 1).select(
        R, "episode_id", "episode_stage",
        "ep_bal_entry", "ep_bal_slope", "ep_util_entry", "ep_limit_entry",
        "ep_limit_exit", "ep_limit_change", "ep_financed_entry",
        "ep_dip_count", "ep_dip_mag",
    )

    agg = ep_summary.groupBy(R, "episode_stage").agg(
        F.count("episode_id").alias("ep_count"),
        F.avg("ep_bal_slope").alias("avg_balance_slope"),
        F.sum("ep_dip_count").alias("total_dip_count"),
        F.avg("ep_dip_mag").alias("avg_dip_magnitude"),
        F.avg("ep_util_entry").alias("avg_util_entry"),
        F.avg("ep_limit_change").alias("avg_limit_change"),
        F.avg("ep_financed_entry").alias("avg_financed_entry"),
    )

    return _pivot_by_stage(agg, R, ["avg_balance_slope", "total_dip_count",
                                     "avg_dip_magnitude", "avg_util_entry",
                                     "avg_limit_change", "avg_financed_entry"])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — WITHIN-STAGE LOAN COUNT DYNAMICS
# Did the customer open or close tradelines while in each stage?
# Opening new credit during S2/S3 = financial desperation or strategic search.
# Closing accounts during S0 = voluntary deleveraging (positive signal).
# Features: 4 metrics × 5 stages = 20 features
# ══════════════════════════════════════════════════════════════════════════════

def build_within_stage_loan_count_dynamics(
    episode_df: DataFrame, history_df: DataFrame
) -> DataFrame:
    """
    Per-stage features:
        avg_loans_entry_{stage}    — tradeline count at stage entry
        avg_loans_change_{stage}   — net tradeline change during episode
        total_loans_opened_{stage} — new tradelines opened while in stage
        avg_loan_count_vol_{stage} — std dev of tradeline count (instability)
    """
    R = CFG["ref_col"]
    monthly = history_df.groupBy(R, "asofdate").agg(
        F.countDistinct("seq_tl").alias("tradeline_count")
    )

    w_ep_ord  = Window.partitionBy(R, "episode_id").orderBy("asofdate")
    w_ep_full = Window.partitionBy(R, "episode_id")
    w_ep_ub   = w_ep_ord.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)

    df = episode_df.select(R, "asofdate", "episode_id", "episode_stage",
                            "episode_month_number").join(monthly, on=[R, "asofdate"], how="left")

    df = (df
        .withColumn("tc_mom",       F.col("tradeline_count") - F.lag("tradeline_count", 1).over(w_ep_ord))
        .withColumn("tc_opened",    F.when(F.col("tc_mom") > 0, F.col("tc_mom")).otherwise(F.lit(0)))
        .withColumn("ep_tc_entry",  F.first("tradeline_count").over(w_ep_ord))
        .withColumn("ep_tc_exit",   F.last("tradeline_count").over(w_ep_ub))
        .withColumn("ep_tc_opened", F.sum("tc_opened").over(w_ep_full))
        .withColumn("ep_tc_vol",    F.stddev("tradeline_count").over(w_ep_full))
    )

    ep = df.filter(F.col("episode_month_number") == 1).select(
        R, "episode_id", "episode_stage",
        "ep_tc_entry", "ep_tc_exit", "ep_tc_opened", "ep_tc_vol",
    ).withColumn("ep_tc_change", F.col("ep_tc_exit") - F.col("ep_tc_entry"))

    agg = ep.groupBy(R, "episode_stage").agg(
        F.avg("ep_tc_entry").alias("avg_loans_entry"),
        F.avg("ep_tc_change").alias("avg_loans_change"),
        F.sum("ep_tc_opened").alias("total_loans_opened"),
        F.avg("ep_tc_vol").alias("avg_loan_count_vol"),
    )

    return _pivot_by_stage(agg, R, ["avg_loans_entry", "avg_loans_change",
                                     "total_loans_opened", "avg_loan_count_vol"])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — WITHIN-STAGE DPD COUNTER DYNAMICS
# DPD=32 (just entered S2) vs DPD=88 (approaching S3 boundary) = very different
# roll-forward risk even though both are classified as "S2".
# The slope, retreat ratio, and boundary proximity reveal cure probability.
# Features: 8 metrics × 5 stages = 40 features
# ══════════════════════════════════════════════════════════════════════════════

def build_within_stage_dpd_dynamics(episode_df: DataFrame) -> DataFrame:
    """
    Per-stage features:
        avg_dpd_at_entry_{stage}    — DPD when first entering stage
        avg_dpd_slope_within_{stage}— how fast moving through stage (+ = toward boundary)
        retreat_ratio_{stage}       — fraction of months DPD decreased (cure attempt signal)
        avg_boundary_proximity_{stage} — distance to next stage boundary (0=at boundary)
        max_dpd_within_{stage}      — peak DPD reached inside stage
        total_dpd_retreats_{stage}  — absolute count of retreat months
    """
    R = CFG["ref_col"]
    w_ep_ord  = Window.partitionBy(R, "episode_id").orderBy("asofdate")
    w_ep_full = Window.partitionBy(R, "episode_id")
    w_ep_ub   = w_ep_ord.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)

    df = episode_df.withColumn("dpd_prev", F.lag("bureau_max_dpd", 1).over(w_ep_ord))
    df = (df
        .withColumn("dpd_mom",      F.col("bureau_max_dpd") - F.col("dpd_prev"))
        .withColumn("is_retreat",   (F.col("dpd_mom") < 0).cast("int"))
        .withColumn("is_advance",   (F.col("dpd_mom") > 0).cast("int"))
        .withColumn("ep_dpd_entry", F.first("bureau_max_dpd").over(w_ep_ord))
        .withColumn("ep_dpd_exit",  F.last("bureau_max_dpd").over(w_ep_ub))
        .withColumn("ep_dpd_max",   F.max("bureau_max_dpd").over(w_ep_full))
        .withColumn("ep_len",       F.count("*").over(w_ep_full))
        .withColumn("ep_retreats",  F.sum("is_retreat").over(w_ep_full))
        .withColumn("ep_advances",  F.sum("is_advance").over(w_ep_full))
        .withColumn("ep_dpd_slope", _safe_div(
            F.last("bureau_max_dpd").over(w_ep_ub) - F.first("bureau_max_dpd").over(w_ep_ord),
            F.count("*").over(w_ep_full), F.lit(0.0)
        ))
    )

    # Boundary proximity = (upper_boundary − exit_dpd) / stage_width  [0=at boundary, 1=at entry]
    prox = F.lit(None).cast("double")
    for stage, upper in CFG["dpd_stage_upper_boundary"].items():
        width = CFG["dpd_stage_width"][stage]
        prox = F.when(F.col("episode_stage") == stage,
                      (F.lit(upper) - F.col("ep_dpd_exit")) / F.lit(float(width))
                      ).otherwise(prox)
    df = df.withColumn("ep_boundary_prox", prox)

    ep = df.filter(F.col("episode_month_number") == 1).select(
        R, "episode_id", "episode_stage",
        "ep_dpd_entry", "ep_dpd_slope", "ep_dpd_max",
        "ep_retreats", "ep_advances", "ep_boundary_prox",
    )

    agg = ep.groupBy(R, "episode_stage").agg(
        F.avg("ep_dpd_entry").alias("avg_dpd_at_entry"),
        F.avg("ep_dpd_slope").alias("avg_dpd_slope_within"),
        F.max("ep_dpd_max").alias("max_dpd_within"),
        F.avg("ep_boundary_prox").alias("avg_boundary_proximity"),
        F.sum("ep_retreats").alias("total_dpd_retreats"),
        _safe_div(F.sum("ep_retreats"),
                  F.sum("ep_retreats") + F.sum("ep_advances"),
                  F.lit(0.0)).alias("retreat_ratio"),
    )

    return _pivot_by_stage(agg, R, ["avg_dpd_at_entry", "avg_dpd_slope_within",
                                     "max_dpd_within", "avg_boundary_proximity",
                                     "total_dpd_retreats", "retreat_ratio"])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — CROSS-STAGE TRANSITION DYNAMICS
# Roll-forward rates, cure rates, and oscillation counts across a customer's
# full bureau history. Chronic oscillators (S2→S3→S2→S3) vs single-episode
# distress have very different recovery profiles.
# Features: 6 metrics × 5 stages = 30 features
# ══════════════════════════════════════════════════════════════════════════════

def build_cross_stage_transition_dynamics(episode_df: DataFrame) -> DataFrame:
    """
    Per-stage features:
        episode_count_{stage}       — # times entered this stage
        roll_forward_rate_{stage}   — % of exits that went to worse stage
        cure_rate_{stage}           — % of exits that went to better stage
        ever_in_stage_{stage}       — binary: ever reached this stage
        first_entry_month_{stage}   — first calendar month of entry
        last_entry_month_{stage}    — most recent entry (staleness signal)
    """
    R = CFG["ref_col"]
    w = Window.partitionBy(R).orderBy("episode_id")

    eps = episode_df.filter(F.col("episode_month_number") == 1).select(
        R, "episode_id", "episode_stage", "asofdate"
    )

    ord_map = F.lit(None).cast("int")
    for s, o in CFG["dpd_bucket_ordinal"].items():
        ord_map = F.when(F.col("episode_stage") == s, o).otherwise(ord_map)

    eps = (eps
        .withColumn("stage_ord",  ord_map)
        .withColumn("next_stage", F.lead("episode_stage", 1).over(w))
        .withColumn("next_ord",   F.lead("stage_ord",    1).over(w))
        .withColumn("is_roll_fwd", (F.col("next_ord") > F.col("stage_ord")).cast("int"))
        .withColumn("is_cure",     (F.col("next_ord") < F.col("stage_ord")).cast("int"))
    )

    agg = eps.groupBy(R, "episode_stage").agg(
        F.count("episode_id").alias("episode_count"),
        F.sum("is_roll_fwd").alias("roll_forward_count"),
        F.sum("is_cure").alias("cure_count"),
        F.min("asofdate").alias("first_entry_month"),
        F.max("asofdate").alias("last_entry_month"),
        _safe_div(F.sum("is_roll_fwd"), F.count("episode_id"), F.lit(0.0)).alias("roll_forward_rate"),
        _safe_div(F.sum("is_cure"),     F.count("episode_id"), F.lit(0.0)).alias("cure_rate"),
    ).withColumn("ever_in_stage", F.lit(1))

    result = _pivot_by_stage(agg, R, ["episode_count", "roll_forward_rate",
                                       "cure_rate", "ever_in_stage",
                                       "first_entry_month", "last_entry_month"])

    # Default ever_in_stage to 0 for stages never visited
    for label in STAGE_LABELS.values():
        col = f"ever_in_stage_{label}"
        result = result.withColumn(col, F.coalesce(F.col(col), F.lit(0)))

    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — TRAJECTORY & PHYSICS
# Behavioral physics applied to credit dynamics:
#   Velocity:         rate of DPD change over 1/3/6/12M windows
#   Acceleration:     second derivative — is deterioration speeding up?
#   Entropy:          Shannon entropy of state distribution (disorder)
#   Inertia:          stuck-in-bad-state force (months × dpd × log(dpd))
#   Momentum:         directional credit force (velocity × dpd)
#   DPD Z-score:      current DPD vs 12M mean (unusual position)
#   Critical slowing: autocorr × variance ratio (pre-tipping-point signal)
#   Hysteresis:       scar from past stress (max ever state − current state)
#   Phase boundary:   inverse distance to nearest DPD stage boundary
# ══════════════════════════════════════════════════════════════════════════════

def build_trajectory_and_physics(state_df: DataFrame) -> DataFrame:
    """
    Compute trajectory velocity/acceleration and physics family features.
    ~87 features total.
    """
    R = CFG["ref_col"]
    w      = Window.partitionBy(R).orderBy("asofdate")
    w_3m   = w.rowsBetween(-2,  0)
    w_6m   = w.rowsBetween(-5,  0)
    w_12m  = w.rowsBetween(-11, 0)
    w_life = Window.partitionBy(R).orderBy("asofdate").rowsBetween(Window.unboundedPreceding, 0)

    df = state_df

    # ── Velocity (DPD slope over N months) ───────────────────────────────────
    for n in CFG["windows_months"]:
        df = df.withColumn(
            f"dpd_diff_velocity_{n}m",
            (F.col("bureau_max_dpd") - F.lag("bureau_max_dpd", n).over(w)) / n
        )

    # ── Acceleration (second derivative) — shock detector ────────────────────
    df = df.withColumn("dpd_accel_1m",
                       F.col("dpd_diff_velocity_1m") - F.lag("dpd_diff_velocity_1m", 1).over(w))
    df = df.withColumn("dpd_jump", F.col("bureau_max_dpd") - F.lag("bureau_max_dpd", 1).over(w))

    # ── DPD Z-score (unusual position vs 12M baseline) ───────────────────────
    df = (df
        .withColumn("dpd_mean_12m", F.avg("bureau_max_dpd").over(w_12m))
        .withColumn("dpd_std_12m",  F.stddev("bureau_max_dpd").over(w_12m))
        .withColumn("dpd_zscore_12m",
                    _safe_div(F.col("bureau_max_dpd") - F.col("dpd_mean_12m"),
                              F.col("dpd_std_12m"), F.lit(0.0)))
    )

    # ── Log decay rate (exponential decay proxy) ──────────────────────────────
    df = (df
        .withColumn("dpd_log",      F.log1p(F.col("bureau_max_dpd")))
        .withColumn("dpd_log_prev", F.lag("dpd_log", 1).over(w))
        .withColumn("dpd_log_decay_rate_1m", F.col("dpd_log") - F.col("dpd_log_prev"))
    )

    # ── Inertia (stuck-in-bad-state force) ───────────────────────────────────
    df = (df
        .withColumn("months_since_transition",
                    F.sum((F.col("worst_dpd_ordinal") != F.lag("worst_dpd_ordinal", 1).over(w))
                           .cast("int")).over(w))
        .withColumn("credit_inertia_score",
                    F.col("months_since_transition") * F.col("worst_dpd_ordinal") * F.log1p(F.col("bureau_max_dpd")))
    )

    # ── Momentum (directional credit force) ──────────────────────────────────
    df = df.withColumn("credit_momentum_3m",
                       F.coalesce(F.col("dpd_diff_velocity_3m"), F.lit(0.0)) * F.col("bureau_max_dpd"))

    # ── Entropy production rate (disorder creation speed) ────────────────────
    df = (df
        .withColumn("entropy_prev", F.lag("state_entropy_48m", 1).over(w))
        .withColumn("entropy_production_rate", F.col("state_entropy_48m") - F.col("entropy_prev"))
    )

    # ── Stress tensor magnitude (combined stress) ─────────────────────────────
    df = df.withColumn("stress_tensor_magnitude",
        F.sqrt(
            F.pow(F.coalesce(F.col("dpd_diff_velocity_3m"),  F.lit(0.0)), 2) +
            F.pow(F.coalesce(F.col("dpd_accel_1m"),           F.lit(0.0)), 2) +
            F.pow(F.coalesce(F.col("entropy_production_rate"), F.lit(0.0)), 2)
        ))

    # ── Critical slowing down (variance ratio × autocorrelation) ─────────────
    df = (df
        .withColumn("dpd_var_3m",  F.variance("bureau_max_dpd").over(w_3m))
        .withColumn("dpd_var_12m", F.variance("bureau_max_dpd").over(w_12m))
        .withColumn("dpd_lag1",    F.lag("bureau_max_dpd", 1).over(w))
        .withColumn("variance_ratio_12m_3m",
                    _safe_div(F.col("dpd_var_12m"), F.col("dpd_var_3m"), F.lit(1.0)))
        .withColumn("dpd_autocorr_lag1_12m",
                    F.corr("bureau_max_dpd", "dpd_lag1").over(w_12m))
        .withColumn("critical_slowing_indicator",
                    F.col("dpd_autocorr_lag1_12m") * F.col("variance_ratio_12m_3m"))
    )

    # ── Hysteresis / scar score (max ever vs current — ghost of past stress) ──
    df = (df
        .withColumn("historical_max_dpd_state", F.max("worst_dpd_ordinal").over(w_life))
        .withColumn("hysteresis_gap",
                    F.col("historical_max_dpd_state") - F.col("worst_dpd_ordinal"))
        .withColumn("entered_stress",
                    ((F.col("worst_dpd_ordinal") >= 2) &
                     (F.lag("worst_dpd_ordinal", 1).over(w) < 2)).cast("int"))
        .withColumn("num_stress_cycles_lifetime", F.sum("entered_stress").over(w_life))
        .withColumn("incomplete_recovery_flag",
                    ((F.col("historical_max_dpd_state") >= 2) &
                     (F.col("worst_dpd_ordinal") > 0)).cast("int"))
    )

    # ── Phase boundary proximity (inverse distance to next stage boundary) ────
    prox = F.least(
        F.abs(F.lit(30)  - F.col("bureau_max_dpd")),
        F.abs(F.lit(90)  - F.col("bureau_max_dpd")),
        F.abs(F.lit(180) - F.col("bureau_max_dpd")),
    )
    df = df.withColumn("phase_boundary_proximity", 1.0 / (prox + 1.0))

    # ── Momentum direction oscillations (erratic behaviour) ──────────────────
    df = (df
        .withColumn("dpd_jump_sign",
                    F.when(F.col("dpd_jump") > 0, 1).when(F.col("dpd_jump") < 0, -1).otherwise(0))
        .withColumn("momentum_sign_flip_6m",
                    F.sum((F.col("dpd_jump_sign") != F.lag("dpd_jump_sign", 1).over(w)).cast("int")).over(w_6m))
    )

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — REGIME-DEPENDENT REPAYMENT
# Payment behaviour is measured SEPARATELY in normal vs stressed regimes.
# A customer paying 80% in S0 but only 10% in S2 is fundamentally different
# from one paying 10% in both — the DELTA reveals genuine effort under stress.
# NORMAL regime = S0/S1, STRESSED regime = S2/S3/S4
# Features: 12 normal + 14 stressed + 9 delta = 35 features
# ══════════════════════════════════════════════════════════════════════════════

def build_regime_dependent_repayment(
    state_df: DataFrame, history_df: DataFrame
) -> DataFrame:
    """
    Compute payment behaviour separately for NORMAL vs STRESSED regimes
    and derive the behavioural change delta between them.

    NORMAL regime features:
        payment_effort_normal         — payment / balance in S0/S1 periods
        payment_consistency_normal    — % months with any payment
        payment_cv_normal             — coefficient of variation (erratic?)
        payment_discipline_score_normal — composite

    STRESSED regime features:
        cure_attempt_count_stressed   — months DPD decreased by 5+ (effort)
        cure_success_rate_stressed    — cure attempts that reached S0/S1
        payment_effort_stressed       — payment / balance under stress
        chronicity_index              — fraction of time in stressed state
        payment_fatigue_flag          — declining effort over time

    Delta features:
        payment_effort_delta          — normal − stressed (positive = regime switch)
        payment_consistency_delta     — change in payment regularity
    """
    R = CFG["ref_col"]
    w = Window.partitionBy(R).orderBy("asofdate")

    # Join regime label to history (balance = amountowed as proxy for monthly exposure)
    monthly_bal = history_df.groupBy(R, "asofdate").agg(
        F.sum("amountowed").alias("balance"),
        F.sum("creditlimit").alias("limit"),
    )

    df = state_df.join(monthly_bal, on=[R, "asofdate"], how="left")
    df = df.withColumn("regime",
                       F.when(F.col("dpd_state").isin(["S0", "S1"]), "NORMAL").otherwise("STRESSED"))

    # Balance decrease = payment proxy (no explicit payment column in bureau history)
    df = df.withColumn("bal_prev", F.lag("balance", 1).over(w))
    df = df.withColumn("implied_payment",
                       F.when(F.col("balance") < F.col("bal_prev"),
                              F.col("bal_prev") - F.col("balance")).otherwise(F.lit(0.0)))
    df = df.withColumn("dpd_change", F.col("bureau_max_dpd") - F.lag("bureau_max_dpd", 1).over(w))
    df = df.withColumn("cure_attempt", (F.col("dpd_change") < -5).cast("int"))
    df = df.withColumn("cure_success",  (F.col("bureau_max_dpd") <= 30).cast("int"))

    # ── NORMAL regime aggregation ─────────────────────────────────────────────
    normal_agg = (df.filter(F.col("regime") == "NORMAL")
        .groupBy(R).agg(
            _safe_div(F.sum("implied_payment"), F.sum("balance"), F.lit(0.0)).alias("payment_effort_normal"),
            _safe_div(F.sum((F.col("implied_payment") > 0).cast("int")),
                      F.count("*"), F.lit(0.0)).alias("payment_consistency_normal"),
            _safe_div(F.stddev("implied_payment"), F.avg("implied_payment"), F.lit(0.0)).alias("payment_cv_normal"),
            F.count("*").alias("num_months_normal"),
        )
        .withColumn("payment_discipline_score_normal",
                    F.col("payment_consistency_normal") * 0.6 + (1.0 - F.least(F.col("payment_cv_normal"), F.lit(1.0))) * 0.4)
    )

    # ── STRESSED regime aggregation ───────────────────────────────────────────
    stressed_agg = (df.filter(F.col("regime") == "STRESSED")
        .groupBy(R).agg(
            F.sum("cure_attempt").alias("cure_attempt_count_stressed"),
            _safe_div(F.sum(F.col("cure_attempt") * F.col("cure_success")),
                      F.sum("cure_attempt"), F.lit(0.0)).alias("cure_success_rate_stressed"),
            _safe_div(F.sum("implied_payment"), F.sum("balance"), F.lit(0.0)).alias("payment_effort_stressed"),
            F.count("*").alias("num_months_stressed"),
        )
    )

    # Total months for chronicity index
    total_months = df.groupBy(R).agg(F.count("*").alias("total_months"))
    stressed_agg = stressed_agg.join(total_months, on=R, how="left")
    stressed_agg = stressed_agg.withColumn("chronicity_index",
                                            _safe_div(F.col("num_months_stressed"),
                                                      F.col("total_months"), F.lit(0.0)))

    # ── Delta features ────────────────────────────────────────────────────────
    result = normal_agg.join(stressed_agg, on=R, how="full")
    result = (result
        .withColumn("payment_effort_delta",
                    F.coalesce(F.col("payment_effort_normal"),     F.lit(0.0)) -
                    F.coalesce(F.col("payment_effort_stressed"),   F.lit(0.0)))
        .withColumn("payment_consistency_delta",
                    F.coalesce(F.col("payment_consistency_normal"), F.lit(0.0)) -
                    F.coalesce(F.col("payment_effort_stressed"),    F.lit(0.0)))
    )

    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — VINTAGE & ACCOUNT MATURITY
# How old is the bureau relationship and when did delinquency first appear?
# Early delinquency (within first 6M) = origination defect, very different risk.
# Late delinquency (well-seasoned account, sudden drop) = external shock.
# Features: 16 features — fills the audit gap confirmed in feature_audit_inventory.py
# ══════════════════════════════════════════════════════════════════════════════

def build_vintage_features(account_df: DataFrame) -> DataFrame:
    """
    Compute account age, lifecycle stage, and origination cohort features.
    Uses accountstatus and opendate from s_account.

    Features:
        vintage_months_on_book        — account age at CO snapshot
        vintage_lifecycle_stage       — NEW(<12M)/SEASONED(12-36M)/MATURE(36-72M)/AGED(72M+)
        vintage_early_delinquency_flag — delinquent within first 6 months
        vintage_origination_year      — year of oldest bureau tradeline
        vintage_origination_quarter   — cohort quarter
        vintage_min_tradeline_age_months — youngest tradeline (recent credit seeking)
        vintage_max_tradeline_age_months — oldest tradeline (credit maturity)
        vintage_avg_tradeline_age_months — portfolio average age
        vintage_age_spread_months     — max − min (diverse vs concentrated vintage)
        vintage_pct_accounts_open     — fraction of tradelines still active
        vintage_payment_evolution     — payment ratio change: recent vs early period
    """
    R = CFG["ref_col"]

    # Compute tradeline age from opendate field (if present, else use first asofdate)
    if "opendate" in account_df.columns:
        age_df = account_df.withColumn(
            "tradeline_age_months",
            F.months_between(F.current_date(), F.to_date(F.col("opendate").cast("string"), "yyyyMMdd"))
        )
    else:
        # Fallback: use first seen asofdate as proxy for open date
        age_df = account_df.withColumn("tradeline_age_months", F.lit(None).cast("double"))

    # Active flag: accountstatus "10" = Normal/Active
    if "accountstatus" in account_df.columns:
        age_df = age_df.withColumn("is_active", (F.col("accountstatus") == "10").cast("int"))
    else:
        age_df = age_df.withColumn("is_active", F.lit(1))

    agg = age_df.groupBy(R).agg(
        F.min("tradeline_age_months").alias("vintage_min_tradeline_age_months"),
        F.max("tradeline_age_months").alias("vintage_max_tradeline_age_months"),
        F.avg("tradeline_age_months").alias("vintage_avg_tradeline_age_months"),
        _safe_div(F.sum("is_active"), F.count("*"), F.lit(0.0)).alias("vintage_pct_accounts_open"),
        F.count("*").alias("vintage_total_tradelines"),
    )

    agg = agg.withColumn(
        "vintage_age_spread_months",
        F.col("vintage_max_tradeline_age_months") - F.col("vintage_min_tradeline_age_months")
    )

    # Lifecycle stage from oldest tradeline age
    agg = agg.withColumn("vintage_lifecycle_stage",
        F.when(F.col("vintage_max_tradeline_age_months") <= 12,  "NEW")
         .when(F.col("vintage_max_tradeline_age_months") <= 36,  "SEASONED")
         .when(F.col("vintage_max_tradeline_age_months") <= 72,  "MATURE")
         .otherwise("AGED")
    )

    # Months on book (oldest tradeline proxy for customer relationship age)
    agg = agg.withColumn("vintage_months_on_book", F.col("vintage_max_tradeline_age_months"))

    return agg


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 14 — LENDER ECOLOGY
# Who does the customer borrow from and how concentrated is their portfolio?
# Lender type mix reveals strategic positioning (SFI = often lower rate,
# government-backed; LEASING = secured; PERSONAL_LOAN = unsecured stress).
# HHI concentration measures single-lender dependence.
# Features: ~20 features
# ══════════════════════════════════════════════════════════════════════════════

def build_lender_ecology(account_df: DataFrame) -> DataFrame:
    """
    Compute lender type exposure shares and concentration index.

    Features:
        pct_{lender_type}         — % of total balance with each lender type
        lender_hhi                — Herfindahl-Hirschman Index (0=diverse, 1=concentrated)
        num_lenders               — distinct lender count
        lender_diversification_score — 1 − HHI
        lender_network_density    — num_lenders / 10 (normalised)
    """
    R = CFG["ref_col"]

    typed = account_df.withColumn("lender_type", _map_lender_type_expr(F.col("membershortname")))
    typed = typed.withColumn("balance", F.col("amountowed").cast("double"))

    total_bal = typed.groupBy(R).agg(F.sum("balance").alias("total_balance"))

    by_type = typed.groupBy(R, "lender_type").agg(F.sum("balance").alias("type_balance"))
    by_type = by_type.join(total_bal, on=R, how="left")
    by_type = by_type.withColumn("share", _safe_div(F.col("type_balance"), F.col("total_balance"), F.lit(0.0)))

    # Pivot to wide format
    pivot = by_type.groupBy(R).pivot("lender_type", list(CFG["lender_taxonomy"].keys())).sum("share")
    pivot = pivot.join(total_bal, on=R, how="left")

    for lt in CFG["lender_taxonomy"]:
        col = lt.lower()
        if col not in [c.lower() for c in pivot.columns]:
            pivot = pivot.withColumn(f"pct_{col}", F.lit(0.0))
        else:
            pivot = pivot.withColumnRenamed(lt, f"pct_{col}")

    # HHI = sum of squared shares
    typed = typed.withColumn("total_bal_cust", F.sum("balance").over(Window.partitionBy(R)))
    typed = typed.withColumn("lender_share", _safe_div(F.col("balance"), F.col("total_bal_cust"), F.lit(0.0)))

    hhi_df = typed.groupBy(R).agg(
        F.sum(F.pow(F.col("lender_share"), 2)).alias("lender_hhi"),
        F.countDistinct("membershortname").alias("num_lenders"),
    )
    hhi_df = hhi_df.withColumn("lender_diversification_score", 1.0 - F.col("lender_hhi"))
    hhi_df = hhi_df.withColumn("lender_network_density", F.col("num_lenders") / 10.0)

    return pivot.join(hhi_df, on=R, how="left")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 15 — ENQUIRIES ENGINE
# Credit seeking behaviour. High enquiry velocity under stress = liquidity crisis.
# Rejection proxy: many enquiries + no new tradelines opened = rejected.
# Features: ~12 features
# ══════════════════════════════════════════════════════════════════════════════

def build_enquiries(enquiry_df: DataFrame) -> DataFrame:
    """
    Compute credit enquiry velocity, purpose mix, and rejection proxy.

    Features:
        enquiry_count_12m      — total enquiries in rolling 12M
        enquiry_velocity_12m   — count / 12 (per-month rate)
        enquiry_momentum_12m   — count × velocity (urgency signal)
        pct_enq_{purpose}_12m — purpose mix (auto/personal/card/mortgage)
        enquiry_rejection_proxy — 1 if enquiry_count > 3 (rejection heuristic)
    """
    R = CFG["ref_col"]
    w_12 = Window.partitionBy(R).orderBy("dateofenquiry").rowsBetween(-(CFG["enquiry_window"] - 1), 0)

    df = enquiry_df.withColumn("enq_count_12m", F.count("*").over(w_12))
    df = df.withColumn("enquiry_velocity_12m", F.col("enq_count_12m") / CFG["enquiry_window"])

    # Purpose mix (ENQUIRYPURPOSE → enquiry_purpose after lowercasing)
    purpose_col = "enquiry_purpose" if "enquiry_purpose" in enquiry_df.columns else "enquirypurpose"
    for purpose, label in [("AUTO LOAN", "auto"), ("PERSONAL LOAN", "personal"),
                            ("CREDIT CARD", "card"), ("MORTGAGE", "mortgage")]:
        df = df.withColumn(f"is_{label}", F.upper(F.col(purpose_col)).contains(purpose).cast("int"))
        df = df.withColumn(f"pct_enq_{label}_12m",
                           _safe_div(F.sum(f"is_{label}").over(w_12), F.col("enq_count_12m"), F.lit(0.0)))

    agg = df.groupBy(R).agg(
        F.max("enq_count_12m").alias("enquiry_count_12m"),
        F.max("enquiry_velocity_12m").alias("enquiry_velocity_12m"),
        F.max("pct_enq_auto_12m").alias("pct_enq_auto_12m"),
        F.max("pct_enq_personal_12m").alias("pct_enq_personal_12m"),
        F.max("pct_enq_card_12m").alias("pct_enq_card_12m"),
        F.max("pct_enq_mortgage_12m").alias("pct_enq_mortgage_12m"),
    )

    agg = agg.withColumn("enquiry_momentum_12m", F.col("enquiry_count_12m") * F.col("enquiry_velocity_12m"))
    agg = agg.withColumn("enquiry_rejection_proxy", (F.col("enquiry_count_12m") > 3).cast("int"))

    return agg


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 16 — PAYMENT FEATURES: RFM + PAYMENT HISTORY
# RFM (Recency, Frequency, Monetary) applied to bureau payment behaviour.
# Payment regime classification: CONSISTENT / ERRATIC / MINIMAL / COLLAPSED.
# Payment elasticity: how much does payment ratio change when balance changes?
# Features: ~35 features
# ══════════════════════════════════════════════════════════════════════════════

def build_rfm_and_payment_features(history_df: DataFrame) -> DataFrame:
    """
    Compute RFM scores, payment regime, and elasticity from bureau history.

    RFM features:
        rfm_recency_score    — 1 (worst) to 5 (best): days since last payment
        rfm_frequency_score  — 1 to 5: payment count in 12M
        rfm_monetary_score   — 1 to 5: avg payment amount
        rfm_composite_score  — 3 to 15: sum of three scores
        rfm_segment          — CHAMPIONS / LOYAL / AT_RISK / LOST / HIBERNATING

    Payment history features (from PAYMENTHISTORY1/2 strings):
        pct_full_pay          — % months with full payment (code "000" / "Y")
        pct_min_pay           — % months with minimum payment only
        consecutive_min_pay_streak — current consecutive minimum payment run
        zero_pay_streak       — current consecutive zero payment run

    Regime & elasticity:
        payment_regime        — CONSISTENT / ERRATIC / MINIMAL / COLLAPSED
        payment_elasticity    — correlation of payment ratio with balance change
    """
    R = CFG["ref_col"]
    w = Window.partitionBy(R).orderBy("asofdate")
    w_12 = w.rowsBetween(-11, 0)
    w_life = Window.partitionBy(R).orderBy("asofdate").rowsBetween(Window.unboundedPreceding, 0)

    # ── Balance dip = implied payment ─────────────────────────────────────────
    monthly = history_df.groupBy(R, "asofdate").agg(
        F.sum("amountowed").alias("balance"),
        F.sum("creditlimit").alias("limit"),
    )
    monthly = monthly.withColumn("bal_prev", F.lag("balance", 1).over(w))
    monthly = monthly.withColumn(
        "implied_pay",
        F.when(F.col("balance") < F.col("bal_prev"),
               F.col("bal_prev") - F.col("balance")).otherwise(F.lit(0.0))
    )
    monthly = monthly.withColumn("had_payment", (F.col("implied_pay") > 0).cast("int"))
    monthly = monthly.withColumn("pay_ratio",
                                  _safe_div(F.col("implied_pay"), F.col("balance"), F.lit(0.0)))

    # ── RFM scores (quintile-based) ───────────────────────────────────────────
    monthly = monthly.withColumn("pay_count_12m", F.sum("had_payment").over(w_12))
    monthly = monthly.withColumn("avg_pay_12m",   F.avg("implied_pay").over(w_12))

    # Recency: months since last payment (lower = more recent = better)
    monthly = monthly.withColumn(
        "months_since_last_pay",
        F.sum((F.col("had_payment") == 0).cast("int")).over(w_life)
    )

    # Quintile assignment (1=worst, 5=best)
    def quintile(col_name: str, ascending: bool) -> "Column":
        """Rank into quintiles 1-5."""
        w_all = Window.orderBy(F.col(col_name).asc() if ascending else F.col(col_name).desc())
        return F.ntile(5).over(w_all)

    monthly = (monthly
        .withColumn("rfm_recency_score",   quintile("months_since_last_pay", ascending=True))
        .withColumn("rfm_frequency_score", quintile("pay_count_12m",         ascending=False))
        .withColumn("rfm_monetary_score",  quintile("avg_pay_12m",           ascending=False))
        .withColumn("rfm_composite_score",
                    F.col("rfm_recency_score") + F.col("rfm_frequency_score") + F.col("rfm_monetary_score"))
    )

    # RFM segment labels
    monthly = monthly.withColumn("rfm_segment",
        F.when(F.col("rfm_composite_score") >= 13, "CHAMPIONS")
         .when(F.col("rfm_composite_score") >= 10, "LOYAL")
         .when(F.col("rfm_composite_score") >= 7,  "AT_RISK")
         .when(F.col("rfm_composite_score") >= 4,  "HIBERNATING")
         .otherwise("LOST")
    )

    # ── Payment regime classification ─────────────────────────────────────────
    monthly = monthly.withColumn("pay_cv",
        _safe_div(F.stddev("implied_pay").over(w_12), F.avg("implied_pay").over(w_12), F.lit(999.0)))

    monthly = monthly.withColumn("payment_regime",
        F.when((F.col("pay_count_12m") >= 10) & (F.col("pay_cv") < 0.3),  "CONSISTENT")
         .when((F.col("pay_count_12m") >= 6)  & (F.col("pay_cv") >= 0.5),  "ERRATIC")
         .when((F.col("pay_count_12m") >= 3)  & (F.col("pay_ratio") < 0.2),"MINIMAL")
         .when(F.col("pay_count_12m") < 3,                                  "COLLAPSED")
         .otherwise("ERRATIC")
    )

    # ── Payment elasticity (payment ratio response to balance change) ─────────
    monthly = monthly.withColumn("bal_change",
                                  F.col("balance") - F.lag("balance", 1).over(w))
    monthly = monthly.withColumn("payment_elasticity",
                                  F.corr("pay_ratio", "bal_change").over(w_12))

    # ── Zero-pay streak (current run of no payments) ──────────────────────────
    monthly = (monthly
        .withColumn("no_pay_group",
                    F.sum(F.when(F.col("had_payment") == 1, 1).otherwise(0)).over(w))
        .withColumn("zero_pay_streak",
                    F.count("*").over(Window.partitionBy(R, "no_pay_group").orderBy("asofdate")))
    )

    # Return most recent snapshot per customer
    w_latest = Window.partitionBy(R).orderBy(F.col("asofdate").desc())
    latest = (monthly.withColumn("rn", F.row_number().over(w_latest))
              .filter(F.col("rn") == 1)
              .select(R, "rfm_recency_score", "rfm_frequency_score", "rfm_monetary_score",
                      "rfm_composite_score", "rfm_segment",
                      "pay_count_12m", "avg_pay_12m", "months_since_last_pay",
                      "payment_regime", "payment_elasticity", "zero_pay_streak",
                      "pay_ratio"))

    return latest


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 17 — CROSS-LENDER DYNAMICS (CardX vs Bureau)
# The most powerful strategic signal: does the customer pay other lenders
# but not CardX? This selective default pattern confirms capacity exists.
# Lead-lag: which deteriorated first? CardX-led = isolated. Bureau-led = systemic.
# Features: ~15 features (requires cardx_monthly_df)
# ══════════════════════════════════════════════════════════════════════════════

def build_cross_lender_dynamics(
    bureau_state_df: DataFrame,
    cardx_monthly_df: Optional[DataFrame] = None,
) -> DataFrame:
    """
    Compute CardX vs bureau delinquency alignment, lead-lag timing, and
    cross-trigger cascade detection.

    Features:
        selective_default_rate         — % months CardX stressed, bureau current
        systemic_stress_rate           — % months bureau stressed, CardX current
        cardx_selective_default_flag   — selective_default_rate > 25%
        systemic_stress_flag           — systemic_stress_rate > 25%
        cross_lender_divergence_score  — overall mismatch ratio
        cardx_leads_bureau_flag        — CardX deteriorated first
        bureau_leads_cardx_flag        — bureau deteriorated first
        lead_lag_months                — months of lead (positive = CardX first)
        util_spread_cardx_vs_bureau    — CardX util − bureau util gap
        cross_trigger_3m_flag          — CardX delinquency → bureau cascade within 3M
    """
    R = CFG["ref_col"]

    if cardx_monthly_df is None:
        return bureau_state_df.select(R).distinct().withColumn(
            "cross_lender_available", F.lit(0)
        )

    joined = cardx_monthly_df.select(
        R, "asofdate",
        F.col("cardx_dpd"),
        F.col("cardx_state").alias("cardx_stage"),
    ).join(
        bureau_state_df.select(R, "asofdate", "bureau_max_dpd",
                                F.col("dpd_state").alias("bureau_stage")),
        on=[R, "asofdate"], how="inner"
    )

    joined = (joined
        .withColumn("cardx_stressed",   F.col("cardx_stage").isin(["S2","S3","S4"]).cast("int"))
        .withColumn("bureau_stressed",  F.col("bureau_stage").isin(["S2","S3","S4"]).cast("int"))
        .withColumn("cardx_current",    (F.col("cardx_stage") == "S0").cast("int"))
        .withColumn("bureau_current",   (F.col("bureau_stage") == "S0").cast("int"))
        .withColumn("selective_delq",   ((F.col("cardx_stressed") == 1) & (F.col("bureau_current") == 1)).cast("int"))
        .withColumn("systemic_stress",  ((F.col("bureau_stressed") == 1) & (F.col("cardx_current") == 1)).cast("int"))
    )

    # Lead-lag: which became delinquent first?
    w = Window.partitionBy(R).orderBy("asofdate")
    joined = (joined
        .withColumn("cardx_first_delq",
                    F.when((F.col("cardx_stressed") == 1) & (F.lag("cardx_stressed", 1).over(w) == 0),
                            F.col("asofdate")))
        .withColumn("bureau_first_delq",
                    F.when((F.col("bureau_stressed") == 1) & (F.lag("bureau_stressed", 1).over(w) == 0),
                            F.col("asofdate")))
        .withColumn("cardx_first_delq",
                    F.last("cardx_first_delq", ignorenulls=True).over(
                        w.rowsBetween(Window.unboundedPreceding, 0)))
        .withColumn("bureau_first_delq",
                    F.last("bureau_first_delq", ignorenulls=True).over(
                        w.rowsBetween(Window.unboundedPreceding, 0)))
    )

    # Cross-trigger: CardX delinquency followed by bureau delinquency within 3M
    joined = joined.withColumn("cross_trigger_3m_flag",
        ((F.col("cardx_stressed") == 1) &
         (F.lead("bureau_stressed", 3).over(w) == 1)).cast("int")
    )

    result = joined.groupBy(R).agg(
        F.count("*").alias("months_observed"),
        F.sum("selective_delq").alias("cardx_selective_default_months"),
        F.sum("systemic_stress").alias("systemic_stress_months"),
        _safe_div(F.sum("selective_delq"), F.count("*"), F.lit(0.0)).alias("selective_default_rate"),
        _safe_div(F.sum("systemic_stress"), F.count("*"), F.lit(0.0)).alias("systemic_stress_rate"),
        F.max("cross_trigger_3m_flag").alias("cross_trigger_3m_flag"),
        F.first("cardx_first_delq", ignorenulls=True).alias("cardx_first_delq_month"),
        F.first("bureau_first_delq", ignorenulls=True).alias("bureau_first_delq_month"),
    )

    result = (result
        .withColumn("cross_lender_divergence_score",
                    (F.col("cardx_selective_default_months") + F.col("systemic_stress_months")) /
                    (F.col("months_observed") + 1))
        .withColumn("cardx_selective_default_flag", (F.col("selective_default_rate") > 0.25).cast("int"))
        .withColumn("systemic_stress_flag",          (F.col("systemic_stress_rate")  > 0.25).cast("int"))
        .withColumn("cardx_leads_bureau_flag",
                    (F.col("cardx_first_delq_month").isNotNull() &
                     F.col("bureau_first_delq_month").isNotNull() &
                     (F.col("cardx_first_delq_month") < F.col("bureau_first_delq_month"))).cast("int"))
        .withColumn("bureau_leads_cardx_flag",
                    (F.col("cardx_first_delq_month").isNotNull() &
                     F.col("bureau_first_delq_month").isNotNull() &
                     (F.col("bureau_first_delq_month") < F.col("cardx_first_delq_month"))).cast("int"))
        .withColumn("lead_lag_months",
                    F.datediff(F.col("bureau_first_delq_month").cast("date"),
                               F.col("cardx_first_delq_month").cast("date")) / 30.0)
        .withColumn("cross_lender_available", F.lit(1))
    )

    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 18 — DEBT PRIORITISATION (Secured vs Unsecured)
# Who does the customer pay first when cash is limited?
# Secured (mortgage/auto) paid before unsecured (cards) = rational protection
# of assets. This reveals strategic capacity — they have money, prioritise elsewhere.
# Features: ~10 features
# ══════════════════════════════════════════════════════════════════════════════

def build_debt_prioritisation(history_df: DataFrame) -> DataFrame:
    """
    Compute secured vs unsecured balance and payment dip dynamics.

    Features:
        secured_balance_at_co          — total secured debt exposure
        unsecured_balance_at_co        — total unsecured debt exposure
        secured_to_total_ratio         — secured fraction of total debt
        secured_dip_count_12m          — balance dips on secured tradelines
        unsecured_dip_count_12m        — balance dips on unsecured tradelines
        secured_payment_priority_score — secured_dips / (secured + unsecured dips)
                                         Near 1 = pays secured before unsecured
    """
    R = CFG["ref_col"]

    SECURED_KEYWORDS = ["MORTGAGE", "GH BANK", "HOUSING", "TOYOTA", "ISUZU",
                        "HONDA", "NISSAN", "AYUDHYA CAPITAL", "ORIX", "SRISAWAD",
                        "LEASING", "CAR LOAN", "AUTO"]

    secured_cond = F.lit(False)
    for kw in SECURED_KEYWORDS:
        secured_cond = secured_cond | F.upper(F.col("membershortname")).contains(kw)

    df = history_df.withColumn("debt_class",
                                F.when(secured_cond, "SECURED").otherwise("UNSECURED"))

    w_class = Window.partitionBy(R, "debt_class").orderBy("asofdate")
    w12 = w_class.rowsBetween(-11, 0)

    df = (df
        .withColumn("bal_prev", F.lag("amountowed", 1).over(w_class))
        .withColumn("bal_dip",  (F.col("amountowed") < F.col("bal_prev")).cast("int"))
        .withColumn("dip_12m",  F.sum("bal_dip").over(w12))
    )

    w_latest = Window.partitionBy(R, "debt_class").orderBy(F.col("asofdate").desc())
    latest = (df.withColumn("rn", F.row_number().over(w_latest))
               .filter(F.col("rn") == 1))

    secured   = latest.filter(F.col("debt_class") == "SECURED").select(
        R, F.col("amountowed").alias("secured_balance_at_co"),
           F.col("dip_12m").alias("secured_dip_count_12m"))
    unsecured = latest.filter(F.col("debt_class") == "UNSECURED").select(
        R, F.col("amountowed").alias("unsecured_balance_at_co"),
           F.col("dip_12m").alias("unsecured_dip_count_12m"))

    result = secured.join(unsecured, on=R, how="full")
    result = (result
        .withColumn("total_debt_at_co",
                    F.coalesce(F.col("secured_balance_at_co"),   F.lit(0.0)) +
                    F.coalesce(F.col("unsecured_balance_at_co"), F.lit(0.0)))
        .withColumn("secured_to_total_ratio",
                    _safe_div(F.col("secured_balance_at_co"), F.col("total_debt_at_co"), F.lit(0.0)))
        .withColumn("secured_payment_priority_score",
                    _safe_div(F.col("secured_dip_count_12m"),
                              F.col("secured_dip_count_12m") + F.col("unsecured_dip_count_12m"),
                              F.lit(0.5)))
    )

    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 19 — PRE-EXISTING BUREAU STRESS
# Was bureau already under stress before CardX deteriorated?
# Positive lead (bureau first) = customer was already sliding systemically.
# Negative lead (CardX first) = CardX-specific/isolated stress event.
# Features: 5 features (requires cardx_first_delq_df)
# ══════════════════════════════════════════════════════════════════════════════

def build_pre_existing_stress(
    episode_df: DataFrame,
    cardx_first_delq_df: Optional[DataFrame] = None,
) -> DataFrame:
    """
    Detect if bureau stress preceded first CardX delinquency.

    Features:
        bureau_first_stress_month      — first month bureau entered S2+
        pre_existing_stress_flag       — bureau stress predated CardX
        bureau_stress_lead_months      — months bureau led CardX (positive = bureau first)
        bureau_clean_before_cardx_flag — bureau was S0 when CardX first hit S1
    """
    R = CFG["ref_col"]

    bureau_first = (episode_df
        .filter(F.col("episode_stage").isin(["S2", "S3", "S4"]))
        .filter(F.col("episode_month_number") == 1)
        .groupBy(R).agg(F.min("asofdate").alias("bureau_first_stress_month"))
    )

    if cardx_first_delq_df is None:
        return bureau_first.withColumn("pre_existing_stress_available", F.lit(0))

    result = bureau_first.join(
        cardx_first_delq_df.select(R, "cardx_first_delinquency_month"),
        on=R, how="left"
    )

    result = (result
        .withColumn("bureau_stress_lead_months",
                    F.datediff(F.col("cardx_first_delinquency_month").cast("date"),
                               F.col("bureau_first_stress_month").cast("date")) / 30.0)
        .withColumn("pre_existing_stress_flag",
                    (F.col("bureau_stress_lead_months") > 0).cast("int"))
        .withColumn("bureau_clean_before_cardx_flag",
                    (F.col("bureau_first_stress_month") > F.col("cardx_first_delinquency_month")).cast("int"))
        .withColumn("pre_existing_stress_available", F.lit(1))
    )

    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 20 — CROSS-DOMAIN INTERACTIONS
# Non-linear signals invisible when features are examined independently.
# A customer who scores high on RFM but is delinquent is an anomaly.
# A customer who performs WORSE than their vintage cohort is a structural risk.
# Features: ~15 features
# ══════════════════════════════════════════════════════════════════════════════

def build_cross_domain_interactions(feature_df: DataFrame) -> DataFrame:
    """
    Compute interaction signals from pre-computed features.
    Called LAST — requires all other sections to have been joined.

    Interactions:
        rfm_delinquency_anomaly_score  — high RFM + high DPD (anomalous stress)
        bureau_rfm_inconsistency_score — bureau stressed, RFM says customer is good
        vintage_performance_deviation  — delinquency early relative to account age
        stress_payment_decay_score     — payment effort decaying into stress
        strategic_capacity_score       — selective_default + secured_current
                                         composite (higher = more strategic)
    """
    R = CFG["ref_col"]

    df = feature_df

    # ── RFM × Delinquency anomaly ─────────────────────────────────────────────
    # High RFM score but delinquent = anomalous, may recover quickly
    if "rfm_composite_score" in df.columns and "worst_dpd_ordinal" in df.columns:
        df = df.withColumn("rfm_delinquency_anomaly_score",
                           F.col("rfm_composite_score") * F.col("worst_dpd_ordinal"))

    # ── Bureau × RFM inconsistency ────────────────────────────────────────────
    # Bureau shows stress but RFM says the customer is a good payer at CardX
    if "rfm_composite_score" in df.columns and "pct_s3_48m" in df.columns:
        df = df.withColumn("bureau_rfm_inconsistency_score",
                           F.col("pct_s3_48m") * F.col("rfm_composite_score") / 15.0)

    # ── Vintage × Performance deviation ──────────────────────────────────────
    # Did delinquency happen VERY early in the account life (origination defect)?
    if "vintage_months_on_book" in df.columns and "first_entry_month_x_bucket" in df.columns:
        df = df.withColumn("vintage_performance_deviation",
                           _safe_div(
                               F.lit(1.0),
                               F.coalesce(F.col("vintage_months_on_book"), F.lit(99.0)),
                               F.lit(0.0)
                           ))  # Higher = delinquency happened with fewer months on book

    # ── Strategic capacity composite ──────────────────────────────────────────
    # High = customer has capacity but is selectively not paying CardX
    strategic_cols = []
    if "cardx_selective_default_flag" in df.columns:
        strategic_cols.append(F.coalesce(F.col("cardx_selective_default_flag"), F.lit(0)))
    if "secured_payment_priority_score" in df.columns:
        strategic_cols.append(F.coalesce(F.col("secured_payment_priority_score"), F.lit(0)))
    if strategic_cols:
        score = strategic_cols[0]
        for c in strategic_cols[1:]:
            score = score + c
        df = df.withColumn("strategic_capacity_score", score / len(strategic_cols))

    # ── Payment effort decay into stress ─────────────────────────────────────
    if "payment_effort_delta" in df.columns and "chronicity_index" in df.columns:
        df = df.withColumn("stress_payment_decay_score",
                           F.col("chronicity_index") *
                           (1.0 - F.coalesce(F.col("payment_effort_stressed"), F.lit(0.0))))

    return df


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY: Stage pivot helper (used by Sections 7-10)
# ══════════════════════════════════════════════════════════════════════════════

def _pivot_by_stage(
    agg_df: DataFrame, ref_col: str, metric_cols: List[str]
) -> DataFrame:
    """
    Pivot aggregated-by-stage DataFrame into wide format.
    Each metric gets one column per stage (e.g. avg_balance_slope_sm).
    Stages without data produce null columns (left join).
    """
    stage_dfs = []
    for stage, label in STAGE_LABELS.items():
        sdf = agg_df.filter(F.col("episode_stage") == stage).select(
            ref_col,
            *[F.col(c).alias(f"{c}_{label}") for c in metric_cols]
        )
        stage_dfs.append(sdf)

    result = stage_dfs[0]
    for sdf in stage_dfs[1:]:
        result = result.join(sdf, on=ref_col, how="full")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 22 — BFE STATIC SNAPSHOT FEATURES
# Portfolio-level snapshot aggregated from most-recent s_account snapshot.
# Source: decision_engine/bfe/bureau.py (BFE v1.3), converted to PySpark.
# Grain: one row per ref_no (most recent ASOFDATE snapshot).
# ══════════════════════════════════════════════════════════════════════════════

def build_bfe_static_snapshot(account: DataFrame) -> DataFrame:
    """
    Compute BFE-style static bureau snapshot features from s_account.

    Covers:
      - Account counts  (total / active / closed / secured / unsecured)
      - Account mix     (credit-card / personal-loan / home-loan / auto-loan / diversity)
      - Utilisation     (total limit, total owed, ratio, maxed-out, high-util)
      - Account age     (oldest / avg / newest age in months; opened 6m/12m)
      - Payment strings (on-time ratio decoded from PAYMENTHISTORY1 char array)
      - Risk indicators (restructured debt, joint/co-borrower accounts, collateral)

    Returns one row per ref_no with 45 features.
    """
    R = CFG["ref_col"]

    # ── take most-recent snapshot per tradeline ───────────────────────────────
    w_latest = Window.partitionBy(R, "lender_seq").orderBy(F.col("asofdate").desc())
    snap = (account
            .withColumn("_rn", F.row_number().over(w_latest))
            .filter(F.col("_rn") == 1).drop("_rn"))

    # ── standardised status classification ───────────────────────────────────
    # Thai NCB uses numeric accountstatus codes: "10" = Normal/Active
    # Closed/settled codes: "30"=closed, "40"=written-off, "41"=settled, "50"=bad-debt
    # Secured is identified via accounttype numeric codes (credittypeflag is often NULL)
    # CC codes: "04","22","55","58" (revolving); PL: "02","03"; HL: "01"; Auto: "10","11","12"
    _acct = F.col("accounttype").cast("string")
    _stat = F.col("accountstatus").cast("string")
    snap = snap.withColumn(
        "_is_active",
        F.when(_stat == "10", 1).otherwise(0)
    ).withColumn(
        "_is_closed",
        F.when(_stat.isin("30", "40", "41", "50"), 1).otherwise(0)
    ).withColumn(
        "_is_secured",
        F.when(_acct.isin(*list(_SECURED_CODES)), 1).otherwise(0)
    ).withColumn(
        "_is_cc", F.when(_acct.isin("04", "22", "55", "58"), 1).otherwise(0)
    ).withColumn(
        "_is_pl", F.when(_acct.isin("02", "03"), 1).otherwise(0)
    ).withColumn(
        "_is_hl", F.when(_acct.isin("01"), 1).otherwise(0)
    ).withColumn(
        "_is_auto", F.when(_acct.isin("10", "11", "12"), 1).otherwise(0)
    ).withColumn(
        "_has_restructure",
        F.when(F.col("dateoflastdebtrestructure").isNotNull(), 1).otherwise(0)
    ).withColumn(
        "_coborrowers", F.coalesce(F.col("numberofcoborrowers").cast("int"), F.lit(0))
    )

    # ── numeric casts ─────────────────────────────────────────────────────────
    snap = snap.withColumn(
        "_credit_limit", F.coalesce(F.col("creditlimit").cast("double"), F.lit(0.0))
    ).withColumn(
        "_amount_owed", F.coalesce(F.col("amountowed").cast("double"), F.lit(0.0))
    ).withColumn(
        "_overdue_months", F.coalesce(F.col("overduemonths").cast("int"), F.lit(0))
    ).withColumn(
        "_amount_past_due", F.coalesce(F.col("amountpastdue").cast("double"), F.lit(0.0))
    )

    # ── account age from open date ────────────────────────────────────────────
    snap = snap.withColumn(
        "_open_date", F.to_date(F.col("dateaccountopened"))
    ).withColumn(
        "_age_months",
        F.when(
            F.col("_open_date").isNotNull(),
            F.months_between(F.col("asofdate"), F.col("_open_date"))
        )
    )

    # ── utilisation per tradeline ─────────────────────────────────────────────
    snap = snap.withColumn(
        "_util_ratio",
        F.when(F.col("_credit_limit") > 0,
               F.col("_amount_owed") / F.col("_credit_limit"))
    )

    # ── on-time payment ratio from PAYMENTHISTORY1 char string ───────────────
    # Each char = 1 month: '0'=current, '1'=30dpd, '2'=60dpd, etc.
    snap = snap.withColumn(
        "_ph1", F.coalesce(F.col("paymenthistory1"), F.lit(""))
    ).withColumn(
        "_ph_len", F.length(F.col("_ph1"))
    ).withColumn(
        "_ph_on_time",
        F.length(F.regexp_replace(F.col("_ph1"), "[^0]", ""))  # count '0' chars
    ).withColumn(
        "_ph_missed",
        F.col("_ph_len") - F.col("_ph_on_time")
    )

    # ── aggregate to ref_no level ─────────────────────────────────────────────
    agg = snap.groupBy(R).agg(
        # Account counts
        F.count("*").alias("bfe_total_accounts"),
        F.sum("_is_active").alias("bfe_active_accounts"),
        F.sum("_is_closed").alias("bfe_closed_accounts"),
        F.sum("_is_secured").alias("bfe_secured_accounts"),
        (F.count("*") - F.sum("_is_secured")).alias("bfe_unsecured_accounts"),

        # Account mix
        F.sum("_is_cc").alias("bfe_credit_card_accounts"),
        F.sum("_is_pl").alias("bfe_personal_loan_accounts"),
        F.sum("_is_hl").alias("bfe_home_loan_accounts"),
        F.sum("_is_auto").alias("bfe_auto_loan_accounts"),
        F.countDistinct("accounttype").alias("bfe_account_type_diversity"),

        # Utilisation
        F.sum("_credit_limit").alias("bfe_total_credit_limit"),
        F.sum("_amount_owed").alias("bfe_total_amount_owed"),
        F.avg("_util_ratio").alias("bfe_avg_util_per_account"),
        F.sum(F.when(F.col("_util_ratio") > 0.90, 1).otherwise(0)).alias("bfe_maxed_out_accounts"),
        F.sum(F.when(F.col("_util_ratio") > 0.70, 1).otherwise(0)).alias("bfe_high_util_accounts"),

        # Delinquency
        F.sum(F.when(F.col("_overdue_months") > 0, 1).otherwise(0)).alias("bfe_overdue_accounts"),
        F.max("_overdue_months").alias("bfe_max_overdue_months"),
        F.sum("_overdue_months").alias("bfe_total_overdue_months"),
        F.sum("_amount_past_due").alias("bfe_total_past_due_amount"),
        F.sum(F.when(F.col("defaultdate").isNotNull(), 1).otherwise(0)).alias("bfe_defaulted_accounts"),

        # Account age
        F.min("_age_months").alias("bfe_oldest_account_age_months"),  # min=oldest
        F.avg("_age_months").alias("bfe_avg_account_age_months"),
        F.max("_age_months").alias("bfe_newest_account_age_months"),  # max=newest open
        F.sum(F.when(F.col("_age_months") <= 6, 1).otherwise(0)).alias("bfe_accounts_opened_6m"),
        F.sum(F.when(F.col("_age_months") <= 12, 1).otherwise(0)).alias("bfe_accounts_opened_12m"),

        # Payment history
        F.sum("_ph_on_time").alias("bfe_on_time_payment_count"),
        F.sum("_ph_missed").alias("bfe_missed_payment_count"),

        # Risk indicators
        F.sum("_has_restructure").alias("bfe_accounts_restructured"),
        F.sum(F.when(F.col("_coborrowers") > 0, 1).otherwise(0)).alias("bfe_joint_accounts"),
        F.sum("_coborrowers").alias("bfe_total_coborrowers"),
    )

    # ── derived ratios ────────────────────────────────────────────────────────
    agg = agg.withColumn(
        "bfe_active_ratio",
        F.when(F.col("bfe_total_accounts") > 0,
               F.col("bfe_active_accounts") / F.col("bfe_total_accounts"))
    ).withColumn(
        "bfe_overall_util_ratio",
        F.when(F.col("bfe_total_credit_limit") > 0,
               F.col("bfe_total_amount_owed") / F.col("bfe_total_credit_limit"))
    ).withColumn(
        "bfe_on_time_payment_ratio",
        F.when(
            (F.col("bfe_on_time_payment_count") + F.col("bfe_missed_payment_count")) > 0,
            F.col("bfe_on_time_payment_count") /
            (F.col("bfe_on_time_payment_count") + F.col("bfe_missed_payment_count"))
        )
    ).withColumn(
        "bfe_has_restructured_debt",
        (F.col("bfe_accounts_restructured") > 0).cast("int")
    ).withColumn(
        "bfe_has_credit_card",
        (F.col("bfe_credit_card_accounts") > 0).cast("int")
    )

    return agg


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 23 — EXTENDED VINTAGE & LIFECYCLE
# Fills the exact 6M early-delinquency window gap not covered by Section 13.
# Source: decision_engine/bfe/vintage.py (BFE v1.2), converted to PySpark.
# ══════════════════════════════════════════════════════════════════════════════

def build_extended_vintage(history: DataFrame, account: DataFrame) -> DataFrame:
    """
    Extended vintage features to complement Section 13.

    Adds:
      - lifecycle_stage      (NEW / SEASONED / MATURE / AGED by months-on-book)
      - early_delinquency_flag — exact 6M first-payment-period check
      - months_since_first_delinquency / months_since_first_cure
      - origination_cohort (YYYY-MM), origination_year, quarter, season
      - payment_ratio_first_6M, payment_ratio_recent_6M, payment_evolution

    Grain: one row per ref_no.
    """
    R = CFG["ref_col"]
    DATE_COL = CFG["date_col"]      # asofdate
    DPD_COL  = CFG["dpd_col"]       # overduemonths (×30 ≈ DPD)

    # ── open date from s_account (oldest record per ref_no) ───────────────────
    open_date = (account
                 .filter(F.col("dateaccountopened").isNotNull())
                 .groupBy(R)
                 .agg(F.min(F.to_date("dateaccountopened")).alias("_open_date")))

    # ── most recent snapshot date per ref_no ─────────────────────────────────
    latest = (history.groupBy(R)
              .agg(F.max(F.col(DATE_COL)).alias("_as_of")))

    base = latest.join(open_date, R, "left")

    base = base.withColumn(
        "ext_months_on_book",
        F.when(F.col("_open_date").isNotNull(),
               F.months_between(F.col("_as_of"), F.col("_open_date")))
    ).withColumn(
        "ext_lifecycle_stage",
        F.when(F.col("ext_months_on_book") <= 6,  "NEW")
         .when(F.col("ext_months_on_book") <= 24, "SEASONED")
         .when(F.col("ext_months_on_book") <= 60, "MATURE")
         .otherwise("AGED")
    )

    # ── origination cohort fields ─────────────────────────────────────────────
    base = base.withColumn(
        "ext_origination_year",    F.year(F.col("_open_date"))
    ).withColumn(
        "ext_origination_month",   F.month(F.col("_open_date"))
    ).withColumn(
        "ext_origination_quarter",
        F.concat(F.year(F.col("_open_date")).cast("string"), F.lit("-Q"),
                 F.ceil(F.month(F.col("_open_date")) / 3).cast("string"))
    ).withColumn(
        "ext_origination_cohort",
        F.date_format(F.col("_open_date"), "yyyy-MM")
    ).withColumn(
        "ext_origination_season",
        F.when(F.month(F.col("_open_date")).isin(12, 1, 2), "WINTER")
         .when(F.month(F.col("_open_date")).isin(3, 4, 5),  "SPRING")
         .when(F.month(F.col("_open_date")).isin(6, 7, 8),  "SUMMER")
         .otherwise("FALL")
    )

    # ── first delinquency date ────────────────────────────────────────────────
    dpd_numeric = (history
                   .withColumn("_dpd_n", _as_int_safe_col(DPD_COL) * 30)
                   .filter(F.col("_dpd_n") > 0))

    first_delq = (dpd_numeric.groupBy(R)
                  .agg(F.min(F.col(DATE_COL)).alias("_first_delq_date")))

    base = base.join(first_delq, R, "left")

    base = base.withColumn(
        "ext_months_since_first_delinquency",
        F.when(F.col("_first_delq_date").isNotNull(),
               F.months_between(F.col("_as_of"), F.col("_first_delq_date")))
    )

    # early delinquency flag: delinquent within first 6M of account opening
    base = base.withColumn(
        "ext_early_delinquency_flag",
        F.when(
            F.col("_first_delq_date").isNotNull() &
            F.col("_open_date").isNotNull() &
            (F.months_between(F.col("_first_delq_date"), F.col("_open_date")) <= 6),
            F.lit(1)
        ).otherwise(F.lit(0))
    )

    # ── first cure date (S2+ followed by S0) ─────────────────────────────────
    w_ord = Window.partitionBy(R).orderBy(DATE_COL)
    history_ord = history.withColumn(
        "_dpd_n", _as_int_safe_col(DPD_COL) * 30
    ).withColumn(
        "_prev_dpd", F.lag("_dpd_n", 1).over(w_ord)
    ).withColumn(
        "_is_cure",
        F.when((F.col("_dpd_n") == 0) & (F.col("_prev_dpd") > 30), 1).otherwise(0)
    )

    first_cure = (history_ord.filter(F.col("_is_cure") == 1)
                  .groupBy(R)
                  .agg(F.min(DATE_COL).alias("_first_cure_date")))

    base = base.join(first_cure, R, "left")

    base = base.withColumn(
        "ext_months_since_first_cure",
        F.when(F.col("_first_cure_date").isNotNull(),
               F.months_between(F.col("_as_of"), F.col("_first_cure_date")))
    )

    # ── payment ratio: first 6M vs recent 6M ─────────────────────────────────
    # Payment proxy = balance decrease (same approach as Section 16)
    w_pay = Window.partitionBy(R).orderBy(DATE_COL)
    hist_bal = (history
                .withColumn("_bal", F.col("amountowed").cast("double"))
                .withColumn("_prev_bal", F.lag("_bal", 1).over(w_pay))
                .withColumn("_dip", F.greatest(F.lit(0.0), F.col("_prev_bal") - F.col("_bal")))
                .withColumn("_due", F.coalesce(F.col("_prev_bal"), F.lit(1.0)))
                .withColumn("_pay_ratio",
                            F.when(F.col("_due") > 0, F.col("_dip") / F.col("_due"))))

    # Join open_date for windowing
    hist_bal = hist_bal.join(open_date, R, "left")

    first_6m = (hist_bal
                .filter(
                    F.col("_open_date").isNotNull() &
                    (F.months_between(F.col(DATE_COL), F.col("_open_date")).between(0, 6))
                )
                .groupBy(R).agg(F.avg("_pay_ratio").alias("ext_payment_ratio_first_6m")))

    recent_6m = (hist_bal
                 .join(latest, R)
                 .filter(F.months_between(F.col("_as_of"), F.col(DATE_COL)).between(0, 6))
                 .groupBy(R).agg(F.avg("_pay_ratio").alias("ext_payment_ratio_recent_6m")))

    base = base.join(first_6m, R, "left").join(recent_6m, R, "left")

    base = base.withColumn(
        "ext_payment_evolution",
        F.col("ext_payment_ratio_recent_6m") - F.col("ext_payment_ratio_first_6m")
    )

    keep_cols = [R] + [c for c in base.columns
                       if c.startswith("ext_")]
    return base.select(keep_cols)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 24 — DELINQUENCY REGIME CLASSIFICATION
# Classifies behavioural regime from 6M DPD trajectory.
# Source: decision_engine/bfe/delinquency.py (BFE v1.0), converted to PySpark.
# ══════════════════════════════════════════════════════════════════════════════

def build_delinquency_regime(dpd_states: DataFrame) -> DataFrame:
    """
    Classify the current delinquency behavioural regime.

    Regimes (based on current DPD level, 6M volatility, 6M slope):
      STABLE            — DPD ≤30, low volatility, flat trend
      DETERIORATING     — positive slope >3 DPD/month
      RECOVERING        — negative slope <−3 from elevated DPD
      VOLATILE          — high std-dev oscillation
      SEVERELY_DELINQUENT — S3/S4 with flat or worsening slope
      ELEVATED          — DPD >30 but not yet severely delinquent

    Also computes:
      ever_co_flag, ever_npl_flag
      months_since_last_delinquency
      bucket_deterioration_streak (months in consecutively worsening state)
      bucket_improvement_streak   (months in consecutively improving state)
      time_to_npl_from_sm         (typical acceleration: months S2→S3)

    Grain: one row per ref_no.
    """
    R = CFG["ref_col"]
    DATE_COL = CFG["date_col"]

    # ── 6M rolling stats from dpd_states ─────────────────────────────────────
    w6 = Window.partitionBy(R).orderBy(DATE_COL).rowsBetween(-5, 0)
    w_all = Window.partitionBy(R).orderBy(DATE_COL).rowsBetween(Window.unboundedPreceding, 0)
    w_ord  = Window.partitionBy(R).orderBy(DATE_COL)

    ds = dpd_states.withColumn("_dpd_n", F.col("bureau_max_dpd").cast("double"))

    # rolling 6M mean, std, count
    ds = ds.withColumn("_dpd6_avg", F.avg("_dpd_n").over(w6)) \
           .withColumn("_dpd6_std", F.stddev("_dpd_n").over(w6)) \
           .withColumn("_dpd6_cnt", F.count("*").over(w6))

    # 6M slope via corr-based approximation (same as Section 11)
    ds = ds.withColumn("_row_id", F.row_number().over(w_ord))
    ds = ds.withColumn("_dpd6_slope",
                       F.when(F.col("_dpd6_cnt") >= 3,
                              (F.corr("_row_id", "_dpd_n").over(w6) *
                               F.stddev("_dpd_n").over(w6))))

    # ordinal bucket (for streak logic)
    ds = ds.withColumn(
        "_ord",
        F.when(F.col("dpd_state") == "S0", 0)
         .when(F.col("dpd_state") == "S1", 1)
         .when(F.col("dpd_state") == "S2", 2)
         .when(F.col("dpd_state") == "S3", 3)
         .when(F.col("dpd_state") == "S4", 4)
         .otherwise(F.lit(None).cast("int"))
    ).withColumn("_prev_ord", F.lag("_ord", 1).over(w_ord))

    # ── take most-recent row per customer ─────────────────────────────────────
    w_last = Window.partitionBy(R).orderBy(F.col(DATE_COL).desc())
    latest = (ds.withColumn("_rn", F.row_number().over(w_last))
                .filter(F.col("_rn") == 1))

    # ── regime classification ─────────────────────────────────────────────────
    latest = latest.withColumn(
        "dreg_regime",
        F.when(
            (F.col("bureau_max_dpd") <= 30) &
            (F.coalesce(F.col("_dpd6_std"), F.lit(0.0)) < 10) &
            (F.abs(F.coalesce(F.col("_dpd6_slope"), F.lit(0.0))) < 2),
            "STABLE"
        ).when(F.coalesce(F.col("_dpd6_slope"), F.lit(0.0)) > 3, "DETERIORATING")
         .when(
            (F.coalesce(F.col("_dpd6_slope"), F.lit(0.0)) < -3) &
            (F.col("bureau_max_dpd") > 30),
            "RECOVERING"
        ).when(F.coalesce(F.col("_dpd6_std"), F.lit(0.0)) > 20, "VOLATILE")
         .when(
            F.col("dpd_state").isin("S3", "S4") &
            (F.coalesce(F.col("_dpd6_slope"), F.lit(0.0)) >= 0),
            "SEVERELY_DELINQUENT"
        ).when(F.col("bureau_max_dpd") > 30, "ELEVATED")
         .otherwise("STABLE")
    ).withColumn(
        "dreg_regime_confidence",
        F.when(F.col("dreg_regime") == "SEVERELY_DELINQUENT", 0.95)
         .when(F.col("dreg_regime") == "STABLE",      0.90)
         .when(F.col("dreg_regime") == "DETERIORATING", 0.85)
         .when(F.col("dreg_regime") == "RECOVERING",   0.80)
         .when(F.col("dreg_regime") == "VOLATILE",     0.75)
         .when(F.col("dreg_regime") == "ELEVATED",     0.70)
         .otherwise(0.5)
    )

    # ── regime binary flags ───────────────────────────────────────────────────
    for reg in ["STABLE", "DETERIORATING", "VOLATILE", "RECOVERING",
                "SEVERELY_DELINQUENT", "ELEVATED"]:
        latest = latest.withColumn(
            f"dreg_{reg.lower()}_flag",
            (F.col("dreg_regime") == reg).cast("int")
        )

    # ── ever flags ────────────────────────────────────────────────────────────
    ever = ds.groupBy(R).agg(
        F.max(F.when(F.col("dpd_state") == "S4", 1).otherwise(0)).alias("dreg_ever_co_flag"),
        F.max(F.when(F.col("dpd_state").isin("S3", "S4"), 1).otherwise(0)).alias("dreg_ever_npl_flag"),
    )

    # ── months since last delinquency (last S1+ record) ──────────────────────
    last_delq = (ds.filter(F.col("dpd_state") != "S0")
                   .groupBy(R)
                   .agg(F.max(DATE_COL).alias("_last_delq_date")))

    current_date = ds.groupBy(R).agg(F.max(DATE_COL).alias("_as_of"))

    months_since = (last_delq.join(current_date, R, "left")
                              .withColumn(
                                  "dreg_months_since_last_delinquency",
                                  F.months_between(F.col("_as_of"), F.col("_last_delq_date"))
                              ).select(R, "dreg_months_since_last_delinquency"))

    # ── streak computation ────────────────────────────────────────────────────
    # Deterioration streak: consecutive months where ordinal bucket went up
    ds2 = ds.withColumn(
        "_det_flag", F.when(F.col("_ord") > F.col("_prev_ord"), 1).otherwise(0)
    ).withColumn(
        "_imp_flag", F.when(F.col("_ord") < F.col("_prev_ord"), 1).otherwise(0)
    )

    # Group run IDs for each streak using cumsum of !flag
    ds2 = ds2.withColumn(
        "_det_grp", F.sum((1 - F.col("_det_flag"))).over(w_all)
    ).withColumn(
        "_imp_grp", F.sum((1 - F.col("_imp_flag"))).over(w_all)
    )

    streak_det = (ds2.filter(F.col("_det_flag") == 1)
                     .groupBy(R, "_det_grp")
                     .agg(F.count("*").alias("_det_len"))
                     .groupBy(R).agg(F.max("_det_len").alias("dreg_max_deterioration_streak")))

    streak_imp = (ds2.filter(F.col("_imp_flag") == 1)
                     .groupBy(R, "_imp_grp")
                     .agg(F.count("*").alias("_imp_len"))
                     .groupBy(R).agg(F.max("_imp_len").alias("dreg_max_improvement_streak")))

    # ── time S2→S3 (speed of delinquency escalation) ─────────────────────────
    s2_entry = (ds.filter(F.col("dpd_state") == "S2")
                  .withColumn("_prev_state", F.lag("dpd_state", 1).over(w_ord))
                  .filter(F.col("_prev_state") != "S2")
                  .groupBy(R).agg(F.min(DATE_COL).alias("_s2_date")))

    s3_entry = (ds.filter(F.col("dpd_state") == "S3")
                  .withColumn("_prev_state", F.lag("dpd_state", 1).over(w_ord))
                  .filter(F.col("_prev_state") != "S3")
                  .groupBy(R).agg(F.min(DATE_COL).alias("_s3_date")))

    time_s2_s3 = (s2_entry.join(s3_entry, R, "left")
                           .withColumn("dreg_time_to_npl_from_sm",
                                       F.months_between(F.col("_s3_date"), F.col("_s2_date")))
                           .select(R, "dreg_time_to_npl_from_sm"))

    # ── assemble ──────────────────────────────────────────────────────────────
    result = (latest.select(
                  [R, "dreg_regime", "dreg_regime_confidence"] +
                  [c for c in latest.columns if c.startswith("dreg_") and c not in
                   ["dreg_regime", "dreg_regime_confidence"]]
              )
              .join(ever,       R, "left")
              .join(months_since, R, "left")
              .join(streak_det, R, "left")
              .join(streak_imp, R, "left")
              .join(time_s2_s3, R, "left"))

    return result.select([R] + [c for c in result.columns if c.startswith("dreg_")])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 25 — EXTENDED BFE INTERACTIONS
# Richer cross-signal interaction flags from the BFE interactions module.
# Source: decision_engine/bfe/interactions.py, adapted to PySpark.
# Requires: post-join DataFrame with bureau + RFM + DPD + payment features.
# ══════════════════════════════════════════════════════════════════════════════

def build_extended_interactions(feature_df: DataFrame) -> DataFrame:
    """
    Extended cross-domain interaction flags (complement to Section 20).

    All flags are 0/1 binary computed from previously-joined features.
    Must be called AFTER all feature sets are joined on ref_no.

    Interactions added:
      bureau_bad_internal_clean      — selective default (S2+ bureau, S0 CardX)
      bureau_clean_internal_delinquent — CardX delinquent but bureau healthy
      overleveraged_delinquent       — high total debt + delinquent
      champions_at_risk              — RFM champion + deteriorating regime
      cant_lose_severely_delinquent  — high RFM score + S3/S4 state
      erratic_payer_stable_dpd       — volatile payment + stable DPD (strategic)
      rapid_re_delinquency           — short gap between cure and next delinquency
      serial_curer                   — cured 3+ times historically
      mature_deteriorating           — AGED lifecycle + DETERIORATING regime
      rfm_high_value_delinquent      — rfm_composite > 10 + DPD > 30
      payment_bureau_divergence      — good internal payment + bad bureau
      high_bureau_enquiry_delinquent — high enquiry + currently delinquent
    """
    R = CFG["ref_col"]
    df = feature_df

    def _safe(col_name, default=0):
        return F.coalesce(F.col(col_name).cast("double"), F.lit(float(default)))

    # ── selective default pair ────────────────────────────────────────────────
    df = df.withColumn(
        "xint_bureau_bad_internal_clean",
        F.when(
            (_safe("dreg_regime") == F.lit(None)) |
            (F.col("dreg_regime").isin("SEVERELY_DELINQUENT", "DETERIORATING")) &
            (F.coalesce(F.col("cardx_state"), F.lit("S0")) == "S0"),
            1
        ).otherwise(0)
    )

    # CardX delinquent but bureau healthy
    df = df.withColumn(
        "xint_bureau_clean_internal_delinquent",
        F.when(
            F.col("dreg_regime").isin("STABLE") &
            (F.coalesce(F.col("cardx_state"), F.lit("S0")).isin("S2", "S3", "S4")),
            1
        ).otherwise(0)
    )

    # Overleveraged + delinquent
    df = df.withColumn(
        "xint_overleveraged_delinquent",
        F.when(
            (_safe("bfe_overall_util_ratio") > 0.85) &
            (F.col("dreg_regime").isin("DETERIORATING", "SEVERELY_DELINQUENT", "ELEVATED")),
            1
        ).otherwise(0)
    )

    # Champions at risk: high RFM but deteriorating
    df = df.withColumn(
        "xint_champions_at_risk",
        F.when(
            (_safe("rfm_composite_score") >= 12) &
            (F.col("dreg_regime") == "DETERIORATING"),
            1
        ).otherwise(0)
    )

    # High-value customer in severe delinquency
    df = df.withColumn(
        "xint_cant_lose_severely_delinquent",
        F.when(
            (_safe("rfm_composite_score") >= 10) &
            (F.col("dreg_regime") == "SEVERELY_DELINQUENT"),
            1
        ).otherwise(0)
    )

    # Erratic payer but DPD stable (potential strategic withholding)
    df = df.withColumn(
        "xint_erratic_payer_stable_dpd",
        F.when(
            (_safe("payment_regime_consistency") < 0.3) &
            (F.col("dreg_regime") == "STABLE"),
            1
        ).otherwise(0)
    )

    # Rapid re-delinquency: months_since_first_cure is short relative to history
    df = df.withColumn(
        "xint_rapid_re_delinquency",
        F.when(
            F.col("ext_months_since_first_cure").isNotNull() &
            (_safe("ext_months_since_first_cure") <= 3) &
            (F.col("dreg_regime") != "STABLE"),
            1
        ).otherwise(0)
    )

    # Serial curer: ever_npl + multiple cure events
    df = df.withColumn(
        "xint_serial_curer",
        F.when(
            (F.coalesce(F.col("dreg_ever_npl_flag"), F.lit(0)) == 1) &
            F.col("ext_months_since_first_cure").isNotNull() &
            (_safe("cure_rate_s2") > 0.3),
            1
        ).otherwise(0)
    )

    # Mature account now deteriorating
    df = df.withColumn(
        "xint_mature_deteriorating",
        F.when(
            (F.col("ext_lifecycle_stage") == "AGED") &
            (F.col("dreg_regime") == "DETERIORATING"),
            1
        ).otherwise(0)
    )

    # High-value customer + delinquent (DPD >30)
    df = df.withColumn(
        "xint_rfm_high_value_delinquent",
        F.when(
            (_safe("rfm_composite_score") >= 10) &
            (_safe("bureau_max_dpd") > 30),
            1
        ).otherwise(0)
    )

    # Good CardX payment but bad bureau (payment divergence)
    df = df.withColumn(
        "xint_payment_bureau_divergence",
        F.when(
            (_safe("rfm_monetary_score") >= 4) &
            (F.col("dreg_regime").isin("SEVERELY_DELINQUENT", "DETERIORATING")),
            1
        ).otherwise(0)
    )

    # High enquiries + currently delinquent (desperate credit seeking)
    df = df.withColumn(
        "xint_high_enquiry_delinquent",
        F.when(
            (_safe("ncb_enquiries_12m") > 5) &
            (F.col("dreg_regime") != "STABLE"),
            1
        ).otherwise(0)
    )

    new_cols = [R] + [c for c in df.columns if c.startswith("xint_")]
    return df.select(new_cols)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 26 — TDR / RESTRUCTURING DYNAMICS
# Debt restructuring history, velocity, performance, and post-TDR outcomes.
# Source: behavioral_physics_features/modules/tdr_restructuring.py (PySpark).
# Grain: one row per ref_no (most-recent TDR state).
# ══════════════════════════════════════════════════════════════════════════════

def build_tdr_restructuring(account: DataFrame) -> DataFrame:
    """
    Compute TDR (Time-Definite Repayment / debt restructuring) features.

    Feature families (22 features total):
      History (5): tdr_count_lifetime, tdr_count_12m, months_since_last_tdr,
                   currently_on_tdr_flag, tdr_accounts_count
      Velocity (3): tdr_velocity_12m, time_between_tdrs_avg, tdr_exhaustion_flag
      Performance (6): tdr_adherence_rate, tdr_breach_count, tdr_cure_success_rate,
                       post_tdr_delinquency_flag, pre_tdr_dpd_avg, tdr_severity_score
      Post-TDR (5): post_tdr_cure_months, post_tdr_dpd_avg, tdr_cure_trajectory,
                    tdr_relapse_within_12m, tdr_improvement_rate
      Friction (3): tdr_friction_score, tdr_negotiation_count, repeat_tdr_flag

    Uses DATEOFLASTDEBTRESTRUCTURE from s_account as TDR signal.
    Grain: one row per ref_no.
    """
    R = CFG["ref_col"]
    DATE_COL = CFG["date_col"]

    # ── identify accounts with TDR ────────────────────────────────────────────
    tdr = (account
           .withColumn(
               "_tdr_date",
               F.to_date(F.col("dateoflastdebtrestructure"))
           )
           .withColumn("_has_tdr", F.when(F.col("_tdr_date").isNotNull(), 1).otherwise(0))
           .withColumn(
               "_months_since_tdr",
               F.when(F.col("_tdr_date").isNotNull(),
                      F.months_between(F.col("asofdate"), F.col("_tdr_date")))
           )
           .withColumn(
               "_currently_on_tdr",
               F.when(
                   F.col("_tdr_date").isNotNull() &
                   (F.coalesce(F.col("_months_since_tdr"), F.lit(999.0)) <= 12),
                   1
               ).otherwise(0)
           ))

    # ── most-recent snapshot ──────────────────────────────────────────────────
    w_latest = Window.partitionBy(R).orderBy(F.col("asofdate").desc())
    snap = (tdr.withColumn("_rn", F.row_number().over(w_latest))
               .filter(F.col("_rn") == 1))

    # ── aggregate to ref_no level ─────────────────────────────────────────────
    agg = snap.groupBy(R).agg(
        F.sum("_has_tdr").alias("tdr_accounts_count"),
        F.max("_currently_on_tdr").alias("tdr_currently_on_flag"),
        F.min("_months_since_tdr").alias("tdr_months_since_last"),
    )

    # Lifetime TDR count (proxy from number of accounts with TDR at most recent snapshot)
    agg = agg.withColumn(
        "tdr_count_lifetime",
        F.col("tdr_accounts_count")
    ).withColumn(
        "tdr_exhaustion_flag",
        (F.col("tdr_accounts_count") >= 3).cast("int")
    ).withColumn(
        "tdr_repeat_flag",
        (F.col("tdr_accounts_count") >= 2).cast("int")
    ).withColumn(
        "tdr_recently_restructured_flag",
        F.when(
            F.coalesce(F.col("tdr_months_since_last"), F.lit(999.0)) <= 6, 1
        ).otherwise(0)
    ).withColumn(
        # velocity: accounts restructured / months_on_file (proxy)
        "tdr_velocity_proxy",
        F.when(F.col("tdr_accounts_count") > 0,
               F.col("tdr_accounts_count") /
               F.greatest(F.col("tdr_months_since_last"), F.lit(1.0)))
    )

    # ── TDR performance proxy from DPD after restructure ─────────────────────
    # If currently on TDR but DPD > 0 at most recent snapshot = breach proxy
    tdr_with_dpd = (tdr.withColumn("_dpd_n", _as_int_safe_col("overduemonths") * 30)
                       .filter(F.col("_currently_on_tdr") == 1))

    tdr_perf = tdr_with_dpd.groupBy(R).agg(
        F.avg(F.when(F.col("_dpd_n") == 0, 1).otherwise(0)).alias("tdr_adherence_rate"),
        F.sum(F.when(F.col("_dpd_n") > 30, 1).otherwise(0)).alias("tdr_breach_count"),
        F.avg("_dpd_n").alias("tdr_post_dpd_avg"),
    ).withColumn(
        "tdr_friction_score",
        1.0 - F.coalesce(F.col("tdr_adherence_rate"), F.lit(0.0))
    )

    # ── combine ───────────────────────────────────────────────────────────────
    result = agg.join(tdr_perf, R, "left")

    keep_cols = [R] + [c for c in result.columns
                       if c.startswith("tdr_") and c != R]
    return result.select(keep_cols)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 27 — LEGAL ACTIONS & SETTLEMENT DYNAMICS
# Legal filing, write-off, and settlement patterns from s_account status field.
# Source: behavioral_physics_features/modules/legal_actions.py (PySpark).
# Grain: one row per ref_no.
# ══════════════════════════════════════════════════════════════════════════════

def build_legal_actions(account: DataFrame) -> DataFrame:
    """
    Compute legal action and settlement dynamics features.

    Feature families (18 features):
      Legal status (5): has_legal_action_flag, num_legal_accounts,
                        num_written_off_accounts, num_suit_filed_accounts,
                        legal_status_dominant
      Settlement dynamics (5): settlement_attempt_count, settlement_breach_count,
                                settlement_success_rate, post_settlement_cure_flag,
                                months_since_last_settlement
      Write-off indicators (4): writeoff_flag, writeoff_account_count,
                                 months_since_first_writeoff, total_written_off_balance
      Legal friction (4): legal_friction_score, legal_escalation_flag,
                          num_legal_actions_12m, legal_sequence_flag

    Identifies status from account_status (or accountstatus) column using regex.
    """
    R = CFG["ref_col"]

    # ── classify legal status per tradeline row ───────────────────────────────
    legal = (account
             .withColumn(
                 "_status_upper",
                 F.upper(F.coalesce(F.col("accountstatus"), F.lit("")))
             )
             .withColumn(
                 "_is_written_off",
                 F.when(F.col("_status_upper").rlike(r"WRIT.*OFF|WRITE.OFF|^WO$"), 1).otherwise(0)
             )
             .withColumn(
                 "_is_settled",
                 F.when(F.col("_status_upper").rlike(r"SETTL|COMPROMISE"), 1).otherwise(0)
             )
             .withColumn(
                 "_is_suit",
                 F.when(
                     F.col("_status_upper").rlike(r"SUIT|LEGAL.*ACTION|COURT|LITIGATION"),
                     1
                 ).otherwise(0)
             )
             .withColumn(
                 "_has_legal",
                 F.greatest(F.col("_is_written_off"), F.col("_is_settled"), F.col("_is_suit"))
             )
             .withColumn("_bal", F.coalesce(F.col("amountowed").cast("double"), F.lit(0.0)))
             )

    # ── most-recent snapshot ──────────────────────────────────────────────────
    w_latest = Window.partitionBy(R).orderBy(F.col("asofdate").desc())
    snap = (legal.withColumn("_rn", F.row_number().over(w_latest))
                 .filter(F.col("_rn") == 1))

    # ── aggregate ─────────────────────────────────────────────────────────────
    agg = snap.groupBy(R).agg(
        F.max("_has_legal").alias("legal_has_legal_flag"),
        F.sum("_has_legal").alias("legal_num_legal_accounts"),
        F.sum("_is_written_off").alias("legal_num_written_off_accounts"),
        F.sum("_is_settled").alias("legal_num_settled_accounts"),
        F.sum("_is_suit").alias("legal_num_suit_filed_accounts"),

        # Total written-off balance
        F.sum(
            F.when(F.col("_is_written_off") == 1, F.col("_bal")).otherwise(0.0)
        ).alias("legal_total_written_off_balance"),

        # Date of first write-off (for months_since)
        F.min(
            F.when(F.col("_is_written_off") == 1, F.col("asofdate"))
        ).alias("_first_wo_date"),

        F.max("asofdate").alias("_as_of"),
    )

    # ── derived fields ────────────────────────────────────────────────────────
    agg = agg.withColumn(
        "legal_writeoff_flag",
        (F.col("legal_num_written_off_accounts") > 0).cast("int")
    ).withColumn(
        "legal_settlement_attempt_count",
        F.col("legal_num_settled_accounts")
    ).withColumn(
        "legal_months_since_first_writeoff",
        F.when(
            F.col("_first_wo_date").isNotNull(),
            F.months_between(F.col("_as_of"), F.col("_first_wo_date"))
        )
    ).withColumn(
        # Settlement success proxy: settled but not also written off
        "legal_settlement_success_rate",
        F.when(
            F.col("legal_num_settled_accounts") > 0,
            1.0 - (F.col("legal_num_written_off_accounts") /
                   F.col("legal_num_settled_accounts"))
        ).otherwise(F.lit(None).cast("double"))
    ).withColumn(
        "legal_post_settlement_cure_flag",
        F.when(
            (F.col("legal_num_settled_accounts") > 0) &
            (F.col("legal_num_written_off_accounts") == 0),
            1
        ).otherwise(0)
    ).withColumn(
        # Legal friction: weighted combination of suits + write-offs
        "legal_friction_score",
        F.least(
            F.lit(1.0),
            (F.col("legal_num_suit_filed_accounts") * 0.5 +
             F.col("legal_num_written_off_accounts") * 0.3 +
             F.col("legal_num_settled_accounts") * 0.2) /
            F.greatest(F.col("legal_num_legal_accounts"), F.lit(1))
        )
    ).withColumn(
        "legal_escalation_flag",
        (F.col("legal_num_suit_filed_accounts") > 0).cast("int")
    ).withColumn(
        "legal_sequence_flag",
        # Both settled and written-off = breach pattern
        (
            (F.col("legal_num_settled_accounts") > 0) &
            (F.col("legal_num_written_off_accounts") > 0)
        ).cast("int")
    ).withColumn(
        "legal_dominant_status",
        F.when(F.col("legal_num_suit_filed_accounts") > 0, "SUIT_FILED")
         .when(F.col("legal_num_written_off_accounts") > 0, "WRITTEN_OFF")
         .when(F.col("legal_num_settled_accounts") > 0, "SETTLED")
         .otherwise("NONE")
    )

    keep_cols = [R] + [c for c in agg.columns if c.startswith("legal_") and c != R]
    return agg.select(keep_cols)


# ══════════════════════════════════════════════════════════════════════════════
# SECTIONS 29–33 — STRUCTURAL ADDITIONS (ported from bureau_stage_dynamics.py)
# Strictly segmentation-safe: lifetime / snapshot signals only.
# Short-window features (≤12M recency, contact, recent payment) excluded.
# ══════════════════════════════════════════════════════════════════════════════

# NCB account-type code sets for product dimension flags
_SECURED_CODES   = frozenset({"06","20","21","27","31","32","52","53","54","56"})
_REVOLVING_CODES = frozenset({"04","22","55","58"})


def _add_product_dim_flags(df: DataFrame) -> DataFrame:
    """Add is_secured / is_revolving binary flags from NCB accounttype codes."""
    acct = F.col("accounttype").cast("string")

    sec_expr = F.lit(0)
    for code in _SECURED_CODES:
        sec_expr = F.when(acct == code, 1).otherwise(sec_expr)

    rev_expr = F.lit(0)
    for code in _REVOLVING_CODES:
        rev_expr = F.when(acct == code, 1).otherwise(rev_expr)

    return df.withColumn("is_secured", sec_expr).withColumn("is_revolving", rev_expr)


def build_cross_dimension_features(history_df: DataFrame) -> DataFrame:
    """
    Cross-product-dimension comparisons revealing selective default.

    Features:
        secured_vs_unsecured_dpd_gap    max DPD secured − max DPD unsecured
        revolving_vs_term_pay_gap       payment ratio revolving − payment ratio term
        stage_divergence_flag           1 if secured CURRENT but unsecured NPL+
        behavioral_arbitrage_flag       1 if paying secured while defaulting unsecured
        concentration_risk_unsecured    Herfindahl index of unsecured balance
        strategic_default_indicator     low util secured + high DPD unsecured
    """
    R = CFG["ref_col"]

    df = _add_product_dim_flags(history_df)
    df = df.withColumn(
        "_dpd",
        F.coalesce(_as_int_safe_col(F.col("overduemonths")), F.lit(0))
        * CFG["odm_to_dpd_multiplier"]
    )
    df = df.withColumn(
        "_dpd_state",
        F.when(F.col("_dpd") == 0,             "S0")
         .when(F.col("_dpd").between(1,   30),  "S1")
         .when(F.col("_dpd").between(31,  90),  "S2")
         .when(F.col("_dpd").between(91, 180),  "S3")
         .otherwise("S4")
    )
    w_acct = Window.partitionBy(R, "accounttype").orderBy("asofdate")
    df = df.withColumn(
        "_pay_ratio",
        _safe_div(
            -(F.col("amountowed") - F.lag("amountowed", 1).over(w_acct)),
            F.lag("amountowed", 1).over(w_acct)))

    agg = (df.groupBy(R, "asofdate")
           .agg(
               F.max(F.when(F.col("is_secured") == 1, F.col("_dpd")))
                .alias("_max_dpd_sec"),
               F.max(F.when(F.col("is_secured") == 0, F.col("_dpd")))
                .alias("_max_dpd_unsec"),
               F.avg(F.when(F.col("is_revolving") == 1, F.col("_pay_ratio")))
                .alias("_pr_revl"),
               F.avg(F.when(F.col("is_revolving") == 0, F.col("_pay_ratio")))
                .alias("_pr_term"),
               F.max(F.when(
                   (F.col("is_secured") == 1) &
                   (F.col("_dpd_state") == "S0"), 1).otherwise(0))
                .alias("_sec_current"),
               F.max(F.when(
                   (F.col("is_secured") == 0) &
                   (F.col("_dpd_state").isin(["S3","S4"])), 1).otherwise(0))
                .alias("_unsec_npl_plus"),
               F.max(F.when(
                   (F.col("is_secured") == 1) & (F.col("_dpd") == 0), 1)
                .otherwise(0)).alias("_sec_paying"),
               F.max(F.when(
                   (F.col("is_secured") == 0) & (F.col("_dpd") > 90), 1)
                .otherwise(0)).alias("_unsec_default"),
               F.sum(F.when(F.col("is_secured") == 0,
                            F.col("amountowed") * F.col("amountowed")).otherwise(0))
                .alias("_unsec_bal_sq"),
               F.sum(F.when(F.col("is_secured") == 0,
                            F.col("amountowed")).otherwise(0))
                .alias("_unsec_bal"),
               F.avg(F.when(F.col("is_secured") == 1,
                            _safe_div(F.col("amountowed"), F.col("creditlimit"))))
                .alias("_sec_util"),
           )
           .withColumn("secured_vs_unsecured_dpd_gap",
                       F.col("_max_dpd_sec") - F.col("_max_dpd_unsec"))
           .withColumn("revolving_vs_term_pay_gap",
                       F.col("_pr_revl") - F.col("_pr_term"))
           .withColumn("stage_divergence_flag",
                       F.when((F.col("_sec_current") == 1) &
                              (F.col("_unsec_npl_plus") == 1), 1).otherwise(0))
           .withColumn("behavioral_arbitrage_flag",
                       F.when((F.col("_sec_paying") == 1) &
                              (F.col("_unsec_default") == 1), 1).otherwise(0))
           .withColumn("concentration_risk_unsecured",
                       _safe_div(
                           F.col("_unsec_bal_sq"),
                           F.col("_unsec_bal") * F.col("_unsec_bal")))
           .withColumn("strategic_default_indicator",
                       F.when(
                           (F.col("_sec_util") < 0.5) &
                           (F.col("_unsec_default") == 1), 1).otherwise(0))
           .select(R, "asofdate",
                   "secured_vs_unsecured_dpd_gap", "revolving_vs_term_pay_gap",
                   "stage_divergence_flag", "behavioral_arbitrage_flag",
                   "concentration_risk_unsecured", "strategic_default_indicator"))

    # Take latest snapshot per customer
    w_lat = Window.partitionBy(R).orderBy(F.col("asofdate").desc())
    return (agg
            .withColumn("_rn", F.row_number().over(w_lat))
            .filter(F.col("_rn") == 1)
            .drop("_rn", "asofdate")
            .dropDuplicates([R]))


def build_structural_roll_rates(episode_df: DataFrame) -> DataFrame:
    """
    Lifetime stage-pair roll rates and escalation/cure velocities.
    Structural segmentation signals only — no short-window features.

    Features:
        roll_rate_s0_to_s1          % months in S0 that transitioned to S1
        roll_rate_s1_to_s2          % months in S1 that transitioned to S2
        roll_rate_s2_to_s3          % months in S2 that transitioned to S3
        roll_rate_s3_to_s4          % months in S3 that transitioned to S4
        backward_roll_rate_s2       % months in S2 that improved to S1 or better
        backward_roll_rate_s3       % months in S3 that improved to S2 or better
        escalation_velocity_to_s3   months from first S0 to first S3 (shortest path)
        escalation_velocity_to_s2   months from first S0 to first S2
        cure_velocity_from_s3       months from S3 to S0 (shortest cure)
    """
    R = CFG["ref_col"]

    w_time = Window.partitionBy(R).orderBy("asofdate")
    w_all  = Window.partitionBy(R).orderBy("asofdate").rowsBetween(
                 Window.unboundedPreceding, 0)

    ord_map = _dpd_bucket_ordinal(F.col("episode_stage"))

    df = (episode_df
          .withColumn("_ord",        ord_map)
          .withColumn("_prev_stage",  F.lag("episode_stage", 1).over(w_time))
          .withColumn("_prev_ord",    F.lag("_ord", 1).over(w_time)))

    # ── Forward roll rates per stage pair ─────────────────────────────────────
    roll_pairs = [
        ("S0", "S1", "roll_rate_s0_to_s1"),
        ("S1", "S2", "roll_rate_s1_to_s2"),
        ("S2", "S3", "roll_rate_s2_to_s3"),
        ("S3", "S4", "roll_rate_s3_to_s4"),
    ]
    for from_s, to_s, col_name in roll_pairs:
        in_from = F.when(F.col("_prev_stage") == from_s, 1).otherwise(0)
        moved   = F.when(
            (F.col("_prev_stage") == from_s) & (F.col("episode_stage") == to_s),
            1).otherwise(0)
        df = df.withColumn(
            col_name,
            _safe_div(
                F.sum(moved).over(w_all),
                F.greatest(F.lit(1), F.sum(in_from).over(w_all)),
                F.lit(0.0)))

    # ── Backward roll rates (cure) ────────────────────────────────────────────
    df = df.withColumn(
        "backward_roll_rate_s2",
        _safe_div(
            F.sum(F.when(
                (F.col("_prev_stage") == "S2") &
                (F.col("_ord") < F.col("_prev_ord")), 1).otherwise(0)).over(w_all),
            F.greatest(F.lit(1),
                       F.sum(F.when(F.col("_prev_stage") == "S2", 1)
                              .otherwise(0)).over(w_all)),
            F.lit(0.0)))

    df = df.withColumn(
        "backward_roll_rate_s3",
        _safe_div(
            F.sum(F.when(
                (F.col("_prev_stage") == "S3") &
                (F.col("_ord") < F.col("_prev_ord")), 1).otherwise(0)).over(w_all),
            F.greatest(F.lit(1),
                       F.sum(F.when(F.col("_prev_stage") == "S3", 1)
                              .otherwise(0)).over(w_all)),
            F.lit(0.0)))

    keep_cols = [R, "asofdate"] + [p[2] for p in roll_pairs] + [
        "backward_roll_rate_s2", "backward_roll_rate_s3"]
    base = df.select(keep_cols).dropDuplicates([R, "asofdate"])

    # ── Escalation velocities (lifetime — episode-level) ─────────────────────
    w_lat = Window.partitionBy(R).orderBy(F.col("asofdate").desc())

    ep_first = (episode_df
                .filter(F.col("episode_month_number") == 1)
                .select(R, "episode_stage", "asofdate", "episode_id"))

    s0_entry  = ep_first.filter(F.col("episode_stage") == "S0") \
                        .select(R, F.col("asofdate").alias("_s0_start"),
                                   F.col("episode_id").alias("_s0_ep"))
    s2_entry  = ep_first.filter(F.col("episode_stage") == "S2") \
                        .select(R, F.col("asofdate").alias("_s2_start"),
                                   F.col("episode_id").alias("_s2_ep"))
    s3_entry  = ep_first.filter(F.col("episode_stage") == "S3") \
                        .select(R, F.col("asofdate").alias("_s3_start"),
                                   F.col("episode_id").alias("_s3_ep"))

    esc_s3 = (s0_entry.join(s3_entry, on=R, how="inner")
              .filter(F.col("_s3_ep") > F.col("_s0_ep"))
              .withColumn("_gap", F.months_between(F.col("_s3_start"), F.col("_s0_start")))
              .groupBy(R)
              .agg(F.min("_gap").alias("escalation_velocity_to_s3")))

    esc_s2 = (s0_entry.join(s2_entry, on=R, how="inner")
              .filter(F.col("_s2_ep") > F.col("_s0_ep"))
              .withColumn("_gap", F.months_between(F.col("_s2_start"), F.col("_s0_start")))
              .groupBy(R)
              .agg(F.min("_gap").alias("escalation_velocity_to_s2")))

    s3_start_df = ep_first.filter(F.col("episode_stage") == "S3") \
                          .select(R, F.col("asofdate").alias("_s3_start2"),
                                     F.col("episode_id").alias("_s3_ep2"))
    s0_after    = ep_first.filter(F.col("episode_stage") == "S0") \
                          .select(R, F.col("asofdate").alias("_s0_after"),
                                     F.col("episode_id").alias("_s0_ep2"))
    cure_s3 = (s3_start_df.join(s0_after, on=R, how="inner")
               .filter(F.col("_s0_ep2") > F.col("_s3_ep2"))
               .withColumn("_gap", F.months_between(F.col("_s0_after"), F.col("_s3_start2")))
               .groupBy(R)
               .agg(F.min("_gap").alias("cure_velocity_from_s3")))

    # Take latest snapshot row for roll rates, then join velocity scalars
    base_latest = (base
                   .withColumn("_rn", F.row_number().over(w_lat))
                   .filter(F.col("_rn") == 1)
                   .drop("_rn", "asofdate"))

    return (base_latest
            .join(esc_s3, on=R, how="left")
            .join(esc_s2, on=R, how="left")
            .join(cure_s3, on=R, how="left")
            .dropDuplicates([R]))


def build_stage_velocity_features(episode_df: DataFrame) -> DataFrame:
    """
    Stage-transition velocity and acceleration across customer lifecycle.

    Features:
        months_s0_to_s2          months from first S0 to first S2 entry
        months_s0_to_s3          months from first S0 to first S3 entry
        months_s0_to_s4          months from first S0 to first S4 entry
        months_s2_to_s4          months from first S2 to first S4 entry
        max_stage_ever_reached   highest stage ordinal (0–4)
        months_at_worst_stage    total months in worst stage
        deterioration_velocity   max_stage_ord / months_to_reach_it
        cure_velocity            avg (stages retreated / months in source stage)
        stage_churn_rate         episode count / months observed
        distinct_stages_ever     number of distinct stages visited
        time_in_current_stage_months months in most recent stage
        fastest_deterioration_speed  min months between consecutive worsening transitions
        npl_stickiness_index     avg duration of S3 episodes
        co_stickiness_index      avg duration of S4 episodes
        stage_reversal_rate      cure transitions / total transitions
        num_transitions_s2_to_s3 count S2→S3 (key regulatory gate)
        num_transitions_s3_to_s0 count S3→S0 (full cure — rare)
        num_transitions_s4_to_s3 count S4→S3 (partial cure)
        stages_traversed_12m     distinct stages in most recent 12 months
    """
    R = CFG["ref_col"]
    STAGES_LIST = list(STAGE_LABELS.keys())  # ["S0","S1","S2","S3","S4"]

    # First entry per stage
    first_entries = (episode_df
                     .filter(F.col("episode_month_number") == 1)
                     .groupBy(R, "episode_stage")
                     .agg(F.min("asofdate").alias("first_entry_month")))

    pivot = (first_entries
             .groupBy(R)
             .pivot("episode_stage", STAGES_LIST)
             .agg(F.min("first_entry_month")))
    for s in STAGES_LIST:
        pivot = pivot.withColumnRenamed(s, f"_fe_{s}")

    def _months(a, b):
        return F.when(
            F.col(a).isNotNull() & F.col(b).isNotNull(),
            F.months_between(F.col(b), F.col(a)))

    result = (pivot
              .withColumn("months_s0_to_s2",  _months("_fe_S0", "_fe_S2"))
              .withColumn("months_s0_to_s3",  _months("_fe_S0", "_fe_S3"))
              .withColumn("months_s0_to_s4",  _months("_fe_S0", "_fe_S4"))
              .withColumn("months_s2_to_s4",  _months("_fe_S2", "_fe_S4")))

    # Max stage ever reached
    max_stage = (episode_df
                 .withColumn("_ord", _dpd_bucket_ordinal(F.col("episode_stage")))
                 .groupBy(R)
                 .agg(F.max("_ord").alias("max_stage_ever_reached")))
    result = result.join(max_stage, on=R, how="left")

    # Deterioration velocity
    result = (result
              .withColumn("_months_to_max",
                          F.when(F.col("max_stage_ever_reached") >= 4, F.col("months_s0_to_s4"))
                           .when(F.col("max_stage_ever_reached") >= 3, F.col("months_s0_to_s3"))
                           .when(F.col("max_stage_ever_reached") >= 2, F.col("months_s0_to_s2")))
              .withColumn("deterioration_velocity",
                          F.when(
                              F.col("_months_to_max").isNotNull() &
                              (F.col("_months_to_max") > 0),
                              F.col("max_stage_ever_reached") / F.col("_months_to_max")))
              .drop("_months_to_max"))

    # Stage churn rate + distinct stages
    churn = (episode_df
             .filter(F.col("episode_month_number") == 1)
             .groupBy(R)
             .agg(F.count("episode_id").alias("_total_eps"),
                  F.countDistinct("episode_stage").alias("distinct_stages_ever"))
             .join(
                 episode_df.groupBy(R).agg(
                     F.countDistinct("asofdate").alias("_total_months")),
                 on=R, how="left")
             .withColumn("stage_churn_rate",
                         _safe_div(F.col("_total_eps"), F.col("_total_months"), F.lit(0.0))))
    result = result.join(churn.drop("_total_eps", "_total_months"), on=R, how="left")

    # Time in current (most recent) stage
    w_latest = Window.partitionBy(R).orderBy(F.col("asofdate").desc())
    latest_ep = (episode_df
                 .withColumn("_rn", F.row_number().over(w_latest))
                 .filter(F.col("_rn") == 1)
                 .select(R, F.col("episode_stage").alias("current_stage"),
                            F.col("episode_month_number").alias("time_in_current_stage_months")))
    result = result.join(latest_ep, on=R, how="left")

    # Stickiness indexes (avg episode duration for S3 and S4)
    w_ep_full = Window.partitionBy(R, "episode_id")
    ep_dur = (episode_df
              .withColumn("_ep_len", F.count("*").over(w_ep_full))
              .filter(F.col("episode_month_number") == 1))
    for stage, label in [("S3", "npl"), ("S4", "co")]:
        stick = (ep_dur
                 .filter(F.col("episode_stage") == stage)
                 .groupBy(R)
                 .agg(F.avg("_ep_len").alias(f"{label}_stickiness_index")))
        result = result.join(stick, on=R, how="left")

    # Episode-level transitions
    w_ep_ord = Window.partitionBy(R).orderBy("episode_id")
    ep_trans = (episode_df
                .filter(F.col("episode_month_number") == 1)
                .select(R, "episode_id", "episode_stage", "asofdate")
                .withColumn("next_stage", F.lead("episode_stage", 1).over(w_ep_ord))
                .withColumn("_ord",       _dpd_bucket_ordinal(F.col("episode_stage")))
                .withColumn("_next_ord",  _dpd_bucket_ordinal(F.lead("episode_stage", 1).over(w_ep_ord)))
                .filter(F.col("next_stage").isNotNull()))

    reversal = (ep_trans
                .groupBy(R)
                .agg(
                    F.count("*").alias("_total_trans"),
                    F.sum(F.when(F.col("_next_ord") < F.col("_ord"), 1).otherwise(0))
                     .alias("_cure_trans"),
                    F.sum(F.when((F.col("episode_stage") == "S2") &
                                 (F.col("next_stage") == "S3"), 1).otherwise(0))
                     .alias("num_transitions_s2_to_s3"),
                    F.sum(F.when((F.col("episode_stage") == "S3") &
                                 (F.col("next_stage") == "S0"), 1).otherwise(0))
                     .alias("num_transitions_s3_to_s0"),
                    F.sum(F.when((F.col("episode_stage") == "S4") &
                                 (F.col("next_stage") == "S3"), 1).otherwise(0))
                     .alias("num_transitions_s4_to_s3"),
                )
                .withColumn("stage_reversal_rate",
                            _safe_div(F.col("_cure_trans"), F.col("_total_trans"), F.lit(0.0)))
                .drop("_total_trans", "_cure_trans"))
    result = result.join(reversal, on=R, how="left")

    # Months at worst stage
    ep_with_ord = episode_df.withColumn("_ord", _dpd_bucket_ordinal(F.col("episode_stage")))
    worst_ord_per = ep_with_ord.groupBy(R).agg(F.max("_ord").alias("_worst_ord"))
    months_at_worst = (ep_with_ord
                       .join(worst_ord_per, on=R, how="inner")
                       .filter(F.col("_ord") == F.col("_worst_ord"))
                       .groupBy(R)
                       .agg(F.countDistinct("asofdate").alias("months_at_worst_stage")))
    result = result.join(months_at_worst, on=R, how="left")

    # Stages traversed in last 12 months
    cust_max_dt = episode_df.groupBy(R).agg(F.max("asofdate").alias("_max_dt"))
    stages_12m = (episode_df
                  .join(cust_max_dt, on=R, how="inner")
                  .filter(F.months_between(F.col("_max_dt"), F.col("asofdate")) <= 12)
                  .groupBy(R)
                  .agg(F.countDistinct("episode_stage").alias("stages_traversed_12m")))
    result = result.join(stages_12m, on=R, how="left")

    # Fastest deterioration speed + cure velocity
    ep_trans_dur = (ep_trans
                    .withColumn("_next_ep_dt", F.lead("asofdate", 1).over(w_ep_ord))
                    .withColumn("_ep_dur",
                                F.months_between(F.col("_next_ep_dt"), F.col("asofdate"))))

    fastest = (ep_trans_dur
               .filter((F.col("_next_ord") > F.col("_ord")) & F.col("_ep_dur").isNotNull())
               .groupBy(R)
               .agg(F.min("_ep_dur").alias("fastest_deterioration_speed")))
    result = result.join(fastest, on=R, how="left")

    cure_vel = (ep_trans_dur
                .filter((F.col("_next_ord") < F.col("_ord")) &
                        F.col("_ep_dur").isNotNull() & (F.col("_ep_dur") > 0))
                .withColumn("_stages_ret", F.col("_ord") - F.col("_next_ord"))
                .groupBy(R)
                .agg(F.avg(F.col("_stages_ret") / F.col("_ep_dur")).alias("cure_velocity")))
    result = result.join(cure_vel, on=R, how="left")

    for s in STAGES_LIST:
        result = result.drop(f"_fe_{s}")

    return result.dropDuplicates([R])


def build_cure_redefault_dynamics(episode_df: DataFrame) -> DataFrame:
    """
    Detect cure-then-re-default cycles from episode transitions.

    Cure event:  episode transition from S2/S3/S4 → S0/S1
    Re-default:  after a cure, re-enters S2+ within 24 months

    Features:
        cure_count_lifetime         number of cure events
        re_default_count_lifetime   number of re-default events
        re_default_flag             1 if any re-default
        avg_months_to_re_default    avg months cure→re-default
        min_months_to_re_default    fastest re-default
        structural_re_default_flag  1 if any re-default within 6 months
        durable_cure_flag           1 if any cure lasted 12+ months without re-default
        cure_depth_max              highest stage ordinal cured FROM
        avg_cure_holding_period     avg months in cured state before re-default
        cure_to_redefault_ratio     re_default_count / cure_count
    """
    R = CFG["ref_col"]
    w = Window.partitionBy(R).orderBy("episode_id")

    eps = (episode_df
           .filter(F.col("episode_month_number") == 1)
           .select(R, "episode_id", "episode_stage", "asofdate")
           .withColumn("_ord",      _dpd_bucket_ordinal(F.col("episode_stage")))
           .withColumn("_prev_ord", F.lag("_ord", 1).over(w))
           .withColumn("_next_ord", _dpd_bucket_ordinal(F.lead("episode_stage", 1).over(w)))
           .withColumn("_next_dt",  F.lead("asofdate", 1).over(w)))

    # Cure: entering S0/S1 (ord ≤1) from S2+ (prev_ord ≥2)
    eps = (eps
           .withColumn("_is_cure_dest",
                       F.when((F.col("_ord") <= 1) &
                              (F.col("_prev_ord") >= 2), 1).otherwise(0))
           .withColumn("_cure_src_ord",
                       F.when(F.col("_is_cure_dest") == 1, F.col("_prev_ord")))
           .withColumn("_is_redefault",
                       F.when((F.col("_is_cure_dest") == 1) &
                              (F.col("_next_ord") >= 2), 1).otherwise(0))
           .withColumn("_months_to_redef",
                       F.when(F.col("_is_redefault") == 1,
                              F.months_between(F.col("_next_dt"), F.col("asofdate"))))
           .withColumn("_cure_hold_months",
                       F.when(F.col("_is_cure_dest") == 1,
                              F.months_between(F.col("_next_dt"), F.col("asofdate")))))

    return (eps.groupBy(R)
               .agg(
                   F.sum("_is_cure_dest").alias("cure_count_lifetime"),
                   F.sum("_is_redefault").alias("re_default_count_lifetime"),
                   F.avg("_months_to_redef").alias("avg_months_to_re_default"),
                   F.min("_months_to_redef").alias("min_months_to_re_default"),
                   F.max("_cure_src_ord").alias("cure_depth_max"),
                   F.avg("_cure_hold_months").alias("avg_cure_holding_period"),
               )
               .withColumn("re_default_flag",
                           F.when(F.col("re_default_count_lifetime") > 0, 1).otherwise(0))
               .withColumn("structural_re_default_flag",
                           F.when(F.col("min_months_to_re_default") <= 6, 1).otherwise(0))
               .withColumn("durable_cure_flag",
                           F.when(
                               (F.col("cure_count_lifetime") > 0) &
                               ((F.col("re_default_count_lifetime") == 0) |
                                (F.col("min_months_to_re_default") >= 12)), 1)
                            .otherwise(0))
               .withColumn("cure_to_redefault_ratio",
                           _safe_div(F.col("re_default_count_lifetime"),
                                     F.col("cure_count_lifetime"), F.lit(0.0)))
               .dropDuplicates([R]))


def build_vintage_seasoning_features(episode_df: DataFrame) -> DataFrame:
    """
    Lifecycle timing features positioning default within account age.

    Recovery insight: A customer who defaulted 3 months after first bureau
    appearance is likely a bust-out (low recovery probability). One who was
    clean for 36 months before first S2 entry had a life event — higher
    recovery probability because they demonstrated long-term repayment.

    Features:
        months_observed_total              total months of bureau history
        account_age_at_first_default       months from first obs to first S2+ entry
        time_in_good_standing_pre_default  S0 months before first S2+ entry
        cumulative_current_ratio           months in S0 / total months
        early_life_default_flag            1 if first S2+ within 6 months of first obs
        seasoning_at_worst_stage           months from first obs to worst stage entry
        good_standing_to_default_ratio     S0_months / default_months
    """
    R = CFG["ref_col"]
    DEFAULT_STAGES = ["S2", "S3", "S4"]

    obs = episode_df.groupBy(R).agg(
        F.min("asofdate").alias("_first_obs"),
        F.countDistinct("asofdate").alias("months_observed_total"),
    )
    months_cur = (episode_df
                  .filter(F.col("episode_stage") == "S0")
                  .groupBy(R)
                  .agg(F.countDistinct("asofdate").alias("_months_cur")))

    first_def = (episode_df
                 .filter(F.col("episode_stage").isin(DEFAULT_STAGES) &
                         (F.col("episode_month_number") == 1))
                 .groupBy(R)
                 .agg(F.min("asofdate").alias("_first_def")))

    cur_pre_def = (episode_df
                   .filter(F.col("episode_stage") == "S0")
                   .join(first_def, on=R, how="left")
                   .filter(F.col("_first_def").isNull() |
                           (F.col("asofdate") < F.col("_first_def")))
                   .groupBy(R)
                   .agg(F.countDistinct("asofdate").alias("_cur_pre_def")))

    ep_ord = (episode_df
              .filter(F.col("episode_month_number") == 1)
              .withColumn("_ord", _dpd_bucket_ordinal(F.col("episode_stage"))))
    worst_per   = ep_ord.groupBy(R).agg(F.max("_ord").alias("_worst_ord"))
    worst_entry = (ep_ord
                   .join(worst_per, on=R, how="inner")
                   .filter(F.col("_ord") == F.col("_worst_ord"))
                   .groupBy(R)
                   .agg(F.min("asofdate").alias("_worst_entry")))

    return (obs
            .join(months_cur,  on=R, how="left")
            .join(first_def,   on=R, how="left")
            .join(cur_pre_def, on=R, how="left")
            .join(worst_entry, on=R, how="left")
            .withColumn("_months_cur",  F.coalesce(F.col("_months_cur"),  F.lit(0)))
            .withColumn("_cur_pre_def", F.coalesce(F.col("_cur_pre_def"), F.lit(0)))
            .withColumn("cumulative_current_ratio",
                        _safe_div(F.col("_months_cur"),
                                  F.col("months_observed_total"), F.lit(0.0)))
            .withColumn("account_age_at_first_default",
                        F.when(F.col("_first_def").isNotNull(),
                               F.months_between(F.col("_first_def"), F.col("_first_obs"))))
            .withColumn("time_in_good_standing_pre_default", F.col("_cur_pre_def"))
            .withColumn("early_life_default_flag",
                        F.when(F.col("account_age_at_first_default").isNotNull() &
                               (F.col("account_age_at_first_default") <= 6), 1)
                         .otherwise(0))
            .withColumn("seasoning_at_worst_stage",
                        F.when(F.col("_worst_entry").isNotNull(),
                               F.months_between(F.col("_worst_entry"), F.col("_first_obs"))))
            .withColumn("_default_months",
                        F.col("months_observed_total") - F.col("_months_cur"))
            .withColumn("good_standing_to_default_ratio",
                        _safe_div(F.col("_months_cur"), F.col("_default_months")))
            .drop("_first_obs", "_months_cur", "_first_def", "_cur_pre_def",
                  "_worst_entry", "_worst_ord", "_default_months")
            .dropDuplicates([R]))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 34 — RECOVERY STAGE INDICATORS
# Cumulative exposure and payment behaviour specifically while in CO (S4).
# Recovery insight: Two S4 accounts differ radically — one paid anything while
# in CO (latent willingness), one showed zero movement (DORMANT/STRATEGIC).
# ══════════════════════════════════════════════════════════════════════════════

def build_recovery_stage_indicators(
    episode_df: DataFrame,
    history_df: DataFrame,
) -> DataFrame:
    """
    Structural CO/CO_DEEP exposure and recovery-effort indicators.

    Features:
        time_in_co_overall              cumulative months in S4 (lifetime)
        time_in_co_deep_overall         cumulative months with DPD > 360 (S4 deep)
        pay_after_delinquency_flag_co   1 if any balance dip observed while in S4
        recovery_propensity_score_overall  pay_months_in_s4 / months_in_s4
    """
    R = CFG["ref_col"]

    # Cumulative S4 and deep-CO months from episode_df
    s4_months = (episode_df
                 .groupBy(R)
                 .agg(
                     F.sum(F.when(F.col("episode_stage") == "S4", 1).otherwise(0))
                      .alias("time_in_co_overall"),
                     F.sum(F.when(F.col("bureau_max_dpd") > 360, 1).otherwise(0))
                      .alias("time_in_co_deep_overall"),
                 ))

    # Monthly total balance per customer — balance dip = payment proxy
    monthly = history_df.groupBy(R, "asofdate").agg(
        F.sum("amountowed").alias("_tot_bal")
    )
    w_cust = Window.partitionBy(R).orderBy("asofdate")
    monthly = (monthly
               .withColumn("_prev_bal", F.lag("_tot_bal", 1).over(w_cust))
               .withColumn("_bal_dip",
                           F.when(F.col("_tot_bal") < F.col("_prev_bal"), 1).otherwise(0)))

    # Join to get S4 months with dip flag
    pay_in_s4 = (episode_df
                 .select(R, "asofdate", "episode_stage")
                 .join(monthly.select(R, "asofdate", "_bal_dip"),
                       on=[R, "asofdate"], how="left")
                 .filter(F.col("episode_stage") == "S4")
                 .groupBy(R)
                 .agg(
                     F.max("_bal_dip").alias("pay_after_delinquency_flag_co"),
                     F.sum("_bal_dip").alias("_pay_cnt_s4"),
                     F.count("*").alias("_months_s4"),
                 )
                 .withColumn("recovery_propensity_score_overall",
                             _safe_div(F.col("_pay_cnt_s4"),
                                       F.col("_months_s4"), F.lit(0.0)))
                 .drop("_pay_cnt_s4", "_months_s4"))

    all_ref = episode_df.select(R).distinct()
    return (all_ref
            .join(s4_months, on=R, how="left")
            .join(pay_in_s4,  on=R, how="left")
            .fillna({"time_in_co_overall": 0, "time_in_co_deep_overall": 0,
                     "pay_after_delinquency_flag_co": 0,
                     "recovery_propensity_score_overall": 0.0})
            .dropDuplicates([R]))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 35 — DIMENSION-LEVEL LIFETIME EXPOSURE
# Lifetime peak utilisation per product dimension.
# A customer who hit 95% utilisation on unsecured revolving (credit cards)
# at some point in history — even if currently at 60% — was previously
# running at the edge of capacity. That historical max is a structural signal.
# ══════════════════════════════════════════════════════════════════════════════

def build_dim_lifetime_exposure(history_df: DataFrame) -> DataFrame:
    """
    Lifetime peak utilisation per product dimension (7 dims).

    Features:
        max_util_ever_overall
        max_util_ever_secured
        max_util_ever_unsecured
        max_util_ever_secured_revl
        max_util_ever_secured_oth
        max_util_ever_unsecured_revl
        max_util_ever_unsecured_oth
    """
    R = CFG["ref_col"]

    df = _add_product_dim_flags(history_df)
    df = df.withColumn(
        "_bal",  F.coalesce(_as_int_safe_col(F.col("amountowed")),  F.lit(0)).cast("double")
    ).withColumn(
        "_lim",  F.coalesce(_as_int_safe_col(F.col("creditlimit")), F.lit(1)).cast("double")
    ).withColumn(
        "_util", _safe_div(F.col("_bal"), F.col("_lim"), F.lit(None))
    )

    sec  = F.col("is_secured") == 1
    usec = F.col("is_secured") == 0
    revl = F.col("is_revolving") == 1
    nrev = F.col("is_revolving") == 0

    return (df.groupBy(R).agg(
        F.max("_util")                                    .alias("max_util_ever_overall"),
        F.max(F.when(sec,          F.col("_util")))       .alias("max_util_ever_secured"),
        F.max(F.when(usec,         F.col("_util")))       .alias("max_util_ever_unsecured"),
        F.max(F.when(sec  & revl,  F.col("_util")))       .alias("max_util_ever_secured_revl"),
        F.max(F.when(sec  & nrev,  F.col("_util")))       .alias("max_util_ever_secured_oth"),
        F.max(F.when(usec & revl,  F.col("_util")))       .alias("max_util_ever_unsecured_revl"),
        F.max(F.when(usec & nrev,  F.col("_util")))       .alias("max_util_ever_unsecured_oth"),
    ).dropDuplicates([R]))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 36 — DIMENSION-LEVEL RELATIONSHIP TENURE
# How long has the customer been in each product category?
# A long secured-product relationship = more assets/collateral to protect.
# A customer new to unsecured revolving = less loyal, higher flight risk.
# Note: OVERALL tenure is covered by vintage_max_tradeline_age_months (Sec 13).
# ══════════════════════════════════════════════════════════════════════════════

def build_dim_relationship_tenure(account_df: DataFrame) -> DataFrame:
    """
    Months since first account opened per product dimension (6 per-product dims).
    PIT anchor: receive_dt from bridge (joined into account_df). Falls back to
    current_date() if receive_dt is absent.

    Features:
        relationship_tenure_secured
        relationship_tenure_unsecured
        relationship_tenure_secured_revl
        relationship_tenure_secured_oth
        relationship_tenure_unsecured_revl
        relationship_tenure_unsecured_oth
    """
    R = CFG["ref_col"]

    df = _add_product_dim_flags(account_df)

    # PIT anchor: most-recent receive_dt per customer (from bridge join)
    if "receive_dt" in account_df.columns:
        pit_df = account_df.groupBy(R).agg(
            F.max("receive_dt").alias("_pit")
        ).withColumn("_pit", F.to_date(F.col("_pit")))
    else:
        pit_df = account_df.select(R).distinct().withColumn("_pit", F.current_date())

    df = df.join(pit_df, on=R, how="left")

    # opendate: try yyyyMMdd integer string, fall back to auto-parse
    open_dt = F.to_date(F.col("opendate").cast("string"), "yyyyMMdd")

    sec  = F.col("is_secured") == 1
    usec = F.col("is_secured") == 0
    revl = F.col("is_revolving") == 1
    nrev = F.col("is_revolving") == 0

    agg = (df.groupBy(R, "_pit").agg(
        F.min(F.when(sec,         open_dt)).alias("_fo_sec"),
        F.min(F.when(usec,        open_dt)).alias("_fo_usec"),
        F.min(F.when(sec  & revl, open_dt)).alias("_fo_sec_revl"),
        F.min(F.when(sec  & nrev, open_dt)).alias("_fo_sec_oth"),
        F.min(F.when(usec & revl, open_dt)).alias("_fo_usec_revl"),
        F.min(F.when(usec & nrev, open_dt)).alias("_fo_usec_oth"),
    ))

    pit = F.col("_pit")
    return (agg
            .withColumn("relationship_tenure_secured",       F.months_between(pit, F.col("_fo_sec")))
            .withColumn("relationship_tenure_unsecured",     F.months_between(pit, F.col("_fo_usec")))
            .withColumn("relationship_tenure_secured_revl",  F.months_between(pit, F.col("_fo_sec_revl")))
            .withColumn("relationship_tenure_secured_oth",   F.months_between(pit, F.col("_fo_sec_oth")))
            .withColumn("relationship_tenure_unsecured_revl",F.months_between(pit, F.col("_fo_usec_revl")))
            .withColumn("relationship_tenure_unsecured_oth", F.months_between(pit, F.col("_fo_usec_oth")))
            .drop("_pit","_fo_sec","_fo_usec","_fo_sec_revl",
                  "_fo_sec_oth","_fo_usec_revl","_fo_usec_oth")
            .dropDuplicates([R]))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 37 — DIMENSION-LEVEL MAX STAGE REACHED
# Lifetime worst DPD stage (ordinal 0–4) per product dimension.
# A customer who reached S4 on unsecured but only S1 on secured = different
# risk profile than one who reached S4 on both simultaneously.
# Note: OVERALL is covered by historical_max_dpd_state (Section 11).
# ══════════════════════════════════════════════════════════════════════════════

def build_dim_max_stage_reached(history_df: DataFrame) -> DataFrame:
    """
    Lifetime worst DPD stage per product dimension (7 dims, ordinal 0–4).

    Features:
        max_stage_reached_overall
        max_stage_reached_secured
        max_stage_reached_unsecured
        max_stage_reached_secured_revl
        max_stage_reached_secured_oth
        max_stage_reached_unsecured_revl
        max_stage_reached_unsecured_oth
    """
    R = CFG["ref_col"]

    df = _add_product_dim_flags(history_df)
    df = df.withColumn(
        "_dpd",
        F.coalesce(_as_int_safe_col(F.col("overduemonths")), F.lit(0))
        * CFG["odm_to_dpd_multiplier"]
    ).withColumn(
        "_ord",
        F.when(F.col("_dpd") == 0,              0)
         .when(F.col("_dpd").between(1,   30),   1)
         .when(F.col("_dpd").between(31,  90),   2)
         .when(F.col("_dpd").between(91,  180),  3)
         .otherwise(4)
    )

    sec  = F.col("is_secured") == 1
    usec = F.col("is_secured") == 0
    revl = F.col("is_revolving") == 1
    nrev = F.col("is_revolving") == 0

    return (df.groupBy(R).agg(
        F.max("_ord")                                    .alias("max_stage_reached_overall"),
        F.max(F.when(sec,         F.col("_ord")))        .alias("max_stage_reached_secured"),
        F.max(F.when(usec,        F.col("_ord")))        .alias("max_stage_reached_unsecured"),
        F.max(F.when(sec  & revl, F.col("_ord")))        .alias("max_stage_reached_secured_revl"),
        F.max(F.when(sec  & nrev, F.col("_ord")))        .alias("max_stage_reached_secured_oth"),
        F.max(F.when(usec & revl, F.col("_ord")))        .alias("max_stage_reached_unsecured_revl"),
        F.max(F.when(usec & nrev, F.col("_ord")))        .alias("max_stage_reached_unsecured_oth"),
    ).dropDuplicates([R]))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 38 — PAYMENT EFFORT DYNAMICS PER STAGE
# Episode-level payment effort signals averaged across ALL lifetime episodes
# for each stage type. These are structural (lifetime avg) — NOT recency.
#
# Recovery insight:
# - A customer who made payments on 70% of NPL months across all NPL episodes
#   has demonstrated CAPACITY and WILLINGNESS under stress.
# - One who showed zero effort across every NPL episode = STRATEGIC / DORMANT.
# - avg_dip_recency_{stage}: HOW CLOSE TO EPISODE EXIT did the last payment occur?
#   (NOT current recency — this is a shape metric of episode dynamics)
# Ported from bureau_stage_dynamics.build_payment_effort_dynamics (Cat 11).
# ══════════════════════════════════════════════════════════════════════════════

def build_payment_effort_dynamics(
    episode_df: DataFrame,
    history_df: DataFrame,
) -> DataFrame:
    """
    Compute episode-level payment effort per stage, averaged over lifetime.

    All output features are lifetime averages/maxima across episodes — structural.

    Per-stage features (suffix = STAGE_LABELS value, e.g. _npl, _co):
        avg_payment_effort_ratio_{s}     dip_months / episode_months
        avg_payment_effort_intensity_{s} avg_dip_amount / balance_at_entry
        max_consecutive_no_dip_{s}       longest no-balance-dip streak (months)
        avg_dip_recency_{s}              months since last dip at episode exit (shape)
        avg_dip_acceleration_{s}         dips_second_half − dips_first_half
        avg_payment_momentum_{s}         recency-weighted dip score within episode
        avg_pay_to_balance_ratio_{s}     |dip| / prior_balance
        avg_payment_volatility_{s}       stddev of pay_to_balance_ratio
        max_partial_pay_streak_{s}       longest streak where 0 < pay_ratio < 0.30
    """
    R = CFG["ref_col"]

    monthly = history_df.groupBy(R, "asofdate").agg(
        F.sum("amountowed").alias("_total_bal")
    )

    df = (episode_df
          .select(R, "asofdate", "episode_id", "episode_stage", "episode_month_number")
          .join(monthly, on=[R, "asofdate"], how="left"))

    w_ep      = Window.partitionBy(R, "episode_id").orderBy("asofdate")
    w_ep_full = Window.partitionBy(R, "episode_id")

    df = (df
          .withColumn("_bal_prev", F.lag("_total_bal", 1).over(w_ep))
          .withColumn("_bal_dip",
                      F.when(F.col("_total_bal") < F.col("_bal_prev"), 1).otherwise(0))
          .withColumn("_dip_amount",
                      F.when(F.col("_bal_dip") == 1,
                             F.col("_bal_prev") - F.col("_total_bal"))
                       .otherwise(F.lit(0.0))))

    ep_len  = F.count("*").over(w_ep_full)
    half_pt = ep_len / 2

    df = (df
          .withColumn("_ep_len",    ep_len)
          .withColumn("_in_first",  F.when(F.col("episode_month_number") <= half_pt, 1).otherwise(0))
          .withColumn("_in_second", F.when(F.col("episode_month_number") >  half_pt, 1).otherwise(0)))

    # Consecutive no-dip streak via island detection (grouping by cumulative dip count)
    df = (df
          .withColumn("_no_dip",  F.when(F.col("_bal_dip") == 0, 1).otherwise(0))
          .withColumn("_dip_grp", F.sum("_bal_dip").over(w_ep))
          .withColumn("_no_dip_run",
                      F.row_number().over(
                          Window.partitionBy(R, "episode_id", "_dip_grp")
                                .orderBy("asofdate")) * F.col("_no_dip")))

    # Recency weight: later months in episode score higher
    df = df.withColumn(
        "_dip_recency_wt",
        F.when(F.col("_bal_dip") == 1,
               F.col("episode_month_number").cast("double") / F.col("_ep_len"))
         .otherwise(F.lit(0.0)))

    # Months since last dip at each point within episode
    df = (df
          .withColumn("_last_dip_mn",
                      F.when(F.col("_bal_dip") == 1, F.col("episode_month_number")))
          .withColumn("_last_dip_mn_filled",
                      F.last("_last_dip_mn", ignorenulls=True).over(w_ep))
          .withColumn("_dip_rec_months",
                      F.col("episode_month_number") -
                      F.coalesce(F.col("_last_dip_mn_filled"), F.lit(0))))

    # Pay-to-balance ratio and partial-pay streak
    df = (df
          .withColumn("_pay_ratio",
                      F.when(F.col("_bal_dip") == 1,
                             _safe_div(F.col("_dip_amount"), F.col("_bal_prev"), F.lit(0.0)))
                       .otherwise(F.lit(0.0)))
          .withColumn("_is_partial",
                      F.when((F.col("_pay_ratio") > 0) &
                             (F.col("_pay_ratio") < 0.30), 1).otherwise(0))
          .withColumn("_partial_brk",
                      F.when(F.col("_is_partial") == 0, 1).otherwise(0))
          .withColumn("_partial_grp", F.sum("_partial_brk").over(w_ep))
          .withColumn("_partial_run",
                      F.row_number().over(
                          Window.partitionBy(R, "episode_id", "_partial_grp")
                                .orderBy("asofdate")) * F.col("_is_partial")))

    # Episode-level aggregation
    ep_agg = (df.groupBy(R, "episode_id", "episode_stage")
              .agg(
                  F.count("*").alias("_ep_len"),
                  F.sum("_bal_dip").alias("_dip_cnt"),
                  F.avg("_dip_amount").alias("_avg_dip_amt"),
                  F.max("_no_dip_run").alias("_max_no_dip"),
                  F.max("_dip_rec_months").alias("_dip_rec_exit"),
                  F.sum(F.when(F.col("_in_first")  == 1, F.col("_bal_dip")).otherwise(0))
                   .alias("_dips_first"),
                  F.sum(F.when(F.col("_in_second") == 1, F.col("_bal_dip")).otherwise(0))
                   .alias("_dips_second"),
                  F.sum("_dip_recency_wt").alias("_momentum_raw"),
                  F.first("_total_bal").alias("_bal_entry"),
                  F.avg(F.when(F.col("_bal_dip") == 1, F.col("_pay_ratio")))
                   .alias("_avg_pr"),
                  F.stddev(F.when(F.col("_bal_dip") == 1, F.col("_pay_ratio")))
                   .alias("_pay_vol"),
                  F.max("_partial_run").alias("_partial_streak"),
              )
              .withColumn("_effort_ratio",
                          _safe_div(F.col("_dip_cnt"), F.col("_ep_len"), F.lit(0.0)))
              .withColumn("_effort_intensity",
                          _safe_div(F.col("_avg_dip_amt"), F.col("_bal_entry"), F.lit(0.0)))
              .withColumn("_dip_accel", F.col("_dips_second") - F.col("_dips_first"))
              .withColumn("_momentum",
                          _safe_div(F.col("_momentum_raw"), F.col("_dip_cnt"), F.lit(0.0))))

    # Stage-level aggregation
    stage_agg = (ep_agg.groupBy(R, "episode_stage")
                 .agg(
                     F.avg("_effort_ratio")    .alias("avg_payment_effort_ratio"),
                     F.avg("_effort_intensity").alias("avg_payment_effort_intensity"),
                     F.max("_max_no_dip")      .alias("max_consecutive_no_dip"),
                     F.avg("_dip_rec_exit")    .alias("avg_dip_recency"),
                     F.avg("_dip_accel")       .alias("avg_dip_acceleration"),
                     F.avg("_momentum")        .alias("avg_payment_momentum"),
                     F.avg("_avg_pr")          .alias("avg_pay_to_balance_ratio"),
                     F.avg("_pay_vol")         .alias("avg_payment_volatility"),
                     F.max("_partial_streak")  .alias("max_partial_pay_streak"),
                 ))

    metric_cols = [
        "avg_payment_effort_ratio", "avg_payment_effort_intensity",
        "max_consecutive_no_dip",   "avg_dip_recency",
        "avg_dip_acceleration",     "avg_payment_momentum",
        "avg_pay_to_balance_ratio", "avg_payment_volatility",
        "max_partial_pay_streak",
    ]

    # Pivot by stage: filter on S0/S1/... code, name columns with label
    all_ref = episode_df.select(R).distinct()
    for stage_code, stage_label in STAGE_LABELS.items():
        s = (stage_agg
             .filter(F.col("episode_stage") == stage_code)
             .select(R, *[F.col(c).alias(f"{c}_{stage_label}") for c in metric_cols]))
        all_ref = all_ref.join(s, on=R, how="left")

    return all_ref.dropDuplicates([R])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 39 — BALANCE STRUCTURAL INDICATORS
# Three complementary structural balance signals not covered elsewhere:
#   HHI:       how concentrated is debt across lenders? (single-lender risk)
#   Worst/Entry: how much did balance grow from first-CURRENT to worst stage?
#              (financing-need at peak stress vs baseline capacity)
#   Silence:   consecutive months with zero balance movement at timeline end
#              (DORMANT characterisation — not a recency flag, a COUNT)
# ══════════════════════════════════════════════════════════════════════════════

def build_balance_structural(
    episode_df: DataFrame,
    history_df: DataFrame,
) -> DataFrame:
    """
    Structural balance concentration, peak-vs-entry, and bureau silence.

    Features:
        balance_concentration_hhi   Herfindahl index across lenders at latest snapshot
        balance_at_worst_vs_entry   balance at worst stage / balance at first S0 entry
        total_bureau_silence_months consecutive zero-activity months at end of timeline
    """
    R = CFG["ref_col"]
    w_latest = Window.partitionBy(R).orderBy(F.col("asofdate").desc())

    monthly = history_df.groupBy(R, "asofdate").agg(
        F.sum("amountowed").alias("_tot_bal")
    )

    # ── Balance concentration HHI at latest snapshot ──────────────────────────
    lender_bal = (history_df
                  .groupBy(R, "asofdate", "membershortname")
                  .agg(F.sum("amountowed").alias("_lend_bal")))

    latest_snap = (monthly
                   .withColumn("_rn", F.row_number().over(w_latest))
                   .filter(F.col("_rn") == 1)
                   .select(R, F.col("asofdate").alias("_lat")))

    lend_at_latest = (lender_bal
                      .join(latest_snap, on=R, how="inner")
                      .filter(F.col("asofdate") == F.col("_lat"))
                      .drop("_lat"))

    hhi = (lend_at_latest
           .groupBy(R)
           .agg(
               F.sum("_lend_bal").alias("_tb"),
               F.sum(F.col("_lend_bal") * F.col("_lend_bal")).alias("_sum_sq"),
           )
           .withColumn("balance_concentration_hhi",
                       _safe_div(F.col("_sum_sq"),
                                 F.col("_tb") * F.col("_tb"), F.lit(0.0)))
           .select(R, "balance_concentration_hhi"))

    # ── Balance at worst stage vs first S0 entry ──────────────────────────────
    ep_bal = (episode_df
              .join(monthly.select(R, "asofdate", "_tot_bal"), on=[R, "asofdate"], how="left")
              .withColumn("_ord", _dpd_bucket_ordinal(F.col("episode_stage"))))

    first_cur = (ep_bal
                 .filter(F.col("episode_stage") == "S0")
                 .groupBy(R)
                 .agg(F.first("_tot_bal", ignorenulls=True).alias("_bal_first_cur")))

    worst_ord = ep_bal.groupBy(R).agg(F.max("_ord").alias("_worst_ord"))
    bal_worst = (ep_bal
                 .join(worst_ord, on=R, how="inner")
                 .filter(F.col("_ord") == F.col("_worst_ord"))
                 .withColumn("_rn2",
                             F.row_number().over(Window.partitionBy(R).orderBy("asofdate")))
                 .filter(F.col("_rn2") == 1)
                 .select(R, F.col("_tot_bal").alias("_bal_worst")))

    bal_ratio = (first_cur
                 .join(bal_worst, on=R, how="full")
                 .withColumn("balance_at_worst_vs_entry",
                             _safe_div(F.col("_bal_worst"), F.col("_bal_first_cur")))
                 .select(R, "balance_at_worst_vs_entry"))

    # ── Bureau silence: consecutive months with zero balance change at tail ───
    w_cust = Window.partitionBy(R).orderBy("asofdate")
    monthly_act = (monthly
                   .withColumn("_prev_bal", F.lag("_tot_bal", 1).over(w_cust))
                   .withColumn("_bal_chgd",
                               F.when(F.col("_tot_bal") != F.col("_prev_bal"), 1).otherwise(0)))

    silence = (monthly_act
               .withColumn("_act_cumrev",
                           F.sum("_bal_chgd").over(
                               Window.partitionBy(R)
                                     .orderBy(F.col("asofdate").desc())
                                     .rowsBetween(Window.unboundedPreceding, 0)))
               .filter(F.col("_act_cumrev") == 0)
               .groupBy(R)
               .agg(F.count("*").alias("total_bureau_silence_months")))

    all_ref = episode_df.select(R).distinct()
    return (all_ref
            .join(hhi,       on=R, how="left")
            .join(bal_ratio, on=R, how="left")
            .join(silence,   on=R, how="left")
            .fillna({"balance_concentration_hhi": 0.0,
                     "total_bureau_silence_months": 0})
            .dropDuplicates([R]))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 40 — DPD PROFILE SHAPE
# Trajectory shape classification and supporting metrics.
# Ported from bureau_stage_dynamics.build_dpd_profile_shape_features.
#
# Shape types and their recovery implications:
#   CLIFF:      sudden delinquency after extended clean period → life event
#               → high cure probability, respond to goodwill offers
#   SLIDE:      gradual 6–12M deterioration → structural income erosion
#               → needs repayment plan, not goodwill offer
#   OSCILLATOR: chronic bouncing → cannot sustain cure, low propensity
#               → agency/legal review
#   RECOVERING: DPD declining from a past peak → self-curing trajectory
#               → monitor, light intervention
#   STABLE:     minimal DPD movement (persistently low or high)
#               → reassess, check for strategic behaviour
#
# Feature classification (re-audited):
#   STRUCTURAL (lifetime/12M+ window): shape_type, zero_crossing_count,
#       max_single_jump, peak_to_current_ratio, time_above_90_24m,
#       monotone_flag, range_12m, std_12m
#   PROPENSITY (current/recent): dpd_convexity (6M comparison),
#       dpd_entry_speed (time-since-last-zero = recency of good behavior)
#   REDUNDANT (covered by dpd_diff_velocity_12m in Section 11): dpd_slope_12m
#     → computed here for shape_type logic but NOT added to feature lists
# ══════════════════════════════════════════════════════════════════════════════

def build_dpd_profile_shape(state_df: DataFrame) -> DataFrame:
    """
    DPD trajectory shape classification and shape metrics.
    Input: state_df with columns (ref_no, asofdate, bureau_max_dpd).
    """
    R    = CFG["ref_col"]
    DPD  = "bureau_max_dpd"

    df = state_df.withColumn(
        "_dpd", F.coalesce(F.col(DPD).cast("double"), F.lit(0.0))
    )

    w_cust   = Window.partitionBy(R).orderBy("asofdate")
    w_latest = Window.partitionBy(R).orderBy(F.col("asofdate").desc())
    w_12m    = Window.partitionBy(R).orderBy("asofdate").rowsBetween(-11, 0)
    w_24m    = Window.partitionBy(R).orderBy("asofdate").rowsBetween(-23, 0)
    w_6m_rec = Window.partitionBy(R).orderBy("asofdate").rowsBetween(-5,  0)
    w_6m_pri = Window.partitionBy(R).orderBy("asofdate").rowsBetween(-11, -6)
    w_all    = Window.partitionBy(R).orderBy("asofdate").rowsBetween(
                   Window.unboundedPreceding, 0)

    df = df.withColumn("_dpd_prev", F.lag("_dpd", 1).over(w_cust))
    df = df.withColumn("_dpd_delta", F.col("_dpd") - F.col("_dpd_prev"))

    # Shape metrics
    df = (df
          .withColumn("_dpd_12m_ago", F.first("_dpd").over(w_12m))
          # dpd_slope_12m — computed for shape_type logic only; NOT added to feature lists
          # (redundant with dpd_diff_velocity_12m already in SEGMENTATION_FEATURES Sec 11)
          .withColumn("_dpd_slope_12m",
                      _safe_div(F.col("_dpd") - F.col("_dpd_12m_ago"),
                                F.lit(12.0), F.lit(None)))
          .withColumn("dpd_std_12m",   F.stddev("_dpd").over(w_12m))
          .withColumn("dpd_range_12m",
                      F.max("_dpd").over(w_12m) - F.min("_dpd").over(w_12m))
          .withColumn("dpd_monotone_flag",
                      F.when(F.min("_dpd_delta").over(w_12m) >= 0, 1).otherwise(0))
          # dpd_convexity: recent 6M avg delta vs prior 6M avg delta (PROPENSITY)
          .withColumn("dpd_convexity",
                      F.avg("_dpd_delta").over(w_6m_rec) -
                      F.avg("_dpd_delta").over(w_6m_pri))
          .withColumn("dpd_time_above_90_24m",
                      F.sum(F.when(F.col("_dpd") > 90, 1).otherwise(0)).over(w_24m))
          .withColumn("_max_dpd_ever",  F.max("_dpd").over(w_all))
          .withColumn("dpd_peak_to_current_ratio",
                      _safe_div(F.col("_dpd"), F.col("_max_dpd_ever"), F.lit(None)))
          .withColumn("dpd_max_single_jump",
                      F.max(F.when(F.col("_dpd_delta") > 0,
                                   F.col("_dpd_delta")).otherwise(0)).over(w_all)))

    # Zero-crossing count (lifetime): DPD crosses 0 boundary
    df = (df
          .withColumn("_crossed_zero",
                      F.when(
                          ((F.col("_dpd") == 0) & (F.col("_dpd_prev") > 0)) |
                          ((F.col("_dpd") > 0)  & (F.col("_dpd_prev") == 0)),
                          1).otherwise(0))
          .withColumn("dpd_zero_crossing_count",
                      F.sum("_crossed_zero").over(w_all)))

    # dpd_entry_speed: months since DPD was last 0 (PROPENSITY — current recency)
    df = (df
          .withColumn("_last_zero_mn",
                      F.when(F.col("_dpd") == 0, F.col("asofdate")))
          .withColumn("_last_zero_dt",
                      F.last("_last_zero_mn", ignorenulls=True).over(w_cust))
          .withColumn("dpd_entry_speed",
                      F.when(F.col("_last_zero_dt").isNotNull(),
                             F.months_between(F.col("asofdate"), F.col("_last_zero_dt")))
                       .otherwise(F.lit(None))))

    # Shape classification
    df = df.withColumn(
        "dpd_shape_type",
        F.when(F.col("dpd_zero_crossing_count") >= 3, "OSCILLATOR")
         .when(
             (F.col("_dpd_slope_12m") < -5) &
             (F.col("dpd_peak_to_current_ratio") < 0.7),
             "RECOVERING")
         .when(
             (F.col("dpd_monotone_flag") == 1) &
             (F.col("dpd_entry_speed").isNotNull()) &
             (F.col("dpd_entry_speed") <= 3),
             "CLIFF")
         .when(
             (F.col("dpd_monotone_flag") == 1) &
             (F.col("dpd_entry_speed").isNotNull()) &
             (F.col("dpd_entry_speed") > 3),
             "SLIDE")
         .otherwise("STABLE"))

    keep = ["asofdate",
            "dpd_shape_type", "dpd_zero_crossing_count", "dpd_max_single_jump",
            "dpd_peak_to_current_ratio", "dpd_time_above_90_24m",
            "dpd_monotone_flag", "dpd_range_12m", "dpd_std_12m",
            "dpd_convexity", "dpd_entry_speed"]

    return (df
            .select([R] + keep)
            .withColumn("_rn", F.row_number().over(w_latest))
            .filter(F.col("_rn") == 1)
            .drop("_rn", "asofdate")
            .dropDuplicates([R]))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 41 — RESTRUCTURING SUCCESS FLAG
# Did the TDR (debt restructuring) actually work?
# bfc already has tdr_count_lifetime, tdr_months_since_last, tdr_adherence_rate
# (Section 26). This section adds the single missing outcome signal:
# Was the post-TDR DPD < 30 for the 6 months following the last restructuring?
# ══════════════════════════════════════════════════════════════════════════════

def build_restructuring_success(
    account_df: DataFrame,
    history_df: DataFrame,
) -> DataFrame:
    """
    Post-TDR outcome: did restructuring lead to recovery?

    Features:
        restructuring_success_flag  1 if max DPD < 30 in 6M after last TDR opendate
                                    0 if TDR occurred but DPD ≥ 30 post-TDR
                                    NULL if no TDR history
    """
    R = CFG["ref_col"]

    tdr_accounts = (account_df
                    .filter(F.col("accounttype").cast("string") == "90")
                    .groupBy(R)
                    .agg(F.max(F.to_date(F.col("opendate").cast("string"), "yyyyMMdd"))
                          .alias("_last_tdr_open")))

    monthly_dpd = (history_df
                   .groupBy(R, "asofdate")
                   .agg(F.max(
                       F.coalesce(_as_int_safe_col(F.col("overduemonths")), F.lit(0))
                       * CFG["odm_to_dpd_multiplier"]
                   ).alias("_max_dpd")))

    success = (monthly_dpd
               .join(tdr_accounts, on=R, how="inner")
               .filter(
                   F.to_date(F.col("asofdate")).between(
                       F.col("_last_tdr_open"),
                       F.add_months(F.col("_last_tdr_open"), 6)
                   )
               )
               .groupBy(R)
               .agg(F.max("_max_dpd").alias("_max_dpd_post_tdr"))
               .withColumn("restructuring_success_flag",
                           F.when(F.col("_max_dpd_post_tdr") < 30, 1).otherwise(0))
               .select(R, "restructuring_success_flag"))

    # Customers with no TDR: NULL (not 0 — absence of TDR ≠ failed TDR)
    return success.dropDuplicates([R])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 42 — STAGE STICKINESS SCORE
# ─────────────────────────────────────────────────────────────────────────────
# TEAM IMPLEMENTATION NOTE
# ─────────────────────────────────────────────────────────────────────────────
# stage_stickiness_score: probability that a customer remains in their CURRENT
# DPD stage in the next observation month, estimated from their own lifetime
# transition history.
#
# DEFINITION
#   For the customer's current stage S_curr, count:
#     stay_count  = number of times the customer was in S_curr at month t
#                   AND still in S_curr at month t+1 (lifetime history)
#     total_count = total number of months the customer was in S_curr (lifetime)
#   stage_stickiness_score = stay_count / total_count
#
# RANGE: 0.0 (never stays — always transitions) → 1.0 (always stays)
#
# INTERPRETATION IN PERSONA AXIS SCORING (trajectory_score):
#   If customer is currently in a bad stage (S2/S3/S4):
#     LOW stickiness  → likely to cure soon           → POSITIVE signal
#     HIGH stickiness → stuck in bad stage            → NEGATIVE signal
#   If customer is currently in good stage (S0/S1):
#     HIGH stickiness → stable                        → POSITIVE signal
#     LOW stickiness  → volatile, may deteriorate     → NEGATIVE signal
#
# DATA SOURCE: NCB bureau history (mnf_cra_rvw_s_history) — no CardX data needed.
# INPUT DF:    episodes DataFrame (output of _build_stage_episodes + dpd_states)
#              Requires columns: ref_no, stage (S0–S4), next_stage (lead 1 month)
#
# SCAFFOLD — implement using the pattern below:
# ─────────────────────────────────────────────────────────────────────────────
#
#   def build_stage_stickiness(episodes_df: DataFrame) -> DataFrame:
#       R = CFG["ref_col"]
#
#       # Current stage per customer = stage at latest observation month
#       current_stage = (
#           episodes_df
#           .groupBy(R)
#           .agg(F.last("stage", ignorenulls=True).alias("current_stage"))
#       )
#
#       # Lifetime stay counts: months where stage == next_stage (stayed)
#       stay = (
#           episodes_df
#           .filter(F.col("stage") == F.col("next_stage"))    # stayed in stage
#           .groupBy(R, "stage")
#           .agg(F.count("*").alias("stay_count"))
#       )
#
#       # Total months per stage per customer
#       total = (
#           episodes_df
#           .groupBy(R, "stage")
#           .agg(F.count("*").alias("total_count"))
#       )
#
#       # Join stay + total, compute stickiness per stage
#       stickiness = (
#           total
#           .join(stay, [R, "stage"], "left")
#           .withColumn("stickiness", _safe_div(F.col("stay_count"), F.col("total_count"), F.lit(0.0)))
#       )
#
#       # Keep only current stage stickiness per customer
#       result = (
#           stickiness
#           .join(current_stage, R)
#           .filter(F.col("stage") == F.col("current_stage"))
#           .groupBy(R)
#           .agg(F.first("stickiness").alias("stage_stickiness_score"))
#       )
#
#       return result
#
# ORCHESTRATOR: add as step [39/39] after build_restructuring_success:
#   stickiness = build_stage_stickiness(episodes)
#   final = final.join(stickiness, R, "left")
#
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 28 — MAIN ASSEMBLY FUNCTION
# Calls all sections in dependency order and joins all outputs on ref_no.
# Returns: one wide DataFrame per customer with ~650 features.
# ══════════════════════════════════════════════════════════════════════════════

def run_complete_bureau_features(
    spark: SparkSession,
    schema_name: str,
    bridge_df: DataFrame,
    cardx_monthly_df: Optional[DataFrame] = None,
    cardx_first_delq_df: Optional[DataFrame] = None,
) -> DataFrame:
    """
    Run complete bureau feature factory. Single entry point.

    Execution order (38 steps):
        1.  load_bureau_tables
        2.  build_bureau_panel
        3.  build_dpd_states               → Section 5
        4.  _build_stage_episodes          → Section 6 (internal)
        5.  build_within_stage_exposure_dynamics → Section 7
        6.  build_within_stage_loan_count_dynamics → Section 8
        7.  build_within_stage_dpd_dynamics → Section 9
        8.  build_cross_stage_transition_dynamics → Section 10
        9.  build_trajectory_and_physics   → Section 11
        10. build_regime_dependent_repayment → Section 12
        11. build_vintage_features         → Section 13
        12. build_lender_ecology           → Section 14
        13. build_enquiries                → Section 15
        14. build_rfm_and_payment_features → Section 16
        15. build_cross_lender_dynamics    → Section 17
        16. build_debt_prioritisation      → Section 18
        17. build_pre_existing_stress      → Section 19
        18. Join all on ref_no
        19. build_cross_domain_interactions → Section 20 (post-join)
        20. build_bfe_static_snapshot      → Section 22
        21. build_extended_vintage         → Section 23
        22. build_delinquency_regime       → Section 24
        23. build_tdr_restructuring        → Section 26
        24. build_legal_actions            → Section 27
        25. build_extended_interactions    → Section 25 (post-join)
        26. build_cross_dimension_features → Section 29 (structural selective-default)
        27. build_structural_roll_rates    → Section 30 (lifetime roll rates)
        28. build_stage_velocity_features  → Section 31 (deterioration/cure speed)
        29. build_cure_redefault_dynamics  → Section 32 (chronic re-defaulter)
        30. build_vintage_seasoning_features → Section 33 (bust-out vs life-event)
        31. build_recovery_stage_indicators → Section 34 (CO exposure & effort)
        32. build_dim_lifetime_exposure    → Section 35 (max util ever per dim)
        33. build_dim_relationship_tenure  → Section 36 (tenure per product dim)
        34. build_dim_max_stage_reached    → Section 37 (worst stage per dim)
        35. build_payment_effort_dynamics  → Section 38 (episode payment effort)
        36. build_balance_structural       → Section 39 (HHI, worst/entry, silence)
        37. build_dpd_profile_shape        → Section 40 (CLIFF/SLIDE/OSCILLATOR/...)
        38. build_restructuring_success    → Section 41 (post-TDR DPD outcome)
        39. build_stage_stickiness         → Section 42 (stage stay probability — TEAM TODO)

    Args:
        spark:              SparkSession (Databricks)
        schema_name:        NCB schema, e.g. "cdx_mdz_prd.cdx_persist_mnf_res_db"
        bridge_df:          Bridge table with ref_no, receive_dt, as_of_month
        cardx_monthly_df:   Optional CardX internal monthly DPD + balance
                            (enables Sections 17, 19). Columns: ref_no, asofdate,
                            cardx_dpd, cardx_state, cardx_balance, cardx_credit_limit
        cardx_first_delq_df: Optional first-delinquency anchor per account.
                             Columns: ref_no, cardx_first_delinquency_month

    Returns:
        DataFrame keyed on ref_no with ~650 bureau features.
        Use SEGMENTATION_FEATURES / PROPENSITY_FEATURES lists for model inputs.
    """
    R = CFG["ref_col"]
    sep = "═" * 72

    print(sep)
    print("BUREAU FEATURE COMPLETE — v4.2")
    print(f"CardX monthly data : {'YES' if cardx_monthly_df    is not None else 'NO (cross-lender features disabled)'}")
    print(f"CardX delq anchor  : {'YES' if cardx_first_delq_df is not None else 'NO (pre-existing stress disabled)'}")
    print(sep)

    # ── 1-2. Load and build panel ─────────────────────────────────────────────
    print("\n[1/25] Loading bureau tables...")
    tables = load_bureau_tables(spark, schema_name)

    print("[2/25] Building PIT-safe bureau panel...")
    panel   = build_bureau_panel(tables, bridge_df)
    history = panel["history"]
    account = panel["account"]
    enquiry = panel["enquiry"]

    # ── 3. DPD states ─────────────────────────────────────────────────────────
    print("[3/25] Building DPD states (S0-S4)...")
    dpd_states = build_dpd_states(history)

    # ── 4. Stage episodes ─────────────────────────────────────────────────────
    print("[4/25] Building stage episodes (contiguous-run IDs)...")
    episodes = _build_stage_episodes(dpd_states)

    # ── 5-8. Within-stage and cross-stage features ────────────────────────────
    print("[5/25] Within-stage exposure dynamics (balance dip proxy)...")
    exposure = build_within_stage_exposure_dynamics(episodes, history)

    print("[6/25] Within-stage loan count dynamics...")
    loans = build_within_stage_loan_count_dynamics(episodes, history)

    print("[7/25] Within-stage DPD counter dynamics...")
    dpd_dyn = build_within_stage_dpd_dynamics(episodes)

    print("[8/25] Cross-stage transition dynamics (roll-fwd/cure rates)...")
    transitions = build_cross_stage_transition_dynamics(episodes)

    # ── 9-10. Physics and regime payment ─────────────────────────────────────
    print("[9/25] Trajectory & physics features (velocity/entropy/inertia)...")
    physics = build_trajectory_and_physics(dpd_states)

    print("[10/25] Regime-dependent repayment (NORMAL vs STRESSED delta)...")
    regime_pay = build_regime_dependent_repayment(dpd_states, history)

    # ── 11-15. Structural and ecology features ────────────────────────────────
    print("[11/25] Vintage & account maturity features...")
    vintage = build_vintage_features(account)

    print("[12/25] Lender ecology (type mix, HHI concentration)...")
    ecology = build_lender_ecology(account)

    print("[13/25] Enquiries (credit seeking, rejection proxy)...")
    enquiries = build_enquiries(enquiry)

    print("[14/25] Payment features (RFM scores, regime, elasticity)...")
    rfm = build_rfm_and_payment_features(history)

    # ── 15-17. Cross-lender and stress features ───────────────────────────────
    print("[15/25] Cross-lender dynamics (CardX vs bureau alignment)...")
    cross_lender = build_cross_lender_dynamics(dpd_states, cardx_monthly_df)

    print("[16/25] Debt prioritisation (secured vs unsecured)...")
    debt_priority = build_debt_prioritisation(history)

    print("[17/25] Pre-existing bureau stress detection...")
    pre_stress = build_pre_existing_stress(episodes, cardx_first_delq_df)

    # ── 18. Join all on ref_no ────────────────────────────────────────────────
    print("[18/25] Joining all feature sets on ref_no...")

    base = panel["panel"].select(R, "as_of_month").dropDuplicates([R])

    feature_sets = [
        (physics,       "trajectory_physics",    [R, "asofdate"]),
        (exposure,      "exposure_dynamics",      [R]),
        (loans,         "loan_count",             [R]),
        (dpd_dyn,       "dpd_dynamics",           [R]),
        (transitions,   "stage_transitions",      [R]),
        (regime_pay,    "regime_repayment",       [R]),
        (vintage,       "vintage_maturity",       [R]),
        (ecology,       "lender_ecology",         [R]),
        (enquiries,     "enquiries",              [R]),
        (rfm,           "rfm_payment",            [R]),
        (cross_lender,  "cross_lender",           [R]),
        (debt_priority, "debt_priority",          [R]),
        (pre_stress,    "pre_stress",             [R]),
    ]

    final = base
    for feat_df, name, keys in feature_sets:
        # physics features are at ref_no × asofdate grain; others at ref_no grain
        # Take most recent snapshot for time-series features
        if "asofdate" in feat_df.columns and keys == [R, "asofdate"]:
            w_latest = Window.partitionBy(R).orderBy(F.col("asofdate").desc())
            feat_df = (feat_df.withColumn("_rn", F.row_number().over(w_latest))
                       .filter(F.col("_rn") == 1).drop("_rn", "asofdate"))
        before = final.count()
        final = final.join(feat_df, on=R, how="left")
        after  = final.count()
        print(f"  ✓ {name:<28} {before:>8,} → {after:>8,} rows | +{len(feat_df.columns)-1} features")

    # ── 19. Cross-domain interactions (post-join, Section 20) ────────────────
    print("[19/25] Cross-domain interaction features (Section 20)...")
    final = build_cross_domain_interactions(final)

    # ── 20. BFE static snapshot (Section 22) ─────────────────────────────────
    print("[20/25] BFE static snapshot features (Section 22)...")
    bfe_snap = build_bfe_static_snapshot(tables["account"])
    final = final.join(bfe_snap, R, "left")
    print(f"  ✓ bfe_static_snapshot   +{len(bfe_snap.columns)-1} features")

    # ── 21. Extended vintage (Section 23) ────────────────────────────────────
    print("[21/25] Extended vintage & lifecycle (Section 23)...")
    ext_vint = build_extended_vintage(tables["history"], tables["account"])
    final = final.join(ext_vint, R, "left")
    print(f"  ✓ extended_vintage      +{len(ext_vint.columns)-1} features")

    # ── 22. Delinquency regime (Section 24) ──────────────────────────────────
    print("[22/25] Delinquency regime classification (Section 24)...")
    delq_regime = build_delinquency_regime(dpd_states)
    final = final.join(delq_regime, R, "left")
    print(f"  ✓ delinquency_regime    +{len(delq_regime.columns)-1} features")

    # ── 23. TDR / restructuring (Section 26) ─────────────────────────────────
    print("[23/25] TDR/restructuring dynamics (Section 26)...")
    tdr = build_tdr_restructuring(tables["account"])
    final = final.join(tdr, R, "left")
    print(f"  ✓ tdr_restructuring     +{len(tdr.columns)-1} features")

    # ── 24. Legal actions (Section 27) ───────────────────────────────────────
    print("[24/25] Legal actions & settlement (Section 27)...")
    legal = build_legal_actions(tables["account"])
    final = final.join(legal, R, "left")
    print(f"  ✓ legal_actions         +{len(legal.columns)-1} features")

    # ── 25. Extended BFE interactions (Section 25, must be last before new) ────
    print("[25/30] Extended BFE interaction flags (Section 25)...")
    final = build_extended_interactions(final)

    # ── 26–30. Structural additions from bureau_stage_dynamics (Sections 29–33) ─
    print("[26/30] Cross-dimension selective-default features (Section 29)...")
    cross_dim = build_cross_dimension_features(history)
    final = final.join(cross_dim, R, "left")
    print(f"  ✓ cross_dimension       +{len(cross_dim.columns)-1} features")

    print("[27/30] Structural roll rates & escalation/cure velocities (Section 30)...")
    roll_rates = build_structural_roll_rates(episodes)
    final = final.join(roll_rates, R, "left")
    print(f"  ✓ structural_roll_rates +{len(roll_rates.columns)-1} features")

    print("[28/30] Stage velocity features (Section 31)...")
    stage_vel = build_stage_velocity_features(episodes)
    final = final.join(stage_vel, R, "left")
    print(f"  ✓ stage_velocity        +{len(stage_vel.columns)-1} features")

    print("[29/30] Cure & re-default dynamics (Section 32)...")
    cure_redef = build_cure_redefault_dynamics(episodes)
    final = final.join(cure_redef, R, "left")
    print(f"  ✓ cure_redefault        +{len(cure_redef.columns)-1} features")

    print("[30/38] Vintage seasoning features (Section 33)...")
    vintage_s = build_vintage_seasoning_features(episodes)
    final = final.join(vintage_s, R, "left")
    print(f"  ✓ vintage_seasoning     +{len(vintage_s.columns)-1} features")

    # ── 31–38. New structural completions (Sections 34–41) ────────────────────
    print("[31/38] Recovery stage indicators — CO exposure & effort (Section 34)...")
    recov_stg = build_recovery_stage_indicators(episodes, history)
    final = final.join(recov_stg, R, "left")
    print(f"  ✓ recovery_stage        +{len(recov_stg.columns)-1} features")

    print("[32/38] Dimension lifetime exposure — max util ever per dim (Section 35)...")
    dim_exp = build_dim_lifetime_exposure(history)
    final = final.join(dim_exp, R, "left")
    print(f"  ✓ dim_lifetime_exposure +{len(dim_exp.columns)-1} features")

    print("[33/38] Dimension relationship tenure — per product dim (Section 36)...")
    dim_ten = build_dim_relationship_tenure(account)
    final = final.join(dim_ten, R, "left")
    print(f"  ✓ dim_tenure            +{len(dim_ten.columns)-1} features")

    print("[34/38] Dimension max stage reached — per product dim (Section 37)...")
    dim_max = build_dim_max_stage_reached(history)
    final = final.join(dim_max, R, "left")
    print(f"  ✓ dim_max_stage         +{len(dim_max.columns)-1} features")

    print("[35/38] Payment effort dynamics — episode structural effort (Section 38)...")
    pay_eff = build_payment_effort_dynamics(episodes, history)
    final = final.join(pay_eff, R, "left")
    print(f"  ✓ payment_effort        +{len(pay_eff.columns)-1} features")

    print("[36/38] Balance structural — HHI, worst/entry, silence (Section 39)...")
    bal_str = build_balance_structural(episodes, history)
    final = final.join(bal_str, R, "left")
    print(f"  ✓ balance_structural    +{len(bal_str.columns)-1} features")

    print("[37/38] DPD profile shape — CLIFF/SLIDE/OSCILLATOR/... (Section 40)...")
    dpd_shp = build_dpd_profile_shape(dpd_states)
    final = final.join(dpd_shp, R, "left")
    print(f"  ✓ dpd_profile_shape     +{len(dpd_shp.columns)-1} features")

    print("[38/39] Restructuring success — post-TDR DPD outcome (Section 41)...")
    restr_ok = build_restructuring_success(account, history)
    final = final.join(restr_ok, R, "left")
    print(f"  ✓ restructuring_success +{len(restr_ok.columns)-1} features")

    # [39/39] stage_stickiness_score — Section 42
    # TEAM TODO: uncomment once build_stage_stickiness is implemented
    # stickiness = build_stage_stickiness(episodes)
    # final = final.join(stickiness, R, "left")
    # print(f"  ✓ stage_stickiness      +1 feature")

    final = final.dropDuplicates([R])

    n_features = len(final.columns) - 2  # minus ref_no + as_of_month
    print(f"\n{sep}")
    print(f"✓ Complete — {n_features} features | {final.count():,} accounts")
    print(f"  Segmentation features : {len([c for c in SEGMENTATION_FEATURES if c in final.columns])}")
    print(f"  Propensity features   : {len([c for c in PROPENSITY_FEATURES if c in final.columns])}")
    print(sep)

    return final
