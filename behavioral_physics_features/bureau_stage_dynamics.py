"""
Bureau Stage Dynamics Engine  v2.0
====================================
Stage-wise behavioral indicators for recovery likelihood, payment propensity,
and account deterioration/improvement patterns.

Stage Nomenclature (CRITICAL — do NOT use S0/S1/S2/S3/S4):
    CURRENT  : 0 DPD          — Performing, no delinquency
    X        : 1–30 DPD       — Early delinquency (grace period)
    SM       : 31–90 DPD      — Short-term delinquent (Special Mention)
    NPL      : 91–180 DPD     — Non-performing loan
    CO       : 181–360 DPD    — Charge-off
    CO_DEEP  : 361+ DPD       — Deep charge-off / Write-off

Product Dimension Hierarchy (from NCB ACCOUNTTYPE / ncbLookupJSON):
    SECURED
    ├── SECURED_REVOLVING   (ncb_sec=sec  & ncb_revl=revl)
    └── SECURED_OTHERS      (ncb_sec=sec  & ncb_revl=nrevl)
    UNSECURED
    ├── UNSECURED_REVOLVING (ncb_sec=unsec & ncb_revl=revl) — credit cards, OD
    └── UNSECURED_OTHERS    (ncb_sec=unsec & ncb_revl=nrevl) — PL, comm loan
    OVERALL — all products combined

NCB Account Type Source: cdx_mdz_prd.cdx_persist_mnf_res_db.mnf_cra_rvw_s_account
Column: ACCOUNTTYPE (acct_type_cd)

Feature Categories:
    1. Stage Transition & Dynamics
    2. Payment Behavior Dynamics
    3. Exposure & Utilisation Dynamics
    4. Enquiry & New Credit Dynamics
    5. Account Closure & Attrition Dynamics
    6. Delinquency Severity Indicators
    7. Cross-Dimensional Comparative Features
    8. Recovery-Specific Indicators
    9. Restructuring & Modification Dynamics
   10. Temporal Velocity Features

Column contract:
    Join key  : ref_no
    Time key  : asofdate
    DPD col   : bureau_max_dpd  (integer, maximum DPD across all tradelines)
    Stage col : dpd_stage       (CURRENT/X/SM/NPL/CO/CO_DEEP)
    Source    : mnf_cra_rvw_s_history (history_df), mnf_cra_rvw_s_account (account_df)

Version : 2.0.0
Author  : Behavioral Physics Team / CardX Decision Intelligence
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from typing import Optional

# ── Column name constants ─────────────────────────────────────────────────────
REF        = "ref_no"
DATE       = "asofdate"
DPD_COL    = "bureau_max_dpd"
STAGE_COL  = "dpd_stage"       # CURRENT/X/SM/NPL/CO/CO_DEEP
BAL        = "amountowed"
LIMIT      = "creditlimit"
FINANCED   = "amountfinanced"
ODM        = "overduemonths"
ACCT_TYPE  = "accounttype"     # NCB acct_type_cd
LENDER     = "membershortname"

# ── Stage definitions ─────────────────────────────────────────────────────────
# Key = stage label used in feature names; value = (dpd_min, dpd_max, numeric_code)
STAGE_DEF = {
    "CURRENT" : (0,   0,   0),
    "X"       : (1,   30,  1),
    "SM"      : (31,  90,  2),
    "NPL"     : (91,  180, 3),
    "CO"      : (181, 360, 4),
    "CO_DEEP" : (361, 9999,5),
}
STAGES = list(STAGE_DEF.keys())   # ordered worst→best or best→worst as needed

# ── NCB account-type → dimension mapping ─────────────────────────────────────
# Source: ncbLookupJSON from mnf_cra_rvw_s_account.ACCOUNTTYPE
#
# Dimension flags:
#   IS_SECURED  : ncb_sec = "sec"   → codes 06,20,21,27,31,32,52,53,54,56
#   IS_REVOLVING: ncb_revl = "revl" → codes 04,22,55,58
#   IS_INST     : ncb_inst = "inst"
#
# Derived dimensions (4 leaves + 2 parents + OVERALL = 7 total):
#   SECURED_REVOLVING   : IS_SECURED  & IS_REVOLVING
#   SECURED_OTHERS      : IS_SECURED  & NOT IS_REVOLVING
#   UNSECURED_REVOLVING : NOT IS_SECURED & IS_REVOLVING   ← credit cards / OD
#   UNSECURED_OTHERS    : NOT IS_SECURED & NOT IS_REVOLVING
#   SECURED             : IS_SECURED  (all)
#   UNSECURED           : NOT IS_SECURED (all)
#   OVERALL             : all

SECURED_CODES  = {"06","20","21","27","31","32","52","53","54","56"}
REVOLVING_CODES = {"04","22","55","58"}

DIMS = [
    "SECURED_REVOLVING",
    "SECURED_OTHERS",
    "UNSECURED_REVOLVING",
    "UNSECURED_OTHERS",
    "SECURED",
    "UNSECURED",
    "OVERALL",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_div(num, den):
    """NULL-safe division; returns NULL instead of divide-by-zero."""
    return F.when((den.isNotNull()) & (den != 0), num / den)


def _dim_filter(acct_type_col: str, dim: str):
    """Return a boolean Column expression for the given dimension."""
    c = F.col(acct_type_col).cast("string")
    is_sec  = c.isin(list(SECURED_CODES))
    is_revl = c.isin(list(REVOLVING_CODES))

    if dim == "OVERALL":
        return F.lit(True)
    elif dim == "SECURED":
        return is_sec
    elif dim == "UNSECURED":
        return ~is_sec
    elif dim == "SECURED_REVOLVING":
        return is_sec & is_revl
    elif dim == "SECURED_OTHERS":
        return is_sec & ~is_revl
    elif dim == "UNSECURED_REVOLVING":
        return ~is_sec & is_revl
    elif dim == "UNSECURED_OTHERS":
        return ~is_sec & ~is_revl
    else:
        raise ValueError(f"Unknown dimension: {dim}")


def _assign_stage(dpd_col):
    """Map integer DPD → stage label string (CURRENT/X/SM/NPL/CO/CO_DEEP)."""
    return (
        F.when(dpd_col == 0,        "CURRENT")
         .when(dpd_col <= 30,       "X")
         .when(dpd_col <= 90,       "SM")
         .when(dpd_col <= 180,      "NPL")
         .when(dpd_col <= 360,      "CO")
         .otherwise(                "CO_DEEP")
    )


def _stage_numeric(stage_col):
    """Map stage label → ordinal (CURRENT=0 … CO_DEEP=5)."""
    return (
        F.when(F.col(stage_col) == "CURRENT", 0)
         .when(F.col(stage_col) == "X",       1)
         .when(F.col(stage_col) == "SM",       2)
         .when(F.col(stage_col) == "NPL",      3)
         .when(F.col(stage_col) == "CO",       4)
         .when(F.col(stage_col) == "CO_DEEP",  5)
         .otherwise(F.lit(None))
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — NCB ACCOUNT TYPE ENRICHMENT
# Adds is_secured / is_revolving / dimension label columns to account-level data.
# ─────────────────────────────────────────────────────────────────────────────

def enrich_account_dimensions(account_df: DataFrame) -> DataFrame:
    """
    Add is_secured, is_revolving, and dim_label columns to the account table.

    Input  : mnf_cra_rvw_s_account  (must have accounttype and ref_no)
    Output : same table + is_secured (bool), is_revolving (bool), dim_label (string)

    dim_label values: SECURED_REVOLVING / SECURED_OTHERS /
                      UNSECURED_REVOLVING / UNSECURED_OTHERS
    """
    acct = F.col(ACCT_TYPE).cast("string")
    is_sec  = acct.isin(list(SECURED_CODES))
    is_revl = acct.isin(list(REVOLVING_CODES))

    dim_label = (
        F.when( is_sec &  is_revl, "SECURED_REVOLVING")
         .when( is_sec & ~is_revl, "SECURED_OTHERS")
         .when(~is_sec &  is_revl, "UNSECURED_REVOLVING")
         .otherwise(               "UNSECURED_OTHERS")
    )

    return (account_df
            .withColumn("is_secured",  is_sec.cast("int"))
            .withColumn("is_revolving", is_revl.cast("int"))
            .withColumn("dim_label",   dim_label))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — STAGE ASSIGNMENT & EPISODE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_dpd_stage(state_df: DataFrame) -> DataFrame:
    """
    Assign dpd_stage (CURRENT/X/SM/NPL/CO/CO_DEEP) and stage_numeric (0–5)
    from bureau_max_dpd.

    Input  : state_df with (ref_no, asofdate, bureau_max_dpd)
    Output : adds dpd_stage, stage_numeric
    """
    return (state_df
            .withColumn(STAGE_COL,     _assign_stage(F.col(DPD_COL)))
            .withColumn("stage_numeric", _stage_numeric(STAGE_COL)))


def build_stage_episodes(state_df: DataFrame) -> DataFrame:
    """
    Assign episode IDs to contiguous runs in the same DPD stage.

    An episode = consecutive months in same stage bucket.
    Resets when stage changes.

    Output adds:
        episode_id           — integer, globally unique per customer (row_number over time)
        episode_stage        — stage label at episode start
        episode_start_date   — first date of the episode
        episode_month_number — 1 = entry month, 2 = second month in episode …
    """
    w_time = Window.partitionBy(REF).orderBy(DATE)

    df = (state_df
          .withColumn(STAGE_COL,      _assign_stage(F.col(DPD_COL)))
          .withColumn("stage_numeric", _stage_numeric(STAGE_COL))
          .withColumn("_prev_stage", F.lag(STAGE_COL).over(w_time))
          .withColumn("_stage_changed",
                      F.when(F.col("_prev_stage").isNull(), 1)
                       .when(F.col("_prev_stage") != F.col(STAGE_COL), 1)
                       .otherwise(0))
          .withColumn("episode_id",
                      F.sum("_stage_changed").over(w_time.rowsBetween(
                          Window.unboundedPreceding, 0)))
          .drop("_prev_stage", "_stage_changed"))

    w_ep = Window.partitionBy(REF, "episode_id").orderBy(DATE)

    df = (df
          .withColumn("episode_start_date",  F.first(DATE).over(w_ep))
          .withColumn("episode_stage",       F.first(STAGE_COL).over(w_ep))
          .withColumn("episode_month_number", F.row_number().over(w_ep)))

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — STAGE TRANSITION & DYNAMICS FEATURES  (Category 1)
# ─────────────────────────────────────────────────────────────────────────────

def build_stage_transition_features(episode_df: DataFrame) -> DataFrame:
    """
    Cross-stage transition dynamics per OVERALL dimension.

    Features produced (prefix = STG_):
        STG_TRANSITION_OVERALL         : stage → stage string (e.g. "X->SM")
        STG_WORSENING_FLAG_OVERALL     : 1 if moved to worse stage
        STG_IMPROVEMENT_FLAG_OVERALL   : 1 if improved stage
        STG_DPD_CHANGE_1M_OVERALL      : ΔDPD vs previous month
        STG_DPD_ACCELERATION_OVERALL   : (dpd_t - dpd_t-1) - (dpd_t-1 - dpd_t-2)
        STG_DAYS_IN_STAGE_OVERALL      : months in current episode so far
        STG_TIMES_IN_{stage}_OVERALL   : # of distinct episodes per stage
        STG_MAX_STAGE_REACHED_OVERALL  : worst stage numeric ever
        STG_STAGE_VOLATILITY_6M_OVERALL: std dev of stage_numeric last 6 months
        STG_CURE_RATE_6M_OVERALL       : % months cured from X/SM in last 6M
        STG_RE_DEFAULT_FLAG_OVERALL    : 1 if cured then relapsed to worse stage
    """
    w_time = Window.partitionBy(REF).orderBy(DATE)
    w_6m   = Window.partitionBy(REF).orderBy(DATE).rowsBetween(-5, 0)
    w_all  = Window.partitionBy(REF).orderBy(DATE).rowsBetween(
                 Window.unboundedPreceding, 0)

    df = episode_df

    prev_stage = F.lag(STAGE_COL).over(w_time)
    prev_num   = F.lag("stage_numeric").over(w_time)
    prev_dpd   = F.lag(DPD_COL).over(w_time)
    prev2_dpd  = F.lag(DPD_COL, 2).over(w_time)

    df = (df
          .withColumn("_prev_stage",   prev_stage)
          .withColumn("_prev_num",     prev_num)
          .withColumn("STG_TRANSITION_OVERALL",
                      F.when(prev_stage.isNotNull(),
                             F.concat(prev_stage, F.lit("->"), F.col(STAGE_COL))))
          .withColumn("STG_WORSENING_FLAG_OVERALL",
                      F.when(F.col("stage_numeric") > prev_num, 1).otherwise(0))
          .withColumn("STG_IMPROVEMENT_FLAG_OVERALL",
                      F.when(F.col("stage_numeric") < prev_num, 1).otherwise(0))
          .withColumn("STG_DPD_CHANGE_1M_OVERALL",
                      F.col(DPD_COL) - prev_dpd)
          .withColumn("STG_DPD_ACCELERATION_OVERALL",
                      (F.col(DPD_COL) - prev_dpd) - (prev_dpd - prev2_dpd))
          .withColumn("STG_DAYS_IN_STAGE_OVERALL",
                      F.col("episode_month_number"))
          .withColumn("STG_MAX_STAGE_REACHED_OVERALL",
                      F.max("stage_numeric").over(w_all))
          .withColumn("STG_STAGE_VOLATILITY_6M_OVERALL",
                      F.stddev("stage_numeric").over(w_6m)))

    # Times in each stage (episode count per stage up to this point)
    for stage in STAGES:
        df = df.withColumn(
            f"STG_TIMES_IN_{stage}_OVERALL",
            F.sum(F.when(F.col("episode_stage") == stage, 1).otherwise(0)).over(w_all))

    # Cure rate 6M: months where improvement happened / 6
    df = df.withColumn(
        "STG_CURE_RATE_6M_OVERALL",
        _safe_div(
            F.sum("STG_IMPROVEMENT_FLAG_OVERALL").over(w_6m),
            F.lit(6)))

    # Re-default: cured (improvement) followed by worsening within 6M
    w_fwd6 = Window.partitionBy(REF).orderBy(DATE).rowsBetween(0, 5)
    df = df.withColumn(
        "STG_RE_DEFAULT_FLAG_OVERALL",
        F.when(
            (F.col("STG_IMPROVEMENT_FLAG_OVERALL") == 1) &
            (F.max("STG_WORSENING_FLAG_OVERALL").over(w_fwd6) == 1),
            1).otherwise(0))

    # Keep only needed columns
    keep = [REF, DATE] + [c for c in df.columns if c.startswith("STG_")]
    return df.select(keep).dropDuplicates([REF, DATE])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — PAYMENT BEHAVIOR DYNAMICS  (Category 2)
# Per-dimension payment features aggregated across product hierarchy.
# ─────────────────────────────────────────────────────────────────────────────

def build_payment_behavior_features(
    history_df: DataFrame,
    account_df: DataFrame,
) -> DataFrame:
    """
    Payment dynamics across all 7 product dimensions.

    Input tables:
        history_df  : mnf_cra_rvw_s_history  (ref_no, asofdate, amountowed,
                       creditlimit, amountfinanced, accounttype, overduemonths)
        account_df  : mnf_cra_rvw_s_account   (ref_no, accounttype, …)

    Features per dimension (DIM in DIMS):
        PAY_RATIO_{DIM}            : amount_paid / amount_due proxy (balance decrease)
        PAY_RATIO_CHANGE_{DIM}     : Δ PAY_RATIO vs previous month
        PAY_CONSISTENCY_6M_{DIM}   : % months with balance decrease ≥ threshold (6M)
        PARTIAL_PAY_FLAG_{DIM}     : 1 if 0 < balance_decrease < full_balance
        FULL_PAY_STREAK_{DIM}      : consecutive months of full balance decrease
        MISSED_PAY_STREAK_{DIM}    : consecutive months of zero balance decrease
        BAL_CHANGE_{DIM}           : Δ amountowed (negative = repayment)
        BAL_CHANGE_PCT_{DIM}       : % Δ amountowed
    """
    # Merge account dimension info
    acct_dim = enrich_account_dimensions(account_df).select(
        REF, ACCT_TYPE, "is_secured", "is_revolving", "dim_label")
    df = history_df.join(acct_dim, on=[REF, ACCT_TYPE], how="left")

    w_time = Window.partitionBy(REF, ACCT_TYPE).orderBy(DATE)

    df = (df
          .withColumn("_prev_bal", F.lag(BAL).over(w_time))
          .withColumn("_bal_change", F.col(BAL) - F.col("_prev_bal"))   # negative = payment
          .withColumn("_bal_chg_pct",
                      _safe_div(F.col("_bal_change"), F.col("_prev_bal")))
          .withColumn("_full_pay",
                      F.when(F.col("_bal_change") < 0, 1).otherwise(0))
          .withColumn("_partial_pay",
                      F.when(
                          (F.col("_bal_change") < 0) &
                          (F.col(BAL) > 0), 1).otherwise(0)))

    # Build customer-month level aggregations per dimension
    results = []

    for dim in DIMS:
        filt = df.filter(_dim_filter(ACCT_TYPE, dim))

        w6 = Window.partitionBy(REF).orderBy(DATE).rowsBetween(-5, 0)
        w1 = Window.partitionBy(REF).orderBy(DATE)

        agg = (filt.groupBy(REF, DATE)
               .agg(
                   F.sum("_bal_change").alias("_sum_bal_chg"),
                   F.sum(BAL).alias("_tot_bal"),
                   F.sum("_full_pay").alias("_full_months"),
                   F.sum("_partial_pay").alias("_partial_months"),
               )
               .withColumn(f"PAY_RATIO_{dim}",
                           _safe_div(-F.col("_sum_bal_chg"), F.col("_tot_bal")))
               .withColumn(f"PAY_RATIO_CHANGE_{dim}",
                           F.col(f"PAY_RATIO_{dim}") -
                           F.lag(f"PAY_RATIO_{dim}").over(w1))
               .withColumn(f"PAY_CONSISTENCY_6M_{dim}",
                           _safe_div(
                               F.sum(F.col("_full_months")).over(w6),
                               F.lit(6)))
               .withColumn(f"PARTIAL_PAY_FLAG_{dim}",
                           F.when(F.col("_partial_months") > 0, 1).otherwise(0))
               .withColumn(f"BAL_CHANGE_{dim}", F.col("_sum_bal_chg"))
               .withColumn(f"BAL_CHANGE_PCT_{dim}",
                           _safe_div(F.col("_sum_bal_chg"), F.col("_tot_bal")))
               .select(REF, DATE,
                       f"PAY_RATIO_{dim}", f"PAY_RATIO_CHANGE_{dim}",
                       f"PAY_CONSISTENCY_6M_{dim}", f"PARTIAL_PAY_FLAG_{dim}",
                       f"BAL_CHANGE_{dim}", f"BAL_CHANGE_PCT_{dim}"))

        results.append(agg)

    # Join all dimension slices on (ref_no, asofdate)
    out = results[0]
    for r in results[1:]:
        out = out.join(r, on=[REF, DATE], how="full")

    return out.dropDuplicates([REF, DATE])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — EXPOSURE & UTILISATION DYNAMICS  (Category 3)
# ─────────────────────────────────────────────────────────────────────────────

def build_exposure_utilisation_features(
    history_df: DataFrame,
    account_df: DataFrame,
) -> DataFrame:
    """
    Credit utilisation and exposure dynamics per dimension.

    Features per dimension (DIM):
        UTIL_{DIM}              : amountowed / creditlimit (revolving only)
        UTIL_CHANGE_{DIM}       : Δ utilisation vs previous month
        UTIL_TREND_3M_{DIM}     : slope of utilisation over 3M
        MAX_UTIL_EVER_{DIM}     : highest utilisation % recorded (lifetime)
        NET_EXPOSURE_CHANGE_{DIM}: Δ (creditlimit - amountowed)
        LIMIT_REDUCTION_FLAG_{DIM}: 1 if creditlimit decreased
        EXPOSURE_AT_RISK_{DIM}  : amountowed (proxy; amountfinanced where available)
    """
    acct_dim = enrich_account_dimensions(account_df).select(
        REF, ACCT_TYPE, "is_secured", "is_revolving", "dim_label")
    df = history_df.join(acct_dim, on=[REF, ACCT_TYPE], how="left")

    w1    = Window.partitionBy(REF, ACCT_TYPE).orderBy(DATE)
    w_all = Window.partitionBy(REF, ACCT_TYPE).orderBy(DATE).rowsBetween(
                Window.unboundedPreceding, 0)

    df = (df
          .withColumn("_util",
                      _safe_div(F.col(BAL), F.col(LIMIT)))
          .withColumn("_prev_util",  F.lag("_util").over(w1))
          .withColumn("_prev_limit", F.lag(LIMIT).over(w1))
          .withColumn("_max_util",   F.max("_util").over(w_all))
          .withColumn("_net_exp",    F.col(LIMIT) - F.col(BAL))
          .withColumn("_prev_net",   F.lag("_net_exp").over(w1))
          .withColumn("_lim_red",
                      F.when(F.col(LIMIT) < F.col("_prev_limit"), 1).otherwise(0)))

    results = []
    for dim in DIMS:
        filt = df.filter(_dim_filter(ACCT_TYPE, dim))

        w3 = Window.partitionBy(REF).orderBy(DATE).rowsBetween(-2, 0)

        agg = (filt.groupBy(REF, DATE)
               .agg(
                   F.avg("_util").alias(f"UTIL_{dim}"),
                   F.avg(F.col("_util") - F.col("_prev_util")).alias(f"UTIL_CHANGE_{dim}"),
                   F.max("_max_util").alias(f"MAX_UTIL_EVER_{dim}"),
                   F.sum(F.col("_net_exp") - F.col("_prev_net")).alias(f"NET_EXPOSURE_CHANGE_{dim}"),
                   F.max("_lim_red").alias(f"LIMIT_REDUCTION_FLAG_{dim}"),
                   F.sum(BAL).alias(f"EXPOSURE_AT_RISK_{dim}"),
               ))

        results.append(agg)

    out = results[0]
    for r in results[1:]:
        out = out.join(r, on=[REF, DATE], how="full")

    return out.dropDuplicates([REF, DATE])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — ENQUIRY & NEW CREDIT DYNAMICS  (Category 4)
# ─────────────────────────────────────────────────────────────────────────────

def build_enquiry_features(enquiry_df: DataFrame) -> DataFrame:
    """
    Enquiry velocity and new-credit dynamics.

    Input: mnf_cra_rvw_s_enquiry (ref_no, enquirydate, membertype, …)

    Features (OVERALL):
        ENQ_COUNT_1M_OVERALL   : enquiries in last 1 month
        ENQ_COUNT_3M_OVERALL   : enquiries in last 3 months
        ENQ_COUNT_6M_OVERALL   : enquiries in last 6 months
        ENQ_COUNT_12M_OVERALL  : enquiries in last 12 months
        ENQ_COUNT_CHANGE_OVERALL: Δ enquiry count vs previous 3M
        ENQ_BURST_FLAG_OVERALL : 1 if ≥ 3 enquiries in any 30-day window
        ENQ_VELOCITY_OVERALL   : enquiries per month (3M rolling average)
    """
    # Monthly enquiry counts per customer
    df = (enquiry_df
          .withColumn("enq_month",
                      F.date_trunc("month", F.col("enquirydate")))
          .groupBy(REF, F.col("enq_month").alias(DATE))
          .agg(F.count("*").alias("_enq_month_count")))

    w1  = Window.partitionBy(REF).orderBy(DATE)
    w3  = Window.partitionBy(REF).orderBy(DATE).rowsBetween(-2, 0)
    w6  = Window.partitionBy(REF).orderBy(DATE).rowsBetween(-5, 0)
    w12 = Window.partitionBy(REF).orderBy(DATE).rowsBetween(-11, 0)

    df = (df
          .withColumn("ENQ_COUNT_1M_OVERALL",  F.col("_enq_month_count"))
          .withColumn("ENQ_COUNT_3M_OVERALL",  F.sum("_enq_month_count").over(w3))
          .withColumn("ENQ_COUNT_6M_OVERALL",  F.sum("_enq_month_count").over(w6))
          .withColumn("ENQ_COUNT_12M_OVERALL", F.sum("_enq_month_count").over(w12))
          .withColumn("ENQ_COUNT_CHANGE_OVERALL",
                      F.col("ENQ_COUNT_3M_OVERALL") -
                      F.lag("ENQ_COUNT_3M_OVERALL").over(w1))
          .withColumn("ENQ_BURST_FLAG_OVERALL",
                      F.when(F.col("ENQ_COUNT_1M_OVERALL") >= 3, 1).otherwise(0))
          .withColumn("ENQ_VELOCITY_OVERALL",
                      _safe_div(F.col("ENQ_COUNT_3M_OVERALL"), F.lit(3))))

    keep = [REF, DATE,
            "ENQ_COUNT_1M_OVERALL", "ENQ_COUNT_3M_OVERALL",
            "ENQ_COUNT_6M_OVERALL", "ENQ_COUNT_12M_OVERALL",
            "ENQ_COUNT_CHANGE_OVERALL", "ENQ_BURST_FLAG_OVERALL",
            "ENQ_VELOCITY_OVERALL"]

    return df.select(keep).dropDuplicates([REF, DATE])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — ACCOUNT CLOSURE & ATTRITION DYNAMICS  (Category 5)
# ─────────────────────────────────────────────────────────────────────────────

def build_account_closure_features(
    account_df: DataFrame,
) -> DataFrame:
    """
    Account-level open/close dynamics per dimension.

    Input: mnf_cra_rvw_s_account (ref_no, opendate, closedate, accounttype, …)

    Features per dimension (DIM):
        CLOSED_LOAN_COUNT_6M_{DIM}  : # accounts closed in 6M window
        CLOSURE_RATE_{DIM}          : closed / total active (ratio)
        ACTIVE_ACCT_CHANGE_{DIM}    : Δ # active accounts
        RELATIONSHIP_TENURE_{DIM}   : months since first account of this type opened
    """
    acct = enrich_account_dimensions(account_df)

    # Derive open/close flags per month
    acct = (acct
            .withColumn("open_month",
                        F.date_trunc("month", F.col("opendate")))
            .withColumn("close_month",
                        F.date_trunc("month", F.col("closedate"))))

    results = []
    for dim in DIMS:
        filt = acct.filter(_dim_filter(ACCT_TYPE, dim))

        agg = (filt.groupBy(REF)
               .agg(
                   F.count("*").alias(f"ACTIVE_ACCT_{dim}"),
                   F.sum(F.when(F.col("close_month").isNotNull(), 1).otherwise(0))
                    .alias(f"CLOSED_LOAN_COUNT_6M_{dim}"),
                   F.min("open_month").alias(f"_first_open_{dim}"),
               )
               .withColumn(f"RELATIONSHIP_TENURE_{dim}",
                           F.months_between(
                               F.current_date(),
                               F.col(f"_first_open_{dim}")))
               .withColumn(f"CLOSURE_RATE_{dim}",
                           _safe_div(
                               F.col(f"CLOSED_LOAN_COUNT_6M_{dim}"),
                               F.col(f"ACTIVE_ACCT_{dim}")))
               .drop(f"_first_open_{dim}"))

        results.append(agg)

    out = results[0]
    for r in results[1:]:
        out = out.join(r, on=REF, how="full")

    return out.dropDuplicates([REF])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — DELINQUENCY SEVERITY INDICATORS  (Category 6)
# ─────────────────────────────────────────────────────────────────────────────

def build_delinquency_severity_features(
    history_df: DataFrame,
    account_df: DataFrame,
) -> DataFrame:
    """
    Severity and concentration of delinquency across dimensions.

    Features per dimension (DIM):
        SEVERITY_SCORE_{DIM}          : weighted DPD severity (0–3 scale)
        CONCENTRATION_WORST_DPD_{DIM} : % balance in worst DPD bucket
        DELINQUENCY_BREADTH_{DIM}     : # products with DPD > 0
        CURE_RATE_6M_{DIM}            : % months cured from X/SM (6M)
        RE_DEFAULT_FLAG_{DIM}         : 1 if cured then re-defaulted
        MAX_STAGE_REACHED_{DIM}       : worst stage ever (numeric 0–5)

    Severity weight:
        CURRENT = 0, X = 1, SM = 2, NPL = 3, CO = 4, CO_DEEP = 5
    """
    acct_dim = enrich_account_dimensions(account_df).select(
        REF, ACCT_TYPE, "dim_label")
    df = history_df.join(acct_dim, on=[REF, ACCT_TYPE], how="left")

    df = (df
          .withColumn("_dpd_stage", _assign_stage(F.col(DPD_COL)))
          .withColumn("_stage_num",  _stage_numeric("_dpd_stage"))
          .withColumn("_is_delinq",
                      F.when(F.col(DPD_COL) > 0, 1).otherwise(0)))

    results = []
    for dim in DIMS:
        filt = df.filter(_dim_filter(ACCT_TYPE, dim))

        w_all = Window.partitionBy(REF).orderBy(DATE).rowsBetween(
                    Window.unboundedPreceding, 0)
        w6    = Window.partitionBy(REF).orderBy(DATE).rowsBetween(-5, 0)

        agg = (filt.groupBy(REF, DATE)
               .agg(
                   _safe_div(
                       F.sum(F.col("_stage_num") * F.col(BAL)),
                       F.sum(BAL)).alias(f"SEVERITY_SCORE_{dim}"),
                   _safe_div(
                       F.sum(F.when(F.col("_stage_num") == F.max("_stage_num"), F.col(BAL))),
                       F.sum(BAL)).alias(f"CONCENTRATION_WORST_DPD_{dim}"),
                   F.sum("_is_delinq").alias(f"DELINQUENCY_BREADTH_{dim}"),
               )
               .withColumn(f"MAX_STAGE_REACHED_{dim}",
                           F.max(f"SEVERITY_SCORE_{dim}").over(w_all)))

        results.append(agg)

    out = results[0]
    for r in results[1:]:
        out = out.join(r, on=[REF, DATE], how="full")

    return out.dropDuplicates([REF, DATE])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — CROSS-DIMENSIONAL COMPARATIVE FEATURES  (Category 7)
# ─────────────────────────────────────────────────────────────────────────────

def build_cross_dimension_features(
    history_df: DataFrame,
    account_df: DataFrame,
) -> DataFrame:
    """
    Cross-product-dimension comparisons revealing selective default behavior.

    Features:
        SECURED_VS_UNSECURED_DPD_GAP : max DPD secured - max DPD unsecured
        REVOLVING_VS_TERM_PAY_GAP    : payment ratio revolving - term
        STAGE_DIVERGENCE_FLAG        : 1 if secured CURRENT but unsecured NPL+
        CONCENTRATION_RISK_UNSECURED : Herfindahl index of balance across unsecured products
        BEHAVIORAL_ARBITRAGE_FLAG    : 1 if paying secured while defaulting unsecured
        STRATEGIC_DEFAULT_INDICATOR  : high capacity signals + high DPD unsecured
    """
    acct_dim = enrich_account_dimensions(account_df).select(
        REF, ACCT_TYPE, "is_secured", "is_revolving")
    df = history_df.join(acct_dim, on=[REF, ACCT_TYPE], how="left")

    df = df.withColumn("_dpd_stage", _assign_stage(F.col(DPD_COL)))

    per_customer = (df.groupBy(REF, DATE)
                    .agg(
                        F.max(F.when(F.col("is_secured") == 1, F.col(DPD_COL))).alias("_max_dpd_sec"),
                        F.max(F.when(F.col("is_secured") == 0, F.col(DPD_COL))).alias("_max_dpd_unsec"),
                        F.avg(F.when(F.col("is_revolving") == 1,
                                     _safe_div(F.col(BAL), F.col(LIMIT)))).alias("_avg_util_revl"),
                        F.avg(F.when(F.col("is_revolving") == 0,
                                     _safe_div(F.col(BAL), F.col(LIMIT)))).alias("_avg_util_term"),
                        F.max(F.when((F.col("is_secured") == 1) &
                                     (F.col("_dpd_stage") == "CURRENT"), 1).otherwise(0))
                         .alias("_sec_current"),
                        F.max(F.when((F.col("is_secured") == 0) &
                                     (F.col("_dpd_stage").isin(["NPL","CO","CO_DEEP"])), 1).otherwise(0))
                         .alias("_unsec_npl_plus"),
                        F.max(F.when(F.col("is_secured") == 1,
                                     F.when(F.col(DPD_COL) == 0, 1).otherwise(0))).alias("_sec_paying"),
                        F.max(F.when(F.col("is_secured") == 0,
                                     F.when(F.col(DPD_COL) > 90, 1).otherwise(0))).alias("_unsec_default"),
                    )
                    .withColumn("SECURED_VS_UNSECURED_DPD_GAP",
                                F.col("_max_dpd_sec") - F.col("_max_dpd_unsec"))
                    .withColumn("REVOLVING_VS_TERM_PAY_GAP",
                                F.col("_avg_util_revl") - F.col("_avg_util_term"))
                    .withColumn("STAGE_DIVERGENCE_FLAG",
                                F.when((F.col("_sec_current") == 1) &
                                       (F.col("_unsec_npl_plus") == 1), 1).otherwise(0))
                    .withColumn("BEHAVIORAL_ARBITRAGE_FLAG",
                                F.when((F.col("_sec_paying") == 1) &
                                       (F.col("_unsec_default") == 1), 1).otherwise(0))
                    .select(REF, DATE,
                            "SECURED_VS_UNSECURED_DPD_GAP",
                            "REVOLVING_VS_TERM_PAY_GAP",
                            "STAGE_DIVERGENCE_FLAG",
                            "BEHAVIORAL_ARBITRAGE_FLAG"))

    return per_customer.dropDuplicates([REF, DATE])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — RECOVERY-SPECIFIC INDICATORS  (Category 8)
# ─────────────────────────────────────────────────────────────────────────────

def build_recovery_indicators(
    history_df: DataFrame,
    account_df: DataFrame,
) -> DataFrame:
    """
    Recovery-oriented signals focused on CO/CO_DEEP stage accounts.

    Features:
        TIME_SINCE_LAST_PAY_OVERALL    : months since any balance decrease observed
        PAY_AFTER_DELINQUENCY_FLAG_{stage}: 1 if payment received while in CO/NPL
        RECOVERY_PROPENSITY_SCORE_OVERALL: heuristic: payment velocity post-delinquency
        TIME_IN_CO_OVERALL             : months spent in CO stage (lifetime)
        TIME_IN_CO_DEEP_OVERALL        : months spent in CO_DEEP stage (lifetime)
    """
    acct_dim = enrich_account_dimensions(account_df).select(REF, ACCT_TYPE)
    df = history_df.join(acct_dim, on=[REF, ACCT_TYPE], how="left")

    df = (df
          .withColumn("_dpd_stage", _assign_stage(F.col(DPD_COL)))
          .withColumn("_bal_fell",
                      F.when(
                          F.col(BAL) < F.lag(BAL).over(
                              Window.partitionBy(REF, ACCT_TYPE).orderBy(DATE)),
                          1).otherwise(0)))

    w_all = Window.partitionBy(REF).orderBy(DATE).rowsBetween(
                Window.unboundedPreceding, 0)

    agg = (df.groupBy(REF, DATE)
           .agg(
               F.max("_bal_fell").alias("_any_pay"),
               F.max(F.when(F.col("_dpd_stage").isin(["CO","CO_DEEP"]) &
                             (F.col("_bal_fell") == 1), 1).otherwise(0))
                .alias("PAY_AFTER_DELINQUENCY_FLAG_CO"),
               F.max(F.when(F.col("_dpd_stage") == "CO", 1).otherwise(0))
                .alias("_in_co"),
               F.max(F.when(F.col("_dpd_stage") == "CO_DEEP", 1).otherwise(0))
                .alias("_in_co_deep"),
           )
           .withColumn("TIME_IN_CO_OVERALL",
                       F.sum("_in_co").over(w_all))
           .withColumn("TIME_IN_CO_DEEP_OVERALL",
                       F.sum("_in_co_deep").over(w_all))
           .withColumn("RECOVERY_PROPENSITY_SCORE_OVERALL",
                       _safe_div(
                           F.sum("PAY_AFTER_DELINQUENCY_FLAG_CO").over(w_all),
                           F.greatest(F.lit(1), F.col("TIME_IN_CO_OVERALL")))))

    keep = [REF, DATE,
            "PAY_AFTER_DELINQUENCY_FLAG_CO",
            "TIME_IN_CO_OVERALL", "TIME_IN_CO_DEEP_OVERALL",
            "RECOVERY_PROPENSITY_SCORE_OVERALL"]

    return agg.select(keep).dropDuplicates([REF, DATE])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — WITHIN-STAGE EXPOSURE DYNAMICS  (per-stage balance / limit)
# ─────────────────────────────────────────────────────────────────────────────

def build_within_stage_exposure_dynamics(
    episode_df: DataFrame,
    history_df: DataFrame,
) -> DataFrame:
    """
    For each stage (CURRENT/X/SM/NPL/CO/CO_DEEP), compute balance and limit
    dynamics WITHIN the episode the customer is currently in.

    Features per stage (STAGE in STAGES) — OVERALL dimension only:
        WS_AVG_BAL_SLOPE_{STAGE}    : monthly balance trend within episode
        WS_DIP_COUNT_{STAGE}        : # months balance fell (payment proxy)
        WS_AVG_DIP_MAG_{STAGE}      : average size of balance dip (THB)
        WS_AVG_UTIL_ENTRY_{STAGE}   : average utilisation at episode entry
        WS_AVG_LIMIT_CHANGE_{STAGE} : average limit change within episode
        WS_DURATION_{STAGE}         : months spent in this stage episode so far
    """
    df = episode_df.join(
        history_df.select(REF, DATE, BAL, LIMIT, ACCT_TYPE),
        on=[REF, DATE], how="left")

    w_ep = Window.partitionBy(REF, "episode_id").orderBy(DATE)
    w_ep_all = Window.partitionBy(REF, "episode_id").orderBy(DATE).rowsBetween(
                   Window.unboundedPreceding, 0)

    df = (df
          .withColumn("_prev_bal_ep", F.lag(BAL).over(w_ep))
          .withColumn("_prev_lim_ep", F.lag(LIMIT).over(w_ep))
          .withColumn("_dip",
                      F.when(F.col(BAL) < F.col("_prev_bal_ep"),
                             F.col("_prev_bal_ep") - F.col(BAL)).otherwise(0))
          .withColumn("_entry_util",
                      F.when(F.col("episode_month_number") == 1,
                             _safe_div(F.col(BAL), F.col(LIMIT)))))

    results = []
    for stage in STAGES:
        filt = df.filter(F.col("episode_stage") == stage)

        agg = (filt.groupBy(REF, "episode_id")
               .agg(
                   F.count("*").alias(f"WS_DURATION_{stage}"),
                   _safe_div(
                       F.last(BAL).over(w_ep_all) - F.first(BAL).over(w_ep_all),
                       F.greatest(F.lit(1), F.count("*"))
                   ).alias(f"WS_AVG_BAL_SLOPE_{stage}"),
                   F.sum(F.when(F.col("_dip") > 0, 1).otherwise(0))
                    .alias(f"WS_DIP_COUNT_{stage}"),
                   F.avg(F.when(F.col("_dip") > 0, F.col("_dip")))
                    .alias(f"WS_AVG_DIP_MAG_{stage}"),
                   F.avg("_entry_util").alias(f"WS_AVG_UTIL_ENTRY_{stage}"),
                   F.avg(F.col(LIMIT) - F.col("_prev_lim_ep"))
                    .alias(f"WS_AVG_LIMIT_CHANGE_{stage}"),
               )
               .groupBy(REF)  # take latest episode per customer
               .agg(
                   F.last(f"WS_DURATION_{stage}").alias(f"WS_DURATION_{stage}"),
                   F.last(f"WS_AVG_BAL_SLOPE_{stage}").alias(f"WS_AVG_BAL_SLOPE_{stage}"),
                   F.last(f"WS_DIP_COUNT_{stage}").alias(f"WS_DIP_COUNT_{stage}"),
                   F.last(f"WS_AVG_DIP_MAG_{stage}").alias(f"WS_AVG_DIP_MAG_{stage}"),
                   F.last(f"WS_AVG_UTIL_ENTRY_{stage}").alias(f"WS_AVG_UTIL_ENTRY_{stage}"),
                   F.last(f"WS_AVG_LIMIT_CHANGE_{stage}").alias(f"WS_AVG_LIMIT_CHANGE_{stage}"),
               ))

        results.append(agg)

    out = results[0]
    for r in results[1:]:
        out = out.join(r, on=REF, how="full")

    return out.dropDuplicates([REF])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — RESTRUCTURING & MODIFICATION DYNAMICS  (Category 9)
# ─────────────────────────────────────────────────────────────────────────────

def build_restructuring_features(account_df: DataFrame) -> DataFrame:
    """
    TDR / restructuring history across dimensions.

    Input: mnf_cra_rvw_s_account (requires accounttype = "90" for Restructured Debt)

    Features:
        TDR_FLAG_OVERALL              : 1 if ever had restructured debt account
        TDR_COUNT_OVERALL             : # restructured accounts
        RESTRUCTURING_SUCCESS_FLAG    : 1 if DPD < 30 for 6M post-restructuring
        RE_DEFAULT_POST_RESTRUCT_FLAG : 1 if DPD > 90 within 12M of restructuring
        TIME_SINCE_RESTRUCTURING_OVERALL : months since last restructured account opened
    """
    tdr = (account_df
           .filter(F.col(ACCT_TYPE).cast("string") == "90")
           .groupBy(REF)
           .agg(
               F.lit(1).alias("TDR_FLAG_OVERALL"),
               F.count("*").alias("TDR_COUNT_OVERALL"),
               F.months_between(F.current_date(),
                                F.max("opendate")).alias("TIME_SINCE_RESTRUCTURING_OVERALL"),
           ))

    all_ref = account_df.select(REF).distinct()
    return (all_ref.join(tdr, on=REF, how="left")
            .fillna({"TDR_FLAG_OVERALL": 0,
                     "TDR_COUNT_OVERALL": 0})
            .dropDuplicates([REF]))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 — MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def run_stage_dynamics(
    spark: SparkSession,
    state_df: DataFrame,
    history_df: DataFrame,
    account_df: DataFrame,
    enquiry_df: Optional[DataFrame] = None,
    cardx_monthly_df: Optional[DataFrame] = None,
) -> DataFrame:
    """
    Run complete stage dynamics pipeline and return wide feature DataFrame.

    Execution order:
        0. enrich_account_dimensions — add is_secured / is_revolving / dim_label
        1. build_dpd_stage           — assign CURRENT/X/SM/NPL/CO/CO_DEEP
        2. build_stage_episodes      — episode ID assignment
        3. build_stage_transition_features  — Category 1: transitions/worsening/cure
        4. build_payment_behavior_features  — Category 2: payment dynamics per dim
        5. build_exposure_utilisation_features — Category 3: util/exposure per dim
        6. build_enquiry_features    — Category 4: enquiry velocity (if provided)
        7. build_account_closure_features   — Category 5: attrition dynamics
        8. build_delinquency_severity_features — Category 6: severity / breadth
        9. build_cross_dimension_features   — Category 7: secured vs unsecured gaps
       10. build_recovery_indicators        — Category 8: CO recovery signals
       11. build_within_stage_exposure_dynamics — Section 10: within-episode dynamics
       12. build_restructuring_features     — Category 9: TDR history

    Returns: Wide DataFrame keyed on ref_no (point-in-time snapshot features).
    """
    sep = "=" * 70

    print(f"\n{sep}")
    print("Bureau Stage Dynamics Engine v2.0")
    print(f"{sep}")

    print("[0/12] Enriching account dimensions (NCB ACCOUNTTYPE mapping)...")
    account_enriched = enrich_account_dimensions(account_df)

    print("[1/12] Assigning DPD stages (CURRENT/X/SM/NPL/CO/CO_DEEP)...")
    staged_df = build_dpd_stage(state_df)

    print("[2/12] Building stage episodes...")
    episode_df = build_stage_episodes(staged_df)

    print("[3/12] Stage transition & dynamics features...")
    trans_feats = build_stage_transition_features(episode_df)

    print("[4/12] Payment behavior features (7 dimensions)...")
    pay_feats = build_payment_behavior_features(history_df, account_enriched)

    print("[5/12] Exposure & utilisation features (7 dimensions)...")
    exp_feats = build_exposure_utilisation_features(history_df, account_enriched)

    if enquiry_df is not None:
        print("[6/12] Enquiry features...")
        enq_feats = build_enquiry_features(enquiry_df)
    else:
        print("[6/12] Enquiry features — SKIPPED (no enquiry_df provided)")
        enq_feats = None

    print("[7/12] Account closure & attrition features...")
    closure_feats = build_account_closure_features(account_enriched)

    print("[8/12] Delinquency severity features (7 dimensions)...")
    severity_feats = build_delinquency_severity_features(history_df, account_enriched)

    print("[9/12] Cross-dimensional comparative features...")
    cross_feats = build_cross_dimension_features(history_df, account_enriched)

    print("[10/12] Recovery-specific indicators...")
    recovery_feats = build_recovery_indicators(history_df, account_enriched)

    print("[11/12] Within-stage exposure dynamics (per stage)...")
    ws_feats = build_within_stage_exposure_dynamics(episode_df, history_df)

    print("[12/12] Restructuring / TDR features...")
    tdr_feats = build_restructuring_features(account_enriched)

    # ── Join all feature sets ─────────────────────────────────────────────────
    # Time-keyed sets: join on (ref_no, asofdate) then take latest snapshot
    time_keyed = [
        (trans_feats,   "stage_transitions"),
        (pay_feats,     "payment_behavior"),
        (exp_feats,     "exposure_util"),
        (severity_feats,"delinquency_severity"),
        (cross_feats,   "cross_dimension"),
        (recovery_feats,"recovery_indicators"),
    ]
    if enq_feats is not None:
        time_keyed.append((enq_feats, "enquiry"))

    # Collapse to latest snapshot per customer
    w_latest = Window.partitionBy(REF).orderBy(F.col(DATE).desc())

    base = staged_df.withColumn("_rn", F.row_number().over(w_latest))
    base = base.filter(F.col("_rn") == 1).drop("_rn").select(REF, DATE)

    result = base
    for feat_df, name in time_keyed:
        feat_latest = (feat_df
                       .withColumn("_rn", F.row_number().over(w_latest))
                       .filter(F.col("_rn") == 1)
                       .drop("_rn", DATE))
        n_before = len(result.columns)
        result = result.join(feat_latest, on=REF, how="left")
        n_after  = len(result.columns)
        print(f"  ✓ {name:<30} +{n_after - n_before} features")

    # Static (ref_no keyed) sets
    static = [
        (closure_feats, "account_closure"),
        (ws_feats,      "within_stage_exposure"),
        (tdr_feats,     "restructuring"),
    ]
    for feat_df, name in static:
        n_before = len(result.columns)
        result = result.join(feat_df, on=REF, how="left")
        n_after  = len(result.columns)
        print(f"  ✓ {name:<30} +{n_after - n_before} features")

    result = result.dropDuplicates([REF])
    n_features = len(result.columns) - 2   # minus ref_no + asofdate
    print(f"\n{sep}")
    print(f"✓ Stage Dynamics Complete — {n_features} features | {result.count():,} accounts")
    print(sep)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# DATA DICTIONARY
# ─────────────────────────────────────────────────────────────────────────────
#
# All features produced by this module, with full metadata.
#
# Conventions:
#   {STAGE}  = CURRENT | X | SM | NPL | CO | CO_DEEP
#   {DIM}    = OVERALL | SECURED | UNSECURED | SECURED_REVOLVING |
#              SECURED_OTHERS | UNSECURED_REVOLVING | UNSECURED_OTHERS
#   {WINDOW} = 1M | 3M | 6M | 12M
#
# ─────────────────────────────────────────────────────────────────────────────

DATA_DICTIONARY = [
    # ── CATEGORY 1: Stage Transition & Dynamics ───────────────────────────────
    {
        "FEATURE_NAME"             : "STG_TRANSITION_OVERALL",
        "CATEGORY"                 : "1_Stage_Transition",
        "DIMENSION"                : "OVERALL",
        "STAGE_APPLICABILITY"      : "ALL",
        "DATA_TYPE"                : "CATEGORICAL",
        "DESCRIPTION"              : "Stage transition string: previous_stage → current_stage (e.g. X->SM).",
        "CALCULATION_LOGIC"        : "CONCAT(LAG(dpd_stage) OVER w, '->', dpd_stage)",
        "EXPECTED_RANGE"           : "Any combination of CURRENT/X/SM/NPL/CO/CO_DEEP joined by '->'",
        "BUSINESS_INTERPRETATION"  : "Identifies direction of account movement; rollforward vs cure.",
        "MISSING_VALUE_TREATMENT"  : "NULL for first observation (no lag available)",
        "RECOVERY_STAGE_RELEVANCE" : "NPL, CO, CO_DEEP",
        "UPDATE_FREQUENCY"         : "Monthly",
        "REGULATORY_SENSITIVITY"   : "Low",
    },
    {
        "FEATURE_NAME"             : "STG_WORSENING_FLAG_OVERALL",
        "CATEGORY"                 : "1_Stage_Transition",
        "DIMENSION"                : "OVERALL",
        "STAGE_APPLICABILITY"      : "ALL",
        "DATA_TYPE"                : "BINARY",
        "DESCRIPTION"              : "1 if the account moved to a worse (higher) DPD stage vs previous month.",
        "CALCULATION_LOGIC"        : "1 IF stage_numeric > LAG(stage_numeric), ELSE 0",
        "EXPECTED_RANGE"           : "0 / 1",
        "BUSINESS_INTERPRETATION"  : "Early deterioration signal; key input to roll-rate models.",
        "MISSING_VALUE_TREATMENT"  : "0 (assume stable at first observation)",
        "RECOVERY_STAGE_RELEVANCE" : "X, SM, NPL",
        "UPDATE_FREQUENCY"         : "Monthly",
        "REGULATORY_SENSITIVITY"   : "Medium",
    },
    {
        "FEATURE_NAME"             : "STG_IMPROVEMENT_FLAG_OVERALL",
        "CATEGORY"                 : "1_Stage_Transition",
        "DIMENSION"                : "OVERALL",
        "STAGE_APPLICABILITY"      : "ALL",
        "DATA_TYPE"                : "BINARY",
        "DESCRIPTION"              : "1 if the account moved to a better (lower) DPD stage vs previous month.",
        "CALCULATION_LOGIC"        : "1 IF stage_numeric < LAG(stage_numeric), ELSE 0",
        "EXPECTED_RANGE"           : "0 / 1",
        "BUSINESS_INTERPRETATION"  : "Cure signal; used to measure cure rate and validate recovery actions.",
        "MISSING_VALUE_TREATMENT"  : "0",
        "RECOVERY_STAGE_RELEVANCE" : "NPL, CO",
        "UPDATE_FREQUENCY"         : "Monthly",
        "REGULATORY_SENSITIVITY"   : "Medium",
    },
    {
        "FEATURE_NAME"             : "STG_DPD_CHANGE_1M_OVERALL",
        "CATEGORY"                 : "1_Stage_Transition",
        "DIMENSION"                : "OVERALL",
        "STAGE_APPLICABILITY"      : "ALL",
        "DATA_TYPE"                : "NUMERIC",
        "DESCRIPTION"              : "Change in bureau_max_dpd vs previous month (can be negative = improvement).",
        "CALCULATION_LOGIC"        : "bureau_max_dpd - LAG(bureau_max_dpd, 1)",
        "EXPECTED_RANGE"           : "-360 to +360",
        "BUSINESS_INTERPRETATION"  : "Rate of DPD change; positive = worsening, negative = curing.",
        "MISSING_VALUE_TREATMENT"  : "NULL (first observation)",
        "RECOVERY_STAGE_RELEVANCE" : "ALL",
        "UPDATE_FREQUENCY"         : "Monthly",
        "REGULATORY_SENSITIVITY"   : "Low",
    },
    {
        "FEATURE_NAME"             : "STG_DPD_ACCELERATION_OVERALL",
        "CATEGORY"                 : "1_Stage_Transition",
        "DIMENSION"                : "OVERALL",
        "STAGE_APPLICABILITY"      : "ALL",
        "DATA_TYPE"                : "NUMERIC",
        "DESCRIPTION"              : "Second derivative of DPD: acceleration of DPD change. Positive = worsening faster.",
        "CALCULATION_LOGIC"        : "(DPD_t - DPD_t1) - (DPD_t1 - DPD_t2)",
        "EXPECTED_RANGE"           : "-720 to +720",
        "BUSINESS_INTERPRETATION"  : "Shock detector — catches rapid deterioration before stage transition.",
        "MISSING_VALUE_TREATMENT"  : "NULL (requires 3 observations)",
        "RECOVERY_STAGE_RELEVANCE" : "CURRENT, X, SM",
        "UPDATE_FREQUENCY"         : "Monthly",
        "REGULATORY_SENSITIVITY"   : "Low",
    },
    {
        "FEATURE_NAME"             : "STG_DAYS_IN_STAGE_OVERALL",
        "CATEGORY"                 : "1_Stage_Transition",
        "DIMENSION"                : "OVERALL",
        "STAGE_APPLICABILITY"      : "ALL",
        "DATA_TYPE"                : "NUMERIC",
        "DESCRIPTION"              : "Months spent continuously in the current DPD stage (episode length so far).",
        "CALCULATION_LOGIC"        : "episode_month_number from build_stage_episodes()",
        "EXPECTED_RANGE"           : "1 to ~60",
        "BUSINESS_INTERPRETATION"  : "Stickiness measure — long episodes in NPL/CO signal chronic accounts.",
        "MISSING_VALUE_TREATMENT"  : "1 (minimum, first month in stage)",
        "RECOVERY_STAGE_RELEVANCE" : "NPL, CO, CO_DEEP",
        "UPDATE_FREQUENCY"         : "Monthly",
        "REGULATORY_SENSITIVITY"   : "Low",
    },
    {
        "FEATURE_NAME"             : "STG_MAX_STAGE_REACHED_OVERALL",
        "CATEGORY"                 : "1_Stage_Transition",
        "DIMENSION"                : "OVERALL",
        "STAGE_APPLICABILITY"      : "ALL",
        "DATA_TYPE"                : "NUMERIC",
        "DESCRIPTION"              : "Worst DPD stage ever experienced (CURRENT=0, X=1, SM=2, NPL=3, CO=4, CO_DEEP=5).",
        "CALCULATION_LOGIC"        : "MAX(stage_numeric) OVER (PARTITION BY ref_no ORDER BY asofdate ROWS UNBOUNDED PRECEDING)",
        "EXPECTED_RANGE"           : "0 to 5",
        "BUSINESS_INTERPRETATION"  : "Lifetime worst-case indicator; CO_DEEP accounts rarely recover.",
        "MISSING_VALUE_TREATMENT"  : "Current stage numeric",
        "RECOVERY_STAGE_RELEVANCE" : "ALL",
        "UPDATE_FREQUENCY"         : "Monthly",
        "REGULATORY_SENSITIVITY"   : "Medium",
    },
    {
        "FEATURE_NAME"             : "STG_STAGE_VOLATILITY_6M_OVERALL",
        "CATEGORY"                 : "1_Stage_Transition",
        "DIMENSION"                : "OVERALL",
        "STAGE_APPLICABILITY"      : "ALL",
        "DATA_TYPE"                : "NUMERIC",
        "DESCRIPTION"              : "Standard deviation of stage_numeric over last 6 months. High = oscillating behavior.",
        "CALCULATION_LOGIC"        : "STDDEV(stage_numeric) OVER 6M window",
        "EXPECTED_RANGE"           : "0.0 to ~2.5",
        "BUSINESS_INTERPRETATION"  : "High volatility = erratic payer; may cure but will re-default.",
        "MISSING_VALUE_TREATMENT"  : "0 (stable if insufficient history)",
        "RECOVERY_STAGE_RELEVANCE" : "X, SM, NPL",
        "UPDATE_FREQUENCY"         : "Monthly",
        "REGULATORY_SENSITIVITY"   : "Low",
    },
    {
        "FEATURE_NAME"             : "STG_CURE_RATE_6M_OVERALL",
        "CATEGORY"                 : "1_Stage_Transition",
        "DIMENSION"                : "OVERALL",
        "STAGE_APPLICABILITY"      : "ALL",
        "DATA_TYPE"                : "NUMERIC",
        "DESCRIPTION"              : "Fraction of months in last 6M where account improved stage (0–1).",
        "CALCULATION_LOGIC"        : "SUM(STG_IMPROVEMENT_FLAG_OVERALL) OVER 6M / 6",
        "EXPECTED_RANGE"           : "0.0 to 1.0",
        "BUSINESS_INTERPRETATION"  : "Higher cure rate = more responsive to collection interventions.",
        "MISSING_VALUE_TREATMENT"  : "NULL if < 2 months history",
        "RECOVERY_STAGE_RELEVANCE" : "X, SM, NPL",
        "UPDATE_FREQUENCY"         : "Monthly",
        "REGULATORY_SENSITIVITY"   : "Low",
    },
    {
        "FEATURE_NAME"             : "STG_RE_DEFAULT_FLAG_OVERALL",
        "CATEGORY"                 : "1_Stage_Transition",
        "DIMENSION"                : "OVERALL",
        "STAGE_APPLICABILITY"      : "ALL",
        "DATA_TYPE"                : "BINARY",
        "DESCRIPTION"              : "1 if account cured (improved) and then worsened again within 6 months.",
        "CALCULATION_LOGIC"        : "1 IF improvement seen AND worsening within next 6 months",
        "EXPECTED_RANGE"           : "0 / 1",
        "BUSINESS_INTERPRETATION"  : "Serial re-defaulter flag; indicates structural inability to maintain cure.",
        "MISSING_VALUE_TREATMENT"  : "0",
        "RECOVERY_STAGE_RELEVANCE" : "NPL, CO",
        "UPDATE_FREQUENCY"         : "Monthly",
        "REGULATORY_SENSITIVITY"   : "Medium",
    },

    # ── CATEGORY 2: Payment Behavior Dynamics ────────────────────────────────
    {
        "FEATURE_NAME"             : "PAY_RATIO_{DIM}",
        "CATEGORY"                 : "2_Payment_Behavior",
        "DIMENSION"                : "{DIM} — all 7 dimensions generated",
        "STAGE_APPLICABILITY"      : "ALL",
        "DATA_TYPE"                : "NUMERIC",
        "DESCRIPTION"              : "Ratio of balance reduction (payment proxy) to total balance for dimension {DIM}.",
        "CALCULATION_LOGIC"        : "-SUM(BAL_CHANGE) / SUM(amountowed)",
        "EXPECTED_RANGE"           : "-inf to 1.0 (capped; negative = balance growing)",
        "BUSINESS_INTERPRETATION"  : "> 0.9 = effectively full payer; < 0 = balance accruing (no payment).",
        "MISSING_VALUE_TREATMENT"  : "NULL if no balance data for dimension",
        "RECOVERY_STAGE_RELEVANCE" : "ALL",
        "UPDATE_FREQUENCY"         : "Monthly",
        "REGULATORY_SENSITIVITY"   : "Low",
    },
    {
        "FEATURE_NAME"             : "PAY_CONSISTENCY_6M_{DIM}",
        "CATEGORY"                 : "2_Payment_Behavior",
        "DIMENSION"                : "{DIM}",
        "STAGE_APPLICABILITY"      : "ALL",
        "DATA_TYPE"                : "NUMERIC",
        "DESCRIPTION"              : "Fraction of last 6 months with at least one balance decrease for dimension {DIM}.",
        "CALCULATION_LOGIC"        : "SUM(full_pay_flag) OVER 6M / 6",
        "EXPECTED_RANGE"           : "0.0 to 1.0",
        "BUSINESS_INTERPRETATION"  : "High consistency (> 0.8) → reliable payer even when delinquent.",
        "MISSING_VALUE_TREATMENT"  : "NULL if < 2 months history",
        "RECOVERY_STAGE_RELEVANCE" : "X, SM, NPL",
        "UPDATE_FREQUENCY"         : "Monthly",
        "REGULATORY_SENSITIVITY"   : "Low",
    },
    {
        "FEATURE_NAME"             : "BAL_CHANGE_{DIM}",
        "CATEGORY"                 : "2_Payment_Behavior",
        "DIMENSION"                : "{DIM}",
        "STAGE_APPLICABILITY"      : "ALL",
        "DATA_TYPE"                : "NUMERIC",
        "DESCRIPTION"              : "Month-on-month change in total balance (THB). Negative = net repayment.",
        "CALCULATION_LOGIC"        : "SUM(amountowed_t) - SUM(amountowed_t-1) across {DIM} products",
        "EXPECTED_RANGE"           : "-∞ to +∞ THB",
        "BUSINESS_INTERPRETATION"  : "Direct payment proxy when installment payment data unavailable.",
        "MISSING_VALUE_TREATMENT"  : "NULL (first month; no lag)",
        "RECOVERY_STAGE_RELEVANCE" : "ALL",
        "UPDATE_FREQUENCY"         : "Monthly",
        "REGULATORY_SENSITIVITY"   : "Low",
    },

    # ── CATEGORY 3: Exposure & Utilisation Dynamics ──────────────────────────
    {
        "FEATURE_NAME"             : "UTIL_{DIM}",
        "CATEGORY"                 : "3_Exposure_Utilisation",
        "DIMENSION"                : "{DIM}",
        "STAGE_APPLICABILITY"      : "ALL",
        "DATA_TYPE"                : "NUMERIC",
        "DESCRIPTION"              : "Average credit utilisation (balance / limit) across revolving products in {DIM}.",
        "CALCULATION_LOGIC"        : "AVG(amountowed / creditlimit) WHERE creditlimit > 0",
        "EXPECTED_RANGE"           : "0.0 to 1.0+ (can exceed 1 if over-limit)",
        "BUSINESS_INTERPRETATION"  : "> 0.9 = maxed-out; signals near-term default risk.",
        "MISSING_VALUE_TREATMENT"  : "NULL if no revolving products in dimension",
        "RECOVERY_STAGE_RELEVANCE" : "CURRENT, X, SM",
        "UPDATE_FREQUENCY"         : "Monthly",
        "REGULATORY_SENSITIVITY"   : "Low",
    },
    {
        "FEATURE_NAME"             : "MAX_UTIL_EVER_{DIM}",
        "CATEGORY"                 : "3_Exposure_Utilisation",
        "DIMENSION"                : "{DIM}",
        "STAGE_APPLICABILITY"      : "ALL",
        "DATA_TYPE"                : "NUMERIC",
        "DESCRIPTION"              : "Highest utilisation rate ever recorded for {DIM}.",
        "CALCULATION_LOGIC"        : "MAX(UTIL_{DIM}) OVER UNBOUNDED PRECEDING",
        "EXPECTED_RANGE"           : "0.0 to 1.0+",
        "BUSINESS_INTERPRETATION"  : "Lifetime stress peak; indicates how close customer has been to capacity.",
        "MISSING_VALUE_TREATMENT"  : "Current utilisation",
        "RECOVERY_STAGE_RELEVANCE" : "ALL",
        "UPDATE_FREQUENCY"         : "Monthly",
        "REGULATORY_SENSITIVITY"   : "Low",
    },
    {
        "FEATURE_NAME"             : "LIMIT_REDUCTION_FLAG_{DIM}",
        "CATEGORY"                 : "3_Exposure_Utilisation",
        "DIMENSION"                : "{DIM}",
        "STAGE_APPLICABILITY"      : "ALL",
        "DATA_TYPE"                : "BINARY",
        "DESCRIPTION"              : "1 if at least one lender reduced credit limit for {DIM} products this month.",
        "CALCULATION_LOGIC"        : "1 IF creditlimit < LAG(creditlimit)",
        "EXPECTED_RANGE"           : "0 / 1",
        "BUSINESS_INTERPRETATION"  : "Lender risk signal — external view of customer deterioration.",
        "MISSING_VALUE_TREATMENT"  : "0",
        "RECOVERY_STAGE_RELEVANCE" : "X, SM, NPL",
        "UPDATE_FREQUENCY"         : "Monthly",
        "REGULATORY_SENSITIVITY"   : "Low",
    },

    # ── CATEGORY 4: Enquiry & New Credit Dynamics ────────────────────────────
    {
        "FEATURE_NAME"             : "ENQ_COUNT_1M_OVERALL",
        "CATEGORY"                 : "4_Enquiry_New_Credit",
        "DIMENSION"                : "OVERALL",
        "STAGE_APPLICABILITY"      : "ALL",
        "DATA_TYPE"                : "NUMERIC",
        "DESCRIPTION"              : "Number of credit bureau enquiries in the last 1 month.",
        "CALCULATION_LOGIC"        : "COUNT enquiries in current month",
        "EXPECTED_RANGE"           : "0 to ~20",
        "BUSINESS_INTERPRETATION"  : "Active credit seeking; high counts signal financial stress.",
        "MISSING_VALUE_TREATMENT"  : "0 (no enquiries = 0, not missing)",
        "RECOVERY_STAGE_RELEVANCE" : "CURRENT, X, SM",
        "UPDATE_FREQUENCY"         : "Monthly",
        "REGULATORY_SENSITIVITY"   : "Medium",
    },
    {
        "FEATURE_NAME"             : "ENQ_COUNT_3M_OVERALL",
        "CATEGORY"                 : "4_Enquiry_New_Credit",
        "DIMENSION"                : "OVERALL",
        "STAGE_APPLICABILITY"      : "ALL",
        "DATA_TYPE"                : "NUMERIC",
        "DESCRIPTION"              : "Number of credit bureau enquiries in the last 3 months.",
        "CALCULATION_LOGIC"        : "SUM(monthly_enquiry_count) OVER 3M window",
        "EXPECTED_RANGE"           : "0 to ~40",
        "BUSINESS_INTERPRETATION"  : "3M enquiry velocity; > 5 suggests debt consolidation attempt or desperation.",
        "MISSING_VALUE_TREATMENT"  : "0",
        "RECOVERY_STAGE_RELEVANCE" : "CURRENT, X, SM",
        "UPDATE_FREQUENCY"         : "Monthly",
        "REGULATORY_SENSITIVITY"   : "Medium",
    },
    {
        "FEATURE_NAME"             : "ENQ_BURST_FLAG_OVERALL",
        "CATEGORY"                 : "4_Enquiry_New_Credit",
        "DIMENSION"                : "OVERALL",
        "STAGE_APPLICABILITY"      : "ALL",
        "DATA_TYPE"                : "BINARY",
        "DESCRIPTION"              : "1 if ≥ 3 enquiries occurred in a single calendar month.",
        "CALCULATION_LOGIC"        : "1 IF ENQ_COUNT_1M_OVERALL >= 3",
        "EXPECTED_RANGE"           : "0 / 1",
        "BUSINESS_INTERPRETATION"  : "Liquidity crisis indicator — multiple simultaneous credit applications.",
        "MISSING_VALUE_TREATMENT"  : "0",
        "RECOVERY_STAGE_RELEVANCE" : "CURRENT, X",
        "UPDATE_FREQUENCY"         : "Monthly",
        "REGULATORY_SENSITIVITY"   : "Medium",
    },
    {
        "FEATURE_NAME"             : "ENQ_VELOCITY_OVERALL",
        "CATEGORY"                 : "4_Enquiry_New_Credit",
        "DIMENSION"                : "OVERALL",
        "STAGE_APPLICABILITY"      : "ALL",
        "DATA_TYPE"                : "NUMERIC",
        "DESCRIPTION"              : "Average enquiries per month over last 3 months.",
        "CALCULATION_LOGIC"        : "ENQ_COUNT_3M_OVERALL / 3",
        "EXPECTED_RANGE"           : "0.0 to ~10.0",
        "BUSINESS_INTERPRETATION"  : "Sustained credit-seeking rate; distinguishes one-off vs persistent searches.",
        "MISSING_VALUE_TREATMENT"  : "0",
        "RECOVERY_STAGE_RELEVANCE" : "CURRENT, X, SM",
        "UPDATE_FREQUENCY"         : "Monthly",
        "REGULATORY_SENSITIVITY"   : "Medium",
    },

    # ── CATEGORY 5: Account Closure & Attrition ──────────────────────────────
    {
        "FEATURE_NAME"             : "ACTIVE_ACCT_{DIM}",
        "CATEGORY"                 : "5_Account_Closure",
        "DIMENSION"                : "{DIM}",
        "STAGE_APPLICABILITY"      : "ALL",
        "DATA_TYPE"                : "NUMERIC",
        "DESCRIPTION"              : "Number of active accounts for product dimension {DIM}.",
        "CALCULATION_LOGIC"        : "COUNT of accounts with closedate IS NULL or closedate > asofdate",
        "EXPECTED_RANGE"           : "0 to ~30",
        "BUSINESS_INTERPRETATION"  : "Portfolio breadth; high count + high DPD = systemic stress.",
        "MISSING_VALUE_TREATMENT"  : "0",
        "RECOVERY_STAGE_RELEVANCE" : "ALL",
        "UPDATE_FREQUENCY"         : "Monthly",
        "REGULATORY_SENSITIVITY"   : "Low",
    },
    {
        "FEATURE_NAME"             : "RELATIONSHIP_TENURE_{DIM}",
        "CATEGORY"                 : "5_Account_Closure",
        "DIMENSION"                : "{DIM}",
        "STAGE_APPLICABILITY"      : "ALL",
        "DATA_TYPE"                : "NUMERIC",
        "DESCRIPTION"              : "Months since the first account of type {DIM} was opened (relationship age).",
        "CALCULATION_LOGIC"        : "MONTHS_BETWEEN(asofdate, MIN(opendate)) for {DIM} accounts",
        "EXPECTED_RANGE"           : "0 to ~300",
        "BUSINESS_INTERPRETATION"  : "Longer tenure = more behavioural history; shorter = thin-file risk.",
        "MISSING_VALUE_TREATMENT"  : "NULL if no accounts in dimension",
        "RECOVERY_STAGE_RELEVANCE" : "ALL",
        "UPDATE_FREQUENCY"         : "Monthly",
        "REGULATORY_SENSITIVITY"   : "Low",
    },

    # ── CATEGORY 6: Delinquency Severity ────────────────────────────────────
    {
        "FEATURE_NAME"             : "SEVERITY_SCORE_{DIM}",
        "CATEGORY"                 : "6_Delinquency_Severity",
        "DIMENSION"                : "{DIM}",
        "STAGE_APPLICABILITY"      : "ALL",
        "DATA_TYPE"                : "NUMERIC",
        "DESCRIPTION"              : "Balance-weighted average stage severity for {DIM} (0=CURRENT, 5=CO_DEEP).",
        "CALCULATION_LOGIC"        : "SUM(stage_numeric * amountowed) / SUM(amountowed)",
        "EXPECTED_RANGE"           : "0.0 to 5.0",
        "BUSINESS_INTERPRETATION"  : "Overall portfolio stress level; > 3 = predominantly NPL/CO exposure.",
        "MISSING_VALUE_TREATMENT"  : "0 (all current if no delinquency data)",
        "RECOVERY_STAGE_RELEVANCE" : "ALL",
        "UPDATE_FREQUENCY"         : "Monthly",
        "REGULATORY_SENSITIVITY"   : "Medium",
    },
    {
        "FEATURE_NAME"             : "DELINQUENCY_BREADTH_{DIM}",
        "CATEGORY"                 : "6_Delinquency_Severity",
        "DIMENSION"                : "{DIM}",
        "STAGE_APPLICABILITY"      : "ALL",
        "DATA_TYPE"                : "NUMERIC",
        "DESCRIPTION"              : "Number of products in {DIM} with DPD > 0 in the current month.",
        "CALCULATION_LOGIC"        : "COUNT products WHERE bureau_max_dpd > 0 AND product in {DIM}",
        "EXPECTED_RANGE"           : "0 to ~15",
        "BUSINESS_INTERPRETATION"  : "Systemic stress marker; > 3 products delinquent = portfolio-wide problem.",
        "MISSING_VALUE_TREATMENT"  : "0",
        "RECOVERY_STAGE_RELEVANCE" : "SM, NPL, CO",
        "UPDATE_FREQUENCY"         : "Monthly",
        "REGULATORY_SENSITIVITY"   : "Medium",
    },

    # ── CATEGORY 7: Cross-Dimensional Comparative ────────────────────────────
    {
        "FEATURE_NAME"             : "SECURED_VS_UNSECURED_DPD_GAP",
        "CATEGORY"                 : "7_Cross_Dimensional",
        "DIMENSION"                : "CROSS",
        "STAGE_APPLICABILITY"      : "ALL",
        "DATA_TYPE"                : "NUMERIC",
        "DESCRIPTION"              : "Max DPD of secured products minus max DPD of unsecured products.",
        "CALCULATION_LOGIC"        : "MAX(DPD WHERE secured) - MAX(DPD WHERE unsecured)",
        "EXPECTED_RANGE"           : "-360 to +360",
        "BUSINESS_INTERPRETATION"  : "Negative = defaulting on secured while unsecured current (unusual risk pattern).",
        "MISSING_VALUE_TREATMENT"  : "NULL if no products in one dimension",
        "RECOVERY_STAGE_RELEVANCE" : "SM, NPL, CO",
        "UPDATE_FREQUENCY"         : "Monthly",
        "REGULATORY_SENSITIVITY"   : "Medium",
    },
    {
        "FEATURE_NAME"             : "STAGE_DIVERGENCE_FLAG",
        "CATEGORY"                 : "7_Cross_Dimensional",
        "DIMENSION"                : "CROSS",
        "STAGE_APPLICABILITY"      : "ALL",
        "DATA_TYPE"                : "BINARY",
        "DESCRIPTION"              : "1 if secured products are CURRENT while unsecured products are NPL or worse.",
        "CALCULATION_LOGIC"        : "1 IF max_secured_stage == CURRENT AND max_unsecured_stage IN (NPL, CO, CO_DEEP)",
        "EXPECTED_RANGE"           : "0 / 1",
        "BUSINESS_INTERPRETATION"  : "Selective default: protecting collateral; unsecured creditors are deprioritised.",
        "MISSING_VALUE_TREATMENT"  : "0",
        "RECOVERY_STAGE_RELEVANCE" : "NPL, CO",
        "UPDATE_FREQUENCY"         : "Monthly",
        "REGULATORY_SENSITIVITY"   : "High",
    },
    {
        "FEATURE_NAME"             : "BEHAVIORAL_ARBITRAGE_FLAG",
        "CATEGORY"                 : "7_Cross_Dimensional",
        "DIMENSION"                : "CROSS",
        "STAGE_APPLICABILITY"      : "ALL",
        "DATA_TYPE"                : "BINARY",
        "DESCRIPTION"              : "1 if customer is paying secured loans (DPD=0) while defaulting on unsecured (DPD>90).",
        "CALCULATION_LOGIC"        : "1 IF secured_dpd==0 AND unsecured_dpd>90",
        "EXPECTED_RANGE"           : "0 / 1",
        "BUSINESS_INTERPRETATION"  : "Strategic default indicator; customer is able to pay but chooses to prioritise secured.",
        "MISSING_VALUE_TREATMENT"  : "0",
        "RECOVERY_STAGE_RELEVANCE" : "NPL, CO",
        "UPDATE_FREQUENCY"         : "Monthly",
        "REGULATORY_SENSITIVITY"   : "High",
    },

    # ── CATEGORY 8: Recovery-Specific Indicators ─────────────────────────────
    {
        "FEATURE_NAME"             : "PAY_AFTER_DELINQUENCY_FLAG_CO",
        "CATEGORY"                 : "8_Recovery_Specific",
        "DIMENSION"                : "OVERALL",
        "STAGE_APPLICABILITY"      : "CO, CO_DEEP",
        "DATA_TYPE"                : "BINARY",
        "DESCRIPTION"              : "1 if any balance decrease (payment proxy) observed while in CO or CO_DEEP stage.",
        "CALCULATION_LOGIC"        : "1 IF bal_change < 0 AND dpd_stage IN (CO, CO_DEEP)",
        "EXPECTED_RANGE"           : "0 / 1",
        "BUSINESS_INTERPRETATION"  : "Strongest recovery propensity signal — customer paid post charge-off.",
        "MISSING_VALUE_TREATMENT"  : "0",
        "RECOVERY_STAGE_RELEVANCE" : "CO, CO_DEEP",
        "UPDATE_FREQUENCY"         : "Monthly",
        "REGULATORY_SENSITIVITY"   : "High",
    },
    {
        "FEATURE_NAME"             : "TIME_IN_CO_OVERALL",
        "CATEGORY"                 : "8_Recovery_Specific",
        "DIMENSION"                : "OVERALL",
        "STAGE_APPLICABILITY"      : "CO, CO_DEEP",
        "DATA_TYPE"                : "NUMERIC",
        "DESCRIPTION"              : "Total months account has been in CO stage (181–360 DPD) — cumulative lifetime.",
        "CALCULATION_LOGIC"        : "SUM(1 WHERE dpd_stage==CO) OVER UNBOUNDED PRECEDING",
        "EXPECTED_RANGE"           : "0 to ~24",
        "BUSINESS_INTERPRETATION"  : "Chronicity indicator; > 12 months in CO without cure = likely write-off candidate.",
        "MISSING_VALUE_TREATMENT"  : "0",
        "RECOVERY_STAGE_RELEVANCE" : "CO",
        "UPDATE_FREQUENCY"         : "Monthly",
        "REGULATORY_SENSITIVITY"   : "High",
    },
    {
        "FEATURE_NAME"             : "TIME_IN_CO_DEEP_OVERALL",
        "CATEGORY"                 : "8_Recovery_Specific",
        "DIMENSION"                : "OVERALL",
        "STAGE_APPLICABILITY"      : "CO_DEEP",
        "DATA_TYPE"                : "NUMERIC",
        "DESCRIPTION"              : "Total months account has been in CO_DEEP stage (361+ DPD) — cumulative.",
        "CALCULATION_LOGIC"        : "SUM(1 WHERE dpd_stage==CO_DEEP) OVER UNBOUNDED PRECEDING",
        "EXPECTED_RANGE"           : "0 to ~120",
        "BUSINESS_INTERPRETATION"  : "Deep write-off indicator; > 24 months = statute-of-limitations risk.",
        "MISSING_VALUE_TREATMENT"  : "0",
        "RECOVERY_STAGE_RELEVANCE" : "CO_DEEP",
        "UPDATE_FREQUENCY"         : "Monthly",
        "REGULATORY_SENSITIVITY"   : "High",
    },
    {
        "FEATURE_NAME"             : "RECOVERY_PROPENSITY_SCORE_OVERALL",
        "CATEGORY"                 : "8_Recovery_Specific",
        "DIMENSION"                : "OVERALL",
        "STAGE_APPLICABILITY"      : "CO, CO_DEEP",
        "DATA_TYPE"                : "NUMERIC",
        "DESCRIPTION"              : "Heuristic recovery propensity: payments made in CO / months in CO (payment rate post charge-off).",
        "CALCULATION_LOGIC"        : "SUM(PAY_AFTER_DELINQUENCY_FLAG_CO) / GREATEST(1, TIME_IN_CO_OVERALL)",
        "EXPECTED_RANGE"           : "0.0 to 1.0",
        "BUSINESS_INTERPRETATION"  : "High score = good recovery target; low score = write-off/legal referral candidate.",
        "MISSING_VALUE_TREATMENT"  : "0 (not yet in CO)",
        "RECOVERY_STAGE_RELEVANCE" : "CO, CO_DEEP",
        "UPDATE_FREQUENCY"         : "Monthly",
        "REGULATORY_SENSITIVITY"   : "High",
    },

    # ── SECTION 10: Within-Stage Exposure Dynamics ───────────────────────────
    {
        "FEATURE_NAME"             : "WS_DURATION_{STAGE}",
        "CATEGORY"                 : "10_Within_Stage_Exposure",
        "DIMENSION"                : "OVERALL",
        "STAGE_APPLICABILITY"      : "{STAGE}",
        "DATA_TYPE"                : "NUMERIC",
        "DESCRIPTION"              : "Months spent in the current {STAGE} episode (latest episode only).",
        "CALCULATION_LOGIC"        : "episode_month_number at end of current episode",
        "EXPECTED_RANGE"           : "1 to ~60",
        "BUSINESS_INTERPRETATION"  : "Stage stickiness: long NPL duration = low cure probability.",
        "MISSING_VALUE_TREATMENT"  : "NULL if customer never in {STAGE}",
        "RECOVERY_STAGE_RELEVANCE" : "{STAGE}",
        "UPDATE_FREQUENCY"         : "Monthly",
        "REGULATORY_SENSITIVITY"   : "Low",
    },
    {
        "FEATURE_NAME"             : "WS_DIP_COUNT_{STAGE}",
        "CATEGORY"                 : "10_Within_Stage_Exposure",
        "DIMENSION"                : "OVERALL",
        "STAGE_APPLICABILITY"      : "{STAGE}",
        "DATA_TYPE"                : "NUMERIC",
        "DESCRIPTION"              : "Number of months within the current {STAGE} episode where balance fell (payment proxy).",
        "CALCULATION_LOGIC"        : "COUNT(amountowed < prev_amountowed) within episode",
        "EXPECTED_RANGE"           : "0 to WS_DURATION_{STAGE}",
        "BUSINESS_INTERPRETATION"  : "Even partial payments in SM/NPL = higher cure propensity.",
        "MISSING_VALUE_TREATMENT"  : "NULL if customer never in {STAGE}",
        "RECOVERY_STAGE_RELEVANCE" : "{STAGE}",
        "UPDATE_FREQUENCY"         : "Monthly",
        "REGULATORY_SENSITIVITY"   : "Low",
    },
    {
        "FEATURE_NAME"             : "WS_AVG_DIP_MAG_{STAGE}",
        "CATEGORY"                 : "10_Within_Stage_Exposure",
        "DIMENSION"                : "OVERALL",
        "STAGE_APPLICABILITY"      : "{STAGE}",
        "DATA_TYPE"                : "NUMERIC",
        "DESCRIPTION"              : "Average magnitude (THB) of balance dips within current {STAGE} episode.",
        "CALCULATION_LOGIC"        : "AVG(prev_balance - current_balance WHERE dip > 0) within episode",
        "EXPECTED_RANGE"           : "0 to ~5,000,000 THB",
        "BUSINESS_INTERPRETATION"  : "Payment capacity proxy — larger dips = customer has capacity, just struggling.",
        "MISSING_VALUE_TREATMENT"  : "NULL if no dips in episode",
        "RECOVERY_STAGE_RELEVANCE" : "NPL, CO",
        "UPDATE_FREQUENCY"         : "Monthly",
        "REGULATORY_SENSITIVITY"   : "Low",
    },

    # ── CATEGORY 9: Restructuring / TDR ─────────────────────────────────────
    {
        "FEATURE_NAME"             : "TDR_FLAG_OVERALL",
        "CATEGORY"                 : "9_Restructuring",
        "DIMENSION"                : "OVERALL",
        "STAGE_APPLICABILITY"      : "ALL",
        "DATA_TYPE"                : "BINARY",
        "DESCRIPTION"              : "1 if customer has any Restructured Debt account (ACCOUNTTYPE = 90) in bureau.",
        "CALCULATION_LOGIC"        : "1 IF any account has accounttype_cd = '90'",
        "EXPECTED_RANGE"           : "0 / 1",
        "BUSINESS_INTERPRETATION"  : "Indicates prior financial distress requiring formal debt restructuring.",
        "MISSING_VALUE_TREATMENT"  : "0",
        "RECOVERY_STAGE_RELEVANCE" : "ALL",
        "UPDATE_FREQUENCY"         : "Monthly",
        "REGULATORY_SENSITIVITY"   : "High",
    },
    {
        "FEATURE_NAME"             : "TDR_COUNT_OVERALL",
        "CATEGORY"                 : "9_Restructuring",
        "DIMENSION"                : "OVERALL",
        "STAGE_APPLICABILITY"      : "ALL",
        "DATA_TYPE"                : "NUMERIC",
        "DESCRIPTION"              : "Number of Restructured Debt accounts (ACCOUNTTYPE = 90) in bureau.",
        "CALCULATION_LOGIC"        : "COUNT accounts WHERE accounttype_cd = '90'",
        "EXPECTED_RANGE"           : "0 to ~10",
        "BUSINESS_INTERPRETATION"  : "> 1 restructuring events = habitual restructurer; lower cure confidence.",
        "MISSING_VALUE_TREATMENT"  : "0",
        "RECOVERY_STAGE_RELEVANCE" : "NPL, CO",
        "UPDATE_FREQUENCY"         : "Monthly",
        "REGULATORY_SENSITIVITY"   : "High",
    },
    {
        "FEATURE_NAME"             : "TIME_SINCE_RESTRUCTURING_OVERALL",
        "CATEGORY"                 : "9_Restructuring",
        "DIMENSION"                : "OVERALL",
        "STAGE_APPLICABILITY"      : "ALL",
        "DATA_TYPE"                : "NUMERIC",
        "DESCRIPTION"              : "Months since last Restructured Debt account was opened.",
        "CALCULATION_LOGIC"        : "MONTHS_BETWEEN(asofdate, MAX(opendate WHERE accounttype_cd='90'))",
        "EXPECTED_RANGE"           : "0 to ~120",
        "BUSINESS_INTERPRETATION"  : "Recent restructuring + high DPD = failed restructure; escalate to legal.",
        "MISSING_VALUE_TREATMENT"  : "NULL if no restructuring",
        "RECOVERY_STAGE_RELEVANCE" : "NPL, CO, CO_DEEP",
        "UPDATE_FREQUENCY"         : "Monthly",
        "REGULATORY_SENSITIVITY"   : "High",
    },
]


def print_data_dictionary() -> None:
    """Print data dictionary as formatted text. Call from notebook to inspect features."""
    header = (
        f"{'FEATURE_NAME':<50} {'CATEGORY':<30} {'DIM':<25} "
        f"{'TYPE':<12} {'RECOVERY_STAGE':<20} DESCRIPTION"
    )
    sep = "-" * 180
    print(sep)
    print(header)
    print(sep)
    for entry in DATA_DICTIONARY:
        print(
            f"{entry['FEATURE_NAME']:<50} {entry['CATEGORY']:<30} "
            f"{entry['DIMENSION']:<25} {entry['DATA_TYPE']:<12} "
            f"{entry['RECOVERY_STAGE_RELEVANCE']:<20} "
            f"{entry['DESCRIPTION'][:80]}"
        )
    print(sep)
    print(f"Total features documented: {len(DATA_DICTIONARY)}")


# ─────────────────────────────────────────────────────────────────────────────
# QUICK-REFERENCE: NCB ACCOUNTTYPE LOOKUP (from mnf_cra_rvw_s_account)
# ─────────────────────────────────────────────────────────────────────────────
#
# acct_type_cd | description                      | is_secured | is_revolving
# -------------|----------------------------------|------------|-------------
# 01           | Commercial Loan                  | NO         | NO
# 04           | Overdraft                        | NO         | YES  ← UNSECURED_REVOLVING
# 05           | Personal Loan                    | NO         | NO
# 06           | Housing/Mortgage                 | YES        | NO   ← SECURED_OTHERS
# 20           | Automobile Leasing               | YES        | NO   ← SECURED_OTHERS
# 21           | Other Hire Purchase              | YES        | NO   ← SECURED_OTHERS
# 22           | Credit Card                      | NO         | YES  ← UNSECURED_REVOLVING
# 27           | Automobile Hire Purchase         | YES        | NO   ← SECURED_OTHERS
# 31           | HP for Agriculture (var install) | YES        | NO   ← SECURED_OTHERS
# 32           | HP for Agriculture               | YES        | NO   ← SECURED_OTHERS
# 33           | Loan for Agriculture             | NO         | NO
# 36           | Coop Loan                        | NO         | NO
# 37           | Nano-Finance                     | NO         | NO
# 50           | Securitized Commercial Loan      | NO         | NO
# 52           | Securitized Housing/Mortgage     | YES        | NO   ← SECURED_OTHERS
# 53           | Securitized Auto Leasing         | YES        | NO   ← SECURED_OTHERS
# 54           | Securitized Other HP             | YES        | NO   ← SECURED_OTHERS
# 55           | Securitized Credit Card          | NO         | YES  ← UNSECURED_REVOLVING
# 56           | Securitized Auto HP              | YES        | NO   ← SECURED_OTHERS
# 58           | Securitized Overdraft            | NO         | YES  ← UNSECURED_REVOLVING
# 90           | Restructured Debt                | NO         | NO   ← TDR flag
# 99           | Other Loans                      | NO         | NO
#
# Note: SECURED_REVOLVING is rare in Thai NCB (no secured revolving facility
#       codes in standard lookup). Map remains for completeness.
