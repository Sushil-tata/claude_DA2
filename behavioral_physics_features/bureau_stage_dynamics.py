"""
Bureau Stage Dynamics Engine  v2.4
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

Feature Categories (all implemented):
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

v2.1 Fixes vs v2.0:
    - Join key: accounttype-only lookup table (no fan-out on multi-account customers)
    - Cat 1: STG_RE_DEFAULT_FLAG backward-looking (removed forward window / leakage)
    - Cat 1: STG_TIMES_IN uses countDistinct(episode_id) not row count
    - Cat 2: Added FULL_PAY_STREAK, MISSED_PAY_STREAK
    - Cat 3: Added UTIL_TREND_3M (was defined but unused)
    - Cat 5: CLOSED_LOAN_COUNT_6M uses real 6M window; tenure PIT-safe via as_of_date
    - Cat 5: Added ACTIVE_ACCT_CHANGE
    - Cat 6: Fixed nested aggregate (pre-compute max_stage per group); MAX_STAGE uses ordinal
    - Cat 6: Added CURE_RATE_6M, RE_DEFAULT_FLAG per dimension
    - Cat 7: Fixed REVOLVING_VS_TERM_PAY_GAP (now payment ratio, not util); added
             CONCENTRATION_RISK_UNSECURED, STRATEGIC_DEFAULT_INDICATOR
    - Cat 8: Added TIME_SINCE_LAST_PAY_OVERALL
    - Cat 9 within-stage: Fixed window-inside-agg (pre-compute first/last via window cols)
    - Cat 9 TDR: Added RESTRUCTURING_SUCCESS_FLAG (needs history_df)
    - Cat 10: New temporal velocity function (was not implemented)

v2.2 Fixes vs v2.1:
    - Cat 3: UTIL_TREND_3M now uses lag(3) not lag(2) — correct 3M reference
    - Cat 10: ESCALATION_VELOCITY_TO_SM implemented (was in docstring only)

v2.3 Additions:
    - Sections 14–19: 6 new recovery dynamics sections (payment effort,
      stage velocity, cure/re-default, balance recovery, engagement, vintage)
    - _stage_ordinal() helper + STAGE_LABELS constant
    - Bug 6 fix: episode builder now detects temporal gaps > 2 months

v2.4 Fixes vs v2.3:
    - Section 12: STAGE_STICKINESS_SCORE fixed (was averaging month-in-episode
      counter instead of episode duration)
    - Section 13: New — TIME_SINCE_RESTRUCTURING_OVERALL added (was documented
      in Section 11 docstring but never computed)
    - Section 11 docstring: removed TIME_SINCE_RESTRUCTURING reference
      (moved to Section 13)

Version : 2.4.0
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
STAGE_DEF = {
    "CURRENT" : (0,   0,   0),
    "X"       : (1,   30,  1),
    "SM"      : (31,  90,  2),
    "NPL"     : (91,  180, 3),
    "CO"      : (181, 360, 4),
    "CO_DEEP" : (361, 9999,5),
}
STAGES = list(STAGE_DEF.keys())

# ── NCB account-type → dimension constants ────────────────────────────────────
SECURED_CODES   = {"06","20","21","27","31","32","52","53","54","56"}
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

# Stage → lowercase label used as column suffix in new sections
STAGE_LABELS = {
    "CURRENT": "current",
    "X":       "x",
    "SM":      "sm",
    "NPL":     "npl",
    "CO":      "co",
    "CO_DEEP": "co_deep",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_div(num, den, default=None):
    """NULL-safe division; returns NULL (or default) on zero/null denominator."""
    result = F.when((den.isNotNull()) & (den != 0), num / den)
    return result.otherwise(default) if default is not None else result


def _dim_filter(acct_type_col: str, dim: str):
    """Return a Column boolean expression for the given product dimension."""
    c = F.col(acct_type_col).cast("string")
    is_sec  = c.isin(list(SECURED_CODES))
    is_revl = c.isin(list(REVOLVING_CODES))
    if dim == "OVERALL":           return F.lit(True)
    elif dim == "SECURED":         return is_sec
    elif dim == "UNSECURED":       return ~is_sec
    elif dim == "SECURED_REVOLVING":   return is_sec & is_revl
    elif dim == "SECURED_OTHERS":      return is_sec & ~is_revl
    elif dim == "UNSECURED_REVOLVING": return ~is_sec & is_revl
    elif dim == "UNSECURED_OTHERS":    return ~is_sec & ~is_revl
    else: raise ValueError(f"Unknown dimension: {dim}")


def _assign_stage(dpd_col):
    """Map integer DPD → stage label (CURRENT/X/SM/NPL/CO/CO_DEEP)."""
    return (
        F.when(dpd_col.isNull(),   "CURRENT")
         .when(dpd_col == 0,       "CURRENT")
         .when(dpd_col <= 30,      "X")
         .when(dpd_col <= 90,      "SM")
         .when(dpd_col <= 180,     "NPL")
         .when(dpd_col <= 360,     "CO")
         .otherwise(               "CO_DEEP")
    )


def _stage_numeric(stage_col):
    """Map stage label Column → ordinal 0–5."""
    return (
        F.when(F.col(stage_col) == "CURRENT", 0)
         .when(F.col(stage_col) == "X",       1)
         .when(F.col(stage_col) == "SM",       2)
         .when(F.col(stage_col) == "NPL",      3)
         .when(F.col(stage_col) == "CO",       4)
         .when(F.col(stage_col) == "CO_DEEP",  5)
         .otherwise(F.lit(None))
    )


def _stage_ordinal(col_name: str):
    """
    Map stage label string column → ordinal integer 0–5.
    Accepts a column *name* string (unlike _stage_numeric which takes column).
    Used by recovery dynamics sections so they share the same mapping.
    """
    c = F.col(col_name)
    return (
        F.when(c == "CURRENT", 0)
         .when(c == "X",       1)
         .when(c == "SM",      2)
         .when(c == "NPL",     3)
         .when(c == "CO",      4)
         .when(c == "CO_DEEP", 5)
         .otherwise(F.lit(None))
    )


def _acct_type_dim_flags(spark: SparkSession) -> DataFrame:
    """
    Build a pure accounttype → (is_secured, is_revolving) lookup DataFrame.

    FIX (v2.1): Previously joined account_df on [ref_no, accounttype], which
    caused row fan-out when customers had multiple accounts of the same type.
    Now we join history_df only on `accounttype` (single column), since the
    dimension flags depend only on account type, not on the customer.
    """
    all_codes = (
        list(SECURED_CODES | REVOLVING_CODES) +
        ["01","05","07","08","09","10","11","12","13","14","15","16","17",
         "18","19","28","29","33","34","35","36","37","38","50","51","57",
         "58","90","99"]
    )
    rows = []
    seen = set()
    for code in all_codes:
        if code in seen:
            continue
        seen.add(code)
        is_sec  = int(code in SECURED_CODES)
        is_revl = int(code in REVOLVING_CODES)
        rows.append((code, is_sec, is_revl))
    return spark.createDataFrame(rows, [ACCT_TYPE, "is_secured", "is_revolving"])


def _streak(df: DataFrame, flag_col: str, ref_col: str, date_col: str,
            out_col: str, value: int = 1) -> DataFrame:
    """
    Compute consecutive streak of `value` in `flag_col` ending at each row.
    Uses island detection (cumsum of breaks).
    """
    w = Window.partitionBy(ref_col).orderBy(date_col)
    is_match = F.when(F.col(flag_col) == value, 1).otherwise(0)
    # Each time is_match breaks, increment group counter
    df = df.withColumn(f"_brk_{out_col}",
                       F.sum(F.when(F.col(flag_col) != value, 1).otherwise(0))
                        .over(w.rowsBetween(Window.unboundedPreceding, 0)))
    w_grp = Window.partitionBy(ref_col, f"_brk_{out_col}").orderBy(date_col)
    df = df.withColumn(out_col,
                       F.when(is_match == 1, F.row_number().over(w_grp)).otherwise(0))
    return df.drop(f"_brk_{out_col}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — NCB ACCOUNT TYPE ENRICHMENT
# ─────────────────────────────────────────────────────────────────────────────

def enrich_account_dimensions(account_df: DataFrame) -> DataFrame:
    """
    Add is_secured, is_revolving, dim_label to the account table.
    Operates on the account table itself (not history), so downstream joins
    on accounttype alone are safe.
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
            .withColumn("is_secured",   is_sec.cast("int"))
            .withColumn("is_revolving", is_revl.cast("int"))
            .withColumn("dim_label",    dim_label))


def _join_dim_flags(history_df: DataFrame, spark: SparkSession) -> DataFrame:
    """
    Join dimension flags onto history_df via accounttype only (no ref_no).

    FIX (v2.1): Using a pure type-level lookup avoids the fan-out bug that
    occurred when joining on [ref_no, accounttype] with multi-account customers.
    """
    lookup = _acct_type_dim_flags(spark).select(
        F.col(ACCT_TYPE).alias("_acct_key"), "is_secured", "is_revolving")
    return history_df.join(
        lookup,
        on=history_df[ACCT_TYPE].cast("string") == lookup["_acct_key"],
        how="left"
    ).drop("_acct_key")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — STAGE ASSIGNMENT & EPISODE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_dpd_stage(state_df: DataFrame) -> DataFrame:
    """
    Assign dpd_stage (CURRENT/X/SM/NPL/CO/CO_DEEP) and stage_numeric (0–5)
    from bureau_max_dpd.
    """
    return (state_df
            .withColumn(STAGE_COL,       _assign_stage(F.col(DPD_COL)))
            .withColumn("stage_numeric", _stage_numeric(STAGE_COL)))


def build_stage_episodes(state_df: DataFrame) -> DataFrame:
    """
    Assign episode IDs to contiguous runs in the same DPD stage.

    Output adds:
        episode_id           — cumulative count of stage changes per customer
        episode_stage        — stage label at episode start
        episode_start_date   — first date of the episode
        episode_month_number — 1 = entry month, 2 = second month …
    """
    w_time = Window.partitionBy(REF).orderBy(DATE)

    df = (state_df
          .withColumn(STAGE_COL,       _assign_stage(F.col(DPD_COL)))
          .withColumn("stage_numeric", _stage_numeric(STAGE_COL))
          .withColumn("_prev_stage", F.lag(STAGE_COL).over(w_time))
          .withColumn("_prev_date",  F.lag(DATE, 1).over(w_time))
          .withColumn("_month_gap",
                      F.when(F.col("_prev_date").isNotNull(),
                             F.months_between(F.col(DATE), F.col("_prev_date")))
                       .otherwise(F.lit(None)))
          .withColumn("_stage_changed",
                      F.when(F.col("_prev_stage").isNull(), 1)
                       .when(F.col("_prev_stage") != F.col(STAGE_COL), 1)
                       .when(F.col("_month_gap") > 2, 1)   # gap > 2M = new episode
                       .otherwise(0))
          .withColumn("episode_id",
                      F.sum("_stage_changed").over(
                          w_time.rowsBetween(Window.unboundedPreceding, 0)))
          .drop("_prev_stage", "_stage_changed", "_prev_date", "_month_gap"))

    w_ep = Window.partitionBy(REF, "episode_id").orderBy(DATE)

    return (df
            .withColumn("episode_start_date",  F.first(DATE).over(w_ep))
            .withColumn("episode_stage",       F.first(STAGE_COL).over(w_ep))
            .withColumn("episode_month_number", F.row_number().over(w_ep)))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — CATEGORY 1: STAGE TRANSITION & DYNAMICS
# ─────────────────────────────────────────────────────────────────────────────

def build_stage_transition_features(episode_df: DataFrame) -> DataFrame:
    """
    Cross-stage transition dynamics (OVERALL dimension).

    Features:
        STG_TRANSITION_OVERALL          stage→stage string e.g. "X->SM"
        STG_WORSENING_FLAG_OVERALL      1 if moved to worse stage
        STG_IMPROVEMENT_FLAG_OVERALL    1 if moved to better stage
        STG_DPD_CHANGE_1M_OVERALL       ΔDPD vs previous month
        STG_DPD_ACCELERATION_OVERALL    second derivative of DPD
        STG_DAYS_IN_STAGE_OVERALL       months in current episode
        STG_TIMES_IN_{STAGE}_OVERALL    distinct episode count per stage (lifetime)
        STG_MAX_STAGE_REACHED_OVERALL   worst stage ordinal ever (0–5)
        STG_STAGE_VOLATILITY_6M_OVERALL std dev of stage_numeric last 6M
        STG_CURE_RATE_6M_OVERALL        fraction of last 6M months with improvement
        STG_RE_DEFAULT_FLAG_OVERALL     1 if improvement + re-worsening in past 12M
                                        (FIX v2.1: backward-looking only — no leakage)
    """
    w_time = Window.partitionBy(REF).orderBy(DATE)
    w_6m   = Window.partitionBy(REF).orderBy(DATE).rowsBetween(-5, 0)
    w_12m  = Window.partitionBy(REF).orderBy(DATE).rowsBetween(-11, 0)
    w_all  = Window.partitionBy(REF).orderBy(DATE).rowsBetween(
                 Window.unboundedPreceding, 0)

    df = episode_df

    prev_stage = F.lag(STAGE_COL).over(w_time)
    prev_num   = F.lag("stage_numeric").over(w_time)
    prev_dpd   = F.lag(DPD_COL).over(w_time)
    prev2_dpd  = F.lag(DPD_COL, 2).over(w_time)

    df = (df
          .withColumn("_prev_num",    prev_num)
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
                      F.stddev("stage_numeric").over(w_6m))
          .withColumn("STG_CURE_RATE_6M_OVERALL",
                      _safe_div(
                          F.sum("STG_IMPROVEMENT_FLAG_OVERALL").over(w_6m),
                          F.lit(6))))

    # FIX v2.1: STG_TIMES_IN — count DISTINCT episodes per stage (not row count)
    for stage in STAGES:
        df = df.withColumn(
            f"STG_TIMES_IN_{stage}_OVERALL",
            F.countDistinct(
                F.when(F.col("episode_stage") == stage, F.col("episode_id"))
            ).over(w_all))

    # FIX v2.1: STG_RE_DEFAULT_FLAG — backward-looking only (no forward window)
    # 1 if there was an improvement AND a subsequent worsening in past 12M
    w_hist12 = Window.partitionBy(REF).orderBy(DATE).rowsBetween(-12, -1)
    w_rec6   = Window.partitionBy(REF).orderBy(DATE).rowsBetween(-6, 0)
    df = df.withColumn(
        "STG_RE_DEFAULT_FLAG_OVERALL",
        F.when(
            (F.max("STG_IMPROVEMENT_FLAG_OVERALL").over(w_hist12) == 1) &
            (F.max("STG_WORSENING_FLAG_OVERALL").over(w_rec6) == 1),
            1).otherwise(0))

    keep = [REF, DATE] + [c for c in df.columns if c.startswith("STG_")]
    return df.select(keep).dropDuplicates([REF, DATE])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — CATEGORY 2: PAYMENT BEHAVIOR DYNAMICS
# ─────────────────────────────────────────────────────────────────────────────

def build_payment_behavior_features(
    history_df: DataFrame,
    spark: SparkSession,
) -> DataFrame:
    """
    Payment dynamics across all 7 product dimensions.

    FIX v2.1: Join on accounttype only (not [ref_no, accounttype]) to prevent
    row fan-out when customers have multiple accounts of the same type.

    Features per dimension (DIM):
        PAY_RATIO_{DIM}           balance-decrease / balance (payment proxy)
        PAY_RATIO_CHANGE_{DIM}    Δ PAY_RATIO vs previous month
        PAY_CONSISTENCY_6M_{DIM}  fraction of 6M months with payment
        PARTIAL_PAY_FLAG_{DIM}    1 if 0 < decrease < full balance
        FULL_PAY_STREAK_{DIM}     consecutive months of balance decrease (FIX v2.1: implemented)
        MISSED_PAY_STREAK_{DIM}   consecutive months of zero payment (FIX v2.1: implemented)
        BAL_CHANGE_{DIM}          Δ amountowed (negative = repayment)
        BAL_CHANGE_PCT_{DIM}      % Δ amountowed
    """
    df = _join_dim_flags(history_df, spark)

    w_acct = Window.partitionBy(REF, ACCT_TYPE).orderBy(DATE)

    df = (df
          .withColumn("_prev_bal",  F.lag(BAL).over(w_acct))
          .withColumn("_bal_chg",   F.col(BAL) - F.col("_prev_bal"))
          .withColumn("_full_pay",  F.when(F.col("_bal_chg") < 0, 1).otherwise(0))
          .withColumn("_miss_pay",  F.when(
              (F.col("_bal_chg") >= 0) | F.col("_bal_chg").isNull(), 1).otherwise(0)))

    results = []
    for dim in DIMS:
        filt = df.filter(_dim_filter(ACCT_TYPE, dim))

        # Aggregate to customer-month level
        w_cust = Window.partitionBy(REF).orderBy(DATE)
        w6     = w_cust.rowsBetween(-5, 0)

        agg = (filt.groupBy(REF, DATE)
               .agg(
                   F.sum("_bal_chg").alias("_sum_bal_chg"),
                   F.sum(BAL).alias("_tot_bal"),
                   F.sum("_full_pay").alias(f"_fp_{dim}"),
                   F.max(F.when((F.col("_bal_chg") < 0) &
                                (F.col(BAL) > 0), 1).otherwise(0))
                    .alias(f"PARTIAL_PAY_FLAG_{dim}"),
               )
               .withColumn(f"PAY_RATIO_{dim}",
                           _safe_div(-F.col("_sum_bal_chg"), F.col("_tot_bal")))
               .withColumn(f"PAY_RATIO_CHANGE_{dim}",
                           F.col(f"PAY_RATIO_{dim}") -
                           F.lag(f"PAY_RATIO_{dim}").over(w_cust))
               .withColumn(f"PAY_CONSISTENCY_6M_{dim}",
                           _safe_div(
                               F.sum(F.when(F.col(f"_fp_{dim}") > 0, 1).otherwise(0))
                                .over(w6),
                               F.lit(6)))
               .withColumn(f"BAL_CHANGE_{dim}", F.col("_sum_bal_chg"))
               .withColumn(f"BAL_CHANGE_PCT_{dim}",
                           _safe_div(F.col("_sum_bal_chg"), F.col("_tot_bal"))))

        # FIX v2.1: FULL_PAY_STREAK / MISSED_PAY_STREAK via island detection
        agg = agg.withColumn(f"_pay_flag_{dim}",
                             F.when(F.col(f"_fp_{dim}") > 0, 1).otherwise(0))
        agg = _streak(agg, f"_pay_flag_{dim}", REF, DATE,
                      f"FULL_PAY_STREAK_{dim}", value=1)
        agg = _streak(agg, f"_pay_flag_{dim}", REF, DATE,
                      f"MISSED_PAY_STREAK_{dim}", value=0)

        keep_cols = [REF, DATE,
                     f"PAY_RATIO_{dim}", f"PAY_RATIO_CHANGE_{dim}",
                     f"PAY_CONSISTENCY_6M_{dim}", f"PARTIAL_PAY_FLAG_{dim}",
                     f"FULL_PAY_STREAK_{dim}", f"MISSED_PAY_STREAK_{dim}",
                     f"BAL_CHANGE_{dim}", f"BAL_CHANGE_PCT_{dim}"]
        results.append(agg.select(keep_cols))

    out = results[0]
    for r in results[1:]:
        out = out.join(r, on=[REF, DATE], how="full")
    return out.dropDuplicates([REF, DATE])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — CATEGORY 3: EXPOSURE & UTILISATION DYNAMICS
# ─────────────────────────────────────────────────────────────────────────────

def build_exposure_utilisation_features(
    history_df: DataFrame,
    spark: SparkSession,
) -> DataFrame:
    """
    Credit utilisation and exposure dynamics per dimension.

    FIX v2.1: join on accounttype only; UTIL_TREND_3M now computed.

    Features per dimension (DIM):
        UTIL_{DIM}                 avg utilisation (balance/limit)
        UTIL_CHANGE_{DIM}          Δ utilisation vs previous month
        UTIL_TREND_3M_{DIM}        slope of utilisation over 3M (FIX v2.1)
        MAX_UTIL_EVER_{DIM}        lifetime peak utilisation
        NET_EXPOSURE_CHANGE_{DIM}  Δ(limit - balance)
        LIMIT_REDUCTION_FLAG_{DIM} 1 if limit decreased
        EXPOSURE_AT_RISK_{DIM}     total balance in dimension
    """
    df = _join_dim_flags(history_df, spark)

    w_acct = Window.partitionBy(REF, ACCT_TYPE).orderBy(DATE)
    w_all  = w_acct.rowsBetween(Window.unboundedPreceding, 0)

    df = (df
          .withColumn("_util",
                      _safe_div(F.col(BAL), F.col(LIMIT)))
          .withColumn("_prev_util",  F.lag("_util").over(w_acct))
          .withColumn("_prev_limit", F.lag(LIMIT).over(w_acct))
          .withColumn("_max_util",   F.max("_util").over(w_all))
          .withColumn("_net_exp",    F.col(LIMIT) - F.col(BAL))
          .withColumn("_prev_net",   F.lag("_net_exp").over(w_acct))
          .withColumn("_lim_red",
                      F.when(F.col(LIMIT) < F.col("_prev_limit"), 1).otherwise(0)))

    results = []
    for dim in DIMS:
        filt = df.filter(_dim_filter(ACCT_TYPE, dim))

        w_cust = Window.partitionBy(REF).orderBy(DATE)
        w3     = w_cust.rowsBetween(-2, 0)

        agg = (filt.groupBy(REF, DATE)
               .agg(
                   F.avg("_util").alias(f"UTIL_{dim}"),
                   F.avg(F.col("_util") - F.col("_prev_util")).alias(f"UTIL_CHANGE_{dim}"),
                   F.max("_max_util").alias(f"MAX_UTIL_EVER_{dim}"),
                   F.sum(F.col("_net_exp") - F.col("_prev_net"))
                    .alias(f"NET_EXPOSURE_CHANGE_{dim}"),
                   F.max("_lim_red").alias(f"LIMIT_REDUCTION_FLAG_{dim}"),
                   F.sum(BAL).alias(f"EXPOSURE_AT_RISK_{dim}"),
               ))

        # UTIL_TREND_3M — (util_now - util_3m_ago) / 3  [lag(3) = 3 months prior]
        agg = agg.withColumn(
            f"UTIL_TREND_3M_{dim}",
            _safe_div(
                F.col(f"UTIL_{dim}") -
                F.lag(f"UTIL_{dim}", 3).over(w_cust),
                F.lit(3)))

        results.append(agg.select(
            REF, DATE,
            f"UTIL_{dim}", f"UTIL_CHANGE_{dim}", f"UTIL_TREND_3M_{dim}",
            f"MAX_UTIL_EVER_{dim}", f"NET_EXPOSURE_CHANGE_{dim}",
            f"LIMIT_REDUCTION_FLAG_{dim}", f"EXPOSURE_AT_RISK_{dim}"))

    out = results[0]
    for r in results[1:]:
        out = out.join(r, on=[REF, DATE], how="full")
    return out.dropDuplicates([REF, DATE])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — CATEGORY 4: ENQUIRY & NEW CREDIT DYNAMICS
# ─────────────────────────────────────────────────────────────────────────────

def build_enquiry_features(enquiry_df: DataFrame) -> DataFrame:
    """
    Enquiry velocity and new-credit dynamics.
    Input: mnf_cra_rvw_s_enquiry (ref_no, enquirydate, …)

    Features (OVERALL):
        ENQ_COUNT_1M_OVERALL    enquiries in last 1M
        ENQ_COUNT_3M_OVERALL    enquiries in last 3M
        ENQ_COUNT_6M_OVERALL    enquiries in last 6M
        ENQ_COUNT_12M_OVERALL   enquiries in last 12M
        ENQ_COUNT_CHANGE_OVERALL Δ 3M count vs previous 3M
        ENQ_BURST_FLAG_OVERALL  1 if ≥ 3 enquiries in current month
        ENQ_VELOCITY_OVERALL    enquiries per month (3M rolling avg)
    """
    df = (enquiry_df
          .withColumn("enq_month", F.date_trunc("month", F.col("enquirydate")))
          .groupBy(REF, F.col("enq_month").alias(DATE))
          .agg(F.count("*").alias("_enq_cnt")))

    w1  = Window.partitionBy(REF).orderBy(DATE)
    w3  = w1.rowsBetween(-2, 0)
    w6  = w1.rowsBetween(-5, 0)
    w12 = w1.rowsBetween(-11, 0)

    df = (df
          .withColumn("ENQ_COUNT_1M_OVERALL",  F.col("_enq_cnt"))
          .withColumn("ENQ_COUNT_3M_OVERALL",  F.sum("_enq_cnt").over(w3))
          .withColumn("ENQ_COUNT_6M_OVERALL",  F.sum("_enq_cnt").over(w6))
          .withColumn("ENQ_COUNT_12M_OVERALL", F.sum("_enq_cnt").over(w12))
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
# SECTION 6 — CATEGORY 5: ACCOUNT CLOSURE & ATTRITION DYNAMICS
# ─────────────────────────────────────────────────────────────────────────────

def build_account_closure_features(
    account_df: DataFrame,
    as_of_date: str,          # FIX v2.1: PIT-safe tenure; format "YYYY-MM-DD"
) -> DataFrame:
    """
    Account-level open/close dynamics per dimension.
    Input: mnf_cra_rvw_s_account (ref_no, opendate, closedate, accounttype)

    FIX v2.1:
    - CLOSED_LOAN_COUNT_6M now uses real 6M window from as_of_date
    - RELATIONSHIP_TENURE uses as_of_date (PIT-safe), not current_date()
    - ACTIVE_ACCT_CHANGE added

    Features per dimension (DIM):
        CLOSED_LOAN_COUNT_6M_{DIM}  accounts closed in 6M prior to as_of_date
        CLOSURE_RATE_{DIM}          closed / total accounts
        ACTIVE_ACCT_{DIM}           count of open accounts
        ACTIVE_ACCT_CHANGE_{DIM}    Δ active accounts (opened - closed in 6M)
        RELATIONSHIP_TENURE_{DIM}   months since first account opened (PIT-safe)
    """
    pit = F.to_date(F.lit(as_of_date))
    six_m_ago = F.add_months(pit, -6)

    acct = enrich_account_dimensions(account_df)

    results = []
    for dim in DIMS:
        filt = acct.filter(_dim_filter(ACCT_TYPE, dim))

        agg = (filt.groupBy(REF)
               .agg(
                   # Active = not closed, or closed after as_of_date
                   F.sum(F.when(
                       F.col("closedate").isNull() |
                       (F.to_date(F.col("closedate")) > pit), 1).otherwise(0))
                    .alias(f"ACTIVE_ACCT_{dim}"),
                   # Closed in 6M window
                   F.sum(F.when(
                       F.to_date(F.col("closedate")).between(six_m_ago, pit), 1
                   ).otherwise(0)).alias(f"CLOSED_LOAN_COUNT_6M_{dim}"),
                   # Opened in 6M window
                   F.sum(F.when(
                       F.to_date(F.col("opendate")).between(six_m_ago, pit), 1
                   ).otherwise(0)).alias(f"_opened_6m_{dim}"),
                   # First open date
                   F.min("opendate").alias(f"_first_open_{dim}"),
               )
               .withColumn(f"ACTIVE_ACCT_CHANGE_{dim}",
                           F.col(f"_opened_6m_{dim}") -
                           F.col(f"CLOSED_LOAN_COUNT_6M_{dim}"))
               .withColumn(f"CLOSURE_RATE_{dim}",
                           _safe_div(
                               F.col(f"CLOSED_LOAN_COUNT_6M_{dim}"),
                               F.greatest(F.lit(1), F.col(f"ACTIVE_ACCT_{dim}"))))
               .withColumn(f"RELATIONSHIP_TENURE_{dim}",
                           F.months_between(pit, F.to_date(F.col(f"_first_open_{dim}"))))
               .drop(f"_first_open_{dim}", f"_opened_6m_{dim}"))

        results.append(agg)

    out = results[0]
    for r in results[1:]:
        out = out.join(r, on=REF, how="full")
    return out.dropDuplicates([REF])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — CATEGORY 6: DELINQUENCY SEVERITY INDICATORS
# ─────────────────────────────────────────────────────────────────────────────

def build_delinquency_severity_features(
    history_df: DataFrame,
    spark: SparkSession,
) -> DataFrame:
    """
    Severity and concentration of delinquency per dimension.

    FIX v2.1:
    - CONCENTRATION_WORST_DPD: pre-computed max_stage_num per group to avoid
      nested aggregate (AnalysisException in PySpark)
    - MAX_STAGE_REACHED: now uses stage ordinal (0–5), not severity float
    - Added CURE_RATE_6M, RE_DEFAULT_FLAG per dimension

    Features per dimension (DIM):
        SEVERITY_SCORE_{DIM}           balance-weighted avg stage severity
        CONCENTRATION_WORST_DPD_{DIM}  % balance in worst DPD bucket
        DELINQUENCY_BREADTH_{DIM}      # products with DPD > 0
        MAX_STAGE_REACHED_{DIM}        worst stage ordinal ever (0–5)
        CURE_RATE_6M_{DIM}             % months with stage improvement in 6M
        RE_DEFAULT_FLAG_{DIM}          1 if improvement then worsening in 12M
    """
    df = _join_dim_flags(history_df, spark)

    df = (df
          .withColumn("_dpd_stage", _assign_stage(F.col(DPD_COL)))
          .withColumn("_stage_num",  _stage_numeric("_dpd_stage"))
          .withColumn("_is_delinq",  F.when(F.col(DPD_COL) > 0, 1).otherwise(0)))

    results = []
    for dim in DIMS:
        filt = df.filter(_dim_filter(ACCT_TYPE, dim))

        # FIX v2.1: pre-compute max_stage_num per (ref, date) BEFORE groupBy
        # so it can be used as a column in the agg without nesting
        w_grp = Window.partitionBy(REF, DATE)
        filt = filt.withColumn("_max_stg_grp", F.max("_stage_num").over(w_grp))

        w_all = Window.partitionBy(REF).orderBy(DATE).rowsBetween(
                    Window.unboundedPreceding, 0)
        w_cust = Window.partitionBy(REF).orderBy(DATE)
        w6     = w_cust.rowsBetween(-5, 0)
        w_hist = w_cust.rowsBetween(-12, -1)
        w_rec  = w_cust.rowsBetween(-6, 0)

        # Stage improvement/worsening at per-tradeline level
        filt = (filt
                .withColumn("_prev_stg_num",
                            F.lag("_stage_num").over(
                                Window.partitionBy(REF, ACCT_TYPE).orderBy(DATE)))
                .withColumn("_tl_improve",
                            F.when(F.col("_stage_num") < F.col("_prev_stg_num"), 1)
                             .otherwise(0))
                .withColumn("_tl_worsen",
                            F.when(F.col("_stage_num") > F.col("_prev_stg_num"), 1)
                             .otherwise(0)))

        agg = (filt.groupBy(REF, DATE)
               .agg(
                   _safe_div(
                       F.sum(F.col("_stage_num") * F.col(BAL)),
                       F.sum(BAL)).alias(f"SEVERITY_SCORE_{dim}"),
                   _safe_div(
                       F.sum(F.when(
                           F.col("_stage_num") == F.col("_max_stg_grp"),
                           F.col(BAL))),
                       F.sum(BAL)).alias(f"CONCENTRATION_WORST_DPD_{dim}"),
                   F.sum("_is_delinq").alias(f"DELINQUENCY_BREADTH_{dim}"),
                   F.max("_stage_num").alias(f"_cur_max_stg_{dim}"),
                   F.max("_tl_improve").alias(f"_improved_{dim}"),
                   F.max("_tl_worsen").alias(f"_worsened_{dim}"),
               ))

        # MAX_STAGE_REACHED: max ordinal ever seen (FIX v2.1: ordinal not float)
        agg = agg.withColumn(
            f"MAX_STAGE_REACHED_{dim}",
            F.max(f"_cur_max_stg_{dim}").over(w_all))

        # CURE_RATE_6M
        agg = agg.withColumn(
            f"CURE_RATE_6M_{dim}",
            _safe_div(
                F.sum(f"_improved_{dim}").over(w6),
                F.lit(6)))

        # RE_DEFAULT_FLAG (backward-looking — no leakage)
        agg = agg.withColumn(
            f"RE_DEFAULT_FLAG_{dim}",
            F.when(
                (F.max(f"_improved_{dim}").over(w_hist) == 1) &
                (F.max(f"_worsened_{dim}").over(w_rec) == 1),
                1).otherwise(0))

        keep = [REF, DATE,
                f"SEVERITY_SCORE_{dim}", f"CONCENTRATION_WORST_DPD_{dim}",
                f"DELINQUENCY_BREADTH_{dim}", f"MAX_STAGE_REACHED_{dim}",
                f"CURE_RATE_6M_{dim}", f"RE_DEFAULT_FLAG_{dim}"]
        results.append(agg.select(keep))

    out = results[0]
    for r in results[1:]:
        out = out.join(r, on=[REF, DATE], how="full")
    return out.dropDuplicates([REF, DATE])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — CATEGORY 7: CROSS-DIMENSIONAL COMPARATIVE FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def build_cross_dimension_features(
    history_df: DataFrame,
    spark: SparkSession,
) -> DataFrame:
    """
    Cross-product-dimension comparisons revealing selective default.

    FIX v2.1:
    - REVOLVING_VS_TERM_PAY_GAP now uses payment ratio (balance-decrease proxy),
      not utilisation (naming was misleading)
    - Added CONCENTRATION_RISK_UNSECURED (Herfindahl index)
    - Added STRATEGIC_DEFAULT_INDICATOR

    Features:
        SECURED_VS_UNSECURED_DPD_GAP    max DPD secured - max DPD unsecured
        REVOLVING_VS_TERM_PAY_GAP       payment ratio revolving - payment ratio term
        STAGE_DIVERGENCE_FLAG           1 if secured CURRENT but unsecured NPL+
        BEHAVIORAL_ARBITRAGE_FLAG       1 if paying secured while defaulting unsecured
        CONCENTRATION_RISK_UNSECURED    Herfindahl index of unsecured balance
        STRATEGIC_DEFAULT_INDICATOR     low util secured + high DPD unsecured
    """
    df = _join_dim_flags(history_df, spark)

    df = (df
          .withColumn("_dpd_stage", _assign_stage(F.col(DPD_COL)))
          .withColumn("_prev_bal",
                      F.lag(BAL).over(
                          Window.partitionBy(REF, ACCT_TYPE).orderBy(DATE)))
          .withColumn("_pay_ratio",
                      _safe_div(-(F.col(BAL) - F.col("_prev_bal")), F.col("_prev_bal"))))

    agg = (df.groupBy(REF, DATE)
           .agg(
               F.max(F.when(F.col("is_secured") == 1, F.col(DPD_COL)))
                .alias("_max_dpd_sec"),
               F.max(F.when(F.col("is_secured") == 0, F.col(DPD_COL)))
                .alias("_max_dpd_unsec"),
               # Payment ratio by revolving vs term
               F.avg(F.when(F.col("is_revolving") == 1, F.col("_pay_ratio")))
                .alias("_pr_revl"),
               F.avg(F.when(F.col("is_revolving") == 0, F.col("_pay_ratio")))
                .alias("_pr_term"),
               # Secured CURRENT flag
               F.max(F.when(
                   (F.col("is_secured") == 1) & (F.col("_dpd_stage") == "CURRENT"),
                   1).otherwise(0)).alias("_sec_current"),
               # Unsecured NPL+ flag
               F.max(F.when(
                   (F.col("is_secured") == 0) &
                   (F.col("_dpd_stage").isin(["NPL","CO","CO_DEEP"])),
                   1).otherwise(0)).alias("_unsec_npl_plus"),
               # Secured paying flag
               F.max(F.when(
                   (F.col("is_secured") == 1) & (F.col(DPD_COL) == 0),
                   1).otherwise(0)).alias("_sec_paying"),
               # Unsecured defaulting flag
               F.max(F.when(
                   (F.col("is_secured") == 0) & (F.col(DPD_COL) > 90),
                   1).otherwise(0)).alias("_unsec_default"),
               # Herfindahl: sum(bal^2) / (sum(bal))^2 for unsecured
               F.sum(F.when(F.col("is_secured") == 0,
                            F.col(BAL) * F.col(BAL)).otherwise(0))
                .alias("_unsec_bal_sq"),
               F.sum(F.when(F.col("is_secured") == 0, F.col(BAL)).otherwise(0))
                .alias("_unsec_bal"),
               # Low util secured (proxy for capacity)
               F.avg(F.when(
                   F.col("is_secured") == 1,
                   _safe_div(F.col(BAL), F.col(LIMIT)))).alias("_sec_util"),
           )
           .withColumn("SECURED_VS_UNSECURED_DPD_GAP",
                       F.col("_max_dpd_sec") - F.col("_max_dpd_unsec"))
           .withColumn("REVOLVING_VS_TERM_PAY_GAP",
                       F.col("_pr_revl") - F.col("_pr_term"))
           .withColumn("STAGE_DIVERGENCE_FLAG",
                       F.when((F.col("_sec_current") == 1) &
                              (F.col("_unsec_npl_plus") == 1), 1).otherwise(0))
           .withColumn("BEHAVIORAL_ARBITRAGE_FLAG",
                       F.when((F.col("_sec_paying") == 1) &
                              (F.col("_unsec_default") == 1), 1).otherwise(0))
           .withColumn("CONCENTRATION_RISK_UNSECURED",
                       _safe_div(
                           F.col("_unsec_bal_sq"),
                           F.col("_unsec_bal") * F.col("_unsec_bal")))
           .withColumn("STRATEGIC_DEFAULT_INDICATOR",
                       F.when(
                           (F.col("_sec_util") < 0.5) &     # capacity exists
                           (F.col("_unsec_default") == 1),  # choosing not to pay unsecured
                           1).otherwise(0))
           .select(REF, DATE,
                   "SECURED_VS_UNSECURED_DPD_GAP",
                   "REVOLVING_VS_TERM_PAY_GAP",
                   "STAGE_DIVERGENCE_FLAG",
                   "BEHAVIORAL_ARBITRAGE_FLAG",
                   "CONCENTRATION_RISK_UNSECURED",
                   "STRATEGIC_DEFAULT_INDICATOR"))

    return agg.dropDuplicates([REF, DATE])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — CATEGORY 8: RECOVERY-SPECIFIC INDICATORS
# ─────────────────────────────────────────────────────────────────────────────

def build_recovery_indicators(
    history_df: DataFrame,
    spark: SparkSession,
) -> DataFrame:
    """
    Recovery-oriented signals for CO/CO_DEEP stage accounts.

    FIX v2.1: Added TIME_SINCE_LAST_PAY_OVERALL.

    Features:
        TIME_SINCE_LAST_PAY_OVERALL      months since any payment (balance decrease)
        PAY_AFTER_DELINQUENCY_FLAG_CO    1 if payment observed while in CO/CO_DEEP
        RECOVERY_PROPENSITY_SCORE_OVERALL payments in CO / months in CO
        TIME_IN_CO_OVERALL               cumulative months in CO stage
        TIME_IN_CO_DEEP_OVERALL          cumulative months in CO_DEEP stage
    """
    df = _join_dim_flags(history_df, spark)

    w_acct = Window.partitionBy(REF, ACCT_TYPE).orderBy(DATE)

    df = (df
          .withColumn("_dpd_stage", _assign_stage(F.col(DPD_COL)))
          .withColumn("_prev_bal",  F.lag(BAL).over(w_acct))
          .withColumn("_bal_fell",
                      F.when(F.col(BAL) < F.col("_prev_bal"), 1).otherwise(0)))

    w_all  = Window.partitionBy(REF).orderBy(DATE).rowsBetween(
                 Window.unboundedPreceding, 0)
    w_cust = Window.partitionBy(REF).orderBy(DATE)

    agg = (df.groupBy(REF, DATE)
           .agg(
               F.max("_bal_fell").alias("_any_pay"),
               F.max(F.when(
                   F.col("_dpd_stage").isin(["CO","CO_DEEP"]) &
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

    # FIX v2.1: TIME_SINCE_LAST_PAY — row_number since last _any_pay=1
    agg = agg.withColumn("_cum_pay_grp",
                         F.sum(F.when(F.col("_any_pay") == 1, 1).otherwise(0))
                          .over(w_all))
    agg = agg.withColumn(
        "TIME_SINCE_LAST_PAY_OVERALL",
        F.row_number().over(
            Window.partitionBy(REF, "_cum_pay_grp").orderBy(DATE)) - 1)
    # When no payment ever observed, set to months on file
    agg = agg.withColumn(
        "TIME_SINCE_LAST_PAY_OVERALL",
        F.when(F.col("_any_pay") == 1, 0)
         .otherwise(F.col("TIME_SINCE_LAST_PAY_OVERALL")))

    keep = [REF, DATE,
            "TIME_SINCE_LAST_PAY_OVERALL",
            "PAY_AFTER_DELINQUENCY_FLAG_CO",
            "TIME_IN_CO_OVERALL", "TIME_IN_CO_DEEP_OVERALL",
            "RECOVERY_PROPENSITY_SCORE_OVERALL"]
    return agg.select(keep).dropDuplicates([REF, DATE])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — CATEGORY 9a: WITHIN-STAGE EXPOSURE DYNAMICS
# ─────────────────────────────────────────────────────────────────────────────

def build_within_stage_exposure_dynamics(
    episode_df: DataFrame,
    history_df: DataFrame,
    spark: SparkSession,
) -> DataFrame:
    """
    Balance and limit dynamics WITHIN each stage episode.

    FIX v2.1: Window functions (first/last) pre-computed as window columns
    BEFORE groupBy, so they are not nested inside agg() — which caused
    AnalysisException in PySpark.

    Features per stage (STAGE in STAGES) — OVERALL dimension:
        WS_DURATION_{STAGE}         months in current episode
        WS_AVG_BAL_SLOPE_{STAGE}    (last_bal - first_bal) / duration
        WS_DIP_COUNT_{STAGE}        months with balance dip (payment proxy)
        WS_AVG_DIP_MAG_{STAGE}      average dip size (THB)
        WS_AVG_UTIL_ENTRY_{STAGE}   utilisation at episode entry
        WS_AVG_LIMIT_CHANGE_{STAGE} avg limit change within episode
    """
    df = _join_dim_flags(history_df, spark)

    df = episode_df.join(
        df.select(REF, DATE, BAL, LIMIT, ACCT_TYPE),
        on=[REF, DATE], how="left")

    w_ep = Window.partitionBy(REF, "episode_id").orderBy(DATE)

    # FIX v2.1: pre-compute first/last as window columns before groupBy
    df = (df
          .withColumn("_prev_bal_ep",  F.lag(BAL).over(w_ep))
          .withColumn("_prev_lim_ep",  F.lag(LIMIT).over(w_ep))
          .withColumn("_first_bal_ep", F.first(BAL).over(w_ep))
          .withColumn("_last_bal_ep",  F.last(BAL).over(
              w_ep.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)))
          .withColumn("_dip",
                      F.when(F.col(BAL) < F.col("_prev_bal_ep"),
                             F.col("_prev_bal_ep") - F.col(BAL)).otherwise(0))
          .withColumn("_entry_util",
                      F.when(F.col("episode_month_number") == 1,
                             _safe_div(F.col(BAL), F.col(LIMIT)))))

    results = []
    for stage in STAGES:
        filt = df.filter(F.col("episode_stage") == stage)

        ep_agg = (filt.groupBy(REF, "episode_id")
                  .agg(
                      F.count("*").alias(f"WS_DURATION_{stage}"),
                      _safe_div(
                          F.last("_last_bal_ep") - F.last("_first_bal_ep"),
                          F.greatest(F.lit(1), F.count("*"))
                      ).alias(f"WS_AVG_BAL_SLOPE_{stage}"),
                      F.sum(F.when(F.col("_dip") > 0, 1).otherwise(0))
                       .alias(f"WS_DIP_COUNT_{stage}"),
                      F.avg(F.when(F.col("_dip") > 0, F.col("_dip")))
                       .alias(f"WS_AVG_DIP_MAG_{stage}"),
                      F.avg("_entry_util").alias(f"WS_AVG_UTIL_ENTRY_{stage}"),
                      F.avg(F.col(LIMIT) - F.col("_prev_lim_ep"))
                       .alias(f"WS_AVG_LIMIT_CHANGE_{stage}"),
                  ))

        # Take the LATEST episode per customer
        w_latest_ep = Window.partitionBy(REF).orderBy(F.col("episode_id").desc())
        per_cust = (ep_agg
                    .withColumn("_rn", F.row_number().over(w_latest_ep))
                    .filter(F.col("_rn") == 1)
                    .drop("_rn", "episode_id"))

        results.append(per_cust)

    out = results[0]
    for r in results[1:]:
        out = out.join(r, on=REF, how="full")
    return out.dropDuplicates([REF])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — CATEGORY 9b: RESTRUCTURING & TDR DYNAMICS
# ─────────────────────────────────────────────────────────────────────────────

def build_restructuring_features(
    account_df: DataFrame,
    history_df: DataFrame,    # FIX v2.1: needed for RESTRUCTURING_SUCCESS_FLAG
    spark: SparkSession,
) -> DataFrame:
    """
    TDR / restructuring history.
    accounttype = "90" → Restructured Debt

    FIX v2.1: Added RESTRUCTURING_SUCCESS_FLAG (DPD < 30 in 6M post-restructuring)

    Features:
        TDR_FLAG_OVERALL                  1 if ever restructured
        TDR_COUNT_OVERALL                 # restructured accounts
        RESTRUCTURING_SUCCESS_FLAG        1 if max DPD < 30 for 6M after restructure
    """
    tdr_accounts = (account_df
                    .filter(F.col(ACCT_TYPE).cast("string") == "90")
                    .groupBy(REF)
                    .agg(
                        F.lit(1).alias("TDR_FLAG_OVERALL"),
                        F.count("*").alias("TDR_COUNT_OVERALL"),
                        F.max("opendate").alias("_last_tdr_open"),
                    ))

    hist_df = _join_dim_flags(history_df, spark)

    # Post-restructure DPD: join history to get DPD in 6M after last TDR open
    success = (hist_df
               .join(tdr_accounts.select(REF, "_last_tdr_open"), on=REF, how="inner")
               .filter(
                   F.to_date(F.col(DATE)).between(
                       F.to_date(F.col("_last_tdr_open")),
                       F.add_months(F.to_date(F.col("_last_tdr_open")), 6)))
               .groupBy(REF)
               .agg(F.max(DPD_COL).alias("_max_dpd_post_tdr"))
               .withColumn("RESTRUCTURING_SUCCESS_FLAG",
                           F.when(F.col("_max_dpd_post_tdr") < 30, 1).otherwise(0))
               .select(REF, "RESTRUCTURING_SUCCESS_FLAG"))

    all_ref = account_df.select(REF).distinct()
    result = (all_ref
              .join(tdr_accounts.drop("_last_tdr_open"), on=REF, how="left")
              .join(success, on=REF, how="left")
              .fillna({"TDR_FLAG_OVERALL": 0,
                       "TDR_COUNT_OVERALL": 0,
                       "RESTRUCTURING_SUCCESS_FLAG": 0}))

    return result.dropDuplicates([REF])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 — CATEGORY 10: TEMPORAL VELOCITY FEATURES  (NEW in v2.1)
# ─────────────────────────────────────────────────────────────────────────────

def build_temporal_velocity_features(episode_df: DataFrame) -> DataFrame:
    """
    Time-sensitive deterioration/improvement velocity patterns.

    FIX v2.1: This category was listed in the module header but had NO
    implementation. Now fully implemented.

    Features:
        ESCALATION_VELOCITY_TO_NPL   months from CURRENT to NPL (shortest path, lifetime)
        CURE_VELOCITY_FROM_NPL       months from NPL to CURRENT (shortest cure, lifetime)
        ESCALATION_VELOCITY_TO_SM    months from CURRENT to SM
        STAGE_STICKINESS_SCORE       avg episode duration across all stages (12M)
        ROLL_RATE_CURRENT_TO_X       % months in CURRENT that transitioned to X
        ROLL_RATE_X_TO_SM            % months in X that transitioned to SM
        ROLL_RATE_SM_TO_NPL          % months in SM that transitioned to NPL
        ROLL_RATE_NPL_TO_CO          % months in NPL that transitioned to CO
        BACKWARD_ROLL_RATE_SM        % months in SM that improved to X or CURRENT
        BACKWARD_ROLL_RATE_NPL       % months in NPL that improved to SM or better
        ESCALATION_ACCELERATION_FLAG 1 if DPD increasing at increasing rate (accel > 0)
        PAYMENT_GAP_MONTHS           months where DPD > 0 (missed payment proxy)
    """
    w_time = Window.partitionBy(REF).orderBy(DATE)
    w_all  = Window.partitionBy(REF).orderBy(DATE).rowsBetween(
                 Window.unboundedPreceding, 0)
    w_12m  = Window.partitionBy(REF).orderBy(DATE).rowsBetween(-11, 0)

    df = episode_df

    # Transition flags
    prev_stage = F.lag(STAGE_COL).over(w_time)
    prev_num   = F.lag("stage_numeric").over(w_time)

    df = (df
          .withColumn("_prev_stage",  prev_stage)
          .withColumn("_prev_num",    prev_num)
          .withColumn("_transitioned", F.when(prev_stage != F.col(STAGE_COL), 1).otherwise(0)))

    # ── Roll rates (forward) ─────────────────────────────────────────────────
    # For each stage, what % of months in that stage transitioned to a worse stage?
    roll_pairs = [
        ("CURRENT", "X",   "ROLL_RATE_CURRENT_TO_X"),
        ("X",       "SM",  "ROLL_RATE_X_TO_SM"),
        ("SM",      "NPL", "ROLL_RATE_SM_TO_NPL"),
        ("NPL",     "CO",  "ROLL_RATE_NPL_TO_CO"),
    ]
    for from_s, to_s, col_name in roll_pairs:
        in_from = F.when(F.col("_prev_stage") == from_s, 1).otherwise(0)
        moved   = F.when(
            (F.col("_prev_stage") == from_s) & (F.col(STAGE_COL) == to_s),
            1).otherwise(0)
        df = df.withColumn(
            col_name,
            _safe_div(F.sum(moved).over(w_all), F.greatest(F.lit(1), F.sum(in_from).over(w_all))))

    # ── Backward roll rates (cure) ────────────────────────────────────────────
    df = df.withColumn(
        "BACKWARD_ROLL_RATE_SM",
        _safe_div(
            F.sum(F.when(
                (F.col("_prev_stage") == "SM") &
                (F.col("stage_numeric") < F.col("_prev_num")), 1).otherwise(0)).over(w_all),
            F.greatest(F.lit(1),
                       F.sum(F.when(F.col("_prev_stage") == "SM", 1).otherwise(0)).over(w_all))))

    df = df.withColumn(
        "BACKWARD_ROLL_RATE_NPL",
        _safe_div(
            F.sum(F.when(
                (F.col("_prev_stage") == "NPL") &
                (F.col("stage_numeric") < F.col("_prev_num")), 1).otherwise(0)).over(w_all),
            F.greatest(F.lit(1),
                       F.sum(F.when(F.col("_prev_stage") == "NPL", 1).otherwise(0)).over(w_all))))

    # ── Escalation velocity to NPL ────────────────────────────────────────────
    # Episodes in CURRENT → find subsequent episode in NPL; compute gap in episode IDs
    w_ep_start = Window.partitionBy(REF, "episode_id")
    df = df.withColumn("_ep_first_row",
                       F.row_number().over(w_ep_start.orderBy(DATE)) == 1)

    # For each episode transition CURRENT→NPL, compute months elapsed
    # Simplified: min episode duration from first CURRENT episode to first NPL episode
    current_entry = (df.filter((F.col("episode_stage") == "CURRENT") &
                               F.col("_ep_first_row"))
                     .select(REF, F.col(DATE).alias("_current_start"),
                             F.col("episode_id").alias("_current_ep")))
    npl_entry = (df.filter((F.col("episode_stage") == "NPL") &
                            F.col("_ep_first_row"))
                 .select(REF, F.col(DATE).alias("_npl_start"),
                         F.col("episode_id").alias("_npl_ep")))

    escalation = (current_entry.join(npl_entry, on=REF, how="inner")
                  .filter(F.col("_npl_ep") > F.col("_current_ep"))
                  .withColumn("_months_gap",
                              F.months_between(
                                  F.col("_npl_start"),
                                  F.col("_current_start")))
                  .groupBy(REF)
                  .agg(F.min("_months_gap").alias("ESCALATION_VELOCITY_TO_NPL")))

    # ── Escalation velocity to SM ─────────────────────────────────────────────
    sm_entry = (df.filter((F.col("episode_stage") == "SM") &
                           F.col("_ep_first_row"))
                .select(REF, F.col(DATE).alias("_sm_start"),
                        F.col("episode_id").alias("_sm_ep")))

    escalation_sm = (current_entry.join(sm_entry, on=REF, how="inner")
                     .filter(F.col("_sm_ep") > F.col("_current_ep"))
                     .withColumn("_months_gap",
                                 F.months_between(
                                     F.col("_sm_start"),
                                     F.col("_current_start")))
                     .groupBy(REF)
                     .agg(F.min("_months_gap").alias("ESCALATION_VELOCITY_TO_SM")))

    # Cure velocity: NPL→CURRENT
    npl_start_df = (df.filter((F.col("episode_stage") == "NPL") &
                               F.col("_ep_first_row"))
                    .select(REF, F.col(DATE).alias("_npl_start2"),
                            F.col("episode_id").alias("_npl_ep2")))
    cur_after = (df.filter((F.col("episode_stage") == "CURRENT") &
                            F.col("_ep_first_row"))
                 .select(REF, F.col(DATE).alias("_cur_after"),
                         F.col("episode_id").alias("_cur_ep2")))

    cure = (npl_start_df.join(cur_after, on=REF, how="inner")
            .filter(F.col("_cur_ep2") > F.col("_npl_ep2"))
            .withColumn("_months_gap",
                        F.months_between(F.col("_cur_after"), F.col("_npl_start2")))
            .groupBy(REF)
            .agg(F.min("_months_gap").alias("CURE_VELOCITY_FROM_NPL")))

    # ── Stage stickiness: avg episode duration in 12M ─────────────────────────
    # Pre-compute ep_length per episode, then take rolling 12M average of
    # episode durations (only at episode-start rows to avoid counting twice)
    _w_ep_full = Window.partitionBy(REF, "episode_id")
    df = df.withColumn("_ep_duration", F.count("*").over(_w_ep_full))
    _ep_dur_at_start = F.when(
        F.col("episode_month_number") == 1, F.col("_ep_duration")
    )
    df = df.withColumn(
        "STAGE_STICKINESS_SCORE",
        _safe_div(
            F.sum(_ep_dur_at_start).over(w_12m),
            F.sum(F.when(F.col("episode_month_number") == 1, 1)
                   .otherwise(0)).over(w_12m),
            F.lit(None)))
    df = df.drop("_ep_duration")

    # ── Escalation acceleration flag ──────────────────────────────────────────
    prev_dpd  = F.lag(DPD_COL).over(w_time)
    prev2_dpd = F.lag(DPD_COL, 2).over(w_time)
    accel = (F.col(DPD_COL) - prev_dpd) - (prev_dpd - prev2_dpd)
    df = df.withColumn(
        "ESCALATION_ACCELERATION_FLAG",
        F.when(accel > 0, 1).otherwise(0))

    # ── Payment gap months ────────────────────────────────────────────────────
    df = df.withColumn(
        "PAYMENT_GAP_MONTHS",
        F.sum(F.when(F.col(DPD_COL) > 0, 1).otherwise(0)).over(w_12m))

    keep_cols = ([REF, DATE]
                 + [p[2] for p in roll_pairs]
                 + ["BACKWARD_ROLL_RATE_SM", "BACKWARD_ROLL_RATE_NPL",
                    "STAGE_STICKINESS_SCORE",
                    "ESCALATION_ACCELERATION_FLAG",
                    "PAYMENT_GAP_MONTHS"])

    base = df.select(keep_cols).dropDuplicates([REF, DATE])

    base = base.join(escalation,    on=REF, how="left")
    base = base.join(escalation_sm, on=REF, how="left")
    base = base.join(cure,          on=REF, how="left")

    return base.dropDuplicates([REF, DATE])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13 — TIME SINCE RESTRUCTURING (fills gap in section numbering)
# ─────────────────────────────────────────────────────────────────────────────

def build_time_since_restructuring(
    account_df: DataFrame,
    as_of_date: str,
) -> DataFrame:
    """
    Compute months since last restructuring (TDR) event.

    This feature was documented in Section 11 (build_restructuring_features)
    but was never computed because that function has no as_of_date parameter.
    Extracted here as a standalone 1-feature builder.

    Features:
        TIME_SINCE_RESTRUCTURING_OVERALL  months from last TDR opendate to as_of_date
    """
    pit = F.to_date(F.lit(as_of_date))

    tdr = (account_df
           .filter(F.col(ACCT_TYPE).cast("string") == "90")
           .groupBy(REF)
           .agg(F.max("opendate").alias("_last_tdr_open")))

    result = (tdr
              .withColumn(
                  "TIME_SINCE_RESTRUCTURING_OVERALL",
                  F.months_between(pit, F.to_date(F.col("_last_tdr_open"))))
              .select(REF, "TIME_SINCE_RESTRUCTURING_OVERALL"))

    return result.dropDuplicates([REF])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14 — PAYMENT EFFORT DYNAMICS PER STAGE
# Balance-dip intensity as payment proxy, weighted by stage severity
# ─────────────────────────────────────────────────────────────────────────────

def build_payment_effort_dynamics(
    episode_df: DataFrame,
    history_df: DataFrame,
) -> DataFrame:
    """
    Compute payment effort signals within each stage episode.

    Recovery insight: A customer who makes partial payments while in NPL
    is fundamentally different from one in NPL with zero balance movement.
    The former has capacity or willingness that recovery can leverage.

    Features per stage label (e.g. _npl, _co, _co_deep …):
        avg_payment_effort_ratio_{stage}     dip_months / episode_months
        avg_payment_effort_intensity_{stage} avg_dip_amount / balance_at_entry
        max_consecutive_no_dip_{stage}       longest streak of NO balance dip
        avg_dip_recency_{stage}              months since last dip at episode exit
        avg_dip_acceleration_{stage}         dips_second_half - dips_first_half
        avg_payment_momentum_{stage}         recency-weighted dip score
        avg_pay_to_balance_ratio_{stage}     |dip| / prior_balance
        avg_payment_volatility_{stage}       stddev of pay_to_balance_ratio
        max_partial_pay_streak_{stage}       longest streak 0 < pay_ratio < 0.30
    """
    monthly = history_df.groupBy(REF, DATE).agg(
        F.sum(BAL).alias("_total_bal"),
    )

    df = (episode_df
          .select(REF, DATE, "episode_id", "episode_stage", "episode_month_number")
          .join(monthly, on=[REF, DATE], how="left"))

    w_ep      = Window.partitionBy(REF, "episode_id").orderBy(DATE)
    w_ep_full = Window.partitionBy(REF, "episode_id")

    df = (df
          .withColumn("_bal_prev", F.lag("_total_bal", 1).over(w_ep))
          .withColumn("_bal_dip",
                      F.when(F.col("_total_bal") < F.col("_bal_prev"), 1).otherwise(0))
          .withColumn("_dip_amount",
                      F.when(F.col("_bal_dip") == 1,
                             F.col("_bal_prev") - F.col("_total_bal"))
                       .otherwise(F.lit(0.0))))

    ep_len    = F.count("*").over(w_ep_full)
    half_pt   = ep_len / 2

    df = (df
          .withColumn("_ep_len",   ep_len)
          .withColumn("_in_first",  F.when(F.col("episode_month_number") <= half_pt, 1).otherwise(0))
          .withColumn("_in_second", F.when(F.col("episode_month_number") >  half_pt, 1).otherwise(0)))

    # Consecutive no-dip streak via island detection
    df = (df
          .withColumn("_no_dip",  F.when(F.col("_bal_dip") == 0, 1).otherwise(0))
          .withColumn("_dip_grp", F.sum("_bal_dip").over(w_ep))
          .withColumn("_no_dip_run",
                      F.row_number().over(
                          Window.partitionBy(REF, "episode_id", "_dip_grp")
                                .orderBy(DATE)) * F.col("_no_dip")))

    # Recency weight: recent dips score higher
    df = df.withColumn(
        "_dip_recency_wt",
        F.when(F.col("_bal_dip") == 1,
               F.col("episode_month_number").cast("double") / F.col("_ep_len"))
         .otherwise(F.lit(0.0)))

    # Months since last dip within episode
    df = (df
          .withColumn("_last_dip_mn",
                      F.when(F.col("_bal_dip") == 1, F.col("episode_month_number")))
          .withColumn("_last_dip_mn_filled",
                      F.last("_last_dip_mn", ignorenulls=True).over(w_ep))
          .withColumn("_dip_rec_months",
                      F.col("episode_month_number") -
                      F.coalesce(F.col("_last_dip_mn_filled"), F.lit(0))))

    # Pay-to-balance ratio + partial-pay streak (0 < ratio < 0.30)
    df = (df
          .withColumn("_pay_ratio",
                      F.when(F.col("_bal_dip") == 1,
                             _safe_div(F.col("_dip_amount"), F.col("_bal_prev"), F.lit(0.0)))
                       .otherwise(F.lit(0.0)))
          .withColumn("_is_partial",
                      F.when((F.col("_pay_ratio") > 0) & (F.col("_pay_ratio") < 0.30), 1)
                       .otherwise(0))
          .withColumn("_partial_brk", F.when(F.col("_is_partial") == 0, 1).otherwise(0))
          .withColumn("_partial_grp", F.sum("_partial_brk").over(w_ep))
          .withColumn("_partial_run",
                      F.row_number().over(
                          Window.partitionBy(REF, "episode_id", "_partial_grp")
                                .orderBy(DATE)) * F.col("_is_partial")))

    # Episode-level aggregation
    ep_agg = (df.groupBy(REF, "episode_id", "episode_stage")
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
                  F.avg(F.when(F.col("_bal_dip") == 1, F.col("_pay_ratio"))).alias("_avg_pr"),
                  F.stddev(F.when(F.col("_bal_dip") == 1, F.col("_pay_ratio"))).alias("_pay_vol"),
                  F.max("_partial_run").alias("_partial_streak"),
              )
              .withColumn("_effort_ratio",
                          _safe_div(F.col("_dip_cnt"), F.col("_ep_len"), F.lit(0.0)))
              .withColumn("_effort_intensity",
                          _safe_div(F.col("_avg_dip_amt"), F.col("_bal_entry"), F.lit(0.0)))
              .withColumn("_dip_accel", F.col("_dips_second") - F.col("_dips_first"))
              .withColumn("_momentum",
                          _safe_div(F.col("_momentum_raw"), F.col("_dip_cnt"), F.lit(0.0))))

    # Stage-level aggregation → pivot by stage
    stage_agg = (ep_agg.groupBy(REF, "episode_stage")
                 .agg(
                     F.avg("_effort_ratio").alias("avg_payment_effort_ratio"),
                     F.avg("_effort_intensity").alias("avg_payment_effort_intensity"),
                     F.max("_max_no_dip").alias("max_consecutive_no_dip"),
                     F.avg("_dip_rec_exit").alias("avg_dip_recency"),
                     F.avg("_dip_accel").alias("avg_dip_acceleration"),
                     F.avg("_momentum").alias("avg_payment_momentum"),
                     F.avg("_avg_pr").alias("avg_pay_to_balance_ratio"),
                     F.avg("_pay_vol").alias("avg_payment_volatility"),
                     F.max("_partial_streak").alias("max_partial_pay_streak"),
                 ))

    metric_cols = [
        "avg_payment_effort_ratio", "avg_payment_effort_intensity",
        "max_consecutive_no_dip", "avg_dip_recency", "avg_dip_acceleration",
        "avg_payment_momentum", "avg_pay_to_balance_ratio",
        "avg_payment_volatility", "max_partial_pay_streak",
    ]

    stage_dfs = []
    for stage, label in STAGE_LABELS.items():
        s = stage_agg.filter(F.col("episode_stage") == stage).select(
            REF, *[F.col(c).alias(f"{c}_{label}") for c in metric_cols])
        stage_dfs.append(s)

    result = stage_dfs[0]
    for s in stage_dfs[1:]:
        result = result.join(s, on=REF, how="full")
    return result.dropDuplicates([REF])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 15 — STAGE VELOCITY & ACCELERATION
# Speed of deterioration and cure across the delinquency lifecycle
# ─────────────────────────────────────────────────────────────────────────────

def build_stage_velocity_features(episode_df: DataFrame) -> DataFrame:
    """
    Stage-transition velocity and acceleration across customer lifecycle.

    Recovery insight: Two customers both at NPL — one took 12 months to get
    there (slow deterioration, structural), the other 3 months (shock). Velocity
    tells you the TYPE of default.

    Features:
        months_current_to_sm       months from first CURRENT to first SM entry
        months_current_to_npl      months from first CURRENT to first NPL entry
        months_current_to_co       months from first CURRENT to first CO entry
        months_current_to_co_deep  months from first CURRENT to first CO_DEEP entry
        months_sm_to_co            months from first SM to first CO entry
        max_stage_ever_reached     highest stage ordinal (0–5)
        months_at_worst_stage      total months in worst stage
        deterioration_velocity     max_stage_ord / months_to_reach_it
        cure_velocity              avg (stages retreated / months in source stage)
        stage_churn_rate           episode count / months observed
        time_in_current_stage_months months in most recent stage
        fastest_deterioration_speed min months between consecutive worsening transitions
        npl_stickiness_index       avg duration of NPL episodes
        co_stickiness_index        avg duration of CO episodes
        co_deep_stickiness_index   avg duration of CO_DEEP episodes
        stage_reversal_rate        cure transitions / total transitions
        num_transitions_sm_to_npl  count SM→NPL (key regulatory gate)
        num_transitions_npl_to_current count NPL→CURRENT (full cure — rare)
        num_transitions_co_to_npl  count CO→NPL (partial cure)
        stages_traversed_12m       distinct stages in most recent 12 months
    """
    # First entry per stage
    first_entries = (episode_df
                     .filter(F.col("episode_month_number") == 1)
                     .groupBy(REF, "episode_stage")
                     .agg(F.min(DATE).alias("first_entry_month"),
                          F.count("*").alias("ep_count_in_stage")))

    # Pivot to wide (one row per customer)
    pivot = (first_entries
             .groupBy(REF)
             .pivot("episode_stage", STAGES)
             .agg(F.min("first_entry_month")))
    for stage in STAGES:
        pivot = pivot.withColumnRenamed(stage, f"_fe_{stage}")

    # Velocity metrics from stage entry dates
    def _months(a_col, b_col):
        return F.when(
            F.col(a_col).isNotNull() & F.col(b_col).isNotNull(),
            F.months_between(F.col(b_col), F.col(a_col)))

    result = (pivot
              .withColumn("months_current_to_sm",       _months("_fe_CURRENT", "_fe_SM"))
              .withColumn("months_current_to_npl",      _months("_fe_CURRENT", "_fe_NPL"))
              .withColumn("months_current_to_co",       _months("_fe_CURRENT", "_fe_CO"))
              .withColumn("months_current_to_co_deep",  _months("_fe_CURRENT", "_fe_CO_DEEP"))
              .withColumn("months_sm_to_co",            _months("_fe_SM",      "_fe_CO")))

    # Max stage ever reached
    max_stage = (episode_df
                 .withColumn("_ord", _stage_ordinal("episode_stage"))
                 .groupBy(REF)
                 .agg(F.max("_ord").alias("max_stage_ever_reached")))
    result = result.join(max_stage, on=REF, how="left")

    # Deterioration velocity
    result = (result
              .withColumn("_months_to_max",
                          F.when(F.col("max_stage_ever_reached") >= 5,
                                 F.col("months_current_to_co_deep"))
                           .when(F.col("max_stage_ever_reached") >= 4,
                                 F.col("months_current_to_co"))
                           .when(F.col("max_stage_ever_reached") >= 3,
                                 F.col("months_current_to_npl"))
                           .when(F.col("max_stage_ever_reached") >= 2,
                                 F.col("months_current_to_sm")))
              .withColumn("deterioration_velocity",
                          F.when(F.col("_months_to_max").isNotNull() &
                                 (F.col("_months_to_max") > 0),
                                 F.col("max_stage_ever_reached") /
                                 F.col("_months_to_max")))
              .drop("_months_to_max"))

    # Stage churn + total months
    churn = (episode_df
             .filter(F.col("episode_month_number") == 1)
             .groupBy(REF)
             .agg(F.count("episode_id").alias("_total_eps"),
                  F.countDistinct("episode_stage").alias("distinct_stages_ever"))
             .join(
                 episode_df.groupBy(REF).agg(
                     F.countDistinct(DATE).alias("_total_months")),
                 on=REF, how="left")
             .withColumn("stage_churn_rate",
                         _safe_div(F.col("_total_eps"), F.col("_total_months"), F.lit(0.0))))
    result = result.join(churn.drop("_total_eps", "_total_months"), on=REF, how="left")

    # Time in current (most recent) stage
    w_latest = Window.partitionBy(REF).orderBy(F.col(DATE).desc())
    latest_ep = (episode_df
                 .withColumn("_rn", F.row_number().over(w_latest))
                 .filter(F.col("_rn") == 1)
                 .select(REF,
                         F.col("episode_stage").alias("current_stage"),
                         F.col("episode_month_number").alias("time_in_current_stage_months")))
    result = result.join(latest_ep, on=REF, how="left")

    # Stickiness indexes (avg episode duration per stage)
    w_ep_full = Window.partitionBy(REF, "episode_id")
    ep_dur = (episode_df
              .withColumn("_ep_len", F.count("*").over(w_ep_full))
              .filter(F.col("episode_month_number") == 1))

    for stage, label in [("NPL", "npl"), ("CO", "co"), ("CO_DEEP", "co_deep")]:
        stick = (ep_dur
                 .filter(F.col("episode_stage") == stage)
                 .groupBy(REF)
                 .agg(F.avg("_ep_len").alias(f"{label}_stickiness_index")))
        result = result.join(stick, on=REF, how="left")

    # Episode-level transitions
    w_ep_ord = Window.partitionBy(REF).orderBy("episode_id")
    ep_trans = (episode_df
                .filter(F.col("episode_month_number") == 1)
                .select(REF, "episode_id", "episode_stage", DATE)
                .withColumn("next_stage", F.lead("episode_stage", 1).over(w_ep_ord))
                .withColumn("next_date",  F.lead(DATE, 1).over(w_ep_ord))
                .withColumn("_ord",       _stage_ordinal("episode_stage"))
                .withColumn("_next_ord",  _stage_ordinal("next_stage"))
                .filter(F.col("next_stage").isNotNull()))

    reversal = (ep_trans
                .groupBy(REF)
                .agg(
                    F.count("*").alias("_total_trans"),
                    F.sum(F.when(F.col("_next_ord") < F.col("_ord"), 1).otherwise(0))
                     .alias("_cure_trans"),
                    F.sum(F.when((F.col("episode_stage") == "SM") &
                                 (F.col("next_stage") == "NPL"), 1).otherwise(0))
                     .alias("num_transitions_sm_to_npl"),
                    F.sum(F.when((F.col("episode_stage") == "NPL") &
                                 (F.col("next_stage") == "CURRENT"), 1).otherwise(0))
                     .alias("num_transitions_npl_to_current"),
                    F.sum(F.when((F.col("episode_stage") == "CO") &
                                 (F.col("next_stage") == "NPL"), 1).otherwise(0))
                     .alias("num_transitions_co_to_npl"),
                )
                .withColumn("stage_reversal_rate",
                            _safe_div(F.col("_cure_trans"),
                                      F.col("_total_trans"), F.lit(0.0)))
                .drop("_total_trans", "_cure_trans"))
    result = result.join(reversal, on=REF, how="left")

    # Months at worst stage
    ep_with_ord = episode_df.withColumn("_ord", _stage_ordinal("episode_stage"))
    worst_ord_per_cust = ep_with_ord.groupBy(REF).agg(
        F.max("_ord").alias("_worst_ord"))
    months_at_worst = (ep_with_ord
                       .join(worst_ord_per_cust, on=REF, how="inner")
                       .filter(F.col("_ord") == F.col("_worst_ord"))
                       .groupBy(REF)
                       .agg(F.countDistinct(DATE).alias("months_at_worst_stage")))
    result = result.join(months_at_worst, on=REF, how="left")

    # Stages traversed in last 12 months
    cust_max_dt = episode_df.groupBy(REF).agg(F.max(DATE).alias("_max_dt"))
    stages_12m = (episode_df
                  .join(cust_max_dt, on=REF, how="inner")
                  .filter(F.months_between(F.col("_max_dt"), F.col(DATE)) <= 12)
                  .groupBy(REF)
                  .agg(F.countDistinct("episode_stage").alias("stages_traversed_12m")))
    result = result.join(stages_12m, on=REF, how="left")

    # Fastest deterioration speed + cure velocity
    ep_trans_with_dur = (ep_trans
                         .withColumn("_next_ep_dt", F.lead(DATE, 1).over(w_ep_ord))
                         .withColumn("_ep_dur",
                                     F.months_between(F.col("_next_ep_dt"), F.col(DATE))))

    fastest = (ep_trans_with_dur
               .filter((F.col("_next_ord") > F.col("_ord")) &
                       F.col("_ep_dur").isNotNull())
               .groupBy(REF)
               .agg(F.min("_ep_dur").alias("fastest_deterioration_speed")))
    result = result.join(fastest, on=REF, how="left")

    cure_vel = (ep_trans_with_dur
                .filter((F.col("_next_ord") < F.col("_ord")) &
                        F.col("_ep_dur").isNotNull() & (F.col("_ep_dur") > 0))
                .withColumn("_stages_ret", F.col("_ord") - F.col("_next_ord"))
                .groupBy(REF)
                .agg(F.avg(F.col("_stages_ret") / F.col("_ep_dur"))
                      .alias("cure_velocity")))
    result = result.join(cure_vel, on=REF, how="left")

    # Drop first-entry pivot columns
    for stage in STAGES:
        result = result.drop(f"_fe_{stage}")

    return result.dropDuplicates([REF])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 16 — CURE & RE-DEFAULT DYNAMICS
# Did the customer cure then re-default? Critical for recovery strategy.
# ─────────────────────────────────────────────────────────────────────────────

def build_cure_redefault_dynamics(episode_df: DataFrame) -> DataFrame:
    """
    Detect cure-then-re-default cycles from episode transitions.

    Cure event:   episode transition from SM/NPL/CO/CO_DEEP → CURRENT/X
    Re-default:   after a cure, re-enters SM+ within 24 months

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
    w = Window.partitionBy(REF).orderBy("episode_id")

    eps = (episode_df
           .filter(F.col("episode_month_number") == 1)
           .select(REF, "episode_id", "episode_stage", DATE)
           .withColumn("_ord",      _stage_ordinal("episode_stage"))
           .withColumn("_prev_ord", F.lag("_ord", 1).over(w))
           .withColumn("_next_ord", F.lead("_ord", 1).over(w))
           .withColumn("_next_dt",  F.lead(DATE, 1).over(w)))

    # Cure destination: ord <= 1 (CURRENT/X) after ord >= 2 (SM+)
    eps = (eps
           .withColumn("_is_cure_dest",
                       F.when((F.col("_ord") <= 1) &
                              (F.col("_prev_ord") >= 2), 1).otherwise(0))
           .withColumn("_cure_src_ord",
                       F.when(F.col("_is_cure_dest") == 1, F.col("_prev_ord")))
           # Re-default: after cure destination, next stage goes back to SM+
           .withColumn("_is_redefault",
                       F.when((F.col("_is_cure_dest") == 1) &
                              (F.col("_next_ord") >= 2), 1).otherwise(0))
           .withColumn("_months_to_redef",
                       F.when(F.col("_is_redefault") == 1,
                              F.months_between(F.col("_next_dt"), F.col(DATE))))
           .withColumn("_cure_hold_months",
                       F.when(F.col("_is_cure_dest") == 1,
                              F.months_between(F.col("_next_dt"), F.col(DATE)))))

    result = (eps.groupBy(REF)
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
                          F.when((F.col("cure_count_lifetime") > 0) &
                                 ((F.col("re_default_count_lifetime") == 0) |
                                  (F.col("min_months_to_re_default") >= 12)), 1)
                           .otherwise(0))
              .withColumn("cure_to_redefault_ratio",
                          _safe_div(F.col("re_default_count_lifetime"),
                                    F.col("cure_count_lifetime"), F.lit(0.0))))

    return result.dropDuplicates([REF])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 17 — BALANCE RECOVERY POTENTIAL
# Residual capacity signals for recovery amount estimation
# ─────────────────────────────────────────────────────────────────────────────

def build_balance_recovery_potential(
    episode_df: DataFrame,
    history_df: DataFrame,
) -> DataFrame:
    """
    Balance and capacity signals that inform recovery amount potential.

    Recovery insight: A customer whose total bureau balance is declining while
    defaulting at CardX has residual capacity. One whose balance is increasing
    everywhere is overleveraged.

    Features:
        bureau_balance_trend_6m         6M slope of total bureau balance (THB/month)
        bureau_balance_trend_12m        12M slope of total bureau balance
        bureau_limit_utilization_latest util = balance / limit at latest snapshot
        available_limit_latest          total_limit - total_balance at latest
        balance_concentration_hhi       HHI across lenders (1 = single lender)
        deleveraging_rate_12m           (bal_12m_ago - bal_now) / bal_12m_ago
        balance_at_worst_vs_entry       balance at worst stage / balance at first CURRENT entry
        top_lender_balance_share        largest lender balance / total balance
        limit_fragmentation_score       largest single lender limit / total limits
        num_active_secured_trades       count secured tradelines at latest snapshot
        num_active_unsecured_trades     count unsecured tradelines at latest snapshot
    """
    monthly = history_df.groupBy(REF, DATE).agg(
        F.sum(BAL).alias("_tot_bal"),
        F.sum(LIMIT).alias("_tot_lim"),
    )
    lender_bal = history_df.groupBy(REF, DATE, LENDER).agg(
        F.sum(BAL).alias("_lend_bal"))
    lender_lim = history_df.groupBy(REF, DATE, LENDER).agg(
        F.sum(LIMIT).alias("_lend_lim"))

    w_latest = Window.partitionBy(REF).orderBy(F.col(DATE).desc())

    latest_month = (monthly
                    .withColumn("_rn", F.row_number().over(w_latest))
                    .filter(F.col("_rn") == 1)
                    .select(REF, DATE))

    # HHI = sum(share^2)
    lend_latest = lender_bal.join(latest_month.withColumnRenamed(DATE, "_ld"),
                                  on=REF, how="inner").filter(
                                      F.col(DATE) == F.col("_ld")).drop("_ld")
    tot_latest  = lend_latest.groupBy(REF).agg(F.sum("_lend_bal").alias("_tb"))
    hhi = (lend_latest
           .join(tot_latest, on=REF, how="left")
           .withColumn("_share_sq",
                       F.pow(_safe_div(F.col("_lend_bal"), F.col("_tb"), F.lit(0.0)), 2))
           .groupBy(REF)
           .agg(F.sum("_share_sq").alias("balance_concentration_hhi"),
                F.max("_lend_bal").alias("_top_lend_bal"),
                F.sum("_lend_bal").alias("_tot_for_share"))
           .withColumn("top_lender_balance_share",
                       _safe_div(F.col("_top_lend_bal"),
                                 F.col("_tot_for_share"), F.lit(0.0)))
           .drop("_top_lend_bal", "_tot_for_share"))

    # Limit fragmentation
    lim_latest = lender_lim.join(latest_month.withColumnRenamed(DATE, "_ld"),
                                  on=REF, how="inner").filter(
                                      F.col(DATE) == F.col("_ld")).drop("_ld")
    frag = (lim_latest
            .groupBy(REF)
            .agg(F.max("_lend_lim").alias("_max_lim"),
                 F.sum("_lend_lim").alias("_tot_lim"))
            .withColumn("limit_fragmentation_score",
                        _safe_div(F.col("_max_lim"), F.col("_tot_lim"), F.lit(0.0)))
            .drop("_max_lim", "_tot_lim"))

    # Active trade counts by secured/unsecured (keyword match on LENDER)
    _sec_kw = ["MORTGAGE","GH BANK","HOUSING","TOYOTA","ISUZU",
               "HONDA","NISSAN","AYUDHYA CAPITAL","ORIX","SRISAWAD","LEASING"]
    _upper = F.upper(F.col(LENDER))
    _is_sec = F.lit(False)
    for kw in _sec_kw:
        _is_sec = _is_sec | _upper.contains(kw)

    trade_cnts = (lim_latest
                  .withColumn("_is_sec", F.when(_is_sec, 1).otherwise(0))
                  .groupBy(REF)
                  .agg(
                      F.sum(F.when(F.col("_is_sec") == 1, 1).otherwise(0))
                       .alias("num_active_secured_trades"),
                      F.sum(F.when(F.col("_is_sec") == 0, 1).otherwise(0))
                       .alias("num_active_unsecured_trades"),
                  ))

    # Balance trends
    w_cust = Window.partitionBy(REF).orderBy(DATE)
    w6     = w_cust.rowsBetween(-5, 0)
    w12    = w_cust.rowsBetween(-11, 0)

    trends = (monthly
              .withColumn("_bal_6m_ago",  F.first("_tot_bal").over(w6))
              .withColumn("_bal_12m_ago", F.first("_tot_bal").over(w12))
              .withColumn("bureau_balance_trend_6m",
                          _safe_div(F.col("_tot_bal") - F.col("_bal_6m_ago"),
                                    F.lit(6.0), F.lit(0.0)))
              .withColumn("bureau_balance_trend_12m",
                          _safe_div(F.col("_tot_bal") - F.col("_bal_12m_ago"),
                                    F.lit(12.0), F.lit(0.0)))
              .withColumn("bureau_limit_utilization_latest",
                          _safe_div(F.col("_tot_bal"), F.col("_tot_lim")))
              .withColumn("available_limit_latest",
                          F.col("_tot_lim") - F.col("_tot_bal"))
              .withColumn("deleveraging_rate_12m",
                          _safe_div(F.col("_bal_12m_ago") - F.col("_tot_bal"),
                                    F.col("_bal_12m_ago"), F.lit(0.0)))
              .withColumn("_rn", F.row_number().over(w_latest))
              .filter(F.col("_rn") == 1)
              .select(REF, "bureau_balance_trend_6m", "bureau_balance_trend_12m",
                      "bureau_limit_utilization_latest", "available_limit_latest",
                      "deleveraging_rate_12m"))

    result = (trends
              .join(hhi,        on=REF, how="left")
              .join(frag,       on=REF, how="left")
              .join(trade_cnts, on=REF, how="left"))

    # Balance at worst stage vs first CURRENT entry
    ep_bal = episode_df.join(
        monthly.select(REF, DATE, "_tot_bal"), on=[REF, DATE], how="left"
    ).withColumn("_ord", _stage_ordinal("episode_stage"))

    first_cur = (ep_bal
                 .filter(F.col("episode_stage") == "CURRENT")
                 .groupBy(REF)
                 .agg(F.first("_tot_bal", ignorenulls=True).alias("_bal_first_cur")))

    worst_ord_df = ep_bal.groupBy(REF).agg(F.max("_ord").alias("_worst_ord"))
    bal_worst = (ep_bal
                 .join(worst_ord_df, on=REF, how="inner")
                 .filter(F.col("_ord") == F.col("_worst_ord"))
                 .withColumn("_rn2",
                             F.row_number().over(
                                 Window.partitionBy(REF).orderBy(DATE)))
                 .filter(F.col("_rn2") == 1)
                 .select(REF, F.col("_tot_bal").alias("_bal_worst")))

    bal_ratio = (first_cur
                 .join(bal_worst, on=REF, how="full")
                 .withColumn("balance_at_worst_vs_entry",
                             _safe_div(F.col("_bal_worst"), F.col("_bal_first_cur")))
                 .select(REF, "balance_at_worst_vs_entry"))

    result = result.join(bal_ratio, on=REF, how="left")
    return result.dropDuplicates([REF])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 18 — ENGAGEMENT RECENCY & DECAY SIGNALS
# How recently did the customer show any sign of life?
# ─────────────────────────────────────────────────────────────────────────────

def build_engagement_recency_signals(
    episode_df: DataFrame,
    history_df: DataFrame,
) -> DataFrame:
    """
    Recency and decay signals across the bureau profile.

    Recovery insight: Time since last ANY positive signal (balance dip,
    new tradeline, stage change) is the strongest predictor of contactability.
    A customer with 9 months of silence across ALL lenders is likely unreachable.

    Features:
        months_since_last_balance_dip    months since last balance decrease (any tradeline)
        months_since_last_new_tradeline  months since last new account
        months_since_last_stage_change   months since last stage transition
        balance_activity_score           proportion of last 12M with any balance change
        dormancy_flag                    1 if zero balance changes in last 6 months
        total_bureau_silence_months      consecutive zero-activity months at end of timeline
    """
    monthly = history_df.groupBy(REF, DATE).agg(
        F.sum(BAL).alias("_tot_bal"),
        F.countDistinct(LENDER).alias("_lend_cnt"),
    )

    w_cust   = Window.partitionBy(REF).orderBy(DATE)
    w_latest = Window.partitionBy(REF).orderBy(F.col(DATE).desc())

    monthly = (monthly
               .withColumn("_bal_prev",  F.lag("_tot_bal", 1).over(w_cust))
               .withColumn("_bal_chgd",
                           F.when(F.col("_tot_bal") != F.col("_bal_prev"), 1).otherwise(0))
               .withColumn("_bal_dip",
                           F.when(F.col("_tot_bal") < F.col("_bal_prev"), 1).otherwise(0))
               .withColumn("_lend_prev", F.lag("_lend_cnt", 1).over(w_cust))
               .withColumn("_new_tl",
                           F.when(F.col("_lend_cnt") > F.col("_lend_prev"), 1).otherwise(0)))

    cust_latest = (monthly
                   .withColumn("_rn", F.row_number().over(w_latest))
                   .filter(F.col("_rn") == 1)
                   .select(REF, F.col(DATE).alias("_latest_dt")))

    last_dip = (monthly.filter(F.col("_bal_dip") == 1)
                .groupBy(REF).agg(F.max(DATE).alias("_last_dip_dt")))
    last_tl  = (monthly.filter(F.col("_new_tl") == 1)
                .groupBy(REF).agg(F.max(DATE).alias("_last_tl_dt")))
    last_stg = (episode_df.filter(F.col("episode_month_number") == 1)
                .groupBy(REF).agg(F.max(DATE).alias("_last_stg_dt")))

    result = (cust_latest
              .join(last_dip, on=REF, how="left")
              .join(last_tl,  on=REF, how="left")
              .join(last_stg, on=REF, how="left")
              .withColumn("months_since_last_balance_dip",
                          F.when(F.col("_last_dip_dt").isNotNull(),
                                 F.months_between(F.col("_latest_dt"), F.col("_last_dip_dt")))
                           .otherwise(F.lit(999)))
              .withColumn("months_since_last_new_tradeline",
                          F.when(F.col("_last_tl_dt").isNotNull(),
                                 F.months_between(F.col("_latest_dt"), F.col("_last_tl_dt")))
                           .otherwise(F.lit(999)))
              .withColumn("months_since_last_stage_change",
                          F.when(F.col("_last_stg_dt").isNotNull(),
                                 F.months_between(F.col("_latest_dt"), F.col("_last_stg_dt")))
                           .otherwise(F.lit(999))))

    # Activity score: proportion of last 12M with any balance change
    w12 = w_cust.rowsBetween(-11, 0)
    w6  = w_cust.rowsBetween(-5,  0)
    act_12m = (monthly
               .withColumn("_act_12m", F.sum("_bal_chgd").over(w12))
               .withColumn("_cnt_12m", F.count("_bal_chgd").over(w12))
               .withColumn("_rn", F.row_number().over(w_latest))
               .filter(F.col("_rn") == 1)
               .select(REF,
                       _safe_div(F.col("_act_12m"), F.col("_cnt_12m"),
                                 F.lit(0.0)).alias("balance_activity_score")))

    dormancy = (monthly
                .withColumn("_chg_6m", F.sum("_bal_chgd").over(w6))
                .withColumn("_rn", F.row_number().over(w_latest))
                .filter(F.col("_rn") == 1)
                .select(REF,
                        F.when(F.col("_chg_6m") == 0, 1).otherwise(0).alias("dormancy_flag")))

    # Bureau silence: consecutive no-activity months at end of timeline
    silence_df = (monthly
                  .withColumn("_act_cumrev",
                              F.sum("_bal_chgd").over(
                                  Window.partitionBy(REF)
                                        .orderBy(F.col(DATE).desc())
                                        .rowsBetween(Window.unboundedPreceding, 0))))
    silence_cnt = (silence_df.filter(F.col("_act_cumrev") == 0)
                   .groupBy(REF)
                   .agg(F.count("*").alias("total_bureau_silence_months")))

    result = (result
              .join(act_12m,    on=REF, how="left")
              .join(dormancy,   on=REF, how="left")
              .join(silence_cnt, on=REF, how="left")
              .withColumn("total_bureau_silence_months",
                          F.coalesce(F.col("total_bureau_silence_months"), F.lit(0)))
              .drop("_latest_dt", "_last_dip_dt", "_last_tl_dt", "_last_stg_dt"))

    return result.dropDuplicates([REF])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 19 — VINTAGE & SEASONING DYNAMICS
# Where in the lifecycle was the borrower when things went wrong?
# ─────────────────────────────────────────────────────────────────────────────

def build_vintage_seasoning_features(episode_df: DataFrame) -> DataFrame:
    """
    Lifecycle timing features that position default within account age.

    Recovery insight: A customer who defaulted 3 months after first bureau
    appearance is likely a bust-out (low recovery probability). One who was
    clean for 36 months before first SM entry had a life event — higher
    recovery probability because they demonstrated long-term repayment.

    Features:
        months_observed_total              total months of bureau history
        account_age_at_first_default       months from first obs to first SM+ entry
        time_in_good_standing_pre_default  S0 months before first SM+ entry
        cumulative_current_ratio           months in CURRENT / total months
        early_life_default_flag            1 if first SM+ within 6 months of first obs
        seasoning_at_worst_stage           months from first obs to worst stage entry
        good_standing_to_default_ratio     good_standing_months / default_months
    """
    obs = episode_df.groupBy(REF).agg(
        F.min(DATE).alias("_first_obs"),
        F.countDistinct(DATE).alias("months_observed_total"),
    )
    months_cur = (episode_df
                  .filter(F.col("episode_stage") == "CURRENT")
                  .groupBy(REF)
                  .agg(F.countDistinct(DATE).alias("_months_cur")))

    default_stages = ["SM", "NPL", "CO", "CO_DEEP"]
    first_def = (episode_df
                 .filter(F.col("episode_stage").isin(default_stages) &
                         (F.col("episode_month_number") == 1))
                 .groupBy(REF)
                 .agg(F.min(DATE).alias("_first_def")))

    # CURRENT months strictly before first default
    cur_pre_def = (episode_df
                   .filter(F.col("episode_stage") == "CURRENT")
                   .join(first_def, on=REF, how="left")
                   .filter(F.col("_first_def").isNull() |
                           (F.col(DATE) < F.col("_first_def")))
                   .groupBy(REF)
                   .agg(F.countDistinct(DATE).alias("_cur_pre_def")))

    # Worst stage entry
    ep_ord = (episode_df
              .filter(F.col("episode_month_number") == 1)
              .withColumn("_ord", _stage_ordinal("episode_stage")))
    worst_per = ep_ord.groupBy(REF).agg(F.max("_ord").alias("_worst_ord"))
    worst_entry = (ep_ord
                   .join(worst_per, on=REF, how="inner")
                   .filter(F.col("_ord") == F.col("_worst_ord"))
                   .groupBy(REF)
                   .agg(F.min(DATE).alias("_worst_entry")))

    result = (obs
              .join(months_cur,   on=REF, how="left")
              .join(first_def,    on=REF, how="left")
              .join(cur_pre_def,  on=REF, how="left")
              .join(worst_entry,  on=REF, how="left")
              .withColumn("_months_cur",     F.coalesce(F.col("_months_cur"), F.lit(0)))
              .withColumn("_cur_pre_def",    F.coalesce(F.col("_cur_pre_def"), F.lit(0)))
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
                    "_worst_entry", "_worst_ord", "_default_months"))

    return result.dropDuplicates([REF])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 20 — MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def run_stage_dynamics(
    spark: SparkSession,
    state_df: DataFrame,
    history_df: DataFrame,
    account_df: DataFrame,
    as_of_date: str,                            # "YYYY-MM-DD" — PIT anchor
    enquiry_df: Optional[DataFrame] = None,
) -> DataFrame:
    """
    Run complete stage dynamics pipeline (~250 features).

    Execution order:
         0. enrich_account_dimensions              — NCB ACCOUNTTYPE dimension flags
         1. build_dpd_stage                        — CURRENT/X/SM/NPL/CO/CO_DEEP assignment
         2. build_stage_episodes                   — episode ID + month-in-episode
         3. build_stage_transition_features        — Cat 1
         4. build_payment_behavior_features        — Cat 2
         5. build_exposure_utilisation_features    — Cat 3
         6. build_enquiry_features                 — Cat 4 (optional)
         7. build_account_closure_features         — Cat 5
         8. build_delinquency_severity_features    — Cat 6
         9. build_cross_dimension_features         — Cat 7
        10. build_recovery_indicators              — Cat 8
        11. build_within_stage_exposure_dynamics   — Cat 9a
        12. build_restructuring_features           — Cat 9b
        13. build_temporal_velocity_features       — Cat 10
        14. build_payment_effort_dynamics          — Cat 11 (NEW)
        15. build_stage_velocity_features          — Cat 12 (NEW)
        16. build_cure_redefault_dynamics          — Cat 13 (NEW)
        17. build_balance_recovery_potential       — Cat 14 (NEW)
        18. build_engagement_recency_signals       — Cat 15 (NEW)
        19. build_vintage_seasoning_features       — Cat 16 (NEW)

    Returns: Wide DataFrame keyed on (ref_no, asofdate).
    """
    sep = "=" * 70
    print(f"\n{sep}\nBureau Stage Dynamics Engine v2.3\n{sep}")

    print("[0] Enriching account dimensions...")
    account_enriched = enrich_account_dimensions(account_df)

    print("[1] Assigning DPD stages...")
    staged_df = build_dpd_stage(state_df)

    print("[2] Building stage episodes...")
    episode_df = build_stage_episodes(staged_df)

    print("[3] Cat 1 — Stage transition features...")
    trans_feats    = build_stage_transition_features(episode_df)

    print("[4] Cat 2 — Payment behavior (7 dims)...")
    pay_feats      = build_payment_behavior_features(history_df, spark)

    print("[5] Cat 3 — Exposure & utilisation (7 dims)...")
    exp_feats      = build_exposure_utilisation_features(history_df, spark)

    if enquiry_df is not None:
        print("[6] Cat 4 — Enquiry features...")
        enq_feats  = build_enquiry_features(enquiry_df)
    else:
        print("[6] Cat 4 — Enquiry SKIPPED (no enquiry_df)")
        enq_feats  = None

    print(f"[7] Cat 5 — Account closure (as_of={as_of_date})...")
    closure_feats  = build_account_closure_features(account_enriched, as_of_date)

    print("[8] Cat 6 — Delinquency severity (7 dims)...")
    severity_feats = build_delinquency_severity_features(history_df, spark)

    print("[9] Cat 7 — Cross-dimensional comparisons...")
    cross_feats    = build_cross_dimension_features(history_df, spark)

    print("[10] Cat 8 — Recovery-specific indicators...")
    recovery_feats = build_recovery_indicators(history_df, spark)

    print("[11] Cat 9a — Within-stage exposure dynamics...")
    ws_feats       = build_within_stage_exposure_dynamics(episode_df, history_df, spark)

    print("[12] Cat 9b — Restructuring / TDR features...")
    tdr_feats      = build_restructuring_features(account_enriched, history_df, spark)

    print("[12b] Time since restructuring...")
    tdr_timing_feats = build_time_since_restructuring(account_enriched, as_of_date)

    print("[13] Cat 10 — Temporal velocity features...")
    vel_feats      = build_temporal_velocity_features(episode_df)

    print("[14] Cat 11 — Payment effort dynamics...")
    effort_feats   = build_payment_effort_dynamics(episode_df, history_df)

    print("[15] Cat 12 — Stage velocity & acceleration...")
    velocity_feats = build_stage_velocity_features(episode_df)

    print("[16] Cat 13 — Cure & re-default dynamics...")
    cure_feats     = build_cure_redefault_dynamics(episode_df)

    print("[17] Cat 14 — Balance recovery potential...")
    recov_pot_feats = build_balance_recovery_potential(episode_df, history_df)

    print("[18] Cat 15 — Engagement recency & decay...")
    engagement_feats = build_engagement_recency_signals(episode_df, history_df)

    print("[19] Cat 16 — Vintage & seasoning dynamics...")
    vintage_feats  = build_vintage_seasoning_features(episode_df)

    # ── Final join: collapse to latest snapshot per customer ─────────────────
    w_latest = Window.partitionBy(REF).orderBy(F.col(DATE).desc())

    base = (staged_df
            .withColumn("_rn", F.row_number().over(w_latest))
            .filter(F.col("_rn") == 1)
            .drop("_rn")
            .select(REF, DATE))

    time_keyed = [
        (trans_feats,    "stage_transitions"),
        (pay_feats,      "payment_behavior"),
        (exp_feats,      "exposure_util"),
        (severity_feats, "delinquency_severity"),
        (cross_feats,    "cross_dimension"),
        (recovery_feats, "recovery_indicators"),
        (vel_feats,      "temporal_velocity"),
    ]
    if enq_feats is not None:
        time_keyed.append((enq_feats, "enquiry"))

    result = base
    for feat_df, name in time_keyed:
        feat_latest = (feat_df
                       .withColumn("_rn", F.row_number().over(w_latest))
                       .filter(F.col("_rn") == 1)
                       .drop("_rn", DATE))
        n_before = len(result.columns)
        result   = result.join(feat_latest, on=REF, how="left")
        print(f"  ✓ {name:<30} +{len(result.columns) - n_before} features")

    # REF-only joins (no DATE key — cross-history aggregates)
    for feat_df, name in [
        (closure_feats,    "account_closure"),
        (ws_feats,         "within_stage_exposure"),
        (tdr_feats,        "restructuring"),
        (tdr_timing_feats, "tdr_timing"),
        (effort_feats,     "payment_effort"),
        (velocity_feats,   "stage_velocity"),
        (cure_feats,       "cure_redefault"),
        (recov_pot_feats,  "recovery_potential"),
        (engagement_feats, "engagement_recency"),
        (vintage_feats,    "vintage_seasoning"),
    ]:
        n_before = len(result.columns)
        result   = result.join(feat_df, on=REF, how="left")
        print(f"  ✓ {name:<30} +{len(result.columns) - n_before} features")

    result = result.dropDuplicates([REF])
    n_features = len(result.columns) - 2
    print(f"\n{sep}\n✓ Complete — {n_features} features\n{sep}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# DATA DICTIONARY
# ─────────────────────────────────────────────────────────────────────────────
#
# Conventions:
#   {STAGE} = CURRENT | X | SM | NPL | CO | CO_DEEP
#   {DIM}   = OVERALL | SECURED | UNSECURED | SECURED_REVOLVING |
#             SECURED_OTHERS | UNSECURED_REVOLVING | UNSECURED_OTHERS
#   {WIN}   = 1M | 3M | 6M | 12M

DATA_DICTIONARY = [
    # ── Category 1: Stage Transition ─────────────────────────────────────────
    {"FEATURE_NAME": "STG_TRANSITION_OVERALL",
     "CATEGORY": "1_Stage_Transition", "DATA_TYPE": "CATEGORICAL",
     "DESCRIPTION": "Stage transition string: prev_stage->curr_stage e.g. X->SM.",
     "CALCULATION_LOGIC": "CONCAT(LAG(dpd_stage), '->', dpd_stage)",
     "EXPECTED_RANGE": "CURRENT/X/SM/NPL/CO/CO_DEEP combinations",
     "BUSINESS_INTERPRETATION": "Direction of movement; rollforward vs cure.",
     "MISSING_VALUE_TREATMENT": "NULL (first observation)",
     "RECOVERY_STAGE_RELEVANCE": "ALL", "REGULATORY_SENSITIVITY": "Low"},

    {"FEATURE_NAME": "STG_WORSENING_FLAG_OVERALL",
     "CATEGORY": "1_Stage_Transition", "DATA_TYPE": "BINARY",
     "DESCRIPTION": "1 if account moved to worse DPD stage vs previous month.",
     "CALCULATION_LOGIC": "1 IF stage_numeric > LAG(stage_numeric)",
     "EXPECTED_RANGE": "0 / 1",
     "BUSINESS_INTERPRETATION": "Early deterioration; key roll-rate input.",
     "MISSING_VALUE_TREATMENT": "0", "RECOVERY_STAGE_RELEVANCE": "X, SM, NPL",
     "REGULATORY_SENSITIVITY": "Medium"},

    {"FEATURE_NAME": "STG_IMPROVEMENT_FLAG_OVERALL",
     "CATEGORY": "1_Stage_Transition", "DATA_TYPE": "BINARY",
     "DESCRIPTION": "1 if account improved stage vs previous month.",
     "CALCULATION_LOGIC": "1 IF stage_numeric < LAG(stage_numeric)",
     "EXPECTED_RANGE": "0 / 1",
     "BUSINESS_INTERPRETATION": "Cure signal for recovery models.",
     "MISSING_VALUE_TREATMENT": "0", "RECOVERY_STAGE_RELEVANCE": "NPL, CO",
     "REGULATORY_SENSITIVITY": "Medium"},

    {"FEATURE_NAME": "STG_DPD_CHANGE_1M_OVERALL",
     "CATEGORY": "1_Stage_Transition", "DATA_TYPE": "NUMERIC",
     "DESCRIPTION": "ΔDPD vs previous month. Negative = improvement.",
     "CALCULATION_LOGIC": "bureau_max_dpd - LAG(bureau_max_dpd)",
     "EXPECTED_RANGE": "-360 to +360",
     "BUSINESS_INTERPRETATION": "Rate of DPD change.",
     "MISSING_VALUE_TREATMENT": "NULL", "RECOVERY_STAGE_RELEVANCE": "ALL",
     "REGULATORY_SENSITIVITY": "Low"},

    {"FEATURE_NAME": "STG_DPD_ACCELERATION_OVERALL",
     "CATEGORY": "1_Stage_Transition", "DATA_TYPE": "NUMERIC",
     "DESCRIPTION": "Second derivative of DPD. Positive = worsening faster.",
     "CALCULATION_LOGIC": "(DPD_t - DPD_t1) - (DPD_t1 - DPD_t2)",
     "EXPECTED_RANGE": "-720 to +720",
     "BUSINESS_INTERPRETATION": "Shock detector — catches rapid deterioration.",
     "MISSING_VALUE_TREATMENT": "NULL", "RECOVERY_STAGE_RELEVANCE": "CURRENT, X, SM",
     "REGULATORY_SENSITIVITY": "Low"},

    {"FEATURE_NAME": "STG_RE_DEFAULT_FLAG_OVERALL",
     "CATEGORY": "1_Stage_Transition", "DATA_TYPE": "BINARY",
     "DESCRIPTION": "1 if improvement in past 12M AND re-worsening in past 6M (backward-looking, no leakage).",
     "CALCULATION_LOGIC": "MAX(improvement in t-12:t-1)==1 AND MAX(worsening in t-6:t)==1",
     "EXPECTED_RANGE": "0 / 1",
     "BUSINESS_INTERPRETATION": "Serial re-defaulter; structural inability to sustain cure.",
     "MISSING_VALUE_TREATMENT": "0", "RECOVERY_STAGE_RELEVANCE": "NPL, CO",
     "REGULATORY_SENSITIVITY": "Medium"},

    {"FEATURE_NAME": "STG_TIMES_IN_{STAGE}_OVERALL",
     "CATEGORY": "1_Stage_Transition", "DATA_TYPE": "NUMERIC",
     "DESCRIPTION": "Count of distinct episodes (contiguous runs) customer has been in {STAGE} (lifetime).",
     "CALCULATION_LOGIC": "COUNTDISTINCT(episode_id WHERE episode_stage={STAGE}) OVER UNBOUNDED",
     "EXPECTED_RANGE": "0 to ~20",
     "BUSINESS_INTERPRETATION": "> 2 NPL episodes = chronic risk.",
     "MISSING_VALUE_TREATMENT": "0", "RECOVERY_STAGE_RELEVANCE": "ALL",
     "REGULATORY_SENSITIVITY": "Low"},

    # ── Category 2: Payment Behavior ─────────────────────────────────────────
    {"FEATURE_NAME": "PAY_RATIO_{DIM}",
     "CATEGORY": "2_Payment_Behavior", "DATA_TYPE": "NUMERIC",
     "DESCRIPTION": "Balance-decrease / balance for dimension {DIM} (payment proxy).",
     "CALCULATION_LOGIC": "-SUM(BAL_CHANGE) / SUM(amountowed)",
     "EXPECTED_RANGE": "-inf to 1.0", "MISSING_VALUE_TREATMENT": "NULL",
     "BUSINESS_INTERPRETATION": "> 0.9 = effectively full payer.",
     "RECOVERY_STAGE_RELEVANCE": "ALL", "REGULATORY_SENSITIVITY": "Low"},

    {"FEATURE_NAME": "FULL_PAY_STREAK_{DIM}",
     "CATEGORY": "2_Payment_Behavior", "DATA_TYPE": "NUMERIC",
     "DESCRIPTION": "Consecutive months of balance decrease (payment) ending at current month for {DIM}.",
     "CALCULATION_LOGIC": "Island detection: row_number within group of consecutive pay months.",
     "EXPECTED_RANGE": "0 to ~60",
     "BUSINESS_INTERPRETATION": "Long streak = reliable payer; break in streak = deterioration signal.",
     "MISSING_VALUE_TREATMENT": "0", "RECOVERY_STAGE_RELEVANCE": "ALL",
     "REGULATORY_SENSITIVITY": "Low"},

    {"FEATURE_NAME": "MISSED_PAY_STREAK_{DIM}",
     "CATEGORY": "2_Payment_Behavior", "DATA_TYPE": "NUMERIC",
     "DESCRIPTION": "Consecutive months of zero payment ending at current month for {DIM}.",
     "CALCULATION_LOGIC": "Island detection: row_number within group of consecutive missed pay months.",
     "EXPECTED_RANGE": "0 to ~60",
     "BUSINESS_INTERPRETATION": "> 3 = collections priority; > 12 = potential write-off.",
     "MISSING_VALUE_TREATMENT": "0", "RECOVERY_STAGE_RELEVANCE": "NPL, CO",
     "REGULATORY_SENSITIVITY": "Medium"},

    # ── Category 3: Exposure & Utilisation ───────────────────────────────────
    {"FEATURE_NAME": "UTIL_{DIM}",
     "CATEGORY": "3_Exposure_Utilisation", "DATA_TYPE": "NUMERIC",
     "DESCRIPTION": "Average utilisation (balance/limit) for {DIM}.",
     "CALCULATION_LOGIC": "AVG(amountowed / creditlimit)",
     "EXPECTED_RANGE": "0.0 to 1.0+", "MISSING_VALUE_TREATMENT": "NULL",
     "BUSINESS_INTERPRETATION": "> 0.9 = maxed out.",
     "RECOVERY_STAGE_RELEVANCE": "CURRENT, X, SM", "REGULATORY_SENSITIVITY": "Low"},

    {"FEATURE_NAME": "UTIL_TREND_3M_{DIM}",
     "CATEGORY": "3_Exposure_Utilisation", "DATA_TYPE": "NUMERIC",
     "DESCRIPTION": "Slope of utilisation over 3 months: (util_now - util_3m_ago) / 3.",
     "CALCULATION_LOGIC": "(UTIL_{DIM} - LAG(UTIL_{DIM}, 2)) / 3",
     "EXPECTED_RANGE": "-0.5 to +0.5 per month", "MISSING_VALUE_TREATMENT": "NULL",
     "BUSINESS_INTERPRETATION": "Positive trend = growing stressed.",
     "RECOVERY_STAGE_RELEVANCE": "CURRENT, X", "REGULATORY_SENSITIVITY": "Low"},

    # ── Category 4: Enquiry ───────────────────────────────────────────────────
    {"FEATURE_NAME": "ENQ_COUNT_3M_OVERALL",
     "CATEGORY": "4_Enquiry", "DATA_TYPE": "NUMERIC",
     "DESCRIPTION": "Number of bureau enquiries in last 3 months.",
     "CALCULATION_LOGIC": "SUM(monthly_count) OVER 3M",
     "EXPECTED_RANGE": "0 to ~40", "MISSING_VALUE_TREATMENT": "0",
     "BUSINESS_INTERPRETATION": "> 5 = active credit-seeking / financial stress.",
     "RECOVERY_STAGE_RELEVANCE": "CURRENT, X", "REGULATORY_SENSITIVITY": "Medium"},

    {"FEATURE_NAME": "ENQ_BURST_FLAG_OVERALL",
     "CATEGORY": "4_Enquiry", "DATA_TYPE": "BINARY",
     "DESCRIPTION": "1 if ≥ 3 enquiries in current month.",
     "CALCULATION_LOGIC": "1 IF ENQ_COUNT_1M >= 3",
     "EXPECTED_RANGE": "0 / 1", "MISSING_VALUE_TREATMENT": "0",
     "BUSINESS_INTERPRETATION": "Liquidity crisis — multiple simultaneous applications.",
     "RECOVERY_STAGE_RELEVANCE": "CURRENT, X", "REGULATORY_SENSITIVITY": "Medium"},

    # ── Category 5: Account Closure ───────────────────────────────────────────
    {"FEATURE_NAME": "CLOSED_LOAN_COUNT_6M_{DIM}",
     "CATEGORY": "5_Account_Closure", "DATA_TYPE": "NUMERIC",
     "DESCRIPTION": "Accounts closed in 6M prior to as_of_date for {DIM} (PIT-safe).",
     "CALCULATION_LOGIC": "COUNT closedate BETWEEN (as_of_date - 6M) AND as_of_date",
     "EXPECTED_RANGE": "0 to ~10", "MISSING_VALUE_TREATMENT": "0",
     "BUSINESS_INTERPRETATION": "High closures = voluntary deleveraging or lender action.",
     "RECOVERY_STAGE_RELEVANCE": "ALL", "REGULATORY_SENSITIVITY": "Low"},

    {"FEATURE_NAME": "RELATIONSHIP_TENURE_{DIM}",
     "CATEGORY": "5_Account_Closure", "DATA_TYPE": "NUMERIC",
     "DESCRIPTION": "Months since first {DIM} account opened (PIT-safe: uses as_of_date).",
     "CALCULATION_LOGIC": "MONTHS_BETWEEN(as_of_date, MIN(opendate))",
     "EXPECTED_RANGE": "0 to ~300", "MISSING_VALUE_TREATMENT": "NULL",
     "BUSINESS_INTERPRETATION": "Shorter tenure = thin-file, higher uncertainty.",
     "RECOVERY_STAGE_RELEVANCE": "ALL", "REGULATORY_SENSITIVITY": "Low"},

    {"FEATURE_NAME": "ACTIVE_ACCT_CHANGE_{DIM}",
     "CATEGORY": "5_Account_Closure", "DATA_TYPE": "NUMERIC",
     "DESCRIPTION": "Net change in active accounts in 6M: opened - closed.",
     "CALCULATION_LOGIC": "COUNT(opendate in 6M) - COUNT(closedate in 6M)",
     "EXPECTED_RANGE": "-10 to +10", "MISSING_VALUE_TREATMENT": "0",
     "BUSINESS_INTERPRETATION": "Negative = net contraction of credit portfolio.",
     "RECOVERY_STAGE_RELEVANCE": "ALL", "REGULATORY_SENSITIVITY": "Low"},

    # ── Category 6: Delinquency Severity ──────────────────────────────────────
    {"FEATURE_NAME": "SEVERITY_SCORE_{DIM}",
     "CATEGORY": "6_Delinquency_Severity", "DATA_TYPE": "NUMERIC",
     "DESCRIPTION": "Balance-weighted avg stage severity for {DIM} (0=CURRENT … 5=CO_DEEP).",
     "CALCULATION_LOGIC": "SUM(stage_numeric * amountowed) / SUM(amountowed)",
     "EXPECTED_RANGE": "0.0 to 5.0", "MISSING_VALUE_TREATMENT": "0",
     "BUSINESS_INTERPRETATION": "> 3 = predominantly NPL/CO.",
     "RECOVERY_STAGE_RELEVANCE": "ALL", "REGULATORY_SENSITIVITY": "Medium"},

    {"FEATURE_NAME": "MAX_STAGE_REACHED_{DIM}",
     "CATEGORY": "6_Delinquency_Severity", "DATA_TYPE": "NUMERIC",
     "DESCRIPTION": "Worst stage ordinal (0–5) ever reached for {DIM} (lifetime max of _stage_num).",
     "CALCULATION_LOGIC": "MAX(stage_numeric) OVER UNBOUNDED PRECEDING",
     "EXPECTED_RANGE": "0 to 5", "MISSING_VALUE_TREATMENT": "0",
     "BUSINESS_INTERPRETATION": "CO_DEEP ever (5) = near-zero recovery probability.",
     "RECOVERY_STAGE_RELEVANCE": "ALL", "REGULATORY_SENSITIVITY": "Medium"},

    {"FEATURE_NAME": "CURE_RATE_6M_{DIM}",
     "CATEGORY": "6_Delinquency_Severity", "DATA_TYPE": "NUMERIC",
     "DESCRIPTION": "Fraction of last 6M months with stage improvement for {DIM}.",
     "CALCULATION_LOGIC": "SUM(improvement_flag) OVER 6M / 6",
     "EXPECTED_RANGE": "0.0 to 1.0", "MISSING_VALUE_TREATMENT": "NULL",
     "BUSINESS_INTERPRETATION": "High = responsive to interventions.",
     "RECOVERY_STAGE_RELEVANCE": "X, SM, NPL", "REGULATORY_SENSITIVITY": "Low"},

    {"FEATURE_NAME": "RE_DEFAULT_FLAG_{DIM}",
     "CATEGORY": "6_Delinquency_Severity", "DATA_TYPE": "BINARY",
     "DESCRIPTION": "1 if improvement and re-worsening both observed in past 12M for {DIM}.",
     "CALCULATION_LOGIC": "Backward-looking only — no forward window.",
     "EXPECTED_RANGE": "0 / 1", "MISSING_VALUE_TREATMENT": "0",
     "BUSINESS_INTERPRETATION": "Structural inability to sustain cure.",
     "RECOVERY_STAGE_RELEVANCE": "NPL, CO", "REGULATORY_SENSITIVITY": "Medium"},

    # ── Category 7: Cross-Dimensional ────────────────────────────────────────
    {"FEATURE_NAME": "STAGE_DIVERGENCE_FLAG",
     "CATEGORY": "7_Cross_Dimensional", "DATA_TYPE": "BINARY",
     "DESCRIPTION": "1 if secured products CURRENT while unsecured NPL or worse.",
     "CALCULATION_LOGIC": "max_secured_stage==CURRENT AND max_unsecured_stage IN (NPL,CO,CO_DEEP)",
     "EXPECTED_RANGE": "0 / 1", "MISSING_VALUE_TREATMENT": "0",
     "BUSINESS_INTERPRETATION": "Selective default: protecting collateral.",
     "RECOVERY_STAGE_RELEVANCE": "NPL, CO", "REGULATORY_SENSITIVITY": "High"},

    {"FEATURE_NAME": "CONCENTRATION_RISK_UNSECURED",
     "CATEGORY": "7_Cross_Dimensional", "DATA_TYPE": "NUMERIC",
     "DESCRIPTION": "Herfindahl index of unsecured balance: SUM(bal^2) / SUM(bal)^2.",
     "CALCULATION_LOGIC": "SUM(unsecured_bal^2) / SUM(unsecured_bal)^2",
     "EXPECTED_RANGE": "0.0 to 1.0 (1.0 = fully concentrated in one product)",
     "MISSING_VALUE_TREATMENT": "NULL", "BUSINESS_INTERPRETATION":
         "High = concentrated unsecured exposure; single-product default risk.",
     "RECOVERY_STAGE_RELEVANCE": "SM, NPL", "REGULATORY_SENSITIVITY": "Low"},

    {"FEATURE_NAME": "STRATEGIC_DEFAULT_INDICATOR",
     "CATEGORY": "7_Cross_Dimensional", "DATA_TYPE": "BINARY",
     "DESCRIPTION": "1 if secured util < 0.5 (capacity exists) AND unsecured DPD > 90.",
     "CALCULATION_LOGIC": "avg_secured_util < 0.5 AND max_unsecured_dpd > 90",
     "EXPECTED_RANGE": "0 / 1", "MISSING_VALUE_TREATMENT": "0",
     "BUSINESS_INTERPRETATION": "Ability-to-pay but choosing not to on unsecured.",
     "RECOVERY_STAGE_RELEVANCE": "NPL, CO", "REGULATORY_SENSITIVITY": "High"},

    # ── Category 8: Recovery-Specific ────────────────────────────────────────
    {"FEATURE_NAME": "TIME_SINCE_LAST_PAY_OVERALL",
     "CATEGORY": "8_Recovery_Specific", "DATA_TYPE": "NUMERIC",
     "DESCRIPTION": "Months since any balance decrease (payment proxy) was observed.",
     "CALCULATION_LOGIC": "Row number within group of non-payment months (island detection).",
     "EXPECTED_RANGE": "0 to ~60", "MISSING_VALUE_TREATMENT": "0",
     "BUSINESS_INTERPRETATION": "> 6 = dormant account; > 12 = likely unrecoverable.",
     "RECOVERY_STAGE_RELEVANCE": "CO, CO_DEEP", "REGULATORY_SENSITIVITY": "High"},

    {"FEATURE_NAME": "RECOVERY_PROPENSITY_SCORE_OVERALL",
     "CATEGORY": "8_Recovery_Specific", "DATA_TYPE": "NUMERIC",
     "DESCRIPTION": "Payments made while in CO / total months in CO (payment rate post charge-off).",
     "CALCULATION_LOGIC": "SUM(PAY_AFTER_DELINQUENCY_FLAG_CO) / GREATEST(1, TIME_IN_CO_OVERALL)",
     "EXPECTED_RANGE": "0.0 to 1.0", "MISSING_VALUE_TREATMENT": "0",
     "BUSINESS_INTERPRETATION": "High = good recovery target; low = write-off/legal.",
     "RECOVERY_STAGE_RELEVANCE": "CO, CO_DEEP", "REGULATORY_SENSITIVITY": "High"},

    # ── Category 9a: Within-Stage Exposure ────────────────────────────────────
    {"FEATURE_NAME": "WS_DIP_COUNT_{STAGE}",
     "CATEGORY": "9a_Within_Stage_Exposure", "DATA_TYPE": "NUMERIC",
     "DESCRIPTION": "Months with balance dip in current {STAGE} episode (payment proxy).",
     "CALCULATION_LOGIC": "COUNT(amountowed < prev_amountowed) within episode",
     "EXPECTED_RANGE": "0 to WS_DURATION_{STAGE}", "MISSING_VALUE_TREATMENT": "NULL",
     "BUSINESS_INTERPRETATION": "Even partial payments in SM/NPL = higher cure propensity.",
     "RECOVERY_STAGE_RELEVANCE": "{STAGE}", "REGULATORY_SENSITIVITY": "Low"},

    {"FEATURE_NAME": "WS_AVG_DIP_MAG_{STAGE}",
     "CATEGORY": "9a_Within_Stage_Exposure", "DATA_TYPE": "NUMERIC",
     "DESCRIPTION": "Average THB dip magnitude within current {STAGE} episode.",
     "CALCULATION_LOGIC": "AVG(prev_bal - curr_bal WHERE dip > 0) within episode",
     "EXPECTED_RANGE": "0 to ~5,000,000 THB", "MISSING_VALUE_TREATMENT": "NULL",
     "BUSINESS_INTERPRETATION": "Larger dips = capacity exists, just struggling with timing.",
     "RECOVERY_STAGE_RELEVANCE": "NPL, CO", "REGULATORY_SENSITIVITY": "Low"},

    # ── Category 9b: Restructuring / TDR ─────────────────────────────────────
    {"FEATURE_NAME": "TDR_FLAG_OVERALL",
     "CATEGORY": "9b_Restructuring", "DATA_TYPE": "BINARY",
     "DESCRIPTION": "1 if customer has any accounttype=90 (Restructured Debt) account.",
     "CALCULATION_LOGIC": "1 IF any accounttype_cd == '90'",
     "EXPECTED_RANGE": "0 / 1", "MISSING_VALUE_TREATMENT": "0",
     "BUSINESS_INTERPRETATION": "Prior financial distress requiring formal restructuring.",
     "RECOVERY_STAGE_RELEVANCE": "ALL", "REGULATORY_SENSITIVITY": "High"},

    {"FEATURE_NAME": "RESTRUCTURING_SUCCESS_FLAG",
     "CATEGORY": "9b_Restructuring", "DATA_TYPE": "BINARY",
     "DESCRIPTION": "1 if max DPD < 30 in 6M following last restructuring event.",
     "CALCULATION_LOGIC": "MAX(bureau_max_dpd) < 30 in [tdr_opendate, tdr_opendate+6M]",
     "EXPECTED_RANGE": "0 / 1", "MISSING_VALUE_TREATMENT": "0",
     "BUSINESS_INTERPRETATION": "Failed restructure + current NPL = escalate to legal.",
     "RECOVERY_STAGE_RELEVANCE": "NPL, CO", "REGULATORY_SENSITIVITY": "High"},

    # ── Category 10: Temporal Velocity ────────────────────────────────────────
    {"FEATURE_NAME": "ESCALATION_VELOCITY_TO_NPL",
     "CATEGORY": "10_Temporal_Velocity", "DATA_TYPE": "NUMERIC",
     "DESCRIPTION": "Minimum months from a CURRENT episode to NPL (shortest escalation path, lifetime).",
     "CALCULATION_LOGIC": "MIN(MONTHS_BETWEEN(npl_start, current_start)) WHERE npl_ep > current_ep",
     "EXPECTED_RANGE": "2 to ~30 months", "MISSING_VALUE_TREATMENT": "NULL (never reached NPL)",
     "BUSINESS_INTERPRETATION": "Fast escalators (< 6M) are highest-risk profiles.",
     "RECOVERY_STAGE_RELEVANCE": "CURRENT, X, SM", "REGULATORY_SENSITIVITY": "Medium"},

    {"FEATURE_NAME": "CURE_VELOCITY_FROM_NPL",
     "CATEGORY": "10_Temporal_Velocity", "DATA_TYPE": "NUMERIC",
     "DESCRIPTION": "Minimum months from NPL to CURRENT episode (shortest cure observed, lifetime).",
     "CALCULATION_LOGIC": "MIN(MONTHS_BETWEEN(current_start_after_npl, npl_start))",
     "EXPECTED_RANGE": "1 to ~24 months", "MISSING_VALUE_TREATMENT": "NULL (never cured from NPL)",
     "BUSINESS_INTERPRETATION": "Fast curers respond to restructuring offers.",
     "RECOVERY_STAGE_RELEVANCE": "NPL", "REGULATORY_SENSITIVITY": "Low"},

    {"FEATURE_NAME": "ROLL_RATE_CURRENT_TO_X",
     "CATEGORY": "10_Temporal_Velocity", "DATA_TYPE": "NUMERIC",
     "DESCRIPTION": "Fraction of CURRENT months that transitioned to X (lifetime).",
     "CALCULATION_LOGIC": "COUNT(CURRENT->X transitions) / COUNT(CURRENT months)",
     "EXPECTED_RANGE": "0.0 to 1.0", "MISSING_VALUE_TREATMENT": "NULL",
     "BUSINESS_INTERPRETATION": "Portfolio-level roll rate; > 0.15 signals stress.",
     "RECOVERY_STAGE_RELEVANCE": "CURRENT", "REGULATORY_SENSITIVITY": "Low"},

    {"FEATURE_NAME": "STAGE_STICKINESS_SCORE",
     "CATEGORY": "10_Temporal_Velocity", "DATA_TYPE": "NUMERIC",
     "DESCRIPTION": "Average episode_month_number over last 12M — proxy for how long accounts stay in stages.",
     "CALCULATION_LOGIC": "AVG(episode_month_number) OVER 12M",
     "EXPECTED_RANGE": "1.0 to ~30.0", "MISSING_VALUE_TREATMENT": "NULL",
     "BUSINESS_INTERPRETATION": "High stickiness in NPL/CO = chronic delinquency.",
     "RECOVERY_STAGE_RELEVANCE": "ALL", "REGULATORY_SENSITIVITY": "Low"},

    {"FEATURE_NAME": "PAYMENT_GAP_MONTHS",
     "CATEGORY": "10_Temporal_Velocity", "DATA_TYPE": "NUMERIC",
     "DESCRIPTION": "Count of months with DPD > 0 in last 12M (missed payment months).",
     "CALCULATION_LOGIC": "SUM(1 WHERE bureau_max_dpd > 0) OVER 12M",
     "EXPECTED_RANGE": "0 to 12", "MISSING_VALUE_TREATMENT": "0",
     "BUSINESS_INTERPRETATION": "> 6 months delinquent in 12M = high charge-off risk.",
     "RECOVERY_STAGE_RELEVANCE": "ALL", "REGULATORY_SENSITIVITY": "Low"},
]


def print_data_dictionary() -> None:
    """Print feature data dictionary as a formatted table (call from notebook)."""
    header = (f"{'FEATURE_NAME':<55} {'CATEGORY':<25} {'TYPE':<12} "
              f"{'RECOVERY_STAGE':<20} DESCRIPTION")
    sep = "-" * 175
    print(sep)
    print(header)
    print(sep)
    for e in DATA_DICTIONARY:
        print(f"{e['FEATURE_NAME']:<55} {e['CATEGORY']:<25} {e['DATA_TYPE']:<12} "
              f"{e['RECOVERY_STAGE_RELEVANCE']:<20} {e['DESCRIPTION'][:70]}")
    print(sep)
    print(f"Documented features: {len(DATA_DICTIONARY)}")


# ─────────────────────────────────────────────────────────────────────────────
# NCB ACCOUNTTYPE QUICK-REFERENCE (from mnf_cra_rvw_s_account.ACCOUNTTYPE)
# ─────────────────────────────────────────────────────────────────────────────
#
# acct_type_cd | description               | is_secured | is_revolving | dim_label
# -------------|---------------------------|------------|--------------|------------------
# 01           | Commercial Loan           | NO         | NO           | UNSECURED_OTHERS
# 04           | Overdraft                 | NO         | YES          | UNSECURED_REVOLVING
# 05           | Personal Loan             | NO         | NO           | UNSECURED_OTHERS
# 06           | Housing/Mortgage          | YES        | NO           | SECURED_OTHERS
# 20           | Automobile Leasing        | YES        | NO           | SECURED_OTHERS
# 21           | Other Hire Purchase       | YES        | NO           | SECURED_OTHERS
# 22           | Credit Card               | NO         | YES          | UNSECURED_REVOLVING
# 27           | Automobile Hire Purchase  | YES        | NO           | SECURED_OTHERS
# 31           | HP Agriculture (var inst) | YES        | NO           | SECURED_OTHERS
# 32           | HP Agriculture            | YES        | NO           | SECURED_OTHERS
# 33           | Loan for Agriculture      | NO         | NO           | UNSECURED_OTHERS
# 36           | Coop Loan                 | NO         | NO           | UNSECURED_OTHERS
# 37           | Nano-Finance              | NO         | NO           | UNSECURED_OTHERS
# 52           | Securitized Mortgage      | YES        | NO           | SECURED_OTHERS
# 53           | Securitized Auto Leasing  | YES        | NO           | SECURED_OTHERS
# 54           | Securitized Other HP      | YES        | NO           | SECURED_OTHERS
# 55           | Securitized Credit Card   | NO         | YES          | UNSECURED_REVOLVING
# 56           | Securitized Auto HP       | YES        | NO           | SECURED_OTHERS
# 58           | Securitized Overdraft     | NO         | YES          | UNSECURED_REVOLVING
# 90           | Restructured Debt         | NO         | NO           | UNSECURED_OTHERS  ← TDR
# 99           | Other Loans               | NO         | NO           | UNSECURED_OTHERS
