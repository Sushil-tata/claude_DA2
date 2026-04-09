"""
Bureau Stage Dynamics Engine
============================
Computes within-stage and cross-stage temporal dynamics across all NCB delinquency
buckets: S0 (Current, 0 DPD), S1 (X, 1-30 DPD), S2 (SM, 31-90 DPD),
S3 (NPL, 91-180 DPD), S4 (CO, 180+ DPD).

Design principle:
    Most feature factories compute WHAT state a customer is in.
    This module computes HOW a customer behaves WITHIN each state and
    HOW they move BETWEEN states — the dynamicity the recovery model needs.

For each stage and each key dimension (balance, limit, amount financed,
loan count, DPD counter), compute:
    - Time spent (duration, oscillation count)
    - Amount dynamics (trend, velocity, dip-as-payment-proxy)
    - DPD trajectory within stage (slope through stage)
    - Structural change (loan additions/closures per stage)

Column contract (matches ncb_feature_factory_v2.py CFG):
    Join key: ref_no
    Time key: asofdate (bureau history month)
    DPD state: dpd_state (S0/S1/S2/S3/S4) — from build_dpd_states()
    Source columns from s_history: amountowed, creditlimit, amountfinanced,
                                   overduemonths, membershortname
    All column names lowercased on load.

Execution: Called from run_ncb_feature_factory_v2() after build_dpd_states()
           and before build_advanced_physics_features().

Output: ~120 additional features joined on ref_no.

Author: Behavioral Physics Team
Version: 1.0.0
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from typing import Dict, List, Optional

# ── Column contract ────────────────────────────────────────────────────────────
REF_COL    = "ref_no"
DATE_COL   = "asofdate"
DPD_COL    = "bureau_max_dpd"
STATE_COL  = "dpd_state"         # S0/S1/S2/S3/S4 — from build_dpd_states()
BAL_COL    = "amountowed"
LIMIT_COL  = "creditlimit"
FINANCED_COL = "amountfinanced"
ODM_COL    = "overduemonths"
LENDER_COL = "membershortname"

STAGES     = ["S0", "S1", "S2", "S3", "S4"]
STAGE_LABELS = {
    "S0": "current",
    "S1": "x_bucket",          # X = 1-30 DPD (Thai regulatory X classification)
    "S2": "sm",                 # Special Mention = 31-90 DPD
    "S3": "npl",                # NPL = 91-180 DPD
    "S4": "co",                 # Charge-Off = 180+ DPD
}


def _safe_div(num, den, default=F.lit(None)):
    return F.when((den.isNotNull()) & (den != 0), num / den).otherwise(default)


def _slope_approx(val_col: str, n_months: int) -> "Column":
    """
    Linear slope approximation: (current - start) / n_months.
    Inline — avoids UDF for Databricks performance.
    """
    w = Window.partitionBy(REF_COL).orderBy(DATE_COL).rowsBetween(-(n_months - 1), 0)
    return (F.col(val_col) - F.first(val_col).over(w)) / n_months


# ============================================================================
# SECTION 1 — STAGE EPISODE BUILDER
# Assigns each customer-month to a stage episode (contiguous run in same stage)
# ============================================================================

def build_stage_episodes(state_df: DataFrame) -> DataFrame:
    """
    Assign episode IDs to contiguous runs in the same DPD stage.

    A "stage episode" is a consecutive sequence of months where the customer
    remains in the same DPD bucket (e.g., 4 months in S2 before rolling to S3).
    Episode boundaries reset when stage changes.

    This is the foundation for within-stage dynamics — all subsequent
    computations operate at the episode level.

    Input cols required: ref_no, asofdate, dpd_state
    Output adds: episode_id, episode_stage, episode_start_month,
                 episode_month_number (1 = entry month)
    """
    w = Window.partitionBy(REF_COL).orderBy(DATE_COL)

    # Flag month where stage changes from previous month
    df = state_df.withColumn(
        "_prev_state",
        F.lag(STATE_COL, 1).over(w)
    ).withColumn(
        "_stage_changed",
        F.when(
            F.col(STATE_COL) != F.col("_prev_state"), 1
        ).otherwise(
            F.when(F.col("_prev_state").isNull(), 1).otherwise(0)
        )
    )

    # Cumulative sum of stage changes = episode ID (unique per customer)
    df = df.withColumn(
        "episode_id",
        F.sum("_stage_changed").over(w)
    ).withColumn(
        "episode_stage", F.col(STATE_COL)
    )

    # Month number within episode (1 = entry month into stage)
    w_ep = Window.partitionBy(REF_COL, "episode_id").orderBy(DATE_COL)
    df = df.withColumn(
        "episode_month_number",
        F.row_number().over(w_ep)
    ).withColumn(
        "episode_start_month",
        F.first(DATE_COL).over(w_ep)
    )

    return df.drop("_prev_state", "_stage_changed")


# ============================================================================
# SECTION 2 — WITHIN-STAGE BALANCE / EXPOSURE DYNAMICS
# Balance dip = payment proxy. Limit change = lender response.
# ============================================================================

def build_within_stage_exposure_dynamics(
    episode_df: DataFrame,
    history_df: DataFrame,
) -> DataFrame:
    """
    Compute balance, limit, and amount-financed dynamics within each stage episode.

    Key insight: A dip in balance while remaining in the same DPD stage is the
    strongest proxy for a payment having been made. The customer has not cured
    (still in S2 for example) but has made a partial payment. This is the
    "effort signal" that distinguishes Sudden-Shock Distressed from Structural
    Defaulters even within the same bucket.

    Features per stage (S0–S4):
        balance_entry_{stage}:      Balance at episode entry (first month in stage)
        balance_exit_{stage}:       Balance at last observed month of episode
        balance_slope_{stage}:      Linear slope of balance WITHIN episode
        balance_dip_count_{stage}:  # months balance fell (payment proxy)
        balance_dip_magnitude_{stage}: Avg month-over-month balance decline when dip occurs
        balance_volatility_{stage}: Std dev of balance within episode
        util_entry_{stage}:         Utilisation at entry
        util_slope_{stage}:         Utilisation slope within episode
        limit_change_{stage}:       (exit_limit − entry_limit) — lender response
        financed_slope_{stage}:     Slope of amount financed within episode
        financed_at_entry_{stage}:  Amount financed on stage entry
    """
    # Aggregate history to ref_no + asofdate level (one row per customer-month)
    monthly = history_df.groupBy(REF_COL, DATE_COL).agg(
        F.sum(BAL_COL).alias("total_balance"),
        F.sum(LIMIT_COL).alias("total_limit"),
        F.sum(FINANCED_COL).alias("total_financed"),
        F.countDistinct(LENDER_COL).alias("num_active_tradelines"),
    )

    # Join episode IDs to monthly exposure
    df = episode_df.select(REF_COL, DATE_COL, "episode_id", "episode_stage",
                            "episode_month_number").join(
        monthly, on=[REF_COL, DATE_COL], how="left"
    )

    # Utilisation (balance / limit)
    df = df.withColumn(
        "util_rate",
        _safe_div(F.col("total_balance"), F.col("total_limit"), F.lit(None))
    )

    # Month-over-month balance change within episode
    w_ep = Window.partitionBy(REF_COL, "episode_id").orderBy(DATE_COL)
    df = df.withColumn(
        "balance_mom_change",
        F.col("total_balance") - F.lag("total_balance", 1).over(w_ep)
    )

    # Flag months where balance FELL (payment proxy)
    df = df.withColumn(
        "balance_dip_flag",
        F.when(F.col("balance_mom_change") < 0, 1).otherwise(0)
    )

    # ── Episode-level aggregation ──────────────────────────────────────────
    w_ep_full = Window.partitionBy(REF_COL, "episode_id")

    df = df.withColumn(
        "ep_balance_entry",
        F.first("total_balance").over(Window.partitionBy(REF_COL, "episode_id").orderBy(DATE_COL))
    ).withColumn(
        "ep_balance_exit",
        F.last("total_balance").over(Window.partitionBy(REF_COL, "episode_id").orderBy(DATE_COL)
                                      .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing))
    ).withColumn(
        "ep_util_entry",
        F.first("util_rate").over(Window.partitionBy(REF_COL, "episode_id").orderBy(DATE_COL))
    ).withColumn(
        "ep_limit_entry",
        F.first("total_limit").over(Window.partitionBy(REF_COL, "episode_id").orderBy(DATE_COL))
    ).withColumn(
        "ep_limit_exit",
        F.last("total_limit").over(Window.partitionBy(REF_COL, "episode_id").orderBy(DATE_COL)
                                    .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing))
    ).withColumn(
        "ep_financed_entry",
        F.first("total_financed").over(Window.partitionBy(REF_COL, "episode_id").orderBy(DATE_COL))
    ).withColumn(
        "ep_episode_length",
        F.count("*").over(w_ep_full)
    ).withColumn(
        "ep_balance_slope",
        _safe_div(
            F.col("ep_balance_exit") - F.col("ep_balance_entry"),
            F.col("ep_episode_length"),
            F.lit(0.0)
        )
    ).withColumn(
        "ep_util_slope",
        _safe_div(
            F.col("ep_util_entry"),
            F.col("ep_episode_length"),
            F.lit(0.0)
        )
    ).withColumn(
        "ep_limit_change",
        F.col("ep_limit_exit") - F.col("ep_limit_entry")
    ).withColumn(
        "ep_dip_count",
        F.sum("balance_dip_flag").over(w_ep_full)
    ).withColumn(
        "ep_dip_magnitude",
        _safe_div(
            F.sum(
                F.when(F.col("balance_dip_flag") == 1, F.abs(F.col("balance_mom_change")))
                 .otherwise(F.lit(0.0))
            ).over(w_ep_full),
            F.sum("balance_dip_flag").over(w_ep_full),
            F.lit(0.0)
        )
    ).withColumn(
        "ep_balance_vol",
        F.stddev("total_balance").over(w_ep_full)
    )

    # ── Keep only episode entry row, then pivot by stage ───────────────────
    # One row per episode (entry month)
    episode_summary = df.filter(F.col("episode_month_number") == 1).select(
        REF_COL,
        "episode_id",
        "episode_stage",
        "ep_balance_entry",
        "ep_balance_exit",
        "ep_balance_slope",
        "ep_util_entry",
        "ep_util_slope",
        "ep_limit_entry",
        "ep_limit_exit",
        "ep_limit_change",
        "ep_financed_entry",
        "ep_dip_count",
        "ep_dip_magnitude",
        "ep_balance_vol",
        "ep_episode_length",
        DATE_COL,
    )

    # ── Aggregate across all episodes per stage per customer ───────────────
    agg_by_stage = episode_summary.groupBy(REF_COL, "episode_stage").agg(
        F.count("episode_id").alias("ep_count"),
        F.avg("ep_episode_length").alias("avg_duration"),
        F.max("ep_episode_length").alias("max_duration"),
        F.avg("ep_balance_entry").alias("avg_balance_entry"),
        F.avg("ep_balance_slope").alias("avg_balance_slope"),
        F.avg("ep_util_entry").alias("avg_util_entry"),
        F.avg("ep_util_slope").alias("avg_util_slope"),
        F.avg("ep_limit_change").alias("avg_limit_change"),
        F.avg("ep_financed_entry").alias("avg_financed_entry"),
        F.sum("ep_dip_count").alias("total_dip_count"),
        F.avg("ep_dip_magnitude").alias("avg_dip_magnitude"),
        F.avg("ep_balance_vol").alias("avg_balance_vol"),
    )

    # ── Pivot: one column per stage ────────────────────────────────────────
    stage_pivots = []
    for stage, label in STAGE_LABELS.items():
        stage_df = agg_by_stage.filter(F.col("episode_stage") == stage).select(
            REF_COL,
            *[
                F.col(c).alias(f"{c}_{label}")
                for c in [
                    "ep_count", "avg_duration", "max_duration",
                    "avg_balance_entry", "avg_balance_slope",
                    "avg_util_entry", "avg_util_slope",
                    "avg_limit_change", "avg_financed_entry",
                    "total_dip_count", "avg_dip_magnitude", "avg_balance_vol",
                ]
            ]
        )
        stage_pivots.append(stage_df)

    # Join all stages into one wide table per customer
    result = stage_pivots[0]
    for sp in stage_pivots[1:]:
        result = result.join(sp, on=REF_COL, how="full")

    return result


# ============================================================================
# SECTION 3 — WITHIN-STAGE LOAN COUNT DYNAMICS
# Number of active tradelines entering/during/exiting each stage
# ============================================================================

def build_within_stage_loan_count_dynamics(
    episode_df: DataFrame,
    history_df: DataFrame,
) -> DataFrame:
    """
    Compute loan count dynamics within and across stage episodes.

    Key insight: A customer who ADDS tradelines while in S2/S3 is behaving
    differently from one who is closing accounts. New loan seeking under
    stress = strategic or desperate. Account closures during S0 = deleveraging.

    Features per stage:
        num_loans_entry_{stage}:   Tradeline count on stage entry
        num_loans_exit_{stage}:    Tradeline count on episode exit
        num_loans_change_{stage}:  Net change in tradelines during stage
        num_loans_opened_{stage}:  New tradelines added during episode
        num_loans_closed_{stage}:  Tradelines closed/written off during episode
        loan_count_volatility_{stage}: Std dev of tradeline count during episode
    """
    # Monthly tradeline count
    monthly_loans = history_df.groupBy(REF_COL, DATE_COL).agg(
        F.countDistinct("seq_tl").alias("active_tradeline_count"),
    )

    w_ep_ord = Window.partitionBy(REF_COL, "episode_id").orderBy(DATE_COL)
    w_ep_full = Window.partitionBy(REF_COL, "episode_id")

    df = episode_df.select(REF_COL, DATE_COL, "episode_id", "episode_stage",
                            "episode_month_number").join(
        monthly_loans, on=[REF_COL, DATE_COL], how="left"
    )

    df = df.withColumn(
        "loan_count_entry",
        F.first("active_tradeline_count").over(w_ep_ord)
    ).withColumn(
        "loan_count_exit",
        F.last("active_tradeline_count").over(
            w_ep_ord.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
        )
    ).withColumn(
        "loan_mom_change",
        F.col("active_tradeline_count") - F.lag("active_tradeline_count", 1).over(w_ep_ord)
    ).withColumn(
        "loans_opened_this_month",
        F.when(F.col("loan_mom_change") > 0, F.col("loan_mom_change")).otherwise(F.lit(0))
    ).withColumn(
        "loans_closed_this_month",
        F.when(F.col("loan_mom_change") < 0, F.abs(F.col("loan_mom_change"))).otherwise(F.lit(0))
    )

    episode_loan_summary = df.filter(F.col("episode_month_number") == 1).select(
        REF_COL, "episode_id", "episode_stage", "loan_count_entry",
        "loan_count_exit",
        (F.col("loan_count_exit") - F.col("loan_count_entry")).alias("loan_count_change"),
    ).join(
        df.groupBy(REF_COL, "episode_id").agg(
            F.sum("loans_opened_this_month").alias("loans_opened_in_ep"),
            F.sum("loans_closed_this_month").alias("loans_closed_in_ep"),
            F.stddev("active_tradeline_count").alias("loan_count_vol"),
        ),
        on=[REF_COL, "episode_id"], how="left"
    )

    agg_by_stage = episode_loan_summary.groupBy(REF_COL, "episode_stage").agg(
        F.avg("loan_count_entry").alias("avg_loans_entry"),
        F.avg("loan_count_exit").alias("avg_loans_exit"),
        F.avg("loan_count_change").alias("avg_loans_change"),
        F.sum("loans_opened_in_ep").alias("total_loans_opened"),
        F.sum("loans_closed_in_ep").alias("total_loans_closed"),
        F.avg("loan_count_vol").alias("avg_loan_count_vol"),
    )

    stage_pivots = []
    for stage, label in STAGE_LABELS.items():
        stage_df = agg_by_stage.filter(F.col("episode_stage") == stage).select(
            REF_COL,
            *[
                F.col(c).alias(f"{c}_{label}")
                for c in [
                    "avg_loans_entry", "avg_loans_exit", "avg_loans_change",
                    "total_loans_opened", "total_loans_closed", "avg_loan_count_vol",
                ]
            ]
        )
        stage_pivots.append(stage_df)

    result = stage_pivots[0]
    for sp in stage_pivots[1:]:
        result = result.join(sp, on=REF_COL, how="full")

    return result


# ============================================================================
# SECTION 4 — DPD COUNTER DYNAMICS WITHIN STAGE
# How fast does DPD move through the bucket? Approaching boundary vs retreating?
# ============================================================================

def build_within_stage_dpd_dynamics(
    episode_df: DataFrame,
) -> DataFrame:
    """
    Compute DPD counter dynamics within each stage episode.

    Key insight: A customer at S2 with DPD=32 (just entered) vs DPD=88
    (approaching S3 boundary) represents very different risk profiles even
    though both are "S2". The velocity and position within the stage reveals
    cure probability and roll-forward risk.

    Features per stage:
        dpd_at_entry_{stage}:     DPD value on stage entry
        dpd_at_exit_{stage}:      DPD value at episode end
        dpd_slope_within_{stage}: DPD slope while in stage (+ = deteriorating)
        dpd_max_within_{stage}:   Maximum DPD reached within stage
        dpd_boundary_proximity_{stage}: (boundary_dpd - current_dpd) / stage_width
        dpd_retreat_count_{stage}: # months DPD decreased within stage
        dpd_advance_count_{stage}: # months DPD increased within stage
    """
    # Stage DPD boundaries (upper boundary = roll-forward threshold)
    STAGE_UPPER_BOUNDARY = {"S0": 30, "S1": 90, "S2": 90, "S3": 180, "S4": 999}
    STAGE_WIDTH = {"S0": 30, "S1": 29, "S2": 60, "S3": 89, "S4": 820}

    w_ep_ord = Window.partitionBy(REF_COL, "episode_id").orderBy(DATE_COL)
    w_ep_full = Window.partitionBy(REF_COL, "episode_id")

    df = episode_df.withColumn(
        "dpd_prev",
        F.lag(DPD_COL, 1).over(w_ep_ord)
    ).withColumn(
        "dpd_mom_change",
        F.col(DPD_COL) - F.col("dpd_prev")
    ).withColumn(
        "dpd_retreat",
        F.when(F.col("dpd_mom_change") < 0, 1).otherwise(0)
    ).withColumn(
        "dpd_advance",
        F.when(F.col("dpd_mom_change") > 0, 1).otherwise(0)
    )

    df = df.withColumn(
        "ep_dpd_entry",
        F.first(DPD_COL).over(w_ep_ord)
    ).withColumn(
        "ep_dpd_exit",
        F.last(DPD_COL).over(
            w_ep_ord.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
        )
    ).withColumn(
        "ep_dpd_max",
        F.max(DPD_COL).over(w_ep_full)
    ).withColumn(
        "ep_dpd_slope",
        _safe_div(
            F.last(DPD_COL).over(
                w_ep_ord.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
            ) - F.first(DPD_COL).over(w_ep_ord),
            F.count("*").over(w_ep_full),
            F.lit(0.0)
        )
    ).withColumn(
        "ep_retreat_count",
        F.sum("dpd_retreat").over(w_ep_full)
    ).withColumn(
        "ep_advance_count",
        F.sum("dpd_advance").over(w_ep_full)
    )

    # Compute boundary proximity for each stage (vectorised)
    stage_when = F.lit(None).cast("double")
    for stage, upper in STAGE_UPPER_BOUNDARY.items():
        width = STAGE_WIDTH[stage]
        stage_when = F.when(
            F.col("episode_stage") == stage,
            (F.lit(upper) - F.col("ep_dpd_exit")) / F.lit(float(width))
        ).otherwise(stage_when)

    df = df.withColumn("ep_boundary_proximity", stage_when)

    episode_dpd = df.filter(F.col("episode_month_number") == 1).select(
        REF_COL, "episode_id", "episode_stage",
        "ep_dpd_entry", "ep_dpd_exit", "ep_dpd_max", "ep_dpd_slope",
        "ep_retreat_count", "ep_advance_count", "ep_boundary_proximity",
    )

    agg_by_stage = episode_dpd.groupBy(REF_COL, "episode_stage").agg(
        F.avg("ep_dpd_entry").alias("avg_dpd_at_entry"),
        F.avg("ep_dpd_exit").alias("avg_dpd_at_exit"),
        F.avg("ep_dpd_slope").alias("avg_dpd_slope_within"),
        F.max("ep_dpd_max").alias("max_dpd_within"),
        F.avg("ep_boundary_proximity").alias("avg_boundary_proximity"),
        F.sum("ep_retreat_count").alias("total_dpd_retreats"),
        F.sum("ep_advance_count").alias("total_dpd_advances"),
        _safe_div(
            F.sum("ep_retreat_count"),
            F.sum("ep_retreat_count") + F.sum("ep_advance_count"),
            F.lit(0.0)
        ).alias("retreat_ratio"),
    )

    stage_pivots = []
    for stage, label in STAGE_LABELS.items():
        stage_df = agg_by_stage.filter(F.col("episode_stage") == stage).select(
            REF_COL,
            *[
                F.col(c).alias(f"{c}_{label}")
                for c in [
                    "avg_dpd_at_entry", "avg_dpd_at_exit", "avg_dpd_slope_within",
                    "max_dpd_within", "avg_boundary_proximity",
                    "total_dpd_retreats", "total_dpd_advances", "retreat_ratio",
                ]
            ]
        )
        stage_pivots.append(stage_df)

    result = stage_pivots[0]
    for sp in stage_pivots[1:]:
        result = result.join(sp, on=REF_COL, how="full")

    return result


# ============================================================================
# SECTION 5 — CROSS-STAGE TRANSITION DYNAMICS
# Roll-forward, cure, oscillation between stages
# ============================================================================

def build_cross_stage_transition_dynamics(episode_df: DataFrame) -> DataFrame:
    """
    Compute cross-stage transitions: roll-forward rates, cure rates, oscillation.

    Each episode transition (S2 → S3, S3 → S2, etc.) is a data point.
    Aggregated across customer lifetime to produce structural risk signals.

    Features:
        roll_fwd_{from}_{to}:         Count of forward transitions (worsening)
        cure_from_{stage}:            Count of cures (backward transitions from stage)
        cure_rate_{stage}:            Cures / total episodes in stage
        oscillation_{stage}:          Times re-entered stage after having left
        avg_months_to_cure_{stage}:   Avg episode duration before backward transition
        avg_months_to_rollforward_{stage}: Avg duration before rolling to worse stage
        ever_in_{stage}:              Binary — ever reached this stage
        first_entry_month_{stage}:    First month the customer entered this stage
        last_exit_month_{stage}:      Most recent exit from this stage

    Roll-forward: S0→S1, S1→S2, S2→S3, S3→S4 (worsening)
    Cure:         S1→S0, S2→S1/S0, S3→S2/S1/S0 (improvement)
    """
    w = Window.partitionBy(REF_COL).orderBy("episode_id")

    # Get ordered episodes (one row per episode)
    episodes = episode_df.filter(F.col("episode_month_number") == 1).select(
        REF_COL, "episode_id", "episode_stage", DATE_COL,
        F.col("ep_episode_length") if "ep_episode_length" in episode_df.columns
        else F.lit(None).cast("int").alias("ep_episode_length")
    )

    episodes = episodes.withColumn(
        "next_stage",
        F.lead("episode_stage", 1).over(w)
    ).withColumn(
        "prev_stage",
        F.lag("episode_stage", 1).over(w)
    ).withColumn(
        "stage_ordinal",
        F.when(F.col("episode_stage") == "S0", 0)
         .when(F.col("episode_stage") == "S1", 1)
         .when(F.col("episode_stage") == "S2", 2)
         .when(F.col("episode_stage") == "S3", 3)
         .when(F.col("episode_stage") == "S4", 4)
         .otherwise(F.lit(None))
    ).withColumn(
        "next_ordinal",
        F.when(F.col("next_stage") == "S0", 0)
         .when(F.col("next_stage") == "S1", 1)
         .when(F.col("next_stage") == "S2", 2)
         .when(F.col("next_stage") == "S3", 3)
         .when(F.col("next_stage") == "S4", 4)
         .otherwise(F.lit(None))
    )

    # Roll-forward: next_ordinal > current_ordinal
    episodes = episodes.withColumn(
        "is_roll_forward",
        F.when(F.col("next_ordinal") > F.col("stage_ordinal"), 1).otherwise(0)
    ).withColumn(
        "is_cure",
        F.when(F.col("next_ordinal") < F.col("stage_ordinal"), 1).otherwise(0)
    ).withColumn(
        "is_same_stage_return",
        F.when(
            (F.col("episode_stage") == F.col("next_stage")) |
            (F.col("episode_stage") == F.col("prev_stage")), 1
        ).otherwise(0)
    )

    # Aggregate per customer-stage
    agg = episodes.groupBy(REF_COL, "episode_stage").agg(
        F.count("episode_id").alias("episode_count"),
        F.sum("is_roll_forward").alias("roll_forward_count"),
        F.sum("is_cure").alias("cure_count"),
        F.min(DATE_COL).alias("first_entry_month"),
        F.max(DATE_COL).alias("last_entry_month"),
        _safe_div(
            F.sum("is_cure"), F.count("episode_id"), F.lit(0.0)
        ).alias("cure_rate"),
        _safe_div(
            F.sum("is_roll_forward"), F.count("episode_id"), F.lit(0.0)
        ).alias("roll_forward_rate"),
    ).withColumn("ever_in_stage", F.lit(1))

    stage_pivots = []
    for stage, label in STAGE_LABELS.items():
        stage_df = agg.filter(F.col("episode_stage") == stage).select(
            REF_COL,
            *[
                F.col(c).alias(f"{c}_{label}")
                for c in [
                    "episode_count", "roll_forward_count", "cure_count",
                    "first_entry_month", "last_entry_month",
                    "cure_rate", "roll_forward_rate", "ever_in_stage",
                ]
            ]
        )
        stage_pivots.append(stage_df)

    result = stage_pivots[0]
    for sp in stage_pivots[1:]:
        result = result.join(sp, on=REF_COL, how="full")

    # Add binary ever_in flags (default 0 for stages never visited)
    for stage, label in STAGE_LABELS.items():
        col = f"ever_in_stage_{label}"
        result = result.withColumn(col, F.coalesce(F.col(col), F.lit(0)))

    return result


# ============================================================================
# SECTION 6 — CROSS-LENDER CONSISTENCY (CardX vs Bureau delinquency alignment)
# ============================================================================

def build_cross_lender_consistency(
    cardx_monthly_df: DataFrame,
    bureau_state_df: DataFrame,
) -> DataFrame:
    """
    Compute alignment between CardX DPD and bureau DPD across months.

    Key insight: A customer who is delinquent at CardX but current at all other
    lenders is a "CardX selective defaulter" — they CAN pay (bureau current) but
    CHOOSE not to pay CardX. This is the most important cross-lender signal for
    recovery strategy.

    The inverse (delinquent at bureau, current at CardX) = systemic distress,
    CardX is protected but customer is overwhelmed overall.

    Features:
        cardx_delinquent_bureau_current_months: # months CardX S2+ but bureau S0
        bureau_delinquent_cardx_current_months: # months bureau S2+ but CardX S0
        aligned_delinquency_months:             Both CardX and bureau in S2+
        aligned_current_months:                 Both in S0
        cross_lender_divergence_score:          Mismatch ratio (0=aligned, 1=opposite)
        cardx_selective_default_flag:           1 if selective default pattern present
        systemic_stress_flag:                   1 if bureau-led stress pattern

    Requires:
        cardx_monthly_df: ref_no, asofdate, cardx_dpd, cardx_state
        bureau_state_df:  ref_no, asofdate, bureau_max_dpd, dpd_state (bureau)
    """
    if cardx_monthly_df is None:
        # No CardX data — return empty DataFrame with schema
        return bureau_state_df.select(REF_COL).distinct().withColumn(
            "cross_lender_consistency_available", F.lit(0)
        )

    # Join CardX and bureau on ref_no + asofdate
    joined = cardx_monthly_df.select(
        REF_COL, DATE_COL,
        F.col("cardx_dpd"),
        F.col("cardx_state").alias("cardx_stage"),
    ).join(
        bureau_state_df.select(REF_COL, DATE_COL, DPD_COL, F.col(STATE_COL).alias("bureau_stage")),
        on=[REF_COL, DATE_COL], how="inner"
    )

    # Classify joint state
    joined = joined.withColumn(
        "cardx_stressed",
        F.when(F.col("cardx_stage").isin(["S2", "S3", "S4"]), 1).otherwise(0)
    ).withColumn(
        "bureau_stressed",
        F.when(F.col("bureau_stage").isin(["S2", "S3", "S4"]), 1).otherwise(0)
    ).withColumn(
        "cardx_current",
        F.when(F.col("cardx_stage") == "S0", 1).otherwise(0)
    ).withColumn(
        "bureau_current",
        F.when(F.col("bureau_stage") == "S0", 1).otherwise(0)
    )

    joined = joined.withColumn(
        "selective_default_month",        # CardX stressed, bureau current
        F.when((F.col("cardx_stressed") == 1) & (F.col("bureau_current") == 1), 1).otherwise(0)
    ).withColumn(
        "systemic_stress_month",          # Bureau stressed, CardX current
        F.when((F.col("bureau_stressed") == 1) & (F.col("cardx_current") == 1), 1).otherwise(0)
    ).withColumn(
        "aligned_delinquent_month",
        F.when((F.col("cardx_stressed") == 1) & (F.col("bureau_stressed") == 1), 1).otherwise(0)
    ).withColumn(
        "aligned_current_month",
        F.when((F.col("cardx_current") == 1) & (F.col("bureau_current") == 1), 1).otherwise(0)
    )

    result = joined.groupBy(REF_COL).agg(
        F.count("*").alias("months_observed"),
        F.sum("selective_default_month").alias("cardx_selective_default_months"),
        F.sum("systemic_stress_month").alias("systemic_stress_months"),
        F.sum("aligned_delinquent_month").alias("aligned_delinquency_months"),
        F.sum("aligned_current_month").alias("aligned_current_months"),
        _safe_div(
            F.sum("selective_default_month"), F.count("*"), F.lit(0.0)
        ).alias("selective_default_rate"),
        _safe_div(
            F.sum("systemic_stress_month"), F.count("*"), F.lit(0.0)
        ).alias("systemic_stress_rate"),
    ).withColumn(
        "cross_lender_divergence_score",
        # Higher = more misalignment between CardX and bureau
        (F.col("cardx_selective_default_months") + F.col("systemic_stress_months"))
        / (F.col("months_observed") + 1)
    ).withColumn(
        "cardx_selective_default_flag",
        F.when(F.col("selective_default_rate") > 0.25, 1).otherwise(0)
    ).withColumn(
        "systemic_stress_flag",
        F.when(F.col("systemic_stress_rate") > 0.25, 1).otherwise(0)
    ).withColumn("cross_lender_consistency_available", F.lit(1))

    return result


# ============================================================================
# SECTION 7 — SECURED vs UNSECURED DEBT PRIORITISATION
# ============================================================================

def build_debt_prioritisation_features(
    history_df: DataFrame,
    account_df: DataFrame,
) -> DataFrame:
    """
    Compute secured vs unsecured repayment behaviour.

    Key insight for recovery: A customer who keeps paying their mortgage
    (secured) while defaulting on unsecured cards is rational — they are
    protecting the asset. This is strong evidence of strategic capacity.
    The question for recovery is whether residual cash flow exists after
    secured obligations are serviced.

    Balance dip (payment proxy) is computed separately for secured vs
    unsecured tradelines to reveal the prioritisation order.

    Features:
        secured_balance_at_co:       Total secured debt at charge-off
        unsecured_balance_at_co:     Total unsecured debt at charge-off
        secured_to_unsecured_ratio:  Secured debt fraction
        secured_dip_count_12m:       Balance dips on secured tradelines (12M pre-CO)
        unsecured_dip_count_12m:     Balance dips on unsecured tradelines (12M pre-CO)
        secured_payment_priority_score: secured_dip_rate / (secured + unsecured dip rate)
        mortgage_current_flag:       Any mortgage currently active and current
        auto_loan_current_flag:      Any auto loan currently active and current
        secured_delinquency_flag:    Any secured tradeline in S2+ (severe stress)
    """
    # Classify secured vs unsecured from lender type (approximation)
    SECURED_TYPES = ["MORTGAGE", "AUTO", "SFI_SECURED", "LEASING"]
    UNSECURED_TYPES = ["CARDX", "CREDIT_CARD", "PERSONAL_LOAN", "FINTECH"]

    from pyspark.sql.functions import udf
    from pyspark.sql.types import StringType

    def _classify_secured(lender_name: Optional[str]) -> str:
        if not lender_name:
            return "UNKNOWN"
        ln = lender_name.upper()
        secured_keywords = ["MORTGAGE", "GH BANK", "HOUSING", "TOYOTA", "ISUZU",
                           "HONDA", "NISSAN", "AYUDHYA CAPITAL", "ORIX",
                           "SRISAWAD", "LEASING"]
        for kw in secured_keywords:
            if kw in ln:
                return "SECURED"
        return "UNSECURED"

    classify_udf = udf(_classify_secured, StringType())

    typed = history_df.withColumn(
        "debt_class",
        classify_udf(F.col(LENDER_COL))
    )

    # Month-over-month balance by debt class
    w_cust_date_class = Window.partitionBy(REF_COL, "debt_class").orderBy(DATE_COL)

    typed = typed.withColumn(
        "bal_prev",
        F.lag(BAL_COL, 1).over(w_cust_date_class)
    ).withColumn(
        "bal_dip",
        F.when(F.col(BAL_COL) < F.col("bal_prev"), 1).otherwise(0)
    )

    # 12-month window (most recent 12 months in history)
    w12 = Window.partitionBy(REF_COL, "debt_class").orderBy(DATE_COL).rowsBetween(-11, 0)

    typed = typed.withColumn("dip_count_12m", F.sum("bal_dip").over(w12))

    # Snapshot at most recent date
    w_latest = Window.partitionBy(REF_COL, "debt_class").orderBy(F.col(DATE_COL).desc())
    latest = typed.withColumn("rn", F.row_number().over(w_latest)).filter(F.col("rn") == 1)

    secured_df = latest.filter(F.col("debt_class") == "SECURED").select(
        REF_COL,
        F.col(BAL_COL).alias("secured_balance_at_co"),
        F.col("dip_count_12m").alias("secured_dip_count_12m"),
    )
    unsecured_df = latest.filter(F.col("debt_class") == "UNSECURED").select(
        REF_COL,
        F.col(BAL_COL).alias("unsecured_balance_at_co"),
        F.col("dip_count_12m").alias("unsecured_dip_count_12m"),
    )

    result = secured_df.join(unsecured_df, on=REF_COL, how="full")

    result = result.withColumn(
        "total_debt_at_co",
        F.coalesce(F.col("secured_balance_at_co"), F.lit(0.0)) +
        F.coalesce(F.col("unsecured_balance_at_co"), F.lit(0.0))
    ).withColumn(
        "secured_to_total_ratio",
        _safe_div(F.col("secured_balance_at_co"), F.col("total_debt_at_co"), F.lit(0.0))
    ).withColumn(
        "secured_payment_priority_score",
        _safe_div(
            F.col("secured_dip_count_12m"),
            F.col("secured_dip_count_12m") + F.col("unsecured_dip_count_12m"),
            F.lit(0.5)
        )
    ).withColumn(
        "debt_prioritization_available", F.lit(1)
    )

    return result


# ============================================================================
# SECTION 8 — PRE-EXISTING BUREAU STRESS (before CardX deterioration)
# ============================================================================

def build_pre_existing_stress_features(
    episode_df: DataFrame,
    cardx_first_delinquency_df: Optional[DataFrame] = None,
) -> DataFrame:
    """
    Detect whether bureau stress preceded CardX deterioration.

    Critical question: Did the customer have bureau problems BEFORE their
    CardX account deteriorated? If yes, CardX deterioration is part of a
    systemic slide (Structural Defaulter). If bureau was clean before
    CardX deteriorated, it may be an isolated shock (Sudden-Shock Distressed).

    Features:
        bureau_stress_pre_cardx_months: Months bureau was in S2+ before
                                         CardX first hit S1
        pre_existing_stress_flag:       1 if bureau was S2+ before CardX S1
        bureau_stress_lead_months:      How many months bureau preceded CardX
                                         (negative = bureau lagged CardX)
        bureau_clean_before_cardx_flag: 1 if bureau was clean (S0) when CardX
                                         first deteriorated

    Args:
        episode_df:               Episode DataFrame from build_stage_episodes()
        cardx_first_delinquency_df: ref_no, cardx_first_delinquency_month
    """
    # First bureau entry into S2+ per customer
    bureau_first_stress = episode_df.filter(
        F.col("episode_stage").isin(["S2", "S3", "S4"])
    ).filter(
        F.col("episode_month_number") == 1
    ).groupBy(REF_COL).agg(
        F.min(DATE_COL).alias("bureau_first_stress_month")
    )

    if cardx_first_delinquency_df is None:
        # No CardX data — return bureau stress timing only
        return bureau_first_stress.withColumn(
            "pre_existing_stress_available", F.lit(0)
        )

    result = bureau_first_stress.join(
        cardx_first_delinquency_df.select(REF_COL, "cardx_first_delinquency_month"),
        on=REF_COL, how="left"
    )

    result = result.withColumn(
        "bureau_stress_lead_months",
        F.datediff(
            F.col("cardx_first_delinquency_month").cast("date"),
            F.col("bureau_first_stress_month").cast("date")
        ) / 30   # approximate months
    ).withColumn(
        "pre_existing_stress_flag",
        F.when(F.col("bureau_stress_lead_months") > 0, 1).otherwise(0)
    ).withColumn(
        "bureau_clean_before_cardx_flag",
        F.when(F.col("bureau_first_stress_month") > F.col("cardx_first_delinquency_month"), 1).otherwise(0)
    ).withColumn("pre_existing_stress_available", F.lit(1))

    return result


# ============================================================================
# SECTION 9 — MAIN ASSEMBLY
# ============================================================================

def run_stage_dynamics(
    spark: SparkSession,
    state_df: DataFrame,
    history_df: DataFrame,
    account_df: DataFrame,
    cardx_monthly_df: Optional[DataFrame] = None,
    cardx_first_delinquency_df: Optional[DataFrame] = None,
) -> DataFrame:
    """
    Run complete stage dynamics pipeline and return joined feature set.

    Execution order:
        1. build_stage_episodes          — episode ID assignment
        2. build_within_stage_exposure_dynamics  — balance/limit/financed per stage
        3. build_within_stage_loan_count_dynamics — tradeline count per stage
        4. build_within_stage_dpd_dynamics        — DPD counter within stage
        5. build_cross_stage_transition_dynamics  — roll-fwd / cure rates
        6. build_cross_lender_consistency         — CardX vs bureau alignment
        7. build_debt_prioritisation_features     — secured vs unsecured
        8. build_pre_existing_stress_features     — bureau stress before CardX

    Returns: Wide DataFrame keyed on ref_no with ~120 features.
    """
    print("[Stage Dynamics] Building stage episodes...")
    episode_df = build_stage_episodes(state_df)

    print("[Stage Dynamics] Exposure dynamics within stage...")
    exposure_feats = build_within_stage_exposure_dynamics(episode_df, history_df)

    print("[Stage Dynamics] Loan count dynamics within stage...")
    loan_feats = build_within_stage_loan_count_dynamics(episode_df, history_df)

    print("[Stage Dynamics] DPD counter dynamics within stage...")
    dpd_feats = build_within_stage_dpd_dynamics(episode_df)

    print("[Stage Dynamics] Cross-stage transitions...")
    transition_feats = build_cross_stage_transition_dynamics(episode_df)

    print("[Stage Dynamics] Cross-lender consistency...")
    cross_lender_feats = build_cross_lender_consistency(cardx_monthly_df, state_df)

    print("[Stage Dynamics] Debt prioritisation (secured vs unsecured)...")
    debt_feats = build_debt_prioritisation_features(history_df, account_df)

    print("[Stage Dynamics] Pre-existing bureau stress...")
    pre_stress_feats = build_pre_existing_stress_features(episode_df, cardx_first_delinquency_df)

    # Join all on ref_no
    result = exposure_feats
    for feat_df, name in [
        (loan_feats,        "loan_count"),
        (dpd_feats,         "dpd_dynamics"),
        (transition_feats,  "transitions"),
        (cross_lender_feats,"cross_lender"),
        (debt_feats,        "debt_priority"),
        (pre_stress_feats,  "pre_stress"),
    ]:
        before = result.count()
        result = result.join(feat_df, on=REF_COL, how="full")
        after  = result.count()
        print(f"  ✓ Joined {name}: {before:,} → {after:,} rows")

    result = result.dropDuplicates([REF_COL])

    n_features = len(result.columns) - 1  # subtract ref_no
    print(f"[Stage Dynamics] Complete — {n_features} features, {result.count():,} accounts")

    return result
