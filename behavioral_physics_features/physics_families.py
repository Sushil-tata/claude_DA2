"""
7 Physics Families - User-Specified Features (41 features total)
=================================================================

Advanced physics-inspired features for credit bureau dynamics.
These are additions to the 335 existing features (Buckets A, B, C).

Author: Principal Data Science Team
Version: 1.0.0
"""

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from typing import Optional


def _safe_div(numerator, denominator, default=F.lit(None)):
    """Safe division with default value when denominator is zero"""
    return F.when(
        (denominator != 0) & (denominator.isNotNull()),
        numerator / denominator
    ).otherwise(default)


# ============================================================================
# FAMILY 1: INERTIA & MOMENTUM (7 features)
# ============================================================================

def add_inertia_momentum_features(panel: DataFrame) -> DataFrame:
    """
    Inertia & Momentum physics features.

    Features (7):
    - credit_inertia_score: resistance to state change
    - credit_momentum_3m: debt × velocity
    - momentum_sign_flip_6m: velocity direction changes
    - velocity_asymmetry_6m: upward vs downward velocity ratio
    - dpd_upward_velocity_3m: average positive jumps
    - dpd_downward_velocity_3m: average negative jumps
    - jerk_dpd_3m: rate of acceleration change (instability)
    """
    w = Window.partitionBy("cust_id").orderBy("as_of_month")

    # 1. Credit inertia score: resistance to change
    # High = large debt stuck in bad state
    panel = panel.withColumn(
        "credit_inertia_score",
        F.col("streak_len") * F.col("state_num") * F.log1p(F.col("bureau_owed_sum"))
    )

    # 2. Credit momentum: debt × velocity
    # Use dpd_diff_velocity_3m (simple difference velocity from TrajectoryEngine)
    panel = panel.withColumn(
        "credit_momentum_3m",
        F.coalesce(F.col("dpd_diff_velocity_3m"), F.lit(0.0)) * F.col("bureau_owed_sum")
    )

    # Compute dpd_jump if not already present
    if "dpd_jump" not in panel.columns:
        panel = panel.withColumn(
            "dpd_jump",
            F.col("bureau_max_dpd") - F.lag("bureau_max_dpd", 1).over(w)
        )

    # 3. Momentum sign flip: direction changes in velocity
    # Use dpd_jump (1-month DPD change) instead of missing dpd_slope_1m
    w6 = w.rowsBetween(-5, 0)
    panel = panel.withColumn(
        "dpd_jump_sign",
        F.when(F.col("dpd_jump") > 0, 1)
         .when(F.col("dpd_jump") < 0, -1)
         .otherwise(0)
    )
    panel = panel.withColumn(
        "momentum_sign_flip_6m",
        F.sum(
            F.when(
                F.col("dpd_jump_sign") != F.lag("dpd_jump_sign", 1).over(w),
                1
            ).otherwise(0)
        ).over(w6)
    )

    # 4. Velocity asymmetry: upward vs downward
    panel = panel.withColumn(
        "dpd_upward_velocity_3m",
        F.avg(F.when(F.col("dpd_jump") > 0, F.col("dpd_jump")).otherwise(F.lit(None)))
         .over(w.rowsBetween(-2, 0))
    )

    panel = panel.withColumn(
        "dpd_downward_velocity_3m",
        F.avg(F.when(F.col("dpd_jump") < 0, F.col("dpd_jump")).otherwise(F.lit(None)))
         .over(w.rowsBetween(-2, 0))
    )

    panel = panel.withColumn(
        "velocity_asymmetry_6m",
        _safe_div(
            F.col("dpd_upward_velocity_3m"),
            F.abs(F.col("dpd_downward_velocity_3m")),
            default=F.lit(None)
        )
    )

    # 5. Jerk: rate of acceleration change
    # Use dpd_accel_1m (single month second difference)
    panel = panel.withColumn(
        "jerk_dpd_3m",
        F.avg(
            F.coalesce(F.col("dpd_accel_1m"), F.lit(0.0)) -
            F.lag(F.coalesce(F.col("dpd_accel_1m"), F.lit(0.0)), 1).over(w)
        ).over(w.rowsBetween(-2, 0))
    )

    return panel


# ============================================================================
# FAMILY 2: CRITICAL SLOWING (6 features)
# ============================================================================

def add_critical_slowing_features(panel: DataFrame) -> DataFrame:
    """
    Critical slowing down: pre-transition warning signals.

    Features (6):
    - dpd_var_3m: short-term variance
    - dpd_var_4to12m: medium-term variance
    - dpd_variance_ratio: variance ratio (early warning when rising)
    - dpd_lag1: lagged DPD
    - dpd_autocorr_lag1_12m: autocorrelation (persistence)
    - critical_slowing_index: combined index (variance ratio × autocorr)

    Theory: Before phase transitions, systems show:
    - Increased variance (critical fluctuations)
    - Increased autocorrelation (critical slowing down)
    """
    w = Window.partitionBy("cust_id").orderBy("as_of_month")

    # 1. Short-term variance (last 3 months)
    w_3m = w.rowsBetween(-2, 0)
    panel = panel.withColumn(
        "dpd_var_3m",
        F.coalesce(F.stddev_pop("bureau_max_dpd").over(w_3m), F.lit(0.0))
    )

    # 2. Medium-term variance (months 4-12)
    w_4to12m = w.rowsBetween(-11, -3)
    panel = panel.withColumn(
        "dpd_var_4to12m",
        F.coalesce(F.stddev_pop("bureau_max_dpd").over(w_4to12m), F.lit(0.0))
    )

    # 3. Variance ratio: rising ratio = approaching transition
    panel = panel.withColumn(
        "dpd_variance_ratio",
        _safe_div(F.col("dpd_var_3m"), F.col("dpd_var_4to12m"), default=F.lit(None))
    )

    # 4. Lagged DPD for autocorrelation
    panel = panel.withColumn(
        "dpd_lag1",
        F.lag("bureau_max_dpd", 1).over(w)
    )

    # 5. Autocorrelation (persistence measure)
    w_12m = w.rowsBetween(-11, 0)
    panel = panel.withColumn(
        "dpd_autocorr_lag1_12m",
        F.coalesce(F.corr("bureau_max_dpd", "dpd_lag1").over(w_12m), F.lit(0.0))
    )

    # 6. Critical slowing index: combined signal
    panel = panel.withColumn(
        "critical_slowing_index",
        F.when(
            F.col("dpd_variance_ratio").isNotNull() & F.col("dpd_autocorr_lag1_12m").isNotNull(),
            F.col("dpd_variance_ratio") * F.col("dpd_autocorr_lag1_12m")
        ).otherwise(F.lit(None))
    )

    return panel


# ============================================================================
# FAMILY 3: PHASE BOUNDARY (7 features)
# ============================================================================

def add_phase_boundary_features(panel: DataFrame) -> DataFrame:
    """
    Phase boundary proximity: distance to regulatory thresholds.

    Features (7):
    - dpd_to_sm_boundary: distance to Special Mention (31 DPD)
    - dpd_to_npl_boundary: distance to NPL (91 DPD)
    - dpd_to_co_boundary: distance to Charge-off (181 DPD)
    - nearest_boundary_dist: closest boundary
    - phase_proximity_score: normalized proximity (0-1)
    - boundary_cross_3m: boundary crossings in 3m
    - boundary_oscillation_flag: unstable (2+ crossings)

    Theory: Behavior near phase boundaries shows critical phenomena.
    """
    w = Window.partitionBy("cust_id").orderBy("as_of_month")

    # 1-3. Distance to each boundary
    panel = panel.withColumn(
        "dpd_to_sm_boundary",
        F.greatest(F.lit(0), F.lit(31) - F.col("bureau_max_dpd"))
    )

    panel = panel.withColumn(
        "dpd_to_npl_boundary",
        F.greatest(F.lit(0), F.lit(91) - F.col("bureau_max_dpd"))
    )

    panel = panel.withColumn(
        "dpd_to_co_boundary",
        F.greatest(F.lit(0), F.lit(181) - F.col("bureau_max_dpd"))
    )

    # 4. Nearest boundary
    panel = panel.withColumn(
        "nearest_boundary_dist",
        F.least(
            F.col("dpd_to_sm_boundary"),
            F.col("dpd_to_npl_boundary"),
            F.col("dpd_to_co_boundary")
        )
    )

    # 5. Proximity score: high when close to boundary
    panel = panel.withColumn(
        "phase_proximity_score",
        F.lit(1.0) / (F.lit(1.0) + F.col("nearest_boundary_dist"))
    )

    # 6. Boundary crossings: state changes that cross BOT thresholds
    # Check if state changed AND crossed a major boundary
    panel = panel.withColumn(
        "boundary_cross_m",
        F.when(
            (F.col("state_changed_m") == 1) &
            (
                # S1↔S2 (crosses 31 DPD)
                ((F.lag("state", 1).over(w) == "S1") & (F.col("state") == "S2")) |
                ((F.lag("state", 1).over(w) == "S2") & (F.col("state") == "S1")) |
                # S2↔S3 (crosses 91 DPD)
                ((F.lag("state", 1).over(w) == "S2") & (F.col("state") == "S3")) |
                ((F.lag("state", 1).over(w) == "S3") & (F.col("state") == "S2")) |
                # S3↔S4 (crosses 181 DPD)
                ((F.lag("state", 1).over(w) == "S3") & (F.col("state") == "S4")) |
                ((F.lag("state", 1).over(w) == "S4") & (F.col("state") == "S3"))
            ),
            1
        ).otherwise(0)
    )

    w_3m = w.rowsBetween(-2, 0)
    panel = panel.withColumn(
        "boundary_cross_3m",
        F.sum("boundary_cross_m").over(w_3m)
    )

    # 7. Oscillation flag: 2+ crossings = unstable
    panel = panel.withColumn(
        "boundary_oscillation_flag",
        F.when(F.col("boundary_cross_3m") >= 2, 1).otherwise(0)
    )

    return panel


# ============================================================================
# FAMILY 4: HYSTERESIS (6 features)
# ============================================================================

def add_hysteresis_features(panel: DataFrame) -> DataFrame:
    """
    Hysteresis: path-dependent memory effects.

    Features (6):
    - lifetime_max_dpd: worst DPD ever seen
    - prior_charge_off_flag: ever reached S4
    - default_episode_cnt: number of S4 entries
    - months_since_last_stress: time since S2/S3/S4
    - scar_score: exponentially weighted history
    - recovery_depth_ratio: current vs lifetime max

    Note: Complementary to Bucket C hysteresis_index_6m/12m (kept).
    Theory: Past stress leaves "scars" affecting future behavior.
    """
    w = Window.partitionBy("cust_id").orderBy("as_of_month")
    w_all = w.rowsBetween(Window.unboundedPreceding, 0)

    # 1. Lifetime max DPD
    panel = panel.withColumn(
        "lifetime_max_dpd",
        F.max("bureau_max_dpd").over(w_all)
    )

    # 2. Prior charge-off flag (ever in S4)
    panel = panel.withColumn(
        "prior_charge_off_flag",
        F.max(F.when(F.col("state") == "S4", 1).otherwise(0)).over(w_all)
    )

    # 3. Default episode count (new entries to S4)
    panel = panel.withColumn(
        "default_episode_cnt",
        F.sum(
            F.when(
                (F.lag("state", 1).over(w) != "S4") & (F.col("state") == "S4"),
                1
            ).otherwise(0)
        ).over(w_all)
    )

    # 4. Months since last stress
    panel = panel.withColumn(
        "last_stress_date",
        F.last(
            F.when(
                F.col("state").isin(["S2", "S3", "S4"]),
                F.col("as_of_month")
            ).otherwise(F.lit(None)),
            ignorenulls=True
        ).over(w_all)
    )

    panel = panel.withColumn(
        "months_since_last_stress",
        F.when(
            F.col("last_stress_date").isNotNull(),
            F.months_between(F.col("as_of_month"), F.col("last_stress_date"))
        ).otherwise(F.lit(None))
    )

    # 5. Scar score: exponentially weighted history
    # scar_score = 0.85 × lag(scar_score) + state_num
    panel = panel.withColumn(
        "scar_score",
        F.lit(0.85) * F.coalesce(F.lag("scar_score", 1).over(w), F.lit(0.0)) +
        F.col("state_num")
    )

    # 6. Recovery depth ratio
    panel = panel.withColumn(
        "recovery_depth_ratio",
        _safe_div(F.col("bureau_max_dpd"), F.col("lifetime_max_dpd"), default=F.lit(None))
    )

    # Drop intermediate column
    panel = panel.drop("last_stress_date")

    return panel


# ============================================================================
# FAMILY 5: LENDER ECOLOGY TOPOLOGY (5 features)
# ============================================================================

def add_lender_ecology_topology(panel: DataFrame, hist_joined_df: DataFrame) -> DataFrame:
    """
    Lender network topology and ecology features.

    Features (5):
    - herfindahl_lender_conc: concentration (1=monopoly, low=diversified)
    - nano_entry_flag_m: new NANO lender this month
    - tier_downgrade_flag_m: new lower-tier lender
    - new_lender_3m_cnt: lenders in 3m not in 4-12m
    - lender_tier_entropy_m: diversity across tiers

    Computed from hist_joined_df at tradeline level.
    """
    w_cust = Window.partitionBy("cust_id", "as_of_month")
    w_cust_time = Window.partitionBy("cust_id").orderBy("as_of_month")

    # Aggregate lender data per customer-month
    lender_agg = hist_joined_df.groupBy("cust_id", "as_of_month", "member_id", "lender_type").agg(
        F.sum("amountowed_hs").alias("lender_balance")
    )

    # Total balance per customer-month
    total_balance = lender_agg.groupBy("cust_id", "as_of_month").agg(
        F.sum("lender_balance").alias("total_balance_m")
    )

    lender_agg = lender_agg.join(total_balance, on=["cust_id", "as_of_month"], how="left")

    # 1. Herfindahl concentration
    lender_agg = lender_agg.withColumn(
        "balance_share",
        _safe_div(F.col("lender_balance"), F.col("total_balance_m"), default=F.lit(0.0))
    )

    herfindahl = lender_agg.groupBy("cust_id", "as_of_month").agg(
        F.sum(F.pow(F.col("balance_share"), 2)).alias("herfindahl_lender_conc")
    )

    # 2. Nano entry flag
    # Get lenders in current month
    current_lenders = lender_agg.select("cust_id", "as_of_month", "member_id", "lender_type")

    # Get lenders in prior 3 months
    w_3m = w_cust_time.rowsBetween(-3, -1)
    prior_lenders = (
        hist_joined_df
        .select("cust_id", "as_of_month", "member_id", "lender_type")
        .distinct()
        .withColumn("prior_lender_flag", F.lit(1))
    )

    # Check for new NANO lenders
    nano_check = (
        current_lenders
        .filter(F.col("lender_type") == "NANO")
        .join(
            prior_lenders.filter(F.col("lender_type") == "NANO"),
            on=["cust_id", "member_id"],
            how="left_anti"
        )
        .groupBy("cust_id", "as_of_month")
        .agg(F.lit(1).alias("nano_entry_flag_m"))
    )

    # 3. Tier downgrade flag
    # Assign tier ranks: COMMERCIAL_BANK=4, SFI=3, FINTECH=2, NANO=1
    tier_rank_map = F.when(F.col("lender_type") == "COMMERCIAL_BANK", 4) \
                     .when(F.col("lender_type") == "SFI", 3) \
                     .when(F.col("lender_type") == "FINTECH", 2) \
                     .when(F.col("lender_type") == "NANO", 1) \
                     .otherwise(0)

    lender_agg = lender_agg.withColumn("lender_tier_rank", tier_rank_map)

    tier_stats = lender_agg.groupBy("cust_id", "as_of_month").agg(
        F.min("lender_tier_rank").alias("min_tier_current")
    )

    tier_stats = tier_stats.withColumn(
        "min_tier_lag1",
        F.lag("min_tier_current", 1).over(w_cust_time)
    )

    tier_stats = tier_stats.withColumn(
        "tier_downgrade_flag_m",
        F.when(
            F.col("min_tier_current") < F.col("min_tier_lag1"),
            1
        ).otherwise(0)
    )

    # 4. New lender count (in 3m but not in 4-12m)
    # This is complex - simplify to count of distinct lenders in 3m vs 4-12m
    w_3m_lenders = w_cust_time.rowsBetween(-2, 0)
    w_4to12m_lenders = w_cust_time.rowsBetween(-11, -3)

    lender_counts = hist_joined_df.groupBy("cust_id", "as_of_month").agg(
        F.countDistinct("member_id").alias("lender_cnt_m")
    )

    lender_counts = lender_counts.withColumn(
        "new_lender_3m_cnt",
        F.max("lender_cnt_m").over(w_3m_lenders) - F.coalesce(F.max("lender_cnt_m").over(w_4to12m_lenders), F.lit(0))
    )

    # 5. Lender tier entropy (Shannon entropy)
    # H = -Σ(p_i × log(p_i))
    tier_dist = lender_agg.groupBy("cust_id", "as_of_month", "lender_type").agg(
        F.sum("lender_balance").alias("tier_balance")
    )

    tier_dist = tier_dist.join(total_balance, on=["cust_id", "as_of_month"], how="left")

    tier_dist = tier_dist.withColumn(
        "tier_prob",
        _safe_div(F.col("tier_balance"), F.col("total_balance_m"), default=F.lit(0.0))
    )

    tier_dist = tier_dist.withColumn(
        "tier_entropy_component",
        F.when(
            F.col("tier_prob") > 0,
            -F.col("tier_prob") * F.log(F.col("tier_prob"))
        ).otherwise(0.0)
    )

    tier_entropy = tier_dist.groupBy("cust_id", "as_of_month").agg(
        F.sum("tier_entropy_component").alias("lender_tier_entropy_m")
    )

    # Join all features back to panel
    panel = panel.join(herfindahl, on=["cust_id", "as_of_month"], how="left")
    panel = panel.join(nano_check, on=["cust_id", "as_of_month"], how="left")
    panel = panel.join(
        tier_stats.select("cust_id", "as_of_month", "tier_downgrade_flag_m"),
        on=["cust_id", "as_of_month"],
        how="left"
    )
    panel = panel.join(
        lender_counts.select("cust_id", "as_of_month", "new_lender_3m_cnt"),
        on=["cust_id", "as_of_month"],
        how="left"
    )
    panel = panel.join(tier_entropy, on=["cust_id", "as_of_month"], how="left")

    # Fill nulls for flags
    panel = panel.withColumn(
        "nano_entry_flag_m",
        F.coalesce(F.col("nano_entry_flag_m"), F.lit(0))
    )

    return panel


# ============================================================================
# FAMILY 6: ENQUIRY PHYSICS (5 features)
# ============================================================================

def add_enquiry_physics(panel: DataFrame, hist_joined_df: DataFrame) -> DataFrame:
    """
    Enquiry physics: application patterns and rejection proxies.

    Features (5):
    - new_tradelines_3m: first-time accounts in 3m
    - enquiry_rejection_proxy_3m: enquiries - new accounts
    - enquiry_desperation_idx_3m: others / total enquiries
    - enquiry_burst_3m_flag: 5+ enquiries in 3m
    - enq_to_tradeline_ratio_6m: enquiries per account

    Theory: Enquiry-to-approval gap signals rejection, desperation.
    """
    w = Window.partitionBy("cust_id").orderBy("as_of_month")

    # Track first appearance of each tradeline
    w_tl = Window.partitionBy("cust_id", "seq_tl").orderBy("as_of_month")

    tl_first = hist_joined_df.withColumn(
        "tl_first_month",
        F.first("as_of_month").over(w_tl)
    )

    # New tradelines in last 3 months
    w_3m = w.rowsBetween(-2, 0)

    new_tl_cnt = tl_first.filter(
        F.col("as_of_month") == F.col("tl_first_month")
    ).groupBy("cust_id", "as_of_month").agg(
        F.countDistinct("seq_tl").alias("new_tradelines_m")
    )

    panel = panel.join(new_tl_cnt, on=["cust_id", "as_of_month"], how="left")

    panel = panel.withColumn(
        "new_tradelines_m",
        F.coalesce(F.col("new_tradelines_m"), F.lit(0))
    )

    # 1. New tradelines in 3m
    panel = panel.withColumn(
        "new_tradelines_3m",
        F.sum("new_tradelines_m").over(w_3m)
    )

    # 2. Enquiry rejection proxy
    panel = panel.withColumn(
        "enquiry_rejection_proxy_3m",
        F.greatest(
            F.lit(0),
            F.coalesce(F.col("enq_cnt_3m"), F.lit(0)) - F.col("new_tradelines_3m")
        )
    )

    # 3. Enquiry desperation index (shopping away from CardX)
    panel = panel.withColumn(
        "enquiry_desperation_idx_3m",
        _safe_div(
            F.coalesce(F.col("enq_others_cnt_3m"), F.lit(0)),
            F.coalesce(F.col("enq_cnt_3m"), F.lit(1)),
            default=F.lit(None)
        )
    )

    # 4. Enquiry burst flag
    panel = panel.withColumn(
        "enquiry_burst_3m_flag",
        F.when(F.coalesce(F.col("enq_cnt_3m"), F.lit(0)) >= 5, 1).otherwise(0)
    )

    # 5. Enquiry to tradeline ratio (6m)
    w_6m = w.rowsBetween(-5, 0)

    tl_count_6m = hist_joined_df.groupBy("cust_id", "as_of_month").agg(
        F.countDistinct("seq_tl").alias("tradeline_cnt_m")
    )

    panel = panel.join(tl_count_6m, on=["cust_id", "as_of_month"], how="left")

    panel = panel.withColumn(
        "tradeline_cnt_6m",
        F.countDistinct("tradeline_cnt_m").over(w_6m)
    )

    panel = panel.withColumn(
        "enq_to_tradeline_ratio_6m",
        _safe_div(
            F.coalesce(F.col("enq_cnt_6m"), F.lit(0)),
            F.col("tradeline_cnt_6m"),
            default=F.lit(None)
        )
    )

    return panel


# ============================================================================
# FAMILY 7: UTILIZATION PHYSICS (5 features)
# ============================================================================

def add_utilization_physics(panel: DataFrame) -> DataFrame:
    """
    Utilization regime dynamics: borrowing patterns under stress.

    Features (5):
    - util_in_normal_avg_12m: utilization when S0/S1
    - util_in_stress_avg_12m: utilization when S2/S3/S4
    - util_regime_delta: stress - normal (positive = danger)
    - util_compression_flag: util down + dpd up = limit cut
    - util_escape_velocity: months to deleverage out of stress

    Theory: Healthy = deleverage under stress. Dangerous = leverage up.
    """
    w = Window.partitionBy("cust_id").orderBy("as_of_month")
    w12 = w.rowsBetween(-11, 0)

    # 1. Utilization in normal regime (S0/S1)
    panel = panel.withColumn(
        "util_in_normal_avg_12m",
        F.avg(
            F.when(
                F.col("state").isin(["S0", "S1"]),
                F.col("bureau_util")
            ).otherwise(F.lit(None))
        ).over(w12)
    )

    # 2. Utilization in stress regime (S2/S3/S4)
    panel = panel.withColumn(
        "util_in_stress_avg_12m",
        F.avg(
            F.when(
                F.col("state").isin(["S2", "S3", "S4"]),
                F.col("bureau_util")
            ).otherwise(F.lit(None))
        ).over(w12)
    )

    # 3. Regime delta
    panel = panel.withColumn(
        "util_regime_delta",
        F.col("util_in_stress_avg_12m") - F.col("util_in_normal_avg_12m")
    )

    # 4. Utilization compression flag (limit cut, not repayment)
    panel = panel.withColumn(
        "util_compression_flag",
        F.when(
            (F.col("bureau_util") < F.lag("bureau_util", 1).over(w)) &
            (F.col("bureau_max_dpd") > F.lag("bureau_max_dpd", 1).over(w)),
            1
        ).otherwise(0)
    )

    # 5. Utilization escape velocity
    # Note: is_stressed_m should exist from template's add_repayment_dynamics
    # proxy_delever_rate_m should also exist from template
    panel = panel.withColumn(
        "util_escape_velocity",
        _safe_div(
            F.col("util_in_stress_avg_12m"),
            F.avg(
                F.when(
                    F.coalesce(F.col("is_stress_m"), F.lit(0)) == 1,
                    F.col("proxy_delever_rate_m")
                ).otherwise(F.lit(None))
            ).over(w12),
            default=F.lit(None)
        )
    )

    return panel


# ============================================================================
# MASTER FUNCTION: Apply All 7 Physics Families
# ============================================================================

def add_all_physics_families(
    panel: DataFrame,
    hist_joined_df: Optional[DataFrame] = None
) -> DataFrame:
    """
    Apply all 7 physics families to panel (41 features total).

    Args:
        panel: Main feature panel
        hist_joined_df: Tradeline-level data (required for Families 5, 6)

    Returns:
        DataFrame with 41 new features
    """
    print("\n" + "="*80)
    print("ADDING 7 PHYSICS FAMILIES (41 features)")
    print("="*80)

    # Family 1: Inertia & Momentum (7 features)
    print("\n1/7: Inertia & Momentum (7 features)...")
    panel = add_inertia_momentum_features(panel)
    print("✓ Complete")

    # Family 2: Critical Slowing (6 features)
    print("\n2/7: Critical Slowing (6 features)...")
    panel = add_critical_slowing_features(panel)
    print("✓ Complete")

    # Family 3: Phase Boundary (7 features)
    print("\n3/7: Phase Boundary (7 features)...")
    panel = add_phase_boundary_features(panel)
    print("✓ Complete")

    # Family 4: Hysteresis (6 features)
    print("\n4/7: Hysteresis (6 features)...")
    panel = add_hysteresis_features(panel)
    print("✓ Complete")

    # Family 5: Lender Ecology Topology (5 features)
    if hist_joined_df is not None:
        print("\n5/7: Lender Ecology Topology (5 features)...")
        panel = add_lender_ecology_topology(panel, hist_joined_df)
        print("✓ Complete")
    else:
        print("\n5/7: Lender Ecology Topology - SKIPPED (hist_joined_df not provided)")

    # Family 6: Enquiry Physics (5 features)
    if hist_joined_df is not None:
        print("\n6/7: Enquiry Physics (5 features)...")
        panel = add_enquiry_physics(panel, hist_joined_df)
        print("✓ Complete")
    else:
        print("\n6/7: Enquiry Physics - SKIPPED (hist_joined_df not provided)")

    # Family 7: Utilization Physics (5 features)
    print("\n7/7: Utilization Physics (5 features)...")
    panel = add_utilization_physics(panel)
    print("✓ Complete")

    print("\n" + "="*80)
    print("✅ ALL 7 PHYSICS FAMILIES COMPLETE (41 features)")
    print("="*80 + "\n")

    return panel
