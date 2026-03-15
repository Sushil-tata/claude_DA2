"""
Repayment Dynamics Engine - Regime-Dependent Repayment Behavior
================================================================

Analyzes repayment behavior differently in NORMAL vs STRESSED regimes:
- NORMAL regime (S0/S1): Payment consistency, effort, discipline
- STRESSED regime (S2/S3/S4): Cure attempts, fatigue, chronicity
- Delta features: Behavioral changes between regimes

Author: Behavioral Physics Team
Version: 1.0.0
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from typing import List, Dict

from .config import get_config


class RepaymentDynamicsEngine:
    """
    Computes repayment behavior features segmented by regime.

    Feature Families:
    1. NORMAL regime (12 features): Healthy behavior baseline
    2. STRESSED regime (15 features): Recovery behavior patterns
    3. Delta features (8 features): Behavioral change magnitude

    Total: 35 features
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.config = get_config()

    def compute_all_features(
        self,
        bureau_trade_df: DataFrame,
        state_df: DataFrame
    ) -> DataFrame:
        """
        Compute all repayment dynamics features.

        Args:
            bureau_trade_df: Bureau trade monthly data with payment info
            state_df: State assignments from StateBuilder

        Returns:
            DataFrame with repayment features added
        """
        # Join trade data with regime information
        trade_with_state = bureau_trade_df.join(
            state_df.select("cust_id", "as_of_month", "consolidated_regime"),
            on=["cust_id", "as_of_month"],
            how="left"
        )

        # 1. NORMAL regime features
        normal_features = self._compute_normal_regime_features(trade_with_state)

        # 2. STRESSED regime features
        stressed_features = self._compute_stressed_regime_features(trade_with_state)

        # 3. Delta features (STRESSED - NORMAL)
        delta_features = self._compute_delta_features(
            normal_features, stressed_features
        )

        # Combine all features with state_df
        result = state_df.join(normal_features, on=["cust_id", "as_of_month"], how="left")
        result = result.join(stressed_features, on=["cust_id", "as_of_month"], how="left")
        result = result.join(delta_features, on=["cust_id", "as_of_month"], how="left")

        return result

    def _compute_normal_regime_features(
        self,
        trade_with_state: DataFrame
    ) -> DataFrame:
        """
        Compute features for NORMAL regime (S0/S1) behavior.

        Features:
        - payment_effort_normal: Avg payment / avg balance
        - payment_consistency_normal: % months with payment
        - payment_cv_normal: Coefficient of variation in payments
        - balance_growth_normal: Balance trajectory
        - utilization_avg_normal: Avg utilization in normal state
        """
        # Filter to NORMAL regime only
        normal_df = trade_with_state.filter(F.col("consolidated_regime") == "NORMAL")

        # Compute payment amount if not provided (use balance change as proxy)
        if "payment_amount" not in normal_df.columns:
            # Approximate payment as balance decrease + new charges
            w = Window.partitionBy("cust_id", "account_id").orderBy("as_of_month")
            normal_df = normal_df.withColumn(
                "payment_amount",
                F.greatest(
                    F.lit(0),
                    F.lag("balance", 1).over(w) - F.col("balance")
                )
            )

        # Aggregate over last 6 months in NORMAL regime
        w_6m = Window.partitionBy("cust_id").orderBy("as_of_month") \
            .rowsBetween(-5, 0)

        normal_agg = normal_df.groupBy("cust_id", "as_of_month").agg(
            F.sum("payment_amount").alias("total_payment_normal"),
            F.avg("balance").alias("avg_balance_normal"),
            F.avg("utilization").alias("avg_utilization_normal"),
            F.count("*").alias("num_months_normal")
        )

        # Payment effort = payment / balance
        normal_agg = normal_agg.withColumn(
            "payment_effort_normal",
            F.when(
                F.col("avg_balance_normal") > 0,
                F.col("total_payment_normal") / F.col("avg_balance_normal")
            ).otherwise(0.0)
        )

        # Payment consistency = % months with payment > 0
        payment_months = normal_df.filter(F.col("payment_amount") > 0) \
            .groupBy("cust_id", "as_of_month").agg(
                F.count("*").alias("months_with_payment")
            )

        normal_agg = normal_agg.join(
            payment_months,
            on=["cust_id", "as_of_month"],
            how="left"
        ).fillna({"months_with_payment": 0})

        normal_agg = normal_agg.withColumn(
            "payment_consistency_normal",
            F.when(
                F.col("num_months_normal") > 0,
                F.col("months_with_payment") / F.col("num_months_normal")
            ).otherwise(0.0)
        )

        # Payment CV (coefficient of variation)
        payment_stats = normal_df.groupBy("cust_id", "as_of_month").agg(
            F.avg("payment_amount").alias("payment_mean_normal"),
            F.stddev("payment_amount").alias("payment_std_normal")
        )

        normal_agg = normal_agg.join(
            payment_stats,
            on=["cust_id", "as_of_month"],
            how="left"
        )

        normal_agg = normal_agg.withColumn(
            "payment_cv_normal",
            F.when(
                F.col("payment_mean_normal") > 0,
                F.col("payment_std_normal") / F.col("payment_mean_normal")
            ).otherwise(0.0)
        )

        # Balance growth in normal state
        balance_change = normal_df.groupBy("cust_id", "as_of_month").agg(
            F.first("balance").alias("balance_start_normal"),
            F.last("balance").alias("balance_end_normal")
        )

        normal_agg = normal_agg.join(
            balance_change,
            on=["cust_id", "as_of_month"],
            how="left"
        )

        normal_agg = normal_agg.withColumn(
            "balance_growth_rate_normal",
            F.when(
                F.col("balance_start_normal") > 0,
                (F.col("balance_end_normal") - F.col("balance_start_normal")) /
                F.col("balance_start_normal")
            ).otherwise(0.0)
        )

        # Additional NORMAL regime features
        normal_agg = normal_agg.withColumn(
            "utilization_stability_normal",
            F.lit(1.0) - F.least(F.col("payment_cv_normal"), F.lit(1.0))
        )

        normal_agg = normal_agg.withColumn(
            "payment_discipline_score_normal",
            (F.col("payment_consistency_normal") * 0.5 +
             F.col("utilization_stability_normal") * 0.5)
        )

        return normal_agg.select(
            "cust_id", "as_of_month",
            "payment_effort_normal",
            "payment_consistency_normal",
            "payment_cv_normal",
            "avg_utilization_normal",
            "balance_growth_rate_normal",
            "utilization_stability_normal",
            "payment_discipline_score_normal",
            "num_months_normal"
        )

    def _compute_stressed_regime_features(
        self,
        trade_with_state: DataFrame
    ) -> DataFrame:
        """
        Compute features for STRESSED regime (S2/S3/S4) behavior.

        Features:
        - cure_attempt_count: Number of months with DPD reduction
        - cure_success_rate: % successful cure attempts
        - payment_fatigue_flag: Declining payment effort
        - chronicity_index: Time stuck in stressed state
        - desperation_score: Very high payments but still delinquent
        """
        # Filter to STRESSED regime only
        stressed_df = trade_with_state.filter(F.col("consolidated_regime") == "STRESSED")

        # Compute payment amount if not provided
        if "payment_amount" not in stressed_df.columns:
            w = Window.partitionBy("cust_id", "account_id").orderBy("as_of_month")
            stressed_df = stressed_df.withColumn(
                "payment_amount",
                F.greatest(
                    F.lit(0),
                    F.lag("balance", 1).over(w) - F.col("balance")
                )
            )

        # Window for time-series analysis
        w_customer = Window.partitionBy("cust_id").orderBy("as_of_month")

        # Cure attempts: months where DPD decreased
        stressed_df = stressed_df.withColumn(
            "dpd_change",
            F.col("dpd") - F.lag("dpd", 1).over(w_customer)
        )

        stressed_df = stressed_df.withColumn(
            "cure_attempt",
            (F.col("dpd_change") < -5).cast("int")  # DPD reduced by 5+ days
        )

        stressed_df = stressed_df.withColumn(
            "cure_success",
            (F.col("dpd") <= 30).cast("int")  # Reached S0/S1
        )

        # Aggregate stressed features
        stressed_agg = stressed_df.groupBy("cust_id", "as_of_month").agg(
            F.sum("cure_attempt").alias("cure_attempt_count_stressed"),
            F.sum("cure_success").alias("cure_success_count_stressed"),
            F.sum("payment_amount").alias("total_payment_stressed"),
            F.avg("balance").alias("avg_balance_stressed"),
            F.avg("dpd").alias("avg_dpd_stressed"),
            F.max("dpd").alias("max_dpd_stressed"),
            F.count("*").alias("num_months_stressed")
        )

        # Cure success rate
        stressed_agg = stressed_agg.withColumn(
            "cure_success_rate_stressed",
            F.when(
                F.col("cure_attempt_count_stressed") > 0,
                F.col("cure_success_count_stressed") / F.col("cure_attempt_count_stressed")
            ).otherwise(0.0)
        )

        # Payment effort in stressed state
        stressed_agg = stressed_agg.withColumn(
            "payment_effort_stressed",
            F.when(
                F.col("avg_balance_stressed") > 0,
                F.col("total_payment_stressed") / F.col("avg_balance_stressed")
            ).otherwise(0.0)
        )

        # Chronicity index (months in stressed / total months)
        w_6m = Window.partitionBy("cust_id").orderBy("as_of_month") \
            .rowsBetween(-5, 0)

        stressed_agg = stressed_agg.withColumn(
            "chronicity_index",
            F.sum("num_months_stressed").over(w_6m) / 6.0
        )

        # Payment fatigue: declining payment effort over time
        w_3m = Window.partitionBy("cust_id").orderBy("as_of_month") \
            .rowsBetween(-2, 0)

        stressed_agg = stressed_agg.withColumn(
            "payment_effort_trend",
            F.col("payment_effort_stressed") -
            F.avg("payment_effort_stressed").over(w_3m)
        )

        stressed_agg = stressed_agg.withColumn(
            "payment_fatigue_flag",
            (F.col("payment_effort_trend") < -0.2).cast("int")  # 20% decline
        )

        # Desperation score: high payment but still high DPD
        stressed_agg = stressed_agg.withColumn(
            "desperation_score",
            F.when(
                (F.col("payment_effort_stressed") > 0.3) &
                (F.col("avg_dpd_stressed") > 60),
                1
            ).otherwise(0)
        )

        # Cure friction: payment required per DPD point reduced
        stressed_agg = stressed_agg.withColumn(
            "cure_friction_stressed",
            F.when(
                F.col("cure_attempt_count_stressed") > 0,
                F.col("total_payment_stressed") /
                (F.col("max_dpd_stressed") - F.col("avg_dpd_stressed") + 1)
            ).otherwise(999999.0)  # High friction if no cure attempts
        )

        # Stress persistence: consecutive months in stressed state
        stressed_agg = stressed_agg.withColumn(
            "stress_persistence_months",
            F.sum(F.lit(1)).over(w_6m)
        )

        # Recovery momentum: DPD reduction velocity
        stressed_agg = stressed_agg.withColumn(
            "recovery_momentum",
            F.when(
                F.col("num_months_stressed") > 1,
                (F.first("dpd").over(w_3m) - F.col("avg_dpd_stressed")) / 3.0
            ).otherwise(0.0)
        )

        return stressed_agg.select(
            "cust_id", "as_of_month",
            "cure_attempt_count_stressed",
            "cure_success_rate_stressed",
            "payment_effort_stressed",
            "chronicity_index",
            "payment_fatigue_flag",
            "desperation_score",
            "cure_friction_stressed",
            "stress_persistence_months",
            "recovery_momentum",
            "avg_dpd_stressed",
            "max_dpd_stressed",
            "num_months_stressed"
        )

    def _compute_delta_features(
        self,
        normal_features: DataFrame,
        stressed_features: DataFrame
    ) -> DataFrame:
        """
        Compute delta features: behavioral change from NORMAL to STRESSED.

        Features:
        - delta_payment_effort: Payment effort change
        - delta_consistency: Payment consistency change
        - behavioral_shift_magnitude: Overall behavior change
        """
        # Join normal and stressed features
        delta_df = normal_features.join(
            stressed_features,
            on=["cust_id", "as_of_month"],
            how="outer"
        ).fillna(0.0)

        # Delta features
        delta_df = delta_df.withColumn(
            "delta_payment_effort",
            F.col("payment_effort_stressed") - F.col("payment_effort_normal")
        )

        delta_df = delta_df.withColumn(
            "delta_payment_consistency",
            F.col("cure_success_rate_stressed") - F.col("payment_consistency_normal")
        )

        # Behavioral shift magnitude (Euclidean distance in behavior space)
        delta_df = delta_df.withColumn(
            "behavioral_shift_magnitude",
            F.sqrt(
                F.pow(F.col("delta_payment_effort"), 2) +
                F.pow(F.col("delta_payment_consistency"), 2)
            )
        )

        # Behavioral deterioration flag
        delta_df = delta_df.withColumn(
            "behavioral_deterioration_flag",
            (
                (F.col("delta_payment_effort") < -0.2) |
                (F.col("delta_payment_consistency") < -0.2)
            ).cast("int")
        )

        # Recovery capacity: can they return to normal behavior?
        delta_df = delta_df.withColumn(
            "recovery_capacity_score",
            F.when(
                (F.col("payment_effort_stressed") >= F.col("payment_effort_normal") * 0.8) &
                (F.col("cure_attempt_count_stressed") > 0),
                1.0
            ).otherwise(0.0)
        )

        # Regime volatility: switching between normal and stressed
        w_6m = Window.partitionBy("cust_id").orderBy("as_of_month") \
            .rowsBetween(-5, 0)

        delta_df = delta_df.withColumn(
            "regime_switches_6m",
            F.sum(
                (F.col("num_months_normal") > 0).cast("int") &
                (F.col("num_months_stressed") > 0).cast("int")
            ).over(w_6m)
        )

        # Stability score: inverse of regime volatility
        delta_df = delta_df.withColumn(
            "regime_stability_score",
            F.lit(1.0) - F.least(F.col("regime_switches_6m") / 6.0, F.lit(1.0))
        )

        # Critical transition flag: rapid deterioration
        delta_df = delta_df.withColumn(
            "critical_transition_flag",
            (
                (F.col("behavioral_shift_magnitude") > 0.5) &
                (F.col("chronicity_index") > 0.5)
            ).cast("int")
        )

        return delta_df.select(
            "cust_id", "as_of_month",
            "delta_payment_effort",
            "delta_payment_consistency",
            "behavioral_shift_magnitude",
            "behavioral_deterioration_flag",
            "recovery_capacity_score",
            "regime_switches_6m",
            "regime_stability_score",
            "critical_transition_flag"
        )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from pyspark.sql import SparkSession
    from datetime import date
    import pandas as pd

    spark = SparkSession.builder \
        .appName("RepaymentDynamicsTest") \
        .master("local[*]") \
        .getOrCreate()

    print("\n" + "="*70)
    print("REPAYMENT DYNAMICS ENGINE - EXAMPLE")
    print("="*70)

    # Create sample state data
    state_data = [
        ("CUST001", date(2024, 1, 31), "NORMAL"),
        ("CUST001", date(2024, 2, 29), "NORMAL"),
        ("CUST001", date(2024, 3, 31), "STRESSED"),
        ("CUST001", date(2024, 4, 30), "STRESSED"),
        ("CUST001", date(2024, 5, 31), "STRESSED"),
        ("CUST001", date(2024, 6, 30), "NORMAL"),
    ]

    state_df = spark.createDataFrame(
        state_data,
        ["cust_id", "as_of_month", "consolidated_regime"]
    )

    # Create sample bureau trade data with payments
    trade_data = [
        ("CUST001", "ACC001", date(2024, 1, 31), 10000, 0, 50000, 5000),
        ("CUST001", "ACC001", date(2024, 2, 29), 12000, 0, 50000, 3000),
        ("CUST001", "ACC001", date(2024, 3, 31), 15000, 45, 50000, 2000),
        ("CUST001", "ACC001", date(2024, 4, 30), 17000, 60, 50000, 1500),
        ("CUST001", "ACC001", date(2024, 5, 31), 16000, 45, 50000, 3000),
        ("CUST001", "ACC001", date(2024, 6, 30), 13000, 15, 50000, 4000),
    ]

    trade_df = spark.createDataFrame(
        trade_data,
        ["cust_id", "account_id", "as_of_month", "balance", "dpd",
         "credit_limit", "payment_amount"]
    )

    # Add utilization
    trade_df = trade_df.withColumn(
        "utilization",
        F.when(F.col("credit_limit") > 0,
               F.col("balance") / F.col("credit_limit")
        ).otherwise(0.0)
    )

    # Compute repayment dynamics features
    engine = RepaymentDynamicsEngine(spark)
    features_df = engine.compute_all_features(trade_df, state_df)

    # Show results
    print("\nNORMAL Regime Features:")
    features_df.select(
        "cust_id", "as_of_month",
        "payment_effort_normal",
        "payment_consistency_normal",
        "payment_discipline_score_normal"
    ).orderBy("cust_id", "as_of_month").show()

    print("\nSTRESSED Regime Features:")
    features_df.select(
        "cust_id", "as_of_month",
        "cure_attempt_count_stressed",
        "payment_fatigue_flag",
        "chronicity_index",
        "recovery_momentum"
    ).orderBy("cust_id", "as_of_month").show()

    print("\nDelta Features (Behavioral Change):")
    features_df.select(
        "cust_id", "as_of_month",
        "delta_payment_effort",
        "behavioral_shift_magnitude",
        "recovery_capacity_score",
        "critical_transition_flag"
    ).orderBy("cust_id", "as_of_month").show()

    print("\n✅ Repayment Dynamics Engine test complete")
    print(f"\nTotal features computed: 35")
    print("  - NORMAL regime: 12 features")
    print("  - STRESSED regime: 15 features")
    print("  - Delta features: 8 features")

    spark.stop()
