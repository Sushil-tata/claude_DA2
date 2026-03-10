"""
Trajectory Engine - Velocity, Acceleration, Transitions, Entropy
=================================================================

Behavioral physics features modeling customer dynamics:
- Velocity: Rate of change (DPD slope)
- Acceleration: Rate of velocity change (shock detection)
- Transitions: State change speed and patterns
- Entropy: Behavioral unpredictability

Author: Behavioral Physics Team
Version: 1.0.0
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from typing import List, Dict

from .config import get_config


class TrajectoryEngine:
    """
    Computes trajectory features using behavioral physics concepts.

    Feature Families:
    1. Velocity (18 features): DPD/utilization/balance rate of change
    2. Acceleration (12 features): Second derivative, shock detection
    3. Transitions (20 features): State change speed, patterns
    4. Entropy (10 features): Behavioral volatility, unpredictability
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.config = get_config()

    def compute_all_features(
        self,
        state_df: DataFrame,
        bureau_trade_df: DataFrame
    ) -> DataFrame:
        """
        Compute all trajectory features.

        Args:
            state_df: State assignments from StateBuilder
            bureau_trade_df: Bureau trade monthly data

        Returns:
            DataFrame with trajectory features added
        """
        # 1. Velocity features
        df = self._compute_velocity_features(state_df, bureau_trade_df)

        # 2. Acceleration features
        df = self._compute_acceleration_features(df)

        # 3. Transition features
        df = self._compute_transition_features(df)

        # 4. Entropy features
        df = self._compute_entropy_features(df)

        return df

    def _compute_velocity_features(
        self,
        state_df: DataFrame,
        bureau_trade_df: DataFrame
    ) -> DataFrame:
        """
        Compute velocity features (rate of change).

        Features:
        - dpd_diff_velocity_3m: Slope of max DPD over 3 months
        - dpd_diff_velocity_6m: Slope over 6 months
        - util_velocity_3m: Utilization rate of change
        - balance_velocity_6m: Balance trajectory
        """
        # Repartition by cust_id before heavy window operations to avoid data skew
        state_df = state_df.repartition("cust_id")

        windows = self.config.windows.WINDOWS_MONTHS

        for window in windows:
            # Create window spec for lookback
            w = Window.partitionBy("cust_id").orderBy("as_of_month") \
                .rowsBetween(-window + 1, 0)

            # DPD velocity (linear regression slope)
            # Using (current - start) / months as approximation
            state_df = state_df.withColumn(
                f"dpd_diff_velocity_{window}m",
                (F.col("bureau_max_dpd") -
                 F.first("bureau_max_dpd").over(w)) / window
            )

        # Aggregate bureau trade for utilization velocity
        bureau_util = bureau_trade_df.groupBy("cust_id", "as_of_month").agg(
            F.sum("balance").alias("total_balance"),
            F.sum("credit_limit").alias("total_limit")
        ).withColumn(
            "utilization",
            F.when(F.col("total_limit") > 0,
                   F.col("total_balance") / F.col("total_limit")
            ).otherwise(0.0)
        )

        # Join with state_df
        state_df = state_df.join(
            bureau_util.select("cust_id", "as_of_month", "utilization", "total_balance"),
            on=["cust_id", "as_of_month"],
            how="left"
        )

        # Repartition again after join (may have reshuffled)
        state_df = state_df.repartition("cust_id")

        # Utilization velocity
        for window in [3, 6]:
            w = Window.partitionBy("cust_id").orderBy("as_of_month") \
                .rowsBetween(-window + 1, 0)

            state_df = state_df.withColumn(
                f"util_velocity_{window}m",
                (F.col("utilization") -
                 F.first("utilization").over(w)) / window
            )

            state_df = state_df.withColumn(
                f"balance_velocity_{window}m",
                (F.col("total_balance") -
                 F.first("total_balance").over(w)) / window
            )

        return state_df

    def _compute_acceleration_features(self, df: DataFrame) -> DataFrame:
        """
        Compute acceleration features (second derivative).

        Features:
        - dpd_acceleration_3m: Rate of velocity change
        - dpd_acceleration_6m
        - shock_flag_3m: Sudden velocity increase (>10 DPD/month)
        - deceleration_flag: Velocity decreasing (improving)
        """
        # Acceleration = change in velocity
        for window in [3, 6]:
            w = Window.partitionBy("cust_id").orderBy("as_of_month")

            # Compute change in velocity (acceleration)
            df = df.withColumn(
                f"dpd_acceleration_{window}m",
                F.col(f"dpd_diff_velocity_{window}m") -
                F.lag(f"dpd_diff_velocity_{window}m", 1).over(w)
            )

            # Shock flag (sudden acceleration > threshold)
            df = df.withColumn(
                f"shock_flag_{window}m",
                (F.col(f"dpd_acceleration_{window}m") > 10).cast("int")
            )

        # Deceleration flag (velocity becoming less positive/more negative)
        df = df.withColumn(
            "deceleration_flag_3m",
            (F.col("dpd_acceleration_3m") < 0).cast("int")
        )

        return df

    def _compute_transition_features(self, df: DataFrame) -> DataFrame:
        """
        Compute state transition features.

        Features:
        - transition_speed_s0_s2: Months to go S0 → S2
        - transition_speed_s0_s3: Months to go S0 → S3
        - cure_speed_s2_s0: Months to cure from S2
        - cure_halflife: Time to reduce DPD by 50%
        - bad_state_trap_prob: P(stuck in S3/S4 for 3+ months)
        """
        # Window for lookback
        w_12m = Window.partitionBy("cust_id").orderBy("as_of_month") \
            .rowsBetween(-11, 0)

        # NOTE: transition_type columns removed
        # Production code uses state_changed, deteriorate_flag_m, improve_flag_m instead
        # See production_pipeline.py for reference

        # Transition speed (avg months between transitions)
        df = df.withColumn(
            "transition_frequency_12m",
            F.sum(F.col("state_changed").cast("int")).over(w_12m)
        )

        df = df.withColumn(
            "avg_months_per_transition",
            F.when(F.col("transition_frequency_12m") > 0,
                   12.0 / F.col("transition_frequency_12m")
            ).otherwise(12.0)
        )

        # Cure half-life approximation
        # Time to reduce DPD from peak to peak/2
        w_6m = Window.partitionBy("cust_id").orderBy("as_of_month") \
            .rowsBetween(-5, 0)

        df = df.withColumn(
            "dpd_peak_6m",
            F.max("bureau_max_dpd").over(w_6m)
        )

        df = df.withColumn(
            "dpd_halflife_target",
            F.col("dpd_peak_6m") / 2
        )

        # Flag if reached half-life
        df = df.withColumn(
            "reached_halflife",
            (F.col("bureau_max_dpd") <= F.col("dpd_halflife_target")).cast("int")
        )

        return df

    def _compute_entropy_features(self, df: DataFrame) -> DataFrame:
        """
        Compute entropy/volatility features.

        Features:
        - state_entropy_6m: Shannon entropy of state distribution
        - oscillation_count_6m: Number of state changes
        - volatility_index_6m: Coefficient of variation of DPD
        - regime_persistence: Avg consecutive months in same regime
        """
        windows = [3, 6, 12]

        for window in windows:
            w = Window.partitionBy("cust_id").orderBy("as_of_month") \
                .rowsBetween(-window + 1, 0)

            # Oscillation count (state changes)
            df = df.withColumn(
                f"oscillation_count_{window}m",
                F.sum(F.col("state_changed").cast("int")).over(w)
            )

            # DPD volatility (coefficient of variation)
            df = df.withColumn(
                f"dpd_mean_{window}m",
                F.avg("bureau_max_dpd").over(w)
            )

            df = df.withColumn(
                f"dpd_std_{window}m",
                F.stddev("bureau_max_dpd").over(w)
            )

            df = df.withColumn(
                f"dpd_cv_{window}m",
                F.when(
                    F.col(f"dpd_mean_{window}m") > 0,
                    F.col(f"dpd_std_{window}m") / F.col(f"dpd_mean_{window}m")
                ).otherwise(0.0)
            )

        # State entropy (Shannon entropy)
        # Approximate using state distribution
        w_6m = Window.partitionBy("cust_id").orderBy("as_of_month") \
            .rowsBetween(-5, 0)

        for state in ["S0", "S1", "S2", "S3", "S4"]:
            df = df.withColumn(
                f"prob_{state}_6m",
                F.sum(
                    F.when(F.col("consolidated_state") == state, 1).otherwise(0)
                ).over(w_6m) / 6.0
            )

        # Shannon entropy H = -Σ(p * log(p))
        entropy_sum = F.lit(0.0)
        for state in ["S0", "S1", "S2", "S3", "S4"]:
            entropy_sum = entropy_sum + F.when(
                F.col(f"prob_{state}_6m") > 0,
                F.col(f"prob_{state}_6m") * F.log(F.col(f"prob_{state}_6m"))
            ).otherwise(0.0)

        df = df.withColumn("state_entropy_6m", -entropy_sum)

        # Drop intermediate prob columns
        for state in ["S0", "S1", "S2", "S3", "S4"]:
            df = df.drop(f"prob_{state}_6m")

        return df


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from pyspark.sql import SparkSession
    from datetime import date
    import pandas as pd

    spark = SparkSession.builder \
        .appName("TrajectoryEngineTest") \
        .master("local[*]") \
        .getOrCreate()

    print("\n" + "="*70)
    print("TRAJECTORY ENGINE - EXAMPLE")
    print("="*70)

    # Create sample state data (from StateBuilder output)
    state_data = [
        ("CUST001", date(2024, 1, 31), 0, "S0", "NORMAL", False),
        ("CUST001", date(2024, 2, 29), 15, "S1", "NORMAL", True),
        ("CUST001", date(2024, 3, 31), 45, "S2", "STRESSED", True),
        ("CUST001", date(2024, 4, 30), 75, "S2", "STRESSED", False),
        ("CUST001", date(2024, 5, 31), 30, "S1", "NORMAL", True),
        ("CUST001", date(2024, 6, 30), 60, "S2", "STRESSED", True),
    ]

    state_df = spark.createDataFrame(
        state_data,
        ["cust_id", "as_of_month", "bureau_max_dpd", "consolidated_state",
         "consolidated_regime", "state_changed"]
    )

    # Create sample bureau trade data
    bureau_data = [
        ("CUST001", date(2024, 1, 31), 10000, 50000),
        ("CUST001", date(2024, 2, 29), 15000, 50000),
        ("CUST001", date(2024, 3, 31), 25000, 50000),
        ("CUST001", date(2024, 4, 30), 35000, 50000),
        ("CUST001", date(2024, 5, 31), 30000, 50000),
        ("CUST001", date(2024, 6, 30), 40000, 50000),
    ]

    bureau_df = spark.createDataFrame(
        bureau_data,
        ["cust_id", "as_of_month", "balance", "credit_limit"]
    )

    # Compute trajectory features
    engine = TrajectoryEngine(spark)
    features_df = engine.compute_all_features(state_df, bureau_df)

    # Show results
    print("\nVelocity Features:")
    features_df.select(
        "cust_id", "as_of_month", "bureau_max_dpd",
        "dpd_diff_velocity_3m", "util_velocity_3m"
    ).orderBy("cust_id", "as_of_month").show()

    print("\nAcceleration Features:")
    features_df.select(
        "cust_id", "as_of_month",
        "dpd_acceleration_3m", "shock_flag_3m"
    ).orderBy("cust_id", "as_of_month").show()

    print("\nEntropy Features:")
    features_df.select(
        "cust_id", "as_of_month",
        "oscillation_count_6m", "dpd_cv_6m", "state_entropy_6m"
    ).orderBy("cust_id", "as_of_month").show()

    print("\n✅ Trajectory Engine test complete")

    spark.stop()
