"""
CardX-Bureau Interactions Engine - Cross-Source Behavioral Intelligence
=======================================================================

Analyzes behavioral differences and interactions between CardX (internal)
and Bureau (external) data sources:

- Lead-Lag Dynamics: Does CardX predict bureau delinquency?
- Performance Divergence: Different trajectories across sources
- Information Asymmetry: What CardX reveals that bureau doesn't
- Cross-Trigger Events: When one source triggers the other

Author: Behavioral Physics Team
Version: 1.0.0
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from typing import List, Dict

from .config import get_config


class CardXBureauInteractionsEngine:
    """
    Computes CardX-Bureau interaction features.

    Feature Families:
    1. Lead-Lag Indicators (5 features): Early warning signals
    2. Performance Divergence (6 features): Different trajectories
    3. Utilization Spread (4 features): Credit dependency patterns
    4. Cross-Trigger Events (5 features): Cascade delinquency

    Total: 20 features
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.config = get_config()

    def compute_all_features(
        self,
        bureau_trade_df: DataFrame,
        cardx_internal_df: DataFrame,
        state_df: DataFrame
    ) -> DataFrame:
        """
        Compute all CardX-Bureau interaction features.

        Args:
            bureau_trade_df: Bureau trade monthly data
            cardx_internal_df: CardX internal monthly data
            state_df: State assignments from StateBuilder

        Returns:
            DataFrame with interaction features at (cust_id, as_of_month) grain
        """
        # 1. Create bureau aggregates
        bureau_agg = self._aggregate_bureau_data(bureau_trade_df)

        # 2. Create CardX aggregates
        cardx_agg = self._aggregate_cardx_data(cardx_internal_df)

        # 3. Join bureau and CardX data
        combined = bureau_agg.join(
            cardx_agg,
            on=["cust_id", "as_of_month"],
            how="outer"
        ).fillna(0.0)

        # 4. Lead-Lag features
        lead_lag_df = self._compute_lead_lag_features(combined)

        # 5. Performance divergence features
        divergence_df = self._compute_divergence_features(combined)

        # 6. Utilization spread features
        util_spread_df = self._compute_utilization_spread(combined)

        # 7. Cross-trigger features
        cross_trigger_df = self._compute_cross_trigger_features(combined)

        # Combine all features
        result = lead_lag_df.join(
            divergence_df, on=["cust_id", "as_of_month"], how="outer"
        ).join(
            util_spread_df, on=["cust_id", "as_of_month"], how="outer"
        ).join(
            cross_trigger_df, on=["cust_id", "as_of_month"], how="outer"
        )

        return result

    def _aggregate_bureau_data(self, bureau_trade_df: DataFrame) -> DataFrame:
        """Aggregate bureau data to customer-month level"""
        bureau_agg = bureau_trade_df.groupBy("cust_id", "as_of_month").agg(
            F.max("dpd").alias("bureau_max_dpd"),
            F.avg("dpd").alias("bureau_avg_dpd"),
            F.sum("balance").alias("bureau_total_balance"),
            F.sum("credit_limit").alias("bureau_total_limit"),
            F.count("*").alias("bureau_account_count")
        )

        # Bureau utilization
        bureau_agg = bureau_agg.withColumn(
            "bureau_utilization",
            F.when(
                F.col("bureau_total_limit") > 0,
                F.col("bureau_total_balance") / F.col("bureau_total_limit")
            ).otherwise(0.0)
        )

        # Bureau delinquency flag
        bureau_agg = bureau_agg.withColumn(
            "bureau_delinquent_flag",
            (F.col("bureau_max_dpd") > 30).cast("int")
        )

        return bureau_agg

    def _aggregate_cardx_data(self, cardx_internal_df: DataFrame) -> DataFrame:
        """Aggregate CardX data to customer-month level"""
        # Expected columns: cust_id, as_of_month, cardx_dpd, cardx_balance, cardx_credit_limit

        cardx_agg = cardx_internal_df.groupBy("cust_id", "as_of_month").agg(
            F.max("cardx_dpd").alias("cardx_max_dpd"),
            F.sum("cardx_balance").alias("cardx_total_balance"),
            F.sum("cardx_credit_limit").alias("cardx_total_limit")
        )

        # CardX utilization
        cardx_agg = cardx_agg.withColumn(
            "cardx_utilization",
            F.when(
                F.col("cardx_total_limit") > 0,
                F.col("cardx_total_balance") / F.col("cardx_total_limit")
            ).otherwise(0.0)
        )

        # CardX delinquency flag
        cardx_agg = cardx_agg.withColumn(
            "cardx_delinquent_flag",
            (F.col("cardx_max_dpd") > 30).cast("int")
        )

        return cardx_agg

    def _compute_lead_lag_features(self, combined_df: DataFrame) -> DataFrame:
        """
        Compute lead-lag indicators.

        Features:
        - cardx_leads_bureau_flag: CardX delinquent before bureau
        - bureau_leads_cardx_flag: Bureau delinquent before CardX
        - lead_lag_months: Time gap between CardX and bureau delinquency
        - cardx_early_warning_flag: CardX signals stress before bureau
        """
        w = Window.partitionBy("cust_id").orderBy("as_of_month")

        # Track when each became delinquent
        combined_df = combined_df.withColumn(
            "cardx_first_delinquent_month",
            F.when(
                (F.col("cardx_delinquent_flag") == 1) &
                (F.lag("cardx_delinquent_flag", 1).over(w) == 0),
                F.col("as_of_month")
            )
        )

        combined_df = combined_df.withColumn(
            "bureau_first_delinquent_month",
            F.when(
                (F.col("bureau_delinquent_flag") == 1) &
                (F.lag("bureau_delinquent_flag", 1).over(w) == 0),
                F.col("as_of_month")
            )
        )

        # Forward-fill first delinquent months
        combined_df = combined_df.withColumn(
            "cardx_first_delinquent_month",
            F.last("cardx_first_delinquent_month", ignorenulls=True).over(
                w.rowsBetween(Window.unboundedPreceding, 0)
            )
        )

        combined_df = combined_df.withColumn(
            "bureau_first_delinquent_month",
            F.last("bureau_first_delinquent_month", ignorenulls=True).over(
                w.rowsBetween(Window.unboundedPreceding, 0)
            )
        )

        # Lead-lag flags
        combined_df = combined_df.withColumn(
            "cardx_leads_bureau_flag",
            (
                F.col("cardx_first_delinquent_month").isNotNull() &
                F.col("bureau_first_delinquent_month").isNotNull() &
                (F.col("cardx_first_delinquent_month") < F.col("bureau_first_delinquent_month"))
            ).cast("int")
        )

        combined_df = combined_df.withColumn(
            "bureau_leads_cardx_flag",
            (
                F.col("cardx_first_delinquent_month").isNotNull() &
                F.col("bureau_first_delinquent_month").isNotNull() &
                (F.col("bureau_first_delinquent_month") < F.col("cardx_first_delinquent_month"))
            ).cast("int")
        )

        # Lead-lag months
        combined_df = combined_df.withColumn(
            "lead_lag_months",
            F.when(
                F.col("cardx_leads_bureau_flag") == 1,
                F.months_between(
                    F.col("bureau_first_delinquent_month"),
                    F.col("cardx_first_delinquent_month")
                )
            ).when(
                F.col("bureau_leads_cardx_flag") == 1,
                -F.months_between(
                    F.col("cardx_first_delinquent_month"),
                    F.col("bureau_first_delinquent_month")
                )
            ).otherwise(0.0)
        )

        # Early warning flag (CardX delinquent but bureau still clean)
        combined_df = combined_df.withColumn(
            "cardx_early_warning_flag",
            (
                (F.col("cardx_delinquent_flag") == 1) &
                (F.col("bureau_delinquent_flag") == 0)
            ).cast("int")
        )

        # Containment success flag (CardX delinquent in past, but bureau stayed clean)
        w_6m = Window.partitionBy("cust_id").orderBy("as_of_month").rowsBetween(-5, 0)

        combined_df = combined_df.withColumn(
            "containment_success_flag",
            (
                (F.sum("cardx_delinquent_flag").over(w_6m) > 0) &
                (F.sum("bureau_delinquent_flag").over(w_6m) == 0)
            ).cast("int")
        )

        return combined_df.select(
            "cust_id", "as_of_month",
            "cardx_leads_bureau_flag",
            "bureau_leads_cardx_flag",
            "lead_lag_months",
            "cardx_early_warning_flag",
            "containment_success_flag"
        )

    def _compute_divergence_features(self, combined_df: DataFrame) -> DataFrame:
        """
        Compute performance divergence features.

        Features:
        - dpd_divergence_score: |CardX_DPD - Bureau_DPD|
        - velocity_divergence: CardX velocity - Bureau velocity
        - trajectory_correlation: Correlation of DPD trajectories
        """
        # DPD divergence (absolute difference)
        combined_df = combined_df.withColumn(
            "dpd_divergence_score",
            F.abs(F.col("cardx_max_dpd") - F.col("bureau_max_dpd"))
        )

        # DPD divergence direction
        combined_df = combined_df.withColumn(
            "cardx_worse_than_bureau_flag",
            (F.col("cardx_max_dpd") > F.col("bureau_max_dpd") + 30).cast("int")
        )

        combined_df = combined_df.withColumn(
            "bureau_worse_than_cardx_flag",
            (F.col("bureau_max_dpd") > F.col("cardx_max_dpd") + 30).cast("int")
        )

        # Velocity divergence
        w = Window.partitionBy("cust_id").orderBy("as_of_month")

        combined_df = combined_df.withColumn(
            "cardx_dpd_velocity",
            F.col("cardx_max_dpd") - F.lag("cardx_max_dpd", 3).over(w)
        )

        combined_df = combined_df.withColumn(
            "bureau_dpd_velocity",
            F.col("bureau_max_dpd") - F.lag("bureau_max_dpd", 3).over(w)
        )

        combined_df = combined_df.withColumn(
            "velocity_divergence",
            F.col("cardx_dpd_velocity") - F.col("bureau_dpd_velocity")
        )

        # Trajectory correlation (simplified - using 6m window)
        w_6m = Window.partitionBy("cust_id").orderBy("as_of_month").rowsBetween(-5, 0)

        combined_df = combined_df.withColumn(
            "trajectory_correlation",
            F.corr("cardx_max_dpd", "bureau_max_dpd").over(w_6m)
        )

        # Behavioral consistency score (inverse of divergence)
        combined_df = combined_df.withColumn(
            "behavioral_consistency_score",
            F.when(
                F.col("dpd_divergence_score") == 0, 1.0
            ).otherwise(
                F.lit(1.0) / (F.lit(1.0) + F.col("dpd_divergence_score") / 100.0)
            )
        )

        return combined_df.select(
            "cust_id", "as_of_month",
            "dpd_divergence_score",
            "cardx_worse_than_bureau_flag",
            "bureau_worse_than_cardx_flag",
            "velocity_divergence",
            "trajectory_correlation",
            "behavioral_consistency_score"
        )

    def _compute_utilization_spread(self, combined_df: DataFrame) -> DataFrame:
        """
        Compute utilization spread features.

        Features:
        - util_spread_cardx_bureau: CardX util - Bureau util
        - util_spread_volatility: Volatility of utilization spread
        - cardx_dependency_flag: CardX util > Bureau util (CardX dependency)
        """
        # Utilization spread
        combined_df = combined_df.withColumn(
            "util_spread_cardx_bureau",
            F.col("cardx_utilization") - F.col("bureau_utilization")
        )

        # Utilization spread volatility (6m window)
        w_6m = Window.partitionBy("cust_id").orderBy("as_of_month").rowsBetween(-5, 0)

        combined_df = combined_df.withColumn(
            "util_spread_volatility",
            F.stddev("util_spread_cardx_bureau").over(w_6m)
        )

        # CardX dependency flag (higher utilization on CardX)
        combined_df = combined_df.withColumn(
            "cardx_dependency_flag",
            (F.col("cardx_utilization") > F.col("bureau_utilization") + 0.1).cast("int")
        )

        # High utilization on both sources flag
        combined_df = combined_df.withColumn(
            "high_util_both_sources_flag",
            (
                (F.col("cardx_utilization") > 0.8) &
                (F.col("bureau_utilization") > 0.8)
            ).cast("int")
        )

        return combined_df.select(
            "cust_id", "as_of_month",
            "util_spread_cardx_bureau",
            "util_spread_volatility",
            "cardx_dependency_flag",
            "high_util_both_sources_flag"
        )

    def _compute_cross_trigger_features(self, combined_df: DataFrame) -> DataFrame:
        """
        Compute cross-trigger event features.

        Features:
        - cross_trigger_count_6m: Times when one triggered delinquency in other
        - cascade_delinquency_flag: Delinquency cascaded from CardX to bureau
        - synchronized_delinquency_flag: Both delinquent simultaneously
        """
        w = Window.partitionBy("cust_id").orderBy("as_of_month")
        w_6m = Window.partitionBy("cust_id").orderBy("as_of_month").rowsBetween(-5, 0)

        # Detect when CardX delinquency triggers bureau delinquency
        combined_df = combined_df.withColumn(
            "cardx_triggered_bureau",
            (
                (F.lag("cardx_delinquent_flag", 1).over(w) == 1) &
                (F.lag("bureau_delinquent_flag", 1).over(w) == 0) &
                (F.col("bureau_delinquent_flag") == 1)
            ).cast("int")
        )

        # Detect when bureau delinquency triggers CardX delinquency
        combined_df = combined_df.withColumn(
            "bureau_triggered_cardx",
            (
                (F.lag("bureau_delinquent_flag", 1).over(w) == 1) &
                (F.lag("cardx_delinquent_flag", 1).over(w) == 0) &
                (F.col("cardx_delinquent_flag") == 1)
            ).cast("int")
        )

        # Cross-trigger count (6m window)
        combined_df = combined_df.withColumn(
            "cross_trigger_count_6m",
            F.sum("cardx_triggered_bureau").over(w_6m) +
            F.sum("bureau_triggered_cardx").over(w_6m)
        )

        # Cascade flag (delinquency spread from one to other)
        combined_df = combined_df.withColumn(
            "cascade_delinquency_flag",
            (F.col("cross_trigger_count_6m") > 0).cast("int")
        )

        # Synchronized delinquency (both delinquent at same time)
        combined_df = combined_df.withColumn(
            "synchronized_delinquency_flag",
            (
                (F.col("cardx_delinquent_flag") == 1) &
                (F.col("bureau_delinquent_flag") == 1)
            ).cast("int")
        )

        # Compute cardx_early_warning_flag locally (CardX delinquent but bureau clean)
        combined_df = combined_df.withColumn(
            "cardx_early_warning_flag",
            (
                (F.col("cardx_delinquent_flag") == 1) &
                (F.col("bureau_delinquent_flag") == 0)
            ).cast("int")
        )

        # Information value (how much CardX adds beyond bureau)
        combined_df = combined_df.withColumn(
            "cardx_information_value",
            F.when(
                (F.col("cardx_early_warning_flag") == 1) |
                (F.col("cardx_triggered_bureau") == 1),
                1.0
            ).otherwise(0.0)
        )

        # Bureau blind spot score (what bureau misses)
        combined_df = combined_df.withColumn(
            "bureau_blind_spot_score",
            F.sum("cardx_early_warning_flag").over(w_6m) / 6.0
        )

        return combined_df.select(
            "cust_id", "as_of_month",
            "cross_trigger_count_6m",
            "cascade_delinquency_flag",
            "synchronized_delinquency_flag",
            "cardx_information_value",
            "bureau_blind_spot_score"
        )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from pyspark.sql import SparkSession
    from datetime import date, timedelta

    spark = SparkSession.builder \
        .appName("CardXBureauInteractionsTest") \
        .master("local[*]") \
        .getOrCreate()

    print("\n" + "="*70)
    print("CARDX-BUREAU INTERACTIONS ENGINE - EXAMPLE")
    print("="*70)

    # Create sample bureau data
    base_date = date(2024, 1, 1)
    bureau_data = []

    for month in range(6):
        as_of = base_date + timedelta(days=30 * month)
        bureau_data.extend([
            ("CUST001", "ACC001", as_of, month * 5, 10000 + month * 1000, 50000),
            ("CUST002", "ACC002", as_of, month * 15, 20000 + month * 2000, 100000),
        ])

    bureau_df = spark.createDataFrame(
        bureau_data,
        ["cust_id", "account_id", "as_of_month", "dpd", "balance", "credit_limit"]
    )

    # Create sample CardX data (CardX deteriorates faster for CUST001)
    cardx_data = []

    for month in range(6):
        as_of = base_date + timedelta(days=30 * month)
        cardx_data.extend([
            ("CUST001", as_of, month * 10, 8000 + month * 2000, 30000),  # Faster deterioration
            ("CUST002", as_of, month * 5, 15000 + month * 1000, 50000),  # Similar to bureau
        ])

    cardx_df = spark.createDataFrame(
        cardx_data,
        ["cust_id", "as_of_month", "cardx_dpd", "cardx_balance", "cardx_credit_limit"]
    )

    # Compute interaction features
    engine = CardXBureauInteractionsEngine(spark)
    features_df = engine.compute_all_features(bureau_df, cardx_df, None)

    # Show results
    print("\nLead-Lag Features:")
    features_df.select(
        "cust_id", "as_of_month",
        "cardx_leads_bureau_flag",
        "cardx_early_warning_flag",
        "lead_lag_months"
    ).orderBy("cust_id", "as_of_month").show()

    print("\nDivergence Features:")
    features_df.select(
        "cust_id", "as_of_month",
        "dpd_divergence_score",
        "velocity_divergence",
        "behavioral_consistency_score"
    ).orderBy("cust_id", "as_of_month").show()

    print("\nCross-Trigger Features:")
    features_df.select(
        "cust_id", "as_of_month",
        "cross_trigger_count_6m",
        "cardx_information_value",
        "bureau_blind_spot_score"
    ).orderBy("cust_id", "as_of_month").show()

    print("\n✅ CardX-Bureau Interactions test complete")
    print(f"\nTotal features computed: 20")
    print("  - Lead-Lag: 5 features")
    print("  - Divergence: 6 features")
    print("  - Utilization Spread: 4 features")
    print("  - Cross-Trigger: 5 features")

    spark.stop()
