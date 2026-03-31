"""
TDR/Restructuring Dynamics Engine - Restructuring Performance Analysis
======================================================================

Analyzes Time-Definite Repayment (TDR) / debt restructuring patterns:
- TDR history and frequency
- TDR performance and adherence
- Post-TDR cure dynamics
- TDR exhaustion and relapse indicators

Author: Behavioral Physics Team
Version: 1.0.0
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from typing import List, Dict

from .config import get_config


class TDRRestructuringEngine:
    """
    Computes TDR/restructuring dynamics features.

    Feature Families:
    1. TDR History (5 features): Restructuring count and recency
    2. TDR Velocity (3 features): Rate of restructuring requests
    3. TDR Performance (6 features): Adherence and breach
    4. Post-TDR Dynamics (5 features): Cure trajectory after TDR
    5. TDR Friction (3 features): Difficulty honoring TDR

    Total: 22 features
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.config = get_config()

    def compute_all_features(
        self,
        bureau_trade_df: DataFrame
    ) -> DataFrame:
        """
        Compute all TDR/restructuring features.

        Args:
            bureau_trade_df: Bureau trade data with TDR fields

        Expected columns:
        - last_tdr_date: Date of last debt restructuring (from DATEOFLASTDEBTRESTRUCTURE)
        - dpd: Days past due
        - balance: Outstanding balance
        - has_tdr: Flag if account has restructuring (added by schema adapter)

        Returns:
            DataFrame with TDR features at (cust_id, as_of_month) grain
        """
        # 1. Identify TDR events
        tdr_df = self._identify_tdr_events(bureau_trade_df)

        # 2. TDR history features
        history_features = self._compute_tdr_history_features(tdr_df)

        # 3. TDR velocity features
        velocity_features = self._compute_tdr_velocity_features(tdr_df)

        # 4. TDR performance features
        performance_features = self._compute_tdr_performance_features(tdr_df)

        # 5. Post-TDR dynamics features
        post_tdr_features = self._compute_post_tdr_dynamics(tdr_df)

        # 6. TDR friction features
        friction_features = self._compute_tdr_friction_features(tdr_df)

        # Combine all features
        result = history_features.join(
            velocity_features, on=["cust_id", "as_of_month"], how="outer"
        ).join(
            performance_features, on=["cust_id", "as_of_month"], how="outer"
        ).join(
            post_tdr_features, on=["cust_id", "as_of_month"], how="outer"
        ).join(
            friction_features, on=["cust_id", "as_of_month"], how="outer"
        )

        return result

    def _identify_tdr_events(self, bureau_trade_df: DataFrame) -> DataFrame:
        """
        Identify TDR events from last_tdr_date field.

        Also infer TDR status from account status if available.
        """
        df = bureau_trade_df

        # Flag if TDR date exists
        if "last_tdr_date" not in df.columns:
            df = df.withColumn("last_tdr_date", F.lit(None).cast("date"))

        df = df.withColumn(
            "has_tdr_flag",
            F.col("last_tdr_date").isNotNull().cast("int")
        )

        # Flag if currently on TDR (TDR date within last 12 months)
        df = df.withColumn(
            "currently_on_tdr",
            F.when(
                F.col("last_tdr_date").isNotNull(),
                (F.months_between(F.col("as_of_month"), F.col("last_tdr_date")) <= 12).cast("int")
            ).otherwise(0)
        )

        # Months since TDR
        df = df.withColumn(
            "months_since_tdr",
            F.when(
                F.col("last_tdr_date").isNotNull(),
                F.months_between(F.col("as_of_month"), F.col("last_tdr_date"))
            )
        )

        return df

    def _compute_tdr_history_features(self, tdr_df: DataFrame) -> DataFrame:
        """
        Compute TDR history features.

        Features:
        - tdr_count_lifetime: Total restructurings
        - tdr_count_12m: Recent restructurings
        - months_since_last_tdr: Time since last restructuring
        - currently_on_tdr_flag: Active restructuring
        - tdr_accounts_count: Number of accounts with TDR
        """
        # Aggregate to customer-month level
        tdr_agg = tdr_df.groupBy("cust_id", "as_of_month").agg(
            F.sum("has_tdr_flag").alias("tdr_accounts_count"),
            F.max("currently_on_tdr").alias("currently_on_tdr_flag"),
            F.min("months_since_tdr").alias("months_since_last_tdr")
        )

        # Count TDRs in lifetime (cumulative)
        w_all = Window.partitionBy("cust_id").orderBy("as_of_month").rowsBetween(
            Window.unboundedPreceding, 0
        )

        tdr_agg = tdr_agg.withColumn(
            "tdr_count_lifetime",
            F.sum("tdr_accounts_count").over(w_all)
        )

        # Count TDRs in last 12 months
        w_12m = Window.partitionBy("cust_id").orderBy("as_of_month").rowsBetween(-11, 0)

        tdr_agg = tdr_agg.withColumn(
            "tdr_count_12m",
            F.sum("tdr_accounts_count").over(w_12m)
        )

        return tdr_agg.select(
            "cust_id", "as_of_month",
            "tdr_count_lifetime",
            "tdr_count_12m",
            "months_since_last_tdr",
            "currently_on_tdr_flag",
            "tdr_accounts_count"
        )

    def _compute_tdr_velocity_features(self, tdr_df: DataFrame) -> DataFrame:
        """
        Compute TDR velocity features.

        Features:
        - tdr_velocity_12m: Rate of restructuring requests
        - time_between_tdrs_avg: Average gap between restructurings
        - tdr_exhaustion_flag: 3+ TDRs in 12 months (exhausted options)
        """
        # Aggregate TDR events
        tdr_agg = tdr_df.groupBy("cust_id", "as_of_month").agg(
            F.sum("has_tdr_flag").alias("new_tdrs")
        )

        # TDR velocity (12m window)
        w_12m = Window.partitionBy("cust_id").orderBy("as_of_month").rowsBetween(-11, 0)

        tdr_agg = tdr_agg.withColumn(
            "tdr_velocity_12m",
            F.sum("new_tdrs").over(w_12m) / 12.0
        )

        # Track TDR dates
        w = Window.partitionBy("cust_id").orderBy("as_of_month")

        tdr_agg = tdr_agg.withColumn(
            "this_month_tdr",
            (F.col("new_tdrs") > 0).cast("int")
        )

        tdr_agg = tdr_agg.withColumn(
            "prev_tdr_month",
            F.when(
                F.col("this_month_tdr") == 1,
                F.lag("as_of_month", 1).over(w)
            )
        )

        # Time between TDRs
        tdr_agg = tdr_agg.withColumn(
            "months_between_tdrs",
            F.when(
                F.col("prev_tdr_month").isNotNull(),
                F.months_between(F.col("as_of_month"), F.col("prev_tdr_month"))
            )
        )

        # Average time between TDRs (6m window)
        w_6m = Window.partitionBy("cust_id").orderBy("as_of_month").rowsBetween(-5, 0)

        tdr_agg = tdr_agg.withColumn(
            "time_between_tdrs_avg",
            F.avg("months_between_tdrs").over(w_6m)
        )

        # TDR exhaustion flag (3+ in 12m)
        tdr_agg = tdr_agg.withColumn(
            "tdr_exhaustion_flag",
            (F.sum("new_tdrs").over(w_12m) >= 3).cast("int")
        )

        return tdr_agg.select(
            "cust_id", "as_of_month",
            "tdr_velocity_12m",
            "time_between_tdrs_avg",
            "tdr_exhaustion_flag"
        )

    def _compute_tdr_performance_features(self, tdr_df: DataFrame) -> DataFrame:
        """
        Compute TDR performance features.

        Features:
        - tdr_adherence_rate: % of months with on-time payment (proxy)
        - tdr_breach_count: Number of breached TDRs
        - tdr_cure_success_rate: % of TDRs that led to full cure
        - post_tdr_delinquency_flag: Delinquent again after TDR
        - pre_tdr_dpd_avg: Average DPD before restructuring
        - tdr_severity_score: How severe was restructuring needed
        """
        # For each customer-month, check performance after TDR
        w = Window.partitionBy("cust_id").orderBy("as_of_month")

        # Flag delinquency after TDR
        tdr_df = tdr_df.withColumn(
            "delinquent_after_tdr",
            F.when(
                (F.col("months_since_tdr").isNotNull()) &
                (F.col("months_since_tdr") > 0) &
                (F.col("months_since_tdr") <= 12) &
                (F.col("dpd") > 30),
                1
            ).otherwise(0)
        )

        # Aggregate to customer-month
        tdr_agg = tdr_df.groupBy("cust_id", "as_of_month").agg(
            F.sum("delinquent_after_tdr").alias("tdr_breach_count"),
            F.avg(
                F.when(F.col("months_since_tdr") < 0, F.col("dpd"))
            ).alias("pre_tdr_dpd_avg")
        )

        # TDR adherence rate (proxy: % of post-TDR months without delinquency)
        w_12m = Window.partitionBy("cust_id").orderBy("as_of_month").rowsBetween(-11, 0)

        tdr_agg = tdr_agg.withColumn(
            "post_tdr_months",
            F.lit(12)  # Assuming 12m TDR period
        )

        tdr_agg = tdr_agg.withColumn(
            "tdr_adherence_rate",
            F.when(
                F.col("post_tdr_months") > 0,
                1.0 - (F.sum("tdr_breach_count").over(w_12m) / F.col("post_tdr_months"))
            ).otherwise(1.0)
        )

        # TDR cure success (no delinquency after TDR for 6+ months)
        w_6m = Window.partitionBy("cust_id").orderBy("as_of_month").rowsBetween(-5, 0)

        tdr_agg = tdr_agg.withColumn(
            "post_tdr_clean_months",
            6 - F.sum("tdr_breach_count").over(w_6m)
        )

        tdr_agg = tdr_agg.withColumn(
            "tdr_cure_success_flag",
            (F.col("post_tdr_clean_months") >= 6).cast("int")
        )

        # TDR cure success rate
        w_all = Window.partitionBy("cust_id").orderBy("as_of_month").rowsBetween(
            Window.unboundedPreceding, 0
        )

        tdr_agg = tdr_agg.withColumn(
            "tdr_cure_success_rate",
            F.avg("tdr_cure_success_flag").over(w_all)
        )

        # Post-TDR delinquency flag
        tdr_agg = tdr_agg.withColumn(
            "post_tdr_delinquency_flag",
            (F.col("tdr_breach_count") > 0).cast("int")
        )

        # TDR severity score (based on pre-TDR DPD)
        tdr_agg = tdr_agg.withColumn(
            "tdr_severity_score",
            F.when(F.col("pre_tdr_dpd_avg") > 180, 1.0)
             .when(F.col("pre_tdr_dpd_avg") > 90, 0.7)
             .when(F.col("pre_tdr_dpd_avg") > 30, 0.4)
             .otherwise(0.0)
        )

        return tdr_agg.select(
            "cust_id", "as_of_month",
            "tdr_adherence_rate",
            "tdr_breach_count",
            "tdr_cure_success_rate",
            "post_tdr_delinquency_flag",
            "pre_tdr_dpd_avg",
            "tdr_severity_score"
        )

    def _compute_post_tdr_dynamics(self, tdr_df: DataFrame) -> DataFrame:
        """
        Compute post-TDR behavioral dynamics.

        Features:
        - post_tdr_cure_velocity: Speed of DPD reduction after TDR
        - post_tdr_payment_consistency: Payment regularity after TDR
        - tdr_cure_halflife: Time to reduce DPD by 50% post-TDR
        - post_tdr_behavioral_stability: Entropy after restructuring
        - tdr_relapse_probability: Probability of re-default after TDR
        """
        # Filter to post-TDR period (0-12 months after TDR)
        post_tdr = tdr_df.filter(
            (F.col("months_since_tdr").isNotNull()) &
            (F.col("months_since_tdr") >= 0) &
            (F.col("months_since_tdr") <= 12)
        )

        # Group by customer-month
        w = Window.partitionBy("cust_id").orderBy("as_of_month")

        # Post-TDR cure velocity (DPD reduction rate)
        post_tdr = post_tdr.withColumn(
            "post_tdr_dpd_change",
            F.col("dpd") - F.lag("dpd", 1).over(w)
        )

        post_tdr_agg = post_tdr.groupBy("cust_id", "as_of_month").agg(
            F.avg("post_tdr_dpd_change").alias("post_tdr_cure_velocity"),
            F.count("*").alias("post_tdr_months_tracked")
        )

        # Post-TDR payment consistency (% of months improving)
        w_6m = Window.partitionBy("cust_id").orderBy("as_of_month").rowsBetween(-5, 0)

        post_tdr_agg = post_tdr_agg.withColumn(
            "improving_months",
            F.when(F.col("post_tdr_cure_velocity") < 0, 1).otherwise(0)
        )

        post_tdr_agg = post_tdr_agg.withColumn(
            "post_tdr_payment_consistency",
            F.sum("improving_months").over(w_6m) / 6.0
        )

        # TDR cure half-life (months to reduce DPD by 50%)
        # Simplified: track if DPD reduced by >50% within 6 months
        post_tdr_agg = post_tdr_agg.withColumn(
            "dpd_reduced_50pct",
            (F.col("post_tdr_cure_velocity") * 6 <= -50).cast("int")
        )

        post_tdr_agg = post_tdr_agg.withColumn(
            "tdr_cure_halflife",
            F.when(
                F.col("dpd_reduced_50pct") == 1, 6.0
            ).otherwise(12.0)  # Default to 12 if not achieved
        )

        # Post-TDR behavioral stability (inverse of volatility)
        post_tdr_agg = post_tdr_agg.withColumn(
            "dpd_volatility",
            F.stddev("post_tdr_cure_velocity").over(w_6m)
        )

        post_tdr_agg = post_tdr_agg.withColumn(
            "post_tdr_behavioral_stability",
            F.lit(1.0) / (F.lit(1.0) + F.coalesce(F.col("dpd_volatility"), F.lit(0.0)))
        )

        # TDR relapse probability (worsening trend)
        post_tdr_agg = post_tdr_agg.withColumn(
            "tdr_relapse_probability",
            F.when(
                F.col("post_tdr_cure_velocity") > 5, 0.8
            ).when(
                F.col("post_tdr_cure_velocity") > 0, 0.5
            ).otherwise(0.2)
        )

        return post_tdr_agg.select(
            "cust_id", "as_of_month",
            "post_tdr_cure_velocity",
            "post_tdr_payment_consistency",
            "tdr_cure_halflife",
            "post_tdr_behavioral_stability",
            "tdr_relapse_probability"
        )

    def _compute_tdr_friction_features(self, tdr_df: DataFrame) -> DataFrame:
        """
        Compute TDR friction (difficulty honoring TDR).

        Features:
        - tdr_friction_score: Difficulty in honoring TDR terms
        - tdr_momentum_score: Positive momentum post-TDR
        - multiple_lender_tdr_flag: Restructured with multiple lenders
        """
        # Aggregate by customer-month
        tdr_agg = tdr_df.groupBy("cust_id", "as_of_month").agg(
            F.sum("has_tdr_flag").alias("tdr_account_count"),
            F.avg(
                F.when(
                    (F.col("months_since_tdr").isNotNull()) &
                    (F.col("months_since_tdr") >= 0),
                    F.col("dpd")
                )
            ).alias("avg_dpd_post_tdr")
        )

        # TDR friction score (high DPD post-TDR = high friction)
        tdr_agg = tdr_agg.withColumn(
            "tdr_friction_score",
            F.when(F.col("avg_dpd_post_tdr") > 60, 1.0)
             .when(F.col("avg_dpd_post_tdr") > 30, 0.5)
             .otherwise(0.0)
        )

        # TDR momentum score (inverse of friction)
        tdr_agg = tdr_agg.withColumn(
            "tdr_momentum_score",
            F.lit(1.0) - F.col("tdr_friction_score")
        )

        # Multiple lender TDR flag
        tdr_agg = tdr_agg.withColumn(
            "multiple_lender_tdr_flag",
            (F.col("tdr_account_count") >= 2).cast("int")
        )

        return tdr_agg.select(
            "cust_id", "as_of_month",
            "tdr_friction_score",
            "tdr_momentum_score",
            "multiple_lender_tdr_flag"
        )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from pyspark.sql import SparkSession
    from datetime import date, timedelta

    spark = SparkSession.builder \
        .appName("TDRRestructuringTest") \
        .master("local[*]") \
        .getOrCreate()

    print("\n" + "="*70)
    print("TDR/RESTRUCTURING ENGINE - EXAMPLE")
    print("="*70)

    # Create sample bureau data with TDR information
    base_date = date(2024, 1, 1)
    tdr_data = []

    for month in range(12):
        as_of = base_date + timedelta(days=30 * month)

        # Customer 001: Got TDR at month 3, improving
        if month < 3:
            tdr_date_001 = None
            dpd_001 = 90 + month * 10
        else:
            tdr_date_001 = base_date + timedelta(days=90)
            dpd_001 = max(0, 120 - (month - 3) * 15)  # Improving

        # Customer 002: Got TDR but relapsed
        if month < 6:
            tdr_date_002 = None
            dpd_002 = 60
        else:
            tdr_date_002 = base_date + timedelta(days=180)
            dpd_002 = 30 + (month - 6) * 20  # Worsening

        tdr_data.extend([
            ("CUST001", "ACC001", as_of, tdr_date_001, dpd_001, 10000),
            ("CUST002", "ACC002", as_of, tdr_date_002, dpd_002, 20000),
        ])

    tdr_df = spark.createDataFrame(
        tdr_data,
        ["cust_id", "account_id", "as_of_month", "last_tdr_date", "dpd", "balance"]
    )

    # Compute TDR features
    engine = TDRRestructuringEngine(spark)
    features_df = engine.compute_all_features(tdr_df)

    # Show results
    print("\nTDR History Features:")
    features_df.select(
        "cust_id", "as_of_month",
        "tdr_count_lifetime",
        "months_since_last_tdr",
        "currently_on_tdr_flag"
    ).orderBy("cust_id", "as_of_month").show()

    print("\nTDR Performance Features:")
    features_df.select(
        "cust_id", "as_of_month",
        "tdr_adherence_rate",
        "tdr_cure_success_rate",
        "post_tdr_delinquency_flag"
    ).orderBy("cust_id", "as_of_month").show()

    print("\nPost-TDR Dynamics:")
    features_df.select(
        "cust_id", "as_of_month",
        "post_tdr_cure_velocity",
        "tdr_relapse_probability",
        "tdr_momentum_score"
    ).orderBy("cust_id", "as_of_month").show()

    print("\n✅ TDR/Restructuring Engine test complete")
    print(f"\nTotal features computed: 22")
    print("  - TDR History: 5 features")
    print("  - TDR Velocity: 3 features")
    print("  - TDR Performance: 6 features")
    print("  - Post-TDR Dynamics: 5 features")
    print("  - TDR Friction: 3 features")

    spark.stop()
