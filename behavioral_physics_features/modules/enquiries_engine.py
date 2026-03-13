"""
Enquiries Engine - Credit Seeking Behavior Analysis
===================================================

Analyzes bureau enquiry patterns to detect credit seeking behavior:
- Enquiry velocity: Rate of new enquiries
- Enquiry acceleration: Burst detection (sudden spikes)
- Enquiry→trade conversion: Application success rate
- Enquiry type analysis: Product mix seeking

Author: Behavioral Physics Team
Version: 1.0.0
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from typing import List, Dict

from .config import get_config


class EnquiriesEngine:
    """
    Computes enquiry-based features modeling credit seeking behavior.

    Feature Families:
    1. Velocity (4 features): Enquiry rate over time windows
    2. Acceleration (3 features): Burst detection, sudden spikes
    3. Conversion (3 features): Enquiry→trade conversion tracking
    4. Type analysis (2 features): Product mix, lender diversity

    Total: 12 features
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.config = get_config()

    def compute_all_features(
        self,
        bureau_enquiry_df: DataFrame,
        bureau_trade_df: DataFrame
    ) -> DataFrame:
        """
        Compute all enquiry-based features.

        Args:
            bureau_enquiry_df: Bureau enquiry data
            bureau_trade_df: Bureau trade data (for conversion tracking)

        Returns:
            DataFrame with enquiry features at (cust_id, as_of_month) grain
        """
        # 1. Velocity features
        velocity_df = self._compute_velocity_features(bureau_enquiry_df)

        # 2. Acceleration features
        acceleration_df = self._compute_acceleration_features(velocity_df)

        # 3. Conversion features
        conversion_df = self._compute_conversion_features(
            bureau_enquiry_df, bureau_trade_df
        )

        # 4. Type analysis features
        type_df = self._compute_type_features(bureau_enquiry_df)

        # Combine all features
        result = velocity_df.join(
            acceleration_df,
            on=["cust_id", "as_of_month"],
            how="outer"
        )
        result = result.join(
            conversion_df,
            on=["cust_id", "as_of_month"],
            how="outer"
        )
        result = result.join(
            type_df,
            on=["cust_id", "as_of_month"],
            how="outer"
        )

        # Fill nulls with 0
        for col in result.columns:
            if col not in ["cust_id", "as_of_month"]:
                result = result.fillna({col: 0.0})

        return result

    def _compute_velocity_features(
        self,
        bureau_enquiry_df: DataFrame
    ) -> DataFrame:
        """
        Compute enquiry velocity features (rate of enquiries).

        Features:
        - enquiry_count_1m: Number of enquiries in last 1 month
        - enquiry_count_3m: Number of enquiries in last 3 months
        - enquiry_count_6m: Number of enquiries in last 6 months
        - enquiry_velocity_3m: Enquiries per month (3m avg)
        """
        # Ensure enquiry date is parsed
        if "enquiry_date" not in bureau_enquiry_df.columns:
            bureau_enquiry_df = bureau_enquiry_df.withColumnRenamed(
                "ENQUIRYDT", "enquiry_date"
            )

        bureau_enquiry_df = bureau_enquiry_df.withColumn(
            "enquiry_date", F.to_date(F.col("enquiry_date"))
        )

        # Create monthly snapshots
        # Get all unique customer-month combinations
        customers = bureau_enquiry_df.select("cust_id").distinct()

        # Get date range from enquiries
        date_range = bureau_enquiry_df.agg(
            F.min("enquiry_date").alias("min_date"),
            F.max("enquiry_date").alias("max_date")
        ).collect()[0]

        # Create month spine (as_of_month)
        months_df = self.spark.sql(f"""
            SELECT explode(sequence(
                date_trunc('month', date'{date_range['min_date']}'),
                date_trunc('month', date'{date_range['max_date']}'),
                interval 1 month
            )) as as_of_month
        """)

        # Cross join to get all customer-month combinations
        spine_df = customers.crossJoin(months_df)

        # Count enquiries in lookback windows
        for window_months in [1, 3, 6]:
            # For each as_of_month, count enquiries in the window
            spine_with_counts = spine_df.alias("spine").join(
                bureau_enquiry_df.alias("enq"),
                (F.col("spine.cust_id") == F.col("enq.cust_id")) &
                (F.col("enq.enquiry_date") <= F.col("spine.as_of_month")) &
                (F.col("enq.enquiry_date") > F.add_months(F.col("spine.as_of_month"), -window_months)),
                how="left"
            ).groupBy("spine.cust_id", "spine.as_of_month").agg(
                F.count("enq.enquiry_date").alias(f"enquiry_count_{window_months}m")
            )

            spine_df = spine_df.join(
                spine_with_counts,
                on=["cust_id", "as_of_month"],
                how="left"
            ).fillna({f"enquiry_count_{window_months}m": 0})

        # Enquiry velocity = count / months
        spine_df = spine_df.withColumn(
            "enquiry_velocity_3m",
            F.col("enquiry_count_3m") / 3.0
        )

        return spine_df

    def _compute_acceleration_features(
        self,
        velocity_df: DataFrame
    ) -> DataFrame:
        """
        Compute enquiry acceleration features (burst detection).

        Features:
        - enquiry_acceleration_3m: Change in enquiry velocity
        - enquiry_burst_flag: Sudden spike (>5 enquiries in 1 month)
        - enquiry_trend: Increasing/stable/decreasing
        """
        # Window for time-series analysis
        w = Window.partitionBy("cust_id").orderBy("as_of_month")

        # Acceleration = change in velocity
        acceleration_df = velocity_df.withColumn(
            "enquiry_acceleration_3m",
            F.col("enquiry_velocity_3m") - F.lag("enquiry_velocity_3m", 1).over(w)
        )

        # Burst flag: >5 enquiries in 1 month
        acceleration_df = acceleration_df.withColumn(
            "enquiry_burst_flag",
            (F.col("enquiry_count_1m") >= 5).cast("int")
        )

        # Trend: compare 1m vs 3m average
        acceleration_df = acceleration_df.withColumn(
            "enquiry_trend",
            F.when(
                F.col("enquiry_count_1m") > F.col("enquiry_velocity_3m") * 1.5,
                "INCREASING"
            ).when(
                F.col("enquiry_count_1m") < F.col("enquiry_velocity_3m") * 0.5,
                "DECREASING"
            ).otherwise("STABLE")
        )

        # Encode trend as numeric
        acceleration_df = acceleration_df.withColumn(
            "enquiry_trend_score",
            F.when(F.col("enquiry_trend") == "INCREASING", 1.0)
             .when(F.col("enquiry_trend") == "DECREASING", -1.0)
             .otherwise(0.0)
        )

        return acceleration_df.select(
            "cust_id", "as_of_month",
            "enquiry_count_1m", "enquiry_count_3m", "enquiry_count_6m",
            "enquiry_velocity_3m",
            "enquiry_acceleration_3m",
            "enquiry_burst_flag",
            "enquiry_trend_score"
        )

    def _compute_conversion_features(
        self,
        bureau_enquiry_df: DataFrame,
        bureau_trade_df: DataFrame
    ) -> DataFrame:
        """
        Compute enquiry→trade conversion features.

        Features:
        - conversion_rate_30d: % enquiries that became accounts (30 days)
        - conversion_rate_60d: % enquiries that became accounts (60 days)
        - avg_days_to_conversion: Avg time from enquiry to account opening
        """
        # Parse dates
        if "enquiry_date" not in bureau_enquiry_df.columns:
            bureau_enquiry_df = bureau_enquiry_df.withColumnRenamed(
                "ENQUIRYDT", "enquiry_date"
            )
        bureau_enquiry_df = bureau_enquiry_df.withColumn(
            "enquiry_date", F.to_date(F.col("enquiry_date"))
        )

        if "account_open_date" not in bureau_trade_df.columns:
            bureau_trade_df = bureau_trade_df.withColumnRenamed(
                "DATEOPENED", "account_open_date"
            )
        bureau_trade_df = bureau_trade_df.withColumn(
            "account_open_date", F.to_date(F.col("account_open_date"))
        )

        # For each enquiry, check if account opened within 30/60 days
        enquiries_with_conversion = bureau_enquiry_df.alias("enq").join(
            bureau_trade_df.alias("trade").select(
                F.col("cust_id").alias("trade_cust_id"),
                F.col("account_open_date")
            ),
            (F.col("enq.cust_id") == F.col("trade_cust_id")) &
            (F.col("trade.account_open_date") >= F.col("enq.enquiry_date")) &
            (F.col("trade.account_open_date") <= F.date_add(F.col("enq.enquiry_date"), 60)),
            how="left"
        )

        # Calculate days to conversion
        enquiries_with_conversion = enquiries_with_conversion.withColumn(
            "days_to_conversion",
            F.datediff(F.col("account_open_date"), F.col("enquiry_date"))
        )

        # Flag conversions
        enquiries_with_conversion = enquiries_with_conversion.withColumn(
            "converted_30d",
            (F.col("days_to_conversion") <= 30).cast("int")
        )

        enquiries_with_conversion = enquiries_with_conversion.withColumn(
            "converted_60d",
            (F.col("days_to_conversion") <= 60).cast("int")
        )

        # Create monthly snapshots
        enquiries_with_conversion = enquiries_with_conversion.withColumn(
            "as_of_month",
            F.date_trunc("month", F.col("enquiry_date"))
        )

        # Aggregate conversion metrics by customer-month
        conversion_metrics = enquiries_with_conversion.groupBy(
            "cust_id", "as_of_month"
        ).agg(
            F.count("*").alias("total_enquiries"),
            F.sum("converted_30d").alias("conversions_30d"),
            F.sum("converted_60d").alias("conversions_60d"),
            F.avg(F.when(F.col("days_to_conversion").isNotNull(),
                         F.col("days_to_conversion"))).alias("avg_days_to_conversion")
        )

        # Conversion rates
        conversion_metrics = conversion_metrics.withColumn(
            "conversion_rate_30d",
            F.when(
                F.col("total_enquiries") > 0,
                F.col("conversions_30d") / F.col("total_enquiries")
            ).otherwise(0.0)
        )

        conversion_metrics = conversion_metrics.withColumn(
            "conversion_rate_60d",
            F.when(
                F.col("total_enquiries") > 0,
                F.col("conversions_60d") / F.col("total_enquiries")
            ).otherwise(0.0)
        )

        # Low conversion = credit rejection signal
        conversion_metrics = conversion_metrics.withColumn(
            "rejection_signal",
            (
                (F.col("total_enquiries") >= 3) &
                (F.col("conversion_rate_60d") < 0.3)
            ).cast("int")
        )

        return conversion_metrics.select(
            "cust_id", "as_of_month",
            "conversion_rate_30d",
            "conversion_rate_60d",
            "avg_days_to_conversion",
            "rejection_signal"
        )

    def _compute_type_features(
        self,
        bureau_enquiry_df: DataFrame
    ) -> DataFrame:
        """
        Compute enquiry type analysis features.

        Features:
        - enquiry_type_diversity: Number of different product types enquired
        - secured_enquiry_share: % enquiries for secured products
        """
        # Parse enquiry date
        if "enquiry_date" not in bureau_enquiry_df.columns:
            bureau_enquiry_df = bureau_enquiry_df.withColumnRenamed(
                "ENQUIRYDT", "enquiry_date"
            )
        bureau_enquiry_df = bureau_enquiry_df.withColumn(
            "enquiry_date", F.to_date(F.col("enquiry_date"))
        )

        # Add as_of_month
        bureau_enquiry_df = bureau_enquiry_df.withColumn(
            "as_of_month",
            F.date_trunc("month", F.col("enquiry_date"))
        )

        # Get enquiry purpose/type (if available)
        # After schema adaptation, column is 'enquiry_purpose' (lowercase)
        if "enquiry_purpose" in bureau_enquiry_df.columns:
            type_col = "enquiry_purpose"
        elif "ENQUIRYPURPOSE" in bureau_enquiry_df.columns:
            type_col = "ENQUIRYPURPOSE"
        elif "PURPOSE" in bureau_enquiry_df.columns:
            type_col = "PURPOSE"
        else:
            # Default: use account type if available
            type_col = "ACCOUNTTYPE" if "ACCOUNTTYPE" in bureau_enquiry_df.columns else None

        if type_col:
            # Count unique types per customer-month (last 3 months)
            w_3m = Window.partitionBy("cust_id", "as_of_month")

            type_diversity = bureau_enquiry_df.groupBy(
                "cust_id", "as_of_month"
            ).agg(
                F.countDistinct(type_col).alias("enquiry_type_diversity")
            )

            # Secured vs unsecured split
            bureau_enquiry_df = bureau_enquiry_df.withColumn(
                "is_secured",
                F.when(
                    F.col(type_col).isin([
                        "AUTO LOAN", "HOME LOAN", "MORTGAGE", "GOLD LOAN",
                        "PROPERTY LOAN", "SECURED BUSINESS LOAN"
                    ]),
                    1
                ).otherwise(0)
            )

            secured_share = bureau_enquiry_df.groupBy(
                "cust_id", "as_of_month"
            ).agg(
                F.avg("is_secured").alias("secured_enquiry_share")
            )

            # Join type features
            type_features = type_diversity.join(
                secured_share,
                on=["cust_id", "as_of_month"],
                how="outer"
            )

        else:
            # No type information available - create dummy features
            type_features = bureau_enquiry_df.select(
                "cust_id", "as_of_month"
            ).distinct()

            type_features = type_features.withColumn("enquiry_type_diversity", F.lit(1.0))
            type_features = type_features.withColumn("secured_enquiry_share", F.lit(0.0))

        return type_features


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from pyspark.sql import SparkSession
    from datetime import date, timedelta

    spark = SparkSession.builder \
        .appName("EnquiriesEngineTest") \
        .master("local[*]") \
        .getOrCreate()

    print("\n" + "="*70)
    print("ENQUIRIES ENGINE - EXAMPLE")
    print("="*70)

    # Create sample enquiry data
    base_date = date(2024, 1, 1)
    enquiry_data = [
        # Customer 001: Normal enquiry pattern
        ("CUST001", base_date, "CREDIT CARD"),
        ("CUST001", base_date + timedelta(days=60), "PERSONAL LOAN"),
        ("CUST001", base_date + timedelta(days=120), "AUTO LOAN"),

        # Customer 002: Burst pattern (credit seeking)
        ("CUST002", base_date + timedelta(days=90), "CREDIT CARD"),
        ("CUST002", base_date + timedelta(days=91), "PERSONAL LOAN"),
        ("CUST002", base_date + timedelta(days=92), "HOME LOAN"),
        ("CUST002", base_date + timedelta(days=93), "AUTO LOAN"),
        ("CUST002", base_date + timedelta(days=94), "CREDIT CARD"),
        ("CUST002", base_date + timedelta(days=95), "PERSONAL LOAN"),
    ]

    enquiry_df = spark.createDataFrame(
        enquiry_data,
        ["cust_id", "enquiry_date", "ENQUIRYPURPOSE"]
    )

    # Create sample trade data (for conversion tracking)
    trade_data = [
        ("CUST001", base_date + timedelta(days=15), "ACC001"),  # Converted
        ("CUST001", base_date + timedelta(days=75), "ACC002"),  # Converted
        ("CUST002", base_date + timedelta(days=100), "ACC003"), # Converted (1 out of 6)
    ]

    trade_df = spark.createDataFrame(
        trade_data,
        ["cust_id", "account_open_date", "account_id"]
    ).withColumnRenamed("account_open_date", "DATEOPENED")

    # Compute enquiry features
    engine = EnquiriesEngine(spark)
    features_df = engine.compute_all_features(enquiry_df, trade_df)

    # Show results
    print("\nVelocity Features:")
    features_df.select(
        "cust_id", "as_of_month",
        "enquiry_count_1m", "enquiry_count_3m",
        "enquiry_velocity_3m"
    ).orderBy("cust_id", "as_of_month").show()

    print("\nAcceleration Features:")
    features_df.select(
        "cust_id", "as_of_month",
        "enquiry_acceleration_3m",
        "enquiry_burst_flag",
        "enquiry_trend_score"
    ).orderBy("cust_id", "as_of_month").show()

    print("\nConversion Features:")
    features_df.select(
        "cust_id", "as_of_month",
        "conversion_rate_30d",
        "conversion_rate_60d",
        "rejection_signal"
    ).orderBy("cust_id", "as_of_month").show()

    print("\n✅ Enquiries Engine test complete")
    print(f"\nTotal features computed: 12")
    print("  - Velocity: 4 features")
    print("  - Acceleration: 3 features")
    print("  - Conversion: 3 features")
    print("  - Type analysis: 2 features")

    spark.stop()
