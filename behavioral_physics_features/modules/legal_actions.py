"""
Legal Actions Engine - Legal Status and Settlement Dynamics
===========================================================

Analyzes legal actions, settlements, and write-offs:
- Legal status tracking and transitions
- Settlement dynamics and breach patterns
- Write-off indicators and velocity
- Legal friction (resistance to resolution)

Author: Behavioral Physics Team
Version: 1.0.0
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from typing import List, Dict

from .config import get_config


class LegalActionsEngine:
    """
    Computes legal action and settlement features.

    Feature Families:
    1. Legal Status (5 features): Legal action tracking
    2. Legal Velocity (3 features): Rate of legal actions
    3. Settlement Dynamics (5 features): Settlement patterns
    4. Write-off Indicators (3 features): Write-off tracking
    5. Legal Friction (2 features): Resistance to resolution

    Total: 18 features
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.config = get_config()

    def compute_all_features(
        self,
        bureau_trade_df: DataFrame
    ) -> DataFrame:
        """
        Compute all legal action features.

        Args:
            bureau_trade_df: Bureau trade data with legal/settlement fields

        Expected columns (from bureau):
        - account_status: May contain "WRITTEN OFF", "SETTLED", "SUIT FILED"
        - default_date: Date of default
        - account_close_date: When account was closed

        Returns:
            DataFrame with legal features at (cust_id, as_of_month) grain
        """
        # 1. Identify legal actions from account status
        legal_df = self._identify_legal_actions(bureau_trade_df)

        # 2. Legal status features
        status_features = self._compute_legal_status_features(legal_df)

        # 3. Legal velocity features
        velocity_features = self._compute_legal_velocity_features(legal_df)

        # 4. Settlement features
        settlement_features = self._compute_settlement_features(legal_df)

        # 5. Write-off features
        writeoff_features = self._compute_writeoff_features(legal_df)

        # 6. Legal friction features
        friction_features = self._compute_legal_friction_features(legal_df)

        # Combine all features
        result = status_features.join(
            velocity_features, on=["cust_id", "as_of_month"], how="outer"
        ).join(
            settlement_features, on=["cust_id", "as_of_month"], how="outer"
        ).join(
            writeoff_features, on=["cust_id", "as_of_month"], how="outer"
        ).join(
            friction_features, on=["cust_id", "as_of_month"], how="outer"
        )

        return result

    def _identify_legal_actions(self, bureau_trade_df: DataFrame) -> DataFrame:
        """
        Identify legal actions from account status field.

        Legal status indicators:
        - WRITTEN OFF, WRITE OFF, WRITTEN-OFF
        - SETTLED, SETTLEMENT
        - SUIT FILED, SUIT-FILED, LEGAL ACTION
        """
        # Create legal status flags
        df = bureau_trade_df.withColumn(
            "is_written_off",
            F.when(
                F.upper(F.col("account_status")).rlike(
                    "WRIT.*OFF|WO|WRITE.OFF"
                ), 1
            ).otherwise(0)
        )

        df = df.withColumn(
            "is_settled",
            F.when(
                F.upper(F.col("account_status")).rlike(
                    "SETTL|COMPROMISE"
                ), 1
            ).otherwise(0)
        )

        df = df.withColumn(
            "is_suit_filed",
            F.when(
                F.upper(F.col("account_status")).rlike(
                    "SUIT|LEGAL.*ACTION|COURT|LITIGATION"
                ), 1
            ).otherwise(0)
        )

        # Any legal action flag
        df = df.withColumn(
            "has_legal_action",
            (
                (F.col("is_written_off") == 1) |
                (F.col("is_settled") == 1) |
                (F.col("is_suit_filed") == 1)
            ).cast("int")
        )

        # Legal status category
        df = df.withColumn(
            "legal_status",
            F.when(F.col("is_written_off") == 1, "WRITTEN_OFF")
             .when(F.col("is_settled") == 1, "SETTLED")
             .when(F.col("is_suit_filed") == 1, "SUIT_FILED")
             .otherwise("NONE")
        )

        return df

    def _compute_legal_status_features(self, legal_df: DataFrame) -> DataFrame:
        """
        Compute legal status tracking features.

        Features:
        - has_legal_action_flag: Any legal action filed
        - num_legal_actions_12m: Count of legal cases
        - months_since_first_legal: Time since first legal action
        - legal_status_current: Current legal status
        """
        # Aggregate to customer-month level
        legal_agg = legal_df.groupBy("cust_id", "as_of_month").agg(
            F.max("has_legal_action").alias("has_legal_action_flag"),
            F.sum("has_legal_action").alias("num_legal_accounts"),
            F.sum("is_written_off").alias("num_written_off_accounts"),
            F.sum("is_settled").alias("num_settled_accounts"),
            F.sum("is_suit_filed").alias("num_suit_filed_accounts"),
            F.first("legal_status").alias("legal_status_current")
        )

        # Count legal actions in last 12 months
        w_12m = Window.partitionBy("cust_id").orderBy("as_of_month").rowsBetween(-11, 0)

        legal_agg = legal_agg.withColumn(
            "num_legal_actions_12m",
            F.sum("has_legal_action_flag").over(w_12m)
        )

        # Track first legal action
        w = Window.partitionBy("cust_id").orderBy("as_of_month")

        legal_agg = legal_agg.withColumn(
            "first_legal_month",
            F.when(
                (F.col("has_legal_action_flag") == 1) &
                (F.lag("has_legal_action_flag", 1).over(w) != 1),
                F.col("as_of_month")
            )
        )

        # Forward-fill first legal month
        legal_agg = legal_agg.withColumn(
            "first_legal_month",
            F.last("first_legal_month", ignorenulls=True).over(
                w.rowsBetween(Window.unboundedPreceding, 0)
            )
        )

        # Months since first legal action
        legal_agg = legal_agg.withColumn(
            "months_since_first_legal",
            F.when(
                F.col("first_legal_month").isNotNull(),
                F.months_between(F.col("as_of_month"), F.col("first_legal_month"))
            ).otherwise(999)  # High value if no legal action
        )

        # Total legal exposure (accounts with legal action)
        legal_agg = legal_agg.withColumn(
            "legal_exposure_accounts",
            F.col("num_written_off_accounts") +
            F.col("num_settled_accounts") +
            F.col("num_suit_filed_accounts")
        )

        return legal_agg.select(
            "cust_id", "as_of_month",
            "has_legal_action_flag",
            "num_legal_actions_12m",
            "months_since_first_legal",
            "legal_status_current",
            "legal_exposure_accounts"
        )

    def _compute_legal_velocity_features(self, legal_df: DataFrame) -> DataFrame:
        """
        Compute legal action velocity.

        Features:
        - legal_action_velocity_6m: Rate of new legal cases
        - legal_acceleration: Sudden spike in legal actions
        - legal_cascade_flag: Multiple legal actions in short period
        """
        # Aggregate to customer-month
        legal_agg = legal_df.groupBy("cust_id", "as_of_month").agg(
            F.sum("has_legal_action").alias("new_legal_actions")
        )

        # Legal action velocity (6m window)
        w_6m = Window.partitionBy("cust_id").orderBy("as_of_month").rowsBetween(-5, 0)

        legal_agg = legal_agg.withColumn(
            "legal_action_velocity_6m",
            F.sum("new_legal_actions").over(w_6m) / 6.0
        )

        # Legal acceleration
        w = Window.partitionBy("cust_id").orderBy("as_of_month")

        legal_agg = legal_agg.withColumn(
            "legal_acceleration",
            F.col("legal_action_velocity_6m") -
            F.lag("legal_action_velocity_6m", 3).over(w)
        )

        # Legal cascade flag (3+ legal actions in 6m)
        legal_agg = legal_agg.withColumn(
            "legal_cascade_flag",
            (F.sum("new_legal_actions").over(w_6m) >= 3).cast("int")
        )

        return legal_agg.select(
            "cust_id", "as_of_month",
            "legal_action_velocity_6m",
            "legal_acceleration",
            "legal_cascade_flag"
        )

    def _compute_settlement_features(self, legal_df: DataFrame) -> DataFrame:
        """
        Compute settlement dynamics features.

        Features:
        - settlement_attempt_count: Number of settlements
        - settlement_success_rate: % of settlements honored (proxy)
        - settlement_breach_count: Settlements that failed (re-delinquent after settlement)
        - post_settlement_cure_flag: Cured after settlement
        """
        # Aggregate settlements
        settlement_agg = legal_df.groupBy("cust_id", "as_of_month").agg(
            F.sum("is_settled").alias("settlement_count"),
            F.sum("is_written_off").alias("writeoff_after_settlement")
        )

        # Count settlements in lifetime
        w_all = Window.partitionBy("cust_id").orderBy("as_of_month").rowsBetween(
            Window.unboundedPreceding, 0
        )

        settlement_agg = settlement_agg.withColumn(
            "settlement_attempt_count",
            F.sum("settlement_count").over(w_all)
        )

        # Settlement breach (write-off after settlement)
        settlement_agg = settlement_agg.withColumn(
            "settlement_breach_count",
            F.sum("writeoff_after_settlement").over(w_all)
        )

        # Settlement success rate (proxy: settlements - breaches) / settlements
        settlement_agg = settlement_agg.withColumn(
            "settlement_success_rate",
            F.when(
                F.col("settlement_attempt_count") > 0,
                (F.col("settlement_attempt_count") - F.col("settlement_breach_count")) /
                F.col("settlement_attempt_count")
            ).otherwise(0.0)
        )

        # Track if settled this month
        w = Window.partitionBy("cust_id").orderBy("as_of_month")

        settlement_agg = settlement_agg.withColumn(
            "settled_this_month",
            (F.col("settlement_count") > 0).cast("int")
        )

        # Post-settlement cure flag (no write-off after settlement)
        settlement_agg = settlement_agg.withColumn(
            "post_settlement_cure_flag",
            (
                (F.col("settlement_attempt_count") > 0) &
                (F.col("settlement_breach_count") == 0)
            ).cast("int")
        )

        # Months since last settlement
        settlement_agg = settlement_agg.withColumn(
            "last_settlement_month",
            F.when(
                F.col("settled_this_month") == 1,
                F.col("as_of_month")
            )
        )

        settlement_agg = settlement_agg.withColumn(
            "last_settlement_month",
            F.last("last_settlement_month", ignorenulls=True).over(
                w.rowsBetween(Window.unboundedPreceding, 0)
            )
        )

        settlement_agg = settlement_agg.withColumn(
            "months_since_settlement",
            F.when(
                F.col("last_settlement_month").isNotNull(),
                F.months_between(F.col("as_of_month"), F.col("last_settlement_month"))
            )
        )

        return settlement_agg.select(
            "cust_id", "as_of_month",
            "settlement_attempt_count",
            "settlement_success_rate",
            "settlement_breach_count",
            "post_settlement_cure_flag",
            "months_since_settlement"
        )

    def _compute_writeoff_features(self, legal_df: DataFrame) -> DataFrame:
        """
        Compute write-off indicator features.

        Features:
        - written_off_accounts_count: Number of accounts written off
        - writeoff_velocity_3m: Rate of write-offs
        - writeoff_to_total_ratio: % of accounts written off
        """
        # Aggregate write-offs
        writeoff_agg = legal_df.groupBy("cust_id", "as_of_month").agg(
            F.sum("is_written_off").alias("new_writeoffs"),
            F.count("*").alias("total_accounts")
        )

        # Cumulative write-offs
        w_all = Window.partitionBy("cust_id").orderBy("as_of_month").rowsBetween(
            Window.unboundedPreceding, 0
        )

        writeoff_agg = writeoff_agg.withColumn(
            "written_off_accounts_count",
            F.sum("new_writeoffs").over(w_all)
        )

        # Write-off velocity (3m window)
        w_3m = Window.partitionBy("cust_id").orderBy("as_of_month").rowsBetween(-2, 0)

        writeoff_agg = writeoff_agg.withColumn(
            "writeoff_velocity_3m",
            F.sum("new_writeoffs").over(w_3m) / 3.0
        )

        # Write-off ratio
        writeoff_agg = writeoff_agg.withColumn(
            "writeoff_to_total_ratio",
            F.when(
                F.col("total_accounts") > 0,
                F.col("written_off_accounts_count") / F.col("total_accounts")
            ).otherwise(0.0)
        )

        return writeoff_agg.select(
            "cust_id", "as_of_month",
            "written_off_accounts_count",
            "writeoff_velocity_3m",
            "writeoff_to_total_ratio"
        )

    def _compute_legal_friction_features(self, legal_df: DataFrame) -> DataFrame:
        """
        Compute legal friction (resistance to resolution).

        Features:
        - legal_friction_score: Difficulty in resolving legal cases
        - legal_state_trap_prob: Probability stuck in legal status
        """
        # Aggregate legal status
        legal_agg = legal_df.groupBy("cust_id", "as_of_month").agg(
            F.max("has_legal_action").alias("has_legal"),
            F.sum("is_suit_filed").alias("suit_count"),
            F.sum("is_settled").alias("settlement_count"),
            F.sum("is_written_off").alias("writeoff_count")
        )

        # Count consecutive months with legal action
        w = Window.partitionBy("cust_id").orderBy("as_of_month")
        w_6m = Window.partitionBy("cust_id").orderBy("as_of_month").rowsBetween(-5, 0)

        legal_agg = legal_agg.withColumn(
            "consecutive_legal_months",
            F.sum("has_legal").over(w_6m)
        )

        # Legal friction score (high if stuck in legal for long time)
        legal_agg = legal_agg.withColumn(
            "legal_friction_score",
            F.when(
                F.col("consecutive_legal_months") >= 6, 1.0
            ).when(
                F.col("consecutive_legal_months") >= 3, 0.5
            ).otherwise(0.0)
        )

        # Legal state trap probability (prob of staying in legal state)
        legal_agg = legal_agg.withColumn(
            "legal_state_trap_prob",
            F.col("consecutive_legal_months") / 6.0
        )

        return legal_agg.select(
            "cust_id", "as_of_month",
            "legal_friction_score",
            "legal_state_trap_prob"
        )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from pyspark.sql import SparkSession
    from datetime import date, timedelta

    spark = SparkSession.builder \
        .appName("LegalActionsTest") \
        .master("local[*]") \
        .getOrCreate()

    print("\n" + "="*70)
    print("LEGAL ACTIONS ENGINE - EXAMPLE")
    print("="*70)

    # Create sample bureau data with legal statuses
    base_date = date(2024, 1, 1)
    legal_data = []

    for month in range(12):
        as_of = base_date + timedelta(days=30 * month)

        # Customer 001: Progressive legal action
        if month < 3:
            status_001 = "ACTIVE"
        elif month < 6:
            status_001 = "SUIT FILED"
        elif month < 9:
            status_001 = "SETTLED"
        else:
            status_001 = "WRITTEN OFF"

        # Customer 002: Settled successfully
        if month < 6:
            status_002 = "ACTIVE"
        else:
            status_002 = "SETTLED"

        legal_data.extend([
            ("CUST001", "ACC001", as_of, status_001, 10000),
            ("CUST002", "ACC002", as_of, status_002, 20000),
        ])

    legal_df = spark.createDataFrame(
        legal_data,
        ["cust_id", "account_id", "as_of_month", "account_status", "balance"]
    )

    # Compute legal features
    engine = LegalActionsEngine(spark)
    features_df = engine.compute_all_features(legal_df)

    # Show results
    print("\nLegal Status Features:")
    features_df.select(
        "cust_id", "as_of_month",
        "has_legal_action_flag",
        "legal_status_current",
        "months_since_first_legal"
    ).orderBy("cust_id", "as_of_month").show()

    print("\nSettlement Features:")
    features_df.select(
        "cust_id", "as_of_month",
        "settlement_attempt_count",
        "settlement_success_rate",
        "post_settlement_cure_flag"
    ).orderBy("cust_id", "as_of_month").show()

    print("\nWrite-off Features:")
    features_df.select(
        "cust_id", "as_of_month",
        "written_off_accounts_count",
        "writeoff_velocity_3m",
        "writeoff_to_total_ratio"
    ).orderBy("cust_id", "as_of_month").show()

    print("\n✅ Legal Actions Engine test complete")
    print(f"\nTotal features computed: 18")
    print("  - Legal Status: 5 features")
    print("  - Legal Velocity: 3 features")
    print("  - Settlement: 5 features")
    print("  - Write-off: 3 features")
    print("  - Legal Friction: 2 features")

    spark.stop()
