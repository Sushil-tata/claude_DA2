"""
Traffic Splitter for A/B Testing

Deterministic, hash-based traffic splitting for gradual rollout.

Key Properties:
- Deterministic: Same customer always sees same model (sticky)
- Fair: Distribution matches rollout percentage
- Hash-based: Uses customer_id hash for assignment
- Configurable: Rollout percentage from config/state

Example:
- 10% rollout: 10% customers → Challenger, 90% → Champion
- 50% rollout: 50% customers → Challenger, 50% → Champion
- 100% rollout: 100% customers → Challenger
"""

import hashlib
import logging
from typing import Dict, Any
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType

logger = logging.getLogger(__name__)


class TrafficSplitter:
    """
    Hash-based traffic splitter for A/B testing.

    Uses deterministic hashing to assign customers to models.
    """

    def __init__(
        self,
        rollout_percentage: int,
        hash_seed: int = 42,
        sticky: bool = True
    ):
        """
        Initialize traffic splitter.

        Args:
            rollout_percentage: Percentage of traffic to Challenger (0-100)
            hash_seed: Seed for hash function (for reproducibility)
            sticky: If True, same customer always sees same model
        """
        if not 0 <= rollout_percentage <= 100:
            raise ValueError(f"Rollout percentage must be 0-100, got {rollout_percentage}")

        self.rollout_percentage = rollout_percentage
        self.hash_seed = hash_seed
        self.sticky = sticky

        logger.info(f"TrafficSplitter initialized:")
        logger.info(f"  Rollout: {rollout_percentage}% to Challenger")
        logger.info(f"  Hash seed: {hash_seed}")
        logger.info(f"  Sticky: {sticky}")

    def assign_model(self, df: DataFrame, customer_id_col: str = "customer_id") -> DataFrame:
        """
        Assign each customer to Champion or Challenger.

        Args:
            df: Input DataFrame
            customer_id_col: Customer ID column name

        Returns:
            DataFrame with 'assigned_model' column added
        """
        logger.info(f"Assigning models with {self.rollout_percentage}% rollout...")

        # Define hash UDF
        @F.udf(returnType=IntegerType())
        def hash_customer_id(customer_id: str) -> int:
            """Hash customer_id to integer in [0, 99]."""
            hash_str = f"{customer_id}_{self.hash_seed}"
            hash_int = int(hashlib.md5(hash_str.encode()).hexdigest(), 16)
            return hash_int % 100

        # Apply hash and assign model
        df_with_assignment = df.withColumn(
            "_hash_bucket", hash_customer_id(F.col(customer_id_col))
        ).withColumn(
            "assigned_model",
            F.when(F.col("_hash_bucket") < self.rollout_percentage, "challenger")
             .otherwise("champion")
        ).drop("_hash_bucket")

        # Log assignment summary
        assignment_counts = df_with_assignment.groupBy("assigned_model").count().collect()
        for row in assignment_counts:
            model = row["assigned_model"]
            count = row["count"]
            percentage = count / df.count() * 100
            logger.info(f"  {model}: {count:,} customers ({percentage:.1f}%)")

        return df_with_assignment

    def verify_split(self, df: DataFrame, customer_id_col: str = "customer_id") -> Dict[str, Any]:
        """
        Verify that traffic split matches expected rollout percentage.

        Args:
            df: DataFrame with assigned_model column
            customer_id_col: Customer ID column

        Returns:
            Verification results
        """
        total_count = df.count()

        assignment_counts = df.groupBy("assigned_model").count().collect()
        counts = {row["assigned_model"]: row["count"] for row in assignment_counts}

        challenger_count = counts.get("challenger", 0)
        champion_count = counts.get("champion", 0)

        actual_challenger_pct = (challenger_count / total_count * 100) if total_count > 0 else 0
        actual_champion_pct = (champion_count / total_count * 100) if total_count > 0 else 0

        # Check if within tolerance (±2%)
        tolerance = 2.0
        expected_challenger = self.rollout_percentage
        expected_champion = 100 - self.rollout_percentage

        challenger_ok = abs(actual_challenger_pct - expected_challenger) <= tolerance
        champion_ok = abs(actual_champion_pct - expected_champion) <= tolerance

        results = {
            "total_customers": total_count,
            "champion_count": champion_count,
            "challenger_count": challenger_count,
            "champion_pct": actual_champion_pct,
            "challenger_pct": actual_challenger_pct,
            "expected_challenger_pct": expected_challenger,
            "expected_champion_pct": expected_champion,
            "within_tolerance": challenger_ok and champion_ok,
            "tolerance": tolerance
        }

        logger.info("Traffic Split Verification:")
        logger.info(f"  Expected: {expected_champion:.1f}% Champion, {expected_challenger:.1f}% Challenger")
        logger.info(f"  Actual: {actual_champion_pct:.1f}% Champion, {actual_challenger_pct:.1f}% Challenger")
        logger.info(f"  Status: {'✓ PASS' if results['within_tolerance'] else '✗ FAIL'}")

        return results

    def verify_stickiness(
        self,
        df1: DataFrame,
        df2: DataFrame,
        customer_id_col: str = "customer_id"
    ) -> Dict[str, Any]:
        """
        Verify that customer assignments are sticky (same across runs).

        Args:
            df1: DataFrame with assignments from run 1
            df2: DataFrame with assignments from run 2
            customer_id_col: Customer ID column

        Returns:
            Stickiness verification results
        """
        # Join on customer_id
        joined = df1.select(
            F.col(customer_id_col),
            F.col("assigned_model").alias("model_run1")
        ).join(
            df2.select(
                F.col(customer_id_col),
                F.col("assigned_model").alias("model_run2")
            ),
            on=customer_id_col,
            how="inner"
        )

        # Check consistency
        total_customers = joined.count()
        consistent_count = joined.filter(F.col("model_run1") == F.col("model_run2")).count()
        inconsistent_count = total_customers - consistent_count

        consistency_pct = (consistent_count / total_customers * 100) if total_customers > 0 else 0

        results = {
            "total_customers": total_customers,
            "consistent_count": consistent_count,
            "inconsistent_count": inconsistent_count,
            "consistency_pct": consistency_pct,
            "is_sticky": inconsistent_count == 0
        }

        logger.info("Stickiness Verification:")
        logger.info(f"  Consistent assignments: {consistent_count:,} / {total_customers:,} ({consistency_pct:.1f}%)")
        logger.info(f"  Status: {'✓ STICKY' if results['is_sticky'] else '✗ NOT STICKY'}")

        return results


def create_traffic_split(
    df: DataFrame,
    rollout_percentage: int,
    customer_id_col: str = "customer_id",
    hash_seed: int = 42
) -> DataFrame:
    """
    Convenience function to create traffic split.

    Args:
        df: Input DataFrame
        rollout_percentage: % to Challenger (0-100)
        customer_id_col: Customer ID column
        hash_seed: Hash seed for reproducibility

    Returns:
        DataFrame with assigned_model column
    """
    splitter = TrafficSplitter(
        rollout_percentage=rollout_percentage,
        hash_seed=hash_seed,
        sticky=True
    )

    return splitter.assign_model(df, customer_id_col)


# Example usage for testing
if __name__ == "__main__":
    from pyspark.sql import SparkSession

    # Initialize Spark
    spark = SparkSession.builder.appName("TrafficSplitterTest").getOrCreate()

    # Create test data
    test_data = spark.createDataFrame([
        {"customer_id": f"CUST_{i:05d}"} for i in range(1000)
    ])

    # Test 10% rollout
    splitter = TrafficSplitter(rollout_percentage=10, hash_seed=42)

    # Assign models
    assigned_df = splitter.assign_model(test_data)
    assigned_df.show(20)

    # Verify split
    verification = splitter.verify_split(assigned_df)
    print("\nVerification Results:")
    for k, v in verification.items():
        print(f"  {k}: {v}")

    # Test stickiness
    assigned_df2 = splitter.assign_model(test_data)
    stickiness = splitter.verify_stickiness(assigned_df, assigned_df2)
    print("\nStickiness Results:")
    for k, v in stickiness.items():
        print(f"  {k}: {v}")
