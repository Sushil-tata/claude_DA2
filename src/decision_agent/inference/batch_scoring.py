"""
Batch Scoring Module

Production-ready batch scoring for daily inference jobs.

Workflow:
1. Load latest features from feature store/Delta
2. Apply business rules and filters
3. Run inference pipeline
4. Apply decision thresholds
5. Write decisions to output table
6. Send summary metrics to Slack

Usage:
    python src/decision_agent/inference/batch_scoring.py \
        --config conf/inference/production_config.yaml \
        --date 2024-12-01
"""

import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

from decision_agent.inference.inference_pipeline import InferencePipeline
from decision_agent.decisions.output_writer import write_decisions

logger = logging.getLogger(__name__)


class BatchScorer:
    """
    Batch scoring orchestrator for production inference.

    Handles:
    - Feature loading with temporal filtering
    - Business rule application
    - Inference pipeline execution
    - Decision threshold application
    - Output writing
    """

    def __init__(
        self,
        spark: SparkSession,
        config: Dict[str, Any]
    ):
        """
        Initialize batch scorer.

        Args:
            spark: Spark session
            config: Batch scoring configuration:
                - feature_table: Delta table with features
                - champion_model_uri: MLflow Champion model URI
                - challenger_model_uri: MLflow Challenger model URI (optional)
                - routing: Routing config
                - business_rules: Decision thresholds, overrides
                - output_table: Output Delta table
        """
        self.spark = spark
        self.config = config

        # Initialize inference pipeline
        self.pipeline = InferencePipeline(
            spark=spark,
            config=config,
            enable_ab_testing=config.get("enable_ab_testing", False)
        )

    def score_batch(
        self,
        score_date: Optional[str] = None,
        lookback_days: int = 1
    ) -> Dict[str, Any]:
        """
        Score a batch of customers.

        Args:
            score_date: Date to score (YYYY-MM-DD). Default: today
            lookback_days: Days to look back for customer data

        Returns:
            Results dictionary with metrics
        """
        if score_date is None:
            score_date = datetime.now().strftime("%Y-%m-%d")

        logger.info("=" * 80)
        logger.info(f"Batch Scoring for {score_date}")
        logger.info("=" * 80)

        results = {
            "score_date": score_date,
            "start_time": datetime.now().isoformat()
        }

        # Step 1: Load features
        logger.info("Step 1: Loading features...")
        features_df = self._load_features(score_date, lookback_days)
        num_customers = features_df.count()
        logger.info(f"  Loaded {num_customers:,} customers")
        results["num_customers_loaded"] = num_customers

        if num_customers == 0:
            logger.warning("No customers to score. Exiting.")
            results["status"] = "no_data"
            return results

        # Step 2: Apply business filters
        logger.info("Step 2: Applying business filters...")
        filtered_df = self._apply_business_filters(features_df)
        num_filtered = filtered_df.count()
        logger.info(f"  {num_filtered:,} customers after filtering")
        results["num_customers_filtered"] = num_filtered

        if num_filtered == 0:
            logger.warning("No customers remaining after filters. Exiting.")
            results["status"] = "all_filtered"
            return results

        # Step 3: Run inference
        logger.info("Step 3: Running inference pipeline...")
        predictions_df = self.pipeline.predict_batch(
            features_df=filtered_df,
            use_case_id=self.config.get("use_case_id", "income_estimation"),
            model_version=self.config.get("model_version", "production")
        )
        logger.info(f"  Generated {predictions_df.count():,} predictions")

        # Step 4: Apply decision thresholds
        logger.info("Step 4: Applying decision thresholds...")
        decisions_df = self._apply_decision_rules(predictions_df)

        # Log decision summary
        decision_summary = decisions_df.groupBy("decision").count().collect()
        results["decision_summary"] = {row["decision"]: row["count"] for row in decision_summary}
        for row in decision_summary:
            logger.info(f"  {row['decision']}: {row['count']:,}")

        # Step 5: Write decisions
        logger.info("Step 5: Writing decisions to output table...")
        output_table = self._write_decisions(decisions_df)
        results["output_table"] = output_table
        results["num_decisions_written"] = decisions_df.count()

        # Summary
        results["end_time"] = datetime.now().isoformat()
        results["status"] = "success"

        logger.info("=" * 80)
        logger.info("Batch Scoring Complete")
        logger.info(f"  Customers loaded: {results['num_customers_loaded']:,}")
        logger.info(f"  Customers scored: {results['num_customers_filtered']:,}")
        logger.info(f"  Decisions written: {results['num_decisions_written']:,}")
        logger.info(f"  Output table: {results['output_table']}")
        logger.info("=" * 80)

        return results

    def _load_features(
        self,
        score_date: str,
        lookback_days: int
    ) -> DataFrame:
        """
        Load customer features for scoring.

        Args:
            score_date: Date to score
            lookback_days: Days to look back

        Returns:
            Features DataFrame
        """
        feature_table = self.config["feature_table"]
        logger.info(f"  Loading from: {feature_table}")

        # Calculate date range
        end_date = datetime.strptime(score_date, "%Y-%m-%d")
        start_date = end_date - timedelta(days=lookback_days)

        logger.info(f"  Date range: {start_date.date()} to {end_date.date()}")

        # Load features with temporal filter
        features_df = self.spark.table(feature_table).filter(
            (F.col("as_of_date") >= start_date.strftime("%Y-%m-%d")) &
            (F.col("as_of_date") <= score_date)
        )

        # Get most recent features per customer
        from pyspark.sql import Window
        window_spec = Window.partitionBy("customer_id").orderBy(F.col("as_of_date").desc())

        latest_features = features_df.withColumn("_rank", F.row_number().over(window_spec)) \
            .filter(F.col("_rank") == 1) \
            .drop("_rank")

        return latest_features

    def _apply_business_filters(self, features_df: DataFrame) -> DataFrame:
        """
        Apply business rules to filter customers.

        Examples:
        - Exclude customers with recent decisions (cooldown period)
        - Exclude opted-out customers
        - Exclude test accounts

        Args:
            features_df: Features DataFrame

        Returns:
            Filtered DataFrame
        """
        filtered_df = features_df

        # Filter 1: Exclude customers with insufficient data (if flagged)
        if "data_quality_flag" in features_df.columns:
            exclude_insufficient = self.config.get("business_rules", {}).get("exclude_insufficient_data", True)
            if exclude_insufficient:
                initial_count = filtered_df.count()
                filtered_df = filtered_df.filter(F.col("data_quality_flag") != "insufficient")
                excluded = initial_count - filtered_df.count()
                if excluded > 0:
                    logger.info(f"    Excluded {excluded:,} customers with insufficient data")

        # Filter 2: Exclude customers below minimum age (if applicable)
        if "customer_age" in features_df.columns:
            min_age = self.config.get("business_rules", {}).get("min_customer_age", 18)
            if min_age:
                initial_count = filtered_df.count()
                filtered_df = filtered_df.filter(F.col("customer_age") >= min_age)
                excluded = initial_count - filtered_df.count()
                if excluded > 0:
                    logger.info(f"    Excluded {excluded:,} customers below age {min_age}")

        # Filter 3: Custom SQL filter (if provided)
        custom_filter = self.config.get("business_rules", {}).get("custom_filter_sql")
        if custom_filter:
            initial_count = filtered_df.count()
            filtered_df = filtered_df.filter(custom_filter)
            excluded = initial_count - filtered_df.count()
            if excluded > 0:
                logger.info(f"    Custom filter excluded {excluded:,} customers")

        return filtered_df

    def _apply_decision_rules(self, predictions_df: DataFrame) -> DataFrame:
        """
        Apply decision thresholds and business rules to predictions.

        Args:
            predictions_df: Predictions DataFrame

        Returns:
            DataFrame with 'decision' column added
        """
        # Get decision threshold from config
        threshold = self.config.get("business_rules", {}).get("decision_threshold", 50000)
        logger.info(f"    Decision threshold: ${threshold:,}")

        # Apply threshold
        decisions_df = predictions_df.withColumn(
            "decision",
            F.when(F.col("prediction") >= threshold, "approve")
             .when(F.col("prediction") < threshold, "decline")
             .otherwise("review")
        )

        # Add decision confidence
        decisions_df = decisions_df.withColumn(
            "decision_confidence",
            F.when(F.col("prediction") >= threshold * 1.2, "high")
             .when(F.col("prediction") >= threshold, "medium")
             .otherwise("low")
        )

        return decisions_df

    def _write_decisions(self, decisions_df: DataFrame) -> str:
        """
        Write decisions to output table.

        Args:
            decisions_df: Decisions DataFrame

        Returns:
            Output table name
        """
        output_table = self.config.get("output_table", "decision_agent.production_decisions")

        # Select output columns
        output_cols = [
            "customer_id",
            "prediction",
            "decision",
            "decision_confidence",
            "selected_model",
            "routing_reason",
            "reason_codes",
            "prediction_timestamp",
            "model_version",
            "use_case_id"
        ]

        # Add metadata
        output_df = decisions_df.select(*[c for c in output_cols if c in decisions_df.columns]) \
            .withColumn("created_timestamp", F.current_timestamp())

        # Write to Delta (append mode)
        output_df.write \
            .format("delta") \
            .mode("append") \
            .saveAsTable(output_table)

        logger.info(f"    ✓ Written to: {output_table}")

        return output_table


def main():
    """CLI entry point for batch scoring."""
    parser = argparse.ArgumentParser(description="Production batch scoring")
    parser.add_argument("--config", required=True, help="Path to inference config YAML")
    parser.add_argument("--date", help="Score date (YYYY-MM-DD), default: today")
    parser.add_argument("--lookback-days", type=int, default=1, help="Days to look back")
    parser.add_argument("--enable-ab", action="store_true", help="Enable A/B testing")

    args = parser.parse_args()

    # Load config
    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Update config with CLI args
    if args.enable_ab:
        config["enable_ab_testing"] = True

    # Initialize Spark
    from decision_agent.utils.spark_utils import get_spark_session
    spark = get_spark_session()

    # Run batch scoring
    scorer = BatchScorer(spark, config)
    results = scorer.score_batch(
        score_date=args.date,
        lookback_days=args.lookback_days
    )

    # Print results
    import json
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
