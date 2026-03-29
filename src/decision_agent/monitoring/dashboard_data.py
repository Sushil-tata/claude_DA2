"""
Dashboard Data Preparation

Prepares aggregated metrics for monitoring dashboards.

Outputs:
- Feature drift trends (PSI over time)
- Prediction drift trends (mean/std over time)
- Model performance trends (if labels available)
- Alert history
- Data quality metrics

Usage:
    prep = DashboardDataPrep(spark, config)
    prep.prepare_all_metrics(lookback_days=30)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

logger = logging.getLogger(__name__)


class DashboardDataPrep:
    """
    Prepare monitoring data for dashboards.
    """

    def __init__(
        self,
        spark: SparkSession,
        config: Dict[str, Any]
    ):
        """
        Initialize dashboard data prep.

        Args:
            spark: Spark session
            config: Dashboard configuration:
                - monitoring_metrics_table: Source table for monitoring metrics
                - prediction_table: Production predictions table
                - dashboard_output_table: Output table for dashboard data
        """
        self.spark = spark
        self.config = config
        self.monitoring_metrics_table = config.get("monitoring_metrics_table", "decision_agent.monitoring_metrics")
        self.prediction_table = config.get("prediction_table", "decision_agent.production_predictions")
        self.dashboard_output_table = config.get("dashboard_output_table", "decision_agent.dashboard_metrics")

    def prepare_all_metrics(
        self,
        lookback_days: int = 30
    ) -> Dict[str, DataFrame]:
        """
        Prepare all dashboard metrics.

        Args:
            lookback_days: Days to look back for historical data

        Returns:
            Dict of {metric_name: DataFrame}
        """
        logger.info(f"Preparing dashboard metrics (lookback: {lookback_days} days)...")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        metrics = {}

        # 1. Feature drift trends
        logger.info("  1. Feature drift trends...")
        metrics["feature_drift"] = self.prepare_feature_drift_trends(start_date, end_date)

        # 2. Prediction drift trends
        logger.info("  2. Prediction drift trends...")
        metrics["prediction_drift"] = self.prepare_prediction_drift_trends(start_date, end_date)

        # 3. Alert history
        logger.info("  3. Alert history...")
        metrics["alerts"] = self.prepare_alert_history(start_date, end_date)

        # 4. Model performance (if labels available)
        logger.info("  4. Model performance...")
        metrics["performance"] = self.prepare_performance_trends(start_date, end_date)

        # 5. Data quality metrics
        logger.info("  5. Data quality metrics...")
        metrics["data_quality"] = self.prepare_data_quality_trends(start_date, end_date)

        # Write to output table
        self._write_to_dashboard_table(metrics)

        logger.info("✓ Dashboard metrics prepared")
        return metrics

    def prepare_feature_drift_trends(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> DataFrame:
        """
        Prepare feature drift trends over time.

        Returns:
            DataFrame with columns: date, feature, psi_score
        """
        # Query monitoring metrics table
        sql = f"""
        SELECT
            DATE(monitoring_timestamp) as date,
            feature,
            psi_score
        FROM {self.monitoring_metrics_table},
        LATERAL VIEW EXPLODE(feature_psi_scores) AS feature, psi_score
        WHERE monitoring_timestamp BETWEEN '{start_date.strftime("%Y-%m-%d")}' AND '{end_date.strftime("%Y-%m-%d")}'
        ORDER BY date, feature
        """

        try:
            df = self.spark.sql(sql)
            logger.info(f"    Feature drift trends: {df.count()} records")
            return df
        except Exception as e:
            logger.warning(f"    Failed to query feature drift: {e}")
            return self.spark.createDataFrame([], "date string, feature string, psi_score double")

    def prepare_prediction_drift_trends(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> DataFrame:
        """
        Prepare prediction drift trends over time.

        Returns:
            DataFrame with columns: date, mean, std, min, max, p50
        """
        sql = f"""
        SELECT
            DATE(prediction_timestamp) as date,
            AVG(prediction) as mean,
            STDDEV(prediction) as std,
            MIN(prediction) as min,
            MAX(prediction) as max,
            PERCENTILE(prediction, 0.50) as p50,
            COUNT(*) as count
        FROM {self.prediction_table}
        WHERE prediction_timestamp BETWEEN '{start_date.strftime("%Y-%m-%d")}' AND '{end_date.strftime("%Y-%m-%d")}'
        GROUP BY DATE(prediction_timestamp)
        ORDER BY date
        """

        try:
            df = self.spark.sql(sql)
            logger.info(f"    Prediction drift trends: {df.count()} records")
            return df
        except Exception as e:
            logger.warning(f"    Failed to query prediction drift: {e}")
            return self.spark.createDataFrame([], "date string, mean double, std double, min double, max double, p50 double, count long")

    def prepare_alert_history(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> DataFrame:
        """
        Prepare alert history.

        Returns:
            DataFrame with columns: date, alert_type, severity, count
        """
        sql = f"""
        SELECT
            DATE(monitoring_timestamp) as date,
            alert_type,
            severity,
            COUNT(*) as count
        FROM {self.monitoring_metrics_table},
        LATERAL VIEW EXPLODE(alerts) AS alert
        WHERE monitoring_timestamp BETWEEN '{start_date.strftime("%Y-%m-%d")}' AND '{end_date.strftime("%Y-%m-%d")}'
        GROUP BY DATE(monitoring_timestamp), alert_type, severity
        ORDER BY date DESC
        """

        try:
            df = self.spark.sql(sql)
            logger.info(f"    Alert history: {df.count()} records")
            return df
        except Exception as e:
            logger.warning(f"    Failed to query alert history: {e}")
            return self.spark.createDataFrame([], "date string, alert_type string, severity string, count long")

    def prepare_performance_trends(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> DataFrame:
        """
        Prepare model performance trends (if labels available).

        Returns:
            DataFrame with columns: date, mae, rmse, r2
        """
        # Check if actual values available
        try:
            has_actuals = self.spark.sql(f"""
                SELECT COUNT(*) as count
                FROM {self.prediction_table}
                WHERE actual_value IS NOT NULL
                LIMIT 1
            """).collect()[0]["count"] > 0

            if not has_actuals:
                logger.info("    No actual values available yet. Skipping performance trends.")
                return self.spark.createDataFrame([], "date string, mae double, rmse double, r2 double")

            sql = f"""
            SELECT
                DATE(prediction_timestamp) as date,
                AVG(ABS(prediction - actual_value)) as mae,
                SQRT(AVG(POW(prediction - actual_value, 2))) as rmse,
                CORR(prediction, actual_value) as correlation,
                COUNT(*) as sample_size
            FROM {self.prediction_table}
            WHERE actual_value IS NOT NULL
              AND prediction_timestamp BETWEEN '{start_date.strftime("%Y-%m-%d")}' AND '{end_date.strftime("%Y-%m-%d")}'
            GROUP BY DATE(prediction_timestamp)
            ORDER BY date
            """

            df = self.spark.sql(sql)
            logger.info(f"    Performance trends: {df.count()} records")
            return df

        except Exception as e:
            logger.warning(f"    Failed to query performance trends: {e}")
            return self.spark.createDataFrame([], "date string, mae double, rmse double, correlation double, sample_size long")

    def prepare_data_quality_trends(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> DataFrame:
        """
        Prepare data quality trends.

        Returns:
            DataFrame with columns: date, metric, value
        """
        sql = f"""
        SELECT
            DATE(prediction_timestamp) as date,
            'null_rate' as metric,
            AVG(CASE WHEN prediction IS NULL THEN 1.0 ELSE 0.0 END) as value
        FROM {self.prediction_table}
        WHERE prediction_timestamp BETWEEN '{start_date.strftime("%Y-%m-%d")}' AND '{end_date.strftime("%Y-%m-%d")}'
        GROUP BY DATE(prediction_timestamp)

        UNION ALL

        SELECT
            DATE(prediction_timestamp) as date,
            'record_count' as metric,
            COUNT(*) as value
        FROM {self.prediction_table}
        WHERE prediction_timestamp BETWEEN '{start_date.strftime("%Y-%m-%d")}' AND '{end_date.strftime("%Y-%m-%d")}'
        GROUP BY DATE(prediction_timestamp)

        ORDER BY date, metric
        """

        try:
            df = self.spark.sql(sql)
            logger.info(f"    Data quality trends: {df.count()} records")
            return df
        except Exception as e:
            logger.warning(f"    Failed to query data quality: {e}")
            return self.spark.createDataFrame([], "date string, metric string, value double")

    def _write_to_dashboard_table(self, metrics: Dict[str, DataFrame]):
        """Write prepared metrics to dashboard output table."""
        logger.info(f"Writing dashboard metrics to: {self.dashboard_output_table}")

        # Combine all metrics with metric_type column
        all_metrics = []

        for metric_name, df in metrics.items():
            if df.count() > 0:
                df_with_type = df.withColumn("metric_type", F.lit(metric_name))
                all_metrics.append(df_with_type)

        if not all_metrics:
            logger.warning("No metrics to write")
            return

        # Union all metrics
        combined_df = all_metrics[0]
        for df in all_metrics[1:]:
            combined_df = combined_df.unionByName(df, allowMissingColumns=True)

        # Add refresh timestamp
        combined_df = combined_df.withColumn("refresh_timestamp", F.current_timestamp())

        # Write to output table
        combined_df.write \
            .format("delta") \
            .mode("overwrite") \
            .option("overwriteSchema", "true") \
            .saveAsTable(self.dashboard_output_table)

        logger.info(f"✓ Dashboard metrics written: {combined_df.count()} records")
