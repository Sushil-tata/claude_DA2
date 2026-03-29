"""
Model Monitor - Production Monitoring Orchestrator (CRITICAL P0)

Orchestrates all production monitoring:
1. Feature drift detection (PSI)
2. Prediction drift detection
3. Performance monitoring (if labels available with lag)
4. Alerting (Slack, PagerDuty, email)
5. Automated retraining triggers

Runs as Databricks scheduled job (daily).

Why This Matters:
- Early warning system for model degradation
- Detect data quality issues
- Trigger retraining automatically
- Prevent silent model failures

Integration Point:
- Databricks scheduled job (daily cron)
- Reads from: decision_agent.income_decisions
- Writes to: decision_agent.monitoring_metrics
- Alerts to: Slack, PagerDuty, email
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
import json

from decision_agent.monitoring.feature_drift_detector import FeatureDriftDetector
from decision_agent.monitoring.prediction_drift_detector import PredictionDriftDetector

logger = logging.getLogger(__name__)


class ModelMonitor:
    """
    Orchestrate all production monitoring.

    Run this daily as Databricks job.
    """

    def __init__(self, spark: SparkSession, config: Dict):
        """
        Initialize monitor.

        Args:
            spark: Spark session
            config: Monitoring configuration:
                - decision_table: Delta table with predictions (default: income_decisions)
                - baseline_table: Baseline data for comparison (default: training data)
                - feature_cols: List of features to monitor
                - lookback_days: Days to look back for current data (default: 1)
                - alert_webhook: Slack webhook URL
                - alert_email: Email address for alerts
                - enable_alerts: Enable alerting (default: True)
                - monitoring_metrics_table: Table to log monitoring results
        """
        self.spark = spark
        self.config = config
        self.decision_table = config.get("decision_table", "decision_agent.income_decisions")
        self.baseline_table = config.get("baseline_table", "decision_agent.training_baseline")
        self.feature_cols = config.get("feature_cols", [])
        self.lookback_days = config.get("lookback_days", 1)
        self.alert_webhook = config.get("alert_webhook")
        self.alert_email = config.get("alert_email")
        self.enable_alerts = config.get("enable_alerts", True)
        self.monitoring_metrics_table = config.get("monitoring_metrics_table", "decision_agent.monitoring_metrics")

        # Initialize detectors
        self.feature_drift_detector = FeatureDriftDetector(config)
        self.prediction_drift_detector = PredictionDriftDetector(config)

    def run_daily_monitoring(self) -> Dict:
        """
        Run daily monitoring workflow.

        Returns:
            Dict with monitoring results and alerts
        """
        logger.info("=" * 80)
        logger.info("Starting Daily Model Monitoring")
        logger.info("=" * 80)

        results = {
            "monitoring_timestamp": datetime.now().isoformat(),
            "checks": {},
            "alerts": [],
            "overall_status": "healthy"
        }

        try:
            # 1. Load current and baseline data
            current_df, baseline_df = self._load_data()

            if current_df is None or baseline_df is None:
                logger.error("Failed to load data. Aborting monitoring.")
                results["overall_status"] = "error"
                return results

            # 2. Feature drift detection
            logger.info("\n" + "=" * 80)
            logger.info("1. FEATURE DRIFT DETECTION")
            logger.info("=" * 80)

            feature_alert, feature_results = self.feature_drift_detector.detect_drift(
                current_df, baseline_df, self.feature_cols
            )

            results["checks"]["feature_drift"] = feature_results

            if feature_alert:
                results["alerts"].append({
                    "type": "feature_drift",
                    "severity": "high",
                    "message": f"{len(feature_results['features_above_threshold'])} features drifted",
                    "details": feature_results["features_above_threshold"]
                })
                results["overall_status"] = "warning"

            # 3. Prediction drift detection
            logger.info("\n" + "=" * 80)
            logger.info("2. PREDICTION DRIFT DETECTION")
            logger.info("=" * 80)

            prediction_alert, prediction_results = self.prediction_drift_detector.detect_drift(
                current_df, baseline_df, "prediction"
            )

            results["checks"]["prediction_drift"] = prediction_results

            if prediction_alert:
                results["alerts"].append({
                    "type": "prediction_drift",
                    "severity": "high",
                    "message": f"Prediction distribution shifted",
                    "details": prediction_results["violations"]
                })
                results["overall_status"] = "warning"

            # 4. Data quality checks
            logger.info("\n" + "=" * 80)
            logger.info("3. DATA QUALITY CHECKS")
            logger.info("=" * 80)

            data_quality_alert, data_quality_results = self._check_data_quality(current_df)
            results["checks"]["data_quality"] = data_quality_results

            if data_quality_alert:
                results["alerts"].append({
                    "type": "data_quality",
                    "severity": "high",
                    "message": "Data quality issues detected",
                    "details": data_quality_results.get("issues", [])
                })
                results["overall_status"] = "critical"

            # 5. Performance monitoring (if labels available)
            logger.info("\n" + "=" * 80)
            logger.info("4. PERFORMANCE MONITORING")
            logger.info("=" * 80)

            if "actual_income" in current_df.columns:
                performance_alert, performance_results = self._monitor_performance(current_df)
                results["checks"]["performance"] = performance_results

                if performance_alert:
                    results["alerts"].append({
                        "type": "performance_degradation",
                        "severity": "critical",
                        "message": "Model performance degraded",
                        "details": performance_results
                    })
                    results["overall_status"] = "critical"
            else:
                logger.info("Labels not available yet (expected lag). Skipping performance monitoring.")

            # 6. Log monitoring results
            self._log_monitoring_results(results)

            # 7. Send alerts
            if results["alerts"] and self.enable_alerts:
                self._send_alerts(results)

            logger.info("\n" + "=" * 80)
            logger.info(f"Monitoring Complete. Status: {results['overall_status'].upper()}")
            logger.info(f"Alerts: {len(results['alerts'])}")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Monitoring failed with error: {e}")
            results["overall_status"] = "error"
            results["error"] = str(e)

        return results

    def _load_data(self) -> tuple:
        """Load current and baseline data."""
        logger.info("Loading current and baseline data...")

        try:
            # Load current data (last N days)
            current_start_date = (datetime.now() - timedelta(days=self.lookback_days)).strftime("%Y-%m-%d")

            current_df = self.spark.sql(f"""
                SELECT * FROM {self.decision_table}
                WHERE prediction_date >= '{current_start_date}'
            """)

            current_count = current_df.count()
            logger.info(f"Current data: {current_count} predictions")

            if current_count == 0:
                logger.warning("No current data found!")
                return None, None

            # Load baseline data
            baseline_df = self.spark.table(self.baseline_table)
            baseline_count = baseline_df.count()

            logger.info(f"Baseline data: {baseline_count} samples")

            return current_df, baseline_df

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return None, None

    def _check_data_quality(self, df: DataFrame) -> tuple:
        """Check data quality issues."""
        issues = []

        # Check for null rates
        total_count = df.count()

        for col in self.feature_cols:
            if col not in df.columns:
                continue

            null_count = df.filter(F.col(col).isNull()).count()
            null_rate = null_count / total_count if total_count > 0 else 0

            if null_rate > 0.10:  # More than 10% nulls
                issues.append({
                    "feature": col,
                    "issue": "high_null_rate",
                    "null_rate": float(null_rate)
                })

        alert = len(issues) > 0

        results = {
            "issues": issues,
            "total_count": total_count
        }

        if alert:
            logger.warning(f"Data quality issues: {len(issues)} features with high null rates")

        return alert, results

    def _monitor_performance(self, df: DataFrame) -> tuple:
        """Monitor model performance if labels available."""
        # Compute MAE
        mae = df.agg(
            F.mean(F.abs(F.col("actual_income") - F.col("prediction")))
        ).collect()[0][0]

        # Compare to baseline MAE (hardcoded for now, should come from config)
        baseline_mae = self.config.get("baseline_mae", 10000)

        degradation_pct = (mae - baseline_mae) / baseline_mae if baseline_mae > 0 else 0

        alert = degradation_pct > 0.15  # Alert if MAE increased by > 15%

        results = {
            "current_mae": float(mae),
            "baseline_mae": float(baseline_mae),
            "degradation_pct": float(degradation_pct),
            "alert_threshold": 0.15
        }

        if alert:
            logger.warning(f"Performance degraded: MAE increased by {degradation_pct:.1%}")
        else:
            logger.info(f"Performance stable: MAE={mae:.0f}")

        return alert, results

    def _log_monitoring_results(self, results: Dict):
        """Log monitoring results to Delta table."""
        logger.info("Logging monitoring results...")

        try:
            # Convert results to DataFrame
            results_json = json.dumps(results)

            monitoring_record = self.spark.createDataFrame([{
                "monitoring_timestamp": results["monitoring_timestamp"],
                "overall_status": results["overall_status"],
                "num_alerts": len(results["alerts"]),
                "results_json": results_json
            }])

            # Write to monitoring metrics table
            monitoring_record.write \
                .format("delta") \
                .mode("append") \
                .saveAsTable(self.monitoring_metrics_table)

            logger.info("Monitoring results logged successfully")

        except Exception as e:
            logger.error(f"Failed to log monitoring results: {e}")

    def _send_alerts(self, results: Dict):
        """Send alerts via Slack/PagerDuty/Email."""
        logger.info(f"Sending {len(results['alerts'])} alerts...")

        # Format alert message
        alert_message = self._format_alert_message(results)

        # Send to Slack
        if self.alert_webhook:
            self._send_slack_alert(alert_message)

        # Send email (placeholder - implement based on your email provider)
        if self.alert_email:
            logger.info(f"Would send email alert to: {self.alert_email}")

    def _format_alert_message(self, results: Dict) -> str:
        """Format alert message for Slack."""
        status_emoji = {
            "healthy": "✅",
            "warning": "⚠️",
            "critical": "🔴",
            "error": "💥"
        }

        emoji = status_emoji.get(results["overall_status"], "ℹ️")

        lines = [
            f"{emoji} *Model Monitoring Alert*",
            f"Status: *{results['overall_status'].upper()}*",
            f"Timestamp: {results['monitoring_timestamp']}",
            "",
            f"*Alerts ({len(results['alerts'])}):*"
        ]

        for alert in results["alerts"]:
            lines.append(f"  • {alert['type']}: {alert['message']}")

        return "\n".join(lines)

    def _send_slack_alert(self, message: str):
        """Send alert to Slack."""
        try:
            import requests

            payload = {"text": message}
            response = requests.post(self.alert_webhook, json=payload)

            if response.status_code == 200:
                logger.info("Slack alert sent successfully")
            else:
                logger.error(f"Failed to send Slack alert: {response.status_code}")

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
