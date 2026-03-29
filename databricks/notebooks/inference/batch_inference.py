# Databricks notebook source
# MAGIC %md
# MAGIC # Production Batch Inference
# MAGIC
# MAGIC Daily batch scoring job for income estimation.
# MAGIC
# MAGIC **Workflow:**
# MAGIC 1. Load features from feature store
# MAGIC 2. Run inference pipeline (Champion + optional Challenger)
# MAGIC 3. Apply business rules
# MAGIC 4. Write decisions to output table
# MAGIC 5. Send summary to Slack
# MAGIC
# MAGIC **Schedule:** Daily at 2 AM UTC
# MAGIC
# MAGIC **Cluster:** production_inference_cluster

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# Install dependencies (if needed)
# %pip install mlflow pyyaml

# COMMAND ----------

# Imports
import sys
import logging
from datetime import datetime
import yaml
import json

from pyspark.sql import SparkSession
from decision_agent.inference.batch_scoring import BatchScorer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Load configuration
config_path = "/Workspace/conf/inference/production_config.yaml"

with open(config_path) as f:
    config = yaml.safe_load(f)

logger.info(f"Loaded config from: {config_path}")
logger.info(f"Use case: {config['use_case_id']}")
logger.info(f"Model version: {config['model_version']}")
logger.info(f"A/B testing enabled: {config.get('enable_ab_testing', False)}")

# COMMAND ----------

# Widget parameters (for manual runs)
dbutils.widgets.text("score_date", "", "Score Date (YYYY-MM-DD)")
dbutils.widgets.dropdown("enable_ab_testing", "false", ["false", "true"], "Enable A/B Testing")
dbutils.widgets.text("lookback_days", "1", "Lookback Days")

# Get widget values
score_date = dbutils.widgets.get("score_date") or None
enable_ab = dbutils.widgets.get("enable_ab_testing") == "true"
lookback_days = int(dbutils.widgets.get("lookback_days") or "1")

# Override config with widget values
if enable_ab:
    config["enable_ab_testing"] = True

logger.info(f"Parameters:")
logger.info(f"  score_date: {score_date or 'today'}")
logger.info(f"  enable_ab_testing: {enable_ab}")
logger.info(f"  lookback_days: {lookback_days}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Batch Inference

# COMMAND ----------

# Initialize batch scorer
scorer = BatchScorer(spark, config)

# Run batch scoring
try:
    results = scorer.score_batch(
        score_date=score_date,
        lookback_days=lookback_days
    )

    logger.info("=" * 80)
    logger.info("Batch Inference Results:")
    logger.info(json.dumps(results, indent=2))
    logger.info("=" * 80)

    # Store results for downstream tasks
    dbutils.jobs.taskValues.set(key="batch_results", value=results)

    # Exit with success
    dbutils.notebook.exit(json.dumps({"status": "success", "results": results}))

except Exception as e:
    logger.error(f"Batch inference failed: {e}")
    import traceback
    traceback.print_exc()

    # Exit with failure
    dbutils.notebook.exit(json.dumps({
        "status": "failure",
        "error": str(e),
        "traceback": traceback.format_exc()
    }))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary Metrics

# COMMAND ----------

# Query output table for summary
output_table = results.get("output_table", "decision_agent.production_predictions")

summary_sql = f"""
SELECT
    COUNT(*) as total_predictions,
    COUNT(DISTINCT customer_id) as unique_customers,
    selected_model,
    decision,
    COUNT(*) as count
FROM {output_table}
WHERE DATE(prediction_timestamp) = CURRENT_DATE
GROUP BY selected_model, decision
ORDER BY selected_model, decision
"""

summary_df = spark.sql(summary_sql)
display(summary_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Slack Notification

# COMMAND ----------

def send_slack_notification(results: dict, webhook_url: str):
    """Send summary to Slack."""
    import requests

    status_emoji = "✅" if results["status"] == "success" else "❌"

    message = f"""
{status_emoji} *Production Batch Inference Complete*

*Use Case:* {config['use_case_id']}
*Date:* {results.get('score_date', 'today')}
*Status:* {results['status']}

*Summary:*
• Customers loaded: {results.get('num_customers_loaded', 0):,}
• Customers scored: {results.get('num_customers_filtered', 0):,}
• Decisions written: {results.get('num_decisions_written', 0):,}

*Output:* `{results.get('output_table', 'N/A')}`
    """.strip()

    if results.get("decision_summary"):
        message += "\n\n*Decisions:*\n"
        for decision, count in results["decision_summary"].items():
            message += f"• {decision}: {count:,}\n"

    if results.get("routing_summary"):
        message += "\n*Routing:*\n"
        for model, count in results["routing_summary"].items():
            message += f"• {model}: {count:,}\n"

    payload = {"text": message}

    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        if response.status_code == 200:
            logger.info("✓ Slack notification sent")
        else:
            logger.warning(f"Slack notification failed: {response.status_code}")
    except Exception as e:
        logger.warning(f"Failed to send Slack notification: {e}")

# Send notification (if webhook configured)
slack_webhook = config.get("slack_webhook")
if slack_webhook and results.get("status") == "success":
    send_slack_notification(results, slack_webhook)
else:
    logger.info("Slack webhook not configured or job failed. Skipping notification.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cleanup

# COMMAND ----------

# Clear cache
spark.catalog.clearCache()

logger.info("✓ Batch inference notebook complete")
