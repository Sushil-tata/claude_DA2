# Databricks notebook source
# MAGIC %md
# MAGIC # Real-Time Feature Streaming Pipeline
# MAGIC
# MAGIC This notebook runs the real-time streaming feature pipeline.
# MAGIC
# MAGIC **Features:**
# MAGIC - Consumes events from Kafka/Kinesis
# MAGIC - Computes windowed aggregations (1min, 5min, 15min)
# MAGIC - Computes stateful customer features
# MAGIC - Writes to Delta Lake (streaming append)
# MAGIC - Populates online feature store (Redis)
# MAGIC
# MAGIC **Performance Targets:**
# MAGIC - Processing latency: < 10 seconds end-to-end
# MAGIC - Throughput: > 10,000 events/sec
# MAGIC - Checkpoint interval: 30 seconds
# MAGIC
# MAGIC **Usage:**
# MAGIC 1. Configure parameters in widgets
# MAGIC 2. Run all cells
# MAGIC 3. Monitor streaming query progress
# MAGIC 4. Stop query when done (or let it run continuously)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# Install dependencies (if needed)
%pip install pyyaml redis

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

import yaml
import logging
from pathlib import Path
from pyspark.sql import SparkSession
from decision_agent.streaming.feature_stream import StreamingFeaturePipeline, create_streaming_pipeline
from decision_agent.streaming.online_store import OnlineFeatureStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Create widgets for runtime configuration
dbutils.widgets.text("config_path", "conf/streaming/streaming_config.yaml", "Config Path")
dbutils.widgets.dropdown("input_source", "kafka", ["kafka", "kinesis", "delta"], "Input Source")
dbutils.widgets.text("checkpoint_location", "dbfs:/decision_agent/streaming/checkpoints/feature_stream", "Checkpoint Location")
dbutils.widgets.text("output_table", "decision_agent.customer_features_streaming", "Output Table")

# COMMAND ----------

# Get widget values
config_path = dbutils.widgets.get("config_path")
input_source_override = dbutils.widgets.get("input_source")
checkpoint_location_override = dbutils.widgets.get("checkpoint_location")
output_table_override = dbutils.widgets.get("output_table")

logger.info(f"Configuration:")
logger.info(f"  Config path: {config_path}")
logger.info(f"  Input source: {input_source_override}")
logger.info(f"  Checkpoint: {checkpoint_location_override}")
logger.info(f"  Output table: {output_table_override}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Configuration

# COMMAND ----------

# Load configuration from YAML
try:
    # Load from DBFS or local path
    if config_path.startswith("dbfs:"):
        config_local_path = config_path.replace("dbfs:", "/dbfs")
    else:
        config_local_path = config_path

    with open(config_local_path) as f:
        config = yaml.safe_load(f)

    # Override with widget values
    config["input_source"] = input_source_override
    config["checkpoint_location"] = checkpoint_location_override
    config["output_table"] = output_table_override

    logger.info("✓ Configuration loaded successfully")
    logger.info(f"Config: {config}")

except Exception as e:
    logger.error(f"✗ Failed to load configuration: {e}")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize Spark Session

# COMMAND ----------

# Get or create Spark session with streaming configuration
spark = SparkSession.builder \
    .appName("DecisionAgent-StreamingFeatures") \
    .config("spark.sql.streaming.checkpointLocation", checkpoint_location_override) \
    .config("spark.sql.shuffle.partitions", config.get("performance", {}).get("shuffle_partitions", 200)) \
    .config("spark.databricks.delta.preview.enabled", "true") \
    .getOrCreate()

logger.info(f"✓ Spark session initialized: {spark.version}")
logger.info(f"  Spark UI: {spark.sparkContext.uiWebUrl}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Streaming Pipeline

# COMMAND ----------

# Create streaming pipeline
try:
    pipeline = create_streaming_pipeline(spark, config)
    logger.info("✓ Streaming pipeline created")

except Exception as e:
    logger.error(f"✗ Failed to create pipeline: {e}")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Start Streaming Query

# COMMAND ----------

# Start the streaming pipeline
try:
    logger.info("=" * 80)
    logger.info("STARTING STREAMING PIPELINE")
    logger.info("=" * 80)

    query = pipeline.start()

    logger.info("=" * 80)
    logger.info("STREAMING PIPELINE STARTED")
    logger.info("=" * 80)
    logger.info(f"Query ID: {query.id}")
    logger.info(f"Query Name: {query.name}")
    logger.info(f"Status: {query.status}")
    logger.info(f"Recent Progress: {query.recentProgress}")

except Exception as e:
    logger.error(f"✗ Failed to start pipeline: {e}")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Monitor Streaming Query

# COMMAND ----------

# Display streaming query metrics
def display_query_metrics(query):
    """Display streaming query metrics in real-time"""
    status = query.status
    recent_progress = query.recentProgress

    print("=" * 80)
    print("STREAMING QUERY METRICS")
    print("=" * 80)
    print(f"Query ID: {query.id}")
    print(f"Status: {status}")
    print()

    if recent_progress:
        latest_progress = recent_progress[-1]
        print("Latest Progress:")
        print(f"  Batch ID: {latest_progress.get('batchId', 'N/A')}")
        print(f"  Num Input Rows: {latest_progress.get('numInputRows', 0):,}")
        print(f"  Input Rows/Sec: {latest_progress.get('inputRowsPerSecond', 0):,.1f}")
        print(f"  Process Rows/Sec: {latest_progress.get('processedRowsPerSecond', 0):,.1f}")
        print(f"  Batch Duration: {latest_progress.get('durationMs', {}).get('triggerExecution', 0):,} ms")
        print()

        # Source statistics
        sources = latest_progress.get('sources', [])
        if sources:
            print("Source Statistics:")
            for source in sources:
                print(f"  Description: {source.get('description', 'N/A')}")
                print(f"  Num Records: {source.get('numInputRows', 0):,}")
                print()

        # Sink statistics
        sink = latest_progress.get('sink', {})
        if sink:
            print("Sink Statistics:")
            print(f"  Description: {sink.get('description', 'N/A')}")
            print(f"  Num Output Rows: {sink.get('numOutputRows', 0):,}")
            print()

    print("=" * 80)

# Display initial metrics
display_query_metrics(query)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optional: Initialize Online Feature Store

# COMMAND ----------

# Initialize online feature store if enabled
online_store = None

if config.get("online_store", {}).get("enabled", False):
    try:
        redis_config = config["online_store"]["redis_config"]

        # Resolve secrets
        if "{{ secrets" in redis_config.get("host", ""):
            # Extract secret key
            secret_key = redis_config["host"].split("secrets.")[1].rstrip(" }}")
            redis_config["host"] = dbutils.secrets.get(scope="decision_agent", key=secret_key)

        if "{{ secrets" in redis_config.get("password", ""):
            secret_key = redis_config["password"].split("secrets.")[1].rstrip(" }}")
            redis_config["password"] = dbutils.secrets.get(scope="decision_agent", key=secret_key)

        delta_config = {
            "feature_table": config["output_table"]
        }

        online_store = OnlineFeatureStore(redis_config, delta_config, spark)
        logger.info("✓ Online feature store initialized")

    except Exception as e:
        logger.warning(f"Online feature store initialization failed: {e}")
        logger.warning("Continuing without online store...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optional: Cache Warming

# COMMAND ----------

# Warm cache with top customers (if online store enabled)
if online_store and config.get("online_store", {}).get("cache_warming", {}).get("enabled", False):
    try:
        top_n = config["online_store"]["cache_warming"].get("top_n_customers", 10000)

        logger.info(f"Warming cache with top {top_n} customers...")
        stats = online_store.warm_cache(top_n=top_n)

        logger.info(f"Cache warming complete:")
        logger.info(f"  Cached: {stats['cached']:,}")
        logger.info(f"  Failed: {stats['failed']:,}")

    except Exception as e:
        logger.warning(f"Cache warming failed: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Continuous Monitoring (Optional)

# COMMAND ----------

# Monitor streaming query continuously (run this cell to see real-time updates)
import time

try:
    while query.isActive:
        display_query_metrics(query)

        # Sleep before next update
        time.sleep(30)  # Update every 30 seconds

except KeyboardInterrupt:
    logger.info("Monitoring stopped by user")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stop Streaming Query (When Done)

# COMMAND ----------

# Uncomment to stop the streaming query
# query.stop()
# logger.info("Streaming query stopped")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Health Checks

# COMMAND ----------

# Check streaming query health
def check_query_health(query):
    """Check streaming query health and report issues"""
    if not query.isActive:
        logger.error("✗ Query is not active!")
        return False

    status = query.status
    recent_progress = query.recentProgress

    if not recent_progress:
        logger.warning("⚠ No recent progress data available")
        return True

    latest_progress = recent_progress[-1]

    # Check processing lag
    input_rate = latest_progress.get('inputRowsPerSecond', 0)
    process_rate = latest_progress.get('processedRowsPerSecond', 0)

    if input_rate > 0 and process_rate < input_rate * 0.8:
        logger.warning(f"⚠ Processing lag detected: {input_rate:.1f} input/sec vs {process_rate:.1f} processed/sec")

    # Check batch duration
    batch_duration_ms = latest_progress.get('durationMs', {}).get('triggerExecution', 0)
    trigger_interval_ms = 30000  # 30 seconds

    if batch_duration_ms > trigger_interval_ms:
        logger.warning(f"⚠ Batch duration ({batch_duration_ms}ms) exceeds trigger interval ({trigger_interval_ms}ms)")

    logger.info("✓ Query health check passed")
    return True

# Run health check
check_query_health(query)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cache Statistics (If Online Store Enabled)

# COMMAND ----------

# Display cache statistics
if online_store:
    try:
        cache_stats = online_store.get_cache_stats()

        print("=" * 80)
        print("CACHE STATISTICS")
        print("=" * 80)
        print(f"Total Keys: {cache_stats.get('total_keys', 0):,}")
        print(f"Keyspace Hits: {cache_stats.get('keyspace_hits', 0):,}")
        print(f"Keyspace Misses: {cache_stats.get('keyspace_misses', 0):,}")
        print(f"Hit Rate: {cache_stats.get('hit_rate', 0):.2%}")
        print("=" * 80)

    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cleanup (When Completely Done)

# COMMAND ----------

# Uncomment to clean up resources when completely done
# if online_store:
#     online_store.clear_cache()
#     logger.info("Cache cleared")
#
# spark.stop()
# logger.info("Spark session stopped")
