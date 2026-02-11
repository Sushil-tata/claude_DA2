"""
Streaming Feature Pipeline

Real-time feature computation using Spark Structured Streaming.

Features:
- Tumbling windows (1min, 5min, 15min aggregations)
- Sliding windows (rolling statistics)
- Stateful processing (customer-level state tracking)
- Delta Lake integration (append streaming writes)
- Exactly-once processing (checkpointing)

Data Flow:
  Kafka/Kinesis Events
    ↓
  Spark Structured Streaming
    ├─ Parse events
    ├─ Compute window aggregations
    ├─ Update customer state
    ├─ Compute derived features
    └─ Write to Delta Lake
    ↓
  Delta Table: customer_features_streaming

Performance Target:
- Processing latency: < 10 seconds end-to-end
- Throughput: > 10,000 events/sec
- Checkpoint interval: 30 seconds
"""

import logging
from typing import Dict, Any, Optional
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType

logger = logging.getLogger(__name__)


class StreamingFeaturePipeline:
    """
    Real-time feature computation pipeline using Spark Structured Streaming.
    """

    def __init__(
        self,
        spark: SparkSession,
        config: Dict[str, Any]
    ):
        """
        Initialize streaming pipeline.

        Args:
            spark: Spark session (must have streaming enabled)
            config: Streaming configuration:
                - input_source: kafka, kinesis, or delta
                - input_config: Source-specific configuration
                - checkpoint_location: DBFS path for checkpoints
                - output_table: Delta table for features
                - window_durations: List of window durations (e.g., ["1 minute", "5 minutes"])
        """
        self.spark = spark
        self.config = config
        self.input_source = config["input_source"]
        self.checkpoint_location = config["checkpoint_location"]
        self.output_table = config["output_table"]
        self.window_durations = config.get("window_durations", ["1 minute", "5 minutes", "15 minutes"])

    def start(self):
        """
        Start streaming pipeline.
        """
        logger.info("=" * 80)
        logger.info("Starting Streaming Feature Pipeline")
        logger.info("=" * 80)
        logger.info(f"Input source: {self.input_source}")
        logger.info(f"Output table: {self.output_table}")
        logger.info(f"Checkpoint: {self.checkpoint_location}")

        # Step 1: Read streaming source
        raw_stream = self._read_stream()

        # Step 2: Parse events
        parsed_stream = self._parse_events(raw_stream)

        # Step 3: Compute windowed aggregations
        windowed_features = self._compute_windowed_features(parsed_stream)

        # Step 4: Compute stateful features
        stateful_features = self._compute_stateful_features(windowed_features)

        # Step 5: Write to Delta Lake
        query = self._write_to_delta(stateful_features)

        logger.info("✓ Streaming pipeline started")
        logger.info(f"Query ID: {query.id}")
        logger.info(f"Status: {query.status}")

        return query

    def _read_stream(self) -> DataFrame:
        """
        Read from streaming source (Kafka, Kinesis, or Delta).
        """
        if self.input_source == "kafka":
            return self._read_kafka()
        elif self.input_source == "kinesis":
            return self._read_kinesis()
        elif self.input_source == "delta":
            return self._read_delta()
        else:
            raise ValueError(f"Unknown input source: {self.input_source}")

    def _read_kafka(self) -> DataFrame:
        """Read from Kafka."""
        kafka_config = self.config["input_config"]

        stream = self.spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", kafka_config["bootstrap_servers"]) \
            .option("subscribe", kafka_config["topic"]) \
            .option("startingOffsets", kafka_config.get("starting_offsets", "latest")) \
            .option("maxOffsetsPerTrigger", kafka_config.get("max_offsets_per_trigger", 10000)) \
            .load()

        logger.info(f"✓ Reading from Kafka: {kafka_config['topic']}")
        return stream

    def _read_kinesis(self) -> DataFrame:
        """Read from Kinesis."""
        kinesis_config = self.config["input_config"]

        stream = self.spark.readStream \
            .format("kinesis") \
            .option("streamName", kinesis_config["stream_name"]) \
            .option("region", kinesis_config["region"]) \
            .option("initialPosition", kinesis_config.get("initial_position", "latest")) \
            .load()

        logger.info(f"✓ Reading from Kinesis: {kinesis_config['stream_name']}")
        return stream

    def _read_delta(self) -> DataFrame:
        """Read from Delta table (for testing)."""
        delta_config = self.config["input_config"]

        stream = self.spark.readStream \
            .format("delta") \
            .table(delta_config["table_name"])

        logger.info(f"✓ Reading from Delta: {delta_config['table_name']}")
        return stream

    def _parse_events(self, raw_stream: DataFrame) -> DataFrame:
        """
        Parse raw events to structured format.

        Expected event schema:
        {
            "customer_id": "CUST12345",
            "event_type": "transaction",
            "timestamp": "2024-12-05T10:30:00Z",
            "amount": 150.50,
            "category": "groceries",
            "balance": 5000.00
        }
        """
        # Define event schema
        event_schema = StructType([
            StructField("customer_id", StringType(), False),
            StructField("event_type", StringType(), False),
            StructField("timestamp", TimestampType(), False),
            StructField("amount", DoubleType(), True),
            StructField("category", StringType(), True),
            StructField("balance", DoubleType(), True)
        ])

        # Parse JSON from Kafka value column
        parsed = raw_stream.select(
            F.from_json(F.col("value").cast("string"), event_schema).alias("data")
        ).select("data.*")

        # Add watermark for late data (10 minutes)
        parsed_with_watermark = parsed.withWatermark("timestamp", "10 minutes")

        logger.info("✓ Events parsed and watermarked")
        return parsed_with_watermark

    def _compute_windowed_features(self, stream: DataFrame) -> DataFrame:
        """
        Compute tumbling window aggregations.

        For each window duration, compute:
        - Transaction count
        - Transaction sum
        - Average transaction amount
        - Deposit count (positive amounts)
        - Withdrawal count (negative amounts)
        """
        windowed_dfs = []

        for window_duration in self.window_durations:
            logger.info(f"  Computing {window_duration} window features...")

            windowed = stream.groupBy(
                "customer_id",
                F.window("timestamp", window_duration)
            ).agg(
                F.count("*").alias(f"txn_count_{window_duration}"),
                F.sum("amount").alias(f"txn_sum_{window_duration}"),
                F.avg("amount").alias(f"txn_avg_{window_duration}"),
                F.sum(F.when(F.col("amount") > 0, F.col("amount")).otherwise(0)).alias(f"deposit_sum_{window_duration}"),
                F.sum(F.when(F.col("amount") < 0, F.abs(F.col("amount"))).otherwise(0)).alias(f"withdrawal_sum_{window_duration}"),
                F.count(F.when(F.col("amount") > 0, 1)).alias(f"deposit_count_{window_duration}"),
                F.count(F.when(F.col("amount") < 0, 1)).alias(f"withdrawal_count_{window_duration}"),
                F.last("balance").alias(f"latest_balance_{window_duration}")
            ).select(
                "customer_id",
                F.col("window.end").alias("window_end"),
                *[c for c in windowed.columns if c not in ["customer_id", "window"]]
            )

            windowed_dfs.append(windowed)

        # Join all window features
        result = windowed_dfs[0]
        for df in windowed_dfs[1:]:
            result = result.join(df, on=["customer_id", "window_end"], how="outer")

        logger.info("✓ Windowed features computed")
        return result

    def _compute_stateful_features(self, stream: DataFrame) -> DataFrame:
        """
        Compute stateful features using customer-level state.

        Stateful features:
        - Days since first transaction
        - Days since last transaction
        - Lifetime transaction count
        - Lifetime deposit sum
        """
        # Group by customer and compute state
        stateful = stream.groupBy("customer_id").agg(
            F.max("window_end").alias("latest_event_time"),
            F.sum("txn_count_1 minute").alias("lifetime_txn_count"),
            F.sum("deposit_sum_1 minute").alias("lifetime_deposit_sum")
        )

        # Join back to stream
        result = stream.join(stateful, on="customer_id", how="left")

        logger.info("✓ Stateful features computed")
        return result

    def _write_to_delta(self, stream: DataFrame):
        """
        Write streaming features to Delta Lake.

        Write mode: append
        Trigger: 30 second intervals
        Checkpointing: enabled
        """
        query = stream.writeStream \
            .format("delta") \
            .outputMode("append") \
            .option("checkpointLocation", self.checkpoint_location) \
            .option("mergeSchema", "true") \
            .trigger(processingTime="30 seconds") \
            .toTable(self.output_table)

        logger.info(f"✓ Writing to Delta table: {self.output_table}")
        return query


def create_streaming_pipeline(
    spark: SparkSession,
    config: Dict[str, Any]
) -> StreamingFeaturePipeline:
    """
    Factory function to create streaming pipeline.

    Args:
        spark: Spark session
        config: Streaming configuration

    Returns:
        Configured streaming pipeline
    """
    return StreamingFeaturePipeline(spark, config)


# Example usage
if __name__ == "__main__":
    import yaml

    # Load config
    with open("conf/streaming/streaming_config.yaml") as f:
        config = yaml.safe_load(f)

    # Initialize Spark with streaming
    spark = SparkSession.builder \
        .appName("StreamingFeaturePipeline") \
        .config("spark.sql.streaming.checkpointLocation", config["checkpoint_location"]) \
        .getOrCreate()

    # Create and start pipeline
    pipeline = create_streaming_pipeline(spark, config)
    query = pipeline.start()

    # Wait for termination (or Ctrl+C)
    query.awaitTermination()
