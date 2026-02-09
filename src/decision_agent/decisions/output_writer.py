"""
Decision output writer to Delta Lake.
"""
import logging
from datetime import datetime
from typing import Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)


def write_decisions(
    spark,
    predictions_df,
    config: Dict[str, Any],
    model_version: str,
    run_id: str,
    as_of_date: str = None,
    prediction_col: str = "prediction"
):
    """
    Write decision output to Delta Lake table (Spark-native).

    Args:
        spark: Spark session
        predictions_df: Spark or pandas DataFrame with customer_id and predictions
        config: Use case configuration
        model_version: Model version identifier
        run_id: MLflow run ID or execution run ID
        as_of_date: As-of date for decisions (default: today)
        prediction_col: Name of prediction column

    Returns:
        Table name where decisions were written

    Decision table schema:
        - customer_id: Customer identifier
        - predicted_value: Model prediction
        - run_id: Execution run identifier
        - model_version: Model version
        - as_of_dt: Decision date
        - use_case_id: Use case identifier
        - created_timestamp: Record creation timestamp
    """
    if as_of_date is None:
        as_of_date = datetime.now().strftime("%Y-%m-%d")

    table_name = config["output"]["table_name"]
    use_case_id = config["use_case_id"]

    logger.info(f"Writing decisions to {table_name}...")

    # Check if Spark DataFrame
    try:
        from pyspark.sql import DataFrame as SparkDataFrame
        is_spark = isinstance(predictions_df, SparkDataFrame)
    except ImportError:
        is_spark = False

    if not is_spark:
        # Convert pandas to Spark if needed
        if spark is None:
            logger.warning("No Spark session available. Saving decisions locally to CSV.")
            import pandas as pd

            decision_data = predictions_df[["customer_id", prediction_col]].copy()
            decision_data = decision_data.rename(columns={prediction_col: "predicted_value"})
            decision_data["run_id"] = run_id
            decision_data["model_version"] = model_version
            decision_data["as_of_dt"] = as_of_date
            decision_data["use_case_id"] = use_case_id
            decision_data["created_timestamp"] = datetime.now().isoformat()

            output_path = f"decisions_{use_case_id}_{run_id}.csv"
            decision_data.to_csv(output_path, index=False)
            logger.info(f"Decisions saved to {output_path}")
            return output_path

        predictions_df = spark.createDataFrame(predictions_df)

    # Prepare decision records (Spark operations only)
    from pyspark.sql import functions as F

    decisions_spark_df = predictions_df.select(
        F.col("customer_id"),
        F.col(prediction_col).alias("predicted_value")
    ).withColumn(
        "run_id", F.lit(run_id)
    ).withColumn(
        "model_version", F.lit(model_version)
    ).withColumn(
        "as_of_dt", F.lit(as_of_date)
    ).withColumn(
        "use_case_id", F.lit(use_case_id)
    ).withColumn(
        "created_timestamp", F.lit(datetime.now().isoformat())
    )

    num_decisions = decisions_spark_df.count()
    logger.info(f"Prepared {num_decisions:,} decision records")

    # Show sample (limited to 5 rows, safe to collect)
    logger.info("Sample decisions:")
    decisions_spark_df.show(5, truncate=False)

    try:
        # Check if table exists
        try:
            spark.sql(f"DESCRIBE TABLE {table_name}")
            table_exists = True
        except Exception:
            table_exists = False

        if table_exists:
            # Append to existing table
            logger.info(f"Appending to existing table: {table_name}")
            decisions_spark_df.write \
                .format("delta") \
                .mode("append") \
                .saveAsTable(table_name)
        else:
            # Create new table
            logger.info(f"Creating new table: {table_name}")

            # Extract catalog and schema from table name
            parts = table_name.split(".")
            if len(parts) == 2:
                catalog_schema = parts[0]
                table = parts[1]

                # Create schema if it doesn't exist
                try:
                    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_schema}")
                    logger.info(f"Created schema: {catalog_schema}")
                except Exception as e:
                    logger.warning(f"Could not create schema: {e}")

            decisions_spark_df.write \
                .format("delta") \
                .mode("overwrite") \
                .saveAsTable(table_name)

        logger.info(f"✓ Decisions written successfully to {table_name}")

        # Show table info
        count = spark.sql(f"SELECT COUNT(*) as count FROM {table_name}").collect()[0]["count"]
        logger.info(f"Total records in {table_name}: {count}")

    except Exception as e:
        logger.error(f"Failed to write to Delta Lake: {e}")
        logger.info("Falling back to local file storage...")

        # Fallback: save as Parquet locally
        output_path = f"decisions_{use_case_id}_{run_id}.parquet"
        decisions_spark_df.write.mode("overwrite").parquet(output_path)
        logger.info(f"Decisions saved to {output_path}")
        return output_path

    return table_name


def read_decisions(
    spark,
    table_name: str,
    use_case_id: str = None,
    as_of_date: str = None,
    limit: int = None,
    as_pandas: bool = False,
    max_rows_pandas: int = 10000
):
    """
    Read decisions from Delta Lake table.

    Args:
        spark: Spark session
        table_name: Decision table name
        use_case_id: Filter by use case (optional)
        as_of_date: Filter by decision date (optional)
        limit: Maximum number of records to return
        as_pandas: If True, convert to pandas (with safety checks)
        max_rows_pandas: Maximum rows allowed for pandas conversion

    Returns:
        Spark DataFrame (default) or pandas DataFrame (if as_pandas=True)
    """
    logger.info(f"Reading decisions from {table_name}...")

    query = f"SELECT * FROM {table_name}"
    filters = []

    if use_case_id:
        filters.append(f"use_case_id = '{use_case_id}'")

    if as_of_date:
        filters.append(f"as_of_dt = '{as_of_date}'")

    if filters:
        query += " WHERE " + " AND ".join(filters)

    if limit:
        query += f" LIMIT {limit}"

    df = spark.sql(query)
    row_count = df.count()

    logger.info(f"Read {row_count:,} decision records")

    if as_pandas:
        # Use safe_to_pandas with guardrails
        from decision_agent.utils.spark_guards import safe_to_pandas

        if row_count > max_rows_pandas:
            logger.warning(
                f"DataFrame has {row_count:,} rows, limiting to {max_rows_pandas:,} for pandas conversion"
            )
            df = df.limit(max_rows_pandas)

        return safe_to_pandas(df, max_rows=max_rows_pandas, sample_for_estimate=False)

    return df
