"""
Spark session utilities.
"""
import logging

logger = logging.getLogger(__name__)


def get_spark_session(app_name: str = "DecisionAgent"):
    """
    Get or create Spark session.

    Args:
        app_name: Application name

    Returns:
        Spark session
    """
    try:
        from pyspark.sql import SparkSession

        spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
            .getOrCreate()

        logger.info(f"Spark session created: {app_name}")
        return spark

    except ImportError:
        logger.error("PySpark not installed. Install with: pip install pyspark")
        raise
    except Exception as e:
        logger.error(f"Failed to create Spark session: {e}")
        raise


def stop_spark_session(spark):
    """
    Stop Spark session.

    Args:
        spark: Spark session to stop
    """
    if spark:
        spark.stop()
        logger.info("Spark session stopped")
