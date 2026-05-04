"""
Guardrails for safe Spark DataFrame operations.

Prevents accidental OOM errors from toPandas() on large DataFrames.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SparkGuardError(Exception):
    """Raised when Spark operation violates safety guardrails"""
    pass


def safe_to_pandas(df, max_rows: int = 10000, sample_for_estimate: bool = True):
    """
    Safely convert Spark DataFrame to pandas with size guards.

    Args:
        df: Spark DataFrame
        max_rows: Maximum allowed rows for conversion
        sample_for_estimate: Whether to sample for size estimation

    Returns:
        Pandas DataFrame

    Raises:
        SparkGuardError: If DataFrame exceeds max_rows

    Example:
        >>> pdf = safe_to_pandas(spark_df, max_rows=5000)
    """
    from pyspark.sql import DataFrame as SparkDataFrame

    if not isinstance(df, SparkDataFrame):
        raise ValueError("Input must be a Spark DataFrame")

    # Get row count (use cached count if available, otherwise compute)
    try:
        if sample_for_estimate:
            # Estimate by sampling (faster for large DFs)
            sample_count = df.sample(False, 0.01).count()
            estimated_count = sample_count * 100

            if estimated_count > max_rows * 1.5:  # 50% buffer
                raise SparkGuardError(
                    f"Estimated DataFrame size ({estimated_count:,} rows) exceeds "
                    f"max_rows ({max_rows:,}). Use .limit() or Spark-native operations."
                )

        # Actual count check
        actual_count = df.count()

        if actual_count > max_rows:
            raise SparkGuardError(
                f"DataFrame has {actual_count:,} rows, exceeds max_rows ({max_rows:,}). "
                f"Use .limit() or Spark-native operations instead of toPandas()."
            )

        logger.info(f"Converting {actual_count:,} rows to pandas (within limit of {max_rows:,})")
        return df.toPandas()

    except Exception as e:
        if isinstance(e, SparkGuardError):
            raise
        logger.error(f"Error during safe_to_pandas: {e}")
        raise


def limit_for_pandas(df, max_rows: int = 10000):
    """
    Limit DataFrame before pandas conversion.

    Args:
        df: Spark DataFrame
        max_rows: Maximum rows to return

    Returns:
        Pandas DataFrame with at most max_rows

    Example:
        >>> pdf = limit_for_pandas(spark_df, max_rows=1000)
    """
    from pyspark.sql import DataFrame as SparkDataFrame

    if not isinstance(df, SparkDataFrame):
        raise ValueError("Input must be a Spark DataFrame")

    limited_df = df.limit(max_rows)
    actual_count = limited_df.count()

    logger.info(f"Converting limited DataFrame to pandas: {actual_count:,} rows")
    return limited_df.toPandas()


def check_spark_df(df, operation: str = "operation"):
    """
    Verify input is a Spark DataFrame.

    Args:
        df: Object to check
        operation: Description of operation for error message

    Raises:
        ValueError: If not a Spark DataFrame
    """
    from pyspark.sql import DataFrame as SparkDataFrame

    if not isinstance(df, SparkDataFrame):
        raise ValueError(
            f"{operation} requires Spark DataFrame, got {type(df).__name__}. "
            f"Avoid toPandas() conversions; use Spark-native operations."
        )


def validate_no_topandas_in_pipeline(module_name: str):
    """
    Runtime check that a module doesn't use toPandas() inappropriately.

    This is a development-time helper to catch violations.

    Args:
        module_name: Name of module to check

    Example:
        >>> validate_no_topandas_in_pipeline("decision_agent.features.tag_pca")
    """
    import sys
    import inspect

    if module_name not in sys.modules:
        logger.warning(f"Module {module_name} not loaded, skipping validation")
        return

    module = sys.modules[module_name]

    # Check source code for toPandas
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) or inspect.ismethod(obj):
            try:
                source = inspect.getsource(obj)
                if 'toPandas()' in source and 'safe_to_pandas' not in source:
                    logger.warning(
                        f"Found unsafe toPandas() in {module_name}.{name}. "
                        f"Use safe_to_pandas() or Spark-native operations."
                    )
            except (OSError, TypeError):
                # Can't get source for built-in functions
                pass
