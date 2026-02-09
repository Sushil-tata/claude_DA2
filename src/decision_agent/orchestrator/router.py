"""
Use case router that maps use_case_id to pipeline functions.
"""
from typing import Dict, Any, Callable
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def income_estimation_pipeline(config: Dict[str, Any], spark=None) -> Dict[str, Any]:
    """
    Income estimation pipeline.

    Args:
        config: Use case configuration
        spark: Spark session (optional, will create if None)

    Returns:
        Pipeline execution results
    """
    from decision_agent.data.synthetic_data import generate_income_synthetic_data
    from decision_agent.data.splits import create_temporal_splits
    from decision_agent.features.income_features import compute_income_features
    from decision_agent.training.training_harness import TrainingHarness
    from decision_agent.validation.segment_eval import SegmentEvaluator
    from decision_agent.validation.calibration_eval import CalibrationEvaluator
    from decision_agent.decisions.output_writer import write_decisions

    logger.info(f"Starting income estimation pipeline: {config['use_case_id']} {config['version']}")

    # Initialize Spark if needed
    if spark is None:
        from decision_agent.utils.spark_utils import get_spark_session
        spark = get_spark_session()

    results = {}

    # Step 1: Load or generate data
    logger.info("Step 1: Loading/generating data...")
    if config.get("data", {}).get("use_synthetic", False):
        df = generate_income_synthetic_data(spark, num_customers=1000, num_days=365)
        results["data_source"] = "synthetic"
        results["num_records"] = df.count()
    else:
        source_table = config.get("data", {}).get("source_table")
        if not source_table:
            raise ValueError("Must specify data.source_table or set data.use_synthetic=true")
        df = spark.table(source_table)
        results["data_source"] = source_table

    # Step 2: Temporal splits
    logger.info("Step 2: Creating temporal train/val/test splits...")
    train_df, val_df, test_df = create_temporal_splits(
        df,
        date_col="transaction_timestamp",
        train_end=config["data"]["train_end_date"],
        val_end=config["data"]["val_end_date"]
    )
    results["split_sizes"] = {
        "train": train_df.count(),
        "val": val_df.count(),
        "test": test_df.count()
    }

    # Step 3: Feature engineering
    logger.info("Step 3: Computing features...")
    train_features = compute_income_features(train_df, config["features"])
    val_features = compute_income_features(val_df, config["features"])
    test_features = compute_income_features(test_df, config["features"])

    # Step 4: Training
    logger.info("Step 4: Training model...")
    harness = TrainingHarness(config["model"])
    model, train_metrics = harness.train(train_features, config["model"]["target_column"])
    results["train_metrics"] = train_metrics

    # Step 5: Validation
    logger.info("Step 5: Validating model...")

    # Segment evaluation
    segment_evaluator = SegmentEvaluator(config["validation"]["segments"])
    val_pred = harness.predict(model, val_features)
    segment_metrics = segment_evaluator.evaluate(
        val_features, val_pred, config["model"]["target_column"]
    )
    results["segment_metrics"] = segment_metrics

    # Calibration evaluation (for classification/probability outputs)
    if config["model"]["algorithm"] in ["logistic_regression"]:
        calibration_evaluator = CalibrationEvaluator(config["validation"]["calibration_config"])
        calibration_metrics = calibration_evaluator.evaluate(val_features, val_pred)
        results["calibration_metrics"] = calibration_metrics

    # Step 6: Test set scoring
    logger.info("Step 6: Scoring test set...")
    test_pred = harness.predict(model, test_features)
    test_metrics = harness.evaluate(test_features, test_pred, config["model"]["target_column"])
    results["test_metrics"] = test_metrics

    # Step 7: Write decisions
    logger.info("Step 7: Writing decisions to Delta Lake...")

    # Add predictions to Spark DataFrame (no toPandas)
    from pyspark.sql.types import DoubleType, LongType, StructType, StructField
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    # Create predictions DataFrame
    predictions_data = [(i, float(pred)) for i, pred in enumerate(test_pred)]
    predictions_schema = StructType([
        StructField("_row_num", LongType(), False),
        StructField("prediction", DoubleType(), False)
    ])

    predictions_spark_df = spark.createDataFrame(predictions_data, predictions_schema)

    # Add row numbers to test_features and join
    window_spec = Window.orderBy(F.monotonically_increasing_id())
    test_with_rownum = test_features.withColumn("_row_num", F.row_number().over(window_spec) - 1)
    test_with_predictions = test_with_rownum.join(predictions_spark_df, "_row_num").drop("_row_num")

    # Write decisions (Spark-native)
    decision_table = write_decisions(
        spark=spark,
        predictions_df=test_with_predictions,
        config=config,
        model_version="v1.0",
        run_id="local_run_001"
    )
    results["output_table"] = config["output"]["table_name"]
    results["decisions_written"] = len(test_pred)

    logger.info("Pipeline completed successfully!")
    return results


def churn_prediction_pipeline(config: Dict[str, Any], spark=None) -> Dict[str, Any]:
    """
    Churn prediction pipeline (placeholder for future implementation).

    Args:
        config: Use case configuration
        spark: Spark session

    Returns:
        Pipeline execution results
    """
    logger.info(f"Churn prediction pipeline: {config['use_case_id']}")
    raise NotImplementedError("Churn prediction pipeline not yet implemented")


def credit_scoring_pipeline(config: Dict[str, Any], spark=None) -> Dict[str, Any]:
    """
    Credit scoring pipeline (placeholder for future implementation).

    Args:
        config: Use case configuration
        spark: Spark session

    Returns:
        Pipeline execution results
    """
    logger.info(f"Credit scoring pipeline: {config['use_case_id']}")
    raise NotImplementedError("Credit scoring pipeline not yet implemented")


# Pipeline registry mapping use_case_id to pipeline functions
PIPELINE_REGISTRY: Dict[str, Callable] = {
    "income_estimation": income_estimation_pipeline,
    "churn_prediction": churn_prediction_pipeline,
    "credit_scoring": credit_scoring_pipeline,
}


def route_use_case(use_case_id: str, config: Dict[str, Any], spark=None) -> Dict[str, Any]:
    """
    Route to appropriate pipeline based on use_case_id.

    Args:
        use_case_id: Unique identifier for the use case
        config: Full configuration dictionary
        spark: Optional Spark session

    Returns:
        Pipeline execution results

    Raises:
        ValueError: If use_case_id is not registered
    """
    pipeline_fn = PIPELINE_REGISTRY.get(use_case_id)

    if not pipeline_fn:
        available = ", ".join(PIPELINE_REGISTRY.keys())
        raise ValueError(
            f"Unknown use_case_id: '{use_case_id}'. Available use cases: {available}"
        )

    logger.info(f"Routing to pipeline: {use_case_id}")
    return pipeline_fn(config, spark)


def list_available_use_cases() -> list:
    """
    List all registered use cases.

    Returns:
        List of available use_case_id strings
    """
    return list(PIPELINE_REGISTRY.keys())
