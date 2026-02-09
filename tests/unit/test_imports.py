"""
Test that all modules can be imported successfully.

These tests do NOT require PySpark and can run in GitHub Actions CI.
"""
import pytest


def test_import_config():
    """Test config module imports"""
    from decision_agent.config import config_loader
    assert config_loader.ConfigLoader is not None


def test_import_orchestrator():
    """Test orchestrator module imports"""
    from decision_agent.orchestrator import router
    assert router.route_use_case is not None
    assert router.PIPELINE_REGISTRY is not None


def test_import_data_splits():
    """Test data splits module (no Spark required)"""
    from decision_agent.data import splits
    assert splits.create_temporal_splits is not None


def test_import_asof_join():
    """Test as-of join module"""
    from decision_agent.data import asof_join
    assert asof_join.PointInTimeJoiner is not None


def test_import_features():
    """Test feature modules"""
    from decision_agent.features import windows
    from decision_agent.features import tags
    from decision_agent.features import tag_pca
    from decision_agent.features import liquidity

    assert windows.compute_rolling_windows is not None
    assert tags.compute_tag_features is not None
    assert tag_pca.compute_tag_pca is not None
    assert liquidity.compute_liquidity_features is not None


def test_import_training():
    """Test training module"""
    from decision_agent.training import training_harness
    assert training_harness.TrainingHarness is not None


def test_import_validation():
    """Test validation modules"""
    from decision_agent.validation import segment_eval
    from decision_agent.validation import calibration_eval

    assert segment_eval.SegmentEvaluator is not None
    assert calibration_eval.CalibrationEvaluator is not None


def test_import_decisions():
    """Test decisions module"""
    from decision_agent.decisions import output_writer
    assert output_writer.write_decisions is not None


def test_import_utils():
    """Test utility modules (skip spark_utils which requires PySpark)"""
    from decision_agent.utils import mlflow_utils
    assert mlflow_utils.setup_mlflow is not None


def test_list_use_cases():
    """Test that use case registry is populated"""
    from decision_agent.orchestrator.router import list_available_use_cases

    use_cases = list_available_use_cases()
    assert len(use_cases) > 0
    assert "income_estimation" in use_cases
