"""
Test configuration loading and validation (no Spark required).
"""
import pytest
import tempfile
import yaml
from pathlib import Path
from decision_agent.config.config_loader import ConfigLoader, load_config


def test_config_loader_initialization():
    """Test ConfigLoader can be initialized with default schema"""
    loader = ConfigLoader()
    assert loader.schema is not None
    assert "properties" in loader.schema


def test_load_valid_config():
    """Test loading a valid configuration"""
    # Use the actual income_estimation.yaml
    repo_root = Path(__file__).parent.parent.parent
    config_path = repo_root / "conf" / "use_cases" / "income_estimation.yaml"

    if config_path.exists():
        config = load_config(str(config_path))
        assert config["use_case_id"] == "income_estimation"
        assert config["version"] == "v1.0.0"
        assert "features" in config
        assert "model" in config
        assert "validation" in config
        assert "output" in config


def test_template_substitution():
    """Test template variable substitution"""
    loader = ConfigLoader()

    # Create a config with template variable
    config = {
        "use_case_id": "test",
        "version": "v1.0.0",
        "features": {
            "snapshot_timestamp": "{{ execution_date }}",
            "lookback_windows": [7, 30],
            "feature_list": []
        },
        "model": {
            "algorithm": "gradient_boosting",
            "hyperparameters": {}
        },
        "validation": {
            "segments": [],
            "calibration_config": {}
        },
        "output": {
            "table_name": "test.decisions"
        }
    }

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        temp_path = f.name

    try:
        # Load with template substitution
        loaded = loader.load_use_case(temp_path, execution_date="2024-12-01")
        assert loaded["features"]["snapshot_timestamp"] == "2024-12-01"
    finally:
        Path(temp_path).unlink()


def test_invalid_version_format():
    """Test that invalid version format is rejected"""
    loader = ConfigLoader()

    invalid_config = {
        "use_case_id": "test",
        "version": "1.0.0",  # Missing 'v' prefix
        "features": {"lookback_windows": [], "feature_list": []},
        "model": {"algorithm": "gradient_boosting", "hyperparameters": {}},
        "validation": {"segments": [], "calibration_config": {}},
        "output": {"table_name": "test.decisions"}
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(invalid_config, f)
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="version"):
            loader.load_use_case(temp_path)
    finally:
        Path(temp_path).unlink()


def test_missing_required_field():
    """Test that missing required fields are detected"""
    loader = ConfigLoader()

    incomplete_config = {
        "use_case_id": "test",
        "version": "v1.0.0"
        # Missing features, model, validation, output
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(incomplete_config, f)
        temp_path = f.name

    try:
        with pytest.raises(ValueError):
            loader.load_use_case(temp_path)
    finally:
        Path(temp_path).unlink()


def test_invalid_algorithm():
    """Test that invalid algorithm is rejected"""
    loader = ConfigLoader()

    config = {
        "use_case_id": "test",
        "version": "v1.0.0",
        "features": {"lookback_windows": [7], "feature_list": []},
        "model": {"algorithm": "invalid_algorithm", "hyperparameters": {}},
        "validation": {"segments": [], "calibration_config": {}},
        "output": {"table_name": "test.decisions"}
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="algorithm"):
            loader.load_use_case(temp_path)
    finally:
        Path(temp_path).unlink()
