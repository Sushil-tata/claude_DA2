"""
Config loader with JSON schema validation.
"""
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import re


class ConfigLoader:
    """Load and validate use case configurations against JSON schema"""

    def __init__(self, schema_path: Optional[str] = None):
        """
        Initialize config loader with JSON schema.

        Args:
            schema_path: Path to JSON schema file. If None, uses default schema.
        """
        if schema_path is None:
            # Default schema path relative to this file
            repo_root = Path(__file__).parent.parent.parent.parent
            schema_path = repo_root / "schemas" / "config_schemas" / "use_case_config.schema.json"

        with open(schema_path) as f:
            self.schema = json.load(f)

    def load_use_case(self, config_path: str, execution_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Load and validate use case YAML configuration.

        Args:
            config_path: Path to YAML config file
            execution_date: Optional execution date for template substitution

        Returns:
            Validated configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML is malformed
            ValueError: If config fails validation
        """
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Template substitution
        if execution_date:
            config = self._substitute_templates(config, {"execution_date": execution_date})

        # Validate against schema
        try:
            import jsonschema
            jsonschema.validate(config, self.schema)
        except ImportError:
            # If jsonschema not available, do basic validation
            self._basic_validation(config)
        except jsonschema.ValidationError as e:
            raise ValueError(f"Config validation failed: {e.message}")

        return config

    def _substitute_templates(self, config: Dict[str, Any], variables: Dict[str, str]) -> Dict[str, Any]:
        """
        Substitute template variables in config (e.g., {{ execution_date }}).

        Args:
            config: Configuration dictionary
            variables: Variables for substitution

        Returns:
            Config with templates substituted
        """
        config_str = json.dumps(config)

        # Replace {{ variable_name }} with actual values
        for var_name, var_value in variables.items():
            pattern = r'\{\{\s*' + var_name + r'\s*\}\}'
            config_str = re.sub(pattern, var_value, config_str)

        return json.loads(config_str)

    def _basic_validation(self, config: Dict[str, Any]) -> None:
        """
        Basic validation when jsonschema is not available.

        Args:
            config: Configuration to validate

        Raises:
            ValueError: If required fields are missing
        """
        required_fields = ["use_case_id", "version", "features", "model", "validation", "output"]

        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")

        # Validate version format
        version = config.get("version", "")
        if not re.match(r'^v\d+\.\d+\.\d+$', version):
            raise ValueError(f"Invalid version format: {version}. Expected format: vX.Y.Z")

        # Validate algorithm
        valid_algorithms = ["gradient_boosting", "random_forest", "logistic_regression", "linear_regression"]
        algorithm = config.get("model", {}).get("algorithm")
        if algorithm not in valid_algorithms:
            raise ValueError(f"Invalid algorithm: {algorithm}. Must be one of {valid_algorithms}")


def load_config(config_path: str, execution_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to load configuration.

    Args:
        config_path: Path to YAML config file
        execution_date: Optional execution date for template substitution

    Returns:
        Validated configuration dictionary
    """
    loader = ConfigLoader()
    return loader.load_use_case(config_path, execution_date)
