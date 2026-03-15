"""
Configuration loader with JSON schema validation.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any
import jsonschema


class ConfigLoader:
    """Load and validate use case configurations against JSON schema."""

    def __init__(self, schema_path: str):
        """
        Initialize config loader with schema.

        Args:
            schema_path: Path to JSON schema file
        """
        with open(schema_path, 'r') as f:
            self.schema = json.load(f)

    def load_use_case(self, config_path: str) -> Dict[str, Any]:
        """
        Load and validate use case YAML configuration.

        Args:
            config_path: Path to YAML config file

        Returns:
            Validated configuration dictionary

        Raises:
            jsonschema.ValidationError: If config doesn't match schema
            FileNotFoundError: If config file doesn't exist
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Validate against JSON schema
        jsonschema.validate(config, self.schema)

        print(f"✓ Configuration loaded and validated: {config['use_case_id']} {config['version']}")
        return config

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate a config dictionary against schema.

        Args:
            config: Configuration dictionary

        Returns:
            True if valid

        Raises:
            jsonschema.ValidationError: If invalid
        """
        jsonschema.validate(config, self.schema)
        return True
