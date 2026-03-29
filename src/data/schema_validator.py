"""
Schema Validator Module

Comprehensive schema definition and validation for data quality assurance.
Supports type checking, range validation, required fields, and custom rules.

Author: Principal Data Science Decision Agent
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd
from loguru import logger


class FieldType(Enum):
    """Supported field types for schema validation."""

    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    DATE = "date"
    CATEGORY = "category"
    EMAIL = "email"
    URL = "url"
    JSON = "json"


@dataclass
class FieldSchema:
    """Schema definition for a single field."""

    name: str
    field_type: FieldType
    required: bool = False
    nullable: bool = True
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    allowed_values: Optional[Set[Any]] = None
    regex_pattern: Optional[str] = None
    custom_validators: List[Callable] = field(default_factory=list)
    description: Optional[str] = None

    def __post_init__(self):
        """Convert allowed_values to set if it's a list."""
        if isinstance(self.allowed_values, list):
            self.allowed_values = set(self.allowed_values)


@dataclass
class DataSchema:
    """Complete schema definition for a dataset."""

    fields: Dict[str, FieldSchema]
    strict: bool = False  # If True, reject unknown columns
    row_validators: List[Callable] = field(default_factory=list)
    description: Optional[str] = None

    def add_field(self, field_schema: FieldSchema) -> None:
        """Add a field to the schema."""
        self.fields[field_schema.name] = field_schema

    def add_row_validator(self, validator: Callable) -> None:
        """Add a row-level validator function."""
        self.row_validators.append(validator)


@dataclass
class ValidationResult:
    """Result of schema validation."""

    is_valid: bool
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    field_results: Dict[str, Dict[str, Any]]
    row_errors: List[Dict[str, Any]]
    summary: Dict[str, Any]

    def get_error_count(self) -> int:
        """Get total number of errors."""
        return len(self.errors)

    def get_warning_count(self) -> int:
        """Get total number of warnings."""
        return len(self.warnings)


class SchemaValidator:
    """
    Schema validator for DataFrames.

    Validates data against defined schemas with support for:
    - Data type checking
    - Range validation
    - Required field validation
    - Custom validation rules
    - Row-level validation
    - Detailed validation reporting

    Examples:
        >>> schema = DataSchema(fields={
        ...     "age": FieldSchema(
        ...         name="age",
        ...         field_type=FieldType.INTEGER,
        ...         required=True,
        ...         min_value=0,
        ...         max_value=120
        ...     )
        ... })
        >>> validator = SchemaValidator(schema)
        >>> result = validator.validate(df)
        >>> if not result.is_valid:
        ...     print(f"Errors: {result.get_error_count()}")
    """

    def __init__(self, schema: DataSchema):
        """
        Initialize SchemaValidator.

        Args:
            schema: DataSchema object defining validation rules
        """
        self.schema = schema
        logger.info("SchemaValidator initialized with {} fields", len(schema.fields))

    def validate(
        self,
        df: pd.DataFrame,
        sample_size: Optional[int] = None,
        raise_on_error: bool = False,
    ) -> ValidationResult:
        """
        Validate DataFrame against schema.

        Args:
            df: DataFrame to validate
            sample_size: Optional sample size for large datasets
            raise_on_error: If True, raise exception on validation failure

        Returns:
            ValidationResult object

        Raises:
            ValueError: If raise_on_error=True and validation fails
        """
        logger.info("Starting schema validation for DataFrame with {} rows", len(df))

        errors = []
        warnings = []
        field_results = {}
        row_errors = []

        # Sample data if needed
        if sample_size and len(df) > sample_size:
            logger.info("Sampling {} rows for validation", sample_size)
            df_validate = df.sample(n=sample_size, random_state=42)
        else:
            df_validate = df

        # Check for unknown columns (if strict mode)
        if self.schema.strict:
            unknown_cols = set(df.columns) - set(self.schema.fields.keys())
            if unknown_cols:
                errors.append(
                    {
                        "type": "unknown_columns",
                        "columns": list(unknown_cols),
                        "message": f"Unknown columns found: {unknown_cols}",
                    }
                )

        # Check for missing required columns
        missing_cols = set(
            name
            for name, field in self.schema.fields.items()
            if field.required and name not in df.columns
        )
        if missing_cols:
            errors.append(
                {
                    "type": "missing_columns",
                    "columns": list(missing_cols),
                    "message": f"Required columns missing: {missing_cols}",
                }
            )

        # Validate each field
        for field_name, field_schema in self.schema.fields.items():
            if field_name not in df.columns:
                continue

            logger.debug("Validating field: {}", field_name)
            field_result = self._validate_field(
                df_validate[field_name], field_schema
            )
            field_results[field_name] = field_result

            errors.extend(field_result.get("errors", []))
            warnings.extend(field_result.get("warnings", []))

        # Row-level validation
        if self.schema.row_validators:
            logger.debug("Running row-level validators")
            row_errors = self._validate_rows(df_validate)
            errors.extend(row_errors)

        # Calculate summary
        summary = self._calculate_summary(
            df_validate, field_results, errors, warnings
        )

        is_valid = len(errors) == 0

        result = ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            field_results=field_results,
            row_errors=row_errors,
            summary=summary,
        )

        logger.info(
            "Validation complete: {} - {} errors, {} warnings",
            "PASSED" if is_valid else "FAILED",
            len(errors),
            len(warnings),
        )

        if raise_on_error and not is_valid:
            raise ValueError(
                f"Schema validation failed with {len(errors)} errors"
            )

        return result

    def validate_field(
        self, data: pd.Series, field_schema: FieldSchema
    ) -> Dict[str, Any]:
        """
        Validate a single field/column.

        Args:
            data: Series to validate
            field_schema: FieldSchema defining validation rules

        Returns:
            Dictionary with validation results
        """
        return self._validate_field(data, field_schema)

    def _validate_field(
        self, data: pd.Series, field_schema: FieldSchema
    ) -> Dict[str, Any]:
        """Internal field validation logic."""
        errors = []
        warnings = []
        stats = {}

        # Check for null values
        null_count = data.isnull().sum()
        null_pct = null_count / len(data)
        stats["null_count"] = int(null_count)
        stats["null_percentage"] = float(null_pct)

        if not field_schema.nullable and null_count > 0:
            errors.append(
                {
                    "type": "null_violation",
                    "field": field_schema.name,
                    "message": f"Field is not nullable but has {null_count} null values",
                    "count": int(null_count),
                }
            )

        # Work with non-null values for remaining checks
        non_null_data = data.dropna()

        if len(non_null_data) == 0:
            return {
                "stats": stats,
                "errors": errors,
                "warnings": warnings,
                "valid_count": 0,
                "invalid_count": int(null_count),
            }

        # Type validation
        type_errors = self._validate_type(non_null_data, field_schema)
        errors.extend(type_errors)

        # Range validation
        if field_schema.min_value is not None or field_schema.max_value is not None:
            range_errors = self._validate_range(non_null_data, field_schema)
            errors.extend(range_errors)

        # Length validation (for strings)
        if field_schema.field_type == FieldType.STRING:
            if field_schema.min_length is not None or field_schema.max_length is not None:
                length_errors = self._validate_length(non_null_data, field_schema)
                errors.extend(length_errors)

        # Allowed values validation
        if field_schema.allowed_values is not None:
            allowed_errors = self._validate_allowed_values(
                non_null_data, field_schema
            )
            errors.extend(allowed_errors)

        # Regex pattern validation
        if field_schema.regex_pattern is not None:
            pattern_errors = self._validate_regex(non_null_data, field_schema)
            errors.extend(pattern_errors)

        # Custom validators
        for validator in field_schema.custom_validators:
            custom_errors = self._run_custom_validator(
                non_null_data, field_schema, validator
            )
            errors.extend(custom_errors)

        # Calculate valid/invalid counts
        total_errors = sum(err.get("count", 1) for err in errors)
        invalid_count = min(total_errors, len(data))
        valid_count = len(data) - invalid_count

        return {
            "stats": stats,
            "errors": errors,
            "warnings": warnings,
            "valid_count": int(valid_count),
            "invalid_count": int(invalid_count),
        }

    def _validate_type(
        self, data: pd.Series, field_schema: FieldSchema
    ) -> List[Dict[str, Any]]:
        """Validate data types."""
        errors = []
        field_type = field_schema.field_type

        if field_type == FieldType.INTEGER:
            invalid = data[~data.apply(lambda x: isinstance(x, (int, np.integer)))]
        elif field_type == FieldType.FLOAT:
            invalid = data[~data.apply(lambda x: isinstance(x, (int, float, np.number)))]
        elif field_type == FieldType.STRING:
            invalid = data[~data.apply(lambda x: isinstance(x, str))]
        elif field_type == FieldType.BOOLEAN:
            invalid = data[~data.apply(lambda x: isinstance(x, (bool, np.bool_)))]
        elif field_type == FieldType.DATETIME:
            invalid = data[~data.apply(lambda x: isinstance(x, (pd.Timestamp, datetime)))]
        elif field_type == FieldType.EMAIL:
            import re
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            invalid = data[~data.apply(lambda x: isinstance(x, str) and re.match(email_pattern, x))]
        elif field_type == FieldType.URL:
            import re
            url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
            invalid = data[~data.apply(lambda x: isinstance(x, str) and re.match(url_pattern, x))]
        else:
            # Default: no specific validation
            return errors

        if len(invalid) > 0:
            errors.append(
                {
                    "type": "type_violation",
                    "field": field_schema.name,
                    "expected_type": field_type.value,
                    "message": f"{len(invalid)} values have incorrect type",
                    "count": int(len(invalid)),
                    "sample_values": invalid.head(5).tolist(),
                }
            )

        return errors

    def _validate_range(
        self, data: pd.Series, field_schema: FieldSchema
    ) -> List[Dict[str, Any]]:
        """Validate numeric ranges."""
        errors = []

        if field_schema.min_value is not None:
            below_min = data[data < field_schema.min_value]
            if len(below_min) > 0:
                errors.append(
                    {
                        "type": "range_violation",
                        "field": field_schema.name,
                        "constraint": f"min_value={field_schema.min_value}",
                        "message": f"{len(below_min)} values below minimum",
                        "count": int(len(below_min)),
                        "min_found": float(below_min.min()),
                    }
                )

        if field_schema.max_value is not None:
            above_max = data[data > field_schema.max_value]
            if len(above_max) > 0:
                errors.append(
                    {
                        "type": "range_violation",
                        "field": field_schema.name,
                        "constraint": f"max_value={field_schema.max_value}",
                        "message": f"{len(above_max)} values above maximum",
                        "count": int(len(above_max)),
                        "max_found": float(above_max.max()),
                    }
                )

        return errors

    def _validate_length(
        self, data: pd.Series, field_schema: FieldSchema
    ) -> List[Dict[str, Any]]:
        """Validate string lengths."""
        errors = []
        lengths = data.str.len()

        if field_schema.min_length is not None:
            too_short = lengths[lengths < field_schema.min_length]
            if len(too_short) > 0:
                errors.append(
                    {
                        "type": "length_violation",
                        "field": field_schema.name,
                        "constraint": f"min_length={field_schema.min_length}",
                        "message": f"{len(too_short)} values too short",
                        "count": int(len(too_short)),
                    }
                )

        if field_schema.max_length is not None:
            too_long = lengths[lengths > field_schema.max_length]
            if len(too_long) > 0:
                errors.append(
                    {
                        "type": "length_violation",
                        "field": field_schema.name,
                        "constraint": f"max_length={field_schema.max_length}",
                        "message": f"{len(too_long)} values too long",
                        "count": int(len(too_long)),
                    }
                )

        return errors

    def _validate_allowed_values(
        self, data: pd.Series, field_schema: FieldSchema
    ) -> List[Dict[str, Any]]:
        """Validate against allowed values."""
        errors = []

        invalid = data[~data.isin(field_schema.allowed_values)]
        if len(invalid) > 0:
            unique_invalid = invalid.unique()
            errors.append(
                {
                    "type": "allowed_values_violation",
                    "field": field_schema.name,
                    "message": f"{len(invalid)} values not in allowed set",
                    "count": int(len(invalid)),
                    "invalid_values": unique_invalid[:10].tolist(),
                    "allowed_values": list(field_schema.allowed_values)[:10],
                }
            )

        return errors

    def _validate_regex(
        self, data: pd.Series, field_schema: FieldSchema
    ) -> List[Dict[str, Any]]:
        """Validate against regex pattern."""
        import re

        errors = []
        pattern = re.compile(field_schema.regex_pattern)

        invalid = data[~data.astype(str).str.match(pattern, na=False)]
        if len(invalid) > 0:
            errors.append(
                {
                    "type": "pattern_violation",
                    "field": field_schema.name,
                    "pattern": field_schema.regex_pattern,
                    "message": f"{len(invalid)} values don't match pattern",
                    "count": int(len(invalid)),
                    "sample_values": invalid.head(5).tolist(),
                }
            )

        return errors

    def _run_custom_validator(
        self, data: pd.Series, field_schema: FieldSchema, validator: Callable
    ) -> List[Dict[str, Any]]:
        """Run custom validator function."""
        errors = []

        try:
            # Custom validator should return boolean Series or raise exception
            is_valid = validator(data)

            if isinstance(is_valid, pd.Series):
                invalid = data[~is_valid]
                if len(invalid) > 0:
                    errors.append(
                        {
                            "type": "custom_validation",
                            "field": field_schema.name,
                            "validator": validator.__name__,
                            "message": f"{len(invalid)} values failed custom validation",
                            "count": int(len(invalid)),
                        }
                    )
            elif not is_valid:
                errors.append(
                    {
                        "type": "custom_validation",
                        "field": field_schema.name,
                        "validator": validator.__name__,
                        "message": "Custom validation failed",
                    }
                )

        except Exception as e:
            logger.error(
                "Custom validator {} failed: {}", validator.__name__, str(e)
            )
            errors.append(
                {
                    "type": "custom_validation_error",
                    "field": field_schema.name,
                    "validator": validator.__name__,
                    "message": f"Validator raised exception: {str(e)}",
                }
            )

        return errors

    def _validate_rows(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Run row-level validators."""
        errors = []

        for validator in self.schema.row_validators:
            try:
                # Row validator should return boolean Series
                is_valid = validator(df)

                if isinstance(is_valid, pd.Series):
                    invalid_rows = df[~is_valid]
                    if len(invalid_rows) > 0:
                        errors.append(
                            {
                                "type": "row_validation",
                                "validator": validator.__name__,
                                "message": f"{len(invalid_rows)} rows failed validation",
                                "count": int(len(invalid_rows)),
                            }
                        )

            except Exception as e:
                logger.error(
                    "Row validator {} failed: {}", validator.__name__, str(e)
                )
                errors.append(
                    {
                        "type": "row_validation_error",
                        "validator": validator.__name__,
                        "message": f"Validator raised exception: {str(e)}",
                    }
                )

        return errors

    def _calculate_summary(
        self,
        df: pd.DataFrame,
        field_results: Dict,
        errors: List,
        warnings: List,
    ) -> Dict[str, Any]:
        """Calculate validation summary statistics."""
        total_errors = sum(
            result.get("invalid_count", 0) for result in field_results.values()
        )
        total_values = len(df) * len(field_results)

        return {
            "total_rows": len(df),
            "total_fields": len(field_results),
            "total_values": total_values,
            "total_errors": total_errors,
            "total_warnings": len(warnings),
            "error_rate": total_errors / total_values if total_values > 0 else 0,
            "fields_with_errors": sum(
                1 for r in field_results.values() if r.get("invalid_count", 0) > 0
            ),
        }


# Helper functions for common schemas
def create_numeric_field(
    name: str,
    required: bool = False,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    allow_integers_only: bool = False,
) -> FieldSchema:
    """Create a numeric field schema."""
    field_type = FieldType.INTEGER if allow_integers_only else FieldType.FLOAT
    return FieldSchema(
        name=name,
        field_type=field_type,
        required=required,
        min_value=min_value,
        max_value=max_value,
    )


def create_string_field(
    name: str,
    required: bool = False,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    allowed_values: Optional[Set[str]] = None,
    regex_pattern: Optional[str] = None,
) -> FieldSchema:
    """Create a string field schema."""
    return FieldSchema(
        name=name,
        field_type=FieldType.STRING,
        required=required,
        min_length=min_length,
        max_length=max_length,
        allowed_values=allowed_values,
        regex_pattern=regex_pattern,
    )


def create_categorical_field(
    name: str, categories: Set[str], required: bool = False
) -> FieldSchema:
    """Create a categorical field schema."""
    return FieldSchema(
        name=name,
        field_type=FieldType.CATEGORY,
        required=required,
        allowed_values=categories,
    )


def create_datetime_field(
    name: str,
    required: bool = False,
    min_value: Optional[datetime] = None,
    max_value: Optional[datetime] = None,
) -> FieldSchema:
    """Create a datetime field schema."""
    return FieldSchema(
        name=name,
        field_type=FieldType.DATETIME,
        required=required,
        min_value=min_value,
        max_value=max_value,
    )


def infer_schema_from_dataframe(
    df: pd.DataFrame, sample_size: int = 1000, strict: bool = False
) -> DataSchema:
    """
    Infer schema from a DataFrame.

    Args:
        df: DataFrame to infer schema from
        sample_size: Number of rows to sample for inference
        strict: Whether to create strict schema

    Returns:
        Inferred DataSchema
    """
    logger.info("Inferring schema from DataFrame")

    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df

    fields = {}

    for col in df.columns:
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        nullable = null_count > 0

        # Infer field type
        if pd.api.types.is_integer_dtype(dtype):
            field_type = FieldType.INTEGER
            min_val = float(df_sample[col].min())
            max_val = float(df_sample[col].max())
            field = FieldSchema(
                name=col,
                field_type=field_type,
                nullable=nullable,
                min_value=min_val,
                max_value=max_val,
            )
        elif pd.api.types.is_float_dtype(dtype):
            field_type = FieldType.FLOAT
            min_val = float(df_sample[col].min())
            max_val = float(df_sample[col].max())
            field = FieldSchema(
                name=col,
                field_type=field_type,
                nullable=nullable,
                min_value=min_val,
                max_value=max_val,
            )
        elif pd.api.types.is_bool_dtype(dtype):
            field = FieldSchema(
                name=col, field_type=FieldType.BOOLEAN, nullable=nullable
            )
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            field = FieldSchema(
                name=col, field_type=FieldType.DATETIME, nullable=nullable
            )
        elif pd.api.types.is_categorical_dtype(dtype):
            categories = set(df[col].cat.categories)
            field = FieldSchema(
                name=col,
                field_type=FieldType.CATEGORY,
                nullable=nullable,
                allowed_values=categories,
            )
        else:
            # String or object type
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.05:  # Low cardinality - treat as category
                categories = set(df_sample[col].dropna().unique())
                field = FieldSchema(
                    name=col,
                    field_type=FieldType.CATEGORY,
                    nullable=nullable,
                    allowed_values=categories,
                )
            else:
                field = FieldSchema(
                    name=col, field_type=FieldType.STRING, nullable=nullable
                )

        fields[col] = field

    schema = DataSchema(fields=fields, strict=strict)
    logger.info("Schema inferred with {} fields", len(fields))

    return schema
