"""
Data Layer Module

Comprehensive data handling including loading, quality assessment,
schema validation, and exploratory data analysis.
"""

from .data_loader import (
    DataLoader,
    DataLoaderConfig,
    detect_delimiter,
    get_file_info,
)
from .data_quality import (
    DataQualityAnalyzer,
    DataQualityConfig,
    QualityReport,
    generate_quality_report_html,
)
from .eda_engine import (
    EDAEngine,
    EDAConfig,
    EDAReport,
    UnivariateAnalysis,
    BivariateAnalysis,
    generate_eda_summary,
)
from .schema_validator import (
    SchemaValidator,
    DataSchema,
    FieldSchema,
    FieldType,
    ValidationResult,
    create_numeric_field,
    create_string_field,
    create_categorical_field,
    create_datetime_field,
    infer_schema_from_dataframe,
)

__all__ = [
    # Data Loader
    "DataLoader",
    "DataLoaderConfig",
    "detect_delimiter",
    "get_file_info",
    # Data Quality
    "DataQualityAnalyzer",
    "DataQualityConfig",
    "QualityReport",
    "generate_quality_report_html",
    # EDA Engine
    "EDAEngine",
    "EDAConfig",
    "EDAReport",
    "UnivariateAnalysis",
    "BivariateAnalysis",
    "generate_eda_summary",
    # Schema Validator
    "SchemaValidator",
    "DataSchema",
    "FieldSchema",
    "FieldType",
    "ValidationResult",
    "create_numeric_field",
    "create_string_field",
    "create_categorical_field",
    "create_datetime_field",
    "infer_schema_from_dataframe",
]
