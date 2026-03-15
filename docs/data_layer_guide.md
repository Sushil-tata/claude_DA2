# Data Layer Documentation

Comprehensive data handling layer for the Principal Data Science Decision Agent.

## Overview

The data layer provides four main modules for robust data processing:

1. **Data Loader** - Multi-format data loading with connection pooling
2. **Data Quality** - Comprehensive quality assessment and scoring
3. **Schema Validator** - Schema definition and validation
4. **EDA Engine** - Automated exploratory data analysis

## Installation

All required dependencies are in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Key dependencies:
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- scipy >= 1.11.0
- sqlalchemy >= 2.0.0
- loguru >= 0.7.0
- openpyxl >= 3.1.0

## Module 1: Data Loader

### Features

- **Multiple Format Support**: CSV, JSON, Excel, SQL databases
- **Connection Pooling**: Efficient database connection management
- **Chunked Loading**: Handle large files without memory issues
- **Type Inference**: Automatic data type optimization
- **Robust Error Handling**: Graceful failure with detailed logging

### Usage Examples

#### Basic File Loading

```python
from data import DataLoader

loader = DataLoader()

# Load CSV
df = loader.load_csv("data.csv")

# Load Excel
df = loader.load_excel("data.xlsx", sheet_name="Sheet1")

# Load JSON
df = loader.load_json("data.json")
```

#### Database Operations

```python
# Load from SQL query
connection_string = "postgresql://user:pass@host/db"
df = loader.load_sql(
    "SELECT * FROM users WHERE active = true",
    connection_string
)

# Load entire table
df = loader.load_table("users", connection_string)

# Get table metadata
info = loader.get_table_info(connection_string, "users")
print(info["columns"])
```

#### Large File Handling

```python
# Process large CSV in chunks
for chunk in loader.load_csv_chunked("large_file.csv", chunk_size=10000):
    process(chunk)

# Chunked SQL loading
for chunk in loader.load_sql_chunked(query, connection_string, chunk_size=5000):
    process(chunk)
```

#### Custom Configuration

```python
from data import DataLoaderConfig

config = DataLoaderConfig(
    chunk_size=20000,
    pool_size=10,
    encoding="utf-8"
)

loader = DataLoader(config)
```

### Advanced Features

#### Database Transactions

```python
with loader.transaction(connection_string) as conn:
    conn.execute("INSERT INTO table VALUES (...)")
    conn.execute("UPDATE table SET ...")
    # Automatically commits or rolls back
```

#### File Information

```python
from data import get_file_info, detect_delimiter

info = get_file_info("data.csv")
print(f"Size: {info['size_mb']:.2f} MB")

delimiter = detect_delimiter("data.csv")
df = loader.load_csv("data.csv", delimiter=delimiter)
```

## Module 2: Data Quality

### Features

- **Missing Value Analysis**: Count, percentage, patterns
- **Outlier Detection**: IQR, Z-score, Isolation Forest
- **Distribution Analysis**: Skewness, kurtosis, normality tests
- **Consistency Checks**: Duplicates, referential integrity
- **Quality Scoring**: Weighted scoring across dimensions

### Usage Examples

#### Basic Quality Analysis

```python
from data import DataQualityAnalyzer

analyzer = DataQualityAnalyzer()
report = analyzer.analyze(df, target_column="target")

print(f"Overall Quality Score: {report.overall_score:.2f}")
print(f"Completeness: {report.dimension_scores['completeness']:.2f}")
print(f"Validity: {report.dimension_scores['validity']:.2f}")
print(f"Issues Found: {len(report.issues)}")
```

#### Detailed Analysis

```python
# Missing value analysis
missing_analysis = analyzer.analyze_missing_values(df)
print(f"Overall Missing Rate: {missing_analysis['overall_missing_rate']:.2%}")

for col_info in missing_analysis['column_details']:
    if col_info['missing_percentage'] > 0.1:
        print(f"{col_info['column']}: {col_info['missing_percentage']:.1%} missing")

# Outlier detection
outlier_analysis = analyzer.detect_outliers(df)
print(f"Total Outlier Flags: {outlier_analysis['summary']['total_outlier_flags']}")

# Distribution analysis
dist_analysis = analyzer.analyze_distributions(df)
for col, stats in dist_analysis['distributions'].items():
    if abs(stats['skewness']) > 1.0:
        print(f"{col}: Skewness = {stats['skewness']:.2f}")

# Consistency checks
consistency = analyzer.check_consistency(df, target_column="target")
print(f"Duplicate Rows: {consistency['duplicate_percentage']:.2%}")
```

#### Custom Configuration

```python
from data import DataQualityConfig

config = DataQualityConfig(
    missing_threshold_warning=0.10,
    missing_threshold_critical=0.40,
    iqr_multiplier=2.0,
    z_score_threshold=3.5
)

analyzer = DataQualityAnalyzer(config)
```

#### Generate Reports

```python
from data import generate_quality_report_html

html = generate_quality_report_html(report)
with open("quality_report.html", "w") as f:
    f.write(html)
```

## Module 3: Schema Validator

### Features

- **Type Validation**: Integer, float, string, datetime, category, etc.
- **Range Validation**: Min/max values for numeric fields
- **Required Fields**: Enforce field presence
- **Allowed Values**: Validate against predefined sets
- **Custom Validators**: User-defined validation logic
- **Row-level Validation**: Cross-field validation rules

### Usage Examples

#### Define Schema

```python
from data import (
    DataSchema,
    FieldSchema,
    FieldType,
    create_numeric_field,
    create_string_field,
    create_categorical_field
)

# Manual schema definition
schema = DataSchema(fields={
    "age": FieldSchema(
        name="age",
        field_type=FieldType.INTEGER,
        required=True,
        nullable=False,
        min_value=0,
        max_value=120
    ),
    "email": FieldSchema(
        name="email",
        field_type=FieldType.EMAIL,
        required=True
    )
})

# Using helper functions
schema = DataSchema(fields={
    "price": create_numeric_field("price", required=True, min_value=0),
    "name": create_string_field("name", required=True, max_length=100),
    "category": create_categorical_field("category", {"A", "B", "C"})
})
```

#### Validate Data

```python
from data import SchemaValidator

validator = SchemaValidator(schema)
result = validator.validate(df)

if result.is_valid:
    print("Validation passed!")
else:
    print(f"Validation failed with {result.get_error_count()} errors")
    for error in result.errors:
        print(f"  - {error['type']}: {error['message']}")
```

#### Custom Validators

```python
def validate_email_domain(series):
    """Custom validator: email must be from company domain."""
    return series.str.endswith("@company.com")

def validate_age_salary_relationship(df):
    """Row-level validator: salary should increase with age."""
    return df['salary'] >= df['age'] * 1000

schema = DataSchema(fields={
    "email": FieldSchema(
        name="email",
        field_type=FieldType.EMAIL,
        custom_validators=[validate_email_domain]
    )
})

schema.add_row_validator(validate_age_salary_relationship)
```

#### Infer Schema

```python
from data import infer_schema_from_dataframe

# Automatically infer schema from data
schema = infer_schema_from_dataframe(df, sample_size=1000)

# Use inferred schema
validator = SchemaValidator(schema)
result = validator.validate(new_df)
```

## Module 4: EDA Engine

### Features

- **Univariate Analysis**: Distribution, statistics, outliers per feature
- **Bivariate Analysis**: Relationship with target variable
- **Correlation Analysis**: Detect multicollinearity
- **Drift Detection**: Temporal distribution changes
- **Leakage Detection**: Identify potential target leakage
- **Bias Detection**: Sampling and class imbalance
- **Regime Changes**: Detect shifts in data patterns

### Usage Examples

#### Comprehensive EDA

```python
from data import EDAEngine, generate_eda_summary

engine = EDAEngine()
report = engine.analyze(
    df,
    target_column="target",
    datetime_column="timestamp",
    categorical_columns=["category", "region"]
)

# Print summary
print(generate_eda_summary(report))

# Access specific analyses
print("\nDataset Overview:")
print(f"  Rows: {report.dataset_overview['n_rows']:,}")
print(f"  Missing: {report.dataset_overview['missing_pct']:.2%}")

print("\nWarnings:")
for warning in report.warnings:
    print(f"  ⚠ {warning}")

print("\nRecommendations:")
for rec in report.recommendations:
    print(f"  → {rec}")
```

#### Univariate Analysis

```python
for analysis in report.univariate_analysis:
    if analysis.missing_pct > 0.1:
        print(f"{analysis.column}:")
        print(f"  Missing: {analysis.missing_pct:.1%}")
        print(f"  Unique: {analysis.unique_pct:.1%}")
        
        if "mean" in analysis.stats:
            print(f"  Mean: {analysis.stats['mean']:.2f}")
            print(f"  Std: {analysis.stats['std']:.2f}")
```

#### Bivariate Analysis

```python
# Sort by importance
for analysis in report.bivariate_analysis[:10]:
    print(f"{analysis.feature}:")
    print(f"  Type: {analysis.relationship_type}")
    
    if analysis.correlation:
        print(f"  Correlation: {analysis.correlation:.3f}")
    
    if analysis.feature_importance:
        print(f"  Importance: {analysis.feature_importance:.3f}")
    
    if analysis.statistical_test['significant']:
        print(f"  Statistically significant (p={analysis.statistical_test['p_value']:.4f})")
```

#### Drift Detection

```python
if report.drift_analysis.get('drifted_features'):
    print("Drifted Features:")
    for feature in report.drift_analysis['drifted_features']:
        details = report.drift_analysis['drift_details'][feature]
        print(f"  {feature}: KS={details['ks_statistic']:.3f}, p={details['p_value']:.4f}")
```

#### Leakage Detection

```python
if report.leakage_risks.get('high_risk_features'):
    print("High Leakage Risk Features:")
    for feature in report.leakage_risks['high_risk_features']:
        scores = report.leakage_risks['leakage_scores'].get(feature, {})
        print(f"  {feature}: {scores}")
```

#### Custom Configuration

```python
from data import EDAConfig

config = EDAConfig(
    high_correlation_threshold=0.8,
    drift_pvalue_threshold=0.01,
    leakage_correlation_threshold=0.98,
    max_rows_for_analysis=50000
)

engine = EDAEngine(config)
```

## Integration Example

Complete workflow combining all modules:

```python
from data import (
    DataLoader,
    DataQualityAnalyzer,
    SchemaValidator,
    infer_schema_from_dataframe,
    EDAEngine,
    generate_eda_summary
)

# 1. Load data
loader = DataLoader()
df = loader.load_csv("data.csv")

# 2. Quality assessment
quality_analyzer = DataQualityAnalyzer()
quality_report = quality_analyzer.analyze(df, target_column="target")

if quality_report.overall_score < 0.7:
    print("⚠ Low quality score, review issues before proceeding")
    for issue in quality_report.issues[:5]:
        print(f"  - {issue['message']}")

# 3. Schema validation
schema = infer_schema_from_dataframe(df)
validator = SchemaValidator(schema)
validation_result = validator.validate(df)

if not validation_result.is_valid:
    print("⚠ Schema validation failed")
    for error in validation_result.errors[:5]:
        print(f"  - {error['message']}")

# 4. EDA
eda_engine = EDAEngine()
eda_report = eda_engine.analyze(df, target_column="target")

print("\n" + generate_eda_summary(eda_report))

# 5. Check for critical issues
if eda_report.leakage_risks.get('high_risk_features'):
    print("\n⚠ CRITICAL: Potential target leakage detected!")
    print("Review these features:", eda_report.leakage_risks['high_risk_features'])

# 6. Apply recommendations
print("\nRecommended Actions:")
for i, rec in enumerate(eda_report.recommendations, 1):
    print(f"{i}. {rec}")
```

## Best Practices

### Data Loading

1. Use chunked loading for files > 100MB
2. Close database connections when done
3. Use connection pooling for repeated queries
4. Enable type inference for better memory usage

### Quality Assessment

1. Always run quality checks before modeling
2. Set thresholds based on your domain
3. Investigate high missing rates (>30%)
4. Address outliers based on business context

### Schema Validation

1. Infer schema from clean sample data
2. Use strict mode for production pipelines
3. Implement custom validators for business rules
4. Version your schemas

### EDA

1. Analyze a representative sample
2. Always check for drift in production data
3. Review leakage risks before modeling
4. Document regime changes

## Logging

All modules use `loguru` for comprehensive logging:

```python
from loguru import logger

# Configure logging
logger.add("data_pipeline.log", rotation="500 MB")

# Modules will automatically log to this file
```

## Error Handling

All modules include robust error handling:

```python
try:
    df = loader.load_csv("data.csv")
except FileNotFoundError:
    logger.error("File not found")
except Exception as e:
    logger.error(f"Unexpected error: {e}")
```

## Performance Tips

1. **Sample large datasets**: Use `max_rows_for_analysis` in EDA
2. **Use chunking**: Process large files in chunks
3. **Connection pooling**: Reuse database connections
4. **Type optimization**: Enable type inference to reduce memory
5. **Parallel processing**: Use built-in parallelization where available

## Contributing

When extending these modules:

1. Add comprehensive docstrings
2. Include type hints
3. Write tests for new functionality
4. Update this documentation
5. Follow PEP 8 style guidelines

## License

Part of the Principal Data Science Decision Agent framework.
