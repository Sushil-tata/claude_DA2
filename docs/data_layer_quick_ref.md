# Data Layer Quick Reference

## Quick Start

```python
# Import everything you need
from data import (
    DataLoader,
    DataQualityAnalyzer,
    SchemaValidator,
    EDAEngine,
    infer_schema_from_dataframe,
    create_numeric_field,
    DataSchema,
    FieldSchema,
    FieldType
)
```

## Common Tasks

### Load Data

```python
loader = DataLoader()

# CSV
df = loader.load_csv("file.csv")

# Excel
df = loader.load_excel("file.xlsx")

# Database
df = loader.load_sql("SELECT * FROM table", connection_string)

# Large files (chunked)
for chunk in loader.load_csv_chunked("large.csv", chunk_size=10000):
    process(chunk)
```

### Check Quality

```python
analyzer = DataQualityAnalyzer()
report = analyzer.analyze(df, target_column="target")

print(f"Quality Score: {report.overall_score:.2f}")
print(f"Issues: {len(report.issues)}")
print(f"Recommendations: {len(report.recommendations)}")
```

### Validate Schema

```python
# Infer schema
schema = infer_schema_from_dataframe(df)

# Or define manually
schema = DataSchema(fields={
    "age": create_numeric_field("age", required=True, min_value=0, max_value=120),
})

# Validate
validator = SchemaValidator(schema)
result = validator.validate(df)

if not result.is_valid:
    for error in result.errors:
        print(error['message'])
```

### Run EDA

```python
engine = EDAEngine()
report = engine.analyze(df, target_column="target", datetime_column="date")

# Print summary
from data import generate_eda_summary
print(generate_eda_summary(report))

# Check warnings
for warning in report.warnings:
    print(f"⚠ {warning}")

# Get recommendations
for rec in report.recommendations:
    print(f"→ {rec}")
```

## Module Cheat Sheet

### DataLoader Methods

| Method | Purpose |
|--------|---------|
| `load_csv(filepath)` | Load CSV file |
| `load_json(filepath)` | Load JSON file |
| `load_excel(filepath, sheet_name)` | Load Excel file |
| `load_sql(query, conn_str)` | Execute SQL query |
| `load_table(table, conn_str)` | Load entire table |
| `load_csv_chunked(filepath, chunk_size)` | Load CSV in chunks |
| `get_table_info(conn_str, table)` | Get table metadata |
| `list_tables(conn_str)` | List all tables |

### DataQualityAnalyzer Methods

| Method | Purpose |
|--------|---------|
| `analyze(df)` | Full quality analysis |
| `analyze_missing_values(df)` | Missing value analysis |
| `detect_outliers(df)` | Outlier detection |
| `analyze_distributions(df)` | Distribution analysis |
| `check_consistency(df)` | Consistency checks |

### SchemaValidator Methods

| Method | Purpose |
|--------|---------|
| `validate(df)` | Validate DataFrame |
| `validate_field(series, schema)` | Validate single field |

### EDAEngine Methods

| Method | Purpose |
|--------|---------|
| `analyze(df, target, datetime)` | Full EDA |

## Field Types

- `FieldType.INTEGER` - Integer numbers
- `FieldType.FLOAT` - Floating point numbers
- `FieldType.STRING` - Text strings
- `FieldType.BOOLEAN` - True/False
- `FieldType.DATETIME` - Date and time
- `FieldType.CATEGORY` - Categorical values
- `FieldType.EMAIL` - Email addresses
- `FieldType.URL` - Web URLs

## Helper Functions

```python
# Schema helpers
create_numeric_field(name, required, min_value, max_value)
create_string_field(name, required, min_length, max_length)
create_categorical_field(name, categories, required)
create_datetime_field(name, required, min_value, max_value)

# Utilities
detect_delimiter(filepath)  # Auto-detect CSV delimiter
get_file_info(filepath)     # Get file metadata
generate_quality_report_html(report)  # HTML report
generate_eda_summary(report)  # Text summary
infer_schema_from_dataframe(df)  # Auto schema
```

## Configuration Objects

### DataLoaderConfig

```python
config = DataLoaderConfig(
    chunk_size=10000,
    max_retries=3,
    pool_size=5,
    encoding="utf-8"
)
```

### DataQualityConfig

```python
config = DataQualityConfig(
    missing_threshold_warning=0.05,
    missing_threshold_critical=0.30,
    iqr_multiplier=1.5,
    z_score_threshold=3.0
)
```

### EDAConfig

```python
config = EDAConfig(
    high_correlation_threshold=0.7,
    drift_pvalue_threshold=0.05,
    leakage_correlation_threshold=0.95,
    max_rows_for_analysis=100000
)
```

## Report Objects

### QualityReport

```python
report.overall_score          # 0-1 quality score
report.dimension_scores       # Dict of dimension scores
report.missing_analysis       # Missing value details
report.outlier_analysis       # Outlier details
report.distribution_analysis  # Distribution details
report.consistency_analysis   # Consistency details
report.issues                 # List of issues
report.recommendations        # List of recommendations
```

### ValidationResult

```python
result.is_valid              # True/False
result.errors                # List of errors
result.warnings              # List of warnings
result.field_results         # Per-field results
result.summary               # Summary statistics
result.get_error_count()     # Error count
result.get_warning_count()   # Warning count
```

### EDAReport

```python
report.dataset_overview       # Basic stats
report.univariate_analysis    # Per-feature analysis
report.bivariate_analysis     # Target relationships
report.correlation_matrix     # Correlation matrix
report.drift_analysis         # Temporal drift
report.leakage_risks         # Leakage detection
report.bias_analysis         # Bias detection
report.regime_changes        # Regime changes
report.recommendations       # Action items
report.warnings              # Important warnings
```

## Common Patterns

### Full Pipeline

```python
# 1. Load
loader = DataLoader()
df = loader.load_csv("data.csv")

# 2. Quality
analyzer = DataQualityAnalyzer()
quality = analyzer.analyze(df)

# 3. Validate
schema = infer_schema_from_dataframe(df)
validator = SchemaValidator(schema)
validation = validator.validate(df)

# 4. EDA
engine = EDAEngine()
eda = engine.analyze(df, target_column="target")

# 5. Review
if quality.overall_score < 0.7:
    print("Low quality!")
if not validation.is_valid:
    print("Schema issues!")
if eda.leakage_risks['high_risk_features']:
    print("Leakage detected!")
```

### Database Workflow

```python
# Connect
loader = DataLoader()
conn_str = "postgresql://user:pass@host/db"

# Explore
tables = loader.list_tables(conn_str)
info = loader.get_table_info(conn_str, "users")

# Load
df = loader.load_table("users", conn_str)

# Clean up
loader.close_all_connections()
```

### Large File Processing

```python
loader = DataLoader()
results = []

for chunk in loader.load_csv_chunked("huge.csv", chunk_size=50000):
    # Process chunk
    processed = process(chunk)
    results.append(processed)

# Combine
final_df = pd.concat(results, ignore_index=True)
```

## Error Handling

```python
from loguru import logger

try:
    df = loader.load_csv("data.csv")
except FileNotFoundError:
    logger.error("File not found")
except Exception as e:
    logger.exception("Failed to load data")
```

## Performance Tips

1. **Use chunking for large files** (>100MB)
2. **Enable type inference** for memory savings
3. **Sample for EDA** on huge datasets
4. **Close connections** when done
5. **Reuse loaders/analyzers** for multiple operations

## Integration with Other Modules

```python
# With feature engineering
from features import FeatureEngineer

eda_report = engine.analyze(df, target_column="target")
engineer = FeatureEngineer()

# Apply recommendations from EDA
for rec in eda_report.recommendations:
    if "transformation" in rec.lower():
        engineer.add_transformation(...)

# With validation
from validation import ModelValidator

quality_report = analyzer.analyze(df)
if quality_report.overall_score > 0.8:
    # Data is good, proceed to modeling
    validator = ModelValidator()
```

---

**Note**: This is a quick reference. See `docs/data_layer_guide.md` for detailed documentation.
