# Data Layer Module

Production-ready data handling layer for the Principal Data Science Decision Agent.

## Overview

The data layer provides comprehensive functionality for:

- **Loading data** from multiple sources (CSV, JSON, Excel, SQL)
- **Assessing quality** with automated checks and scoring
- **Validating schemas** with type checking and business rules
- **Exploratory analysis** with drift, leakage, and bias detection

## Modules

### 1. Data Loader (`data_loader.py`)
- Multi-format support (CSV, JSON, Excel, SQL)
- Connection pooling for databases
- Chunked loading for large files
- Automatic type inference
- 693 lines of production code

### 2. Data Quality (`data_quality.py`)
- Missing value analysis
- Outlier detection (IQR, Z-score, Isolation Forest)
- Distribution analysis (skewness, kurtosis, normality)
- Consistency checks
- Quality scoring (0-1 scale)
- 747 lines of production code

### 3. Schema Validator (`schema_validator.py`)
- Schema definition and inference
- Type validation
- Range and length validation
- Custom validation rules
- Row-level validation
- 766 lines of production code

### 4. EDA Engine (`eda_engine.py`)
- Automated univariate analysis
- Bivariate analysis with target
- Correlation and multicollinearity detection
- Temporal drift detection
- Target leakage identification
- Sampling bias detection
- Regime change detection
- 955 lines of production code

## Total: 3,161 lines of production code

## Quick Start

```python
from data import (
    DataLoader,
    DataQualityAnalyzer,
    SchemaValidator,
    EDAEngine,
)

# Load data
loader = DataLoader()
df = loader.load_csv("data.csv")

# Check quality
analyzer = DataQualityAnalyzer()
quality_report = analyzer.analyze(df)
print(f"Quality: {quality_report.overall_score:.2f}")

# Validate schema
schema = infer_schema_from_dataframe(df)
validator = SchemaValidator(schema)
validation = validator.validate(df)

# Run EDA
engine = EDAEngine()
eda_report = engine.analyze(df, target_column="target")
print(f"Warnings: {len(eda_report.warnings)}")
```

## Documentation

- **Full Guide**: `docs/data_layer_guide.md` (546 lines)
- **Quick Reference**: `docs/data_layer_quick_ref.md` (344 lines)

## Features

### Data Loader
✓ CSV, JSON, Excel, SQL support
✓ Connection pooling
✓ Chunked loading
✓ Type inference
✓ Error handling

### Data Quality
✓ Missing value patterns
✓ Outlier detection
✓ Distribution analysis
✓ Duplicate detection
✓ Quality scoring

### Schema Validator
✓ Type checking
✓ Range validation
✓ Required fields
✓ Allowed values
✓ Custom validators

### EDA Engine
✓ Univariate analysis
✓ Bivariate analysis
✓ Drift detection
✓ Leakage detection
✓ Bias detection
✓ Regime changes

## Best Practices

1. **Always run quality checks** before modeling
2. **Validate schemas** in production pipelines
3. **Check for drift** in production data
4. **Review leakage risks** before training
5. **Use chunking** for large files

## Dependencies

Core requirements (in `requirements.txt`):
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- scipy >= 1.11.0
- sqlalchemy >= 2.0.0
- loguru >= 0.7.0
- openpyxl >= 3.1.0

## Testing

```bash
python tests/test_data_layer.py
```

## Integration

Works seamlessly with other agent modules:
- Feature engineering
- Model validation
- Production deployment
- Simulation framework

## Code Quality

- ✓ Comprehensive docstrings
- ✓ Type hints throughout
- ✓ PEP 8 compliant
- ✓ Extensive error handling
- ✓ Detailed logging
- ✓ Production-ready

## Architecture

```
src/data/
├── __init__.py          # Public API exports
├── data_loader.py       # Data loading (693 lines)
├── data_quality.py      # Quality assessment (747 lines)
├── schema_validator.py  # Schema validation (766 lines)
└── eda_engine.py        # EDA framework (955 lines)

docs/
├── data_layer_guide.md      # Full documentation (546 lines)
└── data_layer_quick_ref.md  # Quick reference (344 lines)

tests/
└── test_data_layer.py   # Module tests
```

## Contributing

When extending:
1. Add comprehensive docstrings
2. Include type hints
3. Write tests
4. Update documentation
5. Follow PEP 8

## License

Part of the Principal Data Science Decision Agent framework.
