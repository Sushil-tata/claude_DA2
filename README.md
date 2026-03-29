# Principal Data Science Decision Agent Platform

A production-grade, Databricks-native ML platform for end-to-end decision workflows from data ingestion to production decisions.

## Overview

This platform provides a **configuration-driven architecture** for building and deploying machine learning decision systems on Databricks. It handles:

- **Point-in-time safe feature engineering** with leakage prevention
- **Distributed training** with MLflow experiment tracking
- **Enhanced validation** with segment analysis and calibration
- **Decision output** to Delta Lake with full audit trail
- **End-to-end orchestration** via Databricks Workflows

## Architecture

### Core Components

1. **Orchestrator** - Routes use cases to appropriate pipelines
2. **Spark Feature Pipelines** - PySpark-based distributed feature engineering
   - Rolling window aggregations (7d, 30d, 90d)
   - Tag-based features with PCA dimensionality reduction
   - Liquidity ratio features
   - Point-in-time safe as-of joins
3. **MLflow Training Harness** - Model training with experiment tracking
4. **Validation Suite** - Segment evaluation and calibration analysis
5. **Decision Output** - Delta Lake tables with metadata and audit trail
6. **YAML Config Schema** - Configuration-driven use case definitions

### Technology Stack

- **Compute**: Databricks (PySpark)
- **Storage**: Delta Lake
- **Experiment Tracking**: MLflow
- **Orchestration**: Databricks Workflows
- **CI/CD**: GitHub Actions

## Quick Start

### Prerequisites

- Python 3.10+
- Databricks workspace (for production) or local PySpark (for development)
- MLflow (optional for local development)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd claude

# Install dependencies for local development (no Spark)
pip install -r requirements.txt

# For Databricks development
pip install -r requirements-databricks.txt
```

### Running Locally

#### 1. Run with Synthetic Data (No Databricks Required)

```bash
# Dry run to validate configuration
python jobs/run_usecase.py \
  --config conf/use_cases/income_estimation.yaml \
  --dry-run

# Run pipeline with synthetic data (local mode)
python jobs/run_usecase.py \
  --config conf/use_cases/income_estimation.yaml \
  --local
```

#### 2. Run with Spark (Requires PySpark)

```bash
# Install PySpark locally
pip install pyspark delta-spark

# Run pipeline
python jobs/run_usecase.py \
  --config conf/use_cases/income_estimation.yaml \
  --execution-date 2024-12-01
```

### Running on Databricks

#### Deploy to Databricks

```bash
# Set Databricks credentials
export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
export DATABRICKS_TOKEN="your-token"

# Upload code to Databricks
databricks workspace import_dir src/decision_agent /Workspace/decision_agent --overwrite
databricks workspace import_dir conf /Workspace/conf --overwrite
databricks workspace import jobs/run_usecase.py /Workspace/jobs/run_usecase --overwrite

# Create workflow
databricks jobs create --json-file databricks/workflows/decision_agent_workflow.yml
```

#### Run Workflow

```bash
# Trigger workflow manually
databricks jobs run-now --job-id <job-id>

# Monitor run
databricks runs list --job-id <job-id>
```

## Project Structure

```
.
├── conf/
│   └── use_cases/              # YAML configuration files
│       └── income_estimation.yaml
├── jobs/
│   └── run_usecase.py          # Main entry point
├── src/decision_agent/
│   ├── orchestrator/           # Use case routing
│   │   └── router.py
│   ├── data/                   # Data loading and splitting
│   │   ├── splits.py
│   │   ├── asof_join.py
│   │   └── synthetic_data.py
│   ├── features/               # Feature engineering
│   │   ├── windows.py          # Rolling window aggregations
│   │   ├── tags.py             # Tag frequency features
│   │   ├── tag_pca.py          # PCA dimensionality reduction
│   │   ├── liquidity.py        # Liquidity ratios
│   │   └── income_features.py  # Income feature pipeline
│   ├── training/               # Model training
│   │   └── training_harness.py
│   ├── validation/             # Model validation
│   │   ├── segment_eval.py
│   │   └── calibration_eval.py
│   ├── decisions/              # Decision output
│   │   └── output_writer.py
│   ├── config/                 # Configuration
│   │   └── config_loader.py
│   └── utils/                  # Utilities
│       ├── spark_utils.py
│       └── mlflow_utils.py
├── databricks/
│   └── workflows/              # Databricks workflow definitions
│       └── decision_agent_workflow.yml
├── tests/
│   └── unit/                   # Unit tests (no Spark)
│       ├── test_imports.py
│       ├── test_config.py
│       └── test_asof_join.py
├── schemas/
│   └── config_schemas/         # JSON schemas for validation
│       └── use_case_config.schema.json
└── .github/workflows/          # CI/CD
    └── ci-tests.yml
```

## Use Cases

### Income Estimation (MVP)

Predict individual income levels based on transaction patterns.

**Features:**
- Rolling window aggregations (7d, 30d, 90d transaction counts/sums)
- Tag frequency features (salary deposits, rent payments, etc.)
- Tag PCA for dimensionality reduction
- Liquidity ratios (income/expense, average balance)

**Model:** Gradient Boosting Regressor

**Validation:** Segment evaluation by income quartile, calibration curves

**Configuration:** `conf/use_cases/income_estimation.yaml`

**Run:**
```bash
python jobs/run_usecase.py --config conf/use_cases/income_estimation.yaml
```

### Adding New Use Cases

1. **Create YAML configuration** in `conf/use_cases/your_use_case.yaml`
2. **Implement pipeline function** in appropriate module
3. **Register in router** at `src/decision_agent/orchestrator/router.py`:
   ```python
   def your_use_case_pipeline(config, spark=None):
       # Implementation
       pass

   PIPELINE_REGISTRY = {
       "your_use_case": your_use_case_pipeline,
       # ...
   }
   ```
4. **Run:** `python jobs/run_usecase.py --config conf/use_cases/your_use_case.yaml`

## Configuration

Use cases are defined via YAML configuration files following the schema in `schemas/config_schemas/use_case_config.schema.json`.

### Example Configuration

```yaml
use_case_id: income_estimation
version: v1.0.0
description: Estimate individual income levels

data:
  use_synthetic: true
  train_end_date: "2024-09-30"
  val_end_date: "2024-11-30"

features:
  lookback_windows: [7, 30, 90]
  feature_list: [...]
  leakage_prevention: true

model:
  algorithm: gradient_boosting
  target_column: income_level
  hyperparameters:
    n_estimators: 100
    max_depth: 5

validation:
  segments:
    - dimension: income_quartile
      values: [Q1, Q2, Q3, Q4]

output:
  table_name: decision_agent.income_decisions
```

## MLflow Integration

All experiments are automatically tracked in MLflow:

### View Experiments

```bash
# Start MLflow UI (local)
mlflow ui

# Navigate to http://localhost:5000
```

### In Databricks

Navigate to **Machine Learning** → **Experiments** → `/decision_agent/experiments`

## Decision Output

Decisions are written to Delta Lake tables with full metadata:

### Schema

```
customer_id          STRING
predicted_value      DOUBLE
run_id              STRING
model_version       STRING
as_of_dt            DATE
use_case_id         STRING
created_timestamp   TIMESTAMP
```

### Query Decisions

```sql
-- View recent decisions
SELECT * FROM decision_agent.income_decisions
WHERE as_of_dt = '2024-12-01'
LIMIT 10;

-- Aggregate by model version
SELECT model_version, COUNT(*) as num_decisions
FROM decision_agent.income_decisions
GROUP BY model_version;

-- Time travel
SELECT * FROM decision_agent.income_decisions
TIMESTAMP AS OF '2024-11-01';
```

## Development

### Running Tests

```bash
# Run unit tests (no Spark required)
pytest tests/unit/ -v

# Run with coverage
pytest tests/unit/ --cov=src/decision_agent --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint
pylint src/decision_agent
flake8 src/ tests/
```

### CI/CD

GitHub Actions automatically runs:
- Unit tests (Python 3.10, 3.11)
- Code formatting checks (black, isort)
- Linting (flake8, pylint)
- Import validation

See `.github/workflows/ci-tests.yml`

## Key Features

### Point-in-Time Safe Feature Engineering

All features use **as-of joins** to prevent data leakage:

```python
from decision_agent.data.asof_join import PointInTimeJoiner

joiner = PointInTimeJoiner(spark)
result = joiner.as_of_join(
    left_df, right_df,
    entity_key="customer_id",
    left_timestamp="prediction_date",
    right_timestamp="feature_date"
)

# Validate no leakage
joiner.validate_no_leakage(features_df, labels_df, ...)
```

### Temporal Splits

Prevent leakage with strict temporal boundaries:

```python
from decision_agent.data.splits import create_temporal_splits

train, val, test = create_temporal_splits(
    df,
    date_col="transaction_timestamp",
    train_end="2024-09-30",
    val_end="2024-11-30"
)
```

### Rolling Window Features

Distributed rolling aggregations:

```python
from decision_agent.features.windows import compute_rolling_windows

df_with_features = compute_rolling_windows(
    df,
    entity_key="customer_id",
    timestamp_col="transaction_timestamp",
    value_col="transaction_amount",
    lookback_windows=[7, 30, 90],
    aggregations=["sum", "avg", "count"]
)
```

## Production Deployment

### Databricks Workflow

The platform uses Databricks Workflows for orchestration:

1. **Load Data** - Load or generate transaction data
2. **Compute Features** - Distributed feature engineering
3. **Train Model** - MLflow-tracked training
4. **Validate Model** - Segment and calibration validation
5. **Score Batch** - Predictions on test set
6. **Write Decisions** - Output to Delta Lake

Workflow configuration: `databricks/workflows/decision_agent_workflow.yml`

### Monitoring

- **MLflow Experiments** - Track all training runs, metrics, parameters
- **Delta Lake Audit** - Full history of decisions with time travel
- **Databricks Workflow UI** - Task execution status and logs

## Troubleshooting

### PySpark not available locally

```bash
# Install PySpark for local development
pip install pyspark delta-spark

# Or run in local mode (pandas only)
python jobs/run_usecase.py --config conf/use_cases/income_estimation.yaml --local
```

### MLflow connection issues

```bash
# Use local MLflow tracking
export MLFLOW_TRACKING_URI="file:./mlruns"

# Or disable MLflow
# Edit training_harness.py to handle mlflow=None gracefully (already implemented)
```

### Import errors

```bash
# Ensure src is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or install in development mode
pip install -e .
```

## Future Enhancements

- **Real-time streaming** feature pipelines (Structured Streaming)
- **Deep learning** support (PyTorch/TensorFlow on Databricks)
- **AutoML** integration (Databricks AutoML)
- **Advanced drift detection** and monitoring
- **A/B testing framework** for model comparison
- **Multi-model ensembles**
- **Online feature computation**

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Run tests: `pytest tests/unit/`
4. Format code: `black src/ tests/`
5. Commit changes: `git commit -m "Add your feature"`
6. Push to branch: `git push origin feature/your-feature`
7. Create Pull Request

## License

[Add license information]

## Contact

For questions or support, contact the Data Science team.

---

**Built with** ♥ **by the Data Science Team**
