# Decision Agent Platform - Principal Data Science Agent

A production-grade, Databricks-native ML platform for end-to-end decision workflows.

**Version:** 1.0.0-MVP  
**Status:** Phase 1 Complete - Runnable Skeleton  
**Branch:** copilot/create-principal-data-science-agent

---

## Overview

The Decision Agent Platform is a configuration-driven ML platform designed for Databricks that handles:
- Point-in-time safe feature engineering
- MLflow-integrated model training
- Segment-based validation and calibration
- Production decision output to Delta Lake
- Complete audit trail and lineage tracking

### Key Principles

- **Databricks-Native**: All processing on Databricks (PySpark, Delta Lake, MLflow)
- **Config-Driven**: YAML configuration with JSON schema validation
- **Point-in-Time Safe**: No look-ahead bias, as-of joins, temporal validation
- **Production-Ready**: ACID transactions, versioning, audit logs

---

## Current Status (Phase 1 Complete)

### ✅ Delivered Components

1. **Directory Structure** - Complete skeleton for all modules
2. **Configuration System**:
   - JSON schema for config validation (`schemas/config_schemas/use_case_config.schema.json`)
   - MVP use case config (`conf/use_cases/income_estimation.yaml`)
   - Config loader with validation (`src/decision_agent/utils/config_loader.py`)

3. **Orchestration**:
   - Use case router with pipeline registry (`src/decision_agent/orchestrator/router.py`)
   - Main entry point (`jobs/run_usecase.py`)

4. **Data Generation**:
   - Synthetic transaction data generator (`src/decision_agent/data/synthetic_data.py`)

5. **Testing**:
   - Import verification ✓
   - Dry-run execution ✓
   - Full pipeline routing ✓

---

## Quick Start

### Prerequisites

- Python 3.10+
- pip package manager

### Installation

```bash
# Clone repository
git clone <repository-url>
cd claude

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### List Available Use Cases

```bash
python3 jobs/run_usecase.py --list-use-cases
```

#### Validate Configuration (Dry Run)

```bash
python3 jobs/run_usecase.py \
  --config conf/use_cases/income_estimation.yaml \
  --dry-run
```

#### Execute Use Case

```bash
python3 jobs/run_usecase.py \
  --config conf/use_cases/income_estimation.yaml
```

---

## Project Structure

```
claude/
├── jobs/
│   └── run_usecase.py              # Main entry point
├── conf/
│   └── use_cases/
│       └── income_estimation.yaml  # MVP use case config
├── src/decision_agent/
│   ├── orchestrator/
│   │   └── router.py               # Use case router
│   ├── data/
│   │   └── synthetic_data.py       # Synthetic data generator
│   ├── utils/
│   │   └── config_loader.py        # Config validation
│   ├── features/                   # Phase 3 (to be implemented)
│   ├── training/                   # Phase 4 (to be implemented)
│   ├── validation/                 # Phase 5 (to be implemented)
│   └── decisions/                  # Phase 6 (to be implemented)
├── tests/
│   ├── unit/                       # Unit tests (no Spark)
│   └── integration/                # Integration tests (Databricks)
├── databricks/
│   └── workflows/                  # Databricks workflow DAGs
├── schemas/
│   └── config_schemas/
│       └── use_case_config.schema.json
└── requirements.txt
```

---

## MVP Use Case: Income Estimation

**Goal**: Estimate individual income levels from transaction patterns

**Features**:
- Rolling window aggregations (7d, 30d, 90d)
- Transaction category features (salary, rent, groceries)
- Tag PCA dimensionality reduction
- Liquidity ratios (income/expense, average balance)

**Model**: Gradient Boosting Regressor

**Output**: Decision table with predicted income + confidence bands

---

## Implementation Roadmap

### ✅ Phase 1: Runnable Skeleton (Complete)
- Directory structure
- Config system with JSON schema validation
- Use case router
- Synthetic data generator
- Main entry point with dry-run capability

### 🔜 Phase 2: Data Layer (Next)
- Temporal train/val/test splits
- Point-in-time safe as-of joins
- Leakage prevention guards

### 🔜 Phase 3: Feature Engineering
- Rolling window aggregations
- Transaction category encoding
- Tag PCA
- Liquidity features

### 🔜 Phase 4: Training Harness
- MLflow integration
- Spark to pandas boundary
- Model logging and registry

### 🔜 Phase 5: Validation
- Segment-based validation
- Calibration evaluation
- Quality gates

### 🔜 Phase 6: Decision Output
- Delta Lake writer
- Audit trail
- Metadata tracking

### 🔜 Phase 7: End-to-End Integration
- Wire all components
- Databricks workflow DAG
- CI/CD pipeline

---

## Configuration Format

### Example: Income Estimation Use Case

```yaml
use_case_id: income_estimation
version: v1.0.0
description: Estimate income levels from transaction patterns

features:
  snapshot_timestamp: "2024-01-31"
  lookback_windows: [7, 30, 90]
  feature_list:
    - transaction_count_7d
    - transaction_sum_30d
    - salary_deposit_frequency
    # ... more features
  leakage_prevention: true

model:
  algorithm: gradient_boosting
  hyperparameters:
    n_estimators: 100
    max_depth: 5
    learning_rate: 0.1
  target_variable: income_level

validation:
  segments:
    - dimension: income_quartile
      values: [Q1, Q2, Q3, Q4]
  quality_gates:
    min_r2: 0.6

output:
  table_name: decision_agent.income_decisions
```

---

## Testing

### Import Verification

```bash
python3 -c "
import sys
sys.path.insert(0, 'src')
from decision_agent.utils.config_loader import ConfigLoader
from decision_agent.orchestrator.router import route_use_case
from decision_agent.data.synthetic_data import SyntheticDataGenerator
print('✓ All imports successful')
"
```

### Generate Synthetic Data

```bash
python3 -c "
import sys
sys.path.insert(0, 'src')
from decision_agent.data.synthetic_data import SyntheticDataGenerator
gen = SyntheticDataGenerator()
df = gen.generate_transactions(n_customers=10, months_history=6)
print(f'Generated {len(df)} transactions')
print(df.head())
"
```

---

## Architecture Decisions

### 1. Databricks-Native
All data processing, training, and inference runs on Databricks. GitHub Actions limited to CI/CD only.

### 2. Config-Driven
All use cases defined in YAML with strict JSON schema validation. No hardcoded parameters.

### 3. Delta Lake for Storage
Features, decisions, and audit logs stored in Delta Lake for ACID transactions and time travel.

### 4. Point-in-Time Safety
Temporal validators and as-of joins prevent look-ahead bias in all feature computations.

### 5. MLflow for Model Lifecycle
Experiment tracking, model registry, and feature lineage managed through MLflow.

---

## Next Steps

1. **Implement Phase 2**: Temporal splits and as-of joins
2. **Implement Phase 3**: Feature engineering modules
3. **Add unit tests**: Config loader, router logic
4. **Create CI/CD pipeline**: GitHub Actions for tests and deployment

---

## Support

For questions or issues, contact the Decision Agent Team.

---

**Built for production ML at scale on Databricks**
