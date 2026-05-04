# Models Package

Production-ready machine learning models for the Principal Data Science Decision Agent.

## Quick Start

```python
from src.models import LightGBMModel, AutoMLEngine

# Quick model training
model = LightGBMModel(config_path='config/model_config.yaml')
model.fit(X_train, y_train, X_val, y_val)
predictions = model.predict_proba(X_test)

# Or use AutoML
automl = AutoMLEngine(config_path='config/model_config.yaml')
best_model = automl.fit(X_train, y_train, X_val, y_val, 
                        models=['lightgbm', 'xgboost'], n_trials=50)
```

## What's Included

### 🌲 Tree-Based Models
- **LightGBMModel** - Fast gradient boosting
- **XGBoostModel** - Powerful gradient boosting
- **CatBoostModel** - Categorical feature handling
- **RandomForestModel** - Bootstrap aggregating

### 🧠 Neural Tabular Models
- **TabNetModel** - Attention-based neural network
- **TabPFNModel** - Pre-trained foundation model
- **NODEModel** - Neural decision ensembles
- **DeepGBMModel** - Deep gradient boosting

### 🎭 Ensemble Methods
- **WeightedAverageEnsemble** - Optimized weighted averaging
- **StackingEnsemble** - Meta-learning with CV
- **BlendingEnsemble** - Simple hold-out blending
- **SegmentWiseEnsemble** - Segment-specific models
- **HybridRuleMLEnsemble** - Rules + ML combination

### 🔍 Unsupervised Learning
- **ClusteringEngine** - KMeans, HDBSCAN, GMM, Spectral, Hierarchical
- **DimensionalityReduction** - PCA, t-SNE, UMAP
- **AutoencoderClustering** - Deep learning clustering

### 🤖 AutoML & Meta-Learning
- **BayesianOptimizer** - Efficient hyperparameter search
- **MultiObjectiveOptimizer** - Multi-metric optimization
- **AutoMLEngine** - Automated model selection

## Documentation

- **[Complete Documentation](../../docs/MODELS_DOCUMENTATION.md)** - Full API reference and usage
- **[Implementation Summary](../../docs/MODELS_IMPLEMENTATION_SUMMARY.md)** - Technical details
- **[Usage Examples](../../examples/models_usage_example.py)** - Code examples

## Features

✅ **Unified API** - Consistent interface across all models  
✅ **Configuration-Driven** - YAML-based hyperparameters  
✅ **Production-Ready** - Error handling, logging, persistence  
✅ **Type-Safe** - Full type hints  
✅ **Well-Tested** - Comprehensive test suite  
✅ **GPU Support** - CUDA detection and usage  
✅ **Graceful Degradation** - Handles missing dependencies  

## Installation

All dependencies are in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train a Single Model

```python
from src.models import LightGBMModel

model = LightGBMModel(config_path='config/model_config.yaml')
model.fit(X_train, y_train, X_val, y_val)

# Get predictions
predictions = model.predict_proba(X_test)

# Feature importance
importance = model.get_feature_importance()
```

### 2. Create an Ensemble

```python
from src.models import WeightedAverageEnsemble, LightGBMModel, XGBoostModel

# Train base models
lgb = LightGBMModel(config_path='config/model_config.yaml')
lgb.fit(X_train, y_train)

xgb = XGBoostModel(config_path='config/model_config.yaml')
xgb.fit(X_train, y_train)

# Create ensemble
ensemble = WeightedAverageEnsemble(models=[lgb, xgb])
ensemble.fit(X_val, y_val, n_trials=50)

predictions = ensemble.predict_proba(X_test)
```

### 3. Use AutoML

```python
from src.models import AutoMLEngine

automl = AutoMLEngine(config_path='config/model_config.yaml')
best_model = automl.fit(
    X_train, y_train, X_val, y_val,
    models=['lightgbm', 'xgboost', 'catboost'],
    n_trials=100
)

# Get leaderboard
print(automl.get_leaderboard())

# Make predictions
predictions = best_model.predict_proba(X_test)
```

### 4. Clustering

```python
from src.models import ClusteringEngine

engine = ClusteringEngine(method='kmeans')
labels = engine.fit_predict(X, optimal_k=True, k_range=(2, 10))

# Visualize
engine.visualize_clusters(X, method='umap')
```

## Testing

Run tests:

```bash
pytest tests/test_models.py -v
```

Run examples:

```bash
python examples/models_usage_example.py
```

## Configuration

Models are configured via `config/model_config.yaml`:

```yaml
lightgbm:
  default:
    learning_rate: 0.05
    num_leaves: 31
    n_estimators: 100
  search_space:
    learning_rate: [0.01, 0.05, 0.1]
    num_leaves: [15, 31, 63, 127]
```

## Model Selection Guide

| Use Case | Recommended Model |
|----------|------------------|
| Speed & Efficiency | LightGBMModel |
| Maximum Performance | XGBoostModel |
| Categorical Features | CatBoostModel |
| Interpretability | TabNetModel |
| Small Datasets | TabPFNModel |
| Quick Prototyping | AutoMLEngine |
| Production Ensemble | WeightedAverageEnsemble |
| Maximum Accuracy | StackingEnsemble |

## Performance Tips

1. **Use validation sets** for early stopping
2. **Start with AutoML** for baseline
3. **Ensemble top models** for production
4. **Monitor feature importance** for insights
5. **Use GPU** for neural models when available

## Support

- Check [documentation](../../docs/MODELS_DOCUMENTATION.md) for detailed info
- See [examples](../../examples/models_usage_example.py) for usage patterns
- Review [tests](../../tests/test_models.py) for edge cases

## License

Part of the Principal Data Science Decision Agent project.
