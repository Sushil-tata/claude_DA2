# Models Layer Documentation

## Overview

The models layer provides a comprehensive suite of machine learning models, ensemble methods, unsupervised learning techniques, and AutoML capabilities for the Principal Data Science Decision Agent.

## Table of Contents

1. [Tree-Based Models](#tree-based-models)
2. [Neural Tabular Models](#neural-tabular-models)
3. [Ensemble Methods](#ensemble-methods)
4. [Unsupervised Learning](#unsupervised-learning)
5. [AutoML and Meta-Learning](#automl-and-meta-learning)
6. [Configuration](#configuration)
7. [Usage Examples](#usage-examples)

---

## Tree-Based Models

### LightGBMModel

Gradient boosting framework optimized for speed and efficiency.

**Features:**
- Fast training and prediction
- Memory efficient
- Built-in categorical feature support
- Multiple importance types (gain, split)
- SHAP value calculation

**Usage:**
```python
from src.models import LightGBMModel

model = LightGBMModel(config_path='config/model_config.yaml')
model.fit(X_train, y_train, X_val, y_val)
predictions = model.predict_proba(X_test)
importance = model.get_feature_importance(importance_type='gain')
```

### XGBoostModel

Powerful gradient boosting implementation with regularization.

**Features:**
- Strong performance on structured data
- Built-in cross-validation
- Tree pruning with max_depth
- Support for custom objectives

**Usage:**
```python
from src.models import XGBoostModel

model = XGBoostModel(config_path='config/model_config.yaml')
model.fit(X_train, y_train, X_val, y_val)
predictions = model.predict_proba(X_test)
```

### CatBoostModel

Gradient boosting with native categorical feature handling.

**Features:**
- Automatic categorical encoding
- Robust to missing values
- Ordered boosting for better generalization
- GPU support

**Usage:**
```python
from src.models import CatBoostModel

model = CatBoostModel(config_path='config/model_config.yaml')
model.fit(X_train, y_train, X_val, y_val, cat_features=['category_col'])
predictions = model.predict_proba(X_test)
```

### RandomForestModel

Ensemble of decision trees with bootstrap aggregating.

**Features:**
- Robust to overfitting
- Out-of-bag error estimation
- Parallel training
- Feature importance

**Usage:**
```python
from src.models import RandomForestModel

model = RandomForestModel(config_path='config/model_config.yaml')
model.fit(X_train, y_train)
predictions = model.predict_proba(X_test)
```

---

## Neural Tabular Models

### TabNetModel

Attention-based neural network for tabular data.

**Features:**
- Sequential attention for feature selection
- Interpretable through attention masks
- Self-supervised pre-training support
- Sparse feature selection

**Usage:**
```python
from src.models import TabNetModel

model = TabNetModel(config_path='config/model_config.yaml')
model.fit(X_train, y_train, X_val, y_val, max_epochs=100, patience=20)
predictions = model.predict_proba(X_test)
importance = model.get_feature_importance()
```

### TabPFNModel

Prior-data fitted network for small tabular datasets.

**Features:**
- Pre-trained on synthetic data
- No training required
- Works best on < 10k samples, < 100 features
- Fast inference

**Usage:**
```python
from src.models import TabPFNModel

model = TabPFNModel()
model.fit(X_train, y_train)  # Just fits data, no actual training
predictions = model.predict_proba(X_test)
```

**Note:** TabPFN is designed for small datasets. For larger datasets, it automatically falls back to gradient boosting.

---

## Ensemble Methods

### WeightedAverageEnsemble

Combines multiple models with optimized weights.

**Features:**
- Automatic weight optimization via Optuna
- Supports any number of base models
- Minimizes overfitting risk

**Usage:**
```python
from src.models import WeightedAverageEnsemble

ensemble = WeightedAverageEnsemble(models=[model1, model2, model3])
ensemble.fit(X_val, y_val, n_trials=100)
predictions = ensemble.predict_proba(X_test)
print(f"Optimal weights: {ensemble.weights}")
```

### StackingEnsemble

Meta-learning with cross-validated base predictions.

**Features:**
- Out-of-fold predictions prevent overfitting
- Customizable meta-learner
- Supports multiple base models
- Cross-validation at training time

**Usage:**
```python
from src.models import StackingEnsemble

# Base models should NOT be fitted
ensemble = StackingEnsemble(
    models=[lgb_model, xgb_model, cat_model],
    meta_learner='logistic',
    cv=5
)
ensemble.fit(X_train, y_train)
predictions = ensemble.predict_proba(X_test)
```

### BlendingEnsemble

Simpler alternative to stacking using hold-out validation.

**Features:**
- Faster than stacking
- Less complex
- Requires separate validation set

**Usage:**
```python
from src.models import BlendingEnsemble

ensemble = BlendingEnsemble(models=[model1, model2])
ensemble.fit(X_train, y_train, X_val, y_val)
predictions = ensemble.predict_proba(X_test)
```

### SegmentWiseEnsemble

Different models for different data segments.

**Features:**
- Specialized models per segment
- Automatic segment handling
- Improved performance on heterogeneous data

**Usage:**
```python
from src.models import SegmentWiseEnsemble

ensemble = SegmentWiseEnsemble(segment_col='customer_type')
ensemble.fit(
    X_train, y_train,
    models={'retail': retail_model, 'corporate': corporate_model}
)
predictions = ensemble.predict_proba(X_test)
```

### HybridRuleMLEnsemble

Combines business rules with ML predictions.

**Features:**
- Rule-based for high-confidence cases
- ML for uncertain cases
- Maintains interpretability

**Usage:**
```python
from src.models import HybridRuleMLEnsemble

def rule_function(X):
    high_risk = X['amount'] > 10000
    return high_risk.astype(int), high_risk

ensemble = HybridRuleMLEnsemble(
    ml_model=ml_model,
    rule_function=rule_function
)
ensemble.fit(X_train, y_train)
predictions = ensemble.predict_proba(X_test)
```

---

## Unsupervised Learning

### ClusteringEngine

Unified interface for multiple clustering algorithms.

**Supported Methods:**
- KMeans (with optimal k selection)
- HDBSCAN (density-based)
- Gaussian Mixture Models
- Spectral Clustering
- Hierarchical Clustering

**Usage:**
```python
from src.models import ClusteringEngine

# Automatic optimal k selection
engine = ClusteringEngine(method='kmeans')
labels = engine.fit_predict(X, optimal_k=True, k_range=(2, 10))

# Visualize clusters
engine.visualize_clusters(X, method='umap')
engine.plot_elbow_curve()
```

### DimensionalityReduction

Reduce feature dimensions while preserving information.

**Supported Methods:**
- PCA (Principal Component Analysis)
- t-SNE (t-distributed Stochastic Neighbor Embedding)
- UMAP (Uniform Manifold Approximation and Projection)

**Usage:**
```python
from src.models import DimensionalityReduction

# PCA
reducer = DimensionalityReduction(method='pca', n_components=10)
X_reduced = reducer.fit_transform(X)
reducer.plot_explained_variance()

# UMAP for visualization
umap_reducer = DimensionalityReduction(method='umap', n_components=2)
X_2d = umap_reducer.fit_transform(X)
```

### AutoencoderClustering

Deep learning-based clustering with representation learning.

**Features:**
- Learns compressed representations
- Applies clustering on embeddings
- Customizable architecture

**Usage:**
```python
from src.models import AutoencoderClustering

model = AutoencoderClustering(
    encoding_dim=10,
    n_clusters=5,
    hidden_dims=[64, 32]
)
model.fit(X, epochs=100)
labels = model.predict(X_test)
embeddings = model.get_embeddings(X_test)
```

---

## AutoML and Meta-Learning

### BayesianOptimizer

Hyperparameter optimization using Bayesian methods.

**Features:**
- Efficient search using TPE sampler
- Automatic early stopping
- Support for multiple parameter types
- Trial history tracking

**Usage:**
```python
from src.models import BayesianOptimizer

search_space = {
    'learning_rate': ('float', 0.01, 0.3),
    'max_depth': ('int', 3, 10),
    'n_estimators': ('categorical', [100, 200, 500])
}

optimizer = BayesianOptimizer(metric='auc', direction='maximize')
best_params, best_score = optimizer.optimize(
    objective_fn, search_space, n_trials=100
)
```

### MultiObjectiveOptimizer

Optimize multiple metrics simultaneously.

**Features:**
- Balance multiple objectives
- Weighted scoring
- Pareto front analysis

**Usage:**
```python
from src.models import MultiObjectiveOptimizer

objectives = {
    'auc': {'weight': 0.4, 'direction': 'maximize'},
    'stability': {'weight': 0.3, 'direction': 'maximize'},
    'calibration': {'weight': 0.2, 'direction': 'minimize'},
    'business_value': {'weight': 0.1, 'direction': 'maximize'}
}

optimizer = MultiObjectiveOptimizer(objectives)
best_params = optimizer.optimize(objective_fn, search_space, n_trials=100)
pareto_front = optimizer.get_pareto_front(top_k=10)
```

### AutoMLEngine

Automated model selection and hyperparameter tuning.

**Features:**
- Tries multiple model types
- Automatic hyperparameter tuning
- Cross-validation
- Model comparison and selection
- Leaderboard tracking

**Usage:**
```python
from src.models import AutoMLEngine

automl = AutoMLEngine(
    config_path='config/model_config.yaml',
    task='classification',
    metric='auc'
)

best_model = automl.fit(
    X_train, y_train, X_val, y_val,
    models=['lightgbm', 'xgboost', 'catboost', 'random_forest'],
    n_trials=50
)

print(f"Best model: {automl.best_model_name}")
print(f"Best score: {automl.best_score:.4f}")

# View all results
leaderboard = automl.get_leaderboard()
print(leaderboard)

# Save results
automl.save_results('models/automl_results.json')
```

---

## Configuration

### model_config.yaml

All models can be configured via `config/model_config.yaml`:

```yaml
lightgbm:
  default:
    boosting_type: 'gbdt'
    objective: 'binary'
    metric: 'auc'
    num_leaves: 31
    learning_rate: 0.05
    n_estimators: 100
  
  search_space:
    num_leaves: [15, 31, 63, 127]
    learning_rate: [0.01, 0.05, 0.1]
    max_depth: [5, 7, 10, -1]

xgboost:
  default:
    objective: 'binary:logistic'
    eval_metric: 'auc'
    max_depth: 6
    learning_rate: 0.05
```

---

## Usage Examples

### Complete Pipeline Example

```python
from src.models import (
    LightGBMModel, XGBoostModel, CatBoostModel,
    WeightedAverageEnsemble, AutoMLEngine
)

# 1. Train individual models
lgb = LightGBMModel(config_path='config/model_config.yaml')
lgb.fit(X_train, y_train, X_val, y_val)

xgb = XGBoostModel(config_path='config/model_config.yaml')
xgb.fit(X_train, y_train, X_val, y_val)

# 2. Create ensemble
ensemble = WeightedAverageEnsemble(models=[lgb, xgb])
ensemble.fit(X_val, y_val, n_trials=50)

# 3. Make predictions
predictions = ensemble.predict_proba(X_test)

# 4. Or use AutoML
automl = AutoMLEngine(config_path='config/model_config.yaml')
best_model = automl.fit(
    X_train, y_train, X_val, y_val,
    models=['lightgbm', 'xgboost', 'catboost'],
    n_trials=100
)
```

### Model Persistence

```python
# Save model
model.save('models/my_model.pkl')

# Load model
loaded_model = LightGBMModel()
loaded_model.load('models/my_model.pkl')

# Use loaded model
predictions = loaded_model.predict_proba(X_test)
```

### Feature Importance

```python
# Get feature importance
importance = model.get_feature_importance(importance_type='gain')
print(importance.head(10))

# SHAP values (if available)
shap_values = model.get_shap_values(X_test)
```

---

## Best Practices

1. **Model Selection:**
   - Use AutoML for quick prototyping
   - LightGBM for speed and memory efficiency
   - XGBoost for best performance on structured data
   - CatBoost for categorical features
   - TabNet for interpretability on tabular data

2. **Ensemble Methods:**
   - Start with WeightedAverageEnsemble (simple, effective)
   - Use StackingEnsemble for maximum performance
   - Consider SegmentWiseEnsemble for heterogeneous data
   - HybridRuleMLEnsemble when business rules exist

3. **Hyperparameter Tuning:**
   - Use BayesianOptimizer for efficient search
   - Start with 50-100 trials
   - Use MultiObjectiveOptimizer for production systems

4. **Cross-Validation:**
   - Always use validation set for early stopping
   - 5-fold CV is a good default
   - Stratified CV for imbalanced data

5. **Feature Importance:**
   - Compare gain, split, and SHAP importances
   - Use SHAP for model explanations
   - Monitor feature importance drift

---

## Dependencies

Required packages (automatically installed with requirements.txt):
- lightgbm>=4.0.0
- xgboost>=2.0.0
- catboost>=1.2.0
- scikit-learn>=1.3.0
- pytorch-tabnet>=4.0
- optuna>=3.3.0
- torch>=2.0.0
- umap-learn>=0.5.4
- hdbscan>=0.8.33

Optional:
- shap>=0.42.0 (for SHAP values)
- tabpfn (for TabPFN model)

---

## Testing

Run the test suite:

```bash
pytest tests/test_models.py -v
```

Run example script:

```bash
python examples/models_usage_example.py
```

---

## Support

For issues or questions:
1. Check this documentation
2. Review example scripts
3. Check test cases for usage patterns
4. Consult model-specific documentation
