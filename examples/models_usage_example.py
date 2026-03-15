"""
Example usage of the models layer.

This script demonstrates how to use all the model types, ensembles,
and AutoML functionality.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

from src.models import (
    # Tree models
    LightGBMModel, XGBoostModel, CatBoostModel, RandomForestModel,
    # Neural models
    TabNetModel,
    # Ensembles
    WeightedAverageEnsemble, StackingEnsemble,
    # Unsupervised
    ClusteringEngine, DimensionalityReduction,
    # AutoML
    AutoMLEngine
)


def example_tree_models():
    """Example: Using tree-based models."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Tree-Based Models")
    print("="*60)
    
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Convert to DataFrame
    X_train = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(20)])
    X_val = pd.DataFrame(X_val, columns=[f'feature_{i}' for i in range(20)])
    X_test = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(20)])
    
    # Train LightGBM
    print("\nTraining LightGBM...")
    lgb = LightGBMModel(config_path='config/model_config.yaml')
    lgb.fit(X_train, y_train, X_val, y_val)
    lgb_preds = lgb.predict_proba(X_test)[:, 1]
    lgb_auc = roc_auc_score(y_test, lgb_preds)
    print(f"LightGBM AUC: {lgb_auc:.4f}")
    
    # Feature importance
    importance = lgb.get_feature_importance()
    print(f"\nTop 5 features:")
    print(importance.head())
    
    # Train XGBoost
    print("\nTraining XGBoost...")
    xgb = XGBoostModel(config_path='config/model_config.yaml')
    xgb.fit(X_train, y_train, X_val, y_val)
    xgb_preds = xgb.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_test, xgb_preds)
    print(f"XGBoost AUC: {xgb_auc:.4f}")
    
    return lgb, xgb, X_train, y_train, X_test, y_test, X_val, y_val


def example_neural_tabular(X_train, y_train, X_val, y_val, X_test, y_test):
    """Example: Using neural tabular models."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Neural Tabular Models")
    print("="*60)
    
    # Train TabNet
    print("\nTraining TabNet...")
    tabnet = TabNetModel(config_path='config/model_config.yaml')
    tabnet.fit(X_train, y_train, X_val, y_val, max_epochs=20)
    tabnet_preds = tabnet.predict_proba(X_test)[:, 1]
    tabnet_auc = roc_auc_score(y_test, tabnet_preds)
    print(f"TabNet AUC: {tabnet_auc:.4f}")
    
    # Feature importance
    importance = tabnet.get_feature_importance()
    print(f"\nTop 5 features:")
    print(importance.head())
    
    return tabnet


def example_ensembles(lgb, xgb, X_val, y_val, X_test, y_test):
    """Example: Using ensemble methods."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Ensemble Methods")
    print("="*60)
    
    # Weighted average ensemble
    print("\nCreating Weighted Average Ensemble...")
    weighted_ensemble = WeightedAverageEnsemble(models=[lgb, xgb])
    weighted_ensemble.fit(X_val, y_val, n_trials=20)
    
    print(f"Optimal weights: {weighted_ensemble.weights}")
    
    weighted_preds = weighted_ensemble.predict_proba(X_test)[:, 1]
    weighted_auc = roc_auc_score(y_test, weighted_preds)
    print(f"Weighted Ensemble AUC: {weighted_auc:.4f}")
    
    return weighted_ensemble


def example_unsupervised(X_train):
    """Example: Using unsupervised learning."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Unsupervised Learning")
    print("="*60)
    
    # Clustering with optimal k
    print("\nPerforming KMeans clustering with optimal k selection...")
    clustering = ClusteringEngine(method='kmeans')
    labels = clustering.fit_predict(X_train, optimal_k=True, k_range=(2, 5))
    
    print(f"Optimal number of clusters: {clustering.n_clusters}")
    print(f"Cluster distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        print(f"  Cluster {cluster}: {count} samples")
    
    # Dimensionality reduction
    print("\nApplying PCA dimensionality reduction...")
    reducer = DimensionalityReduction(method='pca', n_components=2)
    X_reduced = reducer.fit_transform(X_train)
    print(f"Reduced shape: {X_reduced.shape}")
    
    # UMAP
    try:
        print("\nApplying UMAP dimensionality reduction...")
        umap_reducer = DimensionalityReduction(method='umap', n_components=2)
        X_umap = umap_reducer.fit_transform(X_train)
        print(f"UMAP reduced shape: {X_umap.shape}")
    except ImportError:
        print("UMAP not available")
    
    return clustering, reducer


def example_automl():
    """Example: Using AutoML."""
    print("\n" + "="*60)
    print("EXAMPLE 5: AutoML")
    print("="*60)
    
    # Generate data
    X, y = make_classification(n_samples=500, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Convert to DataFrame
    X_train = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(20)])
    X_val = pd.DataFrame(X_val, columns=[f'feature_{i}' for i in range(20)])
    X_test = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(20)])
    
    # Run AutoML
    print("\nRunning AutoML...")
    automl = AutoMLEngine(
        config_path='config/model_config.yaml',
        task='classification',
        metric='auc'
    )
    
    best_model = automl.fit(
        X_train, y_train, X_val, y_val,
        models=['lightgbm', 'xgboost', 'random_forest'],
        n_trials=10  # Small number for demo
    )
    
    print(f"\nBest model: {automl.best_model_name}")
    print(f"Best validation score: {automl.best_score:.4f}")
    
    # Get leaderboard
    print("\nLeaderboard:")
    print(automl.get_leaderboard())
    
    # Test performance
    test_preds = best_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_preds)
    print(f"\nTest AUC: {test_auc:.4f}")
    
    return automl


def example_model_persistence(lgb, tmp_path="/tmp"):
    """Example: Saving and loading models."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Model Persistence")
    print("="*60)
    
    import tempfile
    from pathlib import Path
    
    # Save model
    model_path = Path(tmp_path) / "lgb_model.pkl"
    print(f"\nSaving model to {model_path}...")
    lgb.save(model_path)
    
    # Load model
    print("Loading model...")
    loaded_lgb = LightGBMModel()
    loaded_lgb.load(model_path)
    
    print("Model loaded successfully!")
    print(f"Is fitted: {loaded_lgb.is_fitted}")
    print(f"Number of features: {len(loaded_lgb.feature_names)}")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("MODELS LAYER EXAMPLES")
    print("="*60)
    
    # Example 1: Tree models
    lgb, xgb, X_train, y_train, X_test, y_test, X_val, y_val = example_tree_models()
    
    # Example 2: Neural tabular
    tabnet = example_neural_tabular(X_train, y_train, X_val, y_val, X_test, y_test)
    
    # Example 3: Ensembles
    ensemble = example_ensembles(lgb, xgb, X_val, y_val, X_test, y_test)
    
    # Example 4: Unsupervised
    clustering, reducer = example_unsupervised(X_train)
    
    # Example 5: AutoML
    automl = example_automl()
    
    # Example 6: Model persistence
    example_model_persistence(lgb)
    
    print("\n" + "="*60)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main()
