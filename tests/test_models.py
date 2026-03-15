"""
Test suite for models layer.

Tests all model implementations, ensembles, and AutoML functionality.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split


@pytest.fixture
def classification_data():
    """Generate synthetic classification data."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    return {
        'X_train': pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(20)]),
        'X_val': pd.DataFrame(X_val, columns=[f'feature_{i}' for i in range(20)]),
        'X_test': pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(20)]),
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }


@pytest.fixture
def regression_data():
    """Generate synthetic regression data."""
    X, y = make_regression(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return {
        'X_train': pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(20)]),
        'X_test': pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(20)]),
        'y_train': y_train,
        'y_test': y_test
    }


class TestTreeModels:
    """Test tree-based models."""
    
    def test_lightgbm_classification(self, classification_data):
        """Test LightGBM for classification."""
        from src.models.tree_models import LightGBMModel
        
        model = LightGBMModel(config_path='config/model_config.yaml')
        model.fit(
            classification_data['X_train'],
            classification_data['y_train'],
            classification_data['X_val'],
            classification_data['y_val']
        )
        
        # Predictions
        preds = model.predict_proba(classification_data['X_test'])
        assert preds.shape == (len(classification_data['X_test']), 2)
        
        # Feature importance
        importance = model.get_feature_importance()
        assert len(importance) == 20
        
    def test_xgboost_classification(self, classification_data):
        """Test XGBoost for classification."""
        from src.models.tree_models import XGBoostModel
        
        model = XGBoostModel(config_path='config/model_config.yaml')
        model.fit(
            classification_data['X_train'],
            classification_data['y_train'],
            classification_data['X_val'],
            classification_data['y_val']
        )
        
        preds = model.predict_proba(classification_data['X_test'])
        assert preds.shape == (len(classification_data['X_test']), 2)
    
    def test_catboost_classification(self, classification_data):
        """Test CatBoost for classification."""
        from src.models.tree_models import CatBoostModel
        
        model = CatBoostModel(config_path='config/model_config.yaml')
        model.fit(
            classification_data['X_train'],
            classification_data['y_train'],
            classification_data['X_val'],
            classification_data['y_val']
        )
        
        preds = model.predict_proba(classification_data['X_test'])
        assert preds.shape == (len(classification_data['X_test']), 2)
    
    def test_random_forest_classification(self, classification_data):
        """Test Random Forest for classification."""
        from src.models.tree_models import RandomForestModel
        
        model = RandomForestModel(config_path='config/model_config.yaml')
        model.fit(
            classification_data['X_train'],
            classification_data['y_train']
        )
        
        preds = model.predict_proba(classification_data['X_test'])
        assert preds.shape == (len(classification_data['X_test']), 2)
    
    def test_model_save_load(self, classification_data, tmp_path):
        """Test model save/load functionality."""
        from src.models.tree_models import LightGBMModel
        
        # Train and save
        model = LightGBMModel(config_path='config/model_config.yaml')
        model.fit(
            classification_data['X_train'],
            classification_data['y_train']
        )
        
        model_path = tmp_path / "model.pkl"
        model.save(model_path)
        
        # Load and predict
        loaded_model = LightGBMModel()
        loaded_model.load(model_path)
        
        preds_original = model.predict_proba(classification_data['X_test'])
        preds_loaded = loaded_model.predict_proba(classification_data['X_test'])
        
        np.testing.assert_array_almost_equal(preds_original, preds_loaded)


class TestNeuralTabular:
    """Test neural tabular models."""
    
    def test_tabnet_classification(self, classification_data):
        """Test TabNet for classification."""
        from src.models.neural_tabular import TabNetModel
        
        model = TabNetModel(config_path='config/model_config.yaml')
        model.fit(
            classification_data['X_train'],
            classification_data['y_train'],
            classification_data['X_val'],
            classification_data['y_val'],
            max_epochs=10
        )
        
        preds = model.predict_proba(classification_data['X_test'])
        assert preds.shape == (len(classification_data['X_test']), 2)
        
        # Feature importance
        importance = model.get_feature_importance()
        assert len(importance) == 20


class TestEnsembles:
    """Test ensemble methods."""
    
    def test_weighted_average_ensemble(self, classification_data):
        """Test weighted average ensemble."""
        from src.models.tree_models import LightGBMModel, XGBoostModel
        from src.models.ensemble_engine import WeightedAverageEnsemble
        
        # Train base models
        lgb = LightGBMModel(config_path='config/model_config.yaml')
        lgb.fit(classification_data['X_train'], classification_data['y_train'])
        
        xgb = XGBoostModel(config_path='config/model_config.yaml')
        xgb.fit(classification_data['X_train'], classification_data['y_train'])
        
        # Ensemble
        ensemble = WeightedAverageEnsemble(models=[lgb, xgb])
        ensemble.fit(
            classification_data['X_val'],
            classification_data['y_val'],
            n_trials=10
        )
        
        preds = ensemble.predict_proba(classification_data['X_test'])
        assert preds.shape == (len(classification_data['X_test']), 2)
    
    def test_stacking_ensemble(self, classification_data):
        """Test stacking ensemble."""
        from src.models.tree_models import LightGBMModel, XGBoostModel
        from src.models.ensemble_engine import StackingEnsemble
        
        # Create base models (not fitted)
        lgb = LightGBMModel(config_path='config/model_config.yaml')
        xgb = XGBoostModel(config_path='config/model_config.yaml')
        
        # Stacking ensemble
        ensemble = StackingEnsemble(models=[lgb, xgb], cv=3)
        ensemble.fit(classification_data['X_train'], classification_data['y_train'])
        
        preds = ensemble.predict_proba(classification_data['X_test'])
        assert preds.shape == (len(classification_data['X_test']), 2)


class TestUnsupervised:
    """Test unsupervised learning methods."""
    
    def test_clustering_kmeans(self, classification_data):
        """Test KMeans clustering."""
        from src.models.unsupervised import ClusteringEngine
        
        engine = ClusteringEngine(method='kmeans', n_clusters=3)
        labels = engine.fit_predict(classification_data['X_train'])
        
        assert len(labels) == len(classification_data['X_train'])
        assert len(np.unique(labels)) <= 3
    
    def test_clustering_optimal_k(self, classification_data):
        """Test optimal k selection."""
        from src.models.unsupervised import ClusteringEngine
        
        engine = ClusteringEngine(method='kmeans')
        labels = engine.fit_predict(
            classification_data['X_train'][:500],  # Smaller dataset for speed
            optimal_k=True,
            k_range=(2, 5)
        )
        
        assert engine.n_clusters >= 2
        assert engine.n_clusters <= 5
    
    def test_dimensionality_reduction_pca(self, classification_data):
        """Test PCA dimensionality reduction."""
        from src.models.unsupervised import DimensionalityReduction
        
        reducer = DimensionalityReduction(method='pca', n_components=2)
        X_reduced = reducer.fit_transform(classification_data['X_train'])
        
        assert X_reduced.shape == (len(classification_data['X_train']), 2)
    
    def test_autoencoder_clustering(self, classification_data):
        """Test autoencoder-based clustering."""
        from src.models.unsupervised import AutoencoderClustering
        
        model = AutoencoderClustering(
            encoding_dim=5,
            n_clusters=3,
            hidden_dims=[16, 8]
        )
        model.fit(
            classification_data['X_train'][:500],
            epochs=10,
            verbose=False
        )
        
        labels = model.predict(classification_data['X_test'][:100])
        assert len(labels) == 100


class TestMetaLearner:
    """Test meta-learning and AutoML."""
    
    def test_bayesian_optimizer(self):
        """Test Bayesian optimizer."""
        from src.models.meta_learner import BayesianOptimizer
        
        # Define simple objective
        def objective(params):
            return -(params['x'] - 5) ** 2 + params['y']
        
        # Search space
        search_space = {
            'x': ('float', 0.0, 10.0),
            'y': ('int', 0, 100)
        }
        
        optimizer = BayesianOptimizer(metric='score', direction='maximize')
        best_params, best_score = optimizer.optimize(
            objective, search_space, n_trials=20
        )
        
        assert best_params is not None
        assert best_score is not None
        assert 4.0 <= best_params['x'] <= 6.0  # Should find x near 5
    
    def test_hyperparameter_search_space(self):
        """Test hyperparameter search space builder."""
        from src.models.meta_learner import HyperparameterSearchSpace
        
        search_space = HyperparameterSearchSpace.from_config(
            'config/model_config.yaml',
            'lightgbm'
        )
        
        assert 'num_leaves' in search_space
        assert 'learning_rate' in search_space
    
    def test_automl_engine(self, classification_data):
        """Test AutoML engine."""
        from src.models.meta_learner import AutoMLEngine
        
        automl = AutoMLEngine(
            config_path='config/model_config.yaml',
            task='classification',
            metric='auc'
        )
        
        best_model = automl.fit(
            classification_data['X_train'][:500],
            classification_data['y_train'][:500],
            classification_data['X_val'][:100],
            classification_data['y_val'][:100],
            models=['lightgbm', 'random_forest'],
            n_trials=5
        )
        
        assert best_model is not None
        assert automl.best_model_name in ['lightgbm', 'random_forest']
        
        # Get leaderboard
        leaderboard = automl.get_leaderboard()
        assert len(leaderboard) == 2


class TestIntegration:
    """Integration tests."""
    
    def test_full_pipeline(self, classification_data):
        """Test full ML pipeline."""
        from src.models.tree_models import LightGBMModel, XGBoostModel
        from src.models.ensemble_engine import WeightedAverageEnsemble
        from sklearn.metrics import roc_auc_score
        
        # Train models
        lgb = LightGBMModel(config_path='config/model_config.yaml')
        lgb.fit(
            classification_data['X_train'],
            classification_data['y_train'],
            classification_data['X_val'],
            classification_data['y_val']
        )
        
        xgb = XGBoostModel(config_path='config/model_config.yaml')
        xgb.fit(
            classification_data['X_train'],
            classification_data['y_train'],
            classification_data['X_val'],
            classification_data['y_val']
        )
        
        # Ensemble
        ensemble = WeightedAverageEnsemble(models=[lgb, xgb])
        ensemble.fit(
            classification_data['X_val'],
            classification_data['y_val'],
            n_trials=10
        )
        
        # Evaluate
        preds = ensemble.predict_proba(classification_data['X_test'])[:, 1]
        auc = roc_auc_score(classification_data['y_test'], preds)
        
        assert auc > 0.5  # Should be better than random
        print(f"Ensemble AUC: {auc:.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
