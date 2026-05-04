"""
Unsupervised Anomaly Detection for Fraud

Implements multiple anomaly detection methods for identifying novel fraud patterns:
- Isolation Forest for tree-based anomaly detection
- Local Outlier Factor for density-based detection
- Autoencoder for neural network-based detection
- Ensemble methods combining multiple detectors
- Anomaly score calibration
- Integration with supervised models

Example:
    >>> from src.use_cases.fraud_detection.anomaly_detection import EnsembleAnomalyDetector
    >>> detector = EnsembleAnomalyDetector()
    >>> detector.fit(X_train)
    >>> anomaly_scores = detector.predict_scores(X_test)
    >>> is_anomaly = detector.predict(X_test, contamination=0.01)
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod

from loguru import logger
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import roc_auc_score, average_precision_score

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available. Install with: pip install torch")
    TORCH_AVAILABLE = False

try:
    from sklearn.ensemble import IsolationForest as SKLearnIF
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class BaseAnomalyDetector(ABC):
    """Base class for anomaly detectors."""
    
    def __init__(self):
        """Initialize base anomaly detector."""
        self.is_fitted = False
        self.scaler = StandardScaler()
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit the anomaly detector."""
        pass
    
    @abstractmethod
    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores (higher = more anomalous)."""
        pass
    
    def predict(
        self,
        X: np.ndarray,
        contamination: float = 0.01
    ) -> np.ndarray:
        """
        Predict anomaly labels.
        
        Args:
            X: Features
            contamination: Expected proportion of anomalies
            
        Returns:
            Binary predictions (1 = anomaly, 0 = normal)
        """
        scores = self.predict_scores(X)
        threshold = np.percentile(scores, 100 * (1 - contamination))
        return (scores >= threshold).astype(int)
    
    def save(self, path: Union[str, Path]):
        """Save detector to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Detector saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'BaseAnomalyDetector':
        """Load detector from disk."""
        detector = joblib.load(path)
        logger.info(f"Detector loaded from {path}")
        return detector


class IsolationForestDetector(BaseAnomalyDetector):
    """
    Isolation Forest anomaly detector.
    
    Uses tree-based isolation to identify anomalies. Efficient for high-dimensional
    data and can handle large datasets.
    
    Example:
        >>> detector = IsolationForestDetector(n_estimators=200)
        >>> detector.fit(X_train)
        >>> anomaly_scores = detector.predict_scores(X_test)
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: Union[int, str] = 'auto',
        contamination: float = 0.01,
        max_features: float = 1.0,
        random_state: int = 42
    ):
        """
        Initialize Isolation Forest detector.
        
        Args:
            n_estimators: Number of trees
            max_samples: Number of samples per tree
            contamination: Expected proportion of anomalies
            max_features: Fraction of features per tree
            random_state: Random seed
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.random_state = random_state
        self.model = None
        
        logger.info("Initialized IsolationForestDetector")
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[np.ndarray] = None
    ) -> 'IsolationForestDetector':
        """
        Fit Isolation Forest.
        
        Args:
            X: Training features
            y: Not used (unsupervised)
            
        Returns:
            Self
        """
        logger.info(f"Fitting Isolation Forest on {len(X)} samples")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.model.fit(X)
        self.is_fitted = True
        
        logger.info("Isolation Forest fitting completed")
        return self
    
    def predict_scores(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict anomaly scores.
        
        Args:
            X: Features
            
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Isolation Forest returns negative scores (more negative = more anomalous)
        # We invert to get positive scores
        scores = -self.model.score_samples(X)
        return scores


class LocalOutlierFactorDetector(BaseAnomalyDetector):
    """
    Local Outlier Factor (LOF) anomaly detector.
    
    Uses local density to identify anomalies. Good for finding local outliers
    in datasets with varying density.
    
    Example:
        >>> detector = LocalOutlierFactorDetector(n_neighbors=20)
        >>> detector.fit(X_train)
        >>> anomaly_scores = detector.predict_scores(X_test)
    """
    
    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.01,
        metric: str = 'minkowski',
        novelty: bool = True
    ):
        """
        Initialize LOF detector.
        
        Args:
            n_neighbors: Number of neighbors to use
            contamination: Expected proportion of anomalies
            metric: Distance metric
            novelty: Whether to use novelty detection mode
        """
        super().__init__()
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.metric = metric
        self.novelty = novelty
        self.model = None
        
        logger.info("Initialized LocalOutlierFactorDetector")
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[np.ndarray] = None
    ) -> 'LocalOutlierFactorDetector':
        """
        Fit LOF detector.
        
        Args:
            X: Training features
            y: Not used (unsupervised)
            
        Returns:
            Self
        """
        logger.info(f"Fitting LOF on {len(X)} samples")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            metric=self.metric,
            novelty=self.novelty,
            n_jobs=-1
        )
        
        self.model.fit(X)
        self.is_fitted = True
        
        logger.info("LOF fitting completed")
        return self
    
    def predict_scores(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict anomaly scores.
        
        Args:
            X: Features
            
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if self.novelty:
            # LOF returns negative scores (more negative = more anomalous)
            scores = -self.model.score_samples(X)
        else:
            # If not novelty mode, use fit_predict
            scores = -self.model.negative_outlier_factor_
        
        return scores


class AutoencoderDetector(BaseAnomalyDetector):
    """
    Autoencoder-based anomaly detector.
    
    Uses reconstruction error from a neural network autoencoder to identify
    anomalies. Good for complex, high-dimensional patterns.
    
    Example:
        >>> detector = AutoencoderDetector(
        ...     encoding_dim=32,
        ...     hidden_dims=[64, 32]
        ... )
        >>> detector.fit(X_train)
        >>> anomaly_scores = detector.predict_scores(X_test)
    """
    
    def __init__(
        self,
        encoding_dim: int = 32,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        epochs: int = 50,
        early_stopping_rounds: int = 10,
        validation_split: float = 0.2
    ):
        """
        Initialize Autoencoder detector.
        
        Args:
            encoding_dim: Dimension of encoded representation
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
            learning_rate: Learning rate
            batch_size: Batch size
            epochs: Number of epochs
            early_stopping_rounds: Early stopping patience
            validation_split: Fraction of data for validation
        """
        super().__init__()
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for AutoencoderDetector")
        
        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims or [64, 32]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_split = validation_split
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info("Initialized AutoencoderDetector")
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[np.ndarray] = None
    ) -> 'AutoencoderDetector':
        """
        Fit Autoencoder.
        
        Args:
            X: Training features
            y: Not used (unsupervised)
            
        Returns:
            Self
        """
        logger.info(f"Fitting Autoencoder on {len(X)} samples")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split into train/validation
        n_val = int(len(X_scaled) * self.validation_split)
        indices = np.random.permutation(len(X_scaled))
        X_train = X_scaled[indices[n_val:]]
        X_val = X_scaled[indices[:n_val]]
        
        # Create model
        input_dim = X.shape[1]
        self.model = Autoencoder(
            input_dim=input_dim,
            encoding_dim=self.encoding_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout
        ).to(self.device)
        
        # Train model
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_losses = []
            
            indices = np.random.permutation(len(X_train))
            for i in range(0, len(X_train), self.batch_size):
                batch_indices = indices[i:i+self.batch_size]
                X_batch = torch.FloatTensor(X_train[batch_indices]).to(self.device)
                
                optimizer.zero_grad()
                X_recon = self.model(X_batch)
                loss = criterion(X_recon, X_batch)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                X_val_t = torch.FloatTensor(X_val).to(self.device)
                X_val_recon = self.model(X_val_t)
                val_loss = criterion(X_val_recon, X_val_t).item()
            
            avg_train_loss = np.mean(train_losses)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.early_stopping_rounds:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{self.epochs}: "
                    f"train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}"
                )
        
        self.is_fitted = True
        logger.info("Autoencoder fitting completed")
        return self
    
    def predict_scores(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict anomaly scores based on reconstruction error.
        
        Args:
            X: Features
            
        Returns:
            Anomaly scores (reconstruction error)
        """
        if not self.is_fitted:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.scaler.transform(X)
        
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_scaled).to(self.device)
            X_recon = self.model(X_t)
            
            # Compute reconstruction error
            errors = torch.mean((X_t - X_recon) ** 2, dim=1)
            scores = errors.cpu().numpy()
        
        return scores


class Autoencoder(nn.Module):
    """Autoencoder neural network."""
    
    def __init__(
        self,
        input_dim: int,
        encoding_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.2
    ):
        """
        Initialize Autoencoder.
        
        Args:
            input_dim: Input dimension
            encoding_dim: Encoding dimension
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        encoder_layers.append(nn.Linear(prev_dim, encoding_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = encoding_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input."""
        return self.encoder(x)


class EnsembleAnomalyDetector(BaseAnomalyDetector):
    """
    Ensemble anomaly detector combining multiple methods.
    
    Combines Isolation Forest, LOF, and Autoencoder for robust anomaly detection.
    Uses weighted voting or score averaging.
    
    Example:
        >>> detector = EnsembleAnomalyDetector(
        ...     use_isolation_forest=True,
        ...     use_lof=True,
        ...     use_autoencoder=True
        ... )
        >>> detector.fit(X_train)
        >>> anomaly_scores = detector.predict_scores(X_test)
    """
    
    def __init__(
        self,
        use_isolation_forest: bool = True,
        use_lof: bool = True,
        use_autoencoder: bool = True,
        aggregation: str = 'mean',
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize Ensemble detector.
        
        Args:
            use_isolation_forest: Whether to use Isolation Forest
            use_lof: Whether to use LOF
            use_autoencoder: Whether to use Autoencoder
            aggregation: Score aggregation method ('mean', 'max', 'weighted')
            weights: Weights for each detector (if aggregation='weighted')
        """
        super().__init__()
        
        self.use_isolation_forest = use_isolation_forest
        self.use_lof = use_lof
        self.use_autoencoder = use_autoencoder
        self.aggregation = aggregation
        self.weights = weights or {}
        
        self.detectors: Dict[str, BaseAnomalyDetector] = {}
        
        if use_isolation_forest:
            self.detectors['isolation_forest'] = IsolationForestDetector()
        
        if use_lof:
            self.detectors['lof'] = LocalOutlierFactorDetector()
        
        if use_autoencoder and TORCH_AVAILABLE:
            self.detectors['autoencoder'] = AutoencoderDetector()
        elif use_autoencoder:
            logger.warning("PyTorch not available, skipping Autoencoder")
        
        if not self.detectors:
            raise ValueError("At least one detector must be enabled")
        
        logger.info(f"Initialized EnsembleAnomalyDetector with {len(self.detectors)} detectors")
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[np.ndarray] = None
    ) -> 'EnsembleAnomalyDetector':
        """
        Fit all detectors.
        
        Args:
            X: Training features
            y: Optional labels for calibration
            
        Returns:
            Self
        """
        logger.info(f"Fitting ensemble with {len(self.detectors)} detectors")
        
        for name, detector in self.detectors.items():
            logger.info(f"Fitting {name}")
            detector.fit(X, y)
        
        # Calibrate scores if labels are provided
        if y is not None:
            self._calibrate_scores(X, y)
        
        self.is_fitted = True
        logger.info("Ensemble fitting completed")
        return self
    
    def predict_scores(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict ensemble anomaly scores.
        
        Args:
            X: Features
            
        Returns:
            Aggregated anomaly scores
        """
        if not self.is_fitted:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        # Get scores from all detectors
        all_scores = {}
        for name, detector in self.detectors.items():
            scores = detector.predict_scores(X)
            # Normalize scores to [0, 1]
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
            all_scores[name] = scores
        
        # Aggregate scores
        if self.aggregation == 'mean':
            final_scores = np.mean(list(all_scores.values()), axis=0)
        elif self.aggregation == 'max':
            final_scores = np.max(list(all_scores.values()), axis=0)
        elif self.aggregation == 'weighted':
            weighted_scores = []
            for name, scores in all_scores.items():
                weight = self.weights.get(name, 1.0)
                weighted_scores.append(scores * weight)
            final_scores = np.sum(weighted_scores, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
        
        return final_scores
    
    def _calibrate_scores(self, X: np.ndarray, y: np.ndarray):
        """
        Calibrate detector weights based on labeled data.
        
        Args:
            X: Features
            y: Binary labels (1 = fraud/anomaly, 0 = normal)
        """
        logger.info("Calibrating detector weights")
        
        if y.sum() == 0 or y.sum() == len(y):
            logger.warning("Cannot calibrate with only one class")
            return
        
        # Compute AUC for each detector
        aucs = {}
        for name, detector in self.detectors.items():
            scores = detector.predict_scores(X)
            try:
                auc = roc_auc_score(y, scores)
                aucs[name] = auc
                logger.info(f"{name} AUC: {auc:.4f}")
            except Exception as e:
                logger.warning(f"Could not compute AUC for {name}: {e}")
                aucs[name] = 0.5
        
        # Set weights proportional to AUC
        total_auc = sum(aucs.values())
        if total_auc > 0:
            self.weights = {
                name: auc / total_auc
                for name, auc in aucs.items()
            }
            self.aggregation = 'weighted'
            logger.info(f"Calibrated weights: {self.weights}")
    
    def get_detector_scores(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Get individual scores from each detector.
        
        Args:
            X: Features
            
        Returns:
            Dictionary of detector names to scores
        """
        if not self.is_fitted:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        scores = {}
        for name, detector in self.detectors.items():
            scores[name] = detector.predict_scores(X)
        
        return scores


class AnomalyScoreCalibrator:
    """
    Calibrate anomaly scores for better interpretability.
    
    Transforms raw anomaly scores to calibrated probabilities using
    a small set of labeled examples.
    
    Example:
        >>> calibrator = AnomalyScoreCalibrator()
        >>> calibrator.fit(anomaly_scores, labels)
        >>> calibrated_probs = calibrator.transform(new_scores)
    """
    
    def __init__(self, method: str = 'isotonic'):
        """
        Initialize calibrator.
        
        Args:
            method: Calibration method ('isotonic' or 'sigmoid')
        """
        self.method = method
        self.calibrator = None
        
        logger.info(f"Initialized AnomalyScoreCalibrator with {method} method")
    
    def fit(
        self,
        scores: np.ndarray,
        y: np.ndarray
    ) -> 'AnomalyScoreCalibrator':
        """
        Fit calibrator on labeled data.
        
        Args:
            scores: Anomaly scores
            y: Binary labels (1 = anomaly, 0 = normal)
            
        Returns:
            Self
        """
        logger.info(f"Calibrating scores on {len(scores)} samples")
        
        if self.method == 'isotonic':
            from sklearn.isotonic import IsotonicRegression
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
        elif self.method == 'sigmoid':
            from sklearn.linear_model import LogisticRegression
            self.calibrator = LogisticRegression()
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        
        if self.method == 'sigmoid':
            self.calibrator.fit(scores.reshape(-1, 1), y)
        else:
            self.calibrator.fit(scores, y)
        
        logger.info("Calibration completed")
        return self
    
    def transform(self, scores: np.ndarray) -> np.ndarray:
        """
        Transform scores to calibrated probabilities.
        
        Args:
            scores: Anomaly scores
            
        Returns:
            Calibrated probabilities
        """
        if self.calibrator is None:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        
        if self.method == 'sigmoid':
            return self.calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]
        else:
            return self.calibrator.predict(scores)
    
    def save(self, path: Union[str, Path]):
        """Save calibrator to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Calibrator saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'AnomalyScoreCalibrator':
        """Load calibrator from disk."""
        calibrator = joblib.load(path)
        logger.info(f"Calibrator loaded from {path}")
        return calibrator
