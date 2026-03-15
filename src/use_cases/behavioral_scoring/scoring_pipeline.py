"""
End-to-End Behavioral Scoring Pipeline

Orchestrates complete behavioral scoring workflow:
- Transaction data preprocessing
- Behavioral feature engineering integration
- Multi-model training and evaluation
- Temporal validation (walk-forward)
- Score calibration and monitoring
- API for real-time scoring
- Batch scoring utilities
- Model retraining automation

Example:
    >>> from src.use_cases.behavioral_scoring.scoring_pipeline import BehavioralScoringPipeline
    >>> 
    >>> # Initialize pipeline
    >>> pipeline = BehavioralScoringPipeline(
    ...     feature_config="config/feature_config.yaml",
    ...     use_deep_learning=True
    ... )
    >>> 
    >>> # Train pipeline
    >>> pipeline.fit(transactions_df, labels_df)
    >>> 
    >>> # Real-time scoring
    >>> score = pipeline.score_single(user_id="12345", transaction_data=transaction)
    >>> 
    >>> # Batch scoring
    >>> scores = pipeline.score_batch(test_transactions)
    >>> 
    >>> # Model monitoring
    >>> metrics = pipeline.monitor_performance(validation_data, validation_labels)
"""

import joblib
import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix
)

# Import feature engineering modules
try:
    from src.features.behavioral_features import BehavioralFeatureEngine
    from src.features.temporal_features import TemporalFeatureEngine
    from src.features.persona_features import PersonaFeatureEngine
    FEATURES_AVAILABLE = True
except ImportError:
    logger.warning("Feature modules not available")
    FEATURES_AVAILABLE = False

# Import scoring modules
from src.use_cases.behavioral_scoring.meta_scoring import MetaScoringEngine
from src.use_cases.behavioral_scoring.ensemble_scoring import BehavioralEnsembleScorer

try:
    from src.use_cases.behavioral_scoring.deep_scoring import (
        LSTMScoringModel,
        TransformerScoringModel,
        TemporalCNN,
        SequencePreprocessor
    )
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    logger.warning("Deep learning modules require PyTorch")
    DEEP_LEARNING_AVAILABLE = False


class TransactionPreprocessor:
    """Preprocess transaction data for scoring."""
    
    def __init__(
        self,
        user_col: str = "user_id",
        timestamp_col: str = "timestamp",
        amount_col: str = "amount",
        min_transactions: int = 5
    ):
        """
        Initialize transaction preprocessor.
        
        Args:
            user_col: User identifier column
            timestamp_col: Timestamp column
            amount_col: Transaction amount column
            min_transactions: Minimum transactions per user
            
        Example:
            >>> preprocessor = TransactionPreprocessor()
            >>> clean_df = preprocessor.transform(raw_transactions_df)
        """
        self.user_col = user_col
        self.timestamp_col = timestamp_col
        self.amount_col = amount_col
        self.min_transactions = min_transactions
        
    def transform(
        self,
        df: pd.DataFrame,
        handle_missing: bool = True,
        remove_outliers: bool = True
    ) -> pd.DataFrame:
        """
        Transform transaction data.
        
        Args:
            df: Raw transaction DataFrame
            handle_missing: Whether to handle missing values
            remove_outliers: Whether to remove outliers
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Preprocessing {len(df)} transactions")
        
        df = df.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[self.timestamp_col]):
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
        
        # Sort by user and timestamp
        df = df.sort_values([self.user_col, self.timestamp_col])
        
        # Handle missing values
        if handle_missing:
            # Fill missing amounts with median
            df[self.amount_col] = df.groupby(self.user_col)[self.amount_col].transform(
                lambda x: x.fillna(x.median())
            )
            
            # Drop rows with missing critical fields
            df = df.dropna(subset=[self.user_col, self.timestamp_col])
        
        # Remove outliers
        if remove_outliers:
            # Remove extreme amounts (beyond 99.9th percentile)
            q99 = df[self.amount_col].quantile(0.999)
            df = df[df[self.amount_col] <= q99]
        
        # Filter users with minimum transactions
        user_counts = df[self.user_col].value_counts()
        valid_users = user_counts[user_counts >= self.min_transactions].index
        df = df[df[self.user_col].isin(valid_users)]
        
        logger.info(f"After preprocessing: {len(df)} transactions, {df[self.user_col].nunique()} users")
        
        return df


class TemporalValidator:
    """Temporal validation using walk-forward approach."""
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size_days: int = 30
    ):
        """
        Initialize temporal validator.
        
        Args:
            n_splits: Number of validation splits
            test_size_days: Test period size in days
            
        Example:
            >>> validator = TemporalValidator(n_splits=5)
            >>> for train_idx, test_idx in validator.split(df, timestamp_col="date"):
            ...     train_data = df.iloc[train_idx]
            ...     test_data = df.iloc[test_idx]
        """
        self.n_splits = n_splits
        self.test_size_days = test_size_days
        
    def split(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp"
    ):
        """
        Generate train/test splits.
        
        Args:
            df: DataFrame with timestamp column
            timestamp_col: Name of timestamp column
            
        Yields:
            (train_indices, test_indices) tuples
        """
        df = df.sort_values(timestamp_col).reset_index(drop=True)
        
        min_date = df[timestamp_col].min()
        max_date = df[timestamp_col].max()
        total_days = (max_date - min_date).days
        
        test_days = self.test_size_days
        step_size = (total_days - test_days) // self.n_splits
        
        for i in range(self.n_splits):
            # Define split date
            split_date = min_date + timedelta(days=(i + 1) * step_size)
            test_end_date = split_date + timedelta(days=test_days)
            
            # Create indices
            train_mask = df[timestamp_col] < split_date
            test_mask = (df[timestamp_col] >= split_date) & (df[timestamp_col] < test_end_date)
            
            train_idx = df[train_mask].index.tolist()
            test_idx = df[test_mask].index.tolist()
            
            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx


class PerformanceMonitor:
    """Monitor model performance over time."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics_history = []
        
    def log_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timestamp: Optional[datetime] = None,
        segment: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Log performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted scores
            timestamp: Timestamp for metrics (uses now if None)
            segment: Segment identifier
            
        Returns:
            Dictionary of metrics
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        metrics = {
            'timestamp': timestamp,
            'segment': segment or 'global',
            'auc_roc': roc_auc_score(y_true, y_pred),
            'avg_precision': average_precision_score(y_true, y_pred),
            'n_samples': len(y_true),
            'pos_rate': y_true.mean()
        }
        
        # Add threshold-based metrics at 0.5
        threshold = 0.5
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        
        metrics.update({
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0
        })
        
        self.metrics_history.append(metrics)
        
        return metrics
    
    def get_performance_trend(
        self,
        metric: str = 'auc_roc',
        window: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get performance trend over time.
        
        Args:
            metric: Metric to track
            window: Window size for rolling average (None for all)
            
        Returns:
            DataFrame with performance trend
        """
        df = pd.DataFrame(self.metrics_history)
        
        if window:
            df[f'{metric}_rolling'] = df[metric].rolling(window=window).mean()
        
        return df
    
    def detect_performance_degradation(
        self,
        metric: str = 'auc_roc',
        threshold: float = 0.05,
        window: int = 5
    ) -> bool:
        """
        Detect performance degradation.
        
        Args:
            metric: Metric to monitor
            threshold: Degradation threshold
            window: Window for comparison
            
        Returns:
            True if degradation detected
        """
        if len(self.metrics_history) < window * 2:
            return False
        
        df = pd.DataFrame(self.metrics_history)
        
        recent_avg = df[metric].tail(window).mean()
        baseline_avg = df[metric].head(window).mean()
        
        degradation = (baseline_avg - recent_avg) / baseline_avg
        
        if degradation > threshold:
            logger.warning(
                f"Performance degradation detected: {metric} dropped by {degradation*100:.1f}%"
            )
            return True
        
        return False


class BehavioralScoringPipeline:
    """
    End-to-end behavioral scoring pipeline.
    
    Orchestrates data preprocessing, feature engineering, model training,
    validation, and scoring for production deployment.
    """
    
    def __init__(
        self,
        feature_config: Optional[str] = None,
        use_deep_learning: bool = False,
        use_meta_learning: bool = False,
        ensemble_type: str = "stacking",
        user_col: str = "user_id",
        timestamp_col: str = "timestamp",
        amount_col: str = "amount"
    ):
        """
        Initialize behavioral scoring pipeline.
        
        Args:
            feature_config: Path to feature configuration YAML
            use_deep_learning: Whether to use deep learning models
            use_meta_learning: Whether to use meta-learning
            ensemble_type: Type of ensemble ('voting', 'stacking', 'weighted')
            user_col: User identifier column
            timestamp_col: Timestamp column
            amount_col: Amount column
            
        Example:
            >>> pipeline = BehavioralScoringPipeline(
            ...     feature_config="config/feature_config.yaml",
            ...     use_deep_learning=True,
            ...     ensemble_type="stacking"
            ... )
        """
        self.feature_config = feature_config
        self.use_deep_learning = use_deep_learning and DEEP_LEARNING_AVAILABLE
        self.use_meta_learning = use_meta_learning
        self.ensemble_type = ensemble_type
        self.user_col = user_col
        self.timestamp_col = timestamp_col
        self.amount_col = amount_col
        
        # Initialize components
        self.preprocessor = TransactionPreprocessor(
            user_col=user_col,
            timestamp_col=timestamp_col,
            amount_col=amount_col
        )
        
        if FEATURES_AVAILABLE:
            self.behavioral_features = BehavioralFeatureEngine(
                config_path=feature_config,
                entity_col=user_col,
                timestamp_col=timestamp_col,
                value_col=amount_col
            )
            self.temporal_features = TemporalFeatureEngine(
                config_path=feature_config,
                entity_col=user_col,
                timestamp_col=timestamp_col,
                value_col=amount_col
            )
            self.persona_features = PersonaFeatureEngine(
                config_path=feature_config,
                entity_col=user_col
            )
        else:
            logger.warning("Feature engineering modules not available")
            self.behavioral_features = None
            self.temporal_features = None
            self.persona_features = None
        
        # Models
        self.ensemble_model = None
        self.deep_model = None
        self.meta_model = None
        
        # Monitoring
        self.monitor = PerformanceMonitor()
        
        self.is_fitted = False
        
    def fit(
        self,
        transactions_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        segments_df: Optional[pd.DataFrame] = None,
        validation_strategy: str = "temporal",
        n_splits: int = 5
    ) -> Dict[str, Any]:
        """
        Fit scoring pipeline.
        
        Args:
            transactions_df: Transaction data
            labels_df: Labels (user_id, label)
            segments_df: Segment assignments (user_id, segment)
            validation_strategy: Validation strategy ('temporal', 'cv')
            n_splits: Number of validation splits
            
        Returns:
            Training history and metrics
            
        Example:
            >>> history = pipeline.fit(
            ...     transactions_df=transactions,
            ...     labels_df=labels,
            ...     segments_df=segments
            ... )
            >>> print(f"Validation AUC: {history['val_auc']:.3f}")
        """
        logger.info("Starting behavioral scoring pipeline training")
        
        # Preprocess transactions
        transactions_clean = self.preprocessor.transform(transactions_df)
        
        # Extract features
        logger.info("Extracting features...")
        features_df = self._extract_features(transactions_clean)
        
        # Merge with labels
        data_df = features_df.merge(
            labels_df,
            on=self.user_col,
            how='inner'
        )
        
        if segments_df is not None:
            data_df = data_df.merge(
                segments_df,
                on=self.user_col,
                how='left'
            )
            segment_col = 'segment'
        else:
            segment_col = None
        
        # Prepare data
        feature_cols = [col for col in data_df.columns 
                       if col not in [self.user_col, 'label', 'segment', self.timestamp_col]]
        
        X = data_df[feature_cols].values
        y = data_df['label'].values
        segments = data_df['segment'].values if segment_col in data_df.columns else None
        
        logger.info(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Temporal validation
        if validation_strategy == "temporal" and self.timestamp_col in data_df.columns:
            history = self._temporal_validation(
                data_df, X, y, segments, feature_cols, n_splits
            )
        else:
            history = self._simple_validation(X, y, segments)
        
        # Train final models on all data
        logger.info("Training final models on full dataset...")
        
        if self.use_meta_learning:
            self._train_meta_model(X, y, segments)
        
        self._train_ensemble_model(X, y, segments)
        
        if self.use_deep_learning:
            self._train_deep_model(transactions_clean, y)
        
        self.is_fitted = True
        logger.info("Pipeline training completed")
        
        return history
    
    def _extract_features(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Extract all features from transactions."""
        features_list = []
        
        if self.behavioral_features is not None:
            logger.debug("Extracting behavioral features...")
            behavioral_feat = self.behavioral_features.compute_all_features(transactions_df)
            features_list.append(behavioral_feat)
        
        if self.temporal_features is not None:
            logger.debug("Extracting temporal features...")
            temporal_feat = self.temporal_features.compute_all_features(transactions_df)
            features_list.append(temporal_feat)
        
        if self.persona_features is not None:
            logger.debug("Extracting persona features...")
            persona_feat = self.persona_features.compute_all_features(transactions_df)
            features_list.append(persona_feat)
        
        # Merge all features
        if features_list:
            features_df = features_list[0]
            for feat_df in features_list[1:]:
                features_df = features_df.merge(
                    feat_df,
                    on=self.user_col,
                    how='outer'
                )
            return features_df
        else:
            # Fallback: basic aggregations
            logger.warning("Using basic feature aggregations")
            agg_features = transactions_df.groupby(self.user_col).agg({
                self.amount_col: ['mean', 'std', 'min', 'max', 'sum', 'count']
            }).reset_index()
            agg_features.columns = [self.user_col] + [f'amount_{stat}' for stat in ['mean', 'std', 'min', 'max', 'sum', 'count']]
            return agg_features
    
    def _temporal_validation(
        self,
        data_df: pd.DataFrame,
        X: np.ndarray,
        y: np.ndarray,
        segments: Optional[np.ndarray],
        feature_cols: List[str],
        n_splits: int
    ) -> Dict[str, Any]:
        """Perform temporal validation."""
        logger.info(f"Performing temporal validation with {n_splits} splits")
        
        validator = TemporalValidator(n_splits=n_splits)
        
        val_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(validator.split(data_df, self.timestamp_col)):
            logger.info(f"Fold {fold + 1}/{n_splits}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            seg_train = segments[train_idx] if segments is not None else None
            seg_test = segments[test_idx] if segments is not None else None
            
            # Train ensemble
            ensemble = BehavioralEnsembleScorer(
                ensemble_type=self.ensemble_type,
                segment_aware=(segments is not None)
            )
            ensemble.fit(X_train, y_train, segments=seg_train)
            
            # Evaluate
            y_pred = ensemble.predict(X_test, segments=seg_test)
            
            metrics = {
                'fold': fold + 1,
                'auc_roc': roc_auc_score(y_test, y_pred),
                'avg_precision': average_precision_score(y_test, y_pred)
            }
            
            val_scores.append(metrics)
            logger.info(f"Fold {fold + 1} - AUC: {metrics['auc_roc']:.4f}")
        
        # Aggregate scores
        avg_auc = np.mean([m['auc_roc'] for m in val_scores])
        avg_ap = np.mean([m['avg_precision'] for m in val_scores])
        
        history = {
            'val_auc': avg_auc,
            'val_avg_precision': avg_ap,
            'fold_scores': val_scores
        }
        
        logger.info(f"Temporal validation - Avg AUC: {avg_auc:.4f}, Avg AP: {avg_ap:.4f}")
        
        return history
    
    def _simple_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        segments: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Simple train/test split validation."""
        split_idx = int(0.8 * len(X))
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        seg_train = segments[:split_idx] if segments is not None else None
        seg_test = segments[split_idx:] if segments is not None else None
        
        ensemble = BehavioralEnsembleScorer(
            ensemble_type=self.ensemble_type,
            segment_aware=(segments is not None)
        )
        ensemble.fit(X_train, y_train, segments=seg_train)
        
        y_pred = ensemble.predict(X_test, segments=seg_test)
        
        history = {
            'val_auc': roc_auc_score(y_test, y_pred),
            'val_avg_precision': average_precision_score(y_test, y_pred)
        }
        
        logger.info(f"Validation - AUC: {history['val_auc']:.4f}")
        
        return history
    
    def _train_ensemble_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        segments: Optional[np.ndarray]
    ):
        """Train ensemble model."""
        self.ensemble_model = BehavioralEnsembleScorer(
            ensemble_type=self.ensemble_type,
            segment_aware=(segments is not None)
        )
        self.ensemble_model.fit(X, y, segments=segments)
        logger.info("Ensemble model trained")
    
    def _train_meta_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        segments: Optional[np.ndarray]
    ):
        """Train meta-learning model."""
        if segments is None:
            logger.warning("Meta-learning requires segments, skipping")
            return
        
        # Prepare data for meta-learning
        X_dict = {f"sample_{i}": X[i] for i in range(len(X))}
        y_dict = {f"sample_{i}": y[i] for i in range(len(y))}
        seg_dict = {f"sample_{i}": segments[i] for i in range(len(segments))}
        
        self.meta_model = MetaScoringEngine(
            meta_learning_strategy="transfer",
            base_model_type="lgbm"
        )
        self.meta_model.fit(X_dict, y_dict, seg_dict)
        logger.info("Meta-learning model trained")
    
    def _train_deep_model(
        self,
        transactions_df: pd.DataFrame,
        y: np.ndarray
    ):
        """Train deep learning model on sequences."""
        if not DEEP_LEARNING_AVAILABLE:
            logger.warning("Deep learning not available, skipping")
            return
        
        # Prepare sequences
        sequences = self._prepare_sequences(transactions_df)
        
        if len(sequences) == 0:
            logger.warning("No sequences available for deep learning")
            return
        
        # Train LSTM model
        try:
            from src.use_cases.behavioral_scoring.deep_scoring import LSTMScoringModel
            
            self.deep_model = LSTMScoringModel(
                input_size=sequences[0].shape[1],
                hidden_size=128,
                num_layers=2
            )
            
            # Match sequences with labels
            # NOTE: This assumes y is aligned with unique users in the same order
            # as they appear in transactions_df. If using external labels,
            # ensure they are properly aligned or use a DataFrame merge instead.
            user_ids = transactions_df[self.user_col].unique()
            
            # Build label lookup from y (which should correspond to users in order)
            if len(y) != len(user_ids):
                logger.warning(
                    f"Label count ({len(y)}) doesn't match user count ({len(user_ids)}). "
                    "Using available labels. Ensure y is aligned with user order."
                )
            
            # Create mapping ensuring we only use available labels
            # Positional mapping: y[i] corresponds to user_ids[i]
            user_to_label = {}
            for i, user_id in enumerate(user_ids):
                if i < len(y):
                    user_to_label[user_id] = y[i]
                else:
                    user_to_label[user_id] = 0  # Default for missing labels
            
            sequence_labels = np.array([
                user_to_label.get(user_id, 0)
                for user_id in user_ids[:len(sequences)]
            ])
            
            self.deep_model.fit(sequences, sequence_labels, epochs=20, verbose=False)
            logger.info("Deep learning model trained")
            
        except Exception as e:
            logger.error(f"Failed to train deep model: {e}")
            self.deep_model = None
    
    def _prepare_sequences(
        self,
        transactions_df: pd.DataFrame,
        max_length: int = 50
    ) -> List[np.ndarray]:
        """Prepare transaction sequences for deep learning."""
        sequences = []
        
        for user_id in transactions_df[self.user_col].unique():
            user_trans = transactions_df[transactions_df[self.user_col] == user_id]
            
            # Extract sequence features
            seq_features = user_trans[[self.amount_col]].values
            
            if len(seq_features) > 0:
                sequences.append(seq_features)
        
        return sequences
    
    def score_single(
        self,
        user_id: str,
        transaction_data: Union[pd.DataFrame, Dict],
        segment: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Score single user in real-time.
        
        Args:
            user_id: User identifier
            transaction_data: Transaction data for user
            segment: User segment (optional)
            
        Returns:
            Scoring results
            
        Example:
            >>> result = pipeline.score_single(
            ...     user_id="12345",
            ...     transaction_data=user_transactions,
            ...     segment="premium"
            ... )
            >>> print(f"Risk score: {result['score']:.3f}")
        """
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted")
        
        # Convert to DataFrame if needed
        if isinstance(transaction_data, dict):
            transaction_data = pd.DataFrame([transaction_data])
        
        # Add user_id if not present
        if self.user_col not in transaction_data.columns:
            transaction_data[self.user_col] = user_id
        
        # Extract features
        features_df = self._extract_features(transaction_data)
        
        # Get feature vector
        feature_cols = [col for col in features_df.columns 
                       if col != self.user_col]
        X = features_df[feature_cols].values
        
        # Get ensemble score
        segments_array = np.array([segment] * len(X)) if segment else None
        ensemble_score = self.ensemble_model.predict(X, segments=segments_array)[0]
        
        # Get meta score if available
        meta_score = None
        if self.meta_model is not None and segment is not None:
            try:
                meta_score = self.meta_model.predict(X, segment_id=segment)[0]
            except:
                pass
        
        # Get deep score if available
        deep_score = None
        if self.deep_model is not None:
            try:
                sequences = self._prepare_sequences(transaction_data)
                if len(sequences) > 0:
                    deep_score = self.deep_model.predict(sequences)[0]
            except:
                pass
        
        # Combine scores
        scores = [ensemble_score]
        if meta_score is not None:
            scores.append(meta_score)
        if deep_score is not None:
            scores.append(deep_score)
        
        final_score = np.mean(scores)
        
        result = {
            'user_id': user_id,
            'score': float(final_score),
            'ensemble_score': float(ensemble_score),
            'meta_score': float(meta_score) if meta_score is not None else None,
            'deep_score': float(deep_score) if deep_score is not None else None,
            'segment': segment,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def score_batch(
        self,
        transactions_df: pd.DataFrame,
        segments_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Score batch of users.
        
        Args:
            transactions_df: Transaction data
            segments_df: Segment assignments (optional)
            
        Returns:
            DataFrame with scores
            
        Example:
            >>> scores_df = pipeline.score_batch(test_transactions)
            >>> print(scores_df[['user_id', 'score']].head())
        """
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted")
        
        logger.info(f"Batch scoring for {transactions_df[self.user_col].nunique()} users")
        
        # Preprocess
        transactions_clean = self.preprocessor.transform(transactions_df)
        
        # Extract features
        features_df = self._extract_features(transactions_clean)
        
        # Merge segments if available
        if segments_df is not None:
            features_df = features_df.merge(
                segments_df,
                on=self.user_col,
                how='left'
            )
            segments = features_df['segment'].values
        else:
            segments = None
        
        # Prepare features
        feature_cols = [col for col in features_df.columns 
                       if col not in [self.user_col, 'segment']]
        X = features_df[feature_cols].values
        
        # Get predictions
        scores = self.ensemble_model.predict(X, segments=segments)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            self.user_col: features_df[self.user_col],
            'score': scores,
            'timestamp': datetime.now()
        })
        
        if segments is not None:
            results_df['segment'] = segments
        
        logger.info("Batch scoring completed")
        
        return results_df
    
    def monitor_performance(
        self,
        transactions_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        segments_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Monitor model performance on new data.
        
        Args:
            transactions_df: New transaction data
            labels_df: True labels
            segments_df: Segment assignments (optional)
            
        Returns:
            Performance metrics
            
        Example:
            >>> metrics = pipeline.monitor_performance(
            ...     new_transactions, new_labels
            ... )
            >>> if metrics['auc_roc'] < 0.7:
            ...     print("Model performance degraded, consider retraining")
        """
        # Score batch
        scores_df = self.score_batch(transactions_df, segments_df)
        
        # Merge with labels
        results = scores_df.merge(labels_df, on=self.user_col, how='inner')
        
        # Log metrics
        segments = results['segment'].values if 'segment' in results.columns else None
        
        if segments is not None:
            # Per-segment monitoring
            for segment in np.unique(segments):
                segment_mask = segments == segment
                if np.sum(segment_mask) >= 10:
                    self.monitor.log_metrics(
                        y_true=results['label'].values[segment_mask],
                        y_pred=results['score'].values[segment_mask],
                        segment=segment
                    )
        
        # Global monitoring
        metrics = self.monitor.log_metrics(
            y_true=results['label'].values,
            y_pred=results['score'].values
        )
        
        # Check for degradation
        degraded = self.monitor.detect_performance_degradation()
        metrics['performance_degraded'] = degraded
        
        logger.info(f"Performance monitoring: AUC={metrics['auc_roc']:.4f}, Degraded={degraded}")
        
        return metrics
    
    def save(self, path: Union[str, Path]):
        """Save pipeline to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save pipeline state
        state = {
            'config': {
                'feature_config': self.feature_config,
                'use_deep_learning': self.use_deep_learning,
                'use_meta_learning': self.use_meta_learning,
                'ensemble_type': self.ensemble_type,
                'user_col': self.user_col,
                'timestamp_col': self.timestamp_col,
                'amount_col': self.amount_col
            },
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(state, path / "pipeline_state.pkl")
        
        # Save models
        if self.ensemble_model is not None:
            self.ensemble_model.save(path / "ensemble_model.pkl")
        
        if self.meta_model is not None:
            self.meta_model.save(path / "meta_model.pkl")
        
        if self.deep_model is not None:
            self.deep_model.save(path / "deep_model.pt")
        
        # Save monitor
        joblib.dump(self.monitor, path / "monitor.pkl")
        
        logger.info(f"Pipeline saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'BehavioralScoringPipeline':
        """Load pipeline from disk."""
        path = Path(path)
        
        # Load state
        state = joblib.load(path / "pipeline_state.pkl")
        
        # Create pipeline
        pipeline = cls(**state['config'])
        pipeline.is_fitted = state['is_fitted']
        
        # Load models
        if (path / "ensemble_model.pkl").exists():
            pipeline.ensemble_model = BehavioralEnsembleScorer.load(path / "ensemble_model.pkl")
        
        if (path / "meta_model.pkl").exists():
            pipeline.meta_model = MetaScoringEngine.load(path / "meta_model.pkl")
        
        if (path / "deep_model.pt").exists() and DEEP_LEARNING_AVAILABLE:
            from src.use_cases.behavioral_scoring.deep_scoring import LSTMScoringModel
            pipeline.deep_model = LSTMScoringModel.load(path / "deep_model.pt")
        
        # Load monitor
        if (path / "monitor.pkl").exists():
            pipeline.monitor = joblib.load(path / "monitor.pkl")
        
        logger.info(f"Pipeline loaded from {path}")
        
        return pipeline


# Example usage
if __name__ == "__main__":
    # Example: Complete workflow
    logger.info("=== Behavioral Scoring Pipeline Example ===")
    
    # Generate synthetic data
    np.random.seed(42)
    n_users = 1000
    n_transactions_per_user = 50
    
    transactions_data = []
    for user_id in range(n_users):
        for _ in range(np.random.randint(10, n_transactions_per_user)):
            transactions_data.append({
                'user_id': f'user_{user_id}',
                'timestamp': datetime.now() - timedelta(days=np.random.randint(0, 365)),
                'amount': np.random.lognormal(3, 1),
                'category': np.random.choice(['groceries', 'entertainment', 'bills', 'shopping'])
            })
    
    transactions_df = pd.DataFrame(transactions_data)
    
    # Generate labels (synthetic risk)
    labels_df = pd.DataFrame({
        'user_id': [f'user_{i}' for i in range(n_users)],
        'label': np.random.binomial(1, 0.1, n_users)
    })
    
    # Generate segments
    segments_df = pd.DataFrame({
        'user_id': [f'user_{i}' for i in range(n_users)],
        'segment': np.random.choice(['premium', 'standard', 'basic'], n_users)
    })
    
    # Initialize pipeline
    pipeline = BehavioralScoringPipeline(
        use_deep_learning=False,  # Set to True if PyTorch available
        use_meta_learning=True,
        ensemble_type="stacking"
    )
    
    # Train pipeline
    logger.info("Training pipeline...")
    history = pipeline.fit(
        transactions_df=transactions_df,
        labels_df=labels_df,
        segments_df=segments_df,
        validation_strategy="temporal",
        n_splits=3
    )
    
    logger.info(f"Training completed - Validation AUC: {history['val_auc']:.4f}")
    
    # Real-time scoring example
    test_user_id = 'user_0'
    test_transactions = transactions_df[transactions_df['user_id'] == test_user_id]
    test_segment = segments_df[segments_df['user_id'] == test_user_id]['segment'].values[0]
    
    result = pipeline.score_single(
        user_id=test_user_id,
        transaction_data=test_transactions,
        segment=test_segment
    )
    
    logger.info(f"Single user score: {result['score']:.4f}")
    
    # Batch scoring example
    test_transactions_batch = transactions_df[transactions_df['user_id'].isin([f'user_{i}' for i in range(10)])]
    test_segments_batch = segments_df[segments_df['user_id'].isin([f'user_{i}' for i in range(10)])]
    
    scores_df = pipeline.score_batch(test_transactions_batch, test_segments_batch)
    logger.info(f"Batch scoring completed for {len(scores_df)} users")
    
    # Performance monitoring
    metrics = pipeline.monitor_performance(
        transactions_df=test_transactions_batch,
        labels_df=labels_df[labels_df['user_id'].isin([f'user_{i}' for i in range(10)])],
        segments_df=test_segments_batch
    )
    
    logger.info(f"Monitoring metrics - AUC: {metrics['auc_roc']:.4f}")
    
    logger.info("=== Example completed ===")
