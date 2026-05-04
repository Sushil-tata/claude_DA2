"""
End-to-End Fraud Detection Pipeline

Orchestrates complete fraud detection workflow:
- Data ingestion and preprocessing
- Graph construction from transactions
- Feature engineering (transaction + graph features)
- Model training (supervised + unsupervised)
- Real-time scoring interface
- Model monitoring and alerting
- A/B testing framework for new models
- Complete production deployment

Example:
    >>> from src.use_cases.fraud_detection.fraud_pipeline import FraudDetectionPipeline
    >>> 
    >>> # Initialize pipeline
    >>> pipeline = FraudDetectionPipeline(config_path='config/fraud_config.yaml')
    >>> 
    >>> # Train pipeline
    >>> pipeline.fit(transactions_df, labels=fraud_labels)
    >>> 
    >>> # Real-time scoring
    >>> fraud_score = pipeline.predict_single_transaction(transaction)
    >>> 
    >>> # Batch scoring
    >>> scores = pipeline.predict_batch(transactions_batch)
    >>> 
    >>> # Model monitoring
    >>> metrics = pipeline.get_monitoring_metrics()
"""

import joblib
import numpy as np
import pandas as pd
import networkx as nx
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from collections import defaultdict
import warnings

from loguru import logger

from src.use_cases.fraud_detection.graph_builder import GraphBuilder
from src.use_cases.fraud_detection.supervised_fraud import (
    FraudClassifier, FraudFeatureEngineer
)
from src.use_cases.fraud_detection.anomaly_detection import (
    EnsembleAnomalyDetector, AnomalyScoreCalibrator
)
from src.use_cases.fraud_detection.graph_embeddings import (
    Node2VecEmbeddings, CommunityEmbeddings
)
from src.use_cases.fraud_detection.risk_propagation import (
    PageRankRiskPropagation, IterativeRiskScoring, FraudRingDetector
)


class FraudPipelineConfig:
    """Configuration for fraud detection pipeline."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize pipeline configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {}
        
        # Data processing config
        self.account_col = config.get('account_col', 'account_id')
        self.merchant_col = config.get('merchant_col', 'merchant_id')
        self.device_col = config.get('device_col', 'device_id')
        self.ip_col = config.get('ip_col', 'ip_address')
        self.amount_col = config.get('amount_col', 'amount')
        self.timestamp_col = config.get('timestamp_col', 'timestamp')
        self.fraud_col = config.get('fraud_col', 'is_fraud')
        
        # Model config
        self.supervised_model = config.get('supervised_model', 'lightgbm')
        self.use_anomaly_detection = config.get('use_anomaly_detection', True)
        self.use_graph_features = config.get('use_graph_features', True)
        self.use_risk_propagation = config.get('use_risk_propagation', True)
        
        # Graph config
        self.graph_embedding_dim = config.get('graph_embedding_dim', 64)
        self.use_communities = config.get('use_communities', True)
        
        # Scoring config
        self.score_threshold = config.get('score_threshold', 0.5)
        self.target_recall = config.get('target_recall', 0.8)
        self.max_latency_ms = config.get('max_latency_ms', 100)
        
        # Monitoring config
        self.enable_monitoring = config.get('enable_monitoring', True)
        self.alert_threshold = config.get('alert_threshold', 0.9)
        self.monitoring_window_days = config.get('monitoring_window_days', 7)


class FraudDetectionPipeline:
    """
    Production-ready fraud detection pipeline.
    
    Integrates all fraud detection components into a unified pipeline
    with training, inference, monitoring, and A/B testing capabilities.
    
    Example:
        >>> # Training
        >>> pipeline = FraudDetectionPipeline()
        >>> pipeline.fit(transactions_df)
        >>> pipeline.save('models/fraud_pipeline.pkl')
        >>> 
        >>> # Inference
        >>> pipeline = FraudDetectionPipeline.load('models/fraud_pipeline.pkl')
        >>> score = pipeline.predict_single_transaction(new_transaction)
        >>> 
        >>> # Monitoring
        >>> metrics = pipeline.get_monitoring_metrics()
        >>> alerts = pipeline.check_alerts()
    """
    
    def __init__(self, config: Optional[FraudPipelineConfig] = None):
        """
        Initialize fraud detection pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or FraudPipelineConfig()
        
        # Components
        self.graph_builder = GraphBuilder()
        self.feature_engineer = FraudFeatureEngineer()
        self.supervised_classifier: Optional[FraudClassifier] = None
        self.anomaly_detector: Optional[EnsembleAnomalyDetector] = None
        self.graph_embedder: Optional[Node2VecEmbeddings] = None
        self.community_detector: Optional[CommunityEmbeddings] = None
        self.risk_propagator: Optional[PageRankRiskPropagation] = None
        self.fraud_ring_detector: Optional[FraudRingDetector] = None
        
        # State
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.graph = None
        self.training_metrics: Dict[str, Any] = {}
        self.monitoring_data: Dict[str, List[Any]] = defaultdict(list)
        
        logger.info("Initialized FraudDetectionPipeline")
    
    def fit(
        self,
        transactions: pd.DataFrame,
        labels: Optional[pd.Series] = None,
        validation_split: float = 0.2
    ) -> 'FraudDetectionPipeline':
        """
        Fit the complete fraud detection pipeline.
        
        Args:
            transactions: Transaction DataFrame
            labels: Fraud labels (if None, uses config.fraud_col)
            validation_split: Fraction for validation
            
        Returns:
            Self
            
        Example:
            >>> pipeline.fit(transactions_df)
        """
        logger.info(f"Training fraud detection pipeline on {len(transactions)} transactions")
        
        # Extract labels
        if labels is None:
            if self.config.fraud_col not in transactions.columns:
                raise ValueError(f"Fraud label column '{self.config.fraud_col}' not found")
            labels = transactions[self.config.fraud_col]
        
        fraud_rate = labels.mean()
        logger.info(f"Fraud rate: {fraud_rate:.4%}")
        
        # Split data
        n_val = int(len(transactions) * validation_split)
        indices = np.random.permutation(len(transactions))
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]
        
        train_df = transactions.iloc[train_idx].reset_index(drop=True)
        val_df = transactions.iloc[val_idx].reset_index(drop=True)
        y_train = labels.iloc[train_idx].reset_index(drop=True)
        y_val = labels.iloc[val_idx].reset_index(drop=True)
        
        # Step 1: Build graph
        logger.info("Step 1: Building transaction graph")
        self.graph = self.graph_builder.build_from_transactions(
            transactions,
            account_col=self.config.account_col,
            merchant_col=self.config.merchant_col,
            device_col=self.config.device_col,
            ip_col=self.config.ip_col,
            amount_col=self.config.amount_col,
            timestamp_col=self.config.timestamp_col,
            fraud_col=self.config.fraud_col
        )
        
        # Step 2: Create features
        logger.info("Step 2: Engineering features")
        X_train = self._create_features(train_df, fit=True)
        X_val = self._create_features(val_df, fit=False)
        
        # Step 3: Train supervised model
        logger.info("Step 3: Training supervised classifier")
        self.supervised_classifier = FraudClassifier(
            model_type=self.config.supervised_model,
            calibrate=True
        )
        self.supervised_classifier.fit(X_train, y_train, X_val, y_val)
        
        # Optimize threshold
        optimal_threshold = self.supervised_classifier.optimize_threshold(
            X_val, y_val,
            target_recall=self.config.target_recall
        )
        logger.info(f"Optimal threshold: {optimal_threshold:.4f}")
        
        # Step 4: Train anomaly detector
        if self.config.use_anomaly_detection:
            logger.info("Step 4: Training anomaly detector")
            self.anomaly_detector = EnsembleAnomalyDetector(
                use_isolation_forest=True,
                use_lof=True,
                use_autoencoder=True
            )
            # Train on normal transactions only for better anomaly detection
            normal_idx = y_train == 0
            self.anomaly_detector.fit(
                X_train[normal_idx],
                y_train[normal_idx]
            )
        
        # Step 5: Train graph embeddings
        if self.config.use_graph_features:
            logger.info("Step 5: Training graph embeddings")
            self.graph_embedder = Node2VecEmbeddings(
                dimensions=self.config.graph_embedding_dim
            )
            self.graph_embedder.fit(self.graph)
            
            if self.config.use_communities:
                self.community_detector = CommunityEmbeddings()
                self.community_detector.detect_communities(self.graph)
        
        # Step 6: Setup risk propagation
        if self.config.use_risk_propagation:
            logger.info("Step 6: Setting up risk propagation")
            self.risk_propagator = PageRankRiskPropagation()
            self.fraud_ring_detector = FraudRingDetector()
        
        # Evaluate on validation set
        self._evaluate(X_val, y_val)
        
        self.is_fitted = True
        logger.info("Pipeline training completed successfully")
        
        return self
    
    def _create_features(
        self,
        transactions: pd.DataFrame,
        fit: bool = False
    ) -> np.ndarray:
        """
        Create features from transactions.
        
        Args:
            transactions: Transaction DataFrame
            fit: Whether to fit feature transformations
            
        Returns:
            Feature matrix
        """
        features_list = []
        
        # Velocity features
        velocity_features = self.feature_engineer.create_velocity_features(
            transactions,
            account_col=self.config.account_col,
            timestamp_col=self.config.timestamp_col,
            amount_col=self.config.amount_col
        )
        features_list.append(velocity_features)
        
        # Anomaly scores
        anomaly_features = self.feature_engineer.create_anomaly_scores(
            transactions,
            amount_col=self.config.amount_col,
            merchant_col=self.config.merchant_col
        )
        features_list.append(anomaly_features)
        
        # Graph features (if enabled and graph exists)
        if self.config.use_graph_features and self.graph is not None:
            graph_features = self._extract_graph_features(transactions)
            if graph_features is not None:
                features_list.append(graph_features)
        
        # Combine all features
        X = pd.concat(features_list, axis=1)
        
        if fit:
            self.feature_names = X.columns.tolist()
        
        return X.values
    
    def _extract_graph_features(
        self,
        transactions: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """Extract graph-based features for transactions."""
        if self.graph is None:
            return None
        
        features = {}
        
        # Node degree features
        degree_dict = dict(self.graph.degree())
        features['node_degree'] = transactions[self.config.account_col].map(
            degree_dict
        ).fillna(0)
        
        # PageRank
        pagerank = nx.pagerank(self.graph)
        features['pagerank'] = transactions[self.config.account_col].map(
            pagerank
        ).fillna(0)
        
        # Clustering coefficient
        clustering = nx.clustering(self.graph)
        features['clustering'] = transactions[self.config.account_col].map(
            clustering
        ).fillna(0)
        
        # Graph embeddings (if available)
        if self.graph_embedder is not None:
            account_embeddings = []
            for account in transactions[self.config.account_col]:
                emb = self.graph_embedder.get_embedding(account)
                if emb is not None:
                    account_embeddings.append(emb)
                else:
                    account_embeddings.append(
                        np.zeros(self.config.graph_embedding_dim)
                    )
            
            account_embeddings = np.array(account_embeddings)
            for i in range(account_embeddings.shape[1]):
                features[f'graph_emb_{i}'] = account_embeddings[:, i]
        
        # Community features (if available)
        if self.community_detector is not None and self.community_detector.communities_ is not None:
            features['community_id'] = transactions[self.config.account_col].map(
                self.community_detector.communities_
            ).fillna(-1)
        
        return pd.DataFrame(features)
    
    def predict_single_transaction(
        self,
        transaction: Union[pd.Series, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Predict fraud score for a single transaction (real-time scoring).
        
        Args:
            transaction: Transaction data
            
        Returns:
            Dictionary with fraud score and metadata
            
        Example:
            >>> result = pipeline.predict_single_transaction({
            ...     'account_id': 'acc_123',
            ...     'amount': 1500.00,
            ...     'merchant_id': 'merch_456',
            ...     'timestamp': '2024-01-15 10:30:00'
            ... })
            >>> print(f"Fraud score: {result['fraud_score']:.3f}")
        """
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        start_time = datetime.now()
        
        # Convert to DataFrame
        if isinstance(transaction, dict):
            transaction = pd.Series(transaction)
        
        transaction_df = transaction.to_frame().T
        
        # Create features
        X = self._create_features(transaction_df, fit=False)
        
        # Supervised score
        supervised_score = float(self.supervised_classifier.predict_proba(X)[0])
        
        # Anomaly score (if available)
        anomaly_score = None
        if self.anomaly_detector is not None:
            anomaly_score = float(
                self.anomaly_detector.predict_scores(X)[0]
            )
        
        # Combine scores
        if anomaly_score is not None:
            # Weighted average: 70% supervised, 30% anomaly
            final_score = 0.7 * supervised_score + 0.3 * anomaly_score
        else:
            final_score = supervised_score
        
        # Check latency
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Make decision
        is_fraud = final_score >= self.config.score_threshold
        
        result = {
            'fraud_score': final_score,
            'supervised_score': supervised_score,
            'anomaly_score': anomaly_score,
            'is_fraud': is_fraud,
            'threshold': self.config.score_threshold,
            'latency_ms': latency_ms,
            'timestamp': datetime.now().isoformat()
        }
        
        # Log for monitoring
        if self.config.enable_monitoring:
            self._log_prediction(result)
        
        # Check if latency exceeds limit
        if latency_ms > self.config.max_latency_ms:
            logger.warning(
                f"Prediction latency {latency_ms:.1f}ms exceeds "
                f"limit {self.config.max_latency_ms}ms"
            )
        
        return result
    
    def predict_batch(
        self,
        transactions: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Predict fraud scores for batch of transactions.
        
        Args:
            transactions: Transaction DataFrame
            
        Returns:
            DataFrame with fraud scores
            
        Example:
            >>> scores = pipeline.predict_batch(transactions_df)
        """
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        logger.info(f"Scoring batch of {len(transactions)} transactions")
        
        # Create features
        X = self._create_features(transactions, fit=False)
        
        # Supervised scores
        supervised_scores = self.supervised_classifier.predict_proba(X)
        
        # Anomaly scores
        if self.anomaly_detector is not None:
            anomaly_scores = self.anomaly_detector.predict_scores(X)
            # Combine scores
            final_scores = 0.7 * supervised_scores + 0.3 * anomaly_scores
        else:
            final_scores = supervised_scores
        
        # Create result DataFrame
        results = pd.DataFrame({
            'fraud_score': final_scores,
            'supervised_score': supervised_scores,
            'is_fraud': final_scores >= self.config.score_threshold
        })
        
        if self.anomaly_detector is not None:
            results['anomaly_score'] = anomaly_scores
        
        return results
    
    def detect_fraud_rings(
        self,
        transactions: Optional[pd.DataFrame] = None,
        known_fraud_nodes: Optional[Set[Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect fraud rings in transaction network.
        
        Args:
            transactions: Optional transaction data to update graph
            known_fraud_nodes: Known fraud account IDs
            
        Returns:
            List of detected fraud rings
            
        Example:
            >>> rings = pipeline.detect_fraud_rings()
            >>> for ring in rings[:5]:
            ...     print(f"Ring size: {ring['size']}, Risk: {ring['avg_risk']:.3f}")
        """
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        if self.fraud_ring_detector is None:
            raise ValueError("Fraud ring detection not enabled")
        
        # Update graph if new transactions provided
        if transactions is not None:
            self.graph = self.graph_builder.build_from_transactions(
                transactions,
                account_col=self.config.account_col,
                merchant_col=self.config.merchant_col,
                device_col=self.config.device_col,
                ip_col=self.config.ip_col
            )
        
        # If no known fraud nodes provided, use high-risk nodes from recent predictions
        if known_fraud_nodes is None:
            # Use nodes with high fraud scores (if available from monitoring)
            known_fraud_nodes = set()
        
        # Detect rings
        rings = self.fraud_ring_detector.detect(
            self.graph,
            known_fraud_nodes
        )
        
        logger.info(f"Detected {len(rings)} fraud rings")
        return rings
    
    def propagate_risk(
        self,
        known_fraud_accounts: Set[Any],
        method: str = 'pagerank'
    ) -> Dict[Any, float]:
        """
        Propagate risk through network from known fraud accounts.
        
        Args:
            known_fraud_accounts: Set of known fraud account IDs
            method: Propagation method ('pagerank', 'iterative')
            
        Returns:
            Dictionary of account IDs to risk scores
            
        Example:
            >>> fraud_accounts = {'acc_123', 'acc_456'}
            >>> risk_scores = pipeline.propagate_risk(fraud_accounts)
        """
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        if self.risk_propagator is None:
            raise ValueError("Risk propagation not enabled")
        
        initial_scores = {
            account: 1.0 for account in known_fraud_accounts
            if account in self.graph.nodes()
        }
        
        if not initial_scores:
            logger.warning("No known fraud accounts found in graph")
            return {}
        
        if method == 'pagerank':
            risk_scores = self.risk_propagator.propagate(
                self.graph,
                initial_scores
            )
        elif method == 'iterative':
            scorer = IterativeRiskScoring()
            risk_scores = scorer.score(self.graph, set(initial_scores.keys()))
        else:
            raise ValueError(f"Unknown propagation method: {method}")
        
        return risk_scores
    
    def _evaluate(self, X_val: np.ndarray, y_val: np.ndarray):
        """Evaluate pipeline on validation set."""
        from sklearn.metrics import (
            roc_auc_score, average_precision_score,
            precision_recall_curve, confusion_matrix
        )
        
        # Supervised predictions
        supervised_scores = self.supervised_classifier.predict_proba(X_val)
        supervised_preds = self.supervised_classifier.predict(X_val)
        
        # Metrics
        self.training_metrics = {
            'supervised_auc_roc': float(roc_auc_score(y_val, supervised_scores)),
            'supervised_auc_pr': float(average_precision_score(y_val, supervised_scores)),
            'n_train': len(X_val),
            'fraud_rate': float(y_val.mean())
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_val, supervised_preds)
        self.training_metrics['confusion_matrix'] = cm.tolist()
        
        if len(cm) == 2:
            tn, fp, fn, tp = cm.ravel()
            self.training_metrics['precision'] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0
            self.training_metrics['recall'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0
            self.training_metrics['false_positive_rate'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0
        
        logger.info(f"Validation metrics: {self.training_metrics}")
    
    def _log_prediction(self, result: Dict[str, Any]):
        """Log prediction for monitoring."""
        self.monitoring_data['scores'].append(result['fraud_score'])
        self.monitoring_data['timestamps'].append(result['timestamp'])
        self.monitoring_data['latencies'].append(result['latency_ms'])
        self.monitoring_data['decisions'].append(result['is_fraud'])
    
    def get_monitoring_metrics(self) -> Dict[str, Any]:
        """
        Get monitoring metrics for recent predictions.
        
        Returns:
            Dictionary of monitoring metrics
            
        Example:
            >>> metrics = pipeline.get_monitoring_metrics()
            >>> print(f"Average latency: {metrics['avg_latency_ms']:.1f}ms")
        """
        if not self.monitoring_data['scores']:
            return {
                'message': 'No monitoring data available',
                'n_predictions': 0
            }
        
        scores = self.monitoring_data['scores']
        latencies = self.monitoring_data['latencies']
        decisions = self.monitoring_data['decisions']
        
        metrics = {
            'n_predictions': len(scores),
            'avg_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'fraud_rate': float(np.mean(decisions)),
            'avg_latency_ms': float(np.mean(latencies)),
            'max_latency_ms': float(np.max(latencies)),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'p99_latency_ms': float(np.percentile(latencies, 99))
        }
        
        return metrics
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """
        Check for monitoring alerts.
        
        Returns:
            List of active alerts
            
        Example:
            >>> alerts = pipeline.check_alerts()
            >>> for alert in alerts:
            ...     print(f"{alert['type']}: {alert['message']}")
        """
        alerts = []
        
        if not self.monitoring_data['scores']:
            return alerts
        
        # Check fraud rate
        recent_decisions = self.monitoring_data['decisions'][-1000:]
        fraud_rate = np.mean(recent_decisions)
        
        if fraud_rate > self.config.alert_threshold:
            alerts.append({
                'type': 'HIGH_FRAUD_RATE',
                'severity': 'CRITICAL',
                'message': f'Fraud rate {fraud_rate:.2%} exceeds threshold {self.config.alert_threshold:.2%}',
                'value': fraud_rate
            })
        
        # Check latency
        recent_latencies = self.monitoring_data['latencies'][-100:]
        avg_latency = np.mean(recent_latencies)
        
        if avg_latency > self.config.max_latency_ms:
            alerts.append({
                'type': 'HIGH_LATENCY',
                'severity': 'WARNING',
                'message': f'Average latency {avg_latency:.1f}ms exceeds limit {self.config.max_latency_ms}ms',
                'value': avg_latency
            })
        
        # Check score distribution drift
        recent_scores = self.monitoring_data['scores'][-1000:]
        if len(recent_scores) >= 100:
            recent_avg = np.mean(recent_scores[-100:])
            historical_avg = np.mean(recent_scores[:-100])
            drift = abs(recent_avg - historical_avg) / (historical_avg + 1e-10)
            
            if drift > 0.3:  # 30% drift
                alerts.append({
                    'type': 'SCORE_DRIFT',
                    'severity': 'WARNING',
                    'message': f'Score distribution drift detected: {drift:.1%}',
                    'value': drift
                })
        
        return alerts
    
    def save(self, path: Union[str, Path]):
        """
        Save pipeline to disk.
        
        Args:
            path: Save path
            
        Example:
            >>> pipeline.save('models/fraud_pipeline.pkl')
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        pipeline_data = {
            'config': self.config,
            'graph_builder': self.graph_builder,
            'feature_engineer': self.feature_engineer,
            'supervised_classifier': self.supervised_classifier,
            'anomaly_detector': self.anomaly_detector,
            'graph_embedder': self.graph_embedder,
            'community_detector': self.community_detector,
            'risk_propagator': self.risk_propagator,
            'fraud_ring_detector': self.fraud_ring_detector,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics
        }
        
        joblib.dump(pipeline_data, path)
        logger.info(f"Pipeline saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'FraudDetectionPipeline':
        """
        Load pipeline from disk.
        
        Args:
            path: Model path
            
        Returns:
            Loaded FraudDetectionPipeline
            
        Example:
            >>> pipeline = FraudDetectionPipeline.load('models/fraud_pipeline.pkl')
        """
        pipeline_data = joblib.load(path)
        
        pipeline = cls(config=pipeline_data['config'])
        pipeline.graph_builder = pipeline_data['graph_builder']
        pipeline.feature_engineer = pipeline_data['feature_engineer']
        pipeline.supervised_classifier = pipeline_data['supervised_classifier']
        pipeline.anomaly_detector = pipeline_data['anomaly_detector']
        pipeline.graph_embedder = pipeline_data['graph_embedder']
        pipeline.community_detector = pipeline_data['community_detector']
        pipeline.risk_propagator = pipeline_data['risk_propagator']
        pipeline.fraud_ring_detector = pipeline_data['fraud_ring_detector']
        pipeline.is_fitted = pipeline_data['is_fitted']
        pipeline.feature_names = pipeline_data['feature_names']
        pipeline.training_metrics = pipeline_data['training_metrics']
        
        logger.info(f"Pipeline loaded from {path}")
        return pipeline


class ABTestingFramework:
    """
    A/B testing framework for fraud models.
    
    Allows comparing performance of different models or configurations
    in production with statistical significance testing.
    
    Example:
        >>> ab_test = ABTestingFramework()
        >>> ab_test.add_variant('control', pipeline_v1)
        >>> ab_test.add_variant('treatment', pipeline_v2)
        >>> 
        >>> # Route traffic
        >>> variant = ab_test.get_variant(user_id='user_123')
        >>> result = variant.predict_single_transaction(transaction)
        >>> 
        >>> # Analyze results
        >>> analysis = ab_test.analyze_results()
    """
    
    def __init__(self, traffic_split: Optional[Dict[str, float]] = None):
        """
        Initialize A/B testing framework.
        
        Args:
            traffic_split: Dictionary of variant names to traffic proportions
        """
        self.variants: Dict[str, FraudDetectionPipeline] = {}
        self.traffic_split = traffic_split or {'control': 0.5, 'treatment': 0.5}
        self.results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        logger.info("Initialized ABTestingFramework")
    
    def add_variant(
        self,
        name: str,
        pipeline: FraudDetectionPipeline,
        traffic_proportion: Optional[float] = None
    ):
        """
        Add model variant.
        
        Args:
            name: Variant name
            pipeline: Fraud detection pipeline
            traffic_proportion: Proportion of traffic (optional)
        """
        self.variants[name] = pipeline
        
        if traffic_proportion is not None:
            self.traffic_split[name] = traffic_proportion
        
        logger.info(f"Added variant '{name}' with {traffic_proportion or 'default'} traffic")
    
    def get_variant(self, user_id: str) -> Tuple[str, FraudDetectionPipeline]:
        """
        Get variant for user (deterministic based on user_id).
        
        Args:
            user_id: User identifier
            
        Returns:
            Tuple of (variant_name, pipeline)
        """
        # Hash user_id to get consistent variant assignment
        import hashlib
        hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        rand_val = (hash_val % 10000) / 10000
        
        cumulative = 0
        for name, proportion in self.traffic_split.items():
            cumulative += proportion
            if rand_val < cumulative and name in self.variants:
                return name, self.variants[name]
        
        # Default to first variant
        default_name = list(self.variants.keys())[0]
        return default_name, self.variants[default_name]
    
    def log_result(
        self,
        variant_name: str,
        prediction: Dict[str, Any],
        ground_truth: Optional[bool] = None
    ):
        """
        Log prediction result for analysis.
        
        Args:
            variant_name: Variant name
            prediction: Prediction result
            ground_truth: Optional ground truth label
        """
        result = prediction.copy()
        result['ground_truth'] = ground_truth
        result['variant'] = variant_name
        
        self.results[variant_name].append(result)
    
    def analyze_results(self) -> Dict[str, Any]:
        """
        Analyze A/B test results.
        
        Returns:
            Analysis results with statistical significance
        """
        from scipy import stats
        
        logger.info("Analyzing A/B test results")
        
        analysis = {}
        
        for variant_name, results in self.results.items():
            if not results:
                continue
            
            # Extract metrics
            scores = [r['fraud_score'] for r in results]
            latencies = [r['latency_ms'] for r in results]
            
            # Compute metrics
            analysis[variant_name] = {
                'n_samples': len(results),
                'avg_score': float(np.mean(scores)),
                'std_score': float(np.std(scores)),
                'avg_latency_ms': float(np.mean(latencies)),
                'p95_latency_ms': float(np.percentile(latencies, 95))
            }
            
            # If ground truth available, compute accuracy metrics
            ground_truths = [r['ground_truth'] for r in results if r['ground_truth'] is not None]
            if ground_truths:
                decisions = [r['is_fraud'] for r in results if r['ground_truth'] is not None]
                analysis[variant_name]['accuracy'] = float(
                    np.mean([g == d for g, d in zip(ground_truths, decisions)])
                )
        
        # Statistical comparison (if 2 variants)
        variant_names = list(self.results.keys())
        if len(variant_names) == 2:
            v1, v2 = variant_names
            
            scores1 = [r['fraud_score'] for r in self.results[v1]]
            scores2 = [r['fraud_score'] for r in self.results[v2]]
            
            if scores1 and scores2:
                # T-test for score difference
                t_stat, p_value = stats.ttest_ind(scores1, scores2)
                
                analysis['comparison'] = {
                    'variants': [v1, v2],
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'mean_difference': float(np.mean(scores2) - np.mean(scores1))
                }
        
        logger.info(f"A/B test analysis completed: {len(self.results)} variants")
        return analysis
