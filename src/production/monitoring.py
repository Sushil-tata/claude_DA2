"""
Production Model Monitoring Framework

This module provides comprehensive monitoring for production ML models including
performance monitoring, data drift detection, prediction distribution monitoring,
latency tracking, error rate tracking, and alerting.

Usage Example:
    >>> from monitoring import ModelMonitor
    >>> monitor = ModelMonitor(
    ...     model_name="credit_risk_v1",
    ...     reference_data=train_df,
    ...     reference_predictions=train_preds
    ... )
    >>> # Monitor predictions
    >>> monitor.log_predictions(
    ...     features=prod_features,
    ...     predictions=prod_preds,
    ...     actuals=prod_actuals
    ... )
    >>> # Check for alerts
    >>> alerts = monitor.check_alerts()
    >>> if alerts:
    ...     monitor.send_notifications(alerts)
"""

from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
from enum import Enum
import json
import time

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, mean_squared_error, mean_absolute_error, r2_score
)
from scipy import stats

# Import drift monitor from validation module
import sys
sys.path.append(str(Path(__file__).parent.parent))
from validation.drift_monitor import DriftMonitor, DriftReport


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MetricType(str, Enum):
    """Metric type for classification vs regression."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


@dataclass
class PerformanceMetrics:
    """Container for model performance metrics."""
    timestamp: datetime
    auc: Optional[float] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    ks_statistic: Optional[float] = None
    gini: Optional[float] = None
    # Regression metrics
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None
    # Additional info
    sample_size: int = 0
    positive_rate: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'auc': self.auc,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'ks_statistic': self.ks_statistic,
            'gini': self.gini,
            'mse': self.mse,
            'rmse': self.rmse,
            'mae': self.mae,
            'r2': self.r2,
            'sample_size': self.sample_size,
            'positive_rate': self.positive_rate
        }


@dataclass
class LatencyMetrics:
    """Container for latency metrics."""
    timestamp: datetime
    p50: float  # median
    p95: float
    p99: float
    mean: float
    max: float
    min: float
    sample_size: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'p50': self.p50,
            'p95': self.p95,
            'p99': self.p99,
            'mean': self.mean,
            'max': self.max,
            'min': self.min,
            'sample_size': self.sample_size
        }


@dataclass
class VolumeMetrics:
    """Container for volume metrics."""
    timestamp: datetime
    requests_per_second: float
    total_requests: int
    error_count: int
    error_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'requests_per_second': self.requests_per_second,
            'total_requests': self.total_requests,
            'error_count': self.error_count,
            'error_rate': self.error_rate
        }


@dataclass
class Alert:
    """Container for monitoring alerts."""
    alert_id: str
    severity: AlertSeverity
    category: str  # performance, drift, latency, volume, error
    message: str
    timestamp: datetime
    metrics: Dict[str, Any] = field(default_factory=dict)
    threshold: Optional[float] = None
    current_value: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'alert_id': self.alert_id,
            'severity': self.severity.value,
            'category': self.category,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'metrics': self.metrics,
            'threshold': self.threshold,
            'current_value': self.current_value
        }


@dataclass
class SLAConfig:
    """SLA configuration."""
    max_latency_p95_ms: float = 100.0
    max_latency_p99_ms: float = 500.0
    min_availability: float = 99.9  # percentage
    max_error_rate: float = 0.1  # percentage
    min_auc: Optional[float] = None
    max_auc_degradation: float = 0.05  # 5% degradation
    max_psi_threshold: float = 0.25
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'max_latency_p95_ms': self.max_latency_p95_ms,
            'max_latency_p99_ms': self.max_latency_p99_ms,
            'min_availability': self.min_availability,
            'max_error_rate': self.max_error_rate,
            'min_auc': self.min_auc,
            'max_auc_degradation': self.max_auc_degradation,
            'max_psi_threshold': self.max_psi_threshold
        }


class ModelMonitor:
    """
    Comprehensive production model monitoring system.
    
    Monitors model performance, data drift, prediction distributions,
    latency, error rates, and generates alerts based on SLA violations.
    
    Example:
        >>> monitor = ModelMonitor(
        ...     model_name="fraud_detection",
        ...     reference_data=train_df,
        ...     sla_config=SLAConfig(max_latency_p95_ms=50)
        ... )
        >>> monitor.log_predictions(features, predictions, actuals, latency_ms)
        >>> alerts = monitor.check_alerts()
    """
    
    def __init__(
        self,
        model_name: str,
        model_version: str = "v1.0.0",
        reference_data: Optional[pd.DataFrame] = None,
        reference_predictions: Optional[np.ndarray] = None,
        reference_actuals: Optional[np.ndarray] = None,
        metric_type: MetricType = MetricType.CLASSIFICATION,
        sla_config: Optional[SLAConfig] = None,
        monitoring_window_hours: int = 24,
        storage_path: Optional[Union[str, Path]] = None
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.metric_type = metric_type
        self.sla_config = sla_config or SLAConfig()
        self.monitoring_window = timedelta(hours=monitoring_window_hours)
        
        # Storage
        if storage_path:
            self.storage_path = Path(storage_path)
            self.storage_path.mkdir(parents=True, exist_ok=True)
        else:
            self.storage_path = None
        
        # Reference data and metrics
        self.reference_data = reference_data
        self.reference_predictions = reference_predictions
        self.reference_actuals = reference_actuals
        self.reference_metrics = None
        
        if reference_predictions is not None and reference_actuals is not None:
            self.reference_metrics = self._compute_performance_metrics(
                reference_actuals,
                reference_predictions
            )
            logger.info(f"Computed reference metrics: {self.reference_metrics.to_dict()}")
        
        # Initialize drift monitor
        self.drift_monitor = None
        if reference_data is not None:
            self.drift_monitor = DriftMonitor(
                reference_data=reference_data,
                alert_threshold=0.05,
                critical_threshold=0.01
            )
        
        # Monitoring data structures
        self.prediction_log: deque = deque(maxlen=100000)
        self.latency_log: deque = deque(maxlen=10000)
        self.error_log: deque = deque(maxlen=10000)
        
        # Metrics history
        self.performance_history: List[PerformanceMetrics] = []
        self.latency_history: List[LatencyMetrics] = []
        self.volume_history: List[VolumeMetrics] = []
        self.drift_history: List[DriftReport] = []
        
        # Alerts
        self.active_alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
        
        # Request tracking
        self.total_requests = 0
        self.total_errors = 0
        self.start_time = datetime.now()
        
        logger.info(f"Initialized ModelMonitor for {model_name} {model_version}")
    
    def _compute_performance_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> PerformanceMetrics:
        """Compute performance metrics."""
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            sample_size=len(y_true)
        )
        
        if self.metric_type == MetricType.CLASSIFICATION:
            # Classification metrics
            metrics.accuracy = accuracy_score(y_true, y_pred)
            metrics.precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
            metrics.recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
            metrics.f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
            metrics.positive_rate = np.mean(y_pred)
            
            if y_pred_proba is not None:
                try:
                    metrics.auc = roc_auc_score(y_true, y_pred_proba)
                    metrics.gini = 2 * metrics.auc - 1
                    
                    # KS statistic
                    fpr_list = []
                    tpr_list = []
                    thresholds = np.linspace(0, 1, 100)
                    for threshold in thresholds:
                        y_pred_t = (y_pred_proba >= threshold).astype(int)
                        tn = np.sum((y_true == 0) & (y_pred_t == 0))
                        fp = np.sum((y_true == 0) & (y_pred_t == 1))
                        fn = np.sum((y_true == 1) & (y_pred_t == 0))
                        tp = np.sum((y_true == 1) & (y_pred_t == 1))
                        
                        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                        fpr_list.append(fpr)
                        tpr_list.append(tpr)
                    
                    metrics.ks_statistic = np.max(np.abs(np.array(tpr_list) - np.array(fpr_list)))
                except Exception as e:
                    logger.warning(f"Could not compute AUC/KS: {str(e)}")
        else:
            # Regression metrics
            metrics.mse = mean_squared_error(y_true, y_pred)
            metrics.rmse = np.sqrt(metrics.mse)
            metrics.mae = mean_absolute_error(y_true, y_pred)
            try:
                metrics.r2 = r2_score(y_true, y_pred)
            except:
                metrics.r2 = None
        
        return metrics
    
    def log_predictions(
        self,
        features: pd.DataFrame,
        predictions: np.ndarray,
        actuals: Optional[np.ndarray] = None,
        prediction_probabilities: Optional[np.ndarray] = None,
        latency_ms: Optional[float] = None,
        error: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> None:
        """
        Log predictions for monitoring.
        
        Args:
            features: Input features
            predictions: Model predictions
            actuals: True labels (if available)
            prediction_probabilities: Prediction probabilities
            latency_ms: Prediction latency in milliseconds
            error: Error message if prediction failed
            request_id: Unique request identifier
        """
        timestamp = datetime.now()
        
        # Log prediction
        log_entry = {
            'timestamp': timestamp,
            'request_id': request_id or f"{timestamp.timestamp()}",
            'features': features.to_dict('records') if isinstance(features, pd.DataFrame) else features,
            'predictions': predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
            'actuals': actuals.tolist() if actuals is not None and isinstance(actuals, np.ndarray) else actuals,
            'probabilities': prediction_probabilities.tolist() if prediction_probabilities is not None else None,
            'latency_ms': latency_ms,
            'error': error
        }
        self.prediction_log.append(log_entry)
        
        # Update counters
        self.total_requests += 1
        if error:
            self.total_errors += 1
            self.error_log.append({
                'timestamp': timestamp,
                'error': error,
                'request_id': request_id
            })
        
        # Log latency
        if latency_ms is not None:
            self.latency_log.append({
                'timestamp': timestamp,
                'latency_ms': latency_ms
            })
        
        # Periodic metrics computation (every 1000 requests)
        if self.total_requests % 1000 == 0:
            self._compute_periodic_metrics()
    
    def _compute_periodic_metrics(self) -> None:
        """Compute periodic metrics from logged data."""
        try:
            current_time = datetime.now()
            window_start = current_time - self.monitoring_window
            
            # Filter data within monitoring window
            recent_predictions = [
                p for p in self.prediction_log
                if p['timestamp'] >= window_start
            ]
            
            if not recent_predictions:
                return
            
            # Performance metrics (if actuals available)
            actuals_available = any(p['actuals'] is not None for p in recent_predictions)
            if actuals_available:
                actuals = []
                predictions = []
                probabilities = []
                
                for p in recent_predictions:
                    if p['actuals'] is not None:
                        if isinstance(p['actuals'], list):
                            actuals.extend(p['actuals'])
                            predictions.extend(p['predictions'])
                            if p['probabilities']:
                                probabilities.extend(p['probabilities'])
                        else:
                            actuals.append(p['actuals'])
                            predictions.append(p['predictions'])
                            if p['probabilities']:
                                probabilities.append(p['probabilities'])
                
                if actuals and predictions:
                    actuals = np.array(actuals)
                    predictions = np.array(predictions)
                    probabilities = np.array(probabilities) if probabilities else None
                    
                    perf_metrics = self._compute_performance_metrics(
                        actuals,
                        predictions,
                        probabilities
                    )
                    self.performance_history.append(perf_metrics)
                    
                    # Save metrics
                    if self.storage_path:
                        self._save_metrics('performance', perf_metrics.to_dict())
            
            # Latency metrics
            recent_latencies = [
                l['latency_ms'] for l in self.latency_log
                if l['timestamp'] >= window_start
            ]
            
            if recent_latencies:
                latency_metrics = LatencyMetrics(
                    timestamp=current_time,
                    p50=np.percentile(recent_latencies, 50),
                    p95=np.percentile(recent_latencies, 95),
                    p99=np.percentile(recent_latencies, 99),
                    mean=np.mean(recent_latencies),
                    max=np.max(recent_latencies),
                    min=np.min(recent_latencies),
                    sample_size=len(recent_latencies)
                )
                self.latency_history.append(latency_metrics)
                
                if self.storage_path:
                    self._save_metrics('latency', latency_metrics.to_dict())
            
            # Volume metrics
            time_diff = (current_time - window_start).total_seconds()
            rps = len(recent_predictions) / time_diff if time_diff > 0 else 0
            
            recent_errors = [
                e for e in self.error_log
                if e['timestamp'] >= window_start
            ]
            error_rate = len(recent_errors) / len(recent_predictions) * 100 if recent_predictions else 0
            
            volume_metrics = VolumeMetrics(
                timestamp=current_time,
                requests_per_second=rps,
                total_requests=len(recent_predictions),
                error_count=len(recent_errors),
                error_rate=error_rate
            )
            self.volume_history.append(volume_metrics)
            
            if self.storage_path:
                self._save_metrics('volume', volume_metrics.to_dict())
            
            logger.info(f"Computed periodic metrics: {len(recent_predictions)} requests, "
                       f"{rps:.2f} RPS, {error_rate:.2f}% error rate")
            
        except Exception as e:
            logger.error(f"Error computing periodic metrics: {str(e)}")
    
    def check_drift(
        self,
        production_data: pd.DataFrame,
        production_predictions: Optional[np.ndarray] = None
    ) -> Optional[DriftReport]:
        """
        Check for data drift.
        
        Args:
            production_data: Production features
            production_predictions: Production predictions
            
        Returns:
            DriftReport if drift monitor is configured
        """
        if self.drift_monitor is None:
            logger.warning("Drift monitor not configured (no reference data)")
            return None
        
        drift_report = self.drift_monitor.detect_drift(
            production_data=production_data,
            production_target=production_predictions
        )
        
        self.drift_history.append(drift_report)
        
        if self.storage_path:
            self._save_drift_report(drift_report)
        
        return drift_report
    
    def check_alerts(self) -> List[Alert]:
        """
        Check for SLA violations and generate alerts.
        
        Returns:
            List of active alerts
        """
        alerts = []
        current_time = datetime.now()
        
        # Check performance degradation
        if self.reference_metrics and self.performance_history:
            latest_perf = self.performance_history[-1]
            
            if self.metric_type == MetricType.CLASSIFICATION:
                if self.reference_metrics.auc and latest_perf.auc:
                    auc_degradation = self.reference_metrics.auc - latest_perf.auc
                    
                    if auc_degradation > self.sla_config.max_auc_degradation:
                        severity = AlertSeverity.CRITICAL if auc_degradation > 0.1 else AlertSeverity.WARNING
                        alert = Alert(
                            alert_id=f"perf_auc_{current_time.timestamp()}",
                            severity=severity,
                            category="performance",
                            message=f"AUC degradation detected: {auc_degradation:.4f} "
                                   f"(reference: {self.reference_metrics.auc:.4f}, "
                                   f"current: {latest_perf.auc:.4f})",
                            timestamp=current_time,
                            threshold=self.sla_config.max_auc_degradation,
                            current_value=auc_degradation,
                            metrics={'reference_auc': self.reference_metrics.auc, 'current_auc': latest_perf.auc}
                        )
                        alerts.append(alert)
                
                # Check minimum AUC threshold
                if self.sla_config.min_auc and latest_perf.auc:
                    if latest_perf.auc < self.sla_config.min_auc:
                        alert = Alert(
                            alert_id=f"perf_auc_min_{current_time.timestamp()}",
                            severity=AlertSeverity.CRITICAL,
                            category="performance",
                            message=f"AUC below minimum threshold: {latest_perf.auc:.4f} < {self.sla_config.min_auc}",
                            timestamp=current_time,
                            threshold=self.sla_config.min_auc,
                            current_value=latest_perf.auc
                        )
                        alerts.append(alert)
        
        # Check latency SLA
        if self.latency_history:
            latest_latency = self.latency_history[-1]
            
            if latest_latency.p95 > self.sla_config.max_latency_p95_ms:
                alert = Alert(
                    alert_id=f"latency_p95_{current_time.timestamp()}",
                    severity=AlertSeverity.WARNING,
                    category="latency",
                    message=f"P95 latency exceeded: {latest_latency.p95:.2f}ms > {self.sla_config.max_latency_p95_ms}ms",
                    timestamp=current_time,
                    threshold=self.sla_config.max_latency_p95_ms,
                    current_value=latest_latency.p95
                )
                alerts.append(alert)
            
            if latest_latency.p99 > self.sla_config.max_latency_p99_ms:
                alert = Alert(
                    alert_id=f"latency_p99_{current_time.timestamp()}",
                    severity=AlertSeverity.CRITICAL,
                    category="latency",
                    message=f"P99 latency exceeded: {latest_latency.p99:.2f}ms > {self.sla_config.max_latency_p99_ms}ms",
                    timestamp=current_time,
                    threshold=self.sla_config.max_latency_p99_ms,
                    current_value=latest_latency.p99
                )
                alerts.append(alert)
        
        # Check error rate
        if self.volume_history:
            latest_volume = self.volume_history[-1]
            
            if latest_volume.error_rate > self.sla_config.max_error_rate:
                severity = AlertSeverity.CRITICAL if latest_volume.error_rate > 1.0 else AlertSeverity.WARNING
                alert = Alert(
                    alert_id=f"error_rate_{current_time.timestamp()}",
                    severity=severity,
                    category="error",
                    message=f"Error rate exceeded: {latest_volume.error_rate:.2f}% > {self.sla_config.max_error_rate}%",
                    timestamp=current_time,
                    threshold=self.sla_config.max_error_rate,
                    current_value=latest_volume.error_rate
                )
                alerts.append(alert)
        
        # Check drift
        if self.drift_history:
            latest_drift = self.drift_history[-1]
            
            if latest_drift.has_drift:
                severity = AlertSeverity.CRITICAL if len(latest_drift.drifted_features) > 5 else AlertSeverity.WARNING
                alert = Alert(
                    alert_id=f"drift_{current_time.timestamp()}",
                    severity=severity,
                    category="drift",
                    message=f"Data drift detected in {len(latest_drift.drifted_features)} features: "
                           f"{', '.join(latest_drift.drifted_features[:5])}",
                    timestamp=current_time,
                    metrics={'drifted_features': latest_drift.drifted_features,
                            'drift_scores': latest_drift.drift_scores}
                )
                alerts.append(alert)
        
        # Update active alerts
        self.active_alerts = alerts
        self.alert_history.extend(alerts)
        
        if alerts:
            logger.warning(f"Generated {len(alerts)} alerts")
        
        return alerts
    
    def get_prometheus_metrics(self) -> str:
        """
        Export metrics in Prometheus format.
        
        Returns:
            Prometheus-formatted metrics string
        """
        metrics_lines = []
        
        # Performance metrics
        if self.performance_history:
            latest_perf = self.performance_history[-1]
            
            if latest_perf.auc is not None:
                metrics_lines.append(
                    f'# HELP model_auc Area Under ROC Curve\n'
                    f'# TYPE model_auc gauge\n'
                    f'model_auc{{model="{self.model_name}",version="{self.model_version}"}} {latest_perf.auc:.6f}\n'
                )
            
            if latest_perf.accuracy is not None:
                metrics_lines.append(
                    f'# HELP model_accuracy Model accuracy\n'
                    f'# TYPE model_accuracy gauge\n'
                    f'model_accuracy{{model="{self.model_name}",version="{self.model_version}"}} {latest_perf.accuracy:.6f}\n'
                )
            
            if latest_perf.ks_statistic is not None:
                metrics_lines.append(
                    f'# HELP model_ks_statistic KS statistic\n'
                    f'# TYPE model_ks_statistic gauge\n'
                    f'model_ks_statistic{{model="{self.model_name}",version="{self.model_version}"}} {latest_perf.ks_statistic:.6f}\n'
                )
        
        # Latency metrics
        if self.latency_history:
            latest_latency = self.latency_history[-1]
            
            metrics_lines.append(
                f'# HELP model_latency_ms Prediction latency in milliseconds\n'
                f'# TYPE model_latency_ms gauge\n'
                f'model_latency_ms_p50{{model="{self.model_name}",version="{self.model_version}"}} {latest_latency.p50:.2f}\n'
                f'model_latency_ms_p95{{model="{self.model_name}",version="{self.model_version}"}} {latest_latency.p95:.2f}\n'
                f'model_latency_ms_p99{{model="{self.model_name}",version="{self.model_version}"}} {latest_latency.p99:.2f}\n'
            )
        
        # Volume metrics
        if self.volume_history:
            latest_volume = self.volume_history[-1]
            
            metrics_lines.append(
                f'# HELP model_requests_total Total number of requests\n'
                f'# TYPE model_requests_total counter\n'
                f'model_requests_total{{model="{self.model_name}",version="{self.model_version}"}} {self.total_requests}\n'
            )
            
            metrics_lines.append(
                f'# HELP model_errors_total Total number of errors\n'
                f'# TYPE model_errors_total counter\n'
                f'model_errors_total{{model="{self.model_name}",version="{self.model_version}"}} {self.total_errors}\n'
            )
            
            metrics_lines.append(
                f'# HELP model_error_rate Error rate percentage\n'
                f'# TYPE model_error_rate gauge\n'
                f'model_error_rate{{model="{self.model_name}",version="{self.model_version}"}} {latest_volume.error_rate:.2f}\n'
            )
            
            metrics_lines.append(
                f'# HELP model_rps Requests per second\n'
                f'# TYPE model_rps gauge\n'
                f'model_rps{{model="{self.model_name}",version="{self.model_version}"}} {latest_volume.requests_per_second:.2f}\n'
            )
        
        # Alert metrics
        metrics_lines.append(
            f'# HELP model_active_alerts Number of active alerts\n'
            f'# TYPE model_active_alerts gauge\n'
            f'model_active_alerts{{model="{self.model_name}",version="{self.model_version}"}} {len(self.active_alerts)}\n'
        )
        
        # Uptime
        uptime = (datetime.now() - self.start_time).total_seconds()
        metrics_lines.append(
            f'# HELP model_uptime_seconds Model uptime in seconds\n'
            f'# TYPE model_uptime_seconds counter\n'
            f'model_uptime_seconds{{model="{self.model_name}",version="{self.model_version}"}} {uptime:.2f}\n'
        )
        
        return ''.join(metrics_lines)
    
    def get_sla_compliance(self) -> Dict[str, Any]:
        """
        Get SLA compliance report.
        
        Returns:
            Dictionary with SLA compliance metrics
        """
        compliance = {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'timestamp': datetime.now().isoformat(),
            'compliant': True,
            'violations': []
        }
        
        # Check latency SLA
        if self.latency_history:
            latest = self.latency_history[-1]
            
            if latest.p95 > self.sla_config.max_latency_p95_ms:
                compliance['compliant'] = False
                compliance['violations'].append({
                    'metric': 'latency_p95',
                    'threshold': self.sla_config.max_latency_p95_ms,
                    'actual': latest.p95,
                    'severity': 'warning'
                })
            
            if latest.p99 > self.sla_config.max_latency_p99_ms:
                compliance['compliant'] = False
                compliance['violations'].append({
                    'metric': 'latency_p99',
                    'threshold': self.sla_config.max_latency_p99_ms,
                    'actual': latest.p99,
                    'severity': 'critical'
                })
        
        # Check error rate SLA
        if self.volume_history:
            latest = self.volume_history[-1]
            
            if latest.error_rate > self.sla_config.max_error_rate:
                compliance['compliant'] = False
                compliance['violations'].append({
                    'metric': 'error_rate',
                    'threshold': self.sla_config.max_error_rate,
                    'actual': latest.error_rate,
                    'severity': 'critical' if latest.error_rate > 1.0 else 'warning'
                })
        
        # Check performance SLA
        if self.reference_metrics and self.performance_history:
            latest = self.performance_history[-1]
            
            if self.reference_metrics.auc and latest.auc:
                degradation = self.reference_metrics.auc - latest.auc
                if degradation > self.sla_config.max_auc_degradation:
                    compliance['compliant'] = False
                    compliance['violations'].append({
                        'metric': 'auc_degradation',
                        'threshold': self.sla_config.max_auc_degradation,
                        'actual': degradation,
                        'severity': 'critical' if degradation > 0.1 else 'warning'
                    })
        
        return compliance
    
    def _save_metrics(self, metric_type: str, metrics: Dict[str, Any]) -> None:
        """Save metrics to storage."""
        if not self.storage_path:
            return
        
        metrics_dir = self.storage_path / metric_type
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = metrics_dir / f"{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def _save_drift_report(self, report: DriftReport) -> None:
        """Save drift report to storage."""
        if not self.storage_path:
            return
        
        drift_dir = self.storage_path / "drift"
        drift_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = report.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = drift_dir / f"{timestamp}.json"
        
        report_dict = {
            'has_drift': report.has_drift,
            'drifted_features': report.drifted_features,
            'drift_scores': report.drift_scores,
            'p_values': report.p_values,
            'severity_levels': report.severity_levels,
            'timestamp': report.timestamp.isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(report_dict, f, indent=2)
    
    def send_notifications(
        self,
        alerts: List[Alert],
        webhook_url: Optional[str] = None,
        email_recipients: Optional[List[str]] = None
    ) -> None:
        """
        Send alert notifications.
        
        Args:
            alerts: List of alerts to send
            webhook_url: Webhook URL for notifications (e.g., Slack)
            email_recipients: List of email addresses
        """
        for alert in alerts:
            logger.warning(f"ALERT [{alert.severity.value.upper()}] {alert.category}: {alert.message}")
            
            # Webhook notification (implement as needed)
            if webhook_url:
                # TODO: Implement webhook notification
                pass
            
            # Email notification (implement as needed)
            if email_recipients:
                # TODO: Implement email notification
                pass
    
    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """
        Get data for monitoring dashboard.
        
        Returns:
            Dictionary with dashboard data
        """
        return {
            'model_info': {
                'name': self.model_name,
                'version': self.model_version,
                'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600
            },
            'current_metrics': {
                'performance': self.performance_history[-1].to_dict() if self.performance_history else None,
                'latency': self.latency_history[-1].to_dict() if self.latency_history else None,
                'volume': self.volume_history[-1].to_dict() if self.volume_history else None
            },
            'reference_metrics': self.reference_metrics.to_dict() if self.reference_metrics else None,
            'active_alerts': [alert.to_dict() for alert in self.active_alerts],
            'sla_compliance': self.get_sla_compliance(),
            'drift_status': {
                'has_drift': self.drift_history[-1].has_drift if self.drift_history else False,
                'drifted_features': self.drift_history[-1].drifted_features if self.drift_history else []
            },
            'total_requests': self.total_requests,
            'total_errors': self.total_errors
        }
