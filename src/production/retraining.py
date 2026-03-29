"""
Model Retraining Orchestration Framework

This module provides automated model retraining triggers, pipeline orchestration,
and validation for production ML models. Supports scheduled retraining,
performance-based triggers, drift-based triggers, and manual triggers.

Usage Example:
    >>> from retraining import RetrainingOrchestrator
    >>> orchestrator = RetrainingOrchestrator(
    ...     model_name="credit_risk",
    ...     training_pipeline=my_training_pipeline
    ... )
    >>> # Schedule automatic retraining
    >>> orchestrator.schedule_retraining(schedule="monthly")
    >>> # Or trigger manually
    >>> orchestrator.trigger_retraining(reason="performance_degradation")
"""

from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import json
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, Future

import numpy as np
import pandas as pd
from loguru import logger
import schedule

# Import from existing modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
from validation.drift_monitor import DriftMonitor, DriftReport
from production.monitoring import ModelMonitor, PerformanceMetrics, Alert, AlertSeverity
from production.deployment import ModelRegistry, ModelDeployer, ModelMetadata, EnvironmentType


class TriggerType(str, Enum):
    """Retraining trigger types."""
    SCHEDULED = "scheduled"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_DRIFT = "data_drift"
    PSI_THRESHOLD = "psi_threshold"
    MANUAL = "manual"
    INCIDENT = "incident"


class RetrainingStatus(str, Enum):
    """Retraining job status."""
    PENDING = "pending"
    RUNNING = "running"
    VALIDATING = "validating"
    TESTING = "testing"
    DEPLOYING = "deploying"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class TriggerConfig:
    """Configuration for retraining triggers."""
    # Performance triggers
    enable_performance_trigger: bool = True
    min_auc_threshold: Optional[float] = None  # Absolute threshold
    max_auc_degradation: float = 0.05  # 5% degradation
    min_sample_size: int = 1000
    
    # Drift triggers
    enable_drift_trigger: bool = True
    max_psi_threshold: float = 0.25
    max_drifted_features: int = 5
    drift_severity_threshold: str = "medium"  # low, medium, high, critical
    
    # Scheduled triggers
    enable_scheduled_trigger: bool = True
    schedule_frequency: str = "monthly"  # daily, weekly, monthly, quarterly
    schedule_day: Optional[int] = None  # Day of month for monthly/quarterly
    schedule_time: str = "02:00"  # Time of day (HH:MM)
    
    # Volume triggers
    enable_volume_trigger: bool = False
    min_new_samples: int = 10000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'enable_performance_trigger': self.enable_performance_trigger,
            'min_auc_threshold': self.min_auc_threshold,
            'max_auc_degradation': self.max_auc_degradation,
            'min_sample_size': self.min_sample_size,
            'enable_drift_trigger': self.enable_drift_trigger,
            'max_psi_threshold': self.max_psi_threshold,
            'max_drifted_features': self.max_drifted_features,
            'drift_severity_threshold': self.drift_severity_threshold,
            'enable_scheduled_trigger': self.enable_scheduled_trigger,
            'schedule_frequency': self.schedule_frequency,
            'schedule_day': self.schedule_day,
            'schedule_time': self.schedule_time,
            'enable_volume_trigger': self.enable_volume_trigger,
            'min_new_samples': self.min_new_samples
        }


@dataclass
class RetrainingJob:
    """Container for retraining job information."""
    job_id: str
    model_name: str
    trigger_type: TriggerType
    trigger_reason: str
    status: RetrainingStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    champion_version: Optional[str] = None
    challenger_version: Optional[str] = None
    training_data_size: int = 0
    validation_data_size: int = 0
    champion_metrics: Optional[Dict[str, float]] = None
    challenger_metrics: Optional[Dict[str, float]] = None
    comparison_results: Optional[Dict[str, Any]] = None
    deployed: bool = False
    rolled_back: bool = False
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'job_id': self.job_id,
            'model_name': self.model_name,
            'trigger_type': self.trigger_type.value,
            'trigger_reason': self.trigger_reason,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'champion_version': self.champion_version,
            'challenger_version': self.challenger_version,
            'training_data_size': self.training_data_size,
            'validation_data_size': self.validation_data_size,
            'champion_metrics': self.champion_metrics,
            'challenger_metrics': self.challenger_metrics,
            'comparison_results': self.comparison_results,
            'deployed': self.deployed,
            'rolled_back': self.rolled_back,
            'error_message': self.error_message,
            'metadata': self.metadata
        }


@dataclass
class ValidationConfig:
    """Configuration for model validation."""
    oot_validation_required: bool = True
    min_oot_samples: int = 5000
    oot_period_days: int = 30
    min_improvement_threshold: float = 0.01  # Minimum improvement to replace champion
    max_degradation_threshold: float = 0.02  # Maximum allowed degradation
    business_metrics: List[str] = field(default_factory=lambda: ['auc', 'ks_statistic', 'gini'])
    stability_test_required: bool = True
    adversarial_validation_required: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'oot_validation_required': self.oot_validation_required,
            'min_oot_samples': self.min_oot_samples,
            'oot_period_days': self.oot_period_days,
            'min_improvement_threshold': self.min_improvement_threshold,
            'max_degradation_threshold': self.max_degradation_threshold,
            'business_metrics': self.business_metrics,
            'stability_test_required': self.stability_test_required,
            'adversarial_validation_required': self.adversarial_validation_required
        }


class RetrainingOrchestrator:
    """
    Automated model retraining orchestrator.
    
    Manages retraining triggers, pipeline execution, validation,
    champion-challenger comparison, and automated deployment.
    
    Example:
        >>> def training_pipeline(train_data, val_data):
        ...     # Your training logic
        ...     return trained_model, metrics
        >>> 
        >>> orchestrator = RetrainingOrchestrator(
        ...     model_name="fraud_detection",
        ...     training_pipeline=training_pipeline
        ... )
        >>> orchestrator.schedule_retraining(schedule="monthly")
    """
    
    def __init__(
        self,
        model_name: str,
        training_pipeline: Callable,
        data_pipeline: Optional[Callable] = None,
        feature_pipeline: Optional[Callable] = None,
        trigger_config: Optional[TriggerConfig] = None,
        validation_config: Optional[ValidationConfig] = None,
        model_registry: Optional[ModelRegistry] = None,
        monitor: Optional[ModelMonitor] = None,
        deployer: Optional[ModelDeployer] = None,
        storage_path: Optional[Union[str, Path]] = None,
        notification_callback: Optional[Callable] = None
    ):
        self.model_name = model_name
        self.training_pipeline = training_pipeline
        self.data_pipeline = data_pipeline
        self.feature_pipeline = feature_pipeline
        self.trigger_config = trigger_config or TriggerConfig()
        self.validation_config = validation_config or ValidationConfig()
        
        # Initialize components
        self.registry = model_registry or ModelRegistry()
        self.monitor = monitor
        self.deployer = deployer or ModelDeployer()
        self.notification_callback = notification_callback
        
        # Storage
        if storage_path:
            self.storage_path = Path(storage_path)
            self.storage_path.mkdir(parents=True, exist_ok=True)
        else:
            self.storage_path = Path("./retraining_jobs")
            self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Job tracking
        self.active_jobs: Dict[str, RetrainingJob] = {}
        self.job_history: List[RetrainingJob] = []
        self._load_job_history()
        
        # Scheduling
        self.scheduler_running = False
        self.executor = ThreadPoolExecutor(max_workers=1)
        
        logger.info(f"Initialized RetrainingOrchestrator for {model_name}")
    
    def _load_job_history(self) -> None:
        """Load job history from storage."""
        history_file = self.storage_path / "job_history.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                data = json.load(f)
                self.job_history = [
                    RetrainingJob(
                        job_id=j['job_id'],
                        model_name=j['model_name'],
                        trigger_type=TriggerType(j['trigger_type']),
                        trigger_reason=j['trigger_reason'],
                        status=RetrainingStatus(j['status']),
                        created_at=datetime.fromisoformat(j['created_at']),
                        started_at=datetime.fromisoformat(j['started_at']) if j.get('started_at') else None,
                        completed_at=datetime.fromisoformat(j['completed_at']) if j.get('completed_at') else None,
                        champion_version=j.get('champion_version'),
                        challenger_version=j.get('challenger_version'),
                        training_data_size=j.get('training_data_size', 0),
                        validation_data_size=j.get('validation_data_size', 0),
                        champion_metrics=j.get('champion_metrics'),
                        challenger_metrics=j.get('challenger_metrics'),
                        comparison_results=j.get('comparison_results'),
                        deployed=j.get('deployed', False),
                        rolled_back=j.get('rolled_back', False),
                        error_message=j.get('error_message'),
                        metadata=j.get('metadata', {})
                    )
                    for j in data
                ]
            logger.info(f"Loaded {len(self.job_history)} previous retraining jobs")
    
    def _save_job_history(self) -> None:
        """Save job history to storage."""
        history_file = self.storage_path / "job_history.json"
        with open(history_file, 'w') as f:
            json.dump([j.to_dict() for j in self.job_history], f, indent=2)
    
    def _save_job(self, job: RetrainingJob) -> None:
        """Save individual job to storage."""
        job_file = self.storage_path / f"{job.job_id}.json"
        with open(job_file, 'w') as f:
            json.dump(job.to_dict(), f, indent=2)
    
    def check_triggers(self) -> List[Tuple[TriggerType, str]]:
        """
        Check all enabled triggers.
        
        Returns:
            List of (trigger_type, reason) tuples for triggered conditions
        """
        triggers = []
        
        if not self.monitor:
            logger.warning("Monitor not configured, skipping trigger checks")
            return triggers
        
        # Performance degradation trigger
        if self.trigger_config.enable_performance_trigger:
            if self.monitor.reference_metrics and self.monitor.performance_history:
                latest_perf = self.monitor.performance_history[-1]
                ref_metrics = self.monitor.reference_metrics
                
                if latest_perf.sample_size >= self.trigger_config.min_sample_size:
                    # Check AUC degradation
                    if ref_metrics.auc and latest_perf.auc:
                        degradation = ref_metrics.auc - latest_perf.auc
                        
                        if degradation > self.trigger_config.max_auc_degradation:
                            reason = (f"AUC degradation {degradation:.4f} exceeds threshold "
                                     f"{self.trigger_config.max_auc_degradation}")
                            triggers.append((TriggerType.PERFORMANCE_DEGRADATION, reason))
                            logger.warning(f"Performance trigger activated: {reason}")
                    
                    # Check absolute AUC threshold
                    if self.trigger_config.min_auc_threshold and latest_perf.auc:
                        if latest_perf.auc < self.trigger_config.min_auc_threshold:
                            reason = (f"AUC {latest_perf.auc:.4f} below minimum threshold "
                                     f"{self.trigger_config.min_auc_threshold}")
                            triggers.append((TriggerType.PERFORMANCE_DEGRADATION, reason))
                            logger.warning(f"Performance trigger activated: {reason}")
        
        # Data drift trigger
        if self.trigger_config.enable_drift_trigger:
            if self.monitor.drift_history:
                latest_drift = self.monitor.drift_history[-1]
                
                if latest_drift.has_drift:
                    num_drifted = len(latest_drift.drifted_features)
                    
                    if num_drifted >= self.trigger_config.max_drifted_features:
                        reason = f"Data drift detected in {num_drifted} features"
                        triggers.append((TriggerType.DATA_DRIFT, reason))
                        logger.warning(f"Drift trigger activated: {reason}")
                    
                    # Check PSI if available
                    # Note: PSI calculation would need to be added to drift monitor
                    # For now, using drift scores as proxy
                    max_drift_score = max(latest_drift.drift_scores.values()) if latest_drift.drift_scores else 0
                    if max_drift_score > self.trigger_config.max_psi_threshold:
                        reason = f"Maximum drift score {max_drift_score:.4f} exceeds PSI threshold"
                        triggers.append((TriggerType.PSI_THRESHOLD, reason))
                        logger.warning(f"PSI trigger activated: {reason}")
        
        return triggers
    
    def trigger_retraining(
        self,
        trigger_type: TriggerType = TriggerType.MANUAL,
        reason: str = "Manual trigger",
        training_data: Optional[pd.DataFrame] = None,
        validation_data: Optional[pd.DataFrame] = None,
        async_execution: bool = True
    ) -> RetrainingJob:
        """
        Trigger a retraining job.
        
        Args:
            trigger_type: Type of trigger
            reason: Reason for retraining
            training_data: Training data (optional, will fetch if not provided)
            validation_data: Validation data (optional)
            async_execution: Execute asynchronously
            
        Returns:
            RetrainingJob object
        """
        # Create job
        job_id = f"{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        job = RetrainingJob(
            job_id=job_id,
            model_name=self.model_name,
            trigger_type=trigger_type,
            trigger_reason=reason,
            status=RetrainingStatus.PENDING,
            created_at=datetime.now()
        )
        
        self.active_jobs[job_id] = job
        self._save_job(job)
        
        logger.info(f"Created retraining job {job_id}: {reason}")
        
        # Send notification
        if self.notification_callback:
            self.notification_callback(
                f"Retraining job created for {self.model_name}",
                f"Trigger: {trigger_type.value}\nReason: {reason}"
            )
        
        # Execute
        if async_execution:
            future = self.executor.submit(
                self._execute_retraining_job,
                job,
                training_data,
                validation_data
            )
            job.metadata['future'] = future
        else:
            self._execute_retraining_job(job, training_data, validation_data)
        
        return job
    
    def _execute_retraining_job(
        self,
        job: RetrainingJob,
        training_data: Optional[pd.DataFrame] = None,
        validation_data: Optional[pd.DataFrame] = None
    ) -> None:
        """Execute a retraining job."""
        try:
            job.status = RetrainingStatus.RUNNING
            job.started_at = datetime.now()
            self._save_job(job)
            
            logger.info(f"Starting retraining job {job.job_id}")
            
            # Step 1: Data pipeline
            if training_data is None:
                if self.data_pipeline is None:
                    raise ValueError("No training data provided and no data pipeline configured")
                
                logger.info("Fetching training data...")
                training_data, validation_data = self.data_pipeline()
            
            job.training_data_size = len(training_data)
            job.validation_data_size = len(validation_data) if validation_data is not None else 0
            
            # Step 2: Feature engineering
            if self.feature_pipeline:
                logger.info("Running feature engineering pipeline...")
                training_data = self.feature_pipeline(training_data)
                if validation_data is not None:
                    validation_data = self.feature_pipeline(validation_data)
            
            # Step 3: Train model
            logger.info("Training new model...")
            new_model, challenger_metrics = self.training_pipeline(training_data, validation_data)
            
            # Register challenger model
            challenger_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}_challenger"
            self.registry.register_model(
                model=new_model,
                model_name=self.model_name,
                version=challenger_version,
                metrics=challenger_metrics,
                environment=EnvironmentType.STAGING,
                tags={'job_id': job.job_id, 'type': 'challenger'}
            )
            
            job.challenger_version = challenger_version
            job.challenger_metrics = challenger_metrics
            
            logger.info(f"Trained challenger model {challenger_version} with metrics: {challenger_metrics}")
            
            # Step 4: Validation
            job.status = RetrainingStatus.VALIDATING
            self._save_job(job)
            
            validation_passed = self._validate_challenger(
                job=job,
                challenger_model=new_model,
                validation_data=validation_data
            )
            
            if not validation_passed:
                raise ValueError("Challenger model failed validation")
            
            # Step 5: Champion-Challenger comparison
            job.status = RetrainingStatus.TESTING
            self._save_job(job)
            
            comparison_results = self._compare_champion_challenger(job, validation_data)
            job.comparison_results = comparison_results
            
            # Step 6: Deployment decision
            if comparison_results['deploy_challenger']:
                logger.info("Challenger outperforms champion, deploying...")
                job.status = RetrainingStatus.DEPLOYING
                self._save_job(job)
                
                self._deploy_challenger(job)
                job.deployed = True
                
                logger.info(f"Successfully deployed challenger model {job.challenger_version}")
            else:
                logger.info("Challenger does not outperform champion, keeping current model")
                job.deployed = False
            
            # Complete job
            job.status = RetrainingStatus.COMPLETED
            job.completed_at = datetime.now()
            self._save_job(job)
            
            # Move to history
            self.job_history.append(job)
            self._save_job_history()
            
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]
            
            # Send notification
            if self.notification_callback:
                self.notification_callback(
                    f"Retraining job completed for {self.model_name}",
                    f"Job ID: {job.job_id}\n"
                    f"Deployed: {job.deployed}\n"
                    f"Duration: {(job.completed_at - job.started_at).total_seconds():.1f}s"
                )
            
            logger.info(f"Retraining job {job.job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Retraining job {job.job_id} failed: {str(e)}")
            job.status = RetrainingStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
            self._save_job(job)
            
            # Attempt rollback if deployment was attempted
            if job.deployed:
                logger.info("Attempting rollback...")
                try:
                    self._rollback_deployment(job)
                    job.rolled_back = True
                    logger.info("Rollback successful")
                except Exception as rollback_error:
                    logger.error(f"Rollback failed: {str(rollback_error)}")
            
            # Send notification
            if self.notification_callback:
                self.notification_callback(
                    f"Retraining job FAILED for {self.model_name}",
                    f"Job ID: {job.job_id}\nError: {str(e)}"
                )
    
    def _validate_challenger(
        self,
        job: RetrainingJob,
        challenger_model: Any,
        validation_data: Optional[pd.DataFrame]
    ) -> bool:
        """Validate challenger model."""
        logger.info("Validating challenger model...")
        
        # Basic validation
        if validation_data is None or len(validation_data) == 0:
            logger.warning("No validation data available")
            return True
        
        # Out-of-time validation
        if self.validation_config.oot_validation_required:
            if len(validation_data) < self.validation_config.min_oot_samples:
                logger.warning(f"Insufficient OOT samples: {len(validation_data)} < "
                             f"{self.validation_config.min_oot_samples}")
                return False
        
        # Check for NaN predictions
        try:
            X_val = validation_data.drop(columns=['target']) if 'target' in validation_data.columns else validation_data
            predictions = challenger_model.predict(X_val)
            
            if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                logger.error("Challenger produces NaN or Inf predictions")
                return False
        except Exception as e:
            logger.error(f"Prediction error during validation: {str(e)}")
            return False
        
        logger.info("Challenger validation passed")
        return True
    
    def _compare_champion_challenger(
        self,
        job: RetrainingJob,
        validation_data: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Compare champion and challenger models."""
        logger.info("Comparing champion and challenger models...")
        
        results = {
            'deploy_challenger': False,
            'reason': '',
            'metrics_comparison': {}
        }
        
        # Get current champion
        try:
            champion_version = self.registry.get_latest_version(self.model_name)
            champion_model, champion_metadata = self.registry.get_model(
                self.model_name,
                champion_version
            )
            job.champion_version = champion_version
            job.champion_metrics = champion_metadata.metrics
        except Exception as e:
            logger.warning(f"No champion model found: {str(e)}")
            # No champion exists, deploy challenger
            results['deploy_challenger'] = True
            results['reason'] = "No existing champion model"
            return results
        
        # Compare metrics
        for metric in self.validation_config.business_metrics:
            champion_value = job.champion_metrics.get(metric)
            challenger_value = job.challenger_metrics.get(metric)
            
            if champion_value is not None and challenger_value is not None:
                improvement = challenger_value - champion_value
                improvement_pct = (improvement / champion_value * 100) if champion_value != 0 else 0
                
                results['metrics_comparison'][metric] = {
                    'champion': champion_value,
                    'challenger': challenger_value,
                    'improvement': improvement,
                    'improvement_pct': improvement_pct
                }
        
        # Decision logic
        primary_metric = self.validation_config.business_metrics[0]
        if primary_metric in results['metrics_comparison']:
            comp = results['metrics_comparison'][primary_metric]
            
            if comp['improvement'] > self.validation_config.min_improvement_threshold:
                results['deploy_challenger'] = True
                results['reason'] = (f"Challenger outperforms champion by "
                                   f"{comp['improvement_pct']:.2f}% on {primary_metric}")
            elif comp['improvement'] < -self.validation_config.max_degradation_threshold:
                results['deploy_challenger'] = False
                results['reason'] = (f"Challenger underperforms champion by "
                                   f"{abs(comp['improvement_pct']):.2f}% on {primary_metric}")
            else:
                results['deploy_challenger'] = False
                results['reason'] = "Improvement not significant enough to replace champion"
        
        logger.info(f"Comparison complete: {results['reason']}")
        return results
    
    def _deploy_challenger(self, job: RetrainingJob) -> None:
        """Deploy challenger model to production."""
        if job.challenger_version is None:
            raise ValueError("No challenger version to deploy")
        
        # Promote to production
        self.registry.promote_model(
            model_name=self.model_name,
            version=job.challenger_version,
            to_environment=EnvironmentType.PROD
        )
        
        logger.info(f"Promoted {job.challenger_version} to production")
    
    def _rollback_deployment(self, job: RetrainingJob) -> None:
        """Rollback to champion model."""
        if job.champion_version is None:
            raise ValueError("No champion version to rollback to")
        
        self.registry.promote_model(
            model_name=self.model_name,
            version=job.champion_version,
            to_environment=EnvironmentType.PROD
        )
        
        logger.info(f"Rolled back to champion {job.champion_version}")
    
    def schedule_retraining(self, schedule_str: Optional[str] = None) -> None:
        """
        Schedule automatic retraining.
        
        Args:
            schedule_str: Schedule string (e.g., 'daily', 'weekly', 'monthly')
        """
        if schedule_str:
            self.trigger_config.schedule_frequency = schedule_str
        
        freq = self.trigger_config.schedule_frequency
        time_str = self.trigger_config.schedule_time
        
        if freq == "daily":
            schedule.every().day.at(time_str).do(
                lambda: self.trigger_retraining(
                    trigger_type=TriggerType.SCHEDULED,
                    reason=f"Scheduled retraining (daily)"
                )
            )
        elif freq == "weekly":
            schedule.every().monday.at(time_str).do(
                lambda: self.trigger_retraining(
                    trigger_type=TriggerType.SCHEDULED,
                    reason=f"Scheduled retraining (weekly)"
                )
            )
        elif freq == "monthly":
            # Check monthly on the first of each month
            schedule.every().day.at(time_str).do(
                self._check_monthly_schedule
            )
        elif freq == "quarterly":
            # Check quarterly (Jan 1, Apr 1, Jul 1, Oct 1)
            schedule.every().day.at(time_str).do(
                self._check_quarterly_schedule
            )
        
        logger.info(f"Scheduled retraining: {freq} at {time_str}")
        
        # Start scheduler loop
        if not self.scheduler_running:
            self.scheduler_running = True
            self.executor.submit(self._run_scheduler)
    
    def _check_monthly_schedule(self) -> None:
        """Check if monthly retraining should run."""
        today = datetime.now().day
        schedule_day = self.trigger_config.schedule_day or 1
        
        if today == schedule_day:
            self.trigger_retraining(
                trigger_type=TriggerType.SCHEDULED,
                reason=f"Scheduled retraining (monthly on day {schedule_day})"
            )
    
    def _check_quarterly_schedule(self) -> None:
        """Check if quarterly retraining should run."""
        now = datetime.now()
        if now.month in [1, 4, 7, 10] and now.day == 1:
            self.trigger_retraining(
                trigger_type=TriggerType.SCHEDULED,
                reason=f"Scheduled retraining (quarterly)"
            )
    
    def _run_scheduler(self) -> None:
        """Run scheduler loop."""
        logger.info("Starting scheduler loop")
        while self.scheduler_running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def stop_scheduler(self) -> None:
        """Stop scheduler."""
        self.scheduler_running = False
        logger.info("Stopped scheduler")
    
    def get_job_status(self, job_id: str) -> Optional[RetrainingJob]:
        """Get status of a retraining job."""
        if job_id in self.active_jobs:
            return self.active_jobs[job_id]
        
        for job in self.job_history:
            if job.job_id == job_id:
                return job
        
        return None
    
    def get_recent_jobs(self, limit: int = 10) -> List[RetrainingJob]:
        """Get recent retraining jobs."""
        all_jobs = list(self.active_jobs.values()) + self.job_history
        sorted_jobs = sorted(all_jobs, key=lambda x: x.created_at, reverse=True)
        return sorted_jobs[:limit]
    
    def get_retraining_history(self) -> Dict[str, Any]:
        """Get retraining history statistics."""
        total_jobs = len(self.job_history)
        if total_jobs == 0:
            return {
                'total_jobs': 0,
                'successful_deployments': 0,
                'failed_jobs': 0,
                'rollbacks': 0
            }
        
        successful = sum(1 for j in self.job_history if j.deployed)
        failed = sum(1 for j in self.job_history if j.status == RetrainingStatus.FAILED)
        rolled_back = sum(1 for j in self.job_history if j.rolled_back)
        
        trigger_counts = {}
        for job in self.job_history:
            trigger_counts[job.trigger_type.value] = trigger_counts.get(job.trigger_type.value, 0) + 1
        
        avg_duration = np.mean([
            (j.completed_at - j.started_at).total_seconds()
            for j in self.job_history
            if j.completed_at and j.started_at
        ]) if any(j.completed_at and j.started_at for j in self.job_history) else 0
        
        return {
            'total_jobs': total_jobs,
            'successful_deployments': successful,
            'failed_jobs': failed,
            'rollbacks': rolled_back,
            'success_rate': successful / total_jobs * 100 if total_jobs > 0 else 0,
            'trigger_breakdown': trigger_counts,
            'avg_duration_seconds': avg_duration,
            'last_job': self.job_history[-1].to_dict() if self.job_history else None
        }
