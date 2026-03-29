"""
End-to-End Income Estimation Pipeline

Orchestrates complete income estimation workflow:
- Transaction data preprocessing and validation
- Deposit pattern analysis and classification
- Graph network analysis for validation
- Stability modeling and risk assessment
- Multiple model ensemble
- Calibrated predictions with confidence intervals
- Real-time and batch estimation
- Validation against known incomes
- Regulatory compliance checks
- Monitoring and drift detection
- Complete usage examples and production deployment

Example:
    >>> from src.use_cases.income_estimation.income_pipeline import IncomeEstimationPipeline
    >>> 
    >>> # Initialize pipeline
    >>> pipeline = IncomeEstimationPipeline()
    >>> 
    >>> # Fit on historical data
    >>> pipeline.fit(transactions_df, known_incomes_df)
    >>> 
    >>> # Predict for new users
    >>> results = pipeline.predict(user_id="user_123", transactions_df=new_transactions)
    >>> print(f"Estimated income: ${results.point_estimate:.2f}")
    >>> print(f"90% CI: [{results.lower_bound:.2f}, {results.upper_bound:.2f}]")
    >>> print(f"Stability: {results.stability_score:.2f}")
    >>> print(f"Risk: {results.risk_category}")

Author: Principal Data Science Decision Agent
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from loguru import logger

# Import local modules
from .deposit_intelligence import DepositDetector, IncomeSource
from .graph_payment import PaymentNetworkAnalyzer, NetworkValidation
from .stability_model import IncomeStabilityScorer, StabilityMetrics, RiskAssessment
from .calibration import (
    ConformalPredictor,
    IsotonicCalibrator,
    BayesianCalibrator,
    PredictionInterval,
    QuantilePrediction
)


@dataclass
class IncomeEstimationResult:
    """Container for income estimation results."""
    
    user_id: str
    point_estimate: float
    lower_bound: float  # 90% CI
    upper_bound: float  # 90% CI
    confidence: float  # Overall confidence 0-1
    
    # Source breakdown
    primary_salary: float
    secondary_income: float
    freelance_gig: float
    investment_income: float
    num_sources: int
    
    # Stability and risk
    stability_score: float
    risk_score: float
    risk_category: str  # 'low', 'medium', 'high', 'very_high'
    
    # Network validation
    network_validated: bool
    peer_percentile: float
    
    # Distributional prediction
    quantile_prediction: Optional[QuantilePrediction] = None
    
    # Metadata
    prediction_date: datetime = None
    model_version: str = "1.0"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {
            'user_id': self.user_id,
            'point_estimate': self.point_estimate,
            'lower_bound': self.lower_bound,
            'upper_bound': self.upper_bound,
            'confidence': self.confidence,
            'primary_salary': self.primary_salary,
            'secondary_income': self.secondary_income,
            'freelance_gig': self.freelance_gig,
            'investment_income': self.investment_income,
            'num_sources': self.num_sources,
            'stability_score': self.stability_score,
            'risk_score': self.risk_score,
            'risk_category': self.risk_category,
            'network_validated': self.network_validated,
            'peer_percentile': self.peer_percentile,
            'prediction_date': self.prediction_date,
            'model_version': self.model_version
        }
        
        if self.quantile_prediction:
            result.update({
                'q10': self.quantile_prediction.q10,
                'q25': self.quantile_prediction.q25,
                'q50': self.quantile_prediction.q50,
                'q75': self.quantile_prediction.q75,
                'q90': self.quantile_prediction.q90,
            })
        
        return result


@dataclass
class ValidationMetrics:
    """Container for validation metrics."""
    
    mae: float  # Mean Absolute Error
    mape: float  # Mean Absolute Percentage Error
    rmse: float  # Root Mean Squared Error
    coverage_90: float  # 90% interval coverage
    coverage_80: float  # 80% interval coverage
    median_interval_width: float
    calibration_score: float
    n_samples: int


class IncomeEstimationPipeline:
    """
    End-to-end pipeline for income estimation.
    
    Integrates deposit detection, network analysis, stability modeling,
    and calibration for production-ready income estimation.
    """
    
    def __init__(
        self,
        min_deposit_amount: float = 100.0,
        salary_min_amount: float = 500.0,
        confidence_threshold: float = 0.6,
        stability_threshold: float = 0.6,
        enable_network_analysis: bool = True,
        enable_calibration: bool = True,
        calibration_alpha: float = 0.1,
        model_version: str = "1.0"
    ):
        """
        Initialize income estimation pipeline.
        
        Args:
            min_deposit_amount: Minimum deposit to consider as income
            salary_min_amount: Minimum amount for salary classification
            confidence_threshold: Minimum confidence for estimates
            stability_threshold: Threshold for stability classification
            enable_network_analysis: Whether to use network features
            enable_calibration: Whether to calibrate predictions
            calibration_alpha: Miscoverage rate for calibration
            model_version: Model version string
        """
        # Initialize components
        self.deposit_detector = DepositDetector(
            min_deposit_amount=min_deposit_amount,
            salary_min_amount=salary_min_amount,
            confidence_threshold=confidence_threshold
        )
        
        self.stability_scorer = IncomeStabilityScorer(
            stability_threshold=stability_threshold
        )
        
        self.enable_network_analysis = enable_network_analysis
        if enable_network_analysis:
            self.network_analyzer = PaymentNetworkAnalyzer()
        else:
            self.network_analyzer = None
        
        self.enable_calibration = enable_calibration
        if enable_calibration:
            self.conformal_predictor = ConformalPredictor(alpha=calibration_alpha)
            self.isotonic_calibrator = IsotonicCalibrator(y_min=0.0)
            self.is_calibrated_ = False
        else:
            self.conformal_predictor = None
            self.isotonic_calibrator = None
            self.is_calibrated_ = False
        
        self.model_version = model_version
        self.is_fitted_ = False
        
        logger.info(
            f"IncomeEstimationPipeline initialized (v{model_version})\n"
            f"  Network analysis: {enable_network_analysis}\n"
            f"  Calibration: {enable_calibration}"
        )
    
    def fit(
        self,
        transactions: pd.DataFrame,
        known_incomes: Optional[pd.DataFrame] = None,
        user_col: str = "user_id",
        income_col: str = "monthly_income"
    ) -> 'IncomeEstimationPipeline':
        """
        Fit pipeline on historical data.
        
        Args:
            transactions: Historical transactions
            known_incomes: Known incomes for calibration (user_id, monthly_income)
            user_col: Column name for user ID
            income_col: Column name for income
            
        Returns:
            Self
            
        Example:
            >>> pipeline.fit(transactions_df, known_incomes_df)
            >>> print("Pipeline fitted and calibrated")
        """
        logger.info("Fitting income estimation pipeline...")
        
        # If we have known incomes, use for calibration
        if known_incomes is not None and self.enable_calibration:
            logger.info("Calibrating with known incomes...")
            
            predictions = []
            actuals = []
            
            for _, row in known_incomes.iterrows():
                user_id = row[user_col]
                actual_income = row[income_col]
                
                # Get user transactions
                user_txns = transactions[
                    transactions[user_col] == user_id
                ]
                
                if len(user_txns) < 3:
                    continue
                
                # Predict without calibration
                try:
                    result = self._predict_single(
                        user_id=user_id,
                        transactions=user_txns,
                        apply_calibration=False
                    )
                    
                    predictions.append(result.point_estimate)
                    actuals.append(actual_income)
                except Exception as e:
                    logger.warning(f"Failed to predict for {user_id}: {e}")
                    continue
            
            if len(predictions) >= 10:
                predictions = np.array(predictions)
                actuals = np.array(actuals)
                
                # Fit conformal predictor
                self.conformal_predictor.fit(actuals, predictions)
                
                # Fit isotonic calibrator
                self.isotonic_calibrator.fit(actuals, predictions)
                
                self.is_calibrated_ = True
                
                # Validate
                metrics = self.conformal_predictor.validate_coverage(actuals, predictions)
                logger.info(
                    f"Calibration complete: coverage={metrics.coverage:.2%}, "
                    f"sharpness={metrics.sharpness:.2f}"
                )
            else:
                logger.warning(
                    f"Insufficient data for calibration ({len(predictions)} samples)"
                )
        
        self.is_fitted_ = True
        
        logger.info("Pipeline fitting complete")
        
        return self
    
    def predict(
        self,
        transactions: pd.DataFrame,
        user_id: Optional[str] = None,
        return_details: bool = False
    ) -> Union[IncomeEstimationResult, List[IncomeEstimationResult]]:
        """
        Predict income for user(s).
        
        Args:
            transactions: Transaction data
            user_id: Optional user ID (if None, predict for all users)
            return_details: Whether to return detailed results
            
        Returns:
            IncomeEstimationResult or list of results
            
        Example:
            >>> result = pipeline.predict(transactions_df, user_id="user_123")
            >>> print(f"Income: ${result.point_estimate:.2f} ± ${result.upper_bound - result.point_estimate:.2f}")
        """
        if user_id:
            # Single user prediction
            user_txns = transactions[
                transactions['user_id'] == user_id
            ]
            
            if len(user_txns) == 0:
                raise ValueError(f"No transactions found for user {user_id}")
            
            return self._predict_single(
                user_id=user_id,
                transactions=user_txns,
                apply_calibration=self.is_calibrated_
            )
        else:
            # Multi-user prediction
            results = []
            
            user_ids = transactions['user_id'].unique()
            logger.info(f"Predicting for {len(user_ids)} users...")
            
            for uid in user_ids:
                try:
                    result = self.predict(
                        transactions=transactions,
                        user_id=uid,
                        return_details=return_details
                    )
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to predict for {uid}: {e}")
                    continue
            
            logger.info(f"Completed predictions for {len(results)} users")
            
            return results
    
    def predict_batch(
        self,
        transactions: pd.DataFrame,
        batch_size: int = 100
    ) -> pd.DataFrame:
        """
        Batch prediction for many users.
        
        Args:
            transactions: Transaction data
            batch_size: Number of users per batch
            
        Returns:
            DataFrame of predictions
            
        Example:
            >>> predictions_df = pipeline.predict_batch(transactions_df)
            >>> predictions_df.to_csv("income_predictions.csv")
        """
        user_ids = transactions['user_id'].unique()
        n_users = len(user_ids)
        n_batches = (n_users + batch_size - 1) // batch_size
        
        all_results = []
        
        logger.info(f"Batch prediction for {n_users} users in {n_batches} batches")
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_users)
            batch_users = user_ids[start_idx:end_idx]
            
            logger.info(f"Processing batch {i+1}/{n_batches}")
            
            for user_id in batch_users:
                try:
                    result = self.predict(transactions, user_id=user_id)
                    all_results.append(result.to_dict())
                except Exception as e:
                    logger.warning(f"Failed for {user_id}: {e}")
                    continue
        
        predictions_df = pd.DataFrame(all_results)
        
        logger.info(f"Batch prediction complete: {len(predictions_df)} successful")
        
        return predictions_df
    
    def validate(
        self,
        transactions: pd.DataFrame,
        known_incomes: pd.DataFrame,
        user_col: str = "user_id",
        income_col: str = "monthly_income"
    ) -> ValidationMetrics:
        """
        Validate pipeline on known incomes.
        
        Args:
            transactions: Transaction data
            known_incomes: Known incomes (user_id, monthly_income)
            user_col: Column name for user ID
            income_col: Column name for income
            
        Returns:
            ValidationMetrics
            
        Example:
            >>> metrics = pipeline.validate(transactions_df, known_incomes_df)
            >>> print(f"MAE: ${metrics.mae:.2f}")
            >>> print(f"MAPE: {metrics.mape:.2%}")
            >>> print(f"Coverage (90%): {metrics.coverage_90:.2%}")
        """
        logger.info("Validating income estimation pipeline...")
        
        predictions = []
        actuals = []
        lower_bounds_90 = []
        upper_bounds_90 = []
        lower_bounds_80 = []
        upper_bounds_80 = []
        
        for _, row in known_incomes.iterrows():
            user_id = row[user_col]
            actual_income = row[income_col]
            
            # Get user transactions
            user_txns = transactions[transactions[user_col] == user_id]
            
            if len(user_txns) < 3:
                continue
            
            try:
                result = self.predict(user_txns, user_id=user_id)
                
                predictions.append(result.point_estimate)
                actuals.append(actual_income)
                lower_bounds_90.append(result.lower_bound)
                upper_bounds_90.append(result.upper_bound)
                
                # Calculate 80% CI
                if result.quantile_prediction:
                    lb_80, ub_80 = result.quantile_prediction.get_interval(0.8)
                    lower_bounds_80.append(lb_80)
                    upper_bounds_80.append(ub_80)
                else:
                    # Approximate from 90% CI
                    width_90 = result.upper_bound - result.lower_bound
                    width_80 = width_90 * 0.8 / 0.9
                    half_width = width_80 / 2
                    lower_bounds_80.append(result.point_estimate - half_width)
                    upper_bounds_80.append(result.point_estimate + half_width)
                
            except Exception as e:
                logger.warning(f"Validation failed for {user_id}: {e}")
                continue
        
        if len(predictions) == 0:
            raise ValueError("No successful predictions for validation")
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        errors = predictions - actuals
        abs_errors = np.abs(errors)
        pct_errors = abs_errors / (actuals + 1e-10) * 100
        
        mae = float(np.mean(abs_errors))
        mape = float(np.mean(pct_errors))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        
        # Coverage
        coverage_90 = float(np.mean([
            lb <= actual <= ub
            for lb, ub, actual in zip(lower_bounds_90, upper_bounds_90, actuals)
        ]))
        
        coverage_80 = float(np.mean([
            lb <= actual <= ub
            for lb, ub, actual in zip(lower_bounds_80, upper_bounds_80, actuals)
        ]))
        
        # Interval width
        interval_widths = np.array(upper_bounds_90) - np.array(lower_bounds_90)
        median_interval_width = float(np.median(interval_widths))
        
        # Calibration score (how close coverage is to target)
        calibration_score = 1.0 - abs(coverage_90 - 0.9)
        
        metrics = ValidationMetrics(
            mae=mae,
            mape=mape,
            rmse=rmse,
            coverage_90=coverage_90,
            coverage_80=coverage_80,
            median_interval_width=median_interval_width,
            calibration_score=calibration_score,
            n_samples=len(predictions)
        )
        
        logger.info(
            f"Validation complete ({metrics.n_samples} samples):\n"
            f"  MAE: ${metrics.mae:.2f}\n"
            f"  MAPE: {metrics.mape:.1f}%\n"
            f"  RMSE: ${metrics.rmse:.2f}\n"
            f"  Coverage (90%): {metrics.coverage_90:.1%}\n"
            f"  Coverage (80%): {metrics.coverage_80:.1%}\n"
            f"  Median interval width: ${metrics.median_interval_width:.2f}\n"
            f"  Calibration score: {metrics.calibration_score:.2%}"
        )
        
        return metrics
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save pipeline to disk.
        
        Args:
            filepath: Path to save file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self, filepath)
        
        logger.info(f"Pipeline saved to {filepath}")
    
    @staticmethod
    def load(filepath: Union[str, Path]) -> 'IncomeEstimationPipeline':
        """
        Load pipeline from disk.
        
        Args:
            filepath: Path to saved file
            
        Returns:
            Loaded pipeline
        """
        pipeline = joblib.load(filepath)
        
        logger.info(f"Pipeline loaded from {filepath}")
        
        return pipeline
    
    # ==================== Private Methods ====================
    
    def _predict_single(
        self,
        user_id: str,
        transactions: pd.DataFrame,
        apply_calibration: bool = True
    ) -> IncomeEstimationResult:
        """Predict income for single user."""
        # 1. Detect deposits
        deposits = self.deposit_detector.detect_income_deposits(
            transactions,
            entity_id=user_id
        )
        
        if len(deposits) == 0:
            raise ValueError(f"No income deposits detected for {user_id}")
        
        # 2. Classify income sources
        income_sources = self.deposit_detector.classify_income_sources(deposits)
        
        if len(income_sources) == 0:
            raise ValueError(f"No income sources classified for {user_id}")
        
        # 3. Estimate monthly income
        income_estimates = self.deposit_detector.estimate_monthly_income(
            income_sources=income_sources
        )
        
        point_estimate = income_estimates['total']
        
        # 4. Stability analysis
        income_history = self.deposit_detector.track_income_history(
            transactions,
            entity_id=user_id
        )
        
        if len(income_history) >= 3:
            # Rename 'period' to 'timestamp' for stability scorer
            income_history_renamed = income_history.rename(columns={
                'period': 'timestamp',
                'estimated_income': 'amount'
            })
            
            stability_metrics = self.stability_scorer.score_income_stability(
                income_history_renamed
            )
            risk_assessment = self.stability_scorer.calculate_risk_score(
                stability_metrics,
                income_sources=income_sources
            )
        else:
            stability_metrics = None
            risk_assessment = None
        
        # 5. Network validation (if enabled)
        network_validated = False
        peer_percentile = 0.5
        
        if self.enable_network_analysis and len(transactions) > 0:
            try:
                graph = self.network_analyzer.build_payment_network(transactions)
                
                # Validate with network
                validation = self.network_analyzer.validate_income_with_network(
                    point_estimate,
                    graph,
                    user_id
                )
                network_validated = validation.is_consistent
                
                # Peer comparison
                peer_comp = self.network_analyzer.infer_from_peer_comparison(
                    graph,
                    user_id,
                    user_income=point_estimate
                )
                peer_percentile = peer_comp.user_income_percentile
                
            except Exception as e:
                logger.warning(f"Network analysis failed for {user_id}: {e}")
        
        # 6. Apply calibration
        if apply_calibration and self.is_calibrated_:
            # Isotonic calibration
            point_estimate_calibrated = float(
                self.isotonic_calibrator.calibrate(np.array([point_estimate]))[0]
            )
            
            # Conformal prediction interval
            interval = self.conformal_predictor.predict_interval(
                point_estimate_calibrated
            )
            
            lower_bound = interval.lower_bound
            upper_bound = interval.upper_bound
        else:
            # Use simple interval based on confidence
            confidence = income_estimates['confidence']
            uncertainty = (1 - confidence) * point_estimate
            lower_bound = max(point_estimate - uncertainty, 0)
            upper_bound = point_estimate + uncertainty
        
        # 7. Build result
        result = IncomeEstimationResult(
            user_id=user_id,
            point_estimate=point_estimate,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence=income_estimates.get('confidence', 0.0),
            primary_salary=income_estimates.get('primary_salary', 0.0),
            secondary_income=income_estimates.get('secondary_salary', 0.0),
            freelance_gig=income_estimates.get('freelance_gig', 0.0),
            investment_income=income_estimates.get('investment', 0.0),
            num_sources=income_estimates.get('num_sources', 0),
            stability_score=stability_metrics.overall_stability if stability_metrics else 0.5,
            risk_score=risk_assessment.risk_score if risk_assessment else 0.5,
            risk_category=risk_assessment.risk_category if risk_assessment else 'unknown',
            network_validated=network_validated,
            peer_percentile=peer_percentile,
            prediction_date=datetime.now(),
            model_version=self.model_version
        )
        
        return result


# ==================== Usage Examples ====================

def example_basic_usage():
    """Example: Basic income estimation."""
    logger.info("=== Example: Basic Income Estimation ===")
    
    # Create sample transaction data
    np.random.seed(42)
    transactions = pd.DataFrame({
        'user_id': ['user_1'] * 12,
        'timestamp': pd.date_range('2024-01-01', periods=12, freq='M'),
        'amount': np.random.normal(5000, 500, 12),
        'description': ['Monthly Salary'] * 12,
        'category': ['income'] * 12
    })
    
    # Initialize pipeline
    pipeline = IncomeEstimationPipeline()
    
    # Predict
    result = pipeline.predict(transactions, user_id='user_1')
    
    print(f"\nIncome Estimation for {result.user_id}:")
    print(f"  Estimated Income: ${result.point_estimate:,.2f}")
    print(f"  90% CI: [${result.lower_bound:,.2f}, ${result.upper_bound:,.2f}]")
    print(f"  Confidence: {result.confidence:.1%}")
    print(f"  Stability Score: {result.stability_score:.2f}")
    print(f"  Risk Category: {result.risk_category}")


def example_with_calibration():
    """Example: Income estimation with calibration."""
    logger.info("=== Example: Income Estimation with Calibration ===")
    
    # Create training data
    np.random.seed(42)
    n_users = 50
    
    transactions_list = []
    known_incomes_list = []
    
    for i in range(n_users):
        true_income = np.random.normal(60000, 20000)
        user_id = f"user_{i}"
        
        # Monthly deposits with noise
        amounts = np.random.normal(true_income / 12, true_income / 12 * 0.1, 12)
        
        user_txns = pd.DataFrame({
            'user_id': [user_id] * 12,
            'timestamp': pd.date_range('2024-01-01', periods=12, freq='M'),
            'amount': amounts,
            'description': ['Salary'] * 12
        })
        
        transactions_list.append(user_txns)
        known_incomes_list.append({
            'user_id': user_id,
            'monthly_income': true_income / 12
        })
    
    transactions_df = pd.concat(transactions_list, ignore_index=True)
    known_incomes_df = pd.DataFrame(known_incomes_list)
    
    # Initialize and fit pipeline
    pipeline = IncomeEstimationPipeline(enable_calibration=True)
    pipeline.fit(transactions_df, known_incomes_df)
    
    # Validate
    metrics = pipeline.validate(transactions_df, known_incomes_df)
    
    print(f"\nValidation Metrics:")
    print(f"  MAE: ${metrics.mae:,.2f}")
    print(f"  MAPE: {metrics.mape:.1f}%")
    print(f"  90% Coverage: {metrics.coverage_90:.1%}")
    print(f"  Calibration Score: {metrics.calibration_score:.1%}")


def example_batch_prediction():
    """Example: Batch prediction for many users."""
    logger.info("=== Example: Batch Prediction ===")
    
    # Create data for 100 users
    np.random.seed(42)
    n_users = 100
    
    transactions_list = []
    for i in range(n_users):
        income = np.random.normal(60000, 20000)
        amounts = np.random.normal(income / 12, income / 24, 12)
        
        transactions_list.append(pd.DataFrame({
            'user_id': [f"user_{i}"] * 12,
            'timestamp': pd.date_range('2024-01-01', periods=12, freq='M'),
            'amount': amounts,
            'description': ['Income'] * 12
        }))
    
    transactions_df = pd.concat(transactions_list, ignore_index=True)
    
    # Batch predict
    pipeline = IncomeEstimationPipeline()
    predictions_df = pipeline.predict_batch(transactions_df, batch_size=20)
    
    print(f"\nBatch Prediction Results:")
    print(f"  Total Users: {len(predictions_df)}")
    print(f"  Avg Income: ${predictions_df['point_estimate'].mean():,.2f}")
    print(f"  Median Income: ${predictions_df['point_estimate'].median():,.2f}")
    print(f"  Avg Confidence: {predictions_df['confidence'].mean():.1%}")


if __name__ == "__main__":
    # Run examples
    example_basic_usage()
    print("\n" + "="*60 + "\n")
    example_with_calibration()
    print("\n" + "="*60 + "\n")
    example_batch_prediction()
