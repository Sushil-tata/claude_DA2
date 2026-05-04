"""
Behavioral Stability Modeling for Income Estimation

Implements stability and behavioral pattern analysis:
- Volatility metrics (CV, standard deviation, range)
- Trend analysis (increasing/decreasing/stable)
- Seasonal decomposition for income patterns
- Irregularity detection and scoring
- Employment stability indicators
- Multiple income source stability analysis
- Risk scoring based on stability
- Temporal consistency metrics

Example:
    >>> from src.use_cases.income_estimation.stability_model import IncomeStabilityScorer
    >>> 
    >>> scorer = IncomeStabilityScorer()
    >>> stability = scorer.score_income_stability(income_history_df)
    >>> risk_score = scorer.calculate_risk_score(stability)
    >>> trends = scorer.analyze_trends(income_history_df)

Author: Principal Data Science Decision Agent
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats, signal
from sklearn.linear_model import LinearRegression

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger.warning("statsmodels not available. Install with: pip install statsmodels")


@dataclass
class StabilityMetrics:
    """Container for income stability metrics."""
    
    volatility_cv: float  # Coefficient of variation
    volatility_std: float  # Standard deviation
    range_pct: float  # (max - min) / median
    trend_direction: str  # 'increasing', 'stable', 'decreasing'
    trend_strength: float  # R-squared of trend
    trend_slope: float  # Monthly change amount
    seasonality_strength: float  # 0-1, strength of seasonal pattern
    irregularity_score: float  # 0-1, higher = more irregular
    employment_stability: float  # 0-1, based on source consistency
    overall_stability: float  # 0-1, composite score
    is_stable: bool  # True if overall_stability >= threshold
    

@dataclass
class RiskAssessment:
    """Container for risk assessment based on stability."""
    
    risk_score: float  # 0-1, higher = more risk
    risk_category: str  # 'low', 'medium', 'high', 'very_high'
    income_volatility_risk: float
    trend_risk: float
    irregularity_risk: float
    multi_source_risk: float
    confidence: float
    flags: List[str]


@dataclass
class TrendAnalysis:
    """Container for trend analysis results."""
    
    direction: str  # 'increasing', 'stable', 'decreasing'
    slope: float  # Monthly change
    slope_pct: float  # Monthly % change
    r_squared: float  # Trend strength
    p_value: float  # Statistical significance
    is_significant: bool
    forecast_3m: float  # 3-month forecast
    forecast_6m: float  # 6-month forecast
    forecast_12m: float  # 12-month forecast


class IncomeStabilityScorer:
    """
    Score income stability using multiple behavioral metrics.
    
    Analyzes income patterns over time to assess stability, detect trends,
    and quantify risk associated with income variability.
    """
    
    def __init__(
        self,
        stability_threshold: float = 0.6,
        high_volatility_cv: float = 0.3,
        trend_significance: float = 0.05,
        seasonality_threshold: float = 0.3,
        timestamp_col: str = "timestamp",
        amount_col: str = "amount",
        source_col: Optional[str] = "source_type",
    ):
        """
        Initialize stability scorer.
        
        Args:
            stability_threshold: Minimum score for "stable" classification
            high_volatility_cv: CV threshold for high volatility
            trend_significance: P-value threshold for trend significance
            seasonality_threshold: Minimum strength for seasonal pattern
            timestamp_col: Column name for timestamp
            amount_col: Column name for income amount
            source_col: Column name for income source type
        """
        self.stability_threshold = stability_threshold
        self.high_volatility_cv = high_volatility_cv
        self.trend_significance = trend_significance
        self.seasonality_threshold = seasonality_threshold
        
        self.timestamp_col = timestamp_col
        self.amount_col = amount_col
        self.source_col = source_col
        
        logger.info("IncomeStabilityScorer initialized")
    
    def score_income_stability(
        self,
        income_history: pd.DataFrame,
        min_observations: int = 6
    ) -> StabilityMetrics:
        """
        Calculate comprehensive stability metrics.
        
        Args:
            income_history: Historical income data
            min_observations: Minimum data points required
            
        Returns:
            StabilityMetrics object
            
        Example:
            >>> stability = scorer.score_income_stability(income_df)
            >>> print(f"Overall stability: {stability.overall_stability:.2f}")
            >>> print(f"Is stable: {stability.is_stable}")
        """
        df = income_history.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[self.timestamp_col]):
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
        
        # Sort by timestamp
        df = df.sort_values(self.timestamp_col)
        
        if len(df) < min_observations:
            logger.warning(
                f"Insufficient data: {len(df)} < {min_observations} observations"
            )
            return self._default_stability_metrics()
        
        amounts = df[self.amount_col].values
        
        # 1. Volatility metrics
        volatility_cv = float(np.std(amounts) / np.mean(amounts)) if np.mean(amounts) > 0 else float('inf')
        volatility_std = float(np.std(amounts))
        range_pct = float((np.max(amounts) - np.min(amounts)) / np.median(amounts)) if np.median(amounts) > 0 else float('inf')
        
        # 2. Trend analysis
        trend = self.analyze_trends(df)
        
        # 3. Seasonality
        seasonality_strength = self._calculate_seasonality(df)
        
        # 4. Irregularity score
        irregularity_score = self._calculate_irregularity(df)
        
        # 5. Employment stability (source consistency)
        employment_stability = self._calculate_employment_stability(df)
        
        # 6. Overall stability (composite)
        overall_stability = self._calculate_overall_stability(
            volatility_cv=volatility_cv,
            trend_strength=trend.r_squared,
            irregularity=irregularity_score,
            employment_stability=employment_stability
        )
        
        is_stable = overall_stability >= self.stability_threshold
        
        return StabilityMetrics(
            volatility_cv=volatility_cv,
            volatility_std=volatility_std,
            range_pct=range_pct,
            trend_direction=trend.direction,
            trend_strength=trend.r_squared,
            trend_slope=trend.slope,
            seasonality_strength=seasonality_strength,
            irregularity_score=irregularity_score,
            employment_stability=employment_stability,
            overall_stability=overall_stability,
            is_stable=is_stable
        )
    
    def analyze_trends(
        self,
        income_history: pd.DataFrame
    ) -> TrendAnalysis:
        """
        Analyze income trends and forecast.
        
        Args:
            income_history: Historical income data
            
        Returns:
            TrendAnalysis object
            
        Example:
            >>> trends = scorer.analyze_trends(income_df)
            >>> print(f"Direction: {trends.direction}")
            >>> print(f"12-month forecast: ${trends.forecast_12m:.2f}")
        """
        df = income_history.copy()
        
        # Ensure timestamp is datetime and sorted
        if not pd.api.types.is_datetime64_any_dtype(df[self.timestamp_col]):
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
        df = df.sort_values(self.timestamp_col)
        
        if len(df) < 3:
            return self._default_trend_analysis()
        
        # Prepare data for regression
        df['time_index'] = range(len(df))
        X = df['time_index'].values.reshape(-1, 1)
        y = df[self.amount_col].values
        
        # Fit linear regression
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate metrics
        y_pred = model.predict(X)
        
        # Slope (per period)
        slope = float(model.coef_[0])
        
        # R-squared
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
        
        # Statistical significance
        n = len(df)
        if n > 2:
            # Calculate p-value for slope
            residuals = y - y_pred
            se = np.sqrt(np.sum(residuals**2) / (n - 2))
            se_slope = se / np.sqrt(np.sum((X.flatten() - X.mean())**2))
            t_stat = slope / se_slope if se_slope > 0 else 0
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        else:
            p_value = 1.0
        
        is_significant = p_value < self.trend_significance
        
        # Determine direction
        if not is_significant or abs(slope) < 0.01 * np.mean(y):
            direction = 'stable'
        elif slope > 0:
            direction = 'increasing'
        else:
            direction = 'decreasing'
        
        # Slope as percentage of mean
        slope_pct = float(slope / np.mean(y) * 100) if np.mean(y) > 0 else 0.0
        
        # Forecasts
        last_index = len(df) - 1
        forecast_3m = float(model.predict([[last_index + 3]])[0])
        forecast_6m = float(model.predict([[last_index + 6]])[0])
        forecast_12m = float(model.predict([[last_index + 12]])[0])
        
        return TrendAnalysis(
            direction=direction,
            slope=slope,
            slope_pct=slope_pct,
            r_squared=r_squared,
            p_value=float(p_value),
            is_significant=is_significant,
            forecast_3m=max(forecast_3m, 0),  # Non-negative
            forecast_6m=max(forecast_6m, 0),
            forecast_12m=max(forecast_12m, 0)
        )
    
    def calculate_risk_score(
        self,
        stability_metrics: StabilityMetrics,
        income_sources: Optional[List] = None
    ) -> RiskAssessment:
        """
        Calculate risk score based on stability metrics.
        
        Args:
            stability_metrics: Stability metrics
            income_sources: Optional list of income sources
            
        Returns:
            RiskAssessment object
            
        Example:
            >>> risk = scorer.calculate_risk_score(stability)
            >>> print(f"Risk category: {risk.risk_category}")
            >>> print(f"Risk score: {risk.risk_score:.2f}")
        """
        flags = []
        
        # 1. Volatility risk
        if stability_metrics.volatility_cv > self.high_volatility_cv:
            volatility_risk = min(
                stability_metrics.volatility_cv / (self.high_volatility_cv * 2),
                1.0
            )
            flags.append('high_volatility')
        else:
            volatility_risk = stability_metrics.volatility_cv / self.high_volatility_cv
        
        # 2. Trend risk
        if stability_metrics.trend_direction == 'decreasing':
            trend_risk = 0.8
            flags.append('decreasing_trend')
        elif stability_metrics.trend_direction == 'increasing':
            trend_risk = 0.2
        else:
            trend_risk = 0.4
        
        # Adjust by trend strength
        trend_risk = trend_risk * stability_metrics.trend_strength + 0.4 * (1 - stability_metrics.trend_strength)
        
        # 3. Irregularity risk
        irregularity_risk = stability_metrics.irregularity_score
        if irregularity_risk > 0.7:
            flags.append('high_irregularity')
        
        # 4. Multi-source risk
        if income_sources:
            num_sources = len(income_sources)
            active_sources = sum(1 for s in income_sources if getattr(s, 'is_active', True))
            
            if num_sources > 3:
                # Many sources can indicate instability
                multi_source_risk = 0.6
                flags.append('multiple_income_sources')
            elif active_sources == 0:
                multi_source_risk = 1.0
                flags.append('no_active_sources')
            else:
                # Fewer active sources = higher risk
                multi_source_risk = 1.0 - (active_sources / max(num_sources, 1))
        else:
            multi_source_risk = 0.5  # Unknown
        
        # 5. Overall risk score (weighted)
        risk_score = (
            0.35 * volatility_risk +
            0.25 * trend_risk +
            0.25 * irregularity_risk +
            0.15 * multi_source_risk
        )
        
        # 6. Risk category
        if risk_score < 0.3:
            risk_category = 'low'
        elif risk_score < 0.5:
            risk_category = 'medium'
        elif risk_score < 0.7:
            risk_category = 'high'
        else:
            risk_category = 'very_high'
            flags.append('very_high_risk')
        
        # 7. Confidence in risk assessment
        confidence = min(
            stability_metrics.employment_stability * 0.5 +
            (1.0 if stability_metrics.trend_strength > 0.3 else 0.5) * 0.5,
            1.0
        )
        
        return RiskAssessment(
            risk_score=float(risk_score),
            risk_category=risk_category,
            income_volatility_risk=float(volatility_risk),
            trend_risk=float(trend_risk),
            irregularity_risk=float(irregularity_risk),
            multi_source_risk=float(multi_source_risk),
            confidence=float(confidence),
            flags=flags
        )
    
    def detect_income_shocks(
        self,
        income_history: pd.DataFrame,
        threshold_std: float = 2.0
    ) -> pd.DataFrame:
        """
        Detect sudden changes (shocks) in income.
        
        Args:
            income_history: Historical income data
            threshold_std: Number of std devs for shock detection
            
        Returns:
            DataFrame of detected shocks
            
        Example:
            >>> shocks = scorer.detect_income_shocks(income_df)
            >>> print(f"Detected {len(shocks)} income shocks")
        """
        df = income_history.copy()
        
        if len(df) < 5:
            return pd.DataFrame()
        
        # Calculate rolling statistics
        df['rolling_mean'] = df[self.amount_col].rolling(window=3, center=True).mean()
        df['rolling_std'] = df[self.amount_col].rolling(window=3, center=True).std()
        
        # Calculate z-scores
        df['z_score'] = (
            (df[self.amount_col] - df['rolling_mean']) / df['rolling_std']
        )
        
        # Detect shocks
        df['is_shock'] = df['z_score'].abs() > threshold_std
        
        # Extract shocks
        shocks = df[df['is_shock']].copy()
        shocks['shock_magnitude'] = shocks['z_score'].abs()
        shocks['shock_direction'] = shocks['z_score'].apply(
            lambda x: 'positive' if x > 0 else 'negative'
        )
        
        logger.info(f"Detected {len(shocks)} income shocks")
        
        return shocks[[
            self.timestamp_col,
            self.amount_col,
            'rolling_mean',
            'z_score',
            'shock_magnitude',
            'shock_direction'
        ]]
    
    def assess_employment_consistency(
        self,
        income_history: pd.DataFrame,
        window_months: int = 6
    ) -> Dict[str, float]:
        """
        Assess consistency of employment/income sources.
        
        Args:
            income_history: Historical income with source information
            window_months: Time window for analysis
            
        Returns:
            Dictionary of employment consistency metrics
            
        Example:
            >>> consistency = scorer.assess_employment_consistency(income_df)
            >>> print(f"Source stability: {consistency['source_stability']:.2f}")
        """
        df = income_history.copy()
        
        if self.source_col not in df.columns or len(df) < 3:
            return {
                'source_stability': 0.5,
                'source_diversity': 0.0,
                'primary_source_dominance': 0.0
            }
        
        # Count source changes
        df = df.sort_values(self.timestamp_col)
        source_changes = (df[self.source_col] != df[self.source_col].shift()).sum()
        source_stability = 1.0 - min(source_changes / len(df), 1.0)
        
        # Source diversity (normalized entropy)
        source_counts = df[self.source_col].value_counts()
        source_probs = source_counts / source_counts.sum()
        entropy = -np.sum(source_probs * np.log2(source_probs + 1e-10))
        max_entropy = np.log2(len(source_counts)) if len(source_counts) > 1 else 1.0
        source_diversity = float(entropy / max_entropy) if max_entropy > 0 else 0.0
        
        # Primary source dominance
        primary_source_pct = float(source_counts.iloc[0] / source_counts.sum())
        
        return {
            'source_stability': float(source_stability),
            'source_diversity': source_diversity,
            'primary_source_dominance': primary_source_pct,
            'num_unique_sources': len(source_counts)
        }
    
    # ==================== Private Helper Methods ====================
    
    def _calculate_seasonality(
        self,
        df: pd.DataFrame
    ) -> float:
        """Calculate seasonal pattern strength."""
        if len(df) < 12 or not STATSMODELS_AVAILABLE:
            # Fallback: simple month-based variance
            df_copy = df.copy()
            df_copy['month'] = df_copy[self.timestamp_col].dt.month
            monthly_avg = df_copy.groupby('month')[self.amount_col].mean()
            
            if len(monthly_avg) < 3:
                return 0.0
            
            cv = monthly_avg.std() / monthly_avg.mean() if monthly_avg.mean() > 0 else 0.0
            return float(min(cv, 1.0))
        
        try:
            # Use seasonal decomposition
            # Resample to monthly if needed
            df_copy = df.set_index(self.timestamp_col)
            monthly = df_copy[self.amount_col].resample('M').mean().dropna()
            
            if len(monthly) >= 24:  # Need 2+ years for annual seasonality
                decomposition = seasonal_decompose(
                    monthly,
                    model='additive',
                    period=12,
                    extrapolate_trend='freq'
                )
                
                # Strength of seasonality
                seasonal_var = np.var(decomposition.seasonal)
                residual_var = np.var(decomposition.resid.dropna())
                
                if seasonal_var + residual_var > 0:
                    strength = seasonal_var / (seasonal_var + residual_var)
                else:
                    strength = 0.0
                
                return float(min(strength, 1.0))
            else:
                return 0.0
        except Exception as e:
            logger.warning(f"Seasonality calculation failed: {e}")
            return 0.0
    
    def _calculate_irregularity(
        self,
        df: pd.DataFrame
    ) -> float:
        """Calculate irregularity score based on timing and amount."""
        if len(df) < 3:
            return 1.0
        
        df_sorted = df.sort_values(self.timestamp_col)
        
        # Time irregularity (variance of intervals)
        intervals = df_sorted[self.timestamp_col].diff().dt.days.dropna()
        if len(intervals) > 0:
            interval_cv = intervals.std() / intervals.mean() if intervals.mean() > 0 else 1.0
            time_irregularity = min(interval_cv, 1.0)
        else:
            time_irregularity = 0.5
        
        # Amount irregularity (CV)
        amounts = df_sorted[self.amount_col]
        amount_cv = amounts.std() / amounts.mean() if amounts.mean() > 0 else 1.0
        amount_irregularity = min(amount_cv, 1.0)
        
        # Combined irregularity
        irregularity = 0.6 * time_irregularity + 0.4 * amount_irregularity
        
        return float(irregularity)
    
    def _calculate_employment_stability(
        self,
        df: pd.DataFrame
    ) -> float:
        """Calculate employment stability score."""
        if self.source_col not in df.columns or len(df) < 2:
            return 0.5  # Unknown
        
        # Check source consistency
        df_sorted = df.sort_values(self.timestamp_col)
        source_changes = (
            df_sorted[self.source_col] != df_sorted[self.source_col].shift()
        ).sum()
        
        # Fewer changes = more stable
        stability = 1.0 - min(source_changes / len(df), 1.0)
        
        return float(stability)
    
    def _calculate_overall_stability(
        self,
        volatility_cv: float,
        trend_strength: float,
        irregularity: float,
        employment_stability: float
    ) -> float:
        """Calculate composite stability score."""
        # Convert volatility to stability score
        volatility_stability = 1.0 - min(volatility_cv / self.high_volatility_cv, 1.0)
        
        # Trend stability (higher R² = more predictable)
        trend_stability = trend_strength
        
        # Irregularity to stability
        irregularity_stability = 1.0 - irregularity
        
        # Weighted combination
        overall = (
            0.35 * volatility_stability +
            0.25 * trend_stability +
            0.25 * irregularity_stability +
            0.15 * employment_stability
        )
        
        return float(min(max(overall, 0.0), 1.0))
    
    def _default_stability_metrics(self) -> StabilityMetrics:
        """Return default stability metrics for insufficient data."""
        return StabilityMetrics(
            volatility_cv=0.0,
            volatility_std=0.0,
            range_pct=0.0,
            trend_direction='unknown',
            trend_strength=0.0,
            trend_slope=0.0,
            seasonality_strength=0.0,
            irregularity_score=1.0,
            employment_stability=0.5,
            overall_stability=0.0,
            is_stable=False
        )
    
    def _default_trend_analysis(self) -> TrendAnalysis:
        """Return default trend analysis for insufficient data."""
        return TrendAnalysis(
            direction='unknown',
            slope=0.0,
            slope_pct=0.0,
            r_squared=0.0,
            p_value=1.0,
            is_significant=False,
            forecast_3m=0.0,
            forecast_6m=0.0,
            forecast_12m=0.0
        )
