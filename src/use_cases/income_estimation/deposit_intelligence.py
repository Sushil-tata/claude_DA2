"""
Deposit Transaction Intelligence for Income Estimation

Implements advanced deposit pattern analysis for income detection:
- Salary/income deposit identification
- Regular deposit pattern recognition (monthly, bi-weekly, weekly)
- Income source classification (salary, freelance, gig, investments)
- Amount extraction and normalization
- Confidence scoring for income detection
- Multi-source income aggregation
- Seasonal pattern handling
- Historical income tracking with trend analysis

Example:
    >>> from src.use_cases.income_estimation.deposit_intelligence import DepositDetector
    >>> 
    >>> detector = DepositDetector()
    >>> deposits = detector.detect_income_deposits(transactions_df)
    >>> income_sources = detector.classify_income_sources(deposits)
    >>> monthly_income = detector.estimate_monthly_income(deposits)
    >>> confidence = detector.calculate_confidence(deposits)

Author: Principal Data Science Decision Agent
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN


@dataclass
class DepositPattern:
    """Container for deposit pattern information."""
    
    frequency: str  # 'monthly', 'bi-weekly', 'weekly', 'irregular'
    avg_amount: float
    std_amount: float
    cv_amount: float  # Coefficient of variation
    median_amount: float
    count: int
    confidence: float
    periodicity_days: Optional[float] = None
    source_type: Optional[str] = None  # 'salary', 'freelance', 'gig', 'investment', 'other'
    trend: Optional[str] = None  # 'increasing', 'stable', 'decreasing'
    seasonality: Optional[float] = None  # Seasonal strength 0-1


@dataclass
class IncomeSource:
    """Container for identified income source."""
    
    source_id: str
    source_type: str  # 'primary_salary', 'secondary_salary', 'freelance', 'gig', 'investment'
    pattern: DepositPattern
    transactions: pd.DataFrame
    monthly_estimate: float
    confidence: float
    first_seen: datetime
    last_seen: datetime
    is_active: bool


class DepositDetector:
    """
    Advanced deposit detection and income pattern recognition.
    
    Uses statistical methods, clustering, and periodicity analysis to identify
    and classify income deposits from transaction data.
    """
    
    def __init__(
        self,
        min_deposit_amount: float = 100.0,
        salary_min_amount: float = 500.0,
        regularity_threshold: float = 0.3,
        confidence_threshold: float = 0.6,
        lookback_days: int = 180,
        entity_col: str = "user_id",
        timestamp_col: str = "timestamp",
        amount_col: str = "amount",
        description_col: str = "description",
        category_col: Optional[str] = "category",
    ):
        """
        Initialize deposit detector.
        
        Args:
            min_deposit_amount: Minimum amount to consider as potential income
            salary_min_amount: Minimum amount to classify as salary
            regularity_threshold: CV threshold for regular deposits (lower = more regular)
            confidence_threshold: Minimum confidence for income classification
            lookback_days: Days of history to analyze
            entity_col: Column name for entity/user ID
            timestamp_col: Column name for timestamp
            amount_col: Column name for transaction amount
            description_col: Column name for transaction description
            category_col: Column name for category (optional)
        """
        self.min_deposit_amount = min_deposit_amount
        self.salary_min_amount = salary_min_amount
        self.regularity_threshold = regularity_threshold
        self.confidence_threshold = confidence_threshold
        self.lookback_days = lookback_days
        
        self.entity_col = entity_col
        self.timestamp_col = timestamp_col
        self.amount_col = amount_col
        self.description_col = description_col
        self.category_col = category_col
        
        # Salary keywords for text matching
        self.salary_keywords = [
            'salary', 'payroll', 'wages', 'direct deposit', 'dd', 'paycheck',
            'employer', 'payment from', 'monthly pay', 'biweekly pay', 'weekly pay'
        ]
        
        # Gig economy keywords
        self.gig_keywords = [
            'uber', 'lyft', 'doordash', 'instacart', 'grubhub', 'postmates',
            'upwork', 'fiverr', 'freelance', 'contractor', 'gig'
        ]
        
        # Investment keywords
        self.investment_keywords = [
            'dividend', 'interest', 'capital gain', 'investment', 'broker',
            'stock', 'bond', 'mutual fund', 'etf', 'securities'
        ]
        
        logger.info(f"DepositDetector initialized with lookback={lookback_days} days")
    
    def detect_income_deposits(
        self,
        transactions: pd.DataFrame,
        entity_id: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Identify deposits that are likely income.
        
        Args:
            transactions: Transaction data
            entity_id: Optional entity ID to filter
            
        Returns:
            DataFrame of potential income deposits with confidence scores
            
        Example:
            >>> deposits = detector.detect_income_deposits(transactions_df)
            >>> print(deposits[['amount', 'confidence', 'source_type']].head())
        """
        df = transactions.copy()
        
        # Filter to entity if specified
        if entity_id and self.entity_col in df.columns:
            df = df[df[self.entity_col] == entity_id]
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[self.timestamp_col]):
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
        
        # Filter to lookback period
        cutoff_date = df[self.timestamp_col].max() - pd.Timedelta(days=self.lookback_days)
        df = df[df[self.timestamp_col] >= cutoff_date]
        
        # Filter to deposits only (positive amounts)
        deposits = df[df[self.amount_col] > 0].copy()
        
        # Filter by minimum amount
        deposits = deposits[deposits[self.amount_col] >= self.min_deposit_amount]
        
        if len(deposits) == 0:
            logger.warning("No deposits found meeting criteria")
            return pd.DataFrame()
        
        # Add features for classification
        deposits['is_large_deposit'] = (
            deposits[self.amount_col] >= self.salary_min_amount
        )
        
        # Text-based classification
        if self.description_col in deposits.columns:
            deposits['has_salary_keyword'] = deposits[self.description_col].apply(
                lambda x: self._contains_keywords(x, self.salary_keywords)
            )
            deposits['has_gig_keyword'] = deposits[self.description_col].apply(
                lambda x: self._contains_keywords(x, self.gig_keywords)
            )
            deposits['has_investment_keyword'] = deposits[self.description_col].apply(
                lambda x: self._contains_keywords(x, self.investment_keywords)
            )
        else:
            deposits['has_salary_keyword'] = False
            deposits['has_gig_keyword'] = False
            deposits['has_investment_keyword'] = False
        
        # Sort by timestamp
        deposits = deposits.sort_values(self.timestamp_col)
        
        # Calculate inter-deposit intervals
        deposits['days_since_prev'] = deposits[self.timestamp_col].diff().dt.days
        
        # Cluster similar deposits by amount
        deposits = self._cluster_deposits(deposits)
        
        # Calculate confidence scores
        deposits['confidence'] = deposits.apply(
            lambda row: self._calculate_deposit_confidence(row, deposits),
            axis=1
        )
        
        # Preliminary source type classification
        deposits['source_type'] = deposits.apply(
            self._classify_deposit_type,
            axis=1
        )
        
        logger.info(f"Detected {len(deposits)} potential income deposits")
        
        return deposits
    
    def classify_income_sources(
        self,
        deposits: pd.DataFrame
    ) -> List[IncomeSource]:
        """
        Classify deposits into distinct income sources.
        
        Args:
            deposits: Detected income deposits
            
        Returns:
            List of IncomeSource objects
            
        Example:
            >>> sources = detector.classify_income_sources(deposits)
            >>> for source in sources:
            ...     print(f"{source.source_type}: ${source.monthly_estimate:.2f}")
        """
        if len(deposits) == 0:
            return []
        
        income_sources = []
        
        # Group by cluster and source type
        for (cluster_id, source_type), group in deposits.groupby(
            ['cluster_id', 'source_type'], dropna=False
        ):
            if len(group) < 2:  # Need at least 2 transactions
                continue
            
            # Analyze pattern
            pattern = self._analyze_deposit_pattern(group)
            
            # Skip if low confidence
            if pattern.confidence < self.confidence_threshold:
                continue
            
            # Generate source ID
            source_id = f"{source_type}_{cluster_id}"
            
            # Estimate monthly income
            monthly_estimate = self._estimate_monthly_from_pattern(pattern)
            
            # Create income source
            source = IncomeSource(
                source_id=source_id,
                source_type=source_type,
                pattern=pattern,
                transactions=group,
                monthly_estimate=monthly_estimate,
                confidence=pattern.confidence,
                first_seen=group[self.timestamp_col].min(),
                last_seen=group[self.timestamp_col].max(),
                is_active=self._is_source_active(group)
            )
            
            income_sources.append(source)
        
        # Sort by monthly estimate (descending)
        income_sources.sort(key=lambda x: x.monthly_estimate, reverse=True)
        
        # Relabel primary/secondary salaries
        income_sources = self._relabel_salary_sources(income_sources)
        
        logger.info(f"Identified {len(income_sources)} distinct income sources")
        
        return income_sources
    
    def estimate_monthly_income(
        self,
        deposits: Optional[pd.DataFrame] = None,
        income_sources: Optional[List[IncomeSource]] = None,
        include_inactive: bool = True  # Changed default to True
    ) -> Dict[str, float]:
        """
        Estimate total monthly income from all sources.
        
        Args:
            deposits: Detected deposits (if income_sources not provided)
            income_sources: Classified income sources
            include_inactive: Whether to include inactive sources
            
        Returns:
            Dictionary with income estimates and statistics
            
        Example:
            >>> estimates = detector.estimate_monthly_income(income_sources=sources)
            >>> print(f"Total: ${estimates['total']:.2f}")
            >>> print(f"Confidence: {estimates['confidence']:.2%}")
        """
        if income_sources is None:
            if deposits is None:
                raise ValueError("Must provide either deposits or income_sources")
            income_sources = self.classify_income_sources(deposits)
        
        # Filter to active sources if requested
        if not include_inactive:
            income_sources = [s for s in income_sources if s.is_active]
        
        if len(income_sources) == 0:
            return {
                'total': 0.0,
                'primary_salary': 0.0,
                'secondary_income': 0.0,
                'confidence': 0.0,
                'num_sources': 0
            }
        
        # Calculate totals by type
        primary_salary = sum(
            s.monthly_estimate for s in income_sources
            if s.source_type == 'primary_salary'
        )
        
        secondary_salary = sum(
            s.monthly_estimate for s in income_sources
            if s.source_type == 'secondary_salary'
        )
        
        freelance = sum(
            s.monthly_estimate for s in income_sources
            if s.source_type in ['freelance', 'gig']
        )
        
        investment = sum(
            s.monthly_estimate for s in income_sources
            if s.source_type == 'investment'
        )
        
        total = primary_salary + secondary_salary + freelance + investment
        
        # Calculate weighted confidence
        weighted_confidence = sum(
            s.monthly_estimate * s.confidence for s in income_sources
        ) / total if total > 0 else 0.0
        
        return {
            'total': total,
            'primary_salary': primary_salary,
            'secondary_salary': secondary_salary,
            'freelance_gig': freelance,
            'investment': investment,
            'confidence': weighted_confidence,
            'num_sources': len(income_sources),
            'num_active_sources': sum(1 for s in income_sources if s.is_active)
        }
    
    def calculate_confidence(
        self,
        deposits: pd.DataFrame
    ) -> float:
        """
        Calculate overall confidence in income detection.
        
        Args:
            deposits: Detected deposits
            
        Returns:
            Confidence score 0-1
        """
        if len(deposits) == 0:
            return 0.0
        
        # Factors that increase confidence:
        # 1. Number of deposits (more is better, up to a point)
        count_score = min(len(deposits) / 12, 1.0)  # 12+ deposits = max
        
        # 2. Regularity (low CV of amounts)
        cv = deposits[self.amount_col].std() / deposits[self.amount_col].mean()
        regularity_score = max(0, 1 - cv)
        
        # 3. Consistency of timing
        if 'days_since_prev' in deposits.columns:
            intervals = deposits['days_since_prev'].dropna()
            if len(intervals) > 0:
                interval_cv = intervals.std() / intervals.mean()
                timing_score = max(0, 1 - interval_cv)
            else:
                timing_score = 0.5
        else:
            timing_score = 0.5
        
        # 4. Presence of salary keywords
        keyword_score = deposits.get('has_salary_keyword', pd.Series([False])).mean()
        
        # 5. Large deposit amounts
        amount_score = (
            deposits[self.amount_col] >= self.salary_min_amount
        ).mean()
        
        # Weighted average
        confidence = (
            0.25 * count_score +
            0.25 * regularity_score +
            0.20 * timing_score +
            0.15 * keyword_score +
            0.15 * amount_score
        )
        
        return float(confidence)
    
    def track_income_history(
        self,
        transactions: pd.DataFrame,
        entity_id: Optional[str] = None,
        monthly: bool = True
    ) -> pd.DataFrame:
        """
        Track historical income over time.
        
        Args:
            transactions: Transaction data
            entity_id: Optional entity ID to filter
            monthly: If True, aggregate by month; else by week
            
        Returns:
            DataFrame with historical income estimates
            
        Example:
            >>> history = detector.track_income_history(transactions_df)
            >>> history.plot(x='period', y='estimated_income')
        """
        deposits = self.detect_income_deposits(transactions, entity_id)
        
        if len(deposits) == 0:
            return pd.DataFrame()
        
        # Determine period
        freq = 'M' if monthly else 'W'
        period_label = 'month' if monthly else 'week'
        
        # Group by period
        deposits['period'] = deposits[self.timestamp_col].dt.to_period(freq)
        
        history = []
        for period, group in deposits.groupby('period'):
            # Classify sources for this period
            sources = self.classify_income_sources(group)
            estimates = self.estimate_monthly_income(income_sources=sources)
            
            history.append({
                'period': period.to_timestamp(),
                'period_label': str(period),
                'estimated_income': estimates['total'],
                'primary_salary': estimates.get('primary_salary', 0.0),
                'secondary_income': estimates.get('secondary_salary', 0.0) + estimates.get('freelance_gig', 0.0),
                'investment_income': estimates.get('investment', 0.0),
                'confidence': estimates.get('confidence', 0.0),
                'num_sources': estimates.get('num_sources', 0),
                'num_deposits': len(group)
            })
        
        history_df = pd.DataFrame(history).sort_values('period')
        
        # Calculate trend
        if len(history_df) >= 3:
            history_df['trend'] = self._calculate_trend(
                history_df['estimated_income'].values
            )
        
        logger.info(f"Generated income history with {len(history_df)} periods")
        
        return history_df
    
    # ==================== Private Helper Methods ====================
    
    def _contains_keywords(
        self,
        text: Optional[str],
        keywords: List[str]
    ) -> bool:
        """Check if text contains any of the keywords."""
        if not text or pd.isna(text):
            return False
        text_lower = str(text).lower()
        return any(keyword in text_lower for keyword in keywords)
    
    def _cluster_deposits(
        self,
        deposits: pd.DataFrame
    ) -> pd.DataFrame:
        """Cluster deposits by amount to identify similar deposits."""
        if len(deposits) < 3:
            deposits['cluster_id'] = 0
            return deposits
        
        # Use DBSCAN to cluster by amount
        amounts = deposits[self.amount_col].values.reshape(-1, 1)
        
        # Normalize amounts for clustering
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        amounts_scaled = scaler.fit_transform(amounts)
        
        # Cluster with DBSCAN
        clusterer = DBSCAN(eps=0.3, min_samples=2)
        deposits['cluster_id'] = clusterer.fit_predict(amounts_scaled)
        
        return deposits
    
    def _calculate_deposit_confidence(
        self,
        row: pd.Series,
        deposits: pd.DataFrame
    ) -> float:
        """Calculate confidence score for a single deposit."""
        confidence = 0.0
        
        # Amount-based confidence
        if row[self.amount_col] >= self.salary_min_amount:
            confidence += 0.3
        elif row[self.amount_col] >= self.min_deposit_amount:
            confidence += 0.15
        
        # Keyword-based confidence
        if row.get('has_salary_keyword', False):
            confidence += 0.3
        elif row.get('has_gig_keyword', False):
            confidence += 0.2
        elif row.get('has_investment_keyword', False):
            confidence += 0.15
        
        # Regularity-based confidence (if in a cluster)
        if row.get('cluster_id', -1) >= 0:
            cluster_deposits = deposits[
                deposits['cluster_id'] == row['cluster_id']
            ]
            if len(cluster_deposits) >= 2:
                cv = (
                    cluster_deposits[self.amount_col].std() /
                    cluster_deposits[self.amount_col].mean()
                )
                if cv <= self.regularity_threshold:
                    confidence += 0.25
                else:
                    confidence += 0.1
        
        # Timing regularity
        if row.get('days_since_prev'):
            days = row['days_since_prev']
            # Check if close to common payroll periods
            if 13 <= days <= 17:  # Bi-weekly
                confidence += 0.15
            elif 27 <= days <= 33:  # Monthly
                confidence += 0.15
            elif 6 <= days <= 8:  # Weekly
                confidence += 0.15
        
        return min(confidence, 1.0)
    
    def _classify_deposit_type(self, row: pd.Series) -> str:
        """Classify deposit type based on features."""
        if row.get('has_salary_keyword', False):
            return 'salary'
        elif row.get('has_gig_keyword', False):
            return 'gig'
        elif row.get('has_investment_keyword', False):
            return 'investment'
        elif row.get('is_large_deposit', False):
            return 'salary'
        elif row[self.amount_col] >= self.min_deposit_amount:
            return 'freelance'
        else:
            return 'other'
    
    def _analyze_deposit_pattern(
        self,
        deposits: pd.DataFrame
    ) -> DepositPattern:
        """Analyze pattern of deposit group."""
        amounts = deposits[self.amount_col].values
        
        avg_amount = float(np.mean(amounts))
        std_amount = float(np.std(amounts))
        median_amount = float(np.median(amounts))
        cv_amount = std_amount / avg_amount if avg_amount > 0 else float('inf')
        
        # Determine frequency
        intervals = deposits['days_since_prev'].dropna().values
        if len(intervals) > 0:
            median_interval = np.median(intervals)
            std_interval = np.std(intervals)
            
            # Classify frequency
            if 27 <= median_interval <= 33:
                frequency = 'monthly'
                periodicity_days = median_interval
            elif 13 <= median_interval <= 17:
                frequency = 'bi-weekly'
                periodicity_days = median_interval
            elif 6 <= median_interval <= 8:
                frequency = 'weekly'
                periodicity_days = median_interval
            else:
                frequency = 'irregular'
                periodicity_days = median_interval
        else:
            frequency = 'unknown'
            periodicity_days = None
        
        # Calculate confidence
        confidence_factors = []
        
        # Regularity of amounts
        if cv_amount <= self.regularity_threshold:
            confidence_factors.append(0.9)
        elif cv_amount <= 0.5:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.4)
        
        # Frequency regularity
        if frequency in ['monthly', 'bi-weekly', 'weekly']:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.5)
        
        # Number of observations
        if len(deposits) >= 6:
            confidence_factors.append(0.9)
        elif len(deposits) >= 3:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        confidence = float(np.mean(confidence_factors))
        
        # Detect trend
        trend = self._calculate_trend(amounts)
        
        # Detect seasonality (simplified)
        seasonality = self._detect_seasonality(deposits)
        
        return DepositPattern(
            frequency=frequency,
            avg_amount=avg_amount,
            std_amount=std_amount,
            cv_amount=cv_amount,
            median_amount=median_amount,
            count=len(deposits),
            confidence=confidence,
            periodicity_days=periodicity_days,
            trend=trend,
            seasonality=seasonality
        )
    
    def _estimate_monthly_from_pattern(
        self,
        pattern: DepositPattern
    ) -> float:
        """Estimate monthly income from deposit pattern."""
        if pattern.frequency == 'monthly':
            return pattern.median_amount
        elif pattern.frequency == 'bi-weekly':
            return pattern.median_amount * 2.17  # Avg weeks per month
        elif pattern.frequency == 'weekly':
            return pattern.median_amount * 4.33  # Avg weeks per month
        elif pattern.periodicity_days:
            # Extrapolate based on average interval
            return pattern.median_amount * (30.0 / pattern.periodicity_days)
        else:
            # Fallback: use median amount
            return pattern.median_amount
    
    def _is_source_active(
        self,
        deposits: pd.DataFrame,
        inactive_threshold_days: int = 60
    ) -> bool:
        """Determine if income source is still active."""
        if len(deposits) == 0:
            return False
        
        last_deposit = deposits[self.timestamp_col].max()
        days_since_last = (pd.Timestamp.now() - last_deposit).days
        
        return days_since_last <= inactive_threshold_days
    
    def _relabel_salary_sources(
        self,
        sources: List[IncomeSource]
    ) -> List[IncomeSource]:
        """Relabel salary sources as primary/secondary."""
        salary_sources = [
            s for s in sources if s.source_type == 'salary'
        ]
        
        if len(salary_sources) == 0:
            return sources
        
        # Primary salary is the largest regular salary
        salary_sources.sort(key=lambda x: x.monthly_estimate, reverse=True)
        
        if len(salary_sources) > 0:
            salary_sources[0].source_type = 'primary_salary'
        
        for i in range(1, len(salary_sources)):
            salary_sources[i].source_type = 'secondary_salary'
        
        return sources
    
    def _calculate_trend(self, values: np.ndarray) -> str:
        """Calculate trend direction using linear regression."""
        if len(values) < 3:
            return 'stable'
        
        x = np.arange(len(values))
        slope, _, r_value, _, _ = stats.linregress(x, values)
        
        # Check if trend is significant
        if abs(r_value) < 0.3:
            return 'stable'
        
        # Determine direction
        if slope > 0:
            return 'increasing'
        else:
            return 'decreasing'
    
    def _detect_seasonality(
        self,
        deposits: pd.DataFrame
    ) -> float:
        """Detect seasonal patterns in deposits (simplified)."""
        if len(deposits) < 12:
            return 0.0
        
        # Group by month and calculate variance
        deposits_copy = deposits.copy()
        deposits_copy['month'] = deposits_copy[self.timestamp_col].dt.month
        
        monthly_avg = deposits_copy.groupby('month')[self.amount_col].mean()
        
        if len(monthly_avg) < 3:
            return 0.0
        
        # Calculate coefficient of variation across months
        cv = monthly_avg.std() / monthly_avg.mean()
        
        # Higher CV suggests stronger seasonality
        seasonality_strength = min(cv, 1.0)
        
        return float(seasonality_strength)
