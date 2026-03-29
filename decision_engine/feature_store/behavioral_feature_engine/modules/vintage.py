"""
Vintage Analysis Module - Account Age and Cohort Features

Tracks performance by account age cohorts and lifecycle stages.
Critical for understanding how behavior varies by account maturity.

Version: BFE_v1.2
Author: Behavioral Feature Engineering Team
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, date
import logging

logger = logging.getLogger(__name__)


class VintageFeatureEngine:
    """
    Generate vintage (account age) and cohort analysis features.
    
    Key concepts:
    - Vintage: Account age / months on book
    - Cohort: Origination period (YYYY-MM)
    - Lifecycle: Stage of account maturity (NEW/SEASONED/MATURE/AGED)
    - Performance vs Cohort: Relative performance within origination cohort
    """
    
    VERSION = "BFE_v1.2"
    MODULE_NAME = "bfe.vintage"
    
    def __init__(self):
        """Initialize vintage feature engine."""
        self.logger = logging.getLogger(__name__)
    
    def compute_features(
        self,
        account_history: Dict[str, pd.DataFrame],
        account_id: str,
        as_of_date: date,
        schema_mappings: Dict[str, Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Compute vintage analysis features.
        
        Args:
            account_history: Dictionary with 'delinquency' and 'payment' DataFrames
            account_id: Account identifier
            as_of_date: Point-in-time date for feature computation
            schema_mappings: Field mappings for delinquency and payment domains
            
        Returns:
            Dictionary of vintage features
        """
        features = {}
        
        # Get required data
        delinquency_df = account_history.get("delinquency")
        payment_df = account_history.get("payment")
        
        if delinquency_df is None or len(delinquency_df) == 0:
            self.logger.warning(f"No delinquency data for {account_id}")
            return self._get_null_features()
        
        # Get mappings
        delinq_mappings = schema_mappings.get("delinquency", {})
        payment_mappings = schema_mappings.get("payment", {})
        
        # Filter to account and as_of_date
        delinq_df = delinquency_df[
            delinquency_df[delinq_mappings.get("account_id", "account_id")] == account_id
        ].copy()
        
        if len(delinq_df) == 0:
            return self._get_null_features()
        
        # Convert dates
        date_col = delinq_mappings.get("date", "date")
        delinq_df[date_col] = pd.to_datetime(delinq_df[date_col])
        
        # Filter to as_of_date
        delinq_df = delinq_df[delinq_df[date_col] <= pd.to_datetime(as_of_date)]
        
        if len(delinq_df) == 0:
            return self._get_null_features()
        
        # Sort by date
        delinq_df = delinq_df.sort_values(date_col)
        
        # Get account opening date
        account_open_date = delinq_df[date_col].min()
        
        # Compute vintage features
        features.update(self._compute_account_age_features(
            account_open_date, as_of_date
        ))
        
        features.update(self._compute_cohort_features(
            account_open_date, as_of_date
        ))
        
        features.update(self._compute_lifecycle_features(
            delinq_df, delinq_mappings, as_of_date, account_open_date
        ))
        
        # If payment data available, compute payment vintage features
        if payment_df is not None and len(payment_df) > 0:
            payment_df_filtered = payment_df[
                payment_df[payment_mappings.get("account_id", "account_id")] == account_id
            ].copy()
            
            if len(payment_df_filtered) > 0:
                features.update(self._compute_payment_vintage_features(
                    payment_df_filtered, payment_mappings, as_of_date, account_open_date
                ))
        
        return features
    
    def _compute_account_age_features(
        self,
        account_open_date: pd.Timestamp,
        as_of_date: date
    ) -> Dict[str, Any]:
        """Compute basic account age features."""
        features = {}
        
        as_of_dt = pd.to_datetime(as_of_date)
        
        # Days on book
        days_on_book = (as_of_dt - account_open_date).days
        features["vintage_days_on_book"] = days_on_book
        
        # Months on book
        months_on_book = days_on_book / 30.0
        features["vintage_months_on_book"] = round(months_on_book, 1)
        
        # Years on book
        years_on_book = months_on_book / 12.0
        features["vintage_years_on_book"] = round(years_on_book, 2)
        
        # Age bucket
        if months_on_book <= 6:
            age_bucket = "0-6M"
        elif months_on_book <= 12:
            age_bucket = "6-12M"
        elif months_on_book <= 24:
            age_bucket = "12-24M"
        elif months_on_book <= 36:
            age_bucket = "24-36M"
        else:
            age_bucket = "36M+"
        
        features["vintage_age_bucket"] = age_bucket
        
        return features
    
    def _compute_cohort_features(
        self,
        account_open_date: pd.Timestamp,
        as_of_date: date
    ) -> Dict[str, Any]:
        """Compute origination cohort features."""
        features = {}
        
        # Origination year-month (cohort identifier)
        origination_ym = account_open_date.strftime("%Y-%m")
        features["vintage_origination_cohort"] = origination_ym
        
        # Origination year
        features["vintage_origination_year"] = account_open_date.year
        
        # Origination quarter
        quarter = (account_open_date.month - 1) // 3 + 1
        features["vintage_origination_quarter"] = f"{account_open_date.year}-Q{quarter}"
        
        # Origination month (1-12)
        features["vintage_origination_month"] = account_open_date.month
        
        # Season of origination
        if account_open_date.month in [12, 1, 2]:
            season = "WINTER"
        elif account_open_date.month in [3, 4, 5]:
            season = "SPRING"
        elif account_open_date.month in [6, 7, 8]:
            season = "SUMMER"
        else:
            season = "FALL"
        
        features["vintage_origination_season"] = season
        
        return features
    
    def _compute_lifecycle_features(
        self,
        delinq_df: pd.DataFrame,
        mappings: Dict[str, str],
        as_of_date: date,
        account_open_date: pd.Timestamp
    ) -> Dict[str, Any]:
        """Compute lifecycle stage features."""
        features = {}
        
        months_on_book = features.get(
            "vintage_months_on_book",
            (pd.to_datetime(as_of_date) - account_open_date).days / 30.0
        )
        
        # Lifecycle stage
        if months_on_book <= 6:
            lifecycle_stage = "NEW"
        elif months_on_book <= 24:
            lifecycle_stage = "SEASONED"
        elif months_on_book <= 60:
            lifecycle_stage = "MATURE"
        else:
            lifecycle_stage = "AGED"
        
        features["vintage_lifecycle_stage"] = lifecycle_stage
        
        # Get DPD column
        dpd_col = mappings.get("dpd", "dpd")
        date_col = mappings.get("date", "date")
        
        if dpd_col in delinq_df.columns:
            dpd_series = delinq_df[dpd_col]
            
            # Months since first delinquency
            first_delinquent = delinq_df[dpd_series > 0]
            if len(first_delinquent) > 0:
                first_delinquent_date = first_delinquent[date_col].min()
                months_since_first_delinquency = (
                    (pd.to_datetime(as_of_date) - first_delinquent_date).days / 30.0
                )
                features["vintage_months_since_first_delinquency"] = round(
                    months_since_first_delinquency, 1
                )
                
                # Early delinquency flag (delinquent in first 6 months)
                features["vintage_early_delinquency_flag"] = (
                    months_since_first_delinquency >= months_on_book - 6
                )
            else:
                features["vintage_months_since_first_delinquency"] = np.nan
                features["vintage_early_delinquency_flag"] = False
            
            # Months since first cure (if ever delinquent and cured)
            # Find first instance of DPD > 30 followed by DPD = 0
            cures = []
            was_delinquent = False
            for idx, row in delinq_df.iterrows():
                dpd = row[dpd_col]
                if dpd > 30:
                    was_delinquent = True
                elif dpd == 0 and was_delinquent:
                    cures.append(row[date_col])
                    was_delinquent = False
            
            if len(cures) > 0:
                first_cure_date = min(cures)
                months_since_first_cure = (
                    (pd.to_datetime(as_of_date) - first_cure_date).days / 30.0
                )
                features["vintage_months_since_first_cure"] = round(
                    months_since_first_cure, 1
                )
            else:
                features["vintage_months_since_first_cure"] = np.nan
        
        return features
    
    def _compute_payment_vintage_features(
        self,
        payment_df: pd.DataFrame,
        mappings: Dict[str, str],
        as_of_date: date,
        account_open_date: pd.Timestamp
    ) -> Dict[str, Any]:
        """Compute payment behavior by vintage features."""
        features = {}
        
        date_col = mappings.get("date", "date")
        payment_df[date_col] = pd.to_datetime(payment_df[date_col])
        
        # Filter to as_of_date
        payment_df = payment_df[payment_df[date_col] <= pd.to_datetime(as_of_date)]
        
        if len(payment_df) == 0:
            return features
        
        payment_df = payment_df.sort_values(date_col)
        
        # Payment ratio by vintage period
        payment_amt_col = mappings.get("payment_amount", "payment_amount")
        amount_due_col = mappings.get("amount_due", "amount_due")
        
        if payment_amt_col in payment_df.columns and amount_due_col in payment_df.columns:
            payment_df["payment_ratio"] = (
                payment_df[payment_amt_col] / 
                payment_df[amount_due_col].replace(0, np.nan)
            )
            
            # First 6 months payment ratio
            first_6m_cutoff = account_open_date + pd.DateOffset(months=6)
            first_6m_payments = payment_df[payment_df[date_col] <= first_6m_cutoff]
            
            if len(first_6m_payments) > 0:
                features["vintage_payment_ratio_first_6M"] = first_6m_payments["payment_ratio"].mean()
            else:
                features["vintage_payment_ratio_first_6M"] = np.nan
            
            # Recent 6 months payment ratio
            recent_6m_cutoff = pd.to_datetime(as_of_date) - pd.DateOffset(months=6)
            recent_6m_payments = payment_df[payment_df[date_col] >= recent_6m_cutoff]
            
            if len(recent_6m_payments) > 0:
                features["vintage_payment_ratio_recent_6M"] = recent_6m_payments["payment_ratio"].mean()
            else:
                features["vintage_payment_ratio_recent_6M"] = np.nan
            
            # Payment performance evolution (recent vs early)
            if (features.get("vintage_payment_ratio_first_6M") is not np.nan and 
                features.get("vintage_payment_ratio_recent_6M") is not np.nan):
                features["vintage_payment_evolution"] = (
                    features["vintage_payment_ratio_recent_6M"] - 
                    features["vintage_payment_ratio_first_6M"]
                )
            else:
                features["vintage_payment_evolution"] = np.nan
        
        return features
    
    def _get_null_features(self) -> Dict[str, Any]:
        """Return null/default values for all vintage features."""
        return {
            "vintage_days_on_book": np.nan,
            "vintage_months_on_book": np.nan,
            "vintage_years_on_book": np.nan,
            "vintage_age_bucket": "UNKNOWN",
            "vintage_origination_cohort": "UNKNOWN",
            "vintage_origination_year": np.nan,
            "vintage_origination_quarter": "UNKNOWN",
            "vintage_origination_month": np.nan,
            "vintage_origination_season": "UNKNOWN",
            "vintage_lifecycle_stage": "UNKNOWN",
            "vintage_months_since_first_delinquency": np.nan,
            "vintage_early_delinquency_flag": np.nan,
            "vintage_months_since_first_cure": np.nan,
            "vintage_payment_ratio_first_6M": np.nan,
            "vintage_payment_ratio_recent_6M": np.nan,
            "vintage_payment_evolution": np.nan
        }
    
    @staticmethod
    def get_feature_metadata() -> Dict[str, Dict[str, Any]]:
        """
        Get metadata for all vintage features.
        
        Returns:
            Dictionary mapping feature names to metadata
        """
        return {
            "vintage_days_on_book": {
                "tier": 1,
                "definition": "Number of days since account opening",
                "feature_type": "continuous",
                "signal_direction": "higher = more seasoned",
                "use_cases": ["Vintage analysis", "Lifecycle modeling"]
            },
            "vintage_months_on_book": {
                "tier": 1,
                "definition": "Number of months since account opening",
                "feature_type": "continuous",
                "signal_direction": "higher = more seasoned",
                "use_cases": ["Vintage analysis", "Cohort comparison"]
            },
            "vintage_age_bucket": {
                "tier": 1,
                "definition": "Account age category (0-6M, 6-12M, etc.)",
                "feature_type": "categorical",
                "signal_direction": "varies by bucket",
                "use_cases": ["Segmentation", "Policy rules"]
            },
            "vintage_lifecycle_stage": {
                "tier": 1,
                "definition": "Account lifecycle stage (NEW/SEASONED/MATURE/AGED)",
                "feature_type": "categorical",
                "signal_direction": "varies by stage",
                "use_cases": ["Segmentation", "Collections strategy"]
            },
            "vintage_early_delinquency_flag": {
                "tier": 1,
                "definition": "Delinquent in first 6 months (high risk indicator)",
                "feature_type": "binary",
                "signal_direction": "True = higher risk",
                "use_cases": ["Early warning", "Underwriting quality"]
            },
            "vintage_payment_ratio_first_6M": {
                "tier": 2,
                "definition": "Average payment ratio in first 6 months",
                "feature_type": "continuous",
                "signal_direction": "higher = better early behavior",
                "use_cases": ["Performance tracking", "Cohort analysis"]
            },
            "vintage_payment_evolution": {
                "tier": 2,
                "definition": "Change in payment ratio from early to recent period",
                "feature_type": "continuous",
                "signal_direction": "positive = improving",
                "use_cases": ["Behavior trends", "Recovery prediction"]
            }
        }


# Example usage and testing
if __name__ == "__main__":
    print("\n" + "="*80)
    print("VINTAGE ANALYSIS MODULE - EXAMPLE")
    print("="*80)
    
    # Create sample data
    dates = pd.date_range('2022-01-01', periods=24, freq='MS')
    sample_delinq = pd.DataFrame({
        'account_id': ['ACC001'] * 24,
        'date': dates,
        'dpd': [0, 0, 15, 30, 45, 30, 15, 0, 0, 0, 15, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    })
    
    sample_payment = pd.DataFrame({
        'account_id': ['ACC001'] * 24,
        'date': dates,
        'payment_amount': [5000, 5000, 4500, 4000, 3500, 4000, 4500, 5000] * 3,
        'amount_due': [5000] * 24
    })
    
    # Initialize engine
    engine = VintageFeatureEngine()
    
    # Compute features
    features = engine.compute_features(
        account_history={'delinquency': sample_delinq, 'payment': sample_payment},
        account_id='ACC001',
        as_of_date=date(2024, 1, 31),
        schema_mappings={
            'delinquency': {'account_id': 'account_id', 'date': 'date', 'dpd': 'dpd'},
            'payment': {'account_id': 'account_id', 'date': 'date', 
                       'payment_amount': 'payment_amount', 'amount_due': 'amount_due'}
        }
    )
    
    print("\nVintage Features Computed:")
    print("-" * 80)
    for key, value in sorted(features.items()):
        print(f"  {key:45s} : {value}")
    
    print(f"\nTotal vintage features: {len(features)}")
