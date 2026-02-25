"""
BFE Module 2: Payment Features
===============================

Payment behavior features including:
- RFM Framework (Recency, Frequency, Monetary) - Industry standard
- Payment amounts and ratios
- Payment timing and regularity
- Minimum payment vs full payment behavior
- Elasticity of payment ratio to balance changes
- Payment regimes (consistent payer, erratic, minimal, etc.)

NEW in v1.1: Added comprehensive RFM (Recency, Frequency, Monetary) framework
with 30+ features including RFM scores, segments, and derived metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, date

from ..utils import WindowCalculator, TemporalValidator, NullHandler, get_schema_mapper


class PaymentFeatureEngine:
    """
    Computes payment behavior features with point-in-time safety.

    Features:
    - RFM Framework (Recency, Frequency, Monetary) - 30+ features
      * Recency: Days since last payment, recency score (1-5)
      * Frequency: Payment count over windows, frequency score (1-5)
      * Monetary: Average payment amounts, monetary score (1-5)
      * Composite: RFM score (0-15), customer segments, engagement
    - Payment amounts (statistics over windows)
    - Payment ratios (amount_paid / amount_due)
    - Payment timing (days early/late)
    - Payment consistency and regularity
    - Elasticity (payment response to balance changes)
    - Payment regimes
    """

    VERSION = "BFE_v1.1"  # Updated to v1.1 with RFM framework
    MODULE_NAME = "bfe.payment"

    def __init__(self):
        """Initialize payment feature engine"""
        self.schema_mapper = get_schema_mapper()
        self.field_mappings = {}
        self.window_calc = WindowCalculator()
        self.temporal_validator = TemporalValidator()
        self.null_handler = NullHandler()

    def _get_schema_mappings(self) -> Dict[str, str]:
        """Get or prompt for schema mappings"""
        if not self.field_mappings:
            required_fields = [
                {
                    "bfe_name": "account_id",
                    "description": "Unique account identifier",
                    "type": "string"
                },
                {
                    "bfe_name": "date",
                    "description": "Payment/statement date (monthly)",
                    "type": "date"
                },
                {
                    "bfe_name": "payment_amount",
                    "description": "Amount paid in the period",
                    "type": "float",
                    "example": "5000.00"
                },
                {
                    "bfe_name": "amount_due",
                    "description": "Total amount due for the period",
                    "type": "float",
                    "example": "8000.00"
                }
            ]

            optional_fields = [
                {
                    "bfe_name": "minimum_due",
                    "description": "Minimum payment required",
                    "type": "float"
                },
                {
                    "bfe_name": "statement_balance",
                    "description": "Statement balance for the period",
                    "type": "float"
                },
                {
                    "bfe_name": "due_date",
                    "description": "Payment due date",
                    "type": "date"
                },
                {
                    "bfe_name": "payment_date",
                    "description": "Actual payment date",
                    "type": "date"
                },
                {
                    "bfe_name": "balance",
                    "description": "Outstanding balance",
                    "type": "float"
                }
            ]

            self.field_mappings = self.schema_mapper.get_field_mapping(
                domain="payment",
                required_fields=required_fields,
                optional_fields=optional_fields
            )

        return self.field_mappings

    def compute_features(
        self,
        account_history: pd.DataFrame,
        account_id: str,
        as_of_date: date,
        windows: List[str] = None
    ) -> Dict[str, float]:
        """
        Compute all payment features for a single account.

        Args:
            account_history: Historical payment data for account
            account_id: Account identifier
            as_of_date: Reference date (point-in-time)
            windows: Time windows to compute (default: ["3M", "6M", "12M"])

        Returns:
            Dictionary of features
        """
        # Get schema mappings
        mappings = self._get_schema_mappings()

        # Validate point-in-time safety
        self.temporal_validator.validate_no_future_data(
            account_history,
            as_of_date,
            timestamp_col=mappings["date"]
        )

        # Prepare data
        df = account_history.copy()
        df = df.sort_values(mappings["date"])

        windows = windows or ["3M", "6M", "12M"]

        features = {}

        # ─────────────────────────────────────────────────────────────────
        # TIER 1: Core Payment Drivers
        # ─────────────────────────────────────────────────────────────────

        # Payment amount statistics
        features.update(self._compute_payment_amount_features(df, mappings, windows))

        # Payment ratio statistics (THE KEY METRIC)
        features.update(self._compute_payment_ratio_features(df, mappings, windows))

        # Period-over-period changes
        features.update(self._compute_pop_changes(df, mappings))

        # ─────────────────────────────────────────────────────────────────
        # TIER 1: Payment Behavior Flags
        # ─────────────────────────────────────────────────────────────────

        features.update(self._compute_payment_behavior_flags(df, mappings, windows))

        # ─────────────────────────────────────────────────────────────────
        # TIER 1: RFM Framework (Recency, Frequency, Monetary)
        # ─────────────────────────────────────────────────────────────────

        features.update(self._compute_rfm_features(df, mappings, as_of_date))

        # ─────────────────────────────────────────────────────────────────
        # TIER 2: Payment Timing & Regularity
        # ─────────────────────────────────────────────────────────────────

        features.update(self._compute_payment_timing_features(df, mappings, windows))

        # ─────────────────────────────────────────────────────────────────
        # TIER 3: Payment Elasticity (with lagged independent variable)
        # ─────────────────────────────────────────────────────────────────

        features.update(self._compute_elasticity_features(df, mappings))

        # ─────────────────────────────────────────────────────────────────
        # TIER 1: Payment Regime Classification (MANDATORY)
        # ─────────────────────────────────────────────────────────────────

        features.update(self._compute_payment_regime_features(df, mappings, features))

        # ─────────────────────────────────────────────────────────────────
        # Metadata
        # ─────────────────────────────────────────────────────────────────

        features["module_version"] = self.VERSION
        features["module_name"] = self.MODULE_NAME
        features["as_of_date"] = as_of_date
        features["months_of_history"] = len(df)

        return features

    def _compute_payment_amount_features(
        self,
        df: pd.DataFrame,
        mappings: Dict[str, str],
        windows: List[str]
    ) -> Dict[str, float]:
        """Compute payment amount statistics"""
        features = {}

        payment_series = df[mappings["payment_amount"]]

        # Statistics over windows
        statistics = ["mean", "max", "min", "std", "sum"]
        stats_result = self.window_calc.compute_statistics(
            payment_series,
            windows=windows,
            statistics=statistics
        )

        # Rename with payment_amount prefix
        for key, value in stats_result.items():
            features[f"payment_amount_{key}"] = value

        return features

    def _compute_payment_ratio_features(
        self,
        df: pd.DataFrame,
        mappings: Dict[str, str],
        windows: List[str]
    ) -> Dict[str, float]:
        """
        Compute payment ratio (amount_paid / amount_due) features.

        This is the MOST IMPORTANT payment feature.
        """
        features = {}

        # Compute payment ratio
        payment_ratio_series = self.null_handler.safe_divide(
            df[mappings["payment_amount"]],
            df[mappings["amount_due"]],
            default=0.0
        )

        # Cap at 200% (overpayments beyond 2x are outliers)
        payment_ratio_series = payment_ratio_series.clip(upper=2.0)

        # Statistics over windows
        statistics = ["mean", "max", "min", "std"]
        stats_result = self.window_calc.compute_statistics(
            payment_ratio_series,
            windows=windows,
            statistics=statistics
        )

        # Rename with payment_ratio prefix
        for key, value in stats_result.items():
            features[f"payment_ratio_{key}"] = value

        # Slope and momentum
        slope_result = self.window_calc.compute_slope(payment_ratio_series, windows=windows)
        for key, value in slope_result.items():
            features[f"payment_ratio_{key}"] = value

        momentum_result = self.window_calc.compute_momentum(payment_ratio_series, windows=windows)
        for key, value in momentum_result.items():
            features[f"payment_ratio_{key}"] = value

        return features

    def _compute_pop_changes(
        self,
        df: pd.DataFrame,
        mappings: Dict[str, str]
    ) -> Dict[str, float]:
        """Compute period-over-period changes in payment ratio"""
        features = {}

        # Compute payment ratio
        payment_ratio_series = self.null_handler.safe_divide(
            df[mappings["payment_amount"]],
            df[mappings["amount_due"]],
            default=0.0
        )

        # MoM, QoQ changes
        pop_changes = self.window_calc.compute_period_over_period_change(
            payment_ratio_series,
            periods=["MoM", "QoQ"]
        )

        for key, value in pop_changes.items():
            features[f"payment_ratio_{key}"] = value

        return features

    def _compute_payment_behavior_flags(
        self,
        df: pd.DataFrame,
        mappings: Dict[str, str],
        windows: List[str]
    ) -> Dict[str, float]:
        """Compute binary flags for payment behaviors"""
        features = {}

        payment_amount = df[mappings["payment_amount"]]
        amount_due = df[mappings["amount_due"]]

        # Check if optional fields are available
        has_minimum_due = mappings.get("minimum_due") and mappings["minimum_due"] in df.columns
        has_statement_balance = mappings.get("statement_balance") and mappings["statement_balance"] in df.columns

        for window in windows:
            window_months = self.window_calc.WINDOWS.get(window, int(window.replace("M", "")))
            window_payment = payment_amount.tail(window_months)
            window_due = amount_due.tail(window_months)

            if len(window_payment) == 0:
                continue

            # Missed payment count (payment = 0)
            missed_count = (window_payment == 0).sum()
            features[f"missed_payment_{window}_count"] = missed_count

            # Minimum payment count (if available)
            if has_minimum_due:
                minimum_due = df[mappings["minimum_due"]].tail(window_months)
                min_pay_count = ((window_payment > 0) & (window_payment <= minimum_due * 1.05)).sum()  # 5% tolerance
                features[f"min_payment_flag_{window}_count"] = min_pay_count
            else:
                features[f"min_payment_flag_{window}_count"] = np.nan

            # Full payment count (if statement balance available)
            if has_statement_balance:
                statement_balance = df[mappings["statement_balance"]].tail(window_months)
                full_pay_count = (window_payment >= statement_balance * 0.95).sum()  # 95% threshold
                features[f"full_payment_flag_{window}_count"] = full_pay_count
            else:
                # Alternative: payment >= amount_due
                full_pay_count = (window_payment >= window_due * 0.95).sum()
                features[f"full_payment_flag_{window}_count"] = full_pay_count

            # Overpayment count
            overpay_count = (window_payment > window_due).sum()
            features[f"overpayment_flag_{window}_count"] = overpay_count

            # Payment shortfall (when underpaying)
            underpayment_mask = window_payment < window_due
            if underpayment_mask.any():
                shortfalls = window_due[underpayment_mask] - window_payment[underpayment_mask]
                features[f"payment_shortfall_avg_{window}"] = shortfalls.mean()
            else:
                features[f"payment_shortfall_avg_{window}"] = 0.0

        # Consecutive missed payments (current streak)
        consecutive_missed = self.window_calc.compute_streak(
            payment_amount,
            lambda x: x == 0
        )
        features["consecutive_missed_current"] = consecutive_missed

        # Max consecutive missed (ever)
        max_consecutive_missed = self.window_calc.compute_max_streak(
            payment_amount,
            lambda x: x == 0
        )
        features["max_consecutive_missed_ever"] = max_consecutive_missed

        return features

    def _compute_rfm_features(
        self,
        df: pd.DataFrame,
        mappings: Dict[str, str],
        as_of_date: date
    ) -> Dict[str, float]:
        """
        Compute RFM (Recency, Frequency, Monetary) features.

        RFM is a fundamental credit scoring framework used to segment customers
        based on payment behavior:
        - Recency: How recently did the customer make a payment?
        - Frequency: How often does the customer make payments?
        - Monetary: How much does the customer typically pay?

        RFM scoring uses quintiles (1-5) for each dimension, where:
        - Recency: 5 = most recent (best), 1 = least recent (worst)
        - Frequency: 5 = most frequent (best), 1 = least frequent (worst)
        - Monetary: 5 = highest amount (best), 1 = lowest amount (worst)

        RFM Composite Score ranges from 3 (worst: 1+1+1) to 15 (best: 5+5+5)

        References:
        - RFMS Method for Credit Scoring: https://www3.stat.sinica.edu.tw/statistica/oldpdf/A28n535.pdf
        - Industry standard in behavioral credit scoring
        """
        features = {}

        payment_dates = pd.to_datetime(df[mappings["date"]])
        payment_amounts = df[mappings["payment_amount"]]

        # Filter to non-zero payments only for RFM analysis
        payment_mask = payment_amounts > 0
        payment_dates_nonzero = payment_dates[payment_mask]
        payment_amounts_nonzero = payment_amounts[payment_mask]

        # ─────────────────────────────────────────────────────────────────
        # R - RECENCY
        # ─────────────────────────────────────────────────────────────────

        if len(payment_dates_nonzero) > 0:
            # Days since last payment
            last_payment_date = payment_dates_nonzero.max()
            rfm_recency_days = (pd.to_datetime(as_of_date) - last_payment_date).days

            features["rfm_recency_days"] = rfm_recency_days

            # Recency score (1-5, lower days = higher score)
            # Quintile-based scoring
            if rfm_recency_days <= 7:
                rfm_recency_score = 5  # Excellent: paid within last week
            elif rfm_recency_days <= 30:
                rfm_recency_score = 4  # Good: paid within last month
            elif rfm_recency_days <= 60:
                rfm_recency_score = 3  # Fair: paid within last 2 months
            elif rfm_recency_days <= 90:
                rfm_recency_score = 2  # Poor: paid within last 3 months
            else:
                rfm_recency_score = 1  # Very poor: no payment in 3+ months

            features["rfm_recency_score"] = rfm_recency_score
        else:
            # No payments ever
            features["rfm_recency_days"] = np.nan
            features["rfm_recency_score"] = 0  # Worst possible

        # ─────────────────────────────────────────────────────────────────
        # F - FREQUENCY
        # ─────────────────────────────────────────────────────────────────

        # Frequency over multiple windows
        for window in ["3M", "6M", "12M"]:
            window_months = self.window_calc.WINDOWS.get(window, int(window.replace("M", "")))
            window_payments = payment_amounts_nonzero.tail(window_months)

            frequency = len(window_payments)
            features[f"rfm_frequency_{window}"] = frequency

        # Primary frequency metric (6M)
        rfm_frequency_6m = features.get("rfm_frequency_6M", 0)
        features["rfm_frequency"] = rfm_frequency_6m

        # Frequency score (1-5, based on 6M frequency)
        if rfm_frequency_6m >= 6:
            rfm_frequency_score = 5  # Excellent: pays every month
        elif rfm_frequency_6m >= 4:
            rfm_frequency_score = 4  # Good: pays most months
        elif rfm_frequency_6m >= 2:
            rfm_frequency_score = 3  # Fair: pays occasionally
        elif rfm_frequency_6m >= 1:
            rfm_frequency_score = 2  # Poor: pays rarely
        else:
            rfm_frequency_score = 1  # Very poor: no payments

        features["rfm_frequency_score"] = rfm_frequency_score

        # Frequency rate (payments per month)
        if len(df) > 0:
            features["rfm_frequency_rate"] = rfm_frequency_6m / 6.0
        else:
            features["rfm_frequency_rate"] = 0.0

        # ─────────────────────────────────────────────────────────────────
        # M - MONETARY
        # ─────────────────────────────────────────────────────────────────

        if len(payment_amounts_nonzero) > 0:
            # Average payment amount (6M)
            rfm_monetary_avg_6m = payment_amounts_nonzero.tail(6).mean()
            features["rfm_monetary_avg_6M"] = rfm_monetary_avg_6m

            # Total payment amount (6M)
            rfm_monetary_sum_6m = payment_amounts_nonzero.tail(6).sum()
            features["rfm_monetary_sum_6M"] = rfm_monetary_sum_6m

            # Monetary value (12M)
            rfm_monetary_avg_12m = payment_amounts_nonzero.tail(12).mean()
            features["rfm_monetary_avg_12M"] = rfm_monetary_avg_12m

            # Primary monetary metric
            features["rfm_monetary"] = rfm_monetary_avg_6m

            # Monetary score (1-5, based on average payment amount)
            # This is relative - in production, you'd use quintiles across portfolio
            # For now, we use simple thresholds (adjust based on your data)
            if rfm_monetary_avg_6m >= 10000:
                rfm_monetary_score = 5  # Very high payment amounts
            elif rfm_monetary_avg_6m >= 5000:
                rfm_monetary_score = 4  # High payment amounts
            elif rfm_monetary_avg_6m >= 2000:
                rfm_monetary_score = 3  # Medium payment amounts
            elif rfm_monetary_avg_6m >= 500:
                rfm_monetary_score = 2  # Low payment amounts
            else:
                rfm_monetary_score = 1  # Very low payment amounts

            features["rfm_monetary_score"] = rfm_monetary_score

            # Monetary consistency (CV of payment amounts)
            payment_std = payment_amounts_nonzero.tail(6).std()
            payment_cv = self.null_handler.safe_divide(payment_std, rfm_monetary_avg_6m, default=np.nan)
            features["rfm_monetary_consistency"] = 1.0 / (1.0 + payment_cv) if not np.isnan(payment_cv) else np.nan

        else:
            features["rfm_monetary_avg_6M"] = np.nan
            features["rfm_monetary_sum_6M"] = 0.0
            features["rfm_monetary_avg_12M"] = np.nan
            features["rfm_monetary"] = 0.0
            features["rfm_monetary_score"] = 0
            features["rfm_monetary_consistency"] = np.nan

        # ─────────────────────────────────────────────────────────────────
        # RFM COMPOSITE SCORE & SEGMENT
        # ─────────────────────────────────────────────────────────────────

        # Composite score (sum of R, F, M scores)
        # Range: 0-15 (best is 15, worst is 0)
        rfm_composite_score = (
            features.get("rfm_recency_score", 0) +
            features.get("rfm_frequency_score", 0) +
            features.get("rfm_monetary_score", 0)
        )
        features["rfm_composite_score"] = rfm_composite_score

        # RFM Segment classification
        if rfm_composite_score >= 12:
            rfm_segment = "HIGH_VALUE"          # Best customers (Champions/Loyal)
            rfm_risk_level = "LOW"
        elif rfm_composite_score >= 9:
            rfm_segment = "MEDIUM_VALUE"        # Good customers (Potential Loyalists)
            rfm_risk_level = "MEDIUM"
        elif rfm_composite_score >= 6:
            rfm_segment = "LOW_VALUE"           # At-risk customers (Need Attention)
            rfm_risk_level = "HIGH"
        elif rfm_composite_score >= 3:
            rfm_segment = "VERY_LOW_VALUE"      # Hibernating customers
            rfm_risk_level = "VERY_HIGH"
        else:
            rfm_segment = "LOST"                # Lost customers
            rfm_risk_level = "CRITICAL"

        features["rfm_segment"] = rfm_segment
        features["rfm_risk_level"] = rfm_risk_level

        # Detailed RFM classification (classic RFM segments)
        # Based on combination of R, F, M scores
        r_score = features.get("rfm_recency_score", 0)
        f_score = features.get("rfm_frequency_score", 0)
        m_score = features.get("rfm_monetary_score", 0)

        if r_score >= 4 and f_score >= 4 and m_score >= 4:
            rfm_detailed_segment = "CHAMPIONS"          # Best customers
        elif r_score >= 3 and f_score >= 3 and m_score >= 3:
            rfm_detailed_segment = "LOYAL_CUSTOMERS"    # Regular payers
        elif r_score >= 4 and f_score <= 2:
            rfm_detailed_segment = "NEW_CUSTOMERS"      # Recent but infrequent
        elif r_score <= 2 and f_score >= 3:
            rfm_detailed_segment = "CANT_LOSE_THEM"     # Was loyal, now at risk
        elif r_score <= 2 and f_score <= 2 and m_score >= 3:
            rfm_detailed_segment = "HIBERNATING_HIGH_VALUE"  # Inactive but paid well before
        elif r_score <= 2 and f_score <= 2:
            rfm_detailed_segment = "LOST"               # Completely disengaged
        else:
            rfm_detailed_segment = "NEED_ATTENTION"     # Middle tier

        features["rfm_detailed_segment"] = rfm_detailed_segment

        # Binary flags for each segment
        for segment in ["HIGH_VALUE", "MEDIUM_VALUE", "LOW_VALUE", "VERY_LOW_VALUE", "LOST"]:
            features[f"rfm_segment_{segment.lower()}_flag"] = int(rfm_segment == segment)

        # ─────────────────────────────────────────────────────────────────
        # RFM-BASED DERIVED FEATURES
        # ─────────────────────────────────────────────────────────────────

        # Payment velocity (monetary / recency)
        if features.get("rfm_recency_days", np.nan) > 0 and not np.isnan(features.get("rfm_monetary_avg_6M", np.nan)):
            payment_velocity = features["rfm_monetary_avg_6M"] / features["rfm_recency_days"]
            features["rfm_payment_velocity"] = payment_velocity
        else:
            features["rfm_payment_velocity"] = 0.0

        # RFM balance (R:F:M ratio)
        # Ideally, all three should be high and balanced
        rfm_scores = [r_score, f_score, m_score]
        rfm_scores = [s for s in rfm_scores if s > 0]  # Filter out zeros

        if len(rfm_scores) >= 2:
            rfm_cv = np.std(rfm_scores) / np.mean(rfm_scores)
            features["rfm_balance_score"] = 1.0 / (1.0 + rfm_cv)  # Higher = more balanced
        else:
            features["rfm_balance_score"] = 0.0

        # RFM engagement score (frequency weighted by recency)
        # High engagement = frequent AND recent payments
        rfm_engagement = f_score * r_score / 25.0  # Normalize to 0-1
        features["rfm_engagement_score"] = rfm_engagement

        return features

    def _compute_payment_timing_features(
        self,
        df: pd.DataFrame,
        mappings: Dict[str, str],
        windows: List[str]
    ) -> Dict[str, float]:
        """Compute payment timing features (requires due_date and payment_date)"""
        features = {}

        # Check if timing data is available
        has_timing = (
            mappings.get("due_date") and mappings["due_date"] in df.columns and
            mappings.get("payment_date") and mappings["payment_date"] in df.columns
        )

        if not has_timing:
            # Return NaN for all timing features
            for window in windows:
                features[f"payment_timing_avg_{window}"] = np.nan
                features[f"payment_timing_std_{window}"] = np.nan
                features[f"early_payment_flag_{window}_count"] = np.nan

            features["payment_regularity_score_6M"] = np.nan
            return features

        # Compute days early/late
        due_dates = pd.to_datetime(df[mappings["due_date"]])
        payment_dates = pd.to_datetime(df[mappings["payment_date"]])

        # Days late = payment_date - due_date (positive = late, negative = early)
        days_late_series = (payment_dates - due_dates).dt.days

        for window in windows:
            window_months = self.window_calc.WINDOWS.get(window, int(window.replace("M", "")))
            window_timing = days_late_series.tail(window_months)

            if len(window_timing) == 0:
                continue

            features[f"payment_timing_avg_{window}"] = window_timing.mean()
            features[f"payment_timing_std_{window}"] = window_timing.std()

            # Early payment count (negative days late)
            early_count = (window_timing < 0).sum()
            features[f"early_payment_flag_{window}_count"] = early_count

        # Payment regularity score (inverse of std dev)
        timing_6m = days_late_series.tail(6)
        if len(timing_6m) >= 3:
            timing_std = timing_6m.std()
            regularity_score = 100 / (1 + timing_std)  # Score 0-100, higher = more regular
            features["payment_regularity_score_6M"] = regularity_score
        else:
            features["payment_regularity_score_6M"] = np.nan

        return features

    def _compute_elasticity_features(
        self,
        df: pd.DataFrame,
        mappings: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Compute payment elasticity (with lagged independent variable).

        Elasticity = % change in payment_ratio / % change in balance
        Uses lagged balance to prevent simultaneity bias.
        """
        features = {}

        # Check if balance data is available
        has_balance = mappings.get("balance") and mappings["balance"] in df.columns

        if not has_balance or len(df) < 4:
            features["payment_ratio_elasticity"] = np.nan
            return features

        # Compute payment ratio
        payment_ratio = self.null_handler.safe_divide(
            df[mappings["payment_amount"]],
            df[mappings["amount_due"]],
            default=0.0
        )

        balance = df[mappings["balance"]]

        # Compute elasticity with lag=1 to prevent simultaneity
        elasticity = self.window_calc.compute_elasticity(
            dependent_series=payment_ratio,
            independent_series=balance,
            lag=1
        )

        features["payment_ratio_elasticity"] = elasticity

        return features

    def _compute_payment_regime_features(
        self,
        df: pd.DataFrame,
        mappings: Dict[str, str],
        existing_features: Dict[str, float]
    ) -> Dict[str, float]:
        """
        MANDATORY: Classify accounts into payment behavior regimes.

        Regimes based on:
        - Level: Average payment ratio
        - Consistency: Std dev of payment ratio
        - Trend: Slope of payment ratio

        Regimes:
        - CONSISTENT_FULL_PAYER: Pays full amount consistently
        - CONSISTENT_PARTIAL_PAYER: Pays consistently but partial
        - ERRATIC_PAYER: High volatility in payment behavior
        - MINIMAL_PAYER: Pays minimum or near-minimum
        - NON_PAYER: Frequently misses payments
        - IMPROVING: Payment ratio increasing over time
        - DETERIORATING: Payment ratio decreasing over time
        """
        features = {}

        # Use 6M window for regime classification
        payment_ratio_6m_mean = existing_features.get("payment_ratio_6M_mean", np.nan)
        payment_ratio_6m_std = existing_features.get("payment_ratio_6M_std", np.nan)
        payment_ratio_6m_slope = existing_features.get("payment_ratio_6M_slope", np.nan)
        missed_payment_6m_count = existing_features.get("missed_payment_6M_count", 0)
        full_payment_6m_count = existing_features.get("full_payment_6M_count", 0)

        # Regime classification logic
        regime = "UNKNOWN"
        regime_confidence = 0.0

        if pd.isna(payment_ratio_6m_mean):
            regime = "INSUFFICIENT_DATA"
            regime_confidence = 0.0
        else:
            # Consistent Full Payer: High mean, low std, mostly full payments
            if payment_ratio_6m_mean >= 0.90 and payment_ratio_6m_std < 0.15 and full_payment_6m_count >= 4:
                regime = "CONSISTENT_FULL_PAYER"
                regime_confidence = 0.95

            # Non-Payer: Many missed payments
            elif missed_payment_6m_count >= 3:
                regime = "NON_PAYER"
                regime_confidence = 0.90

            # Improving: Low but increasing
            elif payment_ratio_6m_mean < 0.5 and payment_ratio_6m_slope > 0.05:
                regime = "IMPROVING"
                regime_confidence = 0.80

            # Deteriorating: Decreasing payment ratio
            elif payment_ratio_6m_slope < -0.05:
                regime = "DETERIORATING"
                regime_confidence = 0.85

            # Erratic: High volatility
            elif payment_ratio_6m_std > 0.30:
                regime = "ERRATIC_PAYER"
                regime_confidence = 0.75

            # Consistent Partial: Medium mean, low std
            elif 0.40 <= payment_ratio_6m_mean < 0.90 and payment_ratio_6m_std < 0.20:
                regime = "CONSISTENT_PARTIAL_PAYER"
                regime_confidence = 0.80

            # Minimal Payer: Low mean, low std
            elif payment_ratio_6m_mean < 0.40 and payment_ratio_6m_std < 0.15:
                regime = "MINIMAL_PAYER"
                regime_confidence = 0.75

            # Default
            else:
                regime = "INCONSISTENT"
                regime_confidence = 0.60

        features["payment_regime"] = regime
        features["payment_regime_confidence"] = regime_confidence

        # Regime persistence (simplified)
        features["payment_regime_persistence_months"] = 1  # Placeholder

        # Binary flags for each regime
        for regime_name in ["CONSISTENT_FULL_PAYER", "CONSISTENT_PARTIAL_PAYER", "ERRATIC_PAYER",
                           "MINIMAL_PAYER", "NON_PAYER", "IMPROVING", "DETERIORATING"]:
            features[f"payment_regime_{regime_name.lower()}_flag"] = int(regime == regime_name)

        return features


def get_feature_metadata() -> Dict[str, Dict]:
    """Return metadata for all payment features"""
    metadata = {
        # Payment Ratio Features
        "payment_ratio_6M_mean": {
            "module": "bfe.payment",
            "tier": 1,
            "definition": "Average payment ratio (paid/due) over 6 months",
            "feature_type": "continuous",
            "signal_direction": "higher = less risky",
            "min_history_months": 6,
            "use_cases": ["PD", "EWS", "Collections", "LGD"],
            "product_applicability": ["all"]
        },
        "payment_regime": {
            "module": "bfe.payment",
            "tier": 1,
            "definition": "Payment behavior regime classification",
            "feature_type": "categorical",
            "signal_direction": "varies by regime",
            "min_history_months": 6,
            "use_cases": ["PD", "Collections", "Portfolio_Segmentation"],
            "product_applicability": ["all"]
        },

        # RFM Framework Features
        "rfm_recency_days": {
            "module": "bfe.payment",
            "tier": 1,
            "definition": "Days since last payment (Recency dimension of RFM)",
            "feature_type": "continuous",
            "signal_direction": "lower = less risky (more recent payment)",
            "min_history_months": 1,
            "use_cases": ["PD", "Collections", "Portfolio_Segmentation"],
            "product_applicability": ["all"],
            "framework": "RFM"
        },
        "rfm_recency_score": {
            "module": "bfe.payment",
            "tier": 1,
            "definition": "Recency score (1-5 quintile, 5=best, paid recently)",
            "feature_type": "ordinal",
            "signal_direction": "higher = less risky",
            "min_history_months": 1,
            "use_cases": ["PD", "Collections", "Portfolio_Segmentation"],
            "product_applicability": ["all"],
            "framework": "RFM"
        },
        "rfm_frequency_6M": {
            "module": "bfe.payment",
            "tier": 1,
            "definition": "Number of payments made in trailing 6 months (Frequency dimension of RFM)",
            "feature_type": "continuous",
            "signal_direction": "higher = less risky (more frequent payments)",
            "min_history_months": 6,
            "use_cases": ["PD", "Collections", "LGD"],
            "product_applicability": ["all"],
            "framework": "RFM"
        },
        "rfm_frequency_score": {
            "module": "bfe.payment",
            "tier": 1,
            "definition": "Frequency score (1-5 quintile, 5=best, pays frequently)",
            "feature_type": "ordinal",
            "signal_direction": "higher = less risky",
            "min_history_months": 6,
            "use_cases": ["PD", "Collections"],
            "product_applicability": ["all"],
            "framework": "RFM"
        },
        "rfm_monetary_avg_6M": {
            "module": "bfe.payment",
            "tier": 1,
            "definition": "Average payment amount over trailing 6 months (Monetary dimension of RFM)",
            "feature_type": "continuous",
            "signal_direction": "higher = less risky (pays more)",
            "min_history_months": 6,
            "use_cases": ["PD", "LGD", "Collections"],
            "product_applicability": ["all"],
            "framework": "RFM"
        },
        "rfm_monetary_score": {
            "module": "bfe.payment",
            "tier": 1,
            "definition": "Monetary score (1-5 quintile, 5=best, high payment amounts)",
            "feature_type": "ordinal",
            "signal_direction": "higher = less risky",
            "min_history_months": 6,
            "use_cases": ["PD", "LGD"],
            "product_applicability": ["all"],
            "framework": "RFM"
        },
        "rfm_composite_score": {
            "module": "bfe.payment",
            "tier": 1,
            "definition": "RFM composite score (sum of R+F+M scores, range 0-15, 15=best customer)",
            "feature_type": "continuous",
            "signal_direction": "higher = less risky",
            "min_history_months": 6,
            "use_cases": ["PD", "Collections", "Portfolio_Segmentation", "Customer_Value"],
            "product_applicability": ["all"],
            "framework": "RFM",
            "importance": "HIGH - Industry standard composite metric"
        },
        "rfm_segment": {
            "module": "bfe.payment",
            "tier": 1,
            "definition": "RFM customer segment (HIGH_VALUE/MEDIUM_VALUE/LOW_VALUE/VERY_LOW_VALUE/LOST)",
            "feature_type": "categorical",
            "signal_direction": "HIGH_VALUE = lowest risk",
            "min_history_months": 6,
            "use_cases": ["PD", "Collections", "Portfolio_Segmentation", "Customer_Value"],
            "product_applicability": ["all"],
            "framework": "RFM",
            "importance": "HIGH - Primary customer segmentation"
        },
        "rfm_detailed_segment": {
            "module": "bfe.payment",
            "tier": 1,
            "definition": "RFM detailed segment (CHAMPIONS/LOYAL_CUSTOMERS/CANT_LOSE_THEM/etc.)",
            "feature_type": "categorical",
            "signal_direction": "varies by segment",
            "min_history_months": 6,
            "use_cases": ["Collections", "Portfolio_Segmentation", "Retention_Strategy"],
            "product_applicability": ["all"],
            "framework": "RFM"
        },
        "rfm_engagement_score": {
            "module": "bfe.payment",
            "tier": 2,
            "definition": "RFM engagement score (frequency weighted by recency, 0-1 scale)",
            "feature_type": "continuous",
            "signal_direction": "higher = less risky (engaged customer)",
            "min_history_months": 6,
            "use_cases": ["PD", "Collections"],
            "product_applicability": ["all"],
            "framework": "RFM"
        }
    }

    return metadata
