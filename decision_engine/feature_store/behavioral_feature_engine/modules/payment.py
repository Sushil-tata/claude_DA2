"""
BFE Module 2: Payment Features
===============================

Payment behavior features including:
- Payment amounts and ratios
- Payment timing and regularity
- Minimum payment vs full payment behavior
- Elasticity of payment ratio to balance changes
- Payment regimes (consistent payer, erratic, minimal, etc.)
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
    - Payment amounts (statistics over windows)
    - Payment ratios (amount_paid / amount_due)
    - Payment timing (days early/late)
    - Payment consistency and regularity
    - Elasticity (payment response to balance changes)
    - Payment regimes
    """

    VERSION = "BFE_v1.0"
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
        }
    }

    return metadata
