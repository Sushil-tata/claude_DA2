"""
BFE Module 1: Delinquency Features
===================================

Bucket states:
- Regular (R): 0-30 DPD
- Sub-Standard/SM: 31-90 DPD
- NPL (Non-Performing Loan): 91-180 DPD
- Charge-Off (CO): 181+ DPD

Features include statistics, transitions, trajectories, and behavioral regimes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, date

from ..utils import WindowCalculator, TemporalValidator, NullHandler, get_schema_mapper


class DelinquencyFeatureEngine:
    """
    Computes delinquency-related behavioral features with point-in-time safety.

    Features:
    - Level: Current DPD, bucket state
    - Aggregates: Max, mean, std, CV over windows
    - Trajectory: Slope, momentum, velocity
    - Transitions: Bucket changes, cure/redefault flags
    - Regimes: Stable, deteriorating, volatile, recovering
    """

    VERSION = "BFE_v1.0"
    MODULE_NAME = "bfe.delinquency"

    # Bucket definitions
    BUCKETS = {
        "REGULAR": (0, 30),
        "SM": (31, 90),
        "NPL": (91, 180),
        "CHARGE_OFF": (181, 99999)
    }

    BUCKET_ORDINAL = {
        "REGULAR": 0,
        "SM": 1,
        "NPL": 2,
        "CHARGE_OFF": 3
    }

    def __init__(self):
        """Initialize delinquency feature engine"""
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
                    "type": "string",
                    "example": "ACC123456"
                },
                {
                    "bfe_name": "date",
                    "description": "Observation date (monthly)",
                    "type": "date",
                    "example": "2024-01-31"
                },
                {
                    "bfe_name": "dpd",
                    "description": "Days Past Due (0 = current)",
                    "type": "int",
                    "example": "45"
                }
            ]

            optional_fields = [
                {
                    "bfe_name": "account_open_date",
                    "description": "Date account was opened",
                    "type": "date"
                }
            ]

            self.field_mappings = self.schema_mapper.get_field_mapping(
                domain="delinquency",
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
        Compute all delinquency features for a single account.

        Args:
            account_history: Historical DPD data for account
            account_id: Account identifier
            as_of_date: Reference date (point-in-time)
            windows: Time windows to compute (default: ["3M", "6M", "12M", "24M"])

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

        # Extract DPD series
        dpd_series = df[mappings["dpd"]]

        windows = windows or ["3M", "6M", "12M", "24M"]

        features = {}

        # ─────────────────────────────────────────────────────────────────
        # TIER 1: Core Risk Drivers
        # ─────────────────────────────────────────────────────────────────

        # Spot features
        features.update(self._compute_spot_features(dpd_series))

        # Statistics over windows
        features.update(self._compute_statistics(dpd_series, windows))

        # Trajectory features (slopes, momentum)
        features.update(self._compute_trajectory_features(dpd_series, windows))

        # ─────────────────────────────────────────────────────────────────
        # TIER 1: Bucket-based features
        # ─────────────────────────────────────────────────────────────────

        features.update(self._compute_bucket_features(dpd_series, windows))

        # ─────────────────────────────────────────────────────────────────
        # TIER 1: Transition features
        # ─────────────────────────────────────────────────────────────────

        features.update(self._compute_transition_features(dpd_series, windows))

        # ─────────────────────────────────────────────────────────────────
        # TIER 2: Volatility & Stability
        # ─────────────────────────────────────────────────────────────────

        features.update(self._compute_volatility_features(dpd_series, windows))

        # ─────────────────────────────────────────────────────────────────
        # TIER 1: Regime Classification (MANDATORY)
        # ─────────────────────────────────────────────────────────────────

        features.update(self._compute_regime_features(dpd_series, windows, features))

        # ─────────────────────────────────────────────────────────────────
        # Metadata
        # ─────────────────────────────────────────────────────────────────

        features["module_version"] = self.VERSION
        features["module_name"] = self.MODULE_NAME
        features["as_of_date"] = as_of_date
        features["months_of_history"] = len(dpd_series)

        # Handle insufficient history
        if len(dpd_series) < 3:
            features["new_account_flag"] = True
            features["insufficient_history_flag"] = True

        return features

    def _compute_spot_features(self, dpd_series: pd.Series) -> Dict[str, float]:
        """Compute current (spot) delinquency features"""
        features = {}

        if len(dpd_series) == 0:
            features["dpd_current"] = np.nan
            features["bucket_current"] = None
            features["bucket_current_ordinal"] = np.nan
            return features

        # Current DPD
        dpd_current = dpd_series.iloc[-1]
        features["dpd_current"] = dpd_current

        # Current bucket
        bucket_current = self._dpd_to_bucket(dpd_current)
        features["bucket_current"] = bucket_current
        features["bucket_current_ordinal"] = self.BUCKET_ORDINAL.get(bucket_current, np.nan)

        # Months in current bucket
        features["months_in_current_bucket"] = self._compute_tenure_in_bucket(dpd_series)

        # Ever reached NPL or CO
        features["ever_npl_flag"] = int((dpd_series >= 91).any())
        features["ever_co_flag"] = int((dpd_series >= 181).any())

        # Months since last delinquency (DPD > 0)
        if (dpd_series > 0).any():
            last_delinq_idx = (dpd_series > 0)[::-1].idxmax()
            features["months_since_last_delinquency"] = len(dpd_series) - dpd_series.index.get_loc(last_delinq_idx) - 1
        else:
            features["months_since_last_delinquency"] = len(dpd_series)

        return features

    def _compute_statistics(self, dpd_series: pd.Series, windows: List[str]) -> Dict[str, float]:
        """Compute statistical aggregates over windows"""
        statistics = ["mean", "max", "min", "std", "cv"]

        return self.window_calc.compute_statistics(
            dpd_series,
            windows=windows,
            statistics=statistics
        )

    def _compute_trajectory_features(self, dpd_series: pd.Series, windows: List[str]) -> Dict[str, float]:
        """Compute trajectory features (slopes, momentum)"""
        features = {}

        # Slopes
        slopes = self.window_calc.compute_slope(dpd_series, windows=windows)
        features.update(slopes)

        # Momentum
        momentum = self.window_calc.compute_momentum(dpd_series, windows=windows)
        features.update(momentum)

        return features

    def _compute_bucket_features(self, dpd_series: pd.Series, windows: List[str]) -> Dict[str, float]:
        """Compute bucket-based features"""
        features = {}

        # Create bucket series
        bucket_series = dpd_series.apply(self._dpd_to_bucket)
        bucket_ordinal_series = bucket_series.map(self.BUCKET_ORDINAL)

        for window in windows:
            window_months = self.window_calc.WINDOWS.get(window, int(window.replace("M", "")))
            window_data = bucket_series.tail(window_months)
            window_ordinal = bucket_ordinal_series.tail(window_months)

            if len(window_data) == 0:
                continue

            # Mode (most frequent bucket)
            features[f"bucket_{window}_mode"] = self.window_calc.compute_mode(bucket_series, window)

            # Max (worst bucket)
            features[f"bucket_{window}_max"] = window_data.map(self.BUCKET_ORDINAL).max()

            # Months in each state
            for bucket_name in self.BUCKETS.keys():
                count = (window_data == bucket_name).sum()
                features[f"bucket_months_in_{bucket_name}_{window}"] = count

            # Bucket velocity (slope of ordinal bucket)
            if len(window_ordinal) >= 2:
                bucket_slope = self.window_calc._compute_single_slope(window_ordinal)
                features[f"bucket_velocity_{window}"] = bucket_slope
            else:
                features[f"bucket_velocity_{window}"] = np.nan

            # Bucket oscillation (number of changes)
            bucket_changes = (window_data != window_data.shift()).sum() - 1  # -1 to exclude first
            features[f"bucket_oscillation_{window}"] = bucket_changes

        return features

    def _compute_transition_features(self, dpd_series: pd.Series, windows: List[str]) -> Dict[str, float]:
        """Compute bucket transition features"""
        features = {}

        bucket_series = dpd_series.apply(self._dpd_to_bucket)

        # Transitions in last 3 months
        if len(bucket_series) >= 3:
            recent_3m = bucket_series.tail(3)

            # All possible transitions
            transitions = [
                ("SM", "NPL"),
                ("NPL", "CHARGE_OFF"),
                ("NPL", "SM"),
                ("NPL", "REGULAR"),
                ("SM", "REGULAR"),
                ("CHARGE_OFF", "NPL")
            ]

            for from_bucket, to_bucket in transitions:
                transition_occurred = False
                for i in range(len(recent_3m) - 1):
                    if recent_3m.iloc[i] == from_bucket and recent_3m.iloc[i+1] == to_bucket:
                        transition_occurred = True
                        break

                features[f"bucket_transition_{from_bucket}_{to_bucket}"] = int(transition_occurred)

        # Cure flag (moved from SM/NPL to REGULAR)
        for window in windows:
            window_months = self.window_calc.WINDOWS.get(window, int(window.replace("M", "")))
            window_data = bucket_series.tail(window_months)

            if len(window_data) < 2:
                features[f"cure_flag_{window}"] = 0
                features[f"re_delinquency_flag_{window}"] = 0
                continue

            # Cure: was SM/NPL, became REGULAR
            cure_occurred = False
            re_delinq_occurred = False

            for i in range(len(window_data) - 1):
                current_bucket = window_data.iloc[i]
                next_bucket = window_data.iloc[i+1]

                # Cure
                if current_bucket in ["SM", "NPL"] and next_bucket == "REGULAR":
                    cure_occurred = True

                    # Check for re-delinquency after cure
                    if i+2 < len(window_data):
                        subsequent = window_data.iloc[i+2:]
                        if (subsequent.isin(["SM", "NPL"])).any():
                            re_delinq_occurred = True

            features[f"cure_flag_{window}"] = int(cure_occurred)
            features[f"re_delinquency_flag_{window}"] = int(re_delinq_occurred)

        # Deterioration and improvement streaks
        bucket_ordinal_series = bucket_series.map(self.BUCKET_ORDINAL)

        deterioration_streak = self.window_calc.compute_streak(
            bucket_ordinal_series,
            lambda x: x > bucket_ordinal_series.shift(1).iloc[bucket_ordinal_series.index.get_loc(bucket_ordinal_series[bucket_ordinal_series == x].index[0]) - 1] if bucket_ordinal_series.index.get_loc(bucket_ordinal_series[bucket_ordinal_series == x].index[0]) > 0 else False
        )

        # Simplified: count consecutive months worsening
        deterioration_streak = 0
        for i in range(len(bucket_ordinal_series) - 1, 0, -1):
            if bucket_ordinal_series.iloc[i] > bucket_ordinal_series.iloc[i-1]:
                deterioration_streak += 1
            else:
                break

        features["bucket_deterioration_streak"] = deterioration_streak

        # Improvement streak (stable or improving)
        improvement_streak = 0
        for i in range(len(bucket_ordinal_series) - 1, 0, -1):
            if bucket_ordinal_series.iloc[i] <= bucket_ordinal_series.iloc[i-1]:
                improvement_streak += 1
            else:
                break

        features["bucket_improvement_streak"] = improvement_streak

        # Time to NPL from SM
        if (bucket_series == "SM").any() and (bucket_series == "NPL").any():
            sm_indices = bucket_series[bucket_series == "SM"].index
            npl_indices = bucket_series[bucket_series == "NPL"].index

            # Find first SM followed by NPL
            for sm_idx in sm_indices:
                subsequent_npl = npl_indices[npl_indices > sm_idx]
                if len(subsequent_npl) > 0:
                    first_npl_idx = subsequent_npl[0]
                    months_to_npl = bucket_series.index.get_loc(first_npl_idx) - bucket_series.index.get_loc(sm_idx)
                    features["time_to_npl_from_sm"] = months_to_npl
                    break
            else:
                features["time_to_npl_from_sm"] = np.nan
        else:
            features["time_to_npl_from_sm"] = np.nan

        return features

    def _compute_volatility_features(self, dpd_series: pd.Series, windows: List[str]) -> Dict[str, float]:
        """Compute volatility and stability features"""
        features = {}

        # Already computed: dpd_{W}_std, dpd_{W}_cv from statistics

        # Additional volatility: range
        for window in windows:
            window_months = self.window_calc.WINDOWS.get(window, int(window.replace("M", "")))
            window_data = dpd_series.tail(window_months)

            if len(window_data) >= 2:
                dpd_range = window_data.max() - window_data.min()
                features[f"dpd_{window}_range"] = dpd_range
            else:
                features[f"dpd_{window}_range"] = np.nan

        return features

    def _compute_regime_features(
        self,
        dpd_series: pd.Series,
        windows: List[str],
        existing_features: Dict[str, float]
    ) -> Dict[str, float]:
        """
        MANDATORY: Classify accounts into behavioral regimes.

        Regimes based on:
        - Level: Current DPD/bucket
        - Volatility: Std dev of DPD
        - Trend: Slope of DPD

        Regimes:
        - STABLE: Low DPD, low volatility, flat trend
        - DETERIORATING: Increasing DPD, positive slope
        - VOLATILE: High volatility, oscillating
        - RECOVERING: Decreasing DPD from elevated levels
        - SEVERELY_DELINQUENT: In NPL/CO with no improvement
        """
        features = {}

        # Use 6M window for regime classification
        dpd_current = existing_features.get("dpd_current", np.nan)
        dpd_6m_std = existing_features.get("6M_std", np.nan)
        dpd_6m_slope = existing_features.get("6M_slope", np.nan)
        bucket_current = existing_features.get("bucket_current", "REGULAR")

        # Regime classification logic
        regime = "UNKNOWN"
        regime_confidence = 0.0

        if pd.isna(dpd_current) or pd.isna(dpd_6m_std) or pd.isna(dpd_6m_slope):
            regime = "INSUFFICIENT_DATA"
            regime_confidence = 0.0
        else:
            # Stable: DPD < 30, low volatility, flat trend
            if dpd_current <= 30 and dpd_6m_std < 10 and abs(dpd_6m_slope) < 2:
                regime = "STABLE"
                regime_confidence = 0.9

            # Deteriorating: Positive slope, increasing DPD
            elif dpd_6m_slope > 3:
                regime = "DETERIORATING"
                regime_confidence = 0.85

            # Recovering: Negative slope, was delinquent
            elif dpd_6m_slope < -3 and dpd_current > 30:
                regime = "RECOVERING"
                regime_confidence = 0.8

            # Volatile: High std dev, oscillating
            elif dpd_6m_std > 20:
                regime = "VOLATILE"
                regime_confidence = 0.75

            # Severely delinquent: NPL/CO with flat/worsening trend
            elif bucket_current in ["NPL", "CHARGE_OFF"] and dpd_6m_slope >= 0:
                regime = "SEVERELY_DELINQUENT"
                regime_confidence = 0.95

            # Default: Elevated
            elif dpd_current > 30:
                regime = "ELEVATED"
                regime_confidence = 0.7
            else:
                regime = "STABLE"
                regime_confidence = 0.6

        features["delinquency_regime"] = regime
        features["delinquency_regime_confidence"] = regime_confidence

        # Regime persistence (months in current regime)
        # Simplified: if regime is same as 3 months ago, persistence = 3+
        features["delinquency_regime_persistence_months"] = 1  # Placeholder (requires historical regime tracking)

        # Binary flags for each regime
        for regime_name in ["STABLE", "DETERIORATING", "VOLATILE", "RECOVERING", "SEVERELY_DELINQUENT", "ELEVATED"]:
            features[f"regime_{regime_name.lower()}_flag"] = int(regime == regime_name)

        return features

    # ─────────────────────────────────────────────────────────────────────
    # Helper Methods
    # ─────────────────────────────────────────────────────────────────────

    def _dpd_to_bucket(self, dpd: float) -> str:
        """Convert DPD to bucket name"""
        if pd.isna(dpd):
            return "UNKNOWN"

        for bucket_name, (min_dpd, max_dpd) in self.BUCKETS.items():
            if min_dpd <= dpd <= max_dpd:
                return bucket_name

        return "UNKNOWN"

    def _compute_tenure_in_bucket(self, dpd_series: pd.Series) -> int:
        """Compute number of consecutive months in current bucket"""
        if len(dpd_series) == 0:
            return 0

        bucket_series = dpd_series.apply(self._dpd_to_bucket)
        current_bucket = bucket_series.iloc[-1]

        tenure = 1
        for i in range(len(bucket_series) - 2, -1, -1):
            if bucket_series.iloc[i] == current_bucket:
                tenure += 1
            else:
                break

        return tenure


def get_feature_metadata() -> Dict[str, Dict]:
    """
    Return metadata for all delinquency features.

    This will be used to populate feature_registry.json.
    """
    metadata = {
        "dpd_current": {
            "module": "bfe.delinquency",
            "tier": 1,
            "definition": "Current Days Past Due (spot value)",
            "feature_type": "continuous",
            "signal_direction": "higher = riskier",
            "min_history_months": 1,
            "use_cases": ["PD", "EWS", "Collections"],
            "product_applicability": ["all"]
        },
        "bucket_current": {
            "module": "bfe.delinquency",
            "tier": 1,
            "definition": "Current delinquency bucket (REGULAR/SM/NPL/CO)",
            "feature_type": "categorical",
            "signal_direction": "higher = riskier",
            "min_history_months": 1,
            "use_cases": ["PD", "EWS", "Collections"],
            "product_applicability": ["all"]
        },
        "delinquency_regime": {
            "module": "bfe.delinquency",
            "tier": 1,
            "definition": "Behavioral regime (STABLE/DETERIORATING/VOLATILE/RECOVERING/SEVERELY_DELINQUENT)",
            "feature_type": "categorical",
            "signal_direction": "varies by regime",
            "min_history_months": 6,
            "use_cases": ["PD", "EWS", "Collections", "Portfolio_Segmentation"],
            "product_applicability": ["all"]
        }
        # ... additional features would be listed here
    }

    return metadata
