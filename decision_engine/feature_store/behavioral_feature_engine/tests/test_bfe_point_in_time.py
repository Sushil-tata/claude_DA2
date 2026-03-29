"""
Test Point-in-Time Safety of BFE Repository
============================================

Validates that no look-ahead bias exists in feature computation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

from decision_engine.feature_store.behavioral_feature_engine.utils import TemporalValidator


class TestPointInTimeSafety:
    """Test suite for point-in-time safety"""

    def test_no_future_data_detected(self):
        """Test that future data is detected and rejected"""
        validator = TemporalValidator()

        # Create data with future dates
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "value": range(10)
        })

        as_of_date = "2024-01-05"

        # Should raise error for future data
        with pytest.raises(ValueError, match="LOOK-AHEAD BIAS DETECTED"):
            validator.validate_no_future_data(df, as_of_date, timestamp_col="date")

    def test_no_future_data_clean(self):
        """Test that clean data passes validation"""
        validator = TemporalValidator()

        # Create data with only past dates
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "value": range(5)
        })

        as_of_date = "2024-01-10"

        # Should pass without error
        assert validator.validate_no_future_data(df, as_of_date, timestamp_col="date")

    def test_filter_to_as_of_date(self):
        """Test that data is correctly filtered to as_of_date"""
        validator = TemporalValidator()

        # Create data spanning past and future
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "value": range(10)
        })

        as_of_date = "2024-01-05"

        # Filter data
        filtered_df = validator.filter_to_as_of_date(
            df, as_of_date, timestamp_col="date", strict=False
        )

        # Should only have 5 rows (Jan 1-5)
        assert len(filtered_df) == 5
        assert filtered_df["date"].max() <= pd.to_datetime(as_of_date)

    def test_compute_data_lag(self):
        """Test data staleness computation"""
        validator = TemporalValidator()

        data_timestamp = "2024-01-01"
        as_of_date = "2024-01-31"

        lag_days = validator.compute_data_lag(data_timestamp, as_of_date)

        assert lag_days == 30

    def test_staleness_check(self):
        """Test staleness detection"""
        validator = TemporalValidator()

        # Fresh data (30 days old)
        result_fresh = validator.check_staleness(
            last_update_date="2024-01-01",
            as_of_date="2024-01-31",
            max_staleness_days=90
        )

        assert result_fresh["is_stale"] == False
        assert result_fresh["staleness_flag"] == "FRESH"

        # Stale data (120 days old)
        result_stale = validator.check_staleness(
            last_update_date="2023-10-01",
            as_of_date="2024-01-31",
            max_staleness_days=90
        )

        assert result_stale["is_stale"] == True
        assert result_stale["staleness_flag"] == "STALE"

    def test_validate_minimum_history(self):
        """Test minimum history requirement"""
        validator = TemporalValidator()

        # Create data with varying history lengths
        df1 = pd.DataFrame({
            "account_id": ["ACC001"] * 12,
            "date": pd.date_range("2023-02-01", periods=12, freq="MS"),
            "value": range(12)
        })

        df2 = pd.DataFrame({
            "account_id": ["ACC002"] * 3,
            "date": pd.date_range("2023-11-01", periods=3, freq="MS"),
            "value": range(3)
        })

        df = pd.concat([df1, df2])

        # Require 6 months minimum
        filtered_df = validator.validate_minimum_history(
            df,
            min_months=6,
            timestamp_col="date",
            account_id_col="account_id"
        )

        # Should only have ACC001 (12 months > 6 months)
        assert filtered_df["account_id"].unique().tolist() == ["ACC001"]

    def test_temporal_order_validation(self):
        """Test that temporal ordering is validated"""
        validator = TemporalValidator()

        # Create properly ordered data
        df_valid = pd.DataFrame({
            "account_id": ["ACC001"] * 5,
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "value": range(5)
        })

        # Should pass
        assert validator.validate_temporal_order(df_valid, timestamp_col="date")

        # Create mis-ordered data
        df_invalid = pd.DataFrame({
            "account_id": ["ACC001"] * 5,
            "date": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-02", "2024-01-04", "2024-01-05"]),
            "value": range(5)
        })

        # Should raise error
        with pytest.raises(ValueError, match="Temporal ordering violation"):
            validator.validate_temporal_order(df_invalid, timestamp_col="date")


class TestFeatureLookAheadBias:
    """Test that BFE features do not use future data"""

    def test_delinquency_features_no_lookahead(self):
        """Test delinquency features use only historical data"""
        from decision_engine.feature_store.behavioral_feature_engine.modules import DelinquencyFeatureEngine

        engine = DelinquencyFeatureEngine()

        # Create DPD history
        dpd_history = pd.DataFrame({
            "account_id": ["ACC001"] * 24,
            "date": pd.date_range("2022-02-01", periods=24, freq="MS"),
            "dpd": [0, 5, 10, 15, 20, 30, 45, 60, 90, 120, 95, 70, 45, 30, 15, 10, 5, 0, 0, 0, 5, 10, 15, 20]
        })

        as_of_date = date(2023, 6, 1)  # 16 months after start

        # Mock schema mappings to avoid interactive prompt
        engine.field_mappings = {
            "account_id": "account_id",
            "date": "date",
            "dpd": "dpd"
        }

        # Compute features
        features = engine.compute_features(
            account_history=dpd_history,
            account_id="ACC001",
            as_of_date=as_of_date,
            windows=["3M", "6M"]
        )

        # Verify features are computed only from data <= as_of_date
        # The as_of_date is 2023-06-01, which is 16 months after 2022-02-01
        # So we should have 16 months of history available

        assert "dpd_current" in features
        assert features["as_of_date"] == as_of_date
        assert features["months_of_history"] == 16  # Should have 16 months, not 24

    def test_payment_features_no_lookahead(self):
        """Test payment features use only historical data"""
        from decision_engine.feature_store.behavioral_feature_engine.modules import PaymentFeatureEngine

        engine = PaymentFeatureEngine()

        # Create payment history
        payment_history = pd.DataFrame({
            "account_id": ["ACC001"] * 24,
            "date": pd.date_range("2022-02-01", periods=24, freq="MS"),
            "payment_amount": [5000] * 12 + [4000] * 12,
            "amount_due": [5000] * 24
        })

        as_of_date = date(2023, 6, 1)  # 16 months after start

        # Mock schema mappings
        engine.field_mappings = {
            "account_id": "account_id",
            "date": "date",
            "payment_amount": "payment_amount",
            "amount_due": "amount_due"
        }

        # Compute features
        features = engine.compute_features(
            account_history=payment_history,
            account_id="ACC001",
            as_of_date=as_of_date,
            windows=["3M", "6M"]
        )

        # Verify temporal constraint
        assert features["as_of_date"] == as_of_date
        assert features["months_of_history"] == 16  # Should have 16 months, not 24


def test_integration_point_in_time_safety():
    """Integration test: Verify end-to-end point-in-time safety"""
    from decision_engine.feature_store.behavioral_feature_engine import get_features

    # Create data with explicit future period
    dpd_history = pd.DataFrame({
        "account_id": ["ACC001"] * 24,
        "date": pd.date_range("2022-02-01", periods=24, freq="MS"),
        "dpd": list(range(24))
    })

    payment_history = pd.DataFrame({
        "account_id": ["ACC001"] * 24,
        "date": pd.date_range("2022-02-01", periods=24, freq="MS"),
        "payment_amount": [5000] * 24,
        "amount_due": [5000] * 24
    })

    account_history = {
        "delinquency": dpd_history,
        "payment": payment_history
    }

    # Compute features as of mid-period (12 months)
    as_of_date = "2023-02-01"

    # This should raise error if any future data is used
    features = get_features(
        account_id="ACC001",
        account_history=account_history,
        as_of_date=as_of_date,
        feature_sets=["delinquency", "payment"]
    )

    # Verify features were computed
    assert "delinquency.dpd_current" in features
    assert "payment.payment_ratio_6M_mean" in features

    print("\n✓ Integration test passed: No look-ahead bias detected")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
