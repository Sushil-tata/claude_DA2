"""
Test as-of join logic using pandas (no Spark required).
"""
import pytest
import pandas as pd
from datetime import datetime, timedelta
from decision_agent.data.asof_join import PointInTimeJoiner


def test_asof_join_pandas():
    """Test as-of join with pandas DataFrames"""
    # Create left DataFrame (labels with prediction timestamps)
    left_data = {
        "customer_id": [1, 1, 2, 2],
        "prediction_date": [
            datetime(2024, 1, 5),
            datetime(2024, 1, 10),
            datetime(2024, 1, 6),
            datetime(2024, 1, 12)
        ],
        "label": [1, 0, 1, 0]
    }
    left_df = pd.DataFrame(left_data)

    # Create right DataFrame (features with feature timestamps)
    right_data = {
        "customer_id": [1, 1, 1, 2, 2],
        "feature_date": [
            datetime(2024, 1, 1),
            datetime(2024, 1, 4),
            datetime(2024, 1, 8),
            datetime(2024, 1, 3),
            datetime(2024, 1, 5)
        ],
        "feature_value": [10, 20, 30, 15, 25]
    }
    right_df = pd.DataFrame(right_data)

    # Perform as-of join
    joiner = PointInTimeJoiner(spark=None)
    result = joiner.as_of_join(
        left_df,
        right_df,
        entity_key="customer_id",
        left_timestamp="prediction_date",
        right_timestamp="feature_date"
    )

    # Verify results
    assert len(result) == 4

    # For customer 1, prediction on 2024-01-05 should get feature from 2024-01-04
    row1 = result[(result["customer_id"] == 1) & (result["prediction_date"] == datetime(2024, 1, 5))]
    assert len(row1) == 1
    assert row1.iloc[0]["feature_value"] == 20  # Feature from 2024-01-04

    # For customer 1, prediction on 2024-01-10 should get feature from 2024-01-08
    row2 = result[(result["customer_id"] == 1) & (result["prediction_date"] == datetime(2024, 1, 10))]
    assert len(row2) == 1
    assert row2.iloc[0]["feature_value"] == 30  # Feature from 2024-01-08


def test_asof_join_no_leakage():
    """Test that as-of join doesn't use future data"""
    # Create scenario where feature is after prediction date
    left_data = {
        "customer_id": [1],
        "prediction_date": [datetime(2024, 1, 5)],
        "label": [1]
    }
    left_df = pd.DataFrame(left_data)

    # Feature is AFTER prediction date
    right_data = {
        "customer_id": [1],
        "feature_date": [datetime(2024, 1, 10)],  # Future date
        "feature_value": [100]
    }
    right_df = pd.DataFrame(right_data)

    joiner = PointInTimeJoiner(spark=None)
    result = joiner.as_of_join(
        left_df,
        right_df,
        entity_key="customer_id",
        left_timestamp="prediction_date",
        right_timestamp="feature_date"
    )

    # Should have no match (feature_value should be NaN)
    assert len(result) == 1
    assert pd.isna(result.iloc[0]["feature_value"])


def test_asof_join_with_lookback_window():
    """Test as-of join with lookback window constraint"""
    left_data = {
        "customer_id": [1],
        "prediction_date": [datetime(2024, 1, 30)],
        "label": [1]
    }
    left_df = pd.DataFrame(left_data)

    # Features at different dates
    right_data = {
        "customer_id": [1, 1],
        "feature_date": [
            datetime(2024, 1, 1),   # 29 days ago - outside window
            datetime(2024, 1, 15)   # 15 days ago - inside window
        ],
        "feature_value": [10, 20]
    }
    right_df = pd.DataFrame(right_data)

    joiner = PointInTimeJoiner(spark=None)
    result = joiner.as_of_join(
        left_df,
        right_df,
        entity_key="customer_id",
        left_timestamp="prediction_date",
        right_timestamp="feature_date",
        lookback_window_days=20  # Only look back 20 days
    )

    # Should get feature from 2024-01-15 (inside window)
    assert len(result) == 1
    assert result.iloc[0]["feature_value"] == 20


def test_validate_no_leakage_detection():
    """Test leakage detection logic"""
    # Create features DataFrame
    features_data = {
        "entity_id": [1, 2, 3],
        "feature_timestamp": [
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
            datetime(2024, 1, 3)
        ],
        "feature_value": [10, 20, 30]
    }
    features_df = pd.DataFrame(features_data)

    # Create labels DataFrame where one has leakage
    labels_data = {
        "entity_id": [1, 2, 3],
        "label_timestamp": [
            datetime(2024, 1, 2),  # OK: feature is before label
            datetime(2024, 1, 1),  # LEAKAGE: feature is after label
            datetime(2024, 1, 4)   # OK: feature is before label
        ],
        "label": [1, 0, 1]
    }
    labels_df = pd.DataFrame(labels_data)

    joiner = PointInTimeJoiner(spark=None)

    # Should detect leakage
    with pytest.raises(ValueError, match="DATA LEAKAGE DETECTED"):
        joiner.validate_no_leakage(
            features_df,
            labels_df,
            feature_timestamp="feature_timestamp",
            label_timestamp="label_timestamp",
            entity_key="entity_id"
        )


def test_validate_no_leakage_clean_data():
    """Test leakage validation with clean data"""
    # Create features DataFrame
    features_data = {
        "entity_id": [1, 2, 3],
        "feature_timestamp": [
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
            datetime(2024, 1, 3)
        ],
        "feature_value": [10, 20, 30]
    }
    features_df = pd.DataFrame(features_data)

    # Create labels DataFrame with no leakage
    labels_data = {
        "entity_id": [1, 2, 3],
        "label_timestamp": [
            datetime(2024, 1, 5),
            datetime(2024, 1, 6),
            datetime(2024, 1, 7)
        ],
        "label": [1, 0, 1]
    }
    labels_df = pd.DataFrame(labels_data)

    joiner = PointInTimeJoiner(spark=None)

    # Should pass validation
    result = joiner.validate_no_leakage(
        features_df,
        labels_df,
        feature_timestamp="feature_timestamp",
        label_timestamp="label_timestamp",
        entity_key="entity_id"
    )

    assert result is True
