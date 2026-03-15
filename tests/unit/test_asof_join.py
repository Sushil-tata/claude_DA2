"""
Unit tests for as-of joins and leakage detection (no Spark dependency).
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from decision_agent.data.asof_join import AsOfJoiner, LeakageDetector


class TestAsOfJoiner:
    """Test suite for AsOfJoiner."""
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions DataFrame."""
        return pd.DataFrame({
            'customer_id': ['A', 'B', 'A', 'B'],
            'prediction_date': pd.to_datetime([
                '2024-01-15', '2024-01-15', '2024-01-20', '2024-01-20'
            ]),
            'request_id': [1, 2, 3, 4]
        })
    
    @pytest.fixture
    def sample_features(self):
        """Create sample features DataFrame."""
        return pd.DataFrame({
            'customer_id': ['A', 'A', 'B', 'B', 'A'],
            'feature_date': pd.to_datetime([
                '2024-01-10', '2024-01-18', '2024-01-12',
                '2024-01-19', '2024-01-14'
            ]),
            'credit_score': [700, 720, 650, 660, 710]
        })
    
    def test_as_of_join_basic(self, sample_predictions, sample_features):
        """Test basic as-of join functionality."""
        joiner = AsOfJoiner()
        
        result = joiner.as_of_join(
            sample_predictions,
            sample_features,
            left_on='customer_id',
            right_on='customer_id',
            left_timestamp='prediction_date',
            right_timestamp='feature_date'
        )
        
        # Check we got results
        assert len(result) == len(sample_predictions)
        
        # Check columns exist
        assert 'credit_score' in result.columns
        assert 'prediction_date' in result.columns
    
    def test_as_of_join_respects_timestamp_order(
        self, sample_predictions, sample_features
    ):
        """Test that as-of join only uses past data."""
        joiner = AsOfJoiner()
        
        result = joiner.as_of_join(
            sample_predictions,
            sample_features,
            left_on='customer_id',
            right_on='customer_id',
            left_timestamp='prediction_date',
            right_timestamp='feature_date'
        )
        
        # For each row, feature_date should be <= prediction_date
        result_clean = result.dropna(subset=['feature_date'])
        
        assert all(
            result_clean['feature_date'] <= result_clean['prediction_date']
        ), "As-of join should only use past features"
    
    def test_point_in_time_snapshot(self):
        """Test point-in-time snapshot creation."""
        # Create historical data
        df = pd.DataFrame({
            'customer_id': ['A', 'A', 'B', 'B', 'A'],
            'updated_at': pd.to_datetime([
                '2024-01-10', '2024-01-20', '2024-01-15',
                '2024-01-25', '2024-01-30'
            ]),
            'balance': [1000, 1200, 500, 550, 1300]
        })
        
        joiner = AsOfJoiner()
        
        # Get snapshot as of 2024-01-22
        snapshot = joiner.point_in_time_snapshot(
            df,
            entity_col='customer_id',
            timestamp_col='updated_at',
            as_of_date='2024-01-22'
        )
        
        # Should have one row per customer
        assert len(snapshot) == 2
        
        # Check we got the latest values before cutoff
        customer_a = snapshot[snapshot['customer_id'] == 'A']
        assert customer_a['balance'].values[0] == 1200  # Jan 20 value
        
        customer_b = snapshot[snapshot['customer_id'] == 'B']
        assert customer_b['balance'].values[0] == 500  # Jan 15 value
    
    def test_point_in_time_snapshot_filters_future_data(self):
        """Test that snapshot excludes future data."""
        df = pd.DataFrame({
            'customer_id': ['A', 'A'],
            'updated_at': pd.to_datetime(['2024-01-10', '2024-02-10']),
            'value': [100, 200]
        })
        
        joiner = AsOfJoiner()
        
        snapshot = joiner.point_in_time_snapshot(
            df,
            entity_col='customer_id',
            timestamp_col='updated_at',
            as_of_date='2024-01-15'
        )
        
        # Should only get Jan 10 record
        assert len(snapshot) == 1
        assert snapshot['value'].values[0] == 100
    
    def test_windowed_aggregation(self):
        """Test windowed aggregation with point-in-time safety."""
        predictions = pd.DataFrame({
            'customer_id': ['A', 'B'],
            'prediction_date': pd.to_datetime(['2024-01-31', '2024-01-31'])
        })
        
        transactions = pd.DataFrame({
            'customer_id': ['A', 'A', 'A', 'B', 'B'],
            'transaction_date': pd.to_datetime([
                '2024-01-05', '2024-01-15', '2024-01-25',
                '2024-01-10', '2024-01-20'
            ]),
            'amount': [100, 200, 150, 50, 75]
        })
        
        joiner = AsOfJoiner()
        
        result = joiner.windowed_aggregation_asof(
            predictions,
            transactions,
            entity_col='customer_id',
            left_timestamp='prediction_date',
            right_timestamp='transaction_date',
            lookback_days=30,
            agg_col='amount',
            agg_funcs=['sum', 'mean', 'count']
        )
        
        # Check aggregated features created
        assert 'amount_30d_sum' in result.columns
        assert 'amount_30d_mean' in result.columns
        assert 'amount_30d_count' in result.columns
        
        # Check values for customer A
        customer_a = result[result['customer_id'] == 'A']
        assert customer_a['amount_30d_sum'].values[0] == 450  # 100+200+150
        assert customer_a['amount_30d_count'].values[0] == 3


class TestLeakageDetector:
    """Test suite for LeakageDetector."""
    
    def test_no_leakage_detected(self):
        """Test case with no leakage."""
        features = pd.DataFrame({
            'customer_id': ['A', 'B'],
            'feature_timestamp': pd.to_datetime(['2024-01-10', '2024-01-15']),
            'feature_value': [100, 200]
        })
        
        labels = pd.DataFrame({
            'customer_id': ['A', 'B'],
            'label_timestamp': pd.to_datetime(['2024-01-20', '2024-01-25']),
            'label': [1, 0]
        })
        
        detector = LeakageDetector()
        
        result = detector.detect_leakage(
            features,
            labels,
            feature_timestamp='feature_timestamp',
            label_timestamp='label_timestamp'
        )
        
        assert result['leakage_detected'] == False
        assert result['leakage_rows'] == 0
    
    def test_leakage_detected_raises_error(self):
        """Test that leakage is detected and raises error."""
        features = pd.DataFrame({
            'customer_id': ['A', 'B'],
            'feature_timestamp': pd.to_datetime(['2024-01-25', '2024-01-15']),
            'feature_value': [100, 200]
        })
        
        labels = pd.DataFrame({
            'customer_id': ['A', 'B'],
            'label_timestamp': pd.to_datetime(['2024-01-20', '2024-01-25']),
            'label': [1, 0]
        })
        
        detector = LeakageDetector()
        
        with pytest.raises(ValueError, match="DATA LEAKAGE DETECTED"):
            detector.detect_leakage(
                features,
                labels,
                feature_timestamp='feature_timestamp',
                label_timestamp='label_timestamp'
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
