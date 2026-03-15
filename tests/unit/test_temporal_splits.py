"""
Unit tests for temporal splitting (no Spark dependency).
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from decision_agent.data.splits import TemporalSplitter


class TestTemporalSplitter:
    """Test suite for TemporalSplitter."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame with 100 days of data."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        return pd.DataFrame({
            'date': dates,
            'value': range(100),
            'customer_id': ['CUST001'] * 100
        })
    
    def test_create_splits_basic(self, sample_data):
        """Test basic temporal split functionality."""
        splitter = TemporalSplitter()
        
        train, val, test = splitter.create_splits(
            sample_data,
            date_col='date',
            train_end='2023-03-01',
            val_end='2023-03-20'
        )
        
        # Check sizes
        assert len(train) > 0, "Training set should not be empty"
        assert len(val) > 0, "Validation set should not be empty"
        assert len(test) > 0, "Test set should not be empty"
        
        # Check total rows preserved
        assert len(train) + len(val) + len(test) == len(sample_data)
        
        # Check temporal ordering
        assert train['date'].max() < val['date'].min()
        assert val['date'].max() < test['date'].min()
    
    def test_no_temporal_overlap(self, sample_data):
        """Test that splits have no temporal overlap."""
        splitter = TemporalSplitter()
        
        train, val, test = splitter.create_splits(
            sample_data,
            date_col='date',
            train_end='2023-02-01',
            val_end='2023-03-01'
        )
        
        # Max of train < min of val
        assert train['date'].max() < val['date'].min()
        
        # Max of val < min of test
        assert val['date'].max() < test['date'].min()
    
    def test_invalid_date_order_raises_error(self, sample_data):
        """Test that invalid date ordering raises ValueError."""
        splitter = TemporalSplitter()
        
        with pytest.raises(ValueError, match="must be before"):
            splitter.create_splits(
                sample_data,
                date_col='date',
                train_end='2023-03-20',
                val_end='2023-03-01'  # Invalid: before train_end
            )
    
    def test_create_splits_by_ratio(self, sample_data):
        """Test splitting by ratio."""
        splitter = TemporalSplitter()
        
        train, val, test = splitter.create_splits_by_ratio(
            sample_data,
            date_col='date',
            train_ratio=0.7,
            val_ratio=0.15
        )
        
        total = len(sample_data)
        
        # Check approximate ratios (within 2% tolerance)
        assert abs(len(train) / total - 0.7) < 0.02
        assert abs(len(val) / total - 0.15) < 0.02
        assert abs(len(test) / total - 0.15) < 0.02
        
        # Check temporal ordering
        assert train['date'].max() <= val['date'].min()
        assert val['date'].max() <= test['date'].min()
    
    def test_invalid_ratios_raise_error(self, sample_data):
        """Test that invalid ratios raise ValueError."""
        splitter = TemporalSplitter()
        
        # Test ratio > 1
        with pytest.raises(ValueError, match="between 0 and 1"):
            splitter.create_splits_by_ratio(
                sample_data,
                date_col='date',
                train_ratio=1.5
            )
        
        # Test train + val >= 1
        with pytest.raises(ValueError, match="must be < 1"):
            splitter.create_splits_by_ratio(
                sample_data,
                date_col='date',
                train_ratio=0.8,
                val_ratio=0.3
            )
    
    def test_get_split_statistics(self, sample_data):
        """Test split statistics calculation."""
        splitter = TemporalSplitter()
        
        train, val, test = splitter.create_splits(
            sample_data,
            date_col='date',
            train_end='2023-02-15',
            val_end='2023-03-15'
        )
        
        stats = splitter.get_split_statistics(train, val, test, 'date')
        
        # Check structure
        assert 'train' in stats
        assert 'val' in stats
        assert 'test' in stats
        assert 'total_rows' in stats
        
        # Check proportions sum to 1
        total_prop = (
            stats['train']['proportion'] +
            stats['val']['proportion'] +
            stats['test']['proportion']
        )
        assert abs(total_prop - 1.0) < 0.001
        
        # Check date ranges
        assert stats['train']['date_range'][0] is not None
        assert stats['train']['date_range'][1] is not None
    
    def test_empty_train_raises_error(self):
        """Test that empty training set raises error."""
        df = pd.DataFrame({
            'date': pd.date_range('2023-03-01', periods=10, freq='D'),
            'value': range(10)
        })
        
        splitter = TemporalSplitter()
        
        # train_end before all data
        with pytest.raises(ValueError, match="Training set is empty"):
            splitter.create_splits(
                df,
                date_col='date',
                train_end='2023-02-01',  # Before all data
                val_end='2023-03-05'
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
