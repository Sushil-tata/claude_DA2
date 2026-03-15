"""
Temporal data splitting for time-series safe train/validation/test sets.

Prevents data leakage by ensuring chronological ordering and no overlap.
"""

import pandas as pd
from typing import Tuple, Union, Optional
from datetime import datetime, date
import logging

logger = logging.getLogger(__name__)


class TemporalSplitter:
    """
    Create temporal train/validation/test splits with leakage prevention.
    
    Key principles:
    - Chronological ordering (train < validation < test)
    - No temporal overlap between sets
    - Clear date boundaries
    - Validation of split integrity
    """
    
    def __init__(self):
        """Initialize temporal splitter."""
        pass
    
    def create_splits(
        self,
        df: pd.DataFrame,
        date_col: str,
        train_end: Union[str, datetime, date],
        val_end: Union[str, datetime, date],
        validate_integrity: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split DataFrame into train/val/test by date ranges.
        
        Args:
            df: Input DataFrame
            date_col: Name of date/timestamp column
            train_end: End date for training set (exclusive)
            val_end: End date for validation set (exclusive)
            validate_integrity: Check for temporal integrity
            
        Returns:
            Tuple of (train_df, val_df, test_df)
            
        Raises:
            ValueError: If dates are invalid or splits overlap
            
        Example:
            splitter = TemporalSplitter()
            train, val, test = splitter.create_splits(
                df, 
                date_col='transaction_timestamp',
                train_end='2023-10-31',
                val_end='2023-12-31'
            )
        """
        # Convert dates to pandas datetime
        train_end_dt = pd.to_datetime(train_end)
        val_end_dt = pd.to_datetime(val_end)
        
        # Validate date ordering
        if train_end_dt >= val_end_dt:
            raise ValueError(
                f"train_end ({train_end_dt}) must be before val_end ({val_end_dt})"
            )
        
        # Convert date column to datetime
        df_with_dt = df.copy()
        df_with_dt[date_col] = pd.to_datetime(df_with_dt[date_col])
        
        # Create splits with strict date boundaries
        train_df = df_with_dt[df_with_dt[date_col] < train_end_dt].copy()
        val_df = df_with_dt[
            (df_with_dt[date_col] >= train_end_dt) & 
            (df_with_dt[date_col] < val_end_dt)
        ].copy()
        test_df = df_with_dt[df_with_dt[date_col] >= val_end_dt].copy()
        
        # Validate integrity
        if validate_integrity:
            self._validate_split_integrity(
                train_df, val_df, test_df, 
                date_col, train_end_dt, val_end_dt
            )
        
        logger.info(f"Temporal splits created:")
        logger.info(f"  Train: {len(train_df):,} rows (< {train_end_dt.date()})")
        logger.info(f"  Val:   {len(val_df):,} rows ({train_end_dt.date()} to {val_end_dt.date()})")
        logger.info(f"  Test:  {len(test_df):,} rows (>= {val_end_dt.date()})")
        
        return train_df, val_df, test_df
    
    def create_splits_by_ratio(
        self,
        df: pd.DataFrame,
        date_col: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        validate_integrity: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split by ratio while maintaining temporal ordering.
        
        Args:
            df: Input DataFrame
            date_col: Name of date/timestamp column
            train_ratio: Proportion for training (default 0.7)
            val_ratio: Proportion for validation (default 0.15)
            validate_integrity: Check temporal integrity
            
        Returns:
            Tuple of (train_df, val_df, test_df)
            
        Example:
            # 70% train, 15% val, 15% test
            train, val, test = splitter.create_splits_by_ratio(
                df, date_col='timestamp', train_ratio=0.7, val_ratio=0.15
            )
        """
        if not (0 < train_ratio < 1 and 0 < val_ratio < 1):
            raise ValueError("Ratios must be between 0 and 1")
        
        if train_ratio + val_ratio >= 1:
            raise ValueError("train_ratio + val_ratio must be < 1")
        
        # Sort by date
        df_sorted = df.sort_values(date_col).reset_index(drop=True)
        
        # Calculate split indices
        n = len(df_sorted)
        train_idx = int(n * train_ratio)
        val_idx = int(n * (train_ratio + val_ratio))
        
        # Split by index (temporal ordering preserved)
        train_df = df_sorted.iloc[:train_idx].copy()
        val_df = df_sorted.iloc[train_idx:val_idx].copy()
        test_df = df_sorted.iloc[val_idx:].copy()
        
        # Get date boundaries
        train_end_dt = pd.to_datetime(train_df[date_col].max())
        val_end_dt = pd.to_datetime(val_df[date_col].max())
        
        if validate_integrity:
            self._validate_split_integrity(
                train_df, val_df, test_df,
                date_col, train_end_dt, val_end_dt
            )
        
        logger.info(f"Temporal splits created by ratio:")
        logger.info(f"  Train: {len(train_df):,} rows ({train_ratio:.1%})")
        logger.info(f"  Val:   {len(val_df):,} rows ({val_ratio:.1%})")
        logger.info(f"  Test:  {len(test_df):,} rows ({1-train_ratio-val_ratio:.1%})")
        
        return train_df, val_df, test_df
    
    def _validate_split_integrity(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        date_col: str,
        train_end_dt: pd.Timestamp,
        val_end_dt: pd.Timestamp
    ) -> None:
        """
        Validate that splits have no temporal overlap.
        
        Args:
            train_df: Training set
            val_df: Validation set
            test_df: Test set
            date_col: Date column name
            train_end_dt: Training end date
            val_end_dt: Validation end date
            
        Raises:
            ValueError: If temporal overlap detected
        """
        # Check all splits are non-empty
        if len(train_df) == 0:
            raise ValueError("Training set is empty")
        if len(val_df) == 0:
            logger.warning("Validation set is empty")
        if len(test_df) == 0:
            logger.warning("Test set is empty")
        
        # Check temporal ordering
        train_max = pd.to_datetime(train_df[date_col].max())
        val_min = pd.to_datetime(val_df[date_col].min()) if len(val_df) > 0 else pd.Timestamp.max
        val_max = pd.to_datetime(val_df[date_col].max()) if len(val_df) > 0 else pd.Timestamp.min
        test_min = pd.to_datetime(test_df[date_col].min()) if len(test_df) > 0 else pd.Timestamp.max
        
        # Validate: train_max < val_min
        if len(val_df) > 0 and train_max >= val_min:
            raise ValueError(
                f"Temporal overlap detected: "
                f"train_max={train_max} >= val_min={val_min}"
            )
        
        # Validate: val_max < test_min
        if len(val_df) > 0 and len(test_df) > 0 and val_max >= test_min:
            raise ValueError(
                f"Temporal overlap detected: "
                f"val_max={val_max} >= test_min={test_min}"
            )
        
        logger.info("✓ Temporal integrity validated - no overlap detected")
    
    def get_split_statistics(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        date_col: str
    ) -> dict:
        """
        Get detailed statistics about the splits.
        
        Args:
            train_df: Training set
            val_df: Validation set
            test_df: Test set
            date_col: Date column name
            
        Returns:
            Dictionary with split statistics
        """
        total_rows = len(train_df) + len(val_df) + len(test_df)
        
        stats = {
            "train": {
                "rows": len(train_df),
                "proportion": len(train_df) / total_rows if total_rows > 0 else 0,
                "date_range": (
                    train_df[date_col].min(),
                    train_df[date_col].max()
                ) if len(train_df) > 0 else (None, None)
            },
            "val": {
                "rows": len(val_df),
                "proportion": len(val_df) / total_rows if total_rows > 0 else 0,
                "date_range": (
                    val_df[date_col].min(),
                    val_df[date_col].max()
                ) if len(val_df) > 0 else (None, None)
            },
            "test": {
                "rows": len(test_df),
                "proportion": len(test_df) / total_rows if total_rows > 0 else 0,
                "date_range": (
                    test_df[date_col].min(),
                    test_df[date_col].max()
                ) if len(test_df) > 0 else (None, None)
            },
            "total_rows": total_rows
        }
        
        return stats


# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'value': range(100)
    })
    
    print("\n" + "="*80)
    print("TEMPORAL SPLITTER EXAMPLE")
    print("="*80)
    
    splitter = TemporalSplitter()
    
    # Split by explicit dates
    train, val, test = splitter.create_splits(
        df,
        date_col='date',
        train_end='2023-03-01',
        val_end='2023-03-20'
    )
    
    print(f"\nTrain dates: {train['date'].min()} to {train['date'].max()}")
    print(f"Val dates:   {val['date'].min()} to {val['date'].max()}")
    print(f"Test dates:  {test['date'].min()} to {test['date'].max()}")
    
    # Get statistics
    stats = splitter.get_split_statistics(train, val, test, 'date')
    print(f"\nSplit proportions:")
    print(f"  Train: {stats['train']['proportion']:.1%}")
    print(f"  Val:   {stats['val']['proportion']:.1%}")
    print(f"  Test:  {stats['test']['proportion']:.1%}")
