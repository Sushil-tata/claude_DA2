"""
Point-in-time safe as-of joins with leakage prevention.

Implements temporal join operations that prevent future data leakage.
"""

import pandas as pd
from typing import Optional, List, Union
from datetime import datetime, date, timedelta
import logging

logger = logging.getLogger(__name__)


class AsOfJoiner:
    """
    Perform point-in-time safe as-of joins.
    
    Key principles:
    - Only use data available at prediction time
    - Strict timestamp validation
    - Leakage detection guards
    - Clear time boundaries
    """
    
    def __init__(self):
        """Initialize as-of joiner."""
        pass
    
    def as_of_join(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        left_on: str,
        right_on: str,
        left_timestamp: str,
        right_timestamp: str,
        suffixes: tuple = ('_left', '_right'),
        direction: str = 'backward',
        tolerance: Optional[pd.Timedelta] = None
    ) -> pd.DataFrame:
        """
        Perform as-of join: for each row in left, get latest row from right
        where right_timestamp <= left_timestamp.
        
        Args:
            left_df: Left DataFrame (e.g., prediction requests)
            right_df: Right DataFrame (e.g., features)
            left_on: Join key column in left
            right_on: Join key column in right
            left_timestamp: Timestamp column in left
            right_timestamp: Timestamp column in right
            suffixes: Suffixes for overlapping columns
            direction: 'backward' (default), 'forward', or 'nearest'
            tolerance: Maximum time difference allowed
            
        Returns:
            Joined DataFrame with point-in-time safety
            
        Example:
            # Get latest features for each prediction
            result = joiner.as_of_join(
                predictions_df,
                features_df,
                left_on='customer_id',
                right_on='customer_id',
                left_timestamp='prediction_date',
                right_timestamp='feature_date'
            )
        """
        # Convert timestamps to datetime
        left_df = left_df.copy()
        right_df = right_df.copy()
        
        left_df[left_timestamp] = pd.to_datetime(left_df[left_timestamp])
        right_df[right_timestamp] = pd.to_datetime(right_df[right_timestamp])
        
        # Sort both DataFrames by timestamp (required for merge_asof)
        # When using 'by' parameter, sort by timestamp only
        left_df = left_df.sort_values(left_timestamp)
        right_df = right_df.sort_values(right_timestamp)
        
        # Perform as-of join
        # Note: merge_asof requires 'by' parameter for joining on entity keys
        result = pd.merge_asof(
            left_df,
            right_df,
            left_on=left_timestamp,
            right_on=right_timestamp,
            by=left_on if left_on == right_on else None,
            suffixes=suffixes,
            direction=direction,
            tolerance=tolerance
        )
        
        # Validate no future data leaked
        self._validate_no_future_leakage(
            result, left_timestamp, right_timestamp
        )
        
        logger.info(f"As-of join completed: {len(result):,} rows")
        
        return result
    
    def point_in_time_snapshot(
        self,
        df: pd.DataFrame,
        entity_col: str,
        timestamp_col: str,
        as_of_date: Union[str, datetime, date]
    ) -> pd.DataFrame:
        """
        Create point-in-time snapshot: latest state of each entity as of date.
        
        Args:
            df: Input DataFrame with historical records
            entity_col: Entity identifier column (e.g., 'customer_id')
            timestamp_col: Timestamp column
            as_of_date: Snapshot date
            
        Returns:
            DataFrame with one row per entity (latest as of date)
            
        Example:
            # Get latest customer state as of 2024-01-31
            snapshot = joiner.point_in_time_snapshot(
                customer_history_df,
                entity_col='customer_id',
                timestamp_col='updated_at',
                as_of_date='2024-01-31'
            )
        """
        as_of_dt = pd.to_datetime(as_of_date)
        
        # Convert timestamp to datetime
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Filter to records <= as_of_date
        snapshot_df = df[df[timestamp_col] <= as_of_dt].copy()
        
        if len(snapshot_df) == 0:
            logger.warning(f"No records found <= {as_of_dt}")
            return pd.DataFrame()
        
        # Get latest record per entity
        snapshot_df = snapshot_df.sort_values(timestamp_col)
        snapshot_df = snapshot_df.groupby(entity_col).tail(1).reset_index(drop=True)
        
        logger.info(
            f"Point-in-time snapshot created: "
            f"{len(snapshot_df):,} entities as of {as_of_dt.date()}"
        )
        
        return snapshot_df
    
    def windowed_aggregation_asof(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        entity_col: str,
        left_timestamp: str,
        right_timestamp: str,
        lookback_days: int,
        agg_col: str,
        agg_funcs: List[str] = ['sum', 'mean', 'count']
    ) -> pd.DataFrame:
        """
        Compute windowed aggregations with point-in-time safety.
        
        For each row in left_df, aggregate right_df over lookback window
        ending at left_timestamp.
        
        Args:
            left_df: Left DataFrame (e.g., predictions)
            right_df: Right DataFrame (e.g., transactions)
            entity_col: Entity identifier
            left_timestamp: Timestamp in left
            right_timestamp: Timestamp in right
            lookback_days: Window size in days
            agg_col: Column to aggregate
            agg_funcs: Aggregation functions
            
        Returns:
            left_df with aggregated features added
            
        Example:
            # Get transaction stats over last 30 days
            result = joiner.windowed_aggregation_asof(
                predictions_df,
                transactions_df,
                entity_col='customer_id',
                left_timestamp='prediction_date',
                right_timestamp='transaction_date',
                lookback_days=30,
                agg_col='amount',
                agg_funcs=['sum', 'mean', 'count']
            )
        """
        result = left_df.copy()
        result[left_timestamp] = pd.to_datetime(result[left_timestamp])
        
        right_df = right_df.copy()
        right_df[right_timestamp] = pd.to_datetime(right_df[right_timestamp])
        
        # For each row in left, compute aggregations
        agg_results = []
        
        for idx, row in result.iterrows():
            entity_id = row[entity_col]
            snapshot_dt = row[left_timestamp]
            window_start = snapshot_dt - timedelta(days=lookback_days)
            
            # Filter right_df to window
            window_data = right_df[
                (right_df[entity_col] == entity_id) &
                (right_df[right_timestamp] >= window_start) &
                (right_df[right_timestamp] <= snapshot_dt)
            ]
            
            # Compute aggregations
            agg_row = {}
            for func in agg_funcs:
                if func == 'sum':
                    agg_row[f'{agg_col}_{lookback_days}d_sum'] = window_data[agg_col].sum()
                elif func == 'mean':
                    agg_row[f'{agg_col}_{lookback_days}d_mean'] = window_data[agg_col].mean()
                elif func == 'count':
                    agg_row[f'{agg_col}_{lookback_days}d_count'] = len(window_data)
                elif func == 'std':
                    agg_row[f'{agg_col}_{lookback_days}d_std'] = window_data[agg_col].std()
                elif func == 'min':
                    agg_row[f'{agg_col}_{lookback_days}d_min'] = window_data[agg_col].min()
                elif func == 'max':
                    agg_row[f'{agg_col}_{lookback_days}d_max'] = window_data[agg_col].max()
            
            agg_results.append(agg_row)
        
        # Add aggregated features to result
        agg_df = pd.DataFrame(agg_results)
        result = pd.concat([result.reset_index(drop=True), agg_df], axis=1)
        
        logger.info(
            f"Windowed aggregation completed: "
            f"{len(agg_funcs)} functions over {lookback_days} days"
        )
        
        return result
    
    def _validate_no_future_leakage(
        self,
        df: pd.DataFrame,
        left_timestamp: str,
        right_timestamp: str
    ) -> None:
        """
        Validate that no features use data from after the timestamp.
        
        Args:
            df: Joined DataFrame
            left_timestamp: Left timestamp column
            right_timestamp: Right timestamp column
            
        Raises:
            ValueError: If future leakage detected
        """
        # Check if right_timestamp is in the result
        if right_timestamp not in df.columns:
            # Timestamp might have suffix
            possible_cols = [col for col in df.columns if right_timestamp in col]
            if not possible_cols:
                logger.warning(f"Could not find {right_timestamp} for leakage check")
                return
            right_timestamp = possible_cols[0]
        
        # Check: all right timestamps <= left timestamps
        df_with_timestamps = df[[left_timestamp, right_timestamp]].dropna()
        
        if len(df_with_timestamps) == 0:
            return
        
        leakage = df_with_timestamps[
            df_with_timestamps[right_timestamp] > df_with_timestamps[left_timestamp]
        ]
        
        if len(leakage) > 0:
            raise ValueError(
                f"DATA LEAKAGE DETECTED!\n"
                f"Found {len(leakage)} rows where right_timestamp > left_timestamp\n"
                f"Example: {leakage.head()}"
            )
        
        logger.info("✓ No future data leakage detected")


class LeakageDetector:
    """Detect and prevent data leakage in feature engineering."""
    
    @staticmethod
    def detect_leakage(
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        feature_timestamp: str,
        label_timestamp: str,
        entity_col: str = 'customer_id'
    ) -> dict:
        """
        Detect if features use data from after label timestamp.
        
        Args:
            features_df: Features DataFrame
            labels_df: Labels DataFrame
            feature_timestamp: Feature timestamp column
            label_timestamp: Label timestamp column
            entity_col: Entity identifier for joining
            
        Returns:
            Dictionary with leakage detection results
            
        Raises:
            ValueError: If leakage detected
        """
        # Join features and labels
        joined = features_df.merge(labels_df, on=entity_col, how='inner')
        
        # Convert to datetime
        joined[feature_timestamp] = pd.to_datetime(joined[feature_timestamp])
        joined[label_timestamp] = pd.to_datetime(joined[label_timestamp])
        
        # Check for leakage
        leakage = joined[joined[feature_timestamp] > joined[label_timestamp]]
        
        result = {
            'leakage_detected': len(leakage) > 0,
            'leakage_rows': len(leakage),
            'total_rows': len(joined),
            'leakage_proportion': len(leakage) / len(joined) if len(joined) > 0 else 0
        }
        
        if result['leakage_detected']:
            raise ValueError(
                f"DATA LEAKAGE DETECTED!\n"
                f"  Rows with leakage: {result['leakage_rows']:,}\n"
                f"  Total rows: {result['total_rows']:,}\n"
                f"  Leakage proportion: {result['leakage_proportion']:.2%}"
            )
        
        logger.info("✓ No leakage detected")
        return result


# Example usage
if __name__ == "__main__":
    print("\n" + "="*80)
    print("AS-OF JOIN EXAMPLE")
    print("="*80)
    
    # Create sample data
    predictions = pd.DataFrame({
        'customer_id': ['A', 'B', 'A', 'B'],
        'prediction_date': pd.to_datetime([
            '2024-01-15', '2024-01-15', '2024-01-20', '2024-01-20'
        ])
    })
    
    features = pd.DataFrame({
        'customer_id': ['A', 'A', 'B', 'B'],
        'feature_date': pd.to_datetime([
            '2024-01-10', '2024-01-18', '2024-01-12', '2024-01-19'
        ]),
        'credit_score': [700, 720, 650, 660]
    })
    
    joiner = AsOfJoiner()
    result = joiner.as_of_join(
        predictions,
        features,
        left_on='customer_id',
        right_on='customer_id',
        left_timestamp='prediction_date',
        right_timestamp='feature_date'
    )
    
    print("\nPredictions:")
    print(predictions)
    print("\nFeatures:")
    print(features)
    print("\nJoined (as-of):")
    print(result[['customer_id', 'prediction_date', 'feature_date', 'credit_score']])
