"""
Online Feature Store

Low-latency feature serving using Redis for caching.

Features:
- Redis integration for sub-millisecond feature lookups
- TTL-based cache expiration
- Batch cache warming from Delta Lake
- Fallback to Delta on cache miss
- Cache hit rate monitoring

Performance Targets:
- Cache hit rate: > 80%
- Feature lookup latency: < 5ms (p95)
- Cache warming: < 10 minutes for 1M customers

Architecture:
  Daily Cache Warming:
    Delta Lake (batch features)
      ↓
    Load top customers (by frequency)
      ↓
    Populate Redis cache
      ↓
    Monitor cache hit rate

  Real-Time Lookup:
    Request → Redis lookup (cache)
      ├─ Cache hit → Return features (< 5ms)
      └─ Cache miss → Query Delta → Cache result → Return
"""

import logging
import json
from typing import Dict, Any, Optional, List
import redis
from datetime import timedelta
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


class OnlineFeatureStore:
    """
    Online feature store with Redis caching and Delta Lake fallback.
    """

    def __init__(
        self,
        redis_config: Dict[str, Any],
        delta_config: Dict[str, Any],
        spark: Optional[SparkSession] = None
    ):
        """
        Initialize online feature store.

        Args:
            redis_config: Redis configuration:
                - host: Redis host
                - port: Redis port
                - db: Redis database number
                - password: Redis password (optional)
                - ttl_seconds: TTL for cached features
            delta_config: Delta Lake configuration:
                - feature_table: Delta table with batch features
            spark: Spark session (for Delta fallback)
        """
        self.redis_config = redis_config
        self.delta_config = delta_config
        self.spark = spark

        # Initialize Redis client
        self.redis_client = redis.Redis(
            host=redis_config["host"],
            port=redis_config.get("port", 6379),
            db=redis_config.get("db", 0),
            password=redis_config.get("password"),
            decode_responses=True,
            socket_connect_timeout=5
        )

        self.ttl_seconds = redis_config.get("ttl_seconds", 86400)  # 24 hours default

        # Test connection
        self._test_connection()

        logger.info("OnlineFeatureStore initialized:")
        logger.info(f"  Redis: {redis_config['host']}:{redis_config.get('port', 6379)}")
        logger.info(f"  TTL: {self.ttl_seconds} seconds")
        logger.info(f"  Delta table: {delta_config.get('feature_table')}")

    def _test_connection(self):
        """Test Redis connection."""
        try:
            self.redis_client.ping()
            logger.info("✓ Redis connection successful")
        except redis.ConnectionError as e:
            logger.error(f"✗ Redis connection failed: {e}")
            raise

    def get_features(
        self,
        customer_id: str,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get features for a customer (cache-first strategy).

        Args:
            customer_id: Customer ID
            feature_names: Optional list of feature names to retrieve

        Returns:
            Dict of {feature_name: value}
        """
        # Try cache first
        cached_features = self._get_from_cache(customer_id)

        if cached_features:
            # Cache hit
            if feature_names:
                # Filter to requested features
                return {k: v for k, v in cached_features.items() if k in feature_names}
            return cached_features

        # Cache miss - fallback to Delta Lake
        logger.debug(f"Cache miss for customer: {customer_id}")
        delta_features = self._get_from_delta(customer_id, feature_names)

        if delta_features:
            # Cache result for future
            self._set_in_cache(customer_id, delta_features)

        return delta_features or {}

    def _get_from_cache(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """Get features from Redis cache."""
        try:
            key = self._make_key(customer_id)
            cached = self.redis_client.get(key)

            if cached:
                return json.loads(cached)
            return None

        except Exception as e:
            logger.warning(f"Redis get failed: {e}")
            return None

    def _set_in_cache(
        self,
        customer_id: str,
        features: Dict[str, Any]
    ) -> bool:
        """Set features in Redis cache with TTL."""
        try:
            key = self._make_key(customer_id)
            value = json.dumps(features)

            self.redis_client.setex(
                name=key,
                time=self.ttl_seconds,
                value=value
            )
            return True

        except Exception as e:
            logger.warning(f"Redis set failed: {e}")
            return False

    def _get_from_delta(
        self,
        customer_id: str,
        feature_names: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Get features from Delta Lake (fallback)."""
        if not self.spark:
            logger.warning("Spark session not available. Cannot query Delta Lake.")
            return None

        try:
            feature_table = self.delta_config["feature_table"]

            # Query Delta for customer
            query = f"""
            SELECT *
            FROM {feature_table}
            WHERE customer_id = '{customer_id}'
            ORDER BY as_of_date DESC
            LIMIT 1
            """

            result = self.spark.sql(query).collect()

            if not result:
                return None

            # Convert Row to dict
            features = result[0].asDict()

            # Remove metadata columns
            features.pop("customer_id", None)
            features.pop("as_of_date", None)
            features.pop("created_timestamp", None)

            # Filter to requested features if specified
            if feature_names:
                features = {k: v for k, v in features.items() if k in feature_names}

            return features

        except Exception as e:
            logger.error(f"Delta query failed: {e}")
            return None

    def warm_cache(
        self,
        customer_ids: Optional[List[str]] = None,
        top_n: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Warm cache by loading features from Delta Lake.

        Args:
            customer_ids: Optional list of customer IDs to warm
            top_n: Optional number of top customers to warm (by frequency)

        Returns:
            Stats: {cached: count, failed: count}
        """
        if not self.spark:
            raise ValueError("Spark session required for cache warming")

        logger.info("=" * 80)
        logger.info("Cache Warming")
        logger.info("=" * 80)

        feature_table = self.delta_config["feature_table"]

        # Determine which customers to cache
        if customer_ids:
            # Specific customers
            customer_list = ", ".join([f"'{c}'" for c in customer_ids])
            filter_clause = f"WHERE customer_id IN ({customer_list})"
        elif top_n:
            # Top N customers (would need frequency data)
            filter_clause = f"LIMIT {top_n}"
        else:
            # All customers (use with caution)
            filter_clause = ""

        # Load features from Delta
        query = f"""
        SELECT *
        FROM {feature_table}
        {filter_clause}
        """

        logger.info(f"Loading features from: {feature_table}")
        features_df = self.spark.sql(query)

        total_count = features_df.count()
        logger.info(f"Loaded {total_count:,} customer feature sets")

        # Convert to dict and cache
        cached_count = 0
        failed_count = 0

        for row in features_df.collect():
            customer_id = row["customer_id"]
            features = row.asDict()

            # Remove metadata
            features.pop("customer_id")
            features.pop("as_of_date", None)
            features.pop("created_timestamp", None)

            # Cache features
            success = self._set_in_cache(customer_id, features)

            if success:
                cached_count += 1
            else:
                failed_count += 1

            # Log progress every 10k
            if (cached_count + failed_count) % 10000 == 0:
                logger.info(f"  Progress: {cached_count + failed_count:,} / {total_count:,}")

        logger.info("=" * 80)
        logger.info(f"Cache Warming Complete")
        logger.info(f"  Cached: {cached_count:,}")
        logger.info(f"  Failed: {failed_count:,}")
        logger.info("=" * 80)

        return {
            "cached": cached_count,
            "failed": failed_count
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Stats: cache size, hit rate, etc.
        """
        try:
            info = self.redis_client.info("stats")

            return {
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(info),
                "total_keys": self.redis_client.dbsize()
            }

        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}

    def _calculate_hit_rate(self, info: Dict[str, Any]) -> float:
        """Calculate cache hit rate."""
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)

        total = hits + misses
        if total == 0:
            return 0.0

        return hits / total

    def _make_key(self, customer_id: str) -> str:
        """Make Redis key for customer features."""
        return f"features:{customer_id}"

    def clear_cache(self, customer_id: Optional[str] = None):
        """
        Clear cache (all or specific customer).

        Args:
            customer_id: Optional customer ID to clear (None = clear all)
        """
        if customer_id:
            key = self._make_key(customer_id)
            self.redis_client.delete(key)
            logger.info(f"Cleared cache for customer: {customer_id}")
        else:
            self.redis_client.flushdb()
            logger.warning("Cleared entire cache!")


# Example usage
if __name__ == "__main__":
    from pyspark.sql import SparkSession

    # Configuration
    redis_config = {
        "host": "localhost",
        "port": 6379,
        "db": 0,
        "ttl_seconds": 86400  # 24 hours
    }

    delta_config = {
        "feature_table": "decision_agent.customer_features_latest"
    }

    # Initialize Spark
    spark = SparkSession.builder.appName("OnlineStoreTest").getOrCreate()

    # Initialize online store
    store = OnlineFeatureStore(redis_config, delta_config, spark)

    # Warm cache with top 1000 customers
    stats = store.warm_cache(top_n=1000)
    print(f"Cache warming stats: {stats}")

    # Get features for a customer
    features = store.get_features("CUST12345")
    print(f"Features: {features}")

    # Get cache stats
    cache_stats = store.get_cache_stats()
    print(f"Cache stats: {cache_stats}")
