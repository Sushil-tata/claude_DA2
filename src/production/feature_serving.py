"""
Real-time Feature Serving Framework

This module provides low-latency feature serving for production ML models
with caching, batch precomputation, online/offline feature handling,
and feature versioning support.

Usage Example:
    >>> from feature_serving import FeatureServer
    >>> server = FeatureServer(
    ...     feature_config=feature_config,
    ...     cache_backend="redis"
    ... )
    >>> # Get features for prediction
    >>> features = await server.get_features(
    ...     entity_id="customer_12345",
    ...     feature_names=["age", "income", "credit_score"]
    ... )
    >>> # Precompute batch features
    >>> server.precompute_batch_features(customer_ids)
"""

from typing import Dict, List, Optional, Union, Any, Callable, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
from collections import OrderedDict
import asyncio
import json
import pickle
import time
import hashlib
from functools import lru_cache

import numpy as np
import pandas as pd
from loguru import logger

# Optional dependencies
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available. Install redis for distributed caching.")


class FeatureType(str, Enum):
    """Feature computation type."""
    ONLINE = "online"  # Computed on-demand
    OFFLINE = "offline"  # Precomputed batch
    REALTIME = "realtime"  # Real-time aggregation
    HYBRID = "hybrid"  # Combination


class CacheStrategy(str, Enum):
    """Caching strategy."""
    LRU = "lru"  # Least Recently Used
    TTL = "ttl"  # Time To Live
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"


@dataclass
class FeatureDefinition:
    """Definition of a feature."""
    name: str
    feature_type: FeatureType
    computation_fn: Optional[Callable] = None
    dependencies: List[str] = field(default_factory=list)
    cache_ttl_seconds: int = 3600  # 1 hour default
    version: str = "v1"
    description: str = ""
    data_type: str = "float"
    
    # For offline features
    table_name: Optional[str] = None
    column_name: Optional[str] = None
    
    # For realtime aggregation features
    aggregation_window: Optional[str] = None  # e.g., "1h", "24h", "7d"
    aggregation_function: Optional[str] = None  # sum, avg, count, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'feature_type': self.feature_type.value,
            'dependencies': self.dependencies,
            'cache_ttl_seconds': self.cache_ttl_seconds,
            'version': self.version,
            'description': self.description,
            'data_type': self.data_type,
            'table_name': self.table_name,
            'column_name': self.column_name,
            'aggregation_window': self.aggregation_window,
            'aggregation_function': self.aggregation_function
        }


@dataclass
class FeatureConfig:
    """Configuration for feature serving."""
    features: List[FeatureDefinition]
    entity_key: str = "entity_id"
    cache_strategy: CacheStrategy = CacheStrategy.LRU
    max_cache_size: int = 10000
    default_ttl_seconds: int = 3600
    enable_fallback: bool = True
    latency_target_ms: float = 10.0
    batch_size: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'features': [f.to_dict() for f in self.features],
            'entity_key': self.entity_key,
            'cache_strategy': self.cache_strategy.value,
            'max_cache_size': self.max_cache_size,
            'default_ttl_seconds': self.default_ttl_seconds,
            'enable_fallback': self.enable_fallback,
            'latency_target_ms': self.latency_target_ms,
            'batch_size': self.batch_size
        }


@dataclass
class FeatureMetrics:
    """Metrics for feature serving."""
    feature_name: str
    request_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    compute_count: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    error_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total * 100) if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'feature_name': self.feature_name,
            'request_count': self.request_count,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'compute_count': self.compute_count,
            'cache_hit_rate': self.cache_hit_rate,
            'avg_latency_ms': self.avg_latency_ms,
            'p95_latency_ms': self.p95_latency_ms,
            'p99_latency_ms': self.p99_latency_ms,
            'error_count': self.error_count,
            'last_updated': self.last_updated.isoformat()
        }


class LRUCache:
    """Simple LRU cache implementation."""
    
    def __init__(self, max_size: int = 10000):
        self.cache: OrderedDict = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
    
    def delete(self, key: str) -> None:
        """Delete key from cache."""
        if key in self.cache:
            del self.cache[key]
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0


class TTLCache:
    """Time-based cache with TTL support."""
    
    def __init__(self, default_ttl: int = 3600):
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.default_ttl = default_ttl
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key in self.cache:
            value, expiry = self.cache[key]
            if datetime.now() < expiry:
                self.hits += 1
                return value
            else:
                # Expired
                del self.cache[key]
        
        self.misses += 1
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with TTL."""
        ttl_seconds = ttl or self.default_ttl
        expiry = datetime.now() + timedelta(seconds=ttl_seconds)
        self.cache[key] = (value, expiry)
    
    def delete(self, key: str) -> None:
        """Delete key from cache."""
        if key in self.cache:
            del self.cache[key]
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def cleanup_expired(self) -> int:
        """Remove expired entries."""
        now = datetime.now()
        expired_keys = [
            key for key, (_, expiry) in self.cache.items()
            if now >= expiry
        ]
        for key in expired_keys:
            del self.cache[key]
        return len(expired_keys)
    
    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0


class FeatureServer:
    """
    Real-time feature serving system.
    
    Provides low-latency feature access with caching, batch precomputation,
    and support for online/offline/realtime features.
    
    Example:
        >>> config = FeatureConfig(
        ...     features=[
        ...         FeatureDefinition(
        ...             name="age",
        ...             feature_type=FeatureType.OFFLINE,
        ...             table_name="customers",
        ...             column_name="age"
        ...         ),
        ...         FeatureDefinition(
        ...             name="recent_transactions",
        ...             feature_type=FeatureType.REALTIME,
        ...             aggregation_window="24h",
        ...             aggregation_function="count"
        ...         )
        ...     ]
        ... )
        >>> server = FeatureServer(feature_config=config)
        >>> features = await server.get_features("customer_123", ["age", "recent_transactions"])
    """
    
    def __init__(
        self,
        feature_config: FeatureConfig,
        cache_backend: str = "memory",  # memory, redis
        redis_url: Optional[str] = None,
        offline_store: Optional[Any] = None,
        realtime_store: Optional[Any] = None,
        monitoring_enabled: bool = True
    ):
        self.config = feature_config
        self.cache_backend = cache_backend
        self.offline_store = offline_store
        self.realtime_store = realtime_store
        self.monitoring_enabled = monitoring_enabled
        
        # Build feature registry
        self.features: Dict[str, FeatureDefinition] = {
            f.name: f for f in feature_config.features
        }
        
        # Initialize cache
        if cache_backend == "redis":
            if not REDIS_AVAILABLE:
                raise RuntimeError("Redis not available")
            self.redis_url = redis_url or "redis://localhost:6379"
            self.redis_client = None  # Will be initialized async
            self.local_cache = None
        else:
            if config.cache_strategy == CacheStrategy.LRU:
                self.local_cache = LRUCache(max_size=config.max_cache_size)
            else:
                self.local_cache = TTLCache(default_ttl=config.default_ttl_seconds)
            self.redis_client = None
        
        # Metrics
        self.feature_metrics: Dict[str, FeatureMetrics] = {
            f.name: FeatureMetrics(feature_name=f.name)
            for f in feature_config.features
        }
        
        self.total_requests = 0
        self.total_latencies: List[float] = []
        
        logger.info(f"Initialized FeatureServer with {len(self.features)} features")
    
    async def initialize(self) -> None:
        """Initialize async components."""
        if self.cache_backend == "redis" and self.redis_client is None:
            self.redis_client = await redis.from_url(self.redis_url)
            logger.info(f"Connected to Redis at {self.redis_url}")
    
    async def close(self) -> None:
        """Close connections."""
        if self.redis_client:
            await self.redis_client.close()
    
    def _make_cache_key(self, entity_id: str, feature_name: str, version: str = "v1") -> str:
        """Generate cache key."""
        return f"feature:{entity_id}:{feature_name}:{version}"
    
    async def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get value from cache."""
        if self.redis_client:
            value = await self.redis_client.get(cache_key)
            if value:
                return pickle.loads(value)
        elif self.local_cache:
            return self.local_cache.get(cache_key)
        return None
    
    async def _set_in_cache(
        self,
        cache_key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> None:
        """Set value in cache."""
        if self.redis_client:
            ttl_seconds = ttl or self.config.default_ttl_seconds
            await self.redis_client.setex(
                cache_key,
                ttl_seconds,
                pickle.dumps(value)
            )
        elif self.local_cache:
            if isinstance(self.local_cache, TTLCache):
                self.local_cache.set(cache_key, value, ttl)
            else:
                self.local_cache.set(cache_key, value)
    
    async def get_feature(
        self,
        entity_id: str,
        feature_name: str,
        use_cache: bool = True
    ) -> Optional[Any]:
        """
        Get a single feature value.
        
        Args:
            entity_id: Entity identifier
            feature_name: Feature name
            use_cache: Whether to use cache
            
        Returns:
            Feature value
        """
        start_time = time.time()
        
        if feature_name not in self.features:
            logger.warning(f"Feature {feature_name} not found in registry")
            return None
        
        feature_def = self.features[feature_name]
        metrics = self.feature_metrics[feature_name]
        metrics.request_count += 1
        
        # Check cache
        cache_key = self._make_cache_key(entity_id, feature_name, feature_def.version)
        if use_cache:
            cached_value = await self._get_from_cache(cache_key)
            if cached_value is not None:
                metrics.cache_hits += 1
                latency = (time.time() - start_time) * 1000
                self._update_latency_metrics(metrics, latency)
                return cached_value
            metrics.cache_misses += 1
        
        # Compute feature
        try:
            value = await self._compute_feature(entity_id, feature_def)
            metrics.compute_count += 1
            
            # Cache result
            if use_cache and value is not None:
                await self._set_in_cache(
                    cache_key,
                    value,
                    feature_def.cache_ttl_seconds
                )
            
            latency = (time.time() - start_time) * 1000
            self._update_latency_metrics(metrics, latency)
            
            return value
            
        except Exception as e:
            logger.error(f"Error computing feature {feature_name}: {str(e)}")
            metrics.error_count += 1
            
            # Fallback
            if self.config.enable_fallback:
                return await self._get_fallback_value(entity_id, feature_def)
            
            return None
    
    async def get_features(
        self,
        entity_id: str,
        feature_names: List[str],
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Get multiple feature values.
        
        Args:
            entity_id: Entity identifier
            feature_names: List of feature names
            use_cache: Whether to use cache
            
        Returns:
            Dictionary of feature values
        """
        start_time = time.time()
        
        # Fetch features concurrently
        tasks = [
            self.get_feature(entity_id, name, use_cache)
            for name in feature_names
        ]
        values = await asyncio.gather(*tasks)
        
        result = {
            name: value
            for name, value in zip(feature_names, values)
            if value is not None
        }
        
        # Track overall latency
        latency = (time.time() - start_time) * 1000
        self.total_latencies.append(latency)
        self.total_requests += 1
        
        if latency > self.config.latency_target_ms:
            logger.warning(f"High latency: {latency:.2f}ms > {self.config.latency_target_ms}ms")
        
        return result
    
    async def get_features_batch(
        self,
        entity_ids: List[str],
        feature_names: List[str],
        use_cache: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get features for multiple entities in batch.
        
        Args:
            entity_ids: List of entity identifiers
            feature_names: List of feature names
            use_cache: Whether to use cache
            
        Returns:
            Nested dictionary: {entity_id: {feature_name: value}}
        """
        # Process in batches to avoid overwhelming the system
        batch_size = self.config.batch_size
        results = {}
        
        for i in range(0, len(entity_ids), batch_size):
            batch_ids = entity_ids[i:i+batch_size]
            
            # Fetch features for batch concurrently
            tasks = [
                self.get_features(entity_id, feature_names, use_cache)
                for entity_id in batch_ids
            ]
            batch_results = await asyncio.gather(*tasks)
            
            # Combine results
            for entity_id, features in zip(batch_ids, batch_results):
                results[entity_id] = features
        
        return results
    
    async def _compute_feature(
        self,
        entity_id: str,
        feature_def: FeatureDefinition
    ) -> Any:
        """Compute feature value based on type."""
        if feature_def.feature_type == FeatureType.OFFLINE:
            return await self._get_offline_feature(entity_id, feature_def)
        
        elif feature_def.feature_type == FeatureType.ONLINE:
            return await self._compute_online_feature(entity_id, feature_def)
        
        elif feature_def.feature_type == FeatureType.REALTIME:
            return await self._compute_realtime_feature(entity_id, feature_def)
        
        elif feature_def.feature_type == FeatureType.HYBRID:
            # Try offline first, fallback to online
            value = await self._get_offline_feature(entity_id, feature_def)
            if value is None:
                value = await self._compute_online_feature(entity_id, feature_def)
            return value
        
        else:
            raise ValueError(f"Unknown feature type: {feature_def.feature_type}")
    
    async def _get_offline_feature(
        self,
        entity_id: str,
        feature_def: FeatureDefinition
    ) -> Any:
        """Get precomputed offline feature."""
        if self.offline_store is None:
            logger.warning("Offline store not configured")
            return None
        
        try:
            # Query offline store
            # This is a placeholder - implement based on your offline store
            value = await self.offline_store.get(
                table=feature_def.table_name,
                entity_id=entity_id,
                column=feature_def.column_name
            )
            return value
        except Exception as e:
            logger.error(f"Error fetching offline feature: {str(e)}")
            return None
    
    async def _compute_online_feature(
        self,
        entity_id: str,
        feature_def: FeatureDefinition
    ) -> Any:
        """Compute feature on-demand."""
        if feature_def.computation_fn is None:
            logger.warning(f"No computation function for feature {feature_def.name}")
            return None
        
        try:
            # Get dependencies
            dep_values = {}
            for dep_name in feature_def.dependencies:
                dep_values[dep_name] = await self.get_feature(entity_id, dep_name)
            
            # Compute feature
            if asyncio.iscoroutinefunction(feature_def.computation_fn):
                value = await feature_def.computation_fn(entity_id, **dep_values)
            else:
                value = feature_def.computation_fn(entity_id, **dep_values)
            
            return value
        except Exception as e:
            logger.error(f"Error computing online feature: {str(e)}")
            return None
    
    async def _compute_realtime_feature(
        self,
        entity_id: str,
        feature_def: FeatureDefinition
    ) -> Any:
        """Compute real-time aggregation feature."""
        if self.realtime_store is None:
            logger.warning("Realtime store not configured")
            return None
        
        try:
            # Query realtime store for aggregation
            # This is a placeholder - implement based on your realtime store
            value = await self.realtime_store.aggregate(
                entity_id=entity_id,
                window=feature_def.aggregation_window,
                function=feature_def.aggregation_function
            )
            return value
        except Exception as e:
            logger.error(f"Error computing realtime feature: {str(e)}")
            return None
    
    async def _get_fallback_value(
        self,
        entity_id: str,
        feature_def: FeatureDefinition
    ) -> Any:
        """Get fallback value for missing feature."""
        # Return default value based on data type
        if feature_def.data_type == "float":
            return 0.0
        elif feature_def.data_type == "int":
            return 0
        elif feature_def.data_type == "bool":
            return False
        elif feature_def.data_type == "str":
            return ""
        else:
            return None
    
    def _update_latency_metrics(self, metrics: FeatureMetrics, latency: float) -> None:
        """Update latency metrics."""
        # Simple running average (for production, use proper percentile tracking)
        if metrics.avg_latency_ms == 0:
            metrics.avg_latency_ms = latency
        else:
            alpha = 0.1  # Exponential moving average factor
            metrics.avg_latency_ms = alpha * latency + (1 - alpha) * metrics.avg_latency_ms
        
        metrics.last_updated = datetime.now()
    
    async def precompute_batch_features(
        self,
        entity_ids: List[str],
        feature_names: Optional[List[str]] = None
    ) -> None:
        """
        Precompute and cache features for a batch of entities.
        
        Args:
            entity_ids: List of entity identifiers
            feature_names: List of feature names (defaults to all)
        """
        if feature_names is None:
            feature_names = list(self.features.keys())
        
        logger.info(f"Precomputing {len(feature_names)} features for {len(entity_ids)} entities")
        
        # Fetch and cache
        await self.get_features_batch(entity_ids, feature_names, use_cache=False)
        
        logger.info("Batch precomputation completed")
    
    def invalidate_cache(
        self,
        entity_id: Optional[str] = None,
        feature_name: Optional[str] = None
    ) -> None:
        """
        Invalidate cache entries.
        
        Args:
            entity_id: Entity ID to invalidate (None = all)
            feature_name: Feature name to invalidate (None = all)
        """
        if self.local_cache:
            if entity_id is None and feature_name is None:
                self.local_cache.clear()
                logger.info("Cleared entire cache")
            else:
                # Selective invalidation
                # This is simplified - in production, maintain cache key index
                logger.info(f"Invalidated cache for entity={entity_id}, feature={feature_name}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get feature serving metrics."""
        overall_metrics = {
            'total_requests': self.total_requests,
            'avg_latency_ms': np.mean(self.total_latencies) if self.total_latencies else 0.0,
            'p95_latency_ms': np.percentile(self.total_latencies, 95) if self.total_latencies else 0.0,
            'p99_latency_ms': np.percentile(self.total_latencies, 99) if self.total_latencies else 0.0,
            'feature_metrics': {
                name: metrics.to_dict()
                for name, metrics in self.feature_metrics.items()
            }
        }
        
        # Cache metrics
        if self.local_cache:
            overall_metrics['cache_hit_rate'] = self.local_cache.hit_rate
            overall_metrics['cache_size'] = len(self.local_cache.cache)
        
        return overall_metrics
    
    def get_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        metrics_lines = []
        
        # Overall metrics
        if self.total_latencies:
            avg_latency = np.mean(self.total_latencies)
            p95_latency = np.percentile(self.total_latencies, 95)
            p99_latency = np.percentile(self.total_latencies, 99)
            
            metrics_lines.append(
                f'# HELP feature_serving_requests_total Total number of feature requests\n'
                f'# TYPE feature_serving_requests_total counter\n'
                f'feature_serving_requests_total {self.total_requests}\n'
            )
            
            metrics_lines.append(
                f'# HELP feature_serving_latency_ms Feature serving latency in milliseconds\n'
                f'# TYPE feature_serving_latency_ms gauge\n'
                f'feature_serving_latency_ms_avg {avg_latency:.2f}\n'
                f'feature_serving_latency_ms_p95 {p95_latency:.2f}\n'
                f'feature_serving_latency_ms_p99 {p99_latency:.2f}\n'
            )
        
        # Per-feature metrics
        for name, metrics in self.feature_metrics.items():
            metrics_lines.append(
                f'# HELP feature_requests_total Total requests for feature {name}\n'
                f'# TYPE feature_requests_total counter\n'
                f'feature_requests_total{{feature="{name}"}} {metrics.request_count}\n'
            )
            
            metrics_lines.append(
                f'# HELP feature_cache_hit_rate Cache hit rate for feature {name}\n'
                f'# TYPE feature_cache_hit_rate gauge\n'
                f'feature_cache_hit_rate{{feature="{name}"}} {metrics.cache_hit_rate:.2f}\n'
            )
            
            metrics_lines.append(
                f'# HELP feature_errors_total Error count for feature {name}\n'
                f'# TYPE feature_errors_total counter\n'
                f'feature_errors_total{{feature="{name}"}} {metrics.error_count}\n'
            )
        
        # Cache metrics
        if self.local_cache:
            metrics_lines.append(
                f'# HELP feature_cache_hit_rate_overall Overall cache hit rate\n'
                f'# TYPE feature_cache_hit_rate_overall gauge\n'
                f'feature_cache_hit_rate_overall {self.local_cache.hit_rate:.2f}\n'
            )
        
        return ''.join(metrics_lines)
    
    def check_freshness(self, feature_name: str) -> Dict[str, Any]:
        """Check feature freshness."""
        if feature_name not in self.feature_metrics:
            return {'fresh': False, 'reason': 'Feature not found'}
        
        metrics = self.feature_metrics[feature_name]
        time_since_update = (datetime.now() - metrics.last_updated).total_seconds()
        feature_def = self.features[feature_name]
        
        is_fresh = time_since_update < feature_def.cache_ttl_seconds
        
        return {
            'fresh': is_fresh,
            'last_updated': metrics.last_updated.isoformat(),
            'age_seconds': time_since_update,
            'ttl_seconds': feature_def.cache_ttl_seconds
        }


# Example feature computation functions

def compute_age_from_birthdate(entity_id: str, birthdate: str) -> int:
    """Example: Compute age from birthdate."""
    from datetime import datetime
    birth = datetime.fromisoformat(birthdate)
    age = (datetime.now() - birth).days // 365
    return age


def compute_credit_utilization(entity_id: str, credit_used: float, credit_limit: float) -> float:
    """Example: Compute credit utilization ratio."""
    if credit_limit == 0:
        return 0.0
    return (credit_used / credit_limit) * 100


async def compute_transaction_velocity(entity_id: str, recent_count: int, window_hours: int = 24) -> float:
    """Example: Compute transaction velocity."""
    return recent_count / window_hours
