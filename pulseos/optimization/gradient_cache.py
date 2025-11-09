"""
Enhanced Gradient Cache with explicit metrics and patent compliance

Implements the patent-specified 256-entry circular buffer cache with
comprehensive hit rate tracking and memory efficiency metrics.
"""

from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass, field
from collections import OrderedDict
import time


@dataclass
class CacheMetrics:
    """Comprehensive cache performance metrics"""
    hits: int = 0
    misses: int = 0
    total_requests: int = 0
    evictions: int = 0
    memory_bytes: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0-1)"""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests
    
    @property
    def miss_rate(self) -> float:
        """Cache miss rate (0-1)"""
        return 1.0 - self.hit_rate
    
    @property
    def efficiency(self) -> float:
        """Cache efficiency metric"""
        if self.total_requests == 0:
            return 0.0
        # Efficiency = hit_rate * (1 - eviction_rate)
        eviction_rate = self.evictions / max(self.total_requests, 1)
        return self.hit_rate * (1 - min(eviction_rate, 1.0))


class GradientCache:
    """
    Patent-specified 256-entry circular buffer cache for gradient computation.
    
    Implements:
    - LRU eviction policy
    - Quantized key lookup (O(1) access)
    - Comprehensive hit rate tracking (target: 75%)
    - Memory efficiency metrics
    """
    
    def __init__(self, size: int = 256, quantization_precision: int = 6):
        """
        Initialize gradient cache.
        
        Args:
            size: Maximum cache size (default 256 per patent)
            quantization_precision: Decimal places for key quantization
        """
        self.max_size = size
        self.quantization_precision = quantization_precision
        self.cache: OrderedDict[float, Any] = OrderedDict()
        self.metrics = CacheMetrics()
        
        # Track memory usage (approximate)
        self.entry_size_bytes = 64  # Approximate: float + metadata
    
    def _quantize_key(self, key: float) -> float:
        """Quantize key for cache lookup."""
        return round(key, self.quantization_precision)
    
    def get_or_compute(
        self,
        key: float,
        compute_fn: Callable[[float], Any]
    ) -> Any:
        """
        Get value from cache or compute if miss.
        
        Implements patent-specified caching algorithm:
        1. Quantize key for O(1) lookup
        2. Check cache (LRU order)
        3. Compute on miss and update cache
        4. Evict oldest entry if cache full
        
        Args:
            key: Cache key (delta value)
            compute_fn: Function to compute value on cache miss
            
        Returns:
            Cached or computed value
        """
        self.metrics.total_requests += 1
        
        quantized_key = self._quantize_key(key)
        
        # Cache hit
        if quantized_key in self.cache:
            value = self.cache[quantized_key]
            # Move to end (most recently used)
            self.cache.move_to_end(quantized_key)
            self.metrics.hits += 1
            return value
        
        # Cache miss - compute value
        self.metrics.misses += 1
        value = compute_fn(key)
        
        # Evict if cache is full (LRU: remove oldest)
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)  # Remove oldest
            self.metrics.evictions += 1
        
        # Add to cache (most recently used)
        self.cache[quantized_key] = value
        
        # Update memory estimate
        self.metrics.memory_bytes = len(self.cache) * self.entry_size_bytes
        
        return value
    
    def get(self, key: float) -> Optional[Any]:
        """
        Get value from cache without computing.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if miss
        """
        quantized_key = self._quantize_key(key)
        
        if quantized_key in self.cache:
            value = self.cache[quantized_key]
            self.cache.move_to_end(quantized_key)
            return value
        
        return None
    
    def put(self, key: float, value: Any) -> None:
        """
        Put value into cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        quantized_key = self._quantize_key(key)
        
        # Evict if needed
        if quantized_key not in self.cache and len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
            self.metrics.evictions += 1
        
        self.cache[quantized_key] = value
        self.cache.move_to_end(quantized_key)
        self.metrics.memory_bytes = len(self.cache) * self.entry_size_bytes
    
    def clear(self) -> None:
        """Clear cache and reset metrics."""
        self.cache.clear()
        self.metrics = CacheMetrics()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.
        
        Returns:
            Dictionary with cache metrics
        """
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": self.metrics.hit_rate,
            "miss_rate": self.metrics.miss_rate,
            "hits": self.metrics.hits,
            "misses": self.metrics.misses,
            "total_requests": self.metrics.total_requests,
            "evictions": self.metrics.evictions,
            "efficiency": self.metrics.efficiency,
            "memory_bytes": self.metrics.memory_bytes,
            "memory_kb": self.metrics.memory_bytes / 1024,
            "utilization": len(self.cache) / self.max_size if self.max_size > 0 else 0.0
        }
    
    def meets_target_hit_rate(self, target: float = 0.75) -> bool:
        """
        Check if cache meets target hit rate.
        
        Args:
            target: Target hit rate (default 0.75 per patent)
            
        Returns:
            True if hit rate >= target
        """
        return self.metrics.hit_rate >= target

