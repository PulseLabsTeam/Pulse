"""
Optimization Layer

Memory pools, vectorization utilities, and performance optimizations.
"""

import numpy as np
from typing import Dict, List, Optional, Any, TypeVar, Generic
from dataclasses import dataclass
from collections import deque
import threading


T = TypeVar('T')


class MemoryPool(Generic[T]):
    """
    Memory pool for efficient object allocation/deallocation.
    
    Reduces memory allocation overhead by reusing objects.
    """
    
    def __init__(self, factory: callable, initial_size: int = 100, max_size: int = 1000):
        """
        Initialize memory pool.
        
        Args:
            factory: Factory function to create new objects
            initial_size: Initial pool size
            max_size: Maximum pool size
        """
        self.factory = factory
        self.max_size = max_size
        self.pool: deque = deque()
        self.lock = threading.Lock()
        
        # Pre-allocate initial objects
        for _ in range(initial_size):
            self.pool.append(self.factory())
    
    def acquire(self) -> T:
        """
        Acquire object from pool.
        
        Returns:
            Object from pool or newly created
        """
        with self.lock:
            if self.pool:
                return self.pool.popleft()
            else:
                return self.factory()
    
    def release(self, obj: T) -> None:
        """
        Release object back to pool.
        
        Args:
            obj: Object to release
        """
        with self.lock:
            if len(self.pool) < self.max_size:
                self.pool.append(obj)


class VectorizationUtils:
    """
    Vectorization utilities for numpy operations.
    
    Provides optimized vectorized operations for common patterns.
    """
    
    @staticmethod
    def vectorized_threshold_comparison(
        metrics: np.ndarray,
        thresholds: np.ndarray
    ) -> np.ndarray:
        """
        Vectorized threshold comparison.
        
        Args:
            metrics: Array of metric values
            thresholds: Array of threshold values
            
        Returns:
            Boolean array indicating which metrics meet thresholds
        """
        return metrics >= thresholds
    
    @staticmethod
    def vectorized_normalization(
        metrics: np.ndarray,
        baselines: np.ndarray
    ) -> np.ndarray:
        """
        Vectorized normalization: M_norm = M / M_baseline
        
        Args:
            metrics: Array of metric values
            baselines: Array of baseline values
            
        Returns:
            Normalized metrics array
        """
        # Avoid division by zero
        baselines_safe = np.where(baselines == 0, 1.0, baselines)
        return metrics / baselines_safe
    
    @staticmethod
    def vectorized_sigmoid(deltas: np.ndarray, beta: float = 1.0) -> np.ndarray:
        """
        Vectorized sigmoid computation.
        
        Args:
            deltas: Array of delta values
            beta: Beta parameter
            
        Returns:
            Sigmoid values array
        """
        exponent = -beta * deltas
        exponent = np.clip(exponent, -500, 500)
        return 1.0 / (1.0 + np.exp(exponent))
    
    @staticmethod
    def vectorized_gradient(
        deltas: np.ndarray,
        beta: float = 1.0
    ) -> np.ndarray:
        """
        Vectorized gradient computation.
        
        Args:
            deltas: Array of delta values
            beta: Beta parameter
            
        Returns:
            Gradient values array
        """
        sigmoid = VectorizationUtils.vectorized_sigmoid(deltas, beta)
        return beta * sigmoid * (1.0 - sigmoid)
    
    @staticmethod
    def batch_update(
        values: np.ndarray,
        updates: np.ndarray,
        max_change: Optional[float] = None
    ) -> np.ndarray:
        """
        Batch update with rate limiting.
        
        Args:
            values: Current values array
            updates: Update values array
            max_change: Maximum change per update (None for no limit)
            
        Returns:
            Updated values array
        """
        if max_change is None:
            return values + updates
        
        # Apply rate limiting
        changes = updates
        max_allowed = np.abs(values * max_change)
        
        # Clamp changes
        change_magnitudes = np.abs(changes)
        scale_factors = np.where(
            change_magnitudes > max_allowed,
            max_allowed / change_magnitudes,
            1.0
        )
        
        limited_changes = changes * scale_factors
        return values + limited_changes


class CacheOptimizer:
    """
    Cache optimization utilities.
    """
    
    @staticmethod
    def compute_cache_efficiency(
        hits: int,
        misses: int
    ) -> float:
        """
        Compute cache efficiency (hit rate).
        
        Args:
            hits: Number of cache hits
            misses: Number of cache misses
            
        Returns:
            Cache hit rate (0-1)
        """
        total = hits + misses
        if total == 0:
            return 0.0
        return hits / total
    
    @staticmethod
    def estimate_memory_usage(
        cache_size: int,
        entry_size_bytes: int
    ) -> int:
        """
        Estimate memory usage for cache.
        
        Args:
            cache_size: Number of cache entries
            entry_size_bytes: Size per entry in bytes
            
        Returns:
            Estimated memory usage in bytes
        """
        return cache_size * entry_size_bytes


class ZeroCopyBuffer:
    """
    Zero-copy buffer for efficient message passing.
    """
    
    def __init__(self, size: int):
        """
        Initialize zero-copy buffer.
        
        Args:
            size: Buffer size in bytes
        """
        self.buffer = bytearray(size)
        self.size = size
        self.position = 0
    
    def write(self, data: bytes) -> int:
        """
        Write data to buffer.
        
        Args:
            data: Data to write
            
        Returns:
            Number of bytes written
        """
        available = self.size - self.position
        write_size = min(len(data), available)
        
        self.buffer[self.position:self.position + write_size] = data[:write_size]
        self.position += write_size
        
        return write_size
    
    def read(self, size: int) -> bytes:
        """
        Read data from buffer.
        
        Args:
            size: Number of bytes to read
            
        Returns:
            Data bytes
        """
        read_size = min(size, self.position)
        data = bytes(self.buffer[:read_size])
        self.position = 0  # Reset after read
        
        return data
    
    def reset(self) -> None:
        """Reset buffer position."""
        self.position = 0

