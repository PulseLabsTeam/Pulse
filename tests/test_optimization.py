"""
Test suite for optimization modules

Tests cache, gradient_cache, hardware emulation, and vectorization utilities.
"""

import pytest
import numpy as np
from pulseos.optimization.cache import (
    MemoryPool,
    VectorizationUtils,
    CacheOptimizer,
    ZeroCopyBuffer
)
from pulseos.optimization.gradient_cache import GradientCache, CacheMetrics
from pulseos.optimization.hardware import (
    HardwareEmulationLayer,
    HardwareProfile,
    HardwareMode
)


class TestMemoryPool:
    """Tests for Memory Pool"""
    
    def test_pool_initialization(self):
        """Test memory pool initialization"""
        def factory():
            return {"data": []}
        
        pool = MemoryPool(factory, initial_size=10)
        assert len(pool.pool) == 10
    
    def test_acquire_and_release(self):
        """Test acquiring and releasing objects"""
        def factory():
            return {"id": id({}), "data": []}
        
        pool = MemoryPool(factory, initial_size=5, max_size=10)
        
        # Acquire objects
        obj1 = pool.acquire()
        obj2 = pool.acquire()
        
        assert len(pool.pool) == 3  # 5 - 2
        
        # Release objects
        pool.release(obj1)
        pool.release(obj2)
        
        assert len(pool.pool) == 5
    
    def test_pool_exhaustion(self):
        """Test pool creates new objects when exhausted"""
        def factory():
            return {"id": id({})}
        
        pool = MemoryPool(factory, initial_size=2, max_size=5)
        
        # Acquire all initial objects
        objs = [pool.acquire() for _ in range(3)]
        
        assert len(objs) == 3
        assert len(pool.pool) == 0
    
    def test_max_size_limit(self):
        """Test max size limit"""
        def factory():
            return {"data": []}
        
        pool = MemoryPool(factory, initial_size=2, max_size=3)
        
        # Acquire and release more than max_size
        objs = [pool.acquire() for _ in range(5)]
        for obj in objs:
            pool.release(obj)
        
        # Pool should not exceed max_size
        assert len(pool.pool) <= 3


class TestVectorizationUtils:
    """Tests for Vectorization Utilities"""
    
    def test_vectorized_threshold_comparison(self):
        """Test vectorized threshold comparison"""
        metrics = np.array([0.5, 0.7, 0.9, 1.0])
        thresholds = np.array([0.6, 0.6, 0.8, 0.9])
        
        result = VectorizationUtils.vectorized_threshold_comparison(metrics, thresholds)
        
        expected = np.array([False, True, True, True])
        np.testing.assert_array_equal(result, expected)
    
    def test_vectorized_normalization(self):
        """Test vectorized normalization"""
        metrics = np.array([2.0, 4.0, 6.0])
        baselines = np.array([1.0, 2.0, 3.0])
        
        result = VectorizationUtils.vectorized_normalization(metrics, baselines)
        
        expected = np.array([2.0, 2.0, 2.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_vectorized_normalization_zero_baseline(self):
        """Test normalization handles zero baselines"""
        metrics = np.array([1.0, 2.0])
        baselines = np.array([0.0, 2.0])
        
        result = VectorizationUtils.vectorized_normalization(metrics, baselines)
        
        # Zero baseline should be replaced with 1.0
        assert result[0] == 1.0
        assert result[1] == 1.0
    
    def test_vectorized_sigmoid(self):
        """Test vectorized sigmoid computation"""
        deltas = np.array([0.0, 1.0, -1.0])
        result = VectorizationUtils.vectorized_sigmoid(deltas, beta=1.0)
        
        # At delta=0, sigmoid should be 0.5
        assert result[0] == pytest.approx(0.5, abs=0.01)
        
        # Positive delta -> sigmoid > 0.5
        assert result[1] > 0.5
        
        # Negative delta -> sigmoid < 0.5
        assert result[2] < 0.5
    
    def test_vectorized_gradient(self):
        """Test vectorized gradient computation"""
        deltas = np.array([0.0, 1.0, -1.0])
        result = VectorizationUtils.vectorized_gradient(deltas, beta=1.0)
        
        # At delta=0, gradient should be maximum (0.25 for beta=1.0)
        assert result[0] == pytest.approx(0.25, abs=0.01)
        
        # All gradients should be positive
        assert np.all(result >= 0)
    
    def test_batch_update(self):
        """Test batch update"""
        values = np.array([1.0, 2.0, 3.0])
        updates = np.array([0.1, 0.2, 0.3])
        
        result = VectorizationUtils.batch_update(values, updates)
        
        expected = np.array([1.1, 2.2, 3.3])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_batch_update_rate_limiting(self):
        """Test batch update with rate limiting"""
        values = np.array([1.0, 2.0])
        updates = np.array([0.5, 0.5])  # 50% change
        max_change = 0.1  # Limit to 10%
        
        result = VectorizationUtils.batch_update(values, updates, max_change=max_change)
        
        # Changes should be limited to 10%
        assert abs(result[0] - values[0]) <= values[0] * 0.1 + 0.01
        assert abs(result[1] - values[1]) <= values[1] * 0.1 + 0.01


class TestCacheOptimizer:
    """Tests for Cache Optimizer"""
    
    def test_compute_cache_efficiency(self):
        """Test cache efficiency calculation"""
        hits = 75
        misses = 25
        
        efficiency = CacheOptimizer.compute_cache_efficiency(hits, misses)
        
        assert efficiency == 0.75
    
    def test_compute_cache_efficiency_zero_total(self):
        """Test cache efficiency with zero total"""
        efficiency = CacheOptimizer.compute_cache_efficiency(0, 0)
        assert efficiency == 0.0
    
    def test_estimate_memory_usage(self):
        """Test memory usage estimation"""
        cache_size = 256
        entry_size_bytes = 64
        
        memory_bytes = CacheOptimizer.estimate_memory_usage(cache_size, entry_size_bytes)
        
        assert memory_bytes == 256 * 64


class TestZeroCopyBuffer:
    """Tests for Zero-Copy Buffer"""
    
    def test_buffer_initialization(self):
        """Test buffer initialization"""
        buffer = ZeroCopyBuffer(size=1024)
        assert buffer.size == 1024
        assert buffer.position == 0
    
    def test_write_and_read(self):
        """Test write and read operations"""
        buffer = ZeroCopyBuffer(size=100)
        
        data = b"test data"
        written = buffer.write(data)
        
        assert written == len(data)
        assert buffer.position == len(data)
        
        read_data = buffer.read(len(data))
        assert read_data == data
        assert buffer.position == 0  # Reset after read
    
    def test_write_overflow(self):
        """Test write overflow handling"""
        buffer = ZeroCopyBuffer(size=10)
        
        large_data = b"x" * 20
        written = buffer.write(large_data)
        
        assert written == 10  # Limited by buffer size
        assert buffer.position == 10
    
    def test_read_limits(self):
        """Test read respects buffer limits"""
        buffer = ZeroCopyBuffer(size=100)
        
        buffer.write(b"test")
        read_data = buffer.read(100)  # Request more than available
        
        assert len(read_data) == 4  # Only available data
    
    def test_reset(self):
        """Test buffer reset"""
        buffer = ZeroCopyBuffer(size=100)
        
        buffer.write(b"test")
        assert buffer.position > 0
        
        buffer.reset()
        assert buffer.position == 0


class TestGradientCache:
    """Tests for Gradient Cache"""
    
    def test_cache_initialization(self):
        """Test gradient cache initialization"""
        cache = GradientCache(size=256)
        assert cache.max_size == 256
        assert len(cache.cache) == 0
    
    def test_get_or_compute_cache_hit(self):
        """Test cache hit"""
        cache = GradientCache(size=10)
        
        def compute_fn(key):
            return key * 2
        
        # First call - cache miss
        result1 = cache.get_or_compute(0.5, compute_fn)
        assert result1 == 1.0
        assert cache.metrics.misses == 1
        
        # Second call - cache hit
        result2 = cache.get_or_compute(0.5, compute_fn)
        assert result2 == 1.0
        assert cache.metrics.hits == 1
        assert cache.metrics.misses == 1
    
    def test_cache_eviction(self):
        """Test LRU eviction"""
        cache = GradientCache(size=3)
        
        def compute_fn(key):
            return key
        
        # Fill cache
        for i in range(3):
            cache.get_or_compute(float(i), compute_fn)
        
        assert len(cache.cache) == 3
        assert cache.metrics.evictions == 0
        
        # Add one more - should evict oldest
        cache.get_or_compute(3.0, compute_fn)
        
        assert len(cache.cache) == 3
        assert cache.metrics.evictions == 1
    
    def test_key_quantization(self):
        """Test key quantization"""
        cache = GradientCache(size=10, quantization_precision=2)
        
        def compute_fn(key):
            return key
        
        # These should map to same quantized key
        cache.get_or_compute(0.123, compute_fn)
        cache.get_or_compute(0.124, compute_fn)
        
        # Should be cache hit due to quantization
        assert cache.metrics.hits == 1
    
    def test_get_without_compute(self):
        """Test get without compute function"""
        cache = GradientCache(size=10)
        
        # Put value
        cache.put(0.5, "value")
        
        # Get value
        result = cache.get(0.5)
        assert result == "value"
        
        # Get non-existent
        result = cache.get(1.0)
        assert result is None
    
    def test_cache_statistics(self):
        """Test cache statistics"""
        cache = GradientCache(size=10)
        
        def compute_fn(key):
            return key * 2
        
        cache.get_or_compute(0.5, compute_fn)
        cache.get_or_compute(0.5, compute_fn)  # Hit
        cache.get_or_compute(0.6, compute_fn)
        
        stats = cache.get_statistics()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["total_requests"] == 3
        assert stats["hit_rate"] == pytest.approx(1.0 / 3.0, abs=0.01)
    
    def test_meets_target_hit_rate(self):
        """Test target hit rate check"""
        cache = GradientCache(size=10)
        
        def compute_fn(key):
            return key
        
        # Generate high hit rate
        for _ in range(10):
            cache.get_or_compute(0.5, compute_fn)
        
        assert cache.meets_target_hit_rate(0.75) is True
    
    def test_clear_cache(self):
        """Test cache clearing"""
        cache = GradientCache(size=10)
        
        cache.get_or_compute(0.5, lambda k: k)
        assert len(cache.cache) > 0
        
        cache.clear()
        assert len(cache.cache) == 0
        assert cache.metrics.total_requests == 0


class TestHardwareEmulationLayer:
    """Tests for Hardware Emulation Layer"""
    
    def test_initialization(self):
        """Test hardware emulation initialization"""
        layer = HardwareEmulationLayer()
        assert layer.profile.mode == HardwareMode.SIMULATED_ASIC
        assert layer.operation_count == 0
    
    def test_custom_profile(self):
        """Test custom hardware profile"""
        profile = HardwareProfile(
            mode=HardwareMode.SIMULATED_GPU,
            parallel_units=128,
            clock_frequency_mhz=2000.0
        )
        layer = HardwareEmulationLayer(profile=profile)
        assert layer.profile.mode == HardwareMode.SIMULATED_GPU
        assert layer.profile.parallel_units == 128
    
    def test_parallel_threshold_comparison(self):
        """Test parallel threshold comparison"""
        layer = HardwareEmulationLayer()
        
        metrics = np.array([0.5, 0.7, 0.9])
        thresholds = np.array([0.6, 0.6, 0.8])
        
        result = layer.parallel_threshold_comparison(metrics, thresholds)
        
        expected = np.array([False, True, True])
        np.testing.assert_array_equal(result, expected)
        assert layer.operation_count == 1
    
    def test_vectorized_normalization(self):
        """Test vectorized normalization"""
        layer = HardwareEmulationLayer()
        
        metrics = np.array([2.0, 4.0])
        baselines = np.array([1.0, 2.0])
        
        result = layer.vectorized_normalization(metrics, baselines)
        
        expected = np.array([2.0, 2.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_batch_gradient_computation(self):
        """Test batch gradient computation"""
        layer = HardwareEmulationLayer()
        
        deltas = np.array([0.0, 1.0, -1.0])
        result = layer.batch_gradient_computation(deltas, beta=1.0)
        
        assert len(result) == 3
        assert result[0] == pytest.approx(0.25, abs=0.01)  # Max at delta=0
    
    def test_performance_stats(self):
        """Test performance statistics"""
        layer = HardwareEmulationLayer()
        
        metrics = np.array([0.5, 0.7, 0.9])
        thresholds = np.array([0.6, 0.6, 0.8])
        
        layer.parallel_threshold_comparison(metrics, thresholds)
        
        stats = layer.get_performance_stats()
        assert stats["operation_count"] == 1
        assert stats["average_latency_ns"] > 0
    
    def test_reset_stats(self):
        """Test statistics reset"""
        layer = HardwareEmulationLayer()
        
        metrics = np.array([0.5, 0.7])
        thresholds = np.array([0.6, 0.6])
        
        layer.parallel_threshold_comparison(metrics, thresholds)
        assert layer.operation_count == 1
        
        layer.reset_stats()
        assert layer.operation_count == 0
        assert layer.total_latency_ns == 0.0
    
    def test_different_hardware_modes(self):
        """Test different hardware modes"""
        for mode in HardwareMode:
            profile = HardwareProfile(mode=mode, parallel_units=64)
            layer = HardwareEmulationLayer(profile=profile)
            
            metrics = np.array([0.5, 0.7, 0.9])
            thresholds = np.array([0.6, 0.6, 0.8])
            
            result = layer.parallel_threshold_comparison(metrics, thresholds)
            assert len(result) == 3

