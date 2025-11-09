"""
Test suite for telemetry modules

Tests metrics collection, profiling, and performance monitoring.
"""

import pytest
import time
from pulseos.telemetry.metrics import MetricsCollector, MetricPoint
from pulseos.telemetry.profiler import PerformanceProfiler, ProfileEntry


class TestMetricsCollector:
    """Tests for Metrics Collector"""
    
    def test_initialization(self):
        """Test metrics collector initialization"""
        collector = MetricsCollector(max_retention=1000)
        
        assert collector.max_retention == 1000
        assert collector.total_steps == 0
        assert "survival_signal" in collector.metrics
    
    def test_record_step(self):
        """Test recording step metrics"""
        collector = MetricsCollector()
        
        collector.record_step(
            step=1,
            duration=0.001,
            survival_signal=0.8,
            alpha=0.01,
            epsilon=0.1,
            gradient=0.25,
            agent_count=10
        )
        
        assert collector.total_steps == 1
        assert len(collector.metrics["survival_signal"]) == 1
    
    def test_get_metric(self):
        """Test getting metric data"""
        collector = MetricsCollector()
        
        collector.record_step(1, 0.001, 0.8, 0.01, 0.1, 0.25, 10)
        
        metric_data = collector.get_metric("survival_signal")
        assert len(metric_data) == 1
        assert metric_data[0].value == 0.8
    
    def test_get_statistics(self):
        """Test getting aggregated statistics"""
        collector = MetricsCollector()
        
        # Record multiple steps
        for i in range(5):
            collector.record_step(
                step=i,
                duration=0.001 * (i + 1),
                survival_signal=0.7 + i * 0.05,
                alpha=0.01,
                epsilon=0.1,
                gradient=0.25,
                agent_count=10
            )
        
        stats = collector.get_statistics()
        
        assert stats["total_steps"] == 5
        assert "survival_signal_mean" in stats
        assert "survival_signal_min" in stats
        assert "survival_signal_max" in stats
    
    def test_export_prometheus(self):
        """Test Prometheus export"""
        collector = MetricsCollector()
        
        collector.record_step(1, 0.001, 0.8, 0.01, 0.1, 0.25, 10)
        
        prometheus_output = collector.export_prometheus()
        
        assert "pulseos_survival_signal" in prometheus_output
        assert "0.8" in prometheus_output
    
    def test_export_json(self):
        """Test JSON export"""
        collector = MetricsCollector()
        
        collector.record_step(1, 0.001, 0.8, 0.01, 0.1, 0.25, 10)
        
        json_output = collector.export_json()
        
        import json
        data = json.loads(json_output)
        
        assert "metrics" in data
        assert "statistics" in data
        assert "survival_signal" in data["metrics"]
    
    def test_retention_limit(self):
        """Test retention limit"""
        collector = MetricsCollector(max_retention=5)
        
        # Record more than retention limit
        for i in range(10):
            collector.record_step(i, 0.001, 0.8, 0.01, 0.1, 0.25, 10)
        
        # Should be limited to max_retention
        assert len(collector.metrics["survival_signal"]) <= 5
    
    def test_clear(self):
        """Test clearing metrics"""
        collector = MetricsCollector()
        
        collector.record_step(1, 0.001, 0.8, 0.01, 0.1, 0.25, 10)
        assert collector.total_steps == 1
        
        collector.clear()
        
        assert collector.total_steps == 0
        assert len(collector.metrics["survival_signal"]) == 0


class TestPerformanceProfiler:
    """Tests for Performance Profiler"""
    
    def test_initialization_enabled(self):
        """Test profiler initialization enabled"""
        profiler = PerformanceProfiler(enabled=True)
        
        assert profiler.enabled is True
        assert profiler.profiler is not None
    
    def test_initialization_disabled(self):
        """Test profiler initialization disabled"""
        profiler = PerformanceProfiler(enabled=False)
        
        assert profiler.enabled is False
        assert profiler.profiler is None
    
    def test_start_and_stop(self):
        """Test profiler start and stop"""
        profiler = PerformanceProfiler(enabled=True)
        
        profiler.start()
        # Profiler should be running
        profiler.stop()
        # Profiler should be stopped
    
    def test_timer_operations(self):
        """Test timer start and stop"""
        profiler = PerformanceProfiler()
        
        profiler.start_timer("test_operation")
        time.sleep(0.01)  # Small delay
        duration = profiler.stop_timer("test_operation")
        
        assert duration > 0
        assert "test_operation" in profiler.timings
        assert len(profiler.timings["test_operation"]) == 1
    
    def test_multiple_timings(self):
        """Test multiple timing operations"""
        profiler = PerformanceProfiler()
        
        for i in range(3):
            profiler.start_timer(f"op_{i}")
            time.sleep(0.001)
            profiler.stop_timer(f"op_{i}")
        
        assert len(profiler.timings) == 3
    
    def test_get_timing_statistics(self):
        """Test getting timing statistics"""
        profiler = PerformanceProfiler()
        
        # Record multiple timings
        for _ in range(5):
            profiler.start_timer("test_op")
            time.sleep(0.001)
            profiler.stop_timer("test_op")
        
        stats = profiler.get_timing_statistics()
        
        assert "test_op" in stats
        assert "mean" in stats["test_op"]
        assert "min" in stats["test_op"]
        assert "max" in stats["test_op"]
        assert stats["test_op"]["count"] == 5
    
    def test_get_profile_stats(self):
        """Test getting profile statistics"""
        profiler = PerformanceProfiler(enabled=True)
        
        profiler.start()
        
        # Execute some operations
        def test_function():
            time.sleep(0.001)
            return sum(range(100))
        
        for _ in range(10):
            test_function()
        
        profiler.stop()
        
        stats = profiler.get_profile_stats(top_n=10)
        
        assert isinstance(stats, list)
        # Should have some profile entries
    
    def test_get_bottlenecks(self):
        """Test bottleneck identification"""
        profiler = PerformanceProfiler(enabled=True)
        
        profiler.start()
        
        def slow_function():
            time.sleep(0.01)
        
        for _ in range(5):
            slow_function()
        
        profiler.stop()
        
        bottlenecks = profiler.get_bottlenecks(threshold_percent=1.0)
        
        assert isinstance(bottlenecks, list)
    
    def test_reset(self):
        """Test profiler reset"""
        profiler = PerformanceProfiler()
        
        profiler.start_timer("test")
        profiler.stop_timer("test")
        
        assert len(profiler.timings) > 0
        
        profiler.reset()
        
        assert len(profiler.timings) == 0
        assert len(profiler.active_timers) == 0
    
    def test_stop_timer_nonexistent(self):
        """Test stopping non-existent timer"""
        profiler = PerformanceProfiler()
        
        duration = profiler.stop_timer("nonexistent")
        
        assert duration == 0.0

