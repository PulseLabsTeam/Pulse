"""
Comprehensive test suite for enhanced_metrics.py

Tests all functionality to achieve 90%+ coverage.
"""

import pytest
import time
import json
import numpy as np
from pulseos.telemetry.enhanced_metrics import (
    EnhancedMetricsCollector,
    GradientHistoryPoint,
    CacheMetricsPoint,
    ConvergencePoint
)


class TestGradientHistoryPoint:
    """Tests for GradientHistoryPoint dataclass"""
    
    def test_creation(self):
        """Test creating a gradient history point"""
        point = GradientHistoryPoint(
            timestamp=123.456,
            delta=0.5,
            gradient=0.25,
            sigmoid=0.7,
            cache_hit=True
        )
        
        assert point.timestamp == 123.456
        assert point.delta == 0.5
        assert point.gradient == 0.25
        assert point.sigmoid == 0.7
        assert point.cache_hit is True
    
    def test_cache_miss(self):
        """Test gradient history point with cache miss"""
        point = GradientHistoryPoint(
            timestamp=123.456,
            delta=0.5,
            gradient=0.25,
            sigmoid=0.7,
            cache_hit=False
        )
        
        assert point.cache_hit is False


class TestCacheMetricsPoint:
    """Tests for CacheMetricsPoint dataclass"""
    
    def test_creation(self):
        """Test creating a cache metrics point"""
        point = CacheMetricsPoint(
            timestamp=123.456,
            hit_rate=0.75,
            hits=100,
            misses=33,
            evictions=5,
            memory_bytes=1024
        )
        
        assert point.timestamp == 123.456
        assert point.hit_rate == 0.75
        assert point.hits == 100
        assert point.misses == 33
        assert point.evictions == 5
        assert point.memory_bytes == 1024
    
    def test_zero_metrics(self):
        """Test cache metrics with zero values"""
        point = CacheMetricsPoint(
            timestamp=0.0,
            hit_rate=0.0,
            hits=0,
            misses=0,
            evictions=0,
            memory_bytes=0
        )
        
        assert point.hit_rate == 0.0
        assert point.hits == 0


class TestConvergencePoint:
    """Tests for ConvergencePoint dataclass"""
    
    def test_creation(self):
        """Test creating a convergence point"""
        point = ConvergencePoint(
            timestamp=123.456,
            step=100,
            survival_signal=0.8,
            converged_agents=80,
            total_agents=100,
            convergence_rate=0.8
        )
        
        assert point.timestamp == 123.456
        assert point.step == 100
        assert point.survival_signal == 0.8
        assert point.converged_agents == 80
        assert point.total_agents == 100
        assert point.convergence_rate == 0.8
    
    def test_partial_convergence(self):
        """Test convergence point with partial convergence"""
        point = ConvergencePoint(
            timestamp=123.456,
            step=50,
            survival_signal=0.5,
            converged_agents=50,
            total_agents=100,
            convergence_rate=0.5
        )
        
        assert point.convergence_rate == 0.5


class TestEnhancedMetricsCollector:
    """Comprehensive tests for EnhancedMetricsCollector"""
    
    def test_initialization_default(self):
        """Test initialization with default parameters"""
        collector = EnhancedMetricsCollector()
        
        assert collector.max_history == 10000
        assert len(collector.gradient_history) == 0
        assert len(collector.cache_metrics_history) == 0
        assert len(collector.convergence_history) == 0
        assert collector.total_steps == 0
        assert collector.start_time > 0
    
    def test_initialization_custom_max(self):
        """Test initialization with custom max_history"""
        collector = EnhancedMetricsCollector(max_history=100)
        
        assert collector.max_history == 100
    
    def test_record_gradient(self):
        """Test recording gradient computation"""
        collector = EnhancedMetricsCollector()
        
        collector.record_gradient(
            delta=0.5,
            gradient=0.25,
            sigmoid=0.7,
            cache_hit=True
        )
        
        assert len(collector.gradient_history) == 1
        point = collector.gradient_history[0]
        assert point.delta == 0.5
        assert point.gradient == 0.25
        assert point.sigmoid == 0.7
        assert point.cache_hit is True
        assert point.timestamp > 0
    
    def test_record_gradient_cache_miss(self):
        """Test recording gradient with cache miss"""
        collector = EnhancedMetricsCollector()
        
        collector.record_gradient(
            delta=0.3,
            gradient=0.15,
            sigmoid=0.6,
            cache_hit=False
        )
        
        assert collector.gradient_history[0].cache_hit is False
    
    def test_record_gradient_multiple(self):
        """Test recording multiple gradient computations"""
        collector = EnhancedMetricsCollector()
        
        for i in range(10):
            collector.record_gradient(
                delta=0.1 * i,
                gradient=0.05 * i,
                sigmoid=0.5 + 0.05 * i,
                cache_hit=(i % 2 == 0)
            )
        
        assert len(collector.gradient_history) == 10
        assert collector.gradient_history[0].delta == 0.0
        assert collector.gradient_history[9].delta == 0.9
    
    def test_record_cache_metrics(self):
        """Test recording cache metrics"""
        collector = EnhancedMetricsCollector()
        
        collector.record_cache_metrics(
            hit_rate=0.75,
            hits=100,
            misses=33,
            evictions=5,
            memory_bytes=1024
        )
        
        assert len(collector.cache_metrics_history) == 1
        point = collector.cache_metrics_history[0]
        assert point.hit_rate == 0.75
        assert point.hits == 100
        assert point.misses == 33
        assert point.evictions == 5
        assert point.memory_bytes == 1024
    
    def test_record_cache_metrics_multiple(self):
        """Test recording multiple cache metric snapshots"""
        collector = EnhancedMetricsCollector()
        
        for i in range(5):
            collector.record_cache_metrics(
                hit_rate=0.5 + 0.1 * i,
                hits=50 + 10 * i,
                misses=50 - 10 * i,
                evictions=i,
                memory_bytes=512 + 128 * i
            )
        
        assert len(collector.cache_metrics_history) == 5
        assert collector.cache_metrics_history[0].hit_rate == 0.5
        assert collector.cache_metrics_history[4].hit_rate == 0.9
    
    def test_record_convergence(self):
        """Test recording convergence progress"""
        collector = EnhancedMetricsCollector()
        
        collector.record_convergence(
            step=100,
            survival_signal=0.8,
            converged_agents=80,
            total_agents=100
        )
        
        assert len(collector.convergence_history) == 1
        point = collector.convergence_history[0]
        assert point.step == 100
        assert point.survival_signal == 0.8
        assert point.converged_agents == 80
        assert point.total_agents == 100
        assert point.convergence_rate == 0.8
    
    def test_record_convergence_zero_agents(self):
        """Test recording convergence with zero agents"""
        collector = EnhancedMetricsCollector()
        
        collector.record_convergence(
            step=0,
            survival_signal=0.0,
            converged_agents=0,
            total_agents=0
        )
        
        point = collector.convergence_history[0]
        assert point.convergence_rate == 0.0
    
    def test_record_convergence_partial(self):
        """Test recording partial convergence"""
        collector = EnhancedMetricsCollector()
        
        collector.record_convergence(
            step=50,
            survival_signal=0.5,
            converged_agents=30,
            total_agents=100
        )
        
        point = collector.convergence_history[0]
        assert point.convergence_rate == 0.3
    
    def test_record_step(self):
        """Test recording step metrics"""
        collector = EnhancedMetricsCollector()
        
        collector.record_step(
            step=1,
            duration=0.001,
            survival_signal=0.8,
            alpha=0.01,
            epsilon=0.1
        )
        
        assert collector.total_steps == 1
        assert len(collector.step_durations) == 1
        assert len(collector.survival_signals) == 1
        assert len(collector.alpha_values) == 1
        assert len(collector.epsilon_values) == 1
        
        assert collector.step_durations[0] == 0.001
        assert collector.survival_signals[0] == 0.8
        assert collector.alpha_values[0] == 0.01
        assert collector.epsilon_values[0] == 0.1
    
    def test_record_step_multiple(self):
        """Test recording multiple steps"""
        collector = EnhancedMetricsCollector()
        
        for i in range(10):
            collector.record_step(
                step=i,
                duration=0.001 * (i + 1),
                survival_signal=0.5 + 0.05 * i,
                alpha=0.01 + 0.001 * i,
                epsilon=0.1 - 0.01 * i
            )
        
        assert collector.total_steps == 9  # Last step recorded
        assert len(collector.step_durations) == 10
        assert collector.step_durations[0] == 0.001
        assert collector.step_durations[9] == 0.01
    
    def test_get_gradient_statistics_empty(self):
        """Test getting gradient statistics with no data"""
        collector = EnhancedMetricsCollector()
        
        stats = collector.get_gradient_statistics()
        
        assert stats == {}
    
    def test_get_gradient_statistics_single(self):
        """Test getting gradient statistics with single point"""
        collector = EnhancedMetricsCollector()
        
        collector.record_gradient(0.5, 0.25, 0.7, True)
        
        stats = collector.get_gradient_statistics()
        
        assert stats["total_computations"] == 1
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 0
        assert stats["cache_hit_rate"] == 1.0
        assert stats["gradient_mean"] == 0.25
        assert stats["gradient_min"] == 0.25
        assert stats["gradient_max"] == 0.25
    
    def test_get_gradient_statistics_multiple(self):
        """Test getting gradient statistics with multiple points"""
        collector = EnhancedMetricsCollector()
        
        # Record mix of hits and misses
        for i in range(10):
            collector.record_gradient(
                delta=0.1 * i,
                gradient=0.05 * i,
                sigmoid=0.5,
                cache_hit=(i % 2 == 0)
            )
        
        stats = collector.get_gradient_statistics()
        
        assert stats["total_computations"] == 10
        assert stats["cache_hits"] == 5
        assert stats["cache_misses"] == 5
        assert stats["cache_hit_rate"] == 0.5
        assert stats["gradient_mean"] == pytest.approx(0.225, abs=0.001)
        assert stats["gradient_min"] == 0.0
        assert stats["gradient_max"] == 0.45
    
    def test_get_cache_statistics_empty(self):
        """Test getting cache statistics with no data"""
        collector = EnhancedMetricsCollector()
        
        stats = collector.get_cache_statistics()
        
        assert stats == {}
    
    def test_get_cache_statistics_single(self):
        """Test getting cache statistics with single point"""
        collector = EnhancedMetricsCollector()
        
        collector.record_cache_metrics(0.75, 100, 33, 5, 1024)
        
        stats = collector.get_cache_statistics()
        
        assert stats["average_hit_rate"] == 0.75
        assert stats["current_hit_rate"] == 0.75
        assert stats["hit_rate_std"] == 0.0
        assert stats["average_memory_bytes"] == 1024
        assert stats["peak_memory_bytes"] == 1024
    
    def test_get_cache_statistics_multiple(self):
        """Test getting cache statistics with multiple points"""
        collector = EnhancedMetricsCollector()
        
        hit_rates = [0.5, 0.6, 0.7, 0.8, 0.9]
        for i, rate in enumerate(hit_rates):
            collector.record_cache_metrics(
                hit_rate=rate,
                hits=50 + 10 * i,
                misses=50 - 10 * i,
                evictions=i,
                memory_bytes=512 + 128 * i
            )
        
        stats = collector.get_cache_statistics()
        
        assert stats["average_hit_rate"] == pytest.approx(0.7, abs=0.01)
        assert stats["current_hit_rate"] == 0.9
        assert stats["average_memory_bytes"] == pytest.approx(768, abs=1)
        assert stats["peak_memory_bytes"] == 1024
    
    def test_get_convergence_statistics_empty(self):
        """Test getting convergence statistics with no data"""
        collector = EnhancedMetricsCollector()
        
        stats = collector.get_convergence_statistics()
        
        assert stats == {}
    
    def test_get_convergence_statistics_single(self):
        """Test getting convergence statistics with single point"""
        collector = EnhancedMetricsCollector()
        
        collector.record_convergence(100, 0.8, 80, 100)
        
        stats = collector.get_convergence_statistics()
        
        assert stats["current_convergence_rate"] == 0.8
        assert stats["average_convergence_rate"] == 0.8
        assert stats["convergence_rate_std"] == 0.0
        assert stats["average_survival_signal"] == 0.8
        assert stats["convergence_points"] == 1
    
    def test_get_convergence_statistics_multiple(self):
        """Test getting convergence statistics with multiple points"""
        collector = EnhancedMetricsCollector()
        
        for i in range(10):
            collector.record_convergence(
                step=i * 10,
                survival_signal=0.5 + 0.05 * i,
                converged_agents=50 + 5 * i,
                total_agents=100
            )
        
        stats = collector.get_convergence_statistics()
        
        assert stats["current_convergence_rate"] == 0.95
        assert stats["average_convergence_rate"] == pytest.approx(0.725, abs=0.01)
        assert stats["convergence_points"] == 10
    
    def test_get_performance_statistics_empty(self):
        """Test getting performance statistics with no data"""
        collector = EnhancedMetricsCollector()
        
        stats = collector.get_performance_statistics()
        
        assert stats == {}
    
    def test_get_performance_statistics_single(self):
        """Test getting performance statistics with single step"""
        collector = EnhancedMetricsCollector()
        
        collector.record_step(1, 0.001, 0.8, 0.01, 0.1)
        
        stats = collector.get_performance_statistics()
        
        assert stats["total_steps"] == 1
        assert stats["uptime_seconds"] > 0
        assert stats["average_step_duration_ms"] == pytest.approx(1.0, abs=0.1)
        assert stats["min_step_duration_ms"] == pytest.approx(1.0, abs=0.1)
        assert stats["max_step_duration_ms"] == pytest.approx(1.0, abs=0.1)
        assert stats["average_survival_signal"] == 0.8
        assert stats["average_alpha"] == 0.01
        assert stats["average_epsilon"] == 0.1
    
    def test_get_performance_statistics_multiple(self):
        """Test getting performance statistics with multiple steps"""
        collector = EnhancedMetricsCollector()
        
        for i in range(10):
            collector.record_step(
                step=i,
                duration=0.001 * (i + 1),
                survival_signal=0.5 + 0.05 * i,
                alpha=0.01 + 0.001 * i,
                epsilon=0.1 - 0.01 * i
            )
        
        stats = collector.get_performance_statistics()
        
        assert stats["total_steps"] == 9
        assert stats["average_step_duration_ms"] == pytest.approx(5.5, abs=0.1)
        assert stats["min_step_duration_ms"] == pytest.approx(1.0, abs=0.1)
        assert stats["max_step_duration_ms"] == pytest.approx(10.0, abs=0.1)
        assert stats["average_survival_signal"] == pytest.approx(0.725, abs=0.01)
    
    def test_export_comprehensive_report(self):
        """Test exporting comprehensive report"""
        collector = EnhancedMetricsCollector()
        
        # Add some data
        collector.record_gradient(0.5, 0.25, 0.7, True)
        collector.record_cache_metrics(0.75, 100, 33, 5, 1024)
        collector.record_convergence(100, 0.8, 80, 100)
        collector.record_step(1, 0.001, 0.8, 0.01, 0.1)
        
        report_json = collector.export_comprehensive_report()
        
        assert isinstance(report_json, str)
        
        # Parse JSON
        report = json.loads(report_json)
        
        assert "timestamp" in report
        assert "gradient_statistics" in report
        assert "cache_statistics" in report
        assert "convergence_statistics" in report
        assert "performance_statistics" in report
        
        assert report["gradient_statistics"]["total_computations"] == 1
        assert report["cache_statistics"]["current_hit_rate"] == 0.75
        assert report["convergence_statistics"]["current_convergence_rate"] == 0.8
    
    def test_export_comprehensive_report_empty(self):
        """Test exporting report with no data"""
        collector = EnhancedMetricsCollector()
        
        report_json = collector.export_comprehensive_report()
        report = json.loads(report_json)
        
        assert "timestamp" in report
        assert report["gradient_statistics"] == {}
        assert report["cache_statistics"] == {}
        assert report["convergence_statistics"] == {}
        assert report["performance_statistics"] == {}
    
    def test_clear(self):
        """Test clearing all metrics"""
        collector = EnhancedMetricsCollector()
        
        # Add data
        collector.record_gradient(0.5, 0.25, 0.7, True)
        collector.record_cache_metrics(0.75, 100, 33, 5, 1024)
        collector.record_convergence(100, 0.8, 80, 100)
        collector.record_step(1, 0.001, 0.8, 0.01, 0.1)
        
        # Verify data exists
        assert len(collector.gradient_history) > 0
        assert len(collector.cache_metrics_history) > 0
        assert len(collector.convergence_history) > 0
        assert collector.total_steps > 0
        
        # Clear
        initial_start_time = collector.start_time
        time.sleep(0.01)  # Small delay to ensure time difference
        collector.clear()
        
        # Verify cleared
        assert len(collector.gradient_history) == 0
        assert len(collector.cache_metrics_history) == 0
        assert len(collector.convergence_history) == 0
        assert len(collector.step_durations) == 0
        assert len(collector.survival_signals) == 0
        assert len(collector.alpha_values) == 0
        assert len(collector.epsilon_values) == 0
        assert collector.total_steps == 0
        assert collector.start_time > initial_start_time
    
    def test_max_history_limit(self):
        """Test that max_history limit is enforced"""
        collector = EnhancedMetricsCollector(max_history=5)
        
        # Add more than max_history
        for i in range(10):
            collector.record_gradient(0.1 * i, 0.05 * i, 0.5, True)
        
        # Should be limited to max_history
        assert len(collector.gradient_history) == 5
        
        # Should contain the most recent entries
        assert collector.gradient_history[0].delta == 0.5
        assert collector.gradient_history[4].delta == 0.9
    
    def test_max_history_cache_metrics(self):
        """Test max_history limit for cache metrics"""
        collector = EnhancedMetricsCollector(max_history=3)
        
        for i in range(5):
            collector.record_cache_metrics(0.5 + 0.1 * i, 50, 50, 0, 512)
        
        assert len(collector.cache_metrics_history) == 3
        assert collector.cache_metrics_history[0].hit_rate == 0.7
        assert collector.cache_metrics_history[2].hit_rate == 0.9
    
    def test_max_history_convergence(self):
        """Test max_history limit for convergence history"""
        collector = EnhancedMetricsCollector(max_history=4)
        
        for i in range(6):
            collector.record_convergence(i * 10, 0.5, 50, 100)
        
        assert len(collector.convergence_history) == 4
    
    def test_statistics_with_extreme_values(self):
        """Test statistics with extreme values"""
        collector = EnhancedMetricsCollector()
        
        # Record extreme gradients
        collector.record_gradient(1000.0, 500.0, 0.99, False)
        collector.record_gradient(-1000.0, -500.0, 0.01, False)
        collector.record_gradient(0.0, 0.0, 0.5, True)
        
        stats = collector.get_gradient_statistics()
        
        assert stats["gradient_min"] == -500.0
        assert stats["gradient_max"] == 500.0
        assert stats["gradient_mean"] == pytest.approx(0.0, abs=0.1)
    
    def test_statistics_with_zero_variance(self):
        """Test statistics with zero variance"""
        collector = EnhancedMetricsCollector()
        
        # Record identical values
        for _ in range(5):
            collector.record_gradient(0.5, 0.25, 0.7, True)
        
        stats = collector.get_gradient_statistics()
        
        assert stats["gradient_std"] == pytest.approx(0.0, abs=0.001)
        assert stats["gradient_mean"] == 0.25
    
    def test_timestamp_ordering(self):
        """Test that timestamps are properly ordered"""
        collector = EnhancedMetricsCollector()
        
        timestamps = []
        for i in range(5):
            time.sleep(0.001)  # Small delay
            collector.record_gradient(0.1 * i, 0.05 * i, 0.5, True)
            timestamps.append(collector.gradient_history[-1].timestamp)
        
        # Verify timestamps are increasing
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i-1]
    
    def test_concurrent_operations(self):
        """Test that collector handles rapid operations"""
        collector = EnhancedMetricsCollector()
        
        # Rapid fire operations
        for i in range(100):
            collector.record_gradient(0.01 * i, 0.005 * i, 0.5, i % 2 == 0)
            collector.record_cache_metrics(0.5, 50, 50, 0, 512)
            collector.record_step(i, 0.001, 0.5, 0.01, 0.1)
        
        assert len(collector.gradient_history) == 100
        assert len(collector.cache_metrics_history) == 100
        assert collector.total_steps == 99

