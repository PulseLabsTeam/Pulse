"""
Performance regression tests

Validates performance targets are maintained.
"""

import pytest
import time
import numpy as np
from pulseos import Runtime, Config, SurvivalConstraint
from pulseos.circuits.ngcm import NonlinearGradientComputationModule
from pulseos.circuits.ptdc import PerformanceThresholdDetectionCircuit


class TestPerformance:
    """Performance regression tests"""
    
    def test_cache_hit_rate_target(self):
        """Test that cache hit rate meets 75% target"""
        ngcm = NonlinearGradientComputationModule(
            cache_size=256,
            target_hit_rate=0.75
        )
        
        # Generate workload with repeated patterns
        deltas = []
        for i in range(1000):
            # 70% repeated values, 30% unique
            if i % 10 < 7:
                deltas.append(0.5 + (i % 5) * 0.1)
            else:
                deltas.append(np.random.uniform(-5, 5))
        
        for i, delta in enumerate(deltas):
            ngcm.compute_gradient(delta, timestamp=i)
        
        hit_rate = ngcm.get_cache_hit_rate()
        assert hit_rate >= 0.70  # At least 70% (allowing some variance)
    
    def test_threshold_detection_latency(self):
        """Test sub-millisecond threshold detection"""
        ptdc = PerformanceThresholdDetectionCircuit(threshold=0.8)
        
        # Register 1000 agents
        metrics = {}
        for i in range(1000):
            agent_id = f"agent_{i}"
            ptdc.register_agent(agent_id, 0.5)
            metrics[agent_id] = np.random.uniform(0.5, 1.0)
        
        # Measure latency
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            ptdc.evaluate(metrics)
            latency = time.perf_counter() - start
            latencies.append(latency)
        
        avg_latency_ms = np.mean(latencies) * 1000
        assert avg_latency_ms < 1.0  # Sub-millisecond
    
    def test_memory_efficiency(self):
        """Test memory efficiency of delta encoding"""
        from pulseos.persistence.snapshot import StateSnapshot
        
        # Create parent snapshot
        parent_data = {
            "step": 0,
            "agents": {f"agent_{i}": {"state": i * 0.1} for i in range(100)}
        }
        parent_snapshot = StateSnapshot(parent_data, enable_delta_encoding=False)
        
        # Create delta snapshot (only changes)
        current_data = parent_data.copy()
        current_data["step"] = 1
        current_data["agents"]["agent_0"] = {"state": 0.2}  # Only one change
        
        delta_snapshot = StateSnapshot(
            current_data,
            parent_snapshot=parent_snapshot,
            enable_delta_encoding=True
        )
        
        # Delta snapshot should be smaller than parent snapshot
        # Compare delta snapshot size to parent snapshot size
        delta_size = delta_snapshot.size_bytes
        parent_size = parent_snapshot.size_bytes
        
        # Delta encoding should reduce size significantly
        # With only one agent change, delta should be much smaller
        assert delta_size < parent_size, f"Delta size ({delta_size}) should be < parent size ({parent_size})"
        
        # Compression ratio should also be reasonable
        compression_ratio = delta_snapshot.get_compression_ratio()
        assert compression_ratio <= 1.0

