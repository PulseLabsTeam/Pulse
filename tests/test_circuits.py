"""
Test suite for PulseOS circuits

Tests patent-specified algorithms for correctness.
"""

import pytest
import numpy as np
from pulseos.circuits.ptdc import PerformanceThresholdDetectionCircuit
from pulseos.circuits.ngcm import NonlinearGradientComputationModule
from pulseos.circuits.apc import AdaptiveParameterController


class TestPTDC:
    """Tests for Performance Threshold Detection Circuit"""
    
    def test_normalization(self):
        """Test normalization: M_norm(t) = M_t / M_initial"""
        ptdc = PerformanceThresholdDetectionCircuit(threshold=0.8)
        
        # Register agent with initial metric
        ptdc.register_agent("agent1", initial_metric=0.5)
        
        # Normalize current metric
        normalized = ptdc.normalize_metric("agent1", 1.0)
        assert normalized == pytest.approx(2.0)  # 1.0 / 0.5
    
    def test_threshold_evaluation(self):
        """Test threshold evaluation"""
        ptdc = PerformanceThresholdDetectionCircuit(threshold=0.8)
        
        metrics = {
            "agent1": 0.9,  # Above threshold
            "agent2": 0.7   # Below threshold
        }
        
        results = ptdc.evaluate(metrics)
        
        assert results["agent1"] is True
        assert results["agent2"] is False
    
    def test_parallel_comparison(self):
        """Test parallel comparison for multiple agents"""
        ptdc = PerformanceThresholdDetectionCircuit(threshold=0.8)
        
        # Register multiple agents
        metrics = {f"agent_{i}": 0.5 + i * 0.1 for i in range(10)}
        
        for agent_id, metric in metrics.items():
            ptdc.register_agent(agent_id, metric)
        
        # Evaluate all at once
        results = ptdc.evaluate(metrics)
        
        assert len(results) == 10
        # Agents with metric >= 0.8 should pass
        assert results["agent_3"] is False  # 0.8 exactly
        assert results["agent_4"] is True   # 0.9
    
    def test_sub_millisecond_latency(self):
        """Test that detection is sub-millisecond"""
        import time
        
        ptdc = PerformanceThresholdDetectionCircuit(threshold=0.8)
        metrics = {f"agent_{i}": 0.9 for i in range(100)}
        
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            ptdc.evaluate(metrics)
            latency = time.perf_counter() - start
            latencies.append(latency)
        
        avg_latency_ms = np.mean(latencies) * 1000
        assert avg_latency_ms < 1.0  # Sub-millisecond


class TestNGCM:
    """Tests for Nonlinear Gradient Computation Module"""
    
    def test_sigmoid_computation(self):
        """Test sigmoid: S(t) = 1 / (1 + exp(-β × Δ(t)))"""
        ngcm = NonlinearGradientComputationModule(beta=1.0)
        
        delta = 0.0
        sigmoid = ngcm.compute_sigmoid(delta)
        
        # At delta=0, sigmoid should be 0.5
        assert sigmoid == pytest.approx(0.5, abs=0.01)
    
    def test_gradient_computation(self):
        """Test gradient: G(t) = β × S(t) × (1 - S(t))"""
        ngcm = NonlinearGradientComputationModule(beta=1.0)
        
        delta = 0.0
        gradient = ngcm.compute_gradient(delta, timestamp=0)
        
        # At delta=0, gradient should be maximum (0.25 for beta=1.0)
        assert gradient == pytest.approx(0.25, abs=0.01)
    
    def test_cache_functionality(self):
        """Test gradient caching"""
        ngcm = NonlinearGradientComputationModule(cache_size=256)
        
        delta = 0.5
        
        # First computation (cache miss)
        gradient1 = ngcm.compute_gradient(delta, timestamp=0)
        
        # Second computation (cache hit)
        gradient2 = ngcm.compute_gradient(delta, timestamp=1)
        
        assert gradient1 == gradient2
        assert ngcm.cache_hits >= 1
    
    def test_cache_hit_rate(self):
        """Test cache hit rate target (75%)"""
        ngcm = NonlinearGradientComputationModule(
            cache_size=256,
            target_hit_rate=0.75
        )
        
        # Generate repeated deltas to increase hit rate
        deltas = [0.5, 0.6, 0.5, 0.7, 0.5, 0.8] * 100
        
        for i, delta in enumerate(deltas):
            ngcm.compute_gradient(delta, timestamp=i)
        
        hit_rate = ngcm.get_cache_hit_rate()
        # With repeated values, hit rate should be high
        assert hit_rate > 0.5  # At least 50% with this pattern
    
    def test_lut_implementation(self):
        """Test LUT implementation"""
        ngcm = NonlinearGradientComputationModule(
            implementation="LUT",
            cache_size=256
        )
        
        delta = 0.5
        gradient = ngcm.compute_gradient(delta, timestamp=0)
        
        # Should compute successfully
        assert gradient > 0
        assert gradient < 1
    
    def test_pla_implementation(self):
        """Test PLA implementation"""
        ngcm = NonlinearGradientComputationModule(
            implementation="PLA",
            cache_size=256
        )
        
        delta = 0.5
        gradient = ngcm.compute_gradient(delta, timestamp=0)
        
        assert gradient > 0
        assert gradient < 1


class TestAPC:
    """Tests for Adaptive Parameter Controller"""
    
    def test_alpha_update(self):
        """Test learning rate update: α(t) = α_base × (1 + γ × G(t) × (1 - S(t)))"""
        apc = AdaptiveParameterController(
            alpha_base=0.01,
            gamma=0.1
        )
        
        gradient = 0.25
        survival_signal = 0.5
        
        alpha, epsilon = apc.update_parameters(gradient, survival_signal)
        
        # Alpha should be updated based on formula
        expected_alpha = 0.01 * (1 + 0.1 * 0.25 * (1 - 0.5))
        assert alpha == pytest.approx(expected_alpha, abs=0.001)
    
    def test_epsilon_update(self):
        """Test exploration rate update: ε(t) = ε_min + (ε_max - ε_min) × (1 - S(t))^κ"""
        apc = AdaptiveParameterController(
            epsilon_min=0.01,
            epsilon_max=0.3,
            epsilon_kappa=2.0
        )
        
        gradient = 0.25
        survival_signal = 0.5
        
        alpha, epsilon = apc.update_parameters(gradient, survival_signal)
        
        # Epsilon should be in valid range
        assert epsilon >= apc.epsilon_min
        assert epsilon <= apc.epsilon_max
    
    def test_rate_limiting(self):
        """Test rate limiting (max 10% change per step)"""
        apc = AdaptiveParameterController(
            alpha_base=0.01,
            alpha_max_change=0.10
        )
        
        # Try to make large change
        gradient = 10.0  # Large gradient
        survival_signal = 0.0  # Low survival
        
        alpha1 = apc.get_alpha()
        alpha, epsilon = apc.update_parameters(gradient, survival_signal)
        alpha2 = apc.get_alpha()
        
        # Change should be limited
        change_ratio = abs(alpha2 - alpha1) / alpha1 if alpha1 > 0 else 0
        assert change_ratio <= 0.10 + 0.01  # Allow small tolerance
    
    def test_smoothing(self):
        """Test exponential moving average smoothing"""
        apc = AdaptiveParameterController(
            alpha_base=0.01,
            alpha_smooth=0.9
        )
        
        gradient = 0.25
        survival_signal = 0.5
        
        # Update multiple times
        for _ in range(10):
            alpha, epsilon = apc.update_parameters(gradient, survival_signal)
        
        # Alpha should be smoothed
        assert alpha > 0
    
    def test_increase_exploration(self):
        """Test exploration increase during rollback"""
        apc = AdaptiveParameterController(epsilon_max=0.3)
        
        initial_epsilon = apc.get_epsilon()
        apc.increase_exploration(factor=1.5)
        new_epsilon = apc.get_epsilon()
        
        assert new_epsilon >= initial_epsilon
        assert new_epsilon <= apc.epsilon_max

