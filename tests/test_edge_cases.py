"""
Edge case and error handling tests

Tests boundary conditions, error scenarios, and edge cases.
"""

import pytest
import numpy as np
from pulseos import Runtime, Config, Agent, SurvivalConstraint
from pulseos.circuits.ptdc import PerformanceThresholdDetectionCircuit
from pulseos.circuits.ngcm import NonlinearGradientComputationModule
from pulseos.circuits.apc import AdaptiveParameterController
from pulseos.persistence.snapshot import StateSnapshot, SnapshotManager
from pulseos.persistence.merkle import MerkleTree, SnapshotIntegrity


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""
    
    def test_ptdc_zero_threshold(self):
        """Test PTDC with zero threshold"""
        ptdc = PerformanceThresholdDetectionCircuit(threshold=0.0)
        
        results = ptdc.evaluate({"agent1": 0.1})
        assert results["agent1"] is True  # Any positive value > 0
    
    def test_ptdc_negative_metrics(self):
        """Test PTDC with negative metrics"""
        ptdc = PerformanceThresholdDetectionCircuit(threshold=0.5)
        
        results = ptdc.evaluate({"agent1": -0.1})
        assert results["agent1"] is False  # Negative < threshold
    
    def test_ptdc_empty_metrics(self):
        """Test PTDC with empty metrics"""
        ptdc = PerformanceThresholdDetectionCircuit(threshold=0.8)
        
        results = ptdc.evaluate({})
        assert results == {}
    
    def test_ptdc_zero_initial_metric(self):
        """Test PTDC normalization with zero initial metric"""
        ptdc = PerformanceThresholdDetectionCircuit(threshold=0.8)
        
        ptdc.register_agent("agent1", initial_metric=0.0)
        normalized = ptdc.normalize_metric("agent1", 1.0)
        
        # Should handle division by zero
        assert normalized == float('inf') or normalized == 1.0
    
    def test_ngcm_extreme_deltas(self):
        """Test NGCM with extreme delta values"""
        ngcm = NonlinearGradientComputationModule(beta=1.0)
        
        # Very large delta
        gradient_large = ngcm.compute_gradient(100.0, timestamp=0)
        assert gradient_large >= 0
        assert gradient_large <= 1.0
        
        # Very negative delta
        gradient_negative = ngcm.compute_gradient(-100.0, timestamp=0)
        assert gradient_negative >= 0
        assert gradient_negative <= 1.0
    
    def test_ngcm_zero_beta(self):
        """Test NGCM with zero beta"""
        ngcm = NonlinearGradientComputationModule(beta=0.0)
        
        gradient = ngcm.compute_gradient(1.0, timestamp=0)
        assert gradient == 0.0  # Zero beta -> zero gradient
    
    def test_apc_extreme_gradients(self):
        """Test APC with extreme gradient values"""
        apc = AdaptiveParameterController(alpha_base=0.01, alpha_max_change=0.1)
        
        # Very large gradient
        alpha, epsilon = apc.update_parameters(gradient=1000.0, survival_signal=0.5)
        
        assert alpha >= 0
        assert epsilon >= apc.epsilon_min
        assert epsilon <= apc.epsilon_max
    
    def test_apc_boundary_survival_signals(self):
        """Test APC with boundary survival signals"""
        apc = AdaptiveParameterController()
        
        # Survival signal = 0.0
        alpha1, epsilon1 = apc.update_parameters(0.25, 0.0)
        
        # Survival signal = 1.0
        alpha2, epsilon2 = apc.update_parameters(0.25, 1.0)
        
        assert alpha1 >= 0
        assert alpha2 >= 0
        assert epsilon1 >= apc.epsilon_min
        assert epsilon2 >= apc.epsilon_min
    
    def test_snapshot_empty_data(self):
        """Test snapshot with empty data"""
        snapshot = StateSnapshot({}, enable_delta_encoding=False)
        
        assert snapshot.step == 0
        assert snapshot.agent_count == 0
    
    def test_snapshot_large_data(self):
        """Test snapshot with large data"""
        large_data = {
            "step": 1,
            "agents": {f"agent_{i}": {"state": i * 0.1} for i in range(1000)}
        }
        
        snapshot = StateSnapshot(large_data, enable_delta_encoding=False)
        
        assert snapshot.agent_count == 1000
        assert snapshot.size_bytes > 0
    
    def test_snapshot_no_parent(self):
        """Test delta snapshot without parent"""
        data = {"step": 1, "agents": {"agent1": {"state": 0.5}}}
        
        snapshot = StateSnapshot(data, parent_snapshot=None, enable_delta_encoding=True)
        
        # Should work without parent (no delta encoding)
        assert snapshot.step == 1
    
    def test_snapshot_manager_empty(self):
        """Test snapshot manager with no snapshots"""
        manager = SnapshotManager()
        
        assert manager.get_snapshot_count() == 0
        stats = manager.get_statistics()
        assert stats["snapshot_count"] == 0
    
    @pytest.mark.asyncio
    async def test_runtime_no_agents(self):
        """Test runtime with no agents"""
        constraint = SurvivalConstraint(threshold=0.8)
        runtime = Runtime(constraint=constraint)
        
        result = await runtime.step()
        
        assert result["survival_signal"] == 0.0
        assert runtime.current_step == 1
    
    def test_runtime_max_agents(self):
        """Test runtime at max agents limit"""
        constraint = SurvivalConstraint(threshold=0.8)
        config = Config(max_agents=3)
        runtime = Runtime(constraint=constraint, config=config)
        
        # Register up to limit
        for i in range(3):
            agent = IntegrationAgent(f"agent_{i}")
            runtime.register_agent(f"agent_{i}", agent)
        
        assert len(runtime.agents) == 3
        
        # Should fail on 4th
        with pytest.raises(RuntimeError):
            agent = IntegrationAgent("agent_4")
            runtime.register_agent("agent_4", agent)
    
    def test_constraint_zero_threshold(self):
        """Test constraint with zero threshold"""
        constraint = SurvivalConstraint(threshold=0.0)
        
        assert constraint.evaluate(0.0) is True
        assert constraint.evaluate(-0.1) is False
    
    def test_constraint_negative_threshold(self):
        """Test constraint with negative threshold"""
        constraint = SurvivalConstraint(threshold=-0.5)
        
        assert constraint.evaluate(0.0) is True  # 0 > -0.5
        assert constraint.evaluate(-1.0) is False  # -1 < -0.5
    
    def test_merkle_tree_single_block(self):
        """Test Merkle tree with single block"""
        blocks = [b"single block"]
        tree = MerkleTree(blocks)
        
        assert tree.get_root_hash() is not None
        assert tree.verify(blocks) is True
    
    def test_merkle_tree_odd_blocks(self):
        """Test Merkle tree with odd number of blocks"""
        blocks = [b"block1", b"block2", b"block3"]
        tree = MerkleTree(blocks)
        
        assert tree.verify(blocks) is True
    
    def test_merkle_tree_verification_failure(self):
        """Test Merkle tree verification failure"""
        blocks1 = [b"block1", b"block2"]
        blocks2 = [b"block1", b"modified"]
        
        tree = MerkleTree(blocks1)
        
        assert tree.verify(blocks1) is True
        assert tree.verify(blocks2) is False
    
    def test_snapshot_integrity(self):
        """Test snapshot integrity verification"""
        integrity = SnapshotIntegrity()
        
        snapshot_data = {"step": 1, "agents": {"agent1": {"state": 0.5}}}
        
        integrity.register_snapshot("snapshot1", snapshot_data)
        
        assert integrity.verify_snapshot("snapshot1", snapshot_data) is True
        
        # Modified data should fail
        modified_data = {"step": 1, "agents": {"agent1": {"state": 0.6}}}
        assert integrity.verify_snapshot("snapshot1", modified_data) is False


class IntegrationAgent(Agent):
    """Agent for integration testing"""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.state = 0.0
        self.target = 1.0
        self.performance = 0.5
    
    async def step(self) -> dict:
        error = self.target - self.state
        self.state += self.learning_rate * error
        self.state = max(0.0, min(1.0, self.state))
        return {"state": self.state, "error": abs(error)}
    
    def get_performance_metric(self) -> float:
        return self.performance

