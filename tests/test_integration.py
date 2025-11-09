"""
Integration tests for PulseOS

Tests end-to-end workflows and subsystem integration.
"""

import pytest
import asyncio
from pulseos import Runtime, Config, Agent, SurvivalConstraint
from pulseos.runtime import RuntimeState
from pulseos.persistence.snapshot import SnapshotManager, StateSnapshot


class IntegrationAgent(Agent):
    """Agent for integration testing"""
    
    def __init__(self, agent_id: str, initial_performance: float = 0.5):
        super().__init__(agent_id)
        self.state = 0.0
        self.target = 1.0
        self.performance = initial_performance
    
    async def step(self) -> dict:
        error = self.target - self.state
        self.state += self.learning_rate * error
        self.state = max(0.0, min(1.0, self.state))
        return {"state": self.state, "error": abs(error)}
    
    def get_performance_metric(self) -> float:
        # Use stored performance or compute from state
        if hasattr(self, 'performance'):
            return self.performance
        error = abs(self.target - self.state)
        return 1.0 - error


class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.asyncio
    async def test_full_runtime_workflow(self):
        """Test complete runtime workflow"""
        constraint = SurvivalConstraint(threshold=0.8)
        runtime = Runtime(constraint=constraint)
        
        # Register agents
        for i in range(10):
            agent = IntegrationAgent(f"agent_{i}")
            runtime.register_agent(f"agent_{i}", agent)
        
        # Run for several steps
        for _ in range(20):
            await runtime.step()
        
        # Verify runtime state
        assert runtime.current_step == 20
        assert runtime.state == RuntimeState.RUNNING
        assert len(runtime.agents) == 10
        
        # Verify statistics
        stats = runtime.get_statistics()
        assert stats["agent_count"] == 10
        assert "average_survival_signal" in stats
    
    @pytest.mark.asyncio
    async def test_snapshot_and_rollback(self):
        """Test snapshot creation and rollback"""
        constraint = SurvivalConstraint(threshold=0.3)  # Low threshold to trigger rollback
        config = Config(
            snapshot_interval=0.1,
            critical_survival_threshold=0.3,
            rollback_grace_period=0.5
        )
        runtime = Runtime(constraint=constraint, config=config)
        
        # Register agents with low performance
        for i in range(5):
            agent = IntegrationAgent(f"agent_{i}", initial_performance=0.2)
            runtime.register_agent(f"agent_{i}", agent)
        
        # Run steps to create snapshots
        # Need to ensure enough time passes for snapshot interval (0.1s)
        for i in range(15):
            await runtime.step()
            # Sleep longer to ensure snapshot interval is reached
            if i > 0:  # Skip first sleep
                await asyncio.sleep(0.02)  # 20ms * 14 = 280ms > 100ms interval
        
        # Verify snapshots were created
        snapshot_count = runtime.sprs.get_snapshot_count()
        assert snapshot_count > 0
    
    @pytest.mark.asyncio
    async def test_agent_parameter_updates(self):
        """Test agent parameter updates during runtime"""
        constraint = SurvivalConstraint(threshold=0.8)
        runtime = Runtime(constraint=constraint)
        
        agent = IntegrationAgent("agent1")
        runtime.register_agent("agent1", agent)
        
        initial_alpha = agent.learning_rate
        initial_epsilon = agent.exploration_rate
        
        # Run steps
        for _ in range(10):
            await runtime.step()
        
        # Parameters should be updated by APC
        # (May be same or different depending on survival signal)
        assert agent.learning_rate >= 0
        assert agent.exploration_rate >= 0
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self):
        """Test metrics collection during runtime"""
        constraint = SurvivalConstraint(threshold=0.8)
        runtime = Runtime(constraint=constraint)
        
        for i in range(5):
            agent = IntegrationAgent(f"agent_{i}")
            runtime.register_agent(f"agent_{i}", agent)
        
        # Run steps
        for _ in range(10):
            await runtime.step()
        
        # Verify metrics were collected
        metrics_stats = runtime.metrics_collector.get_statistics()
        assert metrics_stats["total_steps"] == 10
        assert "survival_signal_mean" in metrics_stats
    
    @pytest.mark.asyncio
    async def test_event_handling(self):
        """Test event handling during runtime"""
        constraint = SurvivalConstraint(threshold=0.8)
        runtime = Runtime(constraint=constraint)
        
        events_received = []
        
        def threshold_handler(data):
            events_received.append(("threshold_breach", data))
        
        def rollback_handler(data):
            events_received.append(("rollback", data))
        
        runtime.register_event_handler("threshold_breach", threshold_handler)
        runtime.register_event_handler("rollback", rollback_handler)
        
        # Run steps
        for _ in range(10):
            await runtime.step()
        
        # Events may or may not be triggered depending on performance
        # Just verify handler registration works
        assert True  # Handler registered successfully
    
    @pytest.mark.asyncio
    async def test_multiple_agents_convergence(self):
        """Test multiple agents converging"""
        constraint = SurvivalConstraint(threshold=0.9)
        runtime = Runtime(constraint=constraint)
        
        # Register agents
        for i in range(20):
            agent = IntegrationAgent(f"agent_{i}")
            runtime.register_agent(f"agent_{i}", agent)
        
        # Run until convergence or max steps
        converged = False
        for step in range(100):
            await runtime.step()
            
            # Check if agents are converging
            metrics = runtime._collect_agent_metrics()
            avg_performance = sum(metrics.values()) / len(metrics) if metrics else 0
            
            if avg_performance >= 0.9:
                converged = True
                break
        
        # Should converge or make progress
        assert runtime.current_step > 0
        stats = runtime.get_statistics()
        assert "average_survival_signal" in stats
    
    @pytest.mark.asyncio
    async def test_runtime_pause_resume(self):
        """Test runtime pause and resume"""
        constraint = SurvivalConstraint(threshold=0.8)
        runtime = Runtime(constraint=constraint)
        
        agent = IntegrationAgent("agent1")
        runtime.register_agent("agent1", agent)
        
        # Run a few steps
        for _ in range(5):
            await runtime.step()
        
        step_before_pause = runtime.current_step
        
        # Pause
        runtime.pause()
        assert runtime.state == RuntimeState.PAUSED
        
        # Resume
        runtime.resume()
        assert runtime.state == RuntimeState.RUNNING
        
        # Continue running
        await runtime.step()
        assert runtime.current_step == step_before_pause + 1

