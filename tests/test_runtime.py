"""
Test suite for runtime functionality

Tests runtime orchestrator, state management, and integration with subsystems.
"""

import pytest
import asyncio
import time
from pulseos import Runtime, Config, Agent, SurvivalConstraint
from pulseos.runtime import RuntimeState


class TestRuntimeAgent(Agent):
    """Test agent implementation"""
    
    def __init__(self, agent_id: str, initial_performance: float = 0.5):
        super().__init__(agent_id)
        self.performance = initial_performance
        self.step_count = 0
    
    async def step(self) -> dict:
        """Execute step"""
        self.step_count += 1
        # Gradually improve performance
        self.performance = min(1.0, self.performance + 0.01)
        return {"step": self.step_count, "performance": self.performance}
    
    def get_performance_metric(self) -> float:
        """Get performance metric"""
        return self.performance


class TestRuntime:
    """Tests for Runtime"""
    
    def test_runtime_initialization(self):
        """Test runtime initialization"""
        constraint = SurvivalConstraint(threshold=0.8)
        runtime = Runtime(constraint=constraint)
        
        assert runtime.state == RuntimeState.INITIALIZING
        assert len(runtime.agents) == 0
        assert runtime.current_step == 0
    
    def test_register_agent(self):
        """Test agent registration"""
        constraint = SurvivalConstraint(threshold=0.8)
        runtime = Runtime(constraint=constraint)
        
        agent = TestRuntimeAgent("agent1", initial_performance=0.7)
        runtime.register_agent("agent1", agent)
        
        assert "agent1" in runtime.agents
        assert runtime.agents["agent1"] == agent
    
    def test_register_multiple_agents(self):
        """Test registering multiple agents"""
        constraint = SurvivalConstraint(threshold=0.8)
        runtime = Runtime(constraint=constraint)
        
        for i in range(10):
            agent = TestRuntimeAgent(f"agent_{i}", initial_performance=0.7)
            runtime.register_agent(f"agent_{i}", agent)
        
        assert len(runtime.agents) == 10
    
    def test_max_agents_limit(self):
        """Test maximum agents limit"""
        constraint = SurvivalConstraint(threshold=0.8)
        config = Config(max_agents=5)
        runtime = Runtime(constraint=constraint, config=config)
        
        # Register up to limit
        for i in range(5):
            agent = TestRuntimeAgent(f"agent_{i}")
            runtime.register_agent(f"agent_{i}", agent)
        
        assert len(runtime.agents) == 5
        
        # Should raise error when exceeding limit
        with pytest.raises(RuntimeError, match="Maximum agent limit"):
            agent = TestRuntimeAgent("agent_6")
            runtime.register_agent("agent_6", agent)
    
    def test_unregister_agent(self):
        """Test agent unregistration"""
        constraint = SurvivalConstraint(threshold=0.8)
        runtime = Runtime(constraint=constraint)
        
        agent = TestRuntimeAgent("agent1")
        runtime.register_agent("agent1", agent)
        
        runtime.unregister_agent("agent1")
        assert "agent1" not in runtime.agents
    
    @pytest.mark.asyncio
    async def test_runtime_step(self):
        """Test runtime step execution"""
        constraint = SurvivalConstraint(threshold=0.8)
        runtime = Runtime(constraint=constraint)
        
        agent = TestRuntimeAgent("agent1", initial_performance=0.7)
        runtime.register_agent("agent1", agent)
        
        result = await runtime.step()
        
        assert runtime.current_step == 1
        assert "survival_signal" in result
        assert "alpha" in result
        assert "epsilon" in result
        assert runtime.state == RuntimeState.RUNNING
    
    @pytest.mark.asyncio
    async def test_multiple_steps(self):
        """Test multiple runtime steps"""
        constraint = SurvivalConstraint(threshold=0.8)
        runtime = Runtime(constraint=constraint)
        
        agent = TestRuntimeAgent("agent1", initial_performance=0.7)
        runtime.register_agent("agent1", agent)
        
        for i in range(5):
            await runtime.step()
            assert runtime.current_step == i + 1
    
    @pytest.mark.asyncio
    async def test_runtime_run(self):
        """Test runtime run method"""
        constraint = SurvivalConstraint(threshold=0.8)
        runtime = Runtime(constraint=constraint)
        
        agent = TestRuntimeAgent("agent1", initial_performance=0.7)
        runtime.register_agent("agent1", agent)
        
        # Run for limited steps
        await runtime.run(max_steps=10)
        
        assert runtime.current_step >= 10
        assert runtime.state == RuntimeState.SHUTTING_DOWN
    
    def test_pause_and_resume(self):
        """Test pause and resume"""
        constraint = SurvivalConstraint(threshold=0.8)
        runtime = Runtime(constraint=constraint)
        
        runtime.state = RuntimeState.RUNNING
        runtime.pause()
        assert runtime.state == RuntimeState.PAUSED
        
        runtime.resume()
        assert runtime.state == RuntimeState.RUNNING
    
    def test_shutdown(self):
        """Test runtime shutdown"""
        constraint = SurvivalConstraint(threshold=0.8)
        runtime = Runtime(constraint=constraint)
        
        runtime.state = RuntimeState.RUNNING
        runtime.shutdown()
        assert runtime.state == RuntimeState.SHUTTING_DOWN
    
    def test_get_statistics(self):
        """Test runtime statistics"""
        constraint = SurvivalConstraint(threshold=0.8)
        runtime = Runtime(constraint=constraint)
        
        # Empty runtime
        stats = runtime.get_statistics()
        assert isinstance(stats, dict)
    
    @pytest.mark.asyncio
    async def test_event_handlers(self):
        """Test event handler registration and emission"""
        constraint = SurvivalConstraint(threshold=0.8)
        runtime = Runtime(constraint=constraint)
        
        events_received = []
        
        def handler(data):
            events_received.append(data)
        
        runtime.register_event_handler("threshold_breach", handler)
        
        # Trigger event
        runtime._emit_event("threshold_breach", {"agent": "agent1"})
        
        assert len(events_received) == 1
        assert events_received[0]["agent"] == "agent1"
    
    @pytest.mark.asyncio
    async def test_step_with_no_agents(self):
        """Test step execution with no agents"""
        constraint = SurvivalConstraint(threshold=0.8)
        runtime = Runtime(constraint=constraint)
        
        result = await runtime.step()
        
        assert result["survival_signal"] == 0.0  # No agents = no survival signal
        assert runtime.current_step == 1
    
    @pytest.mark.asyncio
    async def test_step_error_handling(self):
        """Test error handling during step execution"""
        constraint = SurvivalConstraint(threshold=0.8)
        runtime = Runtime(constraint=constraint)
        
        class FailingAgent(Agent):
            async def step(self):
                raise ValueError("Agent error")
            
            def get_performance_metric(self):
                return 0.5
        
        agent = FailingAgent("agent1")
        runtime.register_agent("agent1", agent)
        
        # Step should handle agent errors gracefully
        result = await runtime.step()
        assert "survival_signal" in result
    
    def test_config_customization(self):
        """Test runtime with custom config"""
        constraint = SurvivalConstraint(threshold=0.8)
        config = Config(
            alpha_base=0.1,
            epsilon_max=0.5,
            max_agents=100
        )
        runtime = Runtime(constraint=constraint, config=config)
        
        assert runtime.config.alpha_base == 0.1
        assert runtime.config.epsilon_max == 0.5
        assert runtime.config.max_agents == 100

