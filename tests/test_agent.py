"""
Test suite for PulseOS agent and survival constraint
"""

import pytest
import numpy as np
from pulseos.agent import Agent, SurvivalConstraint


class TestAgent(Agent):
    """Test agent implementation"""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.state = 0.5
    
    async def step(self):
        """Execute step"""
        self.state += 0.01
        return {"state": self.state}
    
    def get_performance_metric(self) -> float:
        """Get performance metric"""
        return self.state


class TestAgentBehaviors:
    """Tests for agent behaviors"""
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        agent = TestAgent("test_agent")
        assert agent.agent_id == "test_agent"
        assert agent.learning_rate == 0.01
        assert agent.exploration_rate == 0.1
    
    def test_update_learning_rate(self):
        """Test learning rate update"""
        agent = TestAgent("test_agent")
        agent.update_learning_rate(0.05)
        assert agent.learning_rate == 0.05
    
    def test_update_exploration_rate(self):
        """Test exploration rate update"""
        agent = TestAgent("test_agent")
        agent.update_exploration_rate(0.2)
        assert agent.exploration_rate == 0.2
    
    def test_get_state(self):
        """Test state retrieval"""
        agent = TestAgent("test_agent")
        state = agent.get_state()
        assert state["agent_id"] == "test_agent"
        assert "learning_rate" in state
        assert "exploration_rate" in state
    
    def test_restore_state(self):
        """Test state restoration"""
        agent = TestAgent("test_agent")
        state = {"learning_rate": 0.05, "exploration_rate": 0.15, "performance_history": []}
        agent.restore_state(state)
        assert agent.learning_rate == 0.05
        assert agent.exploration_rate == 0.15
    
    @pytest.mark.asyncio
    async def test_agent_step(self):
        """Test agent step execution"""
        agent = TestAgent("test_agent")
        result = await agent.step()
        assert "state" in result
        assert agent.state > 0.5


class TestSurvivalConstraint:
    """Tests for survival constraint"""
    
    def test_simple_constraint_evaluation(self):
        """Test simple constraint evaluation"""
        constraint = SurvivalConstraint(threshold=0.8)
        assert constraint.evaluate(0.9) is True
        assert constraint.evaluate(0.7) is False
    
    def test_constraint_history(self):
        """Test constraint history tracking"""
        constraint = SurvivalConstraint(threshold=0.8)
        constraint.evaluate(0.7)
        constraint.evaluate(0.8)
        constraint.evaluate(0.9)
        assert len(constraint.history) == 3
    
    def test_temporal_constraint(self):
        """Test temporal constraint"""
        constraint = SurvivalConstraint(
            threshold=0.8,
            constraint_type="temporal",
            temporal_window=3
        )
        constraint.evaluate(0.9)
        constraint.evaluate(0.9)
        constraint.evaluate(0.9)
        assert constraint.evaluate(0.9) is True
    
    def test_statistical_constraint_mean(self):
        """Test statistical constraint with mean"""
        constraint = SurvivalConstraint(
            threshold=0.8,
            constraint_type="statistical",
            statistical_mode="mean"
        )
        
        constraint.evaluate(0.7)
        constraint.evaluate(0.8)
        constraint.evaluate(0.9)
        
        # Mean should be >= threshold
        # Mean of [0.7, 0.8, 0.9] = 0.8, but due to floating point precision it might be slightly less
        # When we evaluate 0.85, it adds to history: [0.7, 0.8, 0.9, 0.85]
        # Mean of [0.7, 0.8, 0.9, 0.85] = 0.8125 >= 0.8, so result should be True
        result = constraint.evaluate(0.85)
        # The mean of all 4 values should be >= 0.8
        mean = np.mean([0.7, 0.8, 0.9, 0.85])
        # Use approximate comparison to handle floating point precision
        # Check if mean is approximately >= 0.8 (within 0.01 tolerance)
        expected = mean >= (0.8 - 0.01)  # Allow small tolerance
        assert result == expected
    
    def test_statistical_constraint_median(self):
        """Test statistical constraint with median"""
        constraint = SurvivalConstraint(
            threshold=0.8,
            constraint_type="statistical",
            statistical_mode="median"
        )
        constraint.evaluate(0.7)
        constraint.evaluate(0.8)
        constraint.evaluate(0.9)
        result = constraint.evaluate(0.85)
        # Median of [0.7, 0.8, 0.9, 0.85] = 0.825 >= 0.8
        assert result == True
    
    def test_compute_survival_signal(self):
        """Test survival signal computation"""
        constraint = SurvivalConstraint(threshold=0.8)
        signal = constraint.compute_survival_signal(0.5)
        assert signal == 0.5
    
    def test_adapt_threshold(self):
        """Test threshold adaptation"""
        constraint = SurvivalConstraint(threshold=0.8, learning_rate=0.1)
        # Use performance history with median different from initial threshold
        # Median of [0.7, 0.75, 0.85, 0.9] = 0.8, but let's use values where median != 0.8
        performance_history = [0.6, 0.65, 0.7, 0.75, 0.8]  # Median = 0.7
        initial_threshold = constraint.threshold
        constraint.adapt_threshold(performance_history)
        # Threshold should adapt toward median (0.7), so it should decrease
        assert constraint.threshold < initial_threshold  # Should have decreased toward 0.7
    
    def test_temporal_window_limiting(self):
        """Test temporal window limiting"""
        constraint = SurvivalConstraint(
            threshold=0.8,
            constraint_type="temporal",
            temporal_window=3
        )
        # Add more than window size
        for i in range(10):
            constraint.evaluate(0.9)
        # History should be limited to window size
        assert len(constraint.history) == 3
    
    def test_statistical_constraint_insufficient_data(self):
        """Test statistical constraint with insufficient data"""
        constraint = SurvivalConstraint(
            threshold=0.8,
            constraint_type="statistical",
            statistical_mode="mean"
        )
        # Only one value
        result = constraint.evaluate(0.9)
        # Should fall back to simple evaluation
        assert result is True

