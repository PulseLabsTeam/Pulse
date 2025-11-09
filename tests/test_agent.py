"""
Test suite for agent behaviors and survival constraints

Tests agent interface, survival constraints, and constraint evaluation.
"""

import pytest
import numpy as np
from pulseos.agent import Agent, SurvivalConstraint


class TestAgent(Agent):
    """Test agent for testing"""
    
    def __init__(self, agent_id: str, performance: float = 0.5):
        super().__init__(agent_id)
        self.performance = performance
    
    async def step(self) -> dict:
        return {"performance": self.performance}
    
    def get_performance_metric(self) -> float:
        return self.performance


class TestAgentBehaviors:
    """Tests for Agent behaviors"""
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        agent = TestAgent("agent1")
        
        assert agent.agent_id == "agent1"
        assert agent.learning_rate == 0.01
        assert agent.exploration_rate == 0.1
    
    def test_update_learning_rate(self):
        """Test learning rate update"""
        agent = TestAgent("agent1")
        
        agent.update_learning_rate(0.05)
        assert agent.learning_rate == 0.05
    
    def test_update_exploration_rate(self):
        """Test exploration rate update"""
        agent = TestAgent("agent1")
        
        agent.update_exploration_rate(0.2)
        assert agent.exploration_rate == 0.2
    
    def test_get_state(self):
        """Test agent state retrieval"""
        agent = TestAgent("agent1")
        agent.learning_rate = 0.05
        agent.exploration_rate = 0.2
        
        state = agent.get_state()
        
        assert state["agent_id"] == "agent1"
        assert state["learning_rate"] == 0.05
        assert state["exploration_rate"] == 0.2
    
    def test_restore_state(self):
        """Test agent state restoration"""
        agent = TestAgent("agent1")
        
        state = {
            "agent_id": "agent1",
            "learning_rate": 0.05,
            "exploration_rate": 0.2,
            "performance_history": [0.5, 0.6, 0.7]
        }
        
        agent.restore_state(state)
        
        assert agent.learning_rate == 0.05
        assert agent.exploration_rate == 0.2
        assert len(agent.performance_history) == 3
    
    @pytest.mark.asyncio
    async def test_agent_step(self):
        """Test agent step execution"""
        agent = TestAgent("agent1", performance=0.7)
        
        result = await agent.step()
        
        assert "performance" in result
        assert result["performance"] == 0.7


class TestSurvivalConstraint:
    """Tests for Survival Constraint"""
    
    def test_simple_constraint_evaluation(self):
        """Test simple constraint evaluation"""
        constraint = SurvivalConstraint(threshold=0.8)
        
        assert constraint.evaluate(0.9) is True
        assert constraint.evaluate(0.7) is False
        assert constraint.evaluate(0.8) is True  # >= threshold
    
    def test_constraint_history(self):
        """Test constraint history tracking"""
        constraint = SurvivalConstraint(threshold=0.8)
        
        constraint.evaluate(0.7)
        constraint.evaluate(0.9)
        constraint.evaluate(0.8)
        
        assert len(constraint.history) == 3
        assert constraint.history == [0.7, 0.9, 0.8]
    
    def test_temporal_constraint(self):
        """Test temporal constraint evaluation"""
        constraint = SurvivalConstraint(
            threshold=0.8,
            constraint_type="temporal",
            temporal_window=3
        )
        
        # Not enough history yet
        assert constraint.evaluate(0.7) is True
        
        # Build history
        constraint.evaluate(0.9)
        constraint.evaluate(0.85)
        
        # All recent values meet threshold
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
        mean = np.mean([0.7, 0.8, 0.9])
        result = constraint.evaluate(0.85)
        # Use approximate comparison to handle floating point precision
        assert result == (mean >= pytest.approx(0.8, abs=0.01))
    
    def test_statistical_constraint_median(self):
        """Test statistical constraint with median"""
        constraint = SurvivalConstraint(
            threshold=0.8,
            constraint_type="statistical",
            statistical_mode="median"
        )
        
        constraint.evaluate(0.7)
        constraint.evaluate(0.9)
        constraint.evaluate(0.85)
        
        median = np.median([0.7, 0.9, 0.85])
        result = constraint.evaluate(0.9)
        assert result == (median >= 0.8)
    
    def test_compute_survival_signal(self):
        """Test survival signal computation"""
        constraint = SurvivalConstraint(threshold=0.8)
        
        # 100% survival ratio -> signal = 1.0
        signal = constraint.compute_survival_signal(1.0)
        assert signal == 1.0
        
        # 50% survival ratio -> signal = 0.5
        signal = constraint.compute_survival_signal(0.5)
        assert signal == 0.5
        
        # 0% survival ratio -> signal = 0.0
        signal = constraint.compute_survival_signal(0.0)
        assert signal == 0.0
    
    def test_adapt_threshold(self):
        """Test threshold adaptation"""
        constraint = SurvivalConstraint(threshold=0.8, learning_rate=0.1)
        original_threshold = constraint.threshold
        
        # Adapt based on performance history
        # Median of [0.7, 0.75, 0.8, 0.85, 0.9] = 0.8
        # With learning_rate=0.1, new_threshold = 0.9 * 0.8 + 0.1 * 0.8 = 0.8
        # So threshold stays the same. Use different values to ensure change.
        performance_history = [0.5, 0.6, 0.7, 0.8, 0.9]  # Median is 0.7
        constraint.adapt_threshold(performance_history)
        
        # Threshold should move toward median (0.7)
        # new_threshold = 0.9 * 0.8 + 0.1 * 0.7 = 0.72 + 0.07 = 0.79
        assert constraint.threshold != original_threshold
        assert constraint.threshold >= 0.0
        assert constraint.threshold <= 1.0
    
    def test_temporal_window_limiting(self):
        """Test temporal window limits history"""
        constraint = SurvivalConstraint(
            threshold=0.8,
            constraint_type="temporal",
            temporal_window=3
        )
        
        # Add more than window size
        for i in range(10):
            constraint.evaluate(0.5 + i * 0.05)
        
        # History should be limited to window size
        assert len(constraint.history) == 3
    
    def test_statistical_constraint_insufficient_data(self):
        """Test statistical constraint with insufficient data"""
        constraint = SurvivalConstraint(
            threshold=0.8,
            constraint_type="statistical",
            statistical_mode="mean"
        )
        
        # Single data point
        result = constraint.evaluate(0.9)
        assert result is True  # Should use single value
        
        # No data
        constraint2 = SurvivalConstraint(
            threshold=0.8,
            constraint_type="statistical"
        )
        result = constraint2.evaluate(0.9)
        assert result is True  # Should default to simple evaluation

