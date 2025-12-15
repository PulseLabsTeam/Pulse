"""
Custom Constraints Example: Advanced Constraint Types

Demonstrates how to create and use custom constraint types beyond
basic threshold constraints, including temporal and statistical constraints.
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional
from pulseos import Runtime, Agent, Config, SurvivalConstraint


class TemporalConstraint(SurvivalConstraint):
    """
    Temporal constraint: performance must be above threshold for N consecutive steps.
    """
    
    def __init__(self, threshold: float, consecutive_steps: int = 5):
        super().__init__(threshold)
        self.consecutive_steps = consecutive_steps
        self.step_history: Dict[str, List[bool]] = {}
    
    def evaluate(self, agent_id: str, metric: float) -> bool:
        """Evaluate temporal constraint"""
        if agent_id not in self.step_history:
            self.step_history[agent_id] = []
        
        # Check current step
        current_pass = metric >= self.threshold
        self.step_history[agent_id].append(current_pass)
        
        # Keep only recent history
        if len(self.step_history[agent_id]) > self.consecutive_steps:
            self.step_history[agent_id] = self.step_history[agent_id][-self.consecutive_steps:]
        
        # Must pass for consecutive_steps steps
        if len(self.step_history[agent_id]) < self.consecutive_steps:
            return False
        
        return all(self.step_history[agent_id])


class StatisticalConstraint(SurvivalConstraint):
    """
    Statistical constraint: average performance over window must be above threshold.
    """
    
    def __init__(self, threshold: float, window_size: int = 10):
        super().__init__(threshold)
        self.window_size = window_size
        self.metric_history: Dict[str, List[float]] = {}
    
    def evaluate(self, agent_id: str, metric: float) -> bool:
        """Evaluate statistical constraint"""
        if agent_id not in self.metric_history:
            self.metric_history[agent_id] = []
        
        # Add current metric
        self.metric_history[agent_id].append(metric)
        
        # Keep only recent history
        if len(self.metric_history[agent_id]) > self.window_size:
            self.metric_history[agent_id] = self.metric_history[agent_id][-self.window_size:]
        
        # Compute average
        if len(self.metric_history[agent_id]) < self.window_size:
            return False  # Not enough data
        
        average = np.mean(self.metric_history[agent_id])
        return average >= self.threshold


class CompositeConstraint(SurvivalConstraint):
    """
    Composite constraint: combines multiple constraints with AND/OR logic.
    """
    
    def __init__(self, constraints: List[SurvivalConstraint], logic: str = "AND"):
        super().__init__(threshold=0.0)  # Not used
        self.constraints = constraints
        self.logic = logic.upper()
    
    def evaluate(self, agent_id: str, metric: float) -> bool:
        """Evaluate composite constraint"""
        results = [constraint.evaluate(agent_id, metric) for constraint in self.constraints]
        
        if self.logic == "AND":
            return all(results)
        elif self.logic == "OR":
            return any(results)
        else:
            raise ValueError(f"Unknown logic: {self.logic}")


class AdaptiveConstraint(SurvivalConstraint):
    """
    Adaptive constraint: threshold adjusts based on agent performance.
    """
    
    def __init__(self, initial_threshold: float, adaptation_rate: float = 0.01):
        super().__init__(initial_threshold)
        self.initial_threshold = initial_threshold
        self.adaptation_rate = adaptation_rate
        self.performance_history: Dict[str, List[float]] = {}
    
    def evaluate(self, agent_id: str, metric: float) -> bool:
        """Evaluate adaptive constraint"""
        if agent_id not in self.performance_history:
            self.performance_history[agent_id] = []
        
        self.performance_history[agent_id].append(metric)
        
        # Keep recent history
        if len(self.performance_history[agent_id]) > 20:
            self.performance_history[agent_id] = self.performance_history[agent_id][-20:]
        
        # Adapt threshold based on recent performance
        if len(self.performance_history[agent_id]) >= 10:
            recent_avg = np.mean(self.performance_history[agent_id][-10:])
            # Increase threshold if performance is consistently high
            if recent_avg > self.threshold * 1.2:
                self.threshold = min(
                    self.initial_threshold * 1.5,
                    self.threshold + self.adaptation_rate
                )
            # Decrease threshold if performance is consistently low
            elif recent_avg < self.threshold * 0.8:
                self.threshold = max(
                    self.initial_threshold * 0.5,
                    self.threshold - self.adaptation_rate
                )
        
        return metric >= self.threshold


class ExampleAgent(Agent):
    """Simple agent for demonstration"""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.performance = 0.5
        self.trend = 0.01  # Gradually improving
    
    async def step(self):
        """Execute step"""
        # Simulate performance with some noise
        self.performance += self.trend + np.random.normal(0, 0.05)
        self.performance = np.clip(self.performance, 0, 1)
        
        return {"performance": self.performance}
    
    def get_performance_metric(self) -> float:
        return self.performance


async def main():
    """Run custom constraints example"""
    print("🔧 Custom Constraints Example")
    print("=" * 60)
    
    # Create different constraint types
    print("\n1. Temporal Constraint (must pass for 5 consecutive steps)")
    temporal_constraint = TemporalConstraint(threshold=0.6, consecutive_steps=5)
    runtime1 = Runtime(constraint=temporal_constraint, config=Config())
    
    agent1 = ExampleAgent("temporal_agent")
    runtime1.register_agent("temporal_agent", agent1)
    
    await runtime1.run(max_steps=20)
    stats1 = runtime1.get_statistics()
    print(f"   Final survival signal: {stats1['average_survival_signal']:.3f}")
    
    print("\n2. Statistical Constraint (average over 10 steps)")
    statistical_constraint = StatisticalConstraint(threshold=0.6, window_size=10)
    runtime2 = Runtime(constraint=statistical_constraint, config=Config())
    
    agent2 = ExampleAgent("statistical_agent")
    runtime2.register_agent("statistical_agent", agent2)
    
    await runtime2.run(max_steps=20)
    stats2 = runtime2.get_statistics()
    print(f"   Final survival signal: {stats2['average_survival_signal']:.3f}")
    
    print("\n3. Composite Constraint (AND logic)")
    constraint1 = SurvivalConstraint(threshold=0.5)
    constraint2 = SurvivalConstraint(threshold=0.7)
    composite_constraint = CompositeConstraint([constraint1, constraint2], logic="AND")
    runtime3 = Runtime(constraint=composite_constraint, config=Config())
    
    agent3 = ExampleAgent("composite_agent")
    runtime3.register_agent("composite_agent", agent3)
    
    await runtime3.run(max_steps=20)
    stats3 = runtime3.get_statistics()
    print(f"   Final survival signal: {stats3['average_survival_signal']:.3f}")
    
    print("\n4. Adaptive Constraint (threshold adjusts)")
    adaptive_constraint = AdaptiveConstraint(initial_threshold=0.6, adaptation_rate=0.01)
    runtime4 = Runtime(constraint=adaptive_constraint, config=Config())
    
    agent4 = ExampleAgent("adaptive_agent")
    runtime4.register_agent("adaptive_agent", agent4)
    
    initial_threshold = adaptive_constraint.threshold
    await runtime4.run(max_steps=30)
    final_threshold = adaptive_constraint.threshold
    stats4 = runtime4.get_statistics()
    
    print(f"   Initial threshold: {initial_threshold:.3f}")
    print(f"   Final threshold: {final_threshold:.3f}")
    print(f"   Final survival signal: {stats4['average_survival_signal']:.3f}")
    
    print("\n✅ Example completed!")
    print("\n💡 Key Takeaways:")
    print("   - Temporal constraints require sustained performance")
    print("   - Statistical constraints smooth out noise")
    print("   - Composite constraints enable complex logic")
    print("   - Adaptive constraints adjust difficulty dynamically")


if __name__ == "__main__":
    asyncio.run(main())

