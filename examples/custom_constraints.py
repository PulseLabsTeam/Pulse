"""
Custom Constraints Example

Demonstrates advanced constraint configurations including:
- Temporal constraints (time-based performance)
- Statistical constraints (mean, median, percentile, variance)
- Composite constraints
- Adaptive threshold learning
"""

import asyncio
import random
import numpy as np
from pulseos import Runtime, Agent, SurvivalConstraint


class AdaptiveAgent(Agent):
    """Simple agent that adapts to different constraint types."""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.state = random.uniform(0.0, 1.0)
        self.target = 0.8
        self.performance_history = []
    
    async def step(self) -> dict:
        """Execute one step."""
        # Simple learning behavior
        error = self.target - self.state
        
        if random.random() > self.exploration_rate:
            # Exploit
            self.state += self.learning_rate * error
        else:
            # Explore
            self.state += random.uniform(-0.1, 0.1)
        
        self.state = max(0.0, min(1.0, self.state))
        
        # Performance is based on how close to target
        performance = 1.0 - abs(error)
        self.performance_history.append(performance)
        
        return {
            "state": self.state,
            "performance": performance,
            "error": abs(error)
        }
    
    def get_performance_metric(self) -> float:
        """Get current performance metric."""
        if not self.performance_history:
            return 0.0
        return self.performance_history[-1]


async def demonstrate_temporal_constraint():
    """Demonstrate temporal constraint (must maintain performance over time)."""
    print("\n" + "="*70)
    print("TEMPORAL CONSTRAINT EXAMPLE")
    print("="*70)
    print("Constraint: Must maintain performance >= 0.7 over last 10 steps")
    
    constraint = SurvivalConstraint(
        threshold=0.7,
        constraint_type="temporal",
        temporal_window=10
    )
    
    runtime = Runtime(constraint=constraint)
    
    for i in range(10):
        agent = AdaptiveAgent(f"agent_{i}")
        runtime.register_agent(f"agent_{i}", agent)
    
    await runtime.run(max_steps=100)
    
    stats = runtime.get_statistics()
    print(f"\nResults:")
    print(f"  Average survival signal: {stats['average_survival_signal']:.3f}")
    print(f"  Final alpha: {stats['current_alpha']:.6f}")
    print(f"  Final epsilon: {stats['current_epsilon']:.3f}")


async def demonstrate_statistical_constraint():
    """Demonstrate statistical constraint (mean performance)."""
    print("\n" + "="*70)
    print("STATISTICAL CONSTRAINT EXAMPLE (MEAN)")
    print("="*70)
    print("Constraint: Average performance >= 0.75")
    
    constraint = SurvivalConstraint(
        threshold=0.75,
        constraint_type="statistical",
        statistical_mode="mean"
    )
    
    runtime = Runtime(constraint=constraint)
    
    for i in range(10):
        agent = AdaptiveAgent(f"agent_{i}")
        runtime.register_agent(f"agent_{i}", agent)
    
    await runtime.run(max_steps=100)
    
    stats = runtime.get_statistics()
    print(f"\nResults:")
    print(f"  Average survival signal: {stats['average_survival_signal']:.3f}")
    print(f"  Final alpha: {stats['current_alpha']:.6f}")
    print(f"  Final epsilon: {stats['current_epsilon']:.3f}")


async def demonstrate_percentile_constraint():
    """Demonstrate percentile constraint (90th percentile)."""
    print("\n" + "="*70)
    print("STATISTICAL CONSTRAINT EXAMPLE (90TH PERCENTILE)")
    print("="*70)
    print("Constraint: 90th percentile performance >= 0.8")
    
    constraint = SurvivalConstraint(
        threshold=0.8,
        constraint_type="statistical",
        statistical_mode="percentile"
    )
    
    runtime = Runtime(constraint=constraint)
    
    for i in range(10):
        agent = AdaptiveAgent(f"agent_{i}")
        runtime.register_agent(f"agent_{i}", agent)
    
    await runtime.run(max_steps=100)
    
    stats = runtime.get_statistics()
    print(f"\nResults:")
    print(f"  Average survival signal: {stats['average_survival_signal']:.3f}")
    print(f"  Final alpha: {stats['current_alpha']:.6f}")
    print(f"  Final epsilon: {stats['current_epsilon']:.3f}")


async def demonstrate_variance_constraint():
    """Demonstrate variance constraint (low variance = stable performance)."""
    print("\n" + "="*70)
    print("STATISTICAL CONSTRAINT EXAMPLE (VARIANCE)")
    print("="*70)
    print("Constraint: Performance variance <= 0.05 (stable performance)")
    
    constraint = SurvivalConstraint(
        threshold=0.05,  # Low variance threshold
        constraint_type="statistical",
        statistical_mode="variance"
    )
    
    runtime = Runtime(constraint=constraint)
    
    for i in range(10):
        agent = AdaptiveAgent(f"agent_{i}")
        runtime.register_agent(f"agent_{i}", agent)
    
    await runtime.run(max_steps=100)
    
    stats = runtime.get_statistics()
    print(f"\nResults:")
    print(f"  Average survival signal: {stats['average_survival_signal']:.3f}")
    print(f"  Final alpha: {stats['current_alpha']:.6f}")
    print(f"  Final epsilon: {stats['current_epsilon']:.3f}")


async def demonstrate_adaptive_threshold():
    """Demonstrate adaptive threshold learning."""
    print("\n" + "="*70)
    print("ADAPTIVE THRESHOLD EXAMPLE")
    print("="*70)
    print("Constraint: Threshold adapts based on agent performance")
    
    constraint = SurvivalConstraint(
        threshold=0.5,  # Initial threshold
        learning_rate=0.01  # Adaptation rate
    )
    
    runtime = Runtime(constraint=constraint)
    
    for i in range(10):
        agent = AdaptiveAgent(f"agent_{i}")
        runtime.register_agent(f"agent_{i}", agent)
    
    initial_threshold = constraint.threshold
    await runtime.run(max_steps=100)
    final_threshold = constraint.threshold
    
    stats = runtime.get_statistics()
    print(f"\nResults:")
    print(f"  Initial threshold: {initial_threshold:.3f}")
    print(f"  Final threshold: {final_threshold:.3f}")
    print(f"  Threshold change: {final_threshold - initial_threshold:+.3f}")
    print(f"  Average survival signal: {stats['average_survival_signal']:.3f}")


async def main():
    """Run all constraint examples."""
    print("🔧 PulseOS Custom Constraints Example")
    print("Demonstrating various constraint types and configurations\n")
    
    await demonstrate_temporal_constraint()
    await demonstrate_statistical_constraint()
    await demonstrate_percentile_constraint()
    await demonstrate_variance_constraint()
    await demonstrate_adaptive_threshold()
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED")
    print("="*70)
    print("\nKey Takeaways:")
    print("  • Temporal constraints require sustained performance")
    print("  • Statistical constraints evaluate aggregate behavior")
    print("  • Percentile constraints focus on top performers")
    print("  • Variance constraints promote stability")
    print("  • Adaptive thresholds adjust to agent capabilities")


if __name__ == "__main__":
    asyncio.run(main())

