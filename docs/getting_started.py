"""
Getting Started Tutorial

A step-by-step guide to using PulseOS for survival-pressure learning.
"""

import asyncio
from pulseos import Runtime, Agent, SurvivalConstraint


# ============================================================================
# Tutorial Step 1: Creating Your First Agent
# ============================================================================

class MyFirstAgent(Agent):
    """
    A simple agent that learns to maximize a performance metric.
    
    Every agent must implement:
    1. step() - What the agent does each iteration
    2. get_performance_metric() - How well the agent is performing
    """
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.value = 0.0  # Agent's internal state
        self.target = 0.8  # Target performance
    
    async def step(self) -> dict:
        """
        Execute one step of agent behavior.
        
        Returns:
            Dictionary with step results (for logging/debugging)
        """
        # Simple learning: move value toward target
        error = self.target - self.value
        
        # Use learning rate (automatically updated by PulseOS)
        self.value += self.learning_rate * error
        
        # Clamp value to valid range
        self.value = max(0.0, min(1.0, self.value))
        
        return {
            "value": self.value,
            "error": abs(error)
        }
    
    def get_performance_metric(self) -> float:
        """
        Return performance metric (0-1 scale).
        
        PulseOS uses this to:
        - Evaluate if agent meets survival threshold
        - Adapt learning rate and exploration rate
        - Compute survival pressure signal
        """
        error = abs(self.target - self.value)
        return 1.0 - error  # Higher is better


# ============================================================================
# Tutorial Step 2: Setting Up a Runtime
# ============================================================================

async def tutorial_step_2():
    """Demonstrate basic runtime setup."""
    print("Step 2: Setting Up Runtime")
    print("-" * 50)
    
    # Create a survival constraint
    # Agents must maintain performance >= 0.7
    constraint = SurvivalConstraint(threshold=0.7)
    
    # Create runtime with the constraint
    runtime = Runtime(constraint=constraint)
    
    # Register an agent
    agent = MyFirstAgent("agent_1")
    runtime.register_agent("agent_1", agent)
    
    # Run for 50 steps
    await runtime.run(max_steps=50)
    
    # Check results
    stats = runtime.get_statistics()
    print(f"  Steps completed: {stats['current_step']}")
    print(f"  Survival signal: {stats['average_survival_signal']:.3f}")
    print(f"  Agent performance: {agent.get_performance_metric():.3f}\n")


# ============================================================================
# Tutorial Step 3: Multiple Agents
# ============================================================================

async def tutorial_step_3():
    """Demonstrate multiple agents."""
    print("Step 3: Multiple Agents")
    print("-" * 50)
    
    constraint = SurvivalConstraint(threshold=0.7)
    runtime = Runtime(constraint=constraint)
    
    # Register multiple agents
    for i in range(5):
        agent = MyFirstAgent(f"agent_{i}")
        runtime.register_agent(f"agent_{i}", agent)
    
    await runtime.run(max_steps=50)
    
    stats = runtime.get_statistics()
    print(f"  Agents: {stats['agent_count']}")
    print(f"  Average survival signal: {stats['average_survival_signal']:.3f}\n")


# ============================================================================
# Tutorial Step 4: Different Constraint Types
# ============================================================================

async def tutorial_step_4():
    """Demonstrate different constraint types."""
    print("Step 4: Constraint Types")
    print("-" * 50)
    
    # Simple constraint (default)
    simple_constraint = SurvivalConstraint(threshold=0.7)
    print("  Simple constraint: performance >= 0.7")
    
    # Temporal constraint (must maintain over time)
    temporal_constraint = SurvivalConstraint(
        threshold=0.7,
        constraint_type="temporal",
        temporal_window=10  # Last 10 steps
    )
    print("  Temporal constraint: maintain >= 0.7 over last 10 steps")
    
    # Statistical constraint (average performance)
    statistical_constraint = SurvivalConstraint(
        threshold=0.7,
        constraint_type="statistical",
        statistical_mode="mean"
    )
    print("  Statistical constraint: average performance >= 0.7")
    
    # Use temporal constraint
    runtime = Runtime(constraint=temporal_constraint)
    agent = MyFirstAgent("agent_1")
    runtime.register_agent("agent_1", agent)
    await runtime.run(max_steps=50)
    
    print("  ✓ Temporal constraint applied\n")


# ============================================================================
# Tutorial Step 5: Accessing Runtime Statistics
# ============================================================================

async def tutorial_step_5():
    """Demonstrate accessing runtime statistics."""
    print("Step 5: Runtime Statistics")
    print("-" * 50)
    
    constraint = SurvivalConstraint(threshold=0.7)
    runtime = Runtime(constraint=constraint)
    
    for i in range(3):
        agent = MyFirstAgent(f"agent_{i}")
        runtime.register_agent(f"agent_{i}", agent)
    
    await runtime.run(max_steps=50)
    
    stats = runtime.get_statistics()
    
    print("  Available statistics:")
    print(f"    • Current step: {stats['current_step']}")
    print(f"    • Agent count: {stats['agent_count']}")
    print(f"    • Average survival signal: {stats['average_survival_signal']:.3f}")
    print(f"    • Current learning rate (alpha): {stats['current_alpha']:.6f}")
    print(f"    • Current exploration rate (epsilon): {stats['current_epsilon']:.3f}")
    print(f"    • NGCM cache hit rate: {stats.get('ngcm_cache_hit_rate', 0):.2%}\n")


# ============================================================================
# Tutorial Step 6: Custom Agent with Exploration
# ============================================================================

class ExploringAgent(Agent):
    """Agent that uses exploration rate for better learning."""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.value = 0.0
        self.target = 0.8
    
    async def step(self) -> dict:
        import random
        
        error = self.target - self.value
        
        # Use exploration rate (automatically updated by PulseOS)
        if random.random() > self.exploration_rate:
            # Exploit: move toward target
            self.value += self.learning_rate * error
        else:
            # Explore: random movement
            self.value += random.uniform(-0.1, 0.1)
        
        self.value = max(0.0, min(1.0, self.value))
        
        return {"value": self.value}
    
    def get_performance_metric(self) -> float:
        error = abs(self.target - self.value)
        return 1.0 - error


async def tutorial_step_6():
    """Demonstrate exploration."""
    print("Step 6: Exploration vs Exploitation")
    print("-" * 50)
    
    constraint = SurvivalConstraint(threshold=0.7)
    runtime = Runtime(constraint=constraint)
    
    agent = ExploringAgent("agent_1")
    runtime.register_agent("agent_1", agent)
    
    await runtime.run(max_steps=100)
    
    stats = runtime.get_statistics()
    print(f"  Final exploration rate: {stats['current_epsilon']:.3f}")
    print(f"  Final learning rate: {stats['current_alpha']:.6f}")
    print(f"  Agent performance: {agent.get_performance_metric():.3f}\n")


# ============================================================================
# Main Tutorial
# ============================================================================

async def main():
    """Run complete tutorial."""
    print("="*70)
    print("PulseOS Getting Started Tutorial")
    print("="*70)
    print()
    
    await tutorial_step_2()
    await tutorial_step_3()
    await tutorial_step_4()
    await tutorial_step_5()
    await tutorial_step_6()
    
    print("="*70)
    print("Tutorial Complete!")
    print("="*70)
    print("\nNext Steps:")
    print("  1. Check out examples/ directory for more advanced examples")
    print("  2. Read TECHNICAL.md for algorithm details")
    print("  3. Explore custom constraints and configurations")
    print("  4. Review performance tuning guide")


if __name__ == "__main__":
    asyncio.run(main())

