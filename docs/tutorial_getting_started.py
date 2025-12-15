"""
Tutorial: Getting Started with PulseOS

A step-by-step guide to using PulseOS for survival-pressure learning.
"""

import asyncio
from pulseos import Runtime, Config, Agent, SurvivalConstraint


# Step 1: Define Your Agent
class MyFirstAgent(Agent):
    """A simple agent that learns to maximize performance."""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.state = 0.0  # Agent's internal state
        self.target = 1.0  # Target value
    
    async def step(self) -> dict:
        """
        Execute one step of agent behavior.
        
        This method is called by the runtime on each step.
        You should update your agent's state here.
        """
        # Compute error from target
        error = self.target - self.state
        
        # Use learning rate (automatically adjusted by PulseOS)
        # Higher learning rate = faster learning
        self.state += self.learning_rate * error
        
        # Clamp state to valid range
        self.state = max(0.0, min(1.0, self.state))
        
        return {
            "state": self.state,
            "error": abs(error)
        }
    
    def get_performance_metric(self) -> float:
        """
        Return agent's performance metric.
        
        This should be a value between 0 and 1, where:
        - 0 = worst performance
        - 1 = best performance
        
        PulseOS uses this to compute survival pressure.
        """
        # Performance is inverse of error
        error = abs(self.target - self.state)
        return 1.0 - error


async def tutorial_step_1():
    """Step 1: Create a simple agent."""
    print("=== Step 1: Creating Your First Agent ===\n")
    
    agent = MyFirstAgent("agent1")
    print(f"Created agent: {agent.agent_id}")
    print(f"Initial state: {agent.state}")
    print(f"Initial learning rate: {agent.learning_rate}")
    print()


async def tutorial_step_2():
    """Step 2: Create a survival constraint."""
    print("=== Step 2: Creating a Survival Constraint ===\n")
    
    # Survival constraint defines the performance threshold
    # Agents must maintain performance above this threshold
    constraint = SurvivalConstraint(threshold=0.8)
    
    print(f"Survival threshold: {constraint.threshold}")
    print("Agents with performance < 0.8 will experience survival pressure")
    print()


async def tutorial_step_3():
    """Step 3: Create and configure runtime."""
    print("=== Step 3: Creating Runtime ===\n")
    
    constraint = SurvivalConstraint(threshold=0.8)
    
    # Create runtime with default configuration
    runtime = Runtime(constraint=constraint)
    
    print("Runtime created with default configuration")
    print(f"Snapshot interval: {runtime.config.snapshot_interval}s")
    print(f"Max agents: {runtime.config.max_agents}")
    print()


async def tutorial_step_4():
    """Step 4: Register agents and run."""
    print("=== Step 4: Running Your First Simulation ===\n")
    
    constraint = SurvivalConstraint(threshold=0.8)
    runtime = Runtime(constraint=constraint)
    
    # Register agents
    for i in range(5):
        agent = MyFirstAgent(f"agent_{i}")
        runtime.register_agent(f"agent_{i}", agent)
        print(f"Registered: {agent.agent_id}")
    
    print("\nRunning simulation for 50 steps...")
    await runtime.run(max_steps=50)
    
    # Get statistics
    stats = runtime.get_statistics()
    print("\n=== Results ===")
    print(f"Steps executed: {stats['current_step']}")
    print(f"Agents: {stats['agent_count']}")
    print(f"Average survival signal: {stats['average_survival_signal']:.3f}")
    print(f"Current learning rate (alpha): {stats['current_alpha']:.6f}")
    print(f"Current exploration rate (epsilon): {stats['current_epsilon']:.3f}")
    print()


async def tutorial_step_5():
    """Step 5: Custom configuration."""
    print("=== Step 5: Custom Configuration ===\n")
    
    # Create custom configuration
    config = Config(
        snapshot_interval=0.5,  # Snapshot every 0.5 seconds
        max_agents=10,  # Limit to 10 agents
        gradient_cache_size=256,  # Cache size for gradient computation
        threshold_detection_interval=0.1  # Check thresholds every 100ms
    )
    
    constraint = SurvivalConstraint(threshold=0.8)
    runtime = Runtime(constraint=constraint, config=config)
    
    print("Runtime created with custom configuration:")
    print(f"  Snapshot interval: {config.snapshot_interval}s")
    print(f"  Max agents: {config.max_agents}")
    print(f"  Cache size: {config.gradient_cache_size}")
    print()


async def tutorial_step_6():
    """Step 6: Understanding survival pressure."""
    print("=== Step 6: Understanding Survival Pressure ===\n")
    
    constraint = SurvivalConstraint(threshold=0.8)
    runtime = Runtime(constraint=constraint)
    
    # Create agents with different initial states
    agents_data = [
        ("good_agent", 0.9),  # Already above threshold
        ("ok_agent", 0.7),    # Below threshold
        ("poor_agent", 0.3)   # Well below threshold
    ]
    
    for agent_id, initial_state in agents_data:
        agent = MyFirstAgent(agent_id)
        agent.state = initial_state
        runtime.register_agent(agent_id, agent)
    
    print("Running simulation to observe survival pressure...")
    await runtime.run(max_steps=20)
    
    print("\nAgent Performance After Learning:")
    for agent_id, agent in runtime.agents.items():
        metric = agent.get_performance_metric()
        status = "✓ SURVIVING" if metric >= 0.8 else "⚠ PRESSURE"
        print(f"  {agent_id}: {metric:.3f} {status}")
    
    print("\nAgents below threshold experience survival pressure,")
    print("which causes PulseOS to increase learning rate and exploration.")
    print()


async def main():
    """Run all tutorial steps."""
    print("=" * 60)
    print("PulseOS Getting Started Tutorial")
    print("=" * 60)
    print()
    
    await tutorial_step_1()
    await tutorial_step_2()
    await tutorial_step_3()
    await tutorial_step_4()
    await tutorial_step_5()
    await tutorial_step_6()
    
    print("=" * 60)
    print("Tutorial Complete!")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. Try modifying the agent's step() method")
    print("2. Experiment with different threshold values")
    print("3. Add more agents and observe coordination")
    print("4. Check out other examples in the examples/ directory")


if __name__ == "__main__":
    asyncio.run(main())

