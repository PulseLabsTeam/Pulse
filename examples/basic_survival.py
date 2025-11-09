"""
Basic Survival Example

Demonstrates basic usage of PulseOS with a simple agent.
"""

import asyncio
import random
from pulseos import Runtime, Config, Agent, SurvivalConstraint


class SimpleAgent(Agent):
    """Simple agent that learns to maximize a performance metric."""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.state = 0.0
        self.target = 1.0
    
    async def step(self) -> dict:
        """Execute one step of agent behavior."""
        # Simple learning: move state toward target
        error = self.target - self.state
        
        # Use learning rate for exploitation
        if random.random() > self.exploration_rate:
            # Exploit: move toward target
            self.state += self.learning_rate * error
        else:
            # Explore: random movement
            self.state += random.uniform(-0.1, 0.1)
        
        # Clamp state
        self.state = max(0.0, min(1.0, self.state))
        
        return {
            "state": self.state,
            "error": abs(error)
        }
    
    def get_performance_metric(self) -> float:
        """Get performance metric (higher is better)."""
        # Performance is inverse of error
        error = abs(self.target - self.state)
        return 1.0 - error


async def main():
    """Run basic survival example."""
    # Create survival constraint
    constraint = SurvivalConstraint(threshold=0.8)
    
    # Create runtime with default config
    runtime = Runtime(constraint=constraint)
    
    # Register agents
    for i in range(10):
        agent = SimpleAgent(f"agent_{i}")
        runtime.register_agent(f"agent_{i}", agent)
    
    # Run for 100 steps
    print("Running PulseOS runtime for 100 steps...")
    await runtime.run(max_steps=100)
    
    # Print statistics
    stats = runtime.get_statistics()
    print("\nRuntime Statistics:")
    print(f"  Steps: {stats['current_step']}")
    print(f"  Agents: {stats['agent_count']}")
    print(f"  Average Survival Signal: {stats['average_survival_signal']:.3f}")
    print(f"  Current Alpha: {stats['current_alpha']:.6f}")
    print(f"  Current Epsilon: {stats['current_epsilon']:.3f}")
    print(f"  NGCM Cache Hit Rate: {stats['ngcm_cache_hit_rate']:.2%}")
    
    # Print agent states
    print("\nAgent States:")
    for agent_id, agent in runtime.agents.items():
        metric = agent.get_performance_metric()
        print(f"  {agent_id}: metric={metric:.3f}, state={agent.state:.3f}")


if __name__ == "__main__":
    asyncio.run(main())

