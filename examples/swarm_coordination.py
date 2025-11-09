"""
Swarm Coordination Example

Demonstrates PulseOS with 1000+ agents in a swarm configuration.
"""

import asyncio
import random
import numpy as np
from pulseos import Runtime, Config, Agent, SurvivalConstraint


class SwarmAgent(Agent):
    """Agent in a swarm that coordinates with others."""
    
    def __init__(self, agent_id: str, swarm_size: int):
        super().__init__(agent_id)
        self.swarm_size = swarm_size
        
        # Agent position in 2D space
        self.x = random.uniform(0, 1)
        self.y = random.uniform(0, 1)
        
        # Target position (changes over time)
        self.target_x = random.uniform(0, 1)
        self.target_y = random.uniform(0, 1)
        
        # Velocity
        self.vx = 0.0
        self.vy = 0.0
    
    async def step(self) -> dict:
        """Execute one step of swarm behavior."""
        # Compute error
        error_x = self.target_x - self.x
        error_y = self.target_y - self.y
        error_magnitude = np.sqrt(error_x**2 + error_y**2)
        
        # Update velocity based on learning rate
        if random.random() > self.exploration_rate:
            # Exploit: move toward target
            self.vx += self.learning_rate * error_x
            self.vy += self.learning_rate * error_y
        else:
            # Explore: random movement
            self.vx += random.uniform(-0.1, 0.1)
            self.vy += random.uniform(-0.1, 0.1)
        
        # Update position
        self.x += self.vx
        self.y += self.vy
        
        # Wrap around boundaries
        self.x = self.x % 1.0
        self.y = self.y % 1.0
        
        # Damping
        self.vx *= 0.95
        self.vy *= 0.95
        
        # Occasionally update target
        if random.random() < 0.01:
            self.target_x = random.uniform(0, 1)
            self.target_y = random.uniform(0, 1)
        
        return {
            "x": self.x,
            "y": self.y,
            "error": error_magnitude
        }
    
    def get_performance_metric(self) -> float:
        """Get performance metric based on distance to target."""
        error_x = self.target_x - self.x
        error_y = self.target_y - self.y
        error_magnitude = np.sqrt(error_x**2 + error_y**2)
        
        # Performance is inverse of error (normalized)
        return max(0.0, 1.0 - error_magnitude)


async def main():
    """Run swarm coordination example."""
    # Create survival constraint
    constraint = SurvivalConstraint(threshold=0.7)
    
    # Configure for large swarm
    config = Config(
        max_agents=2000,
        parallel_updates=True,
        update_batch_size=100,
        snapshot_interval=5.0
    )
    
    # Create runtime
    runtime = Runtime(constraint=constraint, config=config)
    
    # Register swarm agents
    swarm_size = 1000
    print(f"Registering {swarm_size} swarm agents...")
    
    for i in range(swarm_size):
        agent = SwarmAgent(f"swarm_agent_{i}", swarm_size)
        runtime.register_agent(f"swarm_agent_{i}", agent)
    
    print(f"Running PulseOS runtime with {swarm_size} agents for 200 steps...")
    
    # Run for 200 steps
    await runtime.run(max_steps=200)
    
    # Print statistics
    stats = runtime.get_statistics()
    print("\nRuntime Statistics:")
    print(f"  Steps: {stats['current_step']}")
    print(f"  Agents: {stats['agent_count']}")
    print(f"  Uptime: {stats['uptime']:.2f}s")
    print(f"  Average Step Duration: {stats['average_step_duration']*1000:.3f}ms")
    print(f"  Average Survival Signal: {stats['average_survival_signal']:.3f}")
    print(f"  NGCM Cache Hit Rate: {stats['ngcm_cache_hit_rate']:.2%}")
    print(f"  Snapshot Count: {stats['snapshot_count']}")
    
    # Sample agent metrics
    print("\nSample Agent Metrics:")
    sample_agents = list(runtime.agents.items())[:5]
    for agent_id, agent in sample_agents:
        metric = agent.get_performance_metric()
        print(f"  {agent_id}: metric={metric:.3f}, pos=({agent.x:.2f}, {agent.y:.2f})")


if __name__ == "__main__":
    asyncio.run(main())

