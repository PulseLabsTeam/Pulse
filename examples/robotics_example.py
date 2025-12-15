"""
Robotics Example: Safety-Critical Control with Survival Pressure

Demonstrates PulseOS for robotics applications where agents must maintain
safety constraints while learning optimal control policies.
"""

import asyncio
import numpy as np
from pulseos import Runtime, Agent, SurvivalConstraint, Config


class RobotAgent(Agent):
    """
    Robot agent that learns to navigate while maintaining safety constraints.
    
    Safety constraints:
    - Velocity must stay below max_velocity
    - Distance to obstacles must stay above min_distance
    - Energy consumption must stay below max_energy
    """
    
    def __init__(self, agent_id: str, initial_position: np.ndarray):
        super().__init__(agent_id)
        self.position = initial_position.copy()
        self.velocity = np.array([0.0, 0.0])
        self.energy = 100.0
        self.max_velocity = 2.0
        self.min_distance = 0.5
        self.max_energy = 100.0
        self.target = np.array([10.0, 10.0])
        self.obstacles = [
            np.array([5.0, 5.0]),
            np.array([7.0, 3.0]),
            np.array([3.0, 7.0])
        ]
        
        # Learning parameters
        self.learning_rate = 0.01
        self.exploration_rate = 0.1
    
    async def step(self):
        """Execute one step of robot control"""
        # Compute distance to target
        distance_to_target = np.linalg.norm(self.target - self.position)
        
        # Compute distance to nearest obstacle
        distances_to_obstacles = [
            np.linalg.norm(self.position - obs) for obs in self.obstacles
        ]
        min_distance_to_obstacle = min(distances_to_obstacles)
        
        # Compute control action (simplified policy)
        direction_to_target = (self.target - self.position) / (distance_to_target + 1e-6)
        
        # Add exploration noise
        if np.random.random() < self.exploration_rate:
            direction_to_target += np.random.normal(0, 0.1, size=2)
        
        # Update velocity (respecting max_velocity constraint)
        desired_velocity = direction_to_target * self.max_velocity
        self.velocity = np.clip(desired_velocity, -self.max_velocity, self.max_velocity)
        
        # Update position
        self.position += self.velocity * 0.1  # dt = 0.1
        
        # Consume energy proportional to velocity magnitude
        energy_cost = np.linalg.norm(self.velocity) * 0.1
        self.energy = max(0, self.energy - energy_cost)
        
        return {
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "energy": self.energy,
            "distance_to_target": distance_to_target,
            "min_distance_to_obstacle": min_distance_to_obstacle
        }
    
    def get_performance_metric(self) -> float:
        """
        Performance metric combines:
        - Progress toward target (higher is better)
        - Safety margin (higher is better)
        - Energy efficiency (higher is better)
        """
        # Progress metric (0 to 1)
        initial_distance = np.linalg.norm(self.target - np.array([0.0, 0.0]))
        current_distance = np.linalg.norm(self.target - self.position)
        progress = 1.0 - (current_distance / initial_distance)
        progress = np.clip(progress, 0, 1)
        
        # Safety metric (0 to 1)
        distances_to_obstacles = [
            np.linalg.norm(self.position - obs) for obs in self.obstacles
        ]
        min_distance_to_obstacle = min(distances_to_obstacles)
        safety = min(1.0, min_distance_to_obstacle / self.min_distance)
        
        # Energy metric (0 to 1)
        energy_metric = self.energy / self.max_energy
        
        # Combined performance (weighted average)
        performance = 0.5 * progress + 0.3 * safety + 0.2 * energy_metric
        
        return performance


async def main():
    """Run robotics example"""
    print("🤖 Robotics Example: Safety-Critical Control")
    print("=" * 60)
    
    # Create survival constraint
    # Robots must maintain performance above 0.6 to "survive"
    constraint = SurvivalConstraint(threshold=0.6)
    
    # Configure runtime with snapshot support for rollback
    config = Config(
        snapshot_interval=1.0,
        max_snapshots=10,
        metrics_enabled=True
    )
    
    runtime = Runtime(constraint=constraint, config=config)
    
    # Create robot agents
    num_robots = 5
    for i in range(num_robots):
        initial_position = np.random.uniform(-2, 2, size=2)
        robot = RobotAgent(f"robot_{i}", initial_position)
        runtime.register_agent(f"robot_{i}", robot)
    
    print(f"Created {num_robots} robot agents")
    print("Running simulation...")
    print()
    
    # Run simulation
    await runtime.run(max_steps=100)
    
    # Print results
    stats = runtime.get_statistics()
    print("\n📊 Simulation Results:")
    print(f"  Final Step: {stats['current_step']}")
    print(f"  Average Survival Signal: {stats['average_survival_signal']:.3f}")
    print(f"  Agents Survived: {stats['agent_count']}")
    
    # Print individual robot status
    print("\n🤖 Robot Status:")
    for agent_id, agent in runtime.agents.items():
        if isinstance(agent, RobotAgent):
            distance_to_target = np.linalg.norm(agent.target - agent.position)
            performance = agent.get_performance_metric()
            print(f"  {agent_id}:")
            print(f"    Position: [{agent.position[0]:.2f}, {agent.position[1]:.2f}]")
            print(f"    Distance to Target: {distance_to_target:.2f}")
            print(f"    Performance: {performance:.3f}")
            print(f"    Energy: {agent.energy:.1f}")
    
    print("\n✅ Example completed!")


if __name__ == "__main__":
    asyncio.run(main())

