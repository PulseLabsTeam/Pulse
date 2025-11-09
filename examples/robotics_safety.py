"""
Robotics Example - Safety-Critical Control

Demonstrates PulseOS for robotics applications with safety constraints,
real-time performance requirements, and multi-objective optimization.
"""

import asyncio
import random
import numpy as np
from pulseos import Runtime, Agent, SurvivalConstraint


class RobotAgent(Agent):
    """
    Robot agent with safety-critical constraints.
    
    Simulates a robot that must:
    - Maintain safe distance from obstacles
    - Stay within operational bounds
    - Achieve target performance
    """
    
    def __init__(self, agent_id: str, initial_position: tuple = (0.0, 0.0)):
        super().__init__(agent_id)
        self.position = np.array(initial_position, dtype=np.float32)
        self.velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.target = np.array([10.0, 10.0], dtype=np.float32)
        self.obstacles = [
            np.array([3.0, 3.0]),
            np.array([7.0, 7.0]),
            np.array([5.0, 8.0])
        ]
        self.safety_distance = 1.0
        self.max_speed = 2.0
        self.collisions = 0
        self.distance_to_target = np.linalg.norm(self.target - self.position)
    
    async def step(self) -> dict:
        """Execute one step of robot control."""
        # Compute direction to target
        direction = self.target - self.position
        distance = np.linalg.norm(direction)
        
        if distance > 0.1:
            direction = direction / distance
        
        # Check for obstacles
        obstacle_force = np.array([0.0, 0.0])
        min_obstacle_distance = float('inf')
        
        for obstacle in self.obstacles:
            obstacle_vec = self.position - obstacle
            obstacle_dist = np.linalg.norm(obstacle_vec)
            min_obstacle_distance = min(min_obstacle_distance, obstacle_dist)
            
            if obstacle_dist < self.safety_distance * 2:
                # Repulsion force
                if obstacle_dist > 0.01:
                    obstacle_force += (obstacle_vec / obstacle_dist) * (1.0 / obstacle_dist)
        
        # Combine target attraction and obstacle avoidance
        if random.random() > self.exploration_rate:
            # Exploit: move toward target while avoiding obstacles
            desired_velocity = direction * self.max_speed - obstacle_force * 0.5
        else:
            # Explore: random movement with obstacle awareness
            desired_velocity = np.array([
                random.uniform(-self.max_speed, self.max_speed),
                random.uniform(-self.max_speed, self.max_speed)
            ]) - obstacle_force * 0.3
        
        # Update velocity with learning rate
        self.velocity = (1 - self.learning_rate) * self.velocity + self.learning_rate * desired_velocity
        
        # Limit speed
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = self.velocity / speed * self.max_speed
        
        # Update position
        self.position += self.velocity * 0.1  # Time step
        
        # Check collisions
        for obstacle in self.obstacles:
            if np.linalg.norm(self.position - obstacle) < self.safety_distance:
                self.collisions += 1
                # Push away from obstacle
                push_direction = self.position - obstacle
                if np.linalg.norm(push_direction) > 0.01:
                    push_direction = push_direction / np.linalg.norm(push_direction)
                    self.position = obstacle + push_direction * self.safety_distance
        
        # Update distance to target
        self.distance_to_target = np.linalg.norm(self.target - self.position)
        
        return {
            "position": tuple(self.position),
            "distance_to_target": self.distance_to_target,
            "collisions": self.collisions,
            "min_obstacle_distance": min_obstacle_distance
        }
    
    def get_performance_metric(self) -> float:
        """
        Performance metric combining:
        - Distance to target (closer is better)
        - Safety (no collisions)
        - Speed (reasonable movement)
        """
        # Normalize distance (0-1 scale, 1 = at target)
        distance_score = 1.0 / (1.0 + self.distance_to_target / 10.0)
        
        # Safety score (penalize collisions)
        safety_score = max(0.0, 1.0 - self.collisions * 0.2)
        
        # Movement score (penalize being stuck)
        movement_score = min(1.0, np.linalg.norm(self.velocity) / self.max_speed)
        
        # Weighted combination
        performance = 0.5 * distance_score + 0.3 * safety_score + 0.2 * movement_score
        
        return min(1.0, max(0.0, performance))


async def main():
    """Run robotics example with safety constraints."""
    print("🤖 PulseOS Robotics Example - Safety-Critical Control\n")
    
    # Create survival constraint with high threshold for safety
    constraint = SurvivalConstraint(
        threshold=0.7,  # High performance required
        constraint_type="temporal",
        temporal_window=10  # Must maintain performance over time
    )
    
    # Create runtime
    runtime = Runtime(constraint=constraint)
    
    # Create robot swarm
    num_robots = 20
    print(f"Creating {num_robots} robots...")
    
    for i in range(num_robots):
        # Random starting positions
        start_pos = (
            random.uniform(0.0, 5.0),
            random.uniform(0.0, 5.0)
        )
        robot = RobotAgent(f"robot_{i}", start_pos)
        runtime.register_agent(f"robot_{i}", robot)
    
    print(f"Running simulation for 200 steps...")
    print("Robots must navigate to target while avoiding obstacles.\n")
    
    # Run simulation
    await runtime.run(max_steps=200)
    
    # Print results
    stats = runtime.get_statistics()
    print("\n" + "="*60)
    print("SIMULATION RESULTS")
    print("="*60)
    print(f"Steps completed: {stats['current_step']}")
    print(f"Average survival signal: {stats['average_survival_signal']:.3f}")
    print(f"Final learning rate (alpha): {stats['current_alpha']:.6f}")
    print(f"Final exploration rate (epsilon): {stats['current_epsilon']:.3f}")
    
    print("\nRobot Performance:")
    print("-" * 60)
    
    successful_robots = 0
    for agent_id, agent in runtime.agents.items():
        metric = agent.get_performance_metric()
        distance = agent.distance_to_target
        collisions = agent.collisions
        
        status = "✅" if metric >= 0.7 else "⚠️"
        if distance < 1.0:
            successful_robots += 1
        
        print(f"{status} {agent_id:12s} | Metric: {metric:.3f} | "
              f"Distance: {distance:.2f} | Collisions: {collisions}")
    
    print(f"\n✅ {successful_robots}/{num_robots} robots reached target")
    print(f"📊 Average performance: {np.mean([a.get_performance_metric() for a in runtime.agents.values()]):.3f}")


if __name__ == "__main__":
    asyncio.run(main())

