"""
Game AI Example - Multi-Agent Strategy

Demonstrates PulseOS for game AI with competitive multi-agent scenarios,
emergent strategies, and adaptive behavior.
"""

import asyncio
import random
import numpy as np
from pulseos import Runtime, Agent, SurvivalConstraint


class GameAgent(Agent):
    """
    Game agent with strategic behavior.
    
    Simulates a game agent that must:
    - Compete against other agents
    - Adapt strategy based on opponents
    - Maximize score/performance
    - Learn from experience
    """
    
    def __init__(self, agent_id: str, initial_position: tuple = (0.0, 0.0)):
        super().__init__(agent_id)
        self.position = np.array(initial_position, dtype=np.float32)
        self.score = 0
        self.resources = 100
        
        # Strategy parameters
        self.aggressiveness = 0.5  # 0 = defensive, 1 = aggressive
        self.cooperation = 0.5     # 0 = competitive, 1 = cooperative
        self.efficiency = 0.5      # Resource usage efficiency
        
        # Game state
        self.targets = [
            np.array([5.0, 5.0]),
            np.array([-5.0, 5.0]),
            np.array([5.0, -5.0]),
            np.array([-5.0, -5.0])
        ]
        self.targets_captured = set()
        
        # Performance tracking
        self.actions_taken = 0
        self.successful_actions = 0
    
    async def step(self) -> dict:
        """Execute one step of game agent behavior."""
        # Find nearest uncaptured target
        nearest_target = None
        nearest_distance = float('inf')
        
        for i, target in enumerate(self.targets):
            if i not in self.targets_captured:
                distance = np.linalg.norm(self.position - target)
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_target = (i, target)
        
        # Decide action based on strategy
        if random.random() > self.exploration_rate:
            # Exploit: follow current strategy
            if nearest_target:
                target_idx, target_pos = nearest_target
                
                # Move toward target
                direction = target_pos - self.position
                distance = np.linalg.norm(direction)
                
                if distance > 0.1:
                    direction = direction / distance
                    
                    # Speed based on aggressiveness
                    speed = 0.5 + 0.5 * self.aggressiveness
                    movement = direction * speed * self.learning_rate
                    
                    self.position += movement
                    self.resources -= 1  # Cost of movement
                    
                    # Check if target reached
                    if distance < 0.5:
                        self.targets_captured.add(target_idx)
                        self.score += 100
                        self.successful_actions += 1
                        self.resources += 50  # Reward
                else:
                    # Already at target
                    self.targets_captured.add(target_idx)
                    self.score += 100
                    self.successful_actions += 1
            else:
                # All targets captured, explore
                self.position += np.random.uniform(-1, 1, 2) * self.learning_rate
                self.resources -= 1
        else:
            # Explore: random movement
            self.position += np.random.uniform(-2, 2, 2) * self.learning_rate
            self.resources -= 1
            
            # Randomly adjust strategy
            if random.random() < 0.1:
                self.aggressiveness = max(0, min(1, self.aggressiveness + random.uniform(-0.1, 0.1)))
                self.cooperation = max(0, min(1, self.cooperation + random.uniform(-0.1, 0.1)))
        
        self.actions_taken += 1
        
        # Resource management
        if self.resources < 0:
            self.resources = 0
            self.score -= 10  # Penalty for resource depletion
        
        # Efficiency affects resource consumption
        resource_cost = 1.0 / (0.5 + 0.5 * self.efficiency)
        self.resources = max(0, self.resources - resource_cost + 1)
        
        return {
            "position": tuple(self.position),
            "score": self.score,
            "resources": self.resources,
            "targets_captured": len(self.targets_captured),
            "aggressiveness": self.aggressiveness,
            "cooperation": self.cooperation
        }
    
    def get_performance_metric(self) -> float:
        """
        Performance metric combining:
        - Score (higher is better)
        - Resource efficiency (more resources is better)
        - Action success rate (higher is better)
        """
        # Score component (normalized)
        score_component = min(1.0, self.score / 400.0)  # Max 4 targets * 100
        
        # Resource component
        resource_component = min(1.0, self.resources / 100.0)
        
        # Success rate component
        success_rate = (
            self.successful_actions / self.actions_taken
            if self.actions_taken > 0 else 0.0
        )
        
        # Weighted combination
        performance = 0.5 * score_component + 0.3 * resource_component + 0.2 * success_rate
        
        return min(1.0, max(0.0, performance))


async def main():
    """Run game AI example with competitive agents."""
    print("🎮 PulseOS Game AI Example - Multi-Agent Strategy\n")
    
    # Create survival constraint for competitive environment
    constraint = SurvivalConstraint(
        threshold=0.6,  # Moderate threshold for competitive play
        constraint_type="temporal",
        temporal_window=5  # Must maintain performance
    )
    
    # Create runtime
    runtime = Runtime(constraint=constraint)
    
    # Create game agents
    num_agents = 25
    print(f"Creating {num_agents} game agents...")
    
    for i in range(num_agents):
        # Random starting positions
        start_pos = (
            random.uniform(-10.0, 10.0),
            random.uniform(-10.0, 10.0)
        )
        agent = GameAgent(f"agent_{i}", start_pos)
        runtime.register_agent(f"agent_{i}", agent)
    
    print(f"Running game simulation for 200 steps...")
    print("Agents compete to capture targets and maximize score.\n")
    
    # Run simulation
    await runtime.run(max_steps=200)
    
    # Print results
    stats = runtime.get_statistics()
    print("\n" + "="*70)
    print("GAME SIMULATION RESULTS")
    print("="*70)
    print(f"Steps completed: {stats['current_step']}")
    print(f"Average survival signal: {stats['average_survival_signal']:.3f}")
    print(f"Final learning rate (alpha): {stats['current_alpha']:.6f}")
    print(f"Final exploration rate (epsilon): {stats['current_epsilon']:.3f}")
    
    print("\nAgent Rankings:")
    print("-" * 70)
    
    # Sort agents by performance
    agent_performances = [
        (agent_id, agent.get_performance_metric(), agent.score, agent.resources)
        for agent_id, agent in runtime.agents.items()
    ]
    agent_performances.sort(key=lambda x: x[1], reverse=True)
    
    successful_agents = 0
    for rank, (agent_id, metric, score, resources) in enumerate(agent_performances, 1):
        status = "🏆" if rank == 1 else "✅" if metric >= 0.6 else "⚠️"
        if metric >= 0.6:
            successful_agents += 1
        
        print(f"{status} Rank {rank:2d} | {agent_id:12s} | "
              f"Metric: {metric:.3f} | Score: {score:4d} | Resources: {resources:5.1f}")
    
    print(f"\n✅ {successful_agents}/{num_agents} agents met performance target")
    print(f"🏆 Top performer: {agent_performances[0][0]} with metric {agent_performances[0][1]:.3f}")


if __name__ == "__main__":
    asyncio.run(main())

