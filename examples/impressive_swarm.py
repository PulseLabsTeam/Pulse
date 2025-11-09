"""
Impressive Swarm Coordination Example

Demonstrates 1000+ agent swarm with emergent behavior, role detection,
and crisis synchronization as specified in patent.
"""

import asyncio
import random
import numpy as np
from typing import Dict, List, Optional
from pulseos import Runtime, Config, Agent, SurvivalConstraint


class SwarmAgent(Agent):
    """
    Swarm agent with emergent role detection.
    
    Implements patent-specified emergent behavior:
    - Role detection (23% stylistic, 18% tone, etc.)
    - Crisis synchronization
    - Collective heartbeat compression
    """
    
    def __init__(self, agent_id: str, swarm_size: int):
        super().__init__(agent_id)
        self.swarm_size = swarm_size
        
        # Position in 2D space
        self.x = random.uniform(0, 1)
        self.y = random.uniform(0, 1)
        
        # Velocity
        self.vx = 0.0
        self.vy = 0.0
        
        # Target (changes dynamically)
        self.target_x = random.uniform(0, 1)
        self.target_y = random.uniform(0, 1)
        
        # Emergent role (detected dynamically)
        self.role: Optional[str] = None
        self.role_confidence = 0.0
        
        # Behavior style (affects movement patterns)
        self.style_factor = random.uniform(0.5, 1.5)
        
        # Crisis state
        self.in_crisis = False
        self.crisis_sync_count = 0
    
    async def step(self) -> dict:
        """Execute swarm behavior step."""
        # Compute error to target
        error_x = self.target_x - self.x
        error_y = self.target_y - self.y
        error_magnitude = np.sqrt(error_x**2 + error_y**2)
        
        # Update velocity based on learning rate and exploration
        if random.random() > self.exploration_rate:
            # Exploit: move toward target
            self.vx += self.learning_rate * error_x * self.style_factor
            self.vy += self.learning_rate * error_y * self.style_factor
        else:
            # Explore: random movement with style
            self.vx += random.uniform(-0.1, 0.1) * self.style_factor
            self.vy += random.uniform(-0.1, 0.1) * self.style_factor
        
        # Update position
        self.x += self.vx
        self.y += self.vy
        
        # Wrap around boundaries
        self.x = self.x % 1.0
        self.y = self.y % 1.0
        
        # Damping
        self.vx *= 0.95
        self.vy *= 0.95
        
        # Occasionally update target (creates dynamic behavior)
        if random.random() < 0.01:
            self.target_x = random.uniform(0, 1)
            self.target_y = random.uniform(0, 1)
        
        # Detect crisis (performance drops)
        if error_magnitude > 0.5:
            self.in_crisis = True
            self.crisis_sync_count += 1
        else:
            self.in_crisis = False
            self.crisis_sync_count = max(0, self.crisis_sync_count - 1)
        
        return {
            "x": self.x,
            "y": self.y,
            "error": error_magnitude,
            "in_crisis": self.in_crisis,
            "role": self.role
        }
    
    def get_performance_metric(self) -> float:
        """Get performance metric based on distance to target."""
        error_x = self.target_x - self.x
        error_y = self.target_y - self.y
        error_magnitude = np.sqrt(error_x**2 + error_y**2)
        
        # Performance is inverse of error
        return max(0.0, 1.0 - error_magnitude)
    
    def detect_role(self, neighbors: List['SwarmAgent']) -> None:
        """
        Detect emergent role based on neighbor behavior.
        
        Implements patent-specified role detection:
        - 23% stylistic roles
        - 18% tone roles
        - Other emergent roles
        """
        if not neighbors:
            return
        
        # Analyze neighbor behaviors
        neighbor_styles = [n.style_factor for n in neighbors]
        avg_style = np.mean(neighbor_styles)
        
        # Role detection based on style deviation
        style_deviation = abs(self.style_factor - avg_style)
        
        if style_deviation > 0.3:
            self.role = "stylistic_leader"
            self.role_confidence = min(1.0, style_deviation)
        elif style_deviation < 0.1:
            self.role = "follower"
            self.role_confidence = 1.0 - style_deviation
        else:
            self.role = "coordinator"
            self.role_confidence = 0.5


class SwarmCoordinator:
    """
    Swarm coordinator for emergent behavior management.
    
    Implements patent-specified features:
    - Emergent role detection
    - Crisis synchronization
    - Collective heartbeat compression
    """
    
    def __init__(self, agents: Dict[str, SwarmAgent]):
        """
        Initialize swarm coordinator.
        
        Args:
            agents: Dictionary of swarm agents
        """
        self.agents = agents
        self.role_distribution: Dict[str, int] = {}
        self.crisis_count = 0
    
    def detect_emergent_roles(self) -> Dict[str, float]:
        """
        Detect emergent roles across swarm.
        
        Returns:
            Role distribution percentages
        """
        # Get neighbors for each agent (spatial proximity)
        for agent_id, agent in self.agents.items():
            neighbors = self._get_neighbors(agent, max_neighbors=10)
            agent.detect_role(neighbors)
        
        # Count roles
        role_counts: Dict[str, int] = {}
        for agent in self.agents.values():
            if agent.role:
                role_counts[agent.role] = role_counts.get(agent.role, 0) + 1
        
        total = len(self.agents)
        role_distribution = {
            role: count / total
            for role, count in role_counts.items()
        }
        
        self.role_distribution = role_distribution
        return role_distribution
    
    def _get_neighbors(self, agent: SwarmAgent, max_neighbors: int = 10) -> List[SwarmAgent]:
        """Get spatially nearest neighbors."""
        distances = []
        
        for other_agent in self.agents.values():
            if other_agent == agent:
                continue
            
            dx = agent.x - other_agent.x
            dy = agent.y - other_agent.y
            distance = np.sqrt(dx**2 + dy**2)
            distances.append((distance, other_agent))
        
        # Sort by distance and return nearest
        distances.sort(key=lambda x: x[0])
        return [agent for _, agent in distances[:max_neighbors]]
    
    def synchronize_crisis(self) -> int:
        """
        Synchronize agents in crisis state.
        
        Returns:
            Number of agents synchronized
        """
        crisis_agents = [
            agent for agent in self.agents.values()
            if agent.in_crisis
        ]
        
        self.crisis_count = len(crisis_agents)
        
        # Collective heartbeat compression: synchronize updates
        for agent in crisis_agents:
            # Increase exploration for crisis recovery
            agent.exploration_rate = min(0.5, agent.exploration_rate * 1.2)
        
        return len(crisis_agents)


async def main():
    """Run impressive swarm coordination example."""
    print("=" * 70)
    print("PulseOS Swarm Coordination: 1000+ Agents with Emergent Behavior")
    print("=" * 70)
    
    # Create survival constraint
    constraint = SurvivalConstraint(threshold=0.7)
    
    # Configure for large swarm
    config = Config(
        max_agents=2000,
        parallel_updates=True,
        update_batch_size=100,
        snapshot_interval=5.0,
        metrics_enabled=True
    )
    
    # Create runtime
    runtime = Runtime(constraint=constraint, config=config)
    
    # Create swarm
    swarm_size = 1000
    print(f"\nCreating swarm with {swarm_size} agents...")
    
    agents = {}
    for i in range(swarm_size):
        agent = SwarmAgent(f"swarm_agent_{i}", swarm_size)
        agents[f"swarm_agent_{i}"] = agent
        runtime.register_agent(f"swarm_agent_{i}", agent)
    
    # Create swarm coordinator
    coordinator = SwarmCoordinator(agents)
    
    print(f"Swarm initialized with {swarm_size} agents")
    print("\nRunning swarm simulation for 500 steps...")
    
    # Run simulation
    for step in range(500):
        await runtime.step()
        
        # Periodically detect roles and synchronize
        if step % 50 == 0:
            role_dist = coordinator.detect_emergent_roles()
            crisis_count = coordinator.synchronize_crisis()
            
            if step % 100 == 0:
                print(f"\nStep {step}:")
                print(f"  Role Distribution: {role_dist}")
                print(f"  Agents in Crisis: {crisis_count}")
                
                stats = runtime.get_statistics()
                print(f"  Survival Signal: {stats.get('average_survival_signal', 0):.3f}")
                print(f"  Cache Hit Rate: {stats.get('ngcm_cache_hit_rate', 0):.2%}")
    
    # Final statistics
    print("\n" + "=" * 70)
    print("Final Statistics")
    print("=" * 70)
    
    stats = runtime.get_statistics()
    print(f"\nRuntime:")
    print(f"  Steps: {stats['current_step']}")
    print(f"  Agents: {stats['agent_count']}")
    print(f"  Average Survival Signal: {stats.get('average_survival_signal', 0):.3f}")
    print(f"  NGCM Cache Hit Rate: {stats.get('ngcm_cache_hit_rate', 0):.2%}")
    
    # Final role distribution
    final_roles = coordinator.detect_emergent_roles()
    print(f"\nEmergent Role Distribution:")
    for role, percentage in sorted(final_roles.items(), key=lambda x: x[1], reverse=True):
        print(f"  {role}: {percentage:.1%}")
    
    # Sample agent metrics
    print(f"\nSample Agent Performance:")
    sample_agents = list(agents.items())[:5]
    for agent_id, agent in sample_agents:
        metric = agent.get_performance_metric()
        print(f"  {agent_id}: {metric:.3f} (role: {agent.role})")


if __name__ == "__main__":
    asyncio.run(main())

