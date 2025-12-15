"""
Game AI Example: Multi-Agent Strategy with Survival Pressure

Demonstrates PulseOS for game AI applications where agents must coordinate
and adapt strategies while maintaining survival constraints.
"""

import asyncio
import numpy as np
from pulseos import Runtime, Agent, SurvivalConstraint, Config


class GameAgent(Agent):
    """
    Game agent that learns strategic behavior in a competitive environment.
    
    Agents must:
    - Collect resources
    - Avoid enemies
    - Coordinate with teammates
    - Maintain health above survival threshold
    """
    
    def __init__(self, agent_id: str, team: str, initial_position: np.ndarray):
        super().__init__(agent_id)
        self.team = team
        self.position = initial_position.copy()
        self.health = 100.0
        self.max_health = 100.0
        self.resources = 0
        self.ammo = 50
        self.max_ammo = 50
        
        # Game state
        self.enemies = [
            {"position": np.array([8.0, 8.0]), "health": 100},
            {"position": np.array([12.0, 12.0]), "health": 100}
        ]
        self.resource_nodes = [
            np.array([3.0, 3.0]),
            np.array([15.0, 3.0]),
            np.array([3.0, 15.0]),
            np.array([15.0, 15.0])
        ]
        
        # Strategy parameters
        self.aggressiveness = 0.5  # 0 = defensive, 1 = aggressive
        self.resource_focus = 0.5   # 0 = combat focus, 1 = resource focus
        
        # Learning parameters
        self.learning_rate = 0.01
        self.exploration_rate = 0.1
    
    def distance_to(self, target: np.ndarray) -> float:
        """Compute distance to target"""
        return np.linalg.norm(self.position - target)
    
    def find_nearest_resource(self) -> np.ndarray:
        """Find nearest resource node"""
        distances = [self.distance_to(node) for node in self.resource_nodes]
        nearest_idx = np.argmin(distances)
        return self.resource_nodes[nearest_idx]
    
    def find_nearest_enemy(self) -> np.ndarray:
        """Find nearest enemy"""
        if not self.enemies:
            return None
        
        distances = [self.distance_to(enemy["position"]) for enemy in self.enemies]
        nearest_idx = np.argmin(distances)
        return self.enemies[nearest_idx]["position"]
    
    async def step(self):
        """Execute one step of game agent behavior"""
        # Decide action based on strategy
        action = self._decide_action()
        
        # Execute action
        if action == "collect_resource":
            nearest_resource = self.find_nearest_resource()
            direction = (nearest_resource - self.position) / (self.distance_to(nearest_resource) + 1e-6)
            self.position += direction * 0.5
            
            # Collect resource if close enough
            if self.distance_to(nearest_resource) < 1.0:
                self.resources += 1
                self.health = min(self.max_health, self.health + 5)  # Resource heals
        
        elif action == "attack_enemy":
            nearest_enemy_pos = self.find_nearest_enemy()
            if nearest_enemy_pos is not None and self.ammo > 0:
                direction = (nearest_enemy_pos - self.position) / (self.distance_to(nearest_enemy_pos) + 1e-6)
                self.position += direction * 0.3
                
                # Attack if in range
                if self.distance_to(nearest_enemy_pos) < 2.0:
                    self.ammo -= 1
                    # Damage enemy (simplified)
                    for enemy in self.enemies:
                        if np.linalg.norm(enemy["position"] - nearest_enemy_pos) < 0.1:
                            enemy["health"] -= 10
                            if enemy["health"] <= 0:
                                self.resources += 10  # Reward for kill
        
        elif action == "retreat":
            # Move away from enemies
            nearest_enemy_pos = self.find_nearest_enemy()
            if nearest_enemy_pos is not None:
                direction = (self.position - nearest_enemy_pos) / (self.distance_to(nearest_enemy_pos) + 1e-6)
                self.position += direction * 0.5
        
        # Take damage from enemies if too close
        for enemy in self.enemies:
            if self.distance_to(enemy["position"]) < 1.5:
                self.health -= 2
        
        # Regenerate ammo slowly
        self.ammo = min(self.max_ammo, self.ammo + 0.1)
        
        # Update strategy based on performance
        self._update_strategy()
        
        return {
            "position": self.position.tolist(),
            "health": self.health,
            "resources": self.resources,
            "ammo": self.ammo,
            "aggressiveness": self.aggressiveness,
            "resource_focus": self.resource_focus
        }
    
    def _decide_action(self) -> str:
        """Decide next action based on strategy"""
        if np.random.random() < self.exploration_rate:
            # Exploration: random action
            return np.random.choice(["collect_resource", "attack_enemy", "retreat"])
        
        # Exploitation: strategy-based decision
        health_ratio = self.health / self.max_health
        
        if health_ratio < 0.3:
            return "retreat"  # Low health, retreat
        
        if self.resources < 5 and self.resource_focus > 0.5:
            return "collect_resource"  # Need resources
        
        if self.ammo > 10 and self.aggressiveness > 0.5:
            return "attack_enemy"  # Aggressive and have ammo
        
        # Default: collect resources
        return "collect_resource"
    
    def _update_strategy(self):
        """Update strategy parameters based on performance"""
        # Simple adaptive strategy
        health_ratio = self.health / self.max_health
        
        if health_ratio < 0.5:
            # Low health: become more defensive
            self.aggressiveness = max(0.0, self.aggressiveness - 0.01)
        else:
            # Good health: can be more aggressive
            self.aggressiveness = min(1.0, self.aggressiveness + 0.005)
        
        if self.resources < 3:
            # Need resources: focus on collection
            self.resource_focus = min(1.0, self.resource_focus + 0.01)
        else:
            # Have resources: can focus on combat
            self.resource_focus = max(0.0, self.resource_focus - 0.005)
    
    def get_performance_metric(self) -> float:
        """
        Performance metric combines:
        - Health (survival)
        - Resources collected
        - Combat effectiveness
        """
        # Health metric (0 to 1)
        health_metric = self.health / self.max_health
        
        # Resource metric (0 to 1)
        resource_metric = min(1.0, self.resources / 20.0)
        
        # Combat metric (based on ammo and aggressiveness)
        combat_metric = (self.ammo / self.max_ammo) * self.aggressiveness
        
        # Combined performance (health is most important)
        performance = 0.5 * health_metric + 0.3 * resource_metric + 0.2 * combat_metric
        
        return performance


async def main():
    """Run game AI example"""
    print("🎮 Game AI Example: Multi-Agent Strategy with Survival Pressure")
    print("=" * 70)
    
    # Create survival constraint
    # Agents must maintain performance above 0.4 to "survive"
    constraint = SurvivalConstraint(threshold=0.4)
    
    # Configure runtime
    config = Config(
        snapshot_interval=2.0,
        max_snapshots=15,
        metrics_enabled=True
    )
    
    runtime = Runtime(constraint=constraint, config=config)
    
    # Create game agents (two teams)
    team_a_positions = [
        np.array([1.0, 1.0]),
        np.array([2.0, 1.0]),
        np.array([1.0, 2.0])
    ]
    
    team_b_positions = [
        np.array([19.0, 19.0]),
        np.array([18.0, 19.0]),
        np.array([19.0, 18.0])
    ]
    
    for i, pos in enumerate(team_a_positions):
        agent = GameAgent(f"team_a_{i}", "A", pos)
        runtime.register_agent(f"team_a_{i}", agent)
    
    for i, pos in enumerate(team_b_positions):
        agent = GameAgent(f"team_b_{i}", "B", pos)
        runtime.register_agent(f"team_b_{i}", agent)
    
    print(f"Created {len(team_a_positions) + len(team_b_positions)} game agents")
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
    
    # Print individual agent status
    print("\n🎮 Agent Status:")
    for agent_id, agent in runtime.agents.items():
        if isinstance(agent, GameAgent):
            performance = agent.get_performance_metric()
            print(f"  {agent_id} ({agent.team}):")
            print(f"    Position: [{agent.position[0]:.2f}, {agent.position[1]:.2f}]")
            print(f"    Health: {agent.health:.1f}/{agent.max_health}")
            print(f"    Resources: {agent.resources}")
            print(f"    Ammo: {agent.ammo:.1f}/{agent.max_ammo}")
            print(f"    Aggressiveness: {agent.aggressiveness:.3f}")
            print(f"    Resource Focus: {agent.resource_focus:.3f}")
            print(f"    Performance: {performance:.3f}")
    
    print("\n✅ Example completed!")


if __name__ == "__main__":
    asyncio.run(main())

