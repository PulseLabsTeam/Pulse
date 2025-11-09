"""
Comprehensive Convergence Benchmark

Proves 28% faster convergence claim with baseline comparison.
"""

import asyncio
import time
import random
import numpy as np
from typing import List, Dict, Any
from pulseos import Runtime, Config, Agent, SurvivalConstraint


class BaselineRLAgent(Agent):
    """Baseline RL agent without survival pressure."""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.state = 0.0
        self.target = 1.0
        self.converged = False
        self.convergence_step = None
        
        # Fixed learning rate (no adaptation)
        self.learning_rate = 0.01
        self.exploration_rate = 0.1
    
    async def step(self) -> dict:
        """Execute step with fixed parameters."""
        error = self.target - self.state
        
        if random.random() > self.exploration_rate:
            self.state += self.learning_rate * error
        else:
            self.state += random.uniform(-0.1, 0.1)
        
        self.state = np.clip(self.state, 0.0, 1.0)
        
        if abs(error) < 0.01 and not self.converged:
            self.converged = True
            self.convergence_step = time.time()
        
        return {"state": self.state, "error": abs(error)}
    
    def get_performance_metric(self) -> float:
        """Get performance metric."""
        error = abs(self.target - self.state)
        return 1.0 - error


class PulseOSAgent(Agent):
    """PulseOS agent with survival pressure adaptation."""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.state = 0.0
        self.target = 1.0
        self.converged = False
        self.convergence_step = None
    
    async def step(self) -> dict:
        """Execute step with adaptive parameters."""
        error = self.target - self.state
        
        if random.random() > self.exploration_rate:
            self.state += self.learning_rate * error
        else:
            self.state += random.uniform(-0.1, 0.1)
        
        self.state = np.clip(self.state, 0.0, 1.0)
        
        if abs(error) < 0.01 and not self.converged:
            self.converged = True
            self.convergence_step = time.time()
        
        return {"state": self.state, "error": abs(error)}
    
    def get_performance_metric(self) -> float:
        """Get performance metric."""
        error = abs(self.target - self.state)
        return 1.0 - error


async def run_baseline_rl(num_agents: int = 100, max_steps: int = 1000) -> Dict[str, Any]:
    """
    Run baseline RL without survival pressure.
    
    Args:
        num_agents: Number of agents
        max_steps: Maximum steps
        
    Returns:
        Convergence statistics
    """
    agents = []
    for i in range(num_agents):
        agent = BaselineRLAgent(f"baseline_{i}")
        agents.append(agent)
    
    start_time = time.time()
    converged_count = 0
    convergence_times = []
    
    for step in range(max_steps):
        # Update all agents
        for agent in agents:
            await agent.step()
            
            if agent.converged and agent.convergence_step:
                if agent.convergence_step not in convergence_times:
                    convergence_times.append(agent.convergence_step - start_time)
        
        converged_count = sum(1 for a in agents if a.converged)
        
        # Check if majority converged
        if converged_count >= num_agents * 0.9:
            break
    
    total_time = time.time() - start_time
    
    return {
        "total_time": total_time,
        "steps": step + 1,
        "converged_count": converged_count,
        "convergence_rate": converged_count / num_agents,
        "average_convergence_time": np.mean(convergence_times) if convergence_times else total_time,
        "median_convergence_time": np.median(convergence_times) if convergence_times else total_time
    }


async def run_pulseos(num_agents: int = 100, max_steps: int = 1000) -> Dict[str, Any]:
    """
    Run PulseOS with survival pressure.
    
    Args:
        num_agents: Number of agents
        max_steps: Maximum steps
        
    Returns:
        Convergence statistics
    """
    constraint = SurvivalConstraint(threshold=0.9)
    config = Config(
        max_agents=num_agents,
        parallel_updates=True,
        update_batch_size=50
    )
    
    runtime = Runtime(constraint=constraint, config=config)
    
    # Register agents
    for i in range(num_agents):
        agent = PulseOSAgent(f"pulseos_{i}")
        runtime.register_agent(f"pulseos_{i}", agent)
    
    start_time = time.time()
    converged_count = 0
    convergence_times = []
    
    for step in range(max_steps):
        await runtime.step()
        
        # Check convergence
        converged_count = sum(
            1 for agent in runtime.agents.values()
            if isinstance(agent, PulseOSAgent) and agent.converged
        )
        
        # Track convergence times
        for agent in runtime.agents.values():
            if isinstance(agent, PulseOSAgent) and agent.converged and agent.convergence_step:
                if agent.convergence_step not in convergence_times:
                    convergence_times.append(agent.convergence_step - start_time)
        
        if converged_count >= num_agents * 0.9:
            break
    
    total_time = time.time() - start_time
    
    return {
        "total_time": total_time,
        "steps": step + 1,
        "converged_count": converged_count,
        "convergence_rate": converged_count / num_agents,
        "average_convergence_time": np.mean(convergence_times) if convergence_times else total_time,
        "median_convergence_time": np.median(convergence_times) if convergence_times else total_time,
        "runtime_stats": runtime.get_statistics()
    }


async def benchmark_convergence_improvement(
    num_agents: int = 100,
    num_trials: int = 5
) -> Dict[str, Any]:
    """
    Benchmark convergence improvement with multiple trials.
    
    Args:
        num_agents: Number of agents per trial
        num_trials: Number of trials to run
        
    Returns:
        Comprehensive benchmark results
    """
    print("=" * 70)
    print("Convergence Benchmark: Proving 28% Faster Convergence")
    print("=" * 70)
    
    baseline_results = []
    pulseos_results = []
    
    for trial in range(num_trials):
        print(f"\nTrial {trial + 1}/{num_trials}")
        print("-" * 70)
        
        # Run baseline
        print("Running baseline RL...")
        baseline = await run_baseline_rl(num_agents=num_agents)
        baseline_results.append(baseline)
        print(f"  Time: {baseline['total_time']:.2f}s")
        print(f"  Steps: {baseline['steps']}")
        print(f"  Converged: {baseline['converged_count']}/{num_agents}")
        
        # Run PulseOS
        print("Running PulseOS...")
        pulseos = await run_pulseos(num_agents=num_agents)
        pulseos_results.append(pulseos)
        print(f"  Time: {pulseos['total_time']:.2f}s")
        print(f"  Steps: {pulseos['steps']}")
        print(f"  Converged: {pulseos['converged_count']}/{num_agents}")
    
    # Compute statistics
    baseline_avg_time = np.mean([r['total_time'] for r in baseline_results])
    pulseos_avg_time = np.mean([r['total_time'] for r in pulseos_results])
    
    baseline_avg_steps = np.mean([r['steps'] for r in baseline_results])
    pulseos_avg_steps = np.mean([r['steps'] for r in pulseos_results])
    
    time_improvement = ((baseline_avg_time - pulseos_avg_time) / baseline_avg_time) * 100
    steps_improvement = ((baseline_avg_steps - pulseos_avg_steps) / baseline_avg_steps) * 100
    
    print("\n" + "=" * 70)
    print("Benchmark Results")
    print("=" * 70)
    print(f"\nBaseline RL (Average):")
    print(f"  Time: {baseline_avg_time:.2f}s")
    print(f"  Steps: {baseline_avg_steps:.1f}")
    
    print(f"\nPulseOS (Average):")
    print(f"  Time: {pulseos_avg_time:.2f}s")
    print(f"  Steps: {pulseos_avg_steps:.1f}")
    
    print(f"\nImprovement:")
    print(f"  Time: {time_improvement:.1f}% faster")
    print(f"  Steps: {steps_improvement:.1f}% faster")
    
    # Validate claim
    claim_met = time_improvement >= 28.0
    print(f"\n✓ Target: 28% faster - {'PASS ✓' if claim_met else 'NEEDS IMPROVEMENT ✗'}")
    
    return {
        "baseline_results": baseline_results,
        "pulseos_results": pulseos_results,
        "baseline_avg_time": baseline_avg_time,
        "pulseos_avg_time": pulseos_avg_time,
        "baseline_avg_steps": baseline_avg_steps,
        "pulseos_avg_steps": pulseos_avg_steps,
        "time_improvement_percent": time_improvement,
        "steps_improvement_percent": steps_improvement,
        "claim_met": claim_met
    }


if __name__ == "__main__":
    asyncio.run(benchmark_convergence_improvement(num_agents=100, num_trials=3))

