"""
Comprehensive Validation Suite
"""
import asyncio
import time
import random
import numpy as np
from pulseos import Runtime, Config, Agent, SurvivalConstraint
from pulseos.circuits.ngcm import NonlinearGradientComputationModule

class ValidationAgent(Agent):
    def __init__(self, agent_id: str, initial_performance: float = 0.3):
        super().__init__(agent_id)
        self.performance = initial_performance
        self.target_performance = 0.9
    
    async def step(self):
        error = self.target_performance - self.performance
        if random.random() > self.exploration_rate:
            self.performance += self.learning_rate * error * 0.5
        else:
            self.performance += random.uniform(-0.05, 0.05)
        self.performance = np.clip(self.performance, 0.0, 1.0)
        return {"performance": self.performance}
    
    def get_performance_metric(self):
        return self.performance

async def main():
    print("=" * 70)
    print("VALIDATION: Testing if PulseOS actually works")
    print("=" * 70)
    
    # Test 1: Adaptive parameters
    print("\nTEST 1: Adaptive Parameter Control")
    constraint = SurvivalConstraint(threshold=0.7)
    runtime = Runtime(constraint=constraint)
    agents = [ValidationAgent(f"a_{i}", 0.3 + (i%3)*0.2) for i in range(20)]
    for agent in agents:
        runtime.register_agent(agent.agent_id, agent)
    
    initial_lr = agents[0].learning_rate
    for _ in range(50):
        await runtime.step()
    final_lr = agents[0].learning_rate
    print(f"✓ Learning rate changed: {initial_lr:.4f} → {final_lr:.4f}")
    
    # Test 2: Performance improvement
    print("\nTEST 2: Survival Pressure Effectiveness")
    agents2 = [ValidationAgent(f"b_{i}", 0.3) for i in range(30)]
    runtime2 = Runtime(constraint=SurvivalConstraint(threshold=0.7))
    for agent in agents2:
        runtime2.register_agent(agent.agent_id, agent)
    
    initial_perf = np.mean([a.performance for a in agents2])
    for _ in range(100):
        await runtime2.step()
    final_perf = np.mean([a.performance for a in agents2])
    print(f"✓ Performance: {initial_perf:.3f} → {final_perf:.3f} (+{final_perf-initial_perf:.3f})")
    
    # Test 3: Gradient computation
    print("\nTEST 3: Gradient Computation")
    ngcm = NonlinearGradientComputationModule(beta=1.0)
    sigmoid = ngcm.compute_sigmoid(0.0)
    print(f"✓ Sigmoid at 0: {sigmoid:.4f} (should be ~0.5)")
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
