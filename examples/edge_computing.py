"""
Edge Computing Example - Resource-Constrained Optimization

Demonstrates PulseOS for edge devices with memory and compute constraints.
"""

import asyncio
import numpy as np
from pulseos import Runtime, Config, Agent, SurvivalConstraint


class EdgeDeviceAgent(Agent):
    """Edge device agent with resource constraints."""
    
    def __init__(self, agent_id: str, memory_limit: int, compute_limit: float):
        super().__init__(agent_id)
        self.memory_limit = memory_limit  # MB
        self.compute_limit = compute_limit  # CPU units
        self.memory_used = np.random.uniform(0.3, 0.7) * memory_limit
        self.compute_used = np.random.uniform(0.3, 0.7) * compute_limit
        self.task_queue = []
        self.completed_tasks = 0
        self.failed_tasks = 0
    
    async def step(self) -> dict:
        """Execute edge device step."""
        # Process tasks based on available resources
        tasks_processed = 0
        
        # Compute capacity determines processing rate
        processing_rate = self.compute_used / self.compute_limit
        
        # Memory efficiency affects success rate
        memory_efficiency = 1.0 - (self.memory_used / self.memory_limit)
        
        # Process tasks
        if self.task_queue:
            task = self.task_queue.pop(0)
            if processing_rate > 0.5 and memory_efficiency > 0.3:
                self.completed_tasks += 1
                tasks_processed = 1
            else:
                self.failed_tasks += 1
        
        # Add new tasks
        if np.random.random() < 0.3:
            task_size = np.random.uniform(0.1, 0.5)
            if self.memory_used + task_size * self.memory_limit < self.memory_limit:
                self.task_queue.append(task_size)
                self.memory_used += task_size * self.memory_limit
        
        # Optimize resource usage based on learning
        if np.random.random() > self.exploration_rate:
            # Exploit: optimize resource allocation
            if self.memory_used / self.memory_limit > 0.8:
                # Reduce memory usage
                self.memory_used *= (1 - self.learning_rate * 0.1)
            
            if self.compute_used / self.compute_limit < 0.5:
                # Increase compute usage
                self.compute_used = min(
                    self.compute_limit,
                    self.compute_used * (1 + self.learning_rate * 0.1)
                )
        else:
            # Explore: random resource adjustments
            self.memory_used += np.random.uniform(-0.05, 0.05) * self.memory_limit
            self.compute_used += np.random.uniform(-0.05, 0.05) * self.compute_limit
        
        # Clamp resources
        self.memory_used = np.clip(self.memory_used, 0, self.memory_limit)
        self.compute_used = np.clip(self.compute_used, 0, self.compute_limit)
        
        return {
            "memory_used": self.memory_used,
            "compute_used": self.compute_used,
            "tasks_processed": tasks_processed,
            "queue_length": len(self.task_queue)
        }
    
    def get_performance_metric(self) -> float:
        """Performance: task completion rate and resource efficiency."""
        total_tasks = self.completed_tasks + self.failed_tasks
        if total_tasks == 0:
            return 0.5
        
        completion_rate = self.completed_tasks / total_tasks
        
        # Resource efficiency
        memory_efficiency = 1.0 - abs(self.memory_used / self.memory_limit - 0.7)
        compute_efficiency = self.compute_used / self.compute_limit
        
        return (completion_rate * 0.6 + memory_efficiency * 0.2 + compute_efficiency * 0.2)


async def main():
    """Run edge computing example."""
    # Constraint: devices must maintain > 0.6 performance
    constraint = SurvivalConstraint(threshold=0.6)
    
    # Configure for edge computing (memory-efficient)
    config = Config(
        threshold_detection_interval=0.1,
        snapshot_interval=1.0,
        gradient_cache_size=128,  # Smaller cache for edge
        max_agents=50  # Limit for edge devices
    )
    
    runtime = Runtime(constraint=constraint, config=config)
    
    # Create edge device swarm
    num_devices = 30
    for i in range(num_devices):
        memory_limit = np.random.uniform(100, 500)  # MB
        compute_limit = np.random.uniform(1.0, 4.0)  # CPU units
        device = EdgeDeviceAgent(f"device_{i}", memory_limit, compute_limit)
        runtime.register_agent(f"device_{i}", device)
    
    print(f"Running {num_devices} edge devices for 100 steps...")
    await runtime.run(max_steps=100)
    
    # Print results
    stats = runtime.get_statistics()
    print("\nEdge Computing Statistics:")
    print(f"  Steps: {stats['current_step']}")
    print(f"  Devices: {stats['agent_count']}")
    print(f"  Average Survival Signal: {stats['average_survival_signal']:.3f}")
    
    # Resource usage statistics
    devices = [d for d in runtime.agents.values() if isinstance(d, EdgeDeviceAgent)]
    avg_memory_usage = np.mean([d.memory_used / d.memory_limit for d in devices])
    avg_compute_usage = np.mean([d.compute_used / d.compute_limit for d in devices])
    total_completed = sum([d.completed_tasks for d in devices])
    total_failed = sum([d.failed_tasks for d in devices])
    
    print(f"  Average Memory Usage: {avg_memory_usage:.1%}")
    print(f"  Average Compute Usage: {avg_compute_usage:.1%}")
    print(f"  Total Tasks Completed: {total_completed}")
    print(f"  Total Tasks Failed: {total_failed}")
    if total_completed + total_failed > 0:
        print(f"  Success Rate: {total_completed / (total_completed + total_failed):.1%}")


if __name__ == "__main__":
    asyncio.run(main())

