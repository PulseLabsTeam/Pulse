"""
Performance Tuning Example

Demonstrates how to tune PulseOS for optimal performance:
- Cache configuration
- Batch processing
- Memory optimization
- Hardware acceleration modes
"""

import asyncio
import time
import numpy as np
from pulseos import Runtime, Config, Agent, SurvivalConstraint


class BenchmarkAgent(Agent):
    """Agent for performance benchmarking."""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.state = np.random.rand(100)  # Larger state for realistic benchmark
        self.target = np.ones(100) * 0.8
    
    async def step(self) -> dict:
        """Execute step with computational workload."""
        # Simulate computational work
        error = self.target - self.state
        
        if np.random.random() > self.exploration_rate:
            # Exploit: gradient descent
            self.state += self.learning_rate * error
        else:
            # Explore: random perturbation
            self.state += np.random.normal(0, 0.1, 100)
        
        self.state = np.clip(self.state, 0.0, 1.0)
        
        return {"state_norm": np.linalg.norm(self.state)}
    
    def get_performance_metric(self) -> float:
        """Performance based on distance to target."""
        error = np.linalg.norm(self.target - self.state)
        return max(0.0, 1.0 - error / np.sqrt(len(self.state)))


async def benchmark_configuration(config: Config, name: str, num_agents: int = 100):
    """Benchmark a specific configuration."""
    print(f"\n{'='*70}")
    print(f"Benchmarking: {name}")
    print(f"{'='*70}")
    
    constraint = SurvivalConstraint(threshold=0.7)
    runtime = Runtime(constraint=constraint, config=config)
    
    # Register agents
    for i in range(num_agents):
        agent = BenchmarkAgent(f"agent_{i}")
        runtime.register_agent(f"agent_{i}", agent)
    
    # Benchmark
    start_time = time.time()
    await runtime.run(max_steps=100)
    elapsed_time = time.time() - start_time
    
    stats = runtime.get_statistics()
    
    print(f"  Execution time: {elapsed_time:.3f}s")
    print(f"  Steps/second: {stats['current_step'] / elapsed_time:.1f}")
    print(f"  Cache hit rate: {stats.get('ngcm_cache_hit_rate', 0):.2%}")
    print(f"  Average survival signal: {stats['average_survival_signal']:.3f}")
    
    return elapsed_time, stats


async def main():
    """Run performance tuning examples."""
    print("⚡ PulseOS Performance Tuning Example")
    print("Comparing different configurations for optimal performance\n")
    
    num_agents = 100
    
    # Baseline configuration
    baseline_config = Config()
    baseline_time, baseline_stats = await benchmark_configuration(
        baseline_config, "Baseline (Default)", num_agents
    )
    
    # Optimized cache configuration
    cache_config = Config(
        gradient_cache_size=512,  # Larger cache
        cache_implementation="LUT"  # Fastest implementation
    )
    cache_time, cache_stats = await benchmark_configuration(
        cache_config, "Optimized Cache (512 entries, LUT)", num_agents
    )
    
    # Batch processing configuration
    batch_config = Config(
        update_batch_size=50,  # Larger batches
        parallel_updates=True,
        vectorization_enabled=True
    )
    batch_time, batch_stats = await benchmark_configuration(
        batch_config, "Batch Processing (batch_size=50)", num_agents
    )
    
    # Memory-optimized configuration
    memory_config = Config(
        max_agents=num_agents,
        snapshot_interval=5.0,  # Less frequent snapshots
        enable_delta_encoding=True,
        enable_compression=False  # Trade compression for speed
    )
    memory_time, memory_stats = await benchmark_configuration(
        memory_config, "Memory Optimized", num_agents
    )
    
    # Aggressive optimization (all optimizations)
    aggressive_config = Config(
        gradient_cache_size=512,
        cache_implementation="LUT",
        update_batch_size=50,
        parallel_updates=True,
        vectorization_enabled=True,
        snapshot_interval=5.0,
        enable_delta_encoding=True
    )
    aggressive_time, aggressive_stats = await benchmark_configuration(
        aggressive_config, "Aggressive Optimization (All)", num_agents
    )
    
    # Summary
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*70)
    
    configs = [
        ("Baseline", baseline_time),
        ("Cache Optimized", cache_time),
        ("Batch Processing", batch_time),
        ("Memory Optimized", memory_time),
        ("Aggressive", aggressive_time)
    ]
    
    configs.sort(key=lambda x: x[1])
    
    print("\nRanking by execution time:")
    for rank, (name, time_taken) in enumerate(configs, 1):
        speedup = baseline_time / time_taken
        print(f"  {rank}. {name:25s} | {time_taken:6.3f}s | {speedup:.2f}x speedup")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print("""
1. Cache Configuration:
   • Increase cache_size for repeated computations
   • Use LUT for fastest lookups (if accuracy acceptable)
   • Monitor cache hit rate to validate effectiveness

2. Batch Processing:
   • Increase update_batch_size for better throughput
   • Enable parallel_updates for multi-core systems
   • Use vectorization_enabled for numpy operations

3. Memory Optimization:
   • Adjust snapshot_interval based on rollback needs
   • Enable delta_encoding for state persistence
   • Disable compression if speed > storage

4. Hardware Acceleration:
   • Use LUT/PLA for CPU-bound workloads
   • Consider GPU acceleration for large-scale (future feature)
   • Profile bottlenecks to identify optimization targets
    """)


if __name__ == "__main__":
    asyncio.run(main())

