# PulseOS Tutorials

## Table of Contents

1. [Getting Started](#getting-started)
2. [Custom Constraints](#custom-constraints)
3. [Multi-Agent Coordination](#multi-agent-coordination)
4. [Performance Tuning](#performance-tuning)

## Getting Started

See `docs/tutorial_getting_started.py` for a complete runnable tutorial.

### Quick Overview

1. **Create an Agent:**
   ```python
   class MyAgent(Agent):
       async def step(self):
           # Your logic here
           return {"result": "success"}
       
       def get_performance_metric(self):
           return 0.9  # 0-1 range
   ```

2. **Create Runtime:**
   ```python
   constraint = SurvivalConstraint(threshold=0.8)
   runtime = Runtime(constraint=constraint)
   ```

3. **Register and Run:**
   ```python
   agent = MyAgent("agent1")
   runtime.register_agent("agent1", agent)
   await runtime.run(max_steps=100)
   ```

## Custom Constraints

### Creating Custom Constraints

You can create custom constraints by extending the `Constraint` class:

```python
from pulseos.agent import Constraint, ConstraintOperator

class TemporalConstraint(Constraint):
    """Requires performance over time window."""
    
    def __init__(self, threshold: float, window_size: int = 10):
        super().__init__(threshold=threshold, operator=ConstraintOperator.GREATER_THAN)
        self.window_size = window_size
        self.history = []
    
    def evaluate(self, metric: float) -> bool:
        self.history.append(metric)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        
        if len(self.history) < self.window_size:
            return True
        
        return np.mean(self.history) >= self.threshold
```

### Using Custom Constraints

```python
constraint = TemporalConstraint(threshold=0.7, window_size=10)
runtime = Runtime(constraint=constraint)
```

See `examples/custom_constraints.py` for complete examples.

## Multi-Agent Coordination

### Basic Multi-Agent Setup

```python
constraint = SurvivalConstraint(threshold=0.8)
runtime = Runtime(constraint=constraint)

# Register multiple agents
for i in range(10):
    agent = MyAgent(f"agent_{i}")
    runtime.register_agent(f"agent_{i}", agent)

await runtime.run(max_steps=100)
```

### Agent Communication

Agents can access the runtime through their context:

```python
class CoordinatedAgent(Agent):
    async def step(self):
        # Access other agents through runtime
        # (Note: This requires runtime reference)
        return {"coordinated": True}
```

### Large-Scale Swarms

For 1000+ agents, use optimized configuration:

```python
config = Config(
    gradient_cache_size=512,
    snapshot_interval=1.0,
    threshold_detection_interval=0.1
)
runtime = Runtime(constraint=constraint, config=config)
```

See `examples/impressive_swarm.py` for a complete example.

## Performance Tuning

### Speed Optimization

For maximum speed:

```python
config = Config(
    cache_implementation="LUT",  # Fastest
    enable_normalization=False,  # Skip normalization
    snapshot_interval=10.0,  # Less frequent snapshots
    gradient_cache_size=512  # Larger cache
)
```

### Memory Optimization

For memory-constrained environments:

```python
config = Config(
    gradient_cache_size=64,  # Smaller cache
    max_snapshots=10,  # Fewer snapshots
    max_agents=50,  # Limit agents
    snapshot_interval=5.0
)
```

### Accuracy Optimization

For maximum accuracy:

```python
config = Config(
    cache_implementation="EXACT",  # Most accurate
    enable_normalization=True,  # Enable normalization
    gradient_cache_size=256
)
```

### Benchmarking

Use the performance profiler:

```python
from pulseos.telemetry.profiler import PerformanceProfiler

profiler = PerformanceProfiler(enabled=True)
profiler.start()

# Run your code
await runtime.run(max_steps=100)

profiler.stop()
bottlenecks = profiler.get_bottlenecks()
print(bottlenecks)
```

See `examples/performance_tuning.py` for complete benchmarking examples.

## Advanced Topics

### Custom NGCM Implementation

You can implement custom gradient computation:

```python
from pulseos.circuits.ngcm import NonlinearGradientComputationModule

class CustomNGCM(NonlinearGradientComputationModule):
    def _compute_gradient_exact(self, sigmoid: float) -> float:
        # Your custom implementation
        return super()._compute_gradient_exact(sigmoid)
```

### Snapshot Management

Manual snapshot control:

```python
# Create snapshot manually
snapshot_data = {
    "step": runtime.current_step,
    "agents": {id: agent.get_state() for id, agent in runtime.agents.items()}
}
snapshot = await runtime.sprs.create_snapshot(snapshot_data)

# Find recovery snapshot
recovery = await runtime.sprs.find_best_recovery_snapshot(min_survival_signal=0.5)
```

### Metrics Export

Export metrics for monitoring:

```python
# Prometheus format
prometheus_metrics = runtime.metrics_collector.export_prometheus()

# JSON format
json_metrics = runtime.metrics_collector.export_json()

# Enhanced metrics
from pulseos.telemetry.enhanced_metrics import EnhancedMetricsCollector
enhanced = EnhancedMetricsCollector()
report = enhanced.export_comprehensive_report()
```

## Best Practices

1. **Start Simple:** Begin with basic agents and simple constraints
2. **Iterate:** Gradually add complexity as you understand the system
3. **Monitor:** Use metrics to understand agent behavior
4. **Tune:** Adjust configuration based on your workload
5. **Test:** Test with various agent counts and scenarios

## Next Steps

- Explore the `examples/` directory for more examples
- Read `TECHNICAL.md` for implementation details
- Check `TROUBLESHOOTING.md` for common issues
- Review `ADRs.md` for architectural decisions

