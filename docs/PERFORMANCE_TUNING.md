# PulseOS Performance Tuning Guide

This guide helps you optimize PulseOS for your specific use case and achieve the best performance.

## Table of Contents

1. [Performance Targets](#performance-targets)
2. [Circuit Configuration](#circuit-configuration)
3. [Cache Optimization](#cache-optimization)
4. [Memory Management](#memory-management)
5. [Scalability Tuning](#scalability-tuning)
6. [Profiling and Monitoring](#profiling-and-monitoring)
7. [Common Performance Issues](#common-performance-issues)

---

## Performance Targets

PulseOS is designed to meet these performance targets:

| Metric | Target | How to Validate |
|--------|--------|-----------------|
| Convergence Speed | 28% faster than baseline | Run `benchmarks/convergence_benchmark.py` |
| Cache Hit Rate | 75% | Monitor via `EnhancedMetricsCollector` |
| Storage Reduction | 70-85% | Check snapshot compression ratios |
| Detection Latency | < 1ms | Use `PerformanceProfiler` |
| Scalability | 10K agents < 1GB RAM | Monitor memory usage |

---

## Circuit Configuration

### Performance Threshold Detection Circuit (PTDC)

**Optimization Tips:**

1. **Batch Evaluation**: Evaluate multiple agents together for better vectorization
   ```python
   # Good: Batch evaluation
   metrics = {f"agent_{i}": perf[i] for i in range(100)}
   results = ptdc.evaluate(metrics)
   
   # Avoid: Individual evaluations
   for i in range(100):
       result = ptdc.evaluate({f"agent_{i}": perf[i]})
   ```

2. **Threshold Selection**: Choose thresholds that balance challenge and achievability
   - Too high: Agents struggle to meet threshold, slow convergence
   - Too low: No survival pressure, no adaptation benefit

### Nonlinear Gradient Computation Module (NGCM)

**Implementation Selection:**

| Implementation | Speed | Accuracy | Use Case |
|----------------|-------|----------|----------|
| **EXACT** | Slowest | Highest | When accuracy is critical |
| **LUT** | Fastest | High | Default choice for most cases |
| **PLA** | Fast | Medium | When memory is constrained |
| **CORDIC** | Medium | Medium | Hardware-optimized scenarios |

**Recommendation**: Start with `LUT` (default), switch to `EXACT` if you see numerical issues.

**Cache Configuration:**

```python
from pulseos.circuits.ngcm import NonlinearGradientComputationModule

# Optimize cache size based on workload
ngcm = NonlinearGradientComputationModule(
    cache_size=256,  # Default: 256 entries
    implementation="LUT"
)

# For workloads with many unique delta values:
ngcm = NonlinearGradientComputationModule(
    cache_size=512,  # Increase cache size
    implementation="LUT"
)

# For memory-constrained environments:
ngcm = NonlinearGradientComputationModule(
    cache_size=128,  # Reduce cache size
    implementation="PLA"  # Use memory-efficient implementation
)
```

**Cache Hit Rate Optimization:**

1. **Quantization**: Cache uses quantization - similar delta values map to same cache entry
2. **Workload Patterns**: Cache works best with repeated patterns
3. **Monitor Hit Rate**: Use `EnhancedMetricsCollector` to track cache performance

### Adaptive Parameter Controller (APC)

**Parameter Tuning:**

```python
from pulseos.circuits.apc import AdaptiveParameterController

apc = AdaptiveParameterController(
    alpha_base=0.01,      # Base learning rate
    gamma=0.1,            # Gradient influence (default: 0.1)
    epsilon_min=0.01,     # Minimum exploration
    epsilon_max=0.3,      # Maximum exploration
    kappa=2.0             # Exploration decay (default: 2.0)
)
```

**Guidelines:**

- **alpha_base**: Start with 0.01, increase if convergence is slow
- **gamma**: Controls how much gradient affects learning rate (0.1 is usually good)
- **epsilon_min/max**: Balance exploration vs exploitation
- **kappa**: Higher values = faster exploration decay

---

## Cache Optimization

### Gradient Cache

**Monitoring Cache Performance:**

```python
from pulseos.telemetry.enhanced_metrics import EnhancedMetricsCollector

collector = EnhancedMetricsCollector()

# ... run your simulation ...

cache_stats = collector.get_cache_statistics()
print(f"Cache Hit Rate: {cache_stats['average_hit_rate']:.2%}")
```

**Improving Cache Hit Rate:**

1. **Increase Cache Size**: If hit rate < 50%, try increasing cache size
   ```python
   ngcm = NonlinearGradientComputationModule(cache_size=512)
   ```

2. **Workload Patterns**: Cache works best when delta values repeat
   - If your workload has many unique values, cache may not help much
   - Consider preprocessing to normalize delta values

3. **Implementation Choice**: LUT implementation has better cache behavior than EXACT

### Memory Pool Cache

Memory pools are automatically managed, but you can monitor usage:

```python
from pulseos.optimization.cache import MemoryPool

pool = MemoryPool(object_type=dict, initial_size=100)
# Pool automatically grows/shrinks based on demand
```

---

## Memory Management

### Agent Scaling

**Memory Usage Per Agent:**

- Base agent: ~100 bytes
- State data: Variable (depends on your agent state)
- Runtime overhead: ~50 bytes per agent

**For 10,000 agents:**
- Base: ~1.5 MB
- With state: ~5-10 MB (typical)
- Total: < 1GB (as validated)

**Optimization Tips:**

1. **Minimize Agent State**: Only store essential state in agents
2. **Use Delta Encoding**: Enable for snapshots (default)
3. **Limit History**: Set appropriate `max_history` in metrics collectors

### Snapshot Management

**Storage Optimization:**

```python
from pulseos.persistence.snapshot import SnapshotManager

manager = SnapshotManager(
    snapshot_interval=1.0,        # Create snapshot every 1 second
    max_snapshots=100,            # Keep last 100 snapshots
    enable_delta_encoding=True,   # Enable delta encoding (saves 70-85%)
    enable_compression=True       # Enable compression
)
```

**Storage Reduction:**

- Delta encoding: 70-85% reduction
- Compression: Additional 20-30% reduction
- Combined: ~80-90% total reduction

**When to Create Snapshots:**

- Frequent snapshots: Better recovery, more storage
- Infrequent snapshots: Less storage, longer recovery window
- Recommendation: Create snapshots every 1-10 steps depending on stability

---

## Scalability Tuning

### Large-Scale Deployments

**For 1,000+ Agents:**

```python
from pulseos import Runtime, Config

config = Config(
    max_agents=10000,           # Set appropriate limit
    metrics_enabled=True,        # Enable for monitoring
    snapshot_interval=5.0         # Less frequent snapshots
)

runtime = Runtime(constraint=constraint, config=config)
```

**Performance Considerations:**

1. **Batch Operations**: Process agents in batches for better vectorization
2. **Async Operations**: Use async/await for I/O-bound operations
3. **Memory Monitoring**: Monitor memory usage as you scale

### Multi-Agent Coordination

**Optimization Tips:**

1. **Spatial Partitioning**: Group agents by location for efficient updates
2. **Event-Driven Updates**: Only update agents when needed
3. **Parallel Processing**: Use asyncio for concurrent agent steps

---

## Profiling and Monitoring

### Using Performance Profiler

```python
from pulseos.telemetry.profiler import PerformanceProfiler

profiler = PerformanceProfiler(enabled=True)
profiler.start()

# ... run your code ...

profiler.stop()

# Get bottlenecks
bottlenecks = profiler.get_bottlenecks(threshold_percent=5.0)
for bottleneck in bottlenecks:
    print(f"{bottleneck.name}: {bottleneck.percentage:.2f}%")

# Get timing statistics
timings = profiler.get_timing_statistics()
for operation, stats in timings.items():
    print(f"{operation}: {stats['mean']*1000:.2f}ms")
```

### Using Enhanced Metrics

```python
from pulseos.telemetry.enhanced_metrics import EnhancedMetricsCollector

collector = EnhancedMetricsCollector()

# ... run simulation ...

# Get comprehensive report
report = collector.export_comprehensive_report()
print(report)

# Get specific statistics
gradient_stats = collector.get_gradient_statistics()
cache_stats = collector.get_cache_statistics()
convergence_stats = collector.get_convergence_statistics()
performance_stats = collector.get_performance_statistics()
```

---

## Common Performance Issues

### Issue: Slow Convergence

**Symptoms:**
- Agents take many steps to converge
- Survival signal stays low

**Solutions:**
1. **Increase Learning Rate**: Try `alpha_base=0.02` or higher
2. **Lower Threshold**: Make survival constraint easier to meet initially
3. **Check Gradient**: Ensure gradients are being computed correctly
4. **Monitor Cache**: Low cache hit rate may indicate computation issues

### Issue: High Memory Usage

**Symptoms:**
- Memory usage exceeds expectations
- System runs out of memory

**Solutions:**
1. **Reduce Snapshot Count**: Lower `max_snapshots`
2. **Increase Snapshot Interval**: Create snapshots less frequently
3. **Limit History**: Set `max_history` in metrics collectors
4. **Check Agent State**: Minimize state stored in agents

### Issue: Low Cache Hit Rate

**Symptoms:**
- Cache hit rate < 50%
- No performance benefit from caching

**Solutions:**
1. **Increase Cache Size**: Try `cache_size=512` or higher
2. **Check Workload**: Cache works best with repeated patterns
3. **Normalize Values**: Preprocess delta values to reduce uniqueness
4. **Switch Implementation**: Try `PLA` for better cache behavior

### Issue: High Latency

**Symptoms:**
- Threshold detection > 1ms
- Slow step execution

**Solutions:**
1. **Batch Operations**: Process multiple agents together
2. **Use LUT**: Switch NGCM to LUT implementation
3. **Reduce Agent Count**: If possible, reduce number of agents per runtime
4. **Profile Code**: Use `PerformanceProfiler` to find bottlenecks

---

## Best Practices

1. **Start Simple**: Begin with default configurations, optimize based on profiling
2. **Monitor Metrics**: Always enable metrics collection to track performance
3. **Profile Regularly**: Use profiler to identify bottlenecks
4. **Test at Scale**: Test with realistic agent counts before production
5. **Document Changes**: Keep track of configuration changes and their effects

---

## Performance Checklist

Before deploying to production:

- [ ] Validated convergence speed meets target (28% improvement)
- [ ] Cache hit rate > 70%
- [ ] Memory usage within budget (< 1GB for 10K agents)
- [ ] Latency < 1ms for threshold detection
- [ ] Snapshot storage reduction > 70%
- [ ] Profiled and optimized bottlenecks
- [ ] Tested at production scale
- [ ] Monitoring and alerting configured

---

## Getting Help

If you encounter performance issues:

1. **Check Logs**: Review runtime logs for errors
2. **Profile Code**: Use `PerformanceProfiler` to identify bottlenecks
3. **Review Metrics**: Check `EnhancedMetricsCollector` statistics
4. **Test Isolation**: Test individual components to isolate issues
5. **Open Issue**: Report issues with profiling data and metrics

For more information, see:
- [TECHNICAL.md](TECHNICAL.md) - Technical implementation details
- [README.md](README.md) - Getting started guide
- [Examples](../examples/) - Example implementations

