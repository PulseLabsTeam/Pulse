# PulseOS Troubleshooting Guide

## Common Issues and Solutions

### Issue: Low Cache Hit Rate

**Symptoms:**
- Cache hit rate consistently below 50%
- Performance not meeting targets

**Causes:**
- Workload has high variance in delta values
- Cache size too small
- Cache quantization too fine

**Solutions:**
1. Increase cache size:
   ```python
   config = Config(gradient_cache_size=512)  # Default is 256
   ```

2. Use LUT implementation for faster lookups:
   ```python
   config = Config(cache_implementation="LUT")
   ```

3. Ensure repeated patterns in workload (cache works best with repeated values)

### Issue: Agents Not Converging

**Symptoms:**
- Survival signal remains low
- Agents not improving performance

**Causes:**
- Threshold too high
- Learning rate too low
- Agent step() method not updating state correctly

**Solutions:**
1. Lower survival threshold:
   ```python
   constraint = SurvivalConstraint(threshold=0.5)  # Start lower
   ```

2. Check agent implementation:
   ```python
   # Ensure step() updates state using learning_rate
   self.state += self.learning_rate * error
   ```

3. Verify get_performance_metric() returns correct values (0-1 range)

### Issue: High Memory Usage

**Symptoms:**
- Memory usage exceeds expectations
- Out of memory errors with many agents

**Causes:**
- Too many snapshots retained
- Large agent state
- Cache size too large

**Solutions:**
1. Reduce snapshot retention:
   ```python
   config = Config(
       max_snapshots=10,  # Default is 100
       snapshot_interval=5.0  # Less frequent snapshots
   )
   ```

2. Reduce cache size:
   ```python
   config = Config(gradient_cache_size=64)
   ```

3. Limit number of agents:
   ```python
   config = Config(max_agents=50)
   ```

### Issue: Slow Performance

**Symptoms:**
- Steps taking too long
- Not meeting latency targets

**Causes:**
- Too many agents
- Expensive operations in step()
- Normalization enabled unnecessarily

**Solutions:**
1. Disable normalization if not needed:
   ```python
   config = Config(enable_normalization=False)
   ```

2. Use faster NGCM implementation:
   ```python
   config = Config(cache_implementation="LUT")  # Fastest
   ```

3. Reduce threshold detection frequency:
   ```python
   config = Config(threshold_detection_interval=0.2)  # Less frequent
   ```

### Issue: Rollback Not Working

**Symptoms:**
- Runtime enters error state
- No recovery snapshots available

**Causes:**
- No snapshots created before error
- Snapshot interval too long
- Critical threshold too low

**Solutions:**
1. Ensure snapshots are created:
   ```python
   config = Config(
       snapshot_interval=0.1,  # Create snapshots frequently
       critical_survival_threshold=0.3  # Reasonable threshold
   )
   ```

2. Check snapshot manager:
   ```python
   snapshot_count = runtime.sprs.get_snapshot_count()
   if snapshot_count == 0:
       # Need to run more steps to create snapshots
   ```

### Issue: Agents Not Adapting Parameters

**Symptoms:**
- Learning rate and exploration rate not changing
- No adaptation observed

**Causes:**
- Survival signal not varying
- All agents above/below threshold
- APC not receiving gradient updates

**Solutions:**
1. Check survival signal values:
   ```python
   stats = runtime.get_statistics()
   print(f"Survival signal: {stats['average_survival_signal']}")
   ```

2. Ensure agents have varying performance:
   - Some agents should be below threshold
   - Some agents should be above threshold

3. Verify gradient computation:
   ```python
   stats = runtime.get_statistics()
   print(f"NGCM cache hit rate: {stats['ngcm_cache_hit_rate']}")
   ```

## Performance Tuning Checklist

- [ ] Cache hit rate > 70%? If not, increase cache size
- [ ] Memory usage acceptable? If not, reduce snapshots/cache
- [ ] Latency < 1ms? If not, disable normalization, use LUT
- [ ] Agents converging? If not, check threshold and agent implementation
- [ ] Snapshots being created? If not, reduce snapshot_interval

## Debugging Tips

### Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Runtime State

```python
print(f"Runtime state: {runtime.state}")
print(f"Statistics: {runtime.get_statistics()}")
```

### Monitor Metrics

```python
from pulseos.telemetry.enhanced_metrics import EnhancedMetricsCollector

collector = EnhancedMetricsCollector()
# ... use collector in your code ...
report = collector.export_comprehensive_report()
print(report)
```

### Profile Performance

```python
from pulseos.telemetry.profiler import PerformanceProfiler

profiler = PerformanceProfiler(enabled=True)
profiler.start()
# ... run your code ...
profiler.stop()
bottlenecks = profiler.get_bottlenecks()
print(bottlenecks)
```

## Getting Help

If you encounter issues not covered here:

1. Check the examples in `examples/` directory
2. Review `TECHNICAL.md` for implementation details
3. Check test files in `tests/` for usage examples
4. Open an issue on GitHub with:
   - Error messages
   - Code snippet
   - Configuration used
   - Expected vs actual behavior

