# Getting Started with PulseOS

Welcome to PulseOS! This guide will help you get started with survival-pressure learning.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Creating Your First Agent](#creating-your-first-agent)
5. [Understanding Constraints](#understanding-constraints)
6. [Runtime Configuration](#runtime-configuration)
7. [Next Steps](#next-steps)

## Installation

```bash
# Clone the repository
git clone https://github.com/pulseos/pulseos.git
cd pulseos

# Install dependencies
pip install -r requirements.txt

# Install PulseOS
pip install -e .
```

## Quick Start

```python
import asyncio
from pulseos import Runtime, Agent, SurvivalConstraint

class MyAgent(Agent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.value = 0.0
        self.target = 0.8
    
    async def step(self) -> dict:
        error = self.target - self.value
        self.value += self.learning_rate * error
        self.value = max(0.0, min(1.0, self.value))
        return {"value": self.value}
    
    def get_performance_metric(self) -> float:
        error = abs(self.target - self.value)
        return 1.0 - error

async def main():
    constraint = SurvivalConstraint(threshold=0.7)
    runtime = Runtime(constraint=constraint)
    
    agent = MyAgent("agent_1")
    runtime.register_agent("agent_1", agent)
    
    await runtime.run(max_steps=100)
    
    stats = runtime.get_statistics()
    print(f"Survival signal: {stats['average_survival_signal']:.3f}")

asyncio.run(main())
```

## Core Concepts

### Survival-Pressure Learning

PulseOS implements a novel learning paradigm where agents adapt based on "survival pressure":

- **Survival Signal**: Measures how well agents meet performance thresholds
- **Adaptive Parameters**: Learning rate and exploration rate adjust automatically
- **Constraint Satisfaction**: Agents must maintain performance above thresholds

### Key Components

1. **Agent**: Implements `step()` and `get_performance_metric()`
2. **SurvivalConstraint**: Defines performance thresholds
3. **Runtime**: Orchestrates learning loop and parameter adaptation
4. **Circuits**: Core algorithms (PTDC, NGCM, APC)

## Creating Your First Agent

Every agent must inherit from `Agent` and implement two methods:

```python
class MyAgent(Agent):
    async def step(self) -> dict:
        # Agent behavior here
        return {"result": "success"}
    
    def get_performance_metric(self) -> float:
        # Return performance (0-1 scale)
        return 0.8
```

### Using Learning Rate and Exploration Rate

PulseOS automatically updates these parameters:

```python
async def step(self) -> dict:
    if random.random() > self.exploration_rate:
        # Exploit: use current best strategy
        action = best_action()
    else:
        # Explore: try random actions
        action = random_action()
    
    # Use learning rate for updates
    self.state += self.learning_rate * update
```

## Understanding Constraints

### Simple Constraint

```python
constraint = SurvivalConstraint(threshold=0.7)
# Agents must maintain performance >= 0.7
```

### Temporal Constraint

```python
constraint = SurvivalConstraint(
    threshold=0.7,
    constraint_type="temporal",
    temporal_window=10
)
# Must maintain >= 0.7 over last 10 steps
```

### Statistical Constraint

```python
constraint = SurvivalConstraint(
    threshold=0.7,
    constraint_type="statistical",
    statistical_mode="mean"
)
# Average performance >= 0.7
```

### Adaptive Threshold

```python
constraint = SurvivalConstraint(
    threshold=0.7,
    learning_rate=0.01
)
# Threshold adapts to agent capabilities
```

## Runtime Configuration

Customize runtime behavior:

```python
from pulseos import Config

config = Config(
    gradient_cache_size=512,      # Larger cache
    cache_implementation="LUT",   # Fastest lookup
    update_batch_size=50,         # Batch processing
    snapshot_interval=1.0         # State snapshots
)

runtime = Runtime(constraint=constraint, config=config)
```

## Examples

Check out the `examples/` directory:

- `basic_survival.py` - Simple agent example
- `robotics_safety.py` - Safety-critical robotics
- `finance_portfolio.py` - Portfolio optimization
- `game_ai.py` - Multi-agent game AI
- `custom_constraints.py` - Advanced constraint types
- `performance_tuning.py` - Performance optimization
- `swarm_coordination.py` - Large-scale swarms

## Next Steps

1. **Run Examples**: Try the examples in `examples/` directory
2. **Read Technical Docs**: See `TECHNICAL.md` for algorithm details
3. **Explore API**: Check API documentation (coming soon)
4. **Performance Tuning**: See `examples/performance_tuning.py`
5. **Custom Constraints**: See `examples/custom_constraints.py`

## Common Patterns

### Multi-Agent Coordination

```python
runtime = Runtime(constraint=constraint)

for i in range(100):
    agent = MyAgent(f"agent_{i}")
    runtime.register_agent(f"agent_{i}", agent)

await runtime.run(max_steps=1000)
```

### Accessing Statistics

```python
stats = runtime.get_statistics()
print(f"Survival signal: {stats['average_survival_signal']:.3f}")
print(f"Learning rate: {stats['current_alpha']:.6f}")
print(f"Exploration rate: {stats['current_epsilon']:.3f}")
```

### State Persistence

```python
# Runtime automatically creates snapshots
# Access via runtime.snapshot_manager if needed
```

## Troubleshooting

### Low Survival Signal

- Check if threshold is too high
- Verify agents are improving over time
- Consider adaptive threshold

### Slow Convergence

- Increase learning rate (via config)
- Adjust exploration rate
- Check cache hit rate

### Memory Issues

- Reduce `max_agents` in config
- Increase `snapshot_interval`
- Enable delta encoding

## Resources

- **Documentation**: `README.md`, `TECHNICAL.md`
- **Examples**: `examples/` directory
- **Tests**: `tests/` directory
- **Issues**: GitHub Issues

## Getting Help

- Check examples for similar use cases
- Review `TECHNICAL.md` for algorithm details
- Open an issue on GitHub

Happy learning with PulseOS! 🚀

