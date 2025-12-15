# Minimal Viable Benchmark Suite

Professional benchmark suite comparing PulseOS vs PPO baseline across 4 critical tests.

## Tests

1. **CartPole-v1**: Classic control task (5 trials each)
2. **LunarLander-v2**: Continuous control task (5 trials each)
3. **RLHF Simulation**: Preference learning task (5 trials each)
4. **Multi-Agent PettingZoo**: simple_spread with 3 agents (5 trials each)

## Installation

```bash
# Install benchmark dependencies
pip install -r benchmarks/requirements.txt

# Or install specific packages:
pip install gymnasium pettingzoo[mpe] matplotlib
```

## Usage

```bash
# Run all benchmarks
python benchmarks/minimal_benchmark_suite.py

# Results will be saved to benchmark_results/
```

## Output

The benchmark suite generates:

- **CSV files**: Detailed results for each test (`*_results.csv`)
- **Learning curve plots**: Comparison plots for each test (`*_learning_curves.png`)
- **Report**: Professional markdown report (`BENCHMARK_REPORT.md`)

## Metrics Collected

For each trial:
- Steps to convergence
- Total time
- Final reward
- Learning curve (reward over time)

## Report Summary

The report includes:
- Executive summary with average improvements
- Results table with statistics
- Detailed per-test statistics
- Learning curve visualizations
- Conclusion with overall performance

## Example Output

```
PulseOS achieves X% average step reduction and Y% average time reduction across 4 benchmarks.
```

