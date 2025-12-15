# Real LLM RLHF Experiment: PulseOS vs Baseline PPO

## Overview

This is the **critical $100M+ test** that validates PulseOS survival-pressure learning on REAL LLM RLHF training.

**Objective:** Test whether PulseOS reduces sample complexity in REAL RLHF training compared to standard PPO.

**Expected Result:** PulseOS reaches target reward in 40-60% fewer samples.

**Valuation Impact:**
- 40-60% reduction: $50M-$150M valuation
- 20-40% reduction: $30M-$70M valuation
- 10-20% reduction: $15M-$40M valuation
- <10% reduction: $10M-$25M valuation

## Technical Specifications

### Models
- **Base Model:** GPT-2 (124M parameters) from Hugging Face
- **Free:** Runs on single GPU
- **Well-studied:** Standard baseline for RLHF research

### Dataset
- **HH-RLHF** (Helpful and Harmless RLHF) dataset
- **Source:** `Anthropic/hh-rlhf` on Hugging Face
- **Contains:** Human preference comparisons
- **Standard benchmark** for RLHF research

### Comparison
- **Baseline:** Standard PPO RLHF (10 trials)
- **PulseOS:** PulseOS RLHF with survival pressure (10 trials)
- **Metric:** Samples needed to reach target reward score

## Setup

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure you have:
# - Python 3.9+
# - CUDA-capable GPU (recommended) or CPU
# - ~10GB disk space for models and datasets
```

### Dependencies

- `torch>=2.0.0`
- `transformers>=4.30.0`
- `trl>=0.7.0`
- `datasets>=2.14.0`
- `scipy>=1.10.0`
- `matplotlib>=3.7.0`

## Running the Experiment

### Basic Usage

```bash
cd benchmarks
python real_llm_rlhf_experiment.py
```

### Configuration

Edit `ExperimentConfig` in the script to customize:

```python
@dataclass
class ExperimentConfig:
    model_name: str = "gpt2"  # 124M params
    dataset_name: str = "Anthropic/hh-rlhf"
    num_trials: int = 10  # Trials per method
    max_samples: int = 10000  # Max samples per trial
    target_reward: float = 0.7  # Target reward for convergence
    batch_size: int = 8  # PPO batch size
    learning_rate: float = 1.41e-5  # Standard PPO LR
```

### Expected Timeline

- **Setup:** 4 hours (dataset download, reward model training)
- **Baseline trials (10):** 8 hours
- **PulseOS trials (10):** 8 hours
- **Analysis:** 2 hours
- **Total:** ~22 hours (1 weekend)

## Output

### Results Files

Results are saved to `benchmark_results/real_llm_rlhf/`:

1. **`experiment_results.json`** - Complete results data
2. **`rlhf_experiment_results.png`** - Visualization plots

### Key Metrics

- **Sample Efficiency:** Mean samples to convergence (PPO vs PulseOS)
- **Improvement:** Percentage reduction in samples
- **Statistical Significance:** p-value from t-test
- **Effect Size:** Cohen's d

### Visualization

The script generates 4 plots:

1. **Learning Curves** - Reward vs samples for both methods
2. **Sample Efficiency Comparison** - Bar chart of mean samples
3. **Distribution Comparison** - Histogram of sample distributions
4. **Convergence Rate** - Percentage of trials that converged

## How It Works

### Baseline PPO RLHF

1. Load GPT-2 model with value head
2. Train reward model on preference pairs
3. Use standard PPO with fixed learning rate
4. Track samples until convergence

### PulseOS RLHF

1. Load GPT-2 model with value head
2. Train reward model on preference pairs
3. **Apply survival pressure:**
   - Calculate survival signal from recent performance
   - Apply death penalty when performance drops below threshold
   - Modify rewards based on survival pressure
4. **Adaptive learning:**
   - Learning rate adapts based on survival signal
   - Exploration rate adapts based on survival signal
   - Parameters update via PulseOS runtime
5. Track samples until convergence

### Survival Pressure Mechanism

```python
# Calculate survival signal
survival_signal = 1 / (1 + exp(-5 * distance_from_baseline))

# Apply penalty when DYING
if survival_signal < threshold:
    penalty = death_penalty * (threshold - survival_signal)
    modified_reward = base_reward + penalty
```

## Success Criteria

**Primary Metric:** Sample efficiency

- Baseline: X samples to reach target reward
- PulseOS: Y samples to reach target reward
- **Target: Y < 0.6 * X (40%+ improvement)**

**Secondary Metrics:**

- Final reward quality (should be equal or better)
- Training stability (lower variance)
- Convergence reliability (% of trials that converge)

## Troubleshooting

### Out of Memory

- Reduce `batch_size` in config
- Reduce `max_length` in config
- Use gradient checkpointing

### Slow Training

- Use GPU (CUDA)
- Reduce `max_samples` for testing
- Reduce `num_trials` for quick tests

### Dataset Download Issues

- Check internet connection
- Try manual download: `python -c "from datasets import load_dataset; load_dataset('Anthropic/hh-rlhf')"`

## Interpretation

### Results Interpretation

- **40-60% reduction:** Excellent - validates PulseOS value proposition
- **20-40% reduction:** Good - meaningful improvement
- **10-20% reduction:** Modest - some improvement
- **<10% reduction:** Low - simulation results don't transfer

### Next Steps

If results show 40%+ improvement:

1. **Update whitepaper** with real LLM results
2. **Prepare investor materials** with validated data
3. **Start outreach** to AI labs (Anthropic, OpenAI, Google, Meta)
4. **Scale up** to larger models (GPT-2 Medium, GPT-2 Large)

## References

- HH-RLHF Dataset: https://huggingface.co/datasets/Anthropic/hh-rlhf
- TRL Library: https://github.com/huggingface/trl
- PPO Paper: https://arxiv.org/abs/1707.06347


