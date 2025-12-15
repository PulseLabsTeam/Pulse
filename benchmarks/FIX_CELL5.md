# 🔧 FIX: Use Full Experiment Config

## The Problem:
The experiment is using OLD test settings (1 trial, 20 samples) instead of FULL settings (10 trials, 10000 samples).

## The Fix:

**Replace Cell 5 with this code** (explicitly sets all parameters):

```
import asyncio
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, 'benchmarks')

from real_llm_rlhf_experiment import run_experiment, ExperimentConfig

# FULL EXPERIMENT - EXPLICITLY SET ALL PARAMETERS
config = ExperimentConfig(
    model_name="gpt2",
    dataset_name="Anthropic/hh-rlhf",
    num_trials=10,  # 10 trials for statistical significance
    max_samples=10000,  # Up to 10k samples per trial
    target_reward=0.7,  # Realistic target
    convergence_window=20,  # Stable convergence detection
    batch_size=8,  # Larger batches
    max_length=128,  # Longer sequences
    learning_rate=1.41e-5,
    ppo_epochs=4,  # More training epochs
    ppo_clip_epsilon=0.2,
    seed=42,
    output_dir="benchmark_results/real_llm_rlhf",
    device="cuda",  # Use GPU!
    reward_model_samples=1000,  # Better reward model
    reward_model_epochs=3,  # More reward model training
    dataset_size=None  # Use FULL dataset
)

print("🚀 Starting FULL RLHF Experiment...")
print(f"Device: {config.device}")
print(f"Trials: {config.num_trials} each (baseline + PulseOS)")
print(f"Max samples per trial: {config.max_samples}")
print(f"Target reward: {config.target_reward}")
print(f"Dataset: {'FULL' if config.dataset_size is None else config.dataset_size}")
print("="*80)
print("⚠️ This will take 2-4 hours. Don't close browser!")
print("="*80)

results = await run_experiment(config)
```

## What Changed:
- ✅ `num_trials=10` (was 1)
- ✅ `max_samples=10000` (was 20)
- ✅ `target_reward=0.7` (was 0.5)
- ✅ `dataset_size=None` (was 200)
- ✅ All other full experiment settings

**Stop the current Cell 5, replace it with the code above, and run it!** 🎯


