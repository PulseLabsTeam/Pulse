# 🔧 Quick Fix: Use Large Number Instead of None

## The Problem:
Python/Colab is caching old code or having issues with `None` check.

## The Fix:

**Change Cell 5** to use a large number instead of `None`:

```
import asyncio
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, 'benchmarks')

# IMPORTANT: Reload module to clear cache
import importlib
if 'real_llm_rlhf_experiment' in sys.modules:
    importlib.reload(sys.modules['real_llm_rlhf_experiment'])

from real_llm_rlhf_experiment import run_experiment, ExperimentConfig

# FULL EXPERIMENT - Use large number instead of None
config = ExperimentConfig(
    model_name="gpt2",
    dataset_name="Anthropic/hh-rlhf",
    num_trials=10,
    max_samples=10000,
    target_reward=0.7,
    convergence_window=20,
    batch_size=8,
    max_length=128,
    learning_rate=1.41e-5,
    ppo_epochs=4,
    ppo_clip_epsilon=0.2,
    seed=42,
    output_dir="benchmark_results/real_llm_rlhf",
    device="cuda",
    reward_model_samples=1000,
    reward_model_epochs=3,
    dataset_size=200000  # Large number = effectively full dataset
)

print("🚀 Starting FULL RLHF Experiment...")
print(f"Trials: {config.num_trials} each")
print(f"Max samples: {config.max_samples}")
print(f"Target reward: {config.target_reward}")
print("="*80)
print("⚠️ This will take 2-4 hours!")
print("="*80)

results = await run_experiment(config)
```

**Key change:** `dataset_size=200000` instead of `dataset_size=None`

This loads up to 200k samples (full dataset is ~160k, so this is effectively the full dataset).

---

**OR re-upload the new zip file `rlhf_full_experiment.zip`** which has the improved None handling.


