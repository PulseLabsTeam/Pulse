# 🚀 Cell 5: FINAL FIXED Version

**Copy this ENTIRE code into Cell 5:**

```
import asyncio
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, 'benchmarks')

# IMPORTANT: Reload module to get updated code
import importlib
if 'real_llm_rlhf_experiment' in sys.modules:
    importlib.reload(sys.modules['real_llm_rlhf_experiment'])

from real_llm_rlhf_experiment import run_experiment, ExperimentConfig

# FINAL FIXED CONFIGURATION - Target matches actual reward range
config = ExperimentConfig(
    model_name="gpt2",
    dataset_name="Anthropic/hh-rlhf",
    num_trials=1,  # 1 trial each
    max_samples=10000,
    target_reward=12.0,  # Match actual reward range (rewards are 8-15)
    convergence_window=50,  # Need 50 consecutive samples above threshold
    min_samples=200,  # Don't check convergence before 200 samples
    batch_size=8,
    max_length=128,
    learning_rate=1.41e-5,
    ppo_epochs=4,
    ppo_clip_epsilon=0.2,
    seed=42,
    output_dir="benchmark_results/real_llm_rlhf",
    device="cuda",
    reward_model_samples=5000,  # Better reward model
    reward_model_epochs=10,  # More training
    dataset_size=20000
)

print("🚀 Starting FINAL FIXED Test...")
print(f"Device: {config.device}")
print(f"Trials: {config.num_trials} each")
print(f"Max samples: {config.max_samples}")
print(f"Target reward: {config.target_reward} (matches reward range 8-15)")
print(f"Min samples: {config.min_samples}")
print(f"Convergence window: {config.convergence_window}")
print("="*80)
print("⚠️ Expected time: 45-90 minutes")
print("="*80)

results = await run_experiment(config)
```

## Key Fixes:
- ✅ `target_reward=12.0` - Matches actual reward range (8-15)
- ✅ Fixed code bug (missing return statement)
- ✅ All other fixes included

## Expected Results:
- Baseline PPO: Should take 300-800+ samples to reach 12.0
- PulseOS: Should show improvement if more efficient
- **Real sample complexity differences!**

**Copy-paste this entire code into Cell 5!** 🎯


