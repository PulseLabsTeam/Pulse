# 🚀 Cell 5: FIXED Configuration (Proper Test)

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

# FIXED CONFIGURATION - Proper test settings
config = ExperimentConfig(
    model_name="gpt2",
    dataset_name="Anthropic/hh-rlhf",
    num_trials=1,  # 1 trial each
    max_samples=10000,
    target_reward=1.5,  # Match actual reward range (rewards are 1.5-2.0)
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
    reward_model_samples=5000,  # Better reward model (was 1000)
    reward_model_epochs=10,  # More training (was 3)
    dataset_size=20000
)

print("🚀 Starting FIXED VALIDATION Test...")
print(f"Device: {config.device}")
print(f"Trials: {config.num_trials} each")
print(f"Max samples: {config.max_samples}")
print(f"Target reward: {config.target_reward}")
print(f"Min samples before convergence check: {config.min_samples}")
print(f"Convergence window: {config.convergence_window}")
print(f"Reward model: {config.reward_model_samples} samples, {config.reward_model_epochs} epochs")
print("="*80)
print("⚠️ Expected time: 45-90 minutes (better reward model training)")
print("="*80)

results = await run_experiment(config)
```

## Key Fixes:
- ✅ `target_reward=1.5` - Matches actual reward range
- ✅ `min_samples=200` - No convergence check before 200 samples
- ✅ `convergence_window=50` - Need 50 consecutive samples above threshold
- ✅ `reward_model_samples=5000` - Better reward model training
- ✅ `reward_model_epochs=10` - More epochs for reward model

## Expected Results:
- Baseline PPO: Should take 200-500+ samples
- PulseOS: Should show improvement if more efficient
- **Real sample complexity differences!**

**Copy-paste this entire code into Cell 5!** 🎯


