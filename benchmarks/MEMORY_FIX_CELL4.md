# 🔧 Memory Fix - Updated Cell 4

**Replace Cell 4 with this (reduced agents and batch sizes):**

```
import sys
sys.path.insert(0, '.')

from text_generation_arena import main, ArenaConfig, TextArenaExperiment
import torch

# Clear GPU cache first
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Memory-optimized configuration
config = ArenaConfig(
    n_agents=10,  # Reduced from 20 to fit GPU memory
    model_name="gpt2",
    n_steps=100,
    elimination_interval=20,
    elimination_rate=0.3,
    spawn_rate=0.2,
    batch_size=4,  # Smaller batches
    eval_batch_size=8,  # Smaller eval batches
    device="cuda"
)

print("🚀 Starting Text Generation Arena Test...")
print(f"Agents: {config.n_agents} (reduced for GPU memory)")
print(f"Batch size: {config.batch_size}")
print("="*80)

experiment = TextArenaExperiment(config)
baseline_results = experiment.run_baseline()
pulseos_results = experiment.run_pulseos()
results = experiment.analyze_results(baseline_results, pulseos_results)
```

## Changes:
- ✅ `n_agents=10` (was 20) - Half the models
- ✅ `batch_size=4` (was 8) - Smaller batches
- ✅ `eval_batch_size=8` (was 16) - Smaller eval batches
- ✅ Added `torch.cuda.empty_cache()` to clear GPU memory

**Run this updated Cell 4!** 🎯


