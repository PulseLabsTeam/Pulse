# 🔧 Memory Fix - Clear GPU and Use Fewer Agents

**First, run this cell to clear GPU memory:**

```
import torch
import gc

# Clear GPU cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()
    print("✓ GPU memory cleared")
```

**Then replace Cell 4 with this (uses only 5 agents):**

```
import sys
sys.path.insert(0, '.')

from text_generation_arena import ArenaConfig, TextArenaExperiment
import torch
import gc

# Clear GPU memory again
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

# VERY memory-efficient configuration
config = ArenaConfig(
    n_agents=5,  # Only 5 agents to fit in GPU
    model_name="gpt2",
    n_steps=100,
    elimination_interval=20,
    elimination_rate=0.3,
    spawn_rate=0.2,
    batch_size=2,  # Very small batches
    eval_batch_size=4,  # Small eval batches
    max_length=64,  # Shorter sequences
    device="cuda"
)

print("🚀 Starting Text Generation Arena Test...")
print(f"Agents: {config.n_agents} (reduced for GPU memory)")
print(f"Batch size: {config.batch_size}")
print(f"Max length: {config.max_length}")
print("="*80)

experiment = TextArenaExperiment(config)
baseline_results = experiment.run_baseline()
pulseos_results = experiment.run_pulseos()
results = experiment.analyze_results(baseline_results, pulseos_results)
```

## Changes:
- ✅ `n_agents=5` (was 20) - Much fewer models
- ✅ `batch_size=2` (was 8) - Tiny batches
- ✅ `max_length=64` (was 128) - Shorter sequences
- ✅ Clear GPU memory first

**Run the clear memory cell first, then this Cell 4!** 🎯


