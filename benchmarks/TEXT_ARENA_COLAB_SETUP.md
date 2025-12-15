# 🚀 Text Generation Arena - Complete Colab Setup

## Step-by-Step Instructions

### Step 1: Install Dependencies
**Create Cell 1:**
```
!pip install -q torch transformers datasets matplotlib numpy
print("✓ Dependencies installed")
```

### Step 2: Check GPU
**Create Cell 2:**
```
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ GPU not detected! Enable GPU in Runtime settings")
```

### Step 3: Upload Code File
**Create Cell 3:**
```
from google.colab import files
uploaded = files.upload()
print("✓ File uploaded")
```

Upload `text_generation_arena.py` from Desktop

### Step 4: Import and Run
**Create Cell 4:**
```
import sys
sys.path.insert(0, '.')

from text_generation_arena import main, ArenaConfig

# Quick test configuration (30-45 minutes)
config = ArenaConfig(
    n_agents=20,
    model_name="gpt2",
    n_steps=100,  # Quick test
    elimination_interval=20,
    elimination_rate=0.3,
    spawn_rate=0.2,
    device="cuda"
)

print("🚀 Starting Text Generation Arena Test...")
print(f"Agents: {config.n_agents}")
print(f"Steps: {config.n_steps}")
print(f"Elimination every: {config.elimination_interval} steps")
print("="*80)

results = main()
```

### Step 5: View Results
Results will be displayed automatically, including:
- Learning curves plot
- Improvement percentage
- Valuation assessment

---

## Full Test Configuration (2-3 hours)

If quick test works, run full test:

```
config = ArenaConfig(
    n_agents=20,
    model_name="gpt2",
    n_steps=500,  # Full test
    elimination_interval=50,
    elimination_rate=0.3,
    spawn_rate=0.2,
    device="cuda"
)
```

---

## Expected Output

```
Step    0 | Mean PPL:  45.23 | Best PPL:  42.10 | Worst PPL:  48.50
Step   10 | Mean PPL:  38.15 | Best PPL:  35.20 | Worst PPL:  41.80
...
⚠️  Generation 1: Eliminated 6 agents, Spawned 6 new agents
...
RESULTS
Baseline final perplexity: 32.10
PulseOS final perplexity:  24.70
Final improvement: 23.1%
```

**Ready to run!** 🎯


