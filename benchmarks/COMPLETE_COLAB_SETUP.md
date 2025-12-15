# 🚀 Complete Colab Setup Guide - Text Generation Arena

## Starting Fresh - All Steps

### Step 1: Open Google Colab
1. Go to: https://colab.research.google.com
2. Sign in with your Google account
3. Click **"New notebook"** (or File → New notebook)

---

### Step 2: Enable GPU
1. Click **"Runtime"** in the menu bar
2. Click **"Change runtime type"**
3. Set **Hardware accelerator** to **"GPU"**
4. Set **GPU type** to **"T4"** (if available)
5. Click **"Save"**
6. Wait for GPU to connect (you'll see "T4" in bottom right)

---

### Step 3: Install Dependencies
**Create Cell 1** - Click "+ Code" button, then paste:

```
!pip install -q torch transformers datasets matplotlib numpy
print("✓ Dependencies installed")
```

**Run it** - Click the play button or press Shift+Enter
- Wait 1-2 minutes for installation
- Should see "✓ Dependencies installed"

---

### Step 4: Check GPU
**Create Cell 2** - Click "+ Code", then paste:

```
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU ready!")
else:
    print("⚠️ GPU not detected! Enable GPU in Runtime settings")
```

**Run it** - Should show:
```
CUDA available: True
GPU: Tesla T4
✓ GPU ready!
```

---

### Step 5: Upload Code File
**Create Cell 3** - Click "+ Code", then paste:

```
from google.colab import files
print("📁 Upload 'text_generation_arena.py':")
uploaded = files.upload()
print("✓ File uploaded")
```

**Run it** - Click "Choose Files" button
- Select `text_generation_arena.py` from your Desktop
- Wait for upload to complete
- Should see "✓ File uploaded"

---

### Step 6: Run Experiment
**Create Cell 4** - Click "+ Code", then paste:

```
import sys
sys.path.insert(0, '.')

from text_generation_arena import main

print("🚀 Starting Text Generation Arena Test...")
print("="*80)
print("This will test:")
print("  - Baseline: 20 agents training independently")
print("  - PulseOS: 20 agents with survival pressure (elimination + spawning)")
print("="*80)

results = main()
```

**Run it** - This will take 30-45 minutes (quick test) or 2-3 hours (full test)
- You'll see progress updates every 10 steps
- Will show elimination/spawning events
- Final results with improvement percentage

---

## What You'll See

### During Training:
```
Step    0 | Mean PPL:  45.23 | Best PPL:  42.10 | Worst PPL:  48.50
Step   10 | Mean PPL:  38.15 | Best PPL:  35.20 | Worst PPL:  41.80
...
⚠️  Generation 1: Eliminated 6 agents, Spawned 6 new agents
...
```

### Final Results:
```
RESULTS
Baseline final perplexity: 32.10
PulseOS final perplexity:  24.70
Final improvement: 23.1%

VALUATION ASSESSMENT
✅ GOOD: 23.1% improvement
   Valuation: $25M-$50M
```

---

## Quick Test vs Full Test

### Quick Test (30-45 min):
Uses default settings:
- 100 training steps
- Elimination every 20 steps
- Good for initial validation

### Full Test (2-3 hours):
Modify Cell 4 to use:

```
import sys
sys.path.insert(0, '.')

from text_generation_arena import main, ArenaConfig, TextArenaExperiment

# Full test configuration
config = ArenaConfig(
    n_agents=20,
    model_name="gpt2",
    n_steps=500,  # More steps
    elimination_interval=50,  # Eliminate every 50 steps
    elimination_rate=0.3,
    spawn_rate=0.2,
    device="cuda"
)

experiment = TextArenaExperiment(config)
baseline_results = experiment.run_baseline()
pulseos_results = experiment.run_pulseos()
results = experiment.analyze_results(baseline_results, pulseos_results)
```

---

## Troubleshooting

**"ModuleNotFoundError: No module named 'text_generation_arena'"**
→ Make sure you uploaded the file in Step 5

**"CUDA available: False"**
→ Go to Runtime → Change runtime type → Set GPU → Save

**"Out of memory"**
→ Reduce `n_agents` to 10 or `batch_size` to 4

**Experiment taking too long**
→ Normal! 30-45 min for quick test, 2-3 hours for full test

---

## Summary Checklist

- [ ] Opened Google Colab
- [ ] Enabled GPU (T4)
- [ ] Installed dependencies (Cell 1)
- [ ] Checked GPU (Cell 2)
- [ ] Uploaded `text_generation_arena.py` (Cell 3)
- [ ] Ran experiment (Cell 4)
- [ ] Waited for results

**Ready to start!** 🎯


