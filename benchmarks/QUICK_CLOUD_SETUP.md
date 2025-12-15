# Quick Cloud GPU Setup Guide

## Option 1: Google Colab (Easiest - FREE)

### Step 1: Open Colab
1. Go to: https://colab.research.google.com
2. Sign in with Google account (free)

### Step 2: Upload Notebook
1. Click "File" → "Upload notebook"
2. Upload `rlhf_colab.ipynb` from this folder
3. OR create new notebook and copy code from below

### Step 3: Enable GPU
1. Click "Runtime" → "Change runtime type"
2. Select "GPU" → "T4" (free tier)
3. Click "Save"

### Step 4: Upload Experiment File
1. In Colab, click the folder icon (left sidebar)
2. Click "Upload" button
3. Upload `real_llm_rlhf_experiment.py`
4. Also upload the `pulseos` folder (or install from GitHub)

### Step 5: Run
1. Click "Runtime" → "Run all"
2. Wait ~5 minutes
3. Download results when done

**Cost: FREE** (12+ hours GPU per day)

---

## Option 2: Kaggle Notebooks (Also FREE)

1. Go to: https://www.kaggle.com/code
2. Create new notebook
3. Enable GPU (Settings → Accelerator → GPU)
4. Upload files and run

**Cost: FREE** (30 hours GPU per week)

---

## Option 3: Hugging Face Spaces (FREE)

1. Go to: https://huggingface.co/spaces
2. Create new Space
3. Enable GPU
4. Upload code

**Cost: FREE** (limited hours)

---

## What You Need to Upload:

1. `real_llm_rlhf_experiment.py` - Main experiment file
2. `pulseos/` folder - Or install from GitHub if public
3. (Optional) `requirements.txt` - For dependencies

---

## Quick Colab Code (Copy-Paste):

```python
# Install dependencies
!pip install -q torch transformers trl datasets scipy matplotlib numpy

# Check GPU
import torch
print(f"GPU: {torch.cuda.is_available()}")

# Upload files (use Colab file uploader)
from google.colab import files
files.upload()  # Upload real_llm_rlhf_experiment.py

# Run experiment
import asyncio
import sys
sys.path.insert(0, '.')
from real_llm_rlhf_experiment import run_experiment, ExperimentConfig

config = ExperimentConfig(
    device="cuda",
    num_trials=1,
    max_samples=20
)

results = await run_experiment(config)
print(f"Improvement: {results.improvement_percent:.1f}%")
```

---

## Expected Results:

- **Baseline PPO:** X samples to convergence
- **PulseOS:** Y samples to convergence  
- **Improvement:** (X-Y)/X * 100%

**Target:** 40-60% improvement = $50M-$150M valuation


