# 🚀 Cloud GPU Setup - Step by Step

## Google Colab (FREE - Recommended)

### What You Need:
- Google account (free)
- 5 minutes
- **NO credit card**
- **NO payment**

### Steps:

1. **Go to Colab:**
   - Visit: https://colab.research.google.com
   - Sign in with Google

2. **Create New Notebook:**
   - Click "New notebook"
   - Or upload `rlhf_colab.ipynb` if you have it

3. **Enable GPU (IMPORTANT!):**
   - Click "Runtime" → "Change runtime type"
   - Hardware accelerator: Select **"GPU"**
   - GPU type: **"T4"** (free tier)
   - Click **"Save"**

4. **Install Dependencies:**
   Copy and paste this in first cell:
   ```python
   !pip install -q torch transformers trl datasets scipy matplotlib numpy
   ```
   Click Run (or Shift+Enter)

5. **Upload Files:**
   - Click folder icon (📁) in left sidebar
   - Click "Upload" button
   - Upload these files:
     - `real_llm_rlhf_experiment.py`
     - `pulseos/` folder (or we'll install from GitHub)

6. **Install PulseOS:**
   ```python
   # Option A: If pulseos is on GitHub
   !pip install -q git+https://github.com/yourusername/pulsegithub.git
   
   # Option B: If you uploaded pulseos folder
   !pip install -e ./pulseos
   ```

7. **Run Experiment:**
   ```python
   import asyncio
   import sys
   import os
   
   os.environ["TOKENIZERS_PARALLELISM"] = "false"
   sys.path.insert(0, '.')
   
   from real_llm_rlhf_experiment import run_experiment, ExperimentConfig
   
   config = ExperimentConfig(
       device="cuda",  # Use GPU!
       num_trials=1,
       max_samples=20,
       dataset_size=200
   )
   
   results = await run_experiment(config)
   print(f"\n🎯 Improvement: {results.improvement_percent:.1f}%")
   ```

8. **Download Results:**
   ```python
   from google.colab import files
   files.download('benchmark_results/real_llm_rlhf/experiment_results.json')
   files.download('benchmark_results/real_llm_rlhf/rlhf_experiment_results.png')
   ```

## That's It!

- **Total time:** ~5-10 minutes
- **Cost:** $0
- **GPU:** Free T4 (faster than your Mac!)

## Troubleshooting:

**"GPU not available":**
- Make sure you selected GPU in Runtime settings
- Free tier gets T4 GPU (should work fine)

**"Module not found":**
- Run the pip install cell first
- Make sure files are uploaded

**"Bus error":**
- Won't happen on Colab! This is macOS-specific

## Next Steps After Results:

If you get 40%+ improvement:
- Update whitepaper
- Prepare investor materials
- Start outreach to AI labs

Let me know when you're ready and I'll help you run it!


