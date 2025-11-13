# PulseOS RLHF - Quick Start Guide

## Current Status: macOS Bus Error

We're encountering the macOS bus error during model generation. This is a known issue with macOS ARM + PyTorch.

## Solution: Use Google Colab (Free GPU)

**Recommended:** Use the Colab notebook for all RLHF training.

### Steps:

1. **Open Google Colab**: https://colab.research.google.com/

2. **Upload the notebook**: 
   - File → Upload notebook
   - Upload `pulseos_rlhf/notebooks/01_sentiment_demo.ipynb`

3. **Run the notebook**:
   - Runtime → Change runtime type → GPU (T4 free)
   - Run all cells

4. **Verify it works**:
   - Should see rewards increasing over 10 steps
   - If successful, proceed to reward model training

### Alternative: Fix macOS Issues

If you want to run locally, you can try:

1. **Use Python 3.9** (instead of 3.13):
   ```bash
   source venv39/bin/activate
   python pulseos_rlhf/week1_sentiment.py
   ```

2. **Use CPU-only mode** (slower but avoids bus error):
   - Already using CPU, but bus error still occurs

3. **Wait for PyTorch fix**:
   - Monitor: https://github.com/pytorch/pytorch/issues

## Next Steps After Colab Works

Once the sentiment example works in Colab:

1. **Day 3-4**: Train reward model
   - Upload `train_reward_model.py` to Colab
   - Or create Colab notebook version

2. **Day 5-7**: Baseline PPO
   - Upload `baseline_ppo.py` to Colab
   - Or create Colab notebook version

3. **Continue through all weeks** using Colab

## Files Ready for Colab

All implementation files are ready:
- ✅ Week 1: Setup, sentiment, reward model, baseline PPO
- ✅ Week 2: Death penalty, population
- ✅ Week 3: Runtime, full PulseOS
- ✅ Week 4: Evaluation, scaling, documentation

Just need to adapt them for Colab environment (or run as Python scripts in Colab).


