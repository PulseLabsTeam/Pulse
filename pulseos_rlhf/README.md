# PulseOS RLHF Training - Real LLM Validation

This directory contains the implementation for validating PulseOS on real LLM RLHF training.

## Objective

Get PulseOS working on real LLM RLHF training to validate the 91% simulation improvement.

**Target:** 20%+ improvement over baseline PPO on real RLHF task

**Timeline:** 30 days maximum

## Structure

- `week1_*.py` - Week 1: Baseline setup and reward model training
- `baseline_ppo.py` - Baseline PPO implementation
- `death_penalty.py` - Death penalty mechanism (Week 2)
- `population_training.py` - Population training (Week 2)
- `runtime_adaptation.py` - Runtime alpha adaptation (Week 3)
- `full_pulseos.py` - Full PulseOS integration (Week 3)
- `evaluation_harness.py` - Validation and statistical analysis (Week 4)
- `configs/` - Configuration files for each phase
- `notebooks/` - Colab-compatible notebooks

## Quick Start

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Verify environment
python week1_setup.py
```

### 2. Week 1 Day 1-2: Verify TRL Works

```bash
# Run TRL sentiment example (unchanged)
python week1_sentiment.py
```

**Success criteria:**
- Script runs without errors
- Rewards show learning trend (increasing over time)
- Environment is ready for RLHF

**If local fails (macOS bus errors):**
- Use Colab notebook: `notebooks/01_sentiment_demo.ipynb`
- Upload to Google Colab and run (free T4 GPU)

### 3. Week 1 Day 3-4: Train Reward Model

```bash
# Train reward model on HH-RLHF
python train_reward_model.py
```

### 4. Week 1 Day 5-7: Baseline PPO

```bash
# Run baseline PPO with trained reward model
python baseline_ppo.py
```

## Compute Environment

**Preferred:** Local (your Mac)

**Fallback:** Google Colab free tier
- Upload notebooks to Colab
- Free T4 GPU available
- 12-hour runtime limit (sufficient for testing)

## Week-by-Week Plan

### Week 1: Get Baseline Working (CRITICAL)
- Day 1-2: TRL sentiment example ✓
- Day 3-4: Train reward model on HH-RLHF
- Day 5-7: Baseline PPO with reward model

**Exit condition:** If Week 1 fails, STOP - sell simulation results only.

### Week 2: Add Death Penalty
- Day 8-10: Implement death penalty mechanism
- Day 11-14: Add population training

### Week 3: Add Runtime and Full PulseOS
- Day 15-18: Runtime alpha adaptation
- Day 19-21: Full PulseOS integration

### Week 4: Validation and Scaling
- Day 22-25: Rigorous testing (10 trials per method)
- Day 26-28: Scale to different model sizes
- Day 29-30: Documentation and writeup

## Success Criteria

- **Week 1:** Baseline PPO working, reward model trained, reproducible learning curves
- **Week 2:** Death penalty shows 10-20% improvement
- **Week 3:** Full PulseOS shows 20-40% improvement
- **Week 4:** Validated across 10+ trials, 20%+ improvement, documented

## Troubleshooting

### macOS Bus Errors
If you get bus errors on macOS:
1. Use Colab notebooks instead
2. Or use Linux/cloud GPU

### Missing Dependencies
```bash
pip install -r requirements.txt
```

### CUDA Errors
- Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`
- Use CPU if GPU unavailable (slower but works)

## References

- TRL Library: https://github.com/huggingface/trl
- HH-RLHF Dataset: https://huggingface.co/datasets/Anthropic/hh-rlhf
- PulseOS Runtime: `../pulseos/runtime.py`


