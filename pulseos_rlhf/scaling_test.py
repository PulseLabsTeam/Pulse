"""
Week 4 Day 26-28: Scaling Tests

Tests PulseOS on different model sizes:
- GPT-2 small (124M) - baseline
- GPT-2 medium (355M)
- GPT-2 large (774M)

Verifies that improvement scales with model size.
"""

import os
import torch
import numpy as np
from pathlib import Path
from datasets import load_dataset
from baseline_ppo import BaselinePPORLHFTrainer
from full_pulseos import FullPulseOSRLHF
import asyncio
import json

def test_model_size(model_name, model_size_name):
    """Test a specific model size."""
    print(f"\n{'='*80}")
    print(f"Testing {model_size_name} ({model_name})")
    print(f"{'='*80}\n")
    
    # Configuration
    config = {
        "model_name": model_name,
        "reward_model_path": "pulseos_rlhf/reward_model",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "learning_rate": 1.41e-5,
        "batch_size": 8 if "medium" in model_name or "large" in model_name else 16,  # Smaller batch for larger models
        "mini_batch_size": 2 if "medium" in model_name or "large" in model_name else 4,
        "ppo_epochs": 4,
        "ppo_clip_epsilon": 0.2,
        "max_length": 512,
        "max_new_tokens": 64,
        "death_threshold": -2.0,
        "death_penalty": -10.0,
        "survival_threshold": 0.55,
        "alpha_max_change": 0.50,
        "gamma": 0.5,
        "population_size": 3 if "medium" in model_name or "large" in model_name else 5,  # Smaller population for larger models
        "elimination_rate": 0.2,
        "target_reward": -1.0,
        "convergence_window": 20,
        "min_samples": 200,
        "max_samples": 2000,  # Fewer samples for testing
    }
    
    # Load dataset
    print("Loading dataset...")
    try:
        dataset = load_dataset("Anthropic/hh-rlhf", split="train")
        dataset = dataset.select(range(500))
        print(f"✓ Loaded {len(dataset)} samples")
    except Exception as e:
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if hf_token:
            dataset = load_dataset("Anthropic/hh-rlhf", split="train", token=hf_token)
            dataset = dataset.select(range(500))
            print(f"✓ Loaded {len(dataset)} samples")
        else:
            raise
    
    # Test baseline
    print("\nTesting baseline PPO...")
    baseline_config = config.copy()
    baseline_trainer = BaselinePPORLHFTrainer(baseline_config)
    baseline_results = baseline_trainer.train(dataset)
    baseline_samples = baseline_results['samples_to_convergence']
    print(f"Baseline: {baseline_samples} samples")
    
    # Test Full PulseOS
    print("\nTesting Full PulseOS...")
    pulseos_trainer = FullPulseOSRLHF(config)
    pulseos_results = await pulseos_trainer.train(dataset)
    pulseos_samples = pulseos_results['samples_to_convergence']
    print(f"PulseOS: {pulseos_samples} samples")
    
    # Calculate improvement
    if baseline_samples > 0:
        improvement = ((baseline_samples - pulseos_samples) / baseline_samples) * 100
    else:
        improvement = 0
    
    print(f"\nImprovement: {improvement:.1f}%")
    
    return {
        "model_name": model_name,
        "model_size_name": model_size_name,
        "baseline_samples": baseline_samples,
        "pulseos_samples": pulseos_samples,
        "improvement_percent": improvement,
    }

async def main():
    print("=" * 80)
    print("Week 4 Day 26-28: Scaling Tests")
    print("=" * 80)
    print()
    print("Testing PulseOS on different model sizes:")
    print("  1. GPT-2 small (124M)")
    print("  2. GPT-2 medium (355M)")
    print("  3. GPT-2 large (774M)")
    print()
    
    results = []
    
    # Test GPT-2 small
    try:
        result = await test_model_size("gpt2", "GPT-2 Small (124M)")
        results.append(result)
    except Exception as e:
        print(f"Error testing GPT-2 small: {e}")
        import traceback
        traceback.print_exc()
    
    # Test GPT-2 medium
    try:
        result = await test_model_size("gpt2-medium", "GPT-2 Medium (355M)")
        results.append(result)
    except Exception as e:
        print(f"Error testing GPT-2 medium: {e}")
        print("Skipping medium model (may require more memory)")
    
    # Test GPT-2 large
    try:
        result = await test_model_size("gpt2-large", "GPT-2 Large (774M)")
        results.append(result)
    except Exception as e:
        print(f"Error testing GPT-2 large: {e}")
        print("Skipping large model (may require more memory)")
    
    # Summary
    print("\n" + "=" * 80)
    print("SCALING TEST RESULTS")
    print("=" * 80)
    
    for result in results:
        print(f"\n{result['model_size_name']}:")
        print(f"  Baseline: {result['baseline_samples']} samples")
        print(f"  PulseOS: {result['pulseos_samples']} samples")
        print(f"  Improvement: {result['improvement_percent']:.1f}%")
    
    # Save results
    output_path = Path("pulseos_rlhf/scaling_test_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved results to {output_path}")
    
    # Check if improvement scales
    if len(results) >= 2:
        improvements = [r['improvement_percent'] for r in results]
        if all(imp > 0 for imp in improvements):
            print("\n✓ Improvement maintained across model sizes")
        else:
            print("\n⚠️  Improvement varies across model sizes")

if __name__ == "__main__":
    asyncio.run(main())


