"""
Week 3 Day 15-18: Runtime Alpha Adaptation

Implements variance-based alpha adaptation that adjusts learning pressure
based on performance variance. High variance = increase pressure, low variance = decrease pressure.
"""

import os
import torch
import numpy as np
from pathlib import Path
from copy import deepcopy
from datasets import load_dataset
from death_penalty import PPOWithDeathPenalty
import matplotlib.pyplot as plt

class RuntimeAlphaAdaptation:
    """Runtime alpha adaptation based on variance."""
    
    def __init__(self, config):
        self.config = config
        
        # Runtime adaptation settings
        self.alpha_base = config.get("alpha_base", 0.01)
        self.alpha_max_change = config.get("alpha_max_change", 0.50)  # 50% max change
        self.gamma = config.get("gamma", 0.5)  # Adaptation strength
        
        self.alpha = self.alpha_base
        self.reward_history = []
        self.alpha_history = []
        
        # Create base trainer (with death penalty)
        trainer_config = deepcopy(config)
        trainer_config["learning_rate"] = self.alpha_base  # Will be updated
        self.trainer = PPOWithDeathPenalty(trainer_config)
        
        # Convergence settings
        self.target_reward = config.get("target_reward", -1.0)
        self.convergence_window = config.get("convergence_window", 20)
        self.min_samples = config.get("min_samples", 200)
        self.max_samples = config.get("max_samples", 10000)
    
    def calculate_variance_signal(self, window=10):
        """Calculate variance-based adaptation signal."""
        if len(self.reward_history) < window:
            return 0.5  # Default signal
        
        recent_rewards = np.array(self.reward_history[-window:])
        variance = np.var(recent_rewards)
        mean = np.abs(np.mean(recent_rewards))
        
        if mean > 1e-6:
            cv = np.sqrt(variance) / mean  # Coefficient of variation
        else:
            cv = 1.0
        
        # Map to [0, 1] range
        variance_signal = min(1.0, cv * 3.0)
        return variance_signal
    
    def update_alpha(self):
        """Update alpha based on variance signal."""
        variance_signal = self.calculate_variance_signal()
        
        # High variance = increase pressure (increase alpha)
        # Low variance = decrease pressure (decrease alpha)
        if variance_signal > 0.5:  # High variance
            alpha_change = self.gamma * variance_signal * self.alpha_max_change
            self.alpha = min(self.alpha + alpha_change, 
                           self.alpha_base * (1 + self.alpha_max_change))
        else:  # Low variance
            alpha_change = self.gamma * (1 - variance_signal) * self.alpha_max_change
            self.alpha = max(self.alpha - alpha_change,
                           self.alpha_base * (1 - self.alpha_max_change))
        
        # Update trainer learning rate
        self.trainer.ppo_trainer.optimizer.param_groups[0]['lr'] = self.alpha
        
        return self.alpha, variance_signal
    
    def train_step(self, query):
        """Execute one training step with runtime adaptation."""
        # Train with current alpha
        result = self.trainer.train_step(query)
        
        # Track reward
        self.reward_history.append(result["base_reward"])
        
        # Update alpha every N steps
        if len(self.reward_history) % 10 == 0:
            alpha, variance_signal = self.update_alpha()
            self.alpha_history.append({
                "step": self.trainer.samples_seen,
                "alpha": alpha,
                "variance_signal": variance_signal,
            })
        
        return result
    
    def check_convergence(self):
        """Check if model has converged."""
        return self.trainer.check_convergence()
    
    def train(self, dataset, max_samples=None):
        """Train with runtime adaptation."""
        if max_samples is None:
            max_samples = self.max_samples
        
        print(f"Training with runtime alpha adaptation...")
        print(f"  Base alpha: {self.alpha_base}")
        print(f"  Max change: {self.alpha_max_change * 100:.0f}%")
        print(f"  Gamma: {self.gamma}")
        print()
        
        for i, item in enumerate(dataset):
            if self.trainer.samples_seen >= max_samples:
                print(f"  Reached max samples: {max_samples}")
                break
            
            if self.check_convergence():
                print(f"  ✓ Converged at sample {self.trainer.samples_seen}")
                break
            
            query = item.get("query", "")
            
            # Print progress
            if self.trainer.samples_seen % 50 == 0 and self.trainer.samples_seen > 0:
                recent_avg = np.mean(self.reward_history[-20:]) if len(self.reward_history) >= 20 else np.mean(self.reward_history)
                current_alpha = self.alpha_history[-1]["alpha"] if self.alpha_history else self.alpha_base
                print(f"  Sample {self.trainer.samples_seen}/{max_samples}, Recent avg reward: {recent_avg:.4f}, Alpha: {current_alpha:.6f}")
            
            self.train_step(query)
        
        return {
            "samples_to_convergence": self.trainer.samples_seen,
            "converged": self.check_convergence(),
            "final_reward": np.mean(self.reward_history[-20:]) if len(self.reward_history) >= 20 else np.mean(self.reward_history),
            "rewards_history": self.reward_history.copy(),
            "alpha_history": self.alpha_history.copy(),
            "samples_history": self.trainer.samples_history.copy(),
        }

def main():
    print("=" * 80)
    print("Week 3 Day 15-18: Runtime Alpha Adaptation")
    print("=" * 80)
    print()
    
    # Configuration
    config = {
        "model_name": "gpt2",
        "reward_model_path": "pulseos_rlhf/reward_model",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "learning_rate": 1.41e-5,
        "batch_size": 16,
        "mini_batch_size": 4,
        "ppo_epochs": 4,
        "ppo_clip_epsilon": 0.2,
        "max_length": 512,
        "max_new_tokens": 64,
        "death_threshold": -2.0,
        "death_penalty": -10.0,
        "target_reward": -1.0,
        "convergence_window": 20,
        "min_samples": 200,
        "max_samples": 5000,
        # Runtime adaptation settings
        "alpha_base": 0.01,
        "alpha_max_change": 0.50,
        "gamma": 0.5,
    }
    
    print(f"Using device: {config['device']}")
    print()
    
    # Load dataset
    print("1. Loading dataset...")
    try:
        dataset = load_dataset("Anthropic/hh-rlhf", split="train")
        dataset = dataset.select(range(1000))
        print(f"   ✓ Loaded {len(dataset)} samples")
    except Exception as e:
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if hf_token:
            dataset = load_dataset("Anthropic/hh-rlhf", split="train", token=hf_token)
            dataset = dataset.select(range(1000))
            print(f"   ✓ Loaded {len(dataset)} samples")
        else:
            raise
    print()
    
    # Create trainer
    print("2. Creating runtime adaptation trainer...")
    trainer = RuntimeAlphaAdaptation(config)
    print("   ✓ Trainer created")
    print()
    
    # Train
    print("3. Training with runtime adaptation...")
    results = trainer.train(dataset)
    print()
    
    # Results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Samples to convergence: {results['samples_to_convergence']}")
    print(f"Converged: {results['converged']}")
    print(f"Final reward: {results['final_reward']:.4f}")
    print(f"Alpha changes: {len(results['alpha_history'])}")
    if results['alpha_history']:
        initial_alpha = results['alpha_history'][0]['alpha']
        final_alpha = results['alpha_history'][-1]['alpha']
        print(f"Alpha range: {initial_alpha:.6f} → {final_alpha:.6f}")
    print()
    
    # Plot learning curve and alpha adaptation
    print("4. Plotting results...")
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(results['samples_history'], results['rewards_history'], alpha=0.6, label='Reward', color='blue')
    if len(results['rewards_history']) >= 20:
        window = 20
        moving_avg = [np.mean(results['rewards_history'][max(0, i-window):i+1]) 
                     for i in range(len(results['rewards_history']))]
        plt.plot(results['samples_history'], moving_avg, 'b-', linewidth=2, label='Moving Avg (20)')
    plt.axhline(y=config['target_reward'], color='g', linestyle='--', label='Target Reward')
    plt.xlabel('Samples')
    plt.ylabel('Reward')
    plt.title('Runtime Adaptation Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    if results['alpha_history']:
        alpha_steps = [h['step'] for h in results['alpha_history']]
        alpha_values = [h['alpha'] for h in results['alpha_history']]
        variance_signals = [h['variance_signal'] for h in results['alpha_history']]
        
        ax1 = plt.gca()
        ax1.plot(alpha_steps, alpha_values, 'o-', label='Alpha', color='red', linewidth=2)
        ax1.set_xlabel('Samples')
        ax1.set_ylabel('Alpha (Learning Rate)', color='red')
        ax1.tick_params(axis='y', labelcolor='red')
        ax1.grid(True, alpha=0.3)
        
        ax2 = ax1.twinx()
        ax2.plot(alpha_steps, variance_signals, 's-', label='Variance Signal', color='orange', alpha=0.7)
        ax2.set_ylabel('Variance Signal', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        
        plt.title('Alpha Adaptation Over Time')
    
    plt.tight_layout()
    output_path = Path("pulseos_rlhf/runtime_adaptation_results.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"   ✓ Saved to {output_path}")
    print()
    
    print("=" * 80)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 80)
    
    if results['converged']:
        print(f"✓ Converged in {results['samples_to_convergence']} samples")
    else:
        print(f"⚠️  Did not converge")
    
    if results['alpha_history'] and len(results['alpha_history']) > 1:
        alpha_changed = abs(results['alpha_history'][-1]['alpha'] - results['alpha_history'][0]['alpha']) > 1e-6
        if alpha_changed:
            print(f"✓ Alpha adapted over time (not stuck)")
        else:
            print(f"⚠️  Alpha did not change")
    
    print()
    print("Next steps:")
    print("  1. Compare with baseline and death penalty")
    print("  2. If shows improvement: Proceed to full PulseOS integration")
    print("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise

