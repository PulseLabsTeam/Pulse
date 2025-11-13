"""
Week 2 Day 8-10: Death Penalty Mechanism

This implements death penalty for PPO training - the simplest PulseOS feature.
When agent performance drops below threshold, apply catastrophic penalty.

Death penalty is applied as reward modification, not restart.
"""

import os
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
import matplotlib.pyplot as plt

# Disable tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PPOWithDeathPenalty:
    """PPO trainer with death penalty mechanism."""
    
    def __init__(self, config):
        self.config = config
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # Death penalty settings
        self.death_threshold = config.get("death_threshold", -2.0)  # Below baseline
        self.death_penalty = config.get("death_penalty", -10.0)  # Catastrophic penalty
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load reward model
        reward_model_path = config.get("reward_model_path", "pulseos_rlhf/reward_model")
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_path)
        self.reward_model.to(self.device)
        self.reward_model.eval()
        
        # Load policy model with value head
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(config["model_name"])
        self.model.to(self.device)
        
        # Create reference model
        self.ref_model = create_reference_model(self.model)
        self.ref_model.to(self.device)
        
        # PPO config
        self.ppo_config = PPOConfig(
            learning_rate=config.get("learning_rate", 1.41e-5),
            batch_size=config.get("batch_size", 16),
            mini_batch_size=config.get("mini_batch_size", 4),
            num_ppo_epochs=config.get("ppo_epochs", 4),
            cliprange=config.get("ppo_clip_epsilon", 0.2),
        )
        
        # Create PPO trainer
        self.ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
        )
        
        # Training state
        self.samples_seen = 0
        self.rewards_history = []
        self.modified_rewards_history = []
        self.samples_history = []
        self.death_penalty_applied = []
        
        # Convergence settings
        self.target_reward = config.get("target_reward", -1.0)
        self.convergence_window = config.get("convergence_window", 20)
        self.min_samples = config.get("min_samples", 200)
        self.max_samples = config.get("max_samples", 10000)
    
    def get_reward(self, response_text):
        """Get base reward from reward model."""
        tokens = self.tokenizer(
            response_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.get("max_length", 512),
            padding=True,
        ).to(self.device)
        
        with torch.no_grad():
            reward = self.reward_model(**tokens).logits.item()
        
        return reward
    
    def apply_death_penalty(self, base_reward):
        """Apply death penalty if performance is below threshold."""
        # Calculate recent average reward
        if len(self.rewards_history) >= 10:
            recent_avg = np.mean(self.rewards_history[-10:])
        else:
            recent_avg = base_reward
        
        # Check if dying (below threshold)
        if recent_avg < self.death_threshold:
            penalty = self.death_penalty
            modified_reward = base_reward + penalty
            self.death_penalty_applied.append(True)
            return modified_reward, True
        else:
            self.death_penalty_applied.append(False)
            return base_reward, False
    
    def generate_response(self, query):
        """Generate response using current policy."""
        if not query or len(query.strip()) == 0:
            query = "Human: Hello\nAssistant:"
        
        query_tokens = self.tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.get("max_length", 512),
        ).to(self.device)
        
        # Generate
        response_tensors = self.ppo_trainer.generate(
            query_tokens["input_ids"],
            max_new_tokens=self.config.get("max_new_tokens", 64),
            return_prompt=False,
        )
        
        response_text = self.tokenizer.decode(response_tensors[0], skip_special_tokens=True)
        return response_text, query_tokens["input_ids"][0], response_tensors[0]
    
    def train_step(self, query):
        """Execute one training step with death penalty."""
        # Generate response
        response_text, query_tokens, response_tokens = self.generate_response(query)
        
        # Get base reward
        base_reward = self.get_reward(response_text)
        self.rewards_history.append(base_reward)
        
        # Apply death penalty
        modified_reward, penalty_applied = self.apply_death_penalty(base_reward)
        self.modified_rewards_history.append(modified_reward)
        
        # PPO update with modified reward
        stats = self.ppo_trainer.step(
            [query_tokens],
            [response_tokens],
            [modified_reward],
        )
        
        # Track
        self.samples_seen += 1
        self.samples_history.append(self.samples_seen)
        
        return {
            "base_reward": base_reward,
            "modified_reward": modified_reward,
            "penalty_applied": penalty_applied,
            "samples": self.samples_seen,
            "stats": stats,
        }
    
    def check_convergence(self):
        """Check if model has converged."""
        if len(self.rewards_history) < self.min_samples:
            return False
        
        if len(self.rewards_history) < self.convergence_window:
            return False
        
        recent_rewards = self.rewards_history[-self.convergence_window:]
        avg_reward = np.mean(recent_rewards)
        
        return avg_reward >= self.target_reward
    
    def train(self, dataset, max_samples=None):
        """Train until convergence or max samples."""
        if max_samples is None:
            max_samples = self.max_samples
        
        print(f"Training PPO with death penalty...")
        print(f"  Death threshold: {self.death_threshold}")
        print(f"  Death penalty: {self.death_penalty}")
        print(f"  Target reward: {self.target_reward}")
        print()
        
        for i, item in enumerate(dataset):
            if self.samples_seen >= max_samples:
                print(f"  Reached max samples: {max_samples}")
                break
            
            if self.check_convergence():
                print(f"  ✓ Converged at sample {self.samples_seen}")
                break
            
            query = item.get("query", "")
            
            # Print progress
            if self.samples_seen % 50 == 0 and self.samples_seen > 0:
                recent_avg = np.mean(self.rewards_history[-20:]) if len(self.rewards_history) >= 20 else np.mean(self.rewards_history)
                penalty_count = sum(self.death_penalty_applied[-20:]) if len(self.death_penalty_applied) >= 20 else sum(self.death_penalty_applied)
                print(f"  Sample {self.samples_seen}/{max_samples}, Recent avg reward: {recent_avg:.4f}, Penalties: {penalty_count}/20")
            
            self.train_step(query)
        
        return {
            "samples_to_convergence": self.samples_seen,
            "converged": self.check_convergence(),
            "final_reward": np.mean(self.rewards_history[-20:]) if len(self.rewards_history) >= 20 else np.mean(self.rewards_history),
            "rewards_history": self.rewards_history.copy(),
            "modified_rewards_history": self.modified_rewards_history.copy(),
            "samples_history": self.samples_history.copy(),
            "death_penalty_count": sum(self.death_penalty_applied),
        }

def main():
    print("=" * 80)
    print("Week 2 Day 8-10: Death Penalty Mechanism")
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
        # Death penalty settings
        "death_threshold": -2.0,
        "death_penalty": -10.0,
        # Convergence settings
        "target_reward": -1.0,
        "convergence_window": 20,
        "min_samples": 200,
        "max_samples": 5000,
    }
    
    print(f"Using device: {config['device']}")
    print()
    
    # Check if reward model exists
    reward_model_path = Path(config["reward_model_path"])
    if not reward_model_path.exists():
        print(f"❌ ERROR: Reward model not found at {reward_model_path}")
        print("   Please run train_reward_model.py first")
        return
    
    # Load dataset
    print("1. Loading dataset...")
    try:
        dataset = load_dataset("Anthropic/hh-rlhf", split="train")
        dataset = dataset.select(range(1000))
        print(f"   ✓ Loaded {len(dataset)} samples")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if hf_token:
            dataset = load_dataset("Anthropic/hh-rlhf", split="train", token=hf_token)
            dataset = dataset.select(range(1000))
            print(f"   ✓ Loaded {len(dataset)} samples")
        else:
            raise
    print()
    
    # Create trainer
    print("2. Creating PPO trainer with death penalty...")
    trainer = PPOWithDeathPenalty(config)
    print("   ✓ Trainer created")
    print()
    
    # Train
    print("3. Training PPO with death penalty...")
    results = trainer.train(dataset)
    print()
    
    # Results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Samples to convergence: {results['samples_to_convergence']}")
    print(f"Converged: {results['converged']}")
    print(f"Final reward: {results['final_reward']:.4f}")
    print(f"Death penalties applied: {results['death_penalty_count']}/{len(trainer.death_penalty_applied)}")
    print()
    
    # Plot learning curve
    print("4. Plotting learning curve...")
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(results['samples_history'], results['rewards_history'], alpha=0.6, label='Base Reward', color='blue')
    plt.plot(results['samples_history'], results['modified_rewards_history'], alpha=0.6, label='Modified Reward', color='red')
    if len(results['rewards_history']) >= 20:
        window = 20
        moving_avg = [np.mean(results['rewards_history'][max(0, i-window):i+1]) 
                     for i in range(len(results['rewards_history']))]
        plt.plot(results['samples_history'], moving_avg, 'b-', linewidth=2, label='Moving Avg (20)')
    plt.axhline(y=config['target_reward'], color='g', linestyle='--', label='Target Reward')
    plt.axhline(y=config['death_threshold'], color='r', linestyle='--', label='Death Threshold')
    plt.xlabel('Samples')
    plt.ylabel('Reward')
    plt.title('Death Penalty Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    penalty_applied_array = np.array(trainer.death_penalty_applied).astype(float)
    plt.plot(results['samples_history'], penalty_applied_array, alpha=0.6, color='red')
    plt.xlabel('Samples')
    plt.ylabel('Death Penalty Applied')
    plt.title('Death Penalty Application Over Time')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path("pulseos_rlhf/death_penalty_learning_curve.png")
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
        print(f"⚠️  Did not converge (reached {results['samples_to_convergence']} samples)")
    
    if results['death_penalty_count'] > 0:
        print(f"✓ Death penalty was applied {results['death_penalty_count']} times")
    else:
        print(f"⚠️  Death penalty was never applied (performance never dropped below threshold)")
    
    print()
    print("Next steps:")
    print("  1. Compare with baseline PPO (run comparison_harness.py)")
    print("  2. If shows improvement: Proceed to population training")
    print("  3. If hurts performance: Try different threshold/penalty values")
    print("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise

