"""
Week 1 Day 5-7: Baseline PPO with Trained Reward Model

This script implements baseline PPO RLHF training using TRL PPOTrainer
with the trained reward model from Day 3-4.

Key fixes from previous attempts:
- Use correct reward scaling (negative for HH-RLHF, ~-2.0 to 0.0)
- Proper convergence detection (not instant at min samples)
- Convergence window: 20 samples
- Min samples: 200 before checking
"""

import os
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from datasets import load_dataset
import matplotlib.pyplot as plt

# Disable tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class BaselinePPORLHFTrainer:
    """Baseline PPO RLHF trainer using TRL."""
    
    def __init__(self, config):
        self.config = config
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load reward model
        reward_model_path = config.get("reward_model_path", "pulseos_rlhf/reward_model")
        from transformers import AutoModelForSequenceClassification
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
        self.samples_history = []
        
        # Convergence settings (fixed from previous attempts)
        self.target_reward = config.get("target_reward", -1.0)  # Negative for HH-RLHF
        self.convergence_window = config.get("convergence_window", 20)
        self.min_samples = config.get("min_samples", 200)
        self.max_samples = config.get("max_samples", 10000)
    
    def get_reward(self, response_text):
        """Get reward from reward model."""
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
        """Execute one training step."""
        # Generate response
        response_text, query_tokens, response_tokens = self.generate_response(query)
        
        # Get reward
        reward = self.get_reward(response_text)
        
        # PPO update
        stats = self.ppo_trainer.step(
            [query_tokens],
            [response_tokens],
            [reward],
        )
        
        # Track
        self.samples_seen += 1
        self.rewards_history.append(reward)
        self.samples_history.append(self.samples_seen)
        
        return {
            "reward": reward,
            "samples": self.samples_seen,
            "stats": stats,
        }
    
    def check_convergence(self):
        """Check if model has converged (fixed from previous attempts)."""
        # Don't check until minimum samples
        if len(self.rewards_history) < self.min_samples:
            return False
        
        # Need enough samples for convergence window
        if len(self.rewards_history) < self.convergence_window:
            return False
        
        # Check recent average reward
        recent_rewards = self.rewards_history[-self.convergence_window:]
        avg_reward = np.mean(recent_rewards)
        
        # For HH-RLHF, rewards are negative, so we check if avg >= target
        return avg_reward >= self.target_reward
    
    def train(self, dataset, max_samples=None):
        """Train until convergence or max samples."""
        if max_samples is None:
            max_samples = self.max_samples
        
        print(f"Training baseline PPO...")
        print(f"  Target reward: {self.target_reward}")
        print(f"  Convergence window: {self.convergence_window}")
        print(f"  Min samples: {self.min_samples}")
        print(f"  Max samples: {max_samples}")
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
                print(f"  Sample {self.samples_seen}/{max_samples}, Recent avg reward: {recent_avg:.4f}")
            
            self.train_step(query)
        
        return {
            "samples_to_convergence": self.samples_seen,
            "converged": self.check_convergence(),
            "final_reward": np.mean(self.rewards_history[-20:]) if len(self.rewards_history) >= 20 else np.mean(self.rewards_history),
            "rewards_history": self.rewards_history.copy(),
            "samples_history": self.samples_history.copy(),
        }

def main():
    print("=" * 80)
    print("Week 1 Day 5-7: Baseline PPO with Trained Reward Model")
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
        # Convergence settings (fixed)
        "target_reward": -1.0,  # Negative for HH-RLHF (was positive 0.85)
        "convergence_window": 20,  # Not too short
        "min_samples": 200,  # Not instant
        "max_samples": 5000,  # For testing
    }
    
    print(f"Using device: {config['device']}")
    print()
    
    # Check if reward model exists
    reward_model_path = Path(config["reward_model_path"])
    if not reward_model_path.exists():
        print(f"❌ ERROR: Reward model not found at {reward_model_path}")
        print("   Please run train_reward_model.py first (Day 3-4)")
        return
    
    print("1. Loading dataset...")
    try:
        dataset = load_dataset("Anthropic/hh-rlhf", split="train")
        # Use subset for testing
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
    print("2. Creating baseline PPO trainer...")
    trainer = BaselinePPORLHFTrainer(config)
    print("   ✓ Trainer created")
    print()
    
    # Train
    print("3. Training baseline PPO...")
    results = trainer.train(dataset)
    print()
    
    # Results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Samples to convergence: {results['samples_to_convergence']}")
    print(f"Converged: {results['converged']}")
    print(f"Final reward: {results['final_reward']:.4f}")
    print()
    
    # Plot learning curve
    print("4. Plotting learning curve...")
    plt.figure(figsize=(10, 6))
    plt.plot(results['samples_history'], results['rewards_history'], alpha=0.6, label='Reward')
    if len(results['rewards_history']) >= 20:
        # Moving average
        window = 20
        moving_avg = [np.mean(results['rewards_history'][max(0, i-window):i+1]) 
                     for i in range(len(results['rewards_history']))]
        plt.plot(results['samples_history'], moving_avg, 'r-', linewidth=2, label='Moving Avg (20)')
    plt.axhline(y=config['target_reward'], color='g', linestyle='--', label='Target Reward')
    plt.xlabel('Samples')
    plt.ylabel('Reward')
    plt.title('Baseline PPO Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = Path("pulseos_rlhf/baseline_ppo_learning_curve.png")
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
    
    if results['samples_to_convergence'] >= 200:
        print(f"✓ Took at least {results['min_samples']} samples (not instant)")
    else:
        print(f"⚠️  Converged too quickly (< {results['min_samples']} samples)")
    
    if results['final_reward'] >= config['target_reward']:
        print(f"✓ Final reward ({results['final_reward']:.4f}) >= target ({config['target_reward']})")
    else:
        print(f"⚠️  Final reward ({results['final_reward']:.4f}) < target ({config['target_reward']})")
    
    print()
    print("Next steps:")
    print("  1. If converged properly: Proceed to Week 2 (death penalty)")
    print("  2. If not converged: Check reward model, increase max_samples")
    print("  3. Run 3 trials to verify reproducibility")
    print("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise

