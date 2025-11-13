"""
Week 3 Day 19-21: Full PulseOS Integration

Integrates all components:
- Death penalty
- Population elimination/reproduction
- Runtime alpha adaptation
- PulseOS Runtime class

This is the complete PulseOS system for RLHF training.
"""

import os
import torch
import numpy as np
import asyncio
from pathlib import Path
from copy import deepcopy
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from pulseos import Runtime, Config, Agent, SurvivalConstraint
import matplotlib.pyplot as plt

# Disable tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PulseOSRLHFAgent(Agent):
    """RLHF agent that implements PulseOS Agent interface."""
    
    def __init__(self, agent_id, config, model, ref_model, tokenizer, reward_model):
        super().__init__(agent_id)
        self.config = config
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # PPO trainer
        ppo_config = PPOConfig(
            learning_rate=self.learning_rate,  # Will be updated by Runtime
            batch_size=config.get("batch_size", 16),
            mini_batch_size=config.get("mini_batch_size", 4),
            num_ppo_epochs=config.get("ppo_epochs", 4),
            cliprange=config.get("ppo_clip_epsilon", 0.2),
        )
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
        )
        
        # Training state
        self.rewards_history = []
        self.samples_seen = 0
        
        # Death penalty settings
        self.death_threshold = config.get("death_threshold", -2.0)
        self.death_penalty = config.get("death_penalty", -10.0)
    
    async def step(self):
        """Execute one step (required by Agent interface)."""
        return {
            "samples_seen": self.samples_seen,
            "current_reward": self.rewards_history[-1] if self.rewards_history else 0.0,
        }
    
    def get_performance_metric(self) -> float:
        """Get performance metric for survival constraint."""
        if not self.rewards_history:
            return 0.0
        
        recent = np.mean(self.rewards_history[-10:]) if len(self.rewards_history) >= 10 else self.rewards_history[-1]
        
        # Normalize to [0, 1] for survival constraint
        # HH-RLHF rewards are negative, so shift and scale
        normalized = max(0.0, min(1.0, (recent + 3.0) / 3.0))
        return normalized
    
    def train_step(self, query):
        """Execute one training step."""
        # Generate response
        if not query or len(query.strip()) == 0:
            query = "Human: Hello\nAssistant:"
        
        query_tokens = self.tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.get("max_length", 512),
        ).to(self.device)
        
        response_tensors = self.ppo_trainer.generate(
            query_tokens["input_ids"],
            max_new_tokens=self.config.get("max_new_tokens", 64),
            return_prompt=False,
        )
        
        response_text = self.tokenizer.decode(response_tensors[0], skip_special_tokens=True)
        
        # Get reward
        reward_tokens = self.tokenizer(
            response_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.get("max_length", 512),
            padding=True,
        ).to(self.device)
        
        with torch.no_grad():
            base_reward = self.reward_model(**reward_tokens).logits.item()
        
        # Apply death penalty
        if len(self.rewards_history) >= 10:
            recent_avg = np.mean(self.rewards_history[-10:])
            if recent_avg < self.death_threshold:
                modified_reward = base_reward + self.death_penalty
            else:
                modified_reward = base_reward
        else:
            modified_reward = base_reward
        
        self.rewards_history.append(base_reward)
        
        # Update learning rate from Runtime
        self.ppo_trainer.optimizer.param_groups[0]['lr'] = self.learning_rate
        
        # PPO update
        stats = self.ppo_trainer.step(
            [query_tokens["input_ids"][0]],
            [response_tensors[0]],
            [modified_reward],
        )
        
        self.samples_seen += 1
        
        return {
            "base_reward": base_reward,
            "modified_reward": modified_reward,
            "samples": self.samples_seen,
        }

class FullPulseOSRLHF:
    """Full PulseOS RLHF system with all components."""
    
    def __init__(self, config):
        self.config = config
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # Load models
        model_name = config["model_name"]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            config.get("reward_model_path", "pulseos_rlhf/reward_model")
        )
        reward_model.to(self.device)
        reward_model.eval()
        
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
        model.to(self.device)
        ref_model = create_reference_model(model)
        ref_model.to(self.device)
        
        # Create population of agents
        self.population_size = config.get("population_size", 5)
        self.agents = []
        for i in range(self.population_size):
            agent = PulseOSRLHFAgent(
                agent_id=f"agent_{i}",
                config=config,
                model=model,  # Share model for now (would clone in production)
                ref_model=ref_model,
                tokenizer=tokenizer,
                reward_model=reward_model,
            )
            self.agents.append(agent)
        
        # Create PulseOS Runtime
        constraint = SurvivalConstraint(threshold=config.get("survival_threshold", 0.55))
        runtime_config = Config(
            alpha_base=config.get("learning_rate", 1.41e-5),
            alpha_max_change_per_step=config.get("alpha_max_change", 0.50),
            gamma=config.get("gamma", 0.5),
        )
        self.runtime = Runtime(constraint=constraint, config=runtime_config)
        
        # Register agents with runtime
        for agent in self.agents:
            self.runtime.register_agent(agent.agent_id, agent)
        
        # Training state
        self.generation = 0
        self.elimination_rate = config.get("elimination_rate", 0.2)
        
        # Convergence settings
        self.target_reward = config.get("target_reward", -1.0)
        self.convergence_window = config.get("convergence_window", 20)
        self.min_samples = config.get("min_samples", 200)
        self.max_samples = config.get("max_samples", 10000)
    
    async def train_step(self, query):
        """Execute one training step for all agents."""
        # Train all agents
        for agent in self.agents:
            agent.train_step(query)
        
        # Update PulseOS runtime
        await self.runtime.step()
        
        # Update agents with adaptive parameters from runtime
        for agent in self.agents:
            agent.learning_rate = self.runtime.apc.get_alpha()
            agent.exploration_rate = self.runtime.apc.get_epsilon()
    
    def eliminate_and_reproduce(self):
        """Eliminate worst performers and clone best ones."""
        # Get scores
        scores = [agent.get_performance_metric() for agent in self.agents]
        
        # Sort by score
        agent_score_pairs = list(zip(self.agents, scores))
        agent_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Eliminate worst
        num_eliminate = int(self.population_size * self.elimination_rate)
        num_keep = self.population_size - num_eliminate
        
        survivors = [agent for agent, score in agent_score_pairs[:num_keep]]
        
        # Clone best to replace eliminated
        new_agents = survivors.copy()
        while len(new_agents) < self.population_size:
            best_agent = survivors[0]
            # In production, would clone model weights
            # For now, create new agent (simplified)
            new_agent = PulseOSRLHFAgent(
                agent_id=f"agent_{len(new_agents)}",
                config=self.config,
                model=best_agent.model,
                ref_model=best_agent.ref_model,
                tokenizer=best_agent.tokenizer,
                reward_model=best_agent.reward_model,
            )
            new_agents.append(new_agent)
        
        self.agents = new_agents
        
        # Re-register with runtime
        self.runtime.agents = {}
        for agent in self.agents:
            self.runtime.register_agent(agent.agent_id, agent)
    
    def check_convergence(self):
        """Check if best agent has converged."""
        if len(self.agents) == 0:
            return False
        
        best_agent = max(self.agents, key=lambda a: np.mean(a.rewards_history[-10:]) if len(a.rewards_history) >= 10 else -10.0)
        
        if len(best_agent.rewards_history) < self.min_samples:
            return False
        
        if len(best_agent.rewards_history) < self.convergence_window:
            return False
        
        recent_rewards = best_agent.rewards_history[-self.convergence_window:]
        avg_reward = np.mean(recent_rewards)
        
        return avg_reward >= self.target_reward
    
    async def train(self, dataset, max_samples=None):
        """Train full PulseOS system."""
        if max_samples is None:
            max_samples = self.max_samples
        
        print(f"Training Full PulseOS RLHF system...")
        print(f"  Population size: {self.population_size}")
        print(f"  Elimination rate: {self.elimination_rate * 100:.0f}%")
        print(f"  Target reward: {self.target_reward}")
        print()
        
        total_samples = 0
        generation = 0
        
        for i, item in enumerate(dataset):
            if total_samples >= max_samples:
                break
            
            if self.check_convergence():
                print(f"  ✓ Converged at sample {total_samples}")
                break
            
            query = item.get("query", "")
            
            # Train step
            await self.train_step(query)
            total_samples += self.population_size
            
            # Eliminate and reproduce every N steps
            if total_samples % 200 == 0 and total_samples > 0:
                generation += 1
                self.eliminate_and_reproduce()
                best_score = max([a.get_performance_metric() for a in self.agents])
                print(f"  Generation {generation}, Sample {total_samples}, Best score: {best_score:.4f}")
        
        # Get best agent
        best_agent = max(self.agents, key=lambda a: np.mean(a.rewards_history[-10:]) if len(a.rewards_history) >= 10 else -10.0)
        
        return {
            "samples_to_convergence": total_samples,
            "converged": self.check_convergence(),
            "final_reward": np.mean(best_agent.rewards_history[-20:]) if len(best_agent.rewards_history) >= 20 else np.mean(best_agent.rewards_history),
            "best_agent": best_agent,
            "generations": generation,
        }

async def main():
    print("=" * 80)
    print("Week 3 Day 19-21: Full PulseOS Integration")
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
        "survival_threshold": 0.55,
        "alpha_max_change": 0.50,
        "gamma": 0.5,
        "population_size": 5,
        "elimination_rate": 0.2,
        "target_reward": -1.0,
        "convergence_window": 20,
        "min_samples": 200,
        "max_samples": 5000,
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
    print("2. Creating Full PulseOS trainer...")
    trainer = FullPulseOSRLHF(config)
    print("   ✓ Trainer created")
    print()
    
    # Train
    print("3. Training Full PulseOS system...")
    results = await trainer.train(dataset)
    print()
    
    # Results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Samples to convergence: {results['samples_to_convergence']}")
    print(f"Converged: {results['converged']}")
    print(f"Final reward: {results['final_reward']:.4f}")
    print(f"Generations: {results['generations']}")
    print()
    
    print("=" * 80)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 80)
    
    if results['converged']:
        print(f"✓ Converged in {results['samples_to_convergence']} samples")
    else:
        print(f"⚠️  Did not converge")
    
    print()
    print("Next steps:")
    print("  1. Compare with baseline and other methods")
    print("  2. Run evaluation harness for statistical validation")
    print("=" * 80)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise

