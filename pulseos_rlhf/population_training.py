"""
Week 2 Day 11-14: Population Training with Elimination

Implements population-based training where multiple agents compete,
worst performers are eliminated, and best performers are cloned.
"""

import os
import torch
import numpy as np
from pathlib import Path
from copy import deepcopy
from datasets import load_dataset
from baseline_ppo import BaselinePPORLHFTrainer
import json
import matplotlib.pyplot as plt

class PopulationTrainer:
    """Population-based PPO trainer with elimination/reproduction."""
    
    def __init__(self, config):
        self.config = config
        self.population_size = config.get("population_size", 5)
        self.elimination_rate = config.get("elimination_rate", 0.2)  # Eliminate 20%
        
        # Create population of agents
        self.agents = []
        for i in range(self.population_size):
            agent_config = deepcopy(config)
            agent_config["agent_id"] = f"agent_{i}"
            agent = BaselinePPORLHFTrainer(agent_config)
            self.agents.append(agent)
        
        # Training state
        self.generation = 0
        self.agent_scores = []
        self.generation_history = []
        
        # Convergence settings
        self.target_reward = config.get("target_reward", -1.0)
        self.convergence_window = config.get("convergence_window", 20)
        self.min_samples = config.get("min_samples", 200)
        self.max_samples = config.get("max_samples", 10000)
    
    def evaluate_agents(self, dataset, steps_per_agent=50):
        """Evaluate all agents and return scores."""
        scores = []
        
        for agent in self.agents:
            # Train agent for some steps
            for i, item in enumerate(dataset):
                if i >= steps_per_agent:
                    break
                query = item.get("query", "")
                agent.train_step(query)
            
            # Get recent average reward
            if len(agent.rewards_history) >= 10:
                score = np.mean(agent.rewards_history[-10:])
            else:
                score = np.mean(agent.rewards_history) if agent.rewards_history else -10.0
            
            scores.append(score)
        
        return scores
    
    def eliminate_and_reproduce(self, scores):
        """Eliminate worst performers and clone best ones."""
        # Sort agents by score
        agent_score_pairs = list(zip(self.agents, scores))
        agent_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate how many to eliminate
        num_eliminate = int(self.population_size * self.elimination_rate)
        num_keep = self.population_size - num_eliminate
        
        # Keep best performers
        survivors = [agent for agent, score in agent_score_pairs[:num_keep]]
        
        # Clone best agents to replace eliminated ones
        new_agents = survivors.copy()
        while len(new_agents) < self.population_size:
            # Clone from best agent
            best_agent = survivors[0]
            # Create new agent with same config
            new_config = deepcopy(self.config)
            new_config["agent_id"] = f"agent_{len(new_agents)}"
            new_agent = BaselinePPORLHFTrainer(new_config)
            
            # Copy model weights from best agent
            new_agent.model.load_state_dict(best_agent.model.state_dict())
            
            new_agents.append(new_agent)
        
        self.agents = new_agents
        return num_eliminate
    
    def check_convergence(self):
        """Check if best agent has converged."""
        if len(self.agents) == 0:
            return False
        
        # Check best agent
        best_agent = max(self.agents, key=lambda a: np.mean(a.rewards_history[-10:]) if len(a.rewards_history) >= 10 else -10.0)
        
        if len(best_agent.rewards_history) < self.min_samples:
            return False
        
        if len(best_agent.rewards_history) < self.convergence_window:
            return False
        
        recent_rewards = best_agent.rewards_history[-self.convergence_window:]
        avg_reward = np.mean(recent_rewards)
        
        return avg_reward >= self.target_reward
    
    def train(self, dataset, max_generations=20, steps_per_generation=100):
        """Train population for multiple generations."""
        print(f"Training population of {self.population_size} agents...")
        print(f"  Elimination rate: {self.elimination_rate * 100:.0f}%")
        print(f"  Steps per generation: {steps_per_generation}")
        print(f"  Max generations: {max_generations}")
        print()
        
        for generation in range(max_generations):
            self.generation = generation
            
            # Evaluate all agents
            scores = self.evaluate_agents(dataset, steps_per_generation)
            self.agent_scores.append(scores.copy())
            
            # Get best score
            best_score = max(scores)
            avg_score = np.mean(scores)
            
            print(f"Generation {generation + 1}/{max_generations}:")
            print(f"  Best score: {best_score:.4f}")
            print(f"  Avg score: {avg_score:.4f}")
            print(f"  Scores: {[f'{s:.4f}' for s in scores]}")
            
            # Check convergence
            if self.check_convergence():
                print(f"  ✓ Converged!")
                break
            
            # Eliminate and reproduce
            num_eliminated = self.eliminate_and_reproduce(scores)
            print(f"  Eliminated {num_eliminated} agents, reproduced from best")
            print()
            
            self.generation_history.append({
                "generation": generation,
                "best_score": best_score,
                "avg_score": avg_score,
                "scores": scores.copy(),
            })
        
        # Get best agent
        best_agent = max(self.agents, key=lambda a: np.mean(a.rewards_history[-10:]) if len(a.rewards_history) >= 10 else -10.0)
        
        total_samples = sum(len(agent.rewards_history) for agent in self.agents)
        
        return {
            "generations": self.generation + 1,
            "total_samples": total_samples,
            "converged": self.check_convergence(),
            "best_agent_final_reward": np.mean(best_agent.rewards_history[-20:]) if len(best_agent.rewards_history) >= 20 else np.mean(best_agent.rewards_history),
            "generation_history": self.generation_history,
            "best_agent": best_agent,
        }

def main():
    print("=" * 80)
    print("Week 2 Day 11-14: Population Training")
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
        "target_reward": -1.0,
        "convergence_window": 20,
        "min_samples": 200,
        "max_samples": 5000,
        # Population settings
        "population_size": 5,
        "elimination_rate": 0.2,
    }
    
    print(f"Using device: {config['device']}")
    print()
    
    # Load dataset
    print("1. Loading dataset...")
    try:
        dataset = load_dataset("Anthropic/hh-rlhf", split="train")
        dataset = dataset.select(range(2000))
        print(f"   ✓ Loaded {len(dataset)} samples")
    except Exception as e:
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if hf_token:
            dataset = load_dataset("Anthropic/hh-rlhf", split="train", token=hf_token)
            dataset = dataset.select(range(2000))
            print(f"   ✓ Loaded {len(dataset)} samples")
        else:
            raise
    print()
    
    # Create trainer
    print("2. Creating population trainer...")
    trainer = PopulationTrainer(config)
    print("   ✓ Trainer created")
    print()
    
    # Train
    print("3. Training population...")
    results = trainer.train(dataset, max_generations=10, steps_per_generation=100)
    print()
    
    # Results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Generations: {results['generations']}")
    print(f"Total samples: {results['total_samples']}")
    print(f"Converged: {results['converged']}")
    print(f"Best agent final reward: {results['best_agent_final_reward']:.4f}")
    print()
    
    # Plot evolution
    print("4. Plotting population evolution...")
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    generations = [h["generation"] for h in results["generation_history"]]
    best_scores = [h["best_score"] for h in results["generation_history"]]
    avg_scores = [h["avg_score"] for h in results["generation_history"]]
    
    plt.plot(generations, best_scores, 'o-', label='Best Score', linewidth=2)
    plt.plot(generations, avg_scores, 's-', label='Avg Score', linewidth=2)
    plt.axhline(y=config['target_reward'], color='g', linestyle='--', label='Target Reward')
    plt.xlabel('Generation')
    plt.ylabel('Score')
    plt.title('Population Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Plot score distribution over generations
    for gen_idx, gen_data in enumerate(results["generation_history"]):
        scores = gen_data["scores"]
        plt.scatter([gen_idx] * len(scores), scores, alpha=0.5, s=50)
    plt.xlabel('Generation')
    plt.ylabel('Agent Score')
    plt.title('Score Distribution Across Generations')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path("pulseos_rlhf/population_evolution.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"   ✓ Saved to {output_path}")
    print()
    
    print("=" * 80)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 80)
    
    if results['converged']:
        print(f"✓ Converged in {results['generations']} generations")
    else:
        print(f"⚠️  Did not converge (reached {results['generations']} generations)")
    
    if results['best_agent_final_reward'] >= config['target_reward']:
        print(f"✓ Best agent reached target reward")
    else:
        print(f"⚠️  Best agent did not reach target reward")
    
    print()
    print("Next steps:")
    print("  1. Compare with baseline PPO")
    print("  2. If shows improvement: Proceed to runtime adaptation")
    print("  3. If marginal: Try different population size/elimination rate")
    print("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


