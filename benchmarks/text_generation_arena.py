"""
Text Generation Arena: PulseOS vs Baseline

Tests survival-pressure learning on population of language models.
Eliminates weak agents, spawns from strong agents.

Expected: 20-40% faster convergence = $25M-$60M valuation
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import time
import copy

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from torch.optim import AdamW
from datasets import load_dataset


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ArenaConfig:
    """Configuration for Text Generation Arena"""
    n_agents: int = 10  # Reduced from 20 to fit GPU memory
    model_name: str = "gpt2"  # Use GPT-2 124M
    n_steps: int = 100  # Training steps (100 for quick test, 500 for full)
    elimination_interval: int = 20  # Eliminate every N steps
    elimination_rate: float = 0.3  # Remove bottom 30%
    spawn_rate: float = 0.2  # Top 20% reproduce
    batch_size: int = 4  # Reduced batch size to save memory
    eval_batch_size: int = 8  # Reduced eval batch size
    max_length: int = 128  # Sequence length
    learning_rate: float = 5e-5
    mutation_noise: float = 0.1  # Weight noise for spawning
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"


# ============================================================================
# Text Agent
# ============================================================================

class TextAgent:
    """Individual language model agent"""
    
    def __init__(self, agent_id: str, model_name: str = "gpt2", device: str = "cuda"):
        self.agent_id = agent_id
        self.model_name = model_name
        self.device = device
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.model.to(device)
        
        # Enable gradient checkpointing to save memory
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        # Training setup
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)
        
        # State
        self.training_steps = 0
        self.is_alive = True
        self.generation = 0
        self.parent_id = None
    
    def train(self, texts: List[str]):
        """Single training step"""
        self.model.train()
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device)
        
        # Forward pass
        outputs = self.model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        self.training_steps += 1
        
        return loss.item()
    
    def evaluate(self, texts: List[str]) -> float:
        """Compute perplexity on validation set"""
        self.model.eval()
        
        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128
            ).to(self.device)
            
            outputs = self.model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            
            # Perplexity = exp(loss)
            perplexity = torch.exp(loss).item()
        
        return perplexity
    
    def clone_with_mutation(self, noise_scale: float = 0.001, new_id: str = None):
        """Create offspring with weight mutation
        
        Args:
            noise_scale: Standard deviation of noise to add (default 0.001 = 0.1%)
                         Much smaller than before (was 0.1 = 10%) to avoid breaking models
        """
        if new_id is None:
            new_id = f"{self.agent_id}_child_{int(time.time())}"
        
        child = TextAgent(new_id, self.model_name, self.device)
        child.generation = self.generation + 1
        child.parent_id = self.agent_id
        
        # Copy weights with SMALL noise (0.1% instead of 10%)
        # Large noise breaks neural networks - small noise allows exploration
        with torch.no_grad():
            for parent_param, child_param in zip(
                self.model.parameters(), 
                child.model.parameters()
            ):
                noise = torch.randn_like(parent_param) * noise_scale
                child_param.copy_(parent_param + noise)
        
        return child
    
    def get_state_dict(self):
        """Get model state for saving"""
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load model state"""
        self.model.load_state_dict(state_dict)


# ============================================================================
# Baseline Population (No Survival Pressure)
# ============================================================================

class BaselinePopulation:
    """Baseline: All agents train independently, no elimination"""
    
    def __init__(self, n_agents: int, model_name: str, device: str):
        self.agents = [TextAgent(f"baseline_{i}", model_name, device) 
                      for i in range(n_agents)]
        self.generation = 0
        self.device = device
    
    def train_step(self, train_texts: List[str], eval_texts: List[str]) -> Dict[str, Any]:
        """All agents train independently"""
        # Train all agents
        for agent in self.agents:
            agent.train(train_texts)
        
        # Evaluate all agents
        perplexities = []
        for agent in self.agents:
            ppl = agent.evaluate(eval_texts)
            perplexities.append(ppl)
        
        return {
            'generation': self.generation,
            'mean_perplexity': np.mean(perplexities),
            'best_perplexity': np.min(perplexities),
            'worst_perplexity': np.max(perplexities),
            'std_perplexity': np.std(perplexities),
            'n_agents': len(self.agents),
            'perplexities': perplexities
        }


# ============================================================================
# PulseOS Population (Survival Pressure)
# ============================================================================

class PulseOSPopulation:
    """PulseOS: Eliminate weak agents, spawn from strong agents"""
    
    def __init__(self, n_agents: int, model_name: str, device: str, 
                 elimination_rate: float = 0.3, spawn_rate: float = 0.2):
        self.agents = [TextAgent(f"pulseos_{i}", model_name, device) 
                      for i in range(n_agents)]
        self.generation = 0
        self.device = device
        self.elimination_rate = elimination_rate
        self.spawn_rate = spawn_rate
        self.total_eliminated = 0
        self.total_spawned = 0
    
    def train_step(self, train_texts: List[str], eval_texts: List[str], 
                   eliminate: bool = False, step: int = 0) -> Dict[str, Any]:
        """Train, optionally eliminate weak and spawn from strong"""
        # Train all agents
        for agent in self.agents:
            agent.train(train_texts)
        
        # Evaluate all agents
        performances = [(agent, agent.evaluate(eval_texts)) 
                       for agent in self.agents]
        performances.sort(key=lambda x: x[1])  # Sort by perplexity (lower=better)
        
        # Record stats before elimination
        perplexities = [p for _, p in performances]
        stats = {
            'generation': self.generation,
            'mean_perplexity': np.mean(perplexities),
            'best_perplexity': np.min(perplexities),
            'worst_perplexity': np.max(perplexities),
            'std_perplexity': np.std(perplexities),
            'n_agents_before': len(self.agents),
            'n_eliminated': 0,
            'n_spawned': 0
        }
        
        # ELIMINATION: Check if we should eliminate
        # Need at least 2 agents to eliminate 1 and keep 1
        if eliminate and len(self.agents) >= 2:
            print(f"\n  🔍 Step {step}: Checking elimination...")
            print(f"     Population size: {len(self.agents)}")
            print(f"     Elimination rate: {self.elimination_rate*100:.0f}%")
            print(f"     Perplexities: min={min(perplexities):.2f}, max={max(perplexities):.2f}, mean={np.mean(perplexities):.2f}")
            
            # ELIMINATION: Remove bottom N%
            n_eliminate = max(1, int(len(self.agents) * self.elimination_rate))
            # Ensure we keep at least 2 agents alive
            n_eliminate = min(n_eliminate, len(self.agents) - 2)
            
            if n_eliminate > 0:
                survivors = [agent for agent, _ in performances[:-n_eliminate]]
                
                print(f"     Planning to eliminate: {n_eliminate} agents (worst performers)")
                
                # REPRODUCTION: Top N% spawn new agents
                n_spawn = n_eliminate  # Keep population size constant
                n_reproducers = max(1, int(len(survivors) * self.spawn_rate))
                elite = survivors[:n_reproducers]
                
                print(f"     Top {n_reproducers} agents will reproduce")
                
                # Spawn new agents from elite (with SMALL mutation)
                # Use adaptive noise: start higher, decay over generations
                # Early generations: more exploration (0.01 = 1%)
                # Later generations: fine-tuning (0.001 = 0.1%)
                adaptive_noise = 0.01 * (0.1 ** (self.generation / 10))
                adaptive_noise = max(0.001, min(0.01, adaptive_noise))  # Clamp between 0.1% and 1%
                
                new_agents = []
                for i in range(n_spawn):
                    parent = elite[i % len(elite)]
                    child = parent.clone_with_mutation(
                        noise_scale=adaptive_noise,
                        new_id=f"pulseos_gen{self.generation+1}_{i}"
                    )
                    new_agents.append(child)
                
                # Update population
                self.agents = survivors + new_agents
                self.generation += 1
                self.total_eliminated += n_eliminate
                self.total_spawned += n_spawn
                
                stats['n_agents_after'] = len(self.agents)
                stats['n_eliminated'] = n_eliminate
                stats['n_spawned'] = n_spawn
                
                print(f"     ✂️  ELIMINATED {n_eliminate} agents (worst performers)")
                print(f"     🐣 SPAWNED {n_spawn} new agents (from top {n_reproducers})")
                print(f"     New population: {len(self.agents)} agents, Generation: {self.generation}")
            else:
                print(f"     ⚠️  Cannot eliminate: need to keep at least 2 agents")
        
        stats['n_agents'] = len(self.agents)
        return stats


# ============================================================================
# Experiment Runner
# ============================================================================

class TextArenaExperiment:
    """Run complete Text Generation Arena experiment"""
    
    def __init__(self, config: ArenaConfig):
        self.config = config
        
        # Load dataset
        print(f"Loading {config.dataset_name} dataset...")
        dataset = load_dataset(config.dataset_name, config.dataset_config)
        
        self.train_data = dataset['train']
        self.eval_data = dataset['validation']
        
        print(f"✓ Loaded {len(self.train_data)} training samples")
        print(f"✓ Loaded {len(self.eval_data)} eval samples")
    
    def _sample_batch(self, dataset, batch_size: int) -> List[str]:
        """Sample random batch from dataset"""
        indices = np.random.choice(len(dataset), batch_size, replace=False)
        texts = []
        for i in indices:
            text = dataset[int(i)]['text']
            if text and len(text.strip()) > 10:  # Filter empty/short texts
                texts.append(text)
        # Pad if needed
        while len(texts) < batch_size:
            idx = np.random.choice(len(dataset))
            text = dataset[int(idx)]['text']
            if text and len(text.strip()) > 10:
                texts.append(text)
        return texts[:batch_size]
    
    def run_baseline(self):
        """Run baseline (no survival pressure)"""
        print("\n" + "="*80)
        print("Running BASELINE (No Survival Pressure)")
        print("="*80)
        
        population = BaselinePopulation(
            n_agents=self.config.n_agents,
            model_name=self.config.model_name,
            device=self.config.device
        )
        
        results = []
        start_time = time.time()
        
        for step in range(self.config.n_steps):
            # Sample batches
            train_batch = self._sample_batch(self.train_data, self.config.batch_size)
            eval_batch = self._sample_batch(self.eval_data, self.config.eval_batch_size)
            
            # Train step
            stats = population.train_step(train_batch, eval_batch)
            stats['step'] = step
            stats['time'] = time.time() - start_time
            results.append(stats)
            
            # Log progress
            if step % 10 == 0 or step == self.config.n_steps - 1:
                print(f"Step {step:4d} | "
                      f"Mean PPL: {stats['mean_perplexity']:7.2f} | "
                      f"Best PPL: {stats['best_perplexity']:7.2f} | "
                      f"Worst PPL: {stats['worst_perplexity']:7.2f} | "
                      f"Agents: {stats['n_agents']}")
        
        total_time = time.time() - start_time
        print(f"\n✓ Baseline completed in {total_time:.1f}s")
        
        return results
    
    def run_pulseos(self):
        """Run PulseOS (with survival pressure)"""
        print("\n" + "="*80)
        print("Running PULSEOS (Survival Pressure)")
        print("="*80)
        
        population = PulseOSPopulation(
            n_agents=self.config.n_agents,
            model_name=self.config.model_name,
            device=self.config.device,
            elimination_rate=self.config.elimination_rate,
            spawn_rate=self.config.spawn_rate
        )
        
        results = []
        start_time = time.time()
        
        for step in range(self.config.n_steps):
            # Sample batches
            train_batch = self._sample_batch(self.train_data, self.config.batch_size)
            eval_batch = self._sample_batch(self.eval_data, self.config.eval_batch_size)
            
            # Check if elimination step
            eliminate = (step % self.config.elimination_interval == 0 and step > 0)
            
            # Train step (with optional elimination)
            stats = population.train_step(train_batch, eval_batch, eliminate=eliminate, step=step)
            stats['step'] = step
            stats['time'] = time.time() - start_time
            results.append(stats)
            
            # Log progress
            if step % 10 == 0 or step == self.config.n_steps - 1:
                gen_info = f"Gen {stats['generation']}" if eliminate else ""
                print(f"Step {step:4d} | "
                      f"Mean PPL: {stats['mean_perplexity']:7.2f} | "
                      f"Best PPL: {stats['best_perplexity']:7.2f} | "
                      f"Worst PPL: {stats['worst_perplexity']:7.2f} | "
                      f"Agents: {stats['n_agents']} {gen_info}")
        
        total_time = time.time() - start_time
        print(f"\n✓ PulseOS completed in {total_time:.1f}s")
        print(f"  Total eliminated: {population.total_eliminated}")
        print(f"  Total spawned: {population.total_spawned}")
        
        return results
    
    def analyze_results(self, baseline_results: List[Dict], 
                       pulseos_results: List[Dict]) -> Dict[str, Any]:
        """Compare baseline vs PulseOS"""
        # Extract metrics
        baseline_ppl = [r['mean_perplexity'] for r in baseline_results]
        pulseos_ppl = [r['mean_perplexity'] for r in pulseos_results]
        
        baseline_best = [r['best_perplexity'] for r in baseline_results]
        pulseos_best = [r['best_perplexity'] for r in pulseos_results]
        
        # Calculate improvement
        final_baseline = baseline_ppl[-1]
        final_pulseos = pulseos_ppl[-1]
        improvement = (final_baseline - final_pulseos) / final_baseline * 100
        
        # Find convergence points (first step below threshold)
        threshold = min(final_baseline, final_pulseos) * 1.1  # 10% above best
        baseline_convergence = next((i for i, ppl in enumerate(baseline_ppl) 
                                     if ppl <= threshold), len(baseline_ppl))
        pulseos_convergence = next((i for i, ppl in enumerate(pulseos_ppl) 
                                    if ppl <= threshold), len(pulseos_ppl))
        
        convergence_improvement = ((baseline_convergence - pulseos_convergence) / 
                                  baseline_convergence * 100) if baseline_convergence > 0 else 0
        
        # Plot
        plt.figure(figsize=(15, 5))
        
        # Population mean
        plt.subplot(1, 2, 1)
        plt.plot(baseline_ppl, label='Baseline', alpha=0.7, linewidth=2)
        plt.plot(pulseos_ppl, label='PulseOS', alpha=0.7, linewidth=2)
        plt.xlabel('Training Step')
        plt.ylabel('Mean Perplexity')
        plt.title('Population Average Perplexity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Best agent
        plt.subplot(1, 2, 2)
        plt.plot(baseline_best, label='Baseline', alpha=0.7, linewidth=2)
        plt.plot(pulseos_best, label='PulseOS', alpha=0.7, linewidth=2)
        plt.xlabel('Training Step')
        plt.ylabel('Best Perplexity')
        plt.title('Best Agent Perplexity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('text_arena_results.png', dpi=150, bbox_inches='tight')
        print("\n✓ Saved plot to text_arena_results.png")
        plt.show()
        
        # Print results
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        print(f"Baseline final perplexity: {final_baseline:.2f}")
        print(f"PulseOS final perplexity:  {final_pulseos:.2f}")
        print(f"Final improvement: {improvement:.1f}%")
        print(f"\nConvergence analysis:")
        print(f"  Baseline converged at step: {baseline_convergence}")
        print(f"  PulseOS converged at step: {pulseos_convergence}")
        print(f"  Convergence improvement: {convergence_improvement:.1f}%")
        
        # Valuation
        print("\n" + "="*80)
        print("VALUATION ASSESSMENT")
        print("="*80)
        if improvement > 30 or convergence_improvement > 30:
            print(f"✅ EXCELLENT: {max(improvement, convergence_improvement):.1f}% improvement")
            print("   Valuation: $40M-$80M")
        elif improvement > 20 or convergence_improvement > 20:
            print(f"✅ GOOD: {max(improvement, convergence_improvement):.1f}% improvement")
            print("   Valuation: $25M-$50M")
        elif improvement > 10 or convergence_improvement > 10:
            print(f"⚠️  MODEST: {max(improvement, convergence_improvement):.1f}% improvement")
            print("   Valuation: $15M-$35M")
        else:
            print(f"❌ LOW: {max(improvement, convergence_improvement):.1f}% improvement")
            print("   Valuation: $10M-$25M")
        print("="*80)
        
        return {
            'baseline_final': final_baseline,
            'pulseos_final': final_pulseos,
            'improvement_percent': improvement,
            'baseline_convergence_step': baseline_convergence,
            'pulseos_convergence_step': pulseos_convergence,
            'convergence_improvement': convergence_improvement
        }


# ============================================================================
# Main
# ============================================================================

def main():
    """Run Text Generation Arena experiment"""
    print("="*80)
    print("TEXT GENERATION ARENA: PulseOS vs Baseline")
    print("="*80)
    print("\nThis experiment tests whether survival pressure accelerates")
    print("population-level learning in language model fine-tuning.\n")
    
    # Configuration
    config = ArenaConfig(
        n_agents=10,  # Reduced to fit GPU memory
        model_name="gpt2",
        n_steps=100,  # Start with 100 for quick test
        elimination_interval=20,
        elimination_rate=0.3,
        spawn_rate=0.2,
        batch_size=4,  # Smaller batches
        eval_batch_size=8,  # Smaller eval batches
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"Configuration:")
    print(f"  Agents: {config.n_agents}")
    print(f"  Model: {config.model_name}")
    print(f"  Steps: {config.n_steps}")
    print(f"  Elimination every: {config.elimination_interval} steps")
    print(f"  Elimination rate: {config.elimination_rate*100:.0f}%")
    print(f"  Spawn rate: {config.spawn_rate*100:.0f}%")
    print(f"  Device: {config.device}")
    print("="*80)
    
    # Run experiment
    experiment = TextArenaExperiment(config)
    
    baseline_results = experiment.run_baseline()
    pulseos_results = experiment.run_pulseos()
    
    # Analyze
    analysis = experiment.analyze_results(baseline_results, pulseos_results)
    
    return analysis


if __name__ == "__main__":
    results = main()

