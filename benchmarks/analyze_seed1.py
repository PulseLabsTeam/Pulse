"""
Analyze Seed 1 Success Factors

Seed 1 achieved 3.688 Sharpe (+1.7% vs PPO) with standard initialization.
Let's analyze what makes it successful and improve initialization accordingly.
"""

import numpy as np
from trading_env import TradingEnv
from pulseos_trading_agent import PulseOSTradingAgent
import yfinance as yf

def analyze_seed_initialization(seed: int):
    """
    Analyze initialization characteristics for a given seed.
    """
    print(f"=" * 80)
    print(f"ANALYZING SEED {seed} INITIALIZATION")
    print(f"=" * 80)
    
    # Download data
    symbol = "SPY"
    start = "2023-01-01"
    end = "2024-01-01"
    data = yf.download(symbol, start=start, end=end, progress=False)
    
    # Create environment
    env = TradingEnv(data, initial_capital=100000.0, commission=0.001)
    state = env.reset()
    state_dim = len(state)
    action_dim = 3
    
    # Standard initialization (what seed 1 used)
    np.random.seed(seed)
    scale_standard = np.sqrt(2.0 / (state_dim + action_dim))
    policy_weights_standard = np.random.randn(state_dim, action_dim) * scale_standard * 0.5
    policy_bias_standard = np.zeros(action_dim)
    value_weights_standard = np.random.randn(state_dim) * scale_standard * 0.3
    value_bias_standard = 0.0
    
    # Improved initialization (current)
    np.random.seed(seed)
    scale_improved = np.sqrt(2.0 / (state_dim + action_dim))
    policy_weights_improved = np.random.randn(state_dim, action_dim) * scale_improved * 0.3
    policy_bias_improved = np.random.randn(action_dim) * 0.01
    value_weights_improved = np.random.randn(state_dim) * scale_improved * 0.2
    value_bias_improved = 0.0
    
    # Analyze characteristics
    print(f"\n📊 Initialization Characteristics:")
    print(f"  State Dimension: {state_dim}")
    print(f"  Action Dimension: {action_dim}")
    print(f"  Scale: {scale_standard:.6f}")
    
    print(f"\n🔍 Standard Initialization (Seed {seed}):")
    print(f"  Policy Weights:")
    print(f"    Mean: {np.mean(policy_weights_standard):.6f}")
    print(f"    Std: {np.std(policy_weights_standard):.6f}")
    print(f"    Min: {np.min(policy_weights_standard):.6f}")
    print(f"    Max: {np.max(policy_weights_standard):.6f}")
    print(f"    Magnitude (L2 norm): {np.linalg.norm(policy_weights_standard):.6f}")
    print(f"  Policy Bias:")
    print(f"    Mean: {np.mean(policy_bias_standard):.6f}")
    print(f"    Std: {np.std(policy_bias_standard):.6f}")
    print(f"  Value Weights:")
    print(f"    Mean: {np.mean(value_weights_standard):.6f}")
    print(f"    Std: {np.std(value_weights_standard):.6f}")
    print(f"    Magnitude (L2 norm): {np.linalg.norm(value_weights_standard):.6f}")
    
    print(f"\n🔍 Improved Initialization (Seed {seed}):")
    print(f"  Policy Weights:")
    print(f"    Mean: {np.mean(policy_weights_improved):.6f}")
    print(f"    Std: {np.std(policy_weights_improved):.6f}")
    print(f"    Min: {np.min(policy_weights_improved):.6f}")
    print(f"    Max: {np.max(policy_weights_improved):.6f}")
    print(f"    Magnitude (L2 norm): {np.linalg.norm(policy_weights_improved):.6f}")
    print(f"  Policy Bias:")
    print(f"    Mean: {np.mean(policy_bias_improved):.6f}")
    print(f"    Std: {np.std(policy_bias_improved):.6f}")
    print(f"  Value Weights:")
    print(f"    Mean: {np.mean(value_weights_improved):.6f}")
    print(f"    Std: {np.std(value_weights_improved):.6f}")
    print(f"    Magnitude (L2 norm): {np.linalg.norm(value_weights_improved):.6f}")
    
    # Compare
    print(f"\n📈 Comparison:")
    policy_norm_ratio = np.linalg.norm(policy_weights_improved) / np.linalg.norm(policy_weights_standard)
    value_norm_ratio = np.linalg.norm(value_weights_improved) / np.linalg.norm(value_weights_standard)
    print(f"  Policy Weight Magnitude Ratio: {policy_norm_ratio:.3f} (improved/standard)")
    print(f"  Value Weight Magnitude Ratio: {value_norm_ratio:.3f} (improved/standard)")
    
    # Analyze successful seed characteristics
    print(f"\n💡 Key Insights:")
    print(f"  Standard init (seed {seed}):")
    print(f"    - Larger weights (0.5x multiplier)")
    print(f"    - Zero bias (no exploration bias)")
    print(f"    - Result: 3.688 Sharpe (+1.7% vs PPO)")
    print(f"  Improved init (seed {seed}):")
    print(f"    - Smaller weights (0.3x multiplier)")
    print(f"    - Small random bias (0.01 scale)")
    print(f"    - More conservative")
    
    # Test initial action distribution
    print(f"\n🎯 Initial Action Distribution Test:")
    test_state = np.random.randn(state_dim) * 0.1  # Small test state
    
    # Standard init
    np.random.seed(seed)
    scale_standard = np.sqrt(2.0 / (state_dim + action_dim))
    policy_weights_standard = np.random.randn(state_dim, action_dim) * scale_standard * 0.5
    policy_bias_standard = np.zeros(action_dim)
    logits_standard = test_state @ policy_weights_standard + policy_bias_standard
    probs_standard = np.exp(logits_standard) / np.sum(np.exp(logits_standard))
    
    # Improved init
    np.random.seed(seed)
    scale_improved = np.sqrt(2.0 / (state_dim + action_dim))
    policy_weights_improved = np.random.randn(state_dim, action_dim) * scale_improved * 0.3
    policy_bias_improved = np.random.randn(action_dim) * 0.01
    logits_improved = test_state @ policy_weights_improved + policy_bias_improved
    probs_improved = np.exp(logits_improved) / np.sum(np.exp(logits_improved))
    
    print(f"  Standard Init Action Probs: {probs_standard}")
    print(f"  Improved Init Action Probs: {probs_improved}")
    print(f"  Entropy (Standard): {-np.sum(probs_standard * np.log(probs_standard + 1e-10)):.4f}")
    print(f"  Entropy (Improved): {-np.sum(probs_improved * np.log(probs_improved + 1e-10)):.4f}")
    
    return {
        "seed": seed,
        "standard": {
            "policy_norm": np.linalg.norm(policy_weights_standard),
            "value_norm": np.linalg.norm(value_weights_standard),
            "policy_mean": np.mean(policy_weights_standard),
            "policy_std": np.std(policy_weights_standard),
            "bias_mean": np.mean(policy_bias_standard),
            "bias_std": np.std(policy_bias_standard)
        },
        "improved": {
            "policy_norm": np.linalg.norm(policy_weights_improved),
            "value_norm": np.linalg.norm(value_weights_improved),
            "policy_mean": np.mean(policy_weights_improved),
            "policy_std": np.std(policy_weights_improved),
            "bias_mean": np.mean(policy_bias_improved),
            "bias_std": np.std(policy_bias_improved)
        }
    }

if __name__ == "__main__":
    # Analyze seed 1 (successful)
    seed1_analysis = analyze_seed_initialization(1)
    
    print(f"\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\n💡 Key Finding:")
    print("  Seed 1 succeeded with standard initialization (0.5x multiplier, zero bias)")
    print("  This suggests that:")
    print("    1. Larger initial weights may help some seeds")
    print("    2. Zero bias may be better for some seeds")
    print("    3. Seed 1 happened to get lucky initialization")
    print("\n  Recommendation:")
    print("    - Keep improved initialization (0.3x) for stability")
    print("    - But try slightly larger multiplier (0.35x) as compromise")
    print("    - Keep small bias but make it adaptive based on seed")



