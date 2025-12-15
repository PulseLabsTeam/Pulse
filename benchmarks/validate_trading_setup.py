"""
Quick validation test for trading RL setup
Runs a minimal test to verify everything works
"""

import asyncio
import numpy as np
import pandas as pd
from trading_env import TradingEnv
from ppo_trading_agent import PPOTradingAgent
from pulseos_trading_agent import PulseOSTradingAgent
from pulseos import Runtime, Config, SurvivalConstraint

# Create synthetic data for quick test
def create_synthetic_data(n_days=100):
    """Create synthetic stock price data"""
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    prices = 100 + np.cumsum(np.random.randn(n_days) * 2)
    data = pd.DataFrame({
        'Close': prices,
        'Open': prices * 0.99,
        'High': prices * 1.01,
        'Low': prices * 0.98,
        'Volume': np.random.randint(1000000, 5000000, n_days)
    }, index=dates)
    return data

async def quick_test():
    """Run a quick validation test"""
    print("Running quick validation test...")
    
    # Create synthetic data
    data = create_synthetic_data(100)
    
    # Test environment
    print("Testing TradingEnv...")
    env = TradingEnv(data, initial_capital=100000.0)
    state = env.reset()
    print(f"  ✓ Environment created, state shape: {state.shape}")
    
    # Test PPO agent
    print("Testing PPO Agent...")
    ppo_agent = PPOTradingAgent(env)
    action, log_prob, value = ppo_agent.select_action(state)
    print(f"  ✓ PPO agent created, action: {action}, value: {value:.3f}")
    
    # Test PulseOS agent
    print("Testing PulseOS Agent...")
    pulseos_env = TradingEnv(data, initial_capital=100000.0)
    constraint = SurvivalConstraint(threshold=0.4)
    config = Config()
    runtime = Runtime(constraint=constraint, config=config)
    pulseos_agent = PulseOSTradingAgent("test_agent", pulseos_env)
    runtime.register_agent(pulseos_agent.agent_id, pulseos_agent)
    print(f"  ✓ PulseOS agent created and registered")
    
    # Test one step
    result = await pulseos_agent.step()
    print(f"  ✓ PulseOS step executed: {result.get('action')}")
    
    print("\n✅ All components validated successfully!")
    print("Ready to run full trading RL test.")

if __name__ == "__main__":
    asyncio.run(quick_test())




