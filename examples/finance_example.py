"""
Finance Example: Portfolio Optimization with Risk Constraints

Demonstrates PulseOS for financial applications where agents must optimize
returns while respecting risk constraints and regulatory limits.
"""

import asyncio
import numpy as np
from pulseos import Runtime, Agent, SurvivalConstraint, Config


class PortfolioAgent(Agent):
    """
    Portfolio agent that learns optimal asset allocation.
    
    Constraints:
    - Maximum portfolio risk (volatility)
    - Maximum position size per asset
    - Minimum diversification (number of assets)
    - Regulatory limits (e.g., max leverage)
    """
    
    def __init__(self, agent_id: str, initial_capital: float):
        super().__init__(agent_id)
        self.capital = initial_capital
        self.initial_capital = initial_capital
        
        # Available assets with expected returns and volatilities
        self.assets = {
            "Stock_A": {"return": 0.12, "volatility": 0.20},
            "Stock_B": {"return": 0.10, "volatility": 0.15},
            "Stock_C": {"return": 0.08, "volatility": 0.10},
            "Bond_D": {"return": 0.04, "volatility": 0.05},
            "Bond_E": {"return": 0.03, "volatility": 0.03},
        }
        
        # Portfolio weights (initial equal allocation)
        num_assets = len(self.assets)
        self.weights = np.ones(num_assets) / num_assets
        
        # Constraints
        self.max_portfolio_volatility = 0.15
        self.max_position_size = 0.40  # Max 40% in single asset
        self.min_diversification = 3  # Must hold at least 3 assets
        
        # Learning parameters
        self.learning_rate = 0.01
        self.exploration_rate = 0.1
    
    def compute_portfolio_return(self) -> float:
        """Compute expected portfolio return"""
        returns = np.array([asset["return"] for asset in self.assets.values()])
        return np.dot(self.weights, returns)
    
    def compute_portfolio_volatility(self) -> float:
        """Compute portfolio volatility (simplified)"""
        volatilities = np.array([asset["volatility"] for asset in self.assets.values()])
        # Simplified: assume zero correlation
        portfolio_variance = np.sum(self.weights ** 2 * volatilities ** 2)
        return np.sqrt(portfolio_variance)
    
    def compute_sharpe_ratio(self) -> float:
        """Compute Sharpe ratio (risk-adjusted return)"""
        portfolio_return = self.compute_portfolio_return()
        portfolio_volatility = self.compute_portfolio_volatility()
        risk_free_rate = 0.02
        
        if portfolio_volatility == 0:
            return 0.0
        
        return (portfolio_return - risk_free_rate) / portfolio_volatility
    
    async def step(self):
        """Execute one step of portfolio optimization"""
        # Generate random market returns (simplified)
        market_returns = {}
        for asset_name, asset_info in self.assets.items():
            expected_return = asset_info["return"]
            volatility = asset_info["volatility"]
            # Random return around expected value
            actual_return = np.random.normal(expected_return, volatility / np.sqrt(252))
            market_returns[asset_name] = actual_return
        
        # Update portfolio value
        asset_returns = np.array([market_returns[name] for name in self.assets.keys()])
        portfolio_return = np.dot(self.weights, asset_returns)
        self.capital *= (1 + portfolio_return)
        
        # Update portfolio weights (learning step)
        if np.random.random() < self.exploration_rate:
            # Exploration: random rebalancing
            new_weights = np.random.dirichlet(np.ones(len(self.assets)))
        else:
            # Exploitation: gradient-based update
            # Increase weights for assets with higher returns
            returns = np.array([asset["return"] for asset in self.assets.values()])
            gradient = returns - np.mean(returns)
            new_weights = self.weights + self.learning_rate * gradient
            new_weights = np.clip(new_weights, 0, self.max_position_size)
            new_weights = new_weights / np.sum(new_weights)
        
        # Apply constraints
        new_weights = self._apply_constraints(new_weights)
        self.weights = new_weights
        
        return {
            "capital": self.capital,
            "portfolio_return": portfolio_return,
            "portfolio_volatility": self.compute_portfolio_volatility(),
            "sharpe_ratio": self.compute_sharpe_ratio(),
            "weights": self.weights.tolist()
        }
    
    def _apply_constraints(self, weights: np.ndarray) -> np.ndarray:
        """Apply portfolio constraints"""
        # Enforce max position size
        weights = np.clip(weights, 0, self.max_position_size)
        
        # Enforce minimum diversification
        num_positions = np.sum(weights > 0.01)  # Count significant positions
        if num_positions < self.min_diversification:
            # Redistribute to meet diversification requirement
            top_indices = np.argsort(weights)[-self.min_diversification:]
            weights_new = np.zeros_like(weights)
            weights_new[top_indices] = weights[top_indices]
            weights = weights_new
        
        # Normalize
        weights = weights / np.sum(weights)
        
        # Check volatility constraint
        if self.compute_portfolio_volatility() > self.max_portfolio_volatility:
            # Reduce weights in high-volatility assets
            volatilities = np.array([asset["volatility"] for asset in self.assets.values()])
            reduction_factor = self.max_portfolio_volatility / self.compute_portfolio_volatility()
            weights = weights * (1 - (volatilities - np.mean(volatilities)) * reduction_factor)
            weights = np.clip(weights, 0, None)
            weights = weights / np.sum(weights)
        
        return weights
    
    def get_performance_metric(self) -> float:
        """
        Performance metric combines:
        - Return on investment (higher is better)
        - Risk-adjusted return (Sharpe ratio)
        - Constraint satisfaction
        """
        # Return metric (0 to 1)
        return_metric = (self.capital / self.initial_capital - 1.0) / 2.0  # Normalize
        return_metric = np.clip(return_metric, 0, 1)
        
        # Sharpe ratio metric (0 to 1)
        sharpe = self.compute_sharpe_ratio()
        sharpe_metric = np.clip(sharpe / 2.0, 0, 1)  # Normalize (Sharpe typically 0-2)
        
        # Constraint satisfaction metric
        volatility = self.compute_portfolio_volatility()
        volatility_constraint = 1.0 if volatility <= self.max_portfolio_volatility else 0.5
        
        max_position = np.max(self.weights)
        position_constraint = 1.0 if max_position <= self.max_position_size else 0.5
        
        num_positions = np.sum(self.weights > 0.01)
        diversification_constraint = 1.0 if num_positions >= self.min_diversification else 0.5
        
        constraint_metric = (volatility_constraint + position_constraint + diversification_constraint) / 3.0
        
        # Combined performance
        performance = 0.4 * return_metric + 0.3 * sharpe_metric + 0.3 * constraint_metric
        
        return performance


async def main():
    """Run finance example"""
    print("💰 Finance Example: Portfolio Optimization with Risk Constraints")
    print("=" * 70)
    
    # Create survival constraint
    # Portfolios must maintain performance above 0.5 to "survive"
    constraint = SurvivalConstraint(threshold=0.5)
    
    # Configure runtime
    config = Config(
        snapshot_interval=5.0,
        max_snapshots=20,
        metrics_enabled=True
    )
    
    runtime = Runtime(constraint=constraint, config=config)
    
    # Create portfolio agents
    num_portfolios = 3
    initial_capital = 100000.0
    
    for i in range(num_portfolios):
        portfolio = PortfolioAgent(f"portfolio_{i}", initial_capital)
        runtime.register_agent(f"portfolio_{i}", portfolio)
    
    print(f"Created {num_portfolios} portfolio agents")
    print(f"Initial capital per portfolio: ${initial_capital:,.2f}")
    print("Running optimization...")
    print()
    
    # Run optimization
    await runtime.run(max_steps=50)
    
    # Print results
    stats = runtime.get_statistics()
    print("\n📊 Optimization Results:")
    print(f"  Final Step: {stats['current_step']}")
    print(f"  Average Survival Signal: {stats['average_survival_signal']:.3f}")
    print(f"  Portfolios Survived: {stats['agent_count']}")
    
    # Print individual portfolio status
    print("\n💰 Portfolio Status:")
    for agent_id, agent in runtime.agents.items():
        if isinstance(agent, PortfolioAgent):
            final_capital = agent.capital
            return_pct = (final_capital / agent.initial_capital - 1.0) * 100
            sharpe = agent.compute_sharpe_ratio()
            volatility = agent.compute_portfolio_volatility()
            performance = agent.get_performance_metric()
            
            print(f"  {agent_id}:")
            print(f"    Final Capital: ${final_capital:,.2f}")
            print(f"    Return: {return_pct:+.2f}%")
            print(f"    Sharpe Ratio: {sharpe:.3f}")
            print(f"    Volatility: {volatility:.3f}")
            print(f"    Performance: {performance:.3f}")
            
            # Show top holdings
            asset_names = list(agent.assets.keys())
            top_holdings = sorted(
                zip(asset_names, agent.weights),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            print(f"    Top Holdings:")
            for asset_name, weight in top_holdings:
                print(f"      {asset_name}: {weight*100:.1f}%")
    
    print("\n✅ Example completed!")


if __name__ == "__main__":
    asyncio.run(main())

