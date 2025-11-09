"""
Finance Example - Portfolio Optimization with Risk Constraints

Demonstrates PulseOS for financial applications with risk constraints,
portfolio optimization, and regulatory compliance requirements.
"""

import asyncio
import random
import numpy as np
from pulseos import Runtime, Agent, SurvivalConstraint


class PortfolioAgent(Agent):
    """
    Portfolio management agent with risk constraints.
    
    Simulates a portfolio manager that must:
    - Maximize returns
    - Stay within risk limits
    - Maintain diversification
    - Meet regulatory constraints
    """
    
    def __init__(self, agent_id: str, initial_capital: float = 100000.0):
        super().__init__(agent_id)
        self.capital = initial_capital
        self.initial_capital = initial_capital
        
        # Portfolio: allocation across 5 asset classes
        self.allocations = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # Equal initial allocation
        
        # Asset characteristics (expected return, volatility)
        self.assets = [
            {"return": 0.08, "volatility": 0.15},  # Stocks
            {"return": 0.04, "volatility": 0.08},   # Bonds
            {"return": 0.12, "volatility": 0.25},   # Commodities
            {"return": 0.06, "volatility": 0.10},  # Real Estate
            {"return": 0.03, "volatility": 0.05}   # Cash
        ]
        
        # Risk limits
        self.max_single_allocation = 0.4  # Max 40% in single asset
        self.max_portfolio_volatility = 0.15  # Max 15% portfolio volatility
        self.min_diversification = 0.3  # Min diversification score
        
        # Performance tracking
        self.returns_history = []
        self.risk_violations = 0
    
    async def step(self) -> dict:
        """Execute one step of portfolio management."""
        # Compute current portfolio metrics
        portfolio_return = sum(
            self.allocations[i] * self.assets[i]["return"]
            for i in range(len(self.assets))
        )
        
        portfolio_volatility = np.sqrt(
            sum(
                self.allocations[i] ** 2 * self.assets[i]["volatility"] ** 2
                for i in range(len(self.assets))
            )
        )
        
        # Check risk constraints
        max_allocation = np.max(self.allocations)
        diversification = 1.0 - np.sum(self.allocations ** 2)  # Herfindahl index
        
        # Penalize violations
        if max_allocation > self.max_single_allocation:
            self.risk_violations += 1
        if portfolio_volatility > self.max_portfolio_volatility:
            self.risk_violations += 1
        if diversification < self.min_diversification:
            self.risk_violations += 1
        
        # Rebalance portfolio
        if random.random() > self.exploration_rate:
            # Exploit: optimize allocation based on risk-adjusted returns
            target_allocations = np.array([
                max(0.0, min(self.max_single_allocation, 
                    asset["return"] / asset["volatility"]))
                for asset in self.assets
            ])
            
            # Normalize to sum to 1
            target_allocations = target_allocations / np.sum(target_allocations)
            
            # Smooth transition with learning rate
            self.allocations = (
                (1 - self.learning_rate) * self.allocations +
                self.learning_rate * target_allocations
            )
        else:
            # Explore: random rebalancing
            noise = np.random.normal(0, 0.05, len(self.assets))
            self.allocations = self.allocations + noise
            self.allocations = np.maximum(0, self.allocations)  # No negative allocations
            self.allocations = self.allocations / np.sum(self.allocations)  # Normalize
        
        # Apply returns (with some noise)
        period_return = portfolio_return + np.random.normal(0, portfolio_volatility * 0.1)
        self.capital *= (1 + period_return)
        self.returns_history.append(period_return)
        
        # Keep only recent history
        if len(self.returns_history) > 100:
            self.returns_history = self.returns_history[-100:]
        
        return {
            "capital": self.capital,
            "portfolio_return": portfolio_return,
            "portfolio_volatility": portfolio_volatility,
            "max_allocation": max_allocation,
            "diversification": diversification,
            "risk_violations": self.risk_violations
        }
    
    def get_performance_metric(self) -> float:
        """
        Performance metric combining:
        - Returns (higher is better)
        - Risk management (lower volatility is better)
        - Constraint compliance (no violations is better)
        """
        # Return score (normalized to 0-1)
        total_return = (self.capital - self.initial_capital) / self.initial_capital
        return_score = min(1.0, max(0.0, (total_return + 0.2) / 0.4))  # Scale to 0-1
        
        # Risk score (lower volatility is better)
        portfolio_volatility = np.sqrt(
            sum(
                self.allocations[i] ** 2 * self.assets[i]["volatility"] ** 2
                for i in range(len(self.assets))
            )
        )
        risk_score = max(0.0, 1.0 - portfolio_volatility / self.max_portfolio_volatility)
        
        # Compliance score (no violations is better)
        compliance_score = max(0.0, 1.0 - self.risk_violations * 0.1)
        
        # Weighted combination
        performance = 0.4 * return_score + 0.4 * risk_score + 0.2 * compliance_score
        
        return min(1.0, max(0.0, performance))


async def main():
    """Run finance example with risk constraints."""
    print("💰 PulseOS Finance Example - Portfolio Optimization\n")
    
    # Create survival constraint with risk focus
    constraint = SurvivalConstraint(
        threshold=0.75,  # High performance required
        constraint_type="statistical",
        statistical_mode="mean"  # Average performance over time
    )
    
    # Create runtime
    runtime = Runtime(constraint=constraint)
    
    # Create portfolio managers
    num_portfolios = 15
    print(f"Creating {num_portfolios} portfolio managers...")
    
    for i in range(num_portfolios):
        initial_capital = random.uniform(50000, 200000)
        portfolio = PortfolioAgent(f"portfolio_{i}", initial_capital)
        runtime.register_agent(f"portfolio_{i}", portfolio)
    
    print(f"Running optimization for 150 steps...")
    print("Portfolios must maximize returns while managing risk.\n")
    
    # Run optimization
    await runtime.run(max_steps=150)
    
    # Print results
    stats = runtime.get_statistics()
    print("\n" + "="*70)
    print("PORTFOLIO OPTIMIZATION RESULTS")
    print("="*70)
    print(f"Steps completed: {stats['current_step']}")
    print(f"Average survival signal: {stats['average_survival_signal']:.3f}")
    print(f"Final learning rate (alpha): {stats['current_alpha']:.6f}")
    print(f"Final exploration rate (epsilon): {stats['current_epsilon']:.3f}")
    
    print("\nPortfolio Performance:")
    print("-" * 70)
    
    successful_portfolios = 0
    total_return = 0.0
    
    for agent_id, agent in runtime.agents.items():
        metric = agent.get_performance_metric()
        return_pct = (agent.capital - agent.initial_capital) / agent.initial_capital * 100
        violations = agent.risk_violations
        
        status = "✅" if metric >= 0.75 else "⚠️"
        if metric >= 0.75:
            successful_portfolios += 1
        
        total_return += return_pct
        
        print(f"{status} {agent_id:15s} | Metric: {metric:.3f} | "
              f"Return: {return_pct:+6.2f}% | Violations: {violations:3d}")
    
    avg_return = total_return / num_portfolios
    print(f"\n✅ {successful_portfolios}/{num_portfolios} portfolios met performance target")
    print(f"📊 Average return: {avg_return:+.2f}%")
    print(f"📈 Best portfolio: {max([(a.capital - a.initial_capital) / a.initial_capital * 100 for a in runtime.agents.values()]):+.2f}%")


if __name__ == "__main__":
    asyncio.run(main())

