"""
Trading Environment for Financial RL Testing

Uses real stock market data from Yahoo Finance to create a realistic
trading simulation environment.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class TradingState:
    """State of the trading environment"""
    current_step: int
    balance: float
    shares: int
    portfolio_value: float
    price: float
    position: int  # -1 (short), 0 (hold), 1 (long)
    equity_curve: list
    returns: list


class TradingEnv:
    """
    Trading environment using real stock market data.
    
    Actions:
    - 0: Hold
    - 1: Buy (or close short and go long)
    - 2: Sell (or close long and go short)
    
    State:
    - Price features (normalized)
    - Portfolio value
    - Position
    - Recent returns
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 100000.0,
        commission: float = 0.001,  # 10 basis points
        lookback_window: int = 20,
        episode_length: Optional[int] = None  # If None, use full dataset
    ):
        """
        Initialize trading environment.
        
        Args:
            data: DataFrame with OHLCV data (must have 'Close' column)
            initial_capital: Starting capital
            commission: Commission rate per trade
            lookback_window: Number of past prices to include in state
        """
        self.data = data.copy()
        if 'Close' not in self.data.columns:
            raise ValueError("Data must have 'Close' column")
        
        # Ensure data is sorted by date
        if isinstance(self.data.index, pd.DatetimeIndex):
            self.data = self.data.sort_index()
        
        self.initial_capital = initial_capital
        self.commission = commission
        # Adjust lookback_window if dataset is too small
        max_lookback = min(len(data) - 5, lookback_window)  # Need at least 5 steps after lookback
        self.lookback_window = max(5, max_lookback)  # Minimum lookback of 5
        
        # Episode length optimization: use shorter episodes for faster training
        # Default to 60 days if dataset is large enough, otherwise use full dataset
        if episode_length is None:
            if len(data) > 100:
                self.episode_length = 60  # Use 60-day windows for speed
            else:
                self.episode_length = len(data) - self.lookback_window - 1
        else:
            self.episode_length = min(episode_length, len(data) - self.lookback_window - 1)
        
        # Compute returns
        self.data['Returns'] = self.data['Close'].pct_change().fillna(0)
        
        # Normalize prices for state representation
        close_mean = self.data['Close'].mean()
        close_std = self.data['Close'].std()
        self.price_mean = float(close_mean.iloc[0]) if hasattr(close_mean, 'iloc') else float(close_mean)
        self.price_std = float(close_std.iloc[0]) if hasattr(close_std, 'iloc') else float(close_std)
        
        # Initialize trade tracking
        self._last_trade_price = None
        
        # Reset state
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        # Random start position for episode length optimization
        max_start = len(self.data) - self.episode_length - 1
        if max_start > self.lookback_window:
            start_pos = np.random.randint(self.lookback_window, max_start)
        else:
            start_pos = self.lookback_window
        
        self.current_step = start_pos
        self.episode_start_step = start_pos
        self.balance = self.initial_capital
        self.shares = 0
        self.position = 0  # 0 = no position, 1 = long, -1 = short
        self.equity_curve = [self.initial_capital]
        self.returns = []
        self.trade_history = []
        if self.current_step < len(self.data):
            init_price = self.data['Close'].iloc[self.current_step]
            self._last_trade_price = float(init_price.iloc[0]) if hasattr(init_price, 'iloc') else float(init_price)
        else:
            self._last_trade_price = None
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state vector"""
        if self.current_step < self.lookback_window:
            # Pad with zeros if not enough history
            prices = np.zeros(self.lookback_window)
            if self.current_step > 0:
                prices[-self.current_step:] = np.array(self.data['Close'].iloc[:self.current_step])
        else:
            prices = np.array(self.data['Close'].iloc[
                self.current_step - self.lookback_window:self.current_step
            ])
        
        # Normalize prices
        normalized_prices = (prices - self.price_mean) / (self.price_std + 1e-8)
        
        # Current price
        current_price = self.data['Close'].iloc[self.current_step]
        current_price_val = float(current_price.iloc[0]) if hasattr(current_price, 'iloc') else float(current_price)
        normalized_current = (current_price_val - self.price_mean) / (self.price_std + 1e-8)
        
        # Portfolio value (normalized)
        portfolio_value = self._get_portfolio_value()
        normalized_portfolio = (portfolio_value - self.initial_capital) / self.initial_capital
        
        # Position encoding
        position_encoding = np.array([float(self.position)])
        
        # Recent returns (last 5 steps)
        if len(self.returns) > 0:
            recent_returns = np.array(self.returns[-5:])
            if len(recent_returns) < 5:
                recent_returns = np.pad(
                    recent_returns,
                    (5 - len(recent_returns), 0),
                    mode='constant',
                    constant_values=0.0
                )
        else:
            recent_returns = np.zeros(5)
        
        # Ensure all arrays are 1D
        normalized_prices = np.atleast_1d(normalized_prices).flatten()
        normalized_current = np.atleast_1d(normalized_current).flatten()
        normalized_portfolio = np.atleast_1d(normalized_portfolio).flatten()
        position_encoding = np.atleast_1d(position_encoding).flatten()
        recent_returns = np.atleast_1d(recent_returns).flatten()
        
        # Combine state
        state = np.concatenate([
            normalized_prices,
            normalized_current,
            normalized_portfolio,
            position_encoding,
            recent_returns
        ])
        
        return state
    
    def _get_portfolio_value(self) -> float:
        """Calculate current portfolio value"""
        current_price = self.data['Close'].iloc[self.current_step]
        current_price_val = float(current_price.iloc[0]) if hasattr(current_price, 'iloc') else float(current_price)
        return self.balance + self.shares * current_price_val
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one trading step.
        
        Args:
            action: 0=hold, 1=buy, 2=sell
            
        Returns:
            state: Next state
            reward: Reward for this step
            done: Whether episode is done
            info: Additional information
        """
        if self.current_step >= len(self.data) - 1:
            return self._get_state(), 0.0, True, {"reason": "end_of_data"}
        
        current_price = self.data['Close'].iloc[self.current_step]
        current_price_val = float(current_price.iloc[0]) if hasattr(current_price, 'iloc') else float(current_price)
        next_price = self.data['Close'].iloc[self.current_step + 1]
        next_price_val = float(next_price.iloc[0]) if hasattr(next_price, 'iloc') else float(next_price)
        
        # Execute action
        reward = 0.0
        trade_executed = False
        
        if action == 1:  # Buy
            if self.position <= 0:
                # Close short if exists
                if self.position < 0:
                    # Close short position
                    last_price_val = float(self._last_trade_price.iloc[0]) if hasattr(self._last_trade_price, 'iloc') else float(self._last_trade_price)
                    profit = -self.shares * (current_price_val - last_price_val)
                    reward += profit / self.initial_capital
                    self.balance += -self.shares * current_price_val * (1 - self.commission)
                    self.shares = 0
                    self.position = 0
                    trade_executed = True
                
                # Open long position
                shares_to_buy = int(self.balance / (current_price_val * (1 + self.commission)))
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price_val * (1 + self.commission)
                    self.balance -= cost
                    self.shares = shares_to_buy
                    self.position = 1
                    self._last_trade_price = current_price_val
                    trade_executed = True
        
        elif action == 2:  # Sell
            if self.position >= 0:
                # Close long if exists
                if self.position > 0:
                    # Close long position
                    last_price_val = float(self._last_trade_price.iloc[0]) if hasattr(self._last_trade_price, 'iloc') else float(self._last_trade_price)
                    profit = self.shares * (current_price_val - last_price_val)
                    reward += profit / self.initial_capital
                    self.balance += self.shares * current_price_val * (1 - self.commission)
                    self.shares = 0
                    self.position = 0
                    trade_executed = True
                
                # Open short position (simplified - assume we can short)
                shares_to_short = int(self.balance / (current_price_val * (1 + self.commission)))
                if shares_to_short > 0:
                    proceeds = shares_to_short * current_price_val * (1 - self.commission)
                    self.balance += proceeds
                    self.shares = -shares_to_short
                    self.position = -1
                    self._last_trade_price = current_price_val
                    trade_executed = True
        
        # Move to next step
        self.current_step += 1
        
        # Calculate portfolio return
        portfolio_value = self._get_portfolio_value()
        portfolio_return = (portfolio_value - self.equity_curve[-1]) / self.equity_curve[-1]
        self.equity_curve.append(portfolio_value)
        self.returns.append(portfolio_return)
        
        # Reward is portfolio return
        reward += portfolio_return
        
        # Check if done (episode length optimization)
        episode_steps = self.current_step - self.episode_start_step
        done = (episode_steps >= self.episode_length) or (self.current_step >= len(self.data) - 1)
        
        info = {
            "portfolio_value": portfolio_value,
            "balance": self.balance,
            "shares": self.shares,
            "position": self.position,
            "trade_executed": trade_executed
        }
        
        return self._get_state(), reward, done, info
    
    def get_metrics(self) -> Dict[str, float]:
        """Calculate trading performance metrics"""
        if len(self.equity_curve) < 2:
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "annualized_return": 0.0
            }
        
        equity_array = np.array(self.equity_curve)
        returns_array = np.array(self.returns)
        
        # Total return
        total_return = (equity_array[-1] - equity_array[0]) / equity_array[0]
        
        # Sharpe ratio (annualized)
        if len(returns_array) > 1 and returns_array.std() > 0:
            sharpe_ratio = np.sqrt(252) * returns_array.mean() / (returns_array.std() + 1e-8)
        else:
            sharpe_ratio = 0.0
        
        # Max drawdown
        cumulative = np.cumprod(1 + returns_array)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0
        
        # Annualized return (assuming daily data)
        if len(returns_array) > 0:
            annualized_return = (1 + total_return) ** (252 / len(returns_array)) - 1
        else:
            annualized_return = 0.0
        
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "annualized_return": annualized_return,
            "final_portfolio_value": equity_array[-1]
        }

