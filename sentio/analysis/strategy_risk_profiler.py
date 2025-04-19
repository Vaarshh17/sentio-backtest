# sentio/analysis/performance_analyzer.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import stats

from sentio.execution.portfolio import Portfolio

class PerformanceAnalyzer:
    """
    Calculates performance metrics and statistics for strategy evaluation.
    
    This class analyzes the performance of trading strategies, calculating
    key metrics like returns, drawdowns, Sharpe ratio, and other risk-adjusted
    performance indicators.
    """
    
    def __init__(
        self,
        portfolio: Portfolio,
        benchmark_data: pd.Series = None,
        risk_free_rate: float = 0.02,  # 2% annual risk-free rate
        trading_days_per_year: int = 252
    ):
        """
        Initialize the PerformanceAnalyzer.
        
        Args:
            portfolio: Portfolio to analyze
            benchmark_data: Optional benchmark price data for comparison
            risk_free_rate: Annual risk-free rate
            trading_days_per_year: Number of trading days in a year
        """
        self.portfolio = portfolio
        self.benchmark_data = benchmark_data
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year
        self.daily_risk_free_rate = (1 + risk_free_rate) ** (1 / trading_days_per_year) - 1
        
        self.metrics = {}
        
    def calculate_metrics(self, current_timestamp: datetime) -> Dict[str, Any]:
        """
        Calculate all performance metrics.
        
        Args:
            current_timestamp: Current timestamp for analysis endpoint
            
        Returns:
            Dictionary of performance metrics
        """
        # Get equity curve data
        equity_df = self.portfolio.get_equity_curve()
        
        if equity_df.empty:
            return {
                'total_return': 0,
                'annualized_return': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'max_drawdown': 0,
                'calmar_ratio': 0,
                'volatility': 0
            }
            
        # Convert equity to returns
        equity_values = equity_df['equity']
        returns = equity_values.pct_change().dropna()
        
        # Calculate benchmark returns if available
        benchmark_returns = None
        if self.benchmark_data is not None:
            # Align benchmark data with equity curve
            aligned_benchmark = self.benchmark_data.reindex(equity_df.index, method='ffill')
            benchmark_returns = aligned_benchmark.pct_change().dropna()
            
        # Calculate core metrics
        total_return = (equity_values.iloc[-1] / equity_values.iloc[0]) - 1
        
        # Time period in years
        start_date = equity_df.index[0]
        end_date = equity_df.index[-1]
        
        days = (end_date - start_date).days
        years = days / 365.25
        
        if years > 0:
            annualized_return = (1 + total_return) ** (1 / years) - 1
        else:
            annualized_return = total_return
            
        # Volatility
        daily_volatility = returns.std()
        annualized_volatility = daily_volatility * np.sqrt(self.trading_days_per_year)
        
        # Calculate Sharpe ratio
        excess_returns = returns - self.daily_risk_free_rate
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(self.trading_days_per_year) if excess_returns.std() > 0 else 0
        
        # Calculate Sortino ratio (downside deviation)
        negative_returns = returns[returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(self.trading_days_per_year) if len(negative_returns) > 0 else 0
        sortino_ratio = (returns.mean() - self.daily_risk_free_rate) / downside_deviation * np.sqrt(self.trading_days_per_year) if downside_deviation > 0 else 0
        
        # Calculate maximum drawdown
        peak = equity_values.expanding().max()
        drawdown = (equity_values / peak) - 1
        max_drawdown = drawdown.min()
        
        # Calculate Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Store metrics
        self.metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'trading_days': len(returns),
            'winning_days': (returns > 0).sum(),
            'losing_days': (returns < 0).sum(),
        }
        
        # Add benchmark comparison if available
        if benchmark_returns is not None and not benchmark_returns.empty:
            benchmark_total_return = (aligned_benchmark.iloc[-1] / aligned_benchmark.iloc[0]) - 1
            
            if years > 0:
                benchmark_annual_return = (1 + benchmark_total_return) ** (1 / years) - 1
            else:
                benchmark_annual_return = benchmark_total_return
                
            benchmark_volatility = benchmark_returns.std() * np.sqrt(self.trading_days_per_year)
            
            # Calculate alpha and beta
            beta, alpha, r_value, p_value, std_err = stats.linregress(benchmark_returns, returns)
            
            # Calculate Information Ratio
            tracking_error = (returns - benchmark_returns).std() * np.sqrt(self.trading_days_per_year)
            information_ratio = (annualized_return - benchmark_annual_return) / tracking_error if tracking_error > 0 else 0
            
            self.metrics.update({
                'benchmark_return': benchmark_total_return,
                'benchmark_annual_return': benchmark_annual_return,
                'alpha': alpha * self.trading_days_per_year,
                'beta': beta,
                'r_squared': r_value ** 2,
                'information_ratio': information_ratio,
                'tracking_error': tracking_error
            })
            
        return self.metrics
    
    def calculate_drawdowns(self) -> pd.DataFrame:
        """
        Calculate all drawdown periods.
        
        Returns:
            DataFrame with drawdown periods and depths
        """
        equity_df = self.portfolio.get_equity_curve()
        
        if equity_df.empty:
            return pd.DataFrame(columns=['start_date', 'end_date', 'depth', 'recovery_days'])
            
        equity_values = equity_df['equity']
        
        # Calculate drawdown series
        peak = equity_values.expanding().max()
        drawdown = (equity_values / peak) - 1
        
        # Find drawdown periods
        in_drawdown = False
        drawdown_periods = []
        
        start_date = None
        start_value = None
        max_depth = 0
        
        for date, value in drawdown.items():
            if not in_drawdown and value < 0:
                # Start of drawdown
                in_drawdown = True
                start_date = date
                start_value = equity_values[date]
                max_depth = value
            elif in_drawdown:
                if value < max_depth:
                    # Drawdown getting deeper
                    max_depth = value
                elif value == 0:
                    # End of drawdown
                    end_date = date
                    end_value = equity_values[date]
                    duration = (end_date - start_date).days
                    recovery_days = duration
                    
                    drawdown_periods.append({
                        'start_date': start_date,
                        'end_date': end_date,
                        'depth': max_depth,
                        'duration_days': duration,
                        'recovery_days': recovery_days,
                        'start_value': start_value,
                        'end_value': end_value
                    })
                    
                    in_drawdown = False
                    
        # If still in drawdown at end of data
        if in_drawdown:
            end_date = drawdown.index[-1]
            end_value = equity_values[-1]
            duration = (end_date - start_date).days
            
            drawdown_periods.append({
                'start_date': start_date,
                'end_date': end_date,
                'depth': max_depth,
                'duration_days': duration,
                'recovery_days': None,
                'start_value': start_value,
                'end_value': end_value
            })
            
        return pd.DataFrame(drawdown_periods)
    
    def calculate_monthly_returns(self) -> pd.DataFrame:
        """
        Calculate monthly returns.
        
        Returns:
            DataFrame with monthly returns
        """
        equity_df = self.portfolio.get_equity_curve()
        
        if equity_df.empty:
            return pd.DataFrame(columns=['year', 'month', 'return'])
            
        # Resample to month-end
        monthly_equity = equity_df['equity'].resample('M').last()
        
        # Calculate returns
        monthly_returns = monthly_equity.pct_change().dropna()
        
        # Format into year/month table
        result = pd.DataFrame({
            'return': monthly_returns.values
        }, index=monthly_returns.index)
        
        result['year'] = result.index.year
        result['month'] = result.index.month
        
        return result
    
    def calculate_trade_statistics(self) -> Dict[str, Any]:
        """
        Calculate trade statistics.
        
        Returns:
            Dictionary of trade statistics
        """
        trades_df = self.portfolio.get_transactions()
        
        if trades_df.empty:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_trade': 0,
                'avg_winner': 0,
                'avg_loser': 0,
                'largest_winner': 0,
                'largest_loser': 0
            }
            
        # Calculate trade P&L
        trades_df['pnl'] = trades_df['value']
        
        # Separate winning and losing trades
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        # Calculate statistics
        total_trades = len(trades_df)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        total_profit = winning_trades['pnl'].sum() if not winning_trades.empty else 0
        total_loss = abs(losing_trades['pnl'].sum()) if not losing_trades.empty else 0
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        avg_trade = trades_df['pnl'].mean() if not trades_df.empty else 0
        avg_winner = winning_trades['pnl'].mean() if not winning_trades.empty else 0
        avg_loser = losing_trades['pnl'].mean() if not losing_trades.empty else 0
        
        largest_winner = winning_trades['pnl'].max() if not winning_trades.empty else 0
        largest_loser = losing_trades['pnl'].min() if not losing_trades.empty else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade': avg_trade,
            'avg_winner': avg_winner,
            'avg_loser': avg_loser,
            'largest_winner': largest_winner,
            'largest_loser': largest_loser
        }
    
    def plot_equity_curve(self, benchmark: bool = True, figsize=(12, 6)) -> plt.Figure:
        """
        Plot equity curve and benchmark if available.
        
        Args:
            benchmark: Whether to include benchmark in plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        equity_df = self.portfolio.get_equity_curve()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot strategy equity curve
        ax.plot(equity_df.index, equity_df['equity'], label='Strategy', linewidth=2)
        
        # Plot benchmark if available and requested
        if benchmark and self.benchmark_data is not None:
            # Align benchmark data with equity curve
            aligned_benchmark = self.benchmark_data.reindex(equity_df.index, method='ffill')
            
            # Normalize benchmark to same starting value
            normalized_benchmark = aligned_benchmark / aligned_benchmark.iloc[0] * equity_df['equity'].iloc[0]
            
            ax.plot(equity_df.index, normalized_benchmark, label='Benchmark', linewidth=1, alpha=0.7)
            
        ax.set_title('Equity Curve')
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity')
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def plot_drawdowns(self, figsize=(12, 6)) -> plt.Figure:
        """
        Plot drawdowns over time.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        equity_df = self.portfolio.get_equity_curve()
        
        if equity_df.empty:
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_title('Drawdowns (No Data)')
            return fig
            
        equity_values = equity_df['equity']
        
        # Calculate drawdown series
        peak = equity_values.expanding().max()
        drawdown = (equity_values / peak) - 1
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax.set_title('Drawdowns')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown')
        ax.grid(True)
        
        return fig