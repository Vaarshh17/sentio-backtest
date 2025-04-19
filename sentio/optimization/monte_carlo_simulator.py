# sentio/optimization/monte_carlo_simulator.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
import matplotlib.pyplot as plt
from scipy import stats

class MonteCarloSimulator:
    """
    Performs Monte Carlo simulations for robustness testing.
    
    This class generates probability distributions of outcomes by simulating
    different market scenarios based on historical trade data.
    """
    
    def __init__(
        self, 
        trades: pd.DataFrame,
        initial_capital: float = 100000.0
    ):
        """
        Initialize the Monte Carlo Simulator.
        
        Args:
            trades: DataFrame of historical trades
            initial_capital: Initial portfolio capital
        """
        self.trades = trades
        self.initial_capital = initial_capital
        self.results = {}
        
    def run_simulation(
        self,
        num_simulations: int = 1000,
        random_seed: int = 42
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation.
        
        Args:
            num_simulations: Number of simulations to run
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary of simulation results
        """
        if self.trades.empty:
            return {
                'success': False,
                'error': 'No trade data available for simulation'
            }
            
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        # Extract trade returns
        if 'pnl' in self.trades.columns:
            trade_returns = self.trades['pnl'].values / self.initial_capital
        else:
            # Calculate P&L from quantity, price, and commission
            self.trades['pnl'] = self.trades['quantity'] * self.trades['price'] - self.trades['commission']
            trade_returns = self.trades['pnl'].values / self.initial_capital
        
        # Run simulations
        simulation_results = []
        
        for sim in range(num_simulations):
            # Resample trade returns with replacement (bootstrap)
            resampled_returns = np.random.choice(trade_returns, size=len(trade_returns), replace=True)
            
            # Calculate cumulative returns
            cumulative_returns = np.cumprod(1 + resampled_returns) - 1
            
            # Calculate final return
            final_return = cumulative_returns[-1]
            
            # Calculate max drawdown
            peak = np.maximum.accumulate(1 + cumulative_returns)
            drawdown = (1 + cumulative_returns) / peak - 1
            max_drawdown = drawdown.min()
            
            simulation_results.append({
                'sim_id': sim,
                'final_return': final_return,
                'max_drawdown': max_drawdown,
                'cumulative_returns': cumulative_returns
            })
            
        # Compile results
        sim_df = pd.DataFrame(simulation_results)
        
        # Calculate statistics
        final_returns = sim_df['final_return']
        max_drawdowns = sim_df['max_drawdown']
        
        percentiles = [5, 25, 50, 75, 95]
        return_percentiles = np.percentile(final_returns, percentiles)
        drawdown_percentiles = np.percentile(max_drawdowns, percentiles)
        
        self.results = {
            'success': True,
            'num_simulations': num_simulations,
            'return_mean': final_returns.mean(),
            'return_std': final_returns.std(),
            'return_percentiles': dict(zip(percentiles, return_percentiles)),
            'drawdown_mean': max_drawdowns.mean(),
            'drawdown_std': max_drawdowns.std(),
            'drawdown_percentiles': dict(zip(percentiles, drawdown_percentiles)),
            'simulations': sim_df
        }
        
        return self.results
    
    def plot_simulation_paths(
        self,
        num_paths: int = 100,
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot Monte Carlo simulation paths.
        
        Args:
            num_paths: Number of paths to plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not self.results or 'simulations' not in self.results:
            raise ValueError("Run simulation before plotting paths")
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get random subset of simulations
        sim_df = self.results['simulations']
        
        if len(sim_df) <= num_paths:
            paths = sim_df
        else:
            paths = sim_df.sample(num_paths)
            
        # Plot each path
        for idx, row in paths.iterrows():
            ax.plot(row['cumulative_returns'], alpha=0.1, color='blue')
            
        # Plot median path
        median_idx = sim_df['final_return'].astype(float).argsort()[len(sim_df) // 2]
        median_path = sim_df.iloc[median_idx]['cumulative_returns']
        ax.plot(median_path, color='red', linewidth=2, label='Median Path')
        
        # Plot 5th and 95th percentile paths
        percentile_5_idx = sim_df['final_return'].astype(float).argsort()[int(len(sim_df) * 0.05)]
        percentile_5_path = sim_df.iloc[percentile_5_idx]['cumulative_returns']
        ax.plot(percentile_5_path, color='orange', linewidth=1.5, label='5th Percentile')
        
        percentile_95_idx = sim_df['final_return'].astype(float).argsort()[int(len(sim_df) * 0.95)]
        percentile_95_path = sim_df.iloc[percentile_95_idx]['cumulative_returns']
        ax.plot(percentile_95_path, color='green', linewidth=1.5, label='95th Percentile')
        
        ax.set_title('Monte Carlo Simulation Paths')
        ax.set_xlabel('Trade')
        ax.set_ylabel('Cumulative Return')
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def plot_distribution(self, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot distribution of final returns and max drawdowns.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not self.results or 'simulations' not in self.results:
            raise ValueError("Run simulation before plotting distribution")
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Get return and drawdown data
        sim_df = self.results['simulations']
        
        # Plot return distribution
        ax1.hist(sim_df['final_return'], bins=50, alpha=0.7, color='blue')
        ax1.axvline(x=sim_df['final_return'].mean(), color='red', linestyle='-', linewidth=2, label=f'Mean: {sim_df["final_return"].mean():.2%}')
        ax1.axvline(x=np.percentile(sim_df['final_return'], 5), color='orange', linestyle='--', linewidth=1.5, label=f'5th Percentile: {np.percentile(sim_df["final_return"], 5):.2%}')
        
        ax1.set_title('Distribution of Final Returns')
        ax1.set_xlabel('Return')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        
        # Plot drawdown distribution
        ax2.hist(sim_df['max_drawdown'], bins=50, alpha=0.7, color='red')
        ax2.axvline(x=sim_df['max_drawdown'].mean(), color='blue', linestyle='-', linewidth=2, label=f'Mean: {sim_df["max_drawdown"].mean():.2%}')
        ax2.axvline(x=np.percentile(sim_df['max_drawdown'], 95), color='orange', linestyle='--', linewidth=1.5, label=f'95th Percentile: {np.percentile(sim_df["max_drawdown"], 95):.2%}')
        
        ax2.set_title('Distribution of Max Drawdowns')
        ax2.set_xlabel('Drawdown')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        plt.tight_layout()
        return fig