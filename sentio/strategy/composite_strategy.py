# sentio/strategy/composite_strategy.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging

from sentio.strategy.strategy import Strategy

class CompositeStrategy(Strategy):
    """
    Strategy that combines multiple sub-strategies.
    
    This class allows for combining multiple strategies with different weights,
    enabling strategy diversification and ensemble approaches.
    """
    
    def __init__(
        self,
        name: str,
        strategies: List[Strategy],
        weights: Optional[List[float]] = None,
        parameters: Dict[str, Any] = None
    ):
        """
        Initialize the CompositeStrategy.
        
        Args:
            name: Strategy identifier
            strategies: List of strategy instances to combine
            weights: List of weights for each strategy (must sum to 1.0)
            parameters: Dictionary of strategy parameters
        """
        super().__init__(name, parameters or {})
        
        self.strategies = strategies
        
        if weights is None:
            # Equal weights by default
            self.weights = [1.0 / len(strategies)] * len(strategies)
        else:
            if len(weights) != len(strategies):
                raise ValueError("Number of weights must match number of strategies")
            
            if abs(sum(weights) - 1.0) > 1e-6:
                raise ValueError("Weights must sum to 1.0")
                
            self.weights = weights
            
        self.is_trained = False
    
    def train(self, data: pd.DataFrame) -> None:
        """
        Train all sub-strategies on historical data.
        
        Args:
            data: DataFrame with historical market data
        """
        for i, strategy in enumerate(self.strategies):
            try:
                strategy.train(data)
                logging.info(f"Successfully trained sub-strategy {i}: {strategy.name}")
            except Exception as e:
                logging.error(f"Failed to train sub-strategy {i}: {strategy.name}. Error: {str(e)}")
                raise
                
        self.is_trained = all(strategy.is_trained for strategy in self.strategies)
        logging.info(f"CompositeStrategy training {'completed' if self.is_trained else 'failed'}")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals by combining signals from all sub-strategies.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Series with trading signals (1=buy, -1=sell, 0=hold)
        """
        if not self.is_trained:
            raise ValueError("Strategy must be trained before generating signals")
            
        # Generate signals from each sub-strategy
        all_signals = []
        
        for strategy in self.strategies:
            try:
                signals = strategy.generate_signals(data)
                all_signals.append(signals)
            except Exception as e:
                logging.error(f"Failed to generate signals for {strategy.name}. Error: {str(e)}")
                # Use neutral signals if a strategy fails
                signals = pd.Series(0, index=data.index)
                all_signals.append(signals)
                
        # Combine signals using weights
        combined_signals = pd.Series(0.0, index=data.index)
        
        for i, signals in enumerate(all_signals):
            # Align the index with combined_signals
            weight = self.weights[i]
            signals = signals.reindex(combined_signals.index, fill_value=0)
            combined_signals += signals * weight
            
        # Threshold combined signal to get final trading decisions
        thresholded_signals = pd.Series(0, index=combined_signals.index)
        thresholded_signals[combined_signals > 0.2] = 1
        thresholded_signals[combined_signals < -0.2] = -1
        
        return thresholded_signals
    
    def get_required_data(self) -> List[str]:
        """
        Get the combined list of data fields required by all sub-strategies.
        
        Returns:
            List of required data field names
        """
        required_fields = set()
        
        for strategy in self.strategies:
            required_fields.update(strategy.get_required_data())
            
        return list(required_fields)
    
    def update_strategy_weights(self, weights: List[float]) -> None:
        """
        Update the weights of sub-strategies.
        
        Args:
            weights: New weights for each strategy (must sum to 1.0)
        """
        if len(weights) != len(self.strategies):
            raise ValueError("Number of weights must match number of strategies")
            
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
            
        self.weights = weights
        logging.info(f"Updated strategy weights: {weights}")