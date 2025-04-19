# sentio/strategy/strategy.py
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

class Strategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    This class defines the interface that all strategies must implement.
    It handles the strategy's parameters, internal state, and signal generation.
    """
    
    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        """
        Initialize a new Strategy.
        
        Args:
            name: A unique identifier for the strategy.
            parameters: Dictionary of parameters to configure the strategy.
        """
        self.name = name
        self.parameters = parameters or {}
        self.is_trained = False
        self._last_signal = 0  # 0: no position, 1: long, -1: short
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on the input data.
        
        Args:
            data: DataFrame containing market data and indicators.
            
        Returns:
            Series containing trading signals (1: buy, -1: sell, 0: hold)
        """
        pass
    
    @abstractmethod
    def train(self, data: pd.DataFrame) -> None:
        """
        Train the strategy on historical data.
        
        Args:
            data: DataFrame containing training data.
        """
        pass
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Update the strategy parameters.
        
        Args:
            parameters: Dictionary of parameters to update.
        """
        self.parameters.update(parameters)
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the current strategy parameters.
        
        Returns:
            Dictionary of current parameters.
        """
        return self.parameters
    
    def get_required_data(self) -> List[str]:
        """
        Get the list of data fields required by this strategy.
        
        Returns:
            List of required data field names.
        """
        return []