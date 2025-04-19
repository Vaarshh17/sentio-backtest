# sentio/execution/broker.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import random

from sentio.execution.portfolio import Portfolio

class Broker:
    """
    Simulates order execution in the market.
    
    This class handles the realistic simulation of order execution,
    including slippage, commissions, and other market frictions.
    """
    
    def __init__(
        self,
        commission_rate: float = 0.0006,  # 0.06% commission
        slippage: float = 0.0001,  # 0.01% slippage
        partial_fill_probability: float = 0.0,  # Disabled by default
        random_seed: int = 42
    ):
        """
        Initialize the Broker.
        
        Args:
            commission_rate: Commission rate as a decimal
            slippage: Average slippage as a decimal
            partial_fill_probability: Probability of partial fills
            random_seed: Random seed for reproducibility
        """
        self.commission_rate = commission_rate
        self.base_slippage = slippage
        self.partial_fill_probability = partial_fill_probability
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        self.random_state = np.random.RandomState(random_seed)
        
        # Order tracking
        self.orders = []
        self.order_id_counter = 1
        
    def execute_trade(
        self,
        portfolio: Portfolio,
        timestamp: datetime,
        symbol: str,
        quantity: float,
        market_price: float,
        volatility: float = None,
        order_type: str = 'market'
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a trade and update the portfolio.
        
        Args:
            portfolio: Portfolio to update
            timestamp: Time of execution
            symbol: Asset symbol
            quantity: Quantity to trade (positive for buy, negative for sell)
            market_price: Current market price
            volatility: Current market volatility (for slippage calculation)
            order_type: Type of order ('market', 'limit', etc.)
            
        Returns:
            Trade record or None if execution fails
        """
        if quantity == 0:
            return None
            
        # Calculate execution price with slippage
        execution_price = self._calculate_execution_price(
            market_price, quantity, volatility
        )
        
        # Calculate commission
        commission = abs(quantity * execution_price * self.commission_rate)
        
        # Apply partial fills if enabled
        filled_quantity = self._apply_partial_fills(quantity)
        
        if filled_quantity == 0:
            logging.info(f"Order for {symbol} was not filled")
            return None
            
        # Calculate trade value
        value = filled_quantity * execution_price
        
        # Check if portfolio has enough cash/margin for the trade
        if not self._check_trade_feasibility(portfolio, value, commission):
            logging.warning(f"Trade rejected: Insufficient funds/margin")
            return None
            
        # Update portfolio
        portfolio.update_position(
            timestamp,
            symbol,
            filled_quantity,
            execution_price,
            commission
        )
        
        # Record the order
        order_id = self.order_id_counter
        self.order_id_counter += 1
        
        order = {
            'order_id': order_id,
            'timestamp': timestamp,
            'symbol': symbol,
            'quantity': quantity,
            'filled_quantity': filled_quantity,
            'market_price': market_price,
            'execution_price': execution_price,
            'commission': commission,
            'order_type': order_type,
            'value': value,
            'status': 'filled' if filled_quantity == quantity else 'partial_fill'
        }
        
        self.orders.append(order)
        
        # Create a trade record
        trade = {
            'timestamp': timestamp,
            'symbol': symbol,
            'quantity': filled_quantity,
            'price': execution_price,
            'commission': commission,
            'value': value,
            'order_id': order_id
        }
        
        return trade
    
    def _calculate_execution_price(
        self,
        market_price: float,
        quantity: float,
        volatility: float = None
    ) -> float:
        """
        Calculate the execution price with slippage.
        
        Args:
            market_price: Current market price
            quantity: Trade quantity (positive for buy, negative for sell)
            volatility: Current market volatility
            
        Returns:
            Execution price with slippage
        """
        # Direction-based slippage (positive for buys, negative for sells)
        direction = 1 if quantity > 0 else -1
        
        # Base slippage
        slippage = self.base_slippage
        
        # Adjust slippage based on volatility if provided
        if volatility is not None:
            # Increase slippage proportionally to volatility
            volatility_adjustment = min(5.0, volatility / 0.01)  # Cap at 5x
            slippage *= volatility_adjustment
            
        # Adjust slippage based on quantity (market impact)
        # This is a simple linear model; in reality, market impact is nonlinear
        quantity_adjustment = 1.0 + (abs(quantity) / 10.0)  # Arbitrary scale
        slippage *= quantity_adjustment
        
        # Add some randomness to slippage
        random_factor = self.random_state.normal(1.0, 0.3)  # Mean 1.0, std 0.3
        slippage *= max(0.1, random_factor)  # Ensure minimal slippage
        
        # Calculate execution price
        execution_price = market_price * (1 + direction * slippage)
        
        return execution_price
    
    def _apply_partial_fills(self, quantity: float) -> float:
        """
        Apply partial fills if enabled.
        
        Args:
            quantity: Requested quantity
            
        Returns:
            Filled quantity
        """
        if self.partial_fill_probability <= 0:
            return quantity
            
        # Roll for partial fill
        if random.random() < self.partial_fill_probability:
            # If partial fill occurs, fill between 50-99% of the order
            fill_percentage = random.uniform(0.5, 0.99)
            filled_quantity = quantity * fill_percentage
            
            # Round to 8 decimal places for crypto
            filled_quantity = round(filled_quantity, 8)
            
            return filled_quantity
            
        return quantity
    
    def _check_trade_feasibility(
        self,
        portfolio: Portfolio,
        value: float,
        commission: float
    ) -> bool:
        """
        Check if the trade is feasible given portfolio constraints.
        
        Args:
            portfolio: Portfolio to check against
            value: Value of the trade
            commission: Commission amount
            
        Returns:
            True if trade is feasible, False otherwise
        """
        # Calculate total cost
        total_cost = value + commission
        
        # Check if enough cash for buy orders
        if value > 0 and portfolio.cash < total_cost:
            return False
            
        # Check if margin requirements are met for short sells
        if value < 0:
            # Short sell only needs to check margin availability
            margin_required = abs(value)
            if margin_required > portfolio.get_margin_available():
                return False
                
        return True
    
    def get_orders(self) -> pd.DataFrame:
        """
        Get all orders as a DataFrame.
        
        Returns:
            DataFrame with order data
        """
        if not self.orders:
            return pd.DataFrame(columns=[
                'order_id', 'timestamp', 'symbol', 'quantity',
                'filled_quantity', 'market_price', 'execution_price',
                'commission', 'order_type', 'status'
            ])
            
        return pd.DataFrame(self.orders)