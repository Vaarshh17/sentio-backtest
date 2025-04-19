# sentio/execution/portfolio.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

class Portfolio:
    """
    Tracks positions, cash, and overall account value.
    
    This class manages position sizing, risk allocation, and calculates
    P&L for all positions in the portfolio.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        leverage: float = 1.0
    ):
        """
        Initialize the Portfolio.
        
        Args:
            initial_capital: Starting capital amount
            leverage: Maximum leverage allowed (1.0 = no leverage)
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.leverage = leverage
        
        self.positions = {}  # symbol -> Position object
        self.transactions = []
        self.cash = initial_capital
        
        self.equity_curve = []  # [(timestamp, equity), ...]
        
    def reset(self) -> None:
        """Reset the portfolio to initial state."""
        self.current_capital = self.initial_capital
        self.positions = {}
        self.transactions = []
        self.cash = self.initial_capital
        self.equity_curve = []
        
    def update_position(
        self,
        timestamp: datetime,
        symbol: str,
        quantity: float,
        price: float,
        commission: float
    ) -> Dict[str, Any]:
        """
        Update a position based on a trade.
        
        Args:
            timestamp: Time of the trade
            symbol: Asset symbol
            quantity: Quantity to add (positive) or remove (negative)
            price: Execution price
            commission: Commission amount
            
        Returns:
            Updated position information
        """
        # Calculate cost of trade
        cost = quantity * price
        total_cost = cost + commission
        
        # Update cash
        self.cash -= total_cost
        
        # Update position
        if symbol not in self.positions:
            self.positions[symbol] = {
                'quantity': 0,
                'avg_price': 0,
                'cost_basis': 0
            }
            
        position = self.positions[symbol]
        
        if position['quantity'] == 0:
            # New position
            position['quantity'] = quantity
            position['avg_price'] = price
            position['cost_basis'] = cost
        else:
            # Existing position
            if (position['quantity'] > 0 and quantity > 0) or (position['quantity'] < 0 and quantity < 0):
                # Adding to position
                total_quantity = position['quantity'] + quantity
                position['cost_basis'] += cost
                position['avg_price'] = position['cost_basis'] / total_quantity
                position['quantity'] = total_quantity
            else:
                # Reducing or flipping position
                if abs(quantity) >= abs(position['quantity']):
                    # Closing position entirely or flipping
                    remaining_quantity = quantity + position['quantity']
                    
                    if remaining_quantity != 0:
                        # Flipping to opposite direction
                        position['quantity'] = remaining_quantity
                        position['avg_price'] = price
                        position['cost_basis'] = remaining_quantity * price
                    else:
                        # Closed position
                        position['quantity'] = 0
                        position['avg_price'] = 0
                        position['cost_basis'] = 0
                else:
                    # Partially reducing position
                    position['quantity'] += quantity
                    # Keep avg_price and adjust cost_basis
                    position['cost_basis'] = position['avg_price'] * position['quantity']
                    
        # Record transaction
        transaction = {
            'timestamp': timestamp,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'commission': commission,
            'value': cost,
            'total_cost': total_cost,
            'remaining_cash': self.cash
        }
        
        self.transactions.append(transaction)
        
        # Update equity curve
        self._update_equity_curve(timestamp)
        
        return position
    
    def _update_equity_curve(self, timestamp: datetime) -> None:
        """
        Update the equity curve with current portfolio value.
        
        Args:
            timestamp: Current timestamp
        """
        equity = self.get_equity()
        self.equity_curve.append((timestamp, equity))
        
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific position.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Position information or None if position doesn't exist
        """
        return self.positions.get(symbol, None)
    
    def get_equity(self) -> float:
        """
        Calculate total portfolio equity.
        
        Returns:
            Total equity value
        """
        # Sum up all position values (mark-to-market)
        position_value = 0
        for symbol, position in self.positions.items():
            if position['quantity'] != 0:
                # In a real system, we would get the current market price
                # but for simplicity, we use the average price here
                price = position['avg_price']
                position_value += position['quantity'] * price
                
        return self.cash + position_value
    
    def get_margin_used(self) -> float:
        """
        Calculate margin currently in use.
        
        Returns:
            Amount of margin used
        """
        margin = 0
        for symbol, position in self.positions.items():
            if position['quantity'] != 0:
                price = position['avg_price']
                margin += abs(position['quantity'] * price)
                
        return margin
    
    def get_margin_available(self) -> float:
        """
        Calculate available margin.
        
        Returns:
            Amount of margin available for new positions
        """
        # Maximum margin is initial capital times leverage
        max_margin = self.initial_capital * self.leverage
        
        # Available margin is max minus used
        return max_margin - self.get_margin_used()
    
    def get_equity_curve(self) -> pd.DataFrame:
        """
        Get the equity curve as a DataFrame.
        
        Returns:
            DataFrame with equity curve data
        """
        if not self.equity_curve:
            return pd.DataFrame(columns=['timestamp', 'equity'])
            
        df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity'])
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def get_transactions(self) -> pd.DataFrame:
        """
        Get all transactions as a DataFrame.
        
        Returns:
            DataFrame with transaction data
        """
        if not self.transactions:
            return pd.DataFrame(columns=[
                'timestamp', 'symbol', 'quantity', 'price',
                'commission', 'value', 'total_cost', 'remaining_cash'
            ])
            
        return pd.DataFrame(self.transactions)