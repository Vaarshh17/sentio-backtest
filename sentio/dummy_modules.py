# dummy_modules.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

# Base Strategy
class DummyStrategy(ABC):
    def __init__(self, name, parameters=None):
        self.name = name
        self.parameters = parameters or {}
        self.is_trained = False
    
    def generate_signals(self, data):
        # Return random signals
        return pd.Series(np.random.choice([-1, 0, 1], size=len(data)), index=data.index)
    
    def train(self, data):
        self.is_trained = True
        return True

# Data Source
class DummyDataSource:
    def fetch_data(self, start_date, end_date, metrics):
        # Generate dummy data
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start=start, end=end, freq='D')
        
        # Create dummy data for requested metrics
        data = {}
        for metric in metrics:
            data[metric] = np.random.randn(len(dates))
            
        return pd.DataFrame(data, index=dates)

# Data Manager
class DummyDataManager:
    def __init__(self):
        self.data_sources = {}
    
    def add_data_source(self, name, source):
        self.data_sources[name] = source
    
    def get_combined_data(self, sources_and_metrics, start_date, end_date, use_cache=True):
        # Generate dummy combined data
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start=start, end=end, freq='D')
        
        # Create comprehensive dummy dataset with all possible fields
        data = {
            'price_close': 10000 + np.random.randn(len(dates)).cumsum() * 100,
            'volume': np.abs(100000 + np.random.randn(len(dates)) * 20000),
            'exchange_inflow': np.abs(np.random.randn(len(dates)) * 1000),
            'exchange_outflow': np.abs(np.random.randn(len(dates)) * 1000),
            'net_flow': np.random.randn(len(dates)) * 500,
            'order_book_depth_bid': np.abs(500000 + np.random.randn(len(dates)) * 100000),
            'order_book_depth_ask': np.abs(500000 + np.random.randn(len(dates)) * 100000),
            'social_sentiment': np.clip(np.random.randn(len(dates)).cumsum() * 0.1, -1, 1),
            'fear_greed_index': np.clip(50 + np.random.randn(len(dates)).cumsum(), 0, 100),
            'market_volatility': np.abs(np.random.randn(len(dates)) * 0.02)
        }
        
        return pd.DataFrame(data, index=dates)

# Simplified version of TensorFlow-related classes
class DummyLSTM:
    def __init__(self, units, return_sequences=False, input_shape=None):
        self.units = units
        self.return_sequences = return_sequences
        self.input_shape = input_shape
    
    def __call__(self, inputs):
        return inputs

class DummyDropout:
    def __init__(self, rate):
        self.rate = rate
    
    def __call__(self, inputs):
        return inputs

class DummyDense:
    def __init__(self, units):
        self.units = units
    
    def __call__(self, inputs):
        return inputs

class DummySequential:
    def __init__(self):
        self.layers = []
    
    def add(self, layer):
        self.layers.append(layer)
    
    def compile(self, optimizer, loss):
        pass
    
    def fit(self, x, y, epochs=1, batch_size=32, validation_data=None, verbose=0):
        pass
    
    def predict(self, x, verbose=0):
        # Return random predictions
        return np.random.randn(x.shape[0], 10)

# Monte Carlo simulator
class DummyMonteCarloSimulator:
    def __init__(self, trades=None, initial_capital=100000.0):
        self.trades = trades
        self.initial_capital = initial_capital
    
    def run_simulation(self, num_simulations=1000, random_seed=42):
        return {
            'success': True,
            'num_simulations': num_simulations,
            'return_mean': 0.15,
            'return_std': 0.08,
            'return_percentiles': {5: 0.05, 25: 0.10, 50: 0.15, 75: 0.20, 95: 0.25},
            'drawdown_mean': -0.12,
            'drawdown_std': 0.05,
            'drawdown_percentiles': {5: -0.20, 25: -0.15, 50: -0.12, 75: -0.08, 95: -0.05},
        }