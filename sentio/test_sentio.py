#!/usr/bin/env python
"""
Basic test script for the Sentio backtesting framework.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import Sentio components
from sentio.core.backtest_engine import BacktestEngine
#from sentio.data.crypto_quant_source import CryptoQuantSource
from dummy_modules import DummyDataSource as CryptoQuantSource
from sentio.strategy.hmm_strategy import HMMStrategy
from sentio.strategy.emotional_analysis_strategy import EmotionalAnalysisStrategy
from sentio.strategy.composite_strategy import CompositeStrategy

def generate_synthetic_data(start_date, end_date, interval='1h'):
    """Generate synthetic data for testing."""
    # Parse dates
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Create date range
    if interval == '1h':
        dates = pd.date_range(start=start, end=end, freq='H')
    else:
        dates = pd.date_range(start=start, end=end, freq='D')
    
    # Generate random data
    np.random.seed(42)  # for reproducibility
    price = 10000 + np.random.randn(len(dates)).cumsum() * 100
    volume = np.abs(100000 + np.random.randn(len(dates)) * 20000)
    
    # Generate exchange flow data
    inflow = np.abs(np.random.randn(len(dates)) * 1000)
    outflow = np.abs(np.random.randn(len(dates)) * 1000)
    net_flow = inflow - outflow
    
    # Generate order book data
    bid_depth = np.abs(500000 + np.random.randn(len(dates)) * 100000)
    ask_depth = np.abs(500000 + np.random.randn(len(dates)) * 100000)
    
    # Generate sentiment data
    sentiment = np.clip(np.random.randn(len(dates)).cumsum() * 0.1, -1, 1)
    fear_greed = np.clip(50 + np.random.randn(len(dates)).cumsum(), 0, 100)
    volatility = np.abs(np.random.randn(len(dates)) * 0.02)
    
    # Create DataFrame
    df = pd.DataFrame({
        'price_close': price,
        'volume': volume,
        'exchange_inflow': inflow,
        'exchange_outflow': outflow,
        'net_flow': net_flow,
        'order_book_depth_bid': bid_depth,
        'order_book_depth_ask': ask_depth,
        'social_sentiment': sentiment,
        'fear_greed_index': fear_greed,
        'market_volatility': volatility
    }, index=dates)
    
    return df

class MockCryptoQuantSource(CryptoQuantSource):
    """Mock CryptoQuant data source for testing."""
    
    def __init__(self, api_key, asset, interval):
        super().__init__(api_key, asset, interval)
        self.synthetic_data = None
    
    def fetch_data(self, start_date, end_date, metrics):
        """Fetch synthetic data instead of calling the API."""
        if self.synthetic_data is None:
            self.synthetic_data = generate_synthetic_data(start_date, end_date, self.interval)
        
        # Return only requested metrics
        return self.synthetic_data[metrics]

def test_imports():
    """Test that all components can be imported and instantiated."""
    engine = BacktestEngine()
    hmm = HMMStrategy(name="test_hmm")
    emotional = EmotionalAnalysisStrategy(name="test_emotional")
    composite = CompositeStrategy(
        name="test_composite",
        strategies=[hmm, emotional],
        weights=[0.5, 0.5]
    )
    
    print("All imports successful!")
    print(f"Engine initialized: {engine is not None}")
    print(f"HMM Strategy initialized: {hmm is not None}")
    print(f"Emotional Strategy initialized: {emotional is not None}")
    print(f"Composite Strategy initialized: {composite is not None}")
    
    return all([engine, hmm, emotional, composite])

def test_basic_backtest():
    """Test a basic backtest using synthetic data."""
    print("\nTesting basic backtest with synthetic data...")
    
    # Initialize backtest engine
    engine = BacktestEngine()
    
    # Configure backtest environment
    engine.configure_environment({
        'initial_capital': 100000,
        'commission_rate': 0.0006,
        'slippage': 0.0002
    })
    
    # Set up mock data source
    mock_source = MockCryptoQuantSource(api_key='test_key', asset='btc', interval='1d')
    engine.add_data_source('crypto_quant', mock_source)
    
    # Set up a simple HMM strategy
    hmm_strategy = HMMStrategy(
        name='test_hmm',
        parameters={
            'n_states': 2,  # Simplified for testing
            'flow_lookback': 5,  # Shorter for testing
            'signal_threshold': 0.3
        }
    )
    
    # Set the strategy
    engine.set_strategy(hmm_strategy)
    
    # Define date range (short period for testing)
    start_date = '2022-01-01'
    end_date = '2022-01-31'
    
    # Define data metrics
    sources_and_metrics = {
        'crypto_quant': [
            'exchange_inflow',
            'exchange_outflow',
            'net_flow',
            'order_book_depth_bid',
            'order_book_depth_ask',
            'price_close',
            'volume',
            'social_sentiment',
            'fear_greed_index',
            'market_volatility'
        ]
    }
    
    try:
        # Run backtest
        results = engine.run_backtest(
            start_date=start_date,
            end_date=end_date,
            sources_and_metrics=sources_and_metrics,
            train_test_split=0.5  # 50/50 split for testing
        )
        
        # Check if results contain key metrics
        assert 'performance' in results, "Missing performance metrics"
        assert 'risk_profile' in results, "Missing risk profile"
        
        # Display some results
        perf = results.get('performance', {})
        print(f"Total Return: {perf.get('total_return', 0):.2%}")
        print(f"Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown: {perf.get('max_drawdown', 0):.2%}")
        
        print("Basic backtest test successful!")
        return True
        
    except Exception as e:
        print(f"Backtest failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Sentio Backtesting Framework...")
    imports_ok = test_imports()
    
    if imports_ok:
        backtest_ok = test_basic_backtest()
        
        if imports_ok and backtest_ok:
            print("\nAll tests passed successfully!")
        else:
            print("\nSome tests failed. Please check the output above.")
    else:
        print("Import tests failed. Please check your implementation.")