# simple_test.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import dummy modules
from dummy_modules import DummyDataManager, DummyDataSource 

# Import your actual strategy implementations
from sentio.strategy.hmm_strategy import HMMStrategy
from sentio.strategy.composite_strategy import CompositeStrategy

def generate_test_data(num_days=100):
    """Generate test data for the strategy."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=num_days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create comprehensive dummy dataset
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

def test_hmm_strategy():
    """Test the HMM strategy with dummy data."""
    print("Testing HMM Strategy...")
    
    # Generate test data
    data = generate_test_data(100)
    
    # Create HMM strategy
    hmm = HMMStrategy(
        name="test_hmm",
        parameters={
            'n_states': 2,  # Simplified for testing
            'flow_lookback': 5,  # Shorter for testing
            'signal_threshold': 0.3
        }
    )
    
    try:
        # Train strategy
        print("Training strategy...")
        hmm.train(data)
        
        # Generate signals
        print("Generating signals...")
        signals = hmm.generate_signals(data)
        
        # Check results
        print(f"Generated {len(signals)} signals")
        print(f"Signal distribution: {signals.value_counts()}")
        
        return True
    except Exception as e:
        print(f"Error testing HMM strategy: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run simple tests of the strategy components."""
    print("Running simplified strategy tests...")
    
    hmm_success = test_hmm_strategy()
    
    if hmm_success:
        print("HMM strategy test successful!")
    else:
        print("HMM strategy test failed.")

if __name__ == "__main__":
    main()