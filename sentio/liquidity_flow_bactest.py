#!/usr/bin/env python
"""
Liquidity Flow Hidden Regime Strategy Backtest

This script demonstrates the Sentio backtesting framework with our
Liquidity Flow Hidden Regime strategy implementation. It shows how to
load data, configure strategies, run backtests, and analyze results.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

def main():
    """
    Main backtesting function to demonstrate the Liquidity Flow Hidden Regime strategy.
    """
    print("=" * 80)
    print("Sentio: Liquidity Flow Hidden Regime Strategy Backtest")
    print("=" * 80)

    # Initialize backtest engine
    engine = BacktestEngine()

    # Configure backtest environment
    engine.configure_environment({
        'initial_capital': 1000000,  # 1 million
        'commission_rate': 0.0006,   # 0.06% commission
        'slippage': 0.0002,          # 0.02% slippage
        'leverage': 2.0,             # 2x leverage
        'position_size': 1.0,        # Full position size
    })

    # Set up data sources
    # In a real implementation, you would use your actual API key
    crypto_quant = CryptoQuantSource(api_key='Qb4fOizGZQo81YYQaRmQhnpq5KzJG3uE9Q9Hfbp4CoEAjKuc', asset='btc', interval='1h')
    engine.add_data_source('crypto_quant', crypto_quant)

    # Define date range for backtest
    start_date = '2020-01-01'
    end_date = '2023-12-31'  # This gives a 4-year backtest period

    # Set up strategies
    # 1. HMM Strategy for market regime detection
    hmm_strategy = HMMStrategy(
        name='liquidity_flow_hmm',
        parameters={
            'n_states': 4,
            'flow_lookback': 24,  # 24 hours lookback for liquidity flow
            'signal_threshold': 0.3  # Lower threshold for more signals
        }
    )

    # 2. Emotional Analysis Strategy for sentiment-based adjustments
    emotional_strategy = EmotionalAnalysisStrategy(
        name='emotional_analysis',
        parameters={
            'lstm_units': 64,
            'dropout_rate': 0.3,
            'sequence_length': 48,  # 2 days of hourly data
            'confidence_threshold': 0.4  # Lower threshold for more signals
        }
    )

    # 3. Combine the strategies with weights
    composite_strategy = CompositeStrategy(
        name='liquidity_flow_regime',
        strategies=[hmm_strategy, emotional_strategy],
        weights=[0.7, 0.3]  # More weight on the HMM strategy
    )

    # Set the strategy for backtesting
    engine.set_strategy(composite_strategy)

    # Define what data to use
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

    print(f"Running backtest from {start_date} to {end_date}...")
    
    # Run the backtest
    results = engine.run_backtest(
        start_date=start_date,
        end_date=end_date,
        sources_and_metrics=sources_and_metrics,
        train_test_split=0.3  # Use 30% of data for training, 70% for testing
    )

    # Display performance metrics
    print("\nPerformance Metrics:")
    print("-" * 50)
    
    perf = results.get('performance', {})
    print(f"Total Return: {perf.get('total_return', 0):.2%}")
    print(f"Annualized Return: {perf.get('annualized_return', 0):.2%}")
    print(f"Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
    print(f"Sortino Ratio: {perf.get('sortino_ratio', 0):.2f}")
    print(f"Max Drawdown: {perf.get('max_drawdown', 0):.2%}")
    print(f"Calmar Ratio: {perf.get('calmar_ratio', 0):.2f}")
    print(f"Volatility: {perf.get('volatility', 0):.2%}")
    
    if 'benchmark_return' in perf:
        print(f"Benchmark Return: {perf.get('benchmark_return', 0):.2%}")
        print(f"Alpha: {perf.get('alpha', 0):.2%}")
        print(f"Beta: {perf.get('beta', 0):.2f}")
        print(f"Information Ratio: {perf.get('information_ratio', 0):.2f}")

    # Display trade metrics
    trade_count = len(engine.trades) if engine.trades is not None else 0
    trade_frequency = trade_count / len(engine.signals) if engine.signals is not None and len(engine.signals) > 0 else 0
    
    print("\nTrade Metrics:")
    print("-" * 50)
    print(f"Total Trades: {trade_count}")
    print(f"Trade Frequency: {trade_frequency:.2%}")
    
    # Display risk profile summary
    print("\nRisk Profile Summary:")
    print("-" * 50)
    
    risk_profile = results.get('risk_profile', {})
    
    if 'behavioral_biases' in risk_profile:
        biases = risk_profile['behavioral_biases']
        print("Behavioral Biases:")
        for bias, score in biases.items():
            print(f"  {bias}: {score:.2f}")
    
    if 'regime_vulnerability' in risk_profile:
        regime_vuln = risk_profile['regime_vulnerability']
        print("\nRegime Vulnerability:")
        print(f"  Most Vulnerable Regime: {regime_vuln.get('most_vulnerable_regime', 'N/A')}")
        print(f"  Regime Stability: {regime_vuln.get('regime_stability', 0):.2f}")
    
    if 'tail_risk' in risk_profile:
        tail_risk = risk_profile['tail_risk']
        print("\nTail Risk Assessment:")
        print(f"  VaR (95%): {tail_risk.get('var_95', 0):.2%}")
        print(f"  CVaR (95%): {tail_risk.get('cvar_95', 0):.2%}")
        print(f"  Skewness: {tail_risk.get('skewness', 0):.2f}")
        print(f"  Kurtosis: {tail_risk.get('kurtosis', 0):.2f}")
    
    # Validate strategy against criteria
    validation = engine.validate_strategy_results()
    
    print("\nStrategy Validation:")
    print("-" * 50)
    print(f"Meets Sharpe Ratio Criterion (>= 1.8): {validation.get('meets_sharpe_criterion', False)}")
    print(f"Meets Drawdown Criterion (> -40%): {validation.get('meets_drawdown_criterion', False)}")
    print(f"Meets Trade Frequency Criterion (>= 3%): {validation.get('meets_frequency_criterion', False)}")
    print(f"Overall Valid: {validation.get('overall_valid', False)}")
    
    print("\nMonte Carlo Simulation Results:")
    print("-" * 50)
    
    if 'monte_carlo' in results:
        mc = results['monte_carlo']
        print(f"Mean Return: {mc.get('return_mean', 0):.2%}")
        print(f"Return Std Dev: {mc.get('return_std', 0):.2%}")
        print("Return Percentiles:")
        for pct, value in mc.get('return_percentiles', {}).items():
            print(f"  {pct}%: {value:.2%}")
        
        print(f"Mean Max Drawdown: {mc.get('drawdown_mean', 0):.2%}")
        print("Drawdown Percentiles:")
        for pct, value in mc.get('drawdown_percentiles', {}).items():
            print(f"  {pct}%: {value:.2%}")
    
    print("\nBacktest completed!")
    print("=" * 80)

if __name__ == "__main__":
    main()