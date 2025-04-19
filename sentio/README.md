# Sentio: Advanced Backtesting Library

## Overview

Sentio is a comprehensive, modular backtesting framework designed for quantitative trading strategies with a focus on realism, risk assessment, and adaptability. This library addresses critical limitations of existing backtesting tools by providing:

1. **Advanced Execution Modeling**: Realistic simulation of market mechanics including variable slippage, market impact, and partial fills.

2. **Strategy Risk Profiling**: Behavioral analysis to identify biases, regime vulnerabilities, and potential failure modes.

3. **Adaptive Strategy Framework**: Market regime detection using Hidden Markov Models to enable regime-aware strategy switching.

4. **Robust Validation Pipeline**: Comprehensive walk-forward testing and Monte Carlo simulation to prevent overfitting.

5. **Modular Architecture**: Extensible, plugin-based design that allows easy customization and integration.

## Liquidity Flow Hidden Regime Strategy

Our flagship strategy implements the Liquidity Flow Hidden Regime approach, which combines:

- **Hidden Markov Models** to detect market regimes
- **Liquidity flow analysis** from exchange data
- **Emotional analysis** through LSTM neural networks
- **Adaptive position sizing** based on detected regimes

### Strategy Hypothesis

Market liquidity flows through exchanges (measured by net inflows/outflows and order book depth) precede price movements by 12-24 hours, with different effectiveness depending on hidden market regimes that can be detected using HMMs.

**Formula:** `Liquidity Flow Signal = (Exchange Net Flow 24hr weighted) × (Regime Coefficient)`

Where the Regime Coefficient is determined by the current hidden state identified by the HMM.

## Installation

```bash
# Clone the repository
git clone https://github.com/sentio/sentio-backtesting.git
cd sentio-backtesting

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from sentio.core.backtest_engine import BacktestEngine
from sentio.data.crypto_quant_source import CryptoQuantSource
from sentio.strategy.hmm_strategy import HMMStrategy
from sentio.strategy.emotional_analysis_strategy import EmotionalAnalysisStrategy
from sentio.strategy.composite_strategy import CompositeStrategy

# Initialize backtest engine
engine = BacktestEngine()

# Configure backtest environment
engine.configure_environment({
    'initial_capital': 1000000,  # 1 million
    'commission_rate': 0.0006,   # 0.06% commission
    'slippage': 0.0002           # 0.02% slippage
})

# Set up data source
crypto_quant = CryptoQuantSource(api_key='your_api_key', asset='btc', interval='1h')
engine.add_data_source('crypto_quant', crypto_quant)

# Create strategy
hmm_strategy = HMMStrategy(
    name='liquidity_flow_hmm',
    parameters={
        'n_states': 4,
        'flow_lookback': 24,
        'signal_threshold': 0.3
    }
)

# Set strategy
engine.set_strategy(hmm_strategy)

# Run backtest
results = engine.run_backtest(
    start_date='2022-01-01',
    end_date='2022-12-31',
    sources_and_metrics={
        'crypto_quant': [
            'exchange_inflow',
            'exchange_outflow',
            'net_flow',
            'order_book_depth_bid',
            'order_book_depth_ask',
            'price_close',
            'volume'
        ]
    }
)

# Analyze results
print(f"Sharpe Ratio: {results['performance']['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['performance']['max_drawdown']:.2%}")
```

## Key Features

### 1. Advanced Execution Modeling

- Variable slippage models based on volatility and order size
- Market impact simulation
- Partial fill simulation with probability models
- Realistic commission and fee structures

### 2. Strategy Risk Profiling

- Behavioral bias detection (loss aversion, overconfidence, etc.)
- Market regime vulnerability analysis
- Risk decomposition (systematic vs. idiosyncratic)
- Drawdown characterization
- Tail risk assessment

### 3. Hidden Markov Model Regime Detection

- Unsupervised market state identification
- Forward-looking regime probabilities
- Strategy-regime mapping and performance analysis
- Dynamic strategy weighting based on regime

### 4. Emotional Analysis

- LSTM-based neural networks for sequence pattern recognition
- Analysis of temporal patterns in market data
- Detection of sentiment shifts through derived features
- Integration with fundamental trading decisions

### 5. Comprehensive Performance Analysis

- Standard metrics (Sharpe, Sortino, Calmar, etc.)
- Drawdown analysis and recovery metrics
- Trade statistics and distribution analysis
- Monte Carlo simulation for robustness testing

## Architecture

Sentio follows a modular, object-oriented architecture with clearly defined responsibilities:

- **Core Engine**: Central coordination of the backtesting process
- **Data Management**: Data loading, preprocessing, and feature engineering
- **Strategy Implementation**: Trading logic and signal generation
- **Execution & Portfolio**: Realistic execution and position tracking
- **Analysis & Reporting**: Performance metrics and risk analysis
- **Optimization**: Parameter optimization and walk-forward testing

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- tensorflow
- hmmlearn
- matplotlib
- seaborn

## Project Team

This project was developed by Team "Oops! All Bugs" for the UM Hackathon 2025:

- Ruben Raj - Lead Architect & Backend Developer
- Varsha Selvakumar - ML Specialist & Quant Analyst
- Syivhanii Selvarajan - Frontend Developer & Visual Designer
- Sharveen Raj - Trader & Risk Analyst
- Kedrick Selvanesan - DevOps & Documentation Lead

## License

MIT License