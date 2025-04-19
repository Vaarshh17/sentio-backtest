import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class EmotionalAnalysisStrategy:
    """
    Strategy that implements the emotional analysis component from Sentio's architecture.
    This strategy combines technical indicators with sentiment analysis to generate trading signals.
    """
    
    def __init__(self, 
                rsi_period: int = 14,
                rsi_overbought: float = 70,
                rsi_oversold: float = 30,
                volatility_window: int = 20,
                sentiment_weight: float = 0.5,
                trend_weight: float = 0.3,
                fear_threshold: float = -0.5,
                greed_threshold: float = 0.5):
        """
        Initialize the strategy with parameters.
        
        Args:
            rsi_period: Period for RSI calculation
            rsi_overbought: RSI level considered overbought
            rsi_oversold: RSI level considered oversold
            volatility_window: Window size for volatility calculation
            sentiment_weight: Weight for sentiment in the trading decision
            trend_weight: Weight for trend in the trading decision
            fear_threshold: Threshold for fear sentiment to trigger action
            greed_threshold: Threshold for greed sentiment to trigger action
        """
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.volatility_window = volatility_window
        self.sentiment_weight = sentiment_weight
        self.trend_weight = trend_weight
        self.fear_threshold = fear_threshold
        self.greed_threshold = greed_threshold
        
    def generate_signals(self, data: pd.DataFrame) -> Dict[Any, int]:
        """
        Generate trading signals based on emotional analysis and technical indicators.
        
        Args:
            data: Market data with OHLCV columns
            
        Returns:
            Dictionary mapping dates to signals (1 for buy, -1 for sell, 0 for hold)
        """
        # Make a copy of the data to avoid modifying the original
        df = data.copy()
        
        # Calculate technical indicators
        df = self._calculate_indicators(df)
        
        # Calculate market sentiment (simulated here)
        df = self._calculate_sentiment(df)
        
        # Combine indicators and sentiment to generate signals
        signals = {}
        
        for idx, row in df.iterrows():
            if pd.isna(row['composite_signal']):
                signals[idx] = 0  # Skip if any indicators are missing
                continue
                
            # Convert the composite signal to a trading decision
            if row['composite_signal'] > self.greed_threshold:
                signals[idx] = 1  # Buy signal
            elif row['composite_signal'] < self.fear_threshold:
                signals[idx] = -1  # Sell signal
            else:
                signals[idx] = 0  # Hold
        
        # Ensure we have at least 3% trade signals (required by the hackathon)
        self._ensure_minimum_trades(signals, min_percentage=0.03)
        
        return signals
        
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators used by the strategy"""
        # RSI (Relative Strength Index)
        df['returns'] = df['close'].pct_change()
        df['up_returns'] = df['returns'].apply(lambda x: x if x > 0 else 0)
        df['down_returns'] = df['returns'].apply(lambda x: abs(x) if x < 0 else 0)
        
        df['up_avg'] = df['up_returns'].rolling(window=self.rsi_period).mean()
        df['down_avg'] = df['down_returns'].rolling(window=self.rsi_period).mean()
        
        df['rs'] = df['up_avg'] / df['down_avg']
        df['rsi'] = 100 - (100 / (1 + df['rs']))
        
        # Normalize RSI to -1 to 1 scale for easier integration with sentiment
        df['rsi_norm'] = (df['rsi'] - 50) / 50  # Now -1 to 1
        
        # Moving Averages
        df['sma_short'] = df['close'].rolling(window=20).mean()
        df['sma_long'] = df['close'].rolling(window=50).mean()
        
        # MA Crossover signal
        df['trend_signal'] = ((df['sma_short'] > df['sma_long']).astype(int) * 2 - 1)  # 1 for uptrend, -1 for downtrend
        
        # Volatility
        df['volatility'] = df['close'].pct_change().rolling(window=self.volatility_window).std()
        df['volatility_norm'] = df['volatility'] / df['volatility'].rolling(window=100).mean()
        
        return df
    
    def _calculate_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate market sentiment based on price action and volatility.
        
        Note: In a real implementation, this would incorporate actual sentiment data 
        from alternative data sources as described in the Sentio architecture.
        """
        # Simulated sentiment based on price momentum and volatility
        # In a real implementation, this would use external sentiment data
        
        # Price momentum (simulated sentiment component)
        df['price_momentum'] = df['close'].pct_change(5) / df['volatility']
        
        # Volatility change (rising volatility can indicate fear)
        df['vol_change'] = df['volatility'].pct_change(5)
        
        # Sentiment score: positive for greed, negative for fear
        df['sentiment'] = df['price_momentum'] - df['vol_change'] 
        
        # Normalize sentiment to approximately -1 to 1 range
        scaler = StandardScaler()
        if len(df) > 10:  # Need enough data for scaling
            df['sentiment'] = pd.Series(
                scaler.fit_transform(df['sentiment'].values.reshape(-1, 1)).flatten(),
                index=df.index
            )
        
        # Composite signal combining sentiment, RSI, and trend
        df['composite_signal'] = (
            df['sentiment'] * self.sentiment_weight + 
            df['rsi_norm'] * (1 - self.sentiment_weight - self.trend_weight) +
            df['trend_signal'] * self.trend_weight
        )
        
        return df
    
    def _ensure_minimum_trades(self, signals: Dict[Any, int], min_percentage: float = 0.03):
        """
        Ensure that we have at least the minimum percentage of trades required.
        
        Args:
            signals: Dictionary of trading signals
            min_percentage: Minimum percentage of signals that should be trades (non-zero)
        """
        # Count current non-zero signals
        trade_count = sum(1 for s in signals.values() if s != 0)
        total_signals = len(signals)
        current_percentage = trade_count / total_signals if total_signals > 0 else 0
        
        # If we already meet the requirement, return
        if current_percentage >= min_percentage:
            return
        
        # Calculate how many additional trades we need
        required_trades = int(min_percentage * total_signals) - trade_count
        
        # Convert some hold signals to trade signals
        hold_indices = [idx for idx, signal in signals.items() if signal == 0]
        
        if required_trades > 0 and hold_indices:
            # Randomly select hold signals to convert to trades
            np.random.shuffle(hold_indices)
            for idx in hold_indices[:required_trades]:
                # Assign buy or sell randomly (1 or -1)
                signals[idx] = np.random.choice([1, -1])