# sentio/strategy/hmm_strategy.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import logging

from sentio.strategy.strategy import Strategy

class HMMStrategy(Strategy):
    """
    Hidden Markov Model based trading strategy.
    
    This strategy identifies hidden market regimes using HMM and applies
    regime-specific trading rules based on liquidity flow metrics.
    """
    
    def __init__(
        self,
        name: str,
        parameters: Dict[str, Any] = None,
        n_states: int = 4,
        random_state: int = 42
    ):
        """
        Initialize the HMM Strategy.
        
        Args:
            name: Strategy identifier
            parameters: Dictionary of strategy parameters
            n_states: Number of hidden states in the HMM
            random_state: Random seed for reproducibility
        """
        default_params = {
            'n_states': n_states,
            'flow_lookback': 24,  # Hours to look back for liquidity flow
            'signal_threshold': 0.5,  # Threshold to generate a signal
            'regime_coefficients': None,  # Will be learned during training
            'random_state': random_state,
            'covariance_type': 'full',
            'n_iter': 100
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name, default_params)
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.regime_performance = {}
        
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for HMM from raw data.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Feature array suitable for HMM
        """
        required_columns = [
            'exchange_inflow',
            'exchange_outflow',
            'net_flow',
            'order_book_depth_bid',
            'order_book_depth_ask',
            'price_close'
        ]
        
        # Check if all required columns exist
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            missing_str = ', '.join(missing_columns)
            raise ValueError(f"Missing required columns: {missing_str}")
        
        # Create feature DataFrame
        features_df = pd.DataFrame(index=data.index)
        
        # Add net flow with various lookback periods
        lookback = self.parameters['flow_lookback']
        features_df['net_flow_current'] = data['net_flow']
        features_df[f'net_flow_{lookback}h'] = data['net_flow'].rolling(window=lookback).sum()
        
        # Add order book imbalance
        features_df['ob_imbalance'] = (data['order_book_depth_bid'] - data['order_book_depth_ask']) / (
                data['order_book_depth_bid'] + data['order_book_depth_ask'])
        
        # Add price returns
        features_df['returns_1h'] = data['price_close'].pct_change(1)
        features_df['returns_24h'] = data['price_close'].pct_change(24)
        
        # Add inflow/outflow ratio
        features_df['flow_ratio'] = data['exchange_inflow'] / data['exchange_outflow'].replace(0, 1e-8)
        
        # Store feature columns for later use
        self.feature_columns = features_df.columns.tolist()
        
        # Drop rows with NaN values (from rolling windows)
        features_df = features_df.dropna()
        
        # Standardize features
        features_array = self.scaler.fit_transform(features_df)
        
        return features_array
    
    def train(self, data: pd.DataFrame) -> None:
        """
        Train the HMM model on historical data.
        
        Args:
            data: DataFrame with historical market data
        """
        features = self._prepare_features(data)
        
        if len(features) == 0:
            raise ValueError("No valid features for training after preprocessing")
        
        # Initialize and train HMM
        self.model = hmm.GaussianHMM(
            n_components=self.parameters['n_states'],
            covariance_type=self.parameters['covariance_type'],
            n_iter=self.parameters['n_iter'],
            random_state=self.parameters['random_state']
        )
        
        logging.info(f"Training HMM with {len(features)} samples")
        self.model.fit(features)
        
        # Predict hidden states
        hidden_states = self.model.predict(features)
        
        # Add states back to original data
        valid_indices = data.index[data.index.isin(data.dropna().index)][:len(hidden_states)]
        states_df = pd.DataFrame(
            {'regime': hidden_states},
            index=valid_indices
        )
        
        # Analyze regime performance
        merged_data = pd.concat([data.loc[valid_indices], states_df], axis=1)
        self._learn_regime_coefficients(merged_data)
        
        self.is_trained = True
        logging.info("HMM training completed successfully")
        
    def _learn_regime_coefficients(self, data: pd.DataFrame) -> None:
        """
        Learn the coefficients for each regime based on historical performance.
        
        Args:
            data: DataFrame with market data and regime labels
        """
        # Calculate forward returns for performance evaluation
        data['forward_returns_24h'] = data['price_close'].pct_change(24).shift(-24)
        
        regime_coefficients = {}
        
        # For each regime, analyze how net flows correlate with future returns
        for regime in range(self.parameters['n_states']):
            regime_data = data[data['regime'] == regime].copy()
            
            if len(regime_data) < 24:  # Need enough data points
                regime_coefficients[regime] = 0
                continue
                
            # Calculate correlation between net flow and future returns
            correlation = regime_data['net_flow'].corr(regime_data['forward_returns_24h'])
            
            # Set coefficient based on correlation strength and direction
            regime_coefficients[regime] = np.clip(correlation * 2, -1, 1)
            
            # Store regime performance stats for analysis
            self.regime_performance[regime] = {
                'count': len(regime_data),
                'avg_return': regime_data['forward_returns_24h'].mean(),
                'flow_return_corr': correlation,
                'coefficient': regime_coefficients[regime]
            }
        
        self.parameters['regime_coefficients'] = regime_coefficients
        logging.info(f"Learned regime coefficients: {regime_coefficients}")
    
    def predict_regime(self, data: pd.DataFrame) -> pd.Series:
        """
        Predict the market regime for each data point.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Series with predicted regime for each timestamp
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before predicting regimes")
            
        features = self._prepare_features(data)
        hidden_states = self.model.predict(features)
        
        # Create series with regime labels
        valid_indices = data.index[data.index.isin(data.dropna().index)][:len(hidden_states)]
        regimes = pd.Series(hidden_states, index=valid_indices, name='regime')
        
        return regimes
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on liquidity flow and current regime.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Series with trading signals (1=buy, -1=sell, 0=hold)
        """
        if not self.is_trained:
            raise ValueError("Strategy must be trained before generating signals")
            
        # Predict regimes
        regimes = self.predict_regime(data)
        
        # Prepare features for signal generation
        features_df = pd.DataFrame(index=data.index)
        lookback = self.parameters['flow_lookback']
        features_df['net_flow_current'] = data['net_flow']
        features_df[f'net_flow_{lookback}h'] = data['net_flow'].rolling(window=lookback).sum()
        
        # Generate signals based on flows and regime coefficients
        signals = pd.Series(0, index=data.index)
        regime_coefs = self.parameters['regime_coefficients']
        
        for idx in regimes.index:
            if idx not in features_df.index:
                continue
                
            regime = regimes[idx]
            net_flow = features_df.loc[idx, f'net_flow_{lookback}h']
            
            if pd.isna(net_flow):
                continue
                
            # Apply regime-specific coefficient to determine signal
            coef = regime_coefs.get(regime, 0)
            flow_signal = net_flow * coef
            
            # Generate signal based on threshold
            threshold = self.parameters['signal_threshold']
            if flow_signal > threshold:
                signals[idx] = 1  # Buy signal
            elif flow_signal < -threshold:
                signals[idx] = -1  # Sell signal
        
        return signals
    
    def get_required_data(self) -> List[str]:
        """
        Get the list of data fields required by this strategy.
        
        Returns:
            List of required data field names
        """
        return [
            'exchange_inflow',
            'exchange_outflow', 
            'net_flow',
            'order_book_depth_bid',
            'order_book_depth_ask',
            'price_close'
        ]