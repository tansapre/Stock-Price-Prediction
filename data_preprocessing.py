"""
Data Preprocessing Module
Handles feature engineering and data preparation for LSTM model
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataPreprocessor:
    """Class to preprocess stock data for LSTM model"""

    def __init__(self, lookback_period=60):
        """
        Initialize preprocessor

        Args:
            lookback_period: Number of previous days to use for prediction
        """
        self.lookback_period = lookback_period
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_columns = None
        self.scaled_data = None

    def create_technical_indicators(self, df):
        """
        Create technical indicators as features

        Args:
            df: DataFrame with OHLC data

        Returns:
            DataFrame with additional technical indicators
        """
        df = df.copy()

        # Moving Averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()

        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)

        # Price momentum
        df['Momentum'] = df['Close'] - df['Close'].shift(4)

        # Price rate of change
        df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100

        # Average True Range (ATR)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()

        # Daily returns
        df['Daily_Return'] = df['Close'].pct_change()

        # Volume indicators
        df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
        df['Volume_Change'] = df['Volume'].pct_change()

        # Price range
        df['High_Low_Range'] = df['High'] - df['Low']
        df['Open_Close_Range'] = np.abs(df['Open'] - df['Close'])

        # Drop NaN values created by indicators
        df = df.dropna()

        logger.info(f"Created technical indicators. DataFrame shape: {df.shape}")
        return df

    def prepare_features(self, df):
        """
        Prepare feature set for model training

        Args:
            df: DataFrame with OHLC and technical indicators

        Returns:
            DataFrame with selected features
        """
        # Define feature columns
        self.feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'MA_5', 'MA_10', 'MA_20', 'MA_50',
            'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal',
            'RSI', 'BB_Middle', 'BB_Upper', 'BB_Lower',
            'Momentum', 'ROC', 'ATR', 'Daily_Return',
            'Volume_MA_5', 'Volume_Change',
            'High_Low_Range', 'Open_Close_Range'
        ]

        # Select features
        feature_df = df[self.feature_columns].copy()

        logger.info(f"Prepared {len(self.feature_columns)} features")
        return feature_df

    def scale_data(self, df):
        """
        Scale data using MinMaxScaler

        Args:
            df: DataFrame with features

        Returns:
            Scaled numpy array
        """
        self.scaled_data = self.scaler.fit_transform(df)
        logger.info(f"Data scaled. Shape: {self.scaled_data.shape}")
        return self.scaled_data

    def create_sequences(self, data, target_indices=[3]):
        """
        Create sequences for LSTM model

        Args:
            data: Scaled data array
            target_indices: Indices of target columns (default: [3] for Close price)

        Returns:
            X (features), y (targets) numpy arrays
        """
        X, y = [], []

        for i in range(self.lookback_period, len(data) - 1):
            # Input sequence
            X.append(data[i - self.lookback_period:i, :])

            # Next day's open and close prices
            # Assuming Open is at index 0 and Close is at index 3
            next_day_open = data[i + 1, 0]
            next_day_close = data[i + 1, 3]
            y.append([next_day_open, next_day_close])

        X = np.array(X)
        y = np.array(y)

        logger.info(f"Created sequences. X shape: {X.shape}, y shape: {y.shape}")
        return X, y

    def inverse_transform_predictions(self, predictions):
        """
        Inverse transform predictions to original scale

        Args:
            predictions: Scaled predictions (open, close)

        Returns:
            Original scale predictions
        """
        # Create dummy array with same number of features
        dummy = np.zeros((predictions.shape[0], len(self.feature_columns)))

        # Place open predictions in column 0 and close predictions in column 3
        dummy[:, 0] = predictions[:, 0]  # Open
        dummy[:, 3] = predictions[:, 1]  # Close

        # Inverse transform
        original_scale = self.scaler.inverse_transform(dummy)

        # Extract open and close
        result = np.column_stack([original_scale[:, 0], original_scale[:, 3]])

        return result

    def prepare_data_for_training(self, df, test_split=0.2):
        """
        Complete data preparation pipeline

        Args:
            df: Raw OHLC DataFrame
            test_split: Fraction of data to use for testing

        Returns:
            X_train, X_test, y_train, y_test
        """
        # Create technical indicators
        df_with_indicators = self.create_technical_indicators(df)

        # Prepare features
        feature_df = self.prepare_features(df_with_indicators)

        # Scale data
        scaled_data = self.scale_data(feature_df)

        # Create sequences
        X, y = self.create_sequences(scaled_data)

        # Split into train and test
        split_index = int(len(X) * (1 - test_split))

        X_train = X[:split_index]
        X_test = X[split_index:]
        y_train = y[:split_index]
        y_test = y[split_index:]

        logger.info(f"Training set: X={X_train.shape}, y={y_train.shape}")
        logger.info(f"Test set: X={X_test.shape}, y={y_test.shape}")

        return X_train, X_test, y_train, y_test, feature_df

    def prepare_latest_sequence(self, df):
        """
        Prepare the latest sequence for making predictions

        Args:
            df: DataFrame with OHLC data including latest day

        Returns:
            Sequence ready for prediction
        """
        # Create technical indicators
        df_with_indicators = self.create_technical_indicators(df)

        # Prepare features
        feature_df = self.prepare_features(df_with_indicators)

        # Scale data
        scaled_data = self.scaler.transform(feature_df)

        # Get the latest sequence
        if len(scaled_data) < self.lookback_period:
            raise ValueError(f"Need at least {self.lookback_period} days of data")

        latest_sequence = scaled_data[-self.lookback_period:]
        latest_sequence = np.expand_dims(latest_sequence, axis=0)

        return latest_sequence
