"""
LSTM Model Architecture for Stock Price Prediction
Predicts next day's Open and Close prices
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockPriceLSTM:
    """LSTM Model for predicting next day's Open and Close prices"""

    def __init__(self, input_shape, model_path='models/stock_lstm_model.h5'):
        """
        Initialize LSTM model

        Args:
            input_shape: Shape of input data (lookback_period, n_features)
            model_path: Path to save/load model
        """
        self.input_shape = input_shape
        self.model_path = model_path
        self.model = None
        self.history = None

        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

    def build_model(self, lstm_units=[128, 64, 32], dropout_rate=0.2, learning_rate=0.001):
        """
        Build LSTM model architecture

        Args:
            lstm_units: List of LSTM layer units
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer

        Returns:
            Compiled model
        """
        model = Sequential()

        # First LSTM layer (Bidirectional for better context understanding)
        model.add(Bidirectional(
            LSTM(lstm_units[0], return_sequences=True, input_shape=self.input_shape),
            name='bidirectional_lstm_1'
        ))
        model.add(Dropout(dropout_rate))
        model.add(BatchNormalization())

        # Second LSTM layer
        model.add(Bidirectional(
            LSTM(lstm_units[1], return_sequences=True),
            name='bidirectional_lstm_2'
        ))
        model.add(Dropout(dropout_rate))
        model.add(BatchNormalization())

        # Third LSTM layer
        model.add(LSTM(lstm_units[2], return_sequences=False, name='lstm_3'))
        model.add(Dropout(dropout_rate))
        model.add(BatchNormalization())

        # Dense layers
        model.add(Dense(32, activation='relu', name='dense_1'))
        model.add(Dropout(dropout_rate))

        model.add(Dense(16, activation='relu', name='dense_2'))
        model.add(Dropout(dropout_rate))

        # Output layer: 2 neurons for Open and Close prices
        model.add(Dense(2, activation='linear', name='output'))

        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['mean_absolute_error', 'mean_absolute_percentage_error']
        )

        self.model = model
        logger.info(f"Model built successfully")
        logger.info(f"Total parameters: {model.count_params()}")

        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """
        Train the LSTM model

        Args:
            X_train: Training features
            y_train: Training targets (open, close)
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size

        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )

        model_checkpoint = ModelCheckpoint(
            self.model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )

        callbacks = [early_stopping, model_checkpoint, reduce_lr]

        logger.info("Starting model training...")

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        logger.info("Training completed")
        return self.history

    def predict(self, X):
        """
        Make predictions

        Args:
            X: Input features

        Returns:
            Predictions (open, close)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call build_model() or load_model() first.")

        predictions = self.model.predict(X)
        return predictions

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Loss and metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call build_model() or load_model() first.")

        results = self.model.evaluate(X_test, y_test, verbose=1)
        logger.info(f"Test Loss: {results[0]}")
        logger.info(f"Test MAE: {results[1]}")
        logger.info(f"Test MAPE: {results[2]}")

        return results

    def calculate_rmse(self, y_true, y_pred):
        """
        Calculate RMSE for predictions

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            RMSE value
        """
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        return rmse

    def calculate_rmse_percentage(self, y_true, y_pred):
        """
        Calculate RMSE as percentage of actual price

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            RMSE percentage for Open and Close separately
        """
        # Calculate RMSE for Open (column 0)
        rmse_open = self.calculate_rmse(y_true[:, 0], y_pred[:, 0])
        mean_open = np.mean(y_true[:, 0])
        rmse_pct_open = (rmse_open / mean_open) * 100

        # Calculate RMSE for Close (column 1)
        rmse_close = self.calculate_rmse(y_true[:, 1], y_pred[:, 1])
        mean_close = np.mean(y_true[:, 1])
        rmse_pct_close = (rmse_close / mean_close) * 100

        logger.info(f"RMSE % for Open: {rmse_pct_open:.4f}%")
        logger.info(f"RMSE % for Close: {rmse_pct_close:.4f}%")

        return rmse_pct_open, rmse_pct_close

    def save_model(self, path=None):
        """Save model to disk"""
        if path is None:
            path = self.model_path

        self.model.save(path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path=None):
        """Load model from disk"""
        if path is None:
            path = self.model_path

        self.model = load_model(path)
        logger.info(f"Model loaded from {path}")
        return self.model

    def get_model_summary(self):
        """Print model summary"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        return self.model.summary()


def calculate_detailed_metrics(y_true, y_pred):
    """
    Calculate detailed performance metrics

    Args:
        y_true: True values (N, 2) - Open and Close
        y_pred: Predicted values (N, 2) - Open and Close

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Open price metrics
    mae_open = np.mean(np.abs(y_true[:, 0] - y_pred[:, 0]))
    mse_open = np.mean((y_true[:, 0] - y_pred[:, 0]) ** 2)
    rmse_open = np.sqrt(mse_open)
    mape_open = np.mean(np.abs((y_true[:, 0] - y_pred[:, 0]) / y_true[:, 0])) * 100
    rmse_pct_open = (rmse_open / np.mean(y_true[:, 0])) * 100

    # Close price metrics
    mae_close = np.mean(np.abs(y_true[:, 1] - y_pred[:, 1]))
    mse_close = np.mean((y_true[:, 1] - y_pred[:, 1]) ** 2)
    rmse_close = np.sqrt(mse_close)
    mape_close = np.mean(np.abs((y_true[:, 1] - y_pred[:, 1]) / y_true[:, 1])) * 100
    rmse_pct_close = (rmse_close / np.mean(y_true[:, 1])) * 100

    metrics['open'] = {
        'MAE': mae_open,
        'MSE': mse_open,
        'RMSE': rmse_open,
        'MAPE': mape_open,
        'RMSE_PCT': rmse_pct_open
    }

    metrics['close'] = {
        'MAE': mae_close,
        'MSE': mse_close,
        'RMSE': rmse_close,
        'MAPE': mape_close,
        'RMSE_PCT': rmse_pct_close
    }

    return metrics


def print_metrics(metrics):
    """Print metrics in a formatted way"""
    print("\n" + "="*60)
    print("MODEL PERFORMANCE METRICS")
    print("="*60)

    print("\nOPEN PRICE PREDICTIONS:")
    print(f"  MAE:       {metrics['open']['MAE']:.4f}")
    print(f"  RMSE:      {metrics['open']['RMSE']:.4f}")
    print(f"  RMSE %:    {metrics['open']['RMSE_PCT']:.4f}%")
    print(f"  MAPE:      {metrics['open']['MAPE']:.4f}%")

    print("\nCLOSE PRICE PREDICTIONS:")
    print(f"  MAE:       {metrics['close']['MAE']:.4f}")
    print(f"  RMSE:      {metrics['close']['RMSE']:.4f}")
    print(f"  RMSE %:    {metrics['close']['RMSE_PCT']:.4f}%")
    print(f"  MAPE:      {metrics['close']['MAPE']:.4f}%")

    print("\n" + "="*60)

    # Check if RMSE < 1%
    if metrics['open']['RMSE_PCT'] < 1.0 and metrics['close']['RMSE_PCT'] < 1.0:
        print("SUCCESS: RMSE is less than 1% for both Open and Close prices!")
    else:
        print("Target not met. RMSE should be < 1% for both prices.")
        print("Consider: More training data, feature engineering, or model tuning.")

    print("="*60 + "\n")
