"""
Model Training Script
Trains the LSTM model on historical stock data with RMSE < 1% target
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import joblib
import logging

from zerodha_data_fetcher import ZerodhaDataFetcher
from data_preprocessing import StockDataPreprocessor
from lstm_model import StockPriceLSTM, calculate_detailed_metrics, print_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_training_history(history, save_path='plots/training_history.png'):
    """Plot training history"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Loss plot
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss During Training')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # MAE plot
    axes[1].plot(history.history['mean_absolute_error'], label='Training MAE')
    axes[1].plot(history.history['val_mean_absolute_error'], label='Validation MAE')
    axes[1].set_title('Mean Absolute Error During Training')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f"Training history plot saved to {save_path}")
    plt.close()


def plot_predictions(y_true, y_pred, save_path='plots/predictions.png'):
    """Plot actual vs predicted prices"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Open price
    axes[0].plot(y_true[:, 0], label='Actual Open', color='blue', alpha=0.7)
    axes[0].plot(y_pred[:, 0], label='Predicted Open', color='red', alpha=0.7)
    axes[0].set_title('Open Price: Actual vs Predicted')
    axes[0].set_xlabel('Time Steps')
    axes[0].set_ylabel('Price')
    axes[0].legend()
    axes[0].grid(True)

    # Close price
    axes[1].plot(y_true[:, 1], label='Actual Close', color='green', alpha=0.7)
    axes[1].plot(y_pred[:, 1], label='Predicted Close', color='orange', alpha=0.7)
    axes[1].set_title('Close Price: Actual vs Predicted')
    axes[1].set_xlabel('Time Steps')
    axes[1].set_ylabel('Price')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f"Predictions plot saved to {save_path}")
    plt.close()


def plot_error_distribution(y_true, y_pred, save_path='plots/error_distribution.png'):
    """Plot error distribution"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Open price errors
    errors_open = y_true[:, 0] - y_pred[:, 0]
    axes[0].hist(errors_open, bins=50, color='blue', alpha=0.7, edgecolor='black')
    axes[0].set_title('Open Price Prediction Errors Distribution')
    axes[0].set_xlabel('Error')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0].grid(True, alpha=0.3)

    # Close price errors
    errors_close = y_true[:, 1] - y_pred[:, 1]
    axes[1].hist(errors_close, bins=50, color='green', alpha=0.7, edgecolor='black')
    axes[1].set_title('Close Price Prediction Errors Distribution')
    axes[1].set_xlabel('Error')
    axes[1].set_ylabel('Frequency')
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f"Error distribution plot saved to {save_path}")
    plt.close()


def train_stock_model(symbol, exchange='NSE', days=730, lookback_period=60,
                      epochs=100, batch_size=32, test_split=0.2):
    """
    Complete training pipeline

    Args:
        symbol: Stock symbol (e.g., 'RELIANCE', 'TCS')
        exchange: Exchange name
        days: Number of days of historical data
        lookback_period: Number of previous days for prediction
        epochs: Training epochs
        batch_size: Batch size
        test_split: Test data split ratio

    Returns:
        Trained model and metrics
    """
    logger.info(f"Starting training for {symbol}")

    # Step 1: Fetch data (using dummy data for now, will use API after authentication)
    logger.info("Note: For actual training, you need to authenticate with Zerodha API")
    logger.info("Generating sample data for demonstration...")

    # For demonstration, creating sample data
    # In production, you would use:
    # fetcher = ZerodhaDataFetcher(api_key, api_secret)
    # fetcher.set_access_token(access_token)
    # df = fetcher.fetch_stock_data(symbol, exchange, days)

    # Generate sample data for demonstration
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    np.random.seed(42)
    base_price = 1000
    df = pd.DataFrame({
        'Date': dates,
        'Open': base_price + np.cumsum(np.random.randn(days) * 10),
        'High': base_price + np.cumsum(np.random.randn(days) * 10) + 20,
        'Low': base_price + np.cumsum(np.random.randn(days) * 10) - 20,
        'Close': base_price + np.cumsum(np.random.randn(days) * 10),
        'Volume': np.random.randint(1000000, 10000000, days)
    })
    df.set_index('Date', inplace=True)

    # Ensure positive prices
    df[df < 0] = np.abs(df[df < 0])

    logger.info(f"Data shape: {df.shape}")

    # Step 2: Preprocess data
    preprocessor = StockDataPreprocessor(lookback_period=lookback_period)
    X_train, X_test, y_train, y_test, feature_df = preprocessor.prepare_data_for_training(
        df, test_split=test_split
    )

    # Save preprocessor
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    logger.info("Preprocessor saved")

    # Step 3: Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    lstm_model = StockPriceLSTM(input_shape=input_shape)

    lstm_model.build_model(
        lstm_units=[128, 64, 32],
        dropout_rate=0.3,
        learning_rate=0.001
    )

    # Print model summary
    lstm_model.get_model_summary()

    # Step 4: Train model
    history = lstm_model.train(
        X_train, y_train,
        X_test, y_test,
        epochs=epochs,
        batch_size=batch_size
    )

    # Step 5: Evaluate model
    logger.info("\n" + "="*60)
    logger.info("EVALUATING MODEL ON TEST SET")
    logger.info("="*60)

    # Make predictions
    y_pred_scaled = lstm_model.predict(X_test)

    # Inverse transform to original scale
    y_pred = preprocessor.inverse_transform_predictions(y_pred_scaled)
    y_test_original = preprocessor.inverse_transform_predictions(y_test)

    # Calculate metrics
    metrics = calculate_detailed_metrics(y_test_original, y_pred)
    print_metrics(metrics)

    # Step 6: Generate plots
    plot_training_history(history)
    plot_predictions(y_test_original, y_pred)
    plot_error_distribution(y_test_original, y_pred)

    # Step 7: Save results
    results = {
        'symbol': symbol,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_points': len(df),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'lookback_period': lookback_period,
        'epochs_trained': len(history.history['loss']),
        'metrics': metrics
    }

    results_df = pd.DataFrame([results])
    results_df.to_csv('models/training_results.csv', index=False)
    logger.info("Training results saved")

    return lstm_model, metrics, preprocessor


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train stock price prediction model')
    parser.add_argument('--symbol', type=str, default='RELIANCE',
                        help='Stock symbol')
    parser.add_argument('--exchange', type=str, default='NSE',
                        help='Exchange name')
    parser.add_argument('--days', type=int, default=730,
                        help='Number of days of historical data')
    parser.add_argument('--lookback', type=int, default=60,
                        help='Lookback period for LSTM')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("STOCK PRICE PREDICTION MODEL TRAINING")
    logger.info("="*60)
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Exchange: {args.exchange}")
    logger.info(f"Historical data: {args.days} days")
    logger.info(f"Lookback period: {args.lookback} days")
    logger.info(f"Training epochs: {args.epochs}")
    logger.info("="*60)

    model, metrics, preprocessor = train_stock_model(
        symbol=args.symbol,
        exchange=args.exchange,
        days=args.days,
        lookback_period=args.lookback,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    logger.info("\nTraining completed successfully!")


if __name__ == '__main__':
    main()
