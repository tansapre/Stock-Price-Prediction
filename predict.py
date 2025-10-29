"""
Prediction Interface
Makes predictions for next day's Open and Close prices
"""

import numpy as np
import pandas as pd
import joblib
import logging
from datetime import datetime, timedelta

from zerodha_data_fetcher import ZerodhaDataFetcher
from data_preprocessing import StockDataPreprocessor
from lstm_model import StockPriceLSTM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockPredictor:
    """Class to make stock price predictions"""

    def __init__(self, model_path='models/stock_lstm_model.h5',
                 preprocessor_path='models/preprocessor.pkl'):
        """
        Initialize predictor

        Args:
            model_path: Path to trained model
            preprocessor_path: Path to preprocessor
        """
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model = None
        self.preprocessor = None

    def load_model(self):
        """Load trained model and preprocessor"""
        try:
            # Load preprocessor
            self.preprocessor = joblib.load(self.preprocessor_path)
            logger.info(f"Preprocessor loaded from {self.preprocessor_path}")

            # Load model
            input_shape = (self.preprocessor.lookback_period,
                           len(self.preprocessor.feature_columns))
            self.model = StockPriceLSTM(input_shape=input_shape,
                                       model_path=self.model_path)
            self.model.load_model()
            logger.info(f"Model loaded from {self.model_path}")

            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def predict_next_day(self, historical_data):
        """
        Predict next day's open and close prices

        Args:
            historical_data: DataFrame with OHLC data

        Returns:
            Dictionary with predictions
        """
        if self.model is None or self.preprocessor is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        try:
            # Prepare latest sequence
            latest_sequence = self.preprocessor.prepare_latest_sequence(historical_data)

            # Make prediction
            prediction_scaled = self.model.predict(latest_sequence)

            # Inverse transform to original scale
            prediction = self.preprocessor.inverse_transform_predictions(prediction_scaled)

            result = {
                'next_day_open': float(prediction[0][0]),
                'next_day_close': float(prediction[0][1]),
                'prediction_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
                'last_close': float(historical_data['Close'].iloc[-1])
            }

            # Calculate expected change
            result['open_change_pct'] = ((result['next_day_open'] - result['last_close']) /
                                        result['last_close']) * 100
            result['close_change_pct'] = ((result['next_day_close'] - result['last_close']) /
                                         result['last_close']) * 100

            logger.info(f"Prediction successful: Open={result['next_day_open']:.2f}, "
                       f"Close={result['next_day_close']:.2f}")

            return result

        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise

    def predict_with_confidence(self, historical_data, n_simulations=10):
        """
        Make predictions with confidence intervals using Monte Carlo dropout

        Args:
            historical_data: DataFrame with OHLC data
            n_simulations: Number of simulations for confidence estimation

        Returns:
            Dictionary with predictions and confidence intervals
        """
        if self.model is None or self.preprocessor is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        try:
            # Prepare latest sequence
            latest_sequence = self.preprocessor.prepare_latest_sequence(historical_data)

            # Make multiple predictions
            predictions = []
            for _ in range(n_simulations):
                pred_scaled = self.model.predict(latest_sequence)
                pred = self.preprocessor.inverse_transform_predictions(pred_scaled)
                predictions.append(pred[0])

            predictions = np.array(predictions)

            # Calculate statistics
            mean_pred = np.mean(predictions, axis=0)
            std_pred = np.std(predictions, axis=0)

            result = {
                'next_day_open': float(mean_pred[0]),
                'next_day_close': float(mean_pred[1]),
                'open_std': float(std_pred[0]),
                'close_std': float(std_pred[1]),
                'open_confidence_lower': float(mean_pred[0] - 2 * std_pred[0]),
                'open_confidence_upper': float(mean_pred[0] + 2 * std_pred[0]),
                'close_confidence_lower': float(mean_pred[1] - 2 * std_pred[1]),
                'close_confidence_upper': float(mean_pred[1] + 2 * std_pred[1]),
                'prediction_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
                'last_close': float(historical_data['Close'].iloc[-1])
            }

            # Calculate expected change
            result['open_change_pct'] = ((result['next_day_open'] - result['last_close']) /
                                        result['last_close']) * 100
            result['close_change_pct'] = ((result['next_day_close'] - result['last_close']) /
                                         result['last_close']) * 100

            return result

        except Exception as e:
            logger.error(f"Error making prediction with confidence: {e}")
            raise


def predict_for_symbol(symbol, api_key, api_secret, access_token,
                       exchange='NSE', days=90):
    """
    Make prediction for a given stock symbol

    Args:
        symbol: Stock symbol
        api_key: Zerodha API key
        api_secret: Zerodha API secret
        access_token: Valid access token
        exchange: Exchange name
        days: Days of historical data to fetch

    Returns:
        Prediction results
    """
    try:
        # Fetch data
        fetcher = ZerodhaDataFetcher(api_key, api_secret)
        fetcher.manual_set_access_token(access_token)

        logger.info(f"Fetching data for {symbol}...")
        df = fetcher.fetch_stock_data(symbol, exchange, days)

        if len(df) == 0:
            raise ValueError(f"No data returned for {symbol}")

        # Load predictor
        predictor = StockPredictor()
        predictor.load_model()

        # Make prediction
        prediction = predictor.predict_with_confidence(df)

        return prediction

    except Exception as e:
        logger.error(f"Error in prediction pipeline: {e}")
        raise


def main():
    """Main prediction function"""
    import argparse

    parser = argparse.ArgumentParser(description='Predict next day stock prices')
    parser.add_argument('--symbol', type=str, required=True,
                        help='Stock symbol')
    parser.add_argument('--exchange', type=str, default='NSE',
                        help='Exchange name')

    args = parser.parse_args()

    # Note: In production, get these from secure storage
    API_KEY = "di924ht7h955col3"
    API_SECRET = "4988yz4r2tg1n5relibczd8g98fsja44"

    print("="*60)
    print("STOCK PRICE PREDICTION")
    print("="*60)
    print(f"Symbol: {args.symbol}")
    print(f"Exchange: {args.exchange}")
    print("="*60)

    # Note: You need to authenticate and get access token first
    print("\nNote: Please authenticate with Zerodha and provide access token")
    print("Visit the login URL, authorize, and paste the request token here:")

    # For actual use:
    # fetcher = ZerodhaDataFetcher(API_KEY, API_SECRET)
    # print(fetcher.get_login_url())
    # request_token = input("Enter request token: ")
    # access_token = fetcher.set_access_token(request_token)

    # Then make prediction:
    # prediction = predict_for_symbol(args.symbol, API_KEY, API_SECRET, access_token, args.exchange)
    # print_prediction(prediction)


if __name__ == '__main__':
    main()
