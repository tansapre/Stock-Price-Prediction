"""
Configuration file for Zerodha API credentials
IMPORTANT: Keep this file secure and don't commit to public repositories
"""

# Zerodha API Credentials
API_KEY = "di924ht7h955col3"
API_SECRET = "4988yz4r2tg1n5relibczd8g98fsja44"

# Model Configuration
MODEL_PATH = "models/stock_lstm_model.h5"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

# Data Configuration
DEFAULT_LOOKBACK_PERIOD = 60
DEFAULT_EXCHANGE = "NSE"
DEFAULT_DAYS_HISTORICAL = 730

# Training Configuration
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
DEFAULT_TEST_SPLIT = 0.2
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_DROPOUT_RATE = 0.3

# LSTM Architecture
LSTM_UNITS = [128, 64, 32]
DENSE_UNITS = [32, 16]

# Performance Targets
TARGET_RMSE_PERCENTAGE = 1.0  # Target: RMSE < 1% of stock price
