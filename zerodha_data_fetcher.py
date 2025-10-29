"""
Zerodha API Integration Module
Fetches historical stock data for model training and prediction
"""

from kiteconnect import KiteConnect
import pandas as pd
import datetime
from datetime import timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZerodhaDataFetcher:
    """Class to handle Zerodha API interactions and data fetching"""

    def __init__(self, api_key, api_secret):
        """
        Initialize Zerodha KiteConnect client

        Args:
            api_key: Zerodha API key
            api_secret: Zerodha API secret
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.kite = KiteConnect(api_key=api_key)
        self.access_token = None

    def get_login_url(self):
        """Get the login URL for manual authentication"""
        return self.kite.login_url()

    def set_access_token(self, request_token):
        """
        Generate and set access token using request token

        Args:
            request_token: Request token obtained after login
        """
        try:
            data = self.kite.generate_session(request_token, api_secret=self.api_secret)
            self.access_token = data["access_token"]
            self.kite.set_access_token(self.access_token)
            logger.info("Access token set successfully")
            return self.access_token
        except Exception as e:
            logger.error(f"Error setting access token: {e}")
            raise

    def manual_set_access_token(self, access_token):
        """
        Manually set access token if already available

        Args:
            access_token: Valid access token
        """
        self.access_token = access_token
        self.kite.set_access_token(access_token)
        logger.info("Access token set manually")

    def get_instruments(self, exchange="NSE"):
        """
        Get list of all instruments for a given exchange

        Args:
            exchange: Exchange name (NSE, BSE, NFO, etc.)

        Returns:
            DataFrame of instruments
        """
        try:
            instruments = self.kite.instruments(exchange)
            df = pd.DataFrame(instruments)
            return df
        except Exception as e:
            logger.error(f"Error fetching instruments: {e}")
            raise

    def get_instrument_token(self, symbol, exchange="NSE"):
        """
        Get instrument token for a given symbol

        Args:
            symbol: Stock symbol (e.g., 'RELIANCE', 'TCS')
            exchange: Exchange name

        Returns:
            Instrument token
        """
        try:
            instruments = self.get_instruments(exchange)
            instrument = instruments[instruments['tradingsymbol'] == symbol]
            if len(instrument) == 0:
                raise ValueError(f"Symbol {symbol} not found in {exchange}")
            return instrument.iloc[0]['instrument_token']
        except Exception as e:
            logger.error(f"Error getting instrument token: {e}")
            raise

    def fetch_historical_data(self, instrument_token, from_date, to_date, interval="day"):
        """
        Fetch historical OHLC data for a given instrument

        Args:
            instrument_token: Instrument token
            from_date: Start date (datetime or string YYYY-MM-DD)
            to_date: End date (datetime or string YYYY-MM-DD)
            interval: Candle interval (minute, day, 3minute, 5minute, etc.)

        Returns:
            DataFrame with OHLC data
        """
        try:
            if isinstance(from_date, str):
                from_date = datetime.datetime.strptime(from_date, "%Y-%m-%d")
            if isinstance(to_date, str):
                to_date = datetime.datetime.strptime(to_date, "%Y-%m-%d")

            historical_data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )

            df = pd.DataFrame(historical_data)

            if len(df) == 0:
                logger.warning(f"No data returned for instrument {instrument_token}")
                return df

            # Rename columns for consistency
            df = df.rename(columns={
                'date': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })

            # Set Date as index
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

            logger.info(f"Fetched {len(df)} records from {from_date} to {to_date}")
            return df

        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            raise

    def fetch_stock_data(self, symbol, exchange="NSE", days=365, interval="day"):
        """
        Fetch historical data for a stock symbol

        Args:
            symbol: Stock symbol
            exchange: Exchange name
            days: Number of days of historical data
            interval: Candle interval

        Returns:
            DataFrame with OHLC data
        """
        try:
            instrument_token = self.get_instrument_token(symbol, exchange)
            to_date = datetime.datetime.now()
            from_date = to_date - timedelta(days=days)

            df = self.fetch_historical_data(instrument_token, from_date, to_date, interval)
            return df

        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol}: {e}")
            raise

    def get_quote(self, symbol, exchange="NSE"):
        """
        Get current quote for a symbol

        Args:
            symbol: Stock symbol
            exchange: Exchange name

        Returns:
            Quote data dictionary
        """
        try:
            instrument_key = f"{exchange}:{symbol}"
            quote = self.kite.quote(instrument_key)
            return quote[instrument_key]
        except Exception as e:
            logger.error(f"Error fetching quote: {e}")
            raise


def save_data_to_csv(df, filename):
    """Save DataFrame to CSV file"""
    df.to_csv(filename)
    logger.info(f"Data saved to {filename}")


def load_data_from_csv(filename):
    """Load DataFrame from CSV file"""
    df = pd.read_csv(filename, index_col=0, parse_dates=True)
    logger.info(f"Data loaded from {filename}")
    return df
