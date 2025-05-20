import requests
import logging

class MarketDataAPI:
    def __init__(self, source, api_key=None):
        self.source = source.lower()
        self.api_key = api_key

    def fetch(self, symbol, start_date=None, end_date=None, **kwargs):
        try:
            if self.source == "alphavantage":
                return self._fetch_alphavantage(symbol, start_date, end_date)
            elif self.source == "yahoo":
                return self._fetch_yahoo(symbol, start_date, end_date)
            else:
                raise ValueError(f"Unsupported data source: {self.source}")
        except Exception as e:
            logging.error(f"Error fetching data from {self.source}: {e}")
            return None

    def _fetch_alphavantage(self, symbol, start_date, end_date):
        if not self.api_key:
            raise ValueError("AlphaVantage API key required.")
        url = (
            f"https://www.alphavantage.co/query"
            f"?function=TIME_SERIES_DAILY_ADJUSTED"
            f"&symbol={symbol}"
            f"&outputsize=full"
            f"&apikey={self.api_key}"
        )
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            if "Error Message" in data:
                raise ValueError(data["Error Message"])
            return data
        except Exception as e:
            logging.error(f"AlphaVantage API error: {e}")
            return None

    def _fetch_yahoo(self, symbol, start_date, end_date):
        try:
            import yfinance as yf
        except ImportError:
            logging.error("yfinance package not installed.")
            return None
        try:
            data = yf.download(symbol, start=start_date, end=end_date)
            if data.empty:
                raise ValueError("No data returned from Yahoo Finance.")
            return data
        except Exception as e:
            logging.error(f"Yahoo Finance API error: {e}")
            return None