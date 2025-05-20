import requests
import logging

class FundamentalDataAPI:
    def __init__(self, source, api_key=None):
        self.source = source.lower()
        self.api_key = api_key

    def fetch(self, symbol, **kwargs):
        try:
            if self.source == "alphavantage":
                return self._fetch_alphavantage(symbol)
            elif self.source == "fmp":
                return self._fetch_fmp(symbol)
            else:
                raise ValueError(f"Unsupported data source: {self.source}")
        except Exception as e:
            logging.error(f"Error fetching fundamental data from {self.source}: {e}")
            return None

    def _fetch_alphavantage(self, symbol):
        if not self.api_key:
            raise ValueError("AlphaVantage API key required.")
        url = (
            f"https://www.alphavantage.co/query"
            f"?function=OVERVIEW"
            f"&symbol={symbol}"
            f"&apikey={self.api_key}"
        )
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            if not data or "Note" in data or "Error Message" in data:
                raise ValueError(data.get("Error Message", "No data returned from AlphaVantage."))
            return data
        except Exception as e:
            logging.error(f"AlphaVantage API error: {e}")
            return None

    def _fetch_fmp(self, symbol):
        if not self.api_key:
            raise ValueError("FMP API key required.")
        url = (
            f"https://financialmodelingprep.com/api/v3/profile/{symbol}"
            f"?apikey={self.api_key}"
        )
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            if not data or isinstance(data, dict) and data.get("Error Message"):
                raise ValueError(data.get("Error Message", "No data returned from FMP."))
            return data
        except Exception as e:
            logging.error(f"FMP API error: {e}")
            return None