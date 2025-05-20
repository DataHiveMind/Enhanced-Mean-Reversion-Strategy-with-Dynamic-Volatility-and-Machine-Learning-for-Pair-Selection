import numpy as np
import pandas as pd
import statsmodels.api as sm
import logging

class TimeSeriesAnalysis:
    @staticmethod
    def autocorrelation(series, lags=20):
        """Compute autocorrelation for a given number of lags."""
        try:
            return [series.autocorr(lag) for lag in range(1, lags+1)]
        except Exception as e:
            logging.error(f"Autocorrelation error: {e}")
            return None

    @staticmethod
    def partial_autocorrelation(series, lags=20):
        """Compute partial autocorrelation for a given number of lags."""
        try:
            from statsmodels.tsa.stattools import pacf
            return pacf(series.dropna(), nlags=lags)
        except Exception as e:
            logging.error(f"Partial autocorrelation error: {e}")
            return None

    @staticmethod
    def seasonal_decompose(series, model='additive', freq=None):
        """Decompose time series into trend, seasonal, and residuals."""
        try:
            result = sm.tsa.seasonal_decompose(series.dropna(), model=model, period=freq)
            return result
        except Exception as e:
            logging.error(f"Seasonal decomposition error: {e}")
            return None

    @staticmethod
    def moving_average(series, window=5):
        """Calculate moving average."""
        try:
            return series.rolling(window=window).mean()
        except Exception as e:
            logging.error(f"Moving average error: {e}")
            return None

    @staticmethod
    def exponential_smoothing(series, alpha=0.2):
        """Simple exponential smoothing."""
        try:
            return series.ewm(alpha=alpha, adjust=False).mean()
        except Exception as e:
            logging.error(f"Exponential smoothing error: {e}")
            return None

    @staticmethod
    def arima_fit(series, order=(1,0,0)):
        """Fit an ARIMA model and return the fitted model."""
        try:
            model = sm.tsa.ARIMA(series.dropna(), order=order)
            result = model.fit()
            return result
        except Exception as e:
            logging.error(f"ARIMA fit error: {e}")
            return None

    @staticmethod
    def forecast_arima(series, order=(1,0,0), steps=5):
        """Fit ARIMA and forecast future values."""
        try:
            model = sm.tsa.ARIMA(series.dropna(), order=order)
            result = model.fit()
            forecast = result.forecast(steps=steps)
            return forecast
        except Exception as e:
            logging.error(f"ARIMA forecast error: {e}")
            return None

    @staticmethod
    def volatility(series, window=20, annualize=True, periods_per_year=252):
        """Calculate rolling volatility (standard deviation)."""
        try:
            vol = series.rolling(window=window).std()
            if annualize:
                vol = vol * np.sqrt(periods_per_year)
            return vol
        except Exception as e:
            logging.error(f"Volatility calculation error: {e}")
            return None

    @staticmethod
    def cross_correlation(series1, series2, lag=0):
        """Compute cross-correlation between two series at a given lag."""
        try:
            if lag > 0:
                return series1[lag:].corr(series2[:-lag])
            elif lag < 0:
                return series1[:lag].corr(series2[-lag:])
            else:
                return series1.corr(series2)
        except Exception as e:
            logging.error(f"Cross-correlation error: {e}")
            return None

    @staticmethod
    def detect_outliers_zscore(series, threshold=3):
        """Detect outliers using z-score method."""
        try:
            zscores = (series - series.mean()) / series.std()
            outliers = np.abs(zscores) > threshold
            return outliers
        except Exception as e:
            logging.error(f"Outlier detection error: {e}")
            return None