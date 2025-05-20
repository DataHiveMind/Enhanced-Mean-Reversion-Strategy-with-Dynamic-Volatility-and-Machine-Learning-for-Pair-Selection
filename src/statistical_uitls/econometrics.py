import numpy as np
import pandas as pd
import statsmodels.api as sm
import logging

class Econometrics:
    @staticmethod
    def ols_regression(y, X, add_constant=True):
        """
        Ordinary Least Squares regression.
        y: dependent variable (array-like or pd.Series)
        X: independent variables (array-like or pd.DataFrame)
        add_constant: whether to add intercept term
        Returns: statsmodels regression result
        """
        try:
            if add_constant:
                X = sm.add_constant(X)
            model = sm.OLS(y, X, missing='drop')
            results = model.fit()
            return results
        except Exception as e:
            logging.error(f"OLS regression error: {e}")
            return None

    @staticmethod
    def adf_test(series, maxlag=None, regression='c'):
        """
        Augmented Dickey-Fuller test for stationarity.
        series: pd.Series
        Returns: test statistic, p-value, used lags, nobs, critical values, icbest
        """
        try:
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(series.dropna(), maxlag=maxlag, regression=regression, autolag='AIC')
            return {
                'adf_statistic': result[0],
                'p_value': result[1],
                'used_lag': result[2],
                'nobs': result[3],
                'critical_values': result[4],
                'icbest': result[5]
            }
        except Exception as e:
            logging.error(f"ADF test error: {e}")
            return None

    @staticmethod
    def johansen_test(df, det_order=0, k_ar_diff=1):
        """
        Johansen cointegration test.
        df: pd.DataFrame with multiple time series
        Returns: statsmodels Johansen test result
        """
        try:
            from statsmodels.tsa.vector_ar.vecm import coint_johansen
            result = coint_johansen(df.dropna(), det_order, k_ar_diff)
            return result
        except Exception as e:
            logging.error(f"Johansen test error: {e}")
            return None

    @staticmethod
    def calculate_residuals(y, X, add_constant=True):
        """
        Calculate residuals from OLS regression.
        """
        results = Econometrics.ols_regression(y, X, add_constant=add_constant)
        if results is not None:
            return results.resid
        else:
            return None