import numpy as np
import logging

class PositionSizer:
    @staticmethod
    def fixed_fractional(capital, risk_per_trade):
        """Position size as a fixed fraction of total capital."""
        try:
            size = capital * risk_per_trade
            return size
        except Exception as e:
            logging.error(f"Fixed fractional sizing error: {e}")
            return None

    @staticmethod
    def volatility_position_size(capital, entry_price, stop_price, volatility, risk_per_trade):
        """Size position based on volatility and stop distance."""
        try:
            dollar_risk = abs(entry_price - stop_price)
            if dollar_risk == 0:
                return 0
            size = (capital * risk_per_trade) / dollar_risk
            return size
        except Exception as e:
            logging.error(f"Volatility position sizing error: {e}")
            return None

    @staticmethod
    def atr_position_size(capital, entry_price, atr, risk_per_trade, atr_multiplier=1):
        """Size position based on ATR (Average True Range)."""
        try:
            dollar_risk = atr * atr_multiplier
            if dollar_risk == 0:
                return 0
            size = (capital * risk_per_trade) / dollar_risk
            return size
        except Exception as e:
            logging.error(f"ATR position sizing error: {e}")
            return None

    @staticmethod
    def max_position_limit(size, max_position):
        """Limit position size to a maximum allowed."""
        try:
            return min(size, max_position)
        except Exception as e:
            logging.error(f"Max position limit error: {e}")
            return size

    @staticmethod
    def Kelly_criterion(win_rate, win_loss_ratio):
        """Calculate optimal fraction to bet using Kelly criterion."""
        try:
            kelly = win_rate - (1 - win_rate) / win_loss_ratio
            return max(0, kelly)
        except Exception as e:
            logging.error(f"Kelly criterion error: {e}")
            return 0

    @staticmethod
    def dollar_position_size(capital, dollar_per_trade):
        """Position size as a fixed dollar amount per trade."""
        try:
            size = min(capital, dollar_per_trade)
            return size
        except Exception as e:
            logging.error(f"Dollar position sizing error: {e}")
            return None