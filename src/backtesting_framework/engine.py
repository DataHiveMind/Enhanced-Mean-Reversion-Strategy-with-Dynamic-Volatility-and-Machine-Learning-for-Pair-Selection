import pandas as pd
import numpy as np
import logging

class BacktestEngine:
    def __init__(self, data, strategy, initial_capital=100000, commission=0.0, slippage=0.0):
        """
        data: pd.DataFrame with price data (must include 'close' column)
        strategy: a callable that generates signals (expects data, returns pd.Series of signals)
        initial_capital: starting capital for the backtest
        commission: commission per trade (as a fraction, e.g., 0.001 for 0.1%)
        slippage: slippage per trade (as a fraction)
        """
        self.data = data.copy()
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.results = None

    def run(self):
        """
        Run the backtest using the provided strategy.
        """
        try:
            self.data['signal'] = self.strategy(self.data)
            self.data['position'] = self.data['signal'].shift(1).fillna(0)
            self.data['returns'] = self.data['close'].pct_change().fillna(0)
            self.data['strategy_returns'] = self.data['position'] * self.data['returns']

            # Apply commission and slippage on trades
            trades = self.data['position'].diff().abs()
            self.data['strategy_returns'] -= trades * (self.commission + self.slippage)

            self.data['equity_curve'] = self.initial_capital * (1 + self.data['strategy_returns']).cumprod()
            self.results = self.data
            return self.data
        except Exception as e:
            logging.error(f"Backtest run error: {e}")
            return None

    def summary(self):
        """
        Return a summary of the backtest results.
        """
        if self.results is None:
            logging.warning("No results to summarize. Run the backtest first.")
            return None

        total_return = self.results['equity_curve'].iloc[-1] / self.initial_capital - 1
        annualized_return = (1 + total_return) ** (252 / len(self.results)) - 1
        annualized_vol = self.results['strategy_returns'].std() * np.sqrt(252)
        sharpe = annualized_return / annualized_vol if annualized_vol != 0 else np.nan
        max_drawdown = self._max_drawdown(self.results['equity_curve'])

        return {
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Annualized Volatility': annualized_vol,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_drawdown
        }

    @staticmethod
    def _max_drawdown(equity_curve):
        """
        Calculate the maximum drawdown of an equity curve.
        """
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        return drawdown.min()