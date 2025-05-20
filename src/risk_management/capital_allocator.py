import numpy as np
import pandas as pd
import logging

class CapitalAllocator:
    @staticmethod
    def equal_weight_allocation(n_assets, total_capital):
        """Allocate capital equally among all assets."""
        try:
            allocation = np.full(n_assets, total_capital / n_assets)
            return allocation
        except Exception as e:
            logging.error(f"Equal weight allocation error: {e}")
            return None

    @staticmethod
    def volatility_weighted_allocation(returns, total_capital):
        """Allocate more capital to assets with lower volatility."""
        try:
            vol = returns.std()
            inv_vol = 1 / vol
            weights = inv_vol / inv_vol.sum()
            allocation = weights * total_capital
            return allocation
        except Exception as e:
            logging.error(f"Volatility weighted allocation error: {e}")
            return None

    @staticmethod
    def risk_parity_allocation(returns, total_capital):
        """Allocate capital so each asset contributes equally to portfolio risk."""
        try:
            vol = returns.std()
            inv_vol = 1 / vol
            weights = inv_vol / inv_vol.sum()
            allocation = weights * total_capital
            return allocation
        except Exception as e:
            logging.error(f"Risk parity allocation error: {e}")
            return None

    @staticmethod
    def max_drawdown_control(portfolio_values, max_drawdown=0.2):
        """Check if portfolio drawdown exceeds max_drawdown threshold."""
        try:
            running_max = np.maximum.accumulate(portfolio_values)
            drawdown = (portfolio_values - running_max) / running_max
            exceed = np.min(drawdown) < -max_drawdown
            return exceed, np.min(drawdown)
        except Exception as e:
            logging.error(f"Max drawdown control error: {e}")
            return None, None

    @staticmethod
    def dynamic_leverage(target_vol, realized_vol, base_leverage=1.0, max_leverage=3.0):
        """Adjust leverage based on realized volatility to target a specific volatility."""
        try:
            leverage = base_leverage * (target_vol / realized_vol)
            leverage = np.clip(leverage, 0, max_leverage)
            return leverage
        except Exception as e:
            logging.error(f"Dynamic leverage error: {e}")
            return base_leverage

    @staticmethod
    def capital_allocation_limit(allocation, max_allocation_per_asset):
        """Enforce a maximum allocation per asset."""
        try:
            capped_allocation = np.minimum(allocation, max_allocation_per_asset)
            return capped_allocation
        except Exception as e:
            logging.error(f"Capital allocation limit error: {e}")
            return allocation