"""
Portfolio optimization using Hierarchical Risk Parity (HRP)
"""
import pandas as pd
import numpy as np
from pypfopt import HRPOpt, risk_models
from typing import Dict, List, Optional
from decimal import Decimal


class PortfolioOptimizer:
    """Portfolio optimization using HRP algorithm"""

    @staticmethod
    def optimize_hrp(returns_df: pd.DataFrame) -> Dict[str, float]:
        """
        Optimize portfolio using Hierarchical Risk Parity

        Args:
            returns_df: DataFrame with returns for each asset (columns = assets, rows = dates)

        Returns:
            Dictionary of asset weights
        """
        try:
            # Initialize HRP optimizer
            hrp = HRPOpt(returns_df)

            # Optimize
            weights = hrp.optimize()

            # Clean weights (remove very small allocations)
            cleaned_weights = hrp.clean_weights(cutoff=0.01)

            return dict(cleaned_weights)

        except Exception as e:
            print(f"Error in HRP optimization: {e}")
            # Return equal weights as fallback
            n_assets = len(returns_df.columns)
            equal_weight = 1.0 / n_assets
            return {asset: equal_weight for asset in returns_df.columns}

    @staticmethod
    def calculate_returns_series(
        prices_df: pd.DataFrame,
        method: str = "log"
    ) -> pd.DataFrame:
        """
        Calculate returns from price data

        Args:
            prices_df: DataFrame with prices for each asset
            method: "simple" or "log" returns

        Returns:
            DataFrame of returns
        """
        if method == "log":
            returns = np.log(prices_df / prices_df.shift(1))
        else:
            returns = prices_df.pct_change()

        # Drop NaN values
        returns = returns.dropna()

        return returns

    @staticmethod
    def rebalance_portfolio(
        current_holdings: Dict[str, Decimal],
        target_weights: Dict[str, float],
        total_value: Decimal
    ) -> Dict[str, Dict[str, Decimal]]:
        """
        Calculate rebalancing transactions

        Args:
            current_holdings: Current quantities {symbol: quantity}
            target_weights: Target weights {symbol: weight}
            total_value: Total portfolio value

        Returns:
            Dict with 'buy' and 'sell' transactions
        """
        transactions = {"buy": {}, "sell": {}}

        current_weights = {}
        for symbol, qty in current_holdings.items():
            weight = (qty / total_value) if total_value > 0 else 0
            current_weights[symbol] = float(weight)

        for symbol, target_weight in target_weights.items():
            current_weight = current_weights.get(symbol, 0)
            weight_diff = target_weight - current_weight
            value_diff = Decimal(str(weight_diff)) * total_value

            if weight_diff > 0.01:  # Buy threshold
                transactions["buy"][symbol] = value_diff
            elif weight_diff < -0.01:  # Sell threshold
                transactions["sell"][symbol] = abs(value_diff)

        return transactions


class RiskMetrics:
    """Calculate portfolio risk metrics"""

    @staticmethod
    def calculate_var(
        returns: pd.Series,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Value at Risk (VaR)

        Args:
            returns: Series of returns
            confidence_level: Confidence level (default 95%)

        Returns:
            VaR value
        """
        return float(np.percentile(returns, (1 - confidence_level) * 100))

    @staticmethod
    def calculate_cvar(
        returns: pd.Series,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall

        Args:
            returns: Series of returns
            confidence_level: Confidence level (default 95%)

        Returns:
            CVaR value
        """
        var = RiskMetrics.calculate_var(returns, confidence_level)
        # CVaR is the average of returns below VaR
        cvar = returns[returns <= var].mean()
        return float(cvar)

    @staticmethod
    def calculate_sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.06  # 6% for India
    ) -> float:
        """
        Calculate Sharpe Ratio

        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate (default 6% for India)

        Returns:
            Sharpe ratio
        """
        # Annualize returns and std
        annual_return = returns.mean() * 252
        annual_std = returns.std() * np.sqrt(252)

        if annual_std == 0:
            return 0.0

        sharpe = (annual_return - risk_free_rate) / annual_std
        return float(sharpe)

    @staticmethod
    def calculate_sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.06
    ) -> float:
        """
        Calculate Sortino Ratio (uses downside deviation)

        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate

        Returns:
            Sortino ratio
        """
        # Annualize returns
        annual_return = returns.mean() * 252

        # Downside deviation (only negative returns)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)

        if downside_std == 0:
            return 0.0

        sortino = (annual_return - risk_free_rate) / downside_std
        return float(sortino)

    @staticmethod
    def calculate_beta(
        asset_returns: pd.Series,
        market_returns: pd.Series
    ) -> float:
        """
        Calculate Beta (market correlation)

        Args:
            asset_returns: Asset returns series
            market_returns: Market returns series (e.g., Nifty 50)

        Returns:
            Beta value
        """
        # Align series
        combined = pd.DataFrame({
            "asset": asset_returns,
            "market": market_returns
        }).dropna()

        if len(combined) < 2:
            return 1.0  # Default beta

        covariance = combined["asset"].cov(combined["market"])
        market_variance = combined["market"].var()

        if market_variance == 0:
            return 1.0

        beta = covariance / market_variance
        return float(beta)

    @staticmethod
    def calculate_max_drawdown(prices: pd.Series) -> float:
        """
        Calculate maximum drawdown

        Args:
            prices: Price series

        Returns:
            Maximum drawdown percentage
        """
        # Calculate running maximum
        running_max = prices.expanding().max()

        # Calculate drawdown
        drawdown = (prices - running_max) / running_max

        # Get maximum drawdown
        max_dd = drawdown.min()

        return float(max_dd)

    @staticmethod
    def calculate_volatility(returns: pd.Series, annualize: bool = True) -> float:
        """
        Calculate volatility (standard deviation of returns)

        Args:
            returns: Returns series
            annualize: Whether to annualize the volatility

        Returns:
            Volatility value
        """
        vol = returns.std()

        if annualize:
            vol = vol * np.sqrt(252)

        return float(vol)


# Global instances
portfolio_optimizer = PortfolioOptimizer()
risk_metrics = RiskMetrics()
