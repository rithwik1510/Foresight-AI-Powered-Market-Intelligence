"""
Global Features - Feature engineering from global market data
Generates features from US markets, commodities, forex, and economic indicators
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
import asyncio

from app.integrations.global_markets import global_markets_client, GlobalMarketData
from app.integrations.economic_data import economic_data_client


class GlobalFeatureGenerator:
    """
    Generates features from global market data for ML models

    Features include:
    - US market returns (S&P500, NASDAQ) with various lags
    - Commodity price changes (Gold, Oil)
    - Forex movements (USD/INR, Dollar Index)
    - Volatility indicators (VIX)
    - Economic regime indicators
    """

    def __init__(self):
        self.global_client = global_markets_client
        self.economic_client = economic_data_client

    async def generate_features_async(
        self,
        include_economic: bool = True
    ) -> Dict[str, float]:
        """
        Generate current global features (async)

        Args:
            include_economic: Include FRED economic indicators

        Returns:
            Dictionary of feature name -> value
        """
        features = {}

        # Get global market data
        market_data = await self.global_client.get_current_data()
        features.update(self._extract_market_features(market_data))

        # Get market regime
        regime = await self.global_client.get_market_regime()
        features.update(self._extract_regime_features(regime))

        # Get economic indicators if available
        if include_economic and self.economic_client.is_available():
            indicators = await self.economic_client.get_current_indicators()
            if indicators:
                features.update(self._extract_economic_features(indicators))

            fed_outlook = await self.economic_client.get_fed_outlook()
            features.update(self._extract_fed_features(fed_outlook))

        return features

    def _extract_market_features(self, data: GlobalMarketData) -> Dict[str, float]:
        """Extract features from market data"""
        return {
            # US Markets
            "sp500_return_1d": data.sp500_change_1d,
            "sp500_return_5d": data.sp500_change_5d,
            "nasdaq_return_1d": data.nasdaq_change_1d,
            "nasdaq_return_5d": data.nasdaq_change_5d,

            # Volatility
            "vix_level": data.vix_price,
            "vix_high": 1.0 if data.vix_price > 25 else 0.0,
            "vix_low": 1.0 if data.vix_price < 15 else 0.0,

            # Commodities
            "gold_return_1d": data.gold_change_1d,
            "oil_return_1d": data.oil_change_1d,

            # Forex
            "usdinr_change_1d": data.usdinr_change_1d,
            "dxy_change_1d": data.dxy_change_1d,
            "rupee_weak": 1.0 if data.usdinr_change_1d > 0.003 else 0.0,
            "rupee_strong": 1.0 if data.usdinr_change_1d < -0.003 else 0.0,

            # Bonds
            "us10y_yield": data.us10y_yield,
            "us10y_change_1d": data.us10y_change_1d,
        }

    def _extract_regime_features(self, regime: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from market regime"""
        regime_name = regime.get("regime", "NEUTRAL")

        return {
            "regime_risk_on": 1.0 if regime_name == "RISK_ON" else 0.0,
            "regime_risk_off": 1.0 if regime_name == "RISK_OFF" else 0.0,
            "regime_neutral": 1.0 if regime_name == "NEUTRAL" else 0.0,
            "regime_confidence": regime.get("confidence", 0.5),
            "regime_risk_on_signals": regime.get("signals", {}).get("risk_on", 0),
            "regime_risk_off_signals": regime.get("signals", {}).get("risk_off", 0),
        }

    def _extract_economic_features(self, indicators) -> Dict[str, float]:
        """Extract features from economic indicators"""
        return {
            # Interest rates
            "fed_funds_rate": indicators.fed_funds_rate,
            "fed_funds_change": indicators.fed_funds_change_1m,

            # Inflation
            "inflation_cpi": indicators.cpi_yoy,
            "inflation_pce": indicators.pce_yoy,
            "inflation_high": 1.0 if indicators.cpi_yoy > 3.0 else 0.0,
            "inflation_low": 1.0 if indicators.cpi_yoy < 2.0 else 0.0,

            # Employment
            "unemployment_rate": indicators.unemployment_rate,
            "payrolls_change": indicators.nonfarm_payrolls_change,
            "employment_strong": 1.0 if indicators.unemployment_rate < 4.0 else 0.0,

            # GDP
            "gdp_growth": indicators.gdp_growth_rate,
            "gdp_positive": 1.0 if indicators.gdp_growth_rate > 0 else 0.0,

            # Yield curve
            "yield_curve_spread": indicators.yield_curve_spread,
            "yield_curve_inverted": 1.0 if indicators.yield_curve_spread < 0 else 0.0,

            # Consumer sentiment
            "consumer_sentiment": indicators.consumer_sentiment,
            "consumer_optimistic": 1.0 if indicators.consumer_sentiment > 90 else 0.0,
        }

    def _extract_fed_features(self, outlook: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from Fed outlook"""
        outlook_name = outlook.get("outlook", "NEUTRAL")

        return {
            "fed_hawkish": 1.0 if outlook_name == "HAWKISH" else 0.0,
            "fed_dovish": 1.0 if outlook_name == "DOVISH" else 0.0,
            "fed_neutral": 1.0 if outlook_name == "NEUTRAL" else 0.0,
            "fed_hawkish_signals": outlook.get("hawkish_signals", 0),
            "fed_dovish_signals": outlook.get("dovish_signals", 0),
        }

    async def generate_historical_features(
        self,
        period: str = "2y"
    ) -> pd.DataFrame:
        """
        Generate historical global features for training

        Args:
            period: Historical period

        Returns:
            DataFrame with global features
        """
        # Get historical market data
        market_data = await self.global_client.get_historical_data(period=period)

        if not market_data:
            return pd.DataFrame()

        # Build features DataFrame
        features = pd.DataFrame()

        # S&P500 features
        if "sp500" in market_data:
            sp500 = market_data["sp500"]['Close']
            features["sp500_return_1d"] = sp500.pct_change(1)
            features["sp500_return_5d"] = sp500.pct_change(5)
            features["sp500_return_20d"] = sp500.pct_change(20)
            features["sp500_sma_20"] = sp500.rolling(20).mean() / sp500 - 1
            features["sp500_volatility_20d"] = sp500.pct_change().rolling(20).std()

        # NASDAQ features
        if "nasdaq" in market_data:
            nasdaq = market_data["nasdaq"]['Close']
            features["nasdaq_return_1d"] = nasdaq.pct_change(1)
            features["nasdaq_return_5d"] = nasdaq.pct_change(5)

        # VIX features
        if "vix" in market_data:
            vix = market_data["vix"]['Close']
            features["vix_level"] = vix
            features["vix_change_1d"] = vix.pct_change(1)
            features["vix_sma_10"] = vix.rolling(10).mean()
            features["vix_high"] = (vix > 25).astype(float)
            features["vix_low"] = (vix < 15).astype(float)

        # Gold features
        if "gold" in market_data:
            gold = market_data["gold"]['Close']
            features["gold_return_1d"] = gold.pct_change(1)
            features["gold_return_5d"] = gold.pct_change(5)
            features["gold_sma_20"] = gold.rolling(20).mean() / gold - 1

        # Oil features
        if "oil" in market_data:
            oil = market_data["oil"]['Close']
            features["oil_return_1d"] = oil.pct_change(1)
            features["oil_return_5d"] = oil.pct_change(5)
            features["oil_volatility_20d"] = oil.pct_change().rolling(20).std()

        # USD/INR features
        if "usdinr" in market_data:
            usdinr = market_data["usdinr"]['Close']
            features["usdinr_change_1d"] = usdinr.pct_change(1)
            features["usdinr_change_5d"] = usdinr.pct_change(5)
            features["rupee_weak"] = (usdinr.pct_change(1) > 0.003).astype(float)
            features["rupee_strong"] = (usdinr.pct_change(1) < -0.003).astype(float)

        # Dollar Index features
        if "dxy" in market_data:
            dxy = market_data["dxy"]['Close']
            features["dxy_change_1d"] = dxy.pct_change(1)
            features["dxy_change_5d"] = dxy.pct_change(5)

        # US 10Y Treasury features
        if "us10y" in market_data:
            us10y = market_data["us10y"]['Close']
            features["us10y_yield"] = us10y
            features["us10y_change_1d"] = us10y.diff(1)
            features["us10y_change_5d"] = us10y.diff(5)

        # Add lagged features (global markets affect India next day)
        lag_columns = [
            "sp500_return_1d", "nasdaq_return_1d", "vix_level",
            "gold_return_1d", "oil_return_1d", "dxy_change_1d"
        ]

        for col in lag_columns:
            if col in features.columns:
                features[f"{col}_lag1"] = features[col].shift(1)
                features[f"{col}_lag2"] = features[col].shift(2)

        # Add interaction features
        if "sp500_return_1d" in features.columns and "vix_level" in features.columns:
            features["sp500_vix_interaction"] = features["sp500_return_1d"] * features["vix_level"]

        if "oil_return_1d" in features.columns and "usdinr_change_1d" in features.columns:
            features["oil_inr_interaction"] = features["oil_return_1d"] * features["usdinr_change_1d"]

        return features

    def align_features_with_stock(
        self,
        global_features: pd.DataFrame,
        stock_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Align global features with stock data

        Args:
            global_features: DataFrame with global features
            stock_df: DataFrame with stock data

        Returns:
            Aligned DataFrame
        """
        # Align indices
        aligned = global_features.reindex(stock_df.index)

        # Forward fill missing values (weekends, holidays)
        aligned = aligned.ffill()

        # Backward fill any remaining NaN at start
        aligned = aligned.bfill()

        return aligned

    async def get_india_impact_features(self) -> Dict[str, float]:
        """
        Get features specific to India market impact

        Returns:
            Dictionary of India-specific impact features
        """
        impact = await self.global_client.get_indian_market_impact()

        features = {
            "global_impact_score": impact.get("net_impact_score", 0.0),
            "global_impact_positive": 1.0 if impact.get("overall_impact") == "POSITIVE" else 0.0,
            "global_impact_negative": 1.0 if impact.get("overall_impact") == "NEGATIVE" else 0.0,
        }

        # Add individual factor impacts
        for factor in impact.get("impact_factors", []):
            factor_name = factor["factor"].lower().replace(" ", "_")
            features[f"impact_{factor_name}"] = factor["magnitude"] * (
                1.0 if factor["impact"] == "positive" else -1.0
            )

        return features

    def get_feature_names(self) -> List[str]:
        """Get list of all possible global feature names"""
        return [
            # Market features
            "sp500_return_1d", "sp500_return_5d", "sp500_return_20d",
            "sp500_sma_20", "sp500_volatility_20d",
            "nasdaq_return_1d", "nasdaq_return_5d",
            "vix_level", "vix_change_1d", "vix_sma_10", "vix_high", "vix_low",
            "gold_return_1d", "gold_return_5d", "gold_sma_20",
            "oil_return_1d", "oil_return_5d", "oil_volatility_20d",
            "usdinr_change_1d", "usdinr_change_5d", "rupee_weak", "rupee_strong",
            "dxy_change_1d", "dxy_change_5d",
            "us10y_yield", "us10y_change_1d", "us10y_change_5d",

            # Lagged features
            "sp500_return_1d_lag1", "sp500_return_1d_lag2",
            "nasdaq_return_1d_lag1", "nasdaq_return_1d_lag2",
            "vix_level_lag1", "vix_level_lag2",
            "gold_return_1d_lag1", "gold_return_1d_lag2",
            "oil_return_1d_lag1", "oil_return_1d_lag2",
            "dxy_change_1d_lag1", "dxy_change_1d_lag2",

            # Interaction features
            "sp500_vix_interaction", "oil_inr_interaction",

            # Regime features
            "regime_risk_on", "regime_risk_off", "regime_neutral",
            "regime_confidence", "regime_risk_on_signals", "regime_risk_off_signals",

            # Economic features
            "fed_funds_rate", "fed_funds_change",
            "inflation_cpi", "inflation_pce", "inflation_high", "inflation_low",
            "unemployment_rate", "payrolls_change", "employment_strong",
            "gdp_growth", "gdp_positive",
            "yield_curve_spread", "yield_curve_inverted",
            "consumer_sentiment", "consumer_optimistic",

            # Fed outlook features
            "fed_hawkish", "fed_dovish", "fed_neutral",
            "fed_hawkish_signals", "fed_dovish_signals",

            # India impact features
            "global_impact_score", "global_impact_positive", "global_impact_negative",
        ]


# Global instance
global_feature_generator = GlobalFeatureGenerator()
