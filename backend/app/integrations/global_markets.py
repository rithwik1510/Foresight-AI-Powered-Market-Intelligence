"""
Global Markets Integration - S&P500, Gold, Oil, Forex via yfinance
Fetches data that affects Indian market
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
import asyncio


@dataclass
class GlobalMarketData:
    """Global market data snapshot"""
    timestamp: datetime

    # US Markets
    sp500_price: float
    sp500_change_1d: float
    sp500_change_5d: float
    nasdaq_price: float
    nasdaq_change_1d: float
    nasdaq_change_5d: float
    vix_price: float  # Volatility index

    # Commodities
    gold_price: float
    gold_change_1d: float
    oil_price: float  # Brent crude
    oil_change_1d: float

    # Forex
    usdinr_rate: float
    usdinr_change_1d: float
    dxy_price: float  # US Dollar Index
    dxy_change_1d: float

    # Bonds
    us10y_yield: float
    us10y_change_1d: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "us_markets": {
                "sp500": {"price": self.sp500_price, "change_1d": self.sp500_change_1d, "change_5d": self.sp500_change_5d},
                "nasdaq": {"price": self.nasdaq_price, "change_1d": self.nasdaq_change_1d, "change_5d": self.nasdaq_change_5d},
                "vix": self.vix_price,
            },
            "commodities": {
                "gold": {"price": self.gold_price, "change_1d": self.gold_change_1d},
                "oil_brent": {"price": self.oil_price, "change_1d": self.oil_change_1d},
            },
            "forex": {
                "usdinr": {"rate": self.usdinr_rate, "change_1d": self.usdinr_change_1d},
                "dxy": {"price": self.dxy_price, "change_1d": self.dxy_change_1d},
            },
            "bonds": {
                "us10y": {"yield": self.us10y_yield, "change_1d": self.us10y_change_1d},
            },
        }


class GlobalMarketsClient:
    """
    Client for fetching global market data via yfinance

    Tickers used:
    - US Markets: ^GSPC (S&P500), ^IXIC (NASDAQ), ^VIX (Volatility)
    - Commodities: GC=F (Gold Futures), BZ=F (Brent Crude)
    - Forex: USDINR=X, DX-Y.NYB (Dollar Index)
    - Bonds: ^TNX (10-Year Treasury)
    """

    # Ticker symbols
    TICKERS = {
        "sp500": "^GSPC",
        "nasdaq": "^IXIC",
        "vix": "^VIX",
        "gold": "GC=F",
        "oil": "BZ=F",
        "usdinr": "USDINR=X",
        "dxy": "DX-Y.NYB",
        "us10y": "^TNX",
    }

    def __init__(self):
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_time: Dict[str, datetime] = {}
        self._cache_ttl = 300  # 5 minutes

    def _fetch_ticker_data(
        self,
        symbol: str,
        period: str = "1mo"
    ) -> Optional[pd.DataFrame]:
        """Fetch data for a single ticker"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)

            if df.empty:
                return None

            return df

        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None

    def _calculate_change(
        self,
        df: pd.DataFrame,
        days: int = 1
    ) -> float:
        """Calculate percentage change over N days"""
        if df is None or len(df) < days + 1:
            return 0.0

        try:
            current = df['Close'].iloc[-1]
            previous = df['Close'].iloc[-(days + 1)]
            return (current - previous) / previous
        except:
            return 0.0

    async def get_current_data(self) -> GlobalMarketData:
        """
        Get current global market snapshot

        Returns:
            GlobalMarketData object with current prices and changes
        """
        # Fetch all tickers
        data = {}

        for name, symbol in self.TICKERS.items():
            # Check cache
            cache_key = symbol
            if cache_key in self._cache:
                cache_age = (datetime.now() - self._cache_time[cache_key]).seconds
                if cache_age < self._cache_ttl:
                    data[name] = self._cache[cache_key]
                    continue

            # Fetch fresh data
            df = self._fetch_ticker_data(symbol, period="1mo")
            if df is not None:
                data[name] = df
                self._cache[cache_key] = df
                self._cache_time[cache_key] = datetime.now()

            # Small delay to respect rate limits
            await asyncio.sleep(0.2)

        # Build GlobalMarketData
        return GlobalMarketData(
            timestamp=datetime.now(),
            # US Markets
            sp500_price=self._get_latest_price(data.get("sp500")),
            sp500_change_1d=self._calculate_change(data.get("sp500"), 1),
            sp500_change_5d=self._calculate_change(data.get("sp500"), 5),
            nasdaq_price=self._get_latest_price(data.get("nasdaq")),
            nasdaq_change_1d=self._calculate_change(data.get("nasdaq"), 1),
            nasdaq_change_5d=self._calculate_change(data.get("nasdaq"), 5),
            vix_price=self._get_latest_price(data.get("vix")),
            # Commodities
            gold_price=self._get_latest_price(data.get("gold")),
            gold_change_1d=self._calculate_change(data.get("gold"), 1),
            oil_price=self._get_latest_price(data.get("oil")),
            oil_change_1d=self._calculate_change(data.get("oil"), 1),
            # Forex
            usdinr_rate=self._get_latest_price(data.get("usdinr")),
            usdinr_change_1d=self._calculate_change(data.get("usdinr"), 1),
            dxy_price=self._get_latest_price(data.get("dxy")),
            dxy_change_1d=self._calculate_change(data.get("dxy"), 1),
            # Bonds
            us10y_yield=self._get_latest_price(data.get("us10y")),
            us10y_change_1d=self._calculate_change(data.get("us10y"), 1),
        )

    def _get_latest_price(self, df: Optional[pd.DataFrame]) -> float:
        """Get latest closing price"""
        if df is None or df.empty:
            return 0.0
        try:
            return float(df['Close'].iloc[-1])
        except:
            return 0.0

    async def get_historical_data(
        self,
        period: str = "2y"
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for all global factors

        Args:
            period: Data period (1mo, 3mo, 6mo, 1y, 2y, 5y)

        Returns:
            Dictionary of DataFrames for each ticker
        """
        data = {}

        for name, symbol in self.TICKERS.items():
            df = self._fetch_ticker_data(symbol, period=period)
            if df is not None:
                data[name] = df
            await asyncio.sleep(0.2)

        return data

    async def get_correlation_data(
        self,
        indian_stock_symbol: str,
        period: str = "1y"
    ) -> Dict[str, float]:
        """
        Calculate correlation between Indian stock and global factors

        Args:
            indian_stock_symbol: Indian stock symbol (e.g., RELIANCE.NS)
            period: Period for correlation calculation

        Returns:
            Dictionary of correlations
        """
        # Fetch Indian stock data
        indian_df = self._fetch_ticker_data(indian_stock_symbol, period=period)

        if indian_df is None:
            return {}

        indian_returns = indian_df['Close'].pct_change().dropna()

        # Fetch global data
        global_data = await self.get_historical_data(period=period)

        correlations = {}

        for name, df in global_data.items():
            if df is None or df.empty:
                continue

            try:
                global_returns = df['Close'].pct_change().dropna()

                # Align dates
                aligned = pd.concat([indian_returns, global_returns], axis=1, join='inner')
                aligned.columns = ['indian', 'global']

                if len(aligned) > 30:  # Need enough data points
                    corr = aligned['indian'].corr(aligned['global'])
                    correlations[name] = round(corr, 4)

            except Exception as e:
                print(f"Correlation error for {name}: {e}")
                continue

        return correlations

    async def get_market_regime(self) -> Dict[str, Any]:
        """
        Determine current market regime based on global factors

        Returns:
            Dictionary with regime classification
        """
        data = await self.get_current_data()

        # Classify regime based on multiple factors
        signals = {
            "risk_on": 0,
            "risk_off": 0,
        }

        # VIX below 20 = risk on, above 25 = risk off
        if data.vix_price < 20:
            signals["risk_on"] += 2
        elif data.vix_price > 25:
            signals["risk_off"] += 2
        elif data.vix_price > 30:
            signals["risk_off"] += 3

        # S&P500 positive = risk on
        if data.sp500_change_5d > 0.01:
            signals["risk_on"] += 1
        elif data.sp500_change_5d < -0.02:
            signals["risk_off"] += 2

        # Gold rising with falling equity = risk off
        if data.gold_change_1d > 0.01 and data.sp500_change_1d < 0:
            signals["risk_off"] += 1

        # Oil rising = inflationary
        if data.oil_change_1d > 0.03:
            signals["risk_off"] += 1

        # Dollar strength = risk off for emerging markets
        if data.dxy_change_1d > 0.005:
            signals["risk_off"] += 1
        elif data.dxy_change_1d < -0.005:
            signals["risk_on"] += 1

        # USD/INR weakening = risk on for Indian stocks
        if data.usdinr_change_1d < -0.002:
            signals["risk_on"] += 1
        elif data.usdinr_change_1d > 0.005:
            signals["risk_off"] += 1

        # Rising yields = risk off (growth stocks hurt)
        if data.us10y_change_1d > 0.02:  # 2bps increase
            signals["risk_off"] += 1

        # Determine regime
        if signals["risk_on"] > signals["risk_off"] + 2:
            regime = "RISK_ON"
            confidence = signals["risk_on"] / (signals["risk_on"] + signals["risk_off"])
        elif signals["risk_off"] > signals["risk_on"] + 2:
            regime = "RISK_OFF"
            confidence = signals["risk_off"] / (signals["risk_on"] + signals["risk_off"])
        else:
            regime = "NEUTRAL"
            confidence = 0.5

        return {
            "regime": regime,
            "confidence": min(1.0, confidence),
            "signals": signals,
            "key_factors": {
                "vix": data.vix_price,
                "sp500_5d": data.sp500_change_5d,
                "gold_1d": data.gold_change_1d,
                "usdinr_1d": data.usdinr_change_1d,
                "dxy_1d": data.dxy_change_1d,
            },
            "timestamp": datetime.now().isoformat(),
        }

    async def get_indian_market_impact(self) -> Dict[str, Any]:
        """
        Analyze impact of global factors on Indian market

        Returns:
            Dictionary with impact analysis
        """
        data = await self.get_current_data()
        regime = await self.get_market_regime()

        impact_factors = []

        # Oil impact (India is net importer)
        if data.oil_change_1d > 0.02:
            impact_factors.append({
                "factor": "Oil Price Rise",
                "impact": "negative",
                "description": "Rising oil prices hurt Indian trade deficit and inflation",
                "magnitude": min(1.0, abs(data.oil_change_1d) * 10),
            })
        elif data.oil_change_1d < -0.02:
            impact_factors.append({
                "factor": "Oil Price Decline",
                "impact": "positive",
                "description": "Falling oil prices benefit Indian economy",
                "magnitude": min(1.0, abs(data.oil_change_1d) * 10),
            })

        # Rupee impact
        if data.usdinr_change_1d > 0.005:
            impact_factors.append({
                "factor": "Rupee Weakness",
                "impact": "negative",
                "description": "Weakening rupee indicates FII outflows",
                "magnitude": min(1.0, abs(data.usdinr_change_1d) * 50),
            })
        elif data.usdinr_change_1d < -0.003:
            impact_factors.append({
                "factor": "Rupee Strength",
                "impact": "positive",
                "description": "Strengthening rupee indicates FII inflows",
                "magnitude": min(1.0, abs(data.usdinr_change_1d) * 50),
            })

        # US market correlation
        if data.sp500_change_1d > 0.01:
            impact_factors.append({
                "factor": "US Market Rally",
                "impact": "positive",
                "description": "Positive US session typically leads Indian markets higher",
                "magnitude": min(1.0, abs(data.sp500_change_1d) * 20),
            })
        elif data.sp500_change_1d < -0.01:
            impact_factors.append({
                "factor": "US Market Decline",
                "impact": "negative",
                "description": "Negative US session typically weighs on Indian markets",
                "magnitude": min(1.0, abs(data.sp500_change_1d) * 20),
            })

        # VIX fear gauge
        if data.vix_price > 25:
            impact_factors.append({
                "factor": "High VIX",
                "impact": "negative",
                "description": "Elevated fear in global markets, expect volatility",
                "magnitude": min(1.0, (data.vix_price - 20) / 30),
            })

        # Calculate overall impact score
        positive_impact = sum(
            f["magnitude"] for f in impact_factors if f["impact"] == "positive"
        )
        negative_impact = sum(
            f["magnitude"] for f in impact_factors if f["impact"] == "negative"
        )

        net_impact = positive_impact - negative_impact

        if net_impact > 0.3:
            overall = "POSITIVE"
        elif net_impact < -0.3:
            overall = "NEGATIVE"
        else:
            overall = "NEUTRAL"

        return {
            "overall_impact": overall,
            "net_impact_score": net_impact,
            "market_regime": regime["regime"],
            "impact_factors": impact_factors,
            "global_data_summary": data.to_dict(),
            "timestamp": datetime.now().isoformat(),
        }

    def clear_cache(self):
        """Clear the data cache"""
        self._cache.clear()
        self._cache_time.clear()


# Global instance
global_markets_client = GlobalMarketsClient()
