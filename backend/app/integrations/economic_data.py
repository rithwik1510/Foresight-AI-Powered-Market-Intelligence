"""
Economic Data Integration - FRED API for US economic indicators
Free tier: Unlimited requests (requires free API key)
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
import asyncio

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False

from app.config import get_settings


@dataclass
class EconomicIndicators:
    """Key economic indicators"""
    timestamp: datetime

    # Interest Rates
    fed_funds_rate: float  # Federal Funds Rate
    fed_funds_change_1m: float

    # Inflation
    cpi_yoy: float  # Consumer Price Index (YoY)
    pce_yoy: float  # Personal Consumption Expenditures (YoY)

    # Employment
    unemployment_rate: float
    nonfarm_payrolls_change: float

    # GDP
    gdp_growth_rate: float  # Real GDP Growth Rate (annualized)

    # Market Indicators
    yield_curve_spread: float  # 10Y - 2Y Treasury spread

    # Confidence
    consumer_sentiment: float  # University of Michigan Consumer Sentiment

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "interest_rates": {
                "fed_funds_rate": self.fed_funds_rate,
                "change_1m": self.fed_funds_change_1m,
            },
            "inflation": {
                "cpi_yoy": self.cpi_yoy,
                "pce_yoy": self.pce_yoy,
            },
            "employment": {
                "unemployment_rate": self.unemployment_rate,
                "nonfarm_payrolls_change": self.nonfarm_payrolls_change,
            },
            "gdp": {
                "growth_rate": self.gdp_growth_rate,
            },
            "market_indicators": {
                "yield_curve_spread": self.yield_curve_spread,
            },
            "confidence": {
                "consumer_sentiment": self.consumer_sentiment,
            },
        }


class EconomicDataClient:
    """
    FRED API Client for US Economic Data

    Key series:
    - FEDFUNDS: Federal Funds Rate
    - CPIAUCSL: Consumer Price Index
    - PCEPI: Personal Consumption Expenditures Price Index
    - UNRATE: Unemployment Rate
    - PAYEMS: Total Nonfarm Payrolls
    - GDP: Gross Domestic Product
    - T10Y2Y: 10-Year Treasury Constant Maturity Minus 2-Year
    - UMCSENT: University of Michigan Consumer Sentiment
    """

    # FRED series IDs
    SERIES = {
        "fed_funds": "FEDFUNDS",
        "cpi": "CPIAUCSL",
        "pce": "PCEPI",
        "unemployment": "UNRATE",
        "payrolls": "PAYEMS",
        "gdp": "GDP",
        "yield_spread": "T10Y2Y",
        "consumer_sentiment": "UMCSENT",
    }

    def __init__(self):
        self.settings = get_settings()
        self.fred = None
        self._initialized = False
        self._cache: Dict[str, pd.Series] = {}
        self._cache_time: Dict[str, datetime] = {}
        self._cache_ttl = 3600  # 1 hour (economic data updates infrequently)
        self._init_fred()

    def _init_fred(self):
        """Initialize FRED client"""
        if not FRED_AVAILABLE:
            print("FRED: fredapi not installed")
            return

        api_key = self.settings.FRED_API_KEY

        if not api_key:
            print("FRED: No API key configured")
            return

        try:
            self.fred = Fred(api_key=api_key)
            self._initialized = True
            print("FRED: Client initialized")
        except Exception as e:
            print(f"FRED: Initialization error: {e}")

    def is_available(self) -> bool:
        """Check if FRED client is available"""
        return self._initialized and self.fred is not None

    def _fetch_series(
        self,
        series_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[pd.Series]:
        """Fetch a single FRED series"""
        if not self.is_available():
            return None

        # Check cache
        cache_key = series_id
        if cache_key in self._cache:
            cache_age = (datetime.now() - self._cache_time[cache_key]).seconds
            if cache_age < self._cache_ttl:
                series = self._cache[cache_key]
                if start_date:
                    series = series[series.index >= start_date]
                return series

        try:
            if start_date is None:
                start_date = datetime.now() - timedelta(days=365 * 2)

            series = self.fred.get_series(
                series_id,
                observation_start=start_date,
                observation_end=end_date
            )

            # Cache full series
            self._cache[cache_key] = series
            self._cache_time[cache_key] = datetime.now()

            return series

        except Exception as e:
            print(f"FRED fetch error ({series_id}): {e}")
            return None

    async def get_current_indicators(self) -> Optional[EconomicIndicators]:
        """
        Get current economic indicators

        Returns:
            EconomicIndicators object or None if unavailable
        """
        if not self.is_available():
            return None

        # Fetch all series
        data = {}
        for name, series_id in self.SERIES.items():
            series = self._fetch_series(series_id)
            if series is not None:
                data[name] = series
            await asyncio.sleep(0.1)  # Small delay

        if not data:
            return None

        # Get latest values
        def get_latest(series: Optional[pd.Series]) -> float:
            if series is None or series.empty:
                return 0.0
            return float(series.dropna().iloc[-1])

        def get_yoy_change(series: Optional[pd.Series]) -> float:
            """Calculate year-over-year change"""
            if series is None or len(series) < 12:
                return 0.0
            try:
                current = series.dropna().iloc[-1]
                prev_year = series.dropna().iloc[-12]
                return (current - prev_year) / prev_year * 100
            except:
                return 0.0

        def get_mom_change(series: Optional[pd.Series]) -> float:
            """Calculate month-over-month change"""
            if series is None or len(series) < 2:
                return 0.0
            try:
                current = series.dropna().iloc[-1]
                prev_month = series.dropna().iloc[-2]
                return (current - prev_month) / prev_month * 100
            except:
                return 0.0

        # Calculate CPI YoY
        cpi_yoy = get_yoy_change(data.get("cpi"))

        # Calculate PCE YoY
        pce_yoy = get_yoy_change(data.get("pce"))

        # Fed funds rate change
        fed_series = data.get("fed_funds")
        fed_funds_change = 0.0
        if fed_series is not None and len(fed_series) >= 2:
            try:
                fed_funds_change = float(fed_series.dropna().iloc[-1] - fed_series.dropna().iloc[-2])
            except:
                pass

        # Payrolls change (in thousands)
        payrolls_series = data.get("payrolls")
        payrolls_change = 0.0
        if payrolls_series is not None and len(payrolls_series) >= 2:
            try:
                payrolls_change = float(payrolls_series.dropna().iloc[-1] - payrolls_series.dropna().iloc[-2])
            except:
                pass

        # GDP growth (already annualized rate)
        gdp_series = data.get("gdp")
        gdp_growth = 0.0
        if gdp_series is not None and len(gdp_series) >= 2:
            try:
                current_gdp = gdp_series.dropna().iloc[-1]
                prev_gdp = gdp_series.dropna().iloc[-2]
                gdp_growth = (current_gdp - prev_gdp) / prev_gdp * 400  # Annualized
            except:
                pass

        return EconomicIndicators(
            timestamp=datetime.now(),
            fed_funds_rate=get_latest(data.get("fed_funds")),
            fed_funds_change_1m=fed_funds_change,
            cpi_yoy=cpi_yoy,
            pce_yoy=pce_yoy,
            unemployment_rate=get_latest(data.get("unemployment")),
            nonfarm_payrolls_change=payrolls_change,
            gdp_growth_rate=gdp_growth,
            yield_curve_spread=get_latest(data.get("yield_spread")),
            consumer_sentiment=get_latest(data.get("consumer_sentiment")),
        )

    async def get_economic_regime(self) -> Dict[str, Any]:
        """
        Determine current economic regime

        Returns:
            Dictionary with regime classification
        """
        indicators = await self.get_current_indicators()

        if indicators is None:
            return {
                "regime": "UNKNOWN",
                "confidence": 0.0,
                "message": "Economic data unavailable",
            }

        # Classify regime
        signals = {
            "expansion": 0,
            "contraction": 0,
            "inflationary": 0,
            "deflationary": 0,
        }

        # Unemployment signals
        if indicators.unemployment_rate < 4.0:
            signals["expansion"] += 2
        elif indicators.unemployment_rate > 5.5:
            signals["contraction"] += 2

        # Payrolls signals
        if indicators.nonfarm_payrolls_change > 200:
            signals["expansion"] += 1
        elif indicators.nonfarm_payrolls_change < 0:
            signals["contraction"] += 2

        # GDP signals
        if indicators.gdp_growth_rate > 2.5:
            signals["expansion"] += 2
        elif indicators.gdp_growth_rate < 0:
            signals["contraction"] += 3

        # Inflation signals
        if indicators.cpi_yoy > 3.0:
            signals["inflationary"] += 2
        elif indicators.cpi_yoy < 1.5:
            signals["deflationary"] += 1

        if indicators.pce_yoy > 2.5:
            signals["inflationary"] += 1

        # Yield curve signals (inverted = recession warning)
        if indicators.yield_curve_spread < 0:
            signals["contraction"] += 2
        elif indicators.yield_curve_spread > 1.0:
            signals["expansion"] += 1

        # Consumer sentiment
        if indicators.consumer_sentiment > 90:
            signals["expansion"] += 1
        elif indicators.consumer_sentiment < 70:
            signals["contraction"] += 1

        # Determine primary regime
        if signals["expansion"] > signals["contraction"] + 2:
            if signals["inflationary"] > 2:
                regime = "INFLATIONARY_EXPANSION"
            else:
                regime = "HEALTHY_EXPANSION"
        elif signals["contraction"] > signals["expansion"] + 2:
            regime = "CONTRACTION"
        else:
            if signals["inflationary"] > signals["deflationary"] + 2:
                regime = "STAGFLATION_RISK"
            else:
                regime = "NEUTRAL"

        # Calculate confidence
        total_signals = sum(signals.values())
        max_signal = max(signals.values())
        confidence = max_signal / max(1, total_signals)

        return {
            "regime": regime,
            "confidence": confidence,
            "signals": signals,
            "indicators": indicators.to_dict(),
            "implications": self._get_regime_implications(regime),
            "timestamp": datetime.now().isoformat(),
        }

    def _get_regime_implications(self, regime: str) -> Dict[str, str]:
        """Get investment implications for regime"""
        implications = {
            "HEALTHY_EXPANSION": {
                "stocks": "Favorable - Growth and cyclicals tend to outperform",
                "bonds": "Neutral to Unfavorable - Rising rates pressure prices",
                "gold": "Neutral - Less safe-haven demand",
                "emerging_markets": "Favorable - Risk-on environment",
            },
            "INFLATIONARY_EXPANSION": {
                "stocks": "Mixed - Value may outperform growth",
                "bonds": "Unfavorable - Inflation erodes returns",
                "gold": "Favorable - Inflation hedge",
                "emerging_markets": "Mixed - Depends on commodity exposure",
            },
            "CONTRACTION": {
                "stocks": "Unfavorable - Defensive sectors preferred",
                "bonds": "Favorable - Flight to quality",
                "gold": "Favorable - Safe haven",
                "emerging_markets": "Unfavorable - Risk-off environment",
            },
            "STAGFLATION_RISK": {
                "stocks": "Unfavorable - Both growth and value challenged",
                "bonds": "Unfavorable - Inflation and rate concerns",
                "gold": "Very Favorable - Ultimate hedge",
                "emerging_markets": "Unfavorable - Multiple headwinds",
            },
            "NEUTRAL": {
                "stocks": "Neutral - Focus on quality and fundamentals",
                "bonds": "Neutral - Duration risk exists",
                "gold": "Neutral - Standard allocation",
                "emerging_markets": "Neutral - Selective opportunities",
            },
        }

        return implications.get(regime, implications["NEUTRAL"])

    async def get_fed_outlook(self) -> Dict[str, Any]:
        """
        Analyze Fed policy outlook based on economic data

        Returns:
            Dictionary with Fed policy analysis
        """
        indicators = await self.get_current_indicators()

        if indicators is None:
            return {"outlook": "UNKNOWN", "message": "Data unavailable"}

        # Analyze Fed likely direction
        hawkish_signals = 0
        dovish_signals = 0

        # Inflation above target = hawkish
        if indicators.cpi_yoy > 2.5:
            hawkish_signals += 2
        if indicators.pce_yoy > 2.0:
            hawkish_signals += 1

        # Strong employment = hawkish
        if indicators.unemployment_rate < 4.0:
            hawkish_signals += 1
        if indicators.nonfarm_payrolls_change > 200:
            hawkish_signals += 1

        # Weak employment = dovish
        if indicators.unemployment_rate > 5.0:
            dovish_signals += 2
        if indicators.nonfarm_payrolls_change < 100:
            dovish_signals += 1

        # Low inflation = dovish
        if indicators.cpi_yoy < 2.0:
            dovish_signals += 1

        # Consumer sentiment weak = dovish
        if indicators.consumer_sentiment < 70:
            dovish_signals += 1

        if hawkish_signals > dovish_signals + 2:
            outlook = "HAWKISH"
            direction = "Rate hikes likely or steady at high levels"
        elif dovish_signals > hawkish_signals + 2:
            outlook = "DOVISH"
            direction = "Rate cuts possible"
        else:
            outlook = "NEUTRAL"
            direction = "Rates likely to remain stable"

        return {
            "outlook": outlook,
            "direction": direction,
            "current_rate": indicators.fed_funds_rate,
            "hawkish_signals": hawkish_signals,
            "dovish_signals": dovish_signals,
            "key_factors": {
                "inflation_cpi": indicators.cpi_yoy,
                "inflation_pce": indicators.pce_yoy,
                "unemployment": indicators.unemployment_rate,
                "payrolls": indicators.nonfarm_payrolls_change,
            },
            "impact_on_india": self._get_fed_india_impact(outlook),
            "timestamp": datetime.now().isoformat(),
        }

    def _get_fed_india_impact(self, outlook: str) -> str:
        """Get impact of Fed policy on Indian markets"""
        impacts = {
            "HAWKISH": "Negative for Indian markets - Higher US rates attract capital away from emerging markets, weakening INR",
            "DOVISH": "Positive for Indian markets - Lower US rates increase flows to emerging markets, strengthening INR",
            "NEUTRAL": "Mixed impact - Indian markets to focus on domestic factors",
        }
        return impacts.get(outlook, impacts["NEUTRAL"])

    async def get_historical_indicators(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get historical economic indicators

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with all indicators
        """
        if not self.is_available():
            return pd.DataFrame()

        if start_date is None:
            start_date = datetime.now() - timedelta(days=365 * 5)

        # Fetch all series
        series_data = {}
        for name, series_id in self.SERIES.items():
            series = self._fetch_series(series_id, start_date, end_date)
            if series is not None:
                series_data[name] = series
            await asyncio.sleep(0.1)

        if not series_data:
            return pd.DataFrame()

        # Combine into DataFrame
        df = pd.DataFrame(series_data)

        # Forward fill missing values (economic data is released at different frequencies)
        df = df.ffill()

        return df

    def clear_cache(self):
        """Clear the data cache"""
        self._cache.clear()
        self._cache_time.clear()


# Global instance
economic_data_client = EconomicDataClient()
