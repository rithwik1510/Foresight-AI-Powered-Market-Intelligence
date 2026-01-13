"""
Finnhub Client - Free tier: 60 requests/minute
https://finnhub.io/
Provides news and pre-calculated sentiment scores
"""
import httpx
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import asyncio

from app.config import get_settings
from app.integrations.news.newsapi import NewsArticle


@dataclass
class FinnhubSentiment:
    """Finnhub sentiment data"""
    symbol: str
    buzz_articles_in_last_week: int
    weekly_average: float
    company_news_score: float
    sector_average_bullish_percent: float
    sector_average_news_score: float
    sentiment_score: float  # -1 to 1
    sentiment_label: str  # bullish, bearish, neutral

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "buzz": self.buzz_articles_in_last_week,
            "weekly_average": self.weekly_average,
            "company_news_score": self.company_news_score,
            "sector_average": self.sector_average_bullish_percent,
            "sentiment_score": self.sentiment_score,
            "sentiment_label": self.sentiment_label,
        }


class FinnhubClient:
    """
    Finnhub API Client

    Free tier limits:
    - 60 API calls/minute
    - Company news (US stocks)
    - Market news
    - Basic financials

    Note: Finnhub primarily covers US stocks.
    For Indian stocks, we use it for:
    - Global market news
    - ADR listed companies (INFY, WIT, etc.)
    """

    BASE_URL = "https://finnhub.io/api/v1"

    # Indian ADRs on US exchanges
    INDIAN_ADRS = {
        "INFY": "Infosys",
        "WIT": "Wipro",
        "HDB": "HDFC Bank",
        "IBN": "ICICI Bank",
        "RDY": "Dr. Reddy's",
        "SIFY": "Sify Technologies",
        "TTM": "Tata Motors",
        "VEDL": "Vedanta",
        "WNS": "WNS Holdings",
    }

    def __init__(self):
        self.settings = get_settings()
        self.api_key = self.settings.FINNHUB_API_KEY
        self._request_count = 0
        self._last_minute = datetime.now()

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limit (60/min)"""
        now = datetime.now()
        if (now - self._last_minute).seconds >= 60:
            self._request_count = 0
            self._last_minute = now

        return self._request_count < 60

    async def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make API request with rate limiting"""
        if not self.api_key:
            print("Finnhub: No API key configured")
            return None

        if not self._check_rate_limit():
            print("Finnhub: Rate limit reached (60/min)")
            return None

        if params is None:
            params = {}
        params["token"] = self.api_key

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.BASE_URL}/{endpoint}",
                    params=params
                )
                self._request_count += 1

                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"Finnhub error: {response.status_code}")
                    return None

        except Exception as e:
            print(f"Finnhub request error: {e}")
            return None

    async def get_company_news(
        self,
        symbol: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> List[NewsArticle]:
        """
        Get news for a US-listed company

        Args:
            symbol: Stock symbol (US symbol, not Indian)
            from_date: Start date
            to_date: End date

        Returns:
            List of NewsArticle objects
        """
        if not from_date:
            from_date = datetime.now() - timedelta(days=7)
        if not to_date:
            to_date = datetime.now()

        data = await self._make_request("company-news", {
            "symbol": symbol,
            "from": from_date.strftime("%Y-%m-%d"),
            "to": to_date.strftime("%Y-%m-%d")
        })

        if not data:
            return []

        articles = []
        for item in data[:20]:
            try:
                published = datetime.fromtimestamp(item.get("datetime", 0))
            except:
                published = datetime.now()

            articles.append(NewsArticle(
                title=item.get("headline", ""),
                description=item.get("summary", "")[:500] if item.get("summary") else None,
                source=item.get("source", "Finnhub"),
                url=item.get("url", ""),
                published_at=published,
                content=None,
                author=None,
                image_url=item.get("image")
            ))

        return articles

    async def get_market_news(self, category: str = "general") -> List[NewsArticle]:
        """
        Get market news

        Args:
            category: general, forex, crypto, merger

        Returns:
            List of NewsArticle objects
        """
        data = await self._make_request("news", {"category": category})

        if not data:
            return []

        articles = []
        for item in data[:20]:
            try:
                published = datetime.fromtimestamp(item.get("datetime", 0))
            except:
                published = datetime.now()

            articles.append(NewsArticle(
                title=item.get("headline", ""),
                description=item.get("summary", "")[:500] if item.get("summary") else None,
                source=item.get("source", "Finnhub"),
                url=item.get("url", ""),
                published_at=published,
                content=None,
                author=None,
                image_url=item.get("image")
            ))

        return articles

    async def get_news_sentiment(self, symbol: str) -> Optional[FinnhubSentiment]:
        """
        Get news sentiment for a US-listed company

        Args:
            symbol: US stock symbol

        Returns:
            FinnhubSentiment object or None
        """
        data = await self._make_request("news-sentiment", {"symbol": symbol})

        if not data or "sentiment" not in data:
            return None

        sentiment_data = data.get("sentiment", {})
        buzz_data = data.get("buzz", {})

        # Calculate overall sentiment score (-1 to 1)
        bullish = sentiment_data.get("bullishPercent", 0.5)
        bearish = sentiment_data.get("bearishPercent", 0.5)
        sentiment_score = bullish - bearish

        # Determine label
        if sentiment_score > 0.1:
            label = "bullish"
        elif sentiment_score < -0.1:
            label = "bearish"
        else:
            label = "neutral"

        return FinnhubSentiment(
            symbol=symbol,
            buzz_articles_in_last_week=buzz_data.get("articlesInLastWeek", 0),
            weekly_average=buzz_data.get("weeklyAverage", 0),
            company_news_score=data.get("companyNewsScore", 0),
            sector_average_bullish_percent=sentiment_data.get("sectorAverageBullishPercent", 0.5),
            sector_average_news_score=data.get("sectorAverageNewsScore", 0),
            sentiment_score=sentiment_score,
            sentiment_label=label
        )

    async def get_indian_adr_news(self, limit: int = 20) -> List[NewsArticle]:
        """
        Get news for Indian ADRs listed in US

        Args:
            limit: Maximum articles

        Returns:
            List of NewsArticle objects
        """
        all_articles = []

        for symbol in list(self.INDIAN_ADRS.keys())[:5]:  # Limit to save API calls
            articles = await self.get_company_news(symbol)
            all_articles.extend(articles)
            await asyncio.sleep(0.5)  # Brief delay between requests

        # Sort by date
        all_articles.sort(key=lambda x: x.published_at, reverse=True)
        return all_articles[:limit]

    async def get_indian_adr_sentiment(self) -> Dict[str, FinnhubSentiment]:
        """
        Get sentiment for all Indian ADRs

        Returns:
            Dictionary of symbol -> FinnhubSentiment
        """
        sentiments = {}

        for symbol in self.INDIAN_ADRS.keys():
            sentiment = await self.get_news_sentiment(symbol)
            if sentiment:
                sentiments[symbol] = sentiment
            await asyncio.sleep(0.5)

        return sentiments

    async def get_forex_news(self) -> List[NewsArticle]:
        """
        Get forex news (useful for USD/INR)

        Returns:
            List of NewsArticle objects
        """
        return await self.get_market_news(category="forex")

    def get_remaining_requests(self) -> int:
        """Get remaining API requests for this minute"""
        now = datetime.now()
        if (now - self._last_minute).seconds >= 60:
            return 60
        return max(0, 60 - self._request_count)

    def map_indian_to_adr(self, indian_symbol: str) -> Optional[str]:
        """
        Map Indian stock symbol to US ADR symbol

        Args:
            indian_symbol: Indian stock symbol (e.g., INFY.NS)

        Returns:
            US ADR symbol or None if not listed
        """
        clean_symbol = indian_symbol.replace(".NS", "").replace(".BO", "")

        # Direct mappings
        symbol_map = {
            "INFY": "INFY",
            "WIPRO": "WIT",
            "HDFCBANK": "HDB",
            "ICICIBANK": "IBN",
            "DRREDDY": "RDY",
            "TATAMOTORS": "TTM",
            "VEDL": "VEDL",
        }

        return symbol_map.get(clean_symbol.upper())


# Global instance
finnhub_client = FinnhubClient()
