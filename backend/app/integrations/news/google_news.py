"""
Google News RSS Client - Unlimited requests (no API key needed)
Uses Google News RSS feed for real-time news
"""
import feedparser
from datetime import datetime
from typing import List, Optional, Dict
from urllib.parse import quote
import asyncio
import httpx
from time import mktime

from app.integrations.news.newsapi import NewsArticle


class GoogleNewsClient:
    """
    Google News RSS Feed Client

    No API key required - uses public RSS feeds.
    Good for:
    - Real-time news search
    - Company-specific news
    - Market events

    Note: Google News RSS has no documented rate limits,
    but excessive requests may be blocked.
    """

    BASE_URL = "https://news.google.com/rss"

    def __init__(self):
        self._cache: Dict[str, List[NewsArticle]] = {}
        self._cache_time: Dict[str, datetime] = {}
        self._cache_ttl = 300  # 5 minutes
        self._request_delay = 1.0  # 1 second between requests

    def _parse_date(self, entry: Dict) -> datetime:
        """Parse date from RSS entry"""
        try:
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                return datetime.fromtimestamp(mktime(entry.published_parsed))
        except:
            pass
        return datetime.now()

    def _extract_source(self, title: str) -> tuple:
        """Extract actual title and source from Google News format"""
        # Google News titles format: "Actual Title - Source Name"
        if " - " in title:
            parts = title.rsplit(" - ", 1)
            return parts[0], parts[1] if len(parts) > 1 else "Google News"
        return title, "Google News"

    async def search(
        self,
        query: str,
        language: str = "en",
        country: str = "IN",
        limit: int = 20
    ) -> List[NewsArticle]:
        """
        Search Google News for articles

        Args:
            query: Search query
            language: Language code (en, hi)
            country: Country code (IN for India)
            limit: Maximum results

        Returns:
            List of NewsArticle objects
        """
        # Check cache
        cache_key = f"{query}:{language}:{country}"
        if cache_key in self._cache:
            cache_age = (datetime.now() - self._cache_time[cache_key]).seconds
            if cache_age < self._cache_ttl:
                return self._cache[cache_key][:limit]

        # Build URL
        encoded_query = quote(query)
        url = f"{self.BASE_URL}/search?q={encoded_query}&hl={language}&gl={country}&ceid={country}:{language}"

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url)
                if response.status_code != 200:
                    return []

                feed = feedparser.parse(response.text)
                articles = []

                for entry in feed.entries[:limit]:
                    title, source = self._extract_source(entry.get('title', ''))

                    articles.append(NewsArticle(
                        title=title,
                        description=entry.get('summary', '')[:500] if entry.get('summary') else None,
                        source=source,
                        url=entry.get('link', ''),
                        published_at=self._parse_date(entry),
                        content=None,
                        author=None,
                        image_url=None
                    ))

                # Cache results
                self._cache[cache_key] = articles
                self._cache_time[cache_key] = datetime.now()

                return articles

        except Exception as e:
            print(f"Google News search error: {e}")
            return []

    async def get_stock_news(
        self,
        symbol: str,
        company_name: Optional[str] = None,
        limit: int = 20
    ) -> List[NewsArticle]:
        """
        Get news for a specific stock

        Args:
            symbol: Stock symbol (e.g., RELIANCE.NS)
            company_name: Full company name
            limit: Maximum articles

        Returns:
            List of NewsArticle objects
        """
        # Clean symbol
        clean_symbol = symbol.replace(".NS", "").replace(".BO", "")

        # Search with company name if available
        if company_name:
            query = f'"{company_name}" stock NSE'
        else:
            query = f"{clean_symbol} stock NSE India"

        return await self.search(query, limit=limit)

    async def get_market_news(self, limit: int = 20) -> List[NewsArticle]:
        """
        Get general Indian stock market news

        Args:
            limit: Maximum articles

        Returns:
            List of NewsArticle objects
        """
        queries = [
            "NSE Nifty stock market",
            "Indian stock market today",
            "BSE Sensex news"
        ]

        all_articles = []
        for query in queries:
            articles = await self.search(query, limit=limit // len(queries))
            all_articles.extend(articles)
            await asyncio.sleep(self._request_delay)  # Respect rate limiting

        # Remove duplicates and sort
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique_articles.append(article)

        unique_articles.sort(key=lambda x: x.published_at, reverse=True)
        return unique_articles[:limit]

    async def get_sector_news(self, sector: str, limit: int = 20) -> List[NewsArticle]:
        """
        Get news for a specific sector

        Args:
            sector: Sector name
            limit: Maximum articles

        Returns:
            List of NewsArticle objects
        """
        sector_queries = {
            "banking": "Indian banking sector HDFC ICICI SBI news",
            "it": "Indian IT sector TCS Infosys Wipro news",
            "pharma": "Indian pharma sector news",
            "auto": "Indian automobile sector news Tata Motors Maruti",
            "energy": "Indian energy sector Reliance ONGC news",
            "fmcg": "Indian FMCG sector HUL ITC news",
            "metal": "Indian metal sector Tata Steel JSW news",
            "realty": "Indian real estate sector DLF news",
        }

        query = sector_queries.get(sector.lower(), f"Indian {sector} sector news")
        return await self.search(query, limit=limit)

    async def get_economic_news(self, limit: int = 20) -> List[NewsArticle]:
        """
        Get Indian economic news

        Args:
            limit: Maximum articles

        Returns:
            List of NewsArticle objects
        """
        query = "India economy RBI GDP inflation news"
        return await self.search(query, limit=limit)

    async def get_global_market_news(self, limit: int = 20) -> List[NewsArticle]:
        """
        Get global market news affecting India

        Args:
            limit: Maximum articles

        Returns:
            List of NewsArticle objects
        """
        queries = [
            "US Federal Reserve interest rates",
            "crude oil prices OPEC",
            "US China trade",
            "global stock markets today"
        ]

        all_articles = []
        for query in queries:
            articles = await self.search(query, country="US", limit=limit // len(queries))
            all_articles.extend(articles)
            await asyncio.sleep(self._request_delay)

        all_articles.sort(key=lambda x: x.published_at, reverse=True)
        return all_articles[:limit]


# Global instance
google_news_client = GoogleNewsClient()
