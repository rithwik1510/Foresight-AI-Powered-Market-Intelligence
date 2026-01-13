"""
RSS Parser - Unlimited requests (no API key needed)
Parses RSS feeds from financial news sources
"""
import feedparser
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import asyncio
import httpx
from time import mktime

from app.integrations.news.newsapi import NewsArticle


class RSSParser:
    """
    RSS Feed Parser for Indian Financial News

    No rate limits - these are public RSS feeds.
    Sources:
    - Economic Times
    - Moneycontrol
    - Business Standard
    - Livemint
    """

    # Indian financial news RSS feeds
    FEEDS = {
        "economic_times": {
            "markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
            "stocks": "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
            "industry": "https://economictimes.indiatimes.com/industry/rssfeeds/13352306.cms",
        },
        "moneycontrol": {
            "markets": "https://www.moneycontrol.com/rss/marketreports.xml",
            "news": "https://www.moneycontrol.com/rss/latestnews.xml",
            "business": "https://www.moneycontrol.com/rss/business.xml",
        },
        "business_standard": {
            "markets": "https://www.business-standard.com/rss/markets-106.rss",
            "companies": "https://www.business-standard.com/rss/companies-101.rss",
            "economy": "https://www.business-standard.com/rss/economy-policy-102.rss",
        },
        "livemint": {
            "markets": "https://www.livemint.com/rss/markets",
            "companies": "https://www.livemint.com/rss/companies",
            "money": "https://www.livemint.com/rss/money",
        },
    }

    def __init__(self):
        self._cache: Dict[str, List[NewsArticle]] = {}
        self._cache_time: Dict[str, datetime] = {}
        self._cache_ttl = 300  # 5 minutes

    def _parse_date(self, entry: Dict) -> datetime:
        """Parse date from RSS entry"""
        try:
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                return datetime.fromtimestamp(mktime(entry.published_parsed))
            elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                return datetime.fromtimestamp(mktime(entry.updated_parsed))
        except:
            pass
        return datetime.now()

    def _parse_entry(self, entry: Dict, source: str) -> NewsArticle:
        """Convert RSS entry to NewsArticle"""
        # Get description/summary
        description = None
        if hasattr(entry, 'summary'):
            description = entry.summary
        elif hasattr(entry, 'description'):
            description = entry.description

        # Clean HTML from description
        if description:
            import re
            description = re.sub('<[^<]+?>', '', description)
            description = description[:500]  # Limit length

        return NewsArticle(
            title=entry.get('title', 'No Title'),
            description=description,
            source=source,
            url=entry.get('link', ''),
            published_at=self._parse_date(entry),
            content=None,
            author=entry.get('author'),
            image_url=None
        )

    async def fetch_feed(self, feed_url: str, source: str) -> List[NewsArticle]:
        """
        Fetch and parse a single RSS feed

        Args:
            feed_url: URL of the RSS feed
            source: Source name for attribution

        Returns:
            List of NewsArticle objects
        """
        try:
            # Check cache
            cache_key = feed_url
            if cache_key in self._cache:
                cache_age = (datetime.now() - self._cache_time[cache_key]).seconds
                if cache_age < self._cache_ttl:
                    return self._cache[cache_key]

            # Fetch feed
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(feed_url)
                if response.status_code != 200:
                    return []

                # Parse RSS
                feed = feedparser.parse(response.text)

                articles = []
                for entry in feed.entries[:20]:  # Limit to 20 per feed
                    article = self._parse_entry(entry, source)
                    articles.append(article)

                # Update cache
                self._cache[cache_key] = articles
                self._cache_time[cache_key] = datetime.now()

                return articles

        except Exception as e:
            print(f"RSS fetch error ({source}): {e}")
            return []

    async def get_market_news(self, limit: int = 50) -> List[NewsArticle]:
        """
        Get latest market news from all sources

        Args:
            limit: Maximum articles to return

        Returns:
            List of NewsArticle objects sorted by date
        """
        all_articles = []

        # Fetch from all market feeds in parallel
        tasks = []
        for source_name, feeds in self.FEEDS.items():
            if "markets" in feeds:
                tasks.append(
                    self.fetch_feed(feeds["markets"], source_name.replace("_", " ").title())
                )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)

        # Sort by date and limit
        all_articles.sort(key=lambda x: x.published_at, reverse=True)
        return all_articles[:limit]

    async def get_stock_news(
        self,
        symbol: str,
        company_name: Optional[str] = None,
        limit: int = 20
    ) -> List[NewsArticle]:
        """
        Get news related to a specific stock

        Args:
            symbol: Stock symbol
            company_name: Company name for filtering
            limit: Maximum articles

        Returns:
            Filtered list of NewsArticle objects
        """
        # Get all news first
        all_news = await self.get_all_news(limit=100)

        # Clean symbol
        clean_symbol = symbol.replace(".NS", "").replace(".BO", "").lower()

        # Filter by symbol or company name
        search_terms = [clean_symbol]
        if company_name:
            search_terms.append(company_name.lower())

        filtered = []
        for article in all_news:
            text = f"{article.title} {article.description or ''}".lower()
            if any(term in text for term in search_terms):
                filtered.append(article)
                if len(filtered) >= limit:
                    break

        return filtered

    async def get_all_news(self, limit: int = 100) -> List[NewsArticle]:
        """
        Get news from all feeds

        Args:
            limit: Maximum articles

        Returns:
            List of NewsArticle objects
        """
        all_articles = []

        # Fetch from all feeds
        tasks = []
        for source_name, feeds in self.FEEDS.items():
            for feed_type, feed_url in feeds.items():
                tasks.append(
                    self.fetch_feed(feed_url, f"{source_name.replace('_', ' ').title()} - {feed_type.title()}")
                )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)

        # Remove duplicates by URL
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique_articles.append(article)

        # Sort by date
        unique_articles.sort(key=lambda x: x.published_at, reverse=True)
        return unique_articles[:limit]

    async def get_sector_news(self, sector: str, limit: int = 20) -> List[NewsArticle]:
        """
        Get news for a specific sector

        Args:
            sector: Sector name (banking, it, pharma, etc.)
            limit: Maximum articles

        Returns:
            Filtered list of NewsArticle objects
        """
        # Sector keywords
        sector_keywords = {
            "banking": ["bank", "banking", "rbi", "credit", "loan", "npa"],
            "it": ["technology", "software", "it", "tech", "digital", "ai", "cloud"],
            "pharma": ["pharma", "drug", "healthcare", "medicine", "fda"],
            "auto": ["automobile", "auto", "vehicle", "car", "ev", "electric vehicle"],
            "energy": ["oil", "gas", "energy", "power", "renewable", "solar"],
            "fmcg": ["fmcg", "consumer", "retail", "goods"],
            "metal": ["metal", "steel", "iron", "mining", "aluminium"],
            "realty": ["real estate", "realty", "property", "housing"],
        }

        keywords = sector_keywords.get(sector.lower(), [sector.lower()])

        all_news = await self.get_all_news(limit=100)

        filtered = []
        for article in all_news:
            text = f"{article.title} {article.description or ''}".lower()
            if any(kw in text for kw in keywords):
                filtered.append(article)
                if len(filtered) >= limit:
                    break

        return filtered


# Global instance
rss_parser = RSSParser()
