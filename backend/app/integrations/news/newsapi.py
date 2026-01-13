"""
NewsAPI Client - Free tier: 100 requests/day
https://newsapi.org/
"""
import httpx
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import asyncio

from app.config import get_settings


@dataclass
class NewsArticle:
    """Standardized news article format"""
    title: str
    description: Optional[str]
    source: str
    url: str
    published_at: datetime
    content: Optional[str] = None
    author: Optional[str] = None
    image_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "source": self.source,
            "url": self.url,
            "published_at": self.published_at.isoformat(),
            "content": self.content,
            "author": self.author,
            "image_url": self.image_url,
        }


class NewsAPIClient:
    """
    NewsAPI Client for financial news

    Free tier limits:
    - 100 requests per day
    - Headlines only (no full content)
    - 1 month historical data
    """

    BASE_URL = "https://newsapi.org/v2"

    def __init__(self):
        self.settings = get_settings()
        self.api_key = self.settings.NEWS_API_KEY
        self._request_count = 0
        self._last_reset = datetime.now()

    def _check_rate_limit(self) -> bool:
        """Check if we're within daily rate limit"""
        # Reset counter if new day
        if datetime.now().date() > self._last_reset.date():
            self._request_count = 0
            self._last_reset = datetime.now()

        return self._request_count < 100

    async def get_headlines(
        self,
        query: Optional[str] = None,
        category: str = "business",
        country: str = "in",
        page_size: int = 20
    ) -> List[NewsArticle]:
        """
        Fetch top headlines

        Args:
            query: Search query (stock name, etc.)
            category: News category (business, technology, etc.)
            country: Country code (in for India)
            page_size: Number of results (max 100)

        Returns:
            List of NewsArticle objects
        """
        if not self.api_key:
            print("NewsAPI: No API key configured")
            return []

        if not self._check_rate_limit():
            print("NewsAPI: Daily rate limit reached (100 requests)")
            return []

        params = {
            "apiKey": self.api_key,
            "category": category,
            "country": country,
            "pageSize": min(page_size, 100)
        }

        if query:
            params["q"] = query

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.BASE_URL}/top-headlines",
                    params=params
                )
                self._request_count += 1

                if response.status_code != 200:
                    print(f"NewsAPI error: {response.status_code}")
                    return []

                data = response.json()

                if data.get("status") != "ok":
                    print(f"NewsAPI error: {data.get('message')}")
                    return []

                articles = []
                for article in data.get("articles", []):
                    try:
                        published = datetime.fromisoformat(
                            article["publishedAt"].replace("Z", "+00:00")
                        )
                    except:
                        published = datetime.now()

                    articles.append(NewsArticle(
                        title=article.get("title", ""),
                        description=article.get("description"),
                        source=article.get("source", {}).get("name", "Unknown"),
                        url=article.get("url", ""),
                        published_at=published,
                        content=article.get("content"),
                        author=article.get("author"),
                        image_url=article.get("urlToImage")
                    ))

                return articles

        except Exception as e:
            print(f"NewsAPI fetch error: {e}")
            return []

    async def search_news(
        self,
        query: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        sort_by: str = "relevancy",
        page_size: int = 20
    ) -> List[NewsArticle]:
        """
        Search all news articles

        Args:
            query: Search query (required)
            from_date: Start date (max 1 month ago on free tier)
            to_date: End date
            sort_by: relevancy, popularity, publishedAt
            page_size: Number of results

        Returns:
            List of NewsArticle objects
        """
        if not self.api_key:
            return []

        if not self._check_rate_limit():
            return []

        # Free tier limited to 1 month historical
        if not from_date:
            from_date = datetime.now() - timedelta(days=30)

        params = {
            "apiKey": self.api_key,
            "q": query,
            "from": from_date.strftime("%Y-%m-%d"),
            "sortBy": sort_by,
            "pageSize": min(page_size, 100),
            "language": "en"
        }

        if to_date:
            params["to"] = to_date.strftime("%Y-%m-%d")

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.BASE_URL}/everything",
                    params=params
                )
                self._request_count += 1

                if response.status_code != 200:
                    return []

                data = response.json()

                if data.get("status") != "ok":
                    return []

                articles = []
                for article in data.get("articles", []):
                    try:
                        published = datetime.fromisoformat(
                            article["publishedAt"].replace("Z", "+00:00")
                        )
                    except:
                        published = datetime.now()

                    articles.append(NewsArticle(
                        title=article.get("title", ""),
                        description=article.get("description"),
                        source=article.get("source", {}).get("name", "Unknown"),
                        url=article.get("url", ""),
                        published_at=published,
                        content=article.get("content"),
                        author=article.get("author"),
                        image_url=article.get("urlToImage")
                    ))

                return articles

        except Exception as e:
            print(f"NewsAPI search error: {e}")
            return []

    async def get_stock_news(
        self,
        symbol: str,
        company_name: Optional[str] = None,
        days_back: int = 7
    ) -> List[NewsArticle]:
        """
        Get news for a specific stock

        Args:
            symbol: Stock symbol (e.g., RELIANCE)
            company_name: Full company name for better search
            days_back: Days of news to fetch

        Returns:
            List of NewsArticle objects
        """
        # Clean symbol
        clean_symbol = symbol.replace(".NS", "").replace(".BO", "")

        # Build search query
        query = clean_symbol
        if company_name:
            query = f'"{company_name}" OR "{clean_symbol}"'

        from_date = datetime.now() - timedelta(days=min(days_back, 30))

        return await self.search_news(
            query=query,
            from_date=from_date,
            sort_by="publishedAt",
            page_size=20
        )

    def get_remaining_requests(self) -> int:
        """Get remaining API requests for today"""
        if datetime.now().date() > self._last_reset.date():
            return 100
        return max(0, 100 - self._request_count)


# Global instance
newsapi_client = NewsAPIClient()
