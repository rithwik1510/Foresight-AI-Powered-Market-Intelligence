"""
Sentiment Aggregator - Combines all sentiment sources into unified score
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
import asyncio

from app.integrations.news.newsapi import NewsAPIClient, newsapi_client, NewsArticle
from app.integrations.news.rss_parser import RSSParser, rss_parser
from app.integrations.news.google_news import GoogleNewsClient, google_news_client
from app.integrations.news.finnhub import FinnhubClient, finnhub_client
from app.ml.sentiment.sentiment_analyzer import SentimentAnalyzer, sentiment_analyzer, SentimentResult
from app.ml.sentiment.reddit_scraper import RedditScraper, reddit_scraper


@dataclass
class AggregatedSentiment:
    """Combined sentiment from all sources"""
    symbol: str
    company_name: Optional[str]
    timestamp: datetime

    # Overall metrics
    overall_score: float  # -1 to 1
    overall_label: str  # bullish, bearish, neutral
    confidence: float  # 0 to 1

    # Source-specific scores
    news_score: float
    news_article_count: int
    social_score: float
    social_post_count: int

    # Detailed breakdown
    source_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Top articles/posts
    top_bullish_articles: List[Dict] = field(default_factory=list)
    top_bearish_articles: List[Dict] = field(default_factory=list)

    # Trend (comparing to previous period)
    sentiment_trend: str = "stable"  # improving, deteriorating, stable
    trend_change: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "company_name": self.company_name,
            "timestamp": self.timestamp.isoformat(),
            "overall_score": self.overall_score,
            "overall_label": self.overall_label,
            "confidence": self.confidence,
            "news_score": self.news_score,
            "news_article_count": self.news_article_count,
            "social_score": self.social_score,
            "social_post_count": self.social_post_count,
            "source_breakdown": self.source_breakdown,
            "top_bullish_articles": self.top_bullish_articles,
            "top_bearish_articles": self.top_bearish_articles,
            "sentiment_trend": self.sentiment_trend,
            "trend_change": self.trend_change,
        }


class SentimentAggregator:
    """
    Aggregates sentiment from multiple sources

    Sources (ordered by weight):
    1. RSS Feeds (unlimited) - Primary source
    2. Google News (unlimited) - Real-time news
    3. Reddit (60/min) - Social sentiment
    4. NewsAPI (100/day) - Quality news
    5. Finnhub (60/min) - Pre-calculated sentiment

    Weighting strategy:
    - News: 60% (RSS + Google News + NewsAPI)
    - Social: 30% (Reddit)
    - Pre-calculated: 10% (Finnhub)
    """

    # Source weights
    WEIGHTS = {
        "rss": 0.25,
        "google_news": 0.20,
        "newsapi": 0.15,
        "reddit": 0.30,
        "finnhub": 0.10,
    }

    def __init__(self):
        self.news_api = newsapi_client
        self.rss = rss_parser
        self.google_news = google_news_client
        self.finnhub = finnhub_client
        self.reddit = reddit_scraper
        self.analyzer = sentiment_analyzer

        # Cache for recent results
        self._cache: Dict[str, AggregatedSentiment] = {}
        self._cache_ttl = 1800  # 30 minutes

    async def get_stock_sentiment(
        self,
        symbol: str,
        company_name: Optional[str] = None,
        use_cache: bool = True
    ) -> AggregatedSentiment:
        """
        Get aggregated sentiment for a stock

        Args:
            symbol: Stock symbol (e.g., RELIANCE.NS)
            company_name: Full company name for better search
            use_cache: Whether to use cached results

        Returns:
            AggregatedSentiment object
        """
        # Check cache
        cache_key = symbol.upper()
        if use_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            age = (datetime.now() - cached.timestamp).seconds
            if age < self._cache_ttl:
                return cached

        # Fetch from all sources in parallel
        results = await asyncio.gather(
            self._fetch_rss_sentiment(symbol, company_name),
            self._fetch_google_news_sentiment(symbol, company_name),
            self._fetch_newsapi_sentiment(symbol, company_name),
            self._fetch_reddit_sentiment(symbol, company_name),
            self._fetch_finnhub_sentiment(symbol),
            return_exceptions=True
        )

        # Process results
        source_results = {
            "rss": results[0] if not isinstance(results[0], Exception) else None,
            "google_news": results[1] if not isinstance(results[1], Exception) else None,
            "newsapi": results[2] if not isinstance(results[2], Exception) else None,
            "reddit": results[3] if not isinstance(results[3], Exception) else None,
            "finnhub": results[4] if not isinstance(results[4], Exception) else None,
        }

        # Calculate aggregated sentiment
        aggregated = self._aggregate_results(symbol, company_name, source_results)

        # Cache result
        self._cache[cache_key] = aggregated

        return aggregated

    async def _fetch_rss_sentiment(
        self,
        symbol: str,
        company_name: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Fetch and analyze RSS news"""
        try:
            articles = await self.rss.get_stock_news(symbol, company_name, limit=20)

            if not articles:
                # Fall back to market news if stock-specific not found
                articles = await self.rss.get_market_news(limit=10)

            sentiment_results = self.analyzer.analyze_articles(articles)
            aggregate = self.analyzer.get_aggregate_sentiment(sentiment_results)

            return {
                "score": aggregate["weighted_score"],
                "article_count": len(articles),
                "aggregate": aggregate,
                "top_results": sentiment_results[:5] if sentiment_results else [],
            }

        except Exception as e:
            print(f"RSS sentiment error: {e}")
            return None

    async def _fetch_google_news_sentiment(
        self,
        symbol: str,
        company_name: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Fetch and analyze Google News"""
        try:
            articles = await self.google_news.get_stock_news(symbol, company_name, limit=15)
            sentiment_results = self.analyzer.analyze_articles(articles)
            aggregate = self.analyzer.get_aggregate_sentiment(sentiment_results)

            return {
                "score": aggregate["weighted_score"],
                "article_count": len(articles),
                "aggregate": aggregate,
                "top_results": sentiment_results[:5] if sentiment_results else [],
            }

        except Exception as e:
            print(f"Google News sentiment error: {e}")
            return None

    async def _fetch_newsapi_sentiment(
        self,
        symbol: str,
        company_name: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Fetch and analyze NewsAPI (limited calls)"""
        try:
            # Only use if we have remaining requests
            if self.news_api.get_remaining_requests() < 5:
                return None

            articles = await self.news_api.get_stock_news(symbol, company_name, days_back=7)
            sentiment_results = self.analyzer.analyze_articles(articles)
            aggregate = self.analyzer.get_aggregate_sentiment(sentiment_results)

            return {
                "score": aggregate["weighted_score"],
                "article_count": len(articles),
                "aggregate": aggregate,
                "top_results": sentiment_results[:5] if sentiment_results else [],
            }

        except Exception as e:
            print(f"NewsAPI sentiment error: {e}")
            return None

    async def _fetch_reddit_sentiment(
        self,
        symbol: str,
        company_name: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Fetch and analyze Reddit posts"""
        try:
            if not self.reddit.is_available():
                return None

            posts = await self.reddit.get_stock_mentions(symbol, company_name, days_back=7, limit=30)

            if not posts:
                return None

            # Analyze each post
            sentiment_results = []
            for post in posts:
                text = post.title
                if post.body:
                    text += " " + post.body[:500]

                result = self.analyzer.analyze_text(
                    text=text,
                    source=f"Reddit r/{post.subreddit}",
                    timestamp=post.created_at
                )

                # Weight by engagement
                engagement = self.reddit.calculate_engagement_score(post)
                result.financial_adjusted_score *= (0.5 + engagement * 0.5)

                sentiment_results.append(result)

            aggregate = self.analyzer.get_aggregate_sentiment(sentiment_results)

            return {
                "score": aggregate["weighted_score"],
                "post_count": len(posts),
                "aggregate": aggregate,
                "top_results": sentiment_results[:5] if sentiment_results else [],
            }

        except Exception as e:
            print(f"Reddit sentiment error: {e}")
            return None

    async def _fetch_finnhub_sentiment(
        self,
        symbol: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch Finnhub pre-calculated sentiment (US ADRs only)"""
        try:
            # Check if this is an Indian ADR
            adr_symbol = self.finnhub.map_indian_to_adr(symbol)

            if not adr_symbol:
                return None

            sentiment = await self.finnhub.get_news_sentiment(adr_symbol)

            if not sentiment:
                return None

            return {
                "score": sentiment.sentiment_score,
                "article_count": sentiment.buzz_articles_in_last_week,
                "sentiment": sentiment,
            }

        except Exception as e:
            print(f"Finnhub sentiment error: {e}")
            return None

    def _aggregate_results(
        self,
        symbol: str,
        company_name: Optional[str],
        source_results: Dict[str, Optional[Dict]]
    ) -> AggregatedSentiment:
        """Combine results from all sources"""

        # Calculate weighted score
        total_weight = 0.0
        weighted_score = 0.0
        source_breakdown = {}

        news_score = 0.0
        news_count = 0
        social_score = 0.0
        social_count = 0

        all_sentiment_results = []

        for source, result in source_results.items():
            if result is None:
                continue

            weight = self.WEIGHTS.get(source, 0.1)
            score = result.get("score", 0.0)

            weighted_score += score * weight
            total_weight += weight

            # Track by category
            if source in ["rss", "google_news", "newsapi"]:
                news_score += score * weight
                news_count += result.get("article_count", 0)
            elif source == "reddit":
                social_score = score
                social_count = result.get("post_count", 0)

            # Store breakdown
            source_breakdown[source] = {
                "score": score,
                "weight": weight,
                "count": result.get("article_count", result.get("post_count", 0)),
                "available": True,
            }

            # Collect sentiment results for top articles
            if "top_results" in result:
                all_sentiment_results.extend(result["top_results"])

        # Normalize weighted score
        if total_weight > 0:
            weighted_score /= total_weight

        # Calculate confidence based on data availability
        available_sources = sum(1 for r in source_results.values() if r is not None)
        confidence = available_sources / len(self.WEIGHTS)

        # Boost confidence if we have more articles
        total_articles = news_count + social_count
        if total_articles > 20:
            confidence = min(1.0, confidence + 0.1)
        if total_articles > 50:
            confidence = min(1.0, confidence + 0.1)

        # Determine label
        if weighted_score > 0.15:
            label = "bullish"
        elif weighted_score < -0.15:
            label = "bearish"
        else:
            label = "neutral"

        # Get top bullish/bearish articles
        top_bullish = []
        top_bearish = []

        for result in sorted(all_sentiment_results, key=lambda x: x.financial_adjusted_score, reverse=True):
            if result.financial_adjusted_score > 0.2 and len(top_bullish) < 3:
                top_bullish.append({
                    "title": result.text[:100],
                    "source": result.source,
                    "score": result.financial_adjusted_score,
                })
            elif result.financial_adjusted_score < -0.2 and len(top_bearish) < 3:
                top_bearish.append({
                    "title": result.text[:100],
                    "source": result.source,
                    "score": result.financial_adjusted_score,
                })

        return AggregatedSentiment(
            symbol=symbol,
            company_name=company_name,
            timestamp=datetime.now(),
            overall_score=weighted_score,
            overall_label=label,
            confidence=confidence,
            news_score=news_score / max(1, sum(
                self.WEIGHTS[s] for s in ["rss", "google_news", "newsapi"]
                if source_results.get(s) is not None
            )) if news_count > 0 else 0.0,
            news_article_count=news_count,
            social_score=social_score,
            social_post_count=social_count,
            source_breakdown=source_breakdown,
            top_bullish_articles=top_bullish,
            top_bearish_articles=top_bearish,
        )

    async def get_market_sentiment(self) -> Dict[str, Any]:
        """
        Get overall market sentiment

        Returns:
            Dictionary with market-wide sentiment metrics
        """
        # Fetch market news from multiple sources
        results = await asyncio.gather(
            self.rss.get_market_news(limit=30),
            self.google_news.get_market_news(limit=20),
            return_exceptions=True
        )

        all_articles = []
        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)

        # Analyze all articles
        sentiment_results = self.analyzer.analyze_articles(all_articles)
        aggregate = self.analyzer.get_aggregate_sentiment(sentiment_results)

        return {
            "market_sentiment_score": aggregate["weighted_score"],
            "market_sentiment_label": aggregate["sentiment_label"],
            "article_count": len(all_articles),
            "bullish_articles": aggregate["bullish_count"],
            "bearish_articles": aggregate["bearish_count"],
            "neutral_articles": aggregate["neutral_count"],
            "timestamp": datetime.now().isoformat(),
        }

    async def get_sector_sentiment(self, sector: str) -> Dict[str, Any]:
        """
        Get sentiment for a specific sector

        Args:
            sector: Sector name (banking, it, pharma, etc.)

        Returns:
            Dictionary with sector sentiment metrics
        """
        # Fetch sector news
        results = await asyncio.gather(
            self.rss.get_sector_news(sector, limit=20),
            self.google_news.get_sector_news(sector, limit=15),
            return_exceptions=True
        )

        all_articles = []
        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)

        # Analyze
        sentiment_results = self.analyzer.analyze_articles(all_articles)
        aggregate = self.analyzer.get_aggregate_sentiment(sentiment_results)

        return {
            "sector": sector,
            "sentiment_score": aggregate["weighted_score"],
            "sentiment_label": aggregate["sentiment_label"],
            "article_count": len(all_articles),
            "confidence": aggregate["confidence"],
            "timestamp": datetime.now().isoformat(),
        }

    def clear_cache(self):
        """Clear the sentiment cache"""
        self._cache.clear()


# Global instance
sentiment_aggregator = SentimentAggregator()
