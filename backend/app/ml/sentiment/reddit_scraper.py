"""
Reddit Scraper - PRAW for social sentiment analysis
Free tier: 60 requests/minute
"""
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
import asyncio

try:
    import praw
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False

from app.config import get_settings


@dataclass
class RedditPost:
    """Reddit post/comment data"""
    title: str
    body: str
    subreddit: str
    author: str
    score: int
    upvote_ratio: float
    num_comments: int
    created_at: datetime
    url: str
    is_comment: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "body": self.body[:500] if self.body else "",
            "subreddit": self.subreddit,
            "author": self.author,
            "score": self.score,
            "upvote_ratio": self.upvote_ratio,
            "num_comments": self.num_comments,
            "created_at": self.created_at.isoformat(),
            "url": self.url,
            "is_comment": self.is_comment,
        }


class RedditScraper:
    """
    Reddit Scraper for Indian Stock Market Sentiment

    Uses PRAW (Python Reddit API Wrapper)
    Free tier: 60 requests/minute (OAuth)

    Target subreddits:
    - r/IndiaInvestments - Main Indian investment community
    - r/IndianStreetBets - Retail traders, high sentiment
    - r/IndianStockMarket - Stock-specific discussions
    - r/DalalStreetBets - Memes and sentiment
    """

    # Indian finance subreddits (ordered by quality)
    SUBREDDITS = [
        "IndiaInvestments",      # High quality, moderated
        "IndianStreetBets",      # Retail sentiment, memes
        "IndianStockMarket",     # Stock discussions
        "DalalStreetBets",       # High volatility sentiment
        "indiainvestments",      # Alternative capitalization
    ]

    def __init__(self):
        self.settings = get_settings()
        self.reddit = None
        self._initialized = False
        self._init_reddit()

    def _init_reddit(self):
        """Initialize Reddit client"""
        if not PRAW_AVAILABLE:
            print("Reddit: PRAW not installed")
            return

        client_id = self.settings.REDDIT_CLIENT_ID
        client_secret = self.settings.REDDIT_CLIENT_SECRET
        user_agent = self.settings.REDDIT_USER_AGENT

        if not client_id or not client_secret:
            print("Reddit: No API credentials configured")
            return

        try:
            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent,
            )
            self._initialized = True
            print("Reddit: Client initialized")
        except Exception as e:
            print(f"Reddit: Initialization error: {e}")

    def is_available(self) -> bool:
        """Check if Reddit client is available"""
        return self._initialized and self.reddit is not None

    async def get_subreddit_posts(
        self,
        subreddit_name: str,
        sort: str = "hot",
        limit: int = 25,
        time_filter: str = "week"
    ) -> List[RedditPost]:
        """
        Get posts from a subreddit

        Args:
            subreddit_name: Name of subreddit
            sort: hot, new, top, rising
            limit: Maximum posts
            time_filter: For top: hour, day, week, month, year, all

        Returns:
            List of RedditPost objects
        """
        if not self.is_available():
            return []

        try:
            subreddit = self.reddit.subreddit(subreddit_name)

            # Get posts based on sort method
            if sort == "hot":
                posts = subreddit.hot(limit=limit)
            elif sort == "new":
                posts = subreddit.new(limit=limit)
            elif sort == "top":
                posts = subreddit.top(time_filter=time_filter, limit=limit)
            elif sort == "rising":
                posts = subreddit.rising(limit=limit)
            else:
                posts = subreddit.hot(limit=limit)

            results = []
            for post in posts:
                results.append(RedditPost(
                    title=post.title,
                    body=post.selftext if hasattr(post, 'selftext') else "",
                    subreddit=subreddit_name,
                    author=str(post.author) if post.author else "[deleted]",
                    score=post.score,
                    upvote_ratio=post.upvote_ratio if hasattr(post, 'upvote_ratio') else 0.5,
                    num_comments=post.num_comments,
                    created_at=datetime.fromtimestamp(post.created_utc),
                    url=f"https://reddit.com{post.permalink}",
                    is_comment=False
                ))

            return results

        except Exception as e:
            print(f"Reddit fetch error ({subreddit_name}): {e}")
            return []

    async def search_posts(
        self,
        query: str,
        subreddits: Optional[List[str]] = None,
        sort: str = "relevance",
        time_filter: str = "week",
        limit: int = 25
    ) -> List[RedditPost]:
        """
        Search for posts across subreddits

        Args:
            query: Search query
            subreddits: List of subreddits to search (None for all)
            sort: relevance, hot, top, new, comments
            time_filter: hour, day, week, month, year, all
            limit: Maximum posts

        Returns:
            List of RedditPost objects
        """
        if not self.is_available():
            return []

        try:
            # Build subreddit string
            if subreddits:
                subreddit_str = "+".join(subreddits)
            else:
                subreddit_str = "+".join(self.SUBREDDITS)

            subreddit = self.reddit.subreddit(subreddit_str)

            results = []
            for post in subreddit.search(
                query,
                sort=sort,
                time_filter=time_filter,
                limit=limit
            ):
                results.append(RedditPost(
                    title=post.title,
                    body=post.selftext if hasattr(post, 'selftext') else "",
                    subreddit=str(post.subreddit),
                    author=str(post.author) if post.author else "[deleted]",
                    score=post.score,
                    upvote_ratio=post.upvote_ratio if hasattr(post, 'upvote_ratio') else 0.5,
                    num_comments=post.num_comments,
                    created_at=datetime.fromtimestamp(post.created_utc),
                    url=f"https://reddit.com{post.permalink}",
                    is_comment=False
                ))

            return results

        except Exception as e:
            print(f"Reddit search error: {e}")
            return []

    async def get_stock_mentions(
        self,
        symbol: str,
        company_name: Optional[str] = None,
        days_back: int = 7,
        limit: int = 50
    ) -> List[RedditPost]:
        """
        Get Reddit posts mentioning a specific stock

        Args:
            symbol: Stock symbol (e.g., RELIANCE.NS)
            company_name: Company name for better search
            days_back: Days to look back
            limit: Maximum posts

        Returns:
            List of RedditPost objects
        """
        # Clean symbol
        clean_symbol = symbol.replace(".NS", "").replace(".BO", "")

        # Build search query
        queries = [clean_symbol]
        if company_name:
            queries.append(f'"{company_name}"')

        all_posts = []

        for query in queries:
            posts = await self.search_posts(
                query=query,
                subreddits=self.SUBREDDITS,
                sort="relevance",
                time_filter=self._days_to_time_filter(days_back),
                limit=limit // len(queries)
            )
            all_posts.extend(posts)

        # Remove duplicates by URL
        seen_urls = set()
        unique_posts = []
        for post in all_posts:
            if post.url not in seen_urls:
                seen_urls.add(post.url)
                unique_posts.append(post)

        # Sort by score (engagement)
        unique_posts.sort(key=lambda x: x.score, reverse=True)
        return unique_posts[:limit]

    async def get_market_sentiment_posts(
        self,
        limit: int = 50
    ) -> List[RedditPost]:
        """
        Get general market sentiment posts

        Args:
            limit: Maximum posts

        Returns:
            List of RedditPost objects
        """
        all_posts = []

        # Get hot posts from each subreddit
        posts_per_sub = limit // len(self.SUBREDDITS)
        for subreddit in self.SUBREDDITS:
            posts = await self.get_subreddit_posts(
                subreddit_name=subreddit,
                sort="hot",
                limit=posts_per_sub
            )
            all_posts.extend(posts)

        # Sort by score
        all_posts.sort(key=lambda x: x.score, reverse=True)
        return all_posts[:limit]

    async def get_sector_posts(
        self,
        sector: str,
        limit: int = 25
    ) -> List[RedditPost]:
        """
        Get posts related to a sector

        Args:
            sector: Sector name
            limit: Maximum posts

        Returns:
            List of RedditPost objects
        """
        # Sector keywords
        sector_keywords = {
            "banking": "bank HDFC ICICI SBI",
            "it": "IT TCS Infosys Wipro tech",
            "pharma": "pharma medicine healthcare",
            "auto": "auto Tata Motors Maruti",
            "energy": "oil gas Reliance ONGC",
            "fmcg": "FMCG HUL ITC consumer",
            "metal": "metal steel Tata Steel JSW",
            "realty": "real estate property DLF",
        }

        query = sector_keywords.get(sector.lower(), sector)
        return await self.search_posts(query=query, limit=limit)

    def _days_to_time_filter(self, days: int) -> str:
        """Convert days to Reddit time filter"""
        if days <= 1:
            return "day"
        elif days <= 7:
            return "week"
        elif days <= 30:
            return "month"
        elif days <= 365:
            return "year"
        else:
            return "all"

    def calculate_engagement_score(self, post: RedditPost) -> float:
        """
        Calculate engagement score for a post

        Higher score = more engagement/visibility

        Args:
            post: RedditPost object

        Returns:
            Engagement score (0-1)
        """
        # Factors: score, upvote_ratio, comments
        # Normalize each factor and combine

        # Score (log scale, cap at 1000)
        score_factor = min(1.0, (post.score + 1) / 1000)

        # Upvote ratio (already 0-1)
        ratio_factor = post.upvote_ratio

        # Comments (log scale, cap at 100)
        comment_factor = min(1.0, (post.num_comments + 1) / 100)

        # Weighted combination
        engagement = (
            score_factor * 0.4 +
            ratio_factor * 0.3 +
            comment_factor * 0.3
        )

        return engagement

    def get_weighted_text(self, posts: List[RedditPost]) -> str:
        """
        Combine post texts with engagement weighting

        Args:
            posts: List of RedditPost objects

        Returns:
            Combined text weighted by engagement
        """
        weighted_texts = []

        for post in posts:
            engagement = self.calculate_engagement_score(post)

            # Include title always
            text = post.title

            # Include body for high-engagement posts
            if engagement > 0.3 and post.body:
                text += " " + post.body[:300]

            # Repeat based on engagement (more engagement = more weight)
            repeat = max(1, int(engagement * 3))
            weighted_texts.extend([text] * repeat)

        return " ".join(weighted_texts)


# Global instance
reddit_scraper = RedditScraper()
