"""
Sentiment Analyzer - VADER + TextBlob for financial text analysis
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime
import re

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

from app.integrations.news.newsapi import NewsArticle


@dataclass
class SentimentResult:
    """Result of sentiment analysis"""
    text: str
    source: str
    timestamp: datetime

    # VADER scores (-1 to 1)
    vader_compound: float
    vader_positive: float
    vader_negative: float
    vader_neutral: float

    # TextBlob scores
    textblob_polarity: float  # -1 to 1
    textblob_subjectivity: float  # 0 to 1

    # Combined score
    combined_score: float  # -1 to 1
    label: str  # bullish, bearish, neutral

    # Financial-specific adjustments
    financial_adjusted_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text[:200],  # Truncate for storage
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "vader_compound": self.vader_compound,
            "textblob_polarity": self.textblob_polarity,
            "combined_score": self.combined_score,
            "label": self.label,
            "financial_adjusted_score": self.financial_adjusted_score,
        }


class SentimentAnalyzer:
    """
    Financial Sentiment Analyzer

    Combines VADER (optimized for social media) with TextBlob
    for robust sentiment analysis of financial news.

    Includes financial-specific adjustments for:
    - Earnings keywords (beat, miss, exceed)
    - Market sentiment (bullish, bearish, rally, crash)
    - Analyst actions (upgrade, downgrade, target)
    """

    # Financial keywords with sentiment weights
    FINANCIAL_POSITIVE = {
        # Earnings
        "beat": 0.3, "beats": 0.3, "exceeded": 0.4, "exceeds": 0.4,
        "surprise": 0.2, "outperform": 0.4, "outperforms": 0.4,
        # Market
        "rally": 0.3, "rallies": 0.3, "surge": 0.4, "surges": 0.4,
        "gain": 0.2, "gains": 0.2, "profit": 0.2, "profits": 0.2,
        "bullish": 0.4, "bull": 0.3, "boom": 0.3,
        "growth": 0.2, "growing": 0.2, "expansion": 0.2,
        "record": 0.2, "high": 0.1, "highs": 0.2,
        # Analyst
        "upgrade": 0.4, "upgrades": 0.4, "buy": 0.3,
        "overweight": 0.3, "accumulate": 0.3,
        "target raised": 0.4, "raised target": 0.4,
        # General
        "strong": 0.2, "robust": 0.2, "solid": 0.2,
        "positive": 0.2, "optimistic": 0.3,
    }

    FINANCIAL_NEGATIVE = {
        # Earnings
        "miss": 0.3, "misses": 0.3, "missed": 0.3,
        "disappoint": 0.3, "disappoints": 0.3, "disappointing": 0.3,
        "underperform": 0.4, "underperforms": 0.4,
        # Market
        "crash": 0.5, "crashes": 0.5, "plunge": 0.4, "plunges": 0.4,
        "fall": 0.2, "falls": 0.2, "drop": 0.2, "drops": 0.2,
        "bearish": 0.4, "bear": 0.3, "bust": 0.3,
        "decline": 0.2, "declining": 0.2, "contraction": 0.2,
        "loss": 0.3, "losses": 0.3, "low": 0.1, "lows": 0.2,
        # Analyst
        "downgrade": 0.4, "downgrades": 0.4, "sell": 0.3,
        "underweight": 0.3, "avoid": 0.3,
        "target cut": 0.4, "cut target": 0.4,
        # Risk
        "risk": 0.1, "risky": 0.2, "warning": 0.2,
        "concern": 0.2, "concerns": 0.2, "worried": 0.2,
        "volatile": 0.1, "volatility": 0.1,
        # General
        "weak": 0.2, "poor": 0.2, "negative": 0.2,
        "pessimistic": 0.3, "uncertain": 0.2,
    }

    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()

        # Add financial terms to VADER lexicon
        self._enhance_vader_lexicon()

    def _enhance_vader_lexicon(self):
        """Add financial terms to VADER's lexicon"""
        for word, score in self.FINANCIAL_POSITIVE.items():
            self.vader.lexicon[word] = score * 3  # VADER uses -4 to 4 scale

        for word, score in self.FINANCIAL_NEGATIVE.items():
            self.vader.lexicon[word] = -score * 3

    def _calculate_financial_adjustment(self, text: str) -> float:
        """
        Calculate sentiment adjustment based on financial keywords

        Args:
            text: Text to analyze

        Returns:
            Adjustment value (-1 to 1)
        """
        text_lower = text.lower()
        adjustment = 0.0

        # Check positive keywords
        for keyword, weight in self.FINANCIAL_POSITIVE.items():
            if keyword in text_lower:
                adjustment += weight

        # Check negative keywords
        for keyword, weight in self.FINANCIAL_NEGATIVE.items():
            if keyword in text_lower:
                adjustment -= weight

        # Clamp to -1 to 1
        return max(-1.0, min(1.0, adjustment))

    def analyze_text(
        self,
        text: str,
        source: str = "unknown",
        timestamp: Optional[datetime] = None
    ) -> SentimentResult:
        """
        Analyze sentiment of a single text

        Args:
            text: Text to analyze
            source: Source of the text (for tracking)
            timestamp: When the text was published

        Returns:
            SentimentResult with detailed scores
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Clean text
        clean_text = self._clean_text(text)

        # VADER analysis
        vader_scores = self.vader.polarity_scores(clean_text)

        # TextBlob analysis
        blob = TextBlob(clean_text)

        # Financial adjustment
        financial_adj = self._calculate_financial_adjustment(clean_text)

        # Combine scores (weighted average)
        # VADER compound: 60%, TextBlob polarity: 20%, Financial adjustment: 20%
        combined = (
            vader_scores['compound'] * 0.6 +
            blob.sentiment.polarity * 0.2 +
            financial_adj * 0.2
        )

        # Apply financial adjustment to final score
        financial_adjusted = combined + (financial_adj * 0.3)
        financial_adjusted = max(-1.0, min(1.0, financial_adjusted))

        # Determine label
        if financial_adjusted > 0.15:
            label = "bullish"
        elif financial_adjusted < -0.15:
            label = "bearish"
        else:
            label = "neutral"

        return SentimentResult(
            text=text,
            source=source,
            timestamp=timestamp,
            vader_compound=vader_scores['compound'],
            vader_positive=vader_scores['pos'],
            vader_negative=vader_scores['neg'],
            vader_neutral=vader_scores['neu'],
            textblob_polarity=blob.sentiment.polarity,
            textblob_subjectivity=blob.sentiment.subjectivity,
            combined_score=combined,
            label=label,
            financial_adjusted_score=financial_adjusted
        )

    def analyze_article(self, article: NewsArticle) -> SentimentResult:
        """
        Analyze sentiment of a news article

        Args:
            article: NewsArticle object

        Returns:
            SentimentResult
        """
        # Combine title and description for analysis
        text = article.title
        if article.description:
            text += " " + article.description

        return self.analyze_text(
            text=text,
            source=article.source,
            timestamp=article.published_at
        )

    def analyze_articles(self, articles: List[NewsArticle]) -> List[SentimentResult]:
        """
        Analyze sentiment of multiple articles

        Args:
            articles: List of NewsArticle objects

        Returns:
            List of SentimentResult objects
        """
        return [self.analyze_article(article) for article in articles]

    def get_aggregate_sentiment(
        self,
        results: List[SentimentResult],
        time_decay: bool = True
    ) -> Dict[str, Any]:
        """
        Calculate aggregate sentiment from multiple results

        Args:
            results: List of SentimentResult objects
            time_decay: Apply time-based weighting (recent = higher weight)

        Returns:
            Dictionary with aggregate metrics
        """
        if not results:
            return {
                "average_score": 0.0,
                "weighted_score": 0.0,
                "bullish_count": 0,
                "bearish_count": 0,
                "neutral_count": 0,
                "total_articles": 0,
                "sentiment_label": "neutral",
                "confidence": 0.0,
            }

        # Calculate time weights if enabled
        if time_decay:
            now = datetime.now()
            weights = []
            for r in results:
                hours_old = (now - r.timestamp).total_seconds() / 3600
                # Exponential decay: half-life of 24 hours
                weight = 0.5 ** (hours_old / 24)
                weights.append(weight)
        else:
            weights = [1.0] * len(results)

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Calculate weighted average
        weighted_score = sum(
            r.financial_adjusted_score * w
            for r, w in zip(results, weights)
        )

        # Simple average
        average_score = sum(r.financial_adjusted_score for r in results) / len(results)

        # Count labels
        bullish_count = sum(1 for r in results if r.label == "bullish")
        bearish_count = sum(1 for r in results if r.label == "bearish")
        neutral_count = sum(1 for r in results if r.label == "neutral")

        # Determine overall label
        if weighted_score > 0.15:
            sentiment_label = "bullish"
        elif weighted_score < -0.15:
            sentiment_label = "bearish"
        else:
            sentiment_label = "neutral"

        # Calculate confidence (agreement among articles)
        dominant_count = max(bullish_count, bearish_count, neutral_count)
        confidence = dominant_count / len(results)

        return {
            "average_score": average_score,
            "weighted_score": weighted_score,
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "neutral_count": neutral_count,
            "total_articles": len(results),
            "sentiment_label": sentiment_label,
            "confidence": confidence,
        }

    def _clean_text(self, text: str) -> str:
        """Clean text for analysis"""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def analyze_stock_mention(
        self,
        text: str,
        symbol: str,
        company_name: Optional[str] = None
    ) -> Optional[SentimentResult]:
        """
        Analyze text only if it mentions the stock

        Args:
            text: Text to analyze
            symbol: Stock symbol to look for
            company_name: Company name to look for

        Returns:
            SentimentResult if stock mentioned, None otherwise
        """
        text_lower = text.lower()
        clean_symbol = symbol.replace(".NS", "").replace(".BO", "").lower()

        # Check if stock is mentioned
        mentioned = clean_symbol in text_lower
        if company_name and not mentioned:
            mentioned = company_name.lower() in text_lower

        if mentioned:
            return self.analyze_text(text, source="stock_mention")

        return None


# Global instance
sentiment_analyzer = SentimentAnalyzer()
