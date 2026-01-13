"""
Sentiment Analysis Module
Analyzes news and social media sentiment for stock prediction
"""
from app.ml.sentiment.sentiment_analyzer import (
    SentimentAnalyzer,
    sentiment_analyzer,
    SentimentResult
)
from app.ml.sentiment.reddit_scraper import RedditScraper, reddit_scraper
from app.ml.sentiment.aggregator import SentimentAggregator, sentiment_aggregator

__all__ = [
    "SentimentAnalyzer",
    "sentiment_analyzer",
    "SentimentResult",
    "RedditScraper",
    "reddit_scraper",
    "SentimentAggregator",
    "sentiment_aggregator",
]
