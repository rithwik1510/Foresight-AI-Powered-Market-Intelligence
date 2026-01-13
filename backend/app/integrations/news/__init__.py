"""
News Integration Module
Multi-source news aggregation for sentiment analysis
"""
from app.integrations.news.newsapi import NewsAPIClient, newsapi_client
from app.integrations.news.rss_parser import RSSParser, rss_parser
from app.integrations.news.google_news import GoogleNewsClient, google_news_client
from app.integrations.news.finnhub import FinnhubClient, finnhub_client

__all__ = [
    "NewsAPIClient",
    "newsapi_client",
    "RSSParser",
    "rss_parser",
    "GoogleNewsClient",
    "google_news_client",
    "FinnhubClient",
    "finnhub_client",
]
