"""
Phase 3 Test Script - Sentiment Analysis
Tests news fetchers and sentiment analysis pipeline
"""
import asyncio
import sys
sys.path.insert(0, '.')


async def test_sentiment_pipeline():
    """Test all sentiment analysis components"""
    print("=" * 60)
    print("PHASE 3: SENTIMENT ANALYSIS TEST")
    print("=" * 60)

    # Test 1: Import all modules
    print("\n[1] Testing imports...")
    try:
        from app.integrations.news import (
            newsapi_client,
            rss_parser,
            google_news_client,
            finnhub_client
        )
        from app.ml.sentiment import (
            sentiment_analyzer,
            reddit_scraper,
            sentiment_aggregator
        )
        print("   All imports successful!")
    except ImportError as e:
        print(f"   Import error: {e}")
        return False

    # Test 2: RSS Parser (unlimited, primary source)
    print("\n[2] Testing RSS Parser...")
    try:
        articles = await rss_parser.get_market_news(limit=10)
        print(f"   Fetched {len(articles)} market news articles")
        if articles:
            print(f"   Sample: {articles[0].title[:60]}...")
            print(f"   Source: {articles[0].source}")

        # Test stock-specific news
        reliance_news = await rss_parser.get_stock_news("RELIANCE", "Reliance Industries", limit=5)
        print(f"   Reliance news: {len(reliance_news)} articles")
        print("   RSS Parser: PASSED")
    except Exception as e:
        print(f"   RSS Parser error: {e}")

    # Test 3: Google News (unlimited)
    print("\n[3] Testing Google News...")
    try:
        articles = await google_news_client.get_stock_news("RELIANCE.NS", "Reliance Industries", limit=5)
        print(f"   Fetched {len(articles)} Google News articles")
        if articles:
            print(f"   Sample: {articles[0].title[:60]}...")
            print(f"   Source: {articles[0].source}")
        print("   Google News: PASSED")
    except Exception as e:
        print(f"   Google News error: {e}")

    # Test 4: NewsAPI (100/day limit)
    print("\n[4] Testing NewsAPI...")
    try:
        remaining = newsapi_client.get_remaining_requests()
        print(f"   Remaining daily requests: {remaining}")

        if remaining > 5:
            articles = await newsapi_client.get_headlines(category="business", page_size=5)
            print(f"   Fetched {len(articles)} headlines")
            if articles:
                print(f"   Sample: {articles[0].title[:60]}...")
            print("   NewsAPI: PASSED")
        else:
            print("   NewsAPI: SKIPPED (saving daily quota)")
    except Exception as e:
        print(f"   NewsAPI error: {e}")

    # Test 5: Finnhub (60/min limit)
    print("\n[5] Testing Finnhub...")
    try:
        remaining = finnhub_client.get_remaining_requests()
        print(f"   Remaining minute requests: {remaining}")

        # Get market news
        articles = await finnhub_client.get_market_news(category="general")
        print(f"   Fetched {len(articles)} market news articles")

        # Test Indian ADR sentiment
        adr_sentiment = await finnhub_client.get_news_sentiment("INFY")
        if adr_sentiment:
            print(f"   INFY sentiment: {adr_sentiment.sentiment_label} ({adr_sentiment.sentiment_score:.2f})")
        print("   Finnhub: PASSED")
    except Exception as e:
        print(f"   Finnhub error: {e}")

    # Test 6: Sentiment Analyzer
    print("\n[6] Testing Sentiment Analyzer...")
    try:
        # Test single text
        result = sentiment_analyzer.analyze_text(
            "Reliance Industries reports record profits, stock surges 5%",
            source="test"
        )
        print(f"   Text: 'Reliance Industries reports record profits, stock surges 5%'")
        print(f"   VADER Compound: {result.vader_compound:.3f}")
        print(f"   Financial Adjusted: {result.financial_adjusted_score:.3f}")
        print(f"   Label: {result.label}")

        # Test negative text
        result2 = sentiment_analyzer.analyze_text(
            "Company misses earnings estimates, shares plunge on weak guidance",
            source="test"
        )
        print(f"\n   Text: 'Company misses earnings estimates, shares plunge...'")
        print(f"   Financial Adjusted: {result2.financial_adjusted_score:.3f}")
        print(f"   Label: {result2.label}")

        print("   Sentiment Analyzer: PASSED")
    except Exception as e:
        print(f"   Sentiment Analyzer error: {e}")
        import traceback
        traceback.print_exc()

    # Test 7: Reddit Scraper
    print("\n[7] Testing Reddit Scraper...")
    try:
        if reddit_scraper.is_available():
            posts = await reddit_scraper.get_market_sentiment_posts(limit=5)
            print(f"   Fetched {len(posts)} Reddit posts")
            if posts:
                print(f"   Sample: {posts[0].title[:60]}...")
                print(f"   Subreddit: r/{posts[0].subreddit}")
                print(f"   Score: {posts[0].score}")
            print("   Reddit Scraper: PASSED")
        else:
            print("   Reddit: NOT CONFIGURED (need REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET)")
            print("   Reddit Scraper: SKIPPED")
    except Exception as e:
        print(f"   Reddit Scraper error: {e}")

    # Test 8: Sentiment Aggregator (Full Pipeline)
    print("\n[8] Testing Sentiment Aggregator...")
    try:
        print("   Fetching aggregated sentiment for RELIANCE.NS...")
        sentiment = await sentiment_aggregator.get_stock_sentiment(
            symbol="RELIANCE.NS",
            company_name="Reliance Industries",
            use_cache=False
        )

        print(f"\n   === RELIANCE.NS Sentiment ===")
        print(f"   Overall Score: {sentiment.overall_score:.3f}")
        print(f"   Label: {sentiment.overall_label}")
        print(f"   Confidence: {sentiment.confidence:.2%}")
        print(f"   News Score: {sentiment.news_score:.3f} ({sentiment.news_article_count} articles)")
        print(f"   Social Score: {sentiment.social_score:.3f} ({sentiment.social_post_count} posts)")

        print("\n   Source Breakdown:")
        for source, data in sentiment.source_breakdown.items():
            if data.get("available"):
                print(f"   - {source}: {data['score']:.3f} (weight: {data['weight']:.0%}, count: {data['count']})")

        if sentiment.top_bullish_articles:
            print("\n   Top Bullish:")
            for article in sentiment.top_bullish_articles[:2]:
                print(f"   + {article['title'][:50]}... ({article['score']:.2f})")

        if sentiment.top_bearish_articles:
            print("\n   Top Bearish:")
            for article in sentiment.top_bearish_articles[:2]:
                print(f"   - {article['title'][:50]}... ({article['score']:.2f})")

        print("\n   Sentiment Aggregator: PASSED")
    except Exception as e:
        print(f"   Sentiment Aggregator error: {e}")
        import traceback
        traceback.print_exc()

    # Test 9: Market Sentiment
    print("\n[9] Testing Market-Wide Sentiment...")
    try:
        market = await sentiment_aggregator.get_market_sentiment()
        print(f"   Market Sentiment: {market['market_sentiment_label']} ({market['market_sentiment_score']:.3f})")
        print(f"   Articles analyzed: {market['article_count']}")
        print(f"   Bullish: {market['bullish_articles']}, Bearish: {market['bearish_articles']}, Neutral: {market['neutral_articles']}")
        print("   Market Sentiment: PASSED")
    except Exception as e:
        print(f"   Market Sentiment error: {e}")

    print("\n" + "=" * 60)
    print("PHASE 3 TESTS COMPLETE")
    print("=" * 60)
    return True


if __name__ == "__main__":
    asyncio.run(test_sentiment_pipeline())
