"""
Feature Pipeline - Combines all feature sources into unified dataset
Merges technical, global, and sentiment features for ML models
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List, Tuple
import asyncio
import yfinance as yf

from app.ml.features.technical_features import technical_feature_generator
from app.ml.features.global_features import global_feature_generator
from app.ml.sentiment.aggregator import sentiment_aggregator
from app.integrations.events import events_client


class FeaturePipeline:
    """
    Unified Feature Pipeline

    Combines:
    - 60+ Technical features (price, volume, momentum, volatility)
    - 30+ Global features (US markets, commodities, forex, economic)
    - 10+ Sentiment features (news, social media)
    - 5+ Event features (earnings proximity, dividends)

    Total: 105+ features for ML models
    """

    def __init__(self):
        self.technical = technical_feature_generator
        self.global_gen = global_feature_generator
        self.sentiment = sentiment_aggregator
        self.events = events_client

    async def generate_features(
        self,
        symbol: str,
        company_name: Optional[str] = None,
        period: str = "2y",
        include_sentiment: bool = True,
        include_global: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate complete feature set for a stock

        Args:
            symbol: Stock symbol (e.g., RELIANCE.NS)
            company_name: Company name for sentiment search
            period: Historical data period
            include_sentiment: Include sentiment features
            include_global: Include global market features

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        # Fetch stock data
        print(f"Fetching {symbol} data...")
        ticker = yf.Ticker(symbol)
        stock_df = ticker.history(period=period)

        if stock_df.empty:
            raise ValueError(f"No data found for {symbol}")

        print(f"Data shape: {stock_df.shape}, Date range: {stock_df.index[0]} to {stock_df.index[-1]}")

        # Generate technical features
        print("Generating technical features...")
        technical_features = self.technical.generate_features(stock_df)
        print(f"Technical features: {len(technical_features.columns)} columns")

        # Generate global features
        global_features = pd.DataFrame(index=stock_df.index)
        if include_global:
            print("Generating global features...")
            try:
                global_df = await self.global_gen.generate_historical_features(period=period)
                global_features = self.global_gen.align_features_with_stock(global_df, stock_df)
                print(f"Global features: {len(global_features.columns)} columns")
            except Exception as e:
                print(f"Global features error: {e}")

        # Generate current sentiment features
        sentiment_features = pd.DataFrame(index=stock_df.index)
        if include_sentiment:
            print("Fetching sentiment data...")
            try:
                sentiment_data = await self.sentiment.get_stock_sentiment(
                    symbol=symbol,
                    company_name=company_name,
                    use_cache=True
                )
                sentiment_features = self._create_sentiment_features(
                    stock_df.index,
                    sentiment_data
                )
                print(f"Sentiment features: {len(sentiment_features.columns)} columns")
            except Exception as e:
                print(f"Sentiment features error: {e}")

        # Generate event features (earnings, dividends)
        event_features = pd.DataFrame(index=stock_df.index)
        print("Fetching event data...")
        try:
            events_data = await self.events.get_all_events(symbol)
            event_features = self._create_event_features(
                stock_df.index,
                events_data
            )
            print(f"Event features: {len(event_features.columns)} columns")
        except Exception as e:
            print(f"Event features error: {e}")

        # Combine all features
        print("Combining all features...")
        all_features = pd.concat([
            technical_features,
            global_features,
            sentiment_features,
            event_features
        ], axis=1)

        # Remove duplicate columns
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]

        # Create target variable (forward returns)
        print("Creating target variable...")
        target = self._create_target(stock_df, horizon_days=20)

        # Align features and target
        valid_idx = all_features.index.intersection(target.dropna().index)
        all_features = all_features.loc[valid_idx]
        target = target.loc[valid_idx]

        # Drop rows with too many NaN
        nan_threshold = len(all_features.columns) * 0.5
        valid_rows = all_features.isna().sum(axis=1) < nan_threshold
        all_features = all_features[valid_rows]
        target = target[valid_rows]

        # Fill remaining NaN
        all_features = all_features.ffill().bfill()

        print(f"Final features shape: {all_features.shape}")
        print(f"Final target shape: {target.shape}")

        return all_features, target

    def _create_sentiment_features(
        self,
        index: pd.DatetimeIndex,
        sentiment_data
    ) -> pd.DataFrame:
        """Create sentiment features DataFrame"""
        # For historical data, we only have current sentiment
        # In production, you'd want to store historical sentiment
        features = pd.DataFrame(index=index)

        # Use current sentiment for recent data, fill with neutral for older
        features["sentiment_score"] = sentiment_data.overall_score
        features["sentiment_bullish"] = 1.0 if sentiment_data.overall_label == "bullish" else 0.0
        features["sentiment_bearish"] = 1.0 if sentiment_data.overall_label == "bearish" else 0.0
        features["sentiment_neutral"] = 1.0 if sentiment_data.overall_label == "neutral" else 0.0
        features["sentiment_confidence"] = sentiment_data.confidence
        features["sentiment_news_score"] = sentiment_data.news_score
        features["sentiment_social_score"] = sentiment_data.social_score
        features["sentiment_article_count"] = sentiment_data.news_article_count
        features["sentiment_social_count"] = sentiment_data.social_post_count

        return features

    def _create_event_features(
        self,
        index: pd.DatetimeIndex,
        events_data: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Create event-based features DataFrame

        Args:
            index: DatetimeIndex for alignment
            events_data: Events data from events_client

        Returns:
            DataFrame with event features
        """
        features = pd.DataFrame(index=index)

        # Extract earnings data
        earnings = events_data.get("earnings")
        if earnings:
            days_to_earnings = earnings.get("days_until", 999)
            features["days_to_earnings"] = days_to_earnings
            features["has_upcoming_earnings"] = 1.0 if days_to_earnings <= 30 else 0.0

            # Earnings proximity score: 1.0 at earnings day, decreasing to 0.0 at 90 days
            features["earnings_proximity_score"] = max(0.0, 1.0 - (abs(days_to_earnings) / 90))

            # Earnings week flag (within 7 days)
            features["earnings_week"] = 1.0 if abs(days_to_earnings) <= 7 else 0.0
        else:
            features["days_to_earnings"] = 999
            features["has_upcoming_earnings"] = 0.0
            features["earnings_proximity_score"] = 0.0
            features["earnings_week"] = 0.0

        # Extract dividend data
        dividends = events_data.get("dividends")
        if dividends:
            days_since_dividend = dividends.get("days_since_last", 999)
            days_until_dividend = dividends.get("days_until_next", 999)

            features["days_since_dividend"] = days_since_dividend
            features["days_until_dividend"] = days_until_dividend
            features["has_upcoming_dividend"] = 1.0 if days_until_dividend <= 60 else 0.0

            # Dividend proximity score
            features["dividend_proximity_score"] = max(0.0, 1.0 - (abs(days_until_dividend) / 180))
        else:
            features["days_since_dividend"] = 999
            features["days_until_dividend"] = 999
            features["has_upcoming_dividend"] = 0.0
            features["dividend_proximity_score"] = 0.0

        return features

    def _create_target(
        self,
        df: pd.DataFrame,
        horizon_days: int = 20
    ) -> pd.Series:
        """Create target variable (forward returns)"""
        return df['Close'].pct_change(horizon_days).shift(-horizon_days)

    async def generate_prediction_features(
        self,
        symbol: str,
        company_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate features for real-time prediction

        Args:
            symbol: Stock symbol
            company_name: Company name

        Returns:
            DataFrame with latest features
        """
        # Fetch recent stock data
        ticker = yf.Ticker(symbol)
        stock_df = ticker.history(period="6mo")

        if stock_df.empty:
            raise ValueError(f"No data found for {symbol}")

        # Generate technical features
        technical_features = self.technical.generate_features(stock_df)

        # Generate current global features
        try:
            global_dict = await self.global_gen.generate_features_async(include_economic=True)
            # Convert to single-row DataFrame
            global_features = pd.DataFrame([global_dict], index=[stock_df.index[-1]])
        except Exception as e:
            print(f"Global features error: {e}")
            global_features = pd.DataFrame(index=[stock_df.index[-1]])

        # Get sentiment features
        try:
            sentiment_data = await self.sentiment.get_stock_sentiment(
                symbol=symbol,
                company_name=company_name,
                use_cache=True
            )
            sentiment_features = pd.DataFrame([{
                "sentiment_score": sentiment_data.overall_score,
                "sentiment_bullish": 1.0 if sentiment_data.overall_label == "bullish" else 0.0,
                "sentiment_bearish": 1.0 if sentiment_data.overall_label == "bearish" else 0.0,
                "sentiment_neutral": 1.0 if sentiment_data.overall_label == "neutral" else 0.0,
                "sentiment_confidence": sentiment_data.confidence,
                "sentiment_news_score": sentiment_data.news_score,
                "sentiment_social_score": sentiment_data.social_score,
            }], index=[stock_df.index[-1]])
        except Exception as e:
            print(f"Sentiment features error: {e}")
            sentiment_features = pd.DataFrame(index=[stock_df.index[-1]])

        # Combine (only latest row)
        all_features = pd.concat([
            technical_features.iloc[[-1]],
            global_features,
            sentiment_features
        ], axis=1)

        # Remove duplicate columns
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]

        # Fill NaN
        all_features = all_features.ffill().bfill().fillna(0)

        return all_features

    def get_feature_importance_summary(
        self,
        model_importances: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Summarize feature importance across models

        Args:
            model_importances: Dict of model_name -> feature_importances

        Returns:
            Summary with top features by category
        """
        # Aggregate importances
        all_importances = {}
        for model_name, importances in model_importances.items():
            for feature, importance in importances.items():
                if feature not in all_importances:
                    all_importances[feature] = []
                all_importances[feature].append(importance)

        # Average across models
        avg_importances = {
            feature: np.mean(scores)
            for feature, scores in all_importances.items()
        }

        # Categorize features
        technical_features = {k: v for k, v in avg_importances.items()
                            if any(x in k.lower() for x in ['return', 'sma', 'ema', 'rsi', 'macd', 'volume', 'atr', 'bb_'])}
        global_features = {k: v for k, v in avg_importances.items()
                         if any(x in k.lower() for x in ['sp500', 'nasdaq', 'vix', 'gold', 'oil', 'usd', 'fed', 'inflation'])}
        sentiment_features = {k: v for k, v in avg_importances.items()
                            if 'sentiment' in k.lower()}

        return {
            "top_overall": dict(sorted(avg_importances.items(), key=lambda x: x[1], reverse=True)[:10]),
            "top_technical": dict(sorted(technical_features.items(), key=lambda x: x[1], reverse=True)[:5]),
            "top_global": dict(sorted(global_features.items(), key=lambda x: x[1], reverse=True)[:5]),
            "top_sentiment": dict(sorted(sentiment_features.items(), key=lambda x: x[1], reverse=True)[:5]),
            "category_importance": {
                "technical": sum(technical_features.values()),
                "global": sum(global_features.values()),
                "sentiment": sum(sentiment_features.values()),
            }
        }

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        technical_names = self.technical.feature_config.get_feature_names() if hasattr(self.technical, 'feature_config') else []
        global_names = self.global_gen.get_feature_names()
        sentiment_names = [
            "sentiment_score", "sentiment_bullish", "sentiment_bearish",
            "sentiment_neutral", "sentiment_confidence", "sentiment_news_score",
            "sentiment_social_score", "sentiment_article_count", "sentiment_social_count"
        ]

        return technical_names + global_names + sentiment_names


# Global instance
feature_pipeline = FeaturePipeline()
