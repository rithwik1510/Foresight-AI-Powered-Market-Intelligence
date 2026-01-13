"""
Prediction API Endpoints
ML-powered stock price predictions
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from typing import Optional, List
from datetime import datetime
import asyncio

from app.schemas.prediction import (
    PredictionResponse,
    SentimentResponse,
    GlobalFactorsResponse,
    BacktestRequest,
    BacktestResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelStatusResponse,
    DirectionEnum,
    RiskLevel,
    ModelPrediction
)
from app.ml.prediction import ensemble_predictor, EnsemblePredictor
from app.ml.features import feature_pipeline, technical_feature_generator
from app.ml.sentiment import sentiment_aggregator
from app.integrations.global_markets import global_markets_client
from app.ml.backtesting import backtester

import yfinance as yf
import pandas as pd

router = APIRouter(prefix="/predictions", tags=["predictions"])

# Cache for trained models per symbol
_trained_ensembles = {}


async def get_or_train_ensemble(symbol: str) -> EnsemblePredictor:
    """Get cached ensemble or train new one"""
    global _trained_ensembles

    # Check if we have a recently trained model
    if symbol in _trained_ensembles:
        ensemble, trained_at = _trained_ensembles[symbol]
        # Use cached if less than 1 hour old
        if (datetime.now() - trained_at).seconds < 3600:
            return ensemble

    # Train new ensemble
    print(f"Training ensemble for {symbol}...")

    # Fetch data
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="2y")

    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {symbol}")

    # Generate features
    features = technical_feature_generator.generate_features(df)
    target = df['Close'].pct_change(20).shift(-20)
    prices = df['Close']

    # Align
    valid_idx = features.index.intersection(target.dropna().index)
    features = features.loc[valid_idx]
    target = target.loc[valid_idx]
    prices = prices.loc[valid_idx]

    features = features.ffill().bfill()

    # Train
    ensemble = EnsemblePredictor()
    ensemble.train_all(features, target, prices)

    # Cache
    _trained_ensembles[symbol] = (ensemble, datetime.now())

    return ensemble


@router.get("/{symbol}", response_model=PredictionResponse)
async def get_prediction(
    symbol: str,
    horizon: int = Query(default=30, ge=5, le=90, description="Prediction horizon in days")
):
    """
    Get ML prediction for a stock

    - **symbol**: Stock symbol (e.g., RELIANCE.NS)
    - **horizon**: Prediction horizon in days (5-90)
    """
    try:
        # Add .NS suffix if not present (assume NSE)
        if not symbol.endswith(('.NS', '.BO')):
            symbol = f"{symbol}.NS"

        # Get current price
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d")
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")

        current_price = hist['Close'].iloc[-1]

        # Get or train ensemble
        ensemble = await get_or_train_ensemble(symbol)

        # Generate features for prediction
        pred_features = await feature_pipeline.generate_prediction_features(
            symbol=symbol,
            company_name=None
        )

        # Get prediction
        prediction = ensemble.predict(
            X=pred_features,
            current_price=current_price,
            symbol=symbol,
            horizon_days=horizon
        )

        # Convert to response
        return PredictionResponse(
            symbol=symbol,
            prediction_date=prediction.prediction_date,
            horizon_days=horizon,
            direction=DirectionEnum(prediction.direction.value),
            direction_probability=round(prediction.direction_probability, 4),
            current_price=round(current_price, 2),
            predicted_price=round(prediction.predicted_price, 2),
            predicted_return_pct=round(prediction.predicted_return * 100, 2),
            price_range=[round(prediction.price_lower, 2), round(prediction.price_upper, 2)],
            confidence=round(prediction.confidence, 4),
            model_agreement=round(prediction.model_agreement, 4),
            risk_level=RiskLevel(prediction.risk_level),
            top_bullish_factors=prediction.top_bullish_factors[:3],
            top_bearish_factors=prediction.top_bearish_factors[:3],
            model_breakdown={
                name: ModelPrediction(
                    direction=DirectionEnum(pred.direction.value),
                    predicted_return=round(pred.predicted_return * 100, 2),
                    confidence=round(pred.confidence, 4),
                    weight=round(prediction.model_weights.get(name, 0), 4)
                )
                for name, pred in prediction.model_predictions.items()
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.get("/{symbol}/sentiment", response_model=SentimentResponse)
async def get_sentiment(
    symbol: str,
    company_name: Optional[str] = None
):
    """
    Get sentiment analysis for a stock

    - **symbol**: Stock symbol
    - **company_name**: Company name for better news search
    """
    try:
        if not symbol.endswith(('.NS', '.BO')):
            symbol = f"{symbol}.NS"

        sentiment = await sentiment_aggregator.get_stock_sentiment(
            symbol=symbol,
            company_name=company_name,
            use_cache=True
        )

        return SentimentResponse(
            symbol=symbol,
            company_name=company_name,
            timestamp=sentiment.timestamp,
            overall_score=round(sentiment.overall_score, 4),
            overall_label=sentiment.overall_label,
            confidence=round(sentiment.confidence, 4),
            news_score=round(sentiment.news_score, 4),
            news_article_count=sentiment.news_article_count,
            social_score=round(sentiment.social_score, 4),
            social_post_count=sentiment.social_post_count,
            source_breakdown=sentiment.source_breakdown,
            top_bullish_articles=sentiment.top_bullish_articles,
            top_bearish_articles=sentiment.top_bearish_articles
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment error: {str(e)}")


@router.get("/global-factors", response_model=GlobalFactorsResponse)
async def get_global_factors():
    """
    Get current global market factors affecting Indian markets
    """
    try:
        # Get market data
        market_data = await global_markets_client.get_current_data()
        regime = await global_markets_client.get_market_regime()
        impact = await global_markets_client.get_indian_market_impact()

        return GlobalFactorsResponse(
            timestamp=datetime.now(),
            market_regime=regime["regime"],
            regime_confidence=round(regime["confidence"], 4),
            us_markets={
                "sp500": {
                    "price": round(market_data.sp500_price, 2),
                    "change_1d": round(market_data.sp500_change_1d, 4),
                    "change_5d": round(market_data.sp500_change_5d, 4)
                },
                "nasdaq": {
                    "price": round(market_data.nasdaq_price, 2),
                    "change_1d": round(market_data.nasdaq_change_1d, 4)
                },
                "vix": round(market_data.vix_price, 2)
            },
            commodities={
                "gold": {
                    "price": round(market_data.gold_price, 2),
                    "change_1d": round(market_data.gold_change_1d, 4)
                },
                "oil_brent": {
                    "price": round(market_data.oil_price, 2),
                    "change_1d": round(market_data.oil_change_1d, 4)
                }
            },
            forex={
                "usdinr": {
                    "rate": round(market_data.usdinr_rate, 2),
                    "change_1d": round(market_data.usdinr_change_1d, 4)
                },
                "dxy": {
                    "price": round(market_data.dxy_price, 2),
                    "change_1d": round(market_data.dxy_change_1d, 4)
                }
            },
            india_impact=impact["overall_impact"],
            impact_factors=impact.get("impact_factors", [])
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Global factors error: {str(e)}")


@router.get("/{symbol}/backtest", response_model=BacktestResponse)
async def run_backtest(
    symbol: str,
    test_days: int = Query(default=60, ge=30, le=180),
    horizon_days: int = Query(default=20, ge=5, le=60)
):
    """
    Run backtest for a stock prediction model

    - **symbol**: Stock symbol
    - **test_days**: Number of days to test (30-180)
    - **horizon_days**: Prediction horizon (5-60)

    Note: This may take a few minutes to complete.
    """
    try:
        if not symbol.endswith(('.NS', '.BO')):
            symbol = f"{symbol}.NS"

        result = backtester.quick_backtest(
            symbol=symbol,
            test_days=test_days,
            train_days=365,
            horizon_days=horizon_days
        )

        return BacktestResponse(
            symbol=symbol,
            period={
                "start": result.start_date.isoformat(),
                "end": result.end_date.isoformat(),
                "horizon_days": horizon_days
            },
            total_predictions=result.total_predictions,
            correct_predictions=result.correct_predictions,
            directional_accuracy=round(result.directional_accuracy, 4),
            total_return_pct=round(result.total_return * 100, 2),
            annualized_return_pct=round(result.annualized_return * 100, 2),
            sharpe_ratio=round(result.sharpe_ratio, 2),
            max_drawdown_pct=round(result.max_drawdown * 100, 2),
            by_direction={
                "bullish": round(result.bullish_accuracy, 4),
                "bearish": round(result.bearish_accuracy, 4),
                "neutral": round(result.neutral_accuracy, 4)
            },
            by_confidence={
                "high": round(result.high_confidence_accuracy, 4),
                "medium": round(result.medium_confidence_accuracy, 4),
                "low": round(result.low_confidence_accuracy, 4)
            },
            model_accuracies={
                k: round(v, 4) for k, v in result.model_accuracies.items()
            },
            recent_trades=[t.to_dict() for t in result.trades[-10:]]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest error: {str(e)}")


@router.post("/batch", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """
    Get predictions for multiple stocks

    - **symbols**: List of stock symbols (max 10)
    - **horizon_days**: Prediction horizon
    """
    predictions = []
    failed = []

    for symbol in request.symbols:
        try:
            pred = await get_prediction(symbol, request.horizon_days)
            predictions.append(pred)
        except Exception as e:
            failed.append(symbol)

    return BatchPredictionResponse(
        predictions=predictions,
        total_symbols=len(request.symbols),
        successful=len(predictions),
        failed=failed
    )


@router.get("/models/status", response_model=ModelStatusResponse)
async def get_model_status():
    """
    Get status of ML prediction models
    """
    status = ensemble_predictor.get_model_status()

    return ModelStatusResponse(
        is_trained=status["is_trained"],
        trained_models=status["trained_models"],
        total_models=status["total_models"],
        weights=status["weights"],
        last_trained=None  # Would need to track this
    )


@router.get("/market/sentiment")
async def get_market_sentiment():
    """
    Get overall market sentiment
    """
    try:
        sentiment = await sentiment_aggregator.get_market_sentiment()
        return sentiment
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Market sentiment error: {str(e)}")


@router.get("/sector/{sector}/sentiment")
async def get_sector_sentiment(sector: str):
    """
    Get sentiment for a specific sector

    - **sector**: Sector name (banking, it, pharma, auto, energy, fmcg, metal, realty)
    """
    valid_sectors = ["banking", "it", "pharma", "auto", "energy", "fmcg", "metal", "realty"]
    if sector.lower() not in valid_sectors:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sector. Valid options: {', '.join(valid_sectors)}"
        )

    try:
        sentiment = await sentiment_aggregator.get_sector_sentiment(sector)
        return sentiment
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sector sentiment error: {str(e)}")
