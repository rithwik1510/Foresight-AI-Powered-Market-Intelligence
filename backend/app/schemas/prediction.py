"""
Prediction Schemas - Pydantic models for prediction API
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class DirectionEnum(str, Enum):
    """Prediction direction"""
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"


class RiskLevel(str, Enum):
    """Risk level classification"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ModelPrediction(BaseModel):
    """Individual model prediction"""
    direction: DirectionEnum
    predicted_return: float = Field(..., description="Predicted return percentage")
    confidence: float = Field(..., ge=0, le=1)
    weight: float = Field(..., ge=0, le=1)


class PredictionResponse(BaseModel):
    """Main prediction response"""
    symbol: str
    prediction_date: datetime
    horizon_days: int

    # Direction
    direction: DirectionEnum
    direction_probability: float = Field(..., ge=0, le=1)

    # Price predictions
    current_price: float
    predicted_price: float
    predicted_return_pct: float
    price_range: List[float] = Field(..., min_length=2, max_length=2)

    # Confidence & Risk
    confidence: float = Field(..., ge=0, le=1)
    model_agreement: float = Field(..., ge=0, le=1)
    risk_level: RiskLevel

    # Factors
    top_bullish_factors: List[str] = []
    top_bearish_factors: List[str] = []

    # Model breakdown
    model_breakdown: Dict[str, ModelPrediction] = {}

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "RELIANCE.NS",
                "prediction_date": "2024-01-15T10:30:00",
                "horizon_days": 30,
                "direction": "bullish",
                "direction_probability": 0.73,
                "current_price": 2450.50,
                "predicted_price": 2612.00,
                "predicted_return_pct": 6.59,
                "price_range": [2520, 2710],
                "confidence": 0.68,
                "model_agreement": 0.8,
                "risk_level": "MEDIUM",
                "top_bullish_factors": [
                    "Strong momentum (RSI: 62)",
                    "Positive news sentiment"
                ],
                "top_bearish_factors": [
                    "High P/E ratio",
                    "USD/INR weakening"
                ],
                "model_breakdown": {
                    "xgboost": {
                        "direction": "bullish",
                        "predicted_return": 5.2,
                        "confidence": 0.72,
                        "weight": 0.2
                    }
                }
            }
        }


class SentimentResponse(BaseModel):
    """Sentiment analysis response"""
    symbol: str
    company_name: Optional[str] = None
    timestamp: datetime

    overall_score: float = Field(..., ge=-1, le=1)
    overall_label: str
    confidence: float = Field(..., ge=0, le=1)

    news_score: float = Field(..., ge=-1, le=1)
    news_article_count: int
    social_score: float = Field(..., ge=-1, le=1)
    social_post_count: int

    source_breakdown: Dict[str, Dict[str, Any]] = {}
    top_bullish_articles: List[Dict[str, Any]] = []
    top_bearish_articles: List[Dict[str, Any]] = []

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "RELIANCE.NS",
                "company_name": "Reliance Industries",
                "timestamp": "2024-01-15T10:30:00",
                "overall_score": 0.35,
                "overall_label": "bullish",
                "confidence": 0.72,
                "news_score": 0.42,
                "news_article_count": 15,
                "social_score": 0.28,
                "social_post_count": 8,
                "source_breakdown": {},
                "top_bullish_articles": [],
                "top_bearish_articles": []
            }
        }


class GlobalFactorsResponse(BaseModel):
    """Global market factors response"""
    timestamp: datetime

    # Market regime
    market_regime: str
    regime_confidence: float

    # US Markets
    us_markets: Dict[str, Any]

    # Commodities
    commodities: Dict[str, Any]

    # Forex
    forex: Dict[str, Any]

    # India impact
    india_impact: str
    impact_factors: List[Dict[str, Any]] = []

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2024-01-15T10:30:00",
                "market_regime": "RISK_ON",
                "regime_confidence": 0.75,
                "us_markets": {
                    "sp500": {"price": 4800, "change_1d": 0.008},
                    "nasdaq": {"price": 15000, "change_1d": 0.012},
                    "vix": 14.5
                },
                "commodities": {
                    "gold": {"price": 2050, "change_1d": -0.002},
                    "oil": {"price": 75.5, "change_1d": 0.015}
                },
                "forex": {
                    "usdinr": {"rate": 83.2, "change_1d": 0.001}
                },
                "india_impact": "POSITIVE",
                "impact_factors": []
            }
        }


class BacktestRequest(BaseModel):
    """Backtest request parameters"""
    symbol: str
    test_days: int = Field(default=60, ge=30, le=365)
    train_days: int = Field(default=365, ge=180, le=730)
    horizon_days: int = Field(default=20, ge=5, le=90)


class BacktestResponse(BaseModel):
    """Backtest results response"""
    symbol: str
    period: Dict[str, Any]

    # Summary
    total_predictions: int
    correct_predictions: int
    directional_accuracy: float

    # Returns
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float

    # By direction
    by_direction: Dict[str, float]

    # By confidence
    by_confidence: Dict[str, float]

    # Model accuracies
    model_accuracies: Dict[str, float]

    # Recent trades
    recent_trades: List[Dict[str, Any]] = []

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "RELIANCE.NS",
                "period": {
                    "start": "2023-01-01",
                    "end": "2024-01-01",
                    "horizon_days": 20
                },
                "total_predictions": 100,
                "correct_predictions": 65,
                "directional_accuracy": 0.65,
                "total_return_pct": 12.5,
                "annualized_return_pct": 15.2,
                "sharpe_ratio": 1.2,
                "max_drawdown_pct": 8.3,
                "by_direction": {
                    "bullish": 0.68,
                    "bearish": 0.62,
                    "neutral": 0.55
                },
                "by_confidence": {
                    "high": 0.72,
                    "medium": 0.64,
                    "low": 0.52
                },
                "model_accuracies": {
                    "xgboost": 0.67,
                    "lightgbm": 0.65
                }
            }
        }


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    symbols: List[str] = Field(..., min_length=1, max_length=10)
    horizon_days: int = Field(default=30, ge=5, le=90)


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    total_symbols: int
    successful: int
    failed: List[str] = []


class ModelStatusResponse(BaseModel):
    """Model status response"""
    is_trained: bool
    trained_models: List[str]
    total_models: int
    weights: Dict[str, float]
    last_trained: Optional[datetime] = None
