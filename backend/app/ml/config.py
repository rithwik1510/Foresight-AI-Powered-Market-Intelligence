"""
ML Configuration - Model hyperparameters, feature definitions, and settings
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any
from enum import Enum


class PredictionHorizon(Enum):
    """Prediction time horizons"""
    SHORT = 30   # 30 days
    MEDIUM = 60  # 60 days
    LONG = 90    # 90 days


class Direction(Enum):
    """Prediction direction"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class RiskLevel(Enum):
    """Risk level classification"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class ModelConfig:
    """Configuration for individual ML models"""
    enabled: bool = True
    weight: float = 1.0  # Weight in ensemble
    retrain_days: int = 7  # Days between retraining


@dataclass
class ARIMAConfig(ModelConfig):
    """ARIMA model configuration"""
    max_p: int = 5
    max_d: int = 2
    max_q: int = 5
    seasonal: bool = False
    m: int = 1  # Seasonal period


@dataclass
class ProphetConfig(ModelConfig):
    """Prophet model configuration"""
    yearly_seasonality: bool = True
    weekly_seasonality: bool = True
    daily_seasonality: bool = False
    changepoint_prior_scale: float = 0.05
    seasonality_prior_scale: float = 10.0
    interval_width: float = 0.8  # Confidence interval


@dataclass
class XGBoostConfig(ModelConfig):
    """XGBoost classifier configuration"""
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 1
    objective: str = "multi:softprob"
    num_class: int = 3  # Bullish, Bearish, Neutral


@dataclass
class LightGBMConfig(ModelConfig):
    """LightGBM regressor configuration"""
    n_estimators: int = 100
    max_depth: int = -1  # No limit
    learning_rate: float = 0.1
    num_leaves: int = 31
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    objective: str = "regression"
    metric: str = "rmse"


@dataclass
class RandomForestConfig(ModelConfig):
    """Random Forest classifier configuration"""
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    max_features: str = "sqrt"
    class_weight: str = "balanced"


@dataclass
class EnsembleConfig:
    """Ensemble model configuration"""
    # Model weights (will be adjusted based on backtest performance)
    default_weights: Dict[str, float] = field(default_factory=lambda: {
        "arima": 0.10,
        "prophet": 0.25,
        "xgboost": 0.25,
        "lightgbm": 0.25,
        "random_forest": 0.15,
    })

    # Confidence thresholds
    high_confidence_threshold: float = 0.7
    low_confidence_threshold: float = 0.4

    # Agreement bonus (when models agree)
    agreement_bonus: float = 0.2
    min_models_for_agreement: int = 4


@dataclass
class FeatureConfig:
    """Feature engineering configuration"""

    # Technical indicator periods
    sma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])
    ema_periods: List[int] = field(default_factory=lambda: [5, 12, 26, 50])
    rsi_periods: List[int] = field(default_factory=lambda: [7, 14, 21])

    # Returns lookback periods
    return_periods: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 60])

    # Volatility windows
    volatility_windows: List[int] = field(default_factory=lambda: [10, 20, 60])

    # Volume SMA periods
    volume_sma_periods: List[int] = field(default_factory=lambda: [5, 10, 20])

    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0

    # ATR period
    atr_period: int = 14

    # MACD settings
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Stochastic settings
    stoch_k: int = 14
    stoch_d: int = 3

    # Minimum data points required
    min_data_points: int = 252  # 1 year of trading days


@dataclass
class SentimentConfig:
    """Sentiment analysis configuration"""

    # News sources priority
    news_sources: List[str] = field(default_factory=lambda: [
        "google_news_rss",
        "economic_times_rss",
        "moneycontrol_rss",
        "newsapi",
        "finnhub",
    ])

    # Cache TTL (hours)
    news_cache_ttl: int = 6
    sentiment_cache_ttl: int = 1

    # Sentiment thresholds
    positive_threshold: float = 0.05
    negative_threshold: float = -0.05

    # Maximum articles per source
    max_articles_per_source: int = 50

    # Reddit settings
    reddit_subreddits: List[str] = field(default_factory=lambda: [
        "IndiaInvestments",
        "IndianStreetBets",
        "stocks",
    ])
    reddit_post_limit: int = 50


@dataclass
class GlobalMarketsConfig:
    """Global market factors configuration"""

    # Global indices to track
    indices: Dict[str, str] = field(default_factory=lambda: {
        "sp500": "^GSPC",
        "nasdaq": "^IXIC",
        "dow": "^DJI",
        "vix": "^VIX",
        "nifty": "^NSEI",
        "sensex": "^BSESN",
    })

    # Commodities
    commodities: Dict[str, str] = field(default_factory=lambda: {
        "gold": "GC=F",
        "silver": "SI=F",
        "crude_oil": "CL=F",
        "natural_gas": "NG=F",
    })

    # Forex pairs
    forex: Dict[str, str] = field(default_factory=lambda: {
        "usd_inr": "USDINR=X",
        "eur_usd": "EURUSD=X",
        "gbp_usd": "GBPUSD=X",
    })

    # Lag days for correlation
    lag_days: List[int] = field(default_factory=lambda: [1, 2, 3, 5])

    # Correlation window
    correlation_window: int = 60


@dataclass
class BacktestConfig:
    """Backtesting configuration"""

    # Walk-forward validation
    train_window_days: int = 504  # ~2 years
    test_window_days: int = 63    # ~3 months
    step_days: int = 21           # ~1 month

    # Performance thresholds
    min_accuracy: float = 0.5
    min_sharpe: float = 0.0

    # Direction thresholds for classification
    bullish_threshold: float = 0.02   # >2% = bullish
    bearish_threshold: float = -0.02  # <-2% = bearish


@dataclass
class MLSettings:
    """Master ML settings container"""

    # Model configurations
    arima: ARIMAConfig = field(default_factory=ARIMAConfig)
    prophet: ProphetConfig = field(default_factory=ProphetConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    lightgbm: LightGBMConfig = field(default_factory=LightGBMConfig)
    random_forest: RandomForestConfig = field(default_factory=RandomForestConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)

    # Feature settings
    features: FeatureConfig = field(default_factory=FeatureConfig)

    # Sentiment settings
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)

    # Global markets
    global_markets: GlobalMarketsConfig = field(default_factory=GlobalMarketsConfig)

    # Backtest settings
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    # Risk-free rate for India (for Sharpe ratio)
    risk_free_rate: float = 0.06  # 6%

    # Trading days per year
    trading_days_per_year: int = 252


# Global ML settings instance
ml_settings = MLSettings()
