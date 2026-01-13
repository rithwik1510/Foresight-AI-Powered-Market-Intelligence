"""
Application configuration using Pydantic Settings
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # App
    APP_NAME: str = "Stock Analyzer API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://user:password@localhost:5432/stockanalyzer"

    # Redis
    REDIS_URL: str = "redis://localhost:6379"

    # API Keys
    GOOGLE_API_KEY: Optional[str] = None
    NEWS_API_KEY: Optional[str] = None
    FINNHUB_API_KEY: Optional[str] = None
    FRED_API_KEY: Optional[str] = None

    # Reddit API
    REDDIT_CLIENT_ID: Optional[str] = None
    REDDIT_CLIENT_SECRET: Optional[str] = None
    REDDIT_USER_AGENT: str = "StockAnalyzer/1.0"

    # ML Settings
    ML_MODEL_PATH: str = "./models"
    ML_PREDICTION_CACHE_TTL: int = 3600  # 1 hour
    ML_RETRAIN_SCHEDULE: str = "weekly"  # daily, weekly, monthly

    # JWT Auth
    JWT_SECRET: str = "your-super-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "https://localhost:3000"]

    # Rate Limiting
    YFINANCE_RATE_LIMIT: float = 0.5  # seconds between requests

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


settings = get_settings()
