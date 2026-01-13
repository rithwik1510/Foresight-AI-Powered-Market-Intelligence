"""
Pydantic schemas for corporate events
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class EarningsEvent(BaseModel):
    """Earnings announcement event"""
    symbol: str
    event_type: str = "earnings"
    date: str = Field(..., description="Earnings date in YYYY-MM-DD format")
    days_until: int = Field(..., description="Days until earnings")
    quarter: int = Field(..., description="Quarter number (1-4)")
    source: str = Field(default="yfinance", description="Data source")


class DividendEvent(BaseModel):
    """Dividend event information"""
    symbol: str
    event_type: str = "dividend"
    last_dividend_date: str
    last_dividend_amount: float
    days_since_last: int
    estimated_next_date: str
    days_until_next: int
    frequency_days: int = Field(..., description="Average days between dividends")
    source: str = Field(default="yfinance", description="Data source")


class UpcomingEvent(BaseModel):
    """Simplified upcoming event"""
    type: str = Field(..., description="Event type: earnings or dividend")
    date: str = Field(..., description="Event date in YYYY-MM-DD format")
    days_until: int = Field(..., description="Days until event")
    description: str = Field(..., description="Human-readable description")


class AllEventsResponse(BaseModel):
    """Complete events response for a stock"""
    symbol: str
    earnings: Optional[EarningsEvent] = None
    dividends: Optional[DividendEvent] = None
    upcoming_events: List[UpcomingEvent] = Field(default_factory=list)


class EventFeatures(BaseModel):
    """Event-based features for ML models"""
    symbol: str
    days_to_earnings: Optional[int] = Field(None, description="Days until next earnings, None if unknown")
    days_since_dividend: Optional[int] = Field(None, description="Days since last dividend")
    has_upcoming_earnings: bool = Field(False, description="Has earnings within 30 days")
    has_upcoming_dividend: bool = Field(False, description="Has dividend within 60 days")
    earnings_proximity_score: float = Field(0.0, description="Score 0-1, higher when closer to earnings")
