"""
Stock schemas
"""
from pydantic import BaseModel
from typing import Optional, List
from decimal import Decimal


class StockSearchResult(BaseModel):
    """Schema for stock search result"""
    symbol: str
    name: str


class StockInfo(BaseModel):
    """Schema for basic stock information"""
    symbol: str
    name: str
    sector: str
    industry: str
    market_cap: float
    currency: str
    current_price: float
    previous_close: float
    open: float
    day_high: float
    day_low: float
    volume: int
    fifty_two_week_high: float
    fifty_two_week_low: float


class StockFundamentals(BaseModel):
    """Schema for stock fundamental data"""
    symbol: str
    pe_ratio: float
    forward_pe: float
    pb_ratio: float
    ps_ratio: float
    roe: float
    roa: float
    debt_to_equity: float
    current_ratio: float
    dividend_yield: float
    payout_ratio: float
    earnings_growth: float
    revenue_growth: float
    profit_margins: float
    operating_margins: float


class StockTechnicals(BaseModel):
    """Schema for stock technical indicators"""
    symbol: str
    current_price: float
    sma_20: Optional[float]
    sma_50: Optional[float]
    sma_200: Optional[float]
    ema_12: Optional[float]
    ema_26: Optional[float]
    macd: Optional[float]
    macd_signal: Optional[float]
    macd_histogram: Optional[float]
    rsi: Optional[float]
    bb_upper: Optional[float]
    bb_middle: Optional[float]
    bb_lower: Optional[float]


class StockHistoryPoint(BaseModel):
    """Schema for a single point in stock history"""
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class StockHistoryResponse(BaseModel):
    """Schema for stock historical data response"""
    symbol: str
    period: str
    data: List[StockHistoryPoint]
