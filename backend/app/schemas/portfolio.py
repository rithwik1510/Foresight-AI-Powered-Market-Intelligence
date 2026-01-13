"""
Portfolio schemas
"""
from pydantic import BaseModel, Field
from datetime import datetime, date
from uuid import UUID
from typing import Optional, List
from decimal import Decimal


class PortfolioCreate(BaseModel):
    """Schema for creating a portfolio"""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None


class PortfolioUpdate(BaseModel):
    """Schema for updating a portfolio"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None


class HoldingCreate(BaseModel):
    """Schema for creating a holding"""
    asset_type: str = Field(..., pattern="^(STOCK|MF)$")
    symbol: str = Field(..., min_length=1, max_length=50)
    quantity: Decimal = Field(..., gt=0)
    avg_buy_price: Decimal = Field(..., gt=0)
    buy_date: Optional[date] = None


class HoldingResponse(BaseModel):
    """Schema for holding response"""
    id: UUID
    portfolio_id: UUID
    asset_type: str
    symbol: str
    quantity: Decimal
    avg_buy_price: Decimal
    buy_date: Optional[date]
    created_at: datetime

    class Config:
        from_attributes = True


class PortfolioResponse(BaseModel):
    """Schema for portfolio response"""
    id: UUID
    user_id: UUID
    name: str
    description: Optional[str]
    created_at: datetime
    holdings: List[HoldingResponse] = []

    class Config:
        from_attributes = True


class PortfolioAnalysisResponse(BaseModel):
    """Schema for portfolio analysis response"""
    portfolio_id: UUID
    total_value: Decimal
    total_invested: Decimal
    absolute_returns: Decimal
    returns_percentage: float
    holdings_count: int
    sector_exposure: dict
    risk_metrics: Optional[dict] = None


class PortfolioOptimizationResponse(BaseModel):
    """Schema for portfolio optimization response"""
    portfolio_id: UUID
    current_weights: dict
    optimized_weights: dict
    expected_return: Optional[float] = None
    expected_risk: Optional[float] = None
