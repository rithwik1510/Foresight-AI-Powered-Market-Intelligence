"""
Mutual Fund schemas
"""
from pydantic import BaseModel
from typing import Optional, List
from decimal import Decimal


class FundSearchResult(BaseModel):
    """Schema for fund search result"""
    schemeCode: str
    schemeName: str


class FundInfo(BaseModel):
    """Schema for fund information"""
    scheme_code: str
    scheme_name: str
    nav: float
    date: str


class NAVHistoryPoint(BaseModel):
    """Schema for a single NAV history point"""
    date: str
    nav: float


class FundNAVHistory(BaseModel):
    """Schema for fund NAV history"""
    scheme_code: str
    scheme_name: str
    nav_history: List[NAVHistoryPoint]


class FundReturns(BaseModel):
    """Schema for fund returns"""
    scheme_code: str
    returns: dict  # e.g., {"1_month": 5.2, "3_months": 12.3, ...}


class SIPCalculatorRequest(BaseModel):
    """Schema for SIP calculator request"""
    scheme_code: str
    monthly_investment: Decimal
    years: int


class SIPCalculatorResponse(BaseModel):
    """Schema for SIP calculator response"""
    scheme_code: str
    monthly_investment: float
    period_years: int
    total_investment: float
    total_units: float
    current_value: float
    absolute_returns: float
    returns_percentage: float
    cagr: float


class FundOverlapRequest(BaseModel):
    """Schema for fund overlap calculation request"""
    fund_codes: List[str]


class FundOverlapResponse(BaseModel):
    """Schema for fund overlap response"""
    fund_codes: List[str]
    overlap_percentage: float
    common_holdings: Optional[List[str]] = None
    message: str = "Overlap calculation requires fund holdings data. This is a placeholder."
