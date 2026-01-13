"""
AI Advisor schemas
"""
from pydantic import BaseModel, Field
from typing import Optional


class AIQuestionRequest(BaseModel):
    """Schema for AI question request"""
    question: str = Field(..., min_length=1)
    use_flash: bool = False  # Use Flash model for simple queries


class AIQuestionResponse(BaseModel):
    """Schema for AI question response"""
    question: str
    answer: str
    model_used: str = "gemini-1.5-pro"


class AIStockAnalysisResponse(BaseModel):
    """Schema for AI stock analysis response"""
    symbol: str
    analysis: str


class AIFundAnalysisResponse(BaseModel):
    """Schema for AI fund analysis response"""
    scheme_code: str
    analysis: str


class AIPortfolioAnalysisResponse(BaseModel):
    """Schema for AI portfolio analysis response"""
    portfolio_id: str
    analysis: str
