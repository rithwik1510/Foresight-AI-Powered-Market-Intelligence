"""
API v1 Router - Main router that includes all endpoint modules
"""
from fastapi import APIRouter

from app.api.v1 import stocks, funds, portfolio, ai, auth, predictions, events

api_router = APIRouter()

# Include all sub-routers
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(stocks.router, prefix="/stocks", tags=["Stocks"])
api_router.include_router(funds.router, prefix="/funds", tags=["Mutual Funds"])
api_router.include_router(portfolio.router, prefix="/portfolios", tags=["Portfolio"])
api_router.include_router(ai.router, prefix="/ai", tags=["AI Advisor"])
api_router.include_router(predictions.router, tags=["ML Predictions"])
api_router.include_router(events.router, prefix="/events", tags=["Corporate Events"])
