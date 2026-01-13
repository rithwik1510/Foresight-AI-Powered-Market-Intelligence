"""
Stock analysis endpoints
"""
from fastapi import APIRouter, HTTPException, status
from typing import List

from app.integrations.yahoo_finance import yahoo_client
from app.schemas.stock import (
    StockSearchResult,
    StockInfo,
    StockFundamentals,
    StockTechnicals,
    StockHistoryResponse,
    StockHistoryPoint
)

router = APIRouter()


@router.get("/search", response_model=List[StockSearchResult])
async def search_stocks(q: str):
    """Search for stocks by name or symbol"""
    if not q or len(q) < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query parameter 'q' is required and must be at least 1 character"
        )

    results = await yahoo_client.search_stocks(q)
    return results


@router.get("/{symbol}", response_model=StockInfo)
async def get_stock_info(symbol: str):
    """Get basic stock information"""
    stock_data = await yahoo_client.get_stock_info(symbol)

    if not stock_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stock {symbol} not found"
        )

    return StockInfo(**stock_data)


@router.get("/{symbol}/fundamentals", response_model=StockFundamentals)
async def get_stock_fundamentals(symbol: str):
    """Get stock fundamental data (P/E, P/B, ROE, etc.)"""
    fundamentals = await yahoo_client.get_fundamentals(symbol)

    if not fundamentals:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Fundamentals for {symbol} not found"
        )

    return StockFundamentals(**fundamentals)


@router.get("/{symbol}/technicals", response_model=StockTechnicals)
async def get_stock_technicals(symbol: str):
    """Get stock technical indicators (RSI, MACD, etc.)"""
    technicals = await yahoo_client.get_technicals(symbol)

    if not technicals:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Technical indicators for {symbol} not found"
        )

    return StockTechnicals(**technicals)


@router.get("/{symbol}/history", response_model=StockHistoryResponse)
async def get_stock_history(symbol: str, period: str = "1y"):
    """Get historical price data"""
    valid_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]

    if period not in valid_periods:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid period. Must be one of: {', '.join(valid_periods)}"
        )

    hist = await yahoo_client.get_historical_data(symbol, period=period)

    if hist is None or hist.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Historical data for {symbol} not found"
        )

    # Convert DataFrame to list of dicts
    history_data = []
    for date_idx, row in hist.iterrows():
        history_data.append(
            StockHistoryPoint(
                date=date_idx.strftime("%Y-%m-%d"),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=int(row["Volume"])
            )
        )

    return StockHistoryResponse(
        symbol=symbol,
        period=period,
        data=history_data
    )
