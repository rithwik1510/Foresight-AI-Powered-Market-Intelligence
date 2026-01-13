"""
Corporate events API endpoints
"""
from fastapi import APIRouter, HTTPException, status

from app.integrations.events import events_client
from app.schemas.events import (
    AllEventsResponse,
    EarningsEvent,
    DividendEvent,
    EventFeatures
)

router = APIRouter()


@router.get("/{symbol}", response_model=AllEventsResponse)
async def get_stock_events(symbol: str):
    """
    Get all upcoming events for a stock (earnings + dividends)

    Args:
        symbol: Stock symbol (e.g., RELIANCE, TCS, INFY)

    Returns:
        AllEventsResponse with earnings, dividends, and upcoming events list
    """
    try:
        events_data = await events_client.get_all_events(symbol)
        return AllEventsResponse(**events_data)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching events for {symbol}: {str(e)}"
        )


@router.get("/{symbol}/earnings", response_model=EarningsEvent)
async def get_earnings_date(symbol: str):
    """
    Get next earnings date for a stock

    Args:
        symbol: Stock symbol (e.g., RELIANCE, TCS, INFY)

    Returns:
        EarningsEvent with date and quarter info
    """
    earnings = await events_client.get_earnings_date(symbol)

    if not earnings:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No earnings data found for {symbol}"
        )

    return EarningsEvent(**earnings)


@router.get("/{symbol}/dividends", response_model=DividendEvent)
async def get_dividend_info(symbol: str):
    """
    Get dividend information for a stock

    Args:
        symbol: Stock symbol (e.g., RELIANCE, TCS, INFY)

    Returns:
        DividendEvent with last dividend and estimated next date
    """
    dividend = await events_client.get_dividend_info(symbol)

    if not dividend:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No dividend data found for {symbol}"
        )

    return DividendEvent(**dividend)


@router.get("/{symbol}/features", response_model=EventFeatures)
async def get_event_features(symbol: str):
    """
    Get event-based features for ML prediction models

    Args:
        symbol: Stock symbol (e.g., RELIANCE, TCS, INFY)

    Returns:
        EventFeatures with days_to_earnings, proximity scores, etc.
    """
    try:
        events_data = await events_client.get_all_events(symbol)

        # Extract features
        days_to_earnings = None
        days_since_dividend = None
        has_upcoming_earnings = False
        has_upcoming_dividend = False
        earnings_proximity_score = 0.0

        if events_data.get("earnings"):
            days_to_earnings = events_data["earnings"]["days_until"]
            has_upcoming_earnings = days_to_earnings <= 30

            # Proximity score: 1.0 at earnings day, decreasing to 0.0 at 90 days
            if days_to_earnings is not None:
                earnings_proximity_score = max(0.0, 1.0 - (abs(days_to_earnings) / 90))

        if events_data.get("dividends"):
            days_since_dividend = events_data["dividends"]["days_since_last"]
            days_until_dividend = events_data["dividends"]["days_until_next"]
            has_upcoming_dividend = days_until_dividend <= 60

        return EventFeatures(
            symbol=events_data["symbol"],
            days_to_earnings=days_to_earnings,
            days_since_dividend=days_since_dividend,
            has_upcoming_earnings=has_upcoming_earnings,
            has_upcoming_dividend=has_upcoming_dividend,
            earnings_proximity_score=earnings_proximity_score
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error computing event features for {symbol}: {str(e)}"
        )
