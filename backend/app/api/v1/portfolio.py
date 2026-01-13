"""
Portfolio management endpoints
"""
from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from typing import List
from decimal import Decimal
import pandas as pd
from uuid import UUID

from app.models.base import get_db
from app.models.user import User
from app.models.portfolio import Portfolio
from app.models.holding import Holding, AssetType
from app.services.auth import get_current_user
from app.schemas.portfolio import (
    PortfolioCreate,
    PortfolioUpdate,
    PortfolioResponse,
    HoldingCreate,
    HoldingResponse,
    PortfolioAnalysisResponse,
    PortfolioOptimizationResponse
)
from app.integrations.yahoo_finance import yahoo_client
from app.integrations.mfapi import mf_client
from app.ml.portfolio_optimizer import portfolio_optimizer, risk_metrics

router = APIRouter()


@router.get("", response_model=List[PortfolioResponse])
async def list_portfolios(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List all portfolios for the user"""
    result = await db.execute(
        select(Portfolio)
        .where(Portfolio.user_id == current_user.id)
        .options(selectinload(Portfolio.holdings))
    )
    portfolios = result.scalars().all()

    return [PortfolioResponse.model_validate(p) for p in portfolios]


@router.post("", response_model=PortfolioResponse, status_code=status.HTTP_201_CREATED)
async def create_portfolio(
    portfolio_data: PortfolioCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new portfolio"""
    new_portfolio = Portfolio(
        user_id=current_user.id,
        name=portfolio_data.name,
        description=portfolio_data.description
    )

    db.add(new_portfolio)
    await db.commit()
    await db.refresh(new_portfolio, ["holdings"])

    return PortfolioResponse.model_validate(new_portfolio)


@router.get("/{portfolio_id}", response_model=PortfolioResponse)
async def get_portfolio(
    portfolio_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get portfolio details"""
    result = await db.execute(
        select(Portfolio)
        .where(Portfolio.id == portfolio_id, Portfolio.user_id == current_user.id)
        .options(selectinload(Portfolio.holdings))
    )
    portfolio = result.scalar_one_or_none()

    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found"
        )

    return PortfolioResponse.model_validate(portfolio)


@router.post("/{portfolio_id}/holdings", response_model=HoldingResponse, status_code=status.HTTP_201_CREATED)
async def add_holding(
    portfolio_id: UUID,
    holding_data: HoldingCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Add a holding to the portfolio"""
    # Verify portfolio ownership
    result = await db.execute(
        select(Portfolio).where(Portfolio.id == portfolio_id, Portfolio.user_id == current_user.id)
    )
    portfolio = result.scalar_one_or_none()

    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found"
        )

    # Create holding
    new_holding = Holding(
        portfolio_id=portfolio_id,
        asset_type=AssetType(holding_data.asset_type),
        symbol=holding_data.symbol,
        quantity=holding_data.quantity,
        avg_buy_price=holding_data.avg_buy_price,
        buy_date=holding_data.buy_date
    )

    db.add(new_holding)
    await db.commit()
    await db.refresh(new_holding)

    return HoldingResponse.model_validate(new_holding)


@router.get("/{portfolio_id}/analysis", response_model=PortfolioAnalysisResponse)
async def analyze_portfolio(
    portfolio_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get portfolio analysis (risk metrics, sector exposure, etc.)"""
    # Get portfolio with holdings
    result = await db.execute(
        select(Portfolio)
        .where(Portfolio.id == portfolio_id, Portfolio.user_id == current_user.id)
        .options(selectinload(Portfolio.holdings))
    )
    portfolio = result.scalar_one_or_none()

    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found"
        )

    if not portfolio.holdings:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Portfolio has no holdings"
        )

    # Calculate current values
    total_value = Decimal(0)
    total_invested = Decimal(0)
    sector_exposure = {}
    returns_data = []

    for holding in portfolio.holdings:
        invested_value = holding.quantity * holding.avg_buy_price
        total_invested += invested_value

        if holding.asset_type == AssetType.STOCK:
            # Get current stock price
            stock_data = await yahoo_client.get_stock_info(holding.symbol)
            if stock_data:
                current_price = Decimal(str(stock_data["current_price"]))
                current_value = holding.quantity * current_price
                total_value += current_value

                # Sector exposure
                sector = stock_data.get("sector", "Other")
                if sector not in sector_exposure:
                    sector_exposure[sector] = 0
                sector_exposure[sector] += float(current_value)

                # Get historical data for returns calculation
                hist = await yahoo_client.get_historical_data(holding.symbol, period="1y")
                if hist is not None and not hist.empty:
                    returns_data.append({
                        "symbol": holding.symbol,
                        "returns": hist["Close"].pct_change().dropna()
                    })

        elif holding.asset_type == AssetType.MUTUAL_FUND:
            # Get current NAV
            fund_data = await mf_client.get_latest_nav(holding.symbol)
            if fund_data:
                current_nav = Decimal(str(fund_data["nav"]))
                current_value = holding.quantity * current_nav
                total_value += current_value

                # Category for funds (simplified)
                sector_exposure["Mutual Funds"] = sector_exposure.get("Mutual Funds", 0) + float(current_value)

    # Calculate returns
    absolute_returns = total_value - total_invested
    returns_percentage = float((absolute_returns / total_invested) * 100) if total_invested > 0 else 0

    # Convert sector exposure to percentages
    if total_value > 0:
        sector_exposure = {
            sector: (value / float(total_value)) * 100
            for sector, value in sector_exposure.items()
        }

    # Calculate risk metrics if we have returns data
    risk_metrics_dict = None
    if returns_data:
        try:
            # Combine returns for portfolio-level metrics
            # (simplified - equal-weighted for now)
            portfolio_returns = pd.concat([r["returns"] for r in returns_data], axis=1).mean(axis=1)

            risk_metrics_dict = {
                "sharpe_ratio": risk_metrics.calculate_sharpe_ratio(portfolio_returns),
                "sortino_ratio": risk_metrics.calculate_sortino_ratio(portfolio_returns),
                "volatility": risk_metrics.calculate_volatility(portfolio_returns),
                "var_95": risk_metrics.calculate_var(portfolio_returns, 0.95),
                "cvar_95": risk_metrics.calculate_cvar(portfolio_returns, 0.95),
            }
        except Exception as e:
            print(f"Error calculating risk metrics: {e}")

    return PortfolioAnalysisResponse(
        portfolio_id=portfolio_id,
        total_value=total_value,
        total_invested=total_invested,
        absolute_returns=absolute_returns,
        returns_percentage=returns_percentage,
        holdings_count=len(portfolio.holdings),
        sector_exposure=sector_exposure,
        risk_metrics=risk_metrics_dict
    )


@router.get("/{portfolio_id}/optimize", response_model=PortfolioOptimizationResponse)
async def optimize_portfolio(
    portfolio_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Optimize portfolio using HRP algorithm"""
    # Get portfolio with holdings
    result = await db.execute(
        select(Portfolio)
        .where(Portfolio.id == portfolio_id, Portfolio.user_id == current_user.id)
        .options(selectinload(Portfolio.holdings))
    )
    portfolio = result.scalar_one_or_none()

    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found"
        )

    if len(portfolio.holdings) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Portfolio must have at least 2 holdings for optimization"
        )

    # Collect returns data for stocks only (HRP works best with stocks)
    returns_dict = {}
    current_weights = {}
    total_value = Decimal(0)

    for holding in portfolio.holdings:
        if holding.asset_type == AssetType.STOCK:
            # Get historical data
            hist = await yahoo_client.get_historical_data(holding.symbol, period="1y")
            if hist is not None and not hist.empty:
                returns_dict[holding.symbol] = hist["Close"].pct_change().dropna()

                # Calculate current weight
                stock_data = await yahoo_client.get_stock_info(holding.symbol)
                if stock_data:
                    current_price = Decimal(str(stock_data["current_price"]))
                    value = holding.quantity * current_price
                    total_value += value

    # Calculate current weights
    for holding in portfolio.holdings:
        if holding.asset_type == AssetType.STOCK and holding.symbol in returns_dict:
            stock_data = await yahoo_client.get_stock_info(holding.symbol)
            if stock_data:
                current_price = Decimal(str(stock_data["current_price"]))
                value = holding.quantity * current_price
                current_weights[holding.symbol] = float(value / total_value) if total_value > 0 else 0

    if len(returns_dict) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Need at least 2 stocks with sufficient historical data for optimization"
        )

    # Create returns DataFrame
    returns_df = pd.DataFrame(returns_dict).dropna()

    if returns_df.empty or len(returns_df) < 30:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Insufficient historical data for optimization"
        )

    # Optimize using HRP
    try:
        optimized_weights = portfolio_optimizer.optimize_hrp(returns_df)

        return PortfolioOptimizationResponse(
            portfolio_id=portfolio_id,
            current_weights=current_weights,
            optimized_weights=optimized_weights
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimization failed: {str(e)}"
        )
