"""
Mutual fund analysis endpoints
"""
from fastapi import APIRouter, HTTPException, status
from typing import List

from app.integrations.mfapi import mf_client
from app.schemas.fund import (
    FundSearchResult,
    FundInfo,
    FundNAVHistory,
    NAVHistoryPoint,
    FundReturns,
    SIPCalculatorRequest,
    SIPCalculatorResponse,
    FundOverlapRequest,
    FundOverlapResponse
)

router = APIRouter()


@router.get("/search", response_model=List[FundSearchResult])
async def search_funds(q: str):
    """Search for mutual funds by name"""
    if not q or len(q) < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query parameter 'q' is required and must be at least 1 character"
        )

    results = await mf_client.search_funds(q)
    return results


@router.get("/{scheme_code}", response_model=FundInfo)
async def get_fund_info(scheme_code: str):
    """Get mutual fund information"""
    fund_data = await mf_client.get_latest_nav(scheme_code)

    if not fund_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Fund {scheme_code} not found"
        )

    return FundInfo(**fund_data)


@router.get("/{scheme_code}/nav-history", response_model=FundNAVHistory)
async def get_nav_history(scheme_code: str, days: int = None):
    """Get NAV history for a mutual fund"""
    nav_history = await mf_client.get_nav_history(scheme_code, days=days)

    if not nav_history:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"NAV history for fund {scheme_code} not found"
        )

    # Get fund details for scheme name
    fund_details = await mf_client.get_fund_details(scheme_code)
    scheme_name = fund_details.get("meta", {}).get("scheme_name", scheme_code) if fund_details else scheme_code

    return FundNAVHistory(
        scheme_code=scheme_code,
        scheme_name=scheme_name,
        nav_history=[NAVHistoryPoint(**point) for point in nav_history]
    )


@router.get("/{scheme_code}/returns", response_model=FundReturns)
async def get_fund_returns(scheme_code: str):
    """Get fund returns for different time periods"""
    returns = await mf_client.calculate_returns(scheme_code)

    if not returns:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Returns data for fund {scheme_code} not found"
        )

    return FundReturns(
        scheme_code=scheme_code,
        returns=returns
    )


@router.post("/sip-calculator", response_model=SIPCalculatorResponse)
async def calculate_sip(request: SIPCalculatorRequest):
    """Calculate SIP returns"""
    if request.monthly_investment <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Monthly investment must be greater than 0"
        )

    if request.years <= 0 or request.years > 30:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Years must be between 1 and 30"
        )

    sip_data = await mf_client.calculate_sip_returns(
        request.scheme_code,
        float(request.monthly_investment),
        request.years
    )

    if not sip_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unable to calculate SIP for fund {request.scheme_code}"
        )

    return SIPCalculatorResponse(**sip_data)


@router.post("/overlap", response_model=FundOverlapResponse)
async def calculate_fund_overlap(request: FundOverlapRequest):
    """Calculate portfolio overlap between multiple funds"""
    if len(request.fund_codes) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 2 fund codes are required for overlap calculation"
        )

    # This is a placeholder implementation
    # In a real implementation, you would need access to fund holdings data
    # which is not available from the free mfapi.in API

    return FundOverlapResponse(
        fund_codes=request.fund_codes,
        overlap_percentage=0.0,
        message="Fund overlap calculation requires detailed holdings data which is not available from the free API. Consider integrating with a paid data provider for this feature."
    )
