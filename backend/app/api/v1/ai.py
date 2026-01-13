"""
AI Advisor endpoints (Gemini integration)
"""
from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.integrations.gemini import gemini_client
from app.integrations.yahoo_finance import yahoo_client
from app.integrations.mfapi import mf_client
from app.schemas.ai import (
    AIQuestionRequest,
    AIQuestionResponse,
    AIStockAnalysisResponse,
    AIFundAnalysisResponse
)
from app.models.base import get_db
from app.models.user import User
from app.models.chat_history import ChatHistory
from app.services.auth import get_current_user

router = APIRouter()


@router.post("/ask", response_model=AIQuestionResponse)
async def ask_ai(
    request: AIQuestionRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Ask the AI advisor a question"""
    answer = await gemini_client.ask_general_question(
        request.question,
        use_flash=request.use_flash
    )

    if not answer:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get response from AI"
        )

    # Save to chat history
    chat_entry = ChatHistory(
        user_id=current_user.id,
        question=request.question,
        answer=answer,
        context={"model": "flash" if request.use_flash else "pro"}
    )
    db.add(chat_entry)
    await db.commit()

    model_used = "gemini-1.5-flash" if request.use_flash else "gemini-1.5-pro"

    return AIQuestionResponse(
        question=request.question,
        answer=answer,
        model_used=model_used
    )


@router.post("/analyze-stock/{symbol}", response_model=AIStockAnalysisResponse)
async def analyze_stock_with_ai(
    symbol: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get AI analysis for a stock"""
    # Fetch stock data
    stock_data = await yahoo_client.get_stock_info(symbol)
    if not stock_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stock {symbol} not found"
        )

    # Fetch fundamentals and technicals
    fundamentals = await yahoo_client.get_fundamentals(symbol)
    technicals = await yahoo_client.get_technicals(symbol)

    # Get AI analysis
    analysis = await gemini_client.analyze_stock(
        symbol,
        stock_data,
        fundamentals,
        technicals
    )

    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get AI analysis"
        )

    # Save to chat history
    chat_entry = ChatHistory(
        user_id=current_user.id,
        question=f"Analyze stock {symbol}",
        answer=analysis,
        context={
            "type": "stock_analysis",
            "symbol": symbol,
            "model": "pro"
        }
    )
    db.add(chat_entry)
    await db.commit()

    return AIStockAnalysisResponse(
        symbol=symbol,
        analysis=analysis
    )


@router.post("/analyze-fund/{scheme_code}", response_model=AIFundAnalysisResponse)
async def analyze_fund_with_ai(
    scheme_code: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get AI analysis for a mutual fund"""
    # Fetch fund data
    fund_data = await mf_client.get_latest_nav(scheme_code)
    if not fund_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Fund {scheme_code} not found"
        )

    # Fetch returns data
    returns = await mf_client.calculate_returns(scheme_code)

    # Get AI analysis
    analysis = await gemini_client.analyze_mutual_fund(
        scheme_code,
        fund_data,
        returns
    )

    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get AI analysis"
        )

    # Save to chat history
    chat_entry = ChatHistory(
        user_id=current_user.id,
        question=f"Analyze mutual fund {scheme_code}",
        answer=analysis,
        context={
            "type": "fund_analysis",
            "scheme_code": scheme_code,
            "model": "pro"
        }
    )
    db.add(chat_entry)
    await db.commit()

    return AIFundAnalysisResponse(
        scheme_code=scheme_code,
        analysis=analysis
    )
