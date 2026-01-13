"""
Google Gemini AI integration for investment analysis
"""
import google.generativeai as genai
from typing import Optional, Dict, Any
from app.config import settings


class GeminiClient:
    """Client for Google Gemini AI"""

    def __init__(self):
        if settings.GOOGLE_API_KEY:
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.model_pro = genai.GenerativeModel('gemini-1.5-pro')
            self.model_flash = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.model_pro = None
            self.model_flash = None
            print("Warning: GOOGLE_API_KEY not configured")

    def _get_investment_context(self) -> str:
        """Get system context for investment analysis"""
        return """You are an expert investment advisor specializing in Indian stock markets (NSE/BSE) and mutual funds.

Your role:
- Provide accurate, data-driven investment analysis
- Focus on Indian markets (NSE, BSE, Indian mutual funds)
- Cite specific data points when available
- Give balanced views considering both risks and opportunities
- Use INR (Indian Rupees) for all monetary values
- Never guarantee returns or make absolute predictions

Guidelines:
- Be objective and avoid speculation
- Mention data sources when citing specific numbers
- Explain financial concepts in simple terms
- Consider tax implications (Indian tax laws)
- Recommend diversification
- Acknowledge limitations and uncertainties
"""

    async def ask_general_question(
        self,
        question: str,
        use_flash: bool = False
    ) -> Optional[str]:
        """
        Ask a general investment question

        Args:
            question: User's question
            use_flash: Use Flash model for faster, simpler queries (default: Pro model)
        """
        if not self.model_pro and not self.model_flash:
            return "Gemini API is not configured. Please set GOOGLE_API_KEY."

        try:
            model = self.model_flash if use_flash else self.model_pro

            prompt = f"""{self._get_investment_context()}

User Question: {question}

Provide a helpful, accurate response focused on Indian markets. If the question is about a specific stock or fund, mention that detailed data can be fetched separately for better analysis.
"""

            response = model.generate_content(prompt)
            return response.text

        except Exception as e:
            print(f"Error in Gemini API call: {e}")
            return f"Sorry, I encountered an error: {str(e)}"

    async def analyze_stock(
        self,
        symbol: str,
        stock_data: Dict[str, Any],
        fundamentals: Optional[Dict[str, Any]] = None,
        technicals: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Analyze a stock with provided data

        Args:
            symbol: Stock symbol
            stock_data: Basic stock information
            fundamentals: Fundamental data (optional)
            technicals: Technical indicators (optional)
        """
        if not self.model_pro:
            return "Gemini API is not configured."

        try:
            data_summary = f"Stock: {stock_data.get('name', symbol)} ({symbol})\n"
            data_summary += f"Sector: {stock_data.get('sector', 'N/A')}\n"
            data_summary += f"Current Price: ₹{stock_data.get('current_price', 0):.2f}\n"
            data_summary += f"Market Cap: ₹{stock_data.get('market_cap', 0):,.0f}\n"

            if fundamentals:
                data_summary += f"\nFundamentals:\n"
                data_summary += f"- P/E Ratio: {fundamentals.get('pe_ratio', 'N/A')}\n"
                data_summary += f"- P/B Ratio: {fundamentals.get('pb_ratio', 'N/A')}\n"
                data_summary += f"- ROE: {fundamentals.get('roe', 0) * 100:.2f}%\n"
                data_summary += f"- Debt/Equity: {fundamentals.get('debt_to_equity', 'N/A')}\n"
                data_summary += f"- Dividend Yield: {fundamentals.get('dividend_yield', 0) * 100:.2f}%\n"

            if technicals:
                data_summary += f"\nTechnical Indicators:\n"
                data_summary += f"- RSI: {technicals.get('rsi', 'N/A')}\n"
                data_summary += f"- MACD: {technicals.get('macd', 'N/A')}\n"
                data_summary += f"- SMA 50: ₹{technicals.get('sma_50', 'N/A')}\n"
                data_summary += f"- SMA 200: ₹{technicals.get('sma_200', 'N/A')}\n"

            prompt = f"""{self._get_investment_context()}

Analyze this Indian stock based on the following data:

{data_summary}

Provide:
1. Overview of the company and sector
2. Fundamental analysis (valuation, financial health)
3. Technical analysis (price trends, momentum)
4. Key risks and opportunities
5. Investment perspective (not a buy/sell recommendation, but key considerations)

Keep the analysis concise but insightful."""

            response = self.model_pro.generate_content(prompt)
            return response.text

        except Exception as e:
            print(f"Error analyzing stock: {e}")
            return f"Error analyzing stock: {str(e)}"

    async def analyze_mutual_fund(
        self,
        scheme_code: str,
        fund_data: Dict[str, Any],
        returns: Optional[Dict[str, float]] = None
    ) -> Optional[str]:
        """
        Analyze a mutual fund

        Args:
            scheme_code: Fund scheme code
            fund_data: Basic fund information
            returns: Historical returns data (optional)
        """
        if not self.model_pro:
            return "Gemini API is not configured."

        try:
            data_summary = f"Mutual Fund: {fund_data.get('scheme_name', scheme_code)}\n"
            data_summary += f"Scheme Code: {scheme_code}\n"
            data_summary += f"Latest NAV: ₹{fund_data.get('nav', 0):.2f}\n"
            data_summary += f"Date: {fund_data.get('date', 'N/A')}\n"

            if returns:
                data_summary += f"\nReturns:\n"
                for period, return_pct in returns.items():
                    data_summary += f"- {period.replace('_', ' ').title()}: {return_pct:.2f}%\n"

            prompt = f"""{self._get_investment_context()}

Analyze this Indian mutual fund based on the following data:

{data_summary}

Provide:
1. Fund category and investment strategy (infer from name if possible)
2. Performance analysis based on returns
3. Risk considerations
4. Suitability for different investor profiles
5. Key points to consider

Keep the analysis concise but helpful."""

            response = self.model_pro.generate_content(prompt)
            return response.text

        except Exception as e:
            print(f"Error analyzing mutual fund: {e}")
            return f"Error analyzing mutual fund: {str(e)}"

    async def analyze_portfolio(
        self,
        holdings: list[Dict[str, Any]],
        analysis_data: Dict[str, Any]
    ) -> Optional[str]:
        """
        Analyze a portfolio

        Args:
            holdings: List of portfolio holdings
            analysis_data: Portfolio analysis metrics
        """
        if not self.model_pro:
            return "Gemini API is not configured."

        try:
            holdings_summary = "Portfolio Holdings:\n"
            for holding in holdings:
                holdings_summary += f"- {holding.get('symbol', 'N/A')}: {holding.get('quantity', 0)} units\n"

            metrics_summary = "\nPortfolio Metrics:\n"
            if "total_value" in analysis_data:
                metrics_summary += f"- Total Value: ₹{analysis_data['total_value']:,.2f}\n"
            if "sector_exposure" in analysis_data:
                metrics_summary += "- Sector Exposure:\n"
                for sector, percentage in analysis_data["sector_exposure"].items():
                    metrics_summary += f"  - {sector}: {percentage:.1f}%\n"

            prompt = f"""{self._get_investment_context()}

Analyze this investment portfolio:

{holdings_summary}

{metrics_summary}

Provide:
1. Portfolio composition analysis
2. Diversification assessment
3. Risk evaluation
4. Suggestions for optimization (general guidelines)
5. Key considerations for this portfolio

Keep the analysis actionable and focused."""

            response = self.model_pro.generate_content(prompt)
            return response.text

        except Exception as e:
            print(f"Error analyzing portfolio: {e}")
            return f"Error analyzing portfolio: {str(e)}"


# Global client instance
gemini_client = GeminiClient()
