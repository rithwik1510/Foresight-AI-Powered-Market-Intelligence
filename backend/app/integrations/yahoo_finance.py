"""
Yahoo Finance integration for Indian stocks (NSE/BSE)
"""
import yfinance as yf
import pandas as pd
from typing import Optional, Dict, Any
import asyncio
from functools import lru_cache
import time

from app.config import settings


class YahooFinanceClient:
    """Client for fetching stock data from Yahoo Finance"""

    def __init__(self):
        self.rate_limit_delay = settings.YFINANCE_RATE_LIMIT
        self.last_request_time = 0

    async def _rate_limit(self):
        """Enforce rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()

    def _ensure_indian_suffix(self, symbol: str) -> str:
        """
        Ensure symbol has proper Indian exchange suffix
        NSE: .NS, BSE: .BO
        Default to NSE if no suffix provided
        Skip for indices (symbols starting with ^)
        """
        symbol = symbol.upper()
        # Don't add suffix to indices (^NSEI, ^BSESN, etc.)
        if symbol.startswith('^'):
            return symbol
        if not symbol.endswith(('.NS', '.BO')):
            symbol += '.NS'
        return symbol

    async def get_stock_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get basic stock information"""
        await self._rate_limit()

        symbol = self._ensure_indian_suffix(symbol)

        try:
            ticker = await asyncio.to_thread(yf.Ticker, symbol)
            info = await asyncio.to_thread(lambda: ticker.info)

            return {
                "symbol": symbol,
                "name": info.get("longName", info.get("shortName", "")),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap", 0),
                "currency": info.get("currency", "INR"),
                "current_price": info.get("currentPrice", info.get("regularMarketPrice", 0)),
                "previous_close": info.get("previousClose", 0),
                "open": info.get("open", 0),
                "day_high": info.get("dayHigh", 0),
                "day_low": info.get("dayLow", 0),
                "volume": info.get("volume", 0),
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh", 0),
                "fifty_two_week_low": info.get("fiftyTwoWeekLow", 0),
            }
        except Exception as e:
            print(f"Error fetching stock info for {symbol}: {e}")
            return None

    async def get_fundamentals(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get fundamental data (P/E, P/B, ROE, etc.)"""
        await self._rate_limit()

        symbol = self._ensure_indian_suffix(symbol)

        try:
            ticker = await asyncio.to_thread(yf.Ticker, symbol)
            info = await asyncio.to_thread(lambda: ticker.info)

            return {
                "symbol": symbol,
                "pe_ratio": info.get("trailingPE", 0),
                "forward_pe": info.get("forwardPE", 0),
                "pb_ratio": info.get("priceToBook", 0),
                "ps_ratio": info.get("priceToSalesTrailing12Months", 0),
                "roe": info.get("returnOnEquity", 0),
                "roa": info.get("returnOnAssets", 0),
                "debt_to_equity": info.get("debtToEquity", 0),
                "current_ratio": info.get("currentRatio", 0),
                "dividend_yield": info.get("dividendYield", 0),
                "payout_ratio": info.get("payoutRatio", 0),
                "earnings_growth": info.get("earningsGrowth", 0),
                "revenue_growth": info.get("revenueGrowth", 0),
                "profit_margins": info.get("profitMargins", 0),
                "operating_margins": info.get("operatingMargins", 0),
            }
        except Exception as e:
            print(f"Error fetching fundamentals for {symbol}: {e}")
            return None

    async def get_historical_data(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Get historical price data

        Args:
            symbol: Stock symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        """
        await self._rate_limit()

        symbol = self._ensure_indian_suffix(symbol)

        try:
            ticker = await asyncio.to_thread(yf.Ticker, symbol)
            hist = await asyncio.to_thread(
                lambda: ticker.history(period=period, interval=interval)
            )

            if hist.empty:
                return None

            return hist
        except Exception as e:
            print(f"Error fetching historical data for {symbol}: {e}")
            return None

    async def get_technicals(self, symbol: str, period: str = "3mo") -> Optional[Dict[str, Any]]:
        """
        Calculate technical indicators (RSI, MACD, Moving Averages)
        """
        hist = await self.get_historical_data(symbol, period=period)

        if hist is None or hist.empty:
            return None

        try:
            # Simple Moving Averages
            hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
            hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
            hist['SMA_200'] = hist['Close'].rolling(window=200).mean()

            # Exponential Moving Averages
            hist['EMA_12'] = hist['Close'].ewm(span=12, adjust=False).mean()
            hist['EMA_26'] = hist['Close'].ewm(span=26, adjust=False).mean()

            # MACD
            hist['MACD'] = hist['EMA_12'] - hist['EMA_26']
            hist['Signal'] = hist['MACD'].ewm(span=9, adjust=False).mean()
            hist['MACD_Histogram'] = hist['MACD'] - hist['Signal']

            # RSI
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            hist['RSI'] = 100 - (100 / (1 + rs))

            # Bollinger Bands
            hist['BB_Middle'] = hist['Close'].rolling(window=20).mean()
            bb_std = hist['Close'].rolling(window=20).std()
            hist['BB_Upper'] = hist['BB_Middle'] + (bb_std * 2)
            hist['BB_Lower'] = hist['BB_Middle'] - (bb_std * 2)

            # Get latest values
            latest = hist.iloc[-1]

            return {
                "symbol": symbol,
                "current_price": float(latest['Close']),
                "sma_20": float(latest['SMA_20']) if pd.notna(latest['SMA_20']) else None,
                "sma_50": float(latest['SMA_50']) if pd.notna(latest['SMA_50']) else None,
                "sma_200": float(latest['SMA_200']) if pd.notna(latest['SMA_200']) else None,
                "ema_12": float(latest['EMA_12']) if pd.notna(latest['EMA_12']) else None,
                "ema_26": float(latest['EMA_26']) if pd.notna(latest['EMA_26']) else None,
                "macd": float(latest['MACD']) if pd.notna(latest['MACD']) else None,
                "macd_signal": float(latest['Signal']) if pd.notna(latest['Signal']) else None,
                "macd_histogram": float(latest['MACD_Histogram']) if pd.notna(latest['MACD_Histogram']) else None,
                "rsi": float(latest['RSI']) if pd.notna(latest['RSI']) else None,
                "bb_upper": float(latest['BB_Upper']) if pd.notna(latest['BB_Upper']) else None,
                "bb_middle": float(latest['BB_Middle']) if pd.notna(latest['BB_Middle']) else None,
                "bb_lower": float(latest['BB_Lower']) if pd.notna(latest['BB_Lower']) else None,
            }
        except Exception as e:
            print(f"Error calculating technicals for {symbol}: {e}")
            return None

    async def search_stocks(self, query: str) -> list[Dict[str, str]]:
        """
        Search for stocks (basic implementation - returns common NSE stocks)
        For production, consider integrating with a proper search API
        """
        # This is a placeholder - yfinance doesn't have a search API
        # In production, you'd want to maintain a database of NSE/BSE stocks
        # or integrate with a proper search service

        query = query.upper()

        # Common NSE stocks for demo
        common_stocks = [
            ("RELIANCE.NS", "Reliance Industries"),
            ("TCS.NS", "Tata Consultancy Services"),
            ("HDFCBANK.NS", "HDFC Bank"),
            ("INFY.NS", "Infosys"),
            ("HINDUNILVR.NS", "Hindustan Unilever"),
            ("ITC.NS", "ITC Limited"),
            ("SBIN.NS", "State Bank of India"),
            ("BHARTIARTL.NS", "Bharti Airtel"),
            ("ICICIBANK.NS", "ICICI Bank"),
            ("KOTAKBANK.NS", "Kotak Mahindra Bank"),
        ]

        results = [
            {"symbol": symbol, "name": name}
            for symbol, name in common_stocks
            if query in symbol or query in name.upper()
        ]

        return results[:10]  # Limit to 10 results


# Global client instance
yahoo_client = YahooFinanceClient()
