"""
Corporate events tracking - Earnings dates and Dividends
Focus: Two biggest price catalysts for swing trading
"""
import asyncio
import time
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import yfinance as yf
from bs4 import BeautifulSoup
import aiohttp
from functools import lru_cache

from app.config import settings


class EventsClient:
    """Client for tracking corporate events (earnings, dividends)"""

    def __init__(self):
        self.rate_limit_delay = 0.5  # 500ms between requests
        self.last_request_time = 0
        self.cache_duration = 3600  # 1 hour cache

    async def _rate_limit(self):
        """Enforce rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()

    def _ensure_indian_suffix(self, symbol: str) -> str:
        """Ensure symbol has proper Indian exchange suffix"""
        symbol = symbol.upper()
        if not symbol.endswith(('.NS', '.BO')):
            symbol += '.NS'
        return symbol

    def _clean_symbol(self, symbol: str) -> str:
        """Remove exchange suffix for external APIs"""
        return symbol.replace('.NS', '').replace('.BO', '').upper()

    async def get_earnings_date(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get next earnings date for a stock
        Uses yfinance as primary source
        """
        await self._rate_limit()

        symbol = self._ensure_indian_suffix(symbol)

        try:
            ticker = await asyncio.to_thread(yf.Ticker, symbol)
            calendar = await asyncio.to_thread(lambda: ticker.calendar)

            if calendar is not None and 'Earnings Date' in calendar.index:
                earnings_dates = calendar.loc['Earnings Date']

                # Handle if multiple dates returned
                if hasattr(earnings_dates, '__iter__') and not isinstance(earnings_dates, str):
                    earnings_date = earnings_dates[0] if len(earnings_dates) > 0 else None
                else:
                    earnings_date = earnings_dates

                if earnings_date and earnings_date != 'N/A':
                    # Convert to datetime if it's a string
                    if isinstance(earnings_date, str):
                        earnings_date = datetime.strptime(earnings_date, '%Y-%m-%d')

                    days_until = (earnings_date - datetime.now()).days

                    return {
                        "symbol": symbol,
                        "event_type": "earnings",
                        "date": earnings_date.strftime('%Y-%m-%d'),
                        "days_until": days_until,
                        "quarter": self._estimate_quarter(earnings_date),
                        "source": "yfinance"
                    }

            return None

        except Exception as e:
            print(f"Error fetching earnings date for {symbol}: {e}")
            return None

    async def get_dividend_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get dividend information for a stock
        Uses yfinance for historical dividends
        """
        await self._rate_limit()

        symbol = self._ensure_indian_suffix(symbol)

        try:
            ticker = await asyncio.to_thread(yf.Ticker, symbol)

            # Get dividend history
            dividends = await asyncio.to_thread(lambda: ticker.dividends)

            if dividends is not None and len(dividends) > 0:
                # Get last dividend
                last_dividend_date = dividends.index[-1]
                last_dividend_amount = dividends.iloc[-1]

                # Calculate days since last dividend
                days_since = (datetime.now() - last_dividend_date.to_pydatetime()).days

                # Estimate next dividend (assume annual/semi-annual pattern)
                avg_frequency = self._estimate_dividend_frequency(dividends)
                estimated_next = last_dividend_date + timedelta(days=avg_frequency)
                days_until_next = (estimated_next.to_pydatetime() - datetime.now()).days

                return {
                    "symbol": symbol,
                    "event_type": "dividend",
                    "last_dividend_date": last_dividend_date.strftime('%Y-%m-%d'),
                    "last_dividend_amount": float(last_dividend_amount),
                    "days_since_last": days_since,
                    "estimated_next_date": estimated_next.strftime('%Y-%m-%d'),
                    "days_until_next": days_until_next,
                    "frequency_days": avg_frequency,
                    "source": "yfinance"
                }

            return None

        except Exception as e:
            print(f"Error fetching dividend info for {symbol}: {e}")
            return None

    async def get_all_events(self, symbol: str) -> Dict[str, Any]:
        """
        Get all upcoming events for a stock
        Combines earnings and dividends
        """
        symbol = self._ensure_indian_suffix(symbol)

        # Fetch both events concurrently
        earnings, dividends = await asyncio.gather(
            self.get_earnings_date(symbol),
            self.get_dividend_info(symbol),
            return_exceptions=True
        )

        events = {
            "symbol": symbol,
            "earnings": earnings if not isinstance(earnings, Exception) else None,
            "dividends": dividends if not isinstance(dividends, Exception) else None,
            "upcoming_events": []
        }

        # Build upcoming events list
        if earnings:
            events["upcoming_events"].append({
                "type": "earnings",
                "date": earnings["date"],
                "days_until": earnings["days_until"],
                "description": f"Q{earnings['quarter']} Results"
            })

        if dividends and dividends.get("days_until_next", 999) < 180:  # Within 6 months
            events["upcoming_events"].append({
                "type": "dividend",
                "date": dividends["estimated_next_date"],
                "days_until": dividends["days_until_next"],
                "description": f"Expected Dividend (₹{dividends['last_dividend_amount']:.2f} last time)"
            })

        # Sort by date
        events["upcoming_events"].sort(key=lambda x: x["days_until"])

        return events

    async def scrape_moneycontrol_earnings(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Scrape earnings date from Moneycontrol
        Fallback if yfinance doesn't have data
        """
        clean_symbol = self._clean_symbol(symbol)

        try:
            # Moneycontrol URL pattern (would need to be mapped)
            # This is a placeholder - actual implementation would need symbol mapping
            url = f"https://www.moneycontrol.com/stocks/company_info/stock_news.php?sc_id={clean_symbol}"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')

                        # Parse earnings date from page
                        # This is a placeholder - would need actual HTML parsing logic

                        return {
                            "symbol": symbol,
                            "event_type": "earnings",
                            "source": "moneycontrol",
                            "note": "Scraping implementation pending"
                        }

            return None

        except Exception as e:
            print(f"Error scraping Moneycontrol for {symbol}: {e}")
            return None

    def _estimate_quarter(self, earnings_date: datetime) -> int:
        """Estimate which quarter based on month"""
        month = earnings_date.month
        if month in [4, 5]:
            return 4  # Q4 (Jan-Mar results)
        elif month in [7, 8]:
            return 1  # Q1 (Apr-Jun results)
        elif month in [10, 11]:
            return 2  # Q2 (Jul-Sep results)
        elif month in [1, 2]:
            return 3  # Q3 (Oct-Dec results)
        else:
            return 0  # Unknown

    def _estimate_dividend_frequency(self, dividends) -> int:
        """
        Estimate dividend frequency in days
        Returns average days between dividends
        """
        if len(dividends) < 2:
            return 365  # Assume annual if only one dividend

        # Calculate gaps between dividends
        dates = dividends.index.to_pydatetime()
        gaps = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]

        if len(gaps) > 0:
            return int(sum(gaps) / len(gaps))

        return 365  # Default to annual


# Singleton instance
events_client = EventsClient()
