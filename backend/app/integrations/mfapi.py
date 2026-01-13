"""
Mutual Fund API (mfapi.in) integration for Indian mutual funds
"""
import httpx
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from functools import lru_cache
import asyncio


class MutualFundClient:
    """Client for fetching mutual fund data from mfapi.in"""

    BASE_URL = "https://api.mfapi.in/mf"

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self._cache = {}
        self._cache_expiry = {}

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

    def _is_cache_valid(self, key: str, ttl_hours: int = 24) -> bool:
        """Check if cached data is still valid"""
        if key not in self._cache or key not in self._cache_expiry:
            return False

        expiry_time = self._cache_expiry[key]
        return datetime.now() < expiry_time

    def _set_cache(self, key: str, value: Any, ttl_hours: int = 24):
        """Set cache with expiry"""
        self._cache[key] = value
        self._cache_expiry[key] = datetime.now() + timedelta(hours=ttl_hours)

    def _get_cache(self, key: str) -> Optional[Any]:
        """Get cached value if valid"""
        if self._is_cache_valid(key):
            return self._cache[key]
        return None

    async def get_all_funds(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get list of all mutual funds
        Cached for 7 days as the list doesn't change frequently
        """
        cache_key = "all_funds"
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        try:
            response = await self.client.get(self.BASE_URL)
            response.raise_for_status()
            data = response.json()

            # Cache for 7 days
            self._set_cache(cache_key, data, ttl_hours=24 * 7)

            return data
        except Exception as e:
            print(f"Error fetching all funds: {e}")
            return None

    async def search_funds(self, query: str) -> List[Dict[str, Any]]:
        """Search for mutual funds by name"""
        all_funds = await self.get_all_funds()

        if not all_funds:
            return []

        query_lower = query.lower()

        # Search in scheme name
        results = [
            fund for fund in all_funds
            if query_lower in fund.get("schemeName", "").lower()
        ]

        return results[:20]  # Limit to 20 results

    async def get_fund_details(self, scheme_code: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed fund information including NAV history
        Cached for 24 hours as NAV updates once per day
        """
        cache_key = f"fund_{scheme_code}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        try:
            response = await self.client.get(f"{self.BASE_URL}/{scheme_code}")
            response.raise_for_status()
            data = response.json()

            # Cache for 24 hours
            self._set_cache(cache_key, data, ttl_hours=24)

            return data
        except Exception as e:
            print(f"Error fetching fund details for {scheme_code}: {e}")
            return None

    async def get_latest_nav(self, scheme_code: str) -> Optional[Dict[str, Any]]:
        """Get the latest NAV for a fund"""
        fund_data = await self.get_fund_details(scheme_code)

        if not fund_data or "data" not in fund_data:
            return None

        # Latest NAV is the first item in the data array
        latest = fund_data["data"][0] if fund_data["data"] else None

        if latest:
            return {
                "scheme_code": scheme_code,
                "scheme_name": fund_data.get("meta", {}).get("scheme_name", ""),
                "nav": float(latest.get("nav", 0)),
                "date": latest.get("date", ""),
            }

        return None

    async def get_nav_history(
        self,
        scheme_code: str,
        days: Optional[int] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get NAV history for a fund

        Args:
            scheme_code: Fund scheme code
            days: Number of days of history (None for all available data)
        """
        fund_data = await self.get_fund_details(scheme_code)

        if not fund_data or "data" not in fund_data:
            return None

        history = fund_data["data"]

        if days:
            history = history[:days]

        # Convert to standardized format
        nav_history = []
        for item in history:
            try:
                nav_history.append({
                    "date": item.get("date", ""),
                    "nav": float(item.get("nav", 0)),
                })
            except (ValueError, TypeError):
                continue

        return nav_history

    async def calculate_returns(
        self,
        scheme_code: str,
        periods: List[int] = [30, 90, 180, 365, 365 * 3, 365 * 5]
    ) -> Optional[Dict[str, float]]:
        """
        Calculate returns for different time periods

        Args:
            scheme_code: Fund scheme code
            periods: List of periods in days [30, 90, 180, 365, ...]

        Returns:
            Dict with period names and return percentages
        """
        nav_history = await self.get_nav_history(scheme_code)

        if not nav_history or len(nav_history) < 2:
            return None

        current_nav = float(nav_history[0]["nav"])
        returns = {}

        period_names = {
            30: "1_month",
            90: "3_months",
            180: "6_months",
            365: "1_year",
            365 * 3: "3_years",
            365 * 5: "5_years",
        }

        for period_days in periods:
            if period_days < len(nav_history):
                past_nav = float(nav_history[period_days]["nav"])
                return_pct = ((current_nav - past_nav) / past_nav) * 100
                period_name = period_names.get(period_days, f"{period_days}_days")
                returns[period_name] = round(return_pct, 2)

        return returns

    async def calculate_cagr(
        self,
        scheme_code: str,
        years: int = 3
    ) -> Optional[float]:
        """
        Calculate CAGR (Compound Annual Growth Rate)

        Args:
            scheme_code: Fund scheme code
            years: Number of years for CAGR calculation
        """
        nav_history = await self.get_nav_history(scheme_code)

        if not nav_history or len(nav_history) < (years * 365):
            return None

        current_nav = float(nav_history[0]["nav"])
        past_nav = float(nav_history[years * 365]["nav"])

        # CAGR = (Ending Value / Beginning Value) ^ (1 / Number of Years) - 1
        cagr = (pow((current_nav / past_nav), (1 / years)) - 1) * 100

        return round(cagr, 2)

    async def calculate_sip_returns(
        self,
        scheme_code: str,
        monthly_investment: float,
        years: int
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate SIP returns

        Args:
            scheme_code: Fund scheme code
            monthly_investment: Monthly investment amount
            years: Investment period in years
        """
        nav_history = await self.get_nav_history(scheme_code, days=years * 365)

        if not nav_history:
            return None

        total_investment = 0
        total_units = 0
        months = years * 12

        # Calculate units purchased each month
        for month_idx in range(months):
            day_idx = month_idx * 30  # Approximate month as 30 days

            if day_idx >= len(nav_history):
                break

            nav = float(nav_history[day_idx]["nav"])
            units = monthly_investment / nav
            total_units += units
            total_investment += monthly_investment

        # Current value
        current_nav = float(nav_history[0]["nav"])
        current_value = total_units * current_nav

        # Returns
        absolute_returns = current_value - total_investment
        returns_percentage = (absolute_returns / total_investment) * 100

        # XIRR approximation (simplified CAGR)
        cagr = (pow((current_value / total_investment), (1 / years)) - 1) * 100

        return {
            "scheme_code": scheme_code,
            "monthly_investment": monthly_investment,
            "period_years": years,
            "total_investment": round(total_investment, 2),
            "total_units": round(total_units, 4),
            "current_value": round(current_value, 2),
            "absolute_returns": round(absolute_returns, 2),
            "returns_percentage": round(returns_percentage, 2),
            "cagr": round(cagr, 2),
        }


# Global client instance
mf_client = MutualFundClient()
