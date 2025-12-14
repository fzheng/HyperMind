"""
Funding Rate Provider

Provides funding rate data for EV calculations across multiple exchanges.

Funding rates are periodic payments between longs and shorts on perpetual
contracts. They can significantly impact profitability for positions held
over multiple funding intervals (typically 8 hours).

@module funding_provider
"""

import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Tuple, List

import httpx


# Configuration
FUNDING_CACHE_TTL_SECONDS = int(os.getenv("FUNDING_CACHE_TTL_SECONDS", "300"))  # 5 minutes
FUNDING_API_TIMEOUT_SECONDS = int(os.getenv("FUNDING_API_TIMEOUT_SECONDS", "5"))
FUNDING_INTERVAL_HOURS = int(os.getenv("FUNDING_INTERVAL_HOURS", "8"))  # Most exchanges use 8h

# API URLs
BYBIT_API_URL = os.getenv("BYBIT_API_URL", "https://api.bybit.com")
BYBIT_TESTNET_API_URL = os.getenv("BYBIT_TESTNET_API_URL", "https://api-testnet.bybit.com")
HYPERLIQUID_API_URL = os.getenv("HYPERLIQUID_API_URL", "https://api.hyperliquid.xyz")

# Default funding rates per exchange (annualized bps)
# These are used as fallbacks when API unavailable
DEFAULT_FUNDING_RATES = {
    "hyperliquid": {"BTC": 8.0, "ETH": 10.0},  # ~8-10 bps/8h typical
    "aster": {"BTC": 8.0, "ETH": 10.0},
    "bybit": {"BTC": 5.0, "ETH": 7.0},  # Slightly lower typical
}


@dataclass
class FundingData:
    """Funding rate data for an asset."""
    asset: str
    exchange: str
    rate_pct: float  # Current funding rate as percentage
    rate_bps: float  # Current funding rate in basis points
    interval_hours: int  # Funding interval (typically 8h)
    next_funding_time: Optional[datetime] = None
    fetched_at: datetime = None
    source: str = "static"  # 'api', 'static', 'cached'

    def __post_init__(self):
        if self.fetched_at is None:
            self.fetched_at = datetime.now(timezone.utc)

    @property
    def is_expired(self) -> bool:
        """Check if data has expired."""
        age = (datetime.now(timezone.utc) - self.fetched_at).total_seconds()
        return age > FUNDING_CACHE_TTL_SECONDS

    @property
    def daily_cost_bps(self) -> float:
        """Calculate daily funding cost in bps."""
        intervals_per_day = 24 / self.interval_hours
        return self.rate_bps * intervals_per_day

    @property
    def annual_cost_bps(self) -> float:
        """Calculate annualized funding cost in bps."""
        return self.daily_cost_bps * 365

    def cost_for_hold_time(self, hours: float, side: str = "long") -> float:
        """
        Calculate funding cost for a given hold time in bps.

        Funding is signed based on position direction:
        - Positive rate: longs pay shorts
        - Negative rate: shorts pay longs

        For longs: cost = rate * intervals (pay when rate > 0, receive when rate < 0)
        For shorts: cost = -rate * intervals (receive when rate > 0, pay when rate < 0)

        Args:
            hours: Expected hold time in hours
            side: Position side ("long" or "short")

        Returns:
            Funding cost in basis points (positive = cost, negative = rebate)
        """
        intervals = hours / self.interval_hours
        raw_cost = self.rate_bps * intervals

        # For shorts, funding is inverted (they receive when longs pay)
        if side.lower() == "short":
            return -raw_cost
        return raw_cost


@dataclass
class CachedFunding:
    """Cached funding data."""
    data: FundingData
    fetched_at: datetime

    @property
    def is_expired(self) -> bool:
        """Check if cache has expired."""
        age = (datetime.now(timezone.utc) - self.fetched_at).total_seconds()
        return age > FUNDING_CACHE_TTL_SECONDS


class FundingProvider:
    """
    Multi-exchange funding rate provider with caching.

    Fetches funding rates from exchange APIs when available, caches them
    for a short TTL, and falls back to static defaults on failure.
    """

    def __init__(self, testnet: bool = True):
        """
        Initialize funding provider.

        Args:
            testnet: Whether to use testnet APIs
        """
        self.testnet = testnet
        self._cache: Dict[str, CachedFunding] = {}
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=FUNDING_API_TIMEOUT_SECONDS,
                headers={"Content-Type": "application/json"},
            )
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_cache_key(self, exchange: str, asset: str) -> str:
        """Get cache key for exchange/asset pair."""
        return f"{exchange.lower()}:{asset.upper()}"

    def _is_cache_valid(self, exchange: str, asset: str) -> bool:
        """Check if cached funding is still valid."""
        key = self._get_cache_key(exchange, asset)
        if key not in self._cache:
            return False
        return not self._cache[key].is_expired

    async def get_funding(
        self,
        asset: str,
        exchange: str = "hyperliquid",
        force_refresh: bool = False,
    ) -> FundingData:
        """
        Get funding rate for an asset on an exchange.

        Args:
            asset: Asset symbol (BTC, ETH)
            exchange: Exchange name
            force_refresh: Force API refresh even if cache valid

        Returns:
            FundingData with current rate
        """
        key = self._get_cache_key(exchange, asset)
        asset_upper = asset.upper()
        exchange_lower = exchange.lower()

        # Return cached if valid and not forcing refresh
        if not force_refresh and self._is_cache_valid(exchange, asset):
            return self._cache[key].data

        # Try to fetch from API
        data = None

        if exchange_lower == "bybit":
            data = await self._fetch_bybit_funding(asset_upper)
        elif exchange_lower == "hyperliquid":
            data = await self._fetch_hyperliquid_funding(asset_upper)

        # Fall back to static defaults
        if data is None:
            data = self._get_static_funding(asset_upper, exchange_lower)

        # Cache result
        self._cache[key] = CachedFunding(
            data=data,
            fetched_at=datetime.now(timezone.utc),
        )

        return data

    async def get_funding_cost_bps(
        self,
        asset: str,
        exchange: str,
        hold_hours: float = 24.0,
        side: str = "long",
        force_refresh: bool = False,
    ) -> float:
        """
        Get funding cost for expected hold time in basis points.

        Convenience method for EV calculations.

        Args:
            asset: Asset symbol
            exchange: Exchange name
            hold_hours: Expected hold time in hours
            side: Position side ("long" or "short")
            force_refresh: Force API refresh

        Returns:
            Funding cost in bps (positive = cost, negative = rebate)
        """
        data = await self.get_funding(asset, exchange, force_refresh)
        return data.cost_for_hold_time(hold_hours, side)

    async def _fetch_bybit_funding(self, asset: str) -> Optional[FundingData]:
        """
        Fetch current funding rate from Bybit API.

        Uses v5/market/tickers endpoint for funding info.

        Args:
            asset: Asset symbol (BTC, ETH)

        Returns:
            FundingData or None if unavailable
        """
        try:
            client = await self._get_client()
            symbol = f"{asset}USDT"
            base_url = BYBIT_TESTNET_API_URL if self.testnet else BYBIT_API_URL

            response = await client.get(
                f"{base_url}/v5/market/tickers",
                params={
                    "category": "linear",
                    "symbol": symbol,
                },
            )

            if response.status_code != 200:
                return None

            data = response.json()
            if data.get("retCode") != 0:
                return None

            result = data.get("result", {})
            tickers = result.get("list", [])

            if tickers and len(tickers) > 0:
                ticker = tickers[0]
                # fundingRate is in percentage (e.g., 0.0001 = 0.01%)
                rate_str = ticker.get("fundingRate", "0")
                rate_pct = float(rate_str) * 100  # Convert to percentage

                # Next funding time
                next_funding_ts = ticker.get("nextFundingTime")
                next_funding = None
                if next_funding_ts:
                    next_funding = datetime.fromtimestamp(
                        int(next_funding_ts) / 1000,
                        tz=timezone.utc
                    )

                return FundingData(
                    asset=asset,
                    exchange="bybit",
                    rate_pct=rate_pct,
                    rate_bps=rate_pct * 100,  # Convert % to bps
                    interval_hours=FUNDING_INTERVAL_HOURS,
                    next_funding_time=next_funding,
                    source="api",
                )

        except Exception as e:
            print(f"[funding:bybit] Error fetching funding for {asset}: {e}")

        return None

    async def _fetch_hyperliquid_funding(self, asset: str) -> Optional[FundingData]:
        """
        Fetch current funding rate from Hyperliquid API.

        Uses the meta endpoint for funding info.

        Args:
            asset: Asset symbol (BTC, ETH)

        Returns:
            FundingData or None if unavailable
        """
        try:
            client = await self._get_client()

            response = await client.post(
                f"{HYPERLIQUID_API_URL}/info",
                json={"type": "meta"},
            )

            if response.status_code != 200:
                return None

            data = response.json()
            universe = data.get("universe", [])

            # Find the asset
            for item in universe:
                if item.get("name") == asset:
                    # funding is in decimal (e.g., 0.0001 = 0.01%)
                    funding = item.get("funding", 0)
                    rate_pct = float(funding) * 100

                    return FundingData(
                        asset=asset,
                        exchange="hyperliquid",
                        rate_pct=rate_pct,
                        rate_bps=rate_pct * 100,
                        interval_hours=FUNDING_INTERVAL_HOURS,
                        source="api",
                    )

        except Exception as e:
            print(f"[funding:hyperliquid] Error fetching funding for {asset}: {e}")

        return None

    def _get_static_funding(self, asset: str, exchange: str) -> FundingData:
        """
        Get static fallback funding rate.

        Args:
            asset: Asset symbol
            exchange: Exchange name

        Returns:
            FundingData with static defaults
        """
        exchange_rates = DEFAULT_FUNDING_RATES.get(exchange, {})
        rate_bps = exchange_rates.get(asset, 8.0)  # Default 8 bps/8h

        return FundingData(
            asset=asset,
            exchange=exchange,
            rate_pct=rate_bps / 100,
            rate_bps=rate_bps,
            interval_hours=FUNDING_INTERVAL_HOURS,
            source="static",
        )

    def clear_cache(self) -> None:
        """Clear all cached funding data."""
        self._cache.clear()

    def get_cache_status(self) -> Dict[str, dict]:
        """
        Get cache status for all cached funding rates.

        Returns:
            Dict mapping cache key to cache info
        """
        result = {}
        for key, cached in self._cache.items():
            data = cached.data
            result[key] = {
                "source": data.source,
                "rate_bps": data.rate_bps,
                "daily_cost_bps": data.daily_cost_bps,
                "is_expired": cached.is_expired,
            }
        return result


# Global singleton
_funding_provider: Optional[FundingProvider] = None


def get_funding_provider() -> FundingProvider:
    """Get the global funding provider singleton."""
    global _funding_provider
    if _funding_provider is None:
        _funding_provider = FundingProvider()
    return _funding_provider


def init_funding_provider(testnet: bool = True) -> FundingProvider:
    """
    Initialize the global funding provider.

    Args:
        testnet: Whether to use testnet APIs

    Returns:
        Configured FundingProvider
    """
    global _funding_provider
    _funding_provider = FundingProvider(testnet=testnet)
    print(f"[funding-provider] Initialized with testnet={testnet}, cache_ttl={FUNDING_CACHE_TTL_SECONDS}s")
    return _funding_provider
