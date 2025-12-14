"""
Dynamic Fee Provider

Provides dynamic fee lookup with caching for accurate EV and Kelly calculations.

Features:
- Short-TTL cache (5 minutes default) to avoid stale fees
- API-based fee tier lookup for supported exchanges
- Falls back to static config if API unavailable
- Per-signal fee refresh for real-time accuracy

@module fee_provider
"""

import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Tuple

import httpx

from .exchanges import ExchangeType, FeeConfig, get_fee_config as get_static_fee_config


# Configuration
FEE_CACHE_TTL_SECONDS = int(os.getenv("FEE_CACHE_TTL_SECONDS", "300"))  # 5 minutes
FEE_API_TIMEOUT_SECONDS = int(os.getenv("FEE_API_TIMEOUT_SECONDS", "5"))
BYBIT_API_URL = os.getenv("BYBIT_API_URL", "https://api.bybit.com")
BYBIT_TESTNET_API_URL = os.getenv("BYBIT_TESTNET_API_URL", "https://api-testnet.bybit.com")


@dataclass
class CachedFees:
    """Cached fee data with timestamp."""
    config: FeeConfig
    fetched_at: datetime
    source: str  # 'api', 'static', 'cached'
    vip_level: Optional[str] = None

    @property
    def is_expired(self) -> bool:
        """Check if cache has expired."""
        age = (datetime.now(timezone.utc) - self.fetched_at).total_seconds()
        return age > FEE_CACHE_TTL_SECONDS

    @property
    def age_seconds(self) -> float:
        """Get cache age in seconds."""
        return (datetime.now(timezone.utc) - self.fetched_at).total_seconds()


class FeeProvider:
    """
    Dynamic fee provider with caching.

    Fetches fees from exchange APIs when available, caches them for
    a short TTL, and falls back to static config on failure.
    """

    def __init__(self, testnet: bool = True):
        """
        Initialize fee provider.

        Args:
            testnet: Whether to use testnet APIs
        """
        self.testnet = testnet
        self._cache: Dict[str, CachedFees] = {}
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=FEE_API_TIMEOUT_SECONDS,
                headers={"Content-Type": "application/json"},
            )
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_cache_key(self, exchange: str) -> str:
        """Get cache key for exchange."""
        return exchange.lower()

    def _is_cache_valid(self, exchange: str) -> bool:
        """Check if cached fees are still valid."""
        key = self._get_cache_key(exchange)
        if key not in self._cache:
            return False
        return not self._cache[key].is_expired

    async def get_fees(
        self,
        exchange: str,
        force_refresh: bool = False,
    ) -> Tuple[FeeConfig, str]:
        """
        Get fee config for an exchange.

        Args:
            exchange: Exchange name (hyperliquid, aster, bybit)
            force_refresh: Force API refresh even if cache valid

        Returns:
            Tuple of (FeeConfig, source) where source is 'api', 'static', or 'cached'
        """
        key = self._get_cache_key(exchange)

        # Return cached if valid and not forcing refresh
        if not force_refresh and self._is_cache_valid(exchange):
            cached = self._cache[key]
            return (cached.config, "cached")

        # Try to fetch from API
        exchange_lower = exchange.lower()
        fees = None
        source = "static"

        if exchange_lower == "bybit":
            fees = await self._fetch_bybit_fees()
            if fees:
                source = "api"

        # Fall back to static config
        if fees is None:
            try:
                exchange_type = ExchangeType(exchange_lower)
                fees = get_static_fee_config(exchange_type)
            except ValueError:
                fees = FeeConfig()  # Default

        # Cache result
        self._cache[key] = CachedFees(
            config=fees,
            fetched_at=datetime.now(timezone.utc),
            source=source,
        )

        return (fees, source)

    async def get_fees_bps(
        self,
        exchange: str,
        force_refresh: bool = False,
    ) -> float:
        """
        Get round-trip taker fees in basis points.

        Convenience method for EV calculations.

        Args:
            exchange: Exchange name
            force_refresh: Force API refresh

        Returns:
            Round-trip taker fees in bps
        """
        config, _ = await self.get_fees(exchange, force_refresh)
        return config.round_trip_cost_bps()

    async def _fetch_bybit_fees(self) -> Optional[FeeConfig]:
        """
        Fetch current fee tier from Bybit API.

        Uses account info endpoint to get actual VIP level fees.
        Note: Requires API key for account-specific fees.
        For now, returns None to use static config (no credentials).

        Returns:
            FeeConfig or None if unavailable
        """
        # TODO: Implement Bybit fee tier lookup when credentials available
        # The v5/account/fee-rate endpoint requires authentication
        # For now, return None to use static config
        return None

    def clear_cache(self) -> None:
        """Clear all cached fees."""
        self._cache.clear()

    def get_cache_status(self) -> Dict[str, dict]:
        """
        Get cache status for all exchanges.

        Returns:
            Dict mapping exchange to cache info
        """
        result = {}
        for key, cached in self._cache.items():
            result[key] = {
                "source": cached.source,
                "age_seconds": cached.age_seconds,
                "is_expired": cached.is_expired,
                "maker_bps": cached.config.maker_fee_bps,
                "taker_bps": cached.config.taker_fee_bps,
                "round_trip_bps": cached.config.round_trip_cost_bps(),
            }
        return result


# Global singleton
_fee_provider: Optional[FeeProvider] = None


def get_fee_provider() -> FeeProvider:
    """Get the global fee provider singleton."""
    global _fee_provider
    if _fee_provider is None:
        _fee_provider = FeeProvider()
    return _fee_provider


def init_fee_provider(testnet: bool = True) -> FeeProvider:
    """
    Initialize the global fee provider.

    Args:
        testnet: Whether to use testnet APIs

    Returns:
        Configured FeeProvider
    """
    global _fee_provider
    _fee_provider = FeeProvider(testnet=testnet)
    print(f"[fee-provider] Initialized with testnet={testnet}, cache_ttl={FEE_CACHE_TTL_SECONDS}s")
    return _fee_provider
