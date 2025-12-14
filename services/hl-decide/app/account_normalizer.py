"""
Account State Normalizer

Normalizes account balances and positions across different exchanges
to a common USD denomination for consistent risk calculations.

Key functionality:
- USDT/USD conversion with rate caching
- Normalize Balance objects to USD
- Normalize Position notional values to USD
- Unified exposure calculation across venues

@module account_normalizer
"""

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict

import httpx

from app.exchanges.interface import Balance, Position


# Configuration
USDT_RATE_CACHE_TTL_SECONDS = int(os.getenv("USDT_RATE_CACHE_TTL_SECONDS", "60"))
USDT_RATE_API_TIMEOUT = int(os.getenv("USDT_RATE_API_TIMEOUT", "3"))

# API for USDT/USD rate (CoinGecko public API, no key required)
COINGECKO_API_URL = "https://api.coingecko.com/api/v3/simple/price"

# Fallback rate when API unavailable (USDT typically trades within 0.1% of $1.00)
DEFAULT_USDT_USD_RATE = float(os.getenv("DEFAULT_USDT_USD_RATE", "1.0"))

# Warning threshold for stablecoin depeg (0.5% = 0.005)
DEPEG_WARNING_THRESHOLD = float(os.getenv("DEPEG_WARNING_THRESHOLD", "0.005"))


@dataclass
class CurrencyRate:
    """Cached currency conversion rate."""
    rate: float  # Rate to convert to USD (e.g., USDT -> USD)
    source: str  # 'api', 'fallback', or 'assumed'
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_expired(self) -> bool:
        """Check if rate has expired."""
        age = (datetime.now(timezone.utc) - self.fetched_at).total_seconds()
        return age > USDT_RATE_CACHE_TTL_SECONDS

    @property
    def age_seconds(self) -> float:
        """Get age of rate in seconds."""
        return (datetime.now(timezone.utc) - self.fetched_at).total_seconds()


@dataclass
class NormalizedBalance:
    """
    Account balance normalized to USD.

    Wraps original Balance with USD-normalized values for consistent
    risk calculations across exchanges with different quote currencies.
    """
    original: Balance

    # USD-normalized values
    total_equity_usd: float
    available_balance_usd: float
    margin_used_usd: float
    unrealized_pnl_usd: float
    realized_pnl_today_usd: float

    # Conversion info
    conversion_rate: float  # Rate used for conversion
    conversion_source: str  # 'api', 'fallback', 'identity' (for USD)

    @property
    def margin_ratio(self) -> float:
        """Margin usage ratio (normalized)."""
        if self.total_equity_usd <= 0:
            return 0.0
        return self.margin_used_usd / self.total_equity_usd

    @property
    def is_depeg_warning(self) -> bool:
        """Check if conversion rate indicates potential stablecoin depeg."""
        if self.conversion_source == "identity":
            return False
        return abs(self.conversion_rate - 1.0) > DEPEG_WARNING_THRESHOLD


@dataclass
class NormalizedPosition:
    """
    Position with USD-normalized notional value.

    Useful when aggregating exposure across venues with different
    quote currencies.
    """
    original: Position
    notional_value_usd: float
    conversion_rate: float
    conversion_source: str


class AccountNormalizer:
    """
    Multi-exchange account state normalizer.

    Converts all account balances and position values to USD for
    consistent risk calculations regardless of venue quote currency.

    Supported currencies:
    - USD (no conversion needed)
    - USDT (fetches rate from CoinGecko, fallback to 1.0)

    Usage:
        normalizer = AccountNormalizer()

        # Normalize Bybit balance (USDT)
        bybit_balance = await bybit.get_balance()  # currency="USDT"
        normalized = await normalizer.normalize_balance(bybit_balance)

        # Access USD values
        print(f"Equity: ${normalized.total_equity_usd:.2f}")
    """

    def __init__(self):
        """Initialize account normalizer."""
        self._rate_cache: Dict[str, CurrencyRate] = {}
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=USDT_RATE_API_TIMEOUT,
                headers={"Accept": "application/json"},
            )
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def get_usdt_usd_rate(self, force_refresh: bool = False) -> CurrencyRate:
        """
        Get current USDT/USD conversion rate.

        Args:
            force_refresh: Force API refresh even if cache valid

        Returns:
            CurrencyRate with current rate
        """
        cache_key = "USDT_USD"

        # Return cached if valid
        if not force_refresh and cache_key in self._rate_cache:
            cached = self._rate_cache[cache_key]
            if not cached.is_expired:
                return cached

        # Try to fetch from API
        rate = await self._fetch_usdt_rate()

        if rate is not None:
            cached_rate = CurrencyRate(
                rate=rate,
                source="api",
                fetched_at=datetime.now(timezone.utc),
            )
        else:
            # Fallback to default
            cached_rate = CurrencyRate(
                rate=DEFAULT_USDT_USD_RATE,
                source="fallback",
                fetched_at=datetime.now(timezone.utc),
            )

        self._rate_cache[cache_key] = cached_rate
        return cached_rate

    async def _fetch_usdt_rate(self) -> Optional[float]:
        """
        Fetch USDT/USD rate from CoinGecko API.

        Returns:
            Rate or None if unavailable
        """
        try:
            client = await self._get_client()

            response = await client.get(
                COINGECKO_API_URL,
                params={
                    "ids": "tether",
                    "vs_currencies": "usd",
                },
            )

            if response.status_code != 200:
                return None

            data = response.json()
            rate = data.get("tether", {}).get("usd")

            if rate is not None:
                return float(rate)

        except Exception as e:
            print(f"[account-normalizer] Error fetching USDT rate: {e}")

        return None

    def get_conversion_rate(self, currency: str) -> tuple[float, str]:
        """
        Get conversion rate for currency to USD.

        This is the synchronous version that uses cached rates.
        Call get_usdt_usd_rate() first to populate cache.

        Args:
            currency: Source currency (USD, USDT)

        Returns:
            Tuple of (rate, source) where source is 'identity', 'api', or 'fallback'
        """
        currency_upper = currency.upper()

        # USD is identity
        if currency_upper == "USD":
            return (1.0, "identity")

        # USDT from cache
        if currency_upper == "USDT":
            cached = self._rate_cache.get("USDT_USD")
            if cached:
                return (cached.rate, cached.source)
            # Not in cache, use fallback
            return (DEFAULT_USDT_USD_RATE, "fallback")

        # Unknown currency, assume 1:1
        print(f"[account-normalizer] Unknown currency: {currency}, assuming 1:1 USD")
        return (1.0, "assumed")

    async def normalize_balance(
        self,
        balance: Balance,
        force_refresh_rate: bool = False,
    ) -> NormalizedBalance:
        """
        Normalize balance to USD.

        Args:
            balance: Original balance from exchange
            force_refresh_rate: Force rate refresh

        Returns:
            NormalizedBalance with USD values
        """
        # Pre-fetch USDT rate if needed
        if balance.currency.upper() == "USDT":
            await self.get_usdt_usd_rate(force_refresh=force_refresh_rate)

        rate, source = self.get_conversion_rate(balance.currency)

        return NormalizedBalance(
            original=balance,
            total_equity_usd=balance.total_equity * rate,
            available_balance_usd=balance.available_balance * rate,
            margin_used_usd=balance.margin_used * rate,
            unrealized_pnl_usd=balance.unrealized_pnl * rate,
            realized_pnl_today_usd=balance.realized_pnl_today * rate,
            conversion_rate=rate,
            conversion_source=source,
        )

    def normalize_balance_sync(self, balance: Balance) -> NormalizedBalance:
        """
        Synchronous balance normalization using cached rates.

        For async code, prefer normalize_balance() which can refresh rates.

        Args:
            balance: Original balance from exchange

        Returns:
            NormalizedBalance with USD values
        """
        rate, source = self.get_conversion_rate(balance.currency)

        return NormalizedBalance(
            original=balance,
            total_equity_usd=balance.total_equity * rate,
            available_balance_usd=balance.available_balance * rate,
            margin_used_usd=balance.margin_used * rate,
            unrealized_pnl_usd=balance.unrealized_pnl * rate,
            realized_pnl_today_usd=balance.realized_pnl_today * rate,
            conversion_rate=rate,
            conversion_source=source,
        )

    async def normalize_position(
        self,
        position: Position,
        quote_currency: str = "USD",
        force_refresh_rate: bool = False,
    ) -> NormalizedPosition:
        """
        Normalize position notional value to USD.

        Args:
            position: Original position from exchange
            quote_currency: Quote currency of the position (USD, USDT)
            force_refresh_rate: Force rate refresh

        Returns:
            NormalizedPosition with USD notional
        """
        # Pre-fetch USDT rate if needed
        if quote_currency.upper() == "USDT":
            await self.get_usdt_usd_rate(force_refresh=force_refresh_rate)

        rate, source = self.get_conversion_rate(quote_currency)

        return NormalizedPosition(
            original=position,
            notional_value_usd=position.notional_value * rate,
            conversion_rate=rate,
            conversion_source=source,
        )

    def normalize_position_sync(
        self,
        position: Position,
        quote_currency: str = "USD",
    ) -> NormalizedPosition:
        """
        Synchronous position normalization using cached rates.

        Args:
            position: Original position from exchange
            quote_currency: Quote currency of the position

        Returns:
            NormalizedPosition with USD notional
        """
        rate, source = self.get_conversion_rate(quote_currency)

        return NormalizedPosition(
            original=position,
            notional_value_usd=position.notional_value * rate,
            conversion_rate=rate,
            conversion_source=source,
        )

    def clear_cache(self) -> None:
        """Clear rate cache."""
        self._rate_cache.clear()

    def get_cache_status(self) -> Dict[str, dict]:
        """
        Get cache status for debugging.

        Returns:
            Dict with cache info
        """
        return {
            key: {
                "rate": cached.rate,
                "source": cached.source,
                "age_seconds": cached.age_seconds,
                "is_expired": cached.is_expired,
            }
            for key, cached in self._rate_cache.items()
        }


# Global singleton
_account_normalizer: Optional[AccountNormalizer] = None


def get_account_normalizer() -> AccountNormalizer:
    """Get the global account normalizer singleton."""
    global _account_normalizer
    if _account_normalizer is None:
        _account_normalizer = AccountNormalizer()
    return _account_normalizer


async def init_account_normalizer() -> AccountNormalizer:
    """
    Initialize the global account normalizer and pre-fetch rates.

    Returns:
        Configured AccountNormalizer
    """
    global _account_normalizer
    _account_normalizer = AccountNormalizer()

    # Pre-fetch USDT rate
    rate = await _account_normalizer.get_usdt_usd_rate()
    print(f"[account-normalizer] Initialized with USDT/USD={rate.rate:.4f} (source={rate.source})")

    return _account_normalizer
