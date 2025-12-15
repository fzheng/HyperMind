"""
Account State Normalizer

Normalizes account balances and positions across different exchanges
to a common USD denomination for consistent risk calculations.

Key functionality:
- Treats USDT as equivalent to USD (1:1) - no API calls needed
- Normalize Balance objects
- Normalize Position notional values
- Unified exposure calculation across venues

Note: USDT is a stablecoin pegged 1:1 to USD. Tracking tiny depegs
(e.g., 0.9998) adds complexity without meaningful value for position
sizing or risk calculations.

@module account_normalizer
"""

from dataclasses import dataclass
from typing import Optional, Dict

from .exchanges.interface import Balance, Position


@dataclass
class NormalizedBalance:
    """
    Account balance normalized to USD.

    Wraps original Balance with USD-equivalent values for consistent
    risk calculations across exchanges with different quote currencies.

    Note: USDT is treated as 1:1 with USD (no conversion needed).
    """
    original: Balance

    # USD-equivalent values
    total_equity_usd: float
    available_balance_usd: float
    margin_used_usd: float
    unrealized_pnl_usd: float
    realized_pnl_today_usd: float

    # Conversion info (for API compatibility)
    conversion_rate: float = 1.0  # Always 1.0 (USDT = USD)
    conversion_source: str = "identity"  # Always identity

    @property
    def margin_ratio(self) -> float:
        """Margin usage ratio (normalized)."""
        if self.total_equity_usd <= 0:
            return 0.0
        return self.margin_used_usd / self.total_equity_usd

    @property
    def is_depeg_warning(self) -> bool:
        """Always False - we treat USDT as USD."""
        return False


@dataclass
class NormalizedPosition:
    """
    Position with USD-equivalent notional value.

    Useful when aggregating exposure across venues with different
    quote currencies. USDT is treated as USD (1:1).
    """
    original: Position
    notional_value_usd: float
    conversion_rate: float = 1.0
    conversion_source: str = "identity"


class AccountNormalizer:
    """
    Multi-exchange account state normalizer.

    Treats all stablecoin balances (USD, USDT) as equivalent for
    risk calculations. No external API calls needed.

    Supported currencies:
    - USD (identity)
    - USDT (treated as USD, 1:1)

    Usage:
        normalizer = AccountNormalizer()

        # Normalize Bybit balance (USDT)
        bybit_balance = await bybit.get_balance()  # currency="USDT"
        normalized = normalizer.normalize_balance_sync(bybit_balance)

        # Access USD-equivalent values
        print(f"Equity: ${normalized.total_equity_usd:.2f}")
    """

    def __init__(self):
        """Initialize account normalizer."""
        pass  # No state needed - USDT=USD always

    async def close(self) -> None:
        """Close normalizer (no-op, kept for API compatibility)."""
        pass

    def get_conversion_rate(self, currency: str) -> tuple[float, str]:
        """
        Get conversion rate for currency to USD.

        USDT and USD are both treated as 1:1.

        Args:
            currency: Source currency (USD, USDT)

        Returns:
            Tuple of (rate, source) - always (1.0, "identity")
        """
        currency_upper = currency.upper()

        # USD and USDT are both identity
        if currency_upper in ("USD", "USDT"):
            return (1.0, "identity")

        # Unknown currency, assume 1:1
        print(f"[account-normalizer] Unknown currency: {currency}, assuming 1:1 USD")
        return (1.0, "assumed")

    async def normalize_balance(
        self,
        balance: Balance,
        force_refresh_rate: bool = False,  # Kept for API compatibility
    ) -> NormalizedBalance:
        """
        Normalize balance to USD-equivalent.

        Args:
            balance: Original balance from exchange
            force_refresh_rate: Ignored (no API calls)

        Returns:
            NormalizedBalance with USD-equivalent values
        """
        return self.normalize_balance_sync(balance)

    def normalize_balance_sync(self, balance: Balance) -> NormalizedBalance:
        """
        Synchronous balance normalization.

        USDT is treated as USD (1:1).

        Args:
            balance: Original balance from exchange

        Returns:
            NormalizedBalance with USD-equivalent values
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
        force_refresh_rate: bool = False,  # Kept for API compatibility
    ) -> NormalizedPosition:
        """
        Normalize position notional value to USD-equivalent.

        Args:
            position: Original position from exchange
            quote_currency: Quote currency of the position (USD, USDT)
            force_refresh_rate: Ignored (no API calls)

        Returns:
            NormalizedPosition with USD-equivalent notional
        """
        return self.normalize_position_sync(position, quote_currency)

    def normalize_position_sync(
        self,
        position: Position,
        quote_currency: str = "USD",
    ) -> NormalizedPosition:
        """
        Synchronous position normalization.

        USDT is treated as USD (1:1).

        Args:
            position: Original position from exchange
            quote_currency: Quote currency of the position

        Returns:
            NormalizedPosition with USD-equivalent notional
        """
        rate, source = self.get_conversion_rate(quote_currency)

        return NormalizedPosition(
            original=position,
            notional_value_usd=position.notional_value * rate,
            conversion_rate=rate,
            conversion_source=source,
        )

    def clear_cache(self) -> None:
        """Clear cache (no-op, kept for API compatibility)."""
        pass

    def get_cache_status(self) -> Dict[str, dict]:
        """
        Get cache status for debugging.

        Returns:
            Empty dict (no caching needed)
        """
        return {}


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
    Initialize the global account normalizer.

    Returns:
        Configured AccountNormalizer
    """
    global _account_normalizer
    _account_normalizer = AccountNormalizer()
    print("[account-normalizer] Initialized (USDT=USD, no API calls)")
    return _account_normalizer
