"""
ATR Manager - Multi-Exchange ATR Routing

Routes ATR requests to the appropriate provider based on target exchange.

@module atr_provider.manager
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Tuple

import asyncpg

from .interface import ATRProviderInterface, ATRData
from .hyperliquid import HyperliquidATRProvider
from .bybit import BybitATRProvider


class ATRManager:
    """
    Manages ATR providers for multiple exchanges.

    Routes ATR requests to the appropriate provider based on target exchange.
    Falls back to Hyperliquid if target provider unavailable.
    """

    def __init__(self):
        """Initialize ATR manager with empty providers."""
        self._providers: Dict[str, ATRProviderInterface] = {}
        self._default_exchange = "hyperliquid"
        self._pool: Optional[asyncpg.Pool] = None

    def set_pool(self, pool: asyncpg.Pool) -> None:
        """Set database pool for providers that need it."""
        self._pool = pool
        # Update Hyperliquid provider if registered
        if "hyperliquid" in self._providers:
            hl_provider = self._providers["hyperliquid"]
            if hasattr(hl_provider, "set_pool"):
                hl_provider.set_pool(pool)

    def register_provider(
        self,
        exchange: str,
        provider: ATRProviderInterface,
    ) -> None:
        """
        Register an ATR provider for an exchange.

        Args:
            exchange: Exchange name (hyperliquid, bybit, aster)
            provider: ATRProviderInterface implementation
        """
        self._providers[exchange.lower()] = provider
        print(f"[atr-manager] Registered ATR provider for {exchange}")

    def get_provider(self, exchange: str) -> Optional[ATRProviderInterface]:
        """
        Get ATR provider for an exchange.

        Args:
            exchange: Exchange name

        Returns:
            Provider or None if not registered
        """
        return self._providers.get(exchange.lower())

    def set_default_exchange(self, exchange: str) -> None:
        """Set the default exchange for ATR lookups."""
        self._default_exchange = exchange.lower()

    async def get_atr(
        self,
        asset: str,
        exchange: Optional[str] = None,
        price: Optional[float] = None,
        fallback_to_default: bool = True,
    ) -> ATRData:
        """
        Get ATR data for an asset from the specified exchange.

        Args:
            asset: Asset symbol (BTC, ETH)
            exchange: Target exchange (uses default if None)
            price: Current price (optional)
            fallback_to_default: Whether to fallback to default provider

        Returns:
            ATRData from the target or fallback exchange
        """
        target_exchange = (exchange or self._default_exchange).lower()

        # Try target exchange first
        provider = self._providers.get(target_exchange)
        if provider and provider.is_configured:
            try:
                atr_data = await provider.get_atr(asset, price)
                if atr_data.is_data_driven:
                    return atr_data
                # Provider returned fallback, try default
                print(
                    f"[atr-manager] {target_exchange} returned fallback for {asset}, "
                    f"trying default ({self._default_exchange})"
                )
            except Exception as e:
                print(f"[atr-manager] Error from {target_exchange} for {asset}: {e}")

        # Fallback to default exchange if different
        if fallback_to_default and target_exchange != self._default_exchange:
            default_provider = self._providers.get(self._default_exchange)
            if default_provider and default_provider.is_configured:
                try:
                    atr_data = await default_provider.get_atr(asset, price)
                    # Mark as from fallback exchange
                    print(
                        f"[atr-manager] Using {self._default_exchange} ATR for {asset} "
                        f"(target was {target_exchange})"
                    )
                    return atr_data
                except Exception as e:
                    print(f"[atr-manager] Error from default {self._default_exchange} for {asset}: {e}")

        # No provider available, return hardcoded fallback
        return self._hardcoded_fallback(asset, exchange or self._default_exchange, price)

    async def get_atr_with_staleness_check(
        self,
        asset: str,
        exchange: Optional[str] = None,
        price: Optional[float] = None,
        log_stale: bool = True,
    ) -> Tuple[ATRData, bool]:
        """
        Get ATR data with staleness check.

        Args:
            asset: Asset symbol
            exchange: Target exchange
            price: Current price
            log_stale: Whether to log warnings for stale data

        Returns:
            Tuple of (ATRData, is_stale)
        """
        atr_data = await self.get_atr(asset, exchange, price)

        target_exchange = (exchange or self._default_exchange).lower()
        provider = self._providers.get(atr_data.exchange)

        if provider:
            is_stale, message = provider.check_staleness(atr_data)
        else:
            is_stale = atr_data.is_stale
            message = f"ATR for {asset} on {atr_data.exchange}: stale={is_stale}"

        if is_stale and log_stale:
            print(f"[atr-manager] WARNING: {message}")

        return (atr_data, is_stale)

    def should_block_gate(self, atr_data: ATRData) -> Tuple[bool, str]:
        """
        Check if gating should be blocked due to ATR data quality.

        Args:
            atr_data: ATR data to check

        Returns:
            Tuple of (should_block, reason)
        """
        provider = self._providers.get(atr_data.exchange)
        if provider:
            return provider.should_block_gate(atr_data)

        # No provider, use interface defaults
        from .interface import ATR_STRICT_MODE
        if atr_data.source == "fallback_hardcoded" and ATR_STRICT_MODE:
            return (
                True,
                f"Strict mode: blocking gate - no ATR provider for {atr_data.exchange}"
            )
        return (False, "")

    def get_stop_fraction(self, atr_data: ATRData) -> float:
        """Get stop distance as a fraction (0.01 = 1%)."""
        return atr_data.stop_distance_pct / 100.0

    def _hardcoded_fallback(
        self,
        asset: str,
        exchange: str,
        price: Optional[float],
    ) -> ATRData:
        """Return hardcoded fallback ATR."""
        from .interface import ATR_MULTIPLIERS, ATR_FALLBACK_BY_ASSET, ATR_STRICT_MODE

        multiplier = ATR_MULTIPLIERS.get(asset.upper(), 2.0)
        atr_pct = ATR_FALLBACK_BY_ASSET.get(asset.upper(), 0.5)
        current_price = price or 100000.0

        print(
            f"[atr-manager] WARNING: Using HARDCODED fallback ATR for {asset}: "
            f"{atr_pct:.2f}% (stop={atr_pct * multiplier:.2f}%). "
            f"{'BLOCKING GATE' if ATR_STRICT_MODE else 'Allowing with warning'}."
        )

        return ATRData(
            asset=asset,
            atr=current_price * atr_pct / 100,
            atr_pct=atr_pct,
            price=current_price,
            multiplier=multiplier,
            stop_distance_pct=atr_pct * multiplier,
            timestamp=datetime.now(timezone.utc),
            source="fallback_hardcoded",
            exchange=exchange,
        )

    def clear_all_caches(self) -> None:
        """Clear caches for all providers."""
        for provider in self._providers.values():
            provider.clear_cache()

    async def health_check(self) -> Dict[str, bool]:
        """
        Check health of all registered providers.

        Returns:
            Dict mapping exchange name to health status
        """
        results = {}
        for exchange, provider in self._providers.items():
            try:
                # Try to get ATR for BTC as health check
                atr = await provider.get_atr("BTC")
                results[exchange] = atr.is_data_driven
            except Exception:
                results[exchange] = False
        return results

    @property
    def registered_exchanges(self) -> list:
        """Get list of registered exchanges."""
        return list(self._providers.keys())


# Global singleton
_atr_manager: Optional[ATRManager] = None


def get_atr_manager() -> ATRManager:
    """Get the global ATR manager singleton."""
    global _atr_manager
    if _atr_manager is None:
        _atr_manager = ATRManager()
    return _atr_manager


def init_atr_manager(
    pool: Optional[asyncpg.Pool] = None,
    testnet: bool = True,
) -> ATRManager:
    """
    Initialize the global ATR manager with default providers.

    Args:
        pool: Database pool for Hyperliquid provider
        testnet: Whether to use testnet for Bybit

    Returns:
        Configured ATRManager
    """
    manager = get_atr_manager()

    # Register Hyperliquid provider
    hl_provider = HyperliquidATRProvider(pool)
    manager.register_provider("hyperliquid", hl_provider)

    # Register Bybit provider
    bybit_provider = BybitATRProvider(testnet=testnet)
    manager.register_provider("bybit", bybit_provider)

    # Set pool
    if pool:
        manager.set_pool(pool)

    # Default to Hyperliquid
    manager.set_default_exchange("hyperliquid")

    print(
        f"[atr-manager] Initialized with providers: {manager.registered_exchanges}, "
        f"default={manager._default_exchange}"
    )

    return manager
