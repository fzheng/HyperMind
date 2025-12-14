"""
Abstract ATR Provider Interface

Defines the contract that all ATR providers must implement for multi-exchange support.

@module atr_provider.interface
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List, Dict, Tuple

# Configuration defaults
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
ATR_MULTIPLIER_BTC = float(os.getenv("ATR_MULTIPLIER_BTC", "2.0"))
ATR_MULTIPLIER_ETH = float(os.getenv("ATR_MULTIPLIER_ETH", "1.5"))
ATR_FALLBACK_PCT = float(os.getenv("ATR_FALLBACK_PCT", "1.0"))
ATR_CACHE_TTL_SECONDS = int(os.getenv("ATR_CACHE_TTL_SECONDS", "60"))
ATR_MAX_STALENESS_SECONDS = int(os.getenv("ATR_MAX_STALENESS_SECONDS", "300"))
ATR_STRICT_MODE = os.getenv("ATR_STRICT_MODE", "true").lower() == "true"

# Asset-specific fallback percentages (last resort)
ATR_FALLBACK_BY_ASSET: Dict[str, float] = {
    "BTC": 0.4,  # ~0.4% typical 1-min ATR for BTC
    "ETH": 0.6,  # ~0.6% typical 1-min ATR for ETH
}

# Asset-specific multipliers
ATR_MULTIPLIERS: Dict[str, float] = {
    "BTC": ATR_MULTIPLIER_BTC,
    "ETH": ATR_MULTIPLIER_ETH,
}


@dataclass
class Candle:
    """OHLC candle data."""
    ts: datetime
    open: float
    high: float
    low: float
    close: float


@dataclass
class ATRData:
    """ATR data for an asset."""
    asset: str
    atr: float  # Raw ATR value in price units
    atr_pct: float  # ATR as percentage of price
    price: float  # Current price
    multiplier: float  # Stop multiplier for this asset
    stop_distance_pct: float  # ATR * multiplier as percentage
    timestamp: datetime
    source: str  # 'db', 'api', 'calculated', 'realized_vol', 'fallback_hardcoded'
    exchange: str = "hyperliquid"  # Source exchange

    @property
    def is_stale(self) -> bool:
        """Check if ATR data exceeds max staleness threshold."""
        if self.source == "fallback_hardcoded":
            return True
        if self.source == "realized_vol":
            return True
        age = (datetime.now(timezone.utc) - self.timestamp).total_seconds()
        return age > ATR_MAX_STALENESS_SECONDS

    @property
    def is_data_driven(self) -> bool:
        """Check if ATR data is from a data-driven source."""
        return self.source in ("db", "api", "calculated", "realized_vol")

    @property
    def age_seconds(self) -> float:
        """Get age of data in seconds."""
        return (datetime.now(timezone.utc) - self.timestamp).total_seconds()


def calculate_true_range(current: Candle, prev_close: Optional[float] = None) -> float:
    """
    Calculate True Range for a single candle.

    True Range = max(High - Low, |High - Prev Close|, |Low - Prev Close|)
    """
    hl = current.high - current.low
    if prev_close is None:
        return hl
    hpc = abs(current.high - prev_close)
    lpc = abs(current.low - prev_close)
    return max(hl, hpc, lpc)


def calculate_atr(candles: List[Candle], period: int = ATR_PERIOD) -> Optional[float]:
    """
    Calculate ATR from a list of candles using Wilder's smoothing.

    Args:
        candles: List of candles, any order (will be sorted)
        period: ATR period (default 14)

    Returns:
        ATR value or None if insufficient data
    """
    if len(candles) < period + 1:
        return None

    # Sort by timestamp ascending
    sorted_candles = sorted(candles, key=lambda c: c.ts)

    # Calculate True Ranges
    true_ranges = []
    for i, candle in enumerate(sorted_candles):
        prev_close = sorted_candles[i - 1].close if i > 0 else None
        tr = calculate_true_range(candle, prev_close)
        true_ranges.append(tr)

    # Initial ATR = simple average of first N TRs
    if len(true_ranges) < period:
        return None

    atr = sum(true_ranges[:period]) / period

    # Wilder's smoothing for remaining TRs
    for tr in true_ranges[period:]:
        atr = ((period - 1) * atr + tr) / period

    return atr


class ATRProviderInterface(ABC):
    """
    Abstract base class for ATR providers.

    Each exchange implementation provides ATR data from its own data source.
    """

    def __init__(self, exchange_name: str):
        """
        Initialize ATR provider.

        Args:
            exchange_name: Name of the exchange (hyperliquid, bybit, aster)
        """
        self.exchange_name = exchange_name.lower()
        self._cache: Dict[str, Tuple[ATRData, datetime]] = {}

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """Check if provider is properly configured."""
        pass

    @abstractmethod
    async def get_atr(self, asset: str, price: Optional[float] = None) -> ATRData:
        """
        Get ATR data for an asset.

        Args:
            asset: Asset symbol (BTC, ETH)
            price: Current price (optional, will fetch if not provided)

        Returns:
            ATRData with ATR values and stop distance
        """
        pass

    @abstractmethod
    async def get_candles(
        self,
        asset: str,
        count: int = ATR_PERIOD + 5,
    ) -> List[Candle]:
        """
        Fetch OHLC candles for ATR calculation.

        Args:
            asset: Asset symbol
            count: Number of candles to fetch

        Returns:
            List of Candle objects, newest first
        """
        pass

    def _is_cache_valid(self, asset: str) -> bool:
        """Check if cached data is still valid."""
        if asset not in self._cache:
            return False
        _, cached_at = self._cache[asset]
        age = (datetime.now(timezone.utc) - cached_at).total_seconds()
        return age < ATR_CACHE_TTL_SECONDS

    def _get_cached(self, asset: str) -> Optional[ATRData]:
        """Get cached ATR data if valid."""
        if self._is_cache_valid(asset):
            return self._cache[asset][0]
        return None

    def _set_cached(self, asset: str, data: ATRData) -> None:
        """Cache ATR data."""
        self._cache[asset] = (data, datetime.now(timezone.utc))

    def _get_multiplier(self, asset: str) -> float:
        """Get ATR multiplier for an asset."""
        return ATR_MULTIPLIERS.get(asset.upper(), ATR_MULTIPLIER_BTC)

    def _fallback_atr(self, asset: str, price: Optional[float]) -> ATRData:
        """
        Return hardcoded fallback ATR when no data available.

        This is the LAST RESORT fallback.
        """
        multiplier = self._get_multiplier(asset)
        atr_pct = ATR_FALLBACK_BY_ASSET.get(asset.upper(), 0.5)
        current_price = price or 100000.0

        print(
            f"[atr:{self.exchange_name}] WARNING: Using HARDCODED fallback ATR for {asset}: "
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
            exchange=self.exchange_name,
        )

    def get_stop_fraction(self, atr_data: ATRData) -> float:
        """Get stop distance as a fraction (0.01 = 1%)."""
        return atr_data.stop_distance_pct / 100.0

    def check_staleness(self, atr_data: ATRData) -> Tuple[bool, str]:
        """Check if ATR data is stale and return detailed status."""
        if atr_data.source == "fallback_hardcoded":
            return (True, f"ATR using hardcoded fallback for {atr_data.asset} on {atr_data.exchange}")

        if atr_data.source == "realized_vol":
            return (True, f"ATR using realized vol for {atr_data.asset} on {atr_data.exchange}")

        age = atr_data.age_seconds
        if age > ATR_MAX_STALENESS_SECONDS:
            return (
                True,
                f"ATR for {atr_data.asset} on {atr_data.exchange} is stale: {age:.0f}s old "
                f"(max {ATR_MAX_STALENESS_SECONDS}s)"
            )

        return (False, f"ATR for {atr_data.asset} on {atr_data.exchange} is fresh: {age:.0f}s old")

    def should_block_gate(self, atr_data: ATRData) -> Tuple[bool, str]:
        """Check if gating should be blocked due to ATR data quality."""
        if not ATR_STRICT_MODE:
            if atr_data.source == "fallback_hardcoded":
                return (False, f"Non-strict mode: allowing hardcoded fallback for {atr_data.asset}")
            return (False, "")

        if atr_data.source == "fallback_hardcoded":
            return (
                True,
                f"Strict mode: blocking gate - no fresh ATR for {atr_data.asset} on {atr_data.exchange}"
            )

        return (False, "")

    def clear_cache(self) -> None:
        """Clear the ATR cache."""
        self._cache.clear()
