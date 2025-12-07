"""
ATR (Average True Range) Provider for Dynamic Stop Distances

Provides ATR-based volatility measures for position sizing and stop calculation.
ATR is the gold standard for volatility-adjusted stops because it accounts for
gaps and intraday ranges, not just close-to-close moves.

Key concepts:
- True Range = max(High - Low, |High - Prev Close|, |Low - Prev Close|)
- ATR = Smoothed average of True Range over N periods (default 14)
- Stop distance = ATR multiplier × ATR (e.g., 2× ATR for BTC)

Data sources:
- marks_1m table: 1-minute candles with high, low, close, and pre-computed atr14
- If atr14 not available, calculate from raw candles

Configuration:
- ATR_MULTIPLIER_BTC: Stop distance in ATR units for BTC (default 2.0)
- ATR_MULTIPLIER_ETH: Stop distance in ATR units for ETH (default 1.5)
- ATR_FALLBACK_PCT: Fallback percentage if no ATR data (default 1.0%)

@module atr
"""

import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Tuple
from functools import lru_cache

import asyncpg

# Configuration
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
ATR_MULTIPLIER_BTC = float(os.getenv("ATR_MULTIPLIER_BTC", "2.0"))
ATR_MULTIPLIER_ETH = float(os.getenv("ATR_MULTIPLIER_ETH", "1.5"))
ATR_FALLBACK_PCT = float(os.getenv("ATR_FALLBACK_PCT", "1.0"))  # 1%
ATR_CACHE_TTL_SECONDS = int(os.getenv("ATR_CACHE_TTL_SECONDS", "60"))  # 1 minute
ATR_MAX_STALENESS_SECONDS = int(os.getenv("ATR_MAX_STALENESS_SECONDS", "300"))  # 5 minutes

# Strict mode: block gating when ATR is stale/missing (default: true)
# Set to "false" for non-prod environments to use warn-and-pass
ATR_STRICT_MODE = os.getenv("ATR_STRICT_MODE", "true").lower() == "true"

# Realized volatility fallback configuration
# Use rolling 24h realized vol from marks_1m as data-driven fallback
ATR_REALIZED_VOL_WINDOW_HOURS = int(os.getenv("ATR_REALIZED_VOL_WINDOW_HOURS", "24"))
ATR_REALIZED_VOL_MIN_SAMPLES = int(os.getenv("ATR_REALIZED_VOL_MIN_SAMPLES", "60"))  # At least 60 candles

# Asset-specific fallback percentages (used ONLY when realized vol unavailable)
# These are absolute last resort - prefer data-driven fallbacks
ATR_FALLBACK_BY_ASSET: Dict[str, float] = {
    "BTC": 0.4,   # ~0.4% typical 1-min ATR for BTC
    "ETH": 0.6,   # ~0.6% typical 1-min ATR for ETH (more volatile)
}

# Asset-specific multipliers
ATR_MULTIPLIERS: Dict[str, float] = {
    "BTC": ATR_MULTIPLIER_BTC,
    "ETH": ATR_MULTIPLIER_ETH,
}


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
    source: str  # 'db', 'calculated', 'realized_vol', 'fallback_hardcoded'

    @property
    def is_stale(self) -> bool:
        """Check if ATR data exceeds max staleness threshold."""
        if self.source == "fallback_hardcoded":
            return True  # Hardcoded fallback is always considered stale
        if self.source == "realized_vol":
            # Realized vol is data-driven but still considered "stale" for strict mode
            return True
        age = (datetime.now(timezone.utc) - self.timestamp).total_seconds()
        return age > ATR_MAX_STALENESS_SECONDS

    @property
    def is_data_driven(self) -> bool:
        """Check if ATR data is from a data-driven source (db, calculated, or realized_vol)."""
        return self.source in ("db", "calculated", "realized_vol")

    @property
    def age_seconds(self) -> float:
        """Get age of data in seconds."""
        return (datetime.now(timezone.utc) - self.timestamp).total_seconds()


@dataclass
class Candle:
    """OHLC candle data."""
    ts: datetime
    open: float
    high: float
    low: float
    close: float


def calculate_true_range(current: Candle, prev_close: Optional[float] = None) -> float:
    """
    Calculate True Range for a single candle.

    True Range = max(
        High - Low,
        |High - Prev Close|,
        |Low - Prev Close|
    )

    If no previous close (first candle), TR = High - Low.
    """
    hl = current.high - current.low

    if prev_close is None:
        return hl

    hpc = abs(current.high - prev_close)
    lpc = abs(current.low - prev_close)

    return max(hl, hpc, lpc)


def calculate_atr(candles: List[Candle], period: int = ATR_PERIOD) -> Optional[float]:
    """
    Calculate ATR from a list of candles.

    Uses Wilder's smoothing (exponential moving average):
    ATR_t = ((period - 1) × ATR_{t-1} + TR_t) / period

    Args:
        candles: List of candles, newest first
        period: ATR period (default 14)

    Returns:
        ATR value or None if insufficient data
    """
    if len(candles) < period + 1:
        return None

    # Sort by timestamp ascending for calculation
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


class ATRProvider:
    """
    Provides ATR data for assets.

    Fetches from database with caching to avoid excessive queries.
    Falls back to hardcoded percentage if no data available.
    """

    def __init__(self, pool: Optional[asyncpg.Pool] = None):
        self.pool = pool
        self._cache: Dict[str, Tuple[ATRData, datetime]] = {}

    def set_pool(self, pool: asyncpg.Pool) -> None:
        """Set the database pool (for late initialization)."""
        self.pool = pool

    def _is_cache_valid(self, asset: str) -> bool:
        """Check if cached data is still valid."""
        if asset not in self._cache:
            return False
        _, cached_at = self._cache[asset]
        age = (datetime.now(timezone.utc) - cached_at).total_seconds()
        return age < ATR_CACHE_TTL_SECONDS

    def _get_multiplier(self, asset: str) -> float:
        """Get ATR multiplier for an asset."""
        return ATR_MULTIPLIERS.get(asset.upper(), ATR_MULTIPLIER_BTC)

    async def get_atr(self, asset: str, price: Optional[float] = None) -> ATRData:
        """
        Get ATR data for an asset.

        Args:
            asset: Asset symbol (BTC, ETH)
            price: Current price (optional, will fetch if not provided)

        Returns:
            ATRData with ATR values and stop distance
        """
        asset_upper = asset.upper()

        # Check cache first
        if self._is_cache_valid(asset_upper):
            cached_data, _ = self._cache[asset_upper]
            # Update price if provided
            if price is not None and price != cached_data.price:
                cached_data = ATRData(
                    asset=asset_upper,
                    atr=cached_data.atr,
                    atr_pct=cached_data.atr / price * 100 if price > 0 else cached_data.atr_pct,
                    price=price,
                    multiplier=cached_data.multiplier,
                    stop_distance_pct=cached_data.atr / price * 100 * cached_data.multiplier if price > 0 else cached_data.stop_distance_pct,
                    timestamp=cached_data.timestamp,
                    source=cached_data.source,
                )
            return cached_data

        # Try to fetch from database (fresh ATR)
        atr_data = await self._fetch_atr_from_db(asset_upper, price)

        if atr_data:
            self._cache[asset_upper] = (atr_data, datetime.now(timezone.utc))
            return atr_data

        # Try data-driven fallback: rolling 24h realized volatility
        current_price = price or 100000.0
        realized_vol_data = await self._compute_realized_vol(asset_upper, current_price)
        if realized_vol_data:
            self._cache[asset_upper] = (realized_vol_data, datetime.now(timezone.utc))
            return realized_vol_data

        # Last resort: hardcoded fallback (will fail gate in strict mode)
        return self._fallback_atr(asset_upper, price)

    async def _fetch_atr_from_db(self, asset: str, price: Optional[float]) -> Optional[ATRData]:
        """Fetch ATR from marks_1m table."""
        if self.pool is None:
            return None

        try:
            async with self.pool.acquire() as conn:
                # First try to get pre-computed atr14
                row = await conn.fetchrow(
                    """
                    SELECT atr14, mid, ts
                    FROM marks_1m
                    WHERE asset = $1 AND atr14 IS NOT NULL
                    ORDER BY ts DESC
                    LIMIT 1
                    """,
                    asset,
                )

                if row and row["atr14"] is not None:
                    atr = float(row["atr14"])
                    mid = float(row["mid"])
                    current_price = price or mid
                    multiplier = self._get_multiplier(asset)

                    return ATRData(
                        asset=asset,
                        atr=atr,
                        atr_pct=atr / current_price * 100 if current_price > 0 else 0,
                        price=current_price,
                        multiplier=multiplier,
                        stop_distance_pct=atr / current_price * 100 * multiplier if current_price > 0 else 0,
                        timestamp=row["ts"],
                        source="db",
                    )

                # If no atr14, try to calculate from candles
                rows = await conn.fetch(
                    """
                    SELECT ts, mid as open, high, low, close
                    FROM marks_1m
                    WHERE asset = $1
                      AND high IS NOT NULL
                      AND low IS NOT NULL
                      AND close IS NOT NULL
                    ORDER BY ts DESC
                    LIMIT $2
                    """,
                    asset,
                    ATR_PERIOD + 5,  # Get a few extra for safety
                )

                if len(rows) >= ATR_PERIOD + 1:
                    candles = [
                        Candle(
                            ts=row["ts"],
                            open=float(row["open"]),
                            high=float(row["high"]),
                            low=float(row["low"]),
                            close=float(row["close"]),
                        )
                        for row in rows
                    ]

                    atr = calculate_atr(candles, ATR_PERIOD)
                    if atr is not None and atr > 0:
                        current_price = price or float(rows[0]["close"])
                        multiplier = self._get_multiplier(asset)

                        return ATRData(
                            asset=asset,
                            atr=atr,
                            atr_pct=atr / current_price * 100 if current_price > 0 else 0,
                            price=current_price,
                            multiplier=multiplier,
                            stop_distance_pct=atr / current_price * 100 * multiplier if current_price > 0 else 0,
                            timestamp=rows[0]["ts"],
                            source="calculated",
                        )

        except Exception as e:
            print(f"[atr] Failed to fetch ATR from DB for {asset}: {e}")

        return None

    async def _compute_realized_vol(self, asset: str, price: float) -> Optional[ATRData]:
        """
        Compute realized volatility from rolling 24h marks_1m data.

        Uses standard deviation of log returns × sqrt(samples) to approximate ATR.
        This is a data-driven fallback when fresh ATR is unavailable.

        Args:
            asset: Asset symbol (BTC, ETH)
            price: Current price for percentage calculation

        Returns:
            ATRData with source='realized_vol' or None if insufficient data
        """
        if self.pool is None:
            return None

        try:
            async with self.pool.acquire() as conn:
                # Get closing prices from the last N hours
                cutoff = datetime.now(timezone.utc) - timedelta(hours=ATR_REALIZED_VOL_WINDOW_HOURS)
                rows = await conn.fetch(
                    """
                    SELECT close, ts
                    FROM marks_1m
                    WHERE asset = $1
                      AND ts >= $2
                      AND close IS NOT NULL
                    ORDER BY ts ASC
                    """,
                    asset,
                    cutoff,
                )

                if len(rows) < ATR_REALIZED_VOL_MIN_SAMPLES:
                    return None

                # Calculate log returns
                closes = [float(row["close"]) for row in rows]
                log_returns = []
                for i in range(1, len(closes)):
                    if closes[i - 1] > 0 and closes[i] > 0:
                        log_returns.append(abs(math.log(closes[i] / closes[i - 1])))

                if len(log_returns) < ATR_REALIZED_VOL_MIN_SAMPLES - 1:
                    return None

                # Compute realized volatility as mean absolute log return
                # This approximates ATR% for 1-min candles
                mean_abs_return = sum(log_returns) / len(log_returns)
                realized_vol_pct = mean_abs_return * 100  # Convert to percentage

                multiplier = self._get_multiplier(asset)
                latest_ts = rows[-1]["ts"]

                print(
                    f"[atr] Using 24h realized vol for {asset}: "
                    f"{realized_vol_pct:.3f}% (from {len(log_returns)} samples)"
                )

                return ATRData(
                    asset=asset,
                    atr=price * realized_vol_pct / 100,
                    atr_pct=realized_vol_pct,
                    price=price,
                    multiplier=multiplier,
                    stop_distance_pct=realized_vol_pct * multiplier,
                    timestamp=latest_ts,
                    source="realized_vol",
                )

        except Exception as e:
            print(f"[atr] Failed to compute realized vol for {asset}: {e}")
            return None

    def _fallback_atr(self, asset: str, price: Optional[float]) -> ATRData:
        """
        Return hardcoded fallback ATR when no data available.

        Uses asset-specific typical ATR percentages based on historical analysis:
        - BTC: ~0.4% typical 1-min ATR (lower volatility per candle)
        - ETH: ~0.6% typical 1-min ATR (higher volatility)

        This is the LAST RESORT fallback. In strict mode, this will cause
        gating to fail. In non-strict mode, logs a warning.
        """
        multiplier = self._get_multiplier(asset)
        # Use asset-specific fallback or default
        atr_pct = ATR_FALLBACK_BY_ASSET.get(asset.upper(), 0.5)
        current_price = price or 100000.0  # Placeholder for BTC

        # Log warning about hardcoded fallback usage
        print(
            f"[atr] WARNING: Using HARDCODED fallback ATR for {asset}: "
            f"{atr_pct:.2f}% (stop={atr_pct * multiplier:.2f}%). "
            f"No fresh ATR or realized vol data available. "
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
        )

    def get_stop_fraction(self, atr_data: ATRData) -> float:
        """
        Get stop distance as a fraction (0.01 = 1%).

        This is the value to use in EpisodeBuilderConfig.default_stop_fraction
        and consensus detection.
        """
        return atr_data.stop_distance_pct / 100.0

    def check_staleness(self, atr_data: ATRData) -> Tuple[bool, str]:
        """
        Check if ATR data is stale and return detailed status.

        Args:
            atr_data: ATR data to check

        Returns:
            Tuple of (is_stale, status_message)
        """
        if atr_data.source == "fallback_hardcoded":
            return (True, f"ATR using hardcoded fallback for {atr_data.asset}")

        if atr_data.source == "realized_vol":
            return (True, f"ATR using 24h realized vol for {atr_data.asset} (data-driven fallback)")

        age = atr_data.age_seconds
        if age > ATR_MAX_STALENESS_SECONDS:
            return (
                True,
                f"ATR for {atr_data.asset} is stale: {age:.0f}s old "
                f"(max {ATR_MAX_STALENESS_SECONDS}s)"
            )

        return (False, f"ATR for {atr_data.asset} is fresh: {age:.0f}s old")

    def should_block_gate(self, atr_data: ATRData) -> Tuple[bool, str]:
        """
        Check if gating should be blocked due to ATR data quality.

        In strict mode (default), blocks when:
        - Using hardcoded fallback (no data at all)

        In non-strict mode, always allows but logs warnings.

        Args:
            atr_data: ATR data to check

        Returns:
            Tuple of (should_block, reason)
        """
        if not ATR_STRICT_MODE:
            # Non-strict mode: warn but don't block
            if atr_data.source == "fallback_hardcoded":
                return (False, f"Non-strict mode: allowing hardcoded fallback for {atr_data.asset}")
            return (False, "")

        # Strict mode: block on hardcoded fallback
        if atr_data.source == "fallback_hardcoded":
            return (
                True,
                f"Strict mode: blocking gate - no fresh ATR or realized vol for {atr_data.asset}"
            )

        return (False, "")

    async def get_atr_with_staleness_check(
        self,
        asset: str,
        price: Optional[float] = None,
        log_stale: bool = True,
    ) -> Tuple[ATRData, bool]:
        """
        Get ATR data with staleness check and optional logging.

        Args:
            asset: Asset symbol (BTC, ETH)
            price: Current price (optional)
            log_stale: Whether to log warnings for stale data

        Returns:
            Tuple of (ATRData, is_stale)
        """
        atr_data = await self.get_atr(asset, price)
        is_stale, message = self.check_staleness(atr_data)

        if is_stale and log_stale:
            print(f"[atr] WARNING: {message}")

        return (atr_data, is_stale)

    def clear_cache(self) -> None:
        """Clear the ATR cache."""
        self._cache.clear()


# Global singleton for shared access
_atr_provider: Optional[ATRProvider] = None


def get_atr_provider() -> ATRProvider:
    """Get the global ATR provider singleton."""
    global _atr_provider
    if _atr_provider is None:
        _atr_provider = ATRProvider()
    return _atr_provider


def init_atr_provider(pool: asyncpg.Pool) -> ATRProvider:
    """Initialize the global ATR provider with a database pool."""
    provider = get_atr_provider()
    provider.set_pool(pool)
    return provider
