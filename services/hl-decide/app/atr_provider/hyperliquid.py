"""
Hyperliquid ATR Provider

Provides ATR data from Hyperliquid via the marks_1m database table.

@module atr_provider.hyperliquid
"""

import math
from datetime import datetime, timezone, timedelta
from typing import Optional, List

import asyncpg

from .interface import (
    ATRProviderInterface,
    ATRData,
    Candle,
    calculate_atr,
    ATR_PERIOD,
)


# Realized volatility fallback configuration
ATR_REALIZED_VOL_WINDOW_HOURS = 24
ATR_REALIZED_VOL_MIN_SAMPLES = 60


class HyperliquidATRProvider(ATRProviderInterface):
    """
    ATR provider using Hyperliquid data from marks_1m table.

    This is the primary data source for Hyperliquid executions.
    """

    def __init__(self, pool: Optional[asyncpg.Pool] = None):
        """
        Initialize Hyperliquid ATR provider.

        Args:
            pool: asyncpg connection pool for database access
        """
        super().__init__("hyperliquid")
        self.pool = pool

    def set_pool(self, pool: asyncpg.Pool) -> None:
        """Set the database pool (for late initialization)."""
        self.pool = pool

    @property
    def is_configured(self) -> bool:
        """Check if provider is properly configured."""
        return self.pool is not None

    async def get_atr(self, asset: str, price: Optional[float] = None) -> ATRData:
        """
        Get ATR data for an asset from Hyperliquid marks_1m table.

        Falls back to realized volatility, then hardcoded values.
        """
        asset_upper = asset.upper()

        # Check cache first
        cached = self._get_cached(asset_upper)
        if cached is not None:
            # Update price if provided and different
            if price is not None and price != cached.price and price > 0:
                cached = ATRData(
                    asset=asset_upper,
                    atr=cached.atr,
                    atr_pct=cached.atr / price * 100,
                    price=price,
                    multiplier=cached.multiplier,
                    stop_distance_pct=cached.atr / price * 100 * cached.multiplier,
                    timestamp=cached.timestamp,
                    source=cached.source,
                    exchange=cached.exchange,
                )
            return cached

        # Try to fetch from database
        atr_data = await self._fetch_atr_from_db(asset_upper, price)
        if atr_data:
            self._set_cached(asset_upper, atr_data)
            return atr_data

        # Try data-driven fallback: realized volatility
        current_price = price or 100000.0
        realized_vol_data = await self._compute_realized_vol(asset_upper, current_price)
        if realized_vol_data:
            self._set_cached(asset_upper, realized_vol_data)
            return realized_vol_data

        # Last resort: hardcoded fallback
        return self._fallback_atr(asset_upper, price)

    async def get_candles(
        self,
        asset: str,
        count: int = ATR_PERIOD + 5,
    ) -> List[Candle]:
        """Fetch OHLC candles from marks_1m table."""
        if self.pool is None:
            return []

        try:
            async with self.pool.acquire() as conn:
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
                    asset.upper(),
                    count,
                )

                return [
                    Candle(
                        ts=row["ts"],
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                    )
                    for row in rows
                ]
        except Exception as e:
            print(f"[atr:hyperliquid] Failed to fetch candles for {asset}: {e}")
            return []

    async def _fetch_atr_from_db(
        self, asset: str, price: Optional[float]
    ) -> Optional[ATRData]:
        """Fetch ATR from marks_1m table."""
        if self.pool is None:
            return None

        try:
            async with self.pool.acquire() as conn:
                # First try pre-computed atr14
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
                        exchange="hyperliquid",
                    )

                # If no atr14, calculate from candles
                candles = await self.get_candles(asset, ATR_PERIOD + 5)
                if len(candles) >= ATR_PERIOD + 1:
                    atr = calculate_atr(candles, ATR_PERIOD)
                    if atr is not None and atr > 0:
                        current_price = price or float(candles[0].close)
                        multiplier = self._get_multiplier(asset)

                        return ATRData(
                            asset=asset,
                            atr=atr,
                            atr_pct=atr / current_price * 100 if current_price > 0 else 0,
                            price=current_price,
                            multiplier=multiplier,
                            stop_distance_pct=atr / current_price * 100 * multiplier if current_price > 0 else 0,
                            timestamp=candles[0].ts,
                            source="calculated",
                            exchange="hyperliquid",
                        )

        except Exception as e:
            print(f"[atr:hyperliquid] Failed to fetch ATR from DB for {asset}: {e}")

        return None

    async def _compute_realized_vol(
        self, asset: str, price: float
    ) -> Optional[ATRData]:
        """Compute realized volatility from rolling 24h marks_1m data."""
        if self.pool is None:
            return None

        try:
            async with self.pool.acquire() as conn:
                cutoff = datetime.now(timezone.utc) - timedelta(
                    hours=ATR_REALIZED_VOL_WINDOW_HOURS
                )
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

                mean_abs_return = sum(log_returns) / len(log_returns)
                realized_vol_pct = mean_abs_return * 100

                multiplier = self._get_multiplier(asset)
                latest_ts = rows[-1]["ts"]

                print(
                    f"[atr:hyperliquid] Using 24h realized vol for {asset}: "
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
                    exchange="hyperliquid",
                )

        except Exception as e:
            print(f"[atr:hyperliquid] Failed to compute realized vol for {asset}: {e}")
            return None
