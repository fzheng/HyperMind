"""
Dynamic Hold Time Estimation

Estimates expected hold time for trades based on historical episode data.
Used to improve funding cost accuracy in EV calculations.

Key features:
- Per-asset hold time estimation (BTC vs ETH may differ)
- Regime-adjusted estimates (trending = longer holds, volatile = shorter)
- ATR-weighted adjustments (higher vol = shorter expected holds)
- Fallback to configurable default when insufficient data

@module hold_time_estimator
"""

import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Tuple
from enum import Enum

import asyncpg


# Configuration
DEFAULT_HOLD_HOURS = float(os.getenv("DEFAULT_HOLD_HOURS", "24.0"))
MIN_EPISODES_FOR_ESTIMATE = int(os.getenv("HOLD_TIME_MIN_EPISODES", "10"))
HOLD_TIME_LOOKBACK_DAYS = int(os.getenv("HOLD_TIME_LOOKBACK_DAYS", "30"))
HOLD_TIME_CACHE_TTL_SECONDS = int(os.getenv("HOLD_TIME_CACHE_TTL_SECONDS", "300"))  # 5 min

# Regime adjustments (multipliers)
# Trending markets tend to have longer holds (ride the trend)
# Volatile markets tend to have shorter holds (quick in/out)
REGIME_HOLD_TIME_MULTIPLIERS = {
    "TRENDING": 1.25,   # +25% hold time
    "RANGING": 1.0,     # Baseline
    "VOLATILE": 0.75,   # -25% hold time
    "UNKNOWN": 1.0,     # Default
}


class EstimateSource(Enum):
    """Source of hold time estimate."""
    HISTORICAL = "historical"  # From episode data
    REGIME_ADJUSTED = "regime_adjusted"  # Historical with regime adjustment
    FALLBACK = "fallback"  # Default when no data


@dataclass
class HoldTimeEstimate:
    """Result of hold time estimation."""
    hours: float
    source: EstimateSource
    episode_count: int  # Number of episodes used
    median_hours: Optional[float] = None  # Median from data
    std_hours: Optional[float] = None  # Standard deviation
    regime: Optional[str] = None  # Regime used for adjustment
    asset: str = ""


@dataclass
class CachedHoldTime:
    """Cached hold time estimate."""
    estimate: HoldTimeEstimate
    fetched_at: datetime

    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        age = (datetime.now(timezone.utc) - self.fetched_at).total_seconds()
        return age > HOLD_TIME_CACHE_TTL_SECONDS


class HoldTimeEstimator:
    """
    Estimates expected hold time from historical episode data.

    Usage:
        estimator = HoldTimeEstimator()
        estimate = await estimator.get_hold_time("BTC", db)
        funding_cost = get_funding_cost_bps(asset, exchange, estimate.hours, side)
    """

    def __init__(self):
        """Initialize hold time estimator."""
        self._cache: Dict[str, CachedHoldTime] = {}

    async def get_hold_time(
        self,
        asset: str,
        db: asyncpg.Pool,
        regime: Optional[str] = None,
        force_refresh: bool = False,
    ) -> HoldTimeEstimate:
        """
        Get expected hold time for an asset.

        Args:
            asset: Asset symbol (BTC, ETH)
            db: Database pool
            regime: Optional market regime for adjustment
            force_refresh: Bypass cache

        Returns:
            HoldTimeEstimate with expected hold hours
        """
        cache_key = f"{asset.upper()}:{regime or 'none'}"

        # Check cache
        if not force_refresh and cache_key in self._cache:
            cached = self._cache[cache_key]
            if not cached.is_expired:
                return cached.estimate

        # Fetch from database
        estimate = await self._compute_estimate(asset, db, regime)

        # Update cache
        self._cache[cache_key] = CachedHoldTime(
            estimate=estimate,
            fetched_at=datetime.now(timezone.utc),
        )

        return estimate

    async def _compute_estimate(
        self,
        asset: str,
        db: asyncpg.Pool,
        regime: Optional[str] = None,
    ) -> HoldTimeEstimate:
        """
        Compute hold time estimate from episode data.

        Uses closed episodes from position_signals table.
        """
        try:
            # Query historical hold times
            cutoff = datetime.now(timezone.utc) - timedelta(days=HOLD_TIME_LOOKBACK_DAYS)

            query = """
                SELECT
                    hold_secs,
                    entry_ts,
                    r_clamped
                FROM position_signals
                WHERE asset = $1
                  AND status = 'closed'
                  AND hold_secs IS NOT NULL
                  AND hold_secs > 0
                  AND updated_at >= $2
                ORDER BY updated_at DESC
                LIMIT 1000
            """

            rows = await db.fetch(query, asset.upper(), cutoff)

            if len(rows) < MIN_EPISODES_FOR_ESTIMATE:
                # Not enough data, use fallback
                return HoldTimeEstimate(
                    hours=DEFAULT_HOLD_HOURS,
                    source=EstimateSource.FALLBACK,
                    episode_count=len(rows),
                    asset=asset.upper(),
                    regime=regime,
                )

            # Calculate statistics
            hold_hours_list = [row["hold_secs"] / 3600.0 for row in rows]

            # Use median for robustness against outliers
            sorted_hours = sorted(hold_hours_list)
            n = len(sorted_hours)
            if n % 2 == 0:
                median_hours = (sorted_hours[n//2 - 1] + sorted_hours[n//2]) / 2
            else:
                median_hours = sorted_hours[n//2]

            # Calculate standard deviation
            mean_hours = sum(hold_hours_list) / n
            variance = sum((h - mean_hours) ** 2 for h in hold_hours_list) / n
            std_hours = variance ** 0.5

            # Apply regime adjustment if provided
            base_hours = median_hours
            if regime:
                multiplier = REGIME_HOLD_TIME_MULTIPLIERS.get(regime.upper(), 1.0)
                adjusted_hours = median_hours * multiplier
                return HoldTimeEstimate(
                    hours=adjusted_hours,
                    source=EstimateSource.REGIME_ADJUSTED,
                    episode_count=n,
                    median_hours=median_hours,
                    std_hours=std_hours,
                    asset=asset.upper(),
                    regime=regime,
                )

            return HoldTimeEstimate(
                hours=median_hours,
                source=EstimateSource.HISTORICAL,
                episode_count=n,
                median_hours=median_hours,
                std_hours=std_hours,
                asset=asset.upper(),
            )

        except Exception as e:
            print(f"[hold-time] Error computing estimate for {asset}: {e}")
            return HoldTimeEstimate(
                hours=DEFAULT_HOLD_HOURS,
                source=EstimateSource.FALLBACK,
                episode_count=0,
                asset=asset.upper(),
                regime=regime,
            )

    def get_hold_time_sync(
        self,
        asset: str,
        regime: Optional[str] = None,
    ) -> HoldTimeEstimate:
        """
        Synchronous hold time lookup using cached data.

        For async code, prefer get_hold_time() which can refresh from DB.

        Args:
            asset: Asset symbol (BTC, ETH)
            regime: Optional regime for adjustment

        Returns:
            Cached estimate or fallback
        """
        cache_key = f"{asset.upper()}:{regime or 'none'}"

        if cache_key in self._cache and not self._cache[cache_key].is_expired:
            return self._cache[cache_key].estimate

        # Check without regime
        base_key = f"{asset.upper()}:none"
        if base_key in self._cache and not self._cache[base_key].is_expired:
            base_estimate = self._cache[base_key].estimate
            if regime:
                multiplier = REGIME_HOLD_TIME_MULTIPLIERS.get(regime.upper(), 1.0)
                return HoldTimeEstimate(
                    hours=base_estimate.hours * multiplier,
                    source=EstimateSource.REGIME_ADJUSTED,
                    episode_count=base_estimate.episode_count,
                    median_hours=base_estimate.median_hours,
                    std_hours=base_estimate.std_hours,
                    asset=asset.upper(),
                    regime=regime,
                )
            return base_estimate

        # Fallback
        return HoldTimeEstimate(
            hours=DEFAULT_HOLD_HOURS,
            source=EstimateSource.FALLBACK,
            episode_count=0,
            asset=asset.upper(),
            regime=regime,
        )

    async def get_asset_summary(
        self,
        db: asyncpg.Pool,
    ) -> Dict[str, dict]:
        """
        Get hold time summary for all assets.

        Returns:
            Dict mapping asset -> {median_hours, episode_count, std_hours}
        """
        try:
            query = """
                SELECT
                    asset,
                    COUNT(*) as episode_count,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY hold_secs/3600.0) as median_hours,
                    AVG(hold_secs/3600.0) as mean_hours,
                    STDDEV(hold_secs/3600.0) as std_hours
                FROM position_signals
                WHERE status = 'closed'
                  AND hold_secs IS NOT NULL
                  AND hold_secs > 0
                  AND updated_at >= NOW() - INTERVAL '%s days'
                GROUP BY asset
            """ % HOLD_TIME_LOOKBACK_DAYS

            rows = await db.fetch(query)

            return {
                row["asset"]: {
                    "episode_count": row["episode_count"],
                    "median_hours": round(float(row["median_hours"]), 2) if row["median_hours"] else None,
                    "mean_hours": round(float(row["mean_hours"]), 2) if row["mean_hours"] else None,
                    "std_hours": round(float(row["std_hours"]), 2) if row["std_hours"] else None,
                }
                for row in rows
            }
        except Exception as e:
            print(f"[hold-time] Error getting summary: {e}")
            return {}

    def clear_cache(self) -> None:
        """Clear the hold time cache."""
        self._cache.clear()

    def get_cache_status(self) -> Dict[str, dict]:
        """Get cache status for debugging."""
        return {
            key: {
                "hours": cached.estimate.hours,
                "source": cached.estimate.source.value,
                "episode_count": cached.estimate.episode_count,
                "age_seconds": (datetime.now(timezone.utc) - cached.fetched_at).total_seconds(),
                "is_expired": cached.is_expired,
            }
            for key, cached in self._cache.items()
        }


# Global singleton
_hold_time_estimator: Optional[HoldTimeEstimator] = None


def get_hold_time_estimator() -> HoldTimeEstimator:
    """Get the global hold time estimator singleton."""
    global _hold_time_estimator
    if _hold_time_estimator is None:
        _hold_time_estimator = HoldTimeEstimator()
    return _hold_time_estimator


async def init_hold_time_estimator(db: asyncpg.Pool) -> HoldTimeEstimator:
    """
    Initialize the global hold time estimator and pre-fetch estimates.

    Args:
        db: Database pool

    Returns:
        Configured HoldTimeEstimator
    """
    global _hold_time_estimator
    _hold_time_estimator = HoldTimeEstimator()

    # Pre-fetch estimates for common assets
    for asset in ["BTC", "ETH"]:
        estimate = await _hold_time_estimator.get_hold_time(asset, db)
        print(f"[hold-time] {asset}: {estimate.hours:.1f}h ({estimate.source.value}, {estimate.episode_count} episodes)")

    return _hold_time_estimator
