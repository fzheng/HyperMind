"""
Market Regime Detection

Phase 5: Adaptive strategy parameters based on market conditions.
Phase 6.1: Multi-exchange support via ATR provider infrastructure.

Detects three market regimes:
1. TRENDING: Strong directional moves with momentum
2. RANGING: Sideways consolidation, mean-reverting
3. VOLATILE: High volatility, uncertain direction

Each regime suggests different strategy parameters:
- Stop distances (wider in trending, tighter in ranging)
- Kelly fraction (higher in trending, lower in volatile)
- Consensus thresholds (stricter in volatile)

Detection Methods:
1. Moving Average Relationship: MA20/MA50 spread
2. ATR-based Volatility: Current ATR vs historical average
3. Price Range Compression: Bollinger Band width

@module regime
"""

import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional, Dict, List, Tuple
from functools import lru_cache

import asyncpg

# Import ATR provider infrastructure for multi-exchange candle fetching
from .atr_provider.interface import Candle, ATRProviderInterface
from .atr_provider.manager import get_atr_manager


# Configuration
REGIME_LOOKBACK_MINUTES = int(os.getenv("REGIME_LOOKBACK_MINUTES", "60"))  # 1 hour
REGIME_MA_SHORT = int(os.getenv("REGIME_MA_SHORT", "20"))  # 20-minute MA
REGIME_MA_LONG = int(os.getenv("REGIME_MA_LONG", "50"))  # 50-minute MA
REGIME_TREND_THRESHOLD = float(os.getenv("REGIME_TREND_THRESHOLD", "0.02"))  # 2% MA spread
REGIME_VOLATILITY_HIGH_MULT = float(os.getenv("REGIME_VOLATILITY_HIGH_MULT", "1.5"))  # 1.5x avg vol
REGIME_VOLATILITY_LOW_MULT = float(os.getenv("REGIME_VOLATILITY_LOW_MULT", "0.7"))  # 0.7x avg vol
REGIME_CACHE_TTL_SECONDS = int(os.getenv("REGIME_CACHE_TTL_SECONDS", "60"))  # 1 minute
REGIME_MIN_CANDLES = int(os.getenv("REGIME_MIN_CANDLES", "50"))  # Minimum candles for detection


class MarketRegime(Enum):
    """Market regime types."""
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"  # Insufficient data


@dataclass
class RegimeParams:
    """Strategy parameters for a given regime."""

    # Stop distance multiplier (applied to ATR-based stops)
    stop_multiplier: float

    # Kelly fraction adjustment (multiplied with base Kelly)
    kelly_multiplier: float

    # Consensus confidence threshold adjustment
    min_confidence_adjustment: float

    # Position size cap (as fraction of max)
    max_position_fraction: float

    # Description for logging
    description: str


# Regime-specific parameter presets
REGIME_PARAMS: Dict[MarketRegime, RegimeParams] = {
    MarketRegime.TRENDING: RegimeParams(
        stop_multiplier=1.2,          # Wider stops to ride the trend
        kelly_multiplier=1.0,         # Full Kelly in trending markets
        min_confidence_adjustment=0.0,  # No adjustment
        max_position_fraction=1.0,    # Full position allowed
        description="Trending: Wider stops, full sizing",
    ),
    MarketRegime.RANGING: RegimeParams(
        stop_multiplier=0.8,          # Tighter stops in ranging markets
        kelly_multiplier=0.75,        # Reduced Kelly for choppy markets
        min_confidence_adjustment=0.05,  # Require higher confidence
        max_position_fraction=0.75,   # Cap positions at 75% of max
        description="Ranging: Tighter stops, reduced sizing",
    ),
    MarketRegime.VOLATILE: RegimeParams(
        stop_multiplier=1.5,          # Much wider stops for volatility
        kelly_multiplier=0.5,         # Half Kelly in volatile markets
        min_confidence_adjustment=0.10,  # Require much higher confidence
        max_position_fraction=0.5,    # Cap positions at 50% of max
        description="Volatile: Conservative sizing, wide stops",
    ),
    MarketRegime.UNKNOWN: RegimeParams(
        stop_multiplier=1.0,          # Default stops
        kelly_multiplier=0.5,         # Conservative sizing when uncertain
        min_confidence_adjustment=0.05,  # Slightly higher confidence
        max_position_fraction=0.5,    # Conservative position cap
        description="Unknown: Conservative defaults",
    ),
}


@dataclass
class RegimeAnalysis:
    """Complete regime analysis for an asset."""

    asset: str
    regime: MarketRegime
    params: RegimeParams
    confidence: float  # 0-1, how confident in the regime detection

    # Component signals
    ma_spread_pct: Optional[float]  # MA20 vs MA50 spread
    volatility_ratio: Optional[float]  # Current vol / historical vol
    price_range_pct: Optional[float]  # Recent price range as % of price

    # Metadata
    timestamp: datetime
    candles_used: int
    source: str  # 'full', 'partial', 'fallback'
    exchange: str = "hyperliquid"  # Source exchange for candle data

    @property
    def is_valid(self) -> bool:
        """Check if regime detection has sufficient data."""
        return self.candles_used >= REGIME_MIN_CANDLES and self.regime != MarketRegime.UNKNOWN

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "asset": self.asset,
            "regime": self.regime.value,
            "confidence": round(self.confidence, 2),
            "exchange": self.exchange,
            "params": {
                "stop_multiplier": self.params.stop_multiplier,
                "kelly_multiplier": self.params.kelly_multiplier,
                "min_confidence_adjustment": self.params.min_confidence_adjustment,
                "max_position_fraction": self.params.max_position_fraction,
            },
            "signals": {
                "ma_spread_pct": round(self.ma_spread_pct, 4) if self.ma_spread_pct else None,
                "volatility_ratio": round(self.volatility_ratio, 2) if self.volatility_ratio else None,
                "price_range_pct": round(self.price_range_pct, 4) if self.price_range_pct else None,
            },
            "candles_used": self.candles_used,
            "timestamp": self.timestamp.isoformat(),
        }


class RegimeDetector:
    """
    Market regime detection engine.

    Uses multiple signals to classify market state:
    1. Moving average relationship (trending vs ranging)
    2. Volatility level (high vs low)
    3. Price range compression (breakout potential)

    Phase 6.1: Multi-exchange support via ATR provider infrastructure.
    Candles are fetched via the ATR manager's exchange-specific providers.

    Usage:
        detector = RegimeDetector(db_pool)
        analysis = await detector.detect_regime("BTC")  # Uses default exchange
        analysis = await detector.detect_regime("BTC", exchange="bybit")
        params = analysis.params
    """

    def __init__(self, db: Optional[asyncpg.Pool] = None, default_exchange: str = "hyperliquid"):
        """
        Initialize regime detector.

        Args:
            db: Database pool for price data access (used for HL fallback)
            default_exchange: Default exchange for candle data (hyperliquid, bybit)
        """
        self.db = db
        self.default_exchange = default_exchange.lower()
        self._cache: Dict[str, Tuple[RegimeAnalysis, datetime]] = {}

    async def detect_regime(
        self,
        asset: str,
        exchange: Optional[str] = None,
    ) -> RegimeAnalysis:
        """
        Detect current market regime for an asset.

        Args:
            asset: Asset symbol (BTC, ETH)
            exchange: Exchange to use for candle data (uses default if None)

        Returns:
            RegimeAnalysis with detected regime and parameters
        """
        target_exchange = (exchange or self.default_exchange).lower()

        # Check cache with exchange-specific key
        cache_key = f"{asset.upper()}:{target_exchange}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        # Fetch price data via ATR provider infrastructure
        candles = await self._fetch_candles_multi_exchange(
            asset,
            exchange=target_exchange,
            count=max(REGIME_MA_LONG + 10, REGIME_LOOKBACK_MINUTES),
        )

        if len(candles) < REGIME_MIN_CANDLES:
            analysis = self._create_unknown_regime(asset, len(candles), target_exchange)
            self._cache_result(cache_key, analysis)
            return analysis

        # Convert Candle objects to dict format for calculations
        candle_dicts = self._candles_to_dicts(candles)

        # Calculate signals
        ma_short = self._calculate_ma(candle_dicts, REGIME_MA_SHORT)
        ma_long = self._calculate_ma(candle_dicts, REGIME_MA_LONG)
        current_vol = self._calculate_volatility(candle_dicts, lookback=14)
        historical_vol = self._calculate_volatility(candle_dicts, lookback=min(len(candle_dicts), 50))
        price_range = self._calculate_price_range(candle_dicts, lookback=20)
        current_price = candle_dicts[-1]["close"]

        # Calculate signal values
        ma_spread_pct = None
        if ma_short and ma_long and ma_long > 0:
            ma_spread_pct = (ma_short - ma_long) / ma_long

        volatility_ratio = None
        if current_vol and historical_vol and historical_vol > 0:
            volatility_ratio = current_vol / historical_vol

        price_range_pct = None
        if price_range and current_price and current_price > 0:
            price_range_pct = price_range / current_price

        # Determine regime
        regime, confidence = self._classify_regime(
            ma_spread_pct=ma_spread_pct,
            volatility_ratio=volatility_ratio,
            price_range_pct=price_range_pct,
        )

        analysis = RegimeAnalysis(
            asset=asset,
            regime=regime,
            params=REGIME_PARAMS[regime],
            confidence=confidence,
            ma_spread_pct=ma_spread_pct,
            volatility_ratio=volatility_ratio,
            price_range_pct=price_range_pct,
            timestamp=datetime.now(timezone.utc),
            candles_used=len(candle_dicts),
            source="full" if len(candle_dicts) >= REGIME_MA_LONG else "partial",
            exchange=target_exchange,
        )

        self._cache_result(cache_key, analysis)
        return analysis

    def _classify_regime(
        self,
        ma_spread_pct: Optional[float],
        volatility_ratio: Optional[float],
        price_range_pct: Optional[float],
    ) -> Tuple[MarketRegime, float]:
        """
        Classify regime based on signals.

        Returns:
            Tuple of (regime, confidence)
        """
        scores = {
            MarketRegime.TRENDING: 0.0,
            MarketRegime.RANGING: 0.0,
            MarketRegime.VOLATILE: 0.0,
        }

        # Score based on MA spread
        if ma_spread_pct is not None:
            abs_spread = abs(ma_spread_pct)
            if abs_spread > REGIME_TREND_THRESHOLD:
                # Strong trend signal
                scores[MarketRegime.TRENDING] += 0.4
            elif abs_spread < REGIME_TREND_THRESHOLD * 0.5:
                # MAs converged - ranging
                scores[MarketRegime.RANGING] += 0.3

        # Score based on volatility
        if volatility_ratio is not None:
            if volatility_ratio > REGIME_VOLATILITY_HIGH_MULT:
                # High volatility
                scores[MarketRegime.VOLATILE] += 0.4
            elif volatility_ratio < REGIME_VOLATILITY_LOW_MULT:
                # Low volatility (ranging or trending)
                scores[MarketRegime.RANGING] += 0.2
                scores[MarketRegime.TRENDING] += 0.1
            else:
                # Normal volatility
                scores[MarketRegime.TRENDING] += 0.15
                scores[MarketRegime.RANGING] += 0.15

        # Score based on price range
        if price_range_pct is not None:
            if price_range_pct > 0.03:  # 3% range
                # Wide range - trending or volatile
                scores[MarketRegime.TRENDING] += 0.2
                scores[MarketRegime.VOLATILE] += 0.2
            elif price_range_pct < 0.01:  # 1% range
                # Tight range - ranging
                scores[MarketRegime.RANGING] += 0.3

        # Check for volatility override (always wins if very high)
        if volatility_ratio and volatility_ratio > 2.0:
            return MarketRegime.VOLATILE, 0.9

        # Get highest scoring regime
        max_regime = max(scores, key=scores.get)
        max_score = scores[max_regime]

        # Calculate confidence based on score dominance
        total_score = sum(scores.values())
        if total_score == 0:
            return MarketRegime.UNKNOWN, 0.0

        confidence = max_score / total_score

        # Require minimum score to avoid random classification
        if max_score < 0.3:
            return MarketRegime.UNKNOWN, confidence

        return max_regime, min(confidence, 0.95)

    async def _fetch_candles_multi_exchange(
        self,
        asset: str,
        exchange: str,
        count: int,
    ) -> List[Candle]:
        """
        Fetch candles via ATR provider infrastructure.

        Uses exchange-specific ATR providers which have their own candle fetching.
        Falls back to Hyperliquid DB for historical compatibility.

        Args:
            asset: Asset symbol (BTC, ETH)
            exchange: Target exchange (hyperliquid, bybit)
            count: Number of candles to fetch

        Returns:
            List of Candle objects
        """
        try:
            # Get ATR manager and appropriate provider
            atr_manager = get_atr_manager()
            provider = atr_manager.get_provider(exchange)

            if provider and provider.is_configured:
                # Use provider's candle fetching
                candles = await provider.get_candles(asset, count=count)
                if len(candles) >= REGIME_MIN_CANDLES:
                    # Sort by timestamp ascending for calculations
                    return sorted(candles, key=lambda c: c.ts)

                print(
                    f"[regime] {exchange} provider returned {len(candles)} candles "
                    f"(need {REGIME_MIN_CANDLES}), trying fallback"
                )

            # Fallback to Hyperliquid DB if target not available
            if exchange != "hyperliquid" and self.db:
                print(f"[regime] Falling back to hyperliquid DB for {asset} candles")
                return await self._fetch_candles_from_db(asset, count)

            # Direct DB fetch for hyperliquid
            if self.db:
                return await self._fetch_candles_from_db(asset, count)

            return []

        except Exception as e:
            print(f"[regime] Error fetching candles for {asset} on {exchange}: {e}")
            # Last resort: try direct DB
            if self.db:
                return await self._fetch_candles_from_db(asset, count)
            return []

    async def _fetch_candles_from_db(self, asset: str, count: int) -> List[Candle]:
        """
        Fetch candles from marks_1m database (Hyperliquid data).

        Legacy method for backward compatibility and fallback.
        """
        if not self.db:
            return []

        try:
            async with self.db.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT ts, mid, high, low, close
                    FROM marks_1m
                    WHERE asset = $1
                      AND ts > NOW() - INTERVAL '1 minute' * $2
                    ORDER BY ts ASC
                    """,
                    asset,
                    count,
                )
                candles = []
                for row in rows:
                    if row["mid"] is None:
                        continue

                    mid = float(row["mid"])
                    candles.append(Candle(
                        ts=row["ts"],
                        open=mid,
                        high=float(row["high"]) if row["high"] else mid,
                        low=float(row["low"]) if row["low"] else mid,
                        close=float(row["close"]) if row["close"] else mid,
                    ))
                return candles
        except Exception as e:
            print(f"[regime] Failed to fetch candles from DB: {e}")
            return []

    def _candles_to_dicts(self, candles: List[Candle]) -> List[dict]:
        """Convert Candle objects to dict format for calculation methods."""
        return [
            {
                "ts": c.ts,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
            }
            for c in candles
        ]

    def _calculate_ma(self, candles: List[dict], period: int) -> Optional[float]:
        """Calculate simple moving average of close prices."""
        if len(candles) < period:
            return None

        closes = [c["close"] for c in candles[-period:] if c["close"]]
        if len(closes) < period:
            return None

        return sum(closes) / len(closes)

    def _calculate_volatility(self, candles: List[dict], lookback: int) -> Optional[float]:
        """Calculate volatility as average true range percentage."""
        if len(candles) < lookback + 1:
            return None

        recent = candles[-lookback:]
        true_ranges = []

        for i in range(1, len(recent)):
            curr = recent[i]
            prev = recent[i-1]

            if not all([curr["high"], curr["low"], prev["close"]]):
                continue

            tr = max(
                curr["high"] - curr["low"],
                abs(curr["high"] - prev["close"]),
                abs(curr["low"] - prev["close"]),
            )

            # Normalize by price
            if curr["close"] and curr["close"] > 0:
                true_ranges.append(tr / curr["close"])

        if len(true_ranges) < lookback // 2:
            return None

        return sum(true_ranges) / len(true_ranges)

    def _calculate_price_range(self, candles: List[dict], lookback: int) -> Optional[float]:
        """Calculate high-low range over lookback period."""
        if len(candles) < lookback:
            return None

        recent = candles[-lookback:]
        highs = [c["high"] for c in recent if c["high"]]
        lows = [c["low"] for c in recent if c["low"]]

        if not highs or not lows:
            return None

        return max(highs) - min(lows)

    def _get_cached(self, cache_key: str) -> Optional[RegimeAnalysis]:
        """Get cached regime analysis if still valid."""
        if cache_key not in self._cache:
            return None

        analysis, cached_at = self._cache[cache_key]
        age = (datetime.now(timezone.utc) - cached_at).total_seconds()

        if age > REGIME_CACHE_TTL_SECONDS:
            return None

        return analysis

    def _cache_result(self, cache_key: str, analysis: RegimeAnalysis) -> None:
        """Cache regime analysis."""
        self._cache[cache_key] = (analysis, datetime.now(timezone.utc))

    def _create_unknown_regime(
        self,
        asset: str,
        candles_count: int,
        exchange: str = "hyperliquid",
    ) -> RegimeAnalysis:
        """Create unknown regime analysis when data is insufficient."""
        return RegimeAnalysis(
            asset=asset,
            regime=MarketRegime.UNKNOWN,
            params=REGIME_PARAMS[MarketRegime.UNKNOWN],
            confidence=0.0,
            ma_spread_pct=None,
            volatility_ratio=None,
            price_range_pct=None,
            timestamp=datetime.now(timezone.utc),
            candles_used=candles_count,
            source="fallback",
            exchange=exchange,
        )

    def clear_cache(self, asset: Optional[str] = None, exchange: Optional[str] = None) -> None:
        """
        Clear regime cache.

        Args:
            asset: Specific asset to clear (clears all if None)
            exchange: Specific exchange to clear (clears all if None)
        """
        if asset and exchange:
            cache_key = f"{asset.upper()}:{exchange.lower()}"
            self._cache.pop(cache_key, None)
        elif asset:
            # Clear all exchanges for this asset
            keys_to_clear = [k for k in self._cache if k.startswith(f"{asset.upper()}:")]
            for key in keys_to_clear:
                self._cache.pop(key, None)
        else:
            self._cache.clear()


# Global detector instance
_detector: Optional[RegimeDetector] = None


def get_regime_detector(db: Optional[asyncpg.Pool] = None) -> RegimeDetector:
    """Get or create global regime detector instance."""
    global _detector
    if _detector is None:
        _detector = RegimeDetector(db)
    elif db is not None and _detector.db is None:
        _detector.db = db
    return _detector


async def detect_market_regime(
    asset: str,
    db: Optional[asyncpg.Pool] = None,
    exchange: Optional[str] = None,
) -> RegimeAnalysis:
    """
    Convenience function to detect market regime.

    Args:
        asset: Asset symbol (BTC, ETH)
        db: Database pool
        exchange: Target exchange for candle data (uses default if None)

    Returns:
        RegimeAnalysis with detected regime and parameters
    """
    detector = get_regime_detector(db)
    return await detector.detect_regime(asset, exchange=exchange)


def get_regime_adjusted_kelly(
    base_kelly: float,
    regime: MarketRegime,
) -> float:
    """
    Adjust Kelly fraction based on market regime.

    Args:
        base_kelly: Base Kelly fraction (0-1)
        regime: Current market regime

    Returns:
        Adjusted Kelly fraction
    """
    params = REGIME_PARAMS[regime]
    return base_kelly * params.kelly_multiplier


def get_regime_adjusted_stop(
    base_stop_pct: float,
    regime: MarketRegime,
) -> float:
    """
    Adjust stop distance based on market regime.

    Args:
        base_stop_pct: Base stop distance as percentage
        regime: Current market regime

    Returns:
        Adjusted stop distance as percentage
    """
    params = REGIME_PARAMS[regime]
    return base_stop_pct * params.stop_multiplier


def get_regime_adjusted_confidence(
    min_confidence: float,
    regime: MarketRegime,
) -> float:
    """
    Adjust minimum confidence threshold based on market regime.

    Args:
        min_confidence: Base minimum confidence
        regime: Current market regime

    Returns:
        Adjusted minimum confidence (higher in volatile/unknown regimes)
    """
    params = REGIME_PARAMS[regime]
    return min(min_confidence + params.min_confidence_adjustment, 0.95)
