"""
Tests for Market Regime Detection

Tests the regime classifier, parameter adjustments, and integration.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from app.regime import (
    MarketRegime,
    RegimeParams,
    RegimeAnalysis,
    RegimeDetector,
    REGIME_PARAMS,
    detect_market_regime,
    get_regime_adjusted_kelly,
    get_regime_adjusted_stop,
    get_regime_adjusted_confidence,
    get_regime_detector,
    REGIME_TREND_THRESHOLD,
    REGIME_VOLATILITY_HIGH_MULT,
)


class TestMarketRegime:
    """Tests for MarketRegime enum."""

    def test_regime_values(self):
        """Test regime enum values."""
        assert MarketRegime.TRENDING.value == "trending"
        assert MarketRegime.RANGING.value == "ranging"
        assert MarketRegime.VOLATILE.value == "volatile"
        assert MarketRegime.UNKNOWN.value == "unknown"

    def test_all_regimes_have_params(self):
        """Test all regimes have parameter presets."""
        for regime in MarketRegime:
            assert regime in REGIME_PARAMS
            params = REGIME_PARAMS[regime]
            assert isinstance(params, RegimeParams)


class TestRegimeParams:
    """Tests for regime parameter presets."""

    def test_trending_params(self):
        """Test trending regime parameters."""
        params = REGIME_PARAMS[MarketRegime.TRENDING]
        assert params.stop_multiplier > 1.0  # Wider stops
        assert params.kelly_multiplier == 1.0  # Full Kelly
        assert params.max_position_fraction == 1.0  # Full position

    def test_ranging_params(self):
        """Test ranging regime parameters."""
        params = REGIME_PARAMS[MarketRegime.RANGING]
        assert params.stop_multiplier < 1.0  # Tighter stops
        assert params.kelly_multiplier < 1.0  # Reduced Kelly
        assert params.max_position_fraction < 1.0  # Reduced position

    def test_volatile_params(self):
        """Test volatile regime parameters."""
        params = REGIME_PARAMS[MarketRegime.VOLATILE]
        assert params.stop_multiplier > 1.0  # Wide stops
        assert params.kelly_multiplier < params.stop_multiplier  # Very conservative
        assert params.min_confidence_adjustment > 0  # Higher confidence required

    def test_unknown_is_conservative(self):
        """Test unknown regime is conservative."""
        params = REGIME_PARAMS[MarketRegime.UNKNOWN]
        assert params.kelly_multiplier <= 0.5  # Conservative
        assert params.max_position_fraction <= 0.5  # Conservative


class TestRegimeAnalysis:
    """Tests for RegimeAnalysis dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        analysis = RegimeAnalysis(
            asset="BTC",
            regime=MarketRegime.TRENDING,
            params=REGIME_PARAMS[MarketRegime.TRENDING],
            confidence=0.85,
            ma_spread_pct=0.025,
            volatility_ratio=1.1,
            price_range_pct=0.02,
            timestamp=datetime.now(timezone.utc),
            candles_used=60,
            source="full",
        )

        result = analysis.to_dict()

        assert result["asset"] == "BTC"
        assert result["regime"] == "trending"
        assert result["confidence"] == 0.85
        assert result["signals"]["ma_spread_pct"] == 0.025
        assert result["candles_used"] == 60

    def test_is_valid_with_sufficient_data(self):
        """Test is_valid returns True with sufficient data."""
        analysis = RegimeAnalysis(
            asset="BTC",
            regime=MarketRegime.TRENDING,
            params=REGIME_PARAMS[MarketRegime.TRENDING],
            confidence=0.8,
            ma_spread_pct=0.03,
            volatility_ratio=1.0,
            price_range_pct=0.02,
            timestamp=datetime.now(timezone.utc),
            candles_used=60,  # More than minimum
            source="full",
        )
        assert analysis.is_valid is True

    def test_is_valid_with_insufficient_data(self):
        """Test is_valid returns False with insufficient data."""
        analysis = RegimeAnalysis(
            asset="BTC",
            regime=MarketRegime.UNKNOWN,
            params=REGIME_PARAMS[MarketRegime.UNKNOWN],
            confidence=0.0,
            ma_spread_pct=None,
            volatility_ratio=None,
            price_range_pct=None,
            timestamp=datetime.now(timezone.utc),
            candles_used=10,  # Less than minimum
            source="fallback",
        )
        assert analysis.is_valid is False


class TestRegimeDetector:
    """Tests for RegimeDetector class."""

    @pytest.fixture
    def detector(self):
        """Create detector without database."""
        return RegimeDetector(db=None)

    def test_classify_trending_strong_ma_spread(self, detector):
        """Test trending detection with strong MA spread."""
        regime, confidence = detector._classify_regime(
            ma_spread_pct=0.03,  # 3% spread, above threshold
            volatility_ratio=1.0,  # Normal vol
            price_range_pct=0.025,
        )
        assert regime == MarketRegime.TRENDING
        assert confidence > 0.5

    def test_classify_ranging_converged_mas(self, detector):
        """Test ranging detection with converged MAs."""
        regime, confidence = detector._classify_regime(
            ma_spread_pct=0.005,  # 0.5% spread, below threshold
            volatility_ratio=0.5,  # Low vol
            price_range_pct=0.008,  # Tight range
        )
        assert regime == MarketRegime.RANGING
        assert confidence > 0.3

    def test_classify_volatile_high_vol(self, detector):
        """Test volatile detection with high volatility."""
        regime, confidence = detector._classify_regime(
            ma_spread_pct=0.01,
            volatility_ratio=2.0,  # 2x historical vol
            price_range_pct=0.04,
        )
        assert regime == MarketRegime.VOLATILE
        assert confidence > 0.5

    def test_classify_very_high_vol_override(self, detector):
        """Test very high volatility overrides other signals."""
        # Even with trending MA spread, very high vol wins
        regime, confidence = detector._classify_regime(
            ma_spread_pct=0.04,  # Strong trend signal
            volatility_ratio=2.5,  # Very high vol
            price_range_pct=0.02,
        )
        assert regime == MarketRegime.VOLATILE
        assert confidence >= 0.9

    def test_classify_unknown_insufficient_signals(self, detector):
        """Test unknown when signals are unclear."""
        regime, confidence = detector._classify_regime(
            ma_spread_pct=None,
            volatility_ratio=None,
            price_range_pct=None,
        )
        assert regime == MarketRegime.UNKNOWN

    def test_calculate_ma(self, detector):
        """Test moving average calculation."""
        candles = [
            {"close": 100.0},
            {"close": 102.0},
            {"close": 104.0},
            {"close": 106.0},
            {"close": 108.0},
        ]
        ma = detector._calculate_ma(candles, period=5)
        assert ma == 104.0  # Average of 100-108

    def test_calculate_ma_insufficient_data(self, detector):
        """Test MA returns None with insufficient data."""
        candles = [{"close": 100.0}, {"close": 102.0}]
        ma = detector._calculate_ma(candles, period=5)
        assert ma is None

    def test_calculate_volatility(self, detector):
        """Test volatility calculation."""
        candles = [
            {"high": 101.0, "low": 99.0, "close": 100.0},
            {"high": 103.0, "low": 100.0, "close": 102.0},
            {"high": 104.0, "low": 101.0, "close": 103.0},
            {"high": 105.0, "low": 102.0, "close": 104.0},
            {"high": 106.0, "low": 103.0, "close": 105.0},
        ]
        vol = detector._calculate_volatility(candles, lookback=4)
        assert vol is not None
        assert vol > 0

    def test_calculate_price_range(self, detector):
        """Test price range calculation."""
        candles = [
            {"high": 105.0, "low": 95.0},
            {"high": 107.0, "low": 97.0},
            {"high": 110.0, "low": 98.0},
        ]
        price_range = detector._calculate_price_range(candles, lookback=3)
        assert price_range == 15.0  # 110 - 95

    def test_cache_behavior(self, detector):
        """Test caching of regime analysis."""
        analysis = RegimeAnalysis(
            asset="BTC",
            regime=MarketRegime.TRENDING,
            params=REGIME_PARAMS[MarketRegime.TRENDING],
            confidence=0.8,
            ma_spread_pct=0.03,
            volatility_ratio=1.0,
            price_range_pct=0.02,
            timestamp=datetime.now(timezone.utc),
            candles_used=60,
            source="full",
        )

        detector._cache_result("BTC", analysis)
        cached = detector._get_cached("BTC")

        assert cached is not None
        assert cached.regime == MarketRegime.TRENDING

    def test_cache_expiry(self, detector):
        """Test cache expires after TTL."""
        analysis = RegimeAnalysis(
            asset="BTC",
            regime=MarketRegime.TRENDING,
            params=REGIME_PARAMS[MarketRegime.TRENDING],
            confidence=0.8,
            ma_spread_pct=0.03,
            volatility_ratio=1.0,
            price_range_pct=0.02,
            timestamp=datetime.now(timezone.utc),
            candles_used=60,
            source="full",
        )

        # Cache with old timestamp
        old_time = datetime.now(timezone.utc) - timedelta(seconds=120)
        detector._cache["BTC"] = (analysis, old_time)

        cached = detector._get_cached("BTC")
        assert cached is None  # Should be expired

    def test_clear_cache(self, detector):
        """Test cache clearing."""
        analysis = RegimeAnalysis(
            asset="BTC",
            regime=MarketRegime.TRENDING,
            params=REGIME_PARAMS[MarketRegime.TRENDING],
            confidence=0.8,
            ma_spread_pct=0.03,
            volatility_ratio=1.0,
            price_range_pct=0.02,
            timestamp=datetime.now(timezone.utc),
            candles_used=60,
            source="full",
        )

        detector._cache_result("BTC", analysis)
        detector._cache_result("ETH", analysis)

        detector.clear_cache("BTC")
        assert "BTC" not in detector._cache
        assert "ETH" in detector._cache

        detector.clear_cache()
        assert len(detector._cache) == 0


class TestRegimeDetectorWithDatabase:
    """Tests for RegimeDetector with mocked database."""

    @pytest.mark.asyncio
    async def test_detect_regime_trending(self):
        """Test regime detection returns trending."""
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        # Create strong trending market data
        # MA20/MA50 spread needs to be > 2% (REGIME_TREND_THRESHOLD)
        # We need price to rise significantly from start to end
        now = datetime.now(timezone.utc)
        candles = []
        for i in range(60):
            ts = now - timedelta(minutes=60-i)
            # Strong uptrend: +200 per candle = 12000 total = 12% rise
            base_price = 100000 + i * 200
            candles.append({
                "ts": ts,
                "mid": base_price,
                "high": base_price + 300,
                "low": base_price - 100,
                "close": base_price + 100,
                "atr14": 500,
            })

        mock_conn.fetch.return_value = candles

        detector = RegimeDetector(db=mock_pool)
        analysis = await detector.detect_regime("BTC")

        assert analysis.asset == "BTC"
        # With 12% trend, MA20 should be significantly above MA50
        assert analysis.ma_spread_pct is not None
        assert analysis.ma_spread_pct > REGIME_TREND_THRESHOLD
        assert analysis.regime == MarketRegime.TRENDING
        assert analysis.candles_used == 60

    @pytest.mark.asyncio
    async def test_detect_regime_insufficient_data(self):
        """Test regime detection with insufficient data."""
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.fetch.return_value = []  # No data

        detector = RegimeDetector(db=mock_pool)
        analysis = await detector.detect_regime("BTC")

        assert analysis.regime == MarketRegime.UNKNOWN
        assert analysis.is_valid is False

    @pytest.mark.asyncio
    async def test_detect_regime_database_error(self):
        """Test regime detection handles database errors."""
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.fetch.side_effect = Exception("DB error")

        detector = RegimeDetector(db=mock_pool)
        analysis = await detector.detect_regime("BTC")

        assert analysis.regime == MarketRegime.UNKNOWN


class TestAdjustmentFunctions:
    """Tests for regime adjustment functions."""

    def test_get_regime_adjusted_kelly_trending(self):
        """Test Kelly adjustment for trending market."""
        adjusted = get_regime_adjusted_kelly(0.25, MarketRegime.TRENDING)
        assert adjusted == 0.25  # Full Kelly in trending

    def test_get_regime_adjusted_kelly_volatile(self):
        """Test Kelly adjustment for volatile market."""
        adjusted = get_regime_adjusted_kelly(0.25, MarketRegime.VOLATILE)
        assert adjusted == 0.125  # Half Kelly in volatile

    def test_get_regime_adjusted_kelly_ranging(self):
        """Test Kelly adjustment for ranging market."""
        adjusted = get_regime_adjusted_kelly(0.25, MarketRegime.RANGING)
        assert adjusted == 0.25 * 0.75  # 75% Kelly in ranging

    def test_get_regime_adjusted_stop_trending(self):
        """Test stop adjustment for trending market."""
        adjusted = get_regime_adjusted_stop(2.0, MarketRegime.TRENDING)
        assert adjusted > 2.0  # Wider stops in trending

    def test_get_regime_adjusted_stop_ranging(self):
        """Test stop adjustment for ranging market."""
        adjusted = get_regime_adjusted_stop(2.0, MarketRegime.RANGING)
        assert adjusted < 2.0  # Tighter stops in ranging

    def test_get_regime_adjusted_stop_volatile(self):
        """Test stop adjustment for volatile market."""
        adjusted = get_regime_adjusted_stop(2.0, MarketRegime.VOLATILE)
        assert adjusted == 3.0  # 1.5x wider stops in volatile

    def test_get_regime_adjusted_confidence_trending(self):
        """Test confidence adjustment for trending market."""
        adjusted = get_regime_adjusted_confidence(0.55, MarketRegime.TRENDING)
        assert adjusted == 0.55  # No adjustment in trending

    def test_get_regime_adjusted_confidence_volatile(self):
        """Test confidence adjustment for volatile market."""
        adjusted = get_regime_adjusted_confidence(0.55, MarketRegime.VOLATILE)
        assert adjusted == 0.65  # +10% in volatile

    def test_get_regime_adjusted_confidence_caps_at_95(self):
        """Test confidence adjustment caps at 95%."""
        adjusted = get_regime_adjusted_confidence(0.90, MarketRegime.VOLATILE)
        assert adjusted == 0.95  # Capped at 95%


class TestGlobalDetector:
    """Tests for global detector instance."""

    def test_get_regime_detector_singleton(self):
        """Test get_regime_detector returns same instance."""
        import app.regime as module
        module._detector = None

        d1 = get_regime_detector()
        d2 = get_regime_detector()
        assert d1 is d2

    def test_get_regime_detector_sets_db(self):
        """Test get_regime_detector can set database."""
        import app.regime as module
        module._detector = None

        mock_pool = MagicMock()
        detector = get_regime_detector(mock_pool)
        assert detector.db is mock_pool


class TestQuantAcceptance:
    """Quantitative acceptance tests for regime detection."""

    def test_trending_requires_significant_ma_spread(self):
        """Ensure trending requires MA spread > threshold."""
        detector = RegimeDetector(db=None)

        # Just below threshold should not be trending
        regime, _ = detector._classify_regime(
            ma_spread_pct=REGIME_TREND_THRESHOLD * 0.9,
            volatility_ratio=1.0,
            price_range_pct=0.02,
        )
        assert regime != MarketRegime.TRENDING

    def test_volatile_requires_high_volatility(self):
        """Ensure volatile requires volatility > threshold."""
        detector = RegimeDetector(db=None)

        # Below threshold should not be volatile
        regime, _ = detector._classify_regime(
            ma_spread_pct=0.01,
            volatility_ratio=REGIME_VOLATILITY_HIGH_MULT * 0.9,
            price_range_pct=0.02,
        )
        assert regime != MarketRegime.VOLATILE

    def test_regime_params_are_sensible(self):
        """Ensure regime parameters make economic sense."""
        # In volatile markets:
        # - Stops should be wider (avoid getting stopped out)
        # - Position sizes should be smaller (reduce risk)
        volatile = REGIME_PARAMS[MarketRegime.VOLATILE]
        assert volatile.stop_multiplier >= 1.0
        assert volatile.kelly_multiplier <= 0.5
        assert volatile.max_position_fraction <= 0.5

        # In trending markets:
        # - Full position allowed (capture the trend)
        trending = REGIME_PARAMS[MarketRegime.TRENDING]
        assert trending.max_position_fraction == 1.0
        assert trending.kelly_multiplier == 1.0

    def test_confidence_bounds(self):
        """Ensure confidence is always 0-1."""
        detector = RegimeDetector(db=None)

        test_cases = [
            (0.05, 2.0, 0.04),
            (0.0, 1.0, 0.01),
            (-0.02, 0.5, 0.005),
            (0.03, 1.5, 0.025),
        ]

        for ma_spread, vol_ratio, price_range in test_cases:
            _, confidence = detector._classify_regime(
                ma_spread_pct=ma_spread,
                volatility_ratio=vol_ratio,
                price_range_pct=price_range,
            )
            assert 0.0 <= confidence <= 1.0
