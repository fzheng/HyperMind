"""
Tests for Hold Time Estimator

Tests dynamic hold time estimation from historical episode data.

@module tests.test_hold_time_estimator
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from app.hold_time_estimator import (
    HoldTimeEstimator,
    HoldTimeEstimate,
    EstimateSource,
    CachedHoldTime,
    get_hold_time_estimator,
    init_hold_time_estimator,
    DEFAULT_HOLD_HOURS,
    MIN_EPISODES_FOR_ESTIMATE,
    REGIME_HOLD_TIME_MULTIPLIERS,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def hold_time_estimator():
    """Create a fresh hold time estimator."""
    return HoldTimeEstimator()


@pytest.fixture
def mock_db():
    """Create mock database pool."""
    return AsyncMock()


@pytest.fixture
def sample_episode_data():
    """Create sample episode data with hold times."""
    # Generate 20 episodes with varying hold times
    return [
        {"hold_secs": 3600 * h, "entry_ts": datetime.now(timezone.utc), "r_clamped": 0.5}
        for h in [12, 18, 24, 6, 30, 20, 15, 22, 28, 10,
                  14, 26, 16, 8, 32, 19, 21, 25, 17, 23]
    ]


# =============================================================================
# HoldTimeEstimate Tests
# =============================================================================


class TestHoldTimeEstimate:
    """Tests for HoldTimeEstimate dataclass."""

    def test_estimate_with_historical_source(self):
        """Historical estimate has correct source."""
        estimate = HoldTimeEstimate(
            hours=18.5,
            source=EstimateSource.HISTORICAL,
            episode_count=50,
            median_hours=18.5,
            std_hours=5.2,
            asset="BTC",
        )
        assert estimate.hours == 18.5
        assert estimate.source == EstimateSource.HISTORICAL
        assert estimate.episode_count == 50

    def test_estimate_with_regime_adjustment(self):
        """Regime-adjusted estimate includes regime."""
        estimate = HoldTimeEstimate(
            hours=23.0,
            source=EstimateSource.REGIME_ADJUSTED,
            episode_count=50,
            median_hours=18.5,
            std_hours=5.2,
            asset="BTC",
            regime="TRENDING",
        )
        assert estimate.regime == "TRENDING"
        assert estimate.source == EstimateSource.REGIME_ADJUSTED

    def test_fallback_estimate(self):
        """Fallback estimate uses defaults."""
        estimate = HoldTimeEstimate(
            hours=DEFAULT_HOLD_HOURS,
            source=EstimateSource.FALLBACK,
            episode_count=0,
            asset="BTC",
        )
        assert estimate.hours == DEFAULT_HOLD_HOURS
        assert estimate.source == EstimateSource.FALLBACK


# =============================================================================
# CachedHoldTime Tests
# =============================================================================


class TestCachedHoldTime:
    """Tests for cache expiration."""

    def test_fresh_cache_not_expired(self):
        """Fresh cache entry is not expired."""
        estimate = HoldTimeEstimate(
            hours=20.0,
            source=EstimateSource.HISTORICAL,
            episode_count=30,
            asset="BTC",
        )
        cached = CachedHoldTime(
            estimate=estimate,
            fetched_at=datetime.now(timezone.utc),
        )
        assert not cached.is_expired

    def test_old_cache_is_expired(self):
        """Old cache entry is expired."""
        estimate = HoldTimeEstimate(
            hours=20.0,
            source=EstimateSource.HISTORICAL,
            episode_count=30,
            asset="BTC",
        )
        cached = CachedHoldTime(
            estimate=estimate,
            fetched_at=datetime.now(timezone.utc) - timedelta(minutes=10),
        )
        assert cached.is_expired


# =============================================================================
# HoldTimeEstimator Tests
# =============================================================================


class TestHoldTimeEstimator:
    """Tests for HoldTimeEstimator class."""

    @pytest.mark.asyncio
    async def test_get_hold_time_with_data(self, hold_time_estimator, mock_db, sample_episode_data):
        """Returns historical estimate when data available."""
        mock_db.fetch = AsyncMock(return_value=sample_episode_data)

        estimate = await hold_time_estimator.get_hold_time("BTC", mock_db)

        assert estimate.source == EstimateSource.HISTORICAL
        assert estimate.episode_count == 20
        assert estimate.median_hours is not None
        assert estimate.hours > 0

    @pytest.mark.asyncio
    async def test_get_hold_time_insufficient_data(self, hold_time_estimator, mock_db):
        """Returns fallback when insufficient data."""
        # Less than MIN_EPISODES_FOR_ESTIMATE
        mock_db.fetch = AsyncMock(return_value=[
            {"hold_secs": 3600, "entry_ts": datetime.now(timezone.utc), "r_clamped": 0.5}
            for _ in range(5)
        ])

        estimate = await hold_time_estimator.get_hold_time("BTC", mock_db)

        assert estimate.source == EstimateSource.FALLBACK
        assert estimate.hours == DEFAULT_HOLD_HOURS
        assert estimate.episode_count == 5

    @pytest.mark.asyncio
    async def test_get_hold_time_with_regime_trending(self, hold_time_estimator, mock_db, sample_episode_data):
        """Trending regime increases hold time."""
        mock_db.fetch = AsyncMock(return_value=sample_episode_data)

        estimate_base = await hold_time_estimator.get_hold_time("BTC", mock_db)
        hold_time_estimator.clear_cache()
        estimate_trending = await hold_time_estimator.get_hold_time("BTC", mock_db, regime="TRENDING")

        assert estimate_trending.source == EstimateSource.REGIME_ADJUSTED
        assert estimate_trending.regime == "TRENDING"
        # Trending should increase hold time by 25%
        assert estimate_trending.hours > estimate_base.hours

    @pytest.mark.asyncio
    async def test_get_hold_time_with_regime_volatile(self, hold_time_estimator, mock_db, sample_episode_data):
        """Volatile regime decreases hold time."""
        mock_db.fetch = AsyncMock(return_value=sample_episode_data)

        estimate_base = await hold_time_estimator.get_hold_time("BTC", mock_db)
        hold_time_estimator.clear_cache()
        estimate_volatile = await hold_time_estimator.get_hold_time("BTC", mock_db, regime="VOLATILE")

        assert estimate_volatile.source == EstimateSource.REGIME_ADJUSTED
        assert estimate_volatile.regime == "VOLATILE"
        # Volatile should decrease hold time by 25%
        assert estimate_volatile.hours < estimate_base.hours

    @pytest.mark.asyncio
    async def test_caching_works(self, hold_time_estimator, mock_db, sample_episode_data):
        """Estimates are cached."""
        mock_db.fetch = AsyncMock(return_value=sample_episode_data)

        # First call
        estimate1 = await hold_time_estimator.get_hold_time("BTC", mock_db)

        # Second call should use cache
        estimate2 = await hold_time_estimator.get_hold_time("BTC", mock_db)

        # Only one DB call
        assert mock_db.fetch.call_count == 1
        assert estimate1.hours == estimate2.hours

    @pytest.mark.asyncio
    async def test_force_refresh_bypasses_cache(self, hold_time_estimator, mock_db, sample_episode_data):
        """Force refresh bypasses cache."""
        mock_db.fetch = AsyncMock(return_value=sample_episode_data)

        # First call
        await hold_time_estimator.get_hold_time("BTC", mock_db)

        # Force refresh
        await hold_time_estimator.get_hold_time("BTC", mock_db, force_refresh=True)

        # Two DB calls
        assert mock_db.fetch.call_count == 2

    def test_get_hold_time_sync_from_cache(self, hold_time_estimator):
        """Sync lookup uses cached data."""
        # Pre-populate cache
        estimate = HoldTimeEstimate(
            hours=18.0,
            source=EstimateSource.HISTORICAL,
            episode_count=50,
            median_hours=18.0,
            std_hours=4.0,
            asset="BTC",
        )
        hold_time_estimator._cache["BTC:none"] = CachedHoldTime(
            estimate=estimate,
            fetched_at=datetime.now(timezone.utc),
        )

        result = hold_time_estimator.get_hold_time_sync("BTC")

        assert result.hours == 18.0
        assert result.source == EstimateSource.HISTORICAL

    def test_get_hold_time_sync_fallback(self, hold_time_estimator):
        """Sync lookup returns fallback when no cache."""
        result = hold_time_estimator.get_hold_time_sync("ETH")

        assert result.hours == DEFAULT_HOLD_HOURS
        assert result.source == EstimateSource.FALLBACK

    def test_get_hold_time_sync_with_regime_from_base_cache(self, hold_time_estimator):
        """Sync lookup with regime applies multiplier to base cache."""
        # Pre-populate base cache (no regime)
        estimate = HoldTimeEstimate(
            hours=20.0,
            source=EstimateSource.HISTORICAL,
            episode_count=50,
            median_hours=20.0,
            std_hours=4.0,
            asset="BTC",
        )
        hold_time_estimator._cache["BTC:none"] = CachedHoldTime(
            estimate=estimate,
            fetched_at=datetime.now(timezone.utc),
        )

        result = hold_time_estimator.get_hold_time_sync("BTC", regime="TRENDING")

        # Should apply 1.25x multiplier
        assert result.hours == pytest.approx(25.0, rel=0.01)
        assert result.source == EstimateSource.REGIME_ADJUSTED

    def test_clear_cache(self, hold_time_estimator):
        """Can clear cache."""
        hold_time_estimator._cache["test"] = CachedHoldTime(
            estimate=HoldTimeEstimate(
                hours=20.0,
                source=EstimateSource.HISTORICAL,
                episode_count=50,
                asset="BTC",
            ),
            fetched_at=datetime.now(timezone.utc),
        )

        hold_time_estimator.clear_cache()

        assert len(hold_time_estimator._cache) == 0

    def test_get_cache_status(self, hold_time_estimator):
        """Can get cache status."""
        estimate = HoldTimeEstimate(
            hours=18.0,
            source=EstimateSource.HISTORICAL,
            episode_count=50,
            asset="BTC",
        )
        hold_time_estimator._cache["BTC:none"] = CachedHoldTime(
            estimate=estimate,
            fetched_at=datetime.now(timezone.utc),
        )

        status = hold_time_estimator.get_cache_status()

        assert "BTC:none" in status
        assert status["BTC:none"]["hours"] == 18.0
        assert status["BTC:none"]["source"] == "historical"


# =============================================================================
# Regime Adjustment Tests
# =============================================================================


class TestRegimeAdjustments:
    """Tests for regime-based hold time adjustments."""

    def test_trending_multiplier(self):
        """Trending regime has positive multiplier."""
        assert REGIME_HOLD_TIME_MULTIPLIERS["TRENDING"] > 1.0

    def test_volatile_multiplier(self):
        """Volatile regime has negative multiplier."""
        assert REGIME_HOLD_TIME_MULTIPLIERS["VOLATILE"] < 1.0

    def test_ranging_baseline(self):
        """Ranging regime is baseline."""
        assert REGIME_HOLD_TIME_MULTIPLIERS["RANGING"] == 1.0

    def test_unknown_is_neutral(self):
        """Unknown regime is neutral."""
        assert REGIME_HOLD_TIME_MULTIPLIERS["UNKNOWN"] == 1.0


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingleton:
    """Tests for global singleton."""

    def test_get_hold_time_estimator_singleton(self):
        """get_hold_time_estimator returns same instance."""
        est1 = get_hold_time_estimator()
        est2 = get_hold_time_estimator()
        assert est1 is est2

    @pytest.mark.asyncio
    async def test_init_hold_time_estimator(self, mock_db):
        """init_hold_time_estimator pre-fetches estimates."""
        mock_db.fetch = AsyncMock(return_value=[])

        estimator = await init_hold_time_estimator(mock_db)

        # Should fetch for BTC and ETH
        assert mock_db.fetch.call_count == 2
        assert estimator is not None


# =============================================================================
# Integration Tests
# =============================================================================


class TestHoldTimeIntegration:
    """Integration-style tests."""

    @pytest.mark.asyncio
    async def test_median_calculation(self, hold_time_estimator, mock_db):
        """Median is calculated correctly."""
        # 10 episodes: 1h, 2h, 3h, 4h, 5h, 6h, 7h, 8h, 9h, 10h
        # Median = (5h + 6h) / 2 = 5.5h
        data = [
            {"hold_secs": 3600 * h, "entry_ts": datetime.now(timezone.utc), "r_clamped": 0.5}
            for h in range(1, 11)
        ]
        mock_db.fetch = AsyncMock(return_value=data)

        estimate = await hold_time_estimator.get_hold_time("BTC", mock_db)

        assert estimate.median_hours == pytest.approx(5.5, rel=0.01)
        assert estimate.hours == pytest.approx(5.5, rel=0.01)

    @pytest.mark.asyncio
    async def test_std_calculation(self, hold_time_estimator, mock_db):
        """Standard deviation is calculated."""
        data = [
            {"hold_secs": 3600 * h, "entry_ts": datetime.now(timezone.utc), "r_clamped": 0.5}
            for h in [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]  # All same
        ]
        mock_db.fetch = AsyncMock(return_value=data)

        estimate = await hold_time_estimator.get_hold_time("BTC", mock_db)

        # All same value = 0 std
        assert estimate.std_hours == pytest.approx(0.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_different_assets_different_estimates(self, hold_time_estimator, mock_db):
        """Different assets can have different estimates."""
        # BTC: 24h average
        btc_data = [
            {"hold_secs": 3600 * 24, "entry_ts": datetime.now(timezone.utc), "r_clamped": 0.5}
            for _ in range(15)
        ]
        # ETH: 12h average
        eth_data = [
            {"hold_secs": 3600 * 12, "entry_ts": datetime.now(timezone.utc), "r_clamped": 0.5}
            for _ in range(15)
        ]

        mock_db.fetch = AsyncMock(side_effect=[btc_data, eth_data])

        btc_estimate = await hold_time_estimator.get_hold_time("BTC", mock_db)
        hold_time_estimator.clear_cache()  # Clear to force new fetch
        eth_estimate = await hold_time_estimator.get_hold_time("ETH", mock_db)

        assert btc_estimate.hours == pytest.approx(24.0, rel=0.01)
        assert eth_estimate.hours == pytest.approx(12.0, rel=0.01)

    def test_funding_integration_pattern(self, hold_time_estimator):
        """Test pattern for funding calculation with hold time."""
        # Pre-populate cache
        estimate = HoldTimeEstimate(
            hours=18.0,
            source=EstimateSource.HISTORICAL,
            episode_count=50,
            median_hours=18.0,
            asset="BTC",
        )
        hold_time_estimator._cache["BTC:none"] = CachedHoldTime(
            estimate=estimate,
            fetched_at=datetime.now(timezone.utc),
        )

        # Get hold time
        hold_estimate = hold_time_estimator.get_hold_time_sync("BTC")

        # Use in funding calculation (simulated)
        funding_rate_bps_per_8h = 1.0  # 1 bps per 8h
        intervals = hold_estimate.hours / 8
        funding_cost_bps = funding_rate_bps_per_8h * intervals

        # 18h / 8h = 2.25 intervals × 1 bps = 2.25 bps
        assert funding_cost_bps == pytest.approx(2.25, rel=0.01)
