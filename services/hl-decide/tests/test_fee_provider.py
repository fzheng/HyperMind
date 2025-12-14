"""
Tests for Dynamic Fee Provider

@module tests.test_fee_provider
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from app.fee_provider import (
    FeeProvider,
    CachedFees,
    get_fee_provider,
    init_fee_provider,
    FEE_CACHE_TTL_SECONDS,
)
from app.exchanges import FeeConfig


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def fee_provider():
    """Create fresh fee provider for testing."""
    return FeeProvider(testnet=True)


@pytest.fixture
def mock_fee_config():
    """Create mock fee config."""
    return FeeConfig(maker_fee_bps=2.5, taker_fee_bps=5.0)


# =============================================================================
# CachedFees Tests
# =============================================================================


class TestCachedFees:
    """Tests for CachedFees dataclass."""

    def test_is_expired_fresh(self, mock_fee_config):
        """Fresh cache is not expired."""
        cached = CachedFees(
            config=mock_fee_config,
            fetched_at=datetime.now(timezone.utc),
            source="api",
        )
        assert cached.is_expired is False

    def test_is_expired_old(self, mock_fee_config):
        """Old cache is expired."""
        cached = CachedFees(
            config=mock_fee_config,
            fetched_at=datetime.now(timezone.utc) - timedelta(seconds=FEE_CACHE_TTL_SECONDS + 60),
            source="api",
        )
        assert cached.is_expired is True

    def test_age_seconds(self, mock_fee_config):
        """Age is calculated correctly."""
        past = datetime.now(timezone.utc) - timedelta(seconds=120)
        cached = CachedFees(
            config=mock_fee_config,
            fetched_at=past,
            source="api",
        )
        # Allow some tolerance for execution time
        assert 118 <= cached.age_seconds <= 125


# =============================================================================
# FeeProvider Tests
# =============================================================================


class TestFeeProvider:
    """Tests for FeeProvider class."""

    @pytest.mark.asyncio
    async def test_get_fees_hyperliquid_static(self, fee_provider):
        """Hyperliquid returns static fees (no API)."""
        config, source = await fee_provider.get_fees("hyperliquid")
        assert source == "static"
        assert config.maker_fee_bps == 2.5
        assert config.taker_fee_bps == 5.0

    @pytest.mark.asyncio
    async def test_get_fees_bybit_static(self, fee_provider):
        """Bybit returns static fees when API unavailable."""
        config, source = await fee_provider.get_fees("bybit")
        # Currently returns static since API method returns None
        assert source == "static"
        assert config.taker_fee_bps == 6.0  # Bybit VIP0

    @pytest.mark.asyncio
    async def test_get_fees_aster_static(self, fee_provider):
        """Aster returns static fees."""
        config, source = await fee_provider.get_fees("aster")
        assert source == "static"

    @pytest.mark.asyncio
    async def test_get_fees_unknown_exchange(self, fee_provider):
        """Unknown exchange returns default fees."""
        config, source = await fee_provider.get_fees("unknown_exchange")
        assert source == "static"
        assert config.maker_fee_bps == 2.5  # Default

    @pytest.mark.asyncio
    async def test_caching_works(self, fee_provider):
        """Fees are cached after first fetch."""
        # First fetch
        config1, source1 = await fee_provider.get_fees("hyperliquid")
        assert source1 == "static"

        # Second fetch should be cached
        config2, source2 = await fee_provider.get_fees("hyperliquid")
        assert source2 == "cached"
        assert config1 == config2

    @pytest.mark.asyncio
    async def test_force_refresh_bypasses_cache(self, fee_provider):
        """Force refresh ignores cache."""
        # First fetch
        await fee_provider.get_fees("hyperliquid")

        # Force refresh
        config, source = await fee_provider.get_fees("hyperliquid", force_refresh=True)
        assert source == "static"  # Fresh fetch

    @pytest.mark.asyncio
    async def test_get_fees_bps(self, fee_provider):
        """get_fees_bps returns round-trip cost."""
        bps = await fee_provider.get_fees_bps("hyperliquid")
        # HL: maker=2.5, taker=5.0, round-trip = 5+5 = 10 bps
        assert bps == 10.0

    @pytest.mark.asyncio
    async def test_get_fees_bps_bybit(self, fee_provider):
        """Bybit round-trip fees are higher."""
        bps = await fee_provider.get_fees_bps("bybit")
        # Bybit VIP0: maker=10, taker=6, round-trip = 6+6 = 12 bps
        assert bps == 12.0

    def test_clear_cache(self, fee_provider):
        """clear_cache removes all cached data."""
        # Add to cache directly
        fee_provider._cache["test"] = CachedFees(
            config=FeeConfig(),
            fetched_at=datetime.now(timezone.utc),
            source="test",
        )
        assert len(fee_provider._cache) > 0

        fee_provider.clear_cache()
        assert len(fee_provider._cache) == 0

    @pytest.mark.asyncio
    async def test_get_cache_status(self, fee_provider):
        """get_cache_status returns cache info."""
        await fee_provider.get_fees("hyperliquid")

        status = fee_provider.get_cache_status()
        assert "hyperliquid" in status
        assert status["hyperliquid"]["source"] == "static"
        assert status["hyperliquid"]["is_expired"] is False
        assert "maker_bps" in status["hyperliquid"]
        assert "taker_bps" in status["hyperliquid"]
        assert "round_trip_bps" in status["hyperliquid"]

    @pytest.mark.asyncio
    async def test_cache_key_case_insensitive(self, fee_provider):
        """Cache keys are case-insensitive."""
        await fee_provider.get_fees("Hyperliquid")
        _, source = await fee_provider.get_fees("hyperliquid")
        assert source == "cached"

    @pytest.mark.asyncio
    async def test_expired_cache_refetches(self, fee_provider):
        """Expired cache triggers refetch."""
        # Add expired cache entry
        fee_provider._cache["hyperliquid"] = CachedFees(
            config=FeeConfig(),
            fetched_at=datetime.now(timezone.utc) - timedelta(seconds=FEE_CACHE_TTL_SECONDS + 60),
            source="old",
        )

        # Fetch should detect expired and refetch
        _, source = await fee_provider.get_fees("hyperliquid")
        assert source == "static"  # Fresh fetch, not "old"


class TestFeeProviderSingleton:
    """Tests for global singleton."""

    def test_get_fee_provider_singleton(self):
        """get_fee_provider returns same instance."""
        provider1 = get_fee_provider()
        provider2 = get_fee_provider()
        assert provider1 is provider2

    def test_init_fee_provider(self):
        """init_fee_provider creates new instance."""
        # Reset singleton for test
        import app.fee_provider as module
        module._fee_provider = None

        provider = init_fee_provider(testnet=True)
        assert provider.testnet is True

        # get_fee_provider should now return this instance
        assert get_fee_provider() is provider


class TestFeeProviderIntegration:
    """Integration-style tests."""

    @pytest.mark.asyncio
    async def test_per_signal_refresh_pattern(self, fee_provider):
        """Test pattern for per-signal fee refresh."""
        # First signal - fresh fetch
        fees1 = await fee_provider.get_fees_bps("hyperliquid")

        # Second signal (within TTL) - cached
        fees2 = await fee_provider.get_fees_bps("hyperliquid")
        assert fees1 == fees2

        # Force refresh for important signal
        fees3 = await fee_provider.get_fees_bps("hyperliquid", force_refresh=True)
        assert fees3 == fees1  # Same value, but freshly fetched

    @pytest.mark.asyncio
    async def test_multi_exchange_fees(self, fee_provider):
        """Get fees for multiple exchanges."""
        hl_fees = await fee_provider.get_fees_bps("hyperliquid")
        bybit_fees = await fee_provider.get_fees_bps("bybit")
        aster_fees = await fee_provider.get_fees_bps("aster")

        # All should have values
        assert hl_fees > 0
        assert bybit_fees > 0
        assert aster_fees > 0

        # Bybit is typically higher
        assert bybit_fees >= hl_fees

    @pytest.mark.asyncio
    async def test_close_client(self, fee_provider):
        """Can close HTTP client."""
        await fee_provider.close()
        assert fee_provider._client is None
