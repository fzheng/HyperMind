"""
Tests for Funding Rate Provider

@module tests.test_funding_provider
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from app.funding_provider import (
    FundingProvider,
    FundingData,
    CachedFunding,
    get_funding_provider,
    init_funding_provider,
    FUNDING_CACHE_TTL_SECONDS,
    FUNDING_INTERVAL_HOURS,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def funding_provider():
    """Create fresh funding provider for testing."""
    return FundingProvider(testnet=True)


@pytest.fixture
def mock_funding_data():
    """Create mock funding data."""
    return FundingData(
        asset="BTC",
        exchange="hyperliquid",
        rate_pct=0.01,  # 0.01%
        rate_bps=1.0,  # 1 bps per 8h
        interval_hours=8,
        source="api",
    )


# =============================================================================
# FundingData Tests
# =============================================================================


class TestFundingData:
    """Tests for FundingData dataclass."""

    def test_is_expired_fresh(self, mock_funding_data):
        """Fresh data is not expired."""
        assert mock_funding_data.is_expired is False

    def test_is_expired_old(self):
        """Old data is expired."""
        data = FundingData(
            asset="BTC",
            exchange="test",
            rate_pct=0.01,
            rate_bps=1.0,
            interval_hours=8,
            fetched_at=datetime.now(timezone.utc) - timedelta(seconds=FUNDING_CACHE_TTL_SECONDS + 60),
            source="api",
        )
        assert data.is_expired is True

    def test_daily_cost_bps(self):
        """Daily cost calculated correctly."""
        data = FundingData(
            asset="BTC",
            exchange="test",
            rate_pct=0.01,
            rate_bps=1.0,  # 1 bps per 8h
            interval_hours=8,
            source="api",
        )
        # 24h / 8h = 3 intervals per day
        assert data.daily_cost_bps == 3.0

    def test_annual_cost_bps(self):
        """Annual cost calculated correctly."""
        data = FundingData(
            asset="BTC",
            exchange="test",
            rate_pct=0.01,
            rate_bps=1.0,
            interval_hours=8,
            source="api",
        )
        # 3 bps/day * 365 = 1095 bps/year
        assert data.annual_cost_bps == 1095.0

    def test_cost_for_hold_time(self):
        """Cost for hold time calculated correctly."""
        data = FundingData(
            asset="BTC",
            exchange="test",
            rate_pct=0.01,
            rate_bps=1.0,
            interval_hours=8,
            source="api",
        )
        # 24 hours = 3 intervals (long position pays)
        assert data.cost_for_hold_time(24, "long") == 3.0
        # 4 hours = 0.5 intervals
        assert data.cost_for_hold_time(4, "long") == 0.5
        # Default is long
        assert data.cost_for_hold_time(24) == 3.0

    def test_cost_for_hold_time_short(self):
        """Short positions receive funding when rate is positive."""
        data = FundingData(
            asset="BTC",
            exchange="test",
            rate_pct=0.01,
            rate_bps=1.0,  # Positive: longs pay shorts
            interval_hours=8,
            source="api",
        )
        # Shorts receive (negative cost = rebate)
        assert data.cost_for_hold_time(24, "short") == -3.0
        assert data.cost_for_hold_time(4, "short") == -0.5

    def test_cost_negative_funding_rate(self):
        """Negative funding rate: shorts pay, longs receive."""
        data = FundingData(
            asset="BTC",
            exchange="test",
            rate_pct=-0.01,
            rate_bps=-1.0,  # Negative: shorts pay longs
            interval_hours=8,
            source="api",
        )
        # Longs receive (negative cost = rebate)
        assert data.cost_for_hold_time(24, "long") == -3.0
        # Shorts pay (positive cost)
        assert data.cost_for_hold_time(24, "short") == 3.0


class TestCachedFunding:
    """Tests for CachedFunding dataclass."""

    def test_is_expired_fresh(self, mock_funding_data):
        """Fresh cache is not expired."""
        cached = CachedFunding(
            data=mock_funding_data,
            fetched_at=datetime.now(timezone.utc),
        )
        assert cached.is_expired is False

    def test_is_expired_old(self, mock_funding_data):
        """Old cache is expired."""
        cached = CachedFunding(
            data=mock_funding_data,
            fetched_at=datetime.now(timezone.utc) - timedelta(seconds=FUNDING_CACHE_TTL_SECONDS + 60),
        )
        assert cached.is_expired is True


# =============================================================================
# FundingProvider Tests
# =============================================================================


class TestFundingProvider:
    """Tests for FundingProvider class."""

    @pytest.mark.asyncio
    async def test_get_funding_static_fallback(self, funding_provider):
        """Falls back to static rates when API unavailable."""
        # Mock API to return None (simulating failure)
        with patch.object(funding_provider, "_fetch_hyperliquid_funding", new_callable=AsyncMock, return_value=None):
            data = await funding_provider.get_funding("BTC", "hyperliquid")
            assert data.asset == "BTC"
            assert data.exchange == "hyperliquid"
            assert data.source == "static"
            assert data.rate_bps > 0

    @pytest.mark.asyncio
    async def test_caching_works(self, funding_provider):
        """Funding data is cached after first fetch."""
        # Mock API to return None (use static)
        with patch.object(funding_provider, "_fetch_hyperliquid_funding", new_callable=AsyncMock, return_value=None):
            # First fetch
            data1 = await funding_provider.get_funding("BTC", "hyperliquid")

            # Manually check cache
            key = funding_provider._get_cache_key("hyperliquid", "BTC")
            assert key in funding_provider._cache

            # Second fetch should use cache
            data2 = await funding_provider.get_funding("BTC", "hyperliquid")
            assert data1.rate_bps == data2.rate_bps

    @pytest.mark.asyncio
    async def test_force_refresh_bypasses_cache(self, funding_provider):
        """Force refresh ignores cache."""
        with patch.object(funding_provider, "_fetch_hyperliquid_funding", new_callable=AsyncMock, return_value=None):
            # First fetch
            await funding_provider.get_funding("BTC", "hyperliquid")

            # Force refresh
            data = await funding_provider.get_funding("BTC", "hyperliquid", force_refresh=True)
        # Should still work (falls back to static)
        assert data.asset == "BTC"

    @pytest.mark.asyncio
    async def test_get_funding_cost_bps(self, funding_provider):
        """get_funding_cost_bps returns cost for hold time."""
        with patch.object(funding_provider, "_fetch_hyperliquid_funding", new_callable=AsyncMock, return_value=None):
            cost = await funding_provider.get_funding_cost_bps(
                "BTC", "hyperliquid", hold_hours=24
            )
            assert cost > 0

    @pytest.mark.asyncio
    async def test_different_assets(self, funding_provider):
        """Different assets can have different rates."""
        with patch.object(funding_provider, "_fetch_hyperliquid_funding", new_callable=AsyncMock, return_value=None):
            btc_data = await funding_provider.get_funding("BTC", "hyperliquid")
            eth_data = await funding_provider.get_funding("ETH", "hyperliquid")

            # Both should have valid static data
            assert btc_data.rate_bps > 0
            assert eth_data.rate_bps > 0

    @pytest.mark.asyncio
    async def test_different_exchanges(self, funding_provider):
        """Different exchanges have their own rates."""
        with patch.object(funding_provider, "_fetch_hyperliquid_funding", new_callable=AsyncMock, return_value=None):
            with patch.object(funding_provider, "_fetch_bybit_funding", new_callable=AsyncMock, return_value=None):
                hl_data = await funding_provider.get_funding("BTC", "hyperliquid")
                bybit_data = await funding_provider.get_funding("BTC", "bybit")

                # Both should have valid data
                assert hl_data.exchange == "hyperliquid"
                assert bybit_data.exchange == "bybit"

    def test_clear_cache(self, funding_provider):
        """clear_cache removes all cached data."""
        # Add to cache directly
        funding_provider._cache["test"] = CachedFunding(
            data=FundingData(
                asset="BTC",
                exchange="test",
                rate_pct=0.01,
                rate_bps=1.0,
                interval_hours=8,
                source="test",
            ),
            fetched_at=datetime.now(timezone.utc),
        )
        assert len(funding_provider._cache) > 0

        funding_provider.clear_cache()
        assert len(funding_provider._cache) == 0

    @pytest.mark.asyncio
    async def test_get_cache_status(self, funding_provider):
        """get_cache_status returns cache info."""
        await funding_provider.get_funding("BTC", "hyperliquid")

        status = funding_provider.get_cache_status()
        assert "hyperliquid:BTC" in status
        assert "rate_bps" in status["hyperliquid:BTC"]
        assert "daily_cost_bps" in status["hyperliquid:BTC"]

    @pytest.mark.asyncio
    async def test_cache_key_case_insensitive(self, funding_provider):
        """Cache keys handle case correctly."""
        await funding_provider.get_funding("btc", "Hyperliquid")

        # Should find in cache
        assert funding_provider._is_cache_valid("hyperliquid", "BTC")

    def test_static_funding_returns_defaults(self, funding_provider):
        """Static funding returns reasonable defaults."""
        data = funding_provider._get_static_funding("BTC", "hyperliquid")
        assert data.source == "static"
        assert data.rate_bps > 0
        assert data.interval_hours == FUNDING_INTERVAL_HOURS


class TestBybitFunding:
    """Tests for Bybit funding API."""

    @pytest.mark.asyncio
    async def test_bybit_api_success(self, funding_provider):
        """Successfully parses Bybit API response."""
        with patch.object(funding_provider, "_get_client") as mock_client:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json = MagicMock(
                return_value={
                    "retCode": 0,
                    "result": {
                        "list": [
                            {
                                "fundingRate": "0.0001",  # 0.01%
                                "nextFundingTime": str(int(datetime.now(timezone.utc).timestamp() * 1000 + 3600000)),
                            }
                        ]
                    },
                }
            )

            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_http

            data = await funding_provider._fetch_bybit_funding("BTC")

            assert data is not None
            assert data.exchange == "bybit"
            assert data.source == "api"
            assert data.rate_bps > 0

    @pytest.mark.asyncio
    async def test_bybit_api_error(self, funding_provider):
        """Handles Bybit API errors gracefully."""
        with patch.object(funding_provider, "_get_client") as mock_client:
            mock_response = AsyncMock()
            mock_response.status_code = 500

            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_http

            data = await funding_provider._fetch_bybit_funding("BTC")
            assert data is None


class TestFundingProviderSingleton:
    """Tests for global singleton."""

    def test_get_funding_provider_singleton(self):
        """get_funding_provider returns same instance."""
        provider1 = get_funding_provider()
        provider2 = get_funding_provider()
        assert provider1 is provider2

    def test_init_funding_provider(self):
        """init_funding_provider creates new instance."""
        # Reset singleton for test
        import app.funding_provider as module
        module._funding_provider = None

        provider = init_funding_provider(testnet=True)
        assert provider.testnet is True

        # get_funding_provider should now return this instance
        assert get_funding_provider() is provider


class TestFundingIntegration:
    """Integration-style tests."""

    @pytest.mark.asyncio
    async def test_ev_adjustment_pattern(self, funding_provider):
        """Test pattern for EV adjustment with funding."""
        # Mock API to use static fallback
        with patch.object(funding_provider, "_fetch_hyperliquid_funding", new_callable=AsyncMock, return_value=None):
            # Get funding cost for expected hold time (24h)
            cost_bps = await funding_provider.get_funding_cost_bps(
                "BTC", "hyperliquid", hold_hours=24
            )

            # Should be a reasonable value (3 bps/day typical)
            assert 0 < cost_bps < 50  # Sanity check

            # For a 100 bps stop distance, funding cost in R:
            stop_bps = 100
            funding_r = cost_bps / stop_bps
            assert funding_r > 0

    @pytest.mark.asyncio
    async def test_venue_comparison(self, funding_provider):
        """Compare funding across venues."""
        # Mock APIs to use static fallback
        with patch.object(funding_provider, "_fetch_hyperliquid_funding", new_callable=AsyncMock, return_value=None):
            with patch.object(funding_provider, "_fetch_bybit_funding", new_callable=AsyncMock, return_value=None):
                hl_cost = await funding_provider.get_funding_cost_bps(
                    "BTC", "hyperliquid", hold_hours=24
                )
                bybit_cost = await funding_provider.get_funding_cost_bps(
                    "BTC", "bybit", hold_hours=24
                )

                # Both should have values
                assert hl_cost > 0
                assert bybit_cost > 0

    @pytest.mark.asyncio
    async def test_close_client(self, funding_provider):
        """Can close HTTP client."""
        await funding_provider.close()
        assert funding_provider._client is None
