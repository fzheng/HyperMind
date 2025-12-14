"""
Tests for Slippage Estimation Provider

@module tests.test_slippage_provider
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from app.slippage_provider import (
    SlippageProvider,
    SlippageEstimate,
    OrderbookData,
    OrderbookLevel,
    CachedOrderbook,
    get_slippage_provider,
    init_slippage_provider,
    SLIPPAGE_CACHE_TTL_SECONDS,
    SLIPPAGE_WARNING_THRESHOLD_BPS,
    SIZE_THRESHOLD_SMALL,
    SIZE_THRESHOLD_LARGE,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def slippage_provider():
    """Create fresh slippage provider for testing."""
    return SlippageProvider(testnet=True)


@pytest.fixture
def mock_orderbook():
    """Create mock orderbook data."""
    return OrderbookData(
        asset="BTC",
        exchange="hyperliquid",
        bids=[
            OrderbookLevel(price=100000.0, size=1.0),  # $100k
            OrderbookLevel(price=99990.0, size=2.0),   # $200k
            OrderbookLevel(price=99980.0, size=3.0),   # $300k
        ],
        asks=[
            OrderbookLevel(price=100010.0, size=1.0),  # $100k
            OrderbookLevel(price=100020.0, size=2.0),  # $200k
            OrderbookLevel(price=100030.0, size=3.0),  # $300k
        ],
        mid_price=100005.0,
        spread_bps=1.0,
        source="api",
    )


# =============================================================================
# OrderbookLevel Tests
# =============================================================================


class TestOrderbookLevel:
    """Tests for OrderbookLevel dataclass."""

    def test_create_level(self):
        """Can create orderbook level."""
        level = OrderbookLevel(price=100000.0, size=1.5)
        assert level.price == 100000.0
        assert level.size == 1.5


# =============================================================================
# OrderbookData Tests
# =============================================================================


class TestOrderbookData:
    """Tests for OrderbookData dataclass."""

    def test_is_expired_fresh(self, mock_orderbook):
        """Fresh data is not expired."""
        assert mock_orderbook.is_expired is False

    def test_is_expired_old(self):
        """Old data is expired."""
        data = OrderbookData(
            asset="BTC",
            exchange="test",
            bids=[],
            asks=[],
            mid_price=100000.0,
            spread_bps=1.0,
            fetched_at=datetime.now(timezone.utc) - timedelta(seconds=SLIPPAGE_CACHE_TTL_SECONDS + 60),
            source="api",
        )
        assert data.is_expired is True

    def test_bid_depth_usd(self, mock_orderbook):
        """Calculate bid depth in USD."""
        # 3 bids: $100k + $200k + $300k = $600k
        depth = mock_orderbook.get_bid_depth_usd()
        expected = 100000.0 * 1.0 + 99990.0 * 2.0 + 99980.0 * 3.0
        assert abs(depth - expected) < 0.01

    def test_ask_depth_usd(self, mock_orderbook):
        """Calculate ask depth in USD."""
        depth = mock_orderbook.get_ask_depth_usd()
        expected = 100010.0 * 1.0 + 100020.0 * 2.0 + 100030.0 * 3.0
        assert abs(depth - expected) < 0.01

    def test_depth_with_limit(self, mock_orderbook):
        """Calculate depth with level limit."""
        depth = mock_orderbook.get_bid_depth_usd(levels=2)
        expected = 100000.0 * 1.0 + 99990.0 * 2.0
        assert abs(depth - expected) < 0.01


# =============================================================================
# CachedOrderbook Tests
# =============================================================================


class TestCachedOrderbook:
    """Tests for CachedOrderbook dataclass."""

    def test_is_expired_fresh(self, mock_orderbook):
        """Fresh cache is not expired."""
        cached = CachedOrderbook(
            data=mock_orderbook,
            fetched_at=datetime.now(timezone.utc),
        )
        assert cached.is_expired is False

    def test_is_expired_old(self, mock_orderbook):
        """Old cache is expired."""
        cached = CachedOrderbook(
            data=mock_orderbook,
            fetched_at=datetime.now(timezone.utc) - timedelta(seconds=SLIPPAGE_CACHE_TTL_SECONDS + 60),
        )
        assert cached.is_expired is True


# =============================================================================
# SlippageEstimate Tests
# =============================================================================


class TestSlippageEstimate:
    """Tests for SlippageEstimate dataclass."""

    def test_warning_when_high_slippage(self):
        """Warning flag set when slippage exceeds threshold."""
        estimate = SlippageEstimate(
            asset="BTC",
            exchange="hyperliquid",
            order_size_usd=100000.0,
            side="buy",
            estimated_slippage_bps=SLIPPAGE_WARNING_THRESHOLD_BPS + 5,
            expected_fill_price=100050.0,
            mid_price=100000.0,
            impact_bps=5.0,
            is_warning=True,
            source="static",
        )
        assert estimate.is_warning is True

    def test_no_warning_when_low_slippage(self):
        """No warning when slippage below threshold."""
        estimate = SlippageEstimate(
            asset="BTC",
            exchange="hyperliquid",
            order_size_usd=1000.0,
            side="buy",
            estimated_slippage_bps=1.0,
            expected_fill_price=100001.0,
            mid_price=100000.0,
            impact_bps=1.0,
            is_warning=False,
            source="static",
        )
        assert estimate.is_warning is False


# =============================================================================
# SlippageProvider Tests
# =============================================================================


class TestSlippageProvider:
    """Tests for SlippageProvider class."""

    @pytest.mark.asyncio
    async def test_static_estimate_small_order(self, slippage_provider):
        """Small orders get low slippage estimate."""
        # Mock API to return None to force static fallback
        with patch.object(slippage_provider, "_fetch_hyperliquid_orderbook", new_callable=AsyncMock, return_value=None):
            estimate = await slippage_provider.estimate_slippage(
                asset="BTC",
                exchange="hyperliquid",
                order_size_usd=5000,  # Below SIZE_THRESHOLD_SMALL
                side="buy",
            )
            assert estimate.source == "static"
            assert estimate.estimated_slippage_bps > 0
            assert estimate.estimated_slippage_bps < 5  # Small orders have low slippage

    @pytest.mark.asyncio
    async def test_static_estimate_large_order(self, slippage_provider):
        """Large orders get higher slippage estimate."""
        # Mock API to return None to force static fallback
        with patch.object(slippage_provider, "_fetch_hyperliquid_orderbook", new_callable=AsyncMock, return_value=None):
            estimate = await slippage_provider.estimate_slippage(
                asset="BTC",
                exchange="hyperliquid",
                order_size_usd=100000,  # Above SIZE_THRESHOLD_LARGE
                side="buy",
            )
            assert estimate.source == "static"
            assert estimate.estimated_slippage_bps > 0

    @pytest.mark.asyncio
    async def test_estimate_buy_side(self, slippage_provider):
        """Can estimate buy side slippage."""
        with patch.object(slippage_provider, "_fetch_bybit_orderbook", new_callable=AsyncMock, return_value=None):
            estimate = await slippage_provider.estimate_slippage(
                asset="ETH",
                exchange="bybit",
                order_size_usd=25000,
                side="buy",
            )
            assert estimate.side == "buy"
            assert estimate.asset == "ETH"
            assert estimate.exchange == "bybit"

    @pytest.mark.asyncio
    async def test_estimate_sell_side(self, slippage_provider):
        """Can estimate sell side slippage."""
        with patch.object(slippage_provider, "_fetch_bybit_orderbook", new_callable=AsyncMock, return_value=None):
            estimate = await slippage_provider.estimate_slippage(
                asset="ETH",
                exchange="bybit",
                order_size_usd=25000,
                side="sell",
            )
            assert estimate.side == "sell"

    @pytest.mark.asyncio
    async def test_different_exchanges_different_estimates(self, slippage_provider):
        """Different exchanges have different slippage profiles."""
        # Mock APIs to return None to force static fallback
        with patch.object(slippage_provider, "_fetch_hyperliquid_orderbook", new_callable=AsyncMock, return_value=None):
            hl_estimate = await slippage_provider.estimate_slippage(
                asset="BTC",
                exchange="hyperliquid",
                order_size_usd=25000,
                side="buy",
            )
        with patch.object(slippage_provider, "_fetch_bybit_orderbook", new_callable=AsyncMock, return_value=None):
            bybit_estimate = await slippage_provider.estimate_slippage(
                asset="BTC",
                exchange="bybit",
                order_size_usd=25000,
                side="buy",
            )
        # Both should use static estimates
        assert hl_estimate.source == "static"
        assert bybit_estimate.source == "static"
        # Bybit typically has lower slippage (CEX with more liquidity)
        assert bybit_estimate.estimated_slippage_bps < hl_estimate.estimated_slippage_bps

    def test_estimate_from_orderbook(self, slippage_provider, mock_orderbook):
        """Can estimate slippage from orderbook data."""
        estimate = slippage_provider._estimate_from_orderbook(
            orderbook=mock_orderbook,
            order_size_usd=50000,  # $50k order
            side="buy",
        )
        assert estimate.source == "orderbook"
        assert estimate.mid_price == mock_orderbook.mid_price
        assert estimate.estimated_slippage_bps > 0

    def test_estimate_from_orderbook_small_order(self, slippage_provider, mock_orderbook):
        """Small order mostly crosses spread."""
        estimate = slippage_provider._estimate_from_orderbook(
            orderbook=mock_orderbook,
            order_size_usd=10000,  # $10k, fills at first level
            side="buy",
        )
        # Should be mostly the spread (1 bps) with minimal impact
        assert estimate.estimated_slippage_bps < 3

    def test_estimate_from_orderbook_large_order(self, slippage_provider, mock_orderbook):
        """Large order walks deeper into book."""
        estimate = slippage_provider._estimate_from_orderbook(
            orderbook=mock_orderbook,
            order_size_usd=500000,  # $500k, walks through all levels
            side="buy",
        )
        # Should have more slippage from walking the book
        assert estimate.impact_bps > 0

    def test_clear_cache(self, slippage_provider, mock_orderbook):
        """clear_cache removes all cached data."""
        # Add to cache directly
        slippage_provider._cache["test"] = CachedOrderbook(
            data=mock_orderbook,
            fetched_at=datetime.now(timezone.utc),
        )
        assert len(slippage_provider._cache) > 0

        slippage_provider.clear_cache()
        assert len(slippage_provider._cache) == 0

    @pytest.mark.asyncio
    async def test_close_client(self, slippage_provider):
        """Can close HTTP client."""
        await slippage_provider.close()
        assert slippage_provider._client is None


class TestBybitOrderbook:
    """Tests for Bybit orderbook API."""

    @pytest.mark.asyncio
    async def test_bybit_api_success(self, slippage_provider):
        """Successfully parses Bybit API response."""
        with patch.object(slippage_provider, "_get_client") as mock_client:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json = MagicMock(
                return_value={
                    "retCode": 0,
                    "result": {
                        "b": [
                            ["100000", "1.0"],
                            ["99990", "2.0"],
                        ],
                        "a": [
                            ["100010", "1.0"],
                            ["100020", "2.0"],
                        ],
                    },
                }
            )

            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_http

            data = await slippage_provider._fetch_bybit_orderbook("BTC")

            assert data is not None
            assert data.exchange == "bybit"
            assert data.source == "api"
            assert len(data.bids) == 2
            assert len(data.asks) == 2

    @pytest.mark.asyncio
    async def test_bybit_api_error(self, slippage_provider):
        """Handles Bybit API errors gracefully."""
        with patch.object(slippage_provider, "_get_client") as mock_client:
            mock_response = AsyncMock()
            mock_response.status_code = 500

            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_http

            data = await slippage_provider._fetch_bybit_orderbook("BTC")
            assert data is None


class TestHyperliquidOrderbook:
    """Tests for Hyperliquid orderbook API."""

    @pytest.mark.asyncio
    async def test_hyperliquid_api_success(self, slippage_provider):
        """Successfully parses Hyperliquid API response."""
        with patch.object(slippage_provider, "_get_client") as mock_client:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json = MagicMock(
                return_value={
                    "levels": [
                        [
                            {"px": "100000", "sz": "1.0", "n": 1},
                            {"px": "99990", "sz": "2.0", "n": 2},
                        ],
                        [
                            {"px": "100010", "sz": "1.0", "n": 1},
                            {"px": "100020", "sz": "2.0", "n": 2},
                        ],
                    ],
                }
            )

            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_http

            data = await slippage_provider._fetch_hyperliquid_orderbook("BTC")

            assert data is not None
            assert data.exchange == "hyperliquid"
            assert data.source == "api"
            assert len(data.bids) == 2
            assert len(data.asks) == 2

    @pytest.mark.asyncio
    async def test_hyperliquid_api_error(self, slippage_provider):
        """Handles Hyperliquid API errors gracefully."""
        with patch.object(slippage_provider, "_get_client") as mock_client:
            mock_response = AsyncMock()
            mock_response.status_code = 500

            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_http

            data = await slippage_provider._fetch_hyperliquid_orderbook("BTC")
            assert data is None


class TestSlippageProviderSingleton:
    """Tests for global singleton."""

    def test_get_slippage_provider_singleton(self):
        """get_slippage_provider returns same instance."""
        provider1 = get_slippage_provider()
        provider2 = get_slippage_provider()
        assert provider1 is provider2

    def test_init_slippage_provider(self):
        """init_slippage_provider creates new instance."""
        # Reset singleton for test
        import app.slippage_provider as module
        module._slippage_provider = None

        provider = init_slippage_provider(testnet=True)
        assert provider.testnet is True

        # get_slippage_provider should now return this instance
        assert get_slippage_provider() is provider


class TestSlippageIntegration:
    """Integration-style tests."""

    @pytest.mark.asyncio
    async def test_slippage_warning_threshold(self, slippage_provider):
        """Warning triggered for large orders."""
        # Mock API to return None to force static fallback
        with patch.object(slippage_provider, "_fetch_hyperliquid_orderbook", new_callable=AsyncMock, return_value=None):
            # Very large order should trigger warning (with static fallback)
            estimate = await slippage_provider.estimate_slippage(
                asset="ETH",
                exchange="hyperliquid",
                order_size_usd=200000,  # $200k ETH order
                side="buy",
            )
            # Large ETH orders on DEX should have higher slippage
            assert estimate.source == "static"
            assert estimate.estimated_slippage_bps >= 5

    @pytest.mark.asyncio
    async def test_caching_works(self, slippage_provider, mock_orderbook):
        """Orderbook data is cached after first fetch."""
        # Add mock data to cache
        key = slippage_provider._get_cache_key("hyperliquid", "BTC")
        slippage_provider._cache[key] = CachedOrderbook(
            data=mock_orderbook,
            fetched_at=datetime.now(timezone.utc),
        )

        # Should use cache
        orderbook = await slippage_provider.get_orderbook("BTC", "hyperliquid")
        assert orderbook is not None
        assert orderbook.mid_price == mock_orderbook.mid_price

    @pytest.mark.asyncio
    async def test_force_refresh_bypasses_cache(self, slippage_provider, mock_orderbook):
        """Force refresh ignores cache."""
        # Add mock data to cache
        key = slippage_provider._get_cache_key("hyperliquid", "BTC")
        slippage_provider._cache[key] = CachedOrderbook(
            data=mock_orderbook,
            fetched_at=datetime.now(timezone.utc),
        )

        # Force refresh will try API (and fail, returning None)
        with patch.object(slippage_provider, "_fetch_hyperliquid_orderbook", new_callable=AsyncMock, return_value=None):
            orderbook = await slippage_provider.get_orderbook("BTC", "hyperliquid", force_refresh=True)
            # Should return None since API failed
            assert orderbook is None

    def test_cache_status(self, slippage_provider, mock_orderbook):
        """get_cache_status returns cache info."""
        # Add mock data to cache
        key = slippage_provider._get_cache_key("hyperliquid", "BTC")
        slippage_provider._cache[key] = CachedOrderbook(
            data=mock_orderbook,
            fetched_at=datetime.now(timezone.utc),
        )

        status = slippage_provider.get_cache_status()
        assert "hyperliquid:BTC" in status
        assert "mid_price" in status["hyperliquid:BTC"]
        assert "spread_bps" in status["hyperliquid:BTC"]
