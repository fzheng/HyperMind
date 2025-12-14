"""
Tests for Multi-Exchange ATR Provider Module

@module tests.test_atr_provider
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from app.atr_provider import (
    ATRData,
    Candle,
    ATRProviderInterface,
    HyperliquidATRProvider,
    BybitATRProvider,
    ATRManager,
    get_atr_manager,
    init_atr_manager,
)
from app.atr_provider.interface import (
    calculate_true_range,
    calculate_atr,
    ATR_PERIOD,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_candles():
    """Create sample candle data for testing."""
    now = datetime.now(timezone.utc)
    candles = []
    base_price = 50000.0

    for i in range(20):
        ts = now - timedelta(minutes=19 - i)
        high = base_price + 100 + (i % 5) * 20
        low = base_price - 100 - (i % 3) * 15
        close = (high + low) / 2 + (i % 2 - 0.5) * 50
        candles.append(
            Candle(
                ts=ts,
                open=base_price,
                high=high,
                low=low,
                close=close,
            )
        )
    return candles


@pytest.fixture
def mock_pool():
    """Create mock database pool."""
    pool = MagicMock()
    pool.acquire = MagicMock()
    return pool


# =============================================================================
# Interface / Core Tests
# =============================================================================


class TestTrueRangeCalculation:
    """Tests for true range calculation."""

    def test_true_range_no_prev_close(self):
        """TR with no previous close is just high-low."""
        candle = Candle(
            ts=datetime.now(timezone.utc),
            open=100,
            high=110,
            low=95,
            close=105,
        )
        tr = calculate_true_range(candle, None)
        assert tr == 15  # 110 - 95

    def test_true_range_with_prev_close_high_gap(self):
        """TR accounts for gap up from prev close."""
        candle = Candle(
            ts=datetime.now(timezone.utc),
            open=105,
            high=110,
            low=100,
            close=108,
        )
        # High = 110, Low = 100, Prev close = 90
        # HL = 10, |H-PC| = 20, |L-PC| = 10
        tr = calculate_true_range(candle, prev_close=90)
        assert tr == 20  # |110 - 90|

    def test_true_range_with_prev_close_low_gap(self):
        """TR accounts for gap down from prev close."""
        candle = Candle(
            ts=datetime.now(timezone.utc),
            open=95,
            high=100,
            low=90,
            close=92,
        )
        # High = 100, Low = 90, Prev close = 110
        # HL = 10, |H-PC| = 10, |L-PC| = 20
        tr = calculate_true_range(candle, prev_close=110)
        assert tr == 20  # |90 - 110|


class TestATRCalculation:
    """Tests for ATR calculation."""

    def test_atr_insufficient_data(self):
        """ATR returns None with insufficient candles."""
        candles = [
            Candle(
                ts=datetime.now(timezone.utc),
                open=100,
                high=110,
                low=95,
                close=105,
            )
        ]
        atr = calculate_atr(candles, period=14)
        assert atr is None

    def test_atr_calculation(self, sample_candles):
        """ATR calculates correctly with sufficient data."""
        atr = calculate_atr(sample_candles, period=14)
        assert atr is not None
        assert atr > 0

    def test_atr_wilder_smoothing(self):
        """ATR uses Wilder's smoothing."""
        now = datetime.now(timezone.utc)
        # Create 20 candles with constant TR of 100
        candles = []
        for i in range(20):
            candles.append(
                Candle(
                    ts=now + timedelta(minutes=i),
                    open=1000,
                    high=1050,
                    low=950,
                    close=1000,
                )
            )

        atr = calculate_atr(candles, period=14)
        # With constant TR=100, ATR should converge to 100
        assert atr is not None
        assert 95 <= atr <= 105


class TestATRData:
    """Tests for ATRData dataclass."""

    def test_atr_data_is_stale_fallback(self):
        """Fallback source is always stale."""
        data = ATRData(
            asset="BTC",
            atr=500,
            atr_pct=1.0,
            price=50000,
            multiplier=2.0,
            stop_distance_pct=2.0,
            timestamp=datetime.now(timezone.utc),
            source="fallback_hardcoded",
            exchange="hyperliquid",
        )
        assert data.is_stale is True

    def test_atr_data_is_stale_old(self):
        """Old data is stale."""
        data = ATRData(
            asset="BTC",
            atr=500,
            atr_pct=1.0,
            price=50000,
            multiplier=2.0,
            stop_distance_pct=2.0,
            timestamp=datetime.now(timezone.utc) - timedelta(minutes=10),
            source="db",
            exchange="hyperliquid",
        )
        assert data.is_stale is True

    def test_atr_data_is_fresh(self):
        """Recent data is fresh."""
        data = ATRData(
            asset="BTC",
            atr=500,
            atr_pct=1.0,
            price=50000,
            multiplier=2.0,
            stop_distance_pct=2.0,
            timestamp=datetime.now(timezone.utc) - timedelta(seconds=30),
            source="db",
            exchange="hyperliquid",
        )
        assert data.is_stale is False

    def test_atr_data_is_data_driven(self):
        """Check data-driven sources."""
        for source in ["db", "api", "calculated", "realized_vol"]:
            data = ATRData(
                asset="BTC",
                atr=500,
                atr_pct=1.0,
                price=50000,
                multiplier=2.0,
                stop_distance_pct=2.0,
                timestamp=datetime.now(timezone.utc),
                source=source,
                exchange="test",
            )
            assert data.is_data_driven is True

        # Fallback is not data-driven
        data = ATRData(
            asset="BTC",
            atr=500,
            atr_pct=1.0,
            price=50000,
            multiplier=2.0,
            stop_distance_pct=2.0,
            timestamp=datetime.now(timezone.utc),
            source="fallback_hardcoded",
            exchange="test",
        )
        assert data.is_data_driven is False


# =============================================================================
# Hyperliquid Provider Tests
# =============================================================================


class TestHyperliquidATRProvider:
    """Tests for Hyperliquid ATR provider."""

    def test_is_configured_without_pool(self):
        """Provider is not configured without pool."""
        provider = HyperliquidATRProvider()
        assert provider.is_configured is False

    def test_is_configured_with_pool(self, mock_pool):
        """Provider is configured with pool."""
        provider = HyperliquidATRProvider(mock_pool)
        assert provider.is_configured is True

    def test_set_pool(self, mock_pool):
        """Can set pool after initialization."""
        provider = HyperliquidATRProvider()
        assert provider.is_configured is False
        provider.set_pool(mock_pool)
        assert provider.is_configured is True

    @pytest.mark.asyncio
    async def test_get_atr_no_pool_returns_fallback(self):
        """Returns fallback when no pool configured."""
        provider = HyperliquidATRProvider()
        atr = await provider.get_atr("BTC", price=50000)
        assert atr.source == "fallback_hardcoded"
        assert atr.exchange == "hyperliquid"

    def test_exchange_name(self):
        """Exchange name is set correctly."""
        provider = HyperliquidATRProvider()
        assert provider.exchange_name == "hyperliquid"


# =============================================================================
# Bybit Provider Tests
# =============================================================================


class TestBybitATRProvider:
    """Tests for Bybit ATR provider."""

    def test_is_configured(self):
        """Provider is always configured (public API)."""
        provider = BybitATRProvider(testnet=True)
        assert provider.is_configured is True

    def test_format_symbol_btc(self):
        """BTC formats to BTCUSDT."""
        provider = BybitATRProvider()
        assert provider._format_symbol("BTC") == "BTCUSDT"
        assert provider._format_symbol("btc") == "BTCUSDT"

    def test_format_symbol_eth(self):
        """ETH formats to ETHUSDT."""
        provider = BybitATRProvider()
        assert provider._format_symbol("ETH") == "ETHUSDT"

    def test_testnet_url(self):
        """Testnet uses testnet URL."""
        provider = BybitATRProvider(testnet=True)
        assert "testnet" in provider.base_url

    def test_mainnet_url(self):
        """Mainnet uses mainnet URL."""
        provider = BybitATRProvider(testnet=False)
        assert "testnet" not in provider.base_url

    def test_exchange_name(self):
        """Exchange name is set correctly."""
        provider = BybitATRProvider()
        assert provider.exchange_name == "bybit"

    @pytest.mark.asyncio
    async def test_get_candles_api_error(self):
        """Returns empty list on API error."""
        provider = BybitATRProvider()

        with patch.object(provider, "_get_client") as mock_client:
            mock_response = AsyncMock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"

            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_http

            candles = await provider.get_candles("BTC")
            assert candles == []

    @pytest.mark.asyncio
    async def test_get_atr_api_success(self):
        """Successfully fetches ATR from API."""
        provider = BybitATRProvider()

        # Mock successful kline response
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        mock_klines = []
        for i in range(20):
            ts = now_ms - (19 - i) * 60000
            mock_klines.append(
                [str(ts), "50000", "50100", "49900", "50050", "100", "5000000"]
            )

        with patch.object(provider, "_get_client") as mock_client:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json = MagicMock(
                return_value={
                    "retCode": 0,
                    "result": {"list": mock_klines},
                }
            )

            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_http

            atr = await provider.get_atr("BTC", price=50000)
            assert atr.source == "api"
            assert atr.exchange == "bybit"
            assert atr.atr > 0


# =============================================================================
# ATR Manager Tests
# =============================================================================


class TestATRManager:
    """Tests for ATR Manager."""

    def test_register_provider(self):
        """Can register providers."""
        manager = ATRManager()
        provider = HyperliquidATRProvider()
        manager.register_provider("hyperliquid", provider)
        assert "hyperliquid" in manager.registered_exchanges

    def test_get_provider(self):
        """Can retrieve registered provider."""
        manager = ATRManager()
        provider = HyperliquidATRProvider()
        manager.register_provider("hyperliquid", provider)
        retrieved = manager.get_provider("hyperliquid")
        assert retrieved is provider

    def test_get_provider_not_found(self):
        """Returns None for unregistered exchange."""
        manager = ATRManager()
        assert manager.get_provider("unknown") is None

    def test_set_default_exchange(self):
        """Can set default exchange."""
        manager = ATRManager()
        manager.set_default_exchange("bybit")
        assert manager._default_exchange == "bybit"

    @pytest.mark.asyncio
    async def test_get_atr_no_providers(self):
        """Returns fallback when no providers registered."""
        manager = ATRManager()
        atr = await manager.get_atr("BTC", price=50000)
        assert atr.source == "fallback_hardcoded"

    @pytest.mark.asyncio
    async def test_get_atr_with_provider(self):
        """Uses registered provider for ATR."""
        manager = ATRManager()

        # Mock provider
        mock_provider = MagicMock(spec=ATRProviderInterface)
        mock_provider.is_configured = True
        mock_provider.get_atr = AsyncMock(
            return_value=ATRData(
                asset="BTC",
                atr=500,
                atr_pct=1.0,
                price=50000,
                multiplier=2.0,
                stop_distance_pct=2.0,
                timestamp=datetime.now(timezone.utc),
                source="api",
                exchange="bybit",
            )
        )

        manager.register_provider("bybit", mock_provider)
        atr = await manager.get_atr("BTC", exchange="bybit", price=50000)
        assert atr.exchange == "bybit"
        assert atr.source == "api"

    @pytest.mark.asyncio
    async def test_get_atr_fallback_to_default(self):
        """Falls back to default when target fails."""
        manager = ATRManager()

        # Target provider returns fallback
        mock_target = MagicMock(spec=ATRProviderInterface)
        mock_target.is_configured = True
        mock_target.get_atr = AsyncMock(
            return_value=ATRData(
                asset="BTC",
                atr=500,
                atr_pct=1.0,
                price=50000,
                multiplier=2.0,
                stop_distance_pct=2.0,
                timestamp=datetime.now(timezone.utc),
                source="fallback_hardcoded",  # Not data-driven
                exchange="bybit",
            )
        )

        # Default provider returns real data
        mock_default = MagicMock(spec=ATRProviderInterface)
        mock_default.is_configured = True
        mock_default.get_atr = AsyncMock(
            return_value=ATRData(
                asset="BTC",
                atr=500,
                atr_pct=1.0,
                price=50000,
                multiplier=2.0,
                stop_distance_pct=2.0,
                timestamp=datetime.now(timezone.utc),
                source="db",
                exchange="hyperliquid",
            )
        )

        manager.register_provider("bybit", mock_target)
        manager.register_provider("hyperliquid", mock_default)
        manager.set_default_exchange("hyperliquid")

        atr = await manager.get_atr("BTC", exchange="bybit", price=50000)
        # Should fall back to hyperliquid
        assert atr.exchange == "hyperliquid"
        assert atr.source == "db"

    @pytest.mark.asyncio
    async def test_get_atr_with_staleness_check(self):
        """Can check staleness."""
        manager = ATRManager()

        mock_provider = MagicMock(spec=ATRProviderInterface)
        mock_provider.is_configured = True
        mock_provider.get_atr = AsyncMock(
            return_value=ATRData(
                asset="BTC",
                atr=500,
                atr_pct=1.0,
                price=50000,
                multiplier=2.0,
                stop_distance_pct=2.0,
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=10),
                source="db",
                exchange="hyperliquid",
            )
        )
        mock_provider.check_staleness = MagicMock(return_value=(True, "Stale data"))

        manager.register_provider("hyperliquid", mock_provider)

        atr, is_stale = await manager.get_atr_with_staleness_check(
            "BTC", exchange="hyperliquid", log_stale=False
        )
        assert is_stale is True

    def test_clear_all_caches(self):
        """Can clear all provider caches."""
        manager = ATRManager()

        mock_provider = MagicMock(spec=ATRProviderInterface)
        mock_provider.clear_cache = MagicMock()

        manager.register_provider("test", mock_provider)
        manager.clear_all_caches()

        mock_provider.clear_cache.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Can check health of providers."""
        manager = ATRManager()

        mock_provider = MagicMock(spec=ATRProviderInterface)
        mock_provider.get_atr = AsyncMock(
            return_value=ATRData(
                asset="BTC",
                atr=500,
                atr_pct=1.0,
                price=50000,
                multiplier=2.0,
                stop_distance_pct=2.0,
                timestamp=datetime.now(timezone.utc),
                source="api",
                exchange="test",
            )
        )

        manager.register_provider("test", mock_provider)
        health = await manager.health_check()
        assert health["test"] is True

    def test_get_stop_fraction(self):
        """Correctly calculates stop fraction."""
        manager = ATRManager()
        atr = ATRData(
            asset="BTC",
            atr=500,
            atr_pct=1.0,
            price=50000,
            multiplier=2.0,
            stop_distance_pct=2.0,  # 2%
            timestamp=datetime.now(timezone.utc),
            source="db",
            exchange="test",
        )
        fraction = manager.get_stop_fraction(atr)
        assert fraction == 0.02  # 2% = 0.02


class TestGlobalATRManager:
    """Tests for global ATR manager singleton."""

    def test_get_atr_manager_singleton(self):
        """get_atr_manager returns singleton."""
        manager1 = get_atr_manager()
        manager2 = get_atr_manager()
        assert manager1 is manager2

    def test_init_atr_manager(self, mock_pool):
        """init_atr_manager configures providers."""
        # Reset singleton for test
        import app.atr_provider.manager as manager_module
        manager_module._atr_manager = None

        manager = init_atr_manager(pool=mock_pool, testnet=True)

        assert "hyperliquid" in manager.registered_exchanges
        assert "bybit" in manager.registered_exchanges
        assert manager._default_exchange == "hyperliquid"
