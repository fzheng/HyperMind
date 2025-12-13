"""
Tests for Hyperliquid Exchange Integration

Tests the exchange wrapper without actually placing orders.
Real execution is disabled by default for safety.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.hl_exchange import (
    DEFAULT_SLIPPAGE_PCT,
    HyperliquidExchange,
    OrderParams,
    OrderResult,
    execute_market_order,
    get_exchange,
)


class TestOrderParams:
    """Tests for OrderParams dataclass."""

    def test_default_values(self):
        """Test default parameter values."""
        params = OrderParams(asset="BTC", is_buy=True, size=0.01)
        assert params.asset == "BTC"
        assert params.is_buy is True
        assert params.size == 0.01
        assert params.price is None
        assert params.reduce_only is False
        assert params.slippage_pct == DEFAULT_SLIPPAGE_PCT

    def test_custom_slippage(self):
        """Test custom slippage setting."""
        params = OrderParams(asset="ETH", is_buy=False, size=0.5, slippage_pct=1.0)
        assert params.slippage_pct == 1.0

    def test_reduce_only(self):
        """Test reduce-only order."""
        params = OrderParams(asset="BTC", is_buy=False, size=0.02, reduce_only=True)
        assert params.reduce_only is True


class TestOrderResult:
    """Tests for OrderResult dataclass."""

    def test_success_result(self):
        """Test successful order result."""
        result = OrderResult(
            success=True,
            order_id="12345",
            fill_price=100000.0,
            fill_size=0.01,
            slippage_actual=0.05,
        )
        assert result.success is True
        assert result.order_id == "12345"
        assert result.fill_price == 100000.0
        assert result.error is None

    def test_failure_result(self):
        """Test failed order result."""
        result = OrderResult(success=False, error="Insufficient funds")
        assert result.success is False
        assert result.error == "Insufficient funds"
        assert result.order_id is None


class TestHyperliquidExchangeInit:
    """Tests for HyperliquidExchange initialization."""

    def test_default_init(self):
        """Test default initialization."""
        exchange = HyperliquidExchange()
        assert exchange._private_key_env == "HL_PRIVATE_KEY"
        assert exchange._account_address_env == "HL_ACCOUNT_ADDRESS"
        assert exchange._http_client is None

    def test_custom_env_vars(self):
        """Test custom environment variable names."""
        exchange = HyperliquidExchange(
            private_key_env="MY_KEY",
            account_address_env="MY_ADDR",
        )
        assert exchange._private_key_env == "MY_KEY"
        assert exchange._account_address_env == "MY_ADDR"

    def test_is_configured_without_key(self):
        """Test is_configured returns False without private key."""
        with patch.dict(os.environ, {}, clear=True):
            exchange = HyperliquidExchange(private_key_env="NONEXISTENT_KEY")
            # Without SDK and key, should be False
            assert exchange.is_configured is False

    def test_can_execute_disabled(self):
        """Test can_execute is False when REAL_EXECUTION_ENABLED is false."""
        with patch.dict(os.environ, {"REAL_EXECUTION_ENABLED": "false"}):
            exchange = HyperliquidExchange()
            assert exchange.can_execute is False


class TestHyperliquidExchangeSafetyGates:
    """Tests for safety gates preventing accidental execution."""

    @pytest.mark.asyncio
    async def test_place_order_blocked_without_config(self):
        """Test order placement is blocked without proper configuration."""
        exchange = HyperliquidExchange()
        params = OrderParams(asset="BTC", is_buy=True, size=0.01)
        result = await exchange.place_market_order(params)

        assert result.success is False
        assert "Real execution disabled" in result.error

    @pytest.mark.asyncio
    async def test_close_position_blocked_without_config(self):
        """Test position close is blocked without configuration."""
        exchange = HyperliquidExchange()
        result = await exchange.close_position("BTC")

        assert result.success is False
        assert "Real execution disabled" in result.error

    @pytest.mark.asyncio
    async def test_cancel_blocked_without_config(self):
        """Test order cancel is blocked without configuration."""
        exchange = HyperliquidExchange()
        result = await exchange.cancel_order("BTC", "12345")
        assert result is False


class TestHyperliquidExchangePrices:
    """Tests for price fetching."""

    @pytest.mark.asyncio
    async def test_get_mid_price_success(self):
        """Test successful mid price fetch."""
        exchange = HyperliquidExchange()

        # Mock the HTTP client
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"BTC": "100000.50", "ETH": "3500.25"}

        with patch.object(exchange, "_get_client") as mock_client:
            mock_client.return_value = AsyncMock()
            mock_client.return_value.post = AsyncMock(return_value=mock_response)

            price = await exchange.get_mid_price("BTC")
            assert price == 100000.50

    @pytest.mark.asyncio
    async def test_get_mid_price_failure(self):
        """Test mid price fetch failure returns None."""
        exchange = HyperliquidExchange()

        # Mock failed response
        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch.object(exchange, "_get_client") as mock_client:
            mock_client.return_value = AsyncMock()
            mock_client.return_value.post = AsyncMock(return_value=mock_response)

            price = await exchange.get_mid_price("BTC")
            assert price is None

    @pytest.mark.asyncio
    async def test_get_mid_price_exception(self):
        """Test mid price fetch handles exceptions."""
        exchange = HyperliquidExchange()

        with patch.object(exchange, "_get_client") as mock_client:
            mock_client.return_value = AsyncMock()
            mock_client.return_value.post = AsyncMock(side_effect=Exception("Network error"))

            price = await exchange.get_mid_price("BTC")
            assert price is None


class TestHyperliquidExchangeWithMockedSDK:
    """Tests for exchange operations with mocked SDK."""

    @pytest.mark.asyncio
    async def test_place_order_success_with_mocked_sdk(self):
        """Test successful order placement with mocked SDK."""
        with patch.dict(os.environ, {
            "REAL_EXECUTION_ENABLED": "true",
            "HL_PRIVATE_KEY": "0x" + "a" * 64,
        }):
            with patch("app.hl_exchange.SDK_AVAILABLE", True):
                with patch("app.hl_exchange.REAL_EXECUTION_ENABLED", True):
                    exchange = HyperliquidExchange()

                    # Mock SDK components
                    mock_sdk_exchange = MagicMock()
                    mock_sdk_exchange.market_open.return_value = {
                        "status": "ok",
                        "response": {
                            "data": {
                                "statuses": [{
                                    "filled": {
                                        "oid": 12345,
                                        "avgPx": "100050.00",
                                        "totalSz": "0.01",
                                    }
                                }]
                            }
                        }
                    }
                    exchange._exchange = mock_sdk_exchange

                    # Mock price fetch
                    with patch.object(exchange, "get_mid_price", return_value=100000.0):
                        params = OrderParams(asset="BTC", is_buy=True, size=0.01)
                        result = await exchange.place_market_order(params)

                        assert result.success is True
                        assert result.order_id == "12345"
                        assert result.fill_price == 100050.0
                        assert result.fill_size == 0.01
                        assert result.slippage_actual == pytest.approx(0.05, rel=0.01)

    @pytest.mark.asyncio
    async def test_place_order_error_response(self):
        """Test order placement with error in response."""
        with patch.dict(os.environ, {
            "REAL_EXECUTION_ENABLED": "true",
            "HL_PRIVATE_KEY": "0x" + "a" * 64,
        }):
            with patch("app.hl_exchange.SDK_AVAILABLE", True):
                with patch("app.hl_exchange.REAL_EXECUTION_ENABLED", True):
                    exchange = HyperliquidExchange()

                    mock_sdk_exchange = MagicMock()
                    mock_sdk_exchange.market_open.return_value = {
                        "status": "ok",
                        "response": {
                            "data": {
                                "statuses": [{
                                    "error": "Insufficient margin"
                                }]
                            }
                        }
                    }
                    exchange._exchange = mock_sdk_exchange

                    with patch.object(exchange, "get_mid_price", return_value=100000.0):
                        params = OrderParams(asset="BTC", is_buy=True, size=1.0)
                        result = await exchange.place_market_order(params)

                        assert result.success is False
                        assert "Insufficient margin" in result.error

    @pytest.mark.asyncio
    async def test_close_position_success(self):
        """Test successful position close."""
        with patch.dict(os.environ, {
            "REAL_EXECUTION_ENABLED": "true",
            "HL_PRIVATE_KEY": "0x" + "a" * 64,
        }):
            with patch("app.hl_exchange.SDK_AVAILABLE", True):
                with patch("app.hl_exchange.REAL_EXECUTION_ENABLED", True):
                    exchange = HyperliquidExchange()

                    mock_sdk_exchange = MagicMock()
                    mock_sdk_exchange.market_close.return_value = {
                        "status": "ok",
                        "response": {
                            "data": {
                                "statuses": [{
                                    "filled": {
                                        "oid": 67890,
                                        "avgPx": "99950.00",
                                        "totalSz": "0.05",
                                    }
                                }]
                            }
                        }
                    }
                    exchange._exchange = mock_sdk_exchange

                    with patch.object(exchange, "get_mid_price", return_value=100000.0):
                        result = await exchange.close_position("BTC")

                        assert result.success is True
                        assert result.fill_price == 99950.0
                        assert result.fill_size == 0.05


class TestGlobalExchangeInstance:
    """Tests for global exchange instance management."""

    def test_get_exchange_singleton(self):
        """Test get_exchange returns same instance."""
        # Reset global
        import app.hl_exchange as module
        module._exchange = None

        ex1 = get_exchange()
        ex2 = get_exchange()
        assert ex1 is ex2

    def test_get_exchange_custom_params(self):
        """Test get_exchange with custom parameters."""
        import app.hl_exchange as module
        module._exchange = None

        exchange = get_exchange(
            private_key_env="CUSTOM_KEY",
            account_address_env="CUSTOM_ADDR",
        )
        assert exchange._private_key_env == "CUSTOM_KEY"


class TestExecuteMarketOrder:
    """Tests for convenience function."""

    @pytest.mark.asyncio
    async def test_execute_buy_order(self):
        """Test execute_market_order for buy."""
        import app.hl_exchange as module
        module._exchange = None

        with patch("app.hl_exchange.get_exchange") as mock_get:
            mock_exchange = AsyncMock()
            mock_exchange.place_market_order.return_value = OrderResult(
                success=False, error="Disabled"
            )
            mock_exchange.close_position.return_value = OrderResult(
                success=False, error="Disabled"
            )
            mock_get.return_value = mock_exchange

            result = await execute_market_order("BTC", True, 0.01)
            mock_exchange.place_market_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_reduce_only_calls_close(self):
        """Test execute_market_order with reduce_only calls close_position."""
        import app.hl_exchange as module
        module._exchange = None

        with patch("app.hl_exchange.get_exchange") as mock_get:
            mock_exchange = AsyncMock()
            mock_exchange.close_position.return_value = OrderResult(
                success=True, fill_size=0.01
            )
            mock_get.return_value = mock_exchange

            result = await execute_market_order("BTC", False, 0.01, reduce_only=True)
            mock_exchange.close_position.assert_called_once_with("BTC")


class TestNetworkSelection:
    """Tests for testnet/mainnet selection."""

    def test_default_uses_testnet(self):
        """Test default configuration uses testnet."""
        with patch.dict(os.environ, {"HL_USE_TESTNET": "true"}):
            exchange = HyperliquidExchange()
            assert exchange._use_testnet is True

    def test_explicit_mainnet(self):
        """Test explicit mainnet configuration."""
        exchange = HyperliquidExchange(use_testnet=False)
        assert exchange._use_testnet is False


class TestSlippageCalculation:
    """Tests for slippage handling."""

    def test_default_slippage(self):
        """Test default slippage percentage."""
        # Default is 0.5%
        assert DEFAULT_SLIPPAGE_PCT == 0.50

    @pytest.mark.asyncio
    async def test_slippage_actual_calculation(self):
        """Test actual slippage is calculated correctly."""
        with patch.dict(os.environ, {
            "REAL_EXECUTION_ENABLED": "true",
            "HL_PRIVATE_KEY": "0x" + "a" * 64,
        }):
            with patch("app.hl_exchange.SDK_AVAILABLE", True):
                with patch("app.hl_exchange.REAL_EXECUTION_ENABLED", True):
                    exchange = HyperliquidExchange()

                    mock_sdk_exchange = MagicMock()
                    mock_sdk_exchange.market_open.return_value = {
                        "status": "ok",
                        "response": {
                            "data": {
                                "statuses": [{
                                    "filled": {
                                        "oid": 1,
                                        "avgPx": "100100.00",  # 0.1% higher than mid
                                        "totalSz": "0.01",
                                    }
                                }]
                            }
                        }
                    }
                    exchange._exchange = mock_sdk_exchange

                    with patch.object(exchange, "get_mid_price", return_value=100000.0):
                        params = OrderParams(asset="BTC", is_buy=True, size=0.01)
                        result = await exchange.place_market_order(params)

                        # Slippage should be 0.1%
                        assert result.slippage_actual == pytest.approx(0.1, rel=0.01)


class TestPositionManagement:
    """Tests for position-related functionality."""

    @pytest.mark.asyncio
    async def test_get_position_without_sdk(self):
        """Test get_position returns None when SDK not initialized."""
        exchange = HyperliquidExchange()
        exchange._exchange = None  # Ensure SDK not initialized

        # Mock _init_sdk to fail
        with patch.object(exchange, "_init_sdk", return_value=False):
            result = await exchange.get_position("BTC")
            assert result is None

    @pytest.mark.asyncio
    async def test_get_account_value_without_sdk(self):
        """Test get_account_value returns None when SDK not initialized."""
        exchange = HyperliquidExchange()

        with patch.object(exchange, "_init_sdk", return_value=False):
            result = await exchange.get_account_value()
            assert result is None


class TestHttpClientManagement:
    """Tests for HTTP client lifecycle."""

    @pytest.mark.asyncio
    async def test_close_client(self):
        """Test HTTP client closure."""
        exchange = HyperliquidExchange()

        # Create a mock client
        mock_client = AsyncMock()
        exchange._http_client = mock_client

        await exchange.close()

        mock_client.aclose.assert_called_once()
        assert exchange._http_client is None

    @pytest.mark.asyncio
    async def test_close_when_no_client(self):
        """Test close handles no client gracefully."""
        exchange = HyperliquidExchange()
        exchange._http_client = None

        # Should not raise
        await exchange.close()
        assert exchange._http_client is None
