"""
Tests for Multi-Exchange Integration Module

Tests the abstract interface, factory, and adapter implementations.
Uses mocking to avoid actual API calls.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import os

from app.exchanges import (
    ExchangeConfig,
    ExchangeInterface,
    ExchangeType,
    OrderParams,
    OrderResult,
    OrderSide,
    OrderType,
    Position,
    PositionSide,
    Balance,
    MarginMode,
    MarketData,
    create_exchange,
    get_exchange,
    list_available_exchanges,
    is_exchange_available,
    ExchangeManager,
    AggregatedBalance,
    AggregatedPositions,
    get_exchange_manager,
    init_exchange_manager,
)


class TestExchangeConfig:
    """Tests for ExchangeConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ExchangeConfig(exchange_type=ExchangeType.HYPERLIQUID)

        assert config.exchange_type == ExchangeType.HYPERLIQUID
        assert config.testnet is True
        assert config.default_leverage == 1
        assert config.default_margin_mode == MarginMode.CROSS
        assert config.default_slippage_pct == 0.5

    def test_get_credentials_from_env(self):
        """Test credential retrieval from environment."""
        config = ExchangeConfig(
            exchange_type=ExchangeType.HYPERLIQUID,
            private_key_env="TEST_PRIVATE_KEY",
            api_key_env="TEST_API_KEY",
        )

        with patch.dict(os.environ, {"TEST_PRIVATE_KEY": "secret123"}):
            assert config.get_private_key() == "secret123"

        # Missing env var returns None
        assert config.get_api_key() is None

    def test_empty_env_names(self):
        """Test empty environment variable names return None."""
        config = ExchangeConfig(
            exchange_type=ExchangeType.BYBIT,
            private_key_env="",
            api_key_env="",
        )

        assert config.get_private_key() is None
        assert config.get_api_key() is None


class TestOrderParams:
    """Tests for OrderParams dataclass."""

    def test_market_order(self):
        """Test market order parameters."""
        params = OrderParams(
            symbol="BTC",
            side=OrderSide.BUY,
            size=0.1,
        )

        assert params.symbol == "BTC"
        assert params.side == OrderSide.BUY
        assert params.size == 0.1
        assert params.order_type == OrderType.MARKET
        assert params.price is None
        assert params.reduce_only is False

    def test_limit_order_with_stops(self):
        """Test limit order with stop loss and take profit."""
        params = OrderParams(
            symbol="ETH",
            side=OrderSide.SELL,
            size=1.0,
            order_type=OrderType.LIMIT,
            price=2500.0,
            stop_loss=2400.0,
            take_profit=2700.0,
            leverage=5,
        )

        assert params.order_type == OrderType.LIMIT
        assert params.price == 2500.0
        assert params.stop_loss == 2400.0
        assert params.take_profit == 2700.0
        assert params.leverage == 5


class TestPosition:
    """Tests for Position dataclass."""

    def test_long_position(self):
        """Test long position properties."""
        pos = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            size=0.5,
            entry_price=50000.0,
            mark_price=51000.0,
            leverage=10,
        )

        assert pos.is_long is True
        assert pos.is_short is False
        assert pos.notional_value == 0.5 * 51000.0

    def test_short_position(self):
        """Test short position properties."""
        pos = Position(
            symbol="ETHUSDT",
            side=PositionSide.SHORT,
            size=-2.0,
            entry_price=3000.0,
            mark_price=2900.0,
        )

        assert pos.is_long is False
        assert pos.is_short is True
        assert pos.notional_value == 2.0 * 2900.0


class TestBalance:
    """Tests for Balance dataclass."""

    def test_margin_ratio(self):
        """Test margin ratio calculation."""
        balance = Balance(
            total_equity=10000.0,
            available_balance=8000.0,
            margin_used=2000.0,
        )

        assert balance.margin_ratio == 0.2  # 2000/10000

    def test_zero_equity(self):
        """Test margin ratio with zero equity."""
        balance = Balance(
            total_equity=0.0,
            available_balance=0.0,
            margin_used=0.0,
        )

        assert balance.margin_ratio == 0.0


class TestMarketData:
    """Tests for MarketData dataclass."""

    def test_market_data_properties(self):
        """Test market data calculated properties."""
        data = MarketData(
            symbol="BTCUSDT",
            bid=50000.0,
            ask=50010.0,
            last=50005.0,
            mark_price=50005.0,
        )

        assert data.mid_price == 50005.0
        assert data.spread == 10.0
        assert data.spread_pct == pytest.approx(0.02, rel=0.01)


class TestFactory:
    """Tests for exchange factory functions."""

    def test_list_available_exchanges(self):
        """Test listing available exchanges."""
        exchanges = list_available_exchanges()

        # All three adapters should be available
        assert "hyperliquid" in exchanges
        assert "aster" in exchanges
        assert "bybit" in exchanges

    def test_is_exchange_available(self):
        """Test checking exchange availability."""
        assert is_exchange_available(ExchangeType.HYPERLIQUID) is True
        assert is_exchange_available(ExchangeType.ASTER) is True
        assert is_exchange_available(ExchangeType.BYBIT) is True

    def test_create_exchange_hyperliquid(self):
        """Test creating Hyperliquid adapter."""
        config = ExchangeConfig(
            exchange_type=ExchangeType.HYPERLIQUID,
            testnet=True,
        )

        exchange = create_exchange(config)

        assert exchange is not None
        assert exchange.exchange_type == ExchangeType.HYPERLIQUID
        assert exchange.is_connected is False

    def test_create_exchange_aster(self):
        """Test creating Aster adapter."""
        config = ExchangeConfig(
            exchange_type=ExchangeType.ASTER,
            testnet=True,
        )

        exchange = create_exchange(config)

        assert exchange is not None
        assert exchange.exchange_type == ExchangeType.ASTER

    def test_create_exchange_bybit(self):
        """Test creating Bybit adapter."""
        config = ExchangeConfig(
            exchange_type=ExchangeType.BYBIT,
            testnet=True,
        )

        exchange = create_exchange(config)

        assert exchange is not None
        assert exchange.exchange_type == ExchangeType.BYBIT

    def test_get_exchange_with_defaults(self):
        """Test get_exchange uses correct default env vars."""
        exchange = get_exchange(ExchangeType.HYPERLIQUID, testnet=True)

        # Check default env var names were set
        assert exchange.config.private_key_env == "HL_PRIVATE_KEY"
        assert exchange.config.account_address_env == "HL_ACCOUNT_ADDRESS"


class TestHyperliquidAdapter:
    """Tests for Hyperliquid adapter."""

    def test_format_symbol(self):
        """Test symbol formatting - Hyperliquid just uppercases."""
        config = ExchangeConfig(exchange_type=ExchangeType.HYPERLIQUID)
        from app.exchanges.hyperliquid_adapter import HyperliquidAdapter

        adapter = HyperliquidAdapter(config)

        # Hyperliquid uses simple uppercase symbols
        assert adapter.format_symbol("BTC") == "BTC"
        assert adapter.format_symbol("btc") == "BTC"
        assert adapter.format_symbol("eth") == "ETH"

    def test_format_quantity(self):
        """Test quantity formatting - uses cached precision or default 3."""
        config = ExchangeConfig(exchange_type=ExchangeType.HYPERLIQUID)
        from app.exchanges.hyperliquid_adapter import HyperliquidAdapter

        adapter = HyperliquidAdapter(config)

        # Without cache, uses default sz_decimals=3
        assert adapter.format_quantity("BTC", 0.123456) == 0.123
        assert adapter.format_quantity("ETH", 1.23456) == 1.234

    def test_is_not_configured_without_key(self):
        """Test is_configured returns False without credentials."""
        config = ExchangeConfig(
            exchange_type=ExchangeType.HYPERLIQUID,
            private_key_env="NONEXISTENT_KEY",
        )
        from app.exchanges.hyperliquid_adapter import HyperliquidAdapter

        adapter = HyperliquidAdapter(config)

        assert adapter.is_configured is False


class TestAsterAdapter:
    """Tests for Aster adapter."""

    def test_format_symbol(self):
        """Test symbol formatting adds -PERP suffix."""
        config = ExchangeConfig(exchange_type=ExchangeType.ASTER)
        from app.exchanges.aster_adapter import AsterAdapter

        adapter = AsterAdapter(config)

        # Aster appends -PERP if not present
        assert adapter.format_symbol("BTC") == "BTC-PERP"
        assert adapter.format_symbol("btc") == "BTC-PERP"
        assert adapter.format_symbol("BTC-PERP") == "BTC-PERP"
        assert adapter.format_symbol("ETH") == "ETH-PERP"

    def test_format_quantity(self):
        """Test quantity formatting - uses cached precision or default 4."""
        config = ExchangeConfig(exchange_type=ExchangeType.ASTER)
        from app.exchanges.aster_adapter import AsterAdapter

        adapter = AsterAdapter(config)

        # Without cache, uses default size_precision=4
        assert adapter.format_quantity("BTC", 0.123456) == 0.1234
        assert adapter.format_quantity("ETH", 1.234567) == 1.2345


class TestBybitAdapter:
    """Tests for Bybit adapter."""

    def test_format_symbol(self):
        """Test symbol formatting to USDT format."""
        config = ExchangeConfig(exchange_type=ExchangeType.BYBIT)
        from app.exchanges.bybit_adapter import BybitAdapter

        adapter = BybitAdapter(config)

        assert adapter.format_symbol("BTC") == "BTCUSDT"
        assert adapter.format_symbol("btc") == "BTCUSDT"
        assert adapter.format_symbol("BTC-PERP") == "BTCUSDT"
        assert adapter.format_symbol("BTCUSDT") == "BTCUSDT"
        assert adapter.format_symbol("ETH") == "ETHUSDT"

    def test_format_quantity(self):
        """Test quantity formatting."""
        config = ExchangeConfig(exchange_type=ExchangeType.BYBIT)
        from app.exchanges.bybit_adapter import BybitAdapter

        adapter = BybitAdapter(config)

        # BTCUSDT has 3 decimals
        assert adapter.format_quantity("BTC", 0.1234567) == 0.123

        # ETHUSDT has 2 decimals
        assert adapter.format_quantity("ETH", 1.23456) == 1.23

    def test_format_price(self):
        """Test price formatting."""
        config = ExchangeConfig(exchange_type=ExchangeType.BYBIT)
        from app.exchanges.bybit_adapter import BybitAdapter

        adapter = BybitAdapter(config)

        # BTCUSDT has 1 decimal for price
        assert adapter.format_price("BTC", 50000.123) == 50000.1

        # ETHUSDT has 2 decimals for price
        assert adapter.format_price("ETH", 3000.1234) == 3000.12

    def test_is_not_configured_without_credentials(self):
        """Test is_configured returns False without credentials."""
        config = ExchangeConfig(
            exchange_type=ExchangeType.BYBIT,
            api_key_env="NONEXISTENT_KEY",
            api_secret_env="NONEXISTENT_SECRET",
        )
        from app.exchanges.bybit_adapter import BybitAdapter

        adapter = BybitAdapter(config)

        assert adapter.is_configured is False


class TestExchangeIntegration:
    """Integration tests for exchange adapters (mocked)."""

    @pytest.mark.asyncio
    async def test_hyperliquid_connect_without_creds(self):
        """Test Hyperliquid connection fails without credentials."""
        config = ExchangeConfig(
            exchange_type=ExchangeType.HYPERLIQUID,
            private_key_env="NONEXISTENT",
        )
        from app.exchanges.hyperliquid_adapter import HyperliquidAdapter

        adapter = HyperliquidAdapter(config)
        result = await adapter.connect()

        assert result is False
        assert adapter.is_connected is False

    @pytest.mark.asyncio
    async def test_bybit_connect_without_creds(self):
        """Test Bybit connection fails without credentials."""
        config = ExchangeConfig(
            exchange_type=ExchangeType.BYBIT,
            api_key_env="NONEXISTENT",
            api_secret_env="NONEXISTENT",
        )
        from app.exchanges.bybit_adapter import BybitAdapter

        adapter = BybitAdapter(config)
        result = await adapter.connect()

        assert result is False
        assert adapter.is_connected is False

    @pytest.mark.asyncio
    async def test_get_balance_not_connected(self):
        """Test get_balance returns None when not connected."""
        config = ExchangeConfig(exchange_type=ExchangeType.BYBIT)
        from app.exchanges.bybit_adapter import BybitAdapter

        adapter = BybitAdapter(config)

        balance = await adapter.get_balance()
        assert balance is None

    @pytest.mark.asyncio
    async def test_get_positions_not_connected(self):
        """Test get_positions returns empty list when not connected."""
        config = ExchangeConfig(exchange_type=ExchangeType.BYBIT)
        from app.exchanges.bybit_adapter import BybitAdapter

        adapter = BybitAdapter(config)

        positions = await adapter.get_positions()
        assert positions == []

    @pytest.mark.asyncio
    async def test_place_order_not_connected(self):
        """Test place_order fails when not connected."""
        config = ExchangeConfig(exchange_type=ExchangeType.BYBIT)
        from app.exchanges.bybit_adapter import BybitAdapter

        adapter = BybitAdapter(config)

        result = await adapter.place_order(
            OrderParams(symbol="BTC", side=OrderSide.BUY, size=0.01)
        )

        assert result.success is False
        assert "Not connected" in result.error


class TestOrderResult:
    """Tests for OrderResult dataclass."""

    def test_successful_order(self):
        """Test successful order result."""
        result = OrderResult(
            success=True,
            order_id="12345",
            fill_price=50000.0,
            fill_size=0.1,
            filled_pct=100.0,
            status="filled",
            fees=5.0,
        )

        assert result.success is True
        assert result.order_id == "12345"
        assert result.error is None

    def test_failed_order(self):
        """Test failed order result."""
        result = OrderResult(
            success=False,
            error="Insufficient balance",
            status="rejected",
        )

        assert result.success is False
        assert result.error == "Insufficient balance"


class TestExchangeTypeValidation:
    """Tests for exchange type validation."""

    def test_bybit_adapter_rejects_wrong_type(self):
        """Test Bybit adapter rejects wrong exchange type."""
        config = ExchangeConfig(exchange_type=ExchangeType.HYPERLIQUID)

        with pytest.raises(ValueError, match="Invalid exchange type"):
            from app.exchanges.bybit_adapter import BybitAdapter

            BybitAdapter(config)

    def test_exchange_type_property(self):
        """Test exchange_type property returns correct type."""
        from app.exchanges.hyperliquid_adapter import HyperliquidAdapter
        from app.exchanges.aster_adapter import AsterAdapter
        from app.exchanges.bybit_adapter import BybitAdapter

        hl_config = ExchangeConfig(exchange_type=ExchangeType.HYPERLIQUID)
        assert HyperliquidAdapter(hl_config).exchange_type == ExchangeType.HYPERLIQUID

        aster_config = ExchangeConfig(exchange_type=ExchangeType.ASTER)
        assert AsterAdapter(aster_config).exchange_type == ExchangeType.ASTER

        bybit_config = ExchangeConfig(exchange_type=ExchangeType.BYBIT)
        assert BybitAdapter(bybit_config).exchange_type == ExchangeType.BYBIT


class TestExchangeManager:
    """Tests for ExchangeManager class."""

    def test_initialization(self):
        """Test manager initializes with empty exchange dict."""
        manager = ExchangeManager()

        assert manager.connected_exchanges == []
        assert manager.default_exchange is None

    def test_get_exchange_not_registered(self):
        """Test getting unregistered exchange returns None."""
        manager = ExchangeManager()

        exchange = manager.get_exchange(ExchangeType.HYPERLIQUID)
        assert exchange is None

    def test_set_default_exchange_not_registered(self):
        """Test setting default to unregistered exchange raises error."""
        manager = ExchangeManager()

        with pytest.raises(ValueError, match="not registered"):
            manager.default_exchange = ExchangeType.HYPERLIQUID

    def test_normalize_symbol(self):
        """Test symbol normalization removes exchange suffixes."""
        manager = ExchangeManager()

        assert manager.normalize_symbol("BTC-PERP") == "BTC"
        assert manager.normalize_symbol("BTCUSDT") == "BTC"
        assert manager.normalize_symbol("BTC/USDT") == "BTC"  # Uppercased then /USDT stripped
        assert manager.normalize_symbol("ETH-USD") == "ETH"
        assert manager.normalize_symbol("BTC") == "BTC"

    @pytest.mark.asyncio
    async def test_get_balance_no_exchanges(self):
        """Test getting balance with no exchanges returns None."""
        manager = ExchangeManager()

        balance = await manager.get_balance(ExchangeType.HYPERLIQUID)
        assert balance is None

    @pytest.mark.asyncio
    async def test_get_positions_no_exchanges(self):
        """Test getting positions with no exchanges returns empty list."""
        manager = ExchangeManager()

        positions = await manager.get_positions(ExchangeType.HYPERLIQUID)
        assert positions == []

    @pytest.mark.asyncio
    async def test_execute_order_no_default(self):
        """Test execute order fails without default exchange."""
        manager = ExchangeManager()

        result = await manager.execute_order(
            None,
            OrderParams(symbol="BTC", side=OrderSide.BUY, size=0.01)
        )

        assert result.success is False
        assert "no default set" in result.error.lower()

    @pytest.mark.asyncio
    async def test_disconnect_all_empty(self):
        """Test disconnect all with no exchanges does nothing."""
        manager = ExchangeManager()
        await manager.disconnect_all()  # Should not raise

    @pytest.mark.asyncio
    async def test_get_all_positions_empty(self):
        """Test get all positions with no exchanges."""
        manager = ExchangeManager()

        agg = await manager.get_all_positions()

        assert agg.positions == []
        assert agg.per_exchange == {}
        assert agg.total_notional == 0.0

    @pytest.mark.asyncio
    async def test_get_aggregated_balance_empty(self):
        """Test aggregated balance with no exchanges."""
        manager = ExchangeManager()

        balance = await manager.get_aggregated_balance()
        assert balance is None

    @pytest.mark.asyncio
    async def test_close_position_no_default(self):
        """Test close position fails without default exchange."""
        manager = ExchangeManager()

        result = await manager.close_position("BTC")

        assert result.success is False
        assert "no default set" in result.error.lower()

    @pytest.mark.asyncio
    async def test_set_leverage_no_default(self):
        """Test set leverage fails without default exchange."""
        manager = ExchangeManager()

        result = await manager.set_leverage("BTC", 5)
        assert result is False

    @pytest.mark.asyncio
    async def test_get_market_price_no_default(self):
        """Test get market price without default exchange."""
        manager = ExchangeManager()

        price = await manager.get_market_price("BTC")
        assert price is None


class TestExchangeManagerWithMockedExchange:
    """Tests for ExchangeManager with mocked exchange adapters."""

    @pytest.fixture
    def mock_exchange(self):
        """Create a mock exchange adapter."""
        mock = MagicMock(spec=ExchangeInterface)
        mock.is_connected = True
        mock.is_configured = True
        mock.exchange_type = ExchangeType.HYPERLIQUID
        mock.config = ExchangeConfig(exchange_type=ExchangeType.HYPERLIQUID)
        return mock

    @pytest.mark.asyncio
    async def test_connect_exchange_mocked(self, mock_exchange):
        """Test connecting exchange with mocked adapter."""
        manager = ExchangeManager()

        # Mock the factory
        with patch("app.exchanges.manager.get_exchange", return_value=mock_exchange):
            mock_exchange.connect = AsyncMock(return_value=True)

            result = await manager.connect_exchange(ExchangeType.HYPERLIQUID)

            assert result is True
            assert ExchangeType.HYPERLIQUID in manager.connected_exchanges
            assert manager.default_exchange == ExchangeType.HYPERLIQUID

    @pytest.mark.asyncio
    async def test_get_balance_mocked(self, mock_exchange):
        """Test getting balance with mocked adapter."""
        manager = ExchangeManager()
        manager._exchanges[ExchangeType.HYPERLIQUID] = mock_exchange
        manager._default_exchange = ExchangeType.HYPERLIQUID

        test_balance = Balance(
            total_equity=10000.0,
            available_balance=8000.0,
            margin_used=2000.0,
        )
        mock_exchange.get_balance = AsyncMock(return_value=test_balance)

        balance = await manager.get_balance(ExchangeType.HYPERLIQUID)

        assert balance is not None
        assert balance.total_equity == 10000.0

    @pytest.mark.asyncio
    async def test_get_positions_mocked(self, mock_exchange):
        """Test getting positions with mocked adapter."""
        manager = ExchangeManager()
        manager._exchanges[ExchangeType.HYPERLIQUID] = mock_exchange

        test_positions = [
            Position(
                symbol="BTC",
                side=PositionSide.LONG,
                size=0.5,
                entry_price=50000.0,
                mark_price=51000.0,
            )
        ]
        mock_exchange.get_positions = AsyncMock(return_value=test_positions)

        positions = await manager.get_positions(ExchangeType.HYPERLIQUID)

        assert len(positions) == 1
        assert positions[0].symbol == "BTC"

    @pytest.mark.asyncio
    async def test_execute_order_mocked(self, mock_exchange):
        """Test executing order with mocked adapter."""
        manager = ExchangeManager()
        manager._exchanges[ExchangeType.HYPERLIQUID] = mock_exchange
        manager._default_exchange = ExchangeType.HYPERLIQUID

        test_result = OrderResult(
            success=True,
            order_id="12345",
            fill_price=50000.0,
            fill_size=0.1,
            status="filled",
        )
        mock_exchange.place_order = AsyncMock(return_value=test_result)

        result = await manager.execute_order(
            ExchangeType.HYPERLIQUID,
            OrderParams(symbol="BTC", side=OrderSide.BUY, size=0.1)
        )

        assert result.success is True
        assert result.order_id == "12345"

    @pytest.mark.asyncio
    async def test_aggregated_balance_mocked(self, mock_exchange):
        """Test aggregated balance across exchanges."""
        manager = ExchangeManager()
        manager._exchanges[ExchangeType.HYPERLIQUID] = mock_exchange

        test_balance = Balance(
            total_equity=10000.0,
            available_balance=8000.0,
            margin_used=2000.0,
            unrealized_pnl=500.0,
        )
        mock_exchange.get_balance = AsyncMock(return_value=test_balance)

        agg = await manager.get_aggregated_balance()

        assert agg is not None
        assert agg.total_equity == 10000.0
        assert agg.available_balance == 8000.0
        assert "hyperliquid" in agg.per_exchange

    @pytest.mark.asyncio
    async def test_aggregated_positions_mocked(self, mock_exchange):
        """Test aggregated positions across exchanges."""
        manager = ExchangeManager()
        manager._exchanges[ExchangeType.HYPERLIQUID] = mock_exchange

        test_positions = [
            Position(
                symbol="BTC",
                side=PositionSide.LONG,
                size=0.5,
                entry_price=50000.0,
                mark_price=51000.0,
            )
        ]
        mock_exchange.get_positions = AsyncMock(return_value=test_positions)

        agg = await manager.get_all_positions()

        assert len(agg.positions) == 1
        assert agg.total_notional == 0.5 * 51000.0
        assert "hyperliquid" in agg.per_exchange

    @pytest.mark.asyncio
    async def test_disconnect_exchange(self, mock_exchange):
        """Test disconnecting an exchange."""
        manager = ExchangeManager()
        manager._exchanges[ExchangeType.HYPERLIQUID] = mock_exchange
        manager._default_exchange = ExchangeType.HYPERLIQUID

        mock_exchange.disconnect = AsyncMock()

        await manager.disconnect_exchange(ExchangeType.HYPERLIQUID)

        assert ExchangeType.HYPERLIQUID not in manager._exchanges
        assert manager.default_exchange is None

    @pytest.mark.asyncio
    async def test_format_symbol_with_exchange(self, mock_exchange):
        """Test symbol formatting delegates to exchange."""
        manager = ExchangeManager()
        manager._exchanges[ExchangeType.HYPERLIQUID] = mock_exchange
        manager._default_exchange = ExchangeType.HYPERLIQUID

        mock_exchange.format_symbol = MagicMock(return_value="BTC")

        result = manager.format_symbol("btc", ExchangeType.HYPERLIQUID)

        assert result == "BTC"
        mock_exchange.format_symbol.assert_called_once_with("btc")

    @pytest.mark.asyncio
    async def test_open_position_mocked(self, mock_exchange):
        """Test opening position with mocked adapter."""
        manager = ExchangeManager()
        manager._exchanges[ExchangeType.HYPERLIQUID] = mock_exchange
        manager._default_exchange = ExchangeType.HYPERLIQUID

        test_result = OrderResult(
            success=True,
            order_id="open_123",
            fill_price=50000.0,
            fill_size=0.1,
            status="filled",
        )
        mock_exchange.open_position = AsyncMock(return_value=test_result)

        result = await manager.open_position(
            ExchangeType.HYPERLIQUID,
            OrderParams(symbol="BTC", side=OrderSide.BUY, size=0.1)
        )

        assert result.success is True
        assert result.order_id == "open_123"

    @pytest.mark.asyncio
    async def test_close_position_mocked(self, mock_exchange):
        """Test closing position with mocked adapter."""
        manager = ExchangeManager()
        manager._exchanges[ExchangeType.HYPERLIQUID] = mock_exchange
        manager._default_exchange = ExchangeType.HYPERLIQUID

        test_result = OrderResult(
            success=True,
            order_id="close_123",
            fill_price=50500.0,
            fill_size=0.1,
            status="filled",
        )
        mock_exchange.close_position = AsyncMock(return_value=test_result)

        result = await manager.close_position("BTC")

        assert result.success is True
        mock_exchange.close_position.assert_called_once_with("BTC", None)

    @pytest.mark.asyncio
    async def test_set_stop_loss_mocked(self, mock_exchange):
        """Test setting stop loss with mocked adapter."""
        manager = ExchangeManager()
        manager._exchanges[ExchangeType.HYPERLIQUID] = mock_exchange
        manager._default_exchange = ExchangeType.HYPERLIQUID

        test_result = OrderResult(
            success=True,
            order_id="sl_123",
            status="pending",
        )
        mock_exchange.set_stop_loss = AsyncMock(return_value=test_result)

        result = await manager.set_stop_loss("BTC", 48000.0)

        assert result.success is True
        mock_exchange.set_stop_loss.assert_called_once_with("BTC", 48000.0, None)

    @pytest.mark.asyncio
    async def test_set_take_profit_mocked(self, mock_exchange):
        """Test setting take profit with mocked adapter."""
        manager = ExchangeManager()
        manager._exchanges[ExchangeType.HYPERLIQUID] = mock_exchange
        manager._default_exchange = ExchangeType.HYPERLIQUID

        test_result = OrderResult(
            success=True,
            order_id="tp_123",
            status="pending",
        )
        mock_exchange.set_take_profit = AsyncMock(return_value=test_result)

        result = await manager.set_take_profit("BTC", 55000.0)

        assert result.success is True
        mock_exchange.set_take_profit.assert_called_once_with("BTC", 55000.0, None)


class TestExchangeManagerSingleton:
    """Tests for ExchangeManager singleton and initialization."""

    def test_get_exchange_manager_returns_same_instance(self):
        """Test get_exchange_manager returns singleton."""
        # Reset singleton for test
        import app.exchanges.manager as manager_module
        manager_module._exchange_manager = None

        m1 = get_exchange_manager()
        m2 = get_exchange_manager()

        assert m1 is m2

        # Cleanup
        manager_module._exchange_manager = None

    @pytest.mark.asyncio
    async def test_init_exchange_manager_no_credentials(self):
        """Test init_exchange_manager with no credentials configured."""
        import app.exchanges.manager as manager_module
        manager_module._exchange_manager = None

        # Ensure no credentials are set
        with patch.dict(os.environ, {}, clear=True):
            manager = await init_exchange_manager(testnet=True)

            # Without credentials, no exchanges should connect
            assert manager.connected_exchanges == []

        # Cleanup
        manager_module._exchange_manager = None
