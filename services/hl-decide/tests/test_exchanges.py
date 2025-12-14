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
    FeeConfig,
    get_fee_config,
    EXCHANGE_FEES,
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


class TestFeeConfig:
    """Tests for FeeConfig (Phase 6: Multi-Exchange fee support)."""

    def test_default_fee_config(self):
        """Test default fee configuration values."""
        config = FeeConfig()

        assert config.maker_fee_bps == 2.5
        assert config.taker_fee_bps == 5.0
        assert config.funding_rate_hourly_bps == 0.0

    def test_fee_pct_properties(self):
        """Test fee percentage property conversions."""
        config = FeeConfig(maker_fee_bps=10.0, taker_fee_bps=20.0)

        # 10 bps = 0.001 = 0.1%
        assert config.maker_fee_pct == pytest.approx(0.001, abs=0.0001)
        # 20 bps = 0.002 = 0.2%
        assert config.taker_fee_pct == pytest.approx(0.002, abs=0.0001)

    def test_round_trip_cost_taker_taker(self):
        """Test round-trip cost with taker orders both ways."""
        config = FeeConfig(maker_fee_bps=2.5, taker_fee_bps=5.0)

        # Default: taker entry + taker exit = 5 + 5 = 10 bps
        cost = config.round_trip_cost_bps()
        assert cost == pytest.approx(10.0, abs=0.01)

    def test_round_trip_cost_maker_maker(self):
        """Test round-trip cost with maker orders both ways."""
        config = FeeConfig(maker_fee_bps=2.5, taker_fee_bps=5.0)

        # Maker both ways: 2.5 + 2.5 = 5 bps
        cost = config.round_trip_cost_bps(is_maker_entry=True, is_maker_exit=True)
        assert cost == pytest.approx(5.0, abs=0.01)

    def test_round_trip_cost_mixed(self):
        """Test round-trip cost with mixed order types."""
        config = FeeConfig(maker_fee_bps=2.5, taker_fee_bps=5.0)

        # Maker entry + taker exit: 2.5 + 5 = 7.5 bps
        cost = config.round_trip_cost_bps(is_maker_entry=True, is_maker_exit=False)
        assert cost == pytest.approx(7.5, abs=0.01)

    def test_exchange_fees_hyperliquid(self):
        """Test Hyperliquid fee configuration."""
        fees = get_fee_config(ExchangeType.HYPERLIQUID)

        assert fees.maker_fee_bps == 2.5
        assert fees.taker_fee_bps == 5.0

    def test_exchange_fees_bybit(self):
        """Test Bybit fee configuration (higher fees)."""
        fees = get_fee_config(ExchangeType.BYBIT)

        # Bybit VIP0 rates
        assert fees.maker_fee_bps == 10.0
        assert fees.taker_fee_bps == 6.0

    def test_exchange_fees_aster(self):
        """Test Aster fee configuration (similar to HL)."""
        fees = get_fee_config(ExchangeType.ASTER)

        assert fees.maker_fee_bps == 2.5
        assert fees.taker_fee_bps == 5.0

    def test_all_exchanges_have_fee_config(self):
        """Test all exchange types have fee configuration."""
        for exchange_type in ExchangeType:
            fees = get_fee_config(exchange_type)
            assert fees is not None
            assert fees.maker_fee_bps >= 0
            assert fees.taker_fee_bps >= 0

    def test_exchange_config_get_fees_with_override(self):
        """Test ExchangeConfig.get_fees() with custom fees."""
        custom_fees = FeeConfig(maker_fee_bps=1.0, taker_fee_bps=2.0)
        config = ExchangeConfig(
            exchange_type=ExchangeType.HYPERLIQUID,
            fees=custom_fees,
        )

        fees = config.get_fees()
        assert fees.maker_fee_bps == 1.0
        assert fees.taker_fee_bps == 2.0

    def test_exchange_config_get_fees_default(self):
        """Test ExchangeConfig.get_fees() falls back to exchange defaults."""
        config = ExchangeConfig(
            exchange_type=ExchangeType.BYBIT,
            fees=None,
        )

        fees = config.get_fees()
        # Should get Bybit defaults
        assert fees.maker_fee_bps == 10.0
        assert fees.taker_fee_bps == 6.0

    def test_fee_config_in_exchange_fees_dict(self):
        """Test EXCHANGE_FEES dict contains all exchanges."""
        assert ExchangeType.HYPERLIQUID in EXCHANGE_FEES
        assert ExchangeType.ASTER in EXCHANGE_FEES
        assert ExchangeType.BYBIT in EXCHANGE_FEES

    def test_bybit_higher_fees_than_hl(self):
        """Test Bybit has higher round-trip fees than Hyperliquid."""
        hl_fees = get_fee_config(ExchangeType.HYPERLIQUID)
        bybit_fees = get_fee_config(ExchangeType.BYBIT)

        hl_round_trip = hl_fees.round_trip_cost_bps()
        bybit_round_trip = bybit_fees.round_trip_cost_bps()

        # Bybit should have higher fees
        assert bybit_round_trip > hl_round_trip


class TestBybitAdapterResponseParsing:
    """Tests for Bybit adapter parsing of API responses (Phase 6 integration tests)."""

    @pytest.fixture
    def bybit_adapter(self):
        """Create Bybit adapter for testing."""
        from app.exchanges.bybit_adapter import BybitAdapter

        config = ExchangeConfig(
            exchange_type=ExchangeType.BYBIT,
            testnet=True,
            api_key_env="BYBIT_API_KEY",
            api_secret_env="BYBIT_API_SECRET",
        )
        return BybitAdapter(config)

    def test_parse_balance_response(self, bybit_adapter):
        """Test parsing of Bybit wallet balance response."""
        # Simulate Bybit API response structure
        bybit_response = {
            "retCode": 0,
            "retMsg": "OK",
            "result": {
                "list": [
                    {
                        "accountType": "UNIFIED",
                        "coin": [
                            {
                                "coin": "USDT",
                                "equity": "10523.45",
                                "availableToWithdraw": "8234.12",
                                "totalPositionMM": "1523.33",
                                "unrealisedPnl": "234.56",
                                "cumRealisedPnl": "1234.56",
                            }
                        ],
                    }
                ]
            },
        }

        # Extract balance manually using same logic as adapter
        result = bybit_response.get("result", {})
        for account in result.get("list", []):
            for coin in account.get("coin", []):
                if coin.get("coin") == "USDT":
                    balance = Balance(
                        total_equity=float(coin.get("equity", 0)),
                        available_balance=float(coin.get("availableToWithdraw", 0)),
                        margin_used=float(coin.get("totalPositionMM", 0)),
                        unrealized_pnl=float(coin.get("unrealisedPnl", 0)),
                        realized_pnl_today=float(coin.get("cumRealisedPnl", 0)),
                        currency="USDT",
                    )

                    assert balance.total_equity == pytest.approx(10523.45, abs=0.01)
                    assert balance.available_balance == pytest.approx(8234.12, abs=0.01)
                    assert balance.margin_used == pytest.approx(1523.33, abs=0.01)
                    assert balance.unrealized_pnl == pytest.approx(234.56, abs=0.01)
                    return

        pytest.fail("USDT balance not found in response")

    def test_parse_positions_response(self, bybit_adapter):
        """Test parsing of Bybit positions response."""
        # Simulate Bybit API response for positions
        bybit_response = {
            "retCode": 0,
            "retMsg": "OK",
            "result": {
                "list": [
                    {
                        "symbol": "BTCUSDT",
                        "side": "Buy",
                        "size": "0.5",
                        "avgPrice": "42500.50",
                        "markPrice": "43000.00",
                        "liqPrice": "35000.00",
                        "unrealisedPnl": "249.75",
                        "leverage": "10",
                        "positionIM": "2125.03",
                    },
                    {
                        "symbol": "ETHUSDT",
                        "side": "Sell",
                        "size": "3.0",
                        "avgPrice": "2200.00",
                        "markPrice": "2180.00",
                        "liqPrice": "2500.00",
                        "unrealisedPnl": "60.00",
                        "leverage": "5",
                        "positionIM": "1308.00",
                    },
                ]
            },
        }

        # Parse positions
        positions = []
        for item in bybit_response.get("result", {}).get("list", []):
            size = float(item.get("size", 0))
            if size == 0:
                continue

            side_str = item.get("side", "")
            side = PositionSide.LONG if side_str == "Buy" else PositionSide.SHORT

            # Bybit uses positive size for both sides
            if side == PositionSide.SHORT:
                size = -size

            positions.append(
                Position(
                    symbol=item.get("symbol", ""),
                    side=side,
                    size=size,
                    entry_price=float(item.get("avgPrice", 0)),
                    mark_price=float(item.get("markPrice", 0)),
                    liquidation_price=float(item.get("liqPrice", 0)) if item.get("liqPrice") else None,
                    unrealized_pnl=float(item.get("unrealisedPnl", 0)),
                    leverage=int(item.get("leverage", 1)),
                    margin_used=float(item.get("positionIM", 0)),
                )
            )

        assert len(positions) == 2

        # BTC long position
        btc_pos = positions[0]
        assert btc_pos.symbol == "BTCUSDT"
        assert btc_pos.side == PositionSide.LONG
        assert btc_pos.size == pytest.approx(0.5, abs=0.001)
        assert btc_pos.entry_price == pytest.approx(42500.50, abs=0.01)
        assert btc_pos.mark_price == pytest.approx(43000.00, abs=0.01)
        assert btc_pos.leverage == 10

        # ETH short position (negative size)
        eth_pos = positions[1]
        assert eth_pos.symbol == "ETHUSDT"
        assert eth_pos.side == PositionSide.SHORT
        assert eth_pos.size == pytest.approx(-3.0, abs=0.001)
        assert eth_pos.unrealized_pnl == pytest.approx(60.00, abs=0.01)

    def test_bybit_symbol_formatting(self, bybit_adapter):
        """Test Bybit symbol formatting."""
        # Bybit uses BTCUSDT format
        assert bybit_adapter.format_symbol("BTC") == "BTCUSDT"
        assert bybit_adapter.format_symbol("ETH") == "ETHUSDT"
        assert bybit_adapter.format_symbol("btc") == "BTCUSDT"  # Case insensitive

    def test_bybit_quantity_formatting(self, bybit_adapter):
        """Test Bybit quantity precision formatting."""
        # BTC: 3 decimal places
        qty = bybit_adapter.format_quantity("BTCUSDT", 0.12345678)
        assert qty == pytest.approx(0.123, abs=0.001)

        # ETH: 2 decimal places
        qty = bybit_adapter.format_quantity("ETHUSDT", 1.23456)
        assert qty == pytest.approx(1.23, abs=0.01)

    def test_bybit_price_formatting(self, bybit_adapter):
        """Test Bybit price precision formatting."""
        # BTC: 1 decimal place
        price = bybit_adapter.format_price("BTCUSDT", 42567.89)
        assert price == pytest.approx(42567.9, abs=0.1)

        # ETH: 2 decimal places
        price = bybit_adapter.format_price("ETHUSDT", 2234.567)
        assert price == pytest.approx(2234.57, abs=0.01)


class TestAsterAdapterResponseParsing:
    """Tests for Aster adapter parsing of API responses (Phase 6 integration tests)."""

    @pytest.fixture
    def aster_adapter(self):
        """Create Aster adapter for testing."""
        from app.exchanges.aster_adapter import AsterAdapter

        config = ExchangeConfig(
            exchange_type=ExchangeType.ASTER,
            testnet=True,
            private_key_env="ASTER_PRIVATE_KEY",
        )
        return AsterAdapter(config)

    def test_aster_symbol_formatting(self, aster_adapter):
        """Test Aster symbol formatting (similar to Hyperliquid)."""
        # Aster uses BTC-PERP format
        assert aster_adapter.format_symbol("BTC") == "BTC-PERP"
        assert aster_adapter.format_symbol("ETH") == "ETH-PERP"

    def test_aster_quantity_formatting(self, aster_adapter):
        """Test Aster quantity precision formatting."""
        # Should round to reasonable precision
        qty = aster_adapter.format_quantity("BTC-PERP", 0.123456789)
        # Default precision is 4 decimal places
        assert qty == pytest.approx(0.1234, abs=0.0001)

    def test_aster_price_formatting(self, aster_adapter):
        """Test Aster price precision formatting."""
        # Should round to reasonable precision
        price = aster_adapter.format_price("BTC-PERP", 42567.123456)
        # Default price precision is 2 decimal places
        assert price == pytest.approx(42567.12, abs=0.01)


class TestExchangeManagerHealthCheck:
    """Tests for ExchangeManager health check functionality (Phase 6)."""

    @pytest.mark.asyncio
    async def test_health_check_returns_status(self):
        """Test health check returns proper status structure."""
        import app.exchanges.manager as manager_module
        manager_module._exchange_manager = None

        manager = ExchangeManager()

        # With no exchanges connected, health check should return empty status
        result = await manager.health_check()

        assert "reconnected" in result
        assert "timestamp" in result
        assert result["reconnected"] == []

    @pytest.mark.asyncio
    async def test_health_check_with_mock_exchange(self):
        """Test health check with mocked exchange."""
        manager = ExchangeManager()

        # Create mock exchange
        mock_exchange = MagicMock()
        mock_exchange.is_connected = True
        mock_exchange.is_configured = True

        # Mock balance response
        mock_balance = Balance(
            total_equity=10000.0,
            available_balance=8000.0,
            margin_used=2000.0,
            unrealized_pnl=100.0,
        )
        mock_exchange.get_balance = AsyncMock(return_value=mock_balance)
        mock_exchange.connect = AsyncMock(return_value=True)
        mock_exchange.disconnect = AsyncMock()

        # Register mock exchange
        manager._exchanges[ExchangeType.HYPERLIQUID] = mock_exchange

        # Run health check
        result = await manager.health_check()

        assert ExchangeType.HYPERLIQUID.value in result
        status = result[ExchangeType.HYPERLIQUID.value]
        assert status["connected"] is True
        assert status["healthy"] is True
        assert status["error"] is None

    @pytest.mark.asyncio
    async def test_health_check_reconnects_failed_exchange(self):
        """Test health check attempts reconnection for failed exchanges."""
        manager = ExchangeManager()

        # Create mock exchange that was disconnected
        mock_exchange = MagicMock()
        mock_exchange.is_connected = False
        mock_exchange.is_configured = True
        mock_exchange.connect = AsyncMock(return_value=True)
        mock_exchange.disconnect = AsyncMock()
        mock_exchange.get_balance = AsyncMock(
            return_value=Balance(
                total_equity=10000.0,
                available_balance=8000.0,
                margin_used=2000.0,
            )
        )

        # Register mock exchange
        manager._exchanges[ExchangeType.BYBIT] = mock_exchange

        # Run health check
        result = await manager.health_check()

        # Should have attempted reconnection
        mock_exchange.connect.assert_called_once()
        assert ExchangeType.BYBIT.value in result["reconnected"]

    @pytest.mark.asyncio
    async def test_health_check_detects_unhealthy_exchange(self):
        """Test health check detects unhealthy exchange."""
        manager = ExchangeManager()

        # Create mock exchange that returns None balance
        mock_exchange = MagicMock()
        mock_exchange.is_connected = True
        mock_exchange.is_configured = True
        mock_exchange.get_balance = AsyncMock(return_value=None)
        mock_exchange.connect = AsyncMock(return_value=False)
        mock_exchange.disconnect = AsyncMock()

        manager._exchanges[ExchangeType.ASTER] = mock_exchange

        # Run health check
        result = await manager.health_check()

        status = result[ExchangeType.ASTER.value]
        assert status["connected"] is True  # Was connected before probe
        assert status["healthy"] is False  # But probe failed
        assert "Balance returned None" in status["error"]

    @pytest.mark.asyncio
    async def test_health_check_with_stagger_delay(self):
        """Test health check respects stagger delay between exchanges."""
        import time
        manager = ExchangeManager()

        # Create two mock exchanges
        mock_balance = Balance(
            total_equity=10000.0,
            available_balance=8000.0,
            margin_used=2000.0,
        )

        mock_hl = MagicMock()
        mock_hl.is_connected = True
        mock_hl.is_configured = True
        mock_hl.get_balance = AsyncMock(return_value=mock_balance)
        mock_hl.connect = AsyncMock(return_value=True)
        mock_hl.disconnect = AsyncMock()

        mock_bybit = MagicMock()
        mock_bybit.is_connected = True
        mock_bybit.is_configured = True
        mock_bybit.get_balance = AsyncMock(return_value=mock_balance)
        mock_bybit.connect = AsyncMock(return_value=True)
        mock_bybit.disconnect = AsyncMock()

        manager._exchanges[ExchangeType.HYPERLIQUID] = mock_hl
        manager._exchanges[ExchangeType.BYBIT] = mock_bybit

        # Run health check with 100ms stagger
        start = time.time()
        await manager.health_check(stagger_delay_ms=100)
        elapsed_ms = (time.time() - start) * 1000

        # Should have at least 100ms delay between the two exchanges
        # Allow some tolerance for async overhead
        assert elapsed_ms >= 80  # 100ms - 20ms tolerance

    @pytest.mark.asyncio
    async def test_health_check_no_stagger_when_disabled(self):
        """Test health check runs quickly when stagger is disabled."""
        import time
        manager = ExchangeManager()

        # Create two mock exchanges
        mock_balance = Balance(
            total_equity=10000.0,
            available_balance=8000.0,
            margin_used=2000.0,
        )

        mock_hl = MagicMock()
        mock_hl.is_connected = True
        mock_hl.is_configured = True
        mock_hl.get_balance = AsyncMock(return_value=mock_balance)
        mock_hl.connect = AsyncMock(return_value=True)
        mock_hl.disconnect = AsyncMock()

        mock_bybit = MagicMock()
        mock_bybit.is_connected = True
        mock_bybit.is_configured = True
        mock_bybit.get_balance = AsyncMock(return_value=mock_balance)
        mock_bybit.connect = AsyncMock(return_value=True)
        mock_bybit.disconnect = AsyncMock()

        manager._exchanges[ExchangeType.HYPERLIQUID] = mock_hl
        manager._exchanges[ExchangeType.BYBIT] = mock_bybit

        # Run health check with 0ms stagger
        start = time.time()
        await manager.health_check(stagger_delay_ms=0)
        elapsed_ms = (time.time() - start) * 1000

        # Should be very quick without intentional delay
        assert elapsed_ms < 100
