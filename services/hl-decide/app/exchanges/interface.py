"""
Abstract Exchange Interface

Defines the contract that all exchange adapters must implement.
Inspired by NoFx's trader interface design.

Key operations:
- Account: get_balance(), get_positions(), get_account_state()
- Trading: open_position(), close_position(), place_order()
- Orders: get_order_status(), cancel_order(), cancel_all_orders()
- Risk: set_leverage(), set_stop_loss(), set_take_profit()
- Market: get_market_price(), get_orderbook()

@module exchanges.interface
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class ExchangeType(str, Enum):
    """Supported exchange types."""
    HYPERLIQUID = "hyperliquid"
    ASTER = "aster"
    BYBIT = "bybit"


class OrderSide(str, Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"


class PositionSide(str, Enum):
    """Position side."""
    LONG = "long"
    SHORT = "short"
    NONE = "none"


class MarginMode(str, Enum):
    """Margin mode."""
    CROSS = "cross"
    ISOLATED = "isolated"


@dataclass
class FeeConfig:
    """
    Fee structure for an exchange.

    Fees are specified in basis points (bps). 1 bps = 0.01% = 0.0001.
    """
    maker_fee_bps: float = 2.5  # Default: 2.5 bps = 0.025%
    taker_fee_bps: float = 5.0  # Default: 5.0 bps = 0.05%
    funding_rate_hourly_bps: float = 0.0  # Typically ~1 bps/8h

    @property
    def maker_fee_pct(self) -> float:
        """Maker fee as percentage (0.01 = 1%)."""
        return self.maker_fee_bps / 10000

    @property
    def taker_fee_pct(self) -> float:
        """Taker fee as percentage (0.01 = 1%)."""
        return self.taker_fee_bps / 10000

    def round_trip_cost_bps(self, is_maker_entry: bool = False, is_maker_exit: bool = False) -> float:
        """
        Calculate round-trip fee cost in basis points.

        Most market orders are taker orders. Limit orders may get maker rebates.

        Args:
            is_maker_entry: Whether entry order is maker
            is_maker_exit: Whether exit order is maker

        Returns:
            Total round-trip cost in bps
        """
        entry_fee = self.maker_fee_bps if is_maker_entry else self.taker_fee_bps
        exit_fee = self.maker_fee_bps if is_maker_exit else self.taker_fee_bps
        return entry_fee + exit_fee


# Default fee configs for known exchanges
EXCHANGE_FEES: dict[ExchangeType, FeeConfig] = {
    ExchangeType.HYPERLIQUID: FeeConfig(maker_fee_bps=2.5, taker_fee_bps=5.0),
    ExchangeType.ASTER: FeeConfig(maker_fee_bps=2.5, taker_fee_bps=5.0),  # Similar to HL
    ExchangeType.BYBIT: FeeConfig(maker_fee_bps=10.0, taker_fee_bps=6.0),  # VIP0 rates
}


def get_fee_config(exchange_type: ExchangeType) -> FeeConfig:
    """Get fee config for an exchange type."""
    return EXCHANGE_FEES.get(exchange_type, FeeConfig())


@dataclass
class ExchangeConfig:
    """
    Configuration for exchange connection.

    Credentials are loaded from environment variables for security.
    """
    exchange_type: ExchangeType
    testnet: bool = True

    # Environment variable names for credentials (not the actual values)
    private_key_env: str = ""
    api_key_env: str = ""
    api_secret_env: str = ""
    account_address_env: str = ""

    # Trading parameters
    default_leverage: int = 1
    default_margin_mode: MarginMode = MarginMode.CROSS
    default_slippage_pct: float = 0.5  # 0.5%

    # Fee structure (can be overridden per-user for VIP tiers)
    fees: Optional[FeeConfig] = None

    # Rate limiting
    max_requests_per_second: int = 10

    def get_fees(self) -> FeeConfig:
        """Get fee config, falling back to exchange defaults."""
        if self.fees:
            return self.fees
        return get_fee_config(self.exchange_type)

    def get_private_key(self) -> Optional[str]:
        """Get private key from environment."""
        return os.getenv(self.private_key_env) if self.private_key_env else None

    def get_api_key(self) -> Optional[str]:
        """Get API key from environment."""
        return os.getenv(self.api_key_env) if self.api_key_env else None

    def get_api_secret(self) -> Optional[str]:
        """Get API secret from environment."""
        return os.getenv(self.api_secret_env) if self.api_secret_env else None

    def get_account_address(self) -> Optional[str]:
        """Get account address from environment."""
        return os.getenv(self.account_address_env) if self.account_address_env else None


@dataclass
class OrderParams:
    """Parameters for placing an order."""
    symbol: str  # Trading pair (BTC, ETH, BTC-PERP, etc.)
    side: OrderSide
    size: float  # Size in base currency
    order_type: OrderType = OrderType.MARKET
    price: Optional[float] = None  # Required for limit orders
    reduce_only: bool = False
    slippage_pct: float = 0.5
    leverage: Optional[int] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    client_order_id: Optional[str] = None


@dataclass
class OrderResult:
    """Result of an order operation."""
    success: bool
    order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    fill_price: Optional[float] = None
    fill_size: Optional[float] = None
    filled_pct: float = 0.0
    status: str = ""  # filled, partial, pending, cancelled, rejected
    slippage_actual: Optional[float] = None
    fees: float = 0.0
    error: Optional[str] = None
    raw_response: Optional[dict] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Position:
    """Current position information."""
    symbol: str
    side: PositionSide
    size: float  # Positive for long, negative for short (or use side)
    entry_price: float
    mark_price: float
    liquidation_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    leverage: int = 1
    margin_mode: MarginMode = MarginMode.CROSS
    margin_used: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def notional_value(self) -> float:
        """Position notional value."""
        return abs(self.size) * self.mark_price

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.side == PositionSide.LONG or self.size > 0

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.side == PositionSide.SHORT or self.size < 0


@dataclass
class Balance:
    """Account balance information."""
    total_equity: float  # Total account value
    available_balance: float  # Available for trading
    margin_used: float  # Currently used as margin
    unrealized_pnl: float = 0.0
    realized_pnl_today: float = 0.0
    currency: str = "USD"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def margin_ratio(self) -> float:
        """Margin usage ratio."""
        if self.total_equity <= 0:
            return 0.0
        return self.margin_used / self.total_equity


@dataclass
class MarketData:
    """Market price data."""
    symbol: str
    bid: float
    ask: float
    last: float
    mark_price: float
    index_price: Optional[float] = None
    funding_rate: Optional[float] = None
    volume_24h: Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def mid_price(self) -> float:
        """Mid price between bid and ask."""
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.ask - self.bid

    @property
    def spread_pct(self) -> float:
        """Spread as percentage of mid price."""
        mid = self.mid_price
        return (self.spread / mid * 100) if mid > 0 else 0.0


class ExchangeInterface(ABC):
    """
    Abstract base class for exchange adapters.

    All exchange implementations must implement these methods to ensure
    consistent behavior across different exchanges.

    Usage:
        exchange = HyperliquidAdapter(config)
        await exchange.connect()

        # Get account state
        balance = await exchange.get_balance()
        positions = await exchange.get_positions()

        # Place order
        result = await exchange.open_position(
            OrderParams(symbol="BTC", side=OrderSide.BUY, size=0.01)
        )

        await exchange.disconnect()
    """

    def __init__(self, config: ExchangeConfig):
        """
        Initialize exchange adapter.

        Args:
            config: Exchange configuration
        """
        self.config = config
        self._connected = False

    @property
    def exchange_type(self) -> ExchangeType:
        """Get exchange type."""
        return self.config.exchange_type

    @property
    def is_connected(self) -> bool:
        """Check if connected to exchange."""
        return self._connected

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """Check if exchange is properly configured with credentials."""
        pass

    # Connection management

    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to exchange API.

        Returns:
            True if connection successful
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from exchange API."""
        pass

    # Account operations

    @abstractmethod
    async def get_balance(self) -> Optional[Balance]:
        """
        Get account balance.

        Returns:
            Balance object or None if unavailable
        """
        pass

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """
        Get all open positions.

        Returns:
            List of Position objects
        """
        pass

    @abstractmethod
    async def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for specific symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            Position object or None if no position
        """
        pass

    # Trading operations

    @abstractmethod
    async def open_position(self, params: OrderParams) -> OrderResult:
        """
        Open a new position or add to existing.

        Args:
            params: Order parameters

        Returns:
            OrderResult with fill details
        """
        pass

    @abstractmethod
    async def close_position(
        self,
        symbol: str,
        size: Optional[float] = None,  # None = close all
    ) -> OrderResult:
        """
        Close position partially or fully.

        Args:
            symbol: Trading pair symbol
            size: Size to close (None for full close)

        Returns:
            OrderResult with fill details
        """
        pass

    @abstractmethod
    async def place_order(self, params: OrderParams) -> OrderResult:
        """
        Place a generic order.

        Args:
            params: Order parameters

        Returns:
            OrderResult with order details
        """
        pass

    # Order management

    @abstractmethod
    async def get_order_status(self, order_id: str) -> Optional[dict]:
        """
        Get status of an order.

        Args:
            order_id: Order ID

        Returns:
            Order status dict or None
        """
        pass

    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """
        Cancel a specific order.

        Args:
            symbol: Trading pair symbol
            order_id: Order ID to cancel

        Returns:
            True if cancelled successfully
        """
        pass

    @abstractmethod
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all open orders.

        Args:
            symbol: Optional symbol to filter (None = all symbols)

        Returns:
            Number of orders cancelled
        """
        pass

    # Risk management

    @abstractmethod
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """
        Set leverage for a symbol.

        Args:
            symbol: Trading pair symbol
            leverage: Leverage multiplier

        Returns:
            True if set successfully
        """
        pass

    @abstractmethod
    async def set_stop_loss(
        self,
        symbol: str,
        stop_price: float,
        size: Optional[float] = None,
    ) -> OrderResult:
        """
        Set stop-loss order for position.

        Args:
            symbol: Trading pair symbol
            stop_price: Stop trigger price
            size: Size to close (None = full position)

        Returns:
            OrderResult with stop order details
        """
        pass

    @abstractmethod
    async def set_take_profit(
        self,
        symbol: str,
        take_profit_price: float,
        size: Optional[float] = None,
    ) -> OrderResult:
        """
        Set take-profit order for position.

        Args:
            symbol: Trading pair symbol
            take_profit_price: Take profit trigger price
            size: Size to close (None = full position)

        Returns:
            OrderResult with take profit order details
        """
        pass

    # Market data

    @abstractmethod
    async def get_market_price(self, symbol: str) -> Optional[float]:
        """
        Get current market mid price.

        Args:
            symbol: Trading pair symbol

        Returns:
            Mid price or None if unavailable
        """
        pass

    @abstractmethod
    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """
        Get full market data including bid/ask.

        Args:
            symbol: Trading pair symbol

        Returns:
            MarketData object or None
        """
        pass

    # Utility methods

    @abstractmethod
    def format_symbol(self, symbol: str) -> str:
        """
        Format symbol to exchange-specific format.

        Args:
            symbol: Generic symbol (e.g., "BTC")

        Returns:
            Exchange-specific symbol (e.g., "BTC-PERP", "BTCUSDT")
        """
        pass

    @abstractmethod
    def format_quantity(self, symbol: str, quantity: float) -> float:
        """
        Format quantity to exchange precision.

        Args:
            symbol: Trading pair symbol
            quantity: Raw quantity

        Returns:
            Quantity rounded to exchange precision
        """
        pass

    @abstractmethod
    def format_price(self, symbol: str, price: float) -> float:
        """
        Format price to exchange precision.

        Args:
            symbol: Trading pair symbol
            price: Raw price

        Returns:
            Price rounded to exchange precision
        """
        pass
