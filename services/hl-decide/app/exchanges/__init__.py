"""
Multi-Exchange Integration Module

Phase 6: Provides abstract exchange interface and concrete adapters for:
- Hyperliquid (perp DEX)
- Aster DEX (perp DEX)
- Bybit (CEX)

Each adapter implements the same interface for:
- Account state queries
- Position management
- Order placement/cancellation
- Balance/PnL tracking

@module exchanges
"""

from .interface import (
    ExchangeInterface,
    OrderParams,
    OrderResult,
    Position,
    Balance,
    ExchangeConfig,
    ExchangeType,
    FeeConfig,
    get_fee_config,
    EXCHANGE_FEES,
)
from .interface import (
    MarginMode,
    MarketData,
    OrderSide,
    OrderType,
    PositionSide,
)
from .factory import (
    get_exchange,
    create_exchange,
    list_available_exchanges,
    is_exchange_available,
    connect_exchange,
)
from .manager import (
    ExchangeManager,
    AggregatedBalance,
    AggregatedPositions,
    get_exchange_manager,
    init_exchange_manager,
)

__all__ = [
    # Interface
    "ExchangeInterface",
    "OrderParams",
    "OrderResult",
    "Position",
    "Balance",
    "ExchangeConfig",
    "ExchangeType",
    # Fee config
    "FeeConfig",
    "get_fee_config",
    "EXCHANGE_FEES",
    # Enums
    "OrderSide",
    "OrderType",
    "PositionSide",
    "MarginMode",
    # Data classes
    "MarketData",
    # Factory
    "get_exchange",
    "create_exchange",
    "list_available_exchanges",
    "is_exchange_available",
    "connect_exchange",
    # Manager
    "ExchangeManager",
    "AggregatedBalance",
    "AggregatedPositions",
    "get_exchange_manager",
    "init_exchange_manager",
]
