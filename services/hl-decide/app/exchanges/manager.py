"""
Exchange Manager

Manages multiple exchange connections and routes execution requests
to the appropriate adapter based on configuration.

Bridges the gap between the abstract exchange interface and the
executor/risk management systems.

@module exchanges.manager
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from .interface import (
    Balance,
    ExchangeConfig,
    ExchangeInterface,
    ExchangeType,
    MarketData,
    OrderParams,
    OrderResult,
    OrderSide,
    OrderType,
    Position,
)
from .factory import create_exchange, get_exchange

logger = logging.getLogger(__name__)


@dataclass
class AggregatedBalance:
    """
    Balance aggregated across all connected exchanges.

    All values are USD-normalized (Phase 6.1).
    USDT balances are converted to USD using AccountNormalizer.
    """
    total_equity: float  # USD-normalized
    available_balance: float  # USD-normalized
    margin_used: float  # USD-normalized
    unrealized_pnl: float  # USD-normalized
    per_exchange: dict[str, Balance]  # Original per-exchange balances
    timestamp: datetime


@dataclass
class AggregatedPositions:
    """Positions aggregated across all connected exchanges."""
    positions: list[Position]
    per_exchange: dict[str, list[Position]]
    total_notional: float
    timestamp: datetime


class ExchangeManager:
    """
    Manages connections to multiple exchanges and routes execution.

    Provides:
    - Connection lifecycle management for multiple exchanges
    - Unified position/balance queries across exchanges
    - Execution routing based on configuration
    - Symbol normalization per exchange
    - Database persistence for connection status and balances (Phase 6)

    Usage:
        manager = ExchangeManager()
        await manager.connect_exchange(ExchangeType.HYPERLIQUID)
        await manager.connect_exchange(ExchangeType.BYBIT)

        # Get aggregated state
        balance = await manager.get_aggregated_balance()
        positions = await manager.get_all_positions()

        # Execute on specific exchange
        result = await manager.execute_order(
            ExchangeType.HYPERLIQUID,
            OrderParams(symbol="BTC", side=OrderSide.BUY, size=0.01)
        )
    """

    def __init__(self, db_pool=None):
        """
        Initialize exchange manager.

        Args:
            db_pool: Optional asyncpg connection pool for telemetry persistence
        """
        self._exchanges: dict[ExchangeType, ExchangeInterface] = {}
        self._default_exchange: Optional[ExchangeType] = None
        self._db_pool = db_pool

    @property
    def connected_exchanges(self) -> list[ExchangeType]:
        """List of connected exchange types."""
        return [
            ex_type for ex_type, ex in self._exchanges.items()
            if ex.is_connected
        ]

    @property
    def default_exchange(self) -> Optional[ExchangeType]:
        """Get default exchange for execution."""
        return self._default_exchange

    @default_exchange.setter
    def default_exchange(self, exchange_type: ExchangeType) -> None:
        """Set default exchange for execution."""
        if exchange_type not in self._exchanges:
            raise ValueError(f"Exchange {exchange_type} not registered")
        self._default_exchange = exchange_type

    def get_exchange(self, exchange_type: ExchangeType) -> Optional[ExchangeInterface]:
        """
        Get exchange adapter by type.

        Args:
            exchange_type: Type of exchange

        Returns:
            Exchange adapter or None if not registered
        """
        return self._exchanges.get(exchange_type)

    async def connect_exchange(
        self,
        exchange_type: ExchangeType,
        testnet: bool = True,
        set_as_default: bool = False,
        **config_overrides,
    ) -> bool:
        """
        Connect to an exchange.

        Args:
            exchange_type: Type of exchange to connect
            testnet: Use testnet (default True)
            set_as_default: Set as default execution exchange
            **config_overrides: Override default config values

        Returns:
            True if connected successfully
        """
        try:
            exchange = get_exchange(exchange_type, testnet, **config_overrides)

            if not exchange.is_configured:
                logger.warning(f"Exchange {exchange_type} not configured (missing credentials)")
                await self.persist_connection_status(exchange_type, False, testnet, "Not configured")
                return False

            if await exchange.connect():
                self._exchanges[exchange_type] = exchange

                if set_as_default or self._default_exchange is None:
                    self._default_exchange = exchange_type

                logger.info(f"Connected to {exchange_type.value}")
                await self.persist_connection_status(exchange_type, True, testnet)
                return True
            else:
                logger.error(f"Failed to connect to {exchange_type.value}")
                await self.persist_connection_status(exchange_type, False, testnet, "Connection failed")
                return False

        except Exception as e:
            logger.error(f"Error connecting to {exchange_type}: {e}")
            await self.persist_connection_status(exchange_type, False, testnet, str(e))
            return False

    async def disconnect_exchange(self, exchange_type: ExchangeType) -> None:
        """
        Disconnect from an exchange.

        Args:
            exchange_type: Type of exchange to disconnect
        """
        exchange = self._exchanges.get(exchange_type)
        if exchange:
            await exchange.disconnect()
            del self._exchanges[exchange_type]

            if self._default_exchange == exchange_type:
                # Set new default if available
                if self._exchanges:
                    self._default_exchange = next(iter(self._exchanges.keys()))
                else:
                    self._default_exchange = None

            await self.persist_connection_status(exchange_type, False, error="Disconnected")
            logger.info(f"Disconnected from {exchange_type.value}")

    async def disconnect_all(self) -> None:
        """Disconnect from all exchanges."""
        for exchange_type in list(self._exchanges.keys()):
            await self.disconnect_exchange(exchange_type)

    # Account State Methods

    async def get_balance(self, exchange_type: ExchangeType) -> Optional[Balance]:
        """
        Get balance for specific exchange.

        Args:
            exchange_type: Exchange to query

        Returns:
            Balance or None if unavailable
        """
        exchange = self._exchanges.get(exchange_type)
        if not exchange or not exchange.is_connected:
            return None
        return await exchange.get_balance()

    async def get_aggregated_balance(self) -> Optional[AggregatedBalance]:
        """
        Get aggregated balance across all connected exchanges.

        All values are USD-normalized (Phase 6.1).
        USDT balances from Bybit are converted to USD using AccountNormalizer.

        Returns:
            AggregatedBalance with USD-normalized totals and per-exchange breakdown
        """
        if not self._exchanges:
            return None

        balances: dict[str, Balance] = {}
        total_equity = 0.0
        available_balance = 0.0
        margin_used = 0.0
        unrealized_pnl = 0.0

        # Lazy import to avoid circular dependency (Phase 6.1)
        from ..account_normalizer import get_account_normalizer
        normalizer = get_account_normalizer()

        for ex_type, exchange in self._exchanges.items():
            if not exchange.is_connected:
                continue

            balance = await exchange.get_balance()
            if balance:
                balances[ex_type.value] = balance

                # Normalize balance to USD (handles USDT -> USD for Bybit)
                normalized = normalizer.normalize_balance_sync(balance)

                total_equity += normalized.total_equity_usd
                available_balance += normalized.available_balance_usd
                margin_used += normalized.margin_used_usd
                unrealized_pnl += normalized.unrealized_pnl_usd

        if not balances:
            return None

        return AggregatedBalance(
            total_equity=total_equity,
            available_balance=available_balance,
            margin_used=margin_used,
            unrealized_pnl=unrealized_pnl,
            per_exchange=balances,
            timestamp=datetime.now(timezone.utc),
        )

    async def get_positions(self, exchange_type: ExchangeType) -> list[Position]:
        """
        Get positions for specific exchange.

        Args:
            exchange_type: Exchange to query

        Returns:
            List of positions
        """
        exchange = self._exchanges.get(exchange_type)
        if not exchange or not exchange.is_connected:
            return []
        return await exchange.get_positions()

    async def get_all_positions(self) -> AggregatedPositions:
        """
        Get positions across all connected exchanges.

        Returns:
            AggregatedPositions with all positions and per-exchange breakdown
        """
        all_positions: list[Position] = []
        per_exchange: dict[str, list[Position]] = {}
        total_notional = 0.0

        for ex_type, exchange in self._exchanges.items():
            if not exchange.is_connected:
                continue

            positions = await exchange.get_positions()
            per_exchange[ex_type.value] = positions
            all_positions.extend(positions)
            total_notional += sum(p.notional_value for p in positions)

        return AggregatedPositions(
            positions=all_positions,
            per_exchange=per_exchange,
            total_notional=total_notional,
            timestamp=datetime.now(timezone.utc),
        )

    async def get_position(
        self,
        symbol: str,
        exchange_type: Optional[ExchangeType] = None,
    ) -> Optional[Position]:
        """
        Get position for a symbol, optionally on specific exchange.

        Args:
            symbol: Trading symbol
            exchange_type: Specific exchange (or search all)

        Returns:
            Position or None
        """
        if exchange_type:
            exchange = self._exchanges.get(exchange_type)
            if exchange and exchange.is_connected:
                return await exchange.get_position(symbol)
            return None

        # Search all exchanges
        for exchange in self._exchanges.values():
            if exchange.is_connected:
                position = await exchange.get_position(symbol)
                if position:
                    return position

        return None

    # Market Data Methods

    async def get_market_price(
        self,
        symbol: str,
        exchange_type: Optional[ExchangeType] = None,
    ) -> Optional[float]:
        """
        Get market price for symbol.

        Args:
            symbol: Trading symbol
            exchange_type: Specific exchange (or use default)

        Returns:
            Mid price or None
        """
        ex_type = exchange_type or self._default_exchange
        if not ex_type:
            return None

        exchange = self._exchanges.get(ex_type)
        if not exchange or not exchange.is_connected:
            return None

        return await exchange.get_market_price(symbol)

    async def get_market_data(
        self,
        symbol: str,
        exchange_type: Optional[ExchangeType] = None,
    ) -> Optional[MarketData]:
        """
        Get full market data for symbol.

        Args:
            symbol: Trading symbol
            exchange_type: Specific exchange (or use default)

        Returns:
            MarketData or None
        """
        ex_type = exchange_type or self._default_exchange
        if not ex_type:
            return None

        exchange = self._exchanges.get(ex_type)
        if not exchange or not exchange.is_connected:
            return None

        return await exchange.get_market_data(symbol)

    # Execution Methods

    async def execute_order(
        self,
        exchange_type: Optional[ExchangeType],
        params: OrderParams,
    ) -> OrderResult:
        """
        Execute order on specified exchange.

        Args:
            exchange_type: Exchange to execute on (or use default)
            params: Order parameters

        Returns:
            OrderResult with execution details
        """
        ex_type = exchange_type or self._default_exchange
        if not ex_type:
            return OrderResult(
                success=False,
                error="No exchange specified and no default set",
                timestamp=datetime.now(timezone.utc),
            )

        exchange = self._exchanges.get(ex_type)
        if not exchange:
            return OrderResult(
                success=False,
                error=f"Exchange {ex_type.value} not registered",
                timestamp=datetime.now(timezone.utc),
            )

        if not exchange.is_connected:
            return OrderResult(
                success=False,
                error=f"Exchange {ex_type.value} not connected",
                timestamp=datetime.now(timezone.utc),
            )

        return await exchange.place_order(params)

    async def open_position(
        self,
        exchange_type: Optional[ExchangeType],
        params: OrderParams,
    ) -> OrderResult:
        """
        Open position on specified exchange.

        Args:
            exchange_type: Exchange to execute on (or use default)
            params: Order parameters

        Returns:
            OrderResult with execution details
        """
        ex_type = exchange_type or self._default_exchange
        if not ex_type:
            return OrderResult(
                success=False,
                error="No exchange specified and no default set",
                timestamp=datetime.now(timezone.utc),
            )

        exchange = self._exchanges.get(ex_type)
        if not exchange or not exchange.is_connected:
            return OrderResult(
                success=False,
                error=f"Exchange {ex_type.value} not available",
                timestamp=datetime.now(timezone.utc),
            )

        return await exchange.open_position(params)

    async def close_position(
        self,
        symbol: str,
        exchange_type: Optional[ExchangeType] = None,
        size: Optional[float] = None,
    ) -> OrderResult:
        """
        Close position on specified exchange.

        Args:
            symbol: Trading symbol
            exchange_type: Exchange (or use default)
            size: Size to close (None for full)

        Returns:
            OrderResult with execution details
        """
        ex_type = exchange_type or self._default_exchange
        if not ex_type:
            return OrderResult(
                success=False,
                error="No exchange specified and no default set",
                timestamp=datetime.now(timezone.utc),
            )

        exchange = self._exchanges.get(ex_type)
        if not exchange or not exchange.is_connected:
            return OrderResult(
                success=False,
                error=f"Exchange {ex_type.value} not available",
                timestamp=datetime.now(timezone.utc),
            )

        return await exchange.close_position(symbol, size)

    async def set_leverage(
        self,
        symbol: str,
        leverage: int,
        exchange_type: Optional[ExchangeType] = None,
    ) -> bool:
        """
        Set leverage on specified exchange.

        Args:
            symbol: Trading symbol
            leverage: Leverage multiplier
            exchange_type: Exchange (or use default)

        Returns:
            True if set successfully
        """
        ex_type = exchange_type or self._default_exchange
        if not ex_type:
            return False

        exchange = self._exchanges.get(ex_type)
        if not exchange or not exchange.is_connected:
            return False

        return await exchange.set_leverage(symbol, leverage)

    async def set_stop_loss(
        self,
        symbol: str,
        stop_price: float,
        exchange_type: Optional[ExchangeType] = None,
        size: Optional[float] = None,
    ) -> OrderResult:
        """
        Set stop loss on specified exchange.

        Args:
            symbol: Trading symbol
            stop_price: Stop trigger price
            exchange_type: Exchange (or use default)
            size: Size to close (None for full)

        Returns:
            OrderResult with stop order details
        """
        ex_type = exchange_type or self._default_exchange
        if not ex_type:
            return OrderResult(
                success=False,
                error="No exchange specified",
                timestamp=datetime.now(timezone.utc),
            )

        exchange = self._exchanges.get(ex_type)
        if not exchange or not exchange.is_connected:
            return OrderResult(
                success=False,
                error=f"Exchange {ex_type.value} not available",
                timestamp=datetime.now(timezone.utc),
            )

        return await exchange.set_stop_loss(symbol, stop_price, size)

    async def set_take_profit(
        self,
        symbol: str,
        take_profit_price: float,
        exchange_type: Optional[ExchangeType] = None,
        size: Optional[float] = None,
    ) -> OrderResult:
        """
        Set take profit on specified exchange.

        Args:
            symbol: Trading symbol
            take_profit_price: Take profit trigger price
            exchange_type: Exchange (or use default)
            size: Size to close (None for full)

        Returns:
            OrderResult with take profit order details
        """
        ex_type = exchange_type or self._default_exchange
        if not ex_type:
            return OrderResult(
                success=False,
                error="No exchange specified",
                timestamp=datetime.now(timezone.utc),
            )

        exchange = self._exchanges.get(ex_type)
        if not exchange or not exchange.is_connected:
            return OrderResult(
                success=False,
                error=f"Exchange {ex_type.value} not available",
                timestamp=datetime.now(timezone.utc),
            )

        return await exchange.set_take_profit(symbol, take_profit_price, size)

    # Utility Methods

    def format_symbol(
        self,
        symbol: str,
        exchange_type: Optional[ExchangeType] = None,
    ) -> str:
        """
        Format symbol for specific exchange.

        Args:
            symbol: Generic symbol (e.g., "BTC")
            exchange_type: Target exchange (or use default)

        Returns:
            Exchange-specific symbol format
        """
        ex_type = exchange_type or self._default_exchange
        if not ex_type:
            return symbol

        exchange = self._exchanges.get(ex_type)
        if not exchange:
            return symbol

        return exchange.format_symbol(symbol)

    def normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol to canonical format (e.g., "BTC").

        Strips exchange-specific suffixes.

        Args:
            symbol: Exchange-specific symbol

        Returns:
            Normalized symbol
        """
        # Remove common suffixes (order matters - check longer patterns first)
        symbol = symbol.upper()
        for suffix in ["-PERP", "/USDT", "/USD", "-USD", "USDT"]:
            symbol = symbol.replace(suffix, "")
        return symbol

    # Database Persistence Methods (Phase 6)

    def set_db_pool(self, db_pool) -> None:
        """Set database pool for telemetry persistence."""
        self._db_pool = db_pool

    async def persist_connection_status(
        self,
        exchange_type: ExchangeType,
        is_connected: bool,
        testnet: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """
        Persist exchange connection status to database.

        Args:
            exchange_type: Exchange identifier
            is_connected: Current connection status
            testnet: Whether using testnet
            error: Error message if failed
        """
        if not self._db_pool:
            return

        try:
            async with self._db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO exchange_connections
                    (exchange_type, testnet, is_connected, last_connected_at, last_error, updated_at)
                    VALUES ($1, $2, $3, CASE WHEN $3 THEN NOW() ELSE NULL END, $4, NOW())
                    ON CONFLICT (exchange_type, testnet) DO UPDATE SET
                        is_connected = EXCLUDED.is_connected,
                        last_connected_at = CASE WHEN EXCLUDED.is_connected THEN NOW() ELSE exchange_connections.last_connected_at END,
                        last_error = EXCLUDED.last_error,
                        updated_at = NOW()
                    """,
                    exchange_type.value,
                    testnet,
                    is_connected,
                    error,
                )
        except Exception as e:
            logger.warning(f"Failed to persist connection status: {e}")

    async def persist_balance(
        self,
        exchange_type: ExchangeType,
        balance: Balance,
    ) -> None:
        """
        Persist exchange balance to database.

        Args:
            exchange_type: Exchange identifier
            balance: Balance data
        """
        if not self._db_pool:
            return

        try:
            async with self._db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO exchange_balances
                    (exchange_type, total_equity, available_balance, margin_used, unrealized_pnl, timestamp)
                    VALUES ($1, $2, $3, $4, $5, NOW())
                    ON CONFLICT (exchange_type) DO UPDATE SET
                        total_equity = EXCLUDED.total_equity,
                        available_balance = EXCLUDED.available_balance,
                        margin_used = EXCLUDED.margin_used,
                        unrealized_pnl = EXCLUDED.unrealized_pnl,
                        timestamp = NOW()
                    """,
                    exchange_type.value,
                    balance.total_equity,
                    balance.available_balance,
                    balance.margin_used,
                    balance.unrealized_pnl,
                )
        except Exception as e:
            logger.warning(f"Failed to persist balance: {e}")

    async def update_all_balances(self) -> None:
        """
        Update balance records for all connected exchanges.

        Call periodically to maintain telemetry data.
        """
        for ex_type, exchange in self._exchanges.items():
            if exchange.is_connected:
                try:
                    balance = await exchange.get_balance()
                    if balance:
                        await self.persist_balance(ex_type, balance)
                except Exception as e:
                    logger.warning(f"Failed to update balance for {ex_type}: {e}")

    async def health_check(
        self,
        testnet: bool = True,
        stagger_delay_ms: int = 500,
    ) -> dict[str, Any]:
        """
        Check health of all exchanges and attempt reconnection if needed.

        Probes each connected exchange by fetching balance/positions.
        If an exchange fails the probe, attempts to reconnect.

        Rate limiting: Adds a configurable delay between probing each exchange
        to prevent hitting API rate limits when multiple exchanges are configured.

        Args:
            testnet: Whether using testnet (for reconnection attempts)
            stagger_delay_ms: Delay in ms between checking each exchange (default 500ms)

        Returns:
            Dict with health status per exchange:
            {
                "hyperliquid": {"connected": True, "healthy": True, "error": None},
                "bybit": {"connected": False, "healthy": False, "error": "Timeout"},
                "reconnected": ["bybit"],
                "timestamp": "2024-01-15T10:30:00Z"
            }
        """
        results: dict[str, Any] = {
            "reconnected": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        exchange_list = list(self._exchanges.items())
        for i, (ex_type, exchange) in enumerate(exchange_list):
            # Rate limiting: add delay between exchanges (except first)
            if i > 0 and stagger_delay_ms > 0:
                await asyncio.sleep(stagger_delay_ms / 1000)
            status = {
                "connected": exchange.is_connected,
                "healthy": False,
                "error": None,
            }

            if not exchange.is_connected:
                # Exchange was disconnected - try to reconnect
                logger.info(f"Health check: {ex_type.value} disconnected, attempting reconnect...")
                try:
                    if await exchange.connect():
                        status["connected"] = True
                        results["reconnected"].append(ex_type.value)
                        logger.info(f"Health check: {ex_type.value} reconnected successfully")
                    else:
                        status["error"] = "Reconnection failed"
                except Exception as e:
                    status["error"] = str(e)
                    logger.warning(f"Health check: {ex_type.value} reconnection error: {e}")

            if status["connected"]:
                # Probe the exchange with a lightweight request
                try:
                    balance = await exchange.get_balance()
                    if balance is not None:
                        status["healthy"] = True
                        # Update balance while we have it
                        await self.persist_balance(ex_type, balance)
                    else:
                        status["error"] = "Balance returned None"
                        status["healthy"] = False
                except Exception as e:
                    status["error"] = str(e)
                    status["healthy"] = False
                    logger.warning(f"Health check: {ex_type.value} probe failed: {e}")

                    # If probe failed, the connection may be stale - try reconnect
                    if exchange.is_connected:
                        logger.info(f"Health check: {ex_type.value} connection stale, reconnecting...")
                        try:
                            await exchange.disconnect()
                            if await exchange.connect():
                                results["reconnected"].append(ex_type.value)
                                status["connected"] = True
                                status["healthy"] = True
                                status["error"] = None
                                logger.info(f"Health check: {ex_type.value} reconnected after stale connection")
                        except Exception as reconnect_error:
                            status["error"] = f"Reconnection failed: {reconnect_error}"
                            logger.warning(f"Health check: {ex_type.value} reconnection failed: {reconnect_error}")

            # Persist connection status
            await self.persist_connection_status(
                ex_type,
                status["healthy"],
                testnet,
                status["error"],
            )

            results[ex_type.value] = status

        return results


# Singleton instance
_exchange_manager: Optional[ExchangeManager] = None


def get_exchange_manager() -> ExchangeManager:
    """
    Get the global exchange manager instance.

    Returns:
        ExchangeManager singleton
    """
    global _exchange_manager
    if _exchange_manager is None:
        _exchange_manager = ExchangeManager()
    return _exchange_manager


async def init_exchange_manager(
    exchanges: Optional[list[ExchangeType]] = None,
    testnet: bool = True,
) -> ExchangeManager:
    """
    Initialize exchange manager with configured exchanges.

    Reads exchange configuration from environment variables.

    Args:
        exchanges: List of exchanges to connect (None = auto-detect from env)
        testnet: Use testnet (default True)

    Returns:
        Initialized ExchangeManager
    """
    manager = get_exchange_manager()

    # Auto-detect which exchanges have credentials
    if exchanges is None:
        exchanges = []

        # Check Hyperliquid
        if os.getenv("HL_PRIVATE_KEY"):
            exchanges.append(ExchangeType.HYPERLIQUID)

        # Check Aster
        if os.getenv("ASTER_PRIVATE_KEY"):
            exchanges.append(ExchangeType.ASTER)

        # Check Bybit
        if os.getenv("BYBIT_API_KEY") and os.getenv("BYBIT_API_SECRET"):
            exchanges.append(ExchangeType.BYBIT)

    # Connect to configured exchanges
    for i, ex_type in enumerate(exchanges):
        await manager.connect_exchange(
            ex_type,
            testnet=testnet,
            set_as_default=(i == 0),  # First exchange is default
        )

    return manager
