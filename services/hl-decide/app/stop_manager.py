"""
Stop-Loss and Take-Profit Management

Phase 4.3: Position Management

Monitors open positions and triggers exits when:
- Price hits stop-loss level
- Price hits take-profit level
- Position times out (configurable)

This module implements local stop monitoring with price polling.
Future enhancement: Native Hyperliquid stop orders.

@module stop_manager
"""

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

import asyncpg

from .hl_exchange import get_exchange, OrderResult


# Configuration
STOP_POLL_INTERVAL_S = int(os.getenv("STOP_POLL_INTERVAL_S", "5"))
DEFAULT_RR_RATIO = float(os.getenv("DEFAULT_RR_RATIO", "2.0"))  # 2:1 reward:risk
MAX_POSITION_HOURS = int(os.getenv("MAX_POSITION_HOURS", "168"))  # 7 days
TRAILING_STOP_ENABLED = os.getenv("TRAILING_STOP_ENABLED", "false").lower() == "true"


@dataclass
class StopConfig:
    """Configuration for a position's stop management."""

    decision_id: str
    symbol: str
    direction: str  # "long" or "short"
    entry_price: float
    entry_size: float
    stop_price: float
    take_profit_price: Optional[float]
    trailing_enabled: bool
    trail_distance_pct: float
    timeout_at: Optional[datetime]
    created_at: datetime


@dataclass
class StopTriggerResult:
    """Result when a stop is triggered."""

    decision_id: str
    trigger_reason: str  # "stop_loss", "take_profit", "timeout", "manual"
    trigger_price: float
    order_result: Optional[OrderResult]


class StopManager:
    """
    Manages stop-loss and take-profit levels for open positions.

    Runs as a background task, polling positions and prices every N seconds.
    When price crosses a stop level, triggers a position close.

    Usage:
        manager = StopManager(db_pool)
        await manager.register_stop(decision_id, symbol, entry_price, stop_pct, direction)
        asyncio.create_task(manager.run_loop())
    """

    def __init__(self, db: asyncpg.Pool):
        """
        Initialize StopManager.

        Args:
            db: Database pool for state persistence
        """
        self.db = db
        self._running = False
        self._poll_interval = STOP_POLL_INTERVAL_S

    async def register_stop(
        self,
        decision_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        entry_size: float,
        stop_distance_pct: float,
        take_profit_rr: float = DEFAULT_RR_RATIO,
        trailing_enabled: bool = TRAILING_STOP_ENABLED,
        timeout_hours: int = MAX_POSITION_HOURS,
    ) -> StopConfig:
        """
        Register a stop for a newly opened position.

        Args:
            decision_id: ID of the decision that opened this position
            symbol: Asset symbol (BTC, ETH)
            direction: Position direction (long, short)
            entry_price: Entry price
            entry_size: Position size in coins
            stop_distance_pct: Stop distance as fraction (0.02 = 2%)
            take_profit_rr: Reward:risk ratio for take-profit
            trailing_enabled: Whether to trail the stop
            timeout_hours: Maximum hours to hold position

        Returns:
            StopConfig with all stop parameters
        """
        # Calculate stop and take-profit prices
        stop_price = self._calculate_stop_price(
            entry_price, direction, stop_distance_pct
        )
        take_profit_price = self._calculate_take_profit(
            entry_price, stop_price, direction, take_profit_rr
        )

        # Calculate timeout
        timeout_at = datetime.now(timezone.utc) + timedelta(hours=timeout_hours)

        config = StopConfig(
            decision_id=decision_id,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            entry_size=entry_size,
            stop_price=stop_price,
            take_profit_price=take_profit_price,
            trailing_enabled=trailing_enabled,
            trail_distance_pct=stop_distance_pct,
            timeout_at=timeout_at,
            created_at=datetime.now(timezone.utc),
        )

        # Persist to database
        await self._save_stop(config)

        print(
            f"[stop_manager] Registered stop for {decision_id}: "
            f"{symbol} {direction} entry=${entry_price:.2f}, "
            f"stop=${stop_price:.2f}, tp=${take_profit_price:.2f if take_profit_price else 'None'}"
        )

        return config

    def _calculate_stop_price(
        self,
        entry_price: float,
        direction: str,
        stop_distance_pct: float,
    ) -> float:
        """Calculate stop-loss price based on direction and distance."""
        if direction == "long":
            return entry_price * (1 - stop_distance_pct)
        else:
            return entry_price * (1 + stop_distance_pct)

    def _calculate_take_profit(
        self,
        entry_price: float,
        stop_price: float,
        direction: str,
        rr_ratio: float,
    ) -> Optional[float]:
        """Calculate take-profit price based on reward:risk ratio."""
        if rr_ratio <= 0:
            return None

        stop_distance = abs(entry_price - stop_price)
        profit_distance = stop_distance * rr_ratio

        if direction == "long":
            return entry_price + profit_distance
        else:
            return entry_price - profit_distance

    async def _save_stop(self, config: StopConfig) -> None:
        """Save stop configuration to database."""
        try:
            async with self.db.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO active_stops
                    (decision_id, symbol, direction, entry_price, entry_size,
                     stop_price, take_profit_price, trailing_enabled,
                     trail_distance_pct, timeout_at, status)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, 'active')
                    ON CONFLICT (symbol, decision_id) DO UPDATE SET
                        stop_price = EXCLUDED.stop_price,
                        take_profit_price = EXCLUDED.take_profit_price,
                        status = 'active'
                    """,
                    config.decision_id,
                    config.symbol,
                    config.direction,
                    config.entry_price,
                    config.entry_size,
                    config.stop_price,
                    config.take_profit_price,
                    config.trailing_enabled,
                    config.trail_distance_pct,
                    config.timeout_at,
                )
        except Exception as e:
            print(f"[stop_manager] Failed to save stop: {e}")

    async def get_active_stops(self) -> list[StopConfig]:
        """Get all active stops from database."""
        try:
            async with self.db.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT decision_id, symbol, direction, entry_price, entry_size,
                           stop_price, take_profit_price, trailing_enabled,
                           trail_distance_pct, timeout_at, created_at
                    FROM active_stops
                    WHERE status = 'active'
                    """
                )
                return [
                    StopConfig(
                        decision_id=str(row["decision_id"]),
                        symbol=row["symbol"],
                        direction=row["direction"],
                        entry_price=float(row["entry_price"]),
                        entry_size=float(row["entry_size"]),
                        stop_price=float(row["stop_price"]),
                        take_profit_price=(
                            float(row["take_profit_price"])
                            if row["take_profit_price"]
                            else None
                        ),
                        trailing_enabled=row["trailing_enabled"],
                        trail_distance_pct=float(row["trail_distance_pct"] or 0.02),
                        timeout_at=row["timeout_at"],
                        created_at=row["created_at"],
                    )
                    for row in rows
                ]
        except Exception as e:
            print(f"[stop_manager] Failed to get active stops: {e}")
            return []

    async def check_stops(self) -> list[StopTriggerResult]:
        """
        Check all active stops against current prices.

        Returns:
            List of triggered stops
        """
        triggered = []
        stops = await self.get_active_stops()

        if not stops:
            return triggered

        # Get current prices
        exchange = get_exchange()
        prices = {}
        for symbol in set(s.symbol for s in stops):
            price = await exchange.get_mid_price(symbol)
            if price:
                prices[symbol] = price

        now = datetime.now(timezone.utc)

        for stop in stops:
            current_price = prices.get(stop.symbol)
            if not current_price:
                continue

            trigger_reason = None
            trigger_price = current_price

            # Check timeout first
            if stop.timeout_at and now >= stop.timeout_at:
                trigger_reason = "timeout"

            # Check stop-loss
            elif self._is_stop_hit(stop, current_price):
                trigger_reason = "stop_loss"

            # Check take-profit
            elif stop.take_profit_price and self._is_tp_hit(stop, current_price):
                trigger_reason = "take_profit"

            # Update trailing stop if enabled and price favorable
            elif stop.trailing_enabled:
                await self._maybe_update_trailing(stop, current_price)

            if trigger_reason:
                result = await self._trigger_stop(stop, trigger_reason, trigger_price)
                triggered.append(result)

        return triggered

    def _is_stop_hit(self, stop: StopConfig, current_price: float) -> bool:
        """Check if stop-loss is hit."""
        if stop.direction == "long":
            return current_price <= stop.stop_price
        else:
            return current_price >= stop.stop_price

    def _is_tp_hit(self, stop: StopConfig, current_price: float) -> bool:
        """Check if take-profit is hit."""
        if not stop.take_profit_price:
            return False
        if stop.direction == "long":
            return current_price >= stop.take_profit_price
        else:
            return current_price <= stop.take_profit_price

    async def _maybe_update_trailing(
        self, stop: StopConfig, current_price: float
    ) -> None:
        """Update trailing stop if price has moved favorably."""
        # Calculate new stop based on current price
        new_stop = self._calculate_stop_price(
            current_price, stop.direction, stop.trail_distance_pct
        )

        # Only move stop in favorable direction
        should_update = False
        if stop.direction == "long" and new_stop > stop.stop_price:
            should_update = True
        elif stop.direction == "short" and new_stop < stop.stop_price:
            should_update = True

        if should_update:
            try:
                async with self.db.acquire() as conn:
                    await conn.execute(
                        """
                        UPDATE active_stops
                        SET stop_price = $1
                        WHERE decision_id = $2 AND symbol = $3 AND status = 'active'
                        """,
                        new_stop,
                        stop.decision_id,
                        stop.symbol,
                    )
                print(
                    f"[stop_manager] Trailing stop updated: "
                    f"{stop.symbol} {stop.direction} {stop.stop_price:.2f} -> {new_stop:.2f}"
                )
            except Exception as e:
                print(f"[stop_manager] Failed to update trailing stop: {e}")

    async def _trigger_stop(
        self,
        stop: StopConfig,
        reason: str,
        trigger_price: float,
    ) -> StopTriggerResult:
        """Execute stop trigger and close position."""
        print(
            f"[stop_manager] STOP TRIGGERED: {stop.symbol} {stop.direction} "
            f"reason={reason} price=${trigger_price:.2f}"
        )

        # Close the position
        exchange = get_exchange()
        order_result = await exchange.close_position(stop.symbol)

        # Update database
        try:
            async with self.db.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE active_stops
                    SET status = 'triggered',
                        triggered_at = NOW(),
                        triggered_price = $1,
                        triggered_reason = $2
                    WHERE decision_id = $3 AND symbol = $4
                    """,
                    trigger_price,
                    reason,
                    stop.decision_id,
                    stop.symbol,
                )
        except Exception as e:
            print(f"[stop_manager] Failed to update triggered stop: {e}")

        return StopTriggerResult(
            decision_id=stop.decision_id,
            trigger_reason=reason,
            trigger_price=trigger_price,
            order_result=order_result,
        )

    async def cancel_stop(self, decision_id: str, symbol: str) -> bool:
        """
        Cancel an active stop (e.g., when position manually closed).

        Args:
            decision_id: Decision ID
            symbol: Asset symbol

        Returns:
            True if cancelled
        """
        try:
            async with self.db.acquire() as conn:
                result = await conn.execute(
                    """
                    UPDATE active_stops
                    SET status = 'cancelled'
                    WHERE decision_id = $1 AND symbol = $2 AND status = 'active'
                    """,
                    decision_id,
                    symbol,
                )
                return "UPDATE 1" in result
        except Exception as e:
            print(f"[stop_manager] Failed to cancel stop: {e}")
            return False

    async def run_loop(self) -> None:
        """
        Main monitoring loop.

        Runs continuously, checking stops at regular intervals.
        """
        self._running = True
        print(f"[stop_manager] Starting with {self._poll_interval}s poll interval")

        while self._running:
            try:
                triggered = await self.check_stops()
                if triggered:
                    print(f"[stop_manager] {len(triggered)} stops triggered")
            except Exception as e:
                print(f"[stop_manager] Error in check loop: {e}")

            await asyncio.sleep(self._poll_interval)

    def stop(self) -> None:
        """Stop the monitoring loop."""
        self._running = False
        print("[stop_manager] Stopping...")


# Global stop manager instance
_stop_manager: Optional[StopManager] = None


def get_stop_manager(db: asyncpg.Pool) -> StopManager:
    """Get or create global stop manager instance."""
    global _stop_manager
    if _stop_manager is None:
        _stop_manager = StopManager(db)
    return _stop_manager
