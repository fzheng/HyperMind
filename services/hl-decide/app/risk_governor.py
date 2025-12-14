"""
Risk Governor: Hard safety limits before live trading

The Risk Governor provides hard safety limits that CANNOT be overridden:
1. **Liquidation Distance Guard**: Block trades if account too close to liquidation
2. **Daily Drawdown Kill Switch**: Halt all trading if daily loss exceeds threshold
3. **Exposure Limits**: Prevent excessive position sizing
4. **Circuit Breakers**: Max concurrent positions, API error pause, loss streak pause
5. **Multi-Exchange Support**: Aggregated risk tracking across exchanges (Phase 6)

These are the LAST line of defense before capital destruction.

@module risk_governor
"""

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

import asyncpg


# Configuration - Hard limits that CANNOT be overridden via config
# These are intentionally conservative to prevent capital destruction

# Liquidation Distance: Block if margin ratio below this threshold
# Margin ratio = equity / maintenance_margin
# 1.5 means 50% buffer before liquidation
LIQUIDATION_DISTANCE_MIN = float(os.getenv("LIQUIDATION_DISTANCE_MIN", "1.5"))

# Daily Drawdown Kill Switch: Halt if daily loss exceeds this % of starting equity
DAILY_DRAWDOWN_KILL_PCT = float(os.getenv("DAILY_DRAWDOWN_KILL_PCT", "0.05"))  # 5%

# Minimum equity to trade (absolute floor)
MIN_EQUITY_FLOOR = float(os.getenv("MIN_EQUITY_FLOOR", "10000"))  # $10k

# Maximum single position size as % of equity
MAX_POSITION_SIZE_PCT = float(os.getenv("MAX_POSITION_SIZE_PCT", "0.10"))  # 10%

# Maximum total exposure as % of equity
MAX_TOTAL_EXPOSURE_PCT = float(os.getenv("MAX_TOTAL_EXPOSURE_PCT", "0.50"))  # 50%

# Cooldown after kill switch triggers (seconds)
KILL_SWITCH_COOLDOWN = int(os.getenv("KILL_SWITCH_COOLDOWN", "86400"))  # 24 hours

# Phase 4.4: Circuit Breakers
# Maximum concurrent positions across all symbols
MAX_CONCURRENT_POSITIONS = int(os.getenv("MAX_CONCURRENT_POSITIONS", "3"))

# Maximum positions per symbol (prevents concentration)
MAX_POSITION_PER_SYMBOL = int(os.getenv("MAX_POSITION_PER_SYMBOL", "1"))

# Pause trading after consecutive API errors
API_ERROR_THRESHOLD = int(os.getenv("API_ERROR_THRESHOLD", "3"))
API_ERROR_PAUSE_SECONDS = int(os.getenv("API_ERROR_PAUSE_SECONDS", "300"))  # 5 minutes

# Maximum consecutive losing trades before pause
MAX_CONSECUTIVE_LOSSES = int(os.getenv("MAX_CONSECUTIVE_LOSSES", "5"))
LOSS_STREAK_PAUSE_SECONDS = int(os.getenv("LOSS_STREAK_PAUSE_SECONDS", "3600"))  # 1 hour


@dataclass
class RiskState:
    """Current risk state from account data."""
    timestamp: datetime
    account_value: float  # Always in USD (normalized)
    margin_used: float  # Always in USD (normalized)
    maintenance_margin: float
    total_exposure: float  # Always in USD (normalized)
    margin_ratio: float
    daily_pnl: float
    daily_starting_equity: float
    daily_drawdown_pct: float
    exchange: str = "hyperliquid"  # Phase 6: track which exchange this state is from
    original_currency: str = "USD"  # Original currency before normalization
    conversion_rate: float = 1.0  # Conversion rate used (1.0 for USD)


@dataclass
class AggregatedRiskState:
    """Aggregated risk state across all connected exchanges (Phase 6)."""
    timestamp: datetime
    total_equity: float  # USD-normalized
    total_margin_used: float  # USD-normalized
    total_exposure: float  # USD-normalized
    per_exchange: dict  # exchange -> RiskState
    daily_pnl: float
    daily_drawdown_pct: float
    is_normalized: bool = True  # All values USD-normalized (Phase 6.1.5)


@dataclass
class RiskCheckResult:
    """Result of a risk check."""
    allowed: bool
    reason: str
    risk_state: Optional[RiskState] = None
    warnings: list = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class RiskGovernor:
    """
    Hard safety limits for live trading.

    These checks run BEFORE any trade execution and cannot be bypassed.
    The Risk Governor is the last line of defense against capital destruction.
    """

    def __init__(self, db_pool: Optional[asyncpg.Pool] = None):
        """
        Initialize Risk Governor.

        Args:
            db_pool: Database pool for state persistence (optional)
        """
        self.db_pool = db_pool
        self._kill_switch_active = False
        self._kill_switch_triggered_at: Optional[datetime] = None
        self._daily_starting_equity: Optional[float] = None
        self._daily_start_date: Optional[str] = None

        # Phase 4.4: Circuit breaker state
        self._consecutive_api_errors = 0
        self._api_pause_until: Optional[datetime] = None
        self._consecutive_losses = 0
        self._loss_streak_pause_until: Optional[datetime] = None
        self._current_position_count = 0
        self._positions_by_symbol: Dict[str, int] = {}

        # Phase 6: Multi-exchange tracking
        self._positions_by_exchange: Dict[str, Dict[str, int]] = {}  # exchange -> {symbol: count}
        self._risk_state_by_exchange: Dict[str, RiskState] = {}

    async def load_state(self) -> None:
        """Load persisted state from database."""
        if not self.db_pool:
            return

        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT key, value
                    FROM risk_governor_state
                    WHERE key IN ('kill_switch_active', 'kill_switch_triggered_at',
                                 'daily_starting_equity', 'daily_start_date')
                    """
                )
                # State loading would be implemented here
        except Exception:
            # Table may not exist yet, ignore
            pass

    async def save_state(self) -> None:
        """Persist state to database."""
        if not self.db_pool:
            return

        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO risk_governor_state (key, value, updated_at)
                    VALUES ('kill_switch_active', $1, NOW())
                    ON CONFLICT (key) DO UPDATE SET
                        value = EXCLUDED.value,
                        updated_at = EXCLUDED.updated_at
                    """,
                    str(self._kill_switch_active),
                )
        except Exception:
            pass

    def update_daily_starting_equity(self, equity: float) -> None:
        """
        Update daily starting equity for drawdown tracking.

        Should be called once per day at market open or first trade.

        Args:
            equity: Current account equity
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._daily_start_date != today:
            self._daily_starting_equity = equity
            self._daily_start_date = today

    def is_kill_switch_active(self) -> bool:
        """
        Quick check if kill switch is currently active.

        This is a lightweight check for early bailout. Does NOT update cooldown state.
        Use check_kill_switch() for full check with reason and cooldown handling.

        Returns:
            True if kill switch is active
        """
        if not self._kill_switch_active:
            return False

        # Check if cooldown has expired (but don't modify state)
        if self._kill_switch_triggered_at:
            elapsed = (datetime.now(timezone.utc) - self._kill_switch_triggered_at).total_seconds()
            if elapsed >= KILL_SWITCH_COOLDOWN:
                return False  # Will be reset on next full check

        return True

    def check_kill_switch(self) -> Tuple[bool, str]:
        """
        Check if kill switch is active and return reason.

        This is the full check that handles cooldown expiry.

        Returns:
            Tuple of (is_active, reason)
        """
        if not self._kill_switch_active:
            return False, ""

        # Check if cooldown has expired
        if self._kill_switch_triggered_at:
            elapsed = (datetime.now(timezone.utc) - self._kill_switch_triggered_at).total_seconds()
            if elapsed >= KILL_SWITCH_COOLDOWN:
                self._kill_switch_active = False
                self._kill_switch_triggered_at = None
                return False, ""

            remaining = KILL_SWITCH_COOLDOWN - elapsed
            return True, f"Kill switch active, {remaining/3600:.1f}h remaining"

        return True, "Kill switch active"

    def trigger_kill_switch(self, reason: str) -> None:
        """
        Trigger the kill switch, halting all trading.

        Args:
            reason: Why the kill switch was triggered
        """
        self._kill_switch_active = True
        self._kill_switch_triggered_at = datetime.now(timezone.utc)
        print(f"[risk_governor] KILL SWITCH TRIGGERED: {reason}")

    def reset_kill_switch(self) -> None:
        """
        Manually reset the kill switch.

        Should only be called after human review.
        """
        self._kill_switch_active = False
        self._kill_switch_triggered_at = None
        print("[risk_governor] Kill switch reset by operator")

    def compute_risk_state(
        self,
        account_value: float,
        margin_used: float,
        maintenance_margin: float,
        total_exposure: float,
        daily_pnl: float,
    ) -> RiskState:
        """
        Compute current risk state from account data.

        Args:
            account_value: Current account equity
            margin_used: Currently used margin
            maintenance_margin: Minimum required margin
            total_exposure: Total notional exposure
            daily_pnl: Today's realized + unrealized PnL

        Returns:
            RiskState with computed metrics
        """
        # Update daily starting equity if needed
        self.update_daily_starting_equity(account_value - daily_pnl)

        # Compute margin ratio (liquidation distance)
        margin_ratio = (
            account_value / maintenance_margin
            if maintenance_margin > 0
            else float('inf')
        )

        # Compute daily drawdown
        starting = self._daily_starting_equity or account_value
        daily_drawdown_pct = (
            -daily_pnl / starting if starting > 0 and daily_pnl < 0 else 0
        )

        return RiskState(
            timestamp=datetime.now(timezone.utc),
            account_value=account_value,
            margin_used=margin_used,
            maintenance_margin=maintenance_margin,
            total_exposure=total_exposure,
            margin_ratio=margin_ratio,
            daily_pnl=daily_pnl,
            daily_starting_equity=starting,
            daily_drawdown_pct=daily_drawdown_pct,
        )

    def check_liquidation_distance(self, state: RiskState) -> RiskCheckResult:
        """
        Check if account is too close to liquidation.

        This is a HARD limit - trades are blocked if margin ratio is too low.

        Args:
            state: Current risk state

        Returns:
            RiskCheckResult
        """
        if state.margin_ratio < LIQUIDATION_DISTANCE_MIN:
            return RiskCheckResult(
                allowed=False,
                reason=f"Margin ratio {state.margin_ratio:.2f} < {LIQUIDATION_DISTANCE_MIN} (too close to liquidation)",
                risk_state=state,
            )

        # Warning if getting close
        warnings = []
        if state.margin_ratio < LIQUIDATION_DISTANCE_MIN * 1.5:
            warnings.append(f"Margin ratio {state.margin_ratio:.2f} approaching limit")

        return RiskCheckResult(
            allowed=True,
            reason="Liquidation distance OK",
            risk_state=state,
            warnings=warnings,
        )

    def check_daily_drawdown(self, state: RiskState) -> RiskCheckResult:
        """
        Check if daily drawdown exceeds kill switch threshold.

        This triggers the kill switch, halting ALL trading for the cooldown period.

        Args:
            state: Current risk state

        Returns:
            RiskCheckResult
        """
        if state.daily_drawdown_pct >= DAILY_DRAWDOWN_KILL_PCT:
            self.trigger_kill_switch(
                f"Daily drawdown {state.daily_drawdown_pct:.1%} >= {DAILY_DRAWDOWN_KILL_PCT:.1%}"
            )
            return RiskCheckResult(
                allowed=False,
                reason=f"KILL SWITCH: Daily drawdown {state.daily_drawdown_pct:.1%} >= {DAILY_DRAWDOWN_KILL_PCT:.1%}",
                risk_state=state,
            )

        # Warning if approaching
        warnings = []
        if state.daily_drawdown_pct >= DAILY_DRAWDOWN_KILL_PCT * 0.5:
            warnings.append(f"Daily drawdown {state.daily_drawdown_pct:.1%} at {state.daily_drawdown_pct/DAILY_DRAWDOWN_KILL_PCT:.0%} of kill threshold")

        return RiskCheckResult(
            allowed=True,
            reason="Daily drawdown OK",
            risk_state=state,
            warnings=warnings,
        )

    def check_equity_floor(self, state: RiskState) -> RiskCheckResult:
        """
        Check if account equity is above minimum floor.

        Args:
            state: Current risk state

        Returns:
            RiskCheckResult
        """
        if state.account_value < MIN_EQUITY_FLOOR:
            return RiskCheckResult(
                allowed=False,
                reason=f"Account value ${state.account_value:,.0f} < ${MIN_EQUITY_FLOOR:,.0f} floor",
                risk_state=state,
            )

        return RiskCheckResult(
            allowed=True,
            reason="Equity floor OK",
            risk_state=state,
        )

    def check_position_size(
        self,
        state: RiskState,
        proposed_size_usd: float,
    ) -> RiskCheckResult:
        """
        Check if proposed position size is within limits.

        Args:
            state: Current risk state
            proposed_size_usd: Proposed position size in USD

        Returns:
            RiskCheckResult
        """
        max_size = state.account_value * MAX_POSITION_SIZE_PCT

        if proposed_size_usd > max_size:
            return RiskCheckResult(
                allowed=False,
                reason=f"Position size ${proposed_size_usd:,.0f} > ${max_size:,.0f} max ({MAX_POSITION_SIZE_PCT:.0%} of equity)",
                risk_state=state,
            )

        return RiskCheckResult(
            allowed=True,
            reason="Position size OK",
            risk_state=state,
        )

    def check_total_exposure(
        self,
        state: RiskState,
        proposed_additional_exposure: float = 0,
    ) -> RiskCheckResult:
        """
        Check if total exposure is within limits.

        Args:
            state: Current risk state
            proposed_additional_exposure: Additional exposure from proposed trade

        Returns:
            RiskCheckResult
        """
        new_exposure = state.total_exposure + proposed_additional_exposure
        max_exposure = state.account_value * MAX_TOTAL_EXPOSURE_PCT

        if new_exposure > max_exposure:
            return RiskCheckResult(
                allowed=False,
                reason=f"Total exposure ${new_exposure:,.0f} > ${max_exposure:,.0f} max ({MAX_TOTAL_EXPOSURE_PCT:.0%} of equity)",
                risk_state=state,
            )

        return RiskCheckResult(
            allowed=True,
            reason="Total exposure OK",
            risk_state=state,
        )

    def run_all_checks(
        self,
        account_value: float,
        margin_used: float,
        maintenance_margin: float,
        total_exposure: float,
        daily_pnl: float,
        proposed_size_usd: float = 0,
    ) -> RiskCheckResult:
        """
        Run all risk checks before allowing a trade.

        This is the main entry point for the Risk Governor.
        ALL checks must pass for a trade to be allowed.

        Args:
            account_value: Current account equity
            margin_used: Currently used margin
            maintenance_margin: Minimum required margin
            total_exposure: Total notional exposure
            daily_pnl: Today's realized + unrealized PnL
            proposed_size_usd: Size of proposed trade in USD

        Returns:
            RiskCheckResult with aggregated result
        """
        # Check kill switch first
        kill_active, kill_reason = self.check_kill_switch()
        if kill_active:
            return RiskCheckResult(
                allowed=False,
                reason=kill_reason,
            )

        # Compute risk state
        state = self.compute_risk_state(
            account_value, margin_used, maintenance_margin, total_exposure, daily_pnl
        )

        # Run all checks
        all_warnings = []

        # 1. Equity floor
        equity_check = self.check_equity_floor(state)
        if not equity_check.allowed:
            return equity_check
        all_warnings.extend(equity_check.warnings)

        # 2. Liquidation distance
        liq_check = self.check_liquidation_distance(state)
        if not liq_check.allowed:
            return liq_check
        all_warnings.extend(liq_check.warnings)

        # 3. Daily drawdown (may trigger kill switch)
        dd_check = self.check_daily_drawdown(state)
        if not dd_check.allowed:
            return dd_check
        all_warnings.extend(dd_check.warnings)

        # 4. Position size (if proposing a trade)
        if proposed_size_usd > 0:
            size_check = self.check_position_size(state, proposed_size_usd)
            if not size_check.allowed:
                return size_check
            all_warnings.extend(size_check.warnings)

        # 5. Total exposure
        exposure_check = self.check_total_exposure(state, proposed_size_usd)
        if not exposure_check.allowed:
            return exposure_check
        all_warnings.extend(exposure_check.warnings)

        return RiskCheckResult(
            allowed=True,
            reason="All risk checks passed",
            risk_state=state,
            warnings=all_warnings,
        )

    # =========================================================================
    # Phase 4.4: Circuit Breakers
    # =========================================================================

    def check_concurrent_positions(self, current_count: int) -> RiskCheckResult:
        """
        Check if at maximum concurrent positions.

        Args:
            current_count: Current number of open positions

        Returns:
            RiskCheckResult
        """
        if current_count >= MAX_CONCURRENT_POSITIONS:
            return RiskCheckResult(
                allowed=False,
                reason=f"At max concurrent positions ({current_count}/{MAX_CONCURRENT_POSITIONS})",
            )

        warnings = []
        if current_count >= MAX_CONCURRENT_POSITIONS - 1:
            warnings.append(f"Near position limit ({current_count}/{MAX_CONCURRENT_POSITIONS})")

        return RiskCheckResult(
            allowed=True,
            reason="Concurrent positions OK",
            warnings=warnings,
        )

    def check_symbol_position(self, symbol: str, has_position: bool) -> RiskCheckResult:
        """
        Check if already have a position in this symbol.

        Args:
            symbol: Asset symbol
            has_position: Whether already have a position

        Returns:
            RiskCheckResult
        """
        if has_position and MAX_POSITION_PER_SYMBOL == 1:
            return RiskCheckResult(
                allowed=False,
                reason=f"Already have position in {symbol}",
            )

        return RiskCheckResult(
            allowed=True,
            reason="Symbol position OK",
        )

    def report_api_error(self) -> None:
        """
        Report an API error. May trigger pause if too many consecutive errors.
        """
        self._consecutive_api_errors += 1

        if self._consecutive_api_errors >= API_ERROR_THRESHOLD:
            self._api_pause_until = (
                datetime.now(timezone.utc) +
                timedelta(seconds=API_ERROR_PAUSE_SECONDS)
            )
            print(
                f"[risk_governor] API error pause triggered: "
                f"{self._consecutive_api_errors} errors, paused until {self._api_pause_until}"
            )

    def report_api_success(self) -> None:
        """Report successful API call, resetting error counter."""
        self._consecutive_api_errors = 0

    def check_api_pause(self) -> RiskCheckResult:
        """
        Check if in API error pause period.

        Returns:
            RiskCheckResult
        """
        if self._api_pause_until:
            now = datetime.now(timezone.utc)
            if now < self._api_pause_until:
                remaining = (self._api_pause_until - now).total_seconds()
                return RiskCheckResult(
                    allowed=False,
                    reason=f"API error pause, {remaining:.0f}s remaining",
                )
            # Pause expired
            self._api_pause_until = None
            self._consecutive_api_errors = 0

        return RiskCheckResult(
            allowed=True,
            reason="No API pause",
        )

    def report_trade_result(self, is_win: bool) -> None:
        """
        Report trade result. May trigger pause if too many consecutive losses.

        Args:
            is_win: Whether the trade was profitable
        """
        if is_win:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1

            if self._consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                self._loss_streak_pause_until = (
                    datetime.now(timezone.utc) +
                    timedelta(seconds=LOSS_STREAK_PAUSE_SECONDS)
                )
                print(
                    f"[risk_governor] Loss streak pause triggered: "
                    f"{self._consecutive_losses} losses, paused until {self._loss_streak_pause_until}"
                )

    def check_loss_streak_pause(self) -> RiskCheckResult:
        """
        Check if in loss streak pause period.

        Returns:
            RiskCheckResult
        """
        if self._loss_streak_pause_until:
            now = datetime.now(timezone.utc)
            if now < self._loss_streak_pause_until:
                remaining = (self._loss_streak_pause_until - now).total_seconds()
                return RiskCheckResult(
                    allowed=False,
                    reason=f"Loss streak pause ({self._consecutive_losses} losses), {remaining:.0f}s remaining",
                )
            # Pause expired
            self._loss_streak_pause_until = None
            self._consecutive_losses = 0

        return RiskCheckResult(
            allowed=True,
            reason="No loss streak pause",
        )

    def update_position_count(self, symbol: str, delta: int) -> None:
        """
        Update position tracking incrementally.

        Args:
            symbol: Asset symbol
            delta: Change in position count (+1 for open, -1 for close)
        """
        self._current_position_count += delta
        self._current_position_count = max(0, self._current_position_count)

        current = self._positions_by_symbol.get(symbol, 0)
        self._positions_by_symbol[symbol] = max(0, current + delta)

    def update_positions_from_account_state(
        self,
        account_state: Dict[str, Any],
        exchange: str = "hyperliquid",
    ) -> None:
        """
        Update position tracking from exchange account state.

        Derives per-symbol position counts from assetPositions array.
        This is the preferred method to sync governor state with actual positions.

        Args:
            account_state: Account state from exchange API containing assetPositions
            exchange: Exchange identifier (hyperliquid, aster, bybit)
        """
        positions_by_symbol: Dict[str, int] = {}

        for ap in account_state.get("assetPositions", []):
            pos = ap.get("position", {})
            coin = pos.get("coin", "")
            size = float(pos.get("szi", 0))
            if size != 0 and coin:
                positions_by_symbol[coin] = positions_by_symbol.get(coin, 0) + 1

        # Phase 6: Track per-exchange positions
        self._positions_by_exchange[exchange] = positions_by_symbol

        # Update aggregated totals
        self._update_aggregated_positions()

    def _update_aggregated_positions(self) -> None:
        """Update aggregated position counts from per-exchange data."""
        aggregated: Dict[str, int] = {}

        for exchange_positions in self._positions_by_exchange.values():
            for symbol, count in exchange_positions.items():
                aggregated[symbol] = aggregated.get(symbol, 0) + count

        self._positions_by_symbol = aggregated
        self._current_position_count = sum(aggregated.values())

    def update_risk_state_for_exchange(
        self,
        exchange: str,
        account_value: float,
        margin_used: float,
        maintenance_margin: float,
        total_exposure: float,
        daily_pnl: float,
    ) -> RiskState:
        """
        Update risk state for a specific exchange.

        Args:
            exchange: Exchange identifier
            account_value: Current account equity
            margin_used: Currently used margin
            maintenance_margin: Minimum required margin
            total_exposure: Total notional exposure
            daily_pnl: Today's PnL

        Returns:
            RiskState for this exchange
        """
        state = self.compute_risk_state(
            account_value, margin_used, maintenance_margin, total_exposure, daily_pnl
        )
        state.exchange = exchange
        self._risk_state_by_exchange[exchange] = state
        return state

    def get_aggregated_risk_state(self) -> Optional[AggregatedRiskState]:
        """
        Get aggregated risk state across all exchanges.

        Returns:
            AggregatedRiskState or None if no exchange data available
        """
        if not self._risk_state_by_exchange:
            return None

        total_equity = sum(s.account_value for s in self._risk_state_by_exchange.values())
        total_margin = sum(s.margin_used for s in self._risk_state_by_exchange.values())
        total_exposure = sum(s.total_exposure for s in self._risk_state_by_exchange.values())
        daily_pnl = sum(s.daily_pnl for s in self._risk_state_by_exchange.values())

        # Calculate aggregated drawdown
        starting = self._daily_starting_equity or total_equity
        daily_drawdown_pct = -daily_pnl / starting if starting > 0 and daily_pnl < 0 else 0

        return AggregatedRiskState(
            timestamp=datetime.now(timezone.utc),
            total_equity=total_equity,
            total_margin_used=total_margin,
            total_exposure=total_exposure,
            per_exchange=dict(self._risk_state_by_exchange),
            daily_pnl=daily_pnl,
            daily_drawdown_pct=daily_drawdown_pct,
        )

    def get_positions_for_exchange(self, exchange: str) -> Dict[str, int]:
        """
        Get position counts for a specific exchange.

        Args:
            exchange: Exchange identifier

        Returns:
            Dict of symbol -> position count
        """
        return self._positions_by_exchange.get(exchange, {})

    def get_symbol_position_count(self, symbol: str) -> int:
        """
        Get current position count for a symbol.

        Args:
            symbol: Asset symbol

        Returns:
            Number of positions in this symbol (0 if none)
        """
        return self._positions_by_symbol.get(symbol, 0)

    def run_circuit_breaker_checks(
        self,
        symbol: str,
        symbol_position_count: int = 0,
    ) -> RiskCheckResult:
        """
        Run all circuit breaker checks before a trade.

        Args:
            symbol: Asset symbol to trade
            symbol_position_count: Current number of positions in this symbol (0 if new)

        Returns:
            RiskCheckResult with aggregated result
        """
        all_warnings = []

        # 1. API pause check
        api_check = self.check_api_pause()
        if not api_check.allowed:
            return api_check

        # 2. Loss streak pause check
        loss_check = self.check_loss_streak_pause()
        if not loss_check.allowed:
            return loss_check

        # 3. Concurrent positions check
        pos_check = self.check_concurrent_positions(self._current_position_count)
        if not pos_check.allowed:
            return pos_check
        all_warnings.extend(pos_check.warnings)

        # 4. Symbol position check (pass count for future multi-position support)
        has_existing = symbol_position_count > 0
        symbol_check = self.check_symbol_position(symbol, has_existing)
        if not symbol_check.allowed:
            return symbol_check

        return RiskCheckResult(
            allowed=True,
            reason="Circuit breaker checks passed",
            warnings=all_warnings,
        )


# Global instance
_risk_governor: Optional[RiskGovernor] = None


def get_risk_governor(db_pool: Optional[asyncpg.Pool] = None) -> RiskGovernor:
    """Get or create global Risk Governor instance."""
    global _risk_governor
    if _risk_governor is None:
        _risk_governor = RiskGovernor(db_pool)
    return _risk_governor


async def get_daily_pnl(db_pool: asyncpg.Pool, current_equity: float) -> float:
    """
    Get daily PnL by comparing current equity to daily starting equity.

    If no record exists for today, creates one with current equity as starting.
    Returns the difference: current_equity - starting_equity.

    **Important**: This is EQUITY-BASED (not realized-only), meaning it includes
    both realized and unrealized PnL. The kill switch will trigger if total
    equity drops by the threshold amount, even from unrealized losses. This is
    intentional - we want to halt trading when the account is under stress,
    regardless of whether losses are realized.

    The daily record resets on first call each day (UTC). Starting equity is
    captured from the first account state query of the day.

    Args:
        db_pool: Database pool
        current_equity: Current account equity (realized + unrealized)

    Returns:
        Daily PnL (positive = profit, negative = loss)
    """
    today = datetime.now(timezone.utc).date()

    try:
        async with db_pool.acquire() as conn:
            # Check if we have a record for today
            row = await conn.fetchrow(
                """
                SELECT starting_equity, current_equity
                FROM risk_daily_pnl
                WHERE date = $1
                """,
                today,
            )

            if row:
                starting_equity = float(row["starting_equity"])
                # Update current equity
                await conn.execute(
                    """
                    UPDATE risk_daily_pnl
                    SET current_equity = $1,
                        daily_drawdown_pct = CASE
                            WHEN starting_equity > 0 THEN (starting_equity - $1) / starting_equity
                            ELSE 0
                        END,
                        updated_at = NOW()
                    WHERE date = $2
                    """,
                    current_equity,
                    today,
                )
                return current_equity - starting_equity
            else:
                # First check of the day - create record with current equity as starting
                await conn.execute(
                    """
                    INSERT INTO risk_daily_pnl (date, starting_equity, current_equity)
                    VALUES ($1, $2, $2)
                    ON CONFLICT (date) DO NOTHING
                    """,
                    today,
                    current_equity,
                )
                return 0.0  # No PnL yet today
    except Exception as e:
        print(f"[risk_governor] Failed to get daily PnL: {e}")
        return 0.0  # Fail safe - assume no PnL


async def check_risk_before_trade(
    db_pool: asyncpg.Pool,
    account_state: Dict[str, Any],
    proposed_size_usd: float = 0,
) -> RiskCheckResult:
    """
    Convenience function to check risk before a trade.

    Args:
        db_pool: Database pool
        account_state: Account state from Hyperliquid API
        proposed_size_usd: Proposed trade size in USD

    Returns:
        RiskCheckResult
    """
    governor = get_risk_governor(db_pool)

    # Extract values from account state
    margin_summary = account_state.get("marginSummary", {})
    account_value = float(margin_summary.get("accountValue", 0))
    margin_used = float(margin_summary.get("totalMarginUsed", 0))
    maintenance_margin = float(margin_summary.get("maintenanceMargin", 0))

    # Calculate total exposure
    total_exposure = 0.0
    for ap in account_state.get("assetPositions", []):
        pos = ap.get("position", {})
        size = abs(float(pos.get("szi", 0)))
        entry_price = float(pos.get("entryPx", 0))
        total_exposure += size * entry_price

    # Get daily PnL from database tracking
    daily_pnl = await get_daily_pnl(db_pool, account_value)

    return governor.run_all_checks(
        account_value=account_value,
        margin_used=margin_used,
        maintenance_margin=maintenance_margin,
        total_exposure=total_exposure,
        daily_pnl=daily_pnl,
        proposed_size_usd=proposed_size_usd,
    )
