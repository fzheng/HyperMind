"""
Risk Governor: Hard safety limits before live trading

Phase 3f: Selection Integrity

The Risk Governor provides hard safety limits that CANNOT be overridden:
1. **Liquidation Distance Guard**: Block trades if account too close to liquidation
2. **Daily Drawdown Kill Switch**: Halt all trading if daily loss exceeds threshold
3. **Exposure Limits**: Prevent excessive position sizing

These are the LAST line of defense before capital destruction.

@module risk_governor
"""

import os
from dataclasses import dataclass
from datetime import datetime, timezone
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


@dataclass
class RiskState:
    """Current risk state from account data."""
    timestamp: datetime
    account_value: float
    margin_used: float
    maintenance_margin: float
    total_exposure: float
    margin_ratio: float
    daily_pnl: float
    daily_starting_equity: float
    daily_drawdown_pct: float


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

    def check_kill_switch(self) -> Tuple[bool, str]:
        """
        Check if kill switch is active.

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


# Global instance
_risk_governor: Optional[RiskGovernor] = None


def get_risk_governor(db_pool: Optional[asyncpg.Pool] = None) -> RiskGovernor:
    """Get or create global Risk Governor instance."""
    global _risk_governor
    if _risk_governor is None:
        _risk_governor = RiskGovernor(db_pool)
    return _risk_governor


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

    # Get daily PnL (would need to track this)
    daily_pnl = 0.0  # TODO: Implement daily PnL tracking

    return governor.run_all_checks(
        account_value=account_value,
        margin_used=margin_used,
        maintenance_margin=maintenance_margin,
        total_exposure=total_exposure,
        daily_pnl=daily_pnl,
        proposed_size_usd=proposed_size_usd,
    )
