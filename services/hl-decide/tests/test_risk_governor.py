"""
Tests for Risk Governor module (Phase 3f: Selection Integrity).

These tests verify:
1. Liquidation distance guard
2. Daily drawdown kill switch
3. Position size limits
4. Exposure limits
5. Kill switch behavior
"""
import pytest
from datetime import datetime, timezone, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.risk_governor import (
    RiskGovernor,
    RiskState,
    RiskCheckResult,
    LIQUIDATION_DISTANCE_MIN,
    DAILY_DRAWDOWN_KILL_PCT,
    MIN_EQUITY_FLOOR,
    MAX_POSITION_SIZE_PCT,
    MAX_TOTAL_EXPOSURE_PCT,
    KILL_SWITCH_COOLDOWN,
)


class TestRiskState:
    """Test RiskState dataclass."""

    def test_basic_creation(self):
        """Should create state with all fields."""
        state = RiskState(
            timestamp=datetime.now(timezone.utc),
            account_value=100000,
            margin_used=20000,
            maintenance_margin=15000,
            total_exposure=50000,
            margin_ratio=6.67,
            daily_pnl=-2000,
            daily_starting_equity=102000,
            daily_drawdown_pct=0.0196,
        )

        assert state.account_value == 100000
        assert state.margin_ratio == 6.67
        assert state.daily_drawdown_pct == pytest.approx(0.0196, rel=0.01)


class TestLiquidationDistance:
    """Test liquidation distance guard."""

    def test_healthy_margin_allowed(self):
        """High margin ratio should allow trading."""
        governor = RiskGovernor()
        state = RiskState(
            timestamp=datetime.now(timezone.utc),
            account_value=100000,
            margin_used=10000,
            maintenance_margin=5000,
            total_exposure=50000,
            margin_ratio=20.0,  # Very healthy
            daily_pnl=0,
            daily_starting_equity=100000,
            daily_drawdown_pct=0,
        )

        result = governor.check_liquidation_distance(state)
        assert result.allowed is True

    def test_close_to_liquidation_blocked(self):
        """Low margin ratio should block trading."""
        governor = RiskGovernor()
        state = RiskState(
            timestamp=datetime.now(timezone.utc),
            account_value=100000,
            margin_used=80000,
            maintenance_margin=90000,  # Near liquidation
            total_exposure=200000,
            margin_ratio=1.1,  # Below LIQUIDATION_DISTANCE_MIN (1.5)
            daily_pnl=0,
            daily_starting_equity=100000,
            daily_drawdown_pct=0,
        )

        result = governor.check_liquidation_distance(state)
        assert result.allowed is False
        assert "liquidation" in result.reason.lower()

    def test_warning_when_approaching(self):
        """Should warn when approaching liquidation threshold."""
        governor = RiskGovernor()
        # margin_ratio = 2.0, threshold = 1.5, warning at < 2.25 (1.5 * 1.5)
        state = RiskState(
            timestamp=datetime.now(timezone.utc),
            account_value=100000,
            margin_used=30000,
            maintenance_margin=50000,
            total_exposure=100000,
            margin_ratio=2.0,  # Between threshold and warning
            daily_pnl=0,
            daily_starting_equity=100000,
            daily_drawdown_pct=0,
        )

        result = governor.check_liquidation_distance(state)
        assert result.allowed is True
        assert len(result.warnings) > 0


class TestDailyDrawdown:
    """Test daily drawdown kill switch."""

    def test_positive_pnl_allowed(self):
        """Positive daily PnL should allow trading."""
        governor = RiskGovernor()
        state = RiskState(
            timestamp=datetime.now(timezone.utc),
            account_value=105000,
            margin_used=10000,
            maintenance_margin=5000,
            total_exposure=50000,
            margin_ratio=21.0,
            daily_pnl=5000,  # Up 5%
            daily_starting_equity=100000,
            daily_drawdown_pct=0,
        )

        result = governor.check_daily_drawdown(state)
        assert result.allowed is True

    def test_small_loss_allowed(self):
        """Small daily loss should allow trading."""
        governor = RiskGovernor()
        state = RiskState(
            timestamp=datetime.now(timezone.utc),
            account_value=98000,
            margin_used=10000,
            maintenance_margin=5000,
            total_exposure=50000,
            margin_ratio=19.6,
            daily_pnl=-2000,  # Down 2%
            daily_starting_equity=100000,
            daily_drawdown_pct=0.02,  # 2% < 5% threshold
        )

        result = governor.check_daily_drawdown(state)
        assert result.allowed is True

    def test_large_loss_triggers_kill_switch(self):
        """Large daily loss should trigger kill switch."""
        governor = RiskGovernor()
        state = RiskState(
            timestamp=datetime.now(timezone.utc),
            account_value=94000,
            margin_used=10000,
            maintenance_margin=5000,
            total_exposure=50000,
            margin_ratio=18.8,
            daily_pnl=-6000,  # Down 6%
            daily_starting_equity=100000,
            daily_drawdown_pct=0.06,  # 6% > 5% threshold
        )

        result = governor.check_daily_drawdown(state)
        assert result.allowed is False
        assert "KILL SWITCH" in result.reason
        assert governor._kill_switch_active is True

    def test_warning_at_half_threshold(self):
        """Should warn when at 50% of kill threshold."""
        governor = RiskGovernor()
        state = RiskState(
            timestamp=datetime.now(timezone.utc),
            account_value=97500,
            margin_used=10000,
            maintenance_margin=5000,
            total_exposure=50000,
            margin_ratio=19.5,
            daily_pnl=-2500,  # Down 2.5%
            daily_starting_equity=100000,
            daily_drawdown_pct=0.025,  # 2.5% = 50% of 5% threshold
        )

        result = governor.check_daily_drawdown(state)
        assert result.allowed is True
        assert len(result.warnings) > 0


class TestKillSwitch:
    """Test kill switch behavior."""

    def test_kill_switch_initially_inactive(self):
        """Kill switch should be inactive by default."""
        governor = RiskGovernor()
        active, reason = governor.check_kill_switch()
        assert active is False

    def test_kill_switch_can_be_triggered(self):
        """Kill switch should be triggerable."""
        governor = RiskGovernor()
        governor.trigger_kill_switch("Test trigger")

        active, reason = governor.check_kill_switch()
        assert active is True
        assert "remaining" in reason or "active" in reason.lower()

    def test_kill_switch_blocks_all_trades(self):
        """Active kill switch should block all trades."""
        governor = RiskGovernor()
        governor.trigger_kill_switch("Test trigger")

        # Even healthy account should be blocked
        result = governor.run_all_checks(
            account_value=1000000,
            margin_used=10000,
            maintenance_margin=5000,
            total_exposure=50000,
            daily_pnl=50000,  # Up 5%
            proposed_size_usd=1000,
        )

        assert result.allowed is False

    def test_kill_switch_can_be_reset(self):
        """Kill switch should be resettable by operator."""
        governor = RiskGovernor()
        governor.trigger_kill_switch("Test trigger")
        assert governor._kill_switch_active is True

        governor.reset_kill_switch()
        assert governor._kill_switch_active is False

        active, reason = governor.check_kill_switch()
        assert active is False


class TestEquityFloor:
    """Test minimum equity floor."""

    def test_above_floor_allowed(self):
        """Equity above floor should allow trading."""
        governor = RiskGovernor()
        state = RiskState(
            timestamp=datetime.now(timezone.utc),
            account_value=50000,  # Above $10k floor
            margin_used=5000,
            maintenance_margin=3000,
            total_exposure=25000,
            margin_ratio=16.67,
            daily_pnl=0,
            daily_starting_equity=50000,
            daily_drawdown_pct=0,
        )

        result = governor.check_equity_floor(state)
        assert result.allowed is True

    def test_below_floor_blocked(self):
        """Equity below floor should block trading."""
        governor = RiskGovernor()
        state = RiskState(
            timestamp=datetime.now(timezone.utc),
            account_value=8000,  # Below $10k floor
            margin_used=1000,
            maintenance_margin=500,
            total_exposure=4000,
            margin_ratio=16.0,
            daily_pnl=0,
            daily_starting_equity=8000,
            daily_drawdown_pct=0,
        )

        result = governor.check_equity_floor(state)
        assert result.allowed is False
        assert "floor" in result.reason.lower()


class TestPositionSize:
    """Test position size limits."""

    def test_small_position_allowed(self):
        """Small position should be allowed."""
        governor = RiskGovernor()
        state = RiskState(
            timestamp=datetime.now(timezone.utc),
            account_value=100000,
            margin_used=10000,
            maintenance_margin=5000,
            total_exposure=50000,
            margin_ratio=20.0,
            daily_pnl=0,
            daily_starting_equity=100000,
            daily_drawdown_pct=0,
        )

        # 5% of equity = $5k, well within 10% limit
        result = governor.check_position_size(state, proposed_size_usd=5000)
        assert result.allowed is True

    def test_large_position_blocked(self):
        """Position exceeding limit should be blocked."""
        governor = RiskGovernor()
        state = RiskState(
            timestamp=datetime.now(timezone.utc),
            account_value=100000,
            margin_used=10000,
            maintenance_margin=5000,
            total_exposure=50000,
            margin_ratio=20.0,
            daily_pnl=0,
            daily_starting_equity=100000,
            daily_drawdown_pct=0,
        )

        # 15% of equity = $15k, exceeds 10% limit
        result = governor.check_position_size(state, proposed_size_usd=15000)
        assert result.allowed is False
        assert "Position size" in result.reason


class TestTotalExposure:
    """Test total exposure limits."""

    def test_low_exposure_allowed(self):
        """Low total exposure should be allowed."""
        governor = RiskGovernor()
        state = RiskState(
            timestamp=datetime.now(timezone.utc),
            account_value=100000,
            margin_used=10000,
            maintenance_margin=5000,
            total_exposure=30000,  # 30% < 50% limit
            margin_ratio=20.0,
            daily_pnl=0,
            daily_starting_equity=100000,
            daily_drawdown_pct=0,
        )

        result = governor.check_total_exposure(state, proposed_additional_exposure=10000)
        assert result.allowed is True

    def test_high_exposure_blocked(self):
        """Exposure exceeding limit should be blocked."""
        governor = RiskGovernor()
        state = RiskState(
            timestamp=datetime.now(timezone.utc),
            account_value=100000,
            margin_used=10000,
            maintenance_margin=5000,
            total_exposure=45000,  # 45%
            margin_ratio=20.0,
            daily_pnl=0,
            daily_starting_equity=100000,
            daily_drawdown_pct=0,
        )

        # Adding $10k would make 55% > 50% limit
        result = governor.check_total_exposure(state, proposed_additional_exposure=10000)
        assert result.allowed is False
        assert "exposure" in result.reason.lower()


class TestAllChecks:
    """Test running all checks together."""

    def test_healthy_account_passes_all(self):
        """Healthy account should pass all checks."""
        governor = RiskGovernor()

        result = governor.run_all_checks(
            account_value=100000,
            margin_used=10000,
            maintenance_margin=5000,
            total_exposure=30000,
            daily_pnl=2000,
            proposed_size_usd=5000,
        )

        assert result.allowed is True
        assert "passed" in result.reason.lower()

    def test_unhealthy_account_fails(self):
        """Account with issues should fail appropriate check."""
        governor = RiskGovernor()

        # Very low equity
        result = governor.run_all_checks(
            account_value=5000,  # Below $10k floor
            margin_used=1000,
            maintenance_margin=500,
            total_exposure=2500,
            daily_pnl=0,
            proposed_size_usd=100,
        )

        assert result.allowed is False
        assert "floor" in result.reason.lower()

    def test_checks_run_in_order(self):
        """Checks should run in priority order."""
        governor = RiskGovernor()

        # Multiple issues - should fail on kill switch first
        governor.trigger_kill_switch("Test")

        result = governor.run_all_checks(
            account_value=5000,  # Also below floor
            margin_used=1000,
            maintenance_margin=500,
            total_exposure=2500,
            daily_pnl=-1000,  # Also has drawdown
            proposed_size_usd=100,
        )

        assert result.allowed is False
        # Should fail on kill switch, not equity floor
        assert "kill" in result.reason.lower() or "active" in result.reason.lower()


class TestConfigurationValues:
    """Test configuration values are sensible."""

    def test_liquidation_distance_conservative(self):
        """Liquidation distance should provide safety buffer."""
        # 1.5x means 50% buffer before liquidation
        assert LIQUIDATION_DISTANCE_MIN >= 1.2
        assert LIQUIDATION_DISTANCE_MIN <= 3.0

    def test_daily_drawdown_reasonable(self):
        """Daily drawdown threshold should be conservative."""
        # 5% daily loss is significant but not panic-inducing
        assert DAILY_DRAWDOWN_KILL_PCT >= 0.02  # At least 2%
        assert DAILY_DRAWDOWN_KILL_PCT <= 0.10  # At most 10%

    def test_equity_floor_reasonable(self):
        """Equity floor should prevent trading with tiny accounts."""
        assert MIN_EQUITY_FLOOR >= 1000  # At least $1k
        assert MIN_EQUITY_FLOOR <= 50000  # At most $50k

    def test_position_size_conservative(self):
        """Position size limit should prevent concentration."""
        assert MAX_POSITION_SIZE_PCT >= 0.02  # At least 2%
        assert MAX_POSITION_SIZE_PCT <= 0.25  # At most 25%

    def test_exposure_limit_conservative(self):
        """Exposure limit should prevent over-leveraging."""
        assert MAX_TOTAL_EXPOSURE_PCT >= 0.25  # At least 25%
        assert MAX_TOTAL_EXPOSURE_PCT <= 1.0  # At most 100%

    def test_kill_switch_cooldown_reasonable(self):
        """Kill switch cooldown should be meaningful."""
        # At least 1 hour, at most 1 week
        assert KILL_SWITCH_COOLDOWN >= 3600
        assert KILL_SWITCH_COOLDOWN <= 7 * 86400
