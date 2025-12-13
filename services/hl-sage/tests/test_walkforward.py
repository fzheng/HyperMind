"""
Tests for Walk-Forward Replay module (Phase 3f: Selection Integrity).

These tests verify:
1. Cost estimation in replay
2. Period result computation
3. Summary metric aggregation
4. Dataclass behavior
"""
import pytest
from datetime import date, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.walkforward import (
    ReplayPeriod,
    ReplaySummary,
    compute_period_cost_r,
    format_replay_summary,
    REPLAY_EVALUATION_DAYS,
)
from app.snapshot import ROUND_TRIP_COST_BPS


class TestCostEstimation:
    """Test cost estimation in replay."""

    def test_single_episode_cost(self):
        """Cost should be computed per episode."""
        episodes = [
            {
                "entry_price": 50000,  # BTC at $50k
                "atr_at_entry": 1000,  # $1000 ATR
            }
        ]
        cost_r = compute_period_cost_r(episodes)

        # Cost = entry_price * (30bps / 10000) / ATR
        # = 50000 * 0.003 / 1000 = 0.15R
        expected = 50000 * (ROUND_TRIP_COST_BPS / 10000) / 1000
        assert cost_r == pytest.approx(expected, rel=0.01)

    def test_multiple_episodes_sum(self):
        """Costs should sum across episodes."""
        episodes = [
            {"entry_price": 50000, "atr_at_entry": 1000},
            {"entry_price": 3000, "atr_at_entry": 100},  # ETH
        ]
        cost_r = compute_period_cost_r(episodes)

        # Each episode adds cost
        btc_cost = 50000 * (ROUND_TRIP_COST_BPS / 10000) / 1000
        eth_cost = 3000 * (ROUND_TRIP_COST_BPS / 10000) / 100
        expected = btc_cost + eth_cost

        assert cost_r == pytest.approx(expected, rel=0.01)

    def test_zero_atr_handled(self):
        """Zero ATR should not cause division by zero."""
        episodes = [
            {"entry_price": 50000, "atr_at_entry": 0},
        ]
        cost_r = compute_period_cost_r(episodes)
        assert cost_r == 0

    def test_missing_fields_handled(self):
        """Missing fields should not cause errors."""
        episodes = [
            {},  # Empty episode
            {"entry_price": None, "atr_at_entry": 1000},
            {"entry_price": 50000, "atr_at_entry": None},
        ]
        cost_r = compute_period_cost_r(episodes)
        assert cost_r == 0

    def test_empty_episodes_zero_cost(self):
        """Empty episode list should have zero cost."""
        assert compute_period_cost_r([]) == 0


class TestReplayPeriod:
    """Test ReplayPeriod dataclass."""

    def test_basic_creation(self):
        """Should create period with all fields."""
        period = ReplayPeriod(
            selection_date=date(2025, 12, 1),
            evaluation_start=date(2025, 12, 1),
            evaluation_end=date(2025, 12, 8),
            universe_size=100,
            selected_count=50,
            fdr_qualified_count=40,
            total_r_gross=5.0,
            total_r_net=4.0,
            avg_r_gross=0.1,
            avg_r_net=0.08,
            trader_results=[],
            deaths_during_period=2,
            censored_during_period=1,
        )

        assert period.selection_date == date(2025, 12, 1)
        assert period.selected_count == 50
        assert period.total_r_gross == 5.0
        assert period.deaths_during_period == 2

    def test_net_less_than_gross(self):
        """Net R should typically be less than gross R (due to costs)."""
        period = ReplayPeriod(
            selection_date=date.today(),
            evaluation_start=date.today(),
            evaluation_end=date.today() + timedelta(days=7),
            universe_size=100,
            selected_count=50,
            fdr_qualified_count=40,
            total_r_gross=5.0,
            total_r_net=3.5,
            avg_r_gross=0.1,
            avg_r_net=0.07,
            trader_results=[],
            deaths_during_period=0,
            censored_during_period=0,
        )

        assert period.total_r_net < period.total_r_gross


class TestReplaySummary:
    """Test ReplaySummary dataclass."""

    def test_basic_creation(self):
        """Should create summary with all fields."""
        summary = ReplaySummary(
            start_date=date(2025, 11, 1),
            end_date=date(2025, 12, 1),
            periods=30,
            cumulative_r_gross=15.0,
            cumulative_r_net=12.0,
            avg_period_r_gross=0.5,
            avg_period_r_net=0.4,
            r_gross_std=0.2,
            r_net_std=0.15,
            sharpe_gross=2.5,
            sharpe_net=2.67,
            winning_periods=20,
            losing_periods=10,
            win_rate=0.667,
            total_deaths=5,
            total_censored=3,
            period_results=[],
        )

        assert summary.periods == 30
        assert summary.cumulative_r_gross == 15.0
        assert summary.win_rate == pytest.approx(0.667, rel=0.01)

    def test_sharpe_calculation(self):
        """Sharpe should be mean / std."""
        avg = 0.5
        std = 0.2
        sharpe = avg / std  # 2.5

        summary = ReplaySummary(
            start_date=date.today() - timedelta(days=30),
            end_date=date.today(),
            periods=30,
            cumulative_r_gross=15.0,
            cumulative_r_net=12.0,
            avg_period_r_gross=avg,
            avg_period_r_net=0.4,
            r_gross_std=std,
            r_net_std=0.15,
            sharpe_gross=sharpe,
            sharpe_net=2.67,
            winning_periods=20,
            losing_periods=10,
            win_rate=0.667,
            total_deaths=0,
            total_censored=0,
            period_results=[],
        )

        assert summary.sharpe_gross == pytest.approx(2.5, rel=0.01)


class TestFormatReplaySummary:
    """Test summary formatting for API response."""

    def test_format_basic_summary(self):
        """Should format summary to JSON-serializable dict."""
        summary = ReplaySummary(
            start_date=date(2025, 11, 1),
            end_date=date(2025, 12, 1),
            periods=30,
            cumulative_r_gross=15.0,
            cumulative_r_net=12.0,
            avg_period_r_gross=0.5,
            avg_period_r_net=0.4,
            r_gross_std=0.2,
            r_net_std=0.15,
            sharpe_gross=2.5,
            sharpe_net=2.67,
            winning_periods=20,
            losing_periods=10,
            win_rate=0.667,
            total_deaths=5,
            total_censored=3,
            period_results=[],
        )

        formatted = format_replay_summary(summary)

        assert formatted["start_date"] == "2025-11-01"
        assert formatted["end_date"] == "2025-12-01"
        assert formatted["periods"] == 30
        assert formatted["performance"]["cumulative_r_gross"] == 15.0
        assert formatted["win_rate"]["rate"] == 0.667
        assert formatted["survival"]["total_deaths"] == 5

    def test_format_with_period_results(self):
        """Should include period details in formatted output."""
        period = ReplayPeriod(
            selection_date=date(2025, 12, 1),
            evaluation_start=date(2025, 12, 1),
            evaluation_end=date(2025, 12, 8),
            universe_size=100,
            selected_count=50,
            fdr_qualified_count=40,
            total_r_gross=5.0,
            total_r_net=4.0,
            avg_r_gross=0.1,
            avg_r_net=0.08,
            trader_results=[],
            deaths_during_period=1,
            censored_during_period=0,
        )

        summary = ReplaySummary(
            start_date=date(2025, 12, 1),
            end_date=date(2025, 12, 1),
            periods=1,
            cumulative_r_gross=5.0,
            cumulative_r_net=4.0,
            avg_period_r_gross=5.0,
            avg_period_r_net=4.0,
            r_gross_std=0,
            r_net_std=0,
            sharpe_gross=0,
            sharpe_net=0,
            winning_periods=1,
            losing_periods=0,
            win_rate=1.0,
            total_deaths=1,
            total_censored=0,
            period_results=[period],
        )

        formatted = format_replay_summary(summary)

        assert len(formatted["periods_detail"]) == 1
        assert formatted["periods_detail"][0]["selection_date"] == "2025-12-01"
        assert formatted["periods_detail"][0]["deaths"] == 1


class TestEvaluationWindow:
    """Test evaluation window configuration."""

    def test_default_evaluation_days(self):
        """Default evaluation period should be 7 days."""
        assert REPLAY_EVALUATION_DAYS == 7

    def test_evaluation_end_calculation(self):
        """Evaluation end should be start + evaluation_days."""
        start = date(2025, 12, 1)
        end = start + timedelta(days=REPLAY_EVALUATION_DAYS)
        assert end == date(2025, 12, 8)
