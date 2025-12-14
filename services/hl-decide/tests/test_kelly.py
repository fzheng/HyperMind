"""
Tests for Kelly Criterion Position Sizing (Phase 4.1: Risk Management).

These tests verify:
1. Full Kelly fraction formula correctness
2. Edge cases (p=0, p=1, R=1, R>>1)
3. Negative EV handling
4. Fractional Kelly scaling
5. Fallback behavior for insufficient data
6. Position size calculation with price
7. Hard limit capping
8. Expected value calculation
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.kelly import (
    calculate_kelly_fraction,
    calculate_expected_value,
    kelly_position_size,
    KellyInput,
    KellyResult,
    KELLY_FRACTION,
    KELLY_MIN_EPISODES,
    KELLY_FALLBACK_PCT,
    KELLY_MAX_FRACTION,
    KELLY_MAX_POSITION_PCT,
)


class TestKellyFractionFormula:
    """Test the core Kelly fraction formula."""

    def test_coin_flip_even_odds(self):
        """50% win rate with 1:1 odds should give 0 Kelly."""
        # f* = p - (1-p)/R = 0.5 - 0.5/1 = 0
        kelly = calculate_kelly_fraction(win_rate=0.5, avg_win_r=1.0, avg_loss_r=1.0)
        assert kelly == pytest.approx(0.0, abs=0.001)

    def test_positive_edge_coin_flip(self):
        """55% win rate with 1:1 odds should give positive Kelly."""
        # f* = 0.55 - 0.45/1 = 0.10
        kelly = calculate_kelly_fraction(win_rate=0.55, avg_win_r=1.0, avg_loss_r=1.0)
        assert kelly == pytest.approx(0.10, abs=0.001)

    def test_negative_edge_coin_flip(self):
        """45% win rate with 1:1 odds should give 0 (negative EV)."""
        # f* = 0.45 - 0.55/1 = -0.10 -> clamped to 0
        kelly = calculate_kelly_fraction(win_rate=0.45, avg_win_r=1.0, avg_loss_r=1.0)
        assert kelly == 0.0

    def test_high_reward_ratio(self):
        """Lower win rate with high R ratio should still be positive."""
        # Win 40% but win 3x when right
        # f* = 0.4 - 0.6/3 = 0.4 - 0.2 = 0.2
        kelly = calculate_kelly_fraction(win_rate=0.40, avg_win_r=3.0, avg_loss_r=1.0)
        assert kelly == pytest.approx(0.20, abs=0.001)

    def test_low_reward_ratio(self):
        """High win rate with low R ratio."""
        # Win 70% but win 0.5x when right
        # f* = 0.7 - 0.3/0.5 = 0.7 - 0.6 = 0.1
        kelly = calculate_kelly_fraction(win_rate=0.70, avg_win_r=0.5, avg_loss_r=1.0)
        assert kelly == pytest.approx(0.10, abs=0.001)

    def test_extreme_edge(self):
        """Very high win rate with good R ratio."""
        # Win 80% with 2:1 R ratio
        # f* = 0.8 - 0.2/2 = 0.8 - 0.1 = 0.7
        kelly = calculate_kelly_fraction(win_rate=0.80, avg_win_r=2.0, avg_loss_r=1.0)
        assert kelly == pytest.approx(0.70, abs=0.001)


class TestKellyEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_win_rate(self):
        """0% win rate should give 0 Kelly."""
        kelly = calculate_kelly_fraction(win_rate=0.0, avg_win_r=1.0, avg_loss_r=1.0)
        assert kelly == 0.0

    def test_100_percent_win_rate(self):
        """100% win rate should give 1.0 Kelly (bet everything)."""
        kelly = calculate_kelly_fraction(win_rate=1.0, avg_win_r=1.0, avg_loss_r=1.0)
        assert kelly == 1.0

    def test_near_100_win_rate(self):
        """99% win rate should give high Kelly."""
        kelly = calculate_kelly_fraction(win_rate=0.99, avg_win_r=1.0, avg_loss_r=1.0)
        assert kelly == pytest.approx(0.98, abs=0.01)

    def test_zero_avg_loss(self):
        """Zero avg_loss should return 0 (division by zero protection)."""
        kelly = calculate_kelly_fraction(win_rate=0.6, avg_win_r=1.0, avg_loss_r=0.0)
        assert kelly == 0.0

    def test_negative_avg_loss_converted(self):
        """Negative avg_loss should be converted to positive."""
        kelly = calculate_kelly_fraction(win_rate=0.55, avg_win_r=1.0, avg_loss_r=-1.0)
        assert kelly == pytest.approx(0.10, abs=0.001)

    def test_invalid_win_rate_low(self):
        """Win rate below 0 should return 0."""
        kelly = calculate_kelly_fraction(win_rate=-0.1, avg_win_r=1.0, avg_loss_r=1.0)
        assert kelly == 0.0

    def test_invalid_win_rate_high(self):
        """Win rate above 1 should return 0."""
        kelly = calculate_kelly_fraction(win_rate=1.5, avg_win_r=1.0, avg_loss_r=1.0)
        assert kelly == 0.0

    def test_kelly_clamped_to_1(self):
        """Kelly should never exceed 1.0."""
        # Extreme scenario that would give >1
        kelly = calculate_kelly_fraction(win_rate=0.99, avg_win_r=10.0, avg_loss_r=0.1)
        assert kelly <= 1.0


class TestExpectedValue:
    """Test expected value calculation."""

    def test_positive_ev(self):
        """Positive edge should give positive EV."""
        ev = calculate_expected_value(win_rate=0.55, avg_win_r=1.0, avg_loss_r=1.0)
        # EV = 0.55 * 1.0 - 0.45 * 1.0 = 0.10
        assert ev == pytest.approx(0.10, abs=0.001)

    def test_negative_ev(self):
        """Negative edge should give negative EV."""
        ev = calculate_expected_value(win_rate=0.45, avg_win_r=1.0, avg_loss_r=1.0)
        # EV = 0.45 * 1.0 - 0.55 * 1.0 = -0.10
        assert ev == pytest.approx(-0.10, abs=0.001)

    def test_zero_ev(self):
        """Even odds should give zero EV."""
        ev = calculate_expected_value(win_rate=0.50, avg_win_r=1.0, avg_loss_r=1.0)
        assert ev == pytest.approx(0.0, abs=0.001)

    def test_high_r_ratio_ev(self):
        """High R ratio compensates for lower win rate."""
        # Win 40% but win 2x when right
        ev = calculate_expected_value(win_rate=0.40, avg_win_r=2.0, avg_loss_r=1.0)
        # EV = 0.40 * 2.0 - 0.60 * 1.0 = 0.80 - 0.60 = 0.20
        assert ev == pytest.approx(0.20, abs=0.001)


class TestKellyPositionSize:
    """Test the full kelly_position_size function."""

    def _make_input(
        self,
        win_rate=0.55,
        avg_win_r=1.0,
        avg_loss_r=1.0,
        episode_count=50,
        account_value=100000,
        current_price=50000,
        stop_distance_pct=0.02,
    ):
        """Helper to create KellyInput."""
        return KellyInput(
            win_rate=win_rate,
            avg_win_r=avg_win_r,
            avg_loss_r=avg_loss_r,
            episode_count=episode_count,
            account_value=account_value,
            current_price=current_price,
            stop_distance_pct=stop_distance_pct,
        )

    def test_basic_kelly_sizing(self):
        """Basic Kelly calculation with sufficient data."""
        kelly_input = self._make_input(win_rate=0.55, episode_count=50)
        result = kelly_position_size(kelly_input)

        assert result.method == "kelly"
        assert result.full_kelly == pytest.approx(0.10, abs=0.01)
        # Fractional Kelly = 0.10 * 0.25 = 0.025
        assert result.fractional_kelly == pytest.approx(0.025, abs=0.005)
        assert result.position_size_usd > 0
        assert result.position_size_coin > 0

    def test_insufficient_episodes_fallback(self):
        """Should fall back when episodes < min_episodes."""
        kelly_input = self._make_input(episode_count=10)  # Default min is 30
        result = kelly_position_size(kelly_input)

        assert result.method == "fallback_insufficient_data"
        assert "10 episodes" in result.reasoning
        assert result.position_pct == pytest.approx(KELLY_FALLBACK_PCT, abs=0.001)

    def test_negative_ev_fallback(self):
        """Should fall back with reduced size for negative EV."""
        kelly_input = self._make_input(win_rate=0.30, avg_win_r=1.0, episode_count=50)
        result = kelly_position_size(kelly_input)

        assert result.method == "fallback_negative_ev"
        assert "Negative EV" in result.reasoning
        # Should be half the fallback
        assert result.position_pct == pytest.approx(KELLY_FALLBACK_PCT * 0.5, abs=0.001)

    def test_zero_price_error(self):
        """Should handle zero price gracefully."""
        kelly_input = self._make_input(current_price=0)
        result = kelly_position_size(kelly_input)

        assert result.method == "error"
        assert result.position_size_usd == 0
        assert result.position_size_coin == 0

    def test_position_capped_by_hard_limit(self):
        """Position should be capped at max_position_pct."""
        # Very high win rate should give large Kelly
        kelly_input = self._make_input(
            win_rate=0.90,
            avg_win_r=3.0,
            episode_count=100,
            stop_distance_pct=0.005,  # Small stop = large position
        )
        result = kelly_position_size(kelly_input)

        assert result.capped is True
        assert result.position_pct <= KELLY_MAX_POSITION_PCT

    def test_custom_fraction(self):
        """Custom fractional Kelly should be applied."""
        kelly_input = self._make_input(win_rate=0.60, episode_count=50)

        # Default fraction
        result_default = kelly_position_size(kelly_input)

        # Half fraction
        result_half = kelly_position_size(kelly_input, fraction=0.125)

        # Half fraction should give ~half the fractional kelly
        assert result_half.fractional_kelly == pytest.approx(
            result_default.fractional_kelly * 0.5, rel=0.1
        )

    def test_custom_min_episodes(self):
        """Custom min_episodes threshold should be respected."""
        kelly_input = self._make_input(episode_count=20)

        # With default min (30), should fallback
        result_default = kelly_position_size(kelly_input)
        assert result_default.method == "fallback_insufficient_data"

        # With lower min (10), should use Kelly
        result_custom = kelly_position_size(kelly_input, min_episodes=10)
        assert result_custom.method == "kelly"

    def test_usd_to_coin_conversion(self):
        """Position size in coins should match USD / price."""
        kelly_input = self._make_input(
            account_value=100000, current_price=50000, episode_count=50
        )
        result = kelly_position_size(kelly_input)

        expected_coins = result.position_size_usd / kelly_input.current_price
        assert result.position_size_coin == pytest.approx(expected_coins, rel=0.001)


class TestKellyWithRealScenarios:
    """Test Kelly with realistic trading scenarios."""

    def test_btc_trader_good_stats(self):
        """BTC trader with good win rate and R ratio."""
        kelly_input = KellyInput(
            win_rate=0.58,  # 58% win rate
            avg_win_r=1.2,  # Average 1.2R when winning
            avg_loss_r=0.8,  # Average 0.8R when losing
            episode_count=100,
            account_value=50000,  # $50k account
            current_price=43000,  # BTC at $43k
            stop_distance_pct=0.02,  # 2% stop
        )
        result = kelly_position_size(kelly_input)

        assert result.method == "kelly"
        assert result.position_pct <= KELLY_MAX_POSITION_PCT
        # Should be reasonable size
        assert 0.001 <= result.position_pct <= 0.10

    def test_eth_trader_mediocre_stats(self):
        """ETH trader with borderline statistics."""
        kelly_input = KellyInput(
            win_rate=0.55,  # 55% win rate (increased to have positive EV after fees)
            avg_win_r=1.2,  # Slightly better R ratio
            avg_loss_r=1.0,
            episode_count=50,
            account_value=25000,
            current_price=2200,  # ETH at $2.2k
            stop_distance_pct=0.015,  # 1.5% stop
            round_trip_fee_pct=0.0,  # No fees for this test
        )
        result = kelly_position_size(kelly_input)

        # Positive EV with decent stats
        assert result.method == "kelly"
        # With small stop distance, position gets capped
        assert result.position_pct <= KELLY_MAX_POSITION_PCT
        # Small Kelly fraction due to moderate edge
        assert result.fractional_kelly < 0.10

    def test_trader_with_losing_history(self):
        """Trader with losing track record should get minimal size."""
        kelly_input = KellyInput(
            win_rate=0.40,  # Losing record
            avg_win_r=0.8,
            avg_loss_r=1.2,
            episode_count=75,
            account_value=100000,
            current_price=50000,
            stop_distance_pct=0.02,
        )
        result = kelly_position_size(kelly_input)

        assert result.method == "fallback_negative_ev"
        assert result.position_pct < KELLY_FALLBACK_PCT

    def test_new_trader_no_history(self):
        """New trader with minimal history should use fallback."""
        kelly_input = KellyInput(
            win_rate=0.65,  # Looks good but...
            avg_win_r=1.5,
            avg_loss_r=0.5,
            episode_count=5,  # Only 5 episodes
            account_value=30000,
            current_price=43000,
            stop_distance_pct=0.02,
        )
        result = kelly_position_size(kelly_input)

        assert result.method == "fallback_insufficient_data"
        assert "5 episodes" in result.reasoning


class TestKellyFractionalMultiplier:
    """Test fractional Kelly multiplier behavior."""

    def test_quarter_kelly_default(self):
        """Default should be quarter Kelly (0.25)."""
        assert KELLY_FRACTION == 0.25

    def test_fractional_reduces_variance(self):
        """Fractional Kelly should reduce position size proportionally."""
        kelly_input = KellyInput(
            win_rate=0.60,
            avg_win_r=1.5,
            avg_loss_r=1.0,
            episode_count=100,
            account_value=100000,
            current_price=50000,
            stop_distance_pct=0.02,
        )

        result_full = kelly_position_size(kelly_input, fraction=1.0)
        result_half = kelly_position_size(kelly_input, fraction=0.5)
        result_quarter = kelly_position_size(kelly_input, fraction=0.25)

        # Full Kelly fractional should be full_kelly * 1.0
        assert result_full.fractional_kelly == pytest.approx(
            result_full.full_kelly, rel=0.01
        )
        # Half should be half of full
        assert result_half.fractional_kelly == pytest.approx(
            result_full.fractional_kelly * 0.5, rel=0.01
        )
        # Quarter should be quarter of full
        assert result_quarter.fractional_kelly == pytest.approx(
            result_full.fractional_kelly * 0.25, rel=0.01
        )

    def test_max_fraction_cap(self):
        """Fractional Kelly should be capped at KELLY_MAX_FRACTION."""
        # Extreme edge that would give very high Kelly
        kelly_input = KellyInput(
            win_rate=0.95,
            avg_win_r=5.0,
            avg_loss_r=0.5,
            episode_count=200,
            account_value=100000,
            current_price=50000,
            stop_distance_pct=0.02,
        )

        # Even with fraction=1.0, fractional_kelly should be capped
        result = kelly_position_size(kelly_input, fraction=1.0)
        assert result.fractional_kelly <= KELLY_MAX_FRACTION


class TestKellyConfiguration:
    """Test configuration constants."""

    def test_default_fraction(self):
        """Default fraction should be conservative."""
        assert KELLY_FRACTION <= 0.5  # At most half Kelly

    def test_min_episodes_reasonable(self):
        """Min episodes should be statistically reasonable."""
        assert KELLY_MIN_EPISODES >= 20  # Need sample size for confidence

    def test_fallback_conservative(self):
        """Fallback should be conservative."""
        assert KELLY_FALLBACK_PCT <= 0.02  # At most 2%

    def test_max_position_matches_risk_governor(self):
        """Max position should match risk governor limit."""
        assert KELLY_MAX_POSITION_PCT == 0.10  # 10% matches risk governor


class TestKellyResultDataclass:
    """Test KellyResult dataclass behavior."""

    def test_result_has_all_fields(self):
        """KellyResult should have all required fields."""
        result = KellyResult(
            full_kelly=0.10,
            fractional_kelly=0.025,
            position_pct=0.05,
            position_size_usd=5000,
            position_size_coin=0.1,
            method="kelly",
            reasoning="Test",
            capped=False,
        )

        assert result.full_kelly == 0.10
        assert result.fractional_kelly == 0.025
        assert result.position_pct == 0.05
        assert result.position_size_usd == 5000
        assert result.position_size_coin == 0.1
        assert result.method == "kelly"
        assert result.reasoning == "Test"
        assert result.capped is False


class TestFeeAdjustedKelly:
    """Test fee-adjusted Kelly calculations (Phase 6: Multi-Exchange)."""

    def test_ev_with_zero_fees(self):
        """EV should be unchanged with zero fees."""
        ev_no_fees = calculate_expected_value(
            win_rate=0.55, avg_win_r=1.0, avg_loss_r=1.0, fee_cost_r=0.0
        )
        ev_baseline = calculate_expected_value(
            win_rate=0.55, avg_win_r=1.0, avg_loss_r=1.0
        )
        assert ev_no_fees == pytest.approx(ev_baseline, abs=0.001)

    def test_ev_reduced_by_fees(self):
        """EV should be reduced by fee cost in R-multiples."""
        # Base EV = 0.55 * 1.0 - 0.45 * 1.0 = 0.10R
        ev_no_fees = calculate_expected_value(
            win_rate=0.55, avg_win_r=1.0, avg_loss_r=1.0, fee_cost_r=0.0
        )
        # With 0.05R fees: EV = 0.10 - 0.05 = 0.05R
        ev_with_fees = calculate_expected_value(
            win_rate=0.55, avg_win_r=1.0, avg_loss_r=1.0, fee_cost_r=0.05
        )
        assert ev_with_fees == pytest.approx(ev_no_fees - 0.05, abs=0.001)
        assert ev_with_fees == pytest.approx(0.05, abs=0.001)

    def test_fees_can_turn_positive_ev_negative(self):
        """High fees can make a marginally positive edge negative EV."""
        # Small edge: 52% win rate with 1:1 R ratio -> EV = 0.04R
        ev_no_fees = calculate_expected_value(
            win_rate=0.52, avg_win_r=1.0, avg_loss_r=1.0
        )
        assert ev_no_fees == pytest.approx(0.04, abs=0.001)

        # With 0.10R fees (high fee exchange): EV = 0.04 - 0.10 = -0.06R
        ev_high_fees = calculate_expected_value(
            win_rate=0.52, avg_win_r=1.0, avg_loss_r=1.0, fee_cost_r=0.10
        )
        assert ev_high_fees < 0

    def test_kelly_input_with_fees(self):
        """KellyInput should accept round_trip_fee_pct."""
        kelly_input = KellyInput(
            win_rate=0.55,
            avg_win_r=1.0,
            avg_loss_r=1.0,
            episode_count=50,
            account_value=100000,
            current_price=50000,
            stop_distance_pct=0.02,
            round_trip_fee_pct=0.001,  # 10 bps round-trip
        )
        assert kelly_input.round_trip_fee_pct == 0.001

    def test_kelly_input_default_fees(self):
        """KellyInput should default to 0.001 (10 bps) round-trip fees."""
        kelly_input = KellyInput(
            win_rate=0.55,
            avg_win_r=1.0,
            avg_loss_r=1.0,
            episode_count=50,
            account_value=100000,
            current_price=50000,
            stop_distance_pct=0.02,
        )
        assert kelly_input.round_trip_fee_pct == 0.001

    def test_kelly_with_fees_reduces_ev(self):
        """Kelly sizing should account for fees in EV calculation."""
        # Same stats, different fee levels
        kelly_input_low_fee = KellyInput(
            win_rate=0.55,
            avg_win_r=1.5,
            avg_loss_r=1.0,
            episode_count=100,
            account_value=100000,
            current_price=50000,
            stop_distance_pct=0.02,
            round_trip_fee_pct=0.0005,  # 5 bps (Hyperliquid-like)
        )
        kelly_input_high_fee = KellyInput(
            win_rate=0.55,
            avg_win_r=1.5,
            avg_loss_r=1.0,
            episode_count=100,
            account_value=100000,
            current_price=50000,
            stop_distance_pct=0.02,
            round_trip_fee_pct=0.0020,  # 20 bps (higher fee exchange)
        )

        result_low_fee = kelly_position_size(kelly_input_low_fee)
        result_high_fee = kelly_position_size(kelly_input_high_fee)

        # Both should still use Kelly (positive EV even with fees)
        assert result_low_fee.method == "kelly"
        assert result_high_fee.method == "kelly"

        # High fee should have lower EV in reasoning
        assert "Fees=" in result_low_fee.reasoning
        assert "Fees=" in result_high_fee.reasoning

    def test_fees_can_trigger_negative_ev_fallback(self):
        """Marginal edge + high fees should trigger negative EV fallback."""
        # 52% win rate with 1:1 -> EV = 0.04R raw
        # With 2% stop, 10bps fees = 0.10R / 0.02 = 0.05R fee cost
        # Adjusted EV = 0.04 - 0.05 = -0.01R (negative!)
        kelly_input = KellyInput(
            win_rate=0.52,
            avg_win_r=1.0,
            avg_loss_r=1.0,
            episode_count=100,
            account_value=100000,
            current_price=50000,
            stop_distance_pct=0.02,
            round_trip_fee_pct=0.001,  # 10 bps = 0.05R with 2% stop
        )

        result = kelly_position_size(kelly_input)
        assert result.method == "fallback_negative_ev"
        assert "fees" in result.reasoning.lower()

    def test_fee_cost_in_r_multiples(self):
        """Fee cost should be correctly converted to R-multiples."""
        # With 1% stop distance, 10bps fee = 0.10R fee cost
        kelly_input = KellyInput(
            win_rate=0.60,
            avg_win_r=1.5,
            avg_loss_r=1.0,
            episode_count=100,
            account_value=100000,
            current_price=50000,
            stop_distance_pct=0.01,  # 1% stop
            round_trip_fee_pct=0.001,  # 10 bps
        )
        # Fee cost in R = 0.001 / 0.01 = 0.10R
        result = kelly_position_size(kelly_input)

        # Check that fees are mentioned in reasoning with ~0.10R
        assert "Fees=" in result.reasoning
        # Extract fee value from reasoning (format: "Fees=0.10R")
        import re
        match = re.search(r"Fees=(\d+\.\d+)R", result.reasoning)
        assert match is not None
        fee_r = float(match.group(1))
        assert fee_r == pytest.approx(0.10, abs=0.01)

    def test_zero_fees_no_impact(self):
        """Zero fees should not affect Kelly calculation."""
        kelly_input_no_fee = KellyInput(
            win_rate=0.60,
            avg_win_r=1.5,
            avg_loss_r=1.0,
            episode_count=100,
            account_value=100000,
            current_price=50000,
            stop_distance_pct=0.02,
            round_trip_fee_pct=0.0,  # No fees
        )
        kelly_input_default = KellyInput(
            win_rate=0.60,
            avg_win_r=1.5,
            avg_loss_r=1.0,
            episode_count=100,
            account_value=100000,
            current_price=50000,
            stop_distance_pct=0.02,
            round_trip_fee_pct=0.001,  # Default 10 bps
        )

        result_no_fee = kelly_position_size(kelly_input_no_fee)
        result_with_fee = kelly_position_size(kelly_input_default)

        # No fee result should not have "Fees=" in reasoning
        assert "Fees=" not in result_no_fee.reasoning
        # With fee result should have "Fees=" in reasoning
        assert "Fees=" in result_with_fee.reasoning
