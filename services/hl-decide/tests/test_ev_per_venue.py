"""
Tests for Per-Venue EV Calculation (Phase 6.1 Gap 4)

Tests the ability to calculate and compare EV across different exchanges.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from app.consensus import (
    ConsensusDetector,
    calculate_ev,
    get_exchange_fees_bps,
    get_funding_cost_bps_sync,
    get_slippage_estimate_bps_sync,
    DEFAULT_HOLD_HOURS,
    SLIPPAGE_REFERENCE_SIZE_USD,
)


# =============================================================================
# Test EV Calculation Per Exchange
# =============================================================================


class TestCalculateEVForExchange:
    """Tests for ConsensusDetector.calculate_ev_for_exchange()"""

    @pytest.fixture
    def detector(self):
        """Create a consensus detector for testing."""
        return ConsensusDetector(target_exchange="hyperliquid")

    def test_calculate_ev_for_hyperliquid(self, detector):
        """Calculate EV for Hyperliquid exchange."""
        result = detector.calculate_ev_for_exchange(
            asset="BTC",
            direction="long",
            entry_price=100000,
            stop_price=99000,  # 1% stop
            p_win=0.55,
            exchange="hyperliquid",
            order_size_usd=10000,
        )

        assert "ev_net_r" in result
        assert "ev_gross_r" in result
        assert "ev_cost_r" in result
        assert "fees_bps" in result
        assert "slippage_bps" in result
        assert "funding_bps" in result
        assert result["exchange"] == "hyperliquid"

    def test_calculate_ev_for_bybit(self, detector):
        """Calculate EV for Bybit exchange."""
        result = detector.calculate_ev_for_exchange(
            asset="BTC",
            direction="long",
            entry_price=100000,
            stop_price=99000,
            p_win=0.55,
            exchange="bybit",
            order_size_usd=10000,
        )

        assert result["exchange"] == "bybit"
        assert "ev_net_r" in result
        # Bybit should have different fees than Hyperliquid
        assert result["fees_bps"] >= 0

    def test_ev_includes_direction_aware_funding(self, detector):
        """EV calculation includes direction-aware funding cost."""
        # For the same parameters, long and short may have different funding
        long_result = detector.calculate_ev_for_exchange(
            asset="BTC",
            direction="long",
            entry_price=100000,
            stop_price=99000,
            p_win=0.55,
            exchange="hyperliquid",
        )

        short_result = detector.calculate_ev_for_exchange(
            asset="BTC",
            direction="short",
            entry_price=100000,
            stop_price=101000,  # Stop above for shorts
            p_win=0.55,
            exchange="hyperliquid",
        )

        # Both should have funding_bps defined
        assert "funding_bps" in long_result
        assert "funding_bps" in short_result
        # They may differ due to direction

    def test_ev_uses_dynamic_hold_time(self, detector):
        """EV calculation uses dynamic hold time when not specified."""
        result = detector.calculate_ev_for_exchange(
            asset="BTC",
            direction="long",
            entry_price=100000,
            stop_price=99000,
            p_win=0.55,
            exchange="hyperliquid",
            hold_hours=None,  # Should use dynamic estimate
        )

        # Should have a hold_hours value
        assert "hold_hours" in result
        assert result["hold_hours"] > 0

    def test_ev_uses_specified_hold_time(self, detector):
        """EV calculation uses specified hold time when provided."""
        result = detector.calculate_ev_for_exchange(
            asset="BTC",
            direction="long",
            entry_price=100000,
            stop_price=99000,
            p_win=0.55,
            exchange="hyperliquid",
            hold_hours=48.0,  # Specify 48 hours
        )

        assert result["hold_hours"] == 48.0

    def test_ev_handles_different_order_sizes(self, detector):
        """Larger orders should have higher slippage."""
        small_order = detector.calculate_ev_for_exchange(
            asset="BTC",
            direction="long",
            entry_price=100000,
            stop_price=99000,
            p_win=0.55,
            exchange="hyperliquid",
            order_size_usd=1000,  # Small
        )

        large_order = detector.calculate_ev_for_exchange(
            asset="BTC",
            direction="long",
            entry_price=100000,
            stop_price=99000,
            p_win=0.55,
            exchange="hyperliquid",
            order_size_usd=100000,  # Large
        )

        # Larger order should have higher slippage
        assert large_order["slippage_bps"] >= small_order["slippage_bps"]
        # And thus lower net EV (all else equal)
        assert large_order["ev_net_r"] <= small_order["ev_net_r"]


# =============================================================================
# Test EV Comparison Across Exchanges
# =============================================================================


class TestCompareEVAcrossExchanges:
    """Tests for ConsensusDetector.compare_ev_across_exchanges()"""

    @pytest.fixture
    def detector(self):
        """Create a consensus detector for testing."""
        return ConsensusDetector(target_exchange="hyperliquid")

    def test_compare_default_exchanges(self, detector):
        """Compare EV across default exchanges (hyperliquid, bybit)."""
        result = detector.compare_ev_across_exchanges(
            asset="BTC",
            direction="long",
            entry_price=100000,
            stop_price=99000,
            p_win=0.55,
        )

        # Should have results for both exchanges
        assert "hyperliquid" in result
        assert "bybit" in result
        # Should have best exchange
        assert "best_exchange" in result
        assert "best_ev_net_r" in result
        # Best should be one of the exchanges
        assert result["best_exchange"] in ["hyperliquid", "bybit"]

    def test_compare_specified_exchanges(self, detector):
        """Compare EV for a specified list of exchanges."""
        result = detector.compare_ev_across_exchanges(
            asset="BTC",
            direction="long",
            entry_price=100000,
            stop_price=99000,
            p_win=0.55,
            exchanges=["hyperliquid"],  # Only one exchange
        )

        assert "hyperliquid" in result
        assert "bybit" not in result
        assert result["best_exchange"] == "hyperliquid"

    def test_compare_handles_errors_gracefully(self, detector):
        """Error in one exchange doesn't break comparison."""
        # Mock one exchange to fail
        with patch.object(
            detector,
            'calculate_ev_for_exchange',
            side_effect=[
                {"ev_net_r": 0.5, "exchange": "hyperliquid"},
                Exception("Bybit API error"),
            ]
        ):
            result = detector.compare_ev_across_exchanges(
                asset="BTC",
                direction="long",
                entry_price=100000,
                stop_price=99000,
                p_win=0.55,
                exchanges=["hyperliquid", "bybit"],
            )

        # Should still have both results (one with error)
        assert "hyperliquid" in result
        assert "bybit" in result
        assert "error" in result["bybit"]
        # Best should be the non-error one
        assert result["best_exchange"] == "hyperliquid"

    def test_compare_selects_highest_ev(self, detector):
        """Best exchange is the one with highest net EV."""
        # Mock to return known EV values
        mock_ev_hl = {
            "ev_net_r": 0.3,
            "ev_gross_r": 0.5,
            "ev_cost_r": 0.2,
            "fees_bps": 5,
            "slippage_bps": 2,
            "funding_bps": 1,
            "exchange": "hyperliquid",
            "hold_hours": 24,
        }
        mock_ev_bybit = {
            "ev_net_r": 0.4,  # Higher EV
            "ev_gross_r": 0.5,
            "ev_cost_r": 0.1,  # Lower costs
            "fees_bps": 3,
            "slippage_bps": 1,
            "funding_bps": 0.5,
            "exchange": "bybit",
            "hold_hours": 24,
        }

        with patch.object(
            detector,
            'calculate_ev_for_exchange',
            side_effect=[mock_ev_hl, mock_ev_bybit],
        ):
            result = detector.compare_ev_across_exchanges(
                asset="BTC",
                direction="long",
                entry_price=100000,
                stop_price=99000,
                p_win=0.55,
            )

        assert result["best_exchange"] == "bybit"
        assert result["best_ev_net_r"] == 0.4


# =============================================================================
# Test Exchange Fee Lookup
# =============================================================================


class TestExchangeFeeLookup:
    """Tests for exchange-specific fee lookup."""

    def test_hyperliquid_fees(self):
        """Hyperliquid has expected fee structure."""
        fees = get_exchange_fees_bps("hyperliquid")
        # Hyperliquid typically has lower fees
        assert fees >= 0
        assert fees <= 20  # Should be under 20 bps round-trip

    def test_bybit_fees(self):
        """Bybit has expected fee structure."""
        fees = get_exchange_fees_bps("bybit")
        assert fees >= 0
        # Bybit may have different fees than HL

    def test_unknown_exchange_fallback(self):
        """Unknown exchange falls back to default fees."""
        fees = get_exchange_fees_bps("unknown_exchange")
        # Should return some default, not error
        assert fees >= 0


# =============================================================================
# Test Integration with EV Gate
# =============================================================================


class TestEVGateIntegration:
    """Test that per-venue EV works with consensus detection."""

    def test_target_exchange_affects_ev(self):
        """Changing target exchange affects EV calculation."""
        detector_hl = ConsensusDetector(target_exchange="hyperliquid")
        detector_bybit = ConsensusDetector(target_exchange="bybit")

        ev_hl = detector_hl.calculate_ev_for_exchange(
            asset="BTC",
            direction="long",
            entry_price=100000,
            stop_price=99000,
            p_win=0.55,
            exchange=detector_hl.target_exchange,
        )

        ev_bybit = detector_bybit.calculate_ev_for_exchange(
            asset="BTC",
            direction="long",
            entry_price=100000,
            stop_price=99000,
            p_win=0.55,
            exchange=detector_bybit.target_exchange,
        )

        # Results should reflect different exchanges
        assert ev_hl["exchange"] == "hyperliquid"
        assert ev_bybit["exchange"] == "bybit"

    def test_set_target_exchange_works(self):
        """set_target_exchange changes the default exchange."""
        detector = ConsensusDetector(target_exchange="hyperliquid")
        assert detector.target_exchange == "hyperliquid"

        detector.set_target_exchange("bybit")
        assert detector.target_exchange == "bybit"

        detector.set_target_exchange("ASTER")  # Test case normalization
        assert detector.target_exchange == "aster"


# =============================================================================
# Test Cost Breakdown
# =============================================================================


class TestCostBreakdown:
    """Test that EV result includes detailed cost breakdown."""

    def test_ev_result_has_all_components(self):
        """EV result includes all cost components."""
        detector = ConsensusDetector()
        result = detector.calculate_ev_for_exchange(
            asset="BTC",
            direction="long",
            entry_price=100000,
            stop_price=99000,
            p_win=0.55,
            exchange="hyperliquid",
        )

        required_keys = [
            "ev_gross_r",
            "ev_cost_r",
            "ev_net_r",
            "funding_cost_r",
            "fees_bps",
            "slippage_bps",
            "funding_bps",
            "exchange",
            "hold_hours",
        ]

        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_cost_components_are_positive(self):
        """Cost components should be non-negative."""
        detector = ConsensusDetector()
        result = detector.calculate_ev_for_exchange(
            asset="BTC",
            direction="long",
            entry_price=100000,
            stop_price=99000,
            p_win=0.55,
            exchange="hyperliquid",
        )

        assert result["fees_bps"] >= 0
        assert result["slippage_bps"] >= 0
        # Funding can be negative (you may receive funding)
        assert result["hold_hours"] > 0
