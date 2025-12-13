"""
Tests for decision_logger module.
"""

import pytest
from datetime import datetime, timezone

from app.decision_logger import (
    GateResult,
    generate_reasoning,
)


class TestGenerateReasoning:
    """Tests for generate_reasoning function."""

    def test_signal_reasoning(self):
        """Test reasoning for a successful signal."""
        gates = [
            GateResult(name="supermajority", passed=True, value=0.75, threshold=0.70),
            GateResult(name="effective_k", passed=True, value=3.5, threshold=2.0),
            GateResult(name="ev_gate", passed=True, value=0.25, threshold=0.20),
        ]

        result = generate_reasoning(
            decision_type="signal",
            symbol="BTC",
            direction="long",
            trader_count=5,
            agreement_pct=0.75,
            effective_k=3.5,
            gates=gates,
        )

        assert "5 Alpha Pool traders" in result
        assert "LONG BTC" in result
        assert "75%" in result
        assert "effK=3.5" in result
        assert "All consensus gates passed" in result

    def test_skip_reasoning_supermajority_failed(self):
        """Test reasoning when supermajority gate fails."""
        gates = [
            GateResult(name="supermajority", passed=False, value=0.55, threshold=0.70),
        ]

        result = generate_reasoning(
            decision_type="skip",
            symbol="ETH",
            direction="short",
            trader_count=4,
            agreement_pct=0.55,
            effective_k=0.0,
            gates=gates,
        )

        assert "Skipped" in result
        assert "4 traders" in result
        assert "55% agreement" in result
        assert "need 70%" in result

    def test_skip_reasoning_effk_failed(self):
        """Test reasoning when effective-K gate fails."""
        gates = [
            GateResult(name="supermajority", passed=True, value=0.80, threshold=0.70),
            GateResult(name="effective_k", passed=False, value=1.5, threshold=2.0),
        ]

        result = generate_reasoning(
            decision_type="skip",
            symbol="BTC",
            direction="long",
            trader_count=5,
            agreement_pct=0.80,
            effective_k=1.5,
            gates=gates,
        )

        assert "Skipped" in result
        assert "effK=1.5 too low" in result
        assert "need 2.0" in result

    def test_skip_reasoning_freshness_failed(self):
        """Test reasoning when freshness gate fails."""
        gates = [
            GateResult(name="supermajority", passed=True, value=0.80, threshold=0.70),
            GateResult(name="effective_k", passed=True, value=3.0, threshold=2.0),
            GateResult(name="freshness", passed=False, value=180, threshold=150),
        ]

        result = generate_reasoning(
            decision_type="skip",
            symbol="ETH",
            direction="short",
            trader_count=4,
            agreement_pct=0.80,
            effective_k=3.0,
            gates=gates,
        )

        assert "Skipped" in result
        assert "signal 180s stale" in result
        assert "max 150s" in result

    def test_skip_reasoning_price_band_failed(self):
        """Test reasoning when price band gate fails."""
        gates = [
            GateResult(name="price_band", passed=False, value=0.35, threshold=0.25),
        ]

        result = generate_reasoning(
            decision_type="skip",
            symbol="BTC",
            direction="long",
            trader_count=5,
            agreement_pct=0.80,
            effective_k=3.0,
            gates=gates,
        )

        assert "Skipped" in result
        assert "price drifted 0.35R" in result
        assert "max 0.25R" in result

    def test_skip_reasoning_ev_failed(self):
        """Test reasoning when EV gate fails."""
        gates = [
            GateResult(name="ev_gate", passed=False, value=0.15, threshold=0.20),
        ]

        result = generate_reasoning(
            decision_type="skip",
            symbol="ETH",
            direction="short",
            trader_count=6,
            agreement_pct=0.85,
            effective_k=4.0,
            gates=gates,
        )

        assert "Skipped" in result
        assert "EV=0.15R below threshold" in result
        assert "0.20R" in result

    def test_skip_reasoning_atr_invalid(self):
        """Test reasoning when ATR validity fails."""
        gates = [
            GateResult(
                name="atr_validity",
                passed=False,
                value=0.0,
                threshold=1.0,
                detail="Using hardcoded fallback (strict mode enabled)",
            ),
        ]

        result = generate_reasoning(
            decision_type="skip",
            symbol="BTC",
            direction="long",
            trader_count=5,
            agreement_pct=0.80,
            effective_k=3.0,
            gates=gates,
        )

        assert "Skipped" in result
        assert "ATR data invalid" in result
        assert "hardcoded fallback" in result

    def test_skip_reasoning_min_traders_failed(self):
        """Test reasoning when minimum traders gate fails."""
        gates = [
            GateResult(name="min_traders", passed=False, value=2.0, threshold=3.0),
        ]

        result = generate_reasoning(
            decision_type="skip",
            symbol="BTC",
            direction="none",
            trader_count=2,
            agreement_pct=0.0,
            effective_k=0.0,
            gates=gates,
        )

        assert "Skipped" in result
        assert "only 2 traders" in result
        assert "need 3" in result

    def test_risk_reject_reasoning(self):
        """Test reasoning for risk rejection."""
        gates = [
            GateResult(name="supermajority", passed=True, value=0.80, threshold=0.70),
            GateResult(name="effective_k", passed=True, value=3.5, threshold=2.0),
            GateResult(name="ev_gate", passed=True, value=0.25, threshold=0.20),
            GateResult(name="risk_limits", passed=False, value=0.52, threshold=1.0, detail="Confidence 0.52 < minimum 0.55"),
        ]

        risk_checks = [{"name": "risk_limits", "passed": False, "reason": "Confidence 0.52 < minimum 0.55"}]

        result = generate_reasoning(
            decision_type="risk_reject",
            symbol="BTC",
            direction="long",
            trader_count=5,
            agreement_pct=0.80,
            effective_k=3.5,
            gates=gates,
            risk_checks=risk_checks,
        )

        assert "Consensus detected but rejected by risk limits" in result
        assert "5 traders" in result
        assert "80% agreement" in result
        assert "Confidence 0.52 < minimum 0.55" in result

    def test_skip_reasoning_multiple_failures(self):
        """Test reasoning when multiple gates fail."""
        gates = [
            GateResult(name="supermajority", passed=False, value=0.55, threshold=0.70),
            GateResult(name="effective_k", passed=False, value=1.2, threshold=2.0),
        ]

        result = generate_reasoning(
            decision_type="skip",
            symbol="BTC",
            direction="long",
            trader_count=4,
            agreement_pct=0.55,
            effective_k=1.2,
            gates=gates,
        )

        assert "Skipped" in result
        assert "55% agreement" in result
        assert "need 70%" in result
        assert "effK=1.2 too low" in result


class TestGateResult:
    """Tests for GateResult dataclass."""

    def test_gate_result_creation(self):
        """Test creating a gate result."""
        gate = GateResult(
            name="supermajority",
            passed=True,
            value=0.75,
            threshold=0.70,
        )

        assert gate.name == "supermajority"
        assert gate.passed is True
        assert gate.value == 0.75
        assert gate.threshold == 0.70
        assert gate.detail == ""

    def test_gate_result_with_detail(self):
        """Test creating a gate result with detail."""
        gate = GateResult(
            name="atr_validity",
            passed=False,
            value=0.0,
            threshold=1.0,
            detail="ATR data stale (>1 hour old)",
        )

        assert gate.name == "atr_validity"
        assert gate.passed is False
        assert gate.detail == "ATR data stale (>1 hour old)"
