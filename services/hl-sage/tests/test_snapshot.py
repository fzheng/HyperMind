"""
Tests for Shadow Ledger snapshot module (Phase 3f: Selection Integrity).

These tests verify:
1. Thompson sampling with stored seeds (reproducibility)
2. Benjamini-Hochberg FDR control (correct k* finding)
3. Skill p-value computation
4. Cost-adjusted R-multiple estimation
5. TraderSnapshot dataclass behavior
"""
import pytest
import math
import sys
import os
from datetime import date

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.snapshot import (
    thompson_sample_nig,
    compute_skill_p_value,
    benjamini_hochberg_select,
    estimate_cost_r,
    TraderSnapshot,
    SNAPSHOT_MIN_EPISODES,
    SNAPSHOT_FDR_ALPHA,
)
from app.bandit import (
    NIG_PRIOR_M,
    NIG_PRIOR_KAPPA,
    NIG_PRIOR_ALPHA,
    NIG_PRIOR_BETA,
)


class TestThompsonSamplingWithSeeds:
    """Test Thompson sampling with deterministic seeds for reproducibility."""

    def test_same_seed_same_result(self):
        """Same seed should produce identical samples."""
        params = (0.5, 10.0, 5.0, 1.0)  # m, kappa, alpha, beta
        seed = 12345

        sample1 = thompson_sample_nig(*params, seed)
        sample2 = thompson_sample_nig(*params, seed)

        assert sample1 == sample2

    def test_different_seeds_different_results(self):
        """Different seeds should produce different samples."""
        params = (0.5, 10.0, 5.0, 1.0)

        sample1 = thompson_sample_nig(*params, 12345)
        sample2 = thompson_sample_nig(*params, 54321)

        assert sample1 != sample2

    def test_samples_centered_around_mean(self):
        """Samples should be centered around posterior mean m."""
        m = 0.5
        samples = [thompson_sample_nig(m, 20.0, 10.0, 1.0, seed) for seed in range(1000)]
        sample_mean = sum(samples) / len(samples)

        # With κ=20, samples should be close to m
        assert abs(sample_mean - m) < 0.15

    def test_high_kappa_low_variance(self):
        """High κ (confident) should produce low variance samples."""
        m = 0.5
        samples = [thompson_sample_nig(m, 100.0, 50.0, 1.0, seed) for seed in range(500)]
        variance = sum((s - m) ** 2 for s in samples) / len(samples)

        assert variance < 0.01

    def test_low_kappa_high_variance(self):
        """Low κ (uncertain) should produce high variance samples."""
        m = 0.5
        samples = [thompson_sample_nig(m, 1.0, 3.0, 1.0, seed) for seed in range(500)]
        variance = sum((s - m) ** 2 for s in samples) / len(samples)

        assert variance > 0.1

    def test_prior_params_work(self):
        """Should work with prior parameters."""
        sample = thompson_sample_nig(
            NIG_PRIOR_M,
            NIG_PRIOR_KAPPA,
            NIG_PRIOR_ALPHA,
            NIG_PRIOR_BETA,
            42,
        )
        assert isinstance(sample, float)
        assert not math.isnan(sample)
        assert not math.isinf(sample)

    def test_date_based_seed_reproducibility(self):
        """Date-based seeds should allow walk-forward replay."""
        date_seed = 20251211  # Dec 11, 2025
        trader_seed = date_seed + hash("0x1234") % 1000000

        # Same date + address = same seed = same result
        sample1 = thompson_sample_nig(0.3, 10.0, 5.0, 1.0, trader_seed)
        sample2 = thompson_sample_nig(0.3, 10.0, 5.0, 1.0, trader_seed)

        assert sample1 == sample2


class TestSkillPValue:
    """Test skill p-value computation for FDR qualification."""

    def test_positive_r_values_low_pvalue(self):
        """Consistently positive R-values should give low p-value."""
        # Generate R-values with positive mean
        r_values = [0.1 + (i % 5) * 0.02 for i in range(SNAPSHOT_MIN_EPISODES)]
        p_value = compute_skill_p_value(r_values)

        assert p_value is not None
        assert p_value < 0.05

    def test_negative_r_values_high_pvalue(self):
        """Consistently negative R-values should give high p-value."""
        r_values = [-0.1 - (i % 5) * 0.02 for i in range(SNAPSHOT_MIN_EPISODES)]
        p_value = compute_skill_p_value(r_values)

        assert p_value is not None
        assert p_value > 0.5

    def test_zero_mean_r_values_around_half(self):
        """Zero-mean R-values should give p-value around 0.5."""
        r_values = [0.1 * ((-1) ** i) for i in range(SNAPSHOT_MIN_EPISODES)]
        p_value = compute_skill_p_value(r_values)

        assert p_value is not None
        assert 0.3 < p_value < 0.7

    def test_insufficient_data_returns_none(self):
        """Less than min episodes should return None."""
        r_values = [0.1, 0.2, 0.3]  # Too few
        p_value = compute_skill_p_value(r_values)

        assert p_value is None

    def test_winsorization_applied(self):
        """Extreme R-values should be winsorized."""
        # Include extreme outliers
        r_values = [10.0] + [0.1] * (SNAPSHOT_MIN_EPISODES - 1)  # One huge outlier
        p_value1 = compute_skill_p_value(r_values)

        # Compare to capped values
        r_values_capped = [3.0] + [0.1] * (SNAPSHOT_MIN_EPISODES - 1)  # Capped at R_WINSORIZE_MAX
        p_value2 = compute_skill_p_value(r_values_capped)

        # Both should be similar due to winsorization
        assert abs(p_value1 - p_value2) < 0.1


class TestBenjaminiHochbergSelect:
    """Test Benjamini-Hochberg FDR control procedure."""

    def test_empty_input(self):
        """Empty input should return empty list."""
        result = benjamini_hochberg_select([])
        assert result == []

    def test_single_significant(self):
        """Single significant p-value should be selected."""
        traders = [("0x1", 0.01)]
        result = benjamini_hochberg_select(traders, alpha=0.10)

        assert "0x1" in result

    def test_single_not_significant(self):
        """Single non-significant p-value should not be selected."""
        traders = [("0x1", 0.50)]
        result = benjamini_hochberg_select(traders, alpha=0.10)

        assert result == []

    def test_correct_k_star_finding(self):
        """
        BH should find k* = max{i : p_i <= (i/n)*alpha}, NOT stop at first failure.

        CRITICAL: This tests the corrected BH implementation per Advisor A's feedback.
        """
        # Setup: 10 traders with p-values where BH should select first 5
        # p_1=0.01, p_2=0.02, p_3=0.03, p_4=0.04, p_5=0.05 should all pass
        # p_6=0.08 > (6/10)*0.10 = 0.06 - FAILS
        # But with wrong impl that breaks on first failure, would stop at p_6
        traders = [
            ("0x1", 0.01),  # p_1 <= (1/10)*0.10 = 0.01 ✓
            ("0x2", 0.02),  # p_2 <= (2/10)*0.10 = 0.02 ✓
            ("0x3", 0.025), # p_3 <= (3/10)*0.10 = 0.03 ✓
            ("0x4", 0.035), # p_4 <= (4/10)*0.10 = 0.04 ✓
            ("0x5", 0.045), # p_5 <= (5/10)*0.10 = 0.05 ✓
            ("0x6", 0.08),  # p_6 > (6/10)*0.10 = 0.06 ✗
            ("0x7", 0.09),  # p_7 > (7/10)*0.10 = 0.07 ✗
            ("0x8", 0.10),  # p_8 > (8/10)*0.10 = 0.08 ✗
            ("0x9", 0.15),  # p_9 > (9/10)*0.10 = 0.09 ✗
            ("0x10", 0.20), # p_10 > (10/10)*0.10 = 0.10 ✗
        ]

        result = benjamini_hochberg_select(traders, alpha=0.10)

        # Should select first 5 (k* = 5)
        assert len(result) == 5
        assert "0x1" in result
        assert "0x5" in result
        assert "0x6" not in result

    def test_does_not_break_on_first_failure(self):
        """
        BH should NOT stop at first p_i > threshold.

        Example: If p_5 fails but p_6 passes, should still include p_6.
        """
        # Tricky case: p_5 barely fails, but p_6 passes
        traders = [
            ("0x1", 0.005), # p_1 <= (1/10)*0.10 = 0.01 ✓
            ("0x2", 0.01),  # p_2 <= (2/10)*0.10 = 0.02 ✓
            ("0x3", 0.02),  # p_3 <= (3/10)*0.10 = 0.03 ✓
            ("0x4", 0.03),  # p_4 <= (4/10)*0.10 = 0.04 ✓
            ("0x5", 0.055), # p_5 > (5/10)*0.10 = 0.05 ✗ (barely)
            ("0x6", 0.058), # p_6 <= (6/10)*0.10 = 0.06 ✓ (passes!)
            ("0x7", 0.08),  # p_7 > (7/10)*0.10 = 0.07 ✗
            ("0x8", 0.10),  # etc
            ("0x9", 0.15),
            ("0x10", 0.20),
        ]

        result = benjamini_hochberg_select(traders, alpha=0.10)

        # Should find k* = 6 (0x6 passes), so select 6 traders
        assert len(result) == 6
        assert "0x6" in result

    def test_all_significant_selected(self):
        """All traders with significant p-values should be selected."""
        traders = [
            ("0x1", 0.001),
            ("0x2", 0.002),
            ("0x3", 0.003),
            ("0x4", 0.004),
            ("0x5", 0.005),
        ]
        result = benjamini_hochberg_select(traders, alpha=0.10)

        assert len(result) == 5

    def test_none_significant_empty(self):
        """If no p-values pass, should return empty."""
        traders = [
            ("0x1", 0.50),
            ("0x2", 0.60),
            ("0x3", 0.70),
        ]
        result = benjamini_hochberg_select(traders, alpha=0.10)

        assert result == []


class TestCostEstimation:
    """Test round-trip cost estimation as R-multiple."""

    def test_basic_cost_calculation(self):
        """Test cost calculation with typical values."""
        # BTC: price $50k, ATR $1k (2%)
        # Cost = 30bps = $150 per BTC
        # R-cost = $150 / $1000 = 0.15R
        cost_r = estimate_cost_r(avg_atr=1000, avg_price=50000)
        expected = (50000 * 0.0030) / 1000  # 30bps / ATR
        assert cost_r == pytest.approx(expected, rel=0.01)

    def test_zero_atr_returns_zero(self):
        """Zero ATR should return 0 to avoid division by zero."""
        cost_r = estimate_cost_r(avg_atr=0, avg_price=50000)
        assert cost_r == 0.0

    def test_zero_price_returns_zero(self):
        """Zero price should return 0."""
        cost_r = estimate_cost_r(avg_atr=1000, avg_price=0)
        assert cost_r == 0.0

    def test_negative_values_return_zero(self):
        """Negative values should return 0."""
        assert estimate_cost_r(avg_atr=-100, avg_price=50000) == 0.0
        assert estimate_cost_r(avg_atr=1000, avg_price=-50000) == 0.0


class TestTraderSnapshot:
    """Test TraderSnapshot dataclass behavior."""

    def test_default_values(self):
        """Test default values are set correctly."""
        snapshot = TraderSnapshot(
            address="0x1234",
            snapshot_date=date.today(),
            selection_version="3f.1",
        )

        assert snapshot.is_leaderboard_scanned is False
        assert snapshot.is_pool_selected is False
        assert snapshot.event_type == "active"
        assert snapshot.death_type is None
        assert snapshot.censor_type is None
        assert snapshot.fdr_qualified is False

    def test_nig_defaults_to_prior(self):
        """NIG params should default to prior values."""
        snapshot = TraderSnapshot(
            address="0x1234",
            snapshot_date=date.today(),
            selection_version="3f.1",
        )

        assert snapshot.nig_mu == NIG_PRIOR_M
        assert snapshot.nig_kappa == NIG_PRIOR_KAPPA
        assert snapshot.nig_alpha == NIG_PRIOR_ALPHA
        assert snapshot.nig_beta == NIG_PRIOR_BETA

    def test_thompson_draw_stored(self):
        """Thompson draw and seed should be storable."""
        snapshot = TraderSnapshot(
            address="0x1234",
            snapshot_date=date.today(),
            selection_version="3f.1",
            thompson_draw=0.35,
            thompson_seed=20251211001234,
        )

        assert snapshot.thompson_draw == 0.35
        assert snapshot.thompson_seed == 20251211001234

    def test_death_event_flags(self):
        """Death events should set correct flags."""
        snapshot = TraderSnapshot(
            address="0x1234",
            snapshot_date=date.today(),
            selection_version="3f.1",
            event_type="death",
            death_type="drawdown_80",
        )

        assert snapshot.event_type == "death"
        assert snapshot.death_type == "drawdown_80"
        assert snapshot.censor_type is None

    def test_censor_event_flags(self):
        """Censor events should set correct flags."""
        snapshot = TraderSnapshot(
            address="0x1234",
            snapshot_date=date.today(),
            selection_version="3f.1",
            event_type="censored",
            censor_type="inactive_30d",
        )

        assert snapshot.event_type == "censored"
        assert snapshot.censor_type == "inactive_30d"
        assert snapshot.death_type is None

    def test_gross_vs_net_r_stored(self):
        """Both gross and net R-multiples should be storable."""
        snapshot = TraderSnapshot(
            address="0x1234",
            snapshot_date=date.today(),
            selection_version="3f.1",
            avg_r_gross=0.25,
            avg_r_net=0.10,  # After 30bps round-trip cost
        )

        assert snapshot.avg_r_gross == 0.25
        assert snapshot.avg_r_net == 0.10
        assert snapshot.avg_r_gross > snapshot.avg_r_net


class TestSelectionIntegrity:
    """Integration tests for selection integrity requirements."""

    def test_fdr_controls_false_discovery_rate(self):
        """
        FDR control at α=0.10 means at most 10% of selected traders
        are expected to be false positives (skill = 0).
        """
        # Simulate 100 traders: 20 skilled (p < 0.05), 80 unskilled (p uniform)
        import random
        random.seed(42)

        skilled = [(f"skilled_{i}", random.uniform(0.001, 0.05)) for i in range(20)]
        unskilled = [(f"unskilled_{i}", random.uniform(0.05, 1.0)) for i in range(80)]

        traders = skilled + unskilled
        selected = benjamini_hochberg_select(traders, alpha=0.10)

        # Most selected should be skilled
        skilled_selected = sum(1 for addr in selected if addr.startswith("skilled_"))

        # At 10% FDR, expect ~90% of selected to be truly skilled
        if len(selected) > 0:
            skilled_rate = skilled_selected / len(selected)
            assert skilled_rate > 0.7, f"Skilled rate {skilled_rate:.2%} too low"

    def test_net_r_lower_than_gross(self):
        """Net R should always be lower than gross R due to costs."""
        avg_r_gross = 0.30

        # BTC: ATR $1k, price $50k
        cost_r = estimate_cost_r(avg_atr=1000, avg_price=50000)
        avg_r_net = avg_r_gross - cost_r

        assert avg_r_net < avg_r_gross
        assert avg_r_net > 0  # Should still be positive for good trader
