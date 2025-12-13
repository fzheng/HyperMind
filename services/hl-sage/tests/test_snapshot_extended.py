"""
Extended tests for Shadow Ledger snapshot module (Phase 3f).

These tests cover edge cases and additional scenarios:
1. Death event detection edge cases
2. Censor event detection edge cases
3. Boundary conditions for FDR qualification
4. Thompson sampling statistical properties
5. Snapshot creation edge cases
"""
import pytest
import math
import sys
import os
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

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
    DEATH_DRAWDOWN_PCT,
    DEATH_ACCOUNT_FLOOR,
    CENSOR_INACTIVE_DAYS,
    R_WINSORIZE_MIN,
    R_WINSORIZE_MAX,
    ROUND_TRIP_COST_BPS,
)


class TestThompsonSamplingEdgeCases:
    """Edge cases for Thompson sampling."""

    def test_very_high_kappa_near_deterministic(self):
        """Very high kappa should produce near-deterministic samples."""
        m = 0.5
        # With very high kappa, variance should be very low
        samples = [thompson_sample_nig(m, 10000.0, 5000.0, 1.0, seed) for seed in range(100)]
        variance = sum((s - m) ** 2 for s in samples) / len(samples)
        assert variance < 0.001

    def test_negative_mean_handled(self):
        """Negative posterior mean should work correctly."""
        m = -0.3
        samples = [thompson_sample_nig(m, 10.0, 5.0, 1.0, seed) for seed in range(500)]
        sample_mean = sum(samples) / len(samples)
        assert abs(sample_mean - m) < 0.2

    def test_large_beta_high_variance(self):
        """Large beta should increase sample variance."""
        m = 0.5
        low_beta_samples = [thompson_sample_nig(m, 10.0, 5.0, 0.5, seed) for seed in range(500)]
        high_beta_samples = [thompson_sample_nig(m, 10.0, 5.0, 5.0, seed) for seed in range(500)]

        low_var = sum((s - m) ** 2 for s in low_beta_samples) / len(low_beta_samples)
        high_var = sum((s - m) ** 2 for s in high_beta_samples) / len(high_beta_samples)

        assert high_var > low_var

    def test_seed_zero_works(self):
        """Seed 0 should be valid."""
        sample = thompson_sample_nig(0.5, 10.0, 5.0, 1.0, 0)
        assert isinstance(sample, float)
        assert not math.isnan(sample)

    def test_large_seed_works(self):
        """Very large seeds should work."""
        sample = thompson_sample_nig(0.5, 10.0, 5.0, 1.0, 2**63 - 1)
        assert isinstance(sample, float)


class TestBHEdgeCases:
    """Edge cases for Benjamini-Hochberg procedure."""

    def test_all_same_pvalue(self):
        """All traders with same p-value."""
        traders = [(f"0x{i}", 0.05) for i in range(10)]
        result = benjamini_hochberg_select(traders, alpha=0.10)
        # All should pass: p_i = 0.05 <= (i/10)*0.10 for all i >= 5
        assert len(result) >= 5

    def test_descending_pvalues(self):
        """P-values in descending order (worst case for BH)."""
        # Reversed order - highest p-values first
        traders = [(f"0x{i}", 0.10 - i*0.01) for i in range(10)]
        result = benjamini_hochberg_select(traders, alpha=0.10)
        # BH sorts internally, so should still work
        assert len(result) > 0

    def test_very_small_alpha(self):
        """Very stringent alpha should select fewer traders."""
        traders = [
            ("0x1", 0.001),
            ("0x2", 0.01),
            ("0x3", 0.05),
        ]
        result_loose = benjamini_hochberg_select(traders, alpha=0.10)
        result_strict = benjamini_hochberg_select(traders, alpha=0.01)

        assert len(result_strict) <= len(result_loose)

    def test_pvalue_exactly_at_threshold(self):
        """P-value exactly at BH threshold."""
        # p_1 = 0.01 vs threshold (1/5)*0.10 = 0.02 -> passes
        # p_2 = 0.04 vs threshold (2/5)*0.10 = 0.04 -> passes (exactly at)
        traders = [
            ("0x1", 0.01),
            ("0x2", 0.04),  # Exactly at threshold
            ("0x3", 0.08),
            ("0x4", 0.10),
            ("0x5", 0.15),
        ]
        result = benjamini_hochberg_select(traders, alpha=0.10)
        assert "0x2" in result

    def test_single_trader_significant(self):
        """Single very significant trader."""
        traders = [("0x1", 0.001)]
        result = benjamini_hochberg_select(traders, alpha=0.10)
        assert result == ["0x1"]

    def test_large_number_of_traders(self):
        """BH should handle many traders efficiently."""
        import random
        random.seed(42)
        traders = [(f"0x{i}", random.uniform(0, 1)) for i in range(1000)]
        result = benjamini_hochberg_select(traders, alpha=0.10)
        # Should complete without error
        assert isinstance(result, list)


class TestSkillPValueEdgeCases:
    """Edge cases for skill p-value computation."""

    def test_exactly_min_episodes(self):
        """Exactly minimum episodes should compute p-value."""
        r_values = [0.1] * SNAPSHOT_MIN_EPISODES
        p_value = compute_skill_p_value(r_values)
        assert p_value is not None

    def test_one_less_than_min_episodes(self):
        """One less than min should return None."""
        r_values = [0.1] * (SNAPSHOT_MIN_EPISODES - 1)
        p_value = compute_skill_p_value(r_values)
        assert p_value is None

    def test_all_same_positive_values(self):
        """All identical positive R-values."""
        r_values = [0.5] * SNAPSHOT_MIN_EPISODES
        p_value = compute_skill_p_value(r_values)
        # Should be very significant (low p-value)
        assert p_value is not None
        assert p_value < 0.01

    def test_all_same_negative_values(self):
        """All identical negative R-values."""
        r_values = [-0.5] * SNAPSHOT_MIN_EPISODES
        p_value = compute_skill_p_value(r_values)
        # Should be very insignificant (high p-value for one-sided)
        assert p_value is not None
        assert p_value > 0.9

    def test_extreme_outliers_winsorized(self):
        """Extreme outliers should be clipped."""
        # Mix of normal values with extreme outliers
        r_values = [0.1] * (SNAPSHOT_MIN_EPISODES - 2) + [100.0, -100.0]
        p_value = compute_skill_p_value(r_values)
        assert p_value is not None
        # Should still produce reasonable p-value after winsorization

    def test_very_small_variance(self):
        """Near-zero variance should not cause errors."""
        r_values = [0.1 + i * 0.0001 for i in range(SNAPSHOT_MIN_EPISODES)]
        p_value = compute_skill_p_value(r_values)
        assert p_value is not None


class TestCostEstimationEdgeCases:
    """Edge cases for cost estimation."""

    def test_very_small_atr(self):
        """Very small ATR should produce high cost in R-terms."""
        cost_r = estimate_cost_r(avg_atr=1, avg_price=50000)
        # Cost should be high relative to ATR
        assert cost_r > 1.0

    def test_very_large_atr(self):
        """Very large ATR should produce low cost in R-terms."""
        cost_r = estimate_cost_r(avg_atr=10000, avg_price=50000)
        # Cost should be low relative to ATR
        assert cost_r < 0.05

    def test_eth_typical_values(self):
        """Typical ETH values should produce reasonable cost."""
        # ETH: $3000, ATR ~$100
        cost_r = estimate_cost_r(avg_atr=100, avg_price=3000)
        expected = 3000 * (ROUND_TRIP_COST_BPS / 10000) / 100
        assert cost_r == pytest.approx(expected, rel=0.01)

    def test_negative_inputs_safe(self):
        """Negative inputs should return 0, not error."""
        assert estimate_cost_r(avg_atr=-100, avg_price=50000) == 0
        assert estimate_cost_r(avg_atr=100, avg_price=-50000) == 0
        assert estimate_cost_r(avg_atr=-100, avg_price=-50000) == 0


class TestTraderSnapshotEdgeCases:
    """Edge cases for TraderSnapshot dataclass."""

    def test_all_universes_true(self):
        """Trader in all universes."""
        snapshot = TraderSnapshot(
            address="0x1234",
            snapshot_date=date.today(),
            selection_version="3f.1",
            is_leaderboard_scanned=True,
            is_candidate_filtered=True,
            is_quality_qualified=True,
            is_pool_selected=True,
            is_pinned_custom=True,
        )
        assert all([
            snapshot.is_leaderboard_scanned,
            snapshot.is_candidate_filtered,
            snapshot.is_quality_qualified,
            snapshot.is_pool_selected,
            snapshot.is_pinned_custom,
        ])

    def test_death_and_censor_mutually_exclusive(self):
        """Death and censor should not both be set in practice."""
        # Death takes precedence
        death_snapshot = TraderSnapshot(
            address="0x1234",
            snapshot_date=date.today(),
            selection_version="3f.1",
            event_type="death",
            death_type="liquidation",
            censor_type=None,
        )
        assert death_snapshot.death_type is not None
        assert death_snapshot.censor_type is None

        # Censor only if not dead
        censor_snapshot = TraderSnapshot(
            address="0x5678",
            snapshot_date=date.today(),
            selection_version="3f.1",
            event_type="censored",
            death_type=None,
            censor_type="inactive_30d",
        )
        assert censor_snapshot.death_type is None
        assert censor_snapshot.censor_type is not None

    def test_selection_rank_ordering(self):
        """Selection rank should support ordering."""
        snapshots = [
            TraderSnapshot(address=f"0x{i}", snapshot_date=date.today(),
                          selection_version="3f.1", selection_rank=i)
            for i in [3, 1, 2]
        ]
        sorted_by_rank = sorted(snapshots, key=lambda s: s.selection_rank or 999)
        assert sorted_by_rank[0].address == "0x1"
        assert sorted_by_rank[1].address == "0x2"
        assert sorted_by_rank[2].address == "0x3"

    def test_extreme_r_values(self):
        """Extreme R-multiple values should be storable."""
        snapshot = TraderSnapshot(
            address="0x1234",
            snapshot_date=date.today(),
            selection_version="3f.1",
            avg_r_gross=10.5,  # Very high R
            avg_r_net=-5.0,    # Negative after costs
        )
        assert snapshot.avg_r_gross == 10.5
        assert snapshot.avg_r_net == -5.0


class TestDeathDetectionLogic:
    """Test death detection criteria."""

    def test_drawdown_threshold(self):
        """Drawdown threshold should be 80%."""
        assert DEATH_DRAWDOWN_PCT == 0.80

    def test_account_floor_threshold(self):
        """Account floor should be $10k."""
        assert DEATH_ACCOUNT_FLOOR == 10000

    def test_drawdown_calculation(self):
        """Drawdown should be (peak - current) / peak."""
        peak = 100000
        current = 15000
        drawdown = (peak - current) / peak
        assert drawdown == 0.85
        assert drawdown >= DEATH_DRAWDOWN_PCT  # Would trigger death


class TestCensorDetectionLogic:
    """Test censor detection criteria."""

    def test_inactive_threshold(self):
        """Inactive threshold should be 30 days."""
        assert CENSOR_INACTIVE_DAYS == 30

    def test_inactive_boundary(self):
        """29 days should not be inactive, 30 should."""
        last_fill = date.today() - timedelta(days=29)
        days_since = (date.today() - last_fill).days
        assert days_since < CENSOR_INACTIVE_DAYS

        last_fill_old = date.today() - timedelta(days=30)
        days_since_old = (date.today() - last_fill_old).days
        assert days_since_old >= CENSOR_INACTIVE_DAYS


class TestWinsorization:
    """Test R-value winsorization bounds."""

    def test_winsorize_bounds(self):
        """Winsorization bounds should be symmetric."""
        # Default bounds are ±2.0 (configurable via env)
        assert R_WINSORIZE_MIN == -2.0
        assert R_WINSORIZE_MAX == 2.0
        assert abs(R_WINSORIZE_MIN) == R_WINSORIZE_MAX

    def test_winsorize_effect_on_mean(self):
        """Winsorization should limit impact of outliers."""
        # Without winsorization, mean would be heavily skewed
        values_with_outlier = [0.1] * 29 + [100.0]  # One huge outlier
        raw_mean = sum(values_with_outlier) / len(values_with_outlier)

        # With winsorization at ±2.0
        clipped = [max(R_WINSORIZE_MIN, min(R_WINSORIZE_MAX, v)) for v in values_with_outlier]
        clipped_mean = sum(clipped) / len(clipped)

        assert clipped_mean < raw_mean
        assert clipped_mean < 0.5  # Should be much more reasonable


class TestFDRQualificationIntegration:
    """Integration tests for FDR qualification pipeline."""

    def test_full_qualification_pipeline(self):
        """Test complete pipeline: R-values -> p-value -> BH selection."""
        import random
        random.seed(42)

        # Create 20 traders with varying skill
        traders_data = []
        for i in range(20):
            if i < 5:
                # Skilled traders: positive R-values
                r_values = [random.gauss(0.3, 0.1) for _ in range(SNAPSHOT_MIN_EPISODES)]
            else:
                # Unskilled traders: zero-mean R-values
                r_values = [random.gauss(0, 0.2) for _ in range(SNAPSHOT_MIN_EPISODES)]

            p_value = compute_skill_p_value(r_values)
            if p_value is not None:
                traders_data.append((f"0x{i:04d}", p_value))

        # Run BH selection
        selected = benjamini_hochberg_select(traders_data, alpha=SNAPSHOT_FDR_ALPHA)

        # Skilled traders (0x0000 to 0x0004) should be more likely selected
        skilled_selected = sum(1 for addr in selected if int(addr[2:], 16) < 5)

        # At least some skilled traders should be selected
        assert skilled_selected >= 2

    def test_effect_size_gate_integration(self):
        """Effect size gate should filter marginally significant traders."""
        from app.snapshot import SNAPSHOT_MIN_AVG_R_NET

        # Trader with significant p-value but low effect size
        snapshot = TraderSnapshot(
            address="0x1234",
            snapshot_date=date.today(),
            selection_version="3f.1",
            skill_p_value=0.01,  # Very significant
            avg_r_net=0.02,     # Below 0.05 threshold
            fdr_qualified=True,
        )

        # Even if FDR qualified, effect size should gate
        passes_effect_gate = snapshot.avg_r_net >= SNAPSHOT_MIN_AVG_R_NET
        assert passes_effect_gate is False


class TestConfigurableThresholds:
    """Test that thresholds are configurable via environment variables."""

    def test_min_episodes_is_configurable(self):
        """SNAPSHOT_MIN_EPISODES should be loaded from environment."""
        # The actual value depends on env, but should be a positive int
        assert isinstance(SNAPSHOT_MIN_EPISODES, int)
        assert SNAPSHOT_MIN_EPISODES > 0

    def test_fdr_alpha_is_configurable(self):
        """SNAPSHOT_FDR_ALPHA should be loaded from environment."""
        assert isinstance(SNAPSHOT_FDR_ALPHA, float)
        assert 0 < SNAPSHOT_FDR_ALPHA <= 1.0

    def test_death_drawdown_is_configurable(self):
        """DEATH_DRAWDOWN_PCT should be loaded from environment."""
        assert isinstance(DEATH_DRAWDOWN_PCT, float)
        assert 0 < DEATH_DRAWDOWN_PCT <= 1.0

    def test_death_floor_is_configurable(self):
        """DEATH_ACCOUNT_FLOOR should be loaded from environment."""
        assert isinstance(DEATH_ACCOUNT_FLOOR, (int, float))
        assert DEATH_ACCOUNT_FLOOR >= 0

    def test_censor_inactive_days_is_configurable(self):
        """CENSOR_INACTIVE_DAYS should be loaded from environment."""
        assert isinstance(CENSOR_INACTIVE_DAYS, int)
        assert CENSOR_INACTIVE_DAYS > 0

    def test_round_trip_cost_is_configurable(self):
        """ROUND_TRIP_COST_BPS should be loaded from environment."""
        assert isinstance(ROUND_TRIP_COST_BPS, float)
        assert ROUND_TRIP_COST_BPS >= 0

    def test_winsorize_bounds_are_configurable(self):
        """R_WINSORIZE bounds should be loaded from environment."""
        assert isinstance(R_WINSORIZE_MIN, float)
        assert isinstance(R_WINSORIZE_MAX, float)
        assert R_WINSORIZE_MIN < R_WINSORIZE_MAX

    def test_default_values_are_reasonable(self):
        """Default configuration values should be production-ready."""
        # Production defaults (may be overridden by env)
        # These tests verify the code handles the values correctly

        # Min episodes: should be >= 5 for statistical validity
        assert SNAPSHOT_MIN_EPISODES >= 5

        # FDR alpha: should be reasonable (0.01 to 0.20)
        assert 0.01 <= SNAPSHOT_FDR_ALPHA <= 0.20

        # Death thresholds
        assert DEATH_DRAWDOWN_PCT >= 0.50  # At least 50% drawdown
        assert DEATH_ACCOUNT_FLOOR >= 1000  # At least $1k

        # Censor threshold
        assert CENSOR_INACTIVE_DAYS >= 7  # At least 1 week


class TestThresholdEffectsOnQualification:
    """Test how different threshold values affect qualification."""

    def test_lower_min_episodes_qualifies_more(self):
        """Lower min_episodes threshold should qualify more traders."""
        # With min_episodes=5, a trader with 5 episodes qualifies
        r_values_5 = [0.3] * 5
        p_value_5 = compute_skill_p_value(r_values_5) if len(r_values_5) >= SNAPSHOT_MIN_EPISODES else None

        # With min_episodes=30, same trader doesn't qualify
        r_values_short = [0.3] * 5
        # If SNAPSHOT_MIN_EPISODES > 5, this would return None
        # We test the function behavior with different input sizes

        # Direct test: p-value computation requires enough samples
        if SNAPSHOT_MIN_EPISODES <= 5:
            assert p_value_5 is not None
        else:
            assert p_value_5 is None

    def test_higher_fdr_alpha_selects_more(self):
        """Higher FDR alpha should select more traders (less stringent)."""
        traders = [
            ("0x1", 0.03),  # Significant
            ("0x2", 0.06),  # Marginal
            ("0x3", 0.12),  # Not significant at 0.10
            ("0x4", 0.18),  # Not significant
        ]

        selected_strict = benjamini_hochberg_select(traders, alpha=0.05)
        selected_loose = benjamini_hochberg_select(traders, alpha=0.20)

        # Looser alpha should select at least as many
        assert len(selected_loose) >= len(selected_strict)

    def test_death_drawdown_threshold_effect(self):
        """Different drawdown thresholds change death detection."""
        # At 80% threshold: 85% drawdown = death
        peak = 100000
        current_at_85_drawdown = 15000  # 85% drawdown

        drawdown = (peak - current_at_85_drawdown) / peak
        assert drawdown == 0.85

        # At 80% threshold, this is death
        is_death_at_80 = drawdown >= 0.80
        assert is_death_at_80 is True

        # At 90% threshold, this is not death
        is_death_at_90 = drawdown >= 0.90
        assert is_death_at_90 is False

    def test_cost_affects_net_r(self):
        """Higher costs reduce net R more."""
        from app.snapshot import estimate_cost_r

        # Same trade, different cost assumptions
        low_cost = estimate_cost_r(avg_atr=1000, avg_price=50000)  # Uses ROUND_TRIP_COST_BPS

        # Manually compute what higher cost would be
        # If ROUND_TRIP_COST_BPS is 30, then:
        # cost_usd = 50000 * 0.0030 = 150
        # cost_r = 150 / 1000 = 0.15
        expected_cost_r = (50000 * (ROUND_TRIP_COST_BPS / 10000)) / 1000
        assert low_cost == pytest.approx(expected_cost_r, rel=0.01)

        # Net R would be gross - cost
        gross_r = 0.30
        net_r = gross_r - low_cost
        assert net_r < gross_r


class TestThresholdValidation:
    """Test that invalid threshold values are handled gracefully."""

    def test_p_value_with_zero_variance(self):
        """P-value computation should handle zero variance data."""
        # All identical values = zero variance
        r_values = [0.1] * max(SNAPSHOT_MIN_EPISODES, 30)
        p_value = compute_skill_p_value(r_values)

        # Should return a p-value (likely very significant)
        assert p_value is not None
        assert p_value < 0.01  # Highly significant positive mean

    def test_bh_with_all_same_pvalues(self):
        """BH should handle all identical p-values."""
        traders = [(f"0x{i}", 0.05) for i in range(10)]
        result = benjamini_hochberg_select(traders, alpha=0.10)

        # Should select some (p=0.05 <= (i/10)*0.10 for i >= 5)
        assert len(result) >= 5

    def test_bh_with_extreme_alpha(self):
        """BH should work with extreme alpha values."""
        traders = [(f"0x{i}", 0.01 * (i + 1)) for i in range(10)]

        # Very loose alpha = select all
        result_loose = benjamini_hochberg_select(traders, alpha=1.0)
        assert len(result_loose) == 10

        # Very strict alpha = select few or none
        result_strict = benjamini_hochberg_select(traders, alpha=0.001)
        assert len(result_strict) <= 2
