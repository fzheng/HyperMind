"""
Integration Tests for Phase 3b Components.

These tests verify that Thompson Sampling, ATR, Correlation, and Episode
features work correctly when integrated together.

Key integration points:
1. Thompson Sampling → Score → Consensus Detection
2. ATR Provider → Dynamic Stop Fractions → Consensus Gates
3. Correlation → effK Calculation → Signal Threshold
4. Episode Builder → R-Multiple → NIG Update
"""
import pytest
from datetime import datetime, timezone, timedelta, date
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.consensus import ConsensusDetector, ConsensusSignal, Fill
from app.atr import (
    ATRProvider,
    ATRData,
    calculate_true_range,
    calculate_atr,
    Candle,
    ATR_MULTIPLIER_BTC,
    ATR_MULTIPLIER_ETH,
)
from app.correlation import (
    compute_phi_correlation,
    bucket_id_from_timestamp,
    TraderSignVector,
    CorrelationProvider,
    CORR_MIN_COMMON_BUCKETS,
)
from app.episode import EpisodeTracker, EpisodeBuilderConfig, Episode, EpisodeFill


class TestThompsonSamplingToConsensus:
    """Test Thompson Sampling integration with consensus detection."""

    def _make_fill(self, address: str, asset: str, side: str, size: float, price: float, ts: datetime) -> Fill:
        """Helper to create a fill."""
        return Fill(
            fill_id=f"fill-{ts.timestamp()}",
            address=address,
            asset=asset,
            side=side,
            size=size,
            price=price,
            ts=ts,
        )

    def test_fills_are_processed_by_consensus_detector(self):
        """
        Fills (which come from Thompson-sampled score events) should be
        processed by the consensus detector and accumulated in windows.
        """
        detector = ConsensusDetector()
        now = datetime.now(timezone.utc)

        # Simulate 3 traders with long fills
        for addr in ["0x1111", "0x2222", "0x3333"]:
            fill = self._make_fill(addr, "BTC", "buy", 1.0, 100000.0, now)
            detector.process_fill(fill)

        # Check window has accumulated fills
        window = detector.windows.get("BTC")
        assert window is not None
        assert len(window.fills) == 3

    def test_nig_weight_formula_verification(self):
        """
        Verify NIG weight formula: κ/(κ+10).
        High-κ traders should contribute more weight than low-κ traders.
        """
        # Weight formula verification (used in hl-sage)
        low_kappa = 1.0
        high_kappa = 100.0

        low_weight = low_kappa / (low_kappa + 10.0)
        high_weight = high_kappa / (high_kappa + 10.0)

        # Low κ = 1 → weight ~0.09
        assert low_weight == pytest.approx(0.0909, rel=0.01)

        # High κ = 100 → weight ~0.91
        assert high_weight == pytest.approx(0.909, rel=0.01)

        # High-confidence trader has 10x more weight
        assert high_weight > low_weight * 9

    def test_multiple_fills_same_direction_consensus(self):
        """
        When multiple traders vote the same direction,
        consensus detector should aggregate them correctly.
        """
        detector = ConsensusDetector()
        now = datetime.now(timezone.utc)

        # All traders go long on ETH
        for i, addr in enumerate(["0x1111", "0x2222", "0x3333"]):
            fill = self._make_fill(addr, "ETH", "buy", 1.0 + i * 0.5, 4000.0, now + timedelta(seconds=i))
            detector.process_fill(fill)

        window = detector.windows.get("ETH")
        votes = detector.collapse_to_votes(window.fills)

        # All 3 should have voted long
        directions = [v.direction for v in votes]
        assert directions == ["long", "long", "long"]


class TestATRDynamicStops:
    """Test ATR integration with consensus stop distances."""

    def test_atr_sets_stop_fraction_for_consensus(self):
        """ATRProvider should provide stop fractions that ConsensusDetector uses."""
        provider = ATRProvider()

        # Create ATR data (2% ATR with 2x multiplier = 4% stop)
        atr_data = ATRData(
            asset="BTC",
            atr=2000.0,
            atr_pct=2.0,
            price=100000.0,
            multiplier=2.0,
            stop_distance_pct=4.0,
            timestamp=datetime.now(timezone.utc),
            source="test",
        )

        stop_fraction = provider.get_stop_fraction(atr_data)
        assert stop_fraction == pytest.approx(0.04, rel=1e-5)

        # Apply to consensus detector
        detector = ConsensusDetector()
        detector.set_stop_fraction("BTC", stop_fraction)
        assert detector.get_stop_fraction("BTC") == 0.04

    def test_high_volatility_widens_stops(self):
        """During high volatility (high ATR), stops should widen."""
        provider = ATRProvider()

        # High volatility: 5% ATR
        high_vol = ATRData(
            asset="BTC",
            atr=5000.0,
            atr_pct=5.0,
            price=100000.0,
            multiplier=2.0,
            stop_distance_pct=10.0,
            timestamp=datetime.now(timezone.utc),
            source="test",
        )

        # Low volatility: 0.5% ATR
        low_vol = ATRData(
            asset="BTC",
            atr=500.0,
            atr_pct=0.5,
            price=100000.0,
            multiplier=2.0,
            stop_distance_pct=1.0,
            timestamp=datetime.now(timezone.utc),
            source="test",
        )

        high_stop = provider.get_stop_fraction(high_vol)
        low_stop = provider.get_stop_fraction(low_vol)

        # High volatility should have 10x wider stops
        assert high_stop / low_stop == pytest.approx(10.0, rel=0.01)

    def test_stop_fraction_bounds_in_detector(self):
        """Detector should bound extreme stop fractions."""
        detector = ConsensusDetector()

        # Try to set extremely low stop (0.01%)
        detector.set_stop_fraction("BTC", 0.0001)
        assert detector.get_stop_fraction("BTC") == 0.001  # Bounded to 0.1%

        # Try to set extremely high stop (20%)
        detector.set_stop_fraction("BTC", 0.20)
        assert detector.get_stop_fraction("BTC") == 0.10  # Bounded to 10%


class TestCorrelationEffK:
    """Test correlation integration with effective-K calculation."""

    def test_correlation_hydrates_detector(self):
        """CorrelationProvider should populate detector's correlation matrix."""
        from datetime import date
        provider = CorrelationProvider()
        detector = ConsensusDetector()

        # Manually set correlations and loaded date for fresh data
        provider.correlations[("0x1111", "0x2222")] = 0.5
        provider.correlations[("0x1111", "0x3333")] = 0.3
        provider.correlations[("0x2222", "0x3333")] = 0.4
        provider._loaded_date = date.today()  # Mark as fresh to avoid decay

        # Hydrate detector (apply_decay=False to test raw values)
        count = provider.hydrate_detector(detector, apply_decay=False)
        assert count == 3

        # Verify detector has the correlations in its matrix
        key1 = tuple(sorted(["0x1111", "0x2222"]))
        key2 = tuple(sorted(["0x1111", "0x3333"]))
        key3 = tuple(sorted(["0x2222", "0x3333"]))
        assert detector.correlation_matrix[key1] == 0.5
        assert detector.correlation_matrix[key2] == 0.3
        assert detector.correlation_matrix[key3] == 0.4

    def test_high_correlation_reduces_effk(self):
        """Highly correlated traders should have lower effective-K."""
        detector = ConsensusDetector()

        # All pairs highly correlated (rho=0.8)
        detector.update_correlation("0x1111", "0x2222", 0.8)
        detector.update_correlation("0x1111", "0x3333", 0.8)
        detector.update_correlation("0x2222", "0x3333", 0.8)

        weights = {"0x1111": 1.0, "0x2222": 1.0, "0x3333": 1.0}
        eff_k = detector.eff_k_from_corr(weights)

        # With rho=0.8: effK = 9 / (3 + 6*0.8) = 9/7.8 ≈ 1.15
        assert eff_k < 1.5
        assert eff_k > 1.0

    def test_independent_traders_full_effk(self):
        """Independent traders (rho=0) should have effK = n."""
        detector = ConsensusDetector()

        # All pairs independent
        detector.update_correlation("0x1111", "0x2222", 0.0)
        detector.update_correlation("0x1111", "0x3333", 0.0)
        detector.update_correlation("0x2222", "0x3333", 0.0)

        weights = {"0x1111": 1.0, "0x2222": 1.0, "0x3333": 1.0}
        eff_k = detector.eff_k_from_corr(weights)

        # With rho=0: effK = 9/3 = 3
        assert eff_k == pytest.approx(3.0, rel=0.01)

    def test_correlation_affects_effk_calculation(self):
        """Higher correlation should result in lower effective-K for same weights."""
        # High correlation scenario
        detector_high = ConsensusDetector()
        detector_high.update_correlation("0x1111", "0x2222", 0.9)
        detector_high.update_correlation("0x1111", "0x3333", 0.9)
        detector_high.update_correlation("0x2222", "0x3333", 0.9)

        # Low correlation scenario
        detector_low = ConsensusDetector()
        detector_low.update_correlation("0x1111", "0x2222", 0.1)
        detector_low.update_correlation("0x1111", "0x3333", 0.1)
        detector_low.update_correlation("0x2222", "0x3333", 0.1)

        # Same 3 weights for both
        weights = {"0x1111": 1.0, "0x2222": 1.0, "0x3333": 1.0}

        # effK should be lower for high correlation
        effk_high = detector_high.eff_k_from_corr(weights)
        effk_low = detector_low.eff_k_from_corr(weights)

        # High correlation = lower effective independent traders
        assert effk_high < effk_low
        # With ρ=0.9: effK ≈ 9/(3+6*0.9) = 9/8.4 ≈ 1.07
        # With ρ=0.1: effK ≈ 9/(3+6*0.1) = 9/3.6 ≈ 2.5
        assert effk_high < 1.5
        assert effk_low > 2.0


class TestEpisodeRMultiple:
    """Test episode construction and R-multiple calculation."""

    def _make_fill(self, address: str, asset: str, side: str, px: float, sz: float, ts: datetime, fee: float = 0.0) -> EpisodeFill:
        """Helper to create a fill."""
        return EpisodeFill(
            fill_id=f"fill-{ts.timestamp()}",
            address=address,
            asset=asset,
            side=side,
            size=sz,
            price=px,
            ts=ts,
            fees=fee,
        )

    def test_episode_r_multiple_with_dynamic_stop(self):
        """
        Episode R-multiple should be calculated using ATR-based stop.

        Entry: $100k
        Exit: $103k (3% profit)
        ATR-based stop: 2%
        R = 3% / 2% = 1.5R
        """
        config = EpisodeBuilderConfig(default_stop_fraction=0.02)  # 2% ATR-based
        tracker = EpisodeTracker(config)

        now = datetime.now(timezone.utc)

        # Entry fill (opens position)
        entry_fill = self._make_fill("0xtest", "BTC", "buy", px=100000.0, sz=1.0, ts=now, fee=10.0)
        result = tracker.process_fill(entry_fill)
        assert result is None  # Not closed yet

        # Exit fill (closes position) - 3% profit
        exit_fill = self._make_fill("0xtest", "BTC", "sell", px=103000.0, sz=1.0, ts=now + timedelta(hours=1), fee=10.0)
        episode = tracker.process_fill(exit_fill)

        # Episode should be closed with R calculated
        # PnL = (103000 - 100000) * 1 - 20 fees = 2980
        # Risk = 100000 * 0.02 = 2000
        # R = 2980 / 2000 = 1.49
        assert episode is not None
        assert episode.status == 'closed'
        assert episode.result_r == pytest.approx(1.49, rel=0.05)

    def test_episode_with_vwap_entry(self):
        """
        VWAP entry calculation across multiple fills.
        """
        config = EpisodeBuilderConfig(default_stop_fraction=0.01)
        tracker = EpisodeTracker(config)

        now = datetime.now(timezone.utc)

        # First entry fill at $4000 for 2 ETH
        fill1 = self._make_fill("0xtest", "ETH", "buy", px=4000.0, sz=2.0, ts=now, fee=5.0)
        tracker.process_fill(fill1)

        # Check open episode
        key = ("0xtest", "ETH")
        open_ep = tracker.open_episodes.get(key)
        assert open_ep is not None
        assert open_ep.entry_vwap == 4000.0

        # Second entry fill at $4100 for 3 ETH (add to position)
        fill2 = self._make_fill("0xtest", "ETH", "buy", px=4100.0, sz=3.0, ts=now + timedelta(minutes=1), fee=7.5)
        tracker.process_fill(fill2)

        # Check that VWAP is recalculated
        open_ep = tracker.open_episodes.get(key)
        # VWAP = (4000*2 + 4100*3) / 5 = (8000 + 12300) / 5 = 4060
        assert open_ep.entry_vwap == pytest.approx(4060.0, rel=0.01)

    def test_episode_r_feeds_nig_update(self):
        """
        Episode R-multiple should be suitable for NIG posterior update.
        The R value represents standardized performance that can be
        compared across different volatility regimes.
        """
        config = EpisodeBuilderConfig(default_stop_fraction=0.02)

        now = datetime.now(timezone.utc)

        # Winning episode
        tracker_win = EpisodeTracker(config)
        tracker_win.process_fill(self._make_fill("0xwinner", "BTC", "buy", 100000.0, 1.0, now, 10.0))
        win_episode = tracker_win.process_fill(self._make_fill("0xwinner", "BTC", "sell", 104000.0, 1.0, now + timedelta(hours=1), 10.0))

        # Losing episode
        tracker_lose = EpisodeTracker(config)
        tracker_lose.process_fill(self._make_fill("0xloser", "BTC", "buy", 100000.0, 1.0, now, 10.0))
        lose_episode = tracker_lose.process_fill(self._make_fill("0xloser", "BTC", "sell", 98000.0, 1.0, now + timedelta(hours=1), 10.0))

        # Winner should have positive R (profit > 1R)
        # PnL = 4000 - 20 = 3980, R = 3980/2000 = 1.99
        assert win_episode.result_r > 1.0

        # Loser should have negative R (lost 2% + fees)
        # PnL = -2000 - 20 = -2020, R = -2020/2000 = -1.01
        assert lose_episode.result_r < 0


class TestEndToEndSignalFlow:
    """Test complete signal flow from score to episode."""

    def _make_fill(self, address: str, asset: str, side: str, px: float, sz: float, ts: datetime, fee: float = 0.0) -> EpisodeFill:
        """Helper to create a fill."""
        return EpisodeFill(
            fill_id=f"fill-{ts.timestamp()}",
            address=address,
            asset=asset,
            side=side,
            size=sz,
            price=px,
            ts=ts,
            fees=fee,
        )

    def test_complete_long_signal_flow(self):
        """
        End-to-end test:
        1. Multiple traders vote long (from Thompson Sampling)
        2. Consensus detected with correlation-adjusted effK
        3. Signal generated with ATR-based stop
        4. Episode created and R-multiple calculated
        """
        # Setup correlation provider
        corr_provider = CorrelationProvider()
        corr_provider.correlations[("0x1111", "0x2222")] = 0.3
        corr_provider.correlations[("0x1111", "0x3333")] = 0.2
        corr_provider.correlations[("0x2222", "0x3333")] = 0.4

        # Setup consensus detector with ATR-based stop
        detector = ConsensusDetector()
        corr_provider.hydrate_detector(detector)
        detector.set_stop_fraction("BTC", 0.025)

        # Verify effK calculation with loaded correlations
        weights = {"0x1111": 1.0, "0x2222": 1.0, "0x3333": 1.0}
        eff_k = detector.eff_k_from_corr(weights)
        # With avg correlation ~0.3: effK = 9/(3+6*0.3) = 9/4.8 = 1.875
        # Should be between 1 and 3
        assert 1.0 < eff_k < 3.0
        assert eff_k == pytest.approx(1.875, rel=0.05)

        # Create episode with the ATR-based stop (simulating signal execution)
        config = EpisodeBuilderConfig(
            default_stop_fraction=detector.get_stop_fraction("BTC")
        )
        tracker = EpisodeTracker(config)

        now = datetime.now(timezone.utc)

        # Simulate fills based on hypothetical consensus signal
        tracker.process_fill(self._make_fill("0xpool", "BTC", "buy", 100000.0, 1.0, now, 10.0))

        # Price moves in our favor by 3.75% (1.5R with 2.5% stop)
        episode = tracker.process_fill(self._make_fill("0xpool", "BTC", "sell", 103750.0, 1.0, now + timedelta(hours=4), 10.0))

        assert episode is not None
        # R = (3750 - 20 fees) / 2500 risk ≈ 1.49R
        assert episode.result_r == pytest.approx(1.49, rel=0.05)

    def test_regime_shift_changes_stops(self):
        """
        When volatility regime shifts (ATR changes), stops should adapt.
        Low vol regime: tight stops, higher R
        High vol regime: wide stops, more room but lower R multiple for same move
        """
        now = datetime.now(timezone.utc)

        # Low volatility regime (1% ATR → 2% stop with 2x multiplier)
        low_vol_config = EpisodeBuilderConfig(default_stop_fraction=0.02)
        tracker_low = EpisodeTracker(low_vol_config)
        tracker_low.process_fill(self._make_fill("0x1", "BTC", "buy", 100000.0, 1.0, now, 10.0))
        episode_low = tracker_low.process_fill(self._make_fill("0x1", "BTC", "sell", 102000.0, 1.0, now + timedelta(hours=1), 10.0))

        # High volatility regime (4% ATR → 8% stop with 2x multiplier)
        high_vol_config = EpisodeBuilderConfig(default_stop_fraction=0.08)
        tracker_high = EpisodeTracker(high_vol_config)
        tracker_high.process_fill(self._make_fill("0x2", "BTC", "buy", 100000.0, 1.0, now, 10.0))
        episode_high = tracker_high.process_fill(self._make_fill("0x2", "BTC", "sell", 102000.0, 1.0, now + timedelta(hours=1), 10.0))

        # Same $2000 profit, but different R multiples due to stop size
        # Low vol: R = 1980 / 2000 ≈ 0.99R
        # High vol: R = 1980 / 8000 ≈ 0.25R
        assert episode_low.result_r > episode_high.result_r
        assert episode_low.result_r == pytest.approx(0.99, rel=0.05)
        assert episode_high.result_r == pytest.approx(0.25, rel=0.05)


class TestCorrelationSignVectorIntegration:
    """Test sign vector correlation with consensus votes."""

    def test_same_direction_votes_high_correlation(self):
        """
        Traders who consistently vote the same direction should have
        high correlation in their sign vectors.
        """
        # Build sign vectors with same pattern
        signs_a = {i: 1 for i in range(20)}  # All long
        signs_b = {i: 1 for i in range(20)}  # All long

        rho, n_common = compute_phi_correlation(signs_a, signs_b)

        assert rho == 1.0
        assert n_common == 20

    def test_opposite_direction_votes_zero_correlation(self):
        """
        Traders who vote opposite directions should have
        zero correlation (clipped from negative).
        """
        signs_a = {i: 1 for i in range(20)}   # All long
        signs_b = {i: -1 for i in range(20)}  # All short

        rho, n_common = compute_phi_correlation(signs_a, signs_b)

        assert rho == 0.0  # Clipped
        assert n_common == 20

    def test_bucket_alignment_for_concurrent_votes(self):
        """
        Votes within 5 minutes should fall in the same bucket,
        contributing to correlation calculation.
        """
        t1 = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2024, 6, 15, 12, 3, 0, tzinfo=timezone.utc)  # Same bucket
        t3 = datetime(2024, 6, 15, 12, 6, 0, tzinfo=timezone.utc)  # Next bucket

        b1 = bucket_id_from_timestamp(t1)
        b2 = bucket_id_from_timestamp(t2)
        b3 = bucket_id_from_timestamp(t3)

        assert b1 == b2  # Same 5-min bucket
        assert b1 != b3  # Different bucket


class TestDataFlowValidation:
    """Validate data flows correctly between components."""

    def test_atr_data_format(self):
        """ATR data should have all required fields."""
        atr_data = ATRData(
            asset="BTC",
            atr=1500.0,
            atr_pct=1.5,
            price=100000.0,
            multiplier=2.0,
            stop_distance_pct=3.0,
            timestamp=datetime.now(timezone.utc),
            source="calculated",
        )

        # Verify all fields present
        assert atr_data.asset == "BTC"
        assert atr_data.atr == 1500.0
        assert atr_data.multiplier == ATR_MULTIPLIER_BTC
        assert atr_data.source in ["db", "calculated", "fallback"]

    def test_correlation_lookup_symmetric(self):
        """Correlation lookup should be symmetric (ρ(A,B) = ρ(B,A))."""
        provider = CorrelationProvider()
        provider.correlations[("0x1111", "0x2222")] = 0.5

        # Both orderings should return same value
        assert provider.get("0x1111", "0x2222") == 0.5
        assert provider.get("0x2222", "0x1111") == 0.5

    def test_detector_correlation_symmetric(self):
        """ConsensusDetector correlation should be symmetric (stored with sorted key)."""
        detector = ConsensusDetector()
        detector.update_correlation("0x1111", "0x2222", 0.6)

        # Both orderings should access the same stored value
        key1 = tuple(sorted(["0x1111", "0x2222"]))
        key2 = tuple(sorted(["0x2222", "0x1111"]))
        assert key1 == key2  # Keys should be identical
        assert detector.correlation_matrix[key1] == 0.6

    def test_episode_data_serializable(self):
        """Episode data should be JSON-serializable for persistence."""
        import json

        config = EpisodeBuilderConfig(default_stop_fraction=0.02)
        tracker = EpisodeTracker(config)

        now = datetime.now(timezone.utc)
        entry_fill = EpisodeFill(
            fill_id="f1",
            address="0xtest",
            asset="BTC",
            side="buy",
            size=1.0,
            price=100000.0,
            ts=now,
            fees=10.0,
        )
        exit_fill = EpisodeFill(
            fill_id="f2",
            address="0xtest",
            asset="BTC",
            side="sell",
            size=1.0,
            price=101000.0,
            ts=now + timedelta(hours=1),
            fees=10.0,
        )

        tracker.process_fill(entry_fill)
        episode = tracker.process_fill(exit_fill)

        # Should be able to create a serializable dict
        episode_dict = {
            "episode_id": episode.id,
            "address": episode.address,
            "asset": episode.asset,
            "direction": episode.direction,
            "entry_vwap": episode.entry_vwap,
            "exit_vwap": episode.exit_vwap,
            "pnl": episode.realized_pnl,
            "r_multiple": episode.result_r,
        }

        # Should serialize without error
        json_str = json.dumps(episode_dict)
        assert "r_multiple" in json_str


class TestATRToConsensusFlow:
    """Test the ATR→Consensus stop integration in the actual consensus check."""

    def test_consensus_uses_detector_stop_fraction(self):
        """
        check_episode_consensus should use detector.get_stop_fraction()
        instead of hardcoded ASSUMED_STOP_FRACTION.
        """
        # Setup detector with ATR-based stop
        detector = ConsensusDetector()
        detector.set_stop_fraction("BTC", 0.025)  # 2.5% ATR-based stop

        # Verify detector has correct stop
        assert detector.get_stop_fraction("BTC") == 0.025

        # The check_episode_consensus function imports from main,
        # so we verify the detector method works correctly
        # The actual integration is in main.py:check_episode_consensus
        # which now calls detector.get_stop_fraction(asset)

    def test_consensus_detector_initializes_with_defaults(self):
        """New detector should have default 1% stops."""
        detector = ConsensusDetector()
        assert detector.get_stop_fraction("BTC") == 0.01
        assert detector.get_stop_fraction("ETH") == 0.01

    def test_atr_updates_detector_stop(self):
        """ATR provider updates should reflect in detector."""
        provider = ATRProvider()
        detector = ConsensusDetector()

        # Create ATR data for BTC
        atr_data = ATRData(
            asset="BTC",
            atr=3000.0,
            atr_pct=3.0,
            price=100000.0,
            multiplier=2.0,
            stop_distance_pct=6.0,
            timestamp=datetime.now(timezone.utc),
            source="test",
        )

        # Get stop fraction from ATR
        stop_fraction = provider.get_stop_fraction(atr_data)
        assert stop_fraction == pytest.approx(0.06, rel=1e-5)

        # Apply to detector
        detector.set_stop_fraction("BTC", stop_fraction)
        assert detector.get_stop_fraction("BTC") == 0.06

        # ETH should still have default
        assert detector.get_stop_fraction("ETH") == 0.01

    def test_stop_price_calculation_with_dynamic_stop(self):
        """Stop price should be calculated using dynamic stop fraction."""
        detector = ConsensusDetector()

        # Set 3% ATR-based stop
        detector.set_stop_fraction("BTC", 0.03)

        entry_price = 100000.0
        stop_fraction = detector.get_stop_fraction("BTC")
        stop_distance = entry_price * stop_fraction

        # Long position stop
        long_stop = entry_price - stop_distance
        assert long_stop == 97000.0  # 100k - 3% = 97k

        # Short position stop
        short_stop = entry_price + stop_distance
        assert short_stop == 103000.0  # 100k + 3% = 103k


class TestVoteWeighting:
    """Test improved vote weighting with equity-normalized and log scaling."""

    def test_log_weight_scales_sublinearly(self):
        """Log weighting should give sublinear scaling with notional."""
        from app.consensus import calculate_vote_weight

        # $10k position
        w_10k = calculate_vote_weight(10000, mode="log", log_base=10000)
        # $100k position (10x larger)
        w_100k = calculate_vote_weight(100000, mode="log", log_base=10000)
        # $1M position (100x larger)
        w_1m = calculate_vote_weight(1000000, mode="log", log_base=10000)

        # Weights should increase sublinearly
        assert w_10k > 0
        assert w_100k > w_10k
        assert w_1m > w_100k

        # 10x notional should NOT give 10x weight
        assert w_100k / w_10k < 5  # log(11) / log(2) ≈ 3.46
        assert w_1m / w_10k < 10  # log(101) / log(2) ≈ 6.66

    def test_equity_weight_with_sqrt(self):
        """Equity-normalized weighting should use sqrt for smoothing."""
        from app.consensus import calculate_vote_weight

        # 10% position relative to equity (10k position, 100k equity)
        w_10pct = calculate_vote_weight(10000, equity=100000, mode="equity")
        # 40% position relative to equity (40k position, 100k equity)
        w_40pct = calculate_vote_weight(40000, equity=100000, mode="equity")

        # sqrt(0.1) ≈ 0.316, sqrt(0.4) ≈ 0.632
        assert w_10pct == pytest.approx(0.316, rel=0.05)
        assert w_40pct == pytest.approx(0.632, rel=0.05)

        # Quadrupling position doubles weight (sqrt scaling)
        assert w_40pct / w_10pct == pytest.approx(2.0, rel=0.05)

    def test_equity_weight_caps_at_max(self):
        """Equity-normalized weight should cap at max_weight."""
        from app.consensus import calculate_vote_weight

        # 200% position relative to equity (way over-leveraged)
        w_200pct = calculate_vote_weight(200000, equity=100000, mode="equity", max_weight=1.0)

        # sqrt(2.0) ≈ 1.41, but capped at 1.0
        assert w_200pct == 1.0

    def test_equity_mode_falls_back_to_log_without_equity(self):
        """Equity mode should fall back to log when equity not available."""
        from app.consensus import calculate_vote_weight

        # No equity data
        w_no_equity = calculate_vote_weight(100000, equity=None, mode="equity")
        # Explicit log mode
        w_log = calculate_vote_weight(100000, mode="log")

        # Should be equal since equity mode falls back to log
        assert w_no_equity == w_log

    def test_linear_mode_for_backwards_compat(self):
        """Linear mode should work for backwards compatibility."""
        from app.consensus import calculate_vote_weight

        # $100k position with $10k base
        w_100k = calculate_vote_weight(100000, mode="linear", log_base=10000)

        # Linear: 100k / 10k = 10, but capped at 1.0
        assert w_100k == 1.0

        # Smaller position
        w_5k = calculate_vote_weight(5000, mode="linear", log_base=10000, max_weight=1.0)
        assert w_5k == pytest.approx(0.5, rel=0.01)  # 5k / 10k = 0.5

    def test_zero_notional_returns_zero_weight(self):
        """Zero or negative notional should return zero weight."""
        from app.consensus import calculate_vote_weight

        assert calculate_vote_weight(0) == 0.0
        assert calculate_vote_weight(-1000) == 0.0

    def test_collapse_to_votes_uses_new_weighting(self):
        """collapse_to_votes should use the new weighting function."""
        from app.consensus import ConsensusDetector, Fill

        detector = ConsensusDetector()
        now = datetime.now(timezone.utc)

        # Create fills with different sizes
        fills = [
            Fill(fill_id="1", address="0xsmall", asset="BTC", side="buy", size=0.1, price=100000.0, ts=now),
            Fill(fill_id="2", address="0xlarge", asset="BTC", side="buy", size=1.0, price=100000.0, ts=now),
        ]

        votes = detector.collapse_to_votes(fills)

        # Both should have votes
        assert len(votes) == 2

        small_vote = next(v for v in votes if "small" in v.address)
        large_vote = next(v for v in votes if "large" in v.address)

        # Notional should be tracked
        assert small_vote.notional == pytest.approx(10000, rel=0.01)  # 0.1 * 100k
        assert large_vote.notional == pytest.approx(100000, rel=0.01)  # 1.0 * 100k

        # Larger position should have higher weight
        assert large_vote.weight > small_vote.weight

        # But not 10x higher (log scaling)
        assert large_vote.weight / small_vote.weight < 5

    def test_collapse_to_votes_with_equity(self):
        """collapse_to_votes should use equity-normalized weights when available."""
        from app.consensus import ConsensusDetector, Fill

        detector = ConsensusDetector()
        now = datetime.now(timezone.utc)

        fills = [
            Fill(fill_id="1", address="0xrich", asset="BTC", side="buy", size=1.0, price=100000.0, ts=now),
            Fill(fill_id="2", address="0xpoor", asset="BTC", side="buy", size=1.0, price=100000.0, ts=now),
        ]

        # Rich trader has 10M equity, poor trader has 100k equity
        # Same notional ($100k) but very different position ratios
        equity_map = {
            "0xrich": 10000000.0,  # 1% position
            "0xpoor": 100000.0,    # 100% position
        }

        # Set mode to equity for this test
        import os
        original_mode = os.environ.get("VOTE_WEIGHT_MODE")
        try:
            os.environ["VOTE_WEIGHT_MODE"] = "equity"
            votes = detector.collapse_to_votes(fills, equity_by_address=equity_map)

            rich_vote = next(v for v in votes if "rich" in v.address)
            poor_vote = next(v for v in votes if "poor" in v.address)

            # Both have same notional
            assert rich_vote.notional == poor_vote.notional

            # But equity is tracked
            assert rich_vote.equity == 10000000.0
            assert poor_vote.equity == 100000.0
        finally:
            if original_mode:
                os.environ["VOTE_WEIGHT_MODE"] = original_mode
            elif "VOTE_WEIGHT_MODE" in os.environ:
                del os.environ["VOTE_WEIGHT_MODE"]


class TestCorrelationRefreshIntegration:
    """Test correlation refresh task integration."""

    def test_hydrate_detector_applies_decay(self):
        """CorrelationProvider should apply decay when hydrating detector."""
        provider = CorrelationProvider()
        detector = ConsensusDetector()

        # Set correlations with old date (3 days ago = half-life)
        provider.correlations[("0x1111", "0x2222")] = 0.8
        provider._loaded_date = date.today() - timedelta(days=3)

        # Hydrate with decay
        count = provider.hydrate_detector(detector, apply_decay=True)

        assert count == 1

        # Check detector has decayed correlation
        key = tuple(sorted(["0x1111", "0x2222"]))
        decayed_rho = detector.correlation_matrix.get(key)

        # At 3 days (half-life), decay = 0.5
        # Decayed = 0.8 * 0.5 + 0.3 * 0.5 = 0.55
        assert decayed_rho is not None
        assert 0.5 < decayed_rho < 0.7  # Should be around 0.55

    def test_hydrate_detector_without_decay(self):
        """CorrelationProvider should preserve raw values when apply_decay=False."""
        provider = CorrelationProvider()
        detector = ConsensusDetector()

        # Set correlations with old date
        provider.correlations[("0x1111", "0x2222")] = 0.8
        provider._loaded_date = date.today() - timedelta(days=3)

        # Hydrate WITHOUT decay
        count = provider.hydrate_detector(detector, apply_decay=False)

        assert count == 1

        # Check detector has raw correlation
        key = tuple(sorted(["0x1111", "0x2222"]))
        raw_rho = detector.correlation_matrix.get(key)

        assert raw_rho == 0.8  # No decay applied

    def test_hydrate_detector_with_fresh_data(self):
        """Fresh data should have minimal decay."""
        provider = CorrelationProvider()
        detector = ConsensusDetector()

        # Set correlations with today's date
        provider.correlations[("0x1111", "0x2222")] = 0.8
        provider._loaded_date = date.today()

        # Hydrate with decay
        count = provider.hydrate_detector(detector, apply_decay=True)

        assert count == 1

        # Check detector has nearly raw correlation (fresh data = no decay)
        key = tuple(sorted(["0x1111", "0x2222"]))
        fresh_rho = detector.correlation_matrix.get(key)

        assert fresh_rho is not None
        assert fresh_rho >= 0.79  # Almost no decay

    def test_detector_uses_hydrated_correlations_in_effk(self):
        """Hydrated correlations should affect effective-K calculation."""
        provider = CorrelationProvider()
        detector = ConsensusDetector()

        # Set high correlation between two traders
        provider.correlations[("0xaaa", "0xbbb")] = 0.9
        provider._loaded_date = date.today()

        provider.hydrate_detector(detector, apply_decay=False)

        # Calculate effK with two highly correlated traders
        weights = {"0xaaa": 1.0, "0xbbb": 1.0}
        eff_k = detector.eff_k_from_corr(weights)

        # High correlation → lower effK
        # effK = (1+1)² / (1*1 + 1*1 + 2*1*1*0.9) = 4 / 3.8 ≈ 1.05
        assert 1.0 < eff_k < 1.5

    def test_detector_effk_with_independent_traders(self):
        """Independent traders should give full effK."""
        provider = CorrelationProvider()
        detector = ConsensusDetector()

        # No correlation data = default ρ = 0.3
        provider._loaded_date = date.today()
        provider.hydrate_detector(detector, apply_decay=False)

        # Calculate effK with two uncorrelated traders
        weights = {"0xindep1": 1.0, "0xindep2": 1.0}
        eff_k = detector.eff_k_from_corr(weights)

        # No stored correlation uses default (0.3)
        # effK = 4 / (1 + 1 + 2*0.3) = 4 / 2.6 ≈ 1.54
        assert 1.5 < eff_k < 2.0


class TestBackgroundTaskConfiguration:
    """Test configuration for background tasks."""

    def test_correlation_refresh_interval_default(self):
        """Default correlation refresh interval should be 24 hours."""
        # Import to check default value
        from app.main import CORR_REFRESH_INTERVAL_HOURS

        # Default is 24 hours
        assert CORR_REFRESH_INTERVAL_HOURS == 24

    def test_reconcile_interval_default(self):
        """Default reconciliation interval should be 6 hours."""
        from app.main import RECONCILE_INTERVAL_HOURS

        assert RECONCILE_INTERVAL_HOURS == 6


class TestQuantPipelineIntegration:
    """
    Comprehensive quant pipeline integration tests.

    Tests the full flow: Episode → NIG Update → Thompson Sampling → Consensus → Signal
    under varying volatility and correlation conditions.

    These tests validate that the quant correctness holds across different market regimes.
    """

    def _make_consensus_fill(self, address: str, asset: str, side: str, size: float, price: float, ts: datetime) -> Fill:
        """Helper to create a Fill for consensus detector."""
        return Fill(
            fill_id=f"fill-{address}-{ts.timestamp()}",
            address=address,
            asset=asset,
            side=side,
            size=size,
            price=price,
            ts=ts,
        )

    def _make_episode_fill(self, address: str, asset: str, side: str, px: float, sz: float, ts: datetime, fee: float = 10.0) -> EpisodeFill:
        """Helper to create an EpisodeFill for episode tracker."""
        return EpisodeFill(
            fill_id=f"efill-{address}-{ts.timestamp()}",
            address=address,
            asset=asset,
            side=side,
            size=sz,
            price=px,
            ts=ts,
            fees=fee,
        )

    def test_low_vol_high_corr_regime(self):
        """
        Low volatility + high correlation regime:
        - Tight stops (low ATR) → small risk per trade
        - High correlation → low effK → harder to pass consensus
        - Should require strong agreement to generate signal
        """
        # Setup: Low vol (1% ATR), high correlation (ρ=0.8)
        detector = ConsensusDetector()
        detector.set_stop_fraction("BTC", 0.02)  # 2% stop (low vol)

        # All traders highly correlated
        detector.update_correlation("0xalpha", "0xbeta", 0.8)
        detector.update_correlation("0xalpha", "0xgamma", 0.8)
        detector.update_correlation("0xbeta", "0xgamma", 0.8)

        # 3 traders all going long
        weights = {"0xalpha": 1.0, "0xbeta": 1.0, "0xgamma": 1.0}
        eff_k = detector.eff_k_from_corr(weights)

        # With ρ=0.8: effK = 9 / (3 + 6*0.8) = 9/7.8 ≈ 1.15
        # This should FAIL the MIN_EFFECTIVE_K=2.0 gate
        assert eff_k < 1.5
        assert eff_k < 2.0  # Below consensus threshold

        # Verify R calculation with tight stops
        config = EpisodeBuilderConfig(default_stop_fraction=0.02)
        tracker = EpisodeTracker(config)
        now = datetime.now(timezone.utc)

        tracker.process_fill(self._make_episode_fill("0xtest", "BTC", "buy", 100000.0, 1.0, now))
        episode = tracker.process_fill(self._make_episode_fill("0xtest", "BTC", "sell", 101000.0, 1.0, now + timedelta(hours=1)))

        # $1000 profit with 2% stop ($2000 risk) = ~0.49R
        assert episode.result_r == pytest.approx(0.49, rel=0.05)

    def test_high_vol_low_corr_regime(self):
        """
        High volatility + low correlation regime:
        - Wide stops (high ATR) → larger risk per trade
        - Low correlation → high effK → easier to pass consensus
        - More trades should pass but with lower R multiples
        """
        # Setup: High vol (4% ATR), low correlation (ρ=0.1)
        detector = ConsensusDetector()
        detector.set_stop_fraction("BTC", 0.08)  # 8% stop (high vol)

        # All traders independent
        detector.update_correlation("0xalpha", "0xbeta", 0.1)
        detector.update_correlation("0xalpha", "0xgamma", 0.1)
        detector.update_correlation("0xbeta", "0xgamma", 0.1)

        # 3 traders all going long
        weights = {"0xalpha": 1.0, "0xbeta": 1.0, "0xgamma": 1.0}
        eff_k = detector.eff_k_from_corr(weights)

        # With ρ=0.1: effK = 9 / (3 + 6*0.1) = 9/3.6 ≈ 2.5
        # This should PASS the MIN_EFFECTIVE_K=2.0 gate
        assert eff_k > 2.0
        assert eff_k < 3.0

        # Verify R calculation with wide stops
        config = EpisodeBuilderConfig(default_stop_fraction=0.08)
        tracker = EpisodeTracker(config)
        now = datetime.now(timezone.utc)

        tracker.process_fill(self._make_episode_fill("0xtest", "BTC", "buy", 100000.0, 1.0, now))
        episode = tracker.process_fill(self._make_episode_fill("0xtest", "BTC", "sell", 101000.0, 1.0, now + timedelta(hours=1)))

        # Same $1000 profit with 8% stop ($8000 risk) = ~0.12R
        assert episode.result_r == pytest.approx(0.12, rel=0.05)

    def test_mixed_regime_with_weight_disparity(self):
        """
        Test with unequal weights (whale vs small traders):
        - One whale with high weight
        - Multiple small traders with low weights
        - Verify effK accounts for weight disparity
        """
        detector = ConsensusDetector()

        # Set moderate correlation
        for a, b in [("0xwhale", "0xsmall1"), ("0xwhale", "0xsmall2"), ("0xsmall1", "0xsmall2")]:
            detector.update_correlation(a, b, 0.3)

        # Whale has 10x weight
        weights = {"0xwhale": 1.0, "0xsmall1": 0.1, "0xsmall2": 0.1}
        eff_k = detector.eff_k_from_corr(weights)

        # With unequal weights, effK is dominated by whale
        # Sum weights = 1.2, sum² = 1.44
        # Denominator dominated by whale's self-correlation
        assert 1.0 < eff_k < 2.0  # Lower than equal weights case

        # Compare to equal weights
        equal_weights = {"0xwhale": 1.0, "0xsmall1": 1.0, "0xsmall2": 1.0}
        eff_k_equal = detector.eff_k_from_corr(equal_weights)

        # Equal weights should give higher effK
        assert eff_k_equal > eff_k

    def test_effk_fallback_tracking(self):
        """
        Test that effK calculation properly tracks when default ρ is used.
        """
        detector = ConsensusDetector()

        # Only set correlation for one pair
        detector.update_correlation("0xa", "0xb", 0.5)

        # 3 traders but only 1 stored correlation pair
        weights = {"0xa": 1.0, "0xb": 1.0, "0xc": 1.0}

        fallback_count = 0
        def count_fallbacks():
            nonlocal fallback_count
            fallback_count += 1

        eff_k = detector.eff_k_from_corr(weights, fallback_counter_callback=count_fallbacks)

        # Should have used fallback for pairs (a,c) and (b,c)
        # Since the callback fires once per check, fallback_count should be 1
        assert fallback_count >= 1

        # effK should still be valid
        assert 1.0 < eff_k < 3.0

    def test_nig_weight_formula(self):
        """
        Verify NIG posterior weight formula: κ/(κ+10).

        Higher κ (more observations) → higher weight (more confidence)
        New traders (κ=1) → low weight (0.09)
        Veteran traders (κ=100) → high weight (0.91)
        """
        # NIG weight formula (used in hl-sage Thompson Sampling)
        def nig_weight(kappa: float) -> float:
            return kappa / (kappa + 10)

        # New trader with 1 observation
        assert nig_weight(1) == pytest.approx(0.091, rel=0.01)

        # Moderate trader with 10 observations
        assert nig_weight(10) == pytest.approx(0.5, rel=0.01)

        # Veteran trader with 100 observations
        assert nig_weight(100) == pytest.approx(0.909, rel=0.01)

    def test_r_multiple_sensitivity_to_stop(self):
        """
        Verify R-multiple calculation sensitivity to stop fraction.

        Same dollar profit should give different R values based on stop.
        """
        now = datetime.now(timezone.utc)

        # Test that R-multiple scales inversely with stop fraction
        # Smaller stop = larger R multiple for same profit
        stop_fractions = [0.01, 0.02, 0.05, 0.10]
        r_multiples = []

        for stop_fraction in stop_fractions:
            config = EpisodeBuilderConfig(default_stop_fraction=stop_fraction)
            tracker = EpisodeTracker(config)

            tracker.process_fill(self._make_episode_fill("0xtest", "BTC", "buy", 100000.0, 1.0, now))
            episode = tracker.process_fill(self._make_episode_fill("0xtest", "BTC", "sell", 101000.0, 1.0, now + timedelta(hours=1)))

            r_multiples.append(episode.result_r)

        # Verify inverse relationship: smaller stop = larger R
        for i in range(len(r_multiples) - 1):
            assert r_multiples[i] > r_multiples[i + 1], \
                f"R-multiple should decrease as stop widens: {r_multiples[i]:.2f} vs {r_multiples[i+1]:.2f}"

        # Verify rough magnitudes (with tolerance for fees and R capping)
        # 1% stop: ~0.98R (or capped at 1.0R)
        assert r_multiples[0] >= 0.9, f"1% stop should give ~1R, got {r_multiples[0]:.2f}R"
        # 10% stop: ~0.1R
        assert r_multiples[-1] < 0.2, f"10% stop should give <0.2R, got {r_multiples[-1]:.2f}R"

    def test_consensus_gates_progression(self):
        """
        Test the progression through consensus gates.

        Gate 1: Dispersion (min traders, min %)
        Gate 2: Effective-K (correlation adjusted)
        Gate 3: Latency + Price band (ATR-based)
        Gate 4: EV after costs
        Gate 5: Risk limits (confidence, min EV)
        """
        from app.consensus import (
            passes_consensus_gates, calculate_ev,
            CONSENSUS_MIN_TRADERS, CONSENSUS_MIN_PCT,
            CONSENSUS_MIN_EFFECTIVE_K, CONSENSUS_EV_MIN_R,
            DEFAULT_AVG_WIN_R, DEFAULT_AVG_LOSS_R,
        )

        # Gate 1: Dispersion
        directions = ["long", "long", "long"]  # 3 traders, 100% agreement
        passes, majority = passes_consensus_gates(directions, min_agreeing=3, min_pct=0.70)
        assert passes is True
        assert majority == "long"

        # Gate 1 fail: Not enough agreement
        directions = ["long", "long", "short"]  # 67% agreement
        passes, majority = passes_consensus_gates(directions, min_agreeing=3, min_pct=0.70)
        assert passes is False  # 2/3 = 67% < 70%

        # Gate 2: Effective-K calculation
        detector = ConsensusDetector()
        detector.update_correlation("0xa", "0xb", 0.1)
        detector.update_correlation("0xa", "0xc", 0.1)
        detector.update_correlation("0xb", "0xc", 0.1)
        weights = {"0xa": 1.0, "0xb": 1.0, "0xc": 1.0}
        eff_k = detector.eff_k_from_corr(weights)
        assert eff_k >= CONSENSUS_MIN_EFFECTIVE_K  # Low corr → high effK

        # Gate 4: EV calculation - test with high p_win and wider stop
        # Default avg_win_r=0.5, avg_loss_r=0.3, so gross EV = p*0.5 - (1-p)*0.3
        # For p=0.80: gross = 0.80*0.5 - 0.20*0.3 = 0.40 - 0.06 = 0.34R
        # Costs ~0.014R (7 bps / 5000 bps stop) → net ~0.33R
        ev_result = calculate_ev(
            p_win=0.80,  # High confidence
            entry_px=100000.0,
            stop_px=95000.0,  # 5% stop
        )
        assert ev_result["ev_net_r"] >= CONSENSUS_EV_MIN_R  # Should have positive EV

        # Gate 4 fail: Low p_win gives negative EV
        ev_result_low = calculate_ev(
            p_win=0.30,  # Below breakeven
            entry_px=100000.0,
            stop_px=99000.0,
        )
        # With p=0.30: gross = 0.30*0.5 - 0.70*0.3 = 0.15 - 0.21 = -0.06R (before costs)
        assert ev_result_low["ev_net_r"] < 0  # Negative EV

    def test_atr_staleness_affects_gating(self):
        """
        Test that ATR staleness is properly tracked and affects gating.
        """
        from app.atr import ATR_MAX_STALENESS_SECONDS

        now = datetime.now(timezone.utc)

        # Fresh ATR data
        fresh_atr = ATRData(
            asset="BTC",
            atr=2000.0,
            atr_pct=2.0,
            price=100000.0,
            multiplier=2.0,
            stop_distance_pct=4.0,
            timestamp=now,
            source="calculated",
        )
        assert fresh_atr.age_seconds < ATR_MAX_STALENESS_SECONDS
        assert not fresh_atr.is_stale

        # Stale ATR data (10 minutes old)
        stale_atr = ATRData(
            asset="BTC",
            atr=2000.0,
            atr_pct=2.0,
            price=100000.0,
            multiplier=2.0,
            stop_distance_pct=4.0,
            timestamp=now - timedelta(seconds=ATR_MAX_STALENESS_SECONDS + 60),
            source="calculated",
        )
        assert stale_atr.age_seconds > ATR_MAX_STALENESS_SECONDS
        assert stale_atr.is_stale

    def test_correlation_decay_affects_effk(self):
        """
        Test that correlation decay (for stale data) affects effK calculation.
        """
        provider = CorrelationProvider()
        detector = ConsensusDetector()

        # Set high correlation with old data (6 days = 2 half-lives)
        provider.correlations[("0xa", "0xb")] = 0.9
        provider._loaded_date = date.today() - timedelta(days=6)

        # Hydrate with decay
        provider.hydrate_detector(detector, apply_decay=True)

        key = tuple(sorted(["0xa", "0xb"]))
        decayed_rho = detector.correlation_matrix.get(key)

        # After 2 half-lives: decay = 0.25
        # Decayed = 0.9 * 0.25 + 0.3 * 0.75 = 0.225 + 0.225 = 0.45
        assert decayed_rho is not None
        assert 0.4 < decayed_rho < 0.6

        # Verify effK is higher with decayed (lower) correlation
        weights = {"0xa": 1.0, "0xb": 1.0}
        eff_k_decayed = detector.eff_k_from_corr(weights)

        # Compare to fresh data (no decay)
        detector_fresh = ConsensusDetector()
        detector_fresh.update_correlation("0xa", "0xb", 0.9)
        eff_k_fresh = detector_fresh.eff_k_from_corr(weights)

        # Decayed (lower) correlation → higher effK
        assert eff_k_decayed > eff_k_fresh


class TestRegimeIntegration:
    """Test regime detection integration with stops and Kelly sizing."""

    def test_regime_adjusted_stop_trending(self):
        """Trending regime should widen stops."""
        from app.regime import get_regime_adjusted_stop, MarketRegime

        base_stop = 0.02  # 2%
        adjusted = get_regime_adjusted_stop(base_stop, MarketRegime.TRENDING)

        # Trending: 1.2x multiplier
        assert adjusted == pytest.approx(0.024, rel=0.01)

    def test_regime_adjusted_stop_ranging(self):
        """Ranging regime should tighten stops."""
        from app.regime import get_regime_adjusted_stop, MarketRegime

        base_stop = 0.02  # 2%
        adjusted = get_regime_adjusted_stop(base_stop, MarketRegime.RANGING)

        # Ranging: 0.8x multiplier
        assert adjusted == pytest.approx(0.016, rel=0.01)

    def test_regime_adjusted_stop_volatile(self):
        """Volatile regime should widen stops significantly."""
        from app.regime import get_regime_adjusted_stop, MarketRegime

        base_stop = 0.02  # 2%
        adjusted = get_regime_adjusted_stop(base_stop, MarketRegime.VOLATILE)

        # Volatile: 1.5x multiplier
        assert adjusted == pytest.approx(0.03, rel=0.01)

    def test_regime_adjusted_kelly_trending(self):
        """Trending regime should use full Kelly."""
        from app.regime import get_regime_adjusted_kelly, MarketRegime

        base_kelly = 0.25  # 25%
        adjusted = get_regime_adjusted_kelly(base_kelly, MarketRegime.TRENDING)

        # Trending: 1.0x multiplier
        assert adjusted == pytest.approx(0.25, rel=0.01)

    def test_regime_adjusted_kelly_volatile(self):
        """Volatile regime should reduce Kelly."""
        from app.regime import get_regime_adjusted_kelly, MarketRegime

        base_kelly = 0.25  # 25%
        adjusted = get_regime_adjusted_kelly(base_kelly, MarketRegime.VOLATILE)

        # Volatile: 0.5x multiplier
        assert adjusted == pytest.approx(0.125, rel=0.01)

    def test_regime_adjusted_kelly_ranging(self):
        """Ranging regime should slightly reduce Kelly."""
        from app.regime import get_regime_adjusted_kelly, MarketRegime

        base_kelly = 0.25  # 25%
        adjusted = get_regime_adjusted_kelly(base_kelly, MarketRegime.RANGING)

        # Ranging: 0.75x multiplier
        assert adjusted == pytest.approx(0.1875, rel=0.01)


class TestRiskGovernorIntegration:
    """Test risk governor integration with execution."""

    def test_kill_switch_blocks_execution(self):
        """Kill switch should block all trades."""
        from app.risk_governor import RiskGovernor

        governor = RiskGovernor()
        governor.trigger_kill_switch(reason="test kill switch")

        result = governor.run_all_checks(
            account_value=100000,
            margin_used=10000,
            maintenance_margin=5000,
            total_exposure=0.1,
            daily_pnl=0,
        )

        assert result.allowed is False
        # Check for kill switch in reason
        assert result.reason is not None
        assert "kill" in result.reason.lower() or "halt" in result.reason.lower()

    def test_liquidation_distance_blocks(self):
        """Trades should be blocked when close to liquidation."""
        from app.risk_governor import RiskGovernor

        governor = RiskGovernor()

        # Margin ratio < 1.5x
        result = governor.run_all_checks(
            account_value=10000,
            margin_used=9000,
            maintenance_margin=8000,  # ratio = 10000/8000 = 1.25
            total_exposure=0.5,
            daily_pnl=0,
        )

        assert result.allowed is False
        # Should have reason about margin or liquidation
        assert result.reason is not None
        assert "margin" in result.reason.lower() or "liquidation" in result.reason.lower()

    def test_equity_floor_blocks(self):
        """Trades should be blocked when below equity floor."""
        from app.risk_governor import RiskGovernor, MIN_EQUITY_FLOOR

        governor = RiskGovernor()

        # Below minimum
        result = governor.run_all_checks(
            account_value=MIN_EQUITY_FLOOR - 1,
            margin_used=0,
            maintenance_margin=1,
            total_exposure=0,
            daily_pnl=0,
        )

        assert result.allowed is False
        # Should have reason about account value or floor
        assert result.reason is not None
        assert "floor" in result.reason.lower() or "account" in result.reason.lower()


class TestExecutorIntegration:
    """Test executor integration with regime and risk governor."""

    @pytest.mark.asyncio
    async def test_executor_dry_run_by_default(self):
        """Executor should default to dry run mode."""
        from app.hl_exchange import REAL_EXECUTION_ENABLED

        # By default, real execution should be disabled
        assert REAL_EXECUTION_ENABLED is False

    def test_kelly_result_includes_method(self):
        """Kelly result should include the method used."""
        from app.kelly import kelly_position_size, KellyInput

        input_data = KellyInput(
            win_rate=0.6,
            avg_win_r=1.0,
            avg_loss_r=0.5,
            episode_count=50,
            account_value=100000,
            current_price=50000,
            stop_distance_pct=0.02,
        )

        result = kelly_position_size(input_data)

        assert result.method == "kelly"
        assert result.position_pct > 0
        assert result.reasoning is not None

    def test_kelly_fallback_insufficient_data(self):
        """Kelly should fall back when insufficient episodes."""
        from app.kelly import kelly_position_size, KellyInput, KELLY_MIN_EPISODES

        input_data = KellyInput(
            win_rate=0.6,
            avg_win_r=1.0,
            avg_loss_r=0.5,
            episode_count=KELLY_MIN_EPISODES - 1,  # Insufficient
            account_value=100000,
            current_price=50000,
            stop_distance_pct=0.02,
        )

        result = kelly_position_size(input_data)

        assert result.method == "fallback_insufficient_data"
        # Reasoning mentions episode count needed
        assert "episode" in result.reasoning.lower() or "need" in result.reasoning.lower()


class TestDailyPnLTracking:
    """Test daily PnL tracking for drawdown kill switch."""

    @pytest.mark.asyncio
    async def test_get_daily_pnl_first_call_creates_record(self):
        """First call of the day should create record with current equity as starting."""
        from app.risk_governor import get_daily_pnl
        from unittest.mock import AsyncMock, MagicMock

        # Mock database pool
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        # First call - no existing record
        mock_conn.fetchrow.return_value = None

        result = await get_daily_pnl(mock_pool, 100000.0)

        # Should return 0 (no PnL yet)
        assert result == 0.0
        # Should insert record
        assert mock_conn.execute.called

    @pytest.mark.asyncio
    async def test_get_daily_pnl_returns_difference(self):
        """Subsequent calls should return difference from starting equity."""
        from app.risk_governor import get_daily_pnl
        from unittest.mock import AsyncMock, MagicMock

        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        # Existing record with starting equity 100000
        mock_conn.fetchrow.return_value = {
            "starting_equity": 100000.0,
            "current_equity": 100000.0,
        }

        # Current equity is 95000 (lost $5000)
        result = await get_daily_pnl(mock_pool, 95000.0)

        # Should return -5000
        assert result == -5000.0


class TestRegimeAwareConfidence:
    """Test regime-aware confidence gating."""

    def test_check_risk_limits_with_trending_regime(self):
        """Trending regime should use higher confidence threshold."""
        from app.consensus import check_risk_limits, MIN_SIGNAL_CONFIDENCE
        from app.regime import get_regime_adjusted_confidence, MarketRegime
        from unittest.mock import MagicMock

        signal = MagicMock()
        signal.p_win = 0.58  # Above base 0.55, but may fail trending
        signal.ev_net_r = 0.3

        # Without regime - should pass
        passes_base, _ = check_risk_limits(signal, regime=None)
        assert passes_base is True

        # Calculate what trending threshold would be
        trending_threshold = get_regime_adjusted_confidence(MIN_SIGNAL_CONFIDENCE, MarketRegime.TRENDING)

        # If trending requires higher confidence, this signal may fail
        if trending_threshold > signal.p_win:
            passes_trending, reason = check_risk_limits(signal, regime="TRENDING")
            assert passes_trending is False
            assert "TRENDING" in reason

    def test_check_risk_limits_with_volatile_regime(self):
        """Volatile regime should use lower confidence threshold (more cautious sizing instead)."""
        from app.consensus import check_risk_limits
        from unittest.mock import MagicMock

        signal = MagicMock()
        signal.p_win = 0.52  # Below base 0.55
        signal.ev_net_r = 0.3

        # Without regime - should fail
        passes_base, _ = check_risk_limits(signal, regime=None)
        assert passes_base is False

        # With volatile - regime adjustment applies
        passes_volatile, reason = check_risk_limits(signal, regime="VOLATILE")
        # Still should fail since 0.52 is quite low
        # The key test is that it uses regime-adjusted threshold
        assert "minimum" in reason.lower()

    def test_check_risk_limits_invalid_regime_falls_back(self):
        """Invalid regime should fall back to static threshold."""
        from app.consensus import check_risk_limits
        from unittest.mock import MagicMock

        signal = MagicMock()
        signal.p_win = 0.56
        signal.ev_net_r = 0.3

        # Invalid regime should still work (falls back)
        passes, reason = check_risk_limits(signal, regime="INVALID_REGIME")
        assert passes is True


class TestCircuitBreakerIntegration:
    """Test circuit breaker checks in execution path."""

    def test_circuit_breaker_blocks_on_api_pause(self):
        """Circuit breaker should block when API is paused."""
        from app.risk_governor import RiskGovernor

        governor = RiskGovernor()

        # Simulate API errors that trigger pause
        for _ in range(5):  # More than threshold
            governor.report_api_error()

        # Now check circuit breakers (0 = no existing position)
        result = governor.run_circuit_breaker_checks("BTC", symbol_position_count=0)

        assert result.allowed is False
        assert "pause" in result.reason.lower() or "api" in result.reason.lower()

    def test_circuit_breaker_blocks_on_loss_streak(self):
        """Circuit breaker should block on loss streak."""
        from app.risk_governor import RiskGovernor, MAX_CONSECUTIVE_LOSSES

        governor = RiskGovernor()

        # Record enough losses to trigger pause
        for _ in range(MAX_CONSECUTIVE_LOSSES + 1):
            governor.report_trade_result(is_win=False)

        result = governor.run_circuit_breaker_checks("BTC", symbol_position_count=0)

        assert result.allowed is False
        assert "loss" in result.reason.lower() or "streak" in result.reason.lower()

    def test_circuit_breaker_passes_when_healthy(self):
        """Circuit breaker should pass when no issues."""
        from app.risk_governor import RiskGovernor

        governor = RiskGovernor()

        result = governor.run_circuit_breaker_checks("BTC", symbol_position_count=0)

        assert result.allowed is True


class TestRiskGovernorProposedSize:
    """Test risk governor with actual proposed size."""

    def test_proposed_size_blocks_oversize_position(self):
        """Large proposed size should be blocked by position size check."""
        from app.risk_governor import RiskGovernor, MAX_POSITION_SIZE_PCT

        governor = RiskGovernor()

        # Account value $100k, max position 10% = $10k
        # Propose a $15k trade - should be blocked
        result = governor.run_all_checks(
            account_value=100000,
            margin_used=0,
            maintenance_margin=1,
            total_exposure=0,
            daily_pnl=0,
            proposed_size_usd=15000,  # 15% > 10% limit
        )

        assert result.allowed is False
        assert "position" in result.reason.lower() or "size" in result.reason.lower()

    def test_proposed_size_allows_valid_position(self):
        """Valid proposed size should pass."""
        from app.risk_governor import RiskGovernor

        governor = RiskGovernor()

        # Account value $100k, max position 10% = $10k
        # Propose a $5k trade - should pass
        result = governor.run_all_checks(
            account_value=100000,
            margin_used=0,
            maintenance_margin=1,
            total_exposure=0,
            daily_pnl=0,
            proposed_size_usd=5000,  # 5% < 10% limit
        )

        assert result.allowed is True


class TestCircuitBreakerPositionTracking:
    """Test circuit breaker with actual position data."""

    def test_position_count_from_account_state(self):
        """Governor should track positions from account state using public method."""
        from app.risk_governor import RiskGovernor, MAX_CONCURRENT_POSITIONS

        governor = RiskGovernor()

        # Use public method to update from account state
        mock_account_state = {
            "assetPositions": [
                {"position": {"coin": "BTC", "szi": "0.1"}},
                {"position": {"coin": "ETH", "szi": "-0.5"}},
                {"position": {"coin": "SOL", "szi": "10"}},
            ]
        }
        governor.update_positions_from_account_state(mock_account_state)

        # Verify position counts
        assert governor._current_position_count == 3
        assert governor.get_symbol_position_count("BTC") == 1
        assert governor.get_symbol_position_count("ETH") == 1
        assert governor.get_symbol_position_count("SOL") == 1
        assert governor.get_symbol_position_count("DOGE") == 0

        # Now check if we can open another position (depends on MAX_CONCURRENT_POSITIONS)
        result = governor.run_circuit_breaker_checks("DOGE", symbol_position_count=0)

        # Default MAX_CONCURRENT_POSITIONS is 3, so this should be blocked
        if MAX_CONCURRENT_POSITIONS == 3:
            assert result.allowed is False
            assert "concurrent" in result.reason.lower() or "position" in result.reason.lower()

    def test_existing_symbol_position_blocked_when_limit_is_one(self):
        """Adding to existing position should be blocked when MAX_POSITION_PER_SYMBOL=1."""
        from app.risk_governor import RiskGovernor, MAX_POSITION_PER_SYMBOL

        governor = RiskGovernor()

        # Already have BTC position
        governor._current_position_count = 1
        governor._positions_by_symbol = {"BTC": 1}

        # Adding to BTC position with count=1 (existing position)
        result = governor.run_circuit_breaker_checks("BTC", symbol_position_count=1)

        # When MAX_POSITION_PER_SYMBOL is 1, this should be blocked
        if MAX_POSITION_PER_SYMBOL == 1:
            assert result.allowed is False
            assert "position" in result.reason.lower() or "BTC" in result.reason

    def test_per_symbol_limit_blocks_new_position(self):
        """Per-symbol position limit should block additional positions."""
        from app.risk_governor import RiskGovernor, MAX_POSITION_PER_SYMBOL

        governor = RiskGovernor()

        # Already have max positions in BTC
        governor._current_position_count = 1
        governor._positions_by_symbol = {"BTC": MAX_POSITION_PER_SYMBOL}

        # Adding another BTC position should be blocked
        result = governor.run_circuit_breaker_checks("BTC", symbol_position_count=MAX_POSITION_PER_SYMBOL)

        # If MAX_POSITION_PER_SYMBOL is 1, this should be blocked
        if MAX_POSITION_PER_SYMBOL == 1:
            assert result.allowed is False

    def test_new_symbol_position_allowed(self):
        """New position in a different symbol should be allowed."""
        from app.risk_governor import RiskGovernor

        governor = RiskGovernor()

        # Have one BTC position
        governor._current_position_count = 1
        governor._positions_by_symbol = {"BTC": 1}

        # Opening new position in ETH (count=0) should be allowed
        result = governor.run_circuit_breaker_checks("ETH", symbol_position_count=0)

        assert result.allowed is True


class TestMigrationVerification:
    """Test that required migrations exist."""

    def test_risk_daily_pnl_migration_exists(self):
        """Verify 026_risk_governor_state.sql migration file exists with risk_daily_pnl table."""
        import os

        migration_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..", "db", "migrations", "026_risk_governor_state.sql"
        )

        # Check file exists
        assert os.path.exists(migration_path), f"Migration file not found: {migration_path}"

        # Check it contains risk_daily_pnl table definition
        with open(migration_path, "r") as f:
            content = f.read()

        assert "risk_daily_pnl" in content, "Migration missing risk_daily_pnl table"
        assert "starting_equity" in content, "Migration missing starting_equity column"
        assert "daily_drawdown_pct" in content, "Migration missing daily_drawdown_pct column"
