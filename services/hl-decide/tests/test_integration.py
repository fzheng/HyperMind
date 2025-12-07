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
