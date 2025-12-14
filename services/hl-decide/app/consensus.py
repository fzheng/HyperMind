"""
Consensus Signal Detection

Implements a multi-gate consensus system that generates trading signals
when multiple top traders agree on direction. Key innovations:

1. One vote per trader (collapse fills to net position delta)
2. Correlation-adjusted effective-K (prevents double-counting correlated traders)
3. Latency & price band gates (reject stale or moved-market signals)
4. EV gate with cost conversion (require positive expected value after fees)

@module consensus
"""

import math
import os
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Callable, Dict, List, Optional, Tuple
from uuid import uuid4

# Configuration
CONSENSUS_MIN_TRADERS = int(os.getenv("CONSENSUS_MIN_TRADERS", "3"))
CONSENSUS_MIN_AGREEING = int(os.getenv("CONSENSUS_MIN_AGREEING", "3"))
CONSENSUS_MIN_PCT = float(os.getenv("CONSENSUS_MIN_PCT", "0.70"))
CONSENSUS_MIN_EFFECTIVE_K = float(os.getenv("CONSENSUS_MIN_EFFECTIVE_K", "2.0"))
CONSENSUS_BASE_WINDOW_S = int(os.getenv("CONSENSUS_BASE_WINDOW_S", "120"))
CONSENSUS_MAX_STALENESS_FACTOR = float(os.getenv("CONSENSUS_MAX_STALENESS_FACTOR", "1.25"))
CONSENSUS_MAX_PRICE_BAND_BPS = float(os.getenv("CONSENSUS_MAX_PRICE_BAND_BPS", "8.0"))  # Legacy fallback
CONSENSUS_MAX_PRICE_DRIFT_R = float(os.getenv("CONSENSUS_MAX_PRICE_DRIFT_R", "0.25"))  # ATR-based: 0.25 R-units
CONSENSUS_EV_MIN_R = float(os.getenv("CONSENSUS_EV_MIN_R", "0.20"))
CONSENSUS_SYMBOLS = os.getenv("CONSENSUS_SYMBOLS", "BTC,ETH").split(",")

# EV calculation defaults
DEFAULT_AVG_WIN_R = float(os.getenv("DEFAULT_AVG_WIN_R", "0.5"))
DEFAULT_AVG_LOSS_R = float(os.getenv("DEFAULT_AVG_LOSS_R", "0.3"))
DEFAULT_FEES_BPS = float(os.getenv("DEFAULT_FEES_BPS", "10.0"))  # Round-trip taker fees (conservative default)
DEFAULT_SLIP_BPS = float(os.getenv("DEFAULT_SLIP_BPS", "10.0"))  # Expected slippage

# Per-exchange fee defaults (round-trip taker fees in bps)
# These are conservative estimates; actual VIP tiers may be lower
# NOTE: Phase 6.1 adds dynamic fee lookup via FeeProvider
EXCHANGE_FEES_BPS = {
    "hyperliquid": 10.0,  # 5 bps × 2 = 10 bps round-trip
    "aster": 10.0,        # Similar to HL
    "bybit": 12.0,        # 6 bps × 2 = 12 bps round-trip (VIP0)
}


def get_exchange_fees_bps(exchange: str = "hyperliquid") -> float:
    """
    Get round-trip fee cost in basis points for an exchange (static).

    This returns static fees. For dynamic fee lookup with caching,
    use get_exchange_fees_bps_dynamic() instead (Phase 6.1).

    Args:
        exchange: Exchange name (hyperliquid, aster, bybit)

    Returns:
        Round-trip fee cost in bps
    """
    return EXCHANGE_FEES_BPS.get(exchange.lower(), DEFAULT_FEES_BPS)


async def get_exchange_fees_bps_dynamic(exchange: str = "hyperliquid") -> float:
    """
    Get round-trip fee cost with dynamic lookup (Phase 6.1).

    Uses FeeProvider for cached fees with short TTL, falling back
    to static config if provider unavailable.

    Args:
        exchange: Exchange name (hyperliquid, aster, bybit)

    Returns:
        Round-trip fee cost in bps
    """
    try:
        from .fee_provider import get_fee_provider
        provider = get_fee_provider()
        return await provider.get_fees_bps(exchange)
    except Exception:
        # Fall back to static
        return get_exchange_fees_bps(exchange)


# Default expected hold time for funding calculations (hours)
# Based on typical consensus signal duration
# NOTE: Phase 6.1 adds HoldTimeEstimator for dynamic hold time from episode data
DEFAULT_HOLD_HOURS = float(os.getenv("DEFAULT_HOLD_HOURS", "24.0"))

# Whether to use dynamic hold time estimation (from historical episodes)
USE_DYNAMIC_HOLD_TIME = os.getenv("USE_DYNAMIC_HOLD_TIME", "true").lower() == "true"

# Default slippage estimate (bps) when orderbook unavailable
DEFAULT_SLIP_BPS_STATIC = float(os.getenv("DEFAULT_SLIP_BPS_STATIC", "2.0"))

# Reference size for initial slippage estimate in EV gating (before Kelly sizing)
# Use conservative $10k reference - actual slippage will be recalculated in executor
# with Kelly-sized position for final execution decision
SLIPPAGE_REFERENCE_SIZE_USD = float(os.getenv("SLIPPAGE_REFERENCE_SIZE_USD", "10000.0"))


def get_dynamic_hold_hours_sync(
    asset: str,
    regime: str | None = None,
) -> float:
    """
    Get expected hold time dynamically from cached episode data.

    Uses HoldTimeEstimator's cache if available, otherwise returns default.
    The cache is populated by async calls during startup.

    Args:
        asset: Asset symbol (BTC, ETH)
        regime: Optional market regime for adjustment (TRENDING, VOLATILE, etc.)

    Returns:
        Expected hold time in hours
    """
    if not USE_DYNAMIC_HOLD_TIME:
        return DEFAULT_HOLD_HOURS

    try:
        from .hold_time_estimator import get_hold_time_estimator
        estimator = get_hold_time_estimator()
        estimate = estimator.get_hold_time_sync(asset, regime)
        return estimate.hours
    except Exception:
        return DEFAULT_HOLD_HOURS


async def get_funding_cost_bps(
    asset: str,
    exchange: str = "hyperliquid",
    hold_hours: float = DEFAULT_HOLD_HOURS,
    side: str = "long",
) -> float:
    """
    Get expected funding cost for a position (Phase 6.1) - async version.

    Uses FundingProvider for cached rates with short TTL.

    Args:
        asset: Asset symbol (BTC, ETH)
        exchange: Exchange name (hyperliquid, aster, bybit)
        hold_hours: Expected hold time in hours
        side: Position side ("long" or "short")

    Returns:
        Funding cost in bps (positive = cost, negative = rebate)
    """
    try:
        from .funding_provider import get_funding_provider
        provider = get_funding_provider()
        return await provider.get_funding_cost_bps(asset, exchange, hold_hours, side)
    except Exception:
        # Fall back to conservative default (8 bps per 8h, scaled to hold time)
        # For shorts, negate (they receive when longs pay)
        raw_cost = (hold_hours / 8) * 8.0
        return -raw_cost if side.lower() == "short" else raw_cost


def get_funding_cost_bps_sync(
    asset: str,
    exchange: str = "hyperliquid",
    hold_hours: float = DEFAULT_HOLD_HOURS,
    side: str = "long",
) -> float:
    """
    Get expected funding cost - synchronous version using cached data.

    Uses FundingProvider's cache if available, otherwise returns default.
    The cache is populated by async calls elsewhere in the system.

    Args:
        asset: Asset symbol (BTC, ETH)
        exchange: Exchange name (hyperliquid, aster, bybit)
        hold_hours: Expected hold time in hours
        side: Position side ("long" or "short")

    Returns:
        Funding cost in bps (positive = cost, negative = rebate)
    """
    try:
        from .funding_provider import get_funding_provider, FUNDING_INTERVAL_HOURS
        provider = get_funding_provider()
        key = provider._get_cache_key(exchange, asset)

        # Check if we have valid cached data
        if key in provider._cache and not provider._cache[key].is_expired:
            data = provider._cache[key].data
            return data.cost_for_hold_time(hold_hours, side)

        # No cache - use static default
        # Default: 8 bps per 8h funding interval
        intervals = hold_hours / FUNDING_INTERVAL_HOURS
        default_rate_bps = 8.0  # Conservative default
        raw_cost = default_rate_bps * intervals
        # For shorts, negate (they receive when longs pay)
        return -raw_cost if side.lower() == "short" else raw_cost
    except Exception:
        # Fall back to conservative default
        raw_cost = (hold_hours / 8) * 8.0
        return -raw_cost if side.lower() == "short" else raw_cost


async def get_slippage_estimate_bps(
    asset: str,
    exchange: str = "hyperliquid",
    order_size_usd: float = 10000.0,
    side: str = "buy",
) -> float:
    """
    Get expected slippage for an order (Phase 6.1) - async version.

    Uses SlippageProvider for orderbook-based estimates when available.

    Args:
        asset: Asset symbol (BTC, ETH)
        exchange: Exchange name (hyperliquid, aster, bybit)
        order_size_usd: Order size in USD
        side: "buy" or "sell"

    Returns:
        Estimated slippage in bps
    """
    try:
        from .slippage_provider import get_slippage_provider
        provider = get_slippage_provider()
        estimate = await provider.estimate_slippage(asset, exchange, order_size_usd, side)
        return estimate.estimated_slippage_bps
    except Exception:
        # Fall back to conservative static default
        return DEFAULT_SLIP_BPS_STATIC


def get_slippage_estimate_bps_sync(
    asset: str,
    exchange: str = "hyperliquid",
    order_size_usd: float = 10000.0,
) -> float:
    """
    Get expected slippage - synchronous version using cached orderbook.

    Uses SlippageProvider's cache if available, otherwise returns static default.
    The cache is populated by async calls elsewhere in the system.

    Args:
        asset: Asset symbol (BTC, ETH)
        exchange: Exchange name (hyperliquid, aster, bybit)
        order_size_usd: Order size in USD

    Returns:
        Estimated slippage in bps
    """
    try:
        from .slippage_provider import get_slippage_provider, DEFAULT_SLIPPAGE_BPS, SIZE_THRESHOLD_SMALL, SIZE_THRESHOLD_LARGE
        provider = get_slippage_provider()
        key = provider._get_cache_key(exchange, asset)

        # Check if we have valid cached orderbook
        if key in provider._cache and not provider._cache[key].is_expired:
            # Use cached orderbook for estimation
            orderbook = provider._cache[key].data
            estimate = provider._estimate_from_orderbook(orderbook, order_size_usd, "buy")
            return estimate.estimated_slippage_bps

        # No cache - use static default based on size
        if order_size_usd < SIZE_THRESHOLD_SMALL:
            size_bucket = "small"
        elif order_size_usd < SIZE_THRESHOLD_LARGE:
            size_bucket = "medium"
        else:
            size_bucket = "large"

        exchange_rates = DEFAULT_SLIPPAGE_BPS.get(exchange, DEFAULT_SLIPPAGE_BPS.get("hyperliquid", {}))
        asset_rates = exchange_rates.get(asset, exchange_rates.get("BTC", {"small": 2, "medium": 4, "large": 10}))
        return asset_rates.get(size_bucket, DEFAULT_SLIP_BPS_STATIC)
    except Exception:
        # Fall back to static default
        return DEFAULT_SLIP_BPS_STATIC


# Default correlation (used when pairwise not computed)
DEFAULT_CORRELATION = float(os.getenv("DEFAULT_CORRELATION", "0.3"))

# Weight cap for individual traders (legacy, deprecated)
WEIGHT_CAP = float(os.getenv("CONSENSUS_WEIGHT_CAP", "1.0"))

# Vote weighting configuration
# Mode: "equity" (equity-normalized with sqrt), "log" (logarithmic), or "linear" (legacy)
VOTE_WEIGHT_MODE = os.getenv("VOTE_WEIGHT_MODE", "log")  # Default to log until equity data available
VOTE_WEIGHT_LOG_BASE = float(os.getenv("VOTE_WEIGHT_LOG_BASE", "10000.0"))  # $10k base for log
VOTE_WEIGHT_MAX = float(os.getenv("VOTE_WEIGHT_MAX", "1.0"))  # Max weight per trader

# ============================================================================
# RISK DEFAULTS & FAIL-SAFES
# ============================================================================
# Until Kelly/risk sizing is implemented, enforce conservative static limits.
# These are hard caps that prevent over-exposure regardless of signal quality.
#
# NOTE: These are conservative defaults. Adjust based on account size and risk tolerance.
# These values should be overridden by proper Kelly sizing when Phase 4 is complete.
# ============================================================================

# Maximum position size as fraction of account equity (per position)
# Default 2% = very conservative, prevents catastrophic single-trade losses
MAX_POSITION_SIZE_PCT = float(os.getenv("MAX_POSITION_SIZE_PCT", "2.0"))

# Maximum total exposure (sum of all open positions) as fraction of equity
# Default 10% = allows up to 5 concurrent 2% positions
MAX_TOTAL_EXPOSURE_PCT = float(os.getenv("MAX_TOTAL_EXPOSURE_PCT", "10.0"))

# Maximum daily loss before halting signals (as fraction of equity)
# Default 5% = stop trading after 5% drawdown in a day
MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT", "5.0"))

# Minimum EV required to generate a signal (in R-multiples)
# Higher values = fewer but higher quality signals
# Note: This overrides CONSENSUS_EV_MIN_R for risk management
MIN_SIGNAL_EV_R = float(os.getenv("MIN_SIGNAL_EV_R", str(CONSENSUS_EV_MIN_R)))

# Minimum confidence (p_win) required for signal generation
# Default 0.55 = require at least 55% estimated win probability
MIN_SIGNAL_CONFIDENCE = float(os.getenv("MIN_SIGNAL_CONFIDENCE", "0.55"))

# Maximum leverage allowed (if/when execution layer is added)
# Default 1.0 = no leverage until proper risk sizing is implemented
MAX_LEVERAGE = float(os.getenv("MAX_LEVERAGE", "1.0"))

# Cooldown period between signals for the same symbol (seconds)
# Prevents rapid-fire entries that could compound losses
SIGNAL_COOLDOWN_SECONDS = int(os.getenv("SIGNAL_COOLDOWN_SECONDS", "300"))  # 5 minutes


def check_risk_limits(
    signal: "ConsensusSignal",
    regime: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Check if a signal passes conservative risk limits.

    This is a fail-safe before any position sizing logic.
    Returns (passes, reason).

    Args:
        signal: The consensus signal to check
        regime: Optional market regime for adjusted thresholds
                (TRENDING, RANGING, VOLATILE, UNKNOWN)

    Returns:
        Tuple of (passes_checks, reason_if_failed)
    """
    # Get regime-adjusted confidence threshold
    min_confidence = MIN_SIGNAL_CONFIDENCE

    if regime:
        try:
            from .regime import get_regime_adjusted_confidence, MarketRegime
            # Map string to enum
            regime_enum = MarketRegime[regime.upper()] if regime else MarketRegime.UNKNOWN
            # get_regime_adjusted_confidence adjusts confidence UP for trending, DOWN for volatile
            # For risk limits, we want to RAISE the threshold in volatile regimes (require higher confidence)
            # So we use it directly on the minimum required confidence
            min_confidence = get_regime_adjusted_confidence(MIN_SIGNAL_CONFIDENCE, regime_enum)
        except (KeyError, ImportError, Exception):
            pass  # Fall back to static threshold

    # Check minimum confidence
    if signal.p_win < min_confidence:
        regime_note = f" ({regime})" if regime else ""
        return (
            False,
            f"Confidence {signal.p_win:.2f} < minimum {min_confidence:.2f}{regime_note}"
        )

    # Check minimum EV
    if signal.ev_net_r < MIN_SIGNAL_EV_R:
        return (
            False,
            f"EV {signal.ev_net_r:.3f}R < minimum {MIN_SIGNAL_EV_R:.3f}R"
        )

    # All checks passed
    return (True, "")


@dataclass
class Fill:
    """Represents a single fill from a trader."""
    fill_id: str
    address: str
    asset: str
    side: str  # "long" or "short" or "buy" or "sell"
    size: float
    price: float
    ts: datetime

    @property
    def signed_size(self) -> float:
        """Positive for buys/longs, negative for sells/shorts."""
        if self.side.lower() in ("long", "buy"):
            return self.size
        return -self.size

    @property
    def direction(self) -> str:
        """Infer direction from side."""
        return "long" if self.side.lower() in ("long", "buy") else "short"


@dataclass
class Vote:
    """A trader's vote in a consensus window (one per trader)."""
    address: str
    direction: str  # "long" or "short"
    weight: float  # Computed weight (see calculate_vote_weight)
    price: float  # Entry price (for price band check)
    ts: datetime  # Timestamp of vote
    notional: float = 0.0  # Position notional in USD
    equity: Optional[float] = None  # Trader's account equity (if available)


@dataclass
class ConsensusWindow:
    """Sliding window for collecting fills and detecting consensus."""
    symbol: str
    window_start: datetime
    window_s: int
    fills: List[Fill] = field(default_factory=list)

    @property
    def is_expired(self) -> bool:
        """Check if window has expired."""
        now = datetime.now(timezone.utc)
        return (now - self.window_start).total_seconds() > self.window_s


@dataclass
class ConsensusSignal:
    """A consensus signal ready to be acted upon."""
    id: str
    symbol: str
    direction: str
    entry_price: float
    stop_price: Optional[float]
    # Consensus metrics
    n_traders: int
    n_agreeing: int
    eff_k: float
    dispersion: float
    # Confidence & EV
    p_win: float
    ev_gross_r: float
    ev_cost_r: float
    ev_net_r: float
    # Timing
    latency_ms: int
    median_voter_price: float
    mid_delta_bps: float
    # Metadata
    created_at: datetime
    trigger_addresses: List[str]


class ConsensusDetector:
    """
    Detects consensus among tracked traders and generates signals.

    Uses multiple gates to filter out noise:
    1. Dispersion gate: Require supermajority agreement
    2. Effective-K gate: Correlation-adjusted trader count
    3. Latency gate: Reject stale signals
    4. Price band gate: Reject if market moved too far (requires valid ATR)
    5. EV gate: Require positive expected value after costs
    """

    def __init__(self, target_exchange: str = "hyperliquid"):
        """
        Initialize consensus detector.

        Args:
            target_exchange: Target exchange for fee calculation (hyperliquid, aster, bybit)
        """
        self.windows: Dict[str, ConsensusWindow] = {}
        self.correlation_matrix: Dict[Tuple[str, str], float] = {}
        self.current_prices: Dict[str, float] = {}
        # Target exchange for fee calculation in EV gate
        self._target_exchange = target_exchange.lower()
        # ATR-based stop fractions per asset (updated by ATR provider)
        self.stop_fractions: Dict[str, float] = {
            "BTC": 0.01,  # Default 1%, will be updated by ATR provider
            "ETH": 0.01,
        }
        # Track ATR data quality for strict mode gating
        # Maps symbol -> (is_valid_for_gating, reason)
        self.atr_validity: Dict[str, Tuple[bool, str]] = {}

    @property
    def target_exchange(self) -> str:
        """Get target exchange for fee calculation."""
        return self._target_exchange

    def set_target_exchange(self, exchange: str) -> None:
        """
        Set target exchange for fee calculation in EV gate.

        Args:
            exchange: Exchange name (hyperliquid, aster, bybit)
        """
        self._target_exchange = exchange.lower()

    def calculate_ev_for_exchange(
        self,
        asset: str,
        direction: str,
        entry_price: float,
        stop_price: float,
        p_win: float,
        exchange: str,
        order_size_usd: float = SLIPPAGE_REFERENCE_SIZE_USD,
        hold_hours: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Calculate EV for a specific exchange (Phase 6.1 per-venue awareness).

        This allows comparison of EV across different execution venues
        to find the best place to execute a trade.

        Args:
            asset: Asset symbol (BTC, ETH)
            direction: Position direction (long, short)
            entry_price: Entry price
            stop_price: Stop price
            p_win: Probability of winning
            exchange: Target exchange (hyperliquid, bybit, aster)
            order_size_usd: Order size for slippage estimation
            hold_hours: Expected hold time (uses dynamic estimate if None)

        Returns:
            Dict with ev_gross_r, ev_cost_r, ev_net_r, funding_cost_r,
                   fees_bps, slippage_bps, funding_bps, exchange
        """
        exchange = exchange.lower()

        # Get exchange-specific costs
        fees_bps = get_exchange_fees_bps(exchange)

        # Get dynamic hold time if not specified
        if hold_hours is None:
            hold_hours = get_dynamic_hold_hours_sync(asset)

        # Get funding cost with direction-aware sign
        funding_bps = get_funding_cost_bps_sync(
            asset=asset,
            exchange=exchange,
            hold_hours=hold_hours,
            side=direction,
        )

        # Get slippage estimate for the order size
        slippage_bps = get_slippage_estimate_bps_sync(
            asset=asset,
            exchange=exchange,
            order_size_usd=order_size_usd,
        )

        # Calculate EV
        ev_result = calculate_ev(
            p_win=p_win,
            entry_px=entry_price,
            stop_px=stop_price,
            fees_bps=fees_bps,
            slip_bps=slippage_bps,
            funding_bps=funding_bps,
        )

        # Add cost breakdown
        ev_result["fees_bps"] = fees_bps
        ev_result["slippage_bps"] = slippage_bps
        ev_result["funding_bps"] = funding_bps
        ev_result["exchange"] = exchange
        ev_result["hold_hours"] = hold_hours

        return ev_result

    def compare_ev_across_exchanges(
        self,
        asset: str,
        direction: str,
        entry_price: float,
        stop_price: float,
        p_win: float,
        exchanges: Optional[List[str]] = None,
        order_size_usd: float = SLIPPAGE_REFERENCE_SIZE_USD,
        hold_hours: Optional[float] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare EV across multiple exchanges to find best execution venue.

        Phase 6.1: Per-venue EV comparison for optimal execution routing.

        Args:
            asset: Asset symbol (BTC, ETH)
            direction: Position direction (long, short)
            entry_price: Entry price
            stop_price: Stop price
            p_win: Probability of winning
            exchanges: List of exchanges to compare (defaults to all supported)
            order_size_usd: Order size for slippage estimation
            hold_hours: Expected hold time (uses dynamic estimate if None)

        Returns:
            Dict mapping exchange -> EV result dict, plus 'best_exchange' key
        """
        if exchanges is None:
            exchanges = ["hyperliquid", "bybit"]  # Default supported exchanges

        results = {}
        for exchange in exchanges:
            try:
                ev = self.calculate_ev_for_exchange(
                    asset=asset,
                    direction=direction,
                    entry_price=entry_price,
                    stop_price=stop_price,
                    p_win=p_win,
                    exchange=exchange,
                    order_size_usd=order_size_usd,
                    hold_hours=hold_hours,
                )
                results[exchange] = ev
            except Exception as e:
                print(f"[consensus] Error calculating EV for {exchange}: {e}")
                results[exchange] = {
                    "ev_net_r": float("-inf"),
                    "error": str(e),
                    "exchange": exchange,
                }

        # Find best exchange by net EV
        if results:
            best = max(results.items(), key=lambda x: x[1].get("ev_net_r", float("-inf")))
            results["best_exchange"] = best[0]
            results["best_ev_net_r"] = best[1].get("ev_net_r", 0.0)

        return results

    def set_current_price(self, symbol: str, price: float) -> None:
        """Update current mid price for a symbol."""
        self.current_prices[symbol] = price

    def get_current_mid(self, symbol: str) -> float:
        """Get current mid price for a symbol."""
        return self.current_prices.get(symbol, 0.0)

    def set_stop_fraction(
        self,
        symbol: str,
        stop_fraction: float,
        is_valid_for_gating: bool = True,
        validity_reason: str = "",
    ) -> None:
        """
        Update ATR-based stop fraction for a symbol.

        Called by ATR provider when new ATR data is available.
        Stop fraction = ATR × multiplier / price

        Args:
            symbol: Asset symbol (BTC, ETH)
            stop_fraction: Stop distance as fraction (0.01 = 1%)
            is_valid_for_gating: Whether the ATR data is valid for gating decisions
            validity_reason: Reason for validity status (for logging)
        """
        self.stop_fractions[symbol.upper()] = max(0.001, min(0.10, stop_fraction))
        self.atr_validity[symbol.upper()] = (is_valid_for_gating, validity_reason)

    def get_stop_fraction(self, symbol: str) -> float:
        """Get current stop fraction for a symbol (default 1%)."""
        return self.stop_fractions.get(symbol.upper(), 0.01)

    def is_atr_valid_for_gating(self, symbol: str) -> Tuple[bool, str]:
        """
        Check if ATR data is valid for gating decisions.

        Returns:
            Tuple of (is_valid, reason)
        """
        return self.atr_validity.get(symbol.upper(), (True, ""))

    def update_correlation(self, addr1: str, addr2: str, correlation: float) -> None:
        """Update pairwise correlation between two traders."""
        key = tuple(sorted([addr1.lower(), addr2.lower()]))
        self.correlation_matrix[key] = max(0.0, min(1.0, correlation))

    def process_fill(self, fill: Fill, atr_percentile: float = 0.5) -> Optional[ConsensusSignal]:
        """
        Process an incoming fill and check for consensus.

        Args:
            fill: The incoming fill event
            atr_percentile: Current ATR percentile for adaptive window

        Returns:
            ConsensusSignal if consensus detected, None otherwise
        """
        if fill.asset not in CONSENSUS_SYMBOLS:
            return None

        symbol = fill.asset
        now = datetime.now(timezone.utc)
        window_s = adaptive_window_seconds(atr_percentile)

        # Get or create window
        window = self.windows.get(symbol)
        if window is None or window.is_expired:
            window = ConsensusWindow(
                symbol=symbol,
                window_start=now,
                window_s=window_s,
                fills=[],
            )
            self.windows[symbol] = window

        # Add fill to window
        window.fills.append(fill)

        # Update current price
        self.set_current_price(symbol, fill.price)

        # Check for consensus
        return self.check_consensus(symbol)

    def check_consensus(self, symbol: str) -> Optional[ConsensusSignal]:
        """
        Check if current window has reached consensus.

        Applies all gates in sequence:
        1. Collapse to one vote per trader
        2. Check dispersion (supermajority)
        3. Check effective-K (correlation-adjusted)
        4. Check latency and price band
        5. Check EV after costs

        Returns:
            ConsensusSignal if all gates pass, None otherwise
        """
        window = self.windows.get(symbol)
        if window is None or len(window.fills) == 0:
            return None

        # Collapse to one vote per trader
        votes = self.collapse_to_votes(window.fills)

        if len(votes) < CONSENSUS_MIN_TRADERS:
            return None

        directions = [v.direction for v in votes]

        # Gate 1: Dispersion (supermajority)
        passes, majority_dir = passes_consensus_gates(
            directions,
            min_agreeing=CONSENSUS_MIN_AGREEING,
            min_pct=CONSENSUS_MIN_PCT,
        )
        if not passes:
            return None

        # Gate 2: Correlation-adjusted effective-K
        agreeing_votes = [v for v in votes if v.direction == majority_dir]
        addresses = [v.address for v in agreeing_votes]
        weights = {v.address: v.weight for v in agreeing_votes}
        eff_k = self.eff_k_from_corr(weights)

        if eff_k < CONSENSUS_MIN_EFFECTIVE_K:
            return None

        # Gate 3: Latency + price band
        if not self.passes_latency_and_price_gates(window, agreeing_votes):
            return None

        # Calculate entry price (median of agreeing voters)
        median_entry = statistics.median(v.price for v in agreeing_votes)
        mid_price = self.get_current_mid(symbol)

        # Calculate stop price using ATR-based stop fraction
        # Stop fraction is updated by ATR provider based on current volatility
        stop_fraction = self.get_stop_fraction(symbol)
        stop_distance = median_entry * stop_fraction
        if majority_dir == "long":
            stop_price = median_entry - stop_distance
        else:
            stop_price = median_entry + stop_distance

        # Gate 4: EV after costs (using per-exchange fees, funding, slippage)
        p_win = self.calibrated_p_win(agreeing_votes, eff_k)
        exchange_fees_bps = get_exchange_fees_bps(self._target_exchange)

        # Get funding cost for expected hold time (uses cache or defaults)
        # Funding is signed: positive = cost, negative = rebate
        # Hold time is dynamic: uses historical episode data if available (Phase 6.1)
        asset = symbol.split("-")[0] if "-" in symbol else symbol.replace("USDC", "")
        hold_hours = get_dynamic_hold_hours_sync(asset)
        funding_cost_bps = get_funding_cost_bps_sync(
            asset=asset,
            exchange=self._target_exchange,
            hold_hours=hold_hours,
            side=majority_dir,  # Pass position direction for correct funding sign
        )

        # Get slippage estimate (uses cached orderbook or static defaults)
        # Use REFERENCE SIZE ($10k) for initial EV gating - this is intentional!
        # Actual slippage is recalculated in executor with Kelly-sized position.
        # Using vote notional here would:
        # 1. Underestimate slippage if Kelly sizes up (e.g., $100k Kelly vs $10k vote)
        # 2. Overestimate slippage if Kelly sizes down (e.g., $5k Kelly vs $50k vote)
        # Reference size provides conservative baseline for EV gating decision.
        slippage_bps = get_slippage_estimate_bps_sync(
            asset=asset,
            exchange=self._target_exchange,
            order_size_usd=SLIPPAGE_REFERENCE_SIZE_USD,
        )

        ev_result = calculate_ev(
            p_win=p_win,
            entry_px=median_entry,
            stop_px=stop_price,
            fees_bps=exchange_fees_bps,
            slip_bps=slippage_bps,
            funding_bps=funding_cost_bps,
        )

        if ev_result["ev_net_r"] < CONSENSUS_EV_MIN_R:
            return None

        # All gates passed! Create signal
        now = datetime.now(timezone.utc)
        oldest_fill = min(v.ts for v in agreeing_votes)
        latency_ms = int((now - oldest_fill).total_seconds() * 1000)
        mid_delta_bps = abs(mid_price - median_entry) / median_entry * 10000 if median_entry > 0 else 0

        # Calculate dispersion (std of vote weights by direction)
        dispersion = self._calculate_dispersion(votes, majority_dir)

        signal = ConsensusSignal(
            id=str(uuid4()),
            symbol=symbol,
            direction=majority_dir,
            entry_price=median_entry,
            stop_price=stop_price,
            n_traders=len(votes),
            n_agreeing=len(agreeing_votes),
            eff_k=eff_k,
            dispersion=dispersion,
            p_win=p_win,
            ev_gross_r=ev_result["ev_gross_r"],
            ev_cost_r=ev_result["ev_cost_r"],
            ev_net_r=ev_result["ev_net_r"],
            latency_ms=latency_ms,
            median_voter_price=median_entry,
            mid_delta_bps=mid_delta_bps,
            created_at=now,
            trigger_addresses=addresses,
        )

        # Clear window after generating signal
        self.windows[symbol] = None

        return signal

    def collapse_to_votes(
        self,
        fills: List[Fill],
        equity_by_address: Optional[Dict[str, float]] = None,
    ) -> List[Vote]:
        """
        Collapse multiple fills per trader to one vote each.

        One trader = one vote, based on net position change.
        Weight calculated using calculate_vote_weight() with:
        - Equity-normalized sqrt weighting if equity available
        - Logarithmic scaling as fallback

        Args:
            fills: List of fills in the window
            equity_by_address: Optional dict mapping address -> account equity

        Returns:
            List of votes (one per trader)
        """
        by_trader: Dict[str, List[Fill]] = defaultdict(list)
        for f in fills:
            by_trader[f.address.lower()].append(f)

        equity_lookup = equity_by_address or {}
        votes = []
        for addr, trader_fills in by_trader.items():
            net_delta = sum(f.signed_size for f in trader_fills)
            if abs(net_delta) < 1e-9:
                continue  # No net position change

            direction = "long" if net_delta > 0 else "short"

            # Use weighted average price
            total_size = sum(abs(f.size) for f in trader_fills)
            if total_size > 0:
                avg_price = sum(f.price * abs(f.size) for f in trader_fills) / total_size
            else:
                avg_price = trader_fills[-1].price if trader_fills else 0.0

            # Calculate notional value
            notional = abs(net_delta) * avg_price

            # Get equity if available
            equity = equity_lookup.get(addr.lower())

            # Calculate weight using improved formula
            weight = calculate_vote_weight(notional, equity)

            # Use latest timestamp
            latest_ts = max(f.ts for f in trader_fills)

            votes.append(Vote(
                address=addr,
                direction=direction,
                weight=weight,
                price=avg_price,
                ts=latest_ts,
                notional=notional,
                equity=equity,
            ))

        return votes

    def eff_k_from_corr(
        self,
        weights: Dict[str, float],
        fallback_counter_callback: Optional[Callable[[], None]] = None,
    ) -> float:
        """
        Calculate effective K using correlation matrix.

        Formula: effK = (Σᵢ wᵢ)² / Σᵢ Σⱼ wᵢ wⱼ ρᵢⱼ

        Args:
            weights: Dict mapping address to weight
            fallback_counter_callback: Optional callback to increment when default ρ is used

        Returns:
            Effective number of independent traders
        """
        addrs = list(weights.keys())
        if len(addrs) <= 1:
            return float(len(addrs))

        num = sum(weights.values()) ** 2
        den = 0.0
        fallback_count = 0

        for i, a in enumerate(addrs):
            for j, b in enumerate(addrs):
                if i == j:
                    rho = 1.0
                else:
                    key = tuple(sorted([a, b]))
                    stored_rho = self.correlation_matrix.get(key)
                    if stored_rho is None:
                        rho = DEFAULT_CORRELATION
                        fallback_count += 1
                    else:
                        rho = stored_rho
                    rho = max(0.0, min(1.0, rho))  # Clip negative correlations

                den += weights[a] * weights[b] * rho

        # Report fallback usage (only count unique pairs, not i,j and j,i)
        if fallback_count > 0 and fallback_counter_callback:
            # Each pair is counted twice (i,j and j,i), so divide by 2
            fallback_counter_callback()

        return num / max(den, 1e-9)

    def passes_latency_and_price_gates(
        self,
        window: ConsensusWindow,
        votes: List[Vote],
    ) -> bool:
        """
        Check latency and price band gates.

        Price band now uses ATR-based R-units instead of fixed BPS:
        - Get stop_fraction from ATR provider (e.g., 1% for BTC)
        - Convert price deviation to R-units: deviation_R = bps / (stop_fraction * 10000)
        - Compare against CONSENSUS_MAX_PRICE_DRIFT_R (default 0.25 R)

        IMPORTANT: In strict mode (ATR_STRICT_MODE=true), this gate will FAIL
        if ATR data is not available or is using hardcoded fallback. This prevents
        trading on stale/guessed volatility data.

        Args:
            window: Current consensus window
            votes: Agreeing votes to check

        Returns:
            True if both gates pass
        """
        if not votes:
            return False

        now = datetime.now(timezone.utc)

        # Latency gate: oldest fill must be within window × factor
        oldest_ts = min(v.ts for v in votes)
        staleness_s = (now - oldest_ts).total_seconds()
        max_staleness = window.window_s * CONSENSUS_MAX_STALENESS_FACTOR

        if staleness_s > max_staleness:
            return False

        # ATR validity gate: check if we have valid ATR data for this symbol
        is_valid, reason = self.is_atr_valid_for_gating(window.symbol)
        if not is_valid:
            print(f"[consensus] Price gate BLOCKED: {reason}")
            return False

        # Price band gate: current mid vs median voter entry (ATR-based R-units)
        median_entry = statistics.median(v.price for v in votes)
        mid_price = self.get_current_mid(window.symbol)

        if median_entry <= 0 or mid_price <= 0:
            return False

        # Calculate price deviation in BPS
        bps_deviation = abs(mid_price - median_entry) / median_entry * 10000

        # Convert BPS to R-units using ATR-based stop fraction
        # R-units = bps_deviation / (stop_fraction_pct * 100)
        # E.g., 8 bps with 1% stop = 8 / 100 = 0.08 R
        stop_fraction = self.get_stop_fraction(window.symbol)
        stop_bps = stop_fraction * 10000  # Convert fraction to BPS (0.01 -> 100 bps)

        if stop_bps > 0:
            deviation_r = bps_deviation / stop_bps
            return deviation_r <= CONSENSUS_MAX_PRICE_DRIFT_R
        else:
            # This shouldn't happen if ATR validity check passed
            print(f"[consensus] WARNING: stop_bps=0 for {window.symbol}, using legacy BPS check")
            return bps_deviation <= CONSENSUS_MAX_PRICE_BAND_BPS

    def calibrated_p_win(self, votes: List[Vote], eff_k: float) -> float:
        """
        Calculate calibrated win probability.

        Simple model based on:
        - Direction weight (agreement strength)
        - Effective K (diversity)

        TODO: Fit logistic model on historical data with more features:
        - vol_regime, spread/ATR, latency, etc.

        Args:
            votes: Agreeing votes
            eff_k: Effective K value

        Returns:
            Estimated probability of win (0-1)
        """
        if not votes:
            return 0.5

        # Simple heuristic: base probability + bonuses
        base = 0.5

        # Bonus for more independent signals
        k_bonus = min(0.15, (eff_k - 1) * 0.05)

        # Bonus for stronger agreement (higher total weight)
        total_weight = sum(v.weight for v in votes)
        weight_bonus = min(0.1, total_weight * 0.02)

        p = base + k_bonus + weight_bonus
        return max(0.3, min(0.8, p))  # Clamp to reasonable range

    def _calculate_dispersion(self, votes: List[Vote], majority_dir: str) -> float:
        """
        Calculate dispersion of votes (std of directional weights).

        Lower dispersion = stronger agreement.
        """
        if len(votes) < 2:
            return 0.0

        # Convert to signed weights (-1 for short, +1 for long)
        signed_weights = []
        for v in votes:
            sign = 1.0 if v.direction == "long" else -1.0
            signed_weights.append(sign * v.weight)

        return statistics.stdev(signed_weights)


def passes_consensus_gates(
    directions: List[str],
    min_agreeing: int = CONSENSUS_MIN_AGREEING,
    min_pct: float = CONSENSUS_MIN_PCT,
) -> Tuple[bool, str]:
    """
    Check if directions pass consensus gates.

    Requires both:
    - At least min_agreeing traders in majority direction
    - At least min_pct fraction in majority direction

    Args:
        directions: List of "long" or "short" strings
        min_agreeing: Minimum agreeing traders required
        min_pct: Minimum fraction required (0-1)

    Returns:
        Tuple of (passes, majority_direction)
    """
    if not directions:
        return (False, "")

    long_count = sum(1 for d in directions if d == "long")
    short_count = len(directions) - long_count

    if long_count >= short_count:
        majority_count = long_count
        majority_dir = "long"
    else:
        majority_count = short_count
        majority_dir = "short"

    # Check both conditions
    if majority_count < min_agreeing:
        return (False, "")

    if len(directions) > 0 and majority_count / len(directions) < min_pct:
        return (False, "")

    return (True, majority_dir)


def adaptive_window_seconds(atr_percentile: float) -> int:
    """
    Calculate adaptive window size based on volatility.

    Shorter windows in low-vol (quick signals), longer in high-vol (allow gathering).

    Args:
        atr_percentile: Current ATR percentile (0-1)

    Returns:
        Window size in seconds
    """
    base = CONSENSUS_BASE_WINDOW_S
    lo = 60   # minimum
    hi = 360  # maximum

    if atr_percentile < 0.3:
        return max(lo, base)  # 2 min in low vol
    elif atr_percentile < 0.7:
        return min(hi, base * 2)  # 4 min in medium vol
    else:
        return min(hi, base * 3)  # 6 min in high vol


def bps_to_R(entry_px: float, stop_px: float, bps: float) -> float:
    """
    Convert basis points to R-units based on stop distance.

    Args:
        entry_px: Entry price
        stop_px: Stop price
        bps: Cost in basis points

    Returns:
        Cost in R-units
    """
    if entry_px <= 0:
        return 0.0

    stop_bps = abs(entry_px - stop_px) / entry_px * 10000
    return bps / max(stop_bps, 1.0)


def calculate_ev(
    p_win: float,
    entry_px: float,
    stop_px: float,
    avg_win_r: float = DEFAULT_AVG_WIN_R,
    avg_loss_r: float = DEFAULT_AVG_LOSS_R,
    fees_bps: float = DEFAULT_FEES_BPS,
    slip_bps: float = DEFAULT_SLIP_BPS,
    funding_bps: float = 0.0,
) -> Dict[str, float]:
    """
    Calculate expected value after costs.

    Args:
        p_win: Probability of winning
        entry_px: Entry price
        stop_px: Stop price
        avg_win_r: Average win in R-multiples
        avg_loss_r: Average loss in R-multiples
        fees_bps: Round-trip fees in bps
        slip_bps: Expected slippage in bps
        funding_bps: Expected funding cost in bps (for expected hold time)

    Returns:
        Dict with ev_gross_r, ev_cost_r, ev_net_r, funding_cost_r
    """
    gross_ev = p_win * avg_win_r - (1 - p_win) * avg_loss_r
    total_bps = fees_bps + slip_bps + funding_bps
    cost_r = bps_to_R(entry_px, stop_px, total_bps)
    funding_r = bps_to_R(entry_px, stop_px, funding_bps)
    net_ev = gross_ev - cost_r

    return {
        "ev_gross_r": gross_ev,
        "ev_cost_r": cost_r,
        "ev_net_r": net_ev,
        "funding_cost_r": funding_r,
    }


def calculate_vote_weight(
    notional: float,
    equity: Optional[float] = None,
    mode: str = VOTE_WEIGHT_MODE,
    log_base: float = VOTE_WEIGHT_LOG_BASE,
    max_weight: float = VOTE_WEIGHT_MAX,
) -> float:
    """
    Calculate vote weight for a trader based on their position and equity.

    Three modes are supported:
    1. "equity": Equity-normalized with sqrt smoothing
       - weight = sqrt(notional / equity) capped at max_weight
       - Best reflects risk-adjusted conviction
       - Requires equity data to be available

    2. "log": Logarithmic scaling
       - weight = log(1 + notional / base)
       - Smooths out large positions vs small ones
       - Fallback when equity not available

    3. "linear": Legacy linear scaling (deprecated)
       - weight = min(notional / base, max_weight)
       - Flattens all large accounts

    Args:
        notional: Position notional in USD
        equity: Trader's account equity (optional)
        mode: Weighting mode ("equity", "log", or "linear")
        log_base: Base for logarithmic scaling (default $10k)
        max_weight: Maximum weight cap (default 1.0)

    Returns:
        Computed weight (0 to max_weight)
    """
    if notional <= 0:
        return 0.0

    if mode == "equity" and equity is not None and equity > 0:
        # Equity-normalized with sqrt to soften large positions
        # sqrt(position_ratio) means doubling position only adds 41% weight
        position_ratio = notional / equity
        weight = math.sqrt(position_ratio)
        return min(weight, max_weight)

    elif mode == "log" or (mode == "equity" and equity is None):
        # Logarithmic scaling: log(1 + notional/base)
        # With base=$10k: $10k -> 0.69, $100k -> 2.40, $1M -> 4.62
        weight = math.log(1 + notional / log_base)
        # Normalize to roughly 0-1 range (log(101) ≈ 4.62 for $1M)
        normalized = weight / 4.0  # Roughly normalize to 1.0 at ~$500k
        return min(normalized, max_weight)

    else:
        # Legacy linear mode
        weight = notional / log_base
        return min(weight, max_weight)
