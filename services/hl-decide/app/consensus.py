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
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

# Configuration
CONSENSUS_MIN_TRADERS = int(os.getenv("CONSENSUS_MIN_TRADERS", "3"))
CONSENSUS_MIN_AGREEING = int(os.getenv("CONSENSUS_MIN_AGREEING", "3"))
CONSENSUS_MIN_PCT = float(os.getenv("CONSENSUS_MIN_PCT", "0.70"))
CONSENSUS_MIN_EFFECTIVE_K = float(os.getenv("CONSENSUS_MIN_EFFECTIVE_K", "2.0"))
CONSENSUS_BASE_WINDOW_S = int(os.getenv("CONSENSUS_BASE_WINDOW_S", "120"))
CONSENSUS_MAX_STALENESS_FACTOR = float(os.getenv("CONSENSUS_MAX_STALENESS_FACTOR", "1.25"))
CONSENSUS_MAX_PRICE_BAND_BPS = float(os.getenv("CONSENSUS_MAX_PRICE_BAND_BPS", "8.0"))
CONSENSUS_EV_MIN_R = float(os.getenv("CONSENSUS_EV_MIN_R", "0.20"))
CONSENSUS_SYMBOLS = os.getenv("CONSENSUS_SYMBOLS", "BTC,ETH").split(",")

# EV calculation defaults
DEFAULT_AVG_WIN_R = float(os.getenv("DEFAULT_AVG_WIN_R", "0.5"))
DEFAULT_AVG_LOSS_R = float(os.getenv("DEFAULT_AVG_LOSS_R", "0.3"))
DEFAULT_FEES_BPS = float(os.getenv("DEFAULT_FEES_BPS", "7.0"))  # Round-trip HL fees
DEFAULT_SLIP_BPS = float(os.getenv("DEFAULT_SLIP_BPS", "10.0"))  # Expected slippage

# Default correlation (used when pairwise not computed)
DEFAULT_CORRELATION = float(os.getenv("DEFAULT_CORRELATION", "0.3"))

# Weight cap for individual traders
WEIGHT_CAP = float(os.getenv("CONSENSUS_WEIGHT_CAP", "1.0"))


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
    weight: float  # Based on position size
    price: float  # Entry price (for price band check)
    ts: datetime  # Timestamp of vote


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
    4. Price band gate: Reject if market moved too far
    5. EV gate: Require positive expected value after costs
    """

    def __init__(self):
        self.windows: Dict[str, ConsensusWindow] = {}
        self.correlation_matrix: Dict[Tuple[str, str], float] = {}
        self.current_prices: Dict[str, float] = {}

    def set_current_price(self, symbol: str, price: float) -> None:
        """Update current mid price for a symbol."""
        self.current_prices[symbol] = price

    def get_current_mid(self, symbol: str) -> float:
        """Get current mid price for a symbol."""
        return self.current_prices.get(symbol, 0.0)

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

        # Calculate stop price (simple ATR-based for now)
        # TODO: Use actual ATR when available
        stop_distance = median_entry * 0.01  # 1% as placeholder
        if majority_dir == "long":
            stop_price = median_entry - stop_distance
        else:
            stop_price = median_entry + stop_distance

        # Gate 4: EV after costs
        p_win = self.calibrated_p_win(agreeing_votes, eff_k)
        ev_result = calculate_ev(
            p_win=p_win,
            entry_px=median_entry,
            stop_px=stop_price,
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

    def collapse_to_votes(self, fills: List[Fill]) -> List[Vote]:
        """
        Collapse multiple fills per trader to one vote each.

        One trader = one vote, based on net position change.
        Weight by |Δexposure| (capped).

        Args:
            fills: List of fills in the window

        Returns:
            List of votes (one per trader)
        """
        by_trader: Dict[str, List[Fill]] = defaultdict(list)
        for f in fills:
            by_trader[f.address.lower()].append(f)

        votes = []
        for addr, trader_fills in by_trader.items():
            net_delta = sum(f.signed_size for f in trader_fills)
            if abs(net_delta) < 1e-9:
                continue  # No net position change

            direction = "long" if net_delta > 0 else "short"
            weight = min(abs(net_delta) / WEIGHT_CAP, 1.0)

            # Use weighted average price
            total_size = sum(abs(f.size) for f in trader_fills)
            if total_size > 0:
                avg_price = sum(f.price * abs(f.size) for f in trader_fills) / total_size
            else:
                avg_price = trader_fills[-1].price if trader_fills else 0.0

            # Use latest timestamp
            latest_ts = max(f.ts for f in trader_fills)

            votes.append(Vote(
                address=addr,
                direction=direction,
                weight=weight,
                price=avg_price,
                ts=latest_ts,
            ))

        return votes

    def eff_k_from_corr(self, weights: Dict[str, float]) -> float:
        """
        Calculate effective K using correlation matrix.

        Formula: effK = (Σᵢ wᵢ)² / Σᵢ Σⱼ wᵢ wⱼ ρᵢⱼ

        Args:
            weights: Dict mapping address to weight

        Returns:
            Effective number of independent traders
        """
        addrs = list(weights.keys())
        if len(addrs) <= 1:
            return float(len(addrs))

        num = sum(weights.values()) ** 2
        den = 0.0

        for i, a in enumerate(addrs):
            for j, b in enumerate(addrs):
                if i == j:
                    rho = 1.0
                else:
                    key = tuple(sorted([a, b]))
                    rho = self.correlation_matrix.get(key, DEFAULT_CORRELATION)
                    rho = max(0.0, min(1.0, rho))  # Clip negative correlations

                den += weights[a] * weights[b] * rho

        return num / max(den, 1e-9)

    def passes_latency_and_price_gates(
        self,
        window: ConsensusWindow,
        votes: List[Vote],
    ) -> bool:
        """
        Check latency and price band gates.

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

        # Price band gate: current mid vs median voter entry
        median_entry = statistics.median(v.price for v in votes)
        mid_price = self.get_current_mid(window.symbol)

        if median_entry <= 0 or mid_price <= 0:
            return False

        bps_deviation = abs(mid_price - median_entry) / median_entry * 10000

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

    Returns:
        Dict with ev_gross_r, ev_cost_r, ev_net_r
    """
    gross_ev = p_win * avg_win_r - (1 - p_win) * avg_loss_r
    total_bps = fees_bps + slip_bps
    cost_r = bps_to_R(entry_px, stop_px, total_bps)
    net_ev = gross_ev - cost_r

    return {
        "ev_gross_r": gross_ev,
        "ev_cost_r": cost_r,
        "ev_net_r": net_ev,
    }
