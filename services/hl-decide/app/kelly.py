"""
Kelly Criterion Position Sizing

Phase 4.1: Selection Integrity

Implements Kelly criterion with fractional sizing for conservative risk management.
The Kelly formula optimizes for geometric growth rate of capital.

Full Kelly: f* = p - (1-p)/R where:
- p = probability of winning
- R = avg_win / avg_loss (reward-to-risk ratio)
- f* = optimal fraction of capital to bet

We use Fractional Kelly (25% by default) to reduce variance at cost of lower
expected growth. This is appropriate for trading where:
1. Win rate/edge estimates have uncertainty
2. Drawdown tolerance is limited
3. Psychological factors matter

@module kelly
"""

import os
from dataclasses import dataclass
from typing import Optional


# Configuration defaults
KELLY_ENABLED = os.getenv("KELLY_ENABLED", "false").lower() == "true"
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.25"))  # 25% Kelly
KELLY_MIN_EPISODES = int(os.getenv("KELLY_MIN_EPISODES", "30"))
KELLY_FALLBACK_PCT = float(os.getenv("KELLY_FALLBACK_PCT", "0.01"))  # 1%

# Hard limits - Kelly cannot exceed these
KELLY_MAX_FRACTION = 0.50  # Never bet more than 50% of optimal
KELLY_MAX_POSITION_PCT = 0.10  # Cap at 10% of equity (matches risk_governor)


@dataclass
class KellyInput:
    """Input parameters for Kelly calculation."""

    win_rate: float  # p: probability of winning (0-1)
    avg_win_r: float  # Average win in R-multiples (positive)
    avg_loss_r: float  # Average loss in R-multiples (positive, abs value)
    episode_count: int  # Sample size for reliability
    account_value: float  # Current equity in USD
    current_price: float  # Asset price for size calculation
    stop_distance_pct: float  # Stop distance as fraction (0.01 = 1%)
    round_trip_fee_pct: float = 0.001  # Round-trip fees as fraction (0.001 = 10 bps)


@dataclass
class KellyResult:
    """Result of Kelly position sizing calculation."""

    full_kelly: float  # Raw Kelly fraction (0-1)
    fractional_kelly: float  # Conservative fraction after scaling
    position_pct: float  # Final position size as % of equity
    position_size_usd: float  # Dollar size
    position_size_coin: float  # Asset quantity
    method: str  # 'kelly', 'fallback_insufficient_data', 'fallback_negative_ev'
    reasoning: str  # Human-readable explanation
    capped: bool  # Whether size was capped by hard limits


def calculate_kelly_fraction(
    win_rate: float,
    avg_win_r: float,
    avg_loss_r: float,
) -> float:
    """
    Calculate full Kelly fraction.

    The Kelly criterion formula for unequal win/loss sizes:
    f* = p - (1-p)/R

    Where:
    - p = probability of winning
    - R = avg_win / avg_loss (reward-to-risk ratio)
    - f* = optimal fraction of bankroll to bet

    Args:
        win_rate: Probability of winning (0-1)
        avg_win_r: Average win in R-multiples (positive)
        avg_loss_r: Average loss in R-multiples (positive, will be abs'd)

    Returns:
        Kelly fraction clamped to [0, 1]
        Returns 0 if expected value is negative (don't bet)
    """
    # Validate inputs
    if not 0 <= win_rate <= 1:
        return 0.0

    avg_loss_r = abs(avg_loss_r)  # Ensure positive

    # Avoid division by zero
    if avg_loss_r <= 0:
        return 0.0

    # Calculate R ratio (reward-to-risk)
    R = avg_win_r / avg_loss_r

    # Kelly formula
    kelly = win_rate - (1 - win_rate) / R

    # Clamp to [0, 1]
    return max(0.0, min(kelly, 1.0))


def calculate_expected_value(
    win_rate: float,
    avg_win_r: float,
    avg_loss_r: float,
    fee_cost_r: float = 0.0,
) -> float:
    """
    Calculate expected value per trade in R-multiples.

    EV = (win_rate * avg_win) - ((1 - win_rate) * avg_loss) - fee_cost

    Fee cost is subtracted because every trade incurs fees regardless of outcome.
    This makes higher-fee exchanges less favorable for the same edge.

    Args:
        win_rate: Probability of winning (0-1)
        avg_win_r: Average win in R-multiples
        avg_loss_r: Average loss in R-multiples (positive)
        fee_cost_r: Round-trip fee cost in R-multiples (applied every trade)

    Returns:
        Expected value per trade in R-multiples (fee-adjusted)
    """
    avg_loss_r = abs(avg_loss_r)
    raw_ev = (win_rate * avg_win_r) - ((1 - win_rate) * avg_loss_r)
    return raw_ev - fee_cost_r


def kelly_position_size(
    kelly_input: KellyInput,
    fraction: float = KELLY_FRACTION,
    min_episodes: int = KELLY_MIN_EPISODES,
    fallback_pct: float = KELLY_FALLBACK_PCT,
    max_position_pct: float = KELLY_MAX_POSITION_PCT,
) -> KellyResult:
    """
    Calculate position size using fractional Kelly criterion.

    Applies fractional Kelly for conservative sizing, with fallbacks for:
    - Insufficient data (fewer than min_episodes)
    - Negative expected value
    - Edge cases (0% or 100% win rate)

    The final size is capped by max_position_pct to respect risk limits.

    Args:
        kelly_input: Input parameters for calculation
        fraction: Fractional Kelly multiplier (0.25 = quarter Kelly)
        min_episodes: Minimum episodes for Kelly to be valid
        fallback_pct: Fallback position size as % of equity
        max_position_pct: Hard cap on position size as % of equity

    Returns:
        KellyResult with sizing details and reasoning
    """
    account_value = kelly_input.account_value
    current_price = kelly_input.current_price

    # Handle invalid price
    if current_price <= 0:
        return KellyResult(
            full_kelly=0.0,
            fractional_kelly=0.0,
            position_pct=0.0,
            position_size_usd=0.0,
            position_size_coin=0.0,
            method="error",
            reasoning="Invalid price (<=0)",
            capped=False,
        )

    # Check for insufficient data
    if kelly_input.episode_count < min_episodes:
        position_pct = fallback_pct
        size_usd = account_value * position_pct
        return KellyResult(
            full_kelly=0.0,
            fractional_kelly=0.0,
            position_pct=position_pct,
            position_size_usd=size_usd,
            position_size_coin=size_usd / current_price,
            method="fallback_insufficient_data",
            reasoning=f"Only {kelly_input.episode_count} episodes, need {min_episodes}",
            capped=False,
        )

    # Convert round-trip fees to R-multiples
    # If stop is 1%, then 10bps round-trip = 0.1R fee drag
    fee_cost_r = 0.0
    if kelly_input.stop_distance_pct > 0 and kelly_input.round_trip_fee_pct > 0:
        fee_cost_r = kelly_input.round_trip_fee_pct / kelly_input.stop_distance_pct

    # Calculate expected value with fee adjustment
    ev = calculate_expected_value(
        kelly_input.win_rate,
        kelly_input.avg_win_r,
        kelly_input.avg_loss_r,
        fee_cost_r,
    )

    # If EV is negative, don't trade (or use minimal size for learning)
    if ev <= 0:
        position_pct = fallback_pct * 0.5  # Half fallback for negative EV
        size_usd = account_value * position_pct
        fee_msg = f" (incl {fee_cost_r:.3f}R fees)" if fee_cost_r > 0 else ""
        return KellyResult(
            full_kelly=0.0,
            fractional_kelly=0.0,
            position_pct=position_pct,
            position_size_usd=size_usd,
            position_size_coin=size_usd / current_price,
            method="fallback_negative_ev",
            reasoning=f"Negative EV: {ev:.3f}R per trade{fee_msg}",
            capped=False,
        )

    # Calculate full Kelly
    full_kelly = calculate_kelly_fraction(
        kelly_input.win_rate,
        kelly_input.avg_win_r,
        kelly_input.avg_loss_r,
    )

    # Apply fractional Kelly
    fractional_kelly = full_kelly * fraction

    # Cap fractional Kelly at maximum allowed
    fractional_kelly = min(fractional_kelly, KELLY_MAX_FRACTION)

    # Convert to position percentage
    # Kelly fraction is % of bankroll to risk
    # With stop_distance_pct, we size so that loss = kelly_fraction * account
    # position_size = (kelly_fraction * account) / stop_distance_pct
    if kelly_input.stop_distance_pct > 0:
        position_pct = fractional_kelly / kelly_input.stop_distance_pct
    else:
        # Fallback: treat Kelly directly as position %
        position_pct = fractional_kelly

    # Check if capped
    capped = position_pct > max_position_pct
    position_pct = min(position_pct, max_position_pct)

    # Calculate USD and coin sizes
    size_usd = account_value * position_pct
    size_coin = size_usd / current_price

    # Build reasoning with fee info if applicable
    fee_msg = f", Fees={fee_cost_r:.2f}R" if fee_cost_r > 0 else ""
    reasoning = (
        f"Kelly={full_kelly:.1%}, Fractional={fractional_kelly:.1%}, "
        f"EV={ev:.3f}R, Win={kelly_input.win_rate:.1%}{fee_msg}"
    )

    return KellyResult(
        full_kelly=full_kelly,
        fractional_kelly=fractional_kelly,
        position_pct=position_pct,
        position_size_usd=size_usd,
        position_size_coin=size_coin,
        method="kelly",
        reasoning=reasoning,
        capped=capped,
    )


async def get_kelly_input_from_db(
    db,
    address: str,
    account_value: float,
    current_price: float,
    stop_distance_pct: float,
    round_trip_fee_pct: float = 0.001,
) -> Optional[KellyInput]:
    """
    Fetch trader performance data from database for Kelly calculation.

    Args:
        db: Database connection
        address: Trader address
        account_value: Current account equity
        current_price: Current asset price
        stop_distance_pct: Stop distance as fraction
        round_trip_fee_pct: Round-trip fees as fraction (default 10 bps)

    Returns:
        KellyInput if data found, None otherwise
    """
    query = """
        SELECT
            COALESCE(episode_count, 0) as episode_count,
            COALESCE(win_rate, 0.5) as win_rate,
            COALESCE(avg_r, 0) as avg_r,
            COALESCE(avg_win_r, 0) as avg_win_r,
            COALESCE(avg_loss_r, 1) as avg_loss_r
        FROM trader_performance
        WHERE address = $1
    """
    row = await db.fetchrow(query, address.lower())

    if not row:
        return None

    return KellyInput(
        win_rate=float(row["win_rate"]),
        avg_win_r=float(row["avg_win_r"]) if row["avg_win_r"] else 0.5,
        avg_loss_r=abs(float(row["avg_loss_r"])) if row["avg_loss_r"] else 1.0,
        episode_count=int(row["episode_count"]),
        account_value=account_value,
        current_price=current_price,
        stop_distance_pct=stop_distance_pct,
        round_trip_fee_pct=round_trip_fee_pct,
    )


async def get_consensus_kelly_size(
    db,
    addresses: list[str],
    account_value: float,
    current_price: float,
    stop_distance_pct: float,
    fraction: float = KELLY_FRACTION,
    round_trip_fee_pct: float = 0.001,
) -> KellyResult:
    """
    Calculate Kelly-based position size for a consensus signal.

    When multiple traders agree, we can aggregate their statistics
    for a potentially more robust Kelly estimate.

    Strategy: Use median Kelly across agreeing traders.

    Args:
        db: Database connection
        addresses: List of trader addresses in consensus
        account_value: Current account equity
        current_price: Current asset price
        stop_distance_pct: Stop distance as fraction
        fraction: Fractional Kelly multiplier
        round_trip_fee_pct: Round-trip fees as fraction (passed to Kelly)

    Returns:
        KellyResult with aggregated sizing
    """
    kelly_results = []

    for address in addresses:
        kelly_input = await get_kelly_input_from_db(
            db, address, account_value, current_price, stop_distance_pct,
            round_trip_fee_pct=round_trip_fee_pct,
        )
        if kelly_input and kelly_input.episode_count >= KELLY_MIN_EPISODES:
            result = kelly_position_size(kelly_input, fraction=fraction)
            if result.method == "kelly":
                kelly_results.append(result)

    # If no valid Kelly calculations, use fallback
    if not kelly_results:
        fallback_pct = KELLY_FALLBACK_PCT
        size_usd = account_value * fallback_pct
        return KellyResult(
            full_kelly=0.0,
            fractional_kelly=0.0,
            position_pct=fallback_pct,
            position_size_usd=size_usd,
            position_size_coin=size_usd / current_price if current_price > 0 else 0,
            method="fallback_no_kelly_traders",
            reasoning=f"No traders with {KELLY_MIN_EPISODES}+ episodes",
            capped=False,
        )

    # Use median position size for robustness
    kelly_results.sort(key=lambda r: r.position_pct)
    median_idx = len(kelly_results) // 2
    median_result = kelly_results[median_idx]

    # Adjust reasoning
    return KellyResult(
        full_kelly=median_result.full_kelly,
        fractional_kelly=median_result.fractional_kelly,
        position_pct=median_result.position_pct,
        position_size_usd=median_result.position_size_usd,
        position_size_coin=median_result.position_size_coin,
        method="kelly_consensus",
        reasoning=f"Median of {len(kelly_results)} traders: {median_result.reasoning}",
        capped=median_result.capped,
    )
