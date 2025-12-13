"""
Walk-Forward Replay Engine

Phase 3f: Selection Integrity

Replays trader selection on historical data to validate system performance
without look-ahead bias. Key features:

1. **Universe Freeze**: Uses snapshot table, not current data
2. **Cost-Adjusted Returns**: Includes round-trip costs in performance
3. **Rolling Evaluation**: Period-by-period assessment
4. **Survival Analysis**: Tracks how selected traders performed over time

This enables honest out-of-sample evaluation of the selection algorithm.

@module walkforward
"""

import os
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import asyncpg

from .snapshot import (
    SELECTION_VERSION,
    ROUND_TRIP_COST_BPS,
    load_universe_at_date,
)


# Configuration
REPLAY_LOOKBACK_DAYS = int(os.getenv("REPLAY_LOOKBACK_DAYS", "30"))
REPLAY_EVALUATION_DAYS = int(os.getenv("REPLAY_EVALUATION_DAYS", "7"))


@dataclass
class ReplayPeriod:
    """Results for a single replay period."""
    selection_date: date
    evaluation_start: date
    evaluation_end: date

    # Selection stats
    universe_size: int
    selected_count: int
    fdr_qualified_count: int

    # Performance (R-multiples)
    total_r_gross: float
    total_r_net: float
    avg_r_gross: float
    avg_r_net: float

    # Individual trader results
    trader_results: List[Dict[str, Any]]

    # Survival
    deaths_during_period: int
    censored_during_period: int


@dataclass
class ReplaySummary:
    """Summary of a complete walk-forward replay."""
    start_date: date
    end_date: date
    periods: int

    # Aggregate performance
    cumulative_r_gross: float
    cumulative_r_net: float
    avg_period_r_gross: float
    avg_period_r_net: float

    # Sharpe-like metrics (R-multiple based)
    r_gross_std: float
    r_net_std: float
    sharpe_gross: float
    sharpe_net: float

    # Win rates
    winning_periods: int
    losing_periods: int
    win_rate: float

    # Survival stats
    total_deaths: int
    total_censored: int

    # Per-period details
    period_results: List[ReplayPeriod]


async def get_selected_traders_at_date(
    pool: asyncpg.Pool,
    selection_date: date,
    version: str = SELECTION_VERSION,
) -> List[Dict[str, Any]]:
    """
    Get the traders that were selected on a specific date.

    Uses snapshot table to ensure no look-ahead bias.

    Args:
        pool: Database connection pool
        selection_date: Date to query
        version: Selection version

    Returns:
        List of selected trader records
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                address,
                nig_mu,
                nig_kappa,
                thompson_draw,
                selection_rank,
                avg_r_gross,
                avg_r_net,
                episode_count,
                fdr_qualified
            FROM trader_snapshots
            WHERE snapshot_date = $1
              AND selection_version = $2
              AND is_pool_selected = true
            ORDER BY selection_rank NULLS LAST
            """,
            selection_date,
            version,
        )

    return [dict(row) for row in rows]


async def get_trader_episodes_in_range(
    pool: asyncpg.Pool,
    address: str,
    start_date: date,
    end_date: date,
) -> List[Dict[str, Any]]:
    """
    Get closed episodes for a trader within a date range.

    Only includes episodes that CLOSED within the range (no partial episodes).

    Args:
        pool: Database connection pool
        address: Trader address
        start_date: Start of evaluation period
        end_date: End of evaluation period (exclusive)

    Returns:
        List of episode records with R-multiples
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                id,
                symbol,
                direction,
                entry_price,
                exit_price,
                r_multiple,
                pnl_usd,
                opened_at,
                closed_at,
                atr_at_entry
            FROM position_signals
            WHERE address = $1
              AND closed_at >= $2
              AND closed_at < $3
              AND r_multiple IS NOT NULL
            ORDER BY closed_at
            """,
            address.lower(),
            start_date,
            end_date,
        )

    return [dict(row) for row in rows]


def compute_period_cost_r(episodes: List[Dict[str, Any]]) -> float:
    """
    Compute total round-trip cost as R-multiple for a set of episodes.

    Args:
        episodes: List of episode records

    Returns:
        Total cost in R-multiples
    """
    total_cost_r = 0.0

    for ep in episodes:
        entry_price = float(ep.get("entry_price") or 0)
        atr = float(ep.get("atr_at_entry") or 0)

        if atr > 0 and entry_price > 0:
            # Cost per round-trip in USD
            cost_usd = entry_price * (ROUND_TRIP_COST_BPS / 10000)
            # Express as R-multiple
            cost_r = cost_usd / atr
            total_cost_r += cost_r

    return total_cost_r


async def check_death_during_period(
    pool: asyncpg.Pool,
    address: str,
    start_date: date,
    end_date: date,
) -> Optional[str]:
    """
    Check if trader experienced a death event during the period.

    Args:
        pool: Database connection pool
        address: Trader address
        start_date: Period start
        end_date: Period end

    Returns:
        Death type if occurred, None otherwise
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT death_type
            FROM trader_snapshots
            WHERE address = $1
              AND snapshot_date >= $2
              AND snapshot_date < $3
              AND event_type = 'death'
            ORDER BY snapshot_date
            LIMIT 1
            """,
            address.lower(),
            start_date,
            end_date,
        )

    return row["death_type"] if row else None


async def replay_single_period(
    pool: asyncpg.Pool,
    selection_date: date,
    evaluation_days: int = REPLAY_EVALUATION_DAYS,
    version: str = SELECTION_VERSION,
) -> Optional[ReplayPeriod]:
    """
    Replay a single selection period.

    Process:
    1. Get traders selected on selection_date (from snapshot)
    2. Evaluate their performance in the following evaluation_days
    3. Compute gross and net R-multiples
    4. Track deaths/censors

    Args:
        pool: Database connection pool
        selection_date: Date selection was made
        evaluation_days: Days to evaluate performance
        version: Selection version

    Returns:
        ReplayPeriod with results, or None if no data
    """
    # Get selected traders from snapshot
    selected = await get_selected_traders_at_date(pool, selection_date, version)

    if not selected:
        return None

    # Define evaluation window
    eval_start = selection_date
    eval_end = selection_date + timedelta(days=evaluation_days)

    # Get universe size for context
    async with pool.acquire() as conn:
        universe_size = await conn.fetchval(
            """
            SELECT COUNT(*) FROM trader_snapshots
            WHERE snapshot_date = $1 AND selection_version = $2
            """,
            selection_date,
            version,
        )
        fdr_count = await conn.fetchval(
            """
            SELECT COUNT(*) FROM trader_snapshots
            WHERE snapshot_date = $1
              AND selection_version = $2
              AND fdr_qualified = true
            """,
            selection_date,
            version,
        )

    # Evaluate each trader
    trader_results = []
    total_r_gross = 0.0
    total_r_net = 0.0
    deaths = 0
    censored = 0

    for trader in selected:
        addr = trader["address"]

        # Get episodes that closed during evaluation period
        episodes = await get_trader_episodes_in_range(pool, addr, eval_start, eval_end)

        # Compute gross R
        trader_r_gross = sum(float(ep.get("r_multiple") or 0) for ep in episodes)

        # Compute cost
        cost_r = compute_period_cost_r(episodes)
        trader_r_net = trader_r_gross - cost_r

        # Check for death
        death_type = await check_death_during_period(pool, addr, eval_start, eval_end)
        if death_type:
            deaths += 1

        trader_results.append({
            "address": addr,
            "selection_rank": trader.get("selection_rank"),
            "thompson_draw": trader.get("thompson_draw"),
            "episodes": len(episodes),
            "r_gross": trader_r_gross,
            "r_net": trader_r_net,
            "cost_r": cost_r,
            "death_type": death_type,
        })

        total_r_gross += trader_r_gross
        total_r_net += trader_r_net

    num_traders = len(selected)

    return ReplayPeriod(
        selection_date=selection_date,
        evaluation_start=eval_start,
        evaluation_end=eval_end,
        universe_size=universe_size or 0,
        selected_count=num_traders,
        fdr_qualified_count=fdr_count or 0,
        total_r_gross=total_r_gross,
        total_r_net=total_r_net,
        avg_r_gross=total_r_gross / num_traders if num_traders > 0 else 0,
        avg_r_net=total_r_net / num_traders if num_traders > 0 else 0,
        trader_results=trader_results,
        deaths_during_period=deaths,
        censored_during_period=censored,
    )


async def run_walk_forward_replay(
    pool: asyncpg.Pool,
    start_date: date,
    end_date: date,
    evaluation_days: int = REPLAY_EVALUATION_DAYS,
    version: str = SELECTION_VERSION,
) -> ReplaySummary:
    """
    Run a complete walk-forward replay over a date range.

    For each day with snapshot data:
    1. Load the selected traders from that day's snapshot
    2. Evaluate their performance over the next evaluation_days
    3. Aggregate results

    This provides an honest out-of-sample assessment because:
    - Selection uses only data available at selection_date
    - Performance measured on FUTURE data (not training data)
    - Costs included in net returns

    Args:
        pool: Database connection pool
        start_date: First selection date
        end_date: Last selection date
        evaluation_days: Days to evaluate each selection
        version: Selection version to replay

    Returns:
        ReplaySummary with aggregate and per-period results
    """
    # Find all dates with snapshots in range
    async with pool.acquire() as conn:
        snapshot_dates = await conn.fetch(
            """
            SELECT DISTINCT snapshot_date
            FROM trader_snapshots
            WHERE snapshot_date >= $1
              AND snapshot_date <= $2
              AND selection_version = $3
            ORDER BY snapshot_date
            """,
            start_date,
            end_date,
            version,
        )

    dates = [row["snapshot_date"] for row in snapshot_dates]

    if not dates:
        return ReplaySummary(
            start_date=start_date,
            end_date=end_date,
            periods=0,
            cumulative_r_gross=0,
            cumulative_r_net=0,
            avg_period_r_gross=0,
            avg_period_r_net=0,
            r_gross_std=0,
            r_net_std=0,
            sharpe_gross=0,
            sharpe_net=0,
            winning_periods=0,
            losing_periods=0,
            win_rate=0,
            total_deaths=0,
            total_censored=0,
            period_results=[],
        )

    # Replay each period
    period_results = []
    for selection_date in dates:
        result = await replay_single_period(
            pool, selection_date, evaluation_days, version
        )
        if result:
            period_results.append(result)

    if not period_results:
        return ReplaySummary(
            start_date=start_date,
            end_date=end_date,
            periods=0,
            cumulative_r_gross=0,
            cumulative_r_net=0,
            avg_period_r_gross=0,
            avg_period_r_net=0,
            r_gross_std=0,
            r_net_std=0,
            sharpe_gross=0,
            sharpe_net=0,
            winning_periods=0,
            losing_periods=0,
            win_rate=0,
            total_deaths=0,
            total_censored=0,
            period_results=[],
        )

    # Compute aggregate metrics
    n_periods = len(period_results)
    cumulative_r_gross = sum(p.total_r_gross for p in period_results)
    cumulative_r_net = sum(p.total_r_net for p in period_results)

    avg_r_gross = cumulative_r_gross / n_periods
    avg_r_net = cumulative_r_net / n_periods

    # Compute standard deviations
    gross_values = [p.total_r_gross for p in period_results]
    net_values = [p.total_r_net for p in period_results]

    def std(values: List[float]) -> float:
        if len(values) < 2:
            return 0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return variance ** 0.5

    r_gross_std = std(gross_values)
    r_net_std = std(net_values)

    # Sharpe-like ratio (R-multiple based)
    sharpe_gross = avg_r_gross / r_gross_std if r_gross_std > 0 else 0
    sharpe_net = avg_r_net / r_net_std if r_net_std > 0 else 0

    # Win rate
    winning = sum(1 for p in period_results if p.total_r_net > 0)
    losing = n_periods - winning
    win_rate = winning / n_periods if n_periods > 0 else 0

    # Survival
    total_deaths = sum(p.deaths_during_period for p in period_results)
    total_censored = sum(p.censored_during_period for p in period_results)

    return ReplaySummary(
        start_date=dates[0],
        end_date=dates[-1],
        periods=n_periods,
        cumulative_r_gross=cumulative_r_gross,
        cumulative_r_net=cumulative_r_net,
        avg_period_r_gross=avg_r_gross,
        avg_period_r_net=avg_r_net,
        r_gross_std=r_gross_std,
        r_net_std=r_net_std,
        sharpe_gross=sharpe_gross,
        sharpe_net=sharpe_net,
        winning_periods=winning,
        losing_periods=losing,
        win_rate=win_rate,
        total_deaths=total_deaths,
        total_censored=total_censored,
        period_results=period_results,
    )


def format_replay_summary(summary: ReplaySummary) -> Dict[str, Any]:
    """
    Format replay summary for API response.

    Args:
        summary: ReplaySummary object

    Returns:
        Dict suitable for JSON serialization
    """
    return {
        "start_date": summary.start_date.isoformat(),
        "end_date": summary.end_date.isoformat(),
        "periods": summary.periods,
        "performance": {
            "cumulative_r_gross": round(summary.cumulative_r_gross, 4),
            "cumulative_r_net": round(summary.cumulative_r_net, 4),
            "avg_period_r_gross": round(summary.avg_period_r_gross, 4),
            "avg_period_r_net": round(summary.avg_period_r_net, 4),
            "r_gross_std": round(summary.r_gross_std, 4),
            "r_net_std": round(summary.r_net_std, 4),
            "sharpe_gross": round(summary.sharpe_gross, 4),
            "sharpe_net": round(summary.sharpe_net, 4),
        },
        "win_rate": {
            "winning_periods": summary.winning_periods,
            "losing_periods": summary.losing_periods,
            "rate": round(summary.win_rate, 4),
        },
        "survival": {
            "total_deaths": summary.total_deaths,
            "total_censored": summary.total_censored,
        },
        "periods_detail": [
            {
                "selection_date": p.selection_date.isoformat(),
                "evaluation_start": p.evaluation_start.isoformat(),
                "evaluation_end": p.evaluation_end.isoformat(),
                "universe_size": p.universe_size,
                "selected_count": p.selected_count,
                "fdr_qualified_count": p.fdr_qualified_count,
                "total_r_gross": round(p.total_r_gross, 4),
                "total_r_net": round(p.total_r_net, 4),
                "avg_r_gross": round(p.avg_r_gross, 4),
                "avg_r_net": round(p.avg_r_net, 4),
                "deaths": p.deaths_during_period,
            }
            for p in summary.period_results
        ],
    }
