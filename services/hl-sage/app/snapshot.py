"""
Shadow Ledger: Daily Trader Snapshots

Phase 3f: Selection Integrity

This module captures daily snapshots of all traders for survivorship-bias-free
analysis. Key features:

1. **Multi-universe membership**: Track which stage each trader reached
2. **Thompson sampling storage**: Store actual draws for reproducibility
3. **Death/censor detection**: Distinguish terminal vs non-terminal exits
4. **Gross/net R-multiples**: Separate pre/post-cost performance

The Shadow Ledger enables:
- Survival analysis (who blew up and when)
- Walk-forward validation (replay selection with as-of data)
- FDR control (statistical significance of trader skill)

@module snapshot
"""

import asyncio
import math
import os
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
import scipy.stats

from .bandit import (
    NIG_PRIOR_ALPHA,
    NIG_PRIOR_BETA,
    NIG_PRIOR_KAPPA,
    NIG_PRIOR_M,
    R_WINSORIZE_MAX,
    R_WINSORIZE_MIN,
    TraderPosteriorNIG,
)


# Configuration
SELECTION_VERSION = os.getenv("SELECTION_VERSION", "3f.1")
SNAPSHOT_MIN_EPISODES = int(os.getenv("SNAPSHOT_MIN_EPISODES", "30"))
SNAPSHOT_FDR_ALPHA = float(os.getenv("SNAPSHOT_FDR_ALPHA", "0.10"))
SNAPSHOT_MIN_AVG_R_NET = float(os.getenv("SNAPSHOT_MIN_AVG_R_NET", "0.05"))

# Cost estimation for net R calculation
# Conservative: 10bps spread + 5bps slippage each way = 30bps total round-trip
ROUND_TRIP_COST_BPS = float(os.getenv("ROUND_TRIP_COST_BPS", "30"))

# Death thresholds
DEATH_DRAWDOWN_PCT = float(os.getenv("DEATH_DRAWDOWN_PCT", "0.80"))  # 80% drawdown = death
DEATH_ACCOUNT_FLOOR = float(os.getenv("DEATH_ACCOUNT_FLOOR", "10000"))  # $10k floor

# Censor thresholds
CENSOR_INACTIVE_DAYS = int(os.getenv("CENSOR_INACTIVE_DAYS", "30"))


@dataclass
class TraderSnapshot:
    """A point-in-time snapshot of a trader's state."""
    address: str
    snapshot_date: date
    selection_version: str

    # Universe membership
    is_leaderboard_scanned: bool = False
    is_candidate_filtered: bool = False
    is_quality_qualified: bool = False
    is_pool_selected: bool = False
    is_pinned_custom: bool = False

    # Features
    account_value: Optional[float] = None
    peak_account_value: Optional[float] = None
    pnl_30d: Optional[float] = None
    roi_30d: Optional[float] = None
    win_rate: Optional[float] = None
    episode_count: int = 0
    week_volume: Optional[float] = None
    orders_per_day: Optional[float] = None

    # R-multiple stats
    avg_r_gross: Optional[float] = None
    avg_r_net: Optional[float] = None

    # NIG posterior
    nig_mu: float = NIG_PRIOR_M
    nig_kappa: float = NIG_PRIOR_KAPPA
    nig_alpha: float = NIG_PRIOR_ALPHA
    nig_beta: float = NIG_PRIOR_BETA

    # Thompson sampling
    thompson_draw: Optional[float] = None
    thompson_seed: Optional[int] = None
    selection_rank: Optional[int] = None

    # FDR qualification
    skill_p_value: Optional[float] = None
    fdr_qualified: bool = False

    # Lifecycle
    event_type: str = "active"
    death_type: Optional[str] = None
    censor_type: Optional[str] = None


def thompson_sample_nig(
    m: float,
    kappa: float,
    alpha: float,
    beta: float,
    seed: int,
) -> float:
    """
    Sample mu from NIG posterior for Thompson selection.

    Uses provided seed for reproducibility.

    Args:
        m: Posterior mean
        kappa: Posterior precision scaling
        alpha: Posterior shape
        beta: Posterior rate
        seed: RNG seed for reproducibility

    Returns:
        Sampled mu value
    """
    rng = random.Random(seed)

    # Sample sigma^2 from InverseGamma(alpha, beta)
    # If X ~ Gamma(alpha, 1/beta), then 1/X ~ InverseGamma(alpha, beta)
    gamma_sample = rng.gammavariate(alpha, 1.0 / beta) if beta > 0 else 1.0
    sigma2 = 1.0 / gamma_sample if gamma_sample > 0 else 1.0

    # Sample mu from N(m, sigma^2 / kappa)
    std = math.sqrt(sigma2 / kappa) if kappa > 0 else 1.0
    mu = rng.gauss(m, std)

    return mu


def compute_skill_p_value(r_values: List[float]) -> Optional[float]:
    """
    Compute p-value for H0: mean_r <= 0 using one-sided t-test.

    R-values are winsorized before testing to handle heavy tails.

    Args:
        r_values: List of R-multiples from closed episodes

    Returns:
        One-sided p-value, or None if insufficient data
    """
    if len(r_values) < SNAPSHOT_MIN_EPISODES:
        return None

    # Winsorize to configured bounds
    winsorized = [
        max(R_WINSORIZE_MIN, min(R_WINSORIZE_MAX, r))
        for r in r_values
    ]

    # One-sided t-test: H1: mean > 0
    t_stat, p_two_sided = scipy.stats.ttest_1samp(winsorized, 0)

    # Convert to one-sided
    if t_stat > 0:
        p_one_sided = p_two_sided / 2
    else:
        p_one_sided = 1 - p_two_sided / 2

    return p_one_sided


def benjamini_hochberg_select(
    traders_with_pvalues: List[Tuple[str, float]],
    alpha: float = SNAPSHOT_FDR_ALPHA,
) -> List[str]:
    """
    Select traders using Benjamini-Hochberg FDR control.

    IMPORTANT: BH finds k* = max{i : p_i <= (i/n)*alpha}, then selects all i <= k*.
    It does NOT stop at first failure.

    Args:
        traders_with_pvalues: List of (address, p_value) tuples
        alpha: FDR level (default 0.10)

    Returns:
        List of addresses that pass FDR control
    """
    if not traders_with_pvalues:
        return []

    # Sort by p-value ascending
    sorted_pvals = sorted(traders_with_pvalues, key=lambda x: x[1])
    n = len(sorted_pvals)

    # Find k* = max{i : p_i <= (i/n)*alpha}
    k_star = 0
    for i, (addr, p) in enumerate(sorted_pvals, 1):
        if p <= (i / n) * alpha:
            k_star = i

    # Select all traders with rank <= k_star
    qualified = [addr for addr, _ in sorted_pvals[:k_star]]

    return qualified


def estimate_cost_r(avg_atr: float, avg_price: float) -> float:
    """
    Estimate round-trip cost as R-multiple.

    Args:
        avg_atr: Average ATR for the traded assets
        avg_price: Average entry price

    Returns:
        Cost expressed as R-multiple
    """
    if avg_atr <= 0 or avg_price <= 0:
        return 0.0

    # Cost in USD
    cost_usd = avg_price * (ROUND_TRIP_COST_BPS / 10000)

    # Express as R-multiple (cost / risk)
    r_cost = cost_usd / avg_atr

    return r_cost


async def detect_death_events(
    conn: asyncpg.Connection,
    address: str,
    current_value: float,
    peak_value: float,
) -> Optional[str]:
    """
    Detect if trader has experienced a death event.

    Death types (terminal):
    - liquidation: Account liquidated on Hyperliquid
    - drawdown_80: Current equity < 20% of peak
    - account_value_floor: Account dropped below $10k
    - negative_equity: Account value <= 0

    Args:
        conn: Database connection
        address: Trader address
        current_value: Current account value
        peak_value: Peak account value

    Returns:
        Death type string or None
    """
    # Check for negative equity first (most severe)
    if current_value <= 0:
        return "negative_equity"

    # Check for account floor
    if current_value < DEATH_ACCOUNT_FLOOR:
        return "account_value_floor"

    # Check for 80% drawdown
    if peak_value > 0 and current_value < (1 - DEATH_DRAWDOWN_PCT) * peak_value:
        return "drawdown_80"

    # Check for liquidation events
    liquidation = await conn.fetchval(
        """
        SELECT 1 FROM hl_events
        WHERE address = $1
          AND type = 'liquidation'
          AND at > NOW() - INTERVAL '1 day'
        LIMIT 1
        """,
        address.lower(),
    )
    if liquidation:
        return "liquidation"

    return None


async def detect_censor_events(
    conn: asyncpg.Connection,
    address: str,
    snapshot_date: date,
) -> Optional[str]:
    """
    Detect if trader has experienced a censor event (non-terminal).

    Censor types:
    - inactive_30d: No fills for 30 days
    - stopped_btc_eth: Only trading other assets
    - api_unavailable: HL API returns no data (detected elsewhere)

    Args:
        conn: Database connection
        address: Trader address
        snapshot_date: Date of snapshot

    Returns:
        Censor type string or None
    """
    cutoff = snapshot_date - timedelta(days=CENSOR_INACTIVE_DAYS)

    # Check for any recent fills
    any_fills = await conn.fetchval(
        """
        SELECT 1 FROM hl_events
        WHERE address = $1
          AND type = 'trade'
          AND at > $2
        LIMIT 1
        """,
        address.lower(),
        cutoff,
    )

    if not any_fills:
        return "inactive_30d"

    # Check if they stopped trading BTC/ETH specifically
    btc_eth_fills = await conn.fetchval(
        """
        SELECT 1 FROM hl_events
        WHERE address = $1
          AND type = 'trade'
          AND symbol IN ('BTC', 'ETH')
          AND at > $2
        LIMIT 1
        """,
        address.lower(),
        cutoff,
    )

    if not btc_eth_fills:
        return "stopped_btc_eth"

    return None


async def get_trader_r_values(
    conn: asyncpg.Connection,
    address: str,
    as_of_date: date,
) -> List[float]:
    """
    Get R-multiples for a trader's closed episodes.

    Only includes episodes closed before as_of_date (no look-ahead).

    Args:
        conn: Database connection
        address: Trader address
        as_of_date: Only include episodes closed before this date

    Returns:
        List of R-multiples
    """
    rows = await conn.fetch(
        """
        SELECT r_clamped
        FROM position_signals
        WHERE address = $1
          AND status = 'closed'
          AND exit_ts IS NOT NULL
          AND exit_ts < $2
          AND r_clamped IS NOT NULL
        ORDER BY exit_ts
        """,
        address.lower(),
        as_of_date,
    )

    return [float(row["r_clamped"]) for row in rows]


async def create_daily_snapshot(
    pool: asyncpg.Pool,
    snapshot_date: Optional[date] = None,
) -> Dict[str, Any]:
    """
    Create daily snapshot of all traders in the Shadow Ledger.

    This captures the state of every trader for survivorship-bias-free analysis:
    1. Scans all traders (leaderboard + pool + historical)
    2. Computes features and NIG posteriors as-of snapshot_date
    3. Performs Thompson sampling with stored seeds
    4. Runs FDR qualification
    5. Detects death/censor events

    Args:
        pool: Database connection pool
        snapshot_date: Date for snapshot (default: today)

    Returns:
        Summary of snapshot creation
    """
    if snapshot_date is None:
        snapshot_date = date.today()

    # Generate a deterministic seed from the date for reproducibility
    date_seed = int(snapshot_date.strftime("%Y%m%d"))

    async with pool.acquire() as conn:
        # Get all traders to snapshot:
        # 1. Active Alpha Pool addresses
        # 2. Previously tracked addresses (may have died/censored)
        # 3. Pinned accounts

        active_pool = await conn.fetch(
            """
            SELECT address, account_value, pnl_30d, roi_30d, win_rate
            FROM alpha_pool_addresses
            WHERE is_active = true
            """
        )

        historical = await conn.fetch(
            """
            SELECT DISTINCT address
            FROM trader_snapshots
            WHERE snapshot_date < $1
              AND event_type NOT IN ('death')
            """,
            snapshot_date,
        )

        pinned = await conn.fetch(
            """
            SELECT address
            FROM hl_pinned_accounts
            WHERE pinned_at IS NOT NULL
            """
        )

        # Combine all addresses
        all_addresses = set()
        pool_addresses = {row["address"].lower() for row in active_pool}
        all_addresses.update(pool_addresses)
        all_addresses.update(row["address"].lower() for row in historical)
        all_addresses.update(row["address"].lower() for row in pinned)

        # Build lookup for pool data
        pool_data = {
            row["address"].lower(): {
                "account_value": float(row["account_value"] or 0),
                "pnl_30d": float(row["pnl_30d"] or 0),
                "roi_30d": float(row["roi_30d"] or 0),
                "win_rate": float(row["win_rate"] or 0),
            }
            for row in active_pool
        }

        pinned_addresses = {row["address"].lower() for row in pinned}

        snapshots: List[TraderSnapshot] = []
        traders_with_pvalues: List[Tuple[str, float]] = []

        # Process each trader
        for addr in all_addresses:
            # Get NIG posterior from trader_performance
            perf = await conn.fetchrow(
                """
                SELECT nig_m, nig_kappa, nig_alpha, nig_beta,
                       total_signals, avg_r, total_pnl_r
                FROM trader_performance
                WHERE address = $1
                """,
                addr,
            )

            # Get peak account value for drawdown calculation
            peak_row = await conn.fetchrow(
                """
                SELECT MAX(account_value) as peak
                FROM trader_snapshots
                WHERE address = $1
                """,
                addr,
            )
            peak_value = float(peak_row["peak"]) if peak_row and peak_row["peak"] else 0

            # Get current account value
            current_value = pool_data.get(addr, {}).get("account_value", 0)
            if current_value > peak_value:
                peak_value = current_value

            # Get R-values for FDR testing
            r_values = await get_trader_r_values(conn, addr, snapshot_date)
            episode_count = len(r_values)

            # Compute gross and net R-multiples
            avg_r_gross = sum(r_values) / len(r_values) if r_values else None

            # Estimate cost per trade (simplified: assume BTC avg price $50k, ATR $1000)
            # In production, this should use actual ATR data
            cost_r = estimate_cost_r(avg_atr=1000, avg_price=50000)
            avg_r_net = (avg_r_gross - cost_r) if avg_r_gross is not None else None

            # Build snapshot
            snapshot = TraderSnapshot(
                address=addr,
                snapshot_date=snapshot_date,
                selection_version=SELECTION_VERSION,

                # Universe membership
                is_leaderboard_scanned=addr in pool_addresses,
                is_candidate_filtered=addr in pool_addresses,  # Simplified
                is_quality_qualified=addr in pool_addresses and episode_count >= SNAPSHOT_MIN_EPISODES,
                is_pool_selected=False,  # Set after Thompson sampling
                is_pinned_custom=addr in pinned_addresses,

                # Features
                account_value=current_value,
                peak_account_value=peak_value,
                pnl_30d=pool_data.get(addr, {}).get("pnl_30d"),
                roi_30d=pool_data.get(addr, {}).get("roi_30d"),
                win_rate=pool_data.get(addr, {}).get("win_rate"),
                episode_count=episode_count,

                # R-multiple stats
                avg_r_gross=avg_r_gross,
                avg_r_net=avg_r_net,

                # NIG posterior
                nig_mu=float(perf["nig_m"]) if perf and perf["nig_m"] else NIG_PRIOR_M,
                nig_kappa=float(perf["nig_kappa"]) if perf and perf["nig_kappa"] else NIG_PRIOR_KAPPA,
                nig_alpha=float(perf["nig_alpha"]) if perf and perf["nig_alpha"] else NIG_PRIOR_ALPHA,
                nig_beta=float(perf["nig_beta"]) if perf and perf["nig_beta"] else NIG_PRIOR_BETA,
            )

            # Thompson sampling with reproducible seed
            trader_seed = date_seed + hash(addr) % 1000000
            snapshot.thompson_seed = trader_seed
            snapshot.thompson_draw = thompson_sample_nig(
                snapshot.nig_mu,
                snapshot.nig_kappa,
                snapshot.nig_alpha,
                snapshot.nig_beta,
                trader_seed,
            )

            # Compute skill p-value for FDR
            if episode_count >= SNAPSHOT_MIN_EPISODES:
                p_value = compute_skill_p_value(r_values)
                if p_value is not None:
                    snapshot.skill_p_value = p_value
                    traders_with_pvalues.append((addr, p_value))

            # Detect death events
            death_type = await detect_death_events(conn, addr, current_value, peak_value)
            if death_type:
                snapshot.event_type = "death"
                snapshot.death_type = death_type
            else:
                # Detect censor events
                censor_type = await detect_censor_events(conn, addr, snapshot_date)
                if censor_type:
                    snapshot.event_type = "censored"
                    snapshot.censor_type = censor_type

            snapshots.append(snapshot)

        # Run FDR qualification
        fdr_qualified = set(benjamini_hochberg_select(traders_with_pvalues))
        for snapshot in snapshots:
            snapshot.fdr_qualified = snapshot.address in fdr_qualified

        # Apply effect size gate and determine final selection
        qualified_for_selection = [
            s for s in snapshots
            if s.is_quality_qualified
            and s.fdr_qualified
            and s.avg_r_net is not None
            and s.avg_r_net >= SNAPSHOT_MIN_AVG_R_NET
            and s.event_type not in ("death", "censored")
        ]

        # Rank by Thompson draw and select top 50
        qualified_for_selection.sort(key=lambda s: s.thompson_draw or 0, reverse=True)
        for rank, snapshot in enumerate(qualified_for_selection[:50], 1):
            snapshot.is_pool_selected = True
            snapshot.selection_rank = rank

        # Persist snapshots to database
        inserted = 0
        for snapshot in snapshots:
            try:
                await conn.execute(
                    """
                    INSERT INTO trader_snapshots (
                        snapshot_date, address, selection_version,
                        is_leaderboard_scanned, is_candidate_filtered, is_quality_qualified,
                        is_pool_selected, is_pinned_custom,
                        account_value, peak_account_value, pnl_30d, roi_30d, win_rate,
                        episode_count, week_volume, orders_per_day,
                        avg_r_gross, avg_r_net,
                        nig_mu, nig_kappa, nig_alpha, nig_beta,
                        thompson_draw, thompson_seed, selection_rank,
                        skill_p_value, fdr_qualified,
                        event_type, death_type, censor_type
                    ) VALUES (
                        $1, $2, $3,
                        $4, $5, $6, $7, $8,
                        $9, $10, $11, $12, $13,
                        $14, $15, $16,
                        $17, $18,
                        $19, $20, $21, $22,
                        $23, $24, $25,
                        $26, $27,
                        $28, $29, $30
                    )
                    ON CONFLICT (snapshot_date, address, selection_version) DO UPDATE SET
                        is_leaderboard_scanned = EXCLUDED.is_leaderboard_scanned,
                        is_candidate_filtered = EXCLUDED.is_candidate_filtered,
                        is_quality_qualified = EXCLUDED.is_quality_qualified,
                        is_pool_selected = EXCLUDED.is_pool_selected,
                        is_pinned_custom = EXCLUDED.is_pinned_custom,
                        account_value = EXCLUDED.account_value,
                        peak_account_value = EXCLUDED.peak_account_value,
                        pnl_30d = EXCLUDED.pnl_30d,
                        roi_30d = EXCLUDED.roi_30d,
                        win_rate = EXCLUDED.win_rate,
                        episode_count = EXCLUDED.episode_count,
                        avg_r_gross = EXCLUDED.avg_r_gross,
                        avg_r_net = EXCLUDED.avg_r_net,
                        nig_mu = EXCLUDED.nig_mu,
                        nig_kappa = EXCLUDED.nig_kappa,
                        nig_alpha = EXCLUDED.nig_alpha,
                        nig_beta = EXCLUDED.nig_beta,
                        thompson_draw = EXCLUDED.thompson_draw,
                        thompson_seed = EXCLUDED.thompson_seed,
                        selection_rank = EXCLUDED.selection_rank,
                        skill_p_value = EXCLUDED.skill_p_value,
                        fdr_qualified = EXCLUDED.fdr_qualified,
                        event_type = EXCLUDED.event_type,
                        death_type = EXCLUDED.death_type,
                        censor_type = EXCLUDED.censor_type
                    """,
                    snapshot.snapshot_date,
                    snapshot.address,
                    snapshot.selection_version,
                    snapshot.is_leaderboard_scanned,
                    snapshot.is_candidate_filtered,
                    snapshot.is_quality_qualified,
                    snapshot.is_pool_selected,
                    snapshot.is_pinned_custom,
                    snapshot.account_value,
                    snapshot.peak_account_value,
                    snapshot.pnl_30d,
                    snapshot.roi_30d,
                    snapshot.win_rate,
                    snapshot.episode_count,
                    snapshot.week_volume,
                    snapshot.orders_per_day,
                    snapshot.avg_r_gross,
                    snapshot.avg_r_net,
                    snapshot.nig_mu,
                    snapshot.nig_kappa,
                    snapshot.nig_alpha,
                    snapshot.nig_beta,
                    snapshot.thompson_draw,
                    snapshot.thompson_seed,
                    snapshot.selection_rank,
                    snapshot.skill_p_value,
                    snapshot.fdr_qualified,
                    snapshot.event_type,
                    snapshot.death_type,
                    snapshot.censor_type,
                )
                inserted += 1
            except Exception as e:
                print(f"[snapshot] Failed to insert snapshot for {snapshot.address}: {e}")

        # Summary stats
        death_count = sum(1 for s in snapshots if s.event_type == "death")
        censor_count = sum(1 for s in snapshots if s.event_type == "censored")
        selected_count = sum(1 for s in snapshots if s.is_pool_selected)
        fdr_count = sum(1 for s in snapshots if s.fdr_qualified)

        return {
            "snapshot_date": snapshot_date.isoformat(),
            "selection_version": SELECTION_VERSION,
            "total_traders": len(snapshots),
            "inserted": inserted,
            "selected": selected_count,
            "fdr_qualified": fdr_count,
            "deaths": death_count,
            "censored": censor_count,
            "death_types": {
                "liquidation": sum(1 for s in snapshots if s.death_type == "liquidation"),
                "drawdown_80": sum(1 for s in snapshots if s.death_type == "drawdown_80"),
                "account_value_floor": sum(1 for s in snapshots if s.death_type == "account_value_floor"),
                "negative_equity": sum(1 for s in snapshots if s.death_type == "negative_equity"),
            },
            "censor_types": {
                "inactive_30d": sum(1 for s in snapshots if s.censor_type == "inactive_30d"),
                "stopped_btc_eth": sum(1 for s in snapshots if s.censor_type == "stopped_btc_eth"),
            },
        }


async def get_snapshot_summary(
    pool: asyncpg.Pool,
    snapshot_date: Optional[date] = None,
) -> Dict[str, Any]:
    """
    Get summary statistics for a snapshot date.

    Args:
        pool: Database connection pool
        snapshot_date: Date to summarize (default: today)

    Returns:
        Summary statistics
    """
    if snapshot_date is None:
        snapshot_date = date.today()

    async with pool.acquire() as conn:
        # Get counts by universe
        counts = await conn.fetchrow(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN is_leaderboard_scanned THEN 1 ELSE 0 END) as leaderboard,
                SUM(CASE WHEN is_candidate_filtered THEN 1 ELSE 0 END) as candidates,
                SUM(CASE WHEN is_quality_qualified THEN 1 ELSE 0 END) as qualified,
                SUM(CASE WHEN is_pool_selected THEN 1 ELSE 0 END) as selected,
                SUM(CASE WHEN fdr_qualified THEN 1 ELSE 0 END) as fdr_qualified,
                SUM(CASE WHEN event_type = 'death' THEN 1 ELSE 0 END) as deaths,
                SUM(CASE WHEN event_type = 'censored' THEN 1 ELSE 0 END) as censored
            FROM trader_snapshots
            WHERE snapshot_date = $1
            """,
            snapshot_date,
        )

        # Get top selected traders
        selected = await conn.fetch(
            """
            SELECT address, thompson_draw, nig_mu, avg_r_net, episode_count, selection_rank
            FROM trader_snapshots
            WHERE snapshot_date = $1
              AND is_pool_selected = true
            ORDER BY selection_rank
            LIMIT 10
            """,
            snapshot_date,
        )

        return {
            "snapshot_date": snapshot_date.isoformat(),
            "counts": dict(counts) if counts else {},
            "top_selected": [dict(row) for row in selected],
        }


async def load_universe_at_date(
    pool: asyncpg.Pool,
    evaluation_date: date,
    version: str = SELECTION_VERSION,
) -> List[str]:
    """
    Load EXACTLY the traders known at evaluation_date.

    CRITICAL: Uses snapshot table, NOT current qualification.
    This prevents look-ahead bias in walk-forward replay.

    Args:
        pool: Database connection pool
        evaluation_date: The as-of date
        version: Selection version to use

    Returns:
        List of addresses in the universe at that date
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT address
            FROM trader_snapshots
            WHERE snapshot_date = $1
              AND selection_version = $2
              AND (is_quality_qualified = true OR is_pool_selected = true)
              AND event_type NOT IN ('death', 'censored')
            """,
            evaluation_date,
            version,
        )

    return [row["address"] for row in rows]
