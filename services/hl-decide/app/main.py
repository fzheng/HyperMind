"""
HL-Decide Service

Generates trading signals based on position lifecycles from tracked traders.
Tracks complete position open/close cycles for accurate performance measurement.

Key responsibilities:
- Consume `b.scores.v1` and `c.fills.v1` events from NATS
- Track position opens (signals) and closes (outcomes)
- Update trader NIG posteriors with position-level R-multiples
- Persist position signals to PostgreSQL

Position Lifecycle:
- Open: "Open Long (Open New)" or "Open Short (Open New)" fills
- Close: "Close Long (Close All)" or "Close Short (Close All)" fills
- Ignored: "Increase" and "Decrease" fills (partial position changes)

@module hl-decide
"""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, Optional, Union
from uuid import uuid4
from collections import OrderedDict

import asyncpg
import nats
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
from starlette.responses import Response

from pydantic import BaseModel

from contracts.py.models import FillEvent, ScoreEvent
from .consensus import ConsensusDetector, Fill, ConsensusSignal
from .episode import EpisodeTracker, EpisodeFill, Episode, EpisodeBuilderConfig
from .atr import get_atr_provider, init_atr_provider, ATRProvider
from .atr_provider import (
    ATRManager,
    get_atr_manager,
    init_atr_manager,
)
from .correlation import (
    get_correlation_provider,
    init_correlation_provider,
    CorrelationProvider,
    run_daily_correlation_job,
)
from .decision_logger import (
    log_decision,
    get_decisions,
    get_decision,
    get_decision_stats,
    GateResult,
)
from .portfolio import (
    get_portfolio_summary,
    get_execution_config,
    update_execution_config,
    get_execution_logs,
)
from .regime import (
    detect_market_regime,
    get_regime_detector,
    MarketRegime,
    REGIME_PARAMS,
)
from .exchanges import (
    init_exchange_manager,
    get_exchange_manager,
    ExchangeType,
)
from .fee_provider import (
    init_fee_provider,
    get_fee_provider,
)
from .funding_provider import (
    init_funding_provider,
    get_funding_provider,
)
from .slippage_provider import (
    init_slippage_provider,
    get_slippage_provider,
)
from .account_normalizer import (
    init_account_normalizer,
    get_account_normalizer,
)
from .hold_time_estimator import (
    init_hold_time_estimator,
    get_hold_time_estimator,
)


class ExecutionConfigUpdate(BaseModel):
    """Request body for updating execution config."""
    enabled: Optional[bool] = None
    hl_enabled: Optional[bool] = None
    hl_address: Optional[str] = None
    hl_max_leverage: Optional[int] = None
    hl_max_position_pct: Optional[float] = None
    hl_max_exposure_pct: Optional[float] = None

SERVICE_NAME = "hl-decide"
NATS_URL = os.getenv("NATS_URL", "nats://0.0.0.0:4222")
DB_URL = os.getenv("DATABASE_URL", "postgresql://hlbot:hlbotpassword@0.0.0.0:5432/hlbot")
MAX_SCORES = int(os.getenv("MAX_SCORES", "500"))
MAX_FILLS = int(os.getenv("MAX_FILLS", "500"))

# Daily reconciliation settings
RECONCILE_INTERVAL_HOURS = int(os.getenv("RECONCILE_INTERVAL_HOURS", "6"))  # Run every 6 hours
RECONCILE_ON_STARTUP = os.getenv("RECONCILE_ON_STARTUP", "true").lower() == "true"

# Correlation refresh settings
CORR_REFRESH_INTERVAL_HOURS = int(os.getenv("CORR_REFRESH_INTERVAL_HOURS", "24"))  # Daily by default

# Exchange settings
EXCHANGE_TESTNET = os.getenv("HL_USE_TESTNET", "true").lower() == "true"
EXCHANGE_HEALTH_CHECK_INTERVAL_MINUTES = int(os.getenv("EXCHANGE_HEALTH_CHECK_INTERVAL_MINUTES", "5"))
EXCHANGE_HEALTH_STAGGER_DELAY_MS = int(os.getenv("EXCHANGE_HEALTH_STAGGER_DELAY_MS", "500"))

# R-multiple calculation: assumed stop loss fraction (1% = 0.01)
ASSUMED_STOP_FRACTION = float(os.getenv("ASSUMED_STOP_FRACTION", "0.01"))

# NIG prior parameters
NIG_PRIOR_M = 0.0
NIG_PRIOR_KAPPA = 1.0
NIG_PRIOR_ALPHA = 3.0
NIG_PRIOR_BETA = 1.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Startup
    try:
        # Connect to database first
        app.state.db = await asyncpg.create_pool(DB_URL)

        # Ensure episode_fill_ids table exists for deduplication
        async with app.state.db.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS episode_fill_ids (
                    fill_id TEXT PRIMARY KEY,
                    processed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
                """
            )
        print("[hl-decide] episode_fill_ids table ready for deduplication")

        # Initialize ATR provider for dynamic stop distances (legacy)
        app.state.atr_provider = init_atr_provider(app.state.db)
        print("[hl-decide] ATR provider initialized")

        # Initialize multi-exchange ATR manager (Phase 6.1)
        app.state.atr_manager = init_atr_manager(
            pool=app.state.db,
            testnet=EXCHANGE_TESTNET,
        )
        print("[hl-decide] Multi-exchange ATR manager initialized")

        # Initialize dynamic fee provider (Phase 6.1)
        app.state.fee_provider = init_fee_provider(testnet=EXCHANGE_TESTNET)
        print("[hl-decide] Dynamic fee provider initialized")

        # Initialize funding rate provider (Phase 6.1)
        app.state.funding_provider = init_funding_provider(testnet=EXCHANGE_TESTNET)
        print("[hl-decide] Funding rate provider initialized")

        # Initialize slippage estimation provider (Phase 6.1)
        app.state.slippage_provider = init_slippage_provider(testnet=EXCHANGE_TESTNET)
        print("[hl-decide] Slippage provider initialized")

        # Initialize account normalizer for multi-exchange equity normalization (Phase 6.1.5)
        app.state.account_normalizer = await init_account_normalizer()
        print("[hl-decide] Account normalizer initialized")

        # Initialize correlation provider and hydrate consensus detector
        app.state.corr_provider = init_correlation_provider(app.state.db)
        corr_count = await app.state.corr_provider.load()
        if corr_count > 0:
            hydrated = app.state.corr_provider.hydrate_detector(consensus_detector)
            update_correlation_metrics(app.state.corr_provider)
            print(f"[hl-decide] Loaded {corr_count} correlations, hydrated detector with {hydrated} pairs")
        else:
            # No correlations found - try to compute them on first startup
            print("[hl-decide] No correlations found, computing initial correlations...")
            try:
                summary = await run_daily_correlation_job(app.state.db)
                total_pairs = summary.get("btc_pairs", 0) + summary.get("eth_pairs", 0)
                if total_pairs > 0:
                    corr_count = await app.state.corr_provider.load()
                    hydrated = app.state.corr_provider.hydrate_detector(consensus_detector)
                    update_correlation_metrics(app.state.corr_provider)
                    print(f"[hl-decide] Computed {total_pairs} correlations, hydrated detector with {hydrated} pairs")
                else:
                    print("[hl-decide] No correlations computed (insufficient data, using default ρ=0.3)")
            except Exception as corr_error:
                print(f"[hl-decide] Correlation computation failed (using default ρ=0.3): {corr_error}")

        # Restore state from database
        score_count, fill_count = await restore_state()
        print(f"[hl-decide] Restored {score_count} scores and {fill_count} fills from database")

        # Bootstrap episodes from historical fills (hl_events table)
        # This processes fills that were loaded via backfill, not real-time NATS
        bootstrap_count = await bootstrap_from_historical_fills()

        # Count open positions
        async with app.state.db.acquire() as conn:
            open_count = await conn.fetchval(
                "SELECT COUNT(*) FROM position_signals WHERE status = 'open'"
            )
            print(f"[hl-decide] {open_count} open positions being tracked")

        # Load initial ATR data for consensus detector
        await update_atr_for_consensus()

        # Load initial funding rates for EV calculation
        await update_funding_for_consensus()

        # Load initial orderbooks for slippage estimation
        await update_orderbooks_for_slippage()

        # Initialize hold time estimator (dynamic hold time from episode data)
        app.state.hold_time_estimator = await init_hold_time_estimator(app.state.db)

        # Initialize exchange manager (auto-detects configured exchanges)
        app.state.exchange_manager = await init_exchange_manager(
            exchanges=None,  # Auto-detect from env vars (HL_PRIVATE_KEY, ASTER_PRIVATE_KEY, BYBIT_API_KEY)
            testnet=EXCHANGE_TESTNET,
        )
        # Set database pool for telemetry persistence (Phase 6)
        app.state.exchange_manager.set_db_pool(app.state.db)
        connected = app.state.exchange_manager.connected_exchanges
        if connected:
            print(f"[hl-decide] Exchange manager initialized: {[e.value for e in connected]}")
            # Initial balance snapshot
            await app.state.exchange_manager.update_all_balances()
        else:
            print("[hl-decide] Exchange manager ready (no exchanges configured - execution disabled)")

        # Configure consensus detector with target exchange for accurate fee calculation
        try:
            exec_config = await get_execution_config(db=app.state.db)
            target_exchange = exec_config.get("exchange", "hyperliquid").lower()
            consensus_detector.set_target_exchange(target_exchange)
            print(f"[hl-decide] Consensus EV gate using {target_exchange} fees")
        except Exception as e:
            print(f"[hl-decide] Could not set target exchange for consensus: {e}")

        # Connect to NATS
        app.state.nc = await nats.connect(NATS_URL)
        app.state.js = app.state.nc.jetstream()
        await ensure_stream(app.state.js, "HL_D", ["d.signals.v1", "d.outcomes.v1"])
        await app.state.nc.subscribe("b.scores.v1", cb=handle_score)
        await app.state.nc.subscribe("c.fills.v1", cb=handle_fill)

        # Start periodic background tasks
        app.state.reconcile_task = asyncio.create_task(periodic_reconciliation_task())
        app.state.corr_refresh_task = asyncio.create_task(periodic_correlation_refresh_task())
        app.state.exchange_health_task = asyncio.create_task(periodic_exchange_health_task())
        print(f"[hl-decide] Periodic reconciliation scheduled every {RECONCILE_INTERVAL_HOURS} hours")
        print(f"[hl-decide] Correlation refresh scheduled every {CORR_REFRESH_INTERVAL_HOURS} hours")
        print(f"[hl-decide] Exchange health check scheduled every {EXCHANGE_HEALTH_CHECK_INTERVAL_MINUTES} minutes")

        print("[hl-decide] Started with position-based tracking, ATR stops, and correlation matrix")
    except Exception as e:
        print(f"[hl-decide] Fatal startup error: {e}")
        raise

    yield  # Application runs here

    # Shutdown
    # Cancel periodic background tasks
    for task_name in ["reconcile_task", "corr_refresh_task", "exchange_health_task"]:
        if hasattr(app.state, task_name):
            task = getattr(app.state, task_name)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    if hasattr(app.state, "account_normalizer"):
        await app.state.account_normalizer.close()
    if hasattr(app.state, "exchange_manager"):
        await app.state.exchange_manager.disconnect_all()
    if hasattr(app.state, "nc"):
        await app.state.nc.drain()
    if hasattr(app.state, "db"):
        await app.state.db.close()


app = FastAPI(title="hl-decide", version="0.3.0", lifespan=lifespan)
scores: OrderedDict[str, ScoreEvent] = OrderedDict()
fills: OrderedDict[str, FillEvent] = OrderedDict()

# Consensus detector for Alpha Pool signal generation
consensus_detector = ConsensusDetector()

# Episode tracker for position lifecycle management
episode_config = EpisodeBuilderConfig(
    default_stop_fraction=ASSUMED_STOP_FRACTION,
    r_min=-2.0,
    r_max=2.0,
    timeout_hours=168.0,
)
episode_tracker = EpisodeTracker(config=episode_config)

registry = CollectorRegistry()
position_open_counter = Counter("decide_positions_opened_total", "Positions opened", registry=registry)
position_close_counter = Counter("decide_positions_closed_total", "Positions closed", registry=registry)
fill_counter = Counter("decide_fills_total", "Total fills processed", registry=registry)
consensus_signal_counter = Counter("decide_consensus_signals_total", "Consensus signals generated", registry=registry)
position_pnl_histogram = Histogram(
    "decide_position_pnl_r",
    "Position P&L in R-multiples",
    registry=registry,
    buckets=(-2, -1, -0.5, 0, 0.5, 1, 2, 5),
)

# Data quality observability metrics
from prometheus_client import Gauge

# ATR metrics - track staleness and fallback usage
atr_stale_counter = Counter(
    "decide_atr_stale_total",
    "Times ATR data was stale",
    labelnames=["asset"],
    registry=registry,
)
atr_fallback_counter = Counter(
    "decide_atr_fallback_total",
    "Times ATR used fallback (hardcoded or realized_vol)",
    labelnames=["asset", "source"],
    registry=registry,
)
atr_age_gauge = Gauge(
    "decide_atr_age_seconds",
    "Current age of ATR data in seconds",
    labelnames=["asset"],
    registry=registry,
)
atr_blocked_counter = Counter(
    "decide_atr_blocked_total",
    "Times gating was blocked due to stale ATR (strict mode)",
    labelnames=["asset"],
    registry=registry,
)

# Correlation metrics - track staleness and default usage
corr_stale_gauge = Gauge(
    "decide_correlation_stale",
    "Whether correlation data is stale (1=stale, 0=fresh)",
    registry=registry,
)
corr_age_gauge = Gauge(
    "decide_correlation_age_days",
    "Age of correlation data in days",
    registry=registry,
)
corr_decay_gauge = Gauge(
    "decide_correlation_decay_factor",
    "Current decay factor applied to correlations (1=no decay, 0=full decay)",
    registry=registry,
)
corr_default_used_counter = Counter(
    "decide_correlation_default_used_total",
    "Times default correlation was used (no stored data)",
    registry=registry,
)
corr_pairs_loaded_gauge = Gauge(
    "decide_correlation_pairs_loaded",
    "Number of correlation pairs currently loaded",
    registry=registry,
)
corr_coverage_gauge = Gauge(
    "decide_correlation_coverage_pct",
    "Percentage of pool trader pairs with actual correlation data (0-100)",
    registry=registry,
)
corr_pool_size_gauge = Gauge(
    "decide_correlation_pool_size",
    "Number of traders in the correlation pool",
    registry=registry,
)

# Effective-K metrics - track when default ρ is used
effk_default_fallback_counter = Counter(
    "decide_effk_default_fallback_total",
    "Times effK calculation used default ρ for a pair",
    registry=registry,
)
effk_value_histogram = Histogram(
    "decide_effk_value",
    "Effective-K values in consensus checks",
    registry=registry,
    buckets=(1, 2, 3, 5, 7, 10, 15, 20, 50),
)

# Weight distribution metrics
weight_histogram = Histogram(
    "decide_vote_weight",
    "Vote weight values in consensus",
    registry=registry,
    buckets=(0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0),
)
weight_max_gauge = Gauge(
    "decide_vote_weight_max",
    "Maximum vote weight in recent consensus check",
    registry=registry,
)
weight_gini_gauge = Gauge(
    "decide_vote_weight_gini",
    "Gini coefficient of vote weights (0=equal, 1=concentrated)",
    registry=registry,
)
weight_saturation_counter = Counter(
    "decide_vote_weight_saturated_total",
    "Votes where weight hit the cap (saturation)",
    labelnames=["asset"],
    registry=registry,
)
weight_saturation_pct_gauge = Gauge(
    "decide_vote_weight_saturation_pct",
    "Percentage of weights at or near cap in recent consensus check",
    registry=registry,
)

# Risk limit metrics
signal_risk_rejected_counter = Counter(
    "decide_signal_risk_rejected_total",
    "Signals rejected by risk limits",
    labelnames=["reason"],
    registry=registry,
)
signal_generated_counter = Counter(
    "decide_signal_generated_total",
    "Signals that passed all gates including risk limits",
    labelnames=["symbol", "direction"],
    registry=registry,
)


def calculate_gini(weights: list[float]) -> float:
    """
    Calculate Gini coefficient for a list of weights.

    Gini = 0 means perfect equality (all weights equal)
    Gini = 1 means perfect inequality (one weight = 1, rest = 0)

    Args:
        weights: List of weight values

    Returns:
        Gini coefficient between 0 and 1
    """
    if not weights or len(weights) < 2:
        return 0.0

    sorted_weights = sorted(weights)
    n = len(sorted_weights)
    total = sum(sorted_weights)

    if total == 0:
        return 0.0

    # Calculate using the formula: G = (2 * sum(i * x_i) - (n + 1) * sum(x_i)) / (n * sum(x_i))
    cumulative = sum((i + 1) * w for i, w in enumerate(sorted_weights))
    gini = (2 * cumulative - (n + 1) * total) / (n * total)

    return max(0.0, min(1.0, gini))


def update_weight_metrics(weights: list[float], asset: str = "unknown") -> None:
    """
    Update weight distribution metrics from a consensus check.

    Args:
        weights: List of vote weights from consensus check
        asset: Asset symbol for saturation tracking (BTC/ETH)
    """
    if not weights:
        return

    # Import weight cap from consensus module
    from .consensus import VOTE_WEIGHT_MAX

    # Record each weight in histogram
    for w in weights:
        weight_histogram.observe(w)

    # Update max weight gauge
    weight_max_gauge.set(max(weights))

    # Update Gini coefficient gauge
    gini = calculate_gini(weights)
    weight_gini_gauge.set(gini)

    # Track saturation (weights at or near cap)
    # Consider "near cap" as >= 95% of max weight
    saturation_threshold = VOTE_WEIGHT_MAX * 0.95
    saturated_count = sum(1 for w in weights if w >= saturation_threshold)

    if saturated_count > 0:
        weight_saturation_counter.labels(asset=asset).inc(saturated_count)

    saturation_pct = (saturated_count / len(weights)) * 100 if weights else 0
    weight_saturation_pct_gauge.set(saturation_pct)


def update_correlation_metrics(provider, pool_addresses: list[str] | None = None) -> None:
    """
    Update correlation observability metrics from the provider.

    Args:
        provider: CorrelationProvider instance
        pool_addresses: Optional list of addresses in the current pool (for coverage calc)
    """
    # Update staleness gauge
    corr_stale_gauge.set(1.0 if provider.is_stale else 0.0)

    # Update age gauge
    age = provider.age_days if provider._loaded_date else -1
    if age != float('inf'):
        corr_age_gauge.set(age)

    # Update decay factor
    corr_decay_gauge.set(provider._decay_factor())

    # Update pairs loaded
    corr_pairs_loaded_gauge.set(len(provider.correlations))

    # Calculate coverage if pool addresses provided
    if pool_addresses:
        corr_pool_size_gauge.set(len(pool_addresses))

        # Count how many pairs have actual correlation data
        n = len(pool_addresses)
        total_pairs = n * (n - 1) // 2  # n choose 2

        if total_pairs > 0:
            pairs_with_data = 0
            addrs_lower = [a.lower() for a in pool_addresses]
            for i, a in enumerate(addrs_lower):
                for j in range(i + 1, len(addrs_lower)):
                    b = addrs_lower[j]
                    key = tuple(sorted([a, b]))
                    if key in provider.correlations:
                        pairs_with_data += 1

            coverage_pct = (pairs_with_data / total_pairs) * 100
            corr_coverage_gauge.set(coverage_pct)
        else:
            corr_coverage_gauge.set(0.0)


async def ensure_stream(js, name: str, subjects):
    """Ensures a NATS JetStream stream exists."""
    try:
        await js.stream_info(name)
    except Exception:
        await js.add_stream(name=name, subjects=subjects)


def winsorize_r(r: float, r_min: float = -2.0, r_max: float = 2.0) -> float:
    """Winsorize R-multiple to bounds to tame heavy tails."""
    return max(r_min, min(r_max, r))


def parse_action(fill: FillEvent) -> Optional[str]:
    """
    Parse the action from a fill event's meta.

    Returns:
        'open_long', 'open_short', 'close_long', 'close_short', or None for other actions
    """
    if not isinstance(fill.meta, dict):
        return None

    action = fill.meta.get('action', '')
    if not action:
        return None

    action_lower = action.lower()

    if 'open' in action_lower and 'open new' in action_lower:
        if 'long' in action_lower:
            return 'open_long'
        elif 'short' in action_lower:
            return 'open_short'

    if 'close' in action_lower and 'close all' in action_lower:
        if 'long' in action_lower:
            return 'close_long'
        elif 'short' in action_lower:
            return 'close_short'

    return None


async def create_position_signal(conn, fill: FillEvent, direction: str) -> Optional[str]:
    """
    Create a new position signal when a position opens.

    Args:
        conn: Database connection
        fill: The opening fill event
        direction: 'long' or 'short'

    Returns:
        Position signal ID if created, None if duplicate
    """
    signal_id = str(uuid4())

    try:
        await conn.execute(
            """
            INSERT INTO position_signals (
                id, address, asset, direction,
                entry_fill_id, entry_price, entry_size, entry_ts,
                status
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'open')
            ON CONFLICT (entry_fill_id) DO NOTHING
            """,
            signal_id,
            fill.address.lower(),
            fill.asset,
            direction,
            fill.fill_id,
            fill.price,
            fill.size,
            fill.ts,
        )

        # Update positions_opened count
        await conn.execute(
            """
            INSERT INTO trader_performance (address, positions_opened)
            VALUES ($1, 1)
            ON CONFLICT (address) DO UPDATE SET
                positions_opened = trader_performance.positions_opened + 1,
                updated_at = NOW()
            """,
            fill.address.lower(),
        )

        print(f"[hl-decide] Position opened: {fill.address[:10]}... {direction} {fill.asset} @ {fill.price}")
        return signal_id

    except Exception as e:
        print(f"[hl-decide] Failed to create position signal: {e}")
        return None


async def close_position_signal(conn, fill: FillEvent, direction: str) -> Optional[float]:
    """
    Close an open position signal and calculate R-multiple.

    Args:
        conn: Database connection
        fill: The closing fill event
        direction: 'long' or 'short' (the direction being closed)

    Returns:
        R-multiple if position was closed, None otherwise
    """
    # Find the open position for this address+asset+direction
    open_signal = await conn.fetchrow(
        """
        SELECT id, entry_price, entry_size, entry_ts
        FROM position_signals
        WHERE address = $1 AND asset = $2 AND direction = $3 AND status = 'open'
        ORDER BY entry_ts DESC
        LIMIT 1
        """,
        fill.address.lower(),
        fill.asset,
        direction,
    )

    if not open_signal:
        print(f"[hl-decide] No open {direction} position found for {fill.address[:10]}... {fill.asset}")
        return None

    # Calculate R-multiple from realized_pnl
    entry_price = float(open_signal['entry_price'])
    entry_size = float(open_signal['entry_size'])
    entry_notional = entry_price * entry_size
    risk_amount = entry_notional * ASSUMED_STOP_FRACTION

    # Use realized_pnl from Hyperliquid if available
    if fill.realized_pnl is not None and risk_amount > 0:
        result_r = float(fill.realized_pnl) / risk_amount
    else:
        # Fallback: calculate from price difference
        if direction == 'long':
            pnl = (fill.price - entry_price) * entry_size
        else:  # short
            pnl = (entry_price - fill.price) * entry_size
        result_r = pnl / risk_amount if risk_amount > 0 else 0.0

    # Update the position signal
    await conn.execute(
        """
        UPDATE position_signals SET
            exit_fill_id = $1,
            exit_price = $2,
            exit_ts = $3,
            realized_pnl = $4,
            result_r = $5,
            status = 'closed',
            closed_reason = 'full_close',
            updated_at = NOW()
        WHERE id = $6
        """,
        fill.fill_id,
        fill.price,
        fill.ts,
        fill.realized_pnl,
        result_r,
        open_signal['id'],
    )

    print(f"[hl-decide] Position closed: {fill.address[:10]}... {direction} {fill.asset} R={result_r:.2f}")
    return result_r


async def update_trader_performance(conn, address: str, result_r: float) -> None:
    """
    Update trader performance statistics with a position outcome.
    Updates NIG posterior for Thompson Sampling.

    Args:
        conn: Database connection
        address: Trader's Ethereum address
        result_r: P&L in R-multiples
    """
    # Winsorize for NIG update
    r = winsorize_r(result_r)
    success = result_r > 0

    # First ensure the trader exists with default priors
    await conn.execute(
        """
        INSERT INTO trader_performance (
            address, nig_m, nig_kappa, nig_alpha, nig_beta
        )
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (address) DO NOTHING
        """,
        address.lower(),
        NIG_PRIOR_M,
        NIG_PRIOR_KAPPA,
        NIG_PRIOR_ALPHA,
        NIG_PRIOR_BETA,
    )

    # Update stats and NIG posterior
    await conn.execute(
        """
        UPDATE trader_performance SET
            -- Position stats
            positions_closed = positions_closed + 1,
            positions_won = positions_won + $2,
            -- Legacy stats (for backwards compatibility)
            total_signals = total_signals + 1,
            winning_signals = winning_signals + $2,
            total_pnl_r = total_pnl_r + $3,
            -- Beta update (legacy)
            alpha = alpha + $4,
            beta = beta + $5,
            -- NIG conjugate update
            nig_kappa = COALESCE(nig_kappa, $6) + 1,
            nig_m = (COALESCE(nig_kappa, $6) * COALESCE(nig_m, $7) + $8) / (COALESCE(nig_kappa, $6) + 1),
            nig_alpha = COALESCE(nig_alpha, $9) + 0.5,
            nig_beta = COALESCE(nig_beta, $10) + 0.5 * COALESCE(nig_kappa, $6) * POWER($8 - COALESCE(nig_m, $7), 2) / (COALESCE(nig_kappa, $6) + 1),
            -- Rolling average R
            avg_r = (COALESCE(avg_r, 0) * GREATEST(positions_closed - 1, 0) + $8) / GREATEST(positions_closed, 1),
            last_signal_at = NOW(),
            updated_at = NOW()
        WHERE address = $1
        """,
        address.lower(),
        1 if success else 0,
        result_r,
        1 if success else 0,  # alpha increment
        0 if success else 1,  # beta increment
        NIG_PRIOR_KAPPA,
        NIG_PRIOR_M,
        r,  # winsorized R for NIG
        NIG_PRIOR_ALPHA,
        NIG_PRIOR_BETA,
    )


async def handle_fill_for_positions(fill: FillEvent) -> None:
    """
    Process a fill event for position tracking.

    Only tracks position opens and closes, ignoring increases/decreases.
    """
    action = parse_action(fill)
    if not action:
        return  # Ignore non-position-changing fills

    try:
        async with app.state.db.acquire() as conn:
            if action == 'open_long':
                await create_position_signal(conn, fill, 'long')
                position_open_counter.inc()

            elif action == 'open_short':
                await create_position_signal(conn, fill, 'short')
                position_open_counter.inc()

            elif action == 'close_long':
                result_r = await close_position_signal(conn, fill, 'long')
                if result_r is not None:
                    await update_trader_performance(conn, fill.address, result_r)
                    position_close_counter.inc()
                    position_pnl_histogram.observe(result_r)

            elif action == 'close_short':
                result_r = await close_position_signal(conn, fill, 'short')
                if result_r is not None:
                    await update_trader_performance(conn, fill.address, result_r)
                    position_close_counter.inc()
                    position_pnl_histogram.observe(result_r)

    except Exception as e:
        print(f"[hl-decide] Error handling fill for positions: {e}")


async def handle_fill_via_episodes(fill: Union[FillEvent, EpisodeFill]) -> Optional[Episode]:
    """
    Process a fill through the episode tracker for proper position lifecycle management.

    This is the new episode-based approach that:
    1. Tracks ALL fills (not just Open New / Close All)
    2. Builds complete position episodes with VWAP entry/exit
    3. Calculates R-multiples when positions close
    4. Updates NIG posteriors from episode outcomes

    Accepts both FillEvent (from real-time NATS) and EpisodeFill (from reconciliation).

    Args:
        fill: The incoming fill event (FillEvent or EpisodeFill)

    Returns:
        Closed Episode if position closed, None otherwise
    """
    try:
        # Convert to EpisodeFill if needed
        if isinstance(fill, EpisodeFill):
            episode_fill = fill
        else:
            episode_fill = EpisodeFill(
                fill_id=fill.fill_id,
                address=fill.address,
                asset=fill.asset,
                side=fill.side,
                size=float(fill.size or 0),
                price=float(fill.price) if hasattr(fill, 'price') and fill.price else 0.0,
                ts=fill.ts if isinstance(fill.ts, datetime) else datetime.fromisoformat(str(fill.ts).replace('Z', '+00:00')),
                realized_pnl=float(fill.realized_pnl) if fill.realized_pnl else None,
                fees=0.0,  # Could extract from meta if available
            )

        # Track this fill to prevent double processing during reconciliation
        # This is important for deduplication when real-time and reconciliation overlap
        async with app.state.db.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO episode_fill_ids (fill_id) VALUES ($1) ON CONFLICT DO NOTHING
                """,
                episode_fill.fill_id,
            )

        # Process through episode tracker
        closed_episode = episode_tracker.process_fill(episode_fill)

        if closed_episode:
            # Episode closed - persist and update NIG
            await persist_closed_episode(closed_episode)
            return closed_episode

        # Check if we just opened a new episode
        open_episode = episode_tracker.get_open_episode(episode_fill.address, episode_fill.asset)
        if open_episode and len(open_episode.entry_fills) == 1:
            # New position opened
            await persist_open_episode(open_episode)
            position_open_counter.inc()
            print(f"[hl-decide] Episode opened: {episode_fill.address[:10]}... {open_episode.direction} {episode_fill.asset}")

        return None

    except Exception as e:
        print(f"[hl-decide] Error handling fill via episodes: {e}")
        return None


async def persist_open_episode(episode: Episode) -> None:
    """Persist a newly opened episode to the database."""
    try:
        async with app.state.db.acquire() as conn:
            signal_id = str(uuid4())

            # Insert position signal
            await conn.execute(
                """
                INSERT INTO position_signals (
                    id, address, asset, direction,
                    entry_fill_id, entry_price, entry_size, entry_ts,
                    entry_px_vwap, stop_bps_used, entry_notional_usd, risk_usd,
                    entry_fill_count, status
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, 'open')
                ON CONFLICT (entry_fill_id) DO NOTHING
                """,
                signal_id,
                episode.address,
                episode.asset,
                episode.direction,
                episode.entry_fills[0].fill_id if episode.entry_fills else '',
                episode.entry_vwap,
                episode.entry_size,
                episode.entry_ts,
                episode.entry_vwap,
                episode.stop_bps,
                episode.entry_notional,
                episode.risk_amount,
                len(episode.entry_fills),
            )

            # Insert all entry fills
            for fill in episode.entry_fills:
                await conn.execute(
                    """
                    INSERT INTO episode_fills (episode_id, fill_id, fill_type, side, size, price, ts, fees)
                    VALUES ($1, $2, 'entry', $3, $4, $5, $6, $7)
                    ON CONFLICT (episode_id, fill_id) DO NOTHING
                    """,
                    signal_id,
                    fill.fill_id,
                    fill.side,
                    fill.size,
                    fill.price,
                    fill.ts,
                    fill.fees,
                )

            # Update positions_opened count
            await conn.execute(
                """
                INSERT INTO trader_performance (address, positions_opened)
                VALUES ($1, 1)
                ON CONFLICT (address) DO UPDATE SET
                    positions_opened = trader_performance.positions_opened + 1,
                    updated_at = NOW()
                """,
                episode.address,
            )

    except Exception as e:
        print(f"[hl-decide] Failed to persist open episode: {e}")


async def persist_closed_episode(episode: Episode) -> None:
    """
    Persist a closed episode and update trader NIG posteriors.

    This is where the R-multiple from the episode flows into the
    Bayesian learning system.
    """
    try:
        async with app.state.db.acquire() as conn:
            # Find the open position signal to update
            open_signal = await conn.fetchrow(
                """
                SELECT id FROM position_signals
                WHERE address = $1 AND asset = $2 AND direction = $3 AND status = 'open'
                ORDER BY entry_ts DESC
                LIMIT 1
                """,
                episode.address,
                episode.asset,
                episode.direction,
            )

            if not open_signal:
                print(f"[hl-decide] No open signal found for closed episode {episode.id}")
                return

            signal_id = open_signal['id']

            # Calculate hold time
            hold_secs = None
            if episode.entry_ts and episode.exit_ts:
                hold_secs = int((episode.exit_ts - episode.entry_ts).total_seconds())

            # Update the position signal with exit info
            await conn.execute(
                """
                UPDATE position_signals SET
                    exit_fill_id = $1,
                    exit_price = $2,
                    exit_ts = $3,
                    exit_px_vwap = $4,
                    realized_pnl = $5,
                    realized_pnl_usd = $5,
                    result_r = $6,
                    r_clamped = $6,
                    r_unclamped = $7,
                    hold_secs = $8,
                    exit_fill_count = $9,
                    status = 'closed',
                    closed_reason = $10,
                    updated_at = NOW()
                WHERE id = $11
                """,
                episode.exit_fills[-1].fill_id if episode.exit_fills else None,
                episode.exit_vwap,
                episode.exit_ts,
                episode.exit_vwap,
                episode.realized_pnl,
                episode.result_r,
                episode.result_r_unclamped,
                hold_secs,
                len(episode.exit_fills),
                episode.closed_reason,
                signal_id,
            )

            # Insert all exit fills
            for fill in episode.exit_fills:
                await conn.execute(
                    """
                    INSERT INTO episode_fills (episode_id, fill_id, fill_type, side, size, price, ts, realized_pnl, fees)
                    VALUES ($1, $2, 'exit', $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (episode_id, fill_id) DO NOTHING
                    """,
                    signal_id,
                    fill.fill_id,
                    fill.side,
                    fill.size,
                    fill.price,
                    fill.ts,
                    fill.realized_pnl,
                    fill.fees,
                )

            # Update NIG posterior with episode R-multiple
            if episode.result_r is not None:
                await update_trader_performance(conn, episode.address, episode.result_r)
                position_close_counter.inc()
                position_pnl_histogram.observe(episode.result_r)

                print(f"[hl-decide] Episode closed: {episode.address[:10]}... {episode.direction} {episode.asset} "
                      f"R={episode.result_r:.2f} (unclamped={episode.result_r_unclamped:.2f}) "
                      f"hold={hold_secs}s")

    except Exception as e:
        print(f"[hl-decide] Failed to persist closed episode: {e}")


async def persist_score(address: str, score: ScoreEvent) -> None:
    """Persist score to database for recovery."""
    try:
        async with app.state.db.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO decide_scores (address, score, weight, rank, window_s, ts, meta, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                ON CONFLICT (address) DO UPDATE SET
                    score = EXCLUDED.score,
                    weight = EXCLUDED.weight,
                    rank = EXCLUDED.rank,
                    window_s = EXCLUDED.window_s,
                    ts = EXCLUDED.ts,
                    meta = EXCLUDED.meta,
                    updated_at = EXCLUDED.updated_at
                """,
                address.lower(),
                score.score,
                score.weight,
                score.rank,
                score.window_s,
                score.ts,
                json.dumps(score.meta) if isinstance(score.meta, dict) else "{}",
            )
    except Exception as e:
        print(f"[hl-decide] Failed to persist score for {address}: {e}")


async def persist_fill(address: str, fill: FillEvent) -> None:
    """Persist fill to database for recovery."""
    try:
        async with app.state.db.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO decide_fills (address, fill_id, asset, side, size, price, ts, meta, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
                ON CONFLICT (address) DO UPDATE SET
                    fill_id = EXCLUDED.fill_id,
                    asset = EXCLUDED.asset,
                    side = EXCLUDED.side,
                    size = EXCLUDED.size,
                    price = EXCLUDED.price,
                    ts = EXCLUDED.ts,
                    meta = EXCLUDED.meta,
                    updated_at = EXCLUDED.updated_at
                """,
                address.lower(),
                fill.fill_id,
                fill.asset,
                fill.side,
                fill.size,
                fill.price if hasattr(fill, 'price') else None,
                fill.ts,
                json.dumps(fill.meta) if isinstance(fill.meta, dict) else "{}",
            )
    except Exception as e:
        print(f"[hl-decide] Failed to persist fill for {address}: {e}")


async def update_atr_for_consensus() -> None:
    """
    Update ATR-based stop fractions for consensus detection.

    Called on startup and periodically to refresh volatility data.
    Updates both the consensus detector and episode tracker with
    current ATR-based stop distances, adjusted for market regime.

    Phase 6.1: Uses the multi-exchange ATR manager to fetch ATR from
    the target exchange when possible, falling back to Hyperliquid.
    """
    from .regime import get_regime_detector, get_regime_adjusted_stop, MarketRegime

    try:
        # Get both legacy provider and new manager
        atr_provider = get_atr_provider()
        atr_manager = get_atr_manager()
        regime_detector = get_regime_detector(app.state.db)

        # Get target exchange from consensus detector
        target_exchange = consensus_detector.target_exchange

        for symbol in ["BTC", "ETH"]:
            # Phase 6.1: Try target exchange first via ATR manager
            # Falls back to Hyperliquid if target unavailable
            atr_data = await atr_manager.get_atr(
                symbol,
                exchange=target_exchange,
                fallback_to_default=True,
            )
            base_stop_fraction = atr_manager.get_stop_fraction(atr_data)

            # Get current regime and adjust stop fraction
            regime_analysis = await regime_detector.detect_regime(symbol)
            regime = regime_analysis.regime if regime_analysis else MarketRegime.UNKNOWN
            stop_fraction = get_regime_adjusted_stop(base_stop_fraction, regime)

            # Update ATR observability metrics
            atr_age_gauge.labels(asset=symbol).set(atr_data.age_seconds)

            if atr_data.is_stale:
                atr_stale_counter.labels(asset=symbol).inc()

            if atr_data.source in ("fallback_hardcoded", "realized_vol"):
                atr_fallback_counter.labels(asset=symbol, source=atr_data.source).inc()

            # Check if gating should be blocked (strict mode)
            should_block, block_reason = atr_manager.should_block_gate(atr_data)
            if should_block:
                atr_blocked_counter.labels(asset=symbol).inc()
                print(f"[hl-decide] ATR BLOCKED for {symbol}: {block_reason}")

            # Update consensus detector with ATR validity for price gate
            is_valid_for_gating = not should_block
            validity_reason = block_reason if should_block else f"ATR from {atr_data.exchange}"
            consensus_detector.set_stop_fraction(
                symbol,
                stop_fraction,
                is_valid_for_gating=is_valid_for_gating,
                validity_reason=validity_reason,
            )

            # Update episode tracker config
            # Note: Episode tracker uses a shared config, so update default_stop_fraction
            # This affects new episodes; existing ones keep their original stop
            episode_config.default_stop_fraction = stop_fraction

            regime_str = regime.value if regime else "unknown"
            exchange_note = f"exchange={atr_data.exchange}" if target_exchange != "hyperliquid" else ""
            print(f"[hl-decide] {symbol} ATR stop: {stop_fraction*100:.2f}% (base: {base_stop_fraction*100:.2f}%, regime: {regime_str}, source: {atr_data.source}{', ' + exchange_note if exchange_note else ''})")

    except Exception as e:
        print(f"[hl-decide] Failed to update ATR for consensus: {e}")
        # Keep using default 1% stops on error


async def update_funding_for_consensus() -> None:
    """
    Pre-fetch and cache funding rates for consensus detection.

    Called on startup and periodically to refresh funding rate data.
    The cached data is used by get_funding_cost_bps_sync() in the
    EV calculation during consensus detection.

    Phase 6.1: Uses the funding provider to fetch from exchange APIs.
    """
    try:
        funding_provider = get_funding_provider()
        target_exchange = consensus_detector.target_exchange

        for asset in ["BTC", "ETH"]:
            # Fetch and cache funding rate
            funding_data = await funding_provider.get_funding(
                asset=asset,
                exchange=target_exchange,
                force_refresh=True,
            )

            daily_cost = funding_data.daily_cost_bps
            source = funding_data.source
            print(
                f"[hl-decide] {asset} funding: {funding_data.rate_bps:.2f} bps/{funding_data.interval_hours}h "
                f"({daily_cost:.2f} bps/day, source={source}, exchange={target_exchange})"
            )

    except Exception as e:
        print(f"[hl-decide] Failed to update funding rates: {e}")
        # Sync function will use defaults on cache miss


async def update_orderbooks_for_slippage() -> None:
    """
    Pre-fetch and cache orderbook data for slippage estimation.

    Called on startup and periodically to refresh orderbook data.
    The cached data is used by get_slippage_estimate_bps_sync() in the
    EV calculation during consensus detection.

    Phase 6.1: Uses the slippage provider to fetch orderbooks from exchange APIs.
    """
    try:
        slippage_provider = get_slippage_provider()
        target_exchange = consensus_detector.target_exchange

        for asset in ["BTC", "ETH"]:
            # Fetch and cache orderbook
            orderbook = await slippage_provider.get_orderbook(
                asset=asset,
                exchange=target_exchange,
                force_refresh=True,
            )

            if orderbook is not None:
                print(
                    f"[hl-decide] {asset} orderbook: spread={orderbook.spread_bps:.2f} bps, "
                    f"bid_depth=${orderbook.get_bid_depth_usd(5)/1000:.0f}k, "
                    f"ask_depth=${orderbook.get_ask_depth_usd(5)/1000:.0f}k "
                    f"(source={orderbook.source}, exchange={target_exchange})"
                )
            else:
                print(f"[hl-decide] {asset} orderbook: unavailable, using static slippage estimates")

    except Exception as e:
        print(f"[hl-decide] Failed to update orderbooks: {e}")
        # Sync function will use static defaults on cache miss


async def restore_state() -> tuple[int, int]:
    """Restore scores and fills from database on startup."""
    score_count = 0
    fill_count = 0
    try:
        async with app.state.db.acquire() as conn:
            # Restore scores
            score_rows = await conn.fetch(
                """
                SELECT address, score, weight, rank, window_s, ts, meta
                FROM decide_scores
                WHERE updated_at > NOW() - INTERVAL '24 hours'
                ORDER BY updated_at DESC
                LIMIT $1
                """,
                MAX_SCORES,
            )
            for row in score_rows:
                score = ScoreEvent(
                    address=row["address"],
                    score=float(row["score"]),
                    weight=float(row["weight"]),
                    rank=int(row["rank"]),
                    window_s=int(row["window_s"]),
                    ts=row["ts"],
                    meta=json.loads(row["meta"]) if row["meta"] else {},
                )
                scores[row["address"]] = score
                score_count += 1

            # Restore fills
            fill_rows = await conn.fetch(
                """
                SELECT address, fill_id, asset, side, size, price, ts, meta
                FROM decide_fills
                WHERE updated_at > NOW() - INTERVAL '24 hours'
                ORDER BY updated_at DESC
                LIMIT $1
                """,
                MAX_FILLS,
            )
            for row in fill_rows:
                fill = FillEvent(
                    fill_id=row["fill_id"],
                    address=row["address"],
                    asset=row["asset"],
                    side=row["side"],
                    size=float(row["size"]),
                    price=float(row["price"]) if row["price"] is not None else 0.0,
                    ts=row["ts"],
                    meta=json.loads(row["meta"]) if row["meta"] else {},
                )
                fills[row["address"]] = fill
                fill_count += 1

    except Exception as e:
        print(f"[hl-decide] Failed to restore state: {e}")

    return score_count, fill_count


async def reconcile_historical_fills(force: bool = False) -> dict:
    """
    Reconcile episodes from historical fills in hl_events table.
    This processes fills that were loaded via backfill or missed during downtime.

    Safe to run multiple times - uses episode_fill_ids table to prevent double-booking.

    Args:
        force: If True, reprocess all fills even if already processed

    Returns:
        dict with counts: {"found": N, "new": N, "skipped": N, "errors": N}
    """
    result = {"found": 0, "new": 0, "skipped": 0, "errors": 0}
    try:
        async with app.state.db.acquire() as conn:
            # Ensure tracking table exists for processed fill IDs
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS episode_fill_ids (
                    fill_id TEXT PRIMARY KEY,
                    processed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
                """
            )

            # Get historical fills from hl_events (trades from backfill)
            # Filter to Alpha Pool addresses and BTC/ETH only
            # Exclude already processed fill IDs unless force=True
            if force:
                rows = await conn.fetch(
                    """
                    SELECT
                        e.address,
                        e.payload->>'symbol' as asset,
                        e.payload->>'action' as action,
                        e.payload->>'size' as size,
                        e.payload->>'priceUsd' as price,
                        e.payload->>'fee' as fee,
                        e.payload->>'at' as ts,
                        e.payload->>'hash' as hash
                    FROM hl_events e
                    INNER JOIN alpha_pool_addresses a ON lower(e.address) = lower(a.address)
                    WHERE e.type = 'trade'
                      AND e.payload->>'symbol' IN ('BTC', 'ETH')
                      AND e.payload->>'at' IS NOT NULL
                    ORDER BY (e.payload->>'at')::timestamp ASC
                    LIMIT 10000
                    """
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT
                        e.address,
                        e.payload->>'symbol' as asset,
                        e.payload->>'action' as action,
                        e.payload->>'size' as size,
                        e.payload->>'priceUsd' as price,
                        e.payload->>'fee' as fee,
                        e.payload->>'at' as ts,
                        e.payload->>'hash' as hash
                    FROM hl_events e
                    INNER JOIN alpha_pool_addresses a ON lower(e.address) = lower(a.address)
                    LEFT JOIN episode_fill_ids efi ON e.payload->>'hash' = efi.fill_id
                    WHERE e.type = 'trade'
                      AND e.payload->>'symbol' IN ('BTC', 'ETH')
                      AND e.payload->>'at' IS NOT NULL
                      AND efi.fill_id IS NULL
                    ORDER BY (e.payload->>'at')::timestamp ASC
                    LIMIT 10000
                    """
                )

            result["found"] = len(rows)
            print(f"[hl-decide] Reconciliation found {len(rows)} fills to process")

            for row in rows:
                try:
                    fill_id = row["hash"]
                    if not fill_id:
                        result["skipped"] += 1
                        continue

                    # Parse action to determine side
                    action = row["action"] or ""
                    action_lower = action.lower()

                    # Determine side from action
                    if "long" in action_lower:
                        if "close" in action_lower:
                            side = "sell"  # Closing long = sell
                        else:
                            side = "buy"  # Open/Increase long = buy
                    elif "short" in action_lower:
                        if "close" in action_lower:
                            side = "buy"  # Closing short = buy
                        else:
                            side = "sell"  # Open/Increase short = sell
                    else:
                        result["skipped"] += 1
                        continue  # Skip if we can't determine side

                    # Parse timestamp
                    ts_str = row["ts"]
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))

                    # Create episode fill
                    fill = EpisodeFill(
                        fill_id=fill_id,
                        address=row["address"].lower(),
                        asset=row["asset"],
                        side=side,
                        size=abs(float(row["size"] or 0)),
                        price=float(row["price"] or 0),
                        ts=ts,
                        fees=float(row["fee"] or 0),
                    )

                    # Process through episode tracker (this updates NIG on close)
                    episode = await handle_fill_via_episodes(fill)

                    # Mark fill as processed to prevent double-booking
                    await conn.execute(
                        "INSERT INTO episode_fill_ids (fill_id) VALUES ($1) ON CONFLICT DO NOTHING",
                        fill_id,
                    )

                    result["new"] += 1
                except Exception as fill_err:
                    result["errors"] += 1
                    continue

        print(f"[hl-decide] Reconciliation complete: {result}")

    except Exception as e:
        print(f"[hl-decide] Reconciliation failed: {e}")
        result["errors"] += 1

    return result


# Alias for backward compatibility
async def bootstrap_from_historical_fills() -> int:
    """Bootstrap episodes from historical fills. Alias for reconcile_historical_fills."""
    result = await reconcile_historical_fills(force=False)
    return result.get("new", 0)


async def periodic_reconciliation_task():
    """
    Background task that periodically reconciles historical fills.

    This catches any fills that were missed due to:
    - Service downtime
    - Hyperliquid API rate limits during real-time processing
    - Network issues

    Runs every RECONCILE_INTERVAL_HOURS (default 6 hours).
    """
    interval_seconds = RECONCILE_INTERVAL_HOURS * 3600

    while True:
        try:
            await asyncio.sleep(interval_seconds)

            print(f"[hl-decide] Starting periodic reconciliation (every {RECONCILE_INTERVAL_HOURS}h)...")
            result = await reconcile_historical_fills(force=False)

            if result["new"] > 0:
                print(f"[hl-decide] Periodic reconciliation: found {result['new']} new fills to process")
            else:
                print(f"[hl-decide] Periodic reconciliation: no new fills found")

        except asyncio.CancelledError:
            print("[hl-decide] Periodic reconciliation task cancelled")
            break
        except Exception as e:
            print(f"[hl-decide] Periodic reconciliation error: {e}")
            # Continue running despite errors
            await asyncio.sleep(60)  # Wait 1 minute before retrying


async def periodic_correlation_refresh_task():
    """
    Background task that periodically refreshes trader correlations.

    Runs daily (configurable via CORR_REFRESH_INTERVAL_HOURS) to keep
    the correlation matrix up-to-date with recent trading patterns.

    The correlation decay mechanism ensures stale data is gradually
    blended toward the default ρ=0.3, but refreshing keeps data fresh.
    """
    interval_seconds = CORR_REFRESH_INTERVAL_HOURS * 3600

    while True:
        try:
            await asyncio.sleep(interval_seconds)

            print(f"[hl-decide] Starting correlation refresh (every {CORR_REFRESH_INTERVAL_HOURS}h)...")
            summary = await run_daily_correlation_job(app.state.db)
            total_pairs = summary.get("btc_pairs", 0) + summary.get("eth_pairs", 0)

            if total_pairs > 0:
                # Reload and hydrate detector with fresh correlations
                corr_provider = get_correlation_provider()
                await corr_provider.load()
                hydrated = corr_provider.hydrate_detector(consensus_detector)

                # Update correlation observability metrics
                update_correlation_metrics(corr_provider)

                print(f"[hl-decide] Correlation refresh: computed {total_pairs} pairs, hydrated {hydrated}")
            else:
                print("[hl-decide] Correlation refresh: no pairs computed (insufficient data)")

        except asyncio.CancelledError:
            print("[hl-decide] Correlation refresh task cancelled")
            break
        except Exception as e:
            print(f"[hl-decide] Correlation refresh error: {e}")
            # Continue running despite errors
            await asyncio.sleep(300)  # Wait 5 minutes before retrying


async def periodic_exchange_health_task():
    """
    Background task that periodically checks exchange health and updates balances.

    Runs every EXCHANGE_HEALTH_CHECK_INTERVAL_MINUTES (default 5 minutes) to:
    - Probe each connected exchange with a balance request
    - Attempt reconnection if an exchange is disconnected or stale
    - Update exchange_balances table with latest balances
    - Update exchange_connections table with health status

    This ensures the system can detect and recover from:
    - Network interruptions
    - Exchange API outages
    - Session expiration
    """
    interval_seconds = EXCHANGE_HEALTH_CHECK_INTERVAL_MINUTES * 60

    while True:
        try:
            await asyncio.sleep(interval_seconds)

            if not hasattr(app.state, "exchange_manager"):
                continue

            exchange_manager = app.state.exchange_manager
            if not exchange_manager.connected_exchanges:
                # No exchanges configured - nothing to check
                continue

            # Run health check with reconnection attempts
            # Rate limiting: stagger requests to avoid hitting venue API limits
            result = await exchange_manager.health_check(
                testnet=EXCHANGE_TESTNET,
                stagger_delay_ms=EXCHANGE_HEALTH_STAGGER_DELAY_MS,
            )

            # Log results
            connected_count = sum(
                1 for ex, status in result.items()
                if isinstance(status, dict) and status.get("healthy")
            )
            total_count = sum(
                1 for ex, status in result.items()
                if isinstance(status, dict)
            )
            reconnected = result.get("reconnected", [])

            if reconnected:
                print(
                    f"[hl-decide] Exchange health check: {connected_count}/{total_count} healthy, "
                    f"reconnected: {reconnected}"
                )
            else:
                # Only log if there's something notable (not all healthy)
                if connected_count < total_count:
                    unhealthy = [
                        ex for ex, status in result.items()
                        if isinstance(status, dict) and not status.get("healthy")
                    ]
                    print(
                        f"[hl-decide] Exchange health check: {connected_count}/{total_count} healthy, "
                        f"unhealthy: {unhealthy}"
                    )

        except asyncio.CancelledError:
            print("[hl-decide] Exchange health check task cancelled")
            break
        except Exception as e:
            print(f"[hl-decide] Exchange health check error: {e}")
            # Continue running despite errors
            await asyncio.sleep(60)  # Wait 1 minute before retrying


def enforce_limits():
    """Enforce memory limits on scores and fills using LRU eviction."""
    while len(scores) > MAX_SCORES:
        scores.popitem(last=False)
    while len(fills) > MAX_FILLS:
        fills.popitem(last=False)


async def handle_score(msg):
    """Handle incoming score events from hl-sage."""
    data = ScoreEvent.model_validate_json(msg.data.decode())
    if data.address in scores:
        scores.move_to_end(data.address)
    scores[data.address] = data
    await persist_score(data.address, data)
    enforce_limits()


async def handle_fill(msg):
    """
    Handle incoming fill events from hl-stream.

    Processes fills for:
    1. Episode tracking (position lifecycle with VWAP and R-multiple calculation)
    2. Consensus detection (episode-based votes for Alpha Pool signals)
    """
    data = FillEvent.model_validate_json(msg.data.decode())

    # Update fill cache
    if data.address in fills:
        fills.move_to_end(data.address)
    fills[data.address] = data
    await persist_fill(data.address, data)
    enforce_limits()
    fill_counter.inc()

    # Process for episode tracking (new approach - tracks ALL fills)
    closed_episode = await handle_fill_via_episodes(data)

    # Process for consensus detection using episode-based votes
    await process_fill_for_consensus_via_episodes(data, closed_episode)


async def process_fill_for_consensus_via_episodes(data: FillEvent, closed_episode: Optional[Episode]) -> None:
    """
    Process a fill for consensus detection using episode-based votes.

    Key innovation: One vote per trader derived from their current episode state,
    not from individual fills. This means:
    - A trader with an open position = 1 vote in that direction
    - Multiple fills to the same position = still 1 vote
    - Position close = vote removed

    Args:
        data: FillEvent from hl-stream
        closed_episode: Episode that just closed (if any), for NIG update
    """
    try:
        # Update current price in detector
        price = float(data.price) if hasattr(data, 'price') and data.price else 0.0
        if price > 0:
            consensus_detector.set_current_price(data.asset, price)

        # Get all current open episodes for this asset
        open_episodes = [
            ep for ep in episode_tracker.get_all_open_episodes()
            if ep.asset.upper() == data.asset.upper()
        ]

        # Convert episodes to consensus fills (one per trader)
        # This is the key change: we derive votes from episodes, not raw fills
        episode_fills = []
        for ep in open_episodes:
            episode_fill = Fill(
                fill_id=f"episode-{ep.id}",
                address=ep.address,
                asset=ep.asset,
                side='buy' if ep.direction == 'long' else 'sell',
                size=ep.entry_size,
                price=ep.entry_vwap,
                ts=ep.entry_ts or datetime.now(timezone.utc),
            )
            episode_fills.append(episode_fill)

        # Check for consensus based on episode positions
        if len(episode_fills) >= 3:  # Minimum traders for consensus
            signal = await check_episode_consensus(data.asset, episode_fills)
            if signal:
                await handle_consensus_signal(signal)

    except Exception as e:
        print(f"[hl-decide] Episode consensus processing error: {e}")


async def check_episode_consensus(asset: str, episode_fills: list) -> Optional[ConsensusSignal]:
    """
    Check for consensus among episode-based votes.

    This replaces the fill-by-fill consensus checking with episode-based checking.
    Each episode represents one trader's current position = one vote.

    Uses centralized functions from consensus.py to avoid logic drift:
    - calculate_vote_weight() for vote weighting (log/equity modes)
    - ConsensusDetector.passes_latency_and_price_gates() for ATR-based R-unit drift check

    Now logs all decisions (signal, skip, risk_reject) for auditability.

    Args:
        asset: The asset to check consensus for
        episode_fills: List of Fill objects derived from open episodes

    Returns:
        ConsensusSignal if consensus detected, None otherwise
    """
    from .consensus import (
        passes_consensus_gates, calculate_ev, calculate_vote_weight,
        ConsensusWindow, Vote,
        CONSENSUS_MIN_TRADERS, CONSENSUS_MIN_AGREEING, CONSENSUS_MIN_PCT,
        CONSENSUS_MIN_EFFECTIVE_K, CONSENSUS_EV_MIN_R, CONSENSUS_BASE_WINDOW_S,
        CONSENSUS_MAX_STALENESS_FACTOR, CONSENSUS_MAX_PRICE_DRIFT_R,
    )
    import statistics

    # Track gate results for decision logging
    gate_results: list[GateResult] = []

    # Gate 0: Minimum traders check
    min_traders_passed = len(episode_fills) >= CONSENSUS_MIN_TRADERS
    gate_results.append(GateResult(
        name="min_traders",
        passed=min_traders_passed,
        value=float(len(episode_fills)),
        threshold=float(CONSENSUS_MIN_TRADERS),
    ))

    if not min_traders_passed:
        # Not enough traders - log skip decision
        await log_decision(
            db=app.state.db,
            symbol=asset,
            direction="none",
            decision_type="skip",
            trader_count=len(episode_fills),
            agreement_pct=0.0,
            effective_k=0.0,
            gates=gate_results,
            price=consensus_detector.get_current_mid(asset),
        )
        return None

    # One vote per trader (already deduplicated by episode)
    # Use centralized calculate_vote_weight() for proper log/equity weighting
    votes: list[Vote] = []
    for fill in episode_fills:
        direction = 'long' if fill.side.lower() in ('buy', 'long') else 'short'
        notional = fill.size * fill.price

        # Use centralized weight calculation (respects VOTE_WEIGHT_MODE: log/equity/linear)
        weight = calculate_vote_weight(notional, equity=None)

        votes.append(Vote(
            address=fill.address.lower(),
            direction=direction,
            weight=weight,
            price=fill.price,
            ts=fill.ts,
            notional=notional,
            equity=None,
        ))

    directions = [v.direction for v in votes]

    # Calculate agreement for logging
    long_count = sum(1 for d in directions if d == "long")
    short_count = len(directions) - long_count
    majority_count = max(long_count, short_count)
    agreement_pct = majority_count / len(directions) if directions else 0.0
    majority_dir = "long" if long_count >= short_count else "short"

    # Gate 1: Dispersion (supermajority)
    passes, _ = passes_consensus_gates(
        directions,
        min_agreeing=CONSENSUS_MIN_AGREEING,
        min_pct=CONSENSUS_MIN_PCT,
    )
    gate_results.append(GateResult(
        name="supermajority",
        passed=passes,
        value=agreement_pct,
        threshold=CONSENSUS_MIN_PCT,
    ))

    if not passes:
        await log_decision(
            db=app.state.db,
            symbol=asset,
            direction=majority_dir,
            decision_type="skip",
            trader_count=len(votes),
            agreement_pct=agreement_pct,
            effective_k=0.0,
            gates=gate_results,
            price=consensus_detector.get_current_mid(asset),
        )
        return None

    # Get agreeing votes
    agreeing_votes = [v for v in votes if v.direction == majority_dir]

    # Gate 2: Effective-K (correlation-adjusted)
    weights = {v.address: v.weight for v in agreeing_votes}
    eff_k = consensus_detector.eff_k_from_corr(
        weights,
        fallback_counter_callback=lambda: effk_default_fallback_counter.inc(),
    )

    # Record effK and weight metrics for observability
    effk_value_histogram.observe(eff_k)
    update_weight_metrics(list(weights.values()), asset=asset)

    effk_passed = eff_k >= CONSENSUS_MIN_EFFECTIVE_K
    gate_results.append(GateResult(
        name="effective_k",
        passed=effk_passed,
        value=eff_k,
        threshold=CONSENSUS_MIN_EFFECTIVE_K,
    ))

    if not effk_passed:
        await log_decision(
            db=app.state.db,
            symbol=asset,
            direction=majority_dir,
            decision_type="skip",
            trader_count=len(votes),
            agreement_pct=agreement_pct,
            effective_k=eff_k,
            gates=gate_results,
            price=consensus_detector.get_current_mid(asset),
        )
        return None

    # Calculate entry price (median of agreeing voters)
    median_entry = statistics.median(v.price for v in agreeing_votes)
    mid_price = consensus_detector.get_current_mid(asset)

    # Stop price using ATR-based dynamic stop (or fallback to 1%)
    stop_fraction = consensus_detector.get_stop_fraction(asset)
    stop_distance = median_entry * stop_fraction
    if majority_dir == 'long':
        stop_price = median_entry - stop_distance
    else:
        stop_price = median_entry + stop_distance

    # Gate 3a: ATR validity check
    is_atr_valid, atr_reason = consensus_detector.is_atr_valid_for_gating(asset)
    gate_results.append(GateResult(
        name="atr_validity",
        passed=is_atr_valid,
        value=1.0 if is_atr_valid else 0.0,
        threshold=1.0,
        detail=atr_reason,
    ))

    if not is_atr_valid:
        await log_decision(
            db=app.state.db,
            symbol=asset,
            direction=majority_dir,
            decision_type="skip",
            trader_count=len(votes),
            agreement_pct=agreement_pct,
            effective_k=eff_k,
            gates=gate_results,
            price=mid_price,
        )
        return None

    # Gate 3b: Latency check
    oldest_ts = min(v.ts for v in agreeing_votes)
    now = datetime.now(timezone.utc)
    staleness_s = (now - oldest_ts).total_seconds()
    max_staleness = CONSENSUS_BASE_WINDOW_S * CONSENSUS_MAX_STALENESS_FACTOR
    latency_passed = staleness_s <= max_staleness

    gate_results.append(GateResult(
        name="freshness",
        passed=latency_passed,
        value=staleness_s,
        threshold=max_staleness,
    ))

    if not latency_passed:
        await log_decision(
            db=app.state.db,
            symbol=asset,
            direction=majority_dir,
            decision_type="skip",
            trader_count=len(votes),
            agreement_pct=agreement_pct,
            effective_k=eff_k,
            gates=gate_results,
            price=mid_price,
        )
        return None

    # Gate 3c: Price band check (ATR-based R-units)
    if median_entry > 0 and mid_price > 0:
        bps_deviation = abs(mid_price - median_entry) / median_entry * 10000
        stop_bps = stop_fraction * 10000
        deviation_r = bps_deviation / stop_bps if stop_bps > 0 else 0
        price_band_passed = deviation_r <= CONSENSUS_MAX_PRICE_DRIFT_R
    else:
        deviation_r = 0
        price_band_passed = False

    gate_results.append(GateResult(
        name="price_band",
        passed=price_band_passed,
        value=deviation_r,
        threshold=CONSENSUS_MAX_PRICE_DRIFT_R,
    ))

    if not price_band_passed:
        await log_decision(
            db=app.state.db,
            symbol=asset,
            direction=majority_dir,
            decision_type="skip",
            trader_count=len(votes),
            agreement_pct=agreement_pct,
            effective_k=eff_k,
            gates=gate_results,
            price=mid_price,
        )
        return None

    # Gate 4: EV after costs
    p_win = consensus_detector.calibrated_p_win(agreeing_votes, eff_k)
    ev_result = calculate_ev(
        p_win=p_win,
        entry_px=median_entry,
        stop_px=stop_price,
    )

    ev_passed = ev_result["ev_net_r"] >= CONSENSUS_EV_MIN_R
    gate_results.append(GateResult(
        name="ev_gate",
        passed=ev_passed,
        value=ev_result["ev_net_r"],
        threshold=CONSENSUS_EV_MIN_R,
    ))

    if not ev_passed:
        await log_decision(
            db=app.state.db,
            symbol=asset,
            direction=majority_dir,
            decision_type="skip",
            trader_count=len(votes),
            agreement_pct=agreement_pct,
            effective_k=eff_k,
            gates=gate_results,
            price=mid_price,
            confidence=p_win,
            ev=ev_result["ev_net_r"],
        )
        return None

    # All consensus gates passed! Create signal
    latency_ms = int((now - oldest_ts).total_seconds() * 1000)
    mid_delta_bps = abs(mid_price - median_entry) / median_entry * 10000 if median_entry > 0 else 0

    # Calculate dispersion
    signed_weights = [(1 if v.direction == 'long' else -1) * v.weight for v in votes]
    dispersion = statistics.stdev(signed_weights) if len(signed_weights) > 1 else 0.0

    signal = ConsensusSignal(
        id=str(uuid4()),
        symbol=asset,
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
        trigger_addresses=[v.address for v in agreeing_votes],
    )

    # Gate 5: Risk limits fail-safe (regime-aware)
    from .consensus import check_risk_limits

    # Get current regime for regime-adjusted confidence threshold
    current_regime = None
    try:
        regime_detector = get_regime_detector(app.state.db)
        regime_analysis = await regime_detector.detect_regime(asset)
        if regime_analysis:
            current_regime = regime_analysis.regime.value
    except Exception:
        pass  # Fall back to static threshold

    passes_risk, risk_reason = check_risk_limits(signal, regime=current_regime)

    gate_results.append(GateResult(
        name="risk_limits",
        passed=passes_risk,
        value=p_win if not passes_risk else 1.0,
        threshold=1.0,
        detail=risk_reason if not passes_risk else "",
    ))

    if not passes_risk:
        signal_risk_rejected_counter.labels(reason=risk_reason.split()[0]).inc()
        print(f"[consensus] Signal rejected by risk limits: {risk_reason}")

        # Log as risk_reject
        await log_decision(
            db=app.state.db,
            symbol=asset,
            direction=majority_dir,
            decision_type="risk_reject",
            trader_count=len(votes),
            agreement_pct=agreement_pct,
            effective_k=eff_k,
            gates=gate_results,
            risk_checks=[{"name": "risk_limits", "passed": False, "reason": risk_reason}],
            price=mid_price,
            confidence=p_win,
            ev=ev_result["ev_net_r"],
        )
        return None

    # All gates passed including risk limits - log as signal
    signal_generated_counter.labels(symbol=asset, direction=majority_dir).inc()

    decision_id = await log_decision(
        db=app.state.db,
        symbol=asset,
        direction=majority_dir,
        decision_type="signal",
        trader_count=len(votes),
        agreement_pct=agreement_pct,
        effective_k=eff_k,
        gates=gate_results,
        price=mid_price,
        confidence=p_win,
        ev=ev_result["ev_net_r"],
    )

    # Attempt execution if auto-trading is enabled
    from .executor import maybe_execute_signal
    await maybe_execute_signal(
        db=app.state.db,
        decision_id=decision_id,
        symbol=asset,
        direction=majority_dir,
    )

    return signal


async def handle_consensus_signal(signal: ConsensusSignal) -> None:
    """
    Handle a detected consensus signal.

    Publishes to NATS and persists to database.

    Args:
        signal: The consensus signal to process
    """
    try:
        consensus_signal_counter.inc()

        # Log signal
        print(f"[hl-decide] CONSENSUS SIGNAL: {signal.direction} {signal.symbol} "
              f"@ {signal.entry_price:.2f}, effK={signal.eff_k:.2f}, EV={signal.ev_net_r:.3f}R, "
              f"traders={signal.n_agreeing}/{signal.n_traders}")

        # Build signal payload for NATS
        signal_payload = {
            "id": signal.id,
            "symbol": signal.symbol,
            "direction": signal.direction,
            "entry_price": signal.entry_price,
            "stop_price": signal.stop_price,
            "n_traders": signal.n_traders,
            "n_agreeing": signal.n_agreeing,
            "eff_k": signal.eff_k,
            "dispersion": signal.dispersion,
            "p_win": signal.p_win,
            "ev_gross_r": signal.ev_gross_r,
            "ev_cost_r": signal.ev_cost_r,
            "ev_net_r": signal.ev_net_r,
            "latency_ms": signal.latency_ms,
            "median_voter_price": signal.median_voter_price,
            "mid_delta_bps": signal.mid_delta_bps,
            "created_at": signal.created_at.isoformat(),
            "trigger_addresses": signal.trigger_addresses,
        }

        # Publish to NATS
        await app.state.js.publish(
            "d.signals.v1",
            json.dumps(signal_payload).encode("utf-8"),
        )

        # Persist to database
        async with app.state.db.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO consensus_signals (
                    id, symbol, direction, entry_price, stop_price,
                    n_traders, n_agreeing, eff_k, dispersion,
                    p_win, ev_gross_r, ev_cost_r, ev_net_r,
                    latency_ms, median_voter_price, mid_delta_bps,
                    trigger_addresses, created_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9,
                    $10, $11, $12, $13, $14, $15, $16, $17, $18
                )
                ON CONFLICT (id) DO NOTHING
                """,
                signal.id,
                signal.symbol,
                signal.direction,
                signal.entry_price,
                signal.stop_price,
                signal.n_traders,
                signal.n_agreeing,
                signal.eff_k,
                signal.dispersion,
                signal.p_win,
                signal.ev_gross_r,
                signal.ev_cost_r,
                signal.ev_net_r,
                signal.latency_ms,
                signal.median_voter_price,
                signal.mid_delta_bps,
                signal.trigger_addresses,
                signal.created_at,
            )

    except Exception as e:
        print(f"[hl-decide] Failed to handle consensus signal: {e}")


@app.get("/healthz")
async def health():
    """Health check endpoint."""
    try:
        async with app.state.db.acquire() as conn:
            open_positions = await conn.fetchval(
                "SELECT COUNT(*) FROM position_signals WHERE status = 'open'"
            )
        return {
            "status": "ok",
            "scores": len(scores),
            "fills": len(fills),
            "open_positions": open_positions,
        }
    except Exception:
        return {"status": "ok", "scores": len(scores), "fills": len(fills)}


@app.get("/data-health")
async def data_health():
    """
    Data freshness and health status endpoint.
    Returns warnings for stale data that could affect signal quality.
    Used for observability dashboards and alerting.
    """
    from datetime import datetime, timezone

    warnings = []
    status = "healthy"

    # ATR staleness check
    atr_health = {"btc": {"status": "unknown"}, "eth": {"status": "unknown"}}
    if hasattr(app.state, "atr_provider"):
        provider = app.state.atr_provider
        for asset in ["BTC", "ETH"]:
            atr_data = provider.get_atr(asset)
            if atr_data:
                age_seconds = atr_data.get("age_seconds", 0)
                source = atr_data.get("source", "unknown")
                atr_health[asset.lower()] = {
                    "status": "stale" if age_seconds > ATR_MAX_STALENESS else "fresh",
                    "age_seconds": age_seconds,
                    "source": source,
                    "value": atr_data.get("atr"),
                }
                if age_seconds > ATR_MAX_STALENESS:
                    warnings.append(f"ATR for {asset} is stale ({age_seconds}s old, max {ATR_MAX_STALENESS}s)")
                    status = "degraded"
            else:
                atr_health[asset.lower()] = {"status": "missing"}
                warnings.append(f"No ATR data available for {asset}")
                status = "degraded"

    # Correlation health check
    corr_health = {"status": "unknown", "coverage_pct": 0, "pairs_loaded": 0}
    if hasattr(app.state, "corr_provider"):
        provider = app.state.corr_provider
        pairs_loaded = len(provider.correlations) if hasattr(provider, "correlations") else 0
        pool_size = getattr(provider, "pool_size", 0)

        # Calculate coverage
        expected_pairs = pool_size * (pool_size - 1) // 2 if pool_size > 1 else 0
        coverage_pct = (pairs_loaded / expected_pairs * 100) if expected_pairs > 0 else 0

        corr_health = {
            "status": "healthy" if coverage_pct >= 50 else ("degraded" if coverage_pct > 0 else "missing"),
            "coverage_pct": round(coverage_pct, 1),
            "pairs_loaded": pairs_loaded,
            "pool_size": pool_size,
            "expected_pairs": expected_pairs,
        }

        if coverage_pct < 50:
            warnings.append(f"Correlation coverage low ({coverage_pct:.1f}%, using default ρ=0.3 for missing pairs)")
            if status == "healthy":
                status = "degraded"

    # Weight concentration check (from recent consensus)
    weight_health = {"gini": None, "saturation_pct": None}
    try:
        # Get last Gini from metrics if available
        gini_value = weight_gini_gauge._value.get() if weight_gini_gauge._value else None
        sat_pct = weight_saturation_pct_gauge._value.get() if weight_saturation_pct_gauge._value else None
        weight_health = {
            "gini": round(gini_value, 3) if gini_value is not None else None,
            "saturation_pct": round(sat_pct, 1) if sat_pct is not None else None,
        }
        if gini_value is not None and gini_value > 0.8:
            warnings.append(f"Vote weight concentration high (Gini={gini_value:.2f})")
            if status == "healthy":
                status = "degraded"
    except Exception:
        pass

    return {
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "atr": atr_health,
        "correlation": corr_health,
        "weight_distribution": weight_health,
        "warnings": warnings,
        "warning_count": len(warnings),
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)


@app.post("/reconcile")
async def reconcile_fills(force: bool = False):
    """
    Manually trigger reconciliation of historical fills.
    Safe to run multiple times - tracks processed fill IDs to prevent double-booking.

    Args:
        force: If True, reprocess all fills even if already processed
    """
    result = await reconcile_historical_fills(force=force)
    return result


@app.get("/positions/open")
async def get_open_positions():
    """Get all currently open positions being tracked."""
    try:
        async with app.state.db.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT address, asset, direction, entry_price, entry_size, entry_ts
                FROM position_signals
                WHERE status = 'open'
                ORDER BY entry_ts DESC
                LIMIT 100
                """
            )
            return {
                "count": len(rows),
                "positions": [
                    {
                        "address": row["address"],
                        "asset": row["asset"],
                        "direction": row["direction"],
                        "entry_price": float(row["entry_price"]),
                        "entry_size": float(row["entry_size"]),
                        "entry_ts": row["entry_ts"].isoformat(),
                    }
                    for row in rows
                ]
            }
    except Exception as e:
        return {"error": str(e)}


@app.get("/positions/recent")
async def get_recent_closed_positions():
    """Get recently closed positions with their R-multiples."""
    try:
        async with app.state.db.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT address, asset, direction, entry_price, exit_price,
                       result_r, realized_pnl, entry_ts, exit_ts
                FROM position_signals
                WHERE status = 'closed'
                ORDER BY exit_ts DESC
                LIMIT 50
                """
            )
            return {
                "count": len(rows),
                "positions": [
                    {
                        "address": row["address"],
                        "asset": row["asset"],
                        "direction": row["direction"],
                        "entry_price": float(row["entry_price"]),
                        "exit_price": float(row["exit_price"]) if row["exit_price"] else None,
                        "result_r": float(row["result_r"]) if row["result_r"] else None,
                        "realized_pnl": float(row["realized_pnl"]) if row["realized_pnl"] else None,
                        "entry_ts": row["entry_ts"].isoformat(),
                        "exit_ts": row["exit_ts"].isoformat() if row["exit_ts"] else None,
                    }
                    for row in rows
                ]
            }
    except Exception as e:
        return {"error": str(e)}


# =====================
# Consensus Signal API
# =====================


@app.get("/consensus/signals")
async def get_consensus_signals(limit: int = 20):
    """
    Get recent consensus signals for the Alpha Pool tab.

    Returns signals with their metrics and outcomes.
    """
    try:
        async with app.state.db.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, symbol, direction, entry_price, stop_price,
                       n_traders, n_agreeing, eff_k, dispersion,
                       p_win, ev_gross_r, ev_cost_r, ev_net_r,
                       latency_ms, median_voter_price, mid_delta_bps,
                       trigger_addresses, created_at,
                       outcome, exit_price, result_r, closed_at
                FROM consensus_signals
                ORDER BY created_at DESC
                LIMIT $1
                """,
                limit,
            )
            return {
                "count": len(rows),
                "signals": [
                    {
                        "id": str(row["id"]),
                        "symbol": row["symbol"],
                        "direction": row["direction"],
                        "entry_price": float(row["entry_price"]),
                        "stop_price": float(row["stop_price"]) if row["stop_price"] else None,
                        "n_traders": row["n_traders"],
                        "n_agreeing": row["n_agreeing"],
                        "eff_k": float(row["eff_k"]),
                        "dispersion": float(row["dispersion"]) if row["dispersion"] else None,
                        "p_win": float(row["p_win"]),
                        "ev_gross_r": float(row["ev_gross_r"]),
                        "ev_cost_r": float(row["ev_cost_r"]),
                        "ev_net_r": float(row["ev_net_r"]),
                        "latency_ms": row["latency_ms"],
                        "created_at": row["created_at"].isoformat(),
                        "outcome": row["outcome"],
                        "exit_price": float(row["exit_price"]) if row["exit_price"] else None,
                        "result_r": float(row["result_r"]) if row["result_r"] else None,
                        "closed_at": row["closed_at"].isoformat() if row["closed_at"] else None,
                    }
                    for row in rows
                ]
            }
    except Exception as e:
        return {"error": str(e), "count": 0, "signals": []}


@app.get("/consensus/stats")
async def get_consensus_stats():
    """
    Get aggregate statistics for consensus signals.
    """
    try:
        async with app.state.db.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) as total_signals,
                    COUNT(*) FILTER (WHERE outcome = 'win') as wins,
                    COUNT(*) FILTER (WHERE outcome = 'loss') as losses,
                    COUNT(*) FILTER (WHERE outcome IS NOT NULL) as closed,
                    AVG(eff_k) as avg_eff_k,
                    AVG(ev_net_r) as avg_ev_net_r,
                    AVG(result_r) FILTER (WHERE outcome IS NOT NULL) as avg_result_r
                FROM consensus_signals
                """
            )
            win_rate = (row["wins"] / row["closed"] * 100) if row["closed"] > 0 else 0

            return {
                "total_signals": row["total_signals"],
                "closed": row["closed"],
                "wins": row["wins"],
                "losses": row["losses"],
                "win_rate": round(win_rate, 1),
                "avg_eff_k": round(float(row["avg_eff_k"] or 0), 2),
                "avg_ev_net_r": round(float(row["avg_ev_net_r"] or 0), 3),
                "avg_result_r": round(float(row["avg_result_r"] or 0), 3),
            }
    except Exception as e:
        return {"error": str(e)}


@app.post("/correlation/compute")
async def compute_correlations():
    """
    Trigger daily correlation computation job.

    This computes pairwise correlations between Alpha Pool traders
    based on their position posture (direction) in 5-minute buckets.
    Results are stored in trader_corr table.
    """
    try:
        summary = await run_daily_correlation_job(app.state.db)

        # Reload correlations into provider and hydrate detector
        corr_provider = get_correlation_provider()
        corr_count = await corr_provider.load()
        hydrated = corr_provider.hydrate_detector(consensus_detector)

        return {
            "status": "ok",
            "date": summary["date"],
            "btc_pairs": summary["btc_pairs"],
            "eth_pairs": summary["eth_pairs"],
            "pruned": summary["pruned"],
            "loaded": corr_count,
            "hydrated": hydrated,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/correlation/status")
async def get_correlation_status():
    """
    Get correlation matrix status.

    Returns counts of loaded correlations and sample pairs.
    """
    try:
        corr_provider = get_correlation_provider()

        async with app.state.db.acquire() as conn:
            # Get latest date with correlations
            latest_row = await conn.fetchrow(
                """
                SELECT as_of_date, COUNT(*) as pair_count
                FROM trader_corr
                GROUP BY as_of_date
                ORDER BY as_of_date DESC
                LIMIT 1
                """
            )

            # Get sample pairs
            sample_rows = await conn.fetch(
                """
                SELECT addr_a, addr_b, rho, n_buckets
                FROM trader_corr
                WHERE as_of_date = (SELECT MAX(as_of_date) FROM trader_corr)
                ORDER BY rho DESC
                LIMIT 10
                """
            )

        return {
            "loaded_pairs": len(corr_provider.correlations),
            "latest_date": latest_row["as_of_date"].isoformat() if latest_row else None,
            "db_pair_count": latest_row["pair_count"] if latest_row else 0,
            "detector_pairs": len(consensus_detector.correlation_matrix),
            "sample_highest_correlations": [
                {
                    "addr_a": row["addr_a"][:10] + "...",
                    "addr_b": row["addr_b"][:10] + "...",
                    "rho": round(float(row["rho"]), 3),
                    "n_buckets": row["n_buckets"],
                }
                for row in sample_rows
            ],
        }
    except Exception as e:
        return {"error": str(e)}


# =====================
# Decision Logging API
# =====================


@app.get("/decisions")
async def list_decisions(
    symbol: Optional[str] = None,
    decision_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """
    List decision logs with optional filters.

    Args:
        symbol: Filter by symbol (BTC, ETH)
        decision_type: Filter by type (signal, skip, risk_reject)
        limit: Max results (default 50)
        offset: Pagination offset

    Returns:
        Paginated list of decisions with reasoning
    """
    return await get_decisions(
        db=app.state.db,
        symbol=symbol,
        decision_type=decision_type,
        limit=limit,
        offset=offset,
    )


@app.get("/decisions/stats")
async def decision_stats(days: int = 7):
    """
    Get aggregate statistics for decisions.

    Args:
        days: Number of days to look back (default 7)

    Returns:
        Aggregate stats including signal count, skip rate, win rate
    """
    return await get_decision_stats(db=app.state.db, days=days)


@app.get("/decisions/{decision_id}")
async def get_decision_by_id(decision_id: str):
    """
    Get full details for a single decision.

    Args:
        decision_id: The decision log UUID

    Returns:
        Full decision details including gates, reasoning, and outcome
    """
    result = await get_decision(db=app.state.db, decision_id=decision_id)
    if result is None:
        return {"error": "Decision not found"}
    return result


# =====================
# Portfolio & Execution API
# =====================


@app.get("/portfolio")
async def portfolio_summary(address: Optional[str] = None):
    """
    Get portfolio summary including account value and positions.

    Currently supports Hyperliquid only. Multi-exchange in Phase 4.

    Args:
        address: Optional Hyperliquid address (falls back to config)

    Returns:
        Portfolio summary with equity, positions, and P&L
    """
    return await get_portfolio_summary(db=app.state.db, address=address)


@app.get("/portfolio/positions")
async def list_positions():
    """
    Get live positions from database.

    Returns cached positions from last portfolio sync.
    """
    try:
        async with app.state.db.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT exchange, symbol, side, size, entry_price, mark_price,
                       liquidation_price, unrealized_pnl, margin_used, leverage,
                       opened_at, updated_at
                FROM live_positions
                ORDER BY updated_at DESC
                """
            )
            return {
                "count": len(rows),
                "positions": [
                    {
                        "exchange": row["exchange"],
                        "symbol": row["symbol"],
                        "side": row["side"],
                        "size": float(row["size"]),
                        "entry_price": float(row["entry_price"]),
                        "mark_price": float(row["mark_price"]) if row["mark_price"] else None,
                        "liquidation_price": float(row["liquidation_price"]) if row["liquidation_price"] else None,
                        "unrealized_pnl": float(row["unrealized_pnl"]) if row["unrealized_pnl"] else None,
                        "margin_used": float(row["margin_used"]) if row["margin_used"] else None,
                        "leverage": row["leverage"],
                        "opened_at": row["opened_at"].isoformat() if row["opened_at"] else None,
                        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
                    }
                    for row in rows
                ],
            }
    except Exception as e:
        return {"error": str(e), "count": 0, "positions": []}


@app.get("/execution/config")
async def execution_config():
    """
    Get current execution configuration.

    Returns auto-trade settings and risk limits.
    """
    return await get_execution_config(db=app.state.db)


@app.post("/execution/config")
async def update_config(config: ExecutionConfigUpdate):
    """
    Update execution configuration.

    Requires owner authentication.

    Args:
        config: ExecutionConfigUpdate JSON body containing any of:
            - enabled: Master enable/disable for auto-trading
            - hl_enabled: Enable Hyperliquid auto-trading
            - hl_address: Hyperliquid wallet address
            - hl_max_leverage: Maximum leverage (1-10)
            - hl_max_position_pct: Max position size as % of equity (0-100)
            - hl_max_exposure_pct: Max total exposure as % of equity (0-100)

    Returns:
        Updated config
    """
    return await update_execution_config(
        db=app.state.db,
        enabled=config.enabled,
        hl_enabled=config.hl_enabled,
        hl_address=config.hl_address,
        hl_max_leverage=config.hl_max_leverage,
        hl_max_position_pct=config.hl_max_position_pct,
        hl_max_exposure_pct=config.hl_max_exposure_pct,
    )


@app.get("/execution/logs")
async def execution_logs(limit: int = 50, offset: int = 0):
    """
    Get recent execution logs.

    Shows all trade execution attempts with results.

    Args:
        limit: Max results (default 50)
        offset: Pagination offset

    Returns:
        Paginated execution logs
    """
    return await get_execution_logs(db=app.state.db, limit=limit, offset=offset)


# ============================================================================
# Regime Detection Endpoints
# ============================================================================


@app.get("/regime/{asset}")
async def get_market_regime(asset: str):
    """
    Get current market regime for an asset.

    Detects market regime (trending, ranging, volatile) and returns
    recommended parameter adjustments for strategy execution.

    Args:
        asset: Asset symbol (BTC, ETH)

    Returns:
        {
            "asset": "BTC",
            "regime": "trending",
            "confidence": 0.85,
            "params": {
                "stop_multiplier": 1.2,
                "kelly_multiplier": 1.0,
                "min_confidence_adjustment": 0.0,
                "max_position_fraction": 1.0
            },
            "signals": {
                "ma_spread_pct": 0.025,
                "volatility_ratio": 1.1,
                "price_range_pct": 0.02
            },
            "candles_used": 60,
            "timestamp": "2025-12-13T00:00:00Z"
        }
    """
    asset_upper = asset.upper()
    if asset_upper not in ("BTC", "ETH"):
        return {"error": f"Unsupported asset: {asset}. Use BTC or ETH."}

    analysis = await detect_market_regime(asset_upper, db=app.state.db)
    return analysis.to_dict()


@app.get("/regime")
async def get_all_regimes():
    """
    Get market regimes for all tracked assets.

    Returns regimes for BTC and ETH.

    Returns:
        {
            "BTC": { regime analysis },
            "ETH": { regime analysis },
            "summary": {
                "consensus_regime": "volatile" | "mixed",
                "assets_volatile": 0,
                "assets_trending": 1,
                "assets_ranging": 1
            }
        }
    """
    btc_analysis = await detect_market_regime("BTC", db=app.state.db)
    eth_analysis = await detect_market_regime("ETH", db=app.state.db)

    # Calculate consensus regime
    regimes = [btc_analysis.regime, eth_analysis.regime]
    regime_counts = {
        "volatile": sum(1 for r in regimes if r == MarketRegime.VOLATILE),
        "trending": sum(1 for r in regimes if r == MarketRegime.TRENDING),
        "ranging": sum(1 for r in regimes if r == MarketRegime.RANGING),
        "unknown": sum(1 for r in regimes if r == MarketRegime.UNKNOWN),
    }

    # Consensus: if any volatile, overall is volatile; otherwise majority wins
    if regime_counts["volatile"] > 0:
        consensus = "volatile"
    elif regime_counts["trending"] > regime_counts["ranging"]:
        consensus = "trending"
    elif regime_counts["ranging"] > regime_counts["trending"]:
        consensus = "ranging"
    else:
        consensus = "mixed"

    return {
        "BTC": btc_analysis.to_dict(),
        "ETH": eth_analysis.to_dict(),
        "summary": {
            "consensus_regime": consensus,
            "assets_volatile": regime_counts["volatile"],
            "assets_trending": regime_counts["trending"],
            "assets_ranging": regime_counts["ranging"],
        },
    }


@app.get("/regime/params")
async def get_regime_params():
    """
    Get regime parameter presets.

    Shows the strategy parameter adjustments for each regime type.

    Returns:
        Parameter presets for all regime types
    """
    return {
        regime.value: {
            "stop_multiplier": params.stop_multiplier,
            "kelly_multiplier": params.kelly_multiplier,
            "min_confidence_adjustment": params.min_confidence_adjustment,
            "max_position_fraction": params.max_position_fraction,
            "description": params.description,
        }
        for regime, params in REGIME_PARAMS.items()
    }
