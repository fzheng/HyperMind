"""
HL-Sage Service

Scores and weights candidate addresses based on their trading performance.
Consumes candidate events from hl-scout and fill events from hl-stream,
produces score events for hl-decide to generate trading signals.

Key responsibilities:
- Consume `a.candidates.v1` events from NATS
- Track position changes from `c.fills.v1` events
- Compute scores using NIG (Normal-Inverse-Gamma) Thompson Sampling
- Publish `b.scores.v1` events for downstream signal generation
- Persist state to PostgreSQL for recovery after restarts

## Alpha Pool (Decoupled System)

The Alpha Pool is a standalone trader selection system, fully decoupled from
the legacy leaderboard in hl-scout. Key differences:

| Aspect          | Legacy Leaderboard    | Alpha Pool             |
|-----------------|-----------------------|------------------------|
| Data source     | hl_leaderboard_*      | alpha_pool_addresses   |
| Address refresh | hl-scout daily sync   | /alpha-pool/refresh API|
| PnL curves      | Stored in DB          | Fetched from HL API    |
| Selection       | Rank-based            | NIG Thompson Sampling  |

To populate the Alpha Pool:
1. Call `POST /alpha-pool/refresh` to fetch traders from Hyperliquid
2. Addresses are stored in `alpha_pool_addresses` table
3. `/alpha-pool` API returns traders with NIG posteriors and PnL curves

@module hl-sage
"""

import os
import json
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional
from collections import OrderedDict

import asyncpg
import httpx
import nats
from fastapi import FastAPI, HTTPException, Query
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
from starlette.responses import Response

from contracts.py.models import CandidateEvent, ScoreEvent, FillEvent
from .bandit import (
    get_bandit_status,
    get_bandit_status_nig,
    select_traders_with_exploration,
    thompson_sample_select,
    thompson_sample_select_nig,
    get_trader_posteriors,
    get_trader_posteriors_nig,
    apply_decay,
    BANDIT_SELECT_K,
    BANDIT_POOL_SIZE,
)
from .snapshot import (
    create_daily_snapshot,
    get_snapshot_summary,
    load_universe_at_date,
    SELECTION_VERSION,
)
from .walkforward import (
    run_walk_forward_replay,
    replay_single_period,
    format_replay_summary,
    REPLAY_EVALUATION_DAYS,
)

SERVICE_NAME = "hl-sage"
OWNER_TOKEN = os.getenv("OWNER_TOKEN", "dev-owner")
NATS_URL = os.getenv("NATS_URL", "nats://0.0.0.0:4222")
HL_STREAM_URL = os.getenv("HL_STREAM_URL", "http://hl-stream:8080")
DB_URL = os.getenv("DATABASE_URL", "postgresql://hlbot:hlbotpassword@0.0.0.0:5432/hlbot")
MAX_TRACKED_ADDRESSES = int(os.getenv("MAX_TRACKED_ADDRESSES", "1000"))
MAX_SCORES = int(os.getenv("MAX_SCORES", "500"))
STALE_THRESHOLD_HOURS = int(os.getenv("STALE_THRESHOLD_HOURS", "24"))

# Daily snapshot configuration
SNAPSHOT_ENABLED = os.getenv("SNAPSHOT_ENABLED", "true").lower() == "true"
SNAPSHOT_HOUR_UTC = int(os.getenv("SNAPSHOT_HOUR_UTC", "0"))  # Default: midnight UTC


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Startup
    try:
        # Connect to database first
        app.state.db = await asyncpg.create_pool(DB_URL)

        # Restore tracked addresses from database
        restored = await restore_tracked_addresses()
        print(f"[hl-sage] Restored {restored} tracked addresses from database")

        # Connect to NATS
        app.state.nc = await nats.connect(NATS_URL)
        app.state.js = app.state.nc.jetstream()
        await ensure_stream(app.state.js, "HL_B", ["b.scores.v1"])
        await app.state.nc.subscribe("a.candidates.v1", cb=handle_candidate)
        await app.state.nc.subscribe("c.fills.v1", cb=handle_fill)

        # Auto-refresh Alpha Pool if empty on startup
        if ALPHA_POOL_AUTO_REFRESH:
            asyncio.create_task(auto_refresh_alpha_pool_if_empty())

        # Start periodic fill sync for Alpha Pool addresses
        asyncio.create_task(periodic_alpha_pool_fill_sync())

        # Start subscription sync to register Alpha Pool addresses with hl-stream
        asyncio.create_task(sync_alpha_pool_subscriptions())

        # Start periodic Alpha Pool refresh (every ALPHA_POOL_REFRESH_HOURS)
        asyncio.create_task(periodic_alpha_pool_refresh())

        # Start daily snapshot job (Phase 3f: Shadow Ledger)
        if SNAPSHOT_ENABLED:
            asyncio.create_task(periodic_daily_snapshot())
    except Exception as e:
        print(f"[hl-sage] Fatal startup error: {e}")
        raise

    yield  # Application runs here

    # Shutdown
    if hasattr(app.state, "nc"):
        await app.state.nc.drain()
    if hasattr(app.state, "db"):
        await app.state.db.close()


app = FastAPI(title="hl-sage", version="0.1.0", lifespan=lifespan)

# Use OrderedDict for LRU behavior
scores: OrderedDict[str, ScoreEvent] = OrderedDict()
tracked_addresses: OrderedDict[str, Dict[str, Any]] = OrderedDict()

registry = CollectorRegistry()
candidate_counter = Counter(
    "sage_candidates_total", "Number of candidate messages processed", registry=registry
)
score_counter = Counter(
    "sage_scores_total", "Number of scores published", registry=registry
)
score_latency = Histogram(
    "sage_score_latency_seconds", "Latency to process a candidate", registry=registry, buckets=(0.01, 0.05, 0.1, 0.5)
)


async def ensure_stream(js, name: str, subjects: List[str]) -> None:
    """
    Ensures a NATS JetStream stream exists, creating it if necessary.

    Args:
        js: JetStream client
        name: Stream name
        subjects: List of subject patterns to capture
    """
    try:
        await js.stream_info(name)
    except Exception:
        await js.add_stream(name=name, subjects=subjects)


async def persist_tracked_address(address: str, state: Dict[str, Any]) -> None:
    """Persist tracked address state to database for recovery."""
    try:
        async with app.state.db.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO sage_tracked_addresses (address, weight, rank, period, position, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (address) DO UPDATE SET
                    weight = EXCLUDED.weight,
                    rank = EXCLUDED.rank,
                    period = EXCLUDED.period,
                    position = EXCLUDED.position,
                    updated_at = EXCLUDED.updated_at
                """,
                address,
                state["weight"],
                state["rank"],
                state["period"],
                state["position"],
                state["updated"],
            )
    except Exception as e:
        print(f"[hl-sage] Failed to persist tracked address {address}: {e}")


async def restore_tracked_addresses() -> int:
    """Restore tracked addresses from database on startup."""
    try:
        async with app.state.db.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT address, weight, rank, period, position, updated_at
                FROM sage_tracked_addresses
                WHERE updated_at > NOW() - INTERVAL '24 hours'
                ORDER BY updated_at DESC
                LIMIT $1
                """,
                MAX_TRACKED_ADDRESSES,
            )
            for row in rows:
                addr = row["address"].lower()
                tracked_addresses[addr] = {
                    "weight": float(row["weight"]),
                    "rank": int(row["rank"]),
                    "period": int(row["period"]),
                    "position": float(row["position"]),
                    "updated": row["updated_at"],
                }
            return len(rows)
    except Exception as e:
        print(f"[hl-sage] Failed to restore tracked addresses: {e}")
        return 0


def evict_stale_entries():
    """Remove stale entries to prevent unbounded memory growth."""
    now = datetime.now(timezone.utc)
    stale_cutoff = now - timedelta(hours=STALE_THRESHOLD_HOURS)

    # Remove stale tracked addresses
    stale_addrs = [
        addr for addr, data in tracked_addresses.items()
        if data.get("updated", now) < stale_cutoff
    ]
    for addr in stale_addrs:
        tracked_addresses.pop(addr, None)

    # Enforce max limits using LRU (OrderedDict maintains insertion order)
    while len(tracked_addresses) > MAX_TRACKED_ADDRESSES:
        tracked_addresses.popitem(last=False)  # Remove oldest

    while len(scores) > MAX_SCORES:
        scores.popitem(last=False)  # Remove oldest


async def handle_candidate(msg):
    """
    Handles incoming candidate events from hl-scout.
    Extracts leaderboard weight and rank, stores in tracked_addresses.

    Args:
        msg: NATS message containing CandidateEvent JSON
    """
    with score_latency.time():
        data = CandidateEvent.model_validate_json(msg.data.decode())
        candidate_counter.inc()
        leaderboard_meta = (data.meta.get("leaderboard") if isinstance(data.meta, dict) else None) or {}
        weight = float(leaderboard_meta.get("weight") or data.score_hint or 0.1)
        weight = max(0.05, min(1.0, weight))
        rank = int(leaderboard_meta.get("rank") or 999)
        period = int(leaderboard_meta.get("period_days") or 30)

        addr_lower = data.address.lower()
        # Move to end (most recently used)
        if addr_lower in tracked_addresses:
            tracked_addresses.move_to_end(addr_lower)

        state = {
            "weight": weight,
            "rank": rank,
            "period": period,
            "position": 0.0,
            "updated": datetime.now(timezone.utc),
        }
        tracked_addresses[addr_lower] = state

        # Persist to database for recovery
        await persist_tracked_address(addr_lower, state)

        evict_stale_entries()


async def handle_fill(msg):
    """
    Handles incoming fill events from hl-stream.
    Updates position state and emits score event to hl-decide.

    The score event now includes NIG posterior parameters for the trader,
    enabling hl-decide to make Bayesian-informed consensus decisions.

    Args:
        msg: NATS message containing FillEvent JSON
    """
    data = FillEvent.model_validate_json(msg.data.decode())
    addr_lower = data.address.lower()
    state = tracked_addresses.get(addr_lower)
    if not state:
        return

    # Move to end (most recently used)
    if addr_lower in tracked_addresses:
        tracked_addresses.move_to_end(addr_lower)

    side_multiplier = 1 if data.side == "buy" else -1
    delta = side_multiplier * float(data.size or 0)
    state["position"] = state.get("position", 0.0) + delta
    state["updated"] = datetime.now(timezone.utc)

    # Persist updated position to database
    await persist_tracked_address(addr_lower, state)

    # Fetch NIG posterior for this trader (if available)
    nig_params = await get_trader_nig_params(addr_lower)

    # Thompson Sampling: sample from NIG posterior instead of using mean
    # This enables explore/exploit tradeoff - uncertain traders (low κ) get
    # wider samples, sometimes ranking higher than proven performers.
    if nig_params and nig_params.get("nig_m") is not None:
        # Create TraderPosteriorNIG for sampling
        from .bandit import TraderPosteriorNIG

        posterior = TraderPosteriorNIG(
            address=addr_lower,
            m=nig_params["nig_m"],
            kappa=nig_params["nig_kappa"],
            alpha=nig_params["nig_alpha"],
            beta=nig_params["nig_beta"],
            total_signals=nig_params["total_signals"],
            total_pnl_r=nig_params.get("total_pnl_r", 0.0),
        )

        # Thompson sample from posterior (explore/exploit)
        sampled_mu = posterior.sample()

        # Apply direction to sampled value
        nig_score = sampled_mu * side_multiplier
        # Clamp to [-1, 1] range
        base_score = max(-1.0, min(1.0, nig_score))
        score_source = "thompson"

        # Derive weight from NIG confidence: κ/(κ+10)
        # This gives weight ~0.09 for κ=1 (new trader), ~0.5 for κ=10, ~0.91 for κ=100
        nig_weight = nig_params["nig_kappa"] / (nig_params["nig_kappa"] + 10.0)
    else:
        # Legacy: leaderboard weight * direction
        base_score = max(-1.0, min(1.0, state["weight"] * side_multiplier))
        score_source = "leaderboard"
        nig_weight = None

    # Use NIG-derived weight when available, else legacy weight
    score_weight = nig_weight if nig_weight is not None else state["weight"]

    event = ScoreEvent(
        address=data.address,
        score=base_score,
        weight=score_weight,
        rank=state["rank"],
        window_s=60,
        ts=datetime.now(timezone.utc),
        meta={
            "source": score_source,
            "period": state["period"],
            "position": state["position"],
            "fill": data.model_dump(),
            # Include NIG params for hl-decide consensus detection
            "nig": nig_params if nig_params else None,
            # Include sampled value for debugging/audit
            "thompson_sample": sampled_mu if score_source == "thompson" else None,
        },
    )

    # Move to end (most recently used)
    if data.address in scores:
        scores.move_to_end(data.address)

    scores[data.address] = event
    await app.state.js.publish(
        "b.scores.v1",
        event.model_dump_json().encode("utf-8"),
    )
    score_counter.inc()


async def get_trader_nig_params(address: str) -> Optional[Dict[str, Any]]:
    """
    Fetch NIG posterior parameters for a trader from the database.

    Args:
        address: Trader's Ethereum address (lowercase)

    Returns:
        Dict with nig_m, nig_kappa, nig_alpha, nig_beta, total_signals, avg_r, total_pnl_r
        or None if trader has no posterior data
    """
    try:
        async with app.state.db.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT nig_m, nig_kappa, nig_alpha, nig_beta,
                       total_signals, avg_r, total_pnl_r
                FROM trader_performance
                WHERE address = $1
                """,
                address,
            )
            if row and row["nig_m"] is not None:
                return {
                    "nig_m": float(row["nig_m"]),
                    "nig_kappa": float(row["nig_kappa"]),
                    "nig_alpha": float(row["nig_alpha"]),
                    "nig_beta": float(row["nig_beta"]),
                    "total_signals": int(row["total_signals"] or 0),
                    "avg_r": float(row["avg_r"] or 0),
                    "total_pnl_r": float(row["total_pnl_r"] or 0),
                }
    except Exception as e:
        print(f"[hl-sage] Failed to fetch NIG params for {address}: {e}")
    return None


ALPHA_POOL_AUTO_REFRESH = os.getenv("ALPHA_POOL_AUTO_REFRESH", "true").lower() == "true"
ALPHA_POOL_AUTO_INIT = os.getenv("ALPHA_POOL_AUTO_INIT", "true").lower() == "true"
ALPHA_POOL_AUTO_INIT_DELAY_MS = int(os.getenv("ALPHA_POOL_AUTO_INIT_DELAY_MS", "500"))


async def auto_refresh_alpha_pool_if_empty():
    """
    Auto-initialize Alpha Pool on startup if it has NEVER been refreshed.

    This is the main fresh-install initialization flow:
    1. Detects if this is a fresh database (no records in alpha_pool_addresses)
    2. Refreshes the pool from Hyperliquid leaderboard (auto-backfills new addresses)
    3. Creates an initial snapshot for FDR qualification

    Checks if alpha_pool_addresses table has ANY records (active or not).
    If completely empty, this is a fresh database and we bootstrap the pool.

    Subsequent refreshes should be done via POST /alpha-pool/refresh
    or a scheduled job.

    Runs as a background task to not block service startup.

    IMPORTANT: Uses _background_refresh_task() to properly set is_running=true
    so the dashboard can detect the refresh is in progress.
    """
    try:
        # Wait a bit for database to be fully ready
        await asyncio.sleep(5)

        async with app.state.db.acquire() as conn:
            # Check if pool has EVER been refreshed (any records, active or not)
            total_count = await conn.fetchval(
                "SELECT COUNT(*) FROM alpha_pool_addresses"
            )

        if total_count == 0:
            print(f"[hl-sage] Fresh install detected: Alpha Pool is empty")
            print(f"[hl-sage] Starting automatic initialization...")

            # Step 1: Refresh Alpha Pool from leaderboard (includes auto-backfill)
            print(f"[hl-sage] [1/2] Refreshing Alpha Pool from leaderboard...")
            await _background_refresh_task(limit=ALPHA_POOL_DEFAULT_SIZE)

            # Step 2: Create initial snapshot for FDR qualification
            if ALPHA_POOL_AUTO_INIT:
                print(f"[hl-sage] [2/2] Creating initial snapshot for FDR qualification...")
                try:
                    snapshot_result = await create_daily_snapshot(app.state.db)
                    fdr_count = snapshot_result.get("fdr_qualified", 0)
                    total = snapshot_result.get("total_traders", 0)
                    print(f"[hl-sage] Initial snapshot created: {fdr_count}/{total} traders FDR-qualified")
                except Exception as snapshot_err:
                    print(f"[hl-sage] Initial snapshot failed: {snapshot_err}")

            print(f"[hl-sage] Automatic initialization complete!")
        else:
            print(f"[hl-sage] Alpha Pool has {total_count} records (previously refreshed), skipping auto-init")
    except Exception as e:
        await _refresh_state.fail(str(e))
        print(f"[hl-sage] Alpha Pool auto-init failed: {e}")


# Interval for periodic Alpha Pool fill sync (in seconds)
ALPHA_POOL_FILL_SYNC_INTERVAL = int(os.getenv("ALPHA_POOL_FILL_SYNC_INTERVAL", "300"))  # 5 minutes default


async def get_polling_addresses() -> list[str]:
    """
    Get addresses that need polling (not on WebSocket).

    Fetches subscription methods from hl-stream and returns only addresses
    that are using 'polling' or 'none' method (not real-time WebSocket).
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{HL_STREAM_URL}/subscriptions/methods")
            if response.status_code == 200:
                methods = response.json()
                # Return addresses that are NOT on WebSocket
                polling_addrs = [
                    addr for addr, info in methods.items()
                    if info.get("method") != "websocket"
                ]
                return polling_addrs
    except Exception as e:
        print(f"[hl-sage] Could not fetch subscription methods: {e}")

    # If we can't get subscription info, return empty list to avoid redundant polling
    return []


async def periodic_alpha_pool_fill_sync():
    """
    Periodically sync fills for Alpha Pool addresses that are NOT on WebSocket.

    Only polls fills for addresses using 'polling' method - addresses that
    are subscribed via WebSocket receive real-time fills and don't need polling.

    Runs every ALPHA_POOL_FILL_SYNC_INTERVAL seconds (default: 5 minutes).
    """
    # Wait for startup to complete
    await asyncio.sleep(30)
    print(f"[hl-sage] Starting periodic Alpha Pool fill sync (interval: {ALPHA_POOL_FILL_SYNC_INTERVAL}s)")

    while True:
        try:
            # Get active Alpha Pool addresses
            async with app.state.db.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT address FROM alpha_pool_addresses
                    WHERE is_active = true
                    ORDER BY last_refreshed DESC
                    LIMIT 50
                    """
                )

            if not rows:
                await asyncio.sleep(ALPHA_POOL_FILL_SYNC_INTERVAL)
                continue

            all_addresses = set(row["address"].lower() for row in rows)

            # Only poll addresses that are NOT on WebSocket
            polling_addrs = await get_polling_addresses()
            polling_set = set(addr.lower() for addr in polling_addrs)

            # Filter to Alpha Pool addresses that need polling
            addresses_to_poll = [
                row["address"] for row in rows
                if row["address"].lower() in polling_set
            ]

            # If all addresses are on WebSocket, skip polling
            if not addresses_to_poll:
                websocket_count = len(all_addresses) - len(addresses_to_poll)
                print(f"[hl-sage] All {websocket_count} Alpha Pool addresses on WebSocket, skipping poll")
                await asyncio.sleep(ALPHA_POOL_FILL_SYNC_INTERVAL)
                continue

            websocket_count = len(all_addresses) - len(addresses_to_poll)
            print(f"[hl-sage] Polling fills for {len(addresses_to_poll)} addresses ({websocket_count} on WebSocket)...")

            # Backfill recent fills for polling-only addresses
            results = await backfill_historical_fills_for_addresses(app.state.db, addresses_to_poll)
            total_inserted = sum(results.values())

            if total_inserted > 0:
                print(f"[hl-sage] Fill sync complete: {total_inserted} new fills inserted")
            else:
                print(f"[hl-sage] Fill sync complete: no new fills")

        except Exception as e:
            print(f"[hl-sage] Fill sync error: {e}")

        await asyncio.sleep(ALPHA_POOL_FILL_SYNC_INTERVAL)


# Interval for syncing Alpha Pool addresses to hl-stream subscription manager (in seconds)
ALPHA_POOL_SUBSCRIPTION_SYNC_INTERVAL = int(os.getenv("ALPHA_POOL_SUBSCRIPTION_SYNC_INTERVAL", "60"))  # 1 minute default


async def sync_alpha_pool_subscriptions():
    """
    Sync selected Alpha Pool addresses to hl-stream for real-time WebSocket tracking.

    This registers the top-K selected Alpha Pool addresses with hl-stream's
    centralized subscription manager, enabling real-time fill detection
    instead of relying on periodic backfill.

    Runs every ALPHA_POOL_SUBSCRIPTION_SYNC_INTERVAL seconds (default: 60s).

    The sync only includes addresses that are:
    1. Active in alpha_pool_addresses
    2. Selected by the NIG Thompson Sampling (top-K)
    """
    # Wait for startup to complete
    await asyncio.sleep(15)
    print(f"[hl-sage] Starting Alpha Pool subscription sync (interval: {ALPHA_POOL_SUBSCRIPTION_SYNC_INTERVAL}s)")

    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            try:
                # Get selected Alpha Pool addresses (top-K by posterior mean)
                async with app.state.db.acquire() as conn:
                    # First check if we have any trader_performance data
                    has_performance = await conn.fetchval(
                        "SELECT COUNT(*) FROM trader_performance"
                    )

                    if has_performance > 0:
                        # Get addresses with NIG posteriors, sorted by posterior mean
                        rows = await conn.fetch(
                            """
                            SELECT a.address
                            FROM alpha_pool_addresses a
                            LEFT JOIN trader_performance tp ON LOWER(a.address) = LOWER(tp.address)
                            WHERE a.is_active = true
                            ORDER BY COALESCE(tp.nig_m, 0) DESC
                            LIMIT $1
                            """,
                            BANDIT_SELECT_K,
                        )
                    else:
                        # No performance data yet - use top-K by PnL
                        rows = await conn.fetch(
                            """
                            SELECT address
                            FROM alpha_pool_addresses
                            WHERE is_active = true
                            ORDER BY pnl_30d DESC NULLS LAST
                            LIMIT $1
                            """,
                            BANDIT_SELECT_K,
                        )

                if not rows:
                    await asyncio.sleep(ALPHA_POOL_SUBSCRIPTION_SYNC_INTERVAL)
                    continue

                addresses = [row["address"] for row in rows]

                # Register with hl-stream subscription manager
                response = await client.post(
                    f"{HL_STREAM_URL}/subscriptions/replace",
                    json={"source": "alpha-pool", "addresses": addresses},
                    headers={"x-owner-key": OWNER_TOKEN},
                )

                if response.status_code == 200:
                    result = response.json()
                    total = result.get("totalAddresses", "?")
                    count = result.get("count", len(addresses))
                    print(f"[hl-sage] Synced {count} Alpha Pool addresses to WebSocket (total subscribed: {total})")
                else:
                    print(f"[hl-sage] Failed to sync subscriptions: HTTP {response.status_code}")

            except httpx.ConnectError:
                # hl-stream not available yet, will retry
                pass
            except Exception as e:
                print(f"[hl-sage] Subscription sync error: {e}")

            await asyncio.sleep(ALPHA_POOL_SUBSCRIPTION_SYNC_INTERVAL)


async def periodic_alpha_pool_refresh():
    """
    Periodically refresh the Alpha Pool from Hyperliquid leaderboard.

    Checks if the pool is overdue for refresh (based on ALPHA_POOL_REFRESH_HOURS)
    and triggers a refresh if needed.

    This ensures the pool stays fresh without manual intervention.
    """
    # Wait for startup to complete and initial auto-refresh (if any)
    await asyncio.sleep(120)  # 2 minutes

    refresh_interval_seconds = ALPHA_POOL_REFRESH_HOURS * 3600
    print(f"[hl-sage] Starting periodic Alpha Pool refresh (interval: {ALPHA_POOL_REFRESH_HOURS}h)")

    while True:
        try:
            # Check if refresh is needed
            async with app.state.db.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT MAX(last_refreshed) as last_refreshed
                    FROM alpha_pool_addresses
                    WHERE is_active = true
                    """
                )

            if row and row["last_refreshed"]:
                last_refreshed = row["last_refreshed"]
                now = datetime.now(timezone.utc)
                age_seconds = (now - last_refreshed).total_seconds()

                if age_seconds >= refresh_interval_seconds:
                    print(f"[hl-sage] Alpha Pool is {age_seconds / 3600:.1f}h old, triggering refresh...")
                    # Check if refresh is already running
                    if not _refresh_state.is_running:
                        await _background_refresh_task(limit=ALPHA_POOL_DEFAULT_SIZE)
                    else:
                        print(f"[hl-sage] Refresh already in progress, skipping")
                else:
                    hours_until_refresh = (refresh_interval_seconds - age_seconds) / 3600
                    print(f"[hl-sage] Alpha Pool refresh not needed yet ({hours_until_refresh:.1f}h until due)")
            else:
                # No data, trigger refresh
                print(f"[hl-sage] No Alpha Pool data, triggering initial refresh...")
                if not _refresh_state.is_running:
                    await _background_refresh_task(limit=ALPHA_POOL_DEFAULT_SIZE)

        except Exception as e:
            print(f"[hl-sage] Periodic refresh check error: {e}")

        # Check every hour
        await asyncio.sleep(3600)


async def periodic_daily_snapshot():
    """
    Create daily snapshots for the Shadow Ledger (Phase 3f: Selection Integrity).

    Runs at SNAPSHOT_HOUR_UTC (default: midnight) and captures the state of all
    traders for survivorship-bias-free analysis.

    The snapshot includes:
    - Universe membership (which stage each trader reached)
    - Thompson sampling draws with stored seeds for reproducibility
    - FDR qualification status
    - Death/censor event detection
    """
    # Wait for startup to complete
    await asyncio.sleep(180)  # 3 minutes

    print(f"[hl-sage] Starting daily snapshot job (hour: {SNAPSHOT_HOUR_UTC:02d}:00 UTC)")

    while True:
        try:
            now = datetime.now(timezone.utc)
            target_hour = SNAPSHOT_HOUR_UTC

            # Calculate seconds until next snapshot time
            if now.hour < target_hour:
                # Same day
                next_run = now.replace(hour=target_hour, minute=0, second=0, microsecond=0)
            else:
                # Next day
                next_run = (now + timedelta(days=1)).replace(hour=target_hour, minute=0, second=0, microsecond=0)

            wait_seconds = (next_run - now).total_seconds()

            # Don't wait more than 1 hour at a time (allows for clock adjustments)
            if wait_seconds > 3600:
                print(f"[hl-sage] Next snapshot at {next_run.isoformat()}, waiting...")
                await asyncio.sleep(3600)
                continue

            # Wait until snapshot time
            if wait_seconds > 0:
                print(f"[hl-sage] Snapshot due in {wait_seconds:.0f}s at {next_run.isoformat()}")
                await asyncio.sleep(wait_seconds)

            # Create snapshot
            print(f"[hl-sage] Creating daily snapshot...")
            result = await create_daily_snapshot(app.state.db)
            print(f"[hl-sage] Snapshot complete: {result}")

            # Wait a bit after snapshot to avoid running twice at boundary
            await asyncio.sleep(60)

        except Exception as e:
            print(f"[hl-sage] Snapshot job error: {e}")
            # Wait before retrying
            await asyncio.sleep(300)


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "scores": len(scores), "tracked_addresses": len(tracked_addresses)}


@app.get("/metrics")
async def metrics():
    data = generate_latest(registry)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.get("/ranks/top")
async def ranks_top(n: int = Query(default=20, ge=1, le=100)):
    if not scores:
        raise HTTPException(status_code=503, detail="no scores yet")
    ordered = sorted(scores.values(), key=lambda s: s.score, reverse=True)
    return {"count": len(ordered), "entries": ordered[:n]}


# =====================
# Bandit API Endpoints
# =====================


@app.get("/bandit/status")
async def bandit_status():
    """
    Get current status of the Thompson Sampling bandit algorithm.
    Shows configuration, statistics, and top performers.
    """
    try:
        status = await get_bandit_status(app.state.db)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get bandit status: {e}")


@app.get("/bandit/posteriors")
async def bandit_posteriors(
    limit: int = Query(default=BANDIT_POOL_SIZE, ge=1, le=100),
    min_signals: int = Query(default=0, ge=0),
):
    """
    Get trader posteriors for the bandit algorithm.
    Returns Beta distribution parameters (alpha, beta) for each trader.
    """
    try:
        posteriors = await get_trader_posteriors(app.state.db, limit=limit, min_signals=min_signals)
        return {
            "count": len(posteriors),
            "traders": [
                {
                    "address": p.address,
                    "alpha": p.alpha,
                    "beta": p.beta,
                    "posterior_mean": p.posterior_mean,
                    "posterior_variance": p.posterior_variance,
                    "total_signals": p.total_signals,
                    "winning_signals": p.winning_signals,
                    "total_pnl_r": p.total_pnl_r,
                }
                for p in posteriors
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get posteriors: {e}")


@app.get("/bandit/select")
async def bandit_select(
    k: int = Query(default=BANDIT_SELECT_K, ge=1, le=50),
    exploration_ratio: float = Query(default=0.2, ge=0.0, le=1.0),
):
    """
    Select traders using Thompson Sampling with exploration.
    This is a preview of which traders would be selected - doesn't modify state.
    """
    try:
        selected = await select_traders_with_exploration(
            app.state.db, k=k, exploration_ratio=exploration_ratio
        )
        return {
            "count": len(selected),
            "selected_addresses": selected,
            "config": {
                "k": k,
                "exploration_ratio": exploration_ratio,
                "pool_size": BANDIT_POOL_SIZE,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to select traders: {e}")


@app.post("/bandit/sample")
async def bandit_sample(
    k: int = Query(default=BANDIT_SELECT_K, ge=1, le=50),
):
    """
    Perform Thompson Sampling and return sampled values.
    Shows the actual random samples drawn from each posterior.
    """
    try:
        posteriors = await get_trader_posteriors(app.state.db, limit=BANDIT_POOL_SIZE)
        if not posteriors:
            return {"count": 0, "samples": [], "selected": []}

        # Sample from each posterior
        samples = []
        for p in posteriors:
            sample = p.sample()
            samples.append({
                "address": p.address,
                "sampled_value": sample,
                "posterior_mean": p.posterior_mean,
                "alpha": p.alpha,
                "beta": p.beta,
            })

        # Sort by sampled value
        samples.sort(key=lambda x: x["sampled_value"], reverse=True)

        # Select top k
        selected = thompson_sample_select(posteriors, k=k)

        return {
            "count": len(samples),
            "samples": samples[:20],  # Return top 20 samples for visibility
            "selected": [p.address for p in selected],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to sample: {e}")


@app.post("/bandit/decay")
async def bandit_apply_decay(
    decay_factor: float = Query(default=0.95, ge=0.5, le=1.0),
):
    """
    Apply exponential decay to all trader posteriors.
    This makes recent performance more important than old performance.
    Use sparingly - typically run daily or weekly.
    """
    try:
        count = await apply_decay(app.state.db, decay_factor=decay_factor)
        return {
            "success": True,
            "traders_updated": count,
            "decay_factor": decay_factor,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to apply decay: {e}")


# =====================
# Alpha Pool API (NIG-based Thompson Sampling)
# =====================
#
# The Alpha Pool is a COMPLETELY INDEPENDENT system from the legacy leaderboard.
# It uses:
# - alpha_pool_addresses table (not hl_leaderboard_entries)
# - Direct Hyperliquid API calls for PnL curves (not hl_leaderboard_pnl_points)
# - NIG Thompson Sampling for selection (not rank-based scoring)
#
# To populate the Alpha Pool, call POST /alpha-pool/refresh first.
# =====================

HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"
PNL_CURVE_CONCURRENCY = int(os.getenv("PNL_CURVE_CONCURRENCY", "2"))  # Max concurrent requests
PNL_CURVE_TIMEOUT = 10.0  # Timeout per request in seconds
PNL_CURVE_DELAY = float(os.getenv("PNL_CURVE_DELAY", "0.3"))  # Delay between requests in seconds
PNL_CURVE_MAX_RETRIES = 2  # Max retries on rate limit
PNL_CURVE_CACHE_TTL = int(os.getenv("PNL_CURVE_CACHE_TTL", "86400"))  # Cache TTL in seconds (default 24 hours)

# Historical fill backfill settings
FILL_BACKFILL_CONCURRENCY = int(os.getenv("FILL_BACKFILL_CONCURRENCY", "1"))  # Max concurrent fill fetch requests
FILL_BACKFILL_DELAY = float(os.getenv("FILL_BACKFILL_DELAY", "1.0"))  # Delay between requests in seconds

# Rate limiting settings for Hyperliquid API
# Target: stay under 1200 weight/minute = 20 weight/second
# userFills = weight 20, so max ~1 req/second for fills
HL_API_RATE_LIMIT_DELAY = float(os.getenv("HL_API_RATE_LIMIT_DELAY", "1.0"))  # Min delay between API calls
HL_API_MAX_RETRIES = int(os.getenv("HL_API_MAX_RETRIES", "3"))  # Max retries on 429

# In-memory cache for PnL curves: {address: (timestamp, curve_data)}
_pnl_curve_cache: Dict[str, tuple] = {}

# In-memory cache for user fills during refresh: {address: fills_list}
# This cache is cleared after each refresh cycle to avoid stale data
_fills_cache: Dict[str, List[Dict[str, Any]]] = {}

# Background refresh state
class RefreshState:
    """Tracks the state of background Alpha Pool refresh."""
    def __init__(self):
        self.is_running = False
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.current_step: str = "idle"
        self.progress: int = 0  # 0-100
        self.total_traders: int = 0
        self.processed_traders: int = 0
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self._lock = asyncio.Lock()

    async def start(self):
        async with self._lock:
            self.is_running = True
            self.started_at = datetime.now(timezone.utc)
            self.completed_at = None
            self.current_step = "starting"
            self.progress = 0
            self.total_traders = 0
            self.processed_traders = 0
            self.result = None
            self.error = None

    async def update(self, step: str, progress: int, processed: int = 0, total: int = 0):
        async with self._lock:
            self.current_step = step
            self.progress = progress
            self.processed_traders = processed
            self.total_traders = total

    async def complete(self, result: Dict[str, Any]):
        async with self._lock:
            self.is_running = False
            self.completed_at = datetime.now(timezone.utc)
            self.current_step = "completed"
            self.progress = 100
            self.result = result
            self.error = None

    async def fail(self, error: str):
        async with self._lock:
            self.is_running = False
            self.completed_at = datetime.now(timezone.utc)
            self.current_step = "failed"
            self.error = error

    async def get_status(self) -> Dict[str, Any]:
        async with self._lock:
            elapsed = None
            if self.started_at:
                end = self.completed_at or datetime.now(timezone.utc)
                elapsed = (end - self.started_at).total_seconds()
            return {
                "is_running": self.is_running,
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "completed_at": self.completed_at.isoformat() if self.completed_at else None,
                "elapsed_seconds": elapsed,
                "current_step": self.current_step,
                "progress": self.progress,
                "processed_traders": self.processed_traders,
                "total_traders": self.total_traders,
                "result": self.result,
                "error": self.error,
            }

_refresh_state = RefreshState()


class RateLimiter:
    """
    Token bucket rate limiter for Hyperliquid API calls.

    Hyperliquid limits: 1200 weight/minute per IP
    - userFills: weight 20 (paginated, +1 per 20 items)
    - portfolio: weight 20
    - subAccounts: weight 20 (assumed)
    - clearinghouseState: weight 2

    Safe target: ~50 weight/second = 3000 weight/minute (2.5x safety margin)
    For weight-20 calls: ~2.5 calls/second max
    """

    def __init__(self, calls_per_second: float = 2.0):
        self._min_interval = 1.0 / calls_per_second
        self._last_call = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Wait until a request can be made within rate limits."""
        async with self._lock:
            now = asyncio.get_event_loop().time()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                wait_time = self._min_interval - elapsed
                await asyncio.sleep(wait_time)
            self._last_call = asyncio.get_event_loop().time()


# Global rate limiter for Hyperliquid API
# Hyperliquid: 1200 weight/minute, userFills = weight 20
# Safe limit: 1200/20 = 60 calls/minute = 1 call/second
# Default to 0.8 calls/second (48/min) for safety margin
_hl_rate_limiter = RateLimiter(calls_per_second=float(os.getenv("HL_API_CALLS_PER_SECOND", "0.8")))


async def fetch_user_fills_from_api(
    client: httpx.AsyncClient,
    address: str,
    max_retries: int = HL_API_MAX_RETRIES,
    use_cache: bool = True,
) -> List[Dict[str, Any]]:
    """
    Fetch user fills from Hyperliquid API with rate limiting.

    Returns all BTC/ETH fills for the address.
    Uses rate limiter to stay under Hyperliquid's 1200 weight/minute limit.

    Args:
        client: Async HTTP client
        address: Ethereum address
        max_retries: Max retries on rate limit (uses Retry-After when available)
        use_cache: Whether to check/update the fills cache
    """
    addr_lower = address.lower()

    # Check cache first if enabled
    if use_cache and addr_lower in _fills_cache:
        return _fills_cache[addr_lower]

    for attempt in range(max_retries + 1):
        try:
            # Rate limit before making the call
            await _hl_rate_limiter.acquire()

            response = await client.post(
                HYPERLIQUID_INFO_URL,
                json={"type": "userFills", "user": addr_lower},
                timeout=15.0,
            )

            if response.status_code == 429:
                if attempt < max_retries:
                    # Check for Retry-After header
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = float(retry_after)
                        except ValueError:
                            delay = (attempt + 1) * 2.0
                    else:
                        delay = (attempt + 1) * 2.0
                    print(f"[hl-sage] Rate limited on fills for {addr_lower[:10]}..., retry {attempt + 1} in {delay:.1f}s")
                    await asyncio.sleep(delay)
                    continue
                return []

            if response.status_code != 200:
                return []

            fills = response.json()
            if not isinstance(fills, list):
                return []

            # Filter to BTC/ETH only
            result = []
            for f in fills:
                coin = f.get("coin", "").upper()
                if coin not in ("BTC", "ETH"):
                    continue
                result.append({
                    "coin": coin,
                    "px": float(f.get("px", 0)),
                    "sz": float(f.get("sz", 0)),
                    "side": f.get("side", "B"),  # B=buy, A=ask(sell)
                    "time": int(f.get("time", 0)),
                    "startPosition": float(f.get("startPosition", 0)),
                    "closedPnl": float(f.get("closedPnl", 0)) if f.get("closedPnl") else None,
                    "fee": float(f.get("fee", 0)) if f.get("fee") else None,
                    "hash": f.get("hash"),
                })

            # Cache the result for reuse during same refresh cycle
            if use_cache:
                _fills_cache[addr_lower] = result

            return result
        except Exception as e:
            if attempt < max_retries:
                await asyncio.sleep(1.0)
                continue
            print(f"[hl-sage] Error fetching fills for {address}: {e}")
            return []
    return []


async def publish_fill_to_nats(payload: Dict[str, Any]) -> None:
    """
    Publish a backfilled fill to NATS for hl-decide to process.

    Converts the hl_events payload format to FillEvent format and publishes
    to c.fills.v1 subject for episode building and consensus detection.

    Args:
        payload: Fill payload in hl_events format
    """
    try:
        # Convert to FillEvent format
        fill_event = {
            "fill_id": payload.get("hash") or f"backfill-{payload['address']}-{payload['at']}",
            "source": "hyperliquid",
            "address": payload["address"],
            "asset": payload["symbol"].upper(),
            "side": "buy" if "Long" in payload["action"] and ("Open" in payload["action"] or "Increase" in payload["action"]) else "sell",
            "size": float(payload["size"]),
            "price": float(payload["priceUsd"]),
            "start_position": float(payload.get("startPosition", 0)),
            "realized_pnl": float(payload["realizedPnlUsd"]) if payload.get("realizedPnlUsd") is not None else None,
            "ts": payload["at"],
            "meta": {"backfilled": True},
        }

        # Publish to NATS
        if hasattr(app.state, "nc") and app.state.nc.is_connected:
            await app.state.nc.publish(
                "c.fills.v1",
                json.dumps(fill_event).encode(),
            )
    except Exception as e:
        # Don't fail backfill if NATS publish fails
        print(f"[hl-sage] Failed to publish fill to NATS: {e}")


async def backfill_historical_fills_for_addresses(
    pool: asyncpg.Pool,
    addresses: List[str],
) -> Dict[str, int]:
    """
    Fetch and store historical BTC/ETH fills for the given addresses.

    This is called when new addresses are added to the Alpha Pool to ensure
    we have historical data for episode construction.

    IMPORTANT: This function reuses fills from the _fills_cache populated by
    analyze_user_fills() during the HFT filtering step. This avoids duplicate
    API calls and saves rate limit budget.

    Args:
        pool: Database connection pool
        addresses: List of addresses to backfill

    Returns:
        Dict mapping address -> number of fills inserted
    """
    if not addresses:
        return {}

    results: Dict[str, int] = {}
    semaphore = asyncio.Semaphore(FILL_BACKFILL_CONCURRENCY)

    async def backfill_one(client: httpx.AsyncClient, addr: str, index: int):
        addr_lower = addr.lower()

        # Check cache first - fills were already fetched during HFT filtering
        if addr_lower in _fills_cache:
            fills = _fills_cache[addr_lower]
        else:
            # Stagger requests to avoid rate limiting (only if not cached)
            if index > 0:
                await asyncio.sleep(FILL_BACKFILL_DELAY * index)
            async with semaphore:
                fills = await fetch_user_fills_from_api(client, addr, use_cache=True)

        if not fills:
            results[addr] = 0
            print(f"[hl-sage] Backfill: no fills for {addr[:10]}... (cache={'hit' if addr_lower in _fills_cache else 'miss'})")
            return

        print(f"[hl-sage] Backfill: processing {len(fills)} fills for {addr[:10]}...")
        inserted = 0
        async with pool.acquire() as conn:
            for f in fills:
                # Calculate action from position change
                delta = f["sz"] if f["side"] == "B" else -f["sz"]
                start_pos = f["startPosition"]
                new_pos = start_pos + delta

                if start_pos == 0:
                    action = "Open Long (Open New)" if delta > 0 else "Open Short (Open New)"
                elif start_pos > 0:
                    if delta > 0:
                        action = "Increase Long"
                    elif new_pos == 0:
                        action = "Close Long (Close All)"
                    else:
                        action = "Decrease Long"
                else:  # start_pos < 0
                    if delta < 0:
                        action = "Increase Short"
                    elif new_pos == 0:
                        action = "Close Short (Close All)"
                    else:
                        action = "Decrease Short"

                # Build payload matching hl_events format
                payload = {
                    "at": datetime.fromtimestamp(f["time"] / 1000, tz=timezone.utc).isoformat(),
                    "address": addr_lower,
                    "symbol": f["coin"],
                    "action": action,
                    "size": abs(f["sz"]),
                    "startPosition": start_pos,
                    "priceUsd": f["px"],
                    "realizedPnlUsd": f["closedPnl"],
                    "fee": f["fee"],
                    "hash": f["hash"],
                }

                # Insert into hl_events with dedup on hash
                try:
                    result = await conn.execute(
                        """
                        INSERT INTO hl_events (address, type, symbol, payload)
                        SELECT $1, 'trade', $2, $3::jsonb
                        WHERE NOT EXISTS (
                            SELECT 1 FROM hl_events
                            WHERE type = 'trade' AND payload->>'hash' = $4
                        )
                        """,
                        addr_lower,
                        f["coin"],
                        json.dumps(payload),
                        f["hash"],
                    )
                    # asyncpg returns "INSERT 0 1" for successful insert, "INSERT 0 0" if no rows inserted
                    if result and result.endswith("1"):
                        inserted += 1
                        # Publish to NATS for hl-decide to process
                        await publish_fill_to_nats(payload)
                except asyncpg.exceptions.UniqueViolationError:
                    # Duplicate hash - expected, skip
                    pass
                except Exception as e:
                    # Log unexpected errors once per address
                    if inserted == 0:
                        print(f"[hl-sage] Backfill error for {addr[:10]}...: {type(e).__name__}: {e}")

        results[addr] = inserted
        if inserted > 0:
            print(f"[hl-sage] Backfilled {inserted} fills for {addr[:10]}...")

    async with httpx.AsyncClient() as client:
        tasks = [backfill_one(client, addr, i) for i, addr in enumerate(addresses)]
        await asyncio.gather(*tasks, return_exceptions=True)

    return results


async def fetch_pnl_curve_from_api(
    client: httpx.AsyncClient,
    address: str,
    window: str = "perpMonth",
    retries: int = 0,
) -> List[Dict[str, Any]]:
    """
    Fetch PnL curve for a single address directly from Hyperliquid API.

    Args:
        client: Async HTTP client
        address: Ethereum address
        window: Time window ('perpMonth', 'perpWeek', 'month', 'week', etc.)
        retries: Current retry count

    Returns:
        List of {ts, value} points
    """
    try:
        # Rate limit before making the call
        await _hl_rate_limiter.acquire()

        response = await client.post(
            HYPERLIQUID_INFO_URL,
            json={"type": "portfolio", "user": address.lower()},
            timeout=PNL_CURVE_TIMEOUT,
        )

        # Handle rate limiting with retry and Retry-After header
        if response.status_code == 429 and retries < PNL_CURVE_MAX_RETRIES:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    delay = float(retry_after)
                except ValueError:
                    delay = (retries + 1) * 2.0
            else:
                delay = (retries + 1) * 2.0
            await asyncio.sleep(delay)
            return await fetch_pnl_curve_from_api(client, address, window, retries + 1)

        response.raise_for_status()
        data = response.json()

        # Response is array of [windowName, {pnlHistory, accountValueHistory}]
        if not isinstance(data, list):
            return []

        for row in data:
            if not isinstance(row, list) or len(row) < 2:
                continue
            window_name = str(row[0] or "")
            # Match 'perpMonth' for 30-day perp-only PnL (excludes spot, funding noise)
            # Fall back to 'month' if perpMonth not available
            if window_name == window or window_name == "perpMonth":
                history_data = row[1] or {}
                pnl_history = history_data.get("pnlHistory", [])
                if not isinstance(pnl_history, list):
                    continue
                # Parse [timestamp_ms, "value_string"] pairs
                points = []
                for point in pnl_history:
                    if isinstance(point, list) and len(point) >= 2:
                        ts_ms = point[0]
                        value = point[1]
                        points.append({
                            "ts": ts_ms,  # Keep as timestamp for sparkline
                            "value": str(value) if value is not None else "0",
                        })
                return points
        return []
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429 and retries < PNL_CURVE_MAX_RETRIES:
            delay = (retries + 1) * 2.0
            await asyncio.sleep(delay)
            return await fetch_pnl_curve_from_api(client, address, window, retries + 1)
        print(f"[hl-sage] Failed to fetch PnL curve for {address}: {e}")
        return []
    except Exception as e:
        print(f"[hl-sage] Failed to fetch PnL curve for {address}: {e}")
        return []


async def get_pnl_curves_for_addresses(
    addresses: List[str],
    window: str = "perpMonth",
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetch PnL curves for multiple addresses from Hyperliquid API with caching.

    Uses in-memory cache with TTL to avoid unnecessary API calls.
    30-day perp-only PnL curves don't change frequently, so caching is safe.

    Args:
        addresses: List of Ethereum addresses
        window: Time window (default 'perpMonth' for 30-day perp-only curve)

    Returns:
        Dict mapping address -> list of {ts, value} points
    """
    if not addresses:
        return {}

    now = datetime.now(timezone.utc).timestamp()
    curves: Dict[str, List[Dict[str, Any]]] = {}
    addresses_to_fetch: List[str] = []

    # Check cache first
    for addr in addresses:
        addr_lower = addr.lower()
        if addr_lower in _pnl_curve_cache:
            cached_ts, cached_data = _pnl_curve_cache[addr_lower]
            if now - cached_ts < PNL_CURVE_CACHE_TTL:
                # Cache hit - use cached data
                curves[addr_lower] = cached_data
                continue
        # Cache miss or expired - need to fetch
        addresses_to_fetch.append(addr)

    # Fetch missing curves from API
    if addresses_to_fetch:
        semaphore = asyncio.Semaphore(PNL_CURVE_CONCURRENCY)

        async def fetch_one(client: httpx.AsyncClient, addr: str, index: int):
            # Stagger requests to avoid rate limiting
            if index > 0:
                await asyncio.sleep(PNL_CURVE_DELAY * index)
            async with semaphore:
                points = await fetch_pnl_curve_from_api(client, addr, window)
                if points:
                    addr_lower = addr.lower()
                    curves[addr_lower] = points
                    # Update cache
                    _pnl_curve_cache[addr_lower] = (now, points)

        async with httpx.AsyncClient() as client:
            tasks = [fetch_one(client, addr, i) for i, addr in enumerate(addresses_to_fetch)]
            await asyncio.gather(*tasks, return_exceptions=True)

    return curves


async def get_nicknames_for_addresses(
    pool: asyncpg.Pool,
    addresses: List[str],
) -> Dict[str, str]:
    """
    Fetch nicknames for addresses from alpha_pool_addresses only.

    Fully decoupled from legacy leaderboard.

    Returns a dict mapping address -> nickname.
    """
    if not addresses:
        return {}

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT LOWER(address) as address, nickname
            FROM alpha_pool_addresses
            WHERE LOWER(address) = ANY($1::text[])
              AND nickname IS NOT NULL
              AND nickname != ''
            """,
            [a.lower() for a in addresses],
        )

    return {row["address"]: row["nickname"] for row in rows}


@app.get("/alpha-pool")
async def get_alpha_pool(
    limit: int = Query(default=BANDIT_POOL_SIZE, ge=1, le=100),
    min_signals: int = Query(default=0, ge=0),
):
    """
    Get the Alpha Pool - traders selected by NIG Thompson Sampling.

    Returns top traders ranked by NIG posterior mean (expected R-multiple),
    with full posterior parameters for visualization.

    This is the primary endpoint for the Alpha Pool tab in the dashboard.

    Data sources (fully decoupled from legacy):
    - Traders: alpha_pool_addresses table (populated via /alpha-pool/refresh)
    - PnL curves: Direct Hyperliquid API call (not from DB)
    - Posteriors: trader_performance table (NIG parameters)
    - Nicknames: alpha_pool_addresses.nickname (not hl_leaderboard_entries)

    Note: Call /alpha-pool/refresh first to populate the pool.
    """
    try:
        posteriors = await get_trader_posteriors_nig(
            app.state.db, limit=limit, min_signals=min_signals
        )

        # Also get Thompson samples for current selection
        selected = thompson_sample_select_nig(posteriors, k=min(BANDIT_SELECT_K, len(posteriors)))
        selected_addresses = {p.address for p in selected}

        # Fetch PnL curves directly from Hyperliquid API (decoupled from legacy leaderboard)
        # and nicknames from leaderboard entries
        addresses = [p.address for p in posteriors]

        # Optimization: Only fetch PnL curves for selected traders + top performers
        # This avoids slow API calls for 50 addresses (would take 15+ seconds)
        # Limit to top 20 by NIG posterior mean (most likely to be useful)
        pnl_curve_addresses = [p.address for p in posteriors[:20]]
        pnl_curves = await get_pnl_curves_for_addresses(pnl_curve_addresses)

        nicknames = await get_nicknames_for_addresses(app.state.db, addresses)

        # Get pool refresh timing info
        last_refreshed = None
        next_refresh = None
        async with app.state.db.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT MAX(last_refreshed) as last_refreshed
                FROM alpha_pool_addresses
                WHERE is_active = true
                """
            )
            if row and row["last_refreshed"]:
                last_refreshed = row["last_refreshed"].isoformat()
                next_refresh = (row["last_refreshed"] + timedelta(hours=ALPHA_POOL_REFRESH_HOURS)).isoformat()

        traders = []
        for p in posteriors:
            # Calculate posterior variance for uncertainty display
            var = p.beta / (p.kappa * (p.alpha - 1)) if p.alpha > 1 else float('inf')
            std = var ** 0.5 if var != float('inf') else 99.0

            traders.append({
                "address": p.address,
                "nickname": nicknames.get(p.address.lower()),
                "nig_m": round(p.m, 4),  # Expected R-multiple
                "nig_kappa": round(p.kappa, 2),  # Confidence (effective samples)
                "nig_alpha": round(p.alpha, 2),
                "nig_beta": round(p.beta, 4),
                "posterior_std": round(std, 4),
                "effective_samples": round(p.kappa - 1.0, 1),  # κ - prior κ
                "total_signals": p.total_signals,
                "total_pnl_r": round(p.total_pnl_r, 2),
                "avg_r": round(p.total_pnl_r / max(p.total_signals, 1), 3),
                "is_selected": p.address in selected_addresses,
                "pnl_curve": pnl_curves.get(p.address.lower(), []),
            })

        return {
            "count": len(traders),
            "pool_size": BANDIT_POOL_SIZE,
            "select_k": BANDIT_SELECT_K,
            "refresh_interval_hours": ALPHA_POOL_REFRESH_HOURS,
            "last_refreshed": last_refreshed,
            "next_refresh": next_refresh,
            "traders": traders,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alpha pool: {e}")


@app.get("/alpha-pool/status")
async def get_alpha_pool_status():
    """
    Get Alpha Pool status with NIG model statistics.
    Similar to /bandit/status but uses NIG posteriors.
    """
    try:
        status = await get_bandit_status_nig(app.state.db)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alpha pool status: {e}")


@app.post("/alpha-pool/sample")
async def alpha_pool_sample(
    k: int = Query(default=BANDIT_SELECT_K, ge=1, le=50),
):
    """
    Perform Thompson Sampling from NIG posteriors.

    Returns sampled μ values and selected traders.
    Use this to see exploration/exploitation in action.
    """
    try:
        posteriors = await get_trader_posteriors_nig(app.state.db, limit=BANDIT_POOL_SIZE)
        if not posteriors:
            return {"count": 0, "samples": [], "selected": []}

        # Sample from each posterior
        samples = []
        for p in posteriors:
            sampled_mu = p.sample()
            samples.append({
                "address": p.address,
                "sampled_mu": round(sampled_mu, 4),
                "posterior_mean": round(p.m, 4),
                "nig_kappa": round(p.kappa, 2),
                "total_signals": p.total_signals,
            })

        # Sort by sampled value
        samples.sort(key=lambda x: x["sampled_mu"], reverse=True)

        # Select top k
        selected = thompson_sample_select_nig(posteriors, k=k)

        return {
            "count": len(samples),
            "samples": samples[:20],  # Top 20 for visibility
            "selected": [p.address for p in selected],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to sample: {e}")


# =====================
# Alpha Pool Management API
# =====================
#
# These endpoints manage the alpha_pool_addresses table, which is the source
# of truth for which addresses are in the Alpha Pool.
#
# This is COMPLETELY SEPARATE from the legacy leaderboard system:
# - /alpha-pool/refresh fetches from Hyperliquid API, not hl-scout
# - Addresses stored in alpha_pool_addresses, not hl_leaderboard_entries
# - No dependency on hl-scout's daily leaderboard sync
# =====================

HYPERLIQUID_LEADERBOARD_URL = "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard"
ALPHA_POOL_DEFAULT_SIZE = int(os.getenv("BANDIT_POOL_SIZE", "50"))


async def check_has_subaccounts(client: httpx.AsyncClient, address: str) -> bool:
    """
    Check if an address uses subaccounts.

    Addresses that trade via subaccounts can't be tracked directly since
    the master address has no activity - all trading happens on subaccounts.

    Args:
        client: Async HTTP client
        address: Ethereum address to check

    Returns:
        True if address has subaccounts, False otherwise
    """
    try:
        response = await client.post(
            HYPERLIQUID_INFO_URL,
            json={"type": "subAccounts", "user": address.lower()},
            timeout=5.0,
        )
        if response.status_code == 200:
            data = response.json()
            return isinstance(data, list) and len(data) > 0
        return False
    except Exception:
        return False


# Quality filter thresholds for Alpha Pool
ALPHA_POOL_MIN_PNL = float(os.getenv("ALPHA_POOL_MIN_PNL", "10000"))  # Min $10k 30d PnL
ALPHA_POOL_MIN_ROI = float(os.getenv("ALPHA_POOL_MIN_ROI", "0.10"))  # Min 10% 30d ROI
ALPHA_POOL_MIN_ACCOUNT_VALUE = float(os.getenv("ALPHA_POOL_MIN_ACCOUNT_VALUE", "100000"))  # Min $100k AV
ALPHA_POOL_MIN_WEEK_VLM = float(os.getenv("ALPHA_POOL_MIN_WEEK_VLM", "10000"))  # Min $10k weekly volume (filter inactive)
ALPHA_POOL_MAX_ORDERS_PER_DAY = float(os.getenv("ALPHA_POOL_MAX_ORDERS_PER_DAY", "100"))  # Max 100 orders/day (filter HFT)
ALPHA_POOL_REFRESH_HOURS = int(os.getenv("ALPHA_POOL_REFRESH_HOURS", "24"))  # Refresh interval in hours (default 24h)


async def analyze_user_fills(
    client: httpx.AsyncClient,
    address: str,
    max_retries: int = HL_API_MAX_RETRIES,
) -> Dict[str, Any]:
    """
    Analyze user's fill history for HFT detection and BTC/ETH trading.

    This uses the shared rate limiter and caches fills for later use by backfill.

    Metrics computed:
    - orders_per_day: HFT detection (> 100 orders/day = HFT)
    - has_btc_eth: Whether trader has BTC or ETH fills

    Returns dict with analysis results, or None on error after retries.
    """
    addr_lower = address.lower()

    for attempt in range(max_retries + 1):
        try:
            # Rate limit before making the call
            await _hl_rate_limiter.acquire()

            response = await client.post(
                HYPERLIQUID_INFO_URL,
                json={"type": "userFills", "user": addr_lower},
                timeout=15.0,
            )

            # Handle rate limiting with retry and Retry-After header
            if response.status_code == 429:
                if attempt < max_retries:
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = float(retry_after)
                        except ValueError:
                            delay = (attempt + 1) * 2.0
                    else:
                        delay = (attempt + 1) * 2.0
                    print(f"[hl-sage] Rate limited on fills for {addr_lower[:10]}..., retry {attempt + 1} in {delay:.1f}s")
                    await asyncio.sleep(delay)
                    continue
                return None

            if response.status_code != 200:
                if attempt < max_retries:
                    await asyncio.sleep(1.0)
                    continue
                return None

            fills = response.json()
            if not isinstance(fills, list):
                return None

            # Cache the raw fills for later use by backfill
            # Filter to BTC/ETH and convert to cached format
            btc_eth_fills = []
            for f in fills:
                coin = f.get("coin", "").upper()
                if coin in ("BTC", "ETH"):
                    btc_eth_fills.append({
                        "coin": coin,
                        "px": float(f.get("px", 0)),
                        "sz": float(f.get("sz", 0)),
                        "side": f.get("side", "B"),
                        "time": int(f.get("time", 0)),
                        "startPosition": float(f.get("startPosition", 0)),
                        "closedPnl": float(f.get("closedPnl", 0)) if f.get("closedPnl") else None,
                        "fee": float(f.get("fee", 0)) if f.get("fee") else None,
                        "hash": f.get("hash"),
                    })
            _fills_cache[addr_lower] = btc_eth_fills

            result = {
                "orders_per_day": 0.0,
                "has_btc_eth": len(btc_eth_fills) > 0,
                "fill_count": len(fills),
            }

            if len(fills) == 0:
                return result

            # Calculate orders per day for HFT detection (use all fills, not just BTC/ETH)
            orders_by_id: Dict[str, List[dict]] = {}
            for fill in fills:
                oid = str(fill.get("oid", ""))
                if oid not in orders_by_id:
                    orders_by_id[oid] = []
                orders_by_id[oid].append(fill)

            unique_orders = len(orders_by_id)
            if unique_orders > 0:
                all_times = [f.get("time", 0) for f in fills if f.get("time")]
                if len(all_times) >= 2:
                    first_time = min(all_times)
                    last_time = max(all_times)
                    span_days = (last_time - first_time) / (1000 * 60 * 60 * 24)
                    if span_days >= 0.01:  # At least ~15 minutes of data
                        result["orders_per_day"] = unique_orders / span_days

            return result
        except httpx.TimeoutException:
            if attempt < max_retries:
                print(f"[hl-sage] Timeout on fills for {addr_lower[:10]}..., retry {attempt + 1}")
                await asyncio.sleep(1.0)
                continue
            print(f"[hl-sage] Timeout analyzing fills for {address} after {max_retries + 1} attempts")
            return None
        except Exception as e:
            print(f"[hl-sage] Error analyzing fills for {address}: {e}")
            return None
    return None


async def fetch_leaderboard_from_api(
    limit: int = 100,
    progress_callback: Optional[callable] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch top traders from Hyperliquid leaderboard API with quality filtering.

    Quality filters applied:
    1. Positive 30d PnL (> ALPHA_POOL_MIN_PNL, default $10k)
    2. Positive 30d ROI (> ALPHA_POOL_MIN_ROI, default 10%)
    3. Minimum account value (> ALPHA_POOL_MIN_ACCOUNT_VALUE, default $100k)
    4. Recent activity (week volume > ALPHA_POOL_MIN_WEEK_VLM, default $10k)
    5. Remove HFT (orders/day > ALPHA_POOL_MAX_ORDERS_PER_DAY, default 100)
    6. No subaccounts (master addresses with subaccounts can't be tracked)
    7. BTC/ETH trading history (must have traded BTC or ETH)

    Args:
        limit: Number of qualified traders to return
        progress_callback: Optional async callback(step, progress, processed, total) for progress updates

    Returns:
        List of qualified trader data sorted by 30d PnL descending
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                HYPERLIQUID_LEADERBOARD_URL,
                timeout=15.0,
            )
            response.raise_for_status()
            data = response.json()

            qualified_traders = []
            filtered_counts = {
                "negative_pnl": 0,
                "low_roi": 0,
                "hft": 0,
                "low_account_value": 0,
                "inactive": 0,
                "subaccount": 0,
                "no_btc_eth": 0,
            }

            entries = data.get("leaderboardRows", []) if isinstance(data, dict) else []
            if not isinstance(entries, list):
                return []

            for entry in entries:
                if not isinstance(entry, dict):
                    continue

                address = entry.get("ethAddress", "").lower()
                if not address:
                    continue

                account_value = float(entry.get("accountValue", 0))
                display_name = entry.get("displayName")

                # Extract performance metrics
                window_perfs = entry.get("windowPerformances", [])
                month_pnl = 0.0
                month_roi = 0.0
                week_vlm = 0.0

                for wp in window_perfs:
                    if not isinstance(wp, list) or len(wp) < 2:
                        continue
                    wp_name = wp[0]
                    perf_data = wp[1] if isinstance(wp[1], dict) else {}

                    if wp_name == "month":
                        month_pnl = float(perf_data.get("pnl", 0))
                        month_roi = float(perf_data.get("roi", 0))
                    elif wp_name == "week":
                        week_vlm = float(perf_data.get("vlm", 0))

                # Apply quality filters (fast, no API calls)
                # 1. Positive PnL filter
                if month_pnl < ALPHA_POOL_MIN_PNL:
                    filtered_counts["negative_pnl"] += 1
                    continue

                # 2. Positive ROI filter
                if month_roi < ALPHA_POOL_MIN_ROI:
                    filtered_counts["low_roi"] += 1
                    continue

                # 3. Minimum account value filter
                if account_value < ALPHA_POOL_MIN_ACCOUNT_VALUE:
                    filtered_counts["low_account_value"] += 1
                    continue

                # 4. Activity filter (must have traded in the last week)
                if week_vlm < ALPHA_POOL_MIN_WEEK_VLM:
                    filtered_counts["inactive"] += 1
                    continue

                # Passed initial filters - add to candidates for API-based checks
                qualified_traders.append({
                    "address": address,
                    "account_value": account_value,
                    "pnl": month_pnl,
                    "roi": month_roi,
                    "week_vlm": week_vlm,
                    "win_rate": 0.0,  # Not available in this API
                    "display_name": display_name,
                })

                # Collect extra candidates to account for API-based filtering
                # (subaccount check + BTC/ETH history check)
                # Need many more since ~90% get filtered by BTC/ETH check
                if len(qualified_traders) >= limit * 15:
                    break

            print(f"[hl-sage] Alpha Pool initial filter stats: {len(qualified_traders)} candidates, filtered: {filtered_counts}")

            # Sort by PnL descending before API-based checks
            qualified_traders.sort(key=lambda t: t["pnl"], reverse=True)

            # Report progress: initial filtering complete
            if progress_callback:
                await progress_callback("filtering_candidates", 10, 0, len(qualified_traders))

            # 5, 6, 7. API-based filters: HFT, subaccounts, and BTC/ETH trading history
            # These require individual API calls so we do them last
            final_traders = []
            processed_count = 0
            total_candidates = len(qualified_traders)
            for trader in qualified_traders:
                if len(final_traders) >= limit:
                    break

                processed_count += 1
                # Report progress: processing candidates (10-80% range)
                if progress_callback and processed_count % 5 == 0:
                    progress = 10 + int((processed_count / total_candidates) * 70)
                    await progress_callback("analyzing_traders", min(progress, 80), processed_count, total_candidates)

                # 5 & 7. Combined fill analysis (HFT + BTC/ETH check in one API call)
                fill_analysis = await analyze_user_fills(client, trader["address"])
                if fill_analysis is None:
                    # API error - skip this trader
                    print(f"[hl-sage] Failed to analyze fills for: {trader['address']}")
                    continue

                # 5. HFT filter: check orders per day
                orders_per_day = fill_analysis["orders_per_day"]
                if orders_per_day > ALPHA_POOL_MAX_ORDERS_PER_DAY:
                    filtered_counts["hft"] += 1
                    print(f"[hl-sage] Filtered HFT ({orders_per_day:.1f} orders/day): {trader['address']}")
                    continue

                # 6. Subaccount filter: check if addresses use subaccounts
                # (trading happens on subaccounts which we can't track directly)
                has_subaccounts = await check_has_subaccounts(client, trader["address"])
                if has_subaccounts:
                    filtered_counts["subaccount"] += 1
                    print(f"[hl-sage] Filtered subaccount user: {trader['address']}")
                    continue

                # 7. BTC/ETH filter: must have traded BTC or ETH
                if not fill_analysis["has_btc_eth"]:
                    filtered_counts["no_btc_eth"] += 1
                    print(f"[hl-sage] Filtered no BTC/ETH history: {trader['address']}")
                    continue

                # Store orders_per_day for debugging/display
                trader["orders_per_day"] = orders_per_day
                final_traders.append(trader)

            print(f"[hl-sage] Alpha Pool final: {len(final_traders)} qualified, API filtered: hft={filtered_counts['hft']}, subaccount={filtered_counts['subaccount']}, no_btc_eth={filtered_counts['no_btc_eth']}")
            return final_traders
    except Exception as e:
        print(f"[hl-sage] Failed to fetch leaderboard: {e}")
        return []


async def _do_refresh_alpha_pool(limit: int, report_progress: bool = True) -> Dict[str, Any]:
    """
    Internal function to perform Alpha Pool refresh with progress tracking.

    Args:
        limit: Number of traders to fetch
        report_progress: Whether to update _refresh_state with progress

    Returns:
        Dict with refresh results
    """
    async def progress_callback(step: str, progress: int, processed: int, total: int):
        if report_progress:
            await _refresh_state.update(step, progress, processed, total)

    try:
        # Fetch from Hyperliquid with quality filtering
        if report_progress:
            await _refresh_state.update("fetching_leaderboard", 5, 0, 0)

        traders = await fetch_leaderboard_from_api(limit=limit, progress_callback=progress_callback)
        if not traders:
            raise Exception("Failed to fetch qualified traders from Hyperliquid (check filter thresholds)")

        if report_progress:
            await _refresh_state.update("saving_traders", 85, len(traders), len(traders))

        # Upsert into alpha_pool_addresses and track newly inserted addresses
        newly_inserted_addresses = []
        async with app.state.db.acquire() as conn:
            inserted = 0
            updated = 0
            for trader in traders:
                # Check if address already exists
                existing = await conn.fetchval(
                    "SELECT 1 FROM alpha_pool_addresses WHERE address = $1",
                    trader["address"],
                )

                result = await conn.execute(
                    """
                    INSERT INTO alpha_pool_addresses (address, nickname, account_value, pnl_30d, roi_30d, win_rate, last_refreshed, is_active)
                    VALUES ($1, $2, $3, $4, $5, $6, NOW(), true)
                    ON CONFLICT (address) DO UPDATE SET
                        nickname = COALESCE(EXCLUDED.nickname, alpha_pool_addresses.nickname),
                        account_value = EXCLUDED.account_value,
                        pnl_30d = EXCLUDED.pnl_30d,
                        roi_30d = EXCLUDED.roi_30d,
                        win_rate = EXCLUDED.win_rate,
                        last_refreshed = NOW(),
                        is_active = true
                    """,
                    trader["address"],
                    trader.get("display_name"),
                    trader["account_value"],
                    trader["pnl"],
                    trader.get("roi", 0.0),
                    trader["win_rate"],
                )
                if not existing:
                    inserted += 1
                    newly_inserted_addresses.append(trader["address"])
                else:
                    updated += 1

            # Deactivate addresses not in the new batch
            new_addresses = [t["address"] for t in traders]
            await conn.execute(
                """
                UPDATE alpha_pool_addresses
                SET is_active = false
                WHERE address NOT IN (SELECT UNNEST($1::text[]))
                  AND is_active = true
                """,
                new_addresses,
            )

        if report_progress:
            await _refresh_state.update("backfilling_fills", 90, len(traders), len(traders))

        # Backfill historical fills for NEWLY INSERTED addresses only
        # Existing addresses should already have their fills from previous backfills
        if newly_inserted_addresses:
            print(f"[hl-sage] Starting historical fill backfill for {len(newly_inserted_addresses)} NEW addresses...")
            backfill_results = await backfill_historical_fills_for_addresses(app.state.db, newly_inserted_addresses)
            total_fills_inserted = sum(backfill_results.values())
            print(f"[hl-sage] Backfill complete: {total_fills_inserted} fills inserted for {len(backfill_results)} addresses")
        else:
            print(f"[hl-sage] No new addresses to backfill (all {len(traders)} addresses already existed)")

        # Clear fills cache after backfill - no longer needed and prevents stale data
        _fills_cache.clear()
        print(f"[hl-sage] Cleared fills cache after backfill")

        if report_progress:
            await _refresh_state.update("reconciling", 95, len(traders), len(traders))

        # Notify hl-decide to reconcile the new fills into episodes
        try:
            async with httpx.AsyncClient() as client:
                # Use internal Docker port 8080, not external mapped port
                decide_url = os.getenv("DECIDE_URL", "http://hl-decide:8080")
                response = await client.post(f"{decide_url}/reconcile", timeout=60.0)
                if response.status_code == 200:
                    reconcile_result = response.json()
                    print(f"[hl-sage] hl-decide reconciliation: {reconcile_result}")
                else:
                    print(f"[hl-sage] hl-decide reconciliation failed: {response.status_code}")
        except Exception as e:
            print(f"[hl-sage] Failed to notify hl-decide for reconciliation: {e}")

        return {
            "success": True,
            "fetched": len(traders),
            "inserted": inserted,
            "updated": updated,
            "fills_backfilled": total_fills_inserted,
            "filters": {
                "min_pnl": ALPHA_POOL_MIN_PNL,
                "min_roi": ALPHA_POOL_MIN_ROI,
                "min_account_value": ALPHA_POOL_MIN_ACCOUNT_VALUE,
                "min_week_vlm": ALPHA_POOL_MIN_WEEK_VLM,
                "max_orders_per_day": ALPHA_POOL_MAX_ORDERS_PER_DAY,
            },
        }
    except Exception as e:
        # Clear cache on error too
        _fills_cache.clear()
        raise


async def _background_refresh_task(limit: int):
    """Background task wrapper for refresh with error handling."""
    try:
        await _refresh_state.start()
        result = await _do_refresh_alpha_pool(limit, report_progress=True)
        await _refresh_state.complete(result)
        print(f"[hl-sage] Background refresh completed: {result}")
    except Exception as e:
        await _refresh_state.fail(str(e))
        print(f"[hl-sage] Background refresh failed: {e}")


@app.get("/alpha-pool/refresh/status")
async def get_refresh_status():
    """
    Get the status of the Alpha Pool refresh operation.

    Returns current state including:
    - is_running: Whether a refresh is in progress
    - started_at: When the current/last refresh started
    - completed_at: When the last refresh completed
    - elapsed_seconds: Duration of current/last refresh
    - current_step: Current operation (fetching_leaderboard, analyzing_traders, etc.)
    - progress: Progress percentage (0-100)
    - processed_traders: Number of traders processed so far
    - total_traders: Total traders to process
    - result: Final result (if completed)
    - error: Error message (if failed)
    """
    return await _refresh_state.get_status()


@app.post("/alpha-pool/refresh")
async def refresh_alpha_pool(
    limit: int = Query(default=ALPHA_POOL_DEFAULT_SIZE, ge=10, le=200),
    background: bool = Query(default=True, description="Run refresh in background (non-blocking)"),
):
    """
    Refresh Alpha Pool addresses from Hyperliquid leaderboard API.

    Fetches top traders directly from Hyperliquid (stats-data.hyperliquid.xyz)
    and populates the alpha_pool_addresses table.

    Quality filters applied (7 gates):
    1. ALPHA_POOL_MIN_PNL: Minimum 30d PnL (default $10k)
    2. ALPHA_POOL_MIN_ROI: Minimum 30d ROI (default 10%)
    3. ALPHA_POOL_MIN_ACCOUNT_VALUE: Minimum account value (default $100k)
    4. ALPHA_POOL_MIN_WEEK_VLM: Minimum weekly volume to filter inactive (default $10k)
    5. ALPHA_POOL_MAX_ORDERS_PER_DAY: Maximum orders/day to filter HFT (default 100)
    6. No subaccounts: Filters addresses that use subaccounts (untrackable)
    7. BTC/ETH history: Must have traded BTC or ETH (we only track these)

    This is INDEPENDENT from hl-scout's leaderboard sync:
    - Uses different API endpoint (stats-data vs hyperbot.network)
    - Stores in different table (alpha_pool_addresses vs hl_leaderboard_entries)
    - Can be called on-demand, not tied to daily sync schedule

    The refresh deactivates addresses not in the new batch (is_active=false)
    rather than deleting them, preserving historical data.

    Args:
        limit: Number of traders to fetch (default 50)
        background: If true (default), runs in background and returns immediately.
                   Use GET /alpha-pool/refresh/status to check progress.
    """
    # Check if refresh is already running
    status = await _refresh_state.get_status()
    if status["is_running"]:
        raise HTTPException(
            status_code=409,
            detail="A refresh is already in progress. Check /alpha-pool/refresh/status for progress."
        )

    if background:
        # Start background task and return immediately
        asyncio.create_task(_background_refresh_task(limit))
        return {
            "status": "started",
            "message": "Refresh started in background. Check /alpha-pool/refresh/status for progress.",
            "limit": limit,
        }
    else:
        # Synchronous execution (blocking)
        try:
            await _refresh_state.start()
            result = await _do_refresh_alpha_pool(limit, report_progress=True)
            await _refresh_state.complete(result)
            return result
        except Exception as e:
            await _refresh_state.fail(str(e))
            raise HTTPException(status_code=500, detail=f"Failed to refresh alpha pool: {e}")


@app.get("/alpha-pool/addresses")
async def list_alpha_pool_addresses(
    active_only: bool = Query(default=True),
    limit: int = Query(default=100, ge=1, le=500),
):
    """
    List addresses in the Alpha Pool.

    Returns addresses from alpha_pool_addresses table.
    """
    try:
        async with app.state.db.acquire() as conn:
            if active_only:
                rows = await conn.fetch(
                    """
                    SELECT address, nickname, account_value, pnl_30d, win_rate, last_refreshed, source
                    FROM alpha_pool_addresses
                    WHERE is_active = true
                    ORDER BY pnl_30d DESC NULLS LAST
                    LIMIT $1
                    """,
                    limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT address, nickname, account_value, pnl_30d, win_rate, last_refreshed, source, is_active
                    FROM alpha_pool_addresses
                    ORDER BY is_active DESC, pnl_30d DESC NULLS LAST
                    LIMIT $1
                    """,
                    limit,
                )

        return {
            "count": len(rows),
            "addresses": [dict(row) for row in rows],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list addresses: {e}")


@app.post("/alpha-pool/backfill/{address}")
async def backfill_address_fills(address: str):
    """
    Backfill historical fills for a specific address.

    This fetches fills from Hyperliquid API and stores them in hl_events.
    Useful for:
    - Addresses that were added to the pool but backfill failed
    - Inactive addresses that need historical data
    - Manual data recovery

    The address must exist in alpha_pool_addresses (active or inactive).
    """
    addr_lower = address.lower()

    # Verify address is in the pool (active or inactive)
    async with app.state.db.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT address, is_active FROM alpha_pool_addresses WHERE address = $1",
            addr_lower,
        )
        if not row:
            raise HTTPException(status_code=404, detail=f"Address {address} not found in Alpha Pool")

    # Perform backfill
    try:
        results = await backfill_historical_fills_for_addresses(app.state.db, [addr_lower])
        inserted = results.get(addr_lower, 0)
        return {
            "address": addr_lower,
            "fills_inserted": inserted,
            "is_active": row["is_active"],
            "message": f"Backfilled {inserted} fills for {addr_lower[:10]}...",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to backfill: {e}")


@app.post("/alpha-pool/backfill-all")
async def backfill_all_addresses(
    delay_ms: int = Query(default=500, ge=100, le=5000, description="Delay between requests in milliseconds"),
    active_only: bool = Query(default=True, description="Only backfill active addresses"),
):
    """
    Backfill historical fills for ALL Alpha Pool addresses.

    Use this after a fresh system rebuild to populate historical data.
    Respects rate limits with configurable delay between requests.

    Args:
        delay_ms: Milliseconds to wait between API calls (default 500ms)
        active_only: If true, only backfill active addresses (default true)

    Returns:
        Summary of backfill results for all addresses
    """
    import asyncio

    # Get all addresses
    async with app.state.db.acquire() as conn:
        if active_only:
            rows = await conn.fetch(
                "SELECT address FROM alpha_pool_addresses WHERE is_active = true ORDER BY added_at"
            )
        else:
            rows = await conn.fetch(
                "SELECT address FROM alpha_pool_addresses ORDER BY added_at"
            )

    addresses = [row["address"] for row in rows]

    if not addresses:
        return {
            "total_addresses": 0,
            "total_fills": 0,
            "message": "No addresses found in Alpha Pool",
        }

    # Backfill each address with delay
    results = {}
    total_fills = 0
    errors = []

    for i, addr in enumerate(addresses):
        try:
            print(f"[backfill-all] ({i+1}/{len(addresses)}) Backfilling {addr[:10]}...")
            result = await backfill_historical_fills_for_addresses(app.state.db, [addr])
            inserted = result.get(addr, 0)
            results[addr] = inserted
            total_fills += inserted

            # Rate limit delay (except for last address)
            if i < len(addresses) - 1:
                await asyncio.sleep(delay_ms / 1000.0)

        except Exception as e:
            print(f"[backfill-all] Error backfilling {addr}: {e}")
            errors.append({"address": addr, "error": str(e)})
            results[addr] = 0

    return {
        "total_addresses": len(addresses),
        "total_fills": total_fills,
        "addresses_with_fills": sum(1 for v in results.values() if v > 0),
        "addresses_with_no_fills": sum(1 for v in results.values() if v == 0),
        "errors": len(errors),
        "error_details": errors[:10] if errors else [],  # First 10 errors
        "message": f"Backfilled {total_fills} fills across {len(addresses)} addresses",
    }


# =====================
# Shadow Ledger API (Phase 3f: Selection Integrity)
# =====================
#
# The Shadow Ledger captures daily snapshots of all traders for survivorship-bias-free
# analysis. It enables:
# - Walk-forward replay without look-ahead bias
# - Survival analysis (who blew up and when)
# - FDR-controlled skill qualification
# - Thompson sampling with stored draws for reproducibility
# =====================


@app.post("/snapshots/create")
async def create_snapshot(
    snapshot_date: Optional[str] = Query(default=None, description="Date in YYYY-MM-DD format (default: today)"),
):
    """
    Create a daily snapshot for the Shadow Ledger.

    This captures the current state of all traders including:
    - Universe membership (leaderboard_scanned, candidate_filtered, quality_qualified, pool_selected)
    - NIG posteriors and Thompson sampling draws with stored seeds
    - FDR qualification via Benjamini-Hochberg procedure
    - Death/censor event detection

    Normally runs automatically at SNAPSHOT_HOUR_UTC, but can be triggered manually.

    Args:
        snapshot_date: Optional date to create snapshot for (default: today).
                      Useful for backfilling historical snapshots.
    """
    try:
        from datetime import date as date_type

        target_date = None
        if snapshot_date:
            try:
                target_date = date_type.fromisoformat(snapshot_date)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid date format: {snapshot_date}. Use YYYY-MM-DD.")

        result = await create_daily_snapshot(app.state.db, snapshot_date=target_date)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create snapshot: {e}")


@app.get("/snapshots/summary")
async def get_snapshots_summary(
    snapshot_date: Optional[str] = Query(default=None, description="Date in YYYY-MM-DD format (default: today)"),
):
    """
    Get summary statistics for a snapshot date.

    Returns counts by universe membership and top selected traders.
    """
    try:
        from datetime import date as date_type

        target_date = None
        if snapshot_date:
            try:
                target_date = date_type.fromisoformat(snapshot_date)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid date format: {snapshot_date}. Use YYYY-MM-DD.")

        result = await get_snapshot_summary(app.state.db, snapshot_date=target_date)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get snapshot summary: {e}")


@app.get("/snapshots/universe")
async def get_universe_at_date(
    evaluation_date: str = Query(..., description="Date in YYYY-MM-DD format"),
    version: Optional[str] = Query(default=None, description="Selection version (default: current)"),
):
    """
    Get the trader universe as-of a specific date.

    CRITICAL: This uses the snapshot table, NOT current qualification.
    This prevents look-ahead bias in walk-forward replay.

    Returns the list of addresses that were in the universe on that date.
    """
    try:
        from datetime import date as date_type

        try:
            target_date = date_type.fromisoformat(evaluation_date)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid date format: {evaluation_date}. Use YYYY-MM-DD.")

        addresses = await load_universe_at_date(
            app.state.db,
            evaluation_date=target_date,
            version=version or SELECTION_VERSION,
        )

        return {
            "evaluation_date": target_date.isoformat(),
            "version": version or SELECTION_VERSION,
            "count": len(addresses),
            "addresses": addresses,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get universe: {e}")


@app.get("/snapshots/history")
async def get_snapshot_history(
    address: str = Query(..., description="Trader address"),
    limit: int = Query(default=30, ge=1, le=365),
):
    """
    Get snapshot history for a specific trader.

    Returns the trader's snapshots over time, useful for:
    - Analyzing performance trajectory
    - Understanding universe membership changes
    - Detecting death/censor events
    """
    try:
        addr_lower = address.lower()
        async with app.state.db.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    snapshot_date,
                    selection_version,
                    is_leaderboard_scanned,
                    is_candidate_filtered,
                    is_quality_qualified,
                    is_pool_selected,
                    avg_r_gross,
                    avg_r_net,
                    nig_mu as nig_m,
                    nig_kappa,
                    thompson_draw,
                    skill_p_value,
                    fdr_qualified,
                    event_type,
                    death_type,
                    censor_type,
                    episode_count,
                    selection_rank
                FROM trader_snapshots
                WHERE address = $1
                ORDER BY snapshot_date DESC
                LIMIT $2
                """,
                addr_lower,
                limit,
            )

        return {
            "address": addr_lower,
            "count": len(rows),
            "snapshots": [dict(row) for row in rows],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get snapshot history: {e}")


@app.get("/snapshots/deaths")
async def get_death_events(
    days: int = Query(default=30, ge=1, le=365),
    death_type: Optional[str] = Query(default=None, description="Filter by death type"),
):
    """
    Get recent death events from the Shadow Ledger.

    Death types (terminal events):
    - liquidation: Account liquidated on Hyperliquid
    - drawdown_80: Current equity < 20% of peak
    - account_value_floor: Account dropped below $10k
    - negative_equity: Account value <= 0

    Useful for survival analysis and understanding trader lifecycle.
    """
    try:
        from datetime import date as date_type

        cutoff_date = date_type.today() - timedelta(days=days)

        async with app.state.db.acquire() as conn:
            if death_type:
                rows = await conn.fetch(
                    """
                    SELECT
                        address,
                        snapshot_date,
                        death_type,
                        account_value,
                        peak_account_value,
                        avg_r_net,
                        episode_count
                    FROM trader_snapshots
                    WHERE event_type = 'death'
                      AND death_type = $1
                      AND snapshot_date >= $2
                    ORDER BY snapshot_date DESC
                    """,
                    death_type,
                    cutoff_date,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT
                        address,
                        snapshot_date,
                        death_type,
                        account_value,
                        peak_account_value,
                        avg_r_net,
                        episode_count
                    FROM trader_snapshots
                    WHERE event_type = 'death'
                      AND snapshot_date >= $2
                    ORDER BY snapshot_date DESC
                    """,
                    cutoff_date,
                )

        # Group by death type
        by_type = {}
        for row in rows:
            dt = row["death_type"]
            if dt not in by_type:
                by_type[dt] = []
            by_type[dt].append(dict(row))

        return {
            "period_days": days,
            "total_deaths": len(rows),
            "by_type": {k: len(v) for k, v in by_type.items()},
            "events": [dict(row) for row in rows],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get death events: {e}")


@app.get("/snapshots/config")
async def get_snapshot_config():
    """
    Get current snapshot configuration.
    """
    from .snapshot import (
        SNAPSHOT_FDR_ALPHA,
        SNAPSHOT_MIN_EPISODES,
        SNAPSHOT_MIN_AVG_R_NET,
        DEATH_DRAWDOWN_PCT,
        DEATH_ACCOUNT_FLOOR,
        CENSOR_INACTIVE_DAYS,
        ROUND_TRIP_COST_BPS,
    )

    return {
        "enabled": SNAPSHOT_ENABLED,
        "hour_utc": SNAPSHOT_HOUR_UTC,
        "selection_version": SELECTION_VERSION,
        "fdr_alpha": SNAPSHOT_FDR_ALPHA,
        "min_episodes": SNAPSHOT_MIN_EPISODES,
        "min_avg_r_net": SNAPSHOT_MIN_AVG_R_NET,
        "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
        "death_thresholds": {
            "drawdown_pct": DEATH_DRAWDOWN_PCT,
            "account_floor": DEATH_ACCOUNT_FLOOR,
        },
        "censor_thresholds": {
            "inactive_days": CENSOR_INACTIVE_DAYS,
        },
    }


# =====================
# Walk-Forward Replay API (Phase 3f)
# =====================


@app.post("/replay/run")
async def run_replay(
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: Optional[str] = Query(default=None, description="End date in YYYY-MM-DD (default: today)"),
    evaluation_days: int = Query(default=REPLAY_EVALUATION_DAYS, ge=1, le=30),
    version: Optional[str] = Query(default=None, description="Selection version (default: current)"),
):
    """
    Run a walk-forward replay over a date range.

    This provides an honest out-of-sample assessment:
    - Selection uses only data available at selection_date (from snapshots)
    - Performance measured on FUTURE data (not training data)
    - Costs included in net returns

    Process for each snapshot date:
    1. Load traders selected on that date (from trader_snapshots)
    2. Evaluate their performance over the next evaluation_days
    3. Compute gross and net R-multiples (with costs)
    4. Track deaths/censors

    Args:
        start_date: First selection date to replay
        end_date: Last selection date to replay (default: today)
        evaluation_days: Days to evaluate each selection (default: 7)
        version: Selection version to replay (default: current)
    """
    try:
        from datetime import date as date_type

        try:
            start = date_type.fromisoformat(start_date)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid start_date format: {start_date}")

        if end_date:
            try:
                end = date_type.fromisoformat(end_date)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid end_date format: {end_date}")
        else:
            end = date_type.today()

        if start > end:
            raise HTTPException(status_code=400, detail="start_date must be before end_date")

        summary = await run_walk_forward_replay(
            app.state.db,
            start_date=start,
            end_date=end,
            evaluation_days=evaluation_days,
            version=version or SELECTION_VERSION,
        )

        return format_replay_summary(summary)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run replay: {e}")


@app.get("/replay/period")
async def replay_period(
    selection_date: str = Query(..., description="Date in YYYY-MM-DD format"),
    evaluation_days: int = Query(default=REPLAY_EVALUATION_DAYS, ge=1, le=30),
    version: Optional[str] = Query(default=None, description="Selection version"),
):
    """
    Replay a single selection period.

    Returns detailed results for traders selected on the given date,
    including their performance over the evaluation window.
    """
    try:
        from datetime import date as date_type

        try:
            target_date = date_type.fromisoformat(selection_date)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid date format: {selection_date}")

        result = await replay_single_period(
            app.state.db,
            selection_date=target_date,
            evaluation_days=evaluation_days,
            version=version or SELECTION_VERSION,
        )

        if not result:
            raise HTTPException(status_code=404, detail=f"No snapshot data for {selection_date}")

        return {
            "selection_date": result.selection_date.isoformat(),
            "evaluation_start": result.evaluation_start.isoformat(),
            "evaluation_end": result.evaluation_end.isoformat(),
            "universe_size": result.universe_size,
            "selected_count": result.selected_count,
            "fdr_qualified_count": result.fdr_qualified_count,
            "total_r_gross": round(result.total_r_gross, 4),
            "total_r_net": round(result.total_r_net, 4),
            "avg_r_gross": round(result.avg_r_gross, 4),
            "avg_r_net": round(result.avg_r_net, 4),
            "deaths": result.deaths_during_period,
            "censored": result.censored_during_period,
            "traders": result.trader_results,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to replay period: {e}")
