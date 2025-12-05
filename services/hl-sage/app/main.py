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
import asyncio
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

SERVICE_NAME = "hl-sage"
OWNER_TOKEN = os.getenv("OWNER_TOKEN", "dev-owner")
NATS_URL = os.getenv("NATS_URL", "nats://0.0.0.0:4222")
DB_URL = os.getenv("DATABASE_URL", "postgresql://hlbot:hlbotpassword@0.0.0.0:5432/hlbot")
MAX_TRACKED_ADDRESSES = int(os.getenv("MAX_TRACKED_ADDRESSES", "1000"))
MAX_SCORES = int(os.getenv("MAX_SCORES", "500"))
STALE_THRESHOLD_HOURS = int(os.getenv("STALE_THRESHOLD_HOURS", "24"))

app = FastAPI(title="hl-sage", version="0.1.0")

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

    # Use NIG posterior mean as score if available, else fall back to leaderboard weight
    if nig_params and nig_params.get("nig_m") is not None:
        # NIG-based score: posterior mean * direction
        nig_score = nig_params["nig_m"] * side_multiplier
        # Clamp to [-1, 1] range
        base_score = max(-1.0, min(1.0, nig_score))
        score_source = "nig"
    else:
        # Legacy: leaderboard weight * direction
        base_score = max(-1.0, min(1.0, state["weight"] * side_multiplier))
        score_source = "leaderboard"

    event = ScoreEvent(
        address=data.address,
        score=base_score,
        weight=state["weight"],
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
        Dict with nig_m, nig_kappa, nig_alpha, nig_beta, total_signals, avg_r
        or None if trader has no posterior data
    """
    try:
        async with app.state.db.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT nig_m, nig_kappa, nig_alpha, nig_beta,
                       total_signals, avg_r
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
                }
    except Exception as e:
        print(f"[hl-sage] Failed to fetch NIG params for {address}: {e}")
    return None


ALPHA_POOL_AUTO_REFRESH = os.getenv("ALPHA_POOL_AUTO_REFRESH", "true").lower() == "true"


@app.on_event("startup")
async def startup_event():
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
    except Exception as e:
        print(f"[hl-sage] Fatal startup error: {e}")
        raise


async def auto_refresh_alpha_pool_if_empty():
    """
    Auto-refresh Alpha Pool on startup ONLY if it has NEVER been refreshed.

    Checks if alpha_pool_addresses table has ANY records (active or not).
    If completely empty, this is a fresh database and we bootstrap the pool.

    Subsequent refreshes should be done via POST /alpha-pool/refresh
    or a scheduled job.

    Runs as a background task to not block service startup.
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
            print(f"[hl-sage] Alpha Pool has never been refreshed, auto-refreshing (first-time setup)...")
            traders = await fetch_leaderboard_from_api(limit=ALPHA_POOL_DEFAULT_SIZE)
            if traders:
                async with app.state.db.acquire() as conn:
                    for trader in traders:
                        await conn.execute(
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
                print(f"[hl-sage] Alpha Pool auto-refreshed with {len(traders)} traders")
            else:
                print(f"[hl-sage] Alpha Pool auto-refresh failed - no traders fetched")
        else:
            print(f"[hl-sage] Alpha Pool has {total_count} records (previously refreshed), skipping auto-refresh")
    except Exception as e:
        print(f"[hl-sage] Alpha Pool auto-refresh failed: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    if hasattr(app.state, "nc"):
        await app.state.nc.drain()
    if hasattr(app.state, "db"):
        await app.state.db.close()


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
PNL_CURVE_CONCURRENCY = 2  # Max concurrent requests (reduced to avoid rate limits)
PNL_CURVE_TIMEOUT = 10.0  # Timeout per request in seconds
PNL_CURVE_DELAY = 0.3  # Delay between requests in seconds
PNL_CURVE_MAX_RETRIES = 2  # Max retries on rate limit
PNL_CURVE_CACHE_TTL = int(os.getenv("PNL_CURVE_CACHE_TTL", "86400"))  # Cache TTL in seconds (default 24 hours)

# In-memory cache for PnL curves: {address: (timestamp, curve_data)}
_pnl_curve_cache: Dict[str, tuple] = {}


async def fetch_pnl_curve_from_api(
    client: httpx.AsyncClient,
    address: str,
    window: str = "month",
    retries: int = 0,
) -> List[Dict[str, Any]]:
    """
    Fetch PnL curve for a single address directly from Hyperliquid API.

    Args:
        client: Async HTTP client
        address: Ethereum address
        window: Time window ('day', 'week', 'month', 'allTime', 'period_30', etc.)
        retries: Current retry count

    Returns:
        List of {ts, value} points
    """
    try:
        response = await client.post(
            HYPERLIQUID_INFO_URL,
            json={"type": "portfolio", "user": address.lower()},
            timeout=PNL_CURVE_TIMEOUT,
        )

        # Handle rate limiting with retry
        if response.status_code == 429 and retries < PNL_CURVE_MAX_RETRIES:
            delay = (retries + 1) * 2.0  # Exponential backoff: 2s, 4s
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
            # Match 'month' or 'period_30' for 30-day curve
            if window_name == window or window_name == "month":
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
    window: str = "month",
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetch PnL curves for multiple addresses from Hyperliquid API with caching.

    Uses in-memory cache with TTL to avoid unnecessary API calls.
    30-day PnL curves don't change frequently, so caching is safe.

    Args:
        addresses: List of Ethereum addresses
        window: Time window (default 'month' for 30-day curve)

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
        pnl_curves = await get_pnl_curves_for_addresses(addresses)
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
    max_retries: int = 2,
) -> Dict[str, Any]:
    """
    Analyze user's fill history for HFT detection and BTC/ETH trading.

    This combines multiple checks into one API call to avoid rate limits:
    - orders_per_day: HFT detection (> 100 orders/day = HFT)
    - has_btc_eth: Whether trader has BTC or ETH fills

    Returns dict with analysis results, or None on error after retries.
    """
    for attempt in range(max_retries + 1):
        try:
            response = await client.post(
                HYPERLIQUID_INFO_URL,
                json={"type": "userFills", "user": address.lower()},
                timeout=15.0,  # Increased timeout
            )

            # Handle rate limiting with retry
            if response.status_code == 429:
                if attempt < max_retries:
                    delay = (attempt + 1) * 2.0
                    print(f"[hl-sage] Rate limited on fills for {address}, retry {attempt + 1} in {delay}s")
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

            result = {
                "orders_per_day": 0.0,
                "has_btc_eth": False,
                "fill_count": len(fills),
            }

            if len(fills) == 0:
                return result

            # Check for BTC/ETH fills
            for fill in fills:
                coin = fill.get("coin", "").upper()
                if coin in ("BTC", "ETH"):
                    result["has_btc_eth"] = True
                    break

            # Calculate orders per day for HFT detection
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
                print(f"[hl-sage] Timeout on fills for {address}, retry {attempt + 1}")
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

            # 5, 6, 7. API-based filters: HFT, subaccounts, and BTC/ETH trading history
            # These require individual API calls so we do them last
            final_traders = []
            for trader in qualified_traders:
                if len(final_traders) >= limit:
                    break

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


@app.post("/alpha-pool/refresh")
async def refresh_alpha_pool(
    limit: int = Query(default=ALPHA_POOL_DEFAULT_SIZE, ge=10, le=200),
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
    """
    try:
        # Fetch from Hyperliquid with quality filtering
        traders = await fetch_leaderboard_from_api(limit=limit)
        if not traders:
            raise HTTPException(status_code=502, detail="Failed to fetch qualified traders from Hyperliquid (check filter thresholds)")

        # Upsert into alpha_pool_addresses
        async with app.state.db.acquire() as conn:
            inserted = 0
            updated = 0
            for trader in traders:
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
                if "INSERT" in result:
                    inserted += 1
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

        return {
            "success": True,
            "fetched": len(traders),
            "inserted": inserted,
            "updated": updated,
            "filters": {
                "min_pnl": ALPHA_POOL_MIN_PNL,
                "min_roi": ALPHA_POOL_MIN_ROI,
                "min_account_value": ALPHA_POOL_MIN_ACCOUNT_VALUE,
                "min_week_vlm": ALPHA_POOL_MIN_WEEK_VLM,
                "max_orders_per_day": ALPHA_POOL_MAX_ORDERS_PER_DAY,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
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
