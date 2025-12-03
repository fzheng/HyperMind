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
from datetime import datetime, timezone
from typing import Dict, Optional
from uuid import uuid4
from collections import OrderedDict

import asyncpg
import nats
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
from starlette.responses import Response

from contracts.py.models import FillEvent, ScoreEvent
from .consensus import ConsensusDetector, Fill, ConsensusSignal

SERVICE_NAME = "hl-decide"
NATS_URL = os.getenv("NATS_URL", "nats://localhost:4222")
DB_URL = os.getenv("DATABASE_URL", "postgresql://hlbot:hlbotpassword@localhost:5432/hlbot")
MAX_SCORES = int(os.getenv("MAX_SCORES", "500"))
MAX_FILLS = int(os.getenv("MAX_FILLS", "500"))

# R-multiple calculation: assumed stop loss fraction (1% = 0.01)
ASSUMED_STOP_FRACTION = float(os.getenv("ASSUMED_STOP_FRACTION", "0.01"))

# NIG prior parameters
NIG_PRIOR_M = 0.0
NIG_PRIOR_KAPPA = 1.0
NIG_PRIOR_ALPHA = 3.0
NIG_PRIOR_BETA = 1.0

app = FastAPI(title="hl-decide", version="0.3.0")
scores: OrderedDict[str, ScoreEvent] = OrderedDict()
fills: OrderedDict[str, FillEvent] = OrderedDict()

# Consensus detector for Alpha Pool signal generation
consensus_detector = ConsensusDetector()

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
    1. Position tracking (open/close detection for R-multiple calculation)
    2. Consensus detection (Alpha Pool signal generation)
    """
    data = FillEvent.model_validate_json(msg.data.decode())

    # Update fill cache
    if data.address in fills:
        fills.move_to_end(data.address)
    fills[data.address] = data
    await persist_fill(data.address, data)
    enforce_limits()
    fill_counter.inc()

    # Process for position tracking (this is where the magic happens)
    await handle_fill_for_positions(data)

    # Process for consensus detection
    await process_fill_for_consensus(data)


async def process_fill_for_consensus(data: FillEvent) -> None:
    """
    Process a fill through the consensus detector.

    Converts FillEvent to Fill dataclass and checks for consensus.
    If consensus is detected, publishes signal to NATS and persists to DB.

    Args:
        data: FillEvent from hl-stream
    """
    try:
        # Convert FillEvent to consensus Fill
        fill = Fill(
            fill_id=data.fill_id,
            address=data.address,
            asset=data.asset,
            side=data.side,
            size=float(data.size or 0),
            price=float(data.price) if hasattr(data, 'price') and data.price else 0.0,
            ts=data.ts if isinstance(data.ts, datetime) else datetime.fromisoformat(str(data.ts).replace('Z', '+00:00')),
        )

        # Update current price in detector
        if fill.price > 0:
            consensus_detector.set_current_price(fill.asset, fill.price)

        # Process fill and check for consensus
        signal = consensus_detector.process_fill(fill)

        if signal:
            await handle_consensus_signal(signal)

    except Exception as e:
        print(f"[hl-decide] Consensus processing error: {e}")


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


@app.on_event("startup")
async def startup():
    try:
        # Connect to database first
        app.state.db = await asyncpg.create_pool(DB_URL)

        # Restore state from database
        score_count, fill_count = await restore_state()
        print(f"[hl-decide] Restored {score_count} scores and {fill_count} fills from database")

        # Count open positions
        async with app.state.db.acquire() as conn:
            open_count = await conn.fetchval(
                "SELECT COUNT(*) FROM position_signals WHERE status = 'open'"
            )
            print(f"[hl-decide] {open_count} open positions being tracked")

        # Connect to NATS
        app.state.nc = await nats.connect(NATS_URL)
        app.state.js = app.state.nc.jetstream()
        await ensure_stream(app.state.js, "HL_D", ["d.signals.v1", "d.outcomes.v1"])
        await app.state.nc.subscribe("b.scores.v1", cb=handle_score)
        await app.state.nc.subscribe("c.fills.v1", cb=handle_fill)

        print("[hl-decide] Started with position-based tracking")
    except Exception as e:
        print(f"[hl-decide] Fatal startup error: {e}")
        raise


@app.on_event("shutdown")
async def shutdown():
    if hasattr(app.state, "nc"):
        await app.state.nc.drain()
    if hasattr(app.state, "db"):
        await app.state.db.close()


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


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)


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
