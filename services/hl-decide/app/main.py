import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict
from uuid import uuid4
from collections import OrderedDict

import asyncpg
import nats
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
from starlette.responses import Response

from contracts.py.models import FillEvent, ScoreEvent, SignalEvent, OutcomeEvent

SERVICE_NAME = "hl-decide"
NATS_URL = os.getenv("NATS_URL", "nats://localhost:4222")
DB_URL = os.getenv("DATABASE_URL", "postgresql://hlbot:hlbotpassword@localhost:5432/hlbot")
MAX_SCORES = int(os.getenv("MAX_SCORES", "500"))
MAX_FILLS = int(os.getenv("MAX_FILLS", "500"))

app = FastAPI(title="hl-decide", version="0.1.0")
scores: OrderedDict[str, ScoreEvent] = OrderedDict()
fills: OrderedDict[str, FillEvent] = OrderedDict()
pending_outcomes: Dict[str, asyncio.Task] = {}  # Track pending outcome tasks by ticket_id
registry = CollectorRegistry()
signal_counter = Counter("decide_signals_total", "Signals emitted", registry=registry)
outcome_counter = Counter("decide_outcomes_total", "Outcomes emitted", registry=registry)
decision_latency = Histogram(
    "decide_latency_seconds",
    "Latency between receiving score and emitting signal",
    registry=registry,
    buckets=(0.01, 0.05, 0.1, 0.5),
)


async def ensure_stream(js, name: str, subjects):
    try:
        await js.stream_info(name)
    except Exception:
        await js.add_stream(name=name, subjects=subjects)


def pick_side(score: ScoreEvent) -> str:
    return "long" if score.score >= 0 else "short"


async def persist_ticket(conn, ticket_id: str, signal: SignalEvent):
    await conn.execute(
        """
        INSERT INTO tickets (id, ts, address, asset, side, payload)
        VALUES ($1, $2, $3, $4, $5, $6)
        """,
        ticket_id,
        signal.signal_ts,
        signal.address,
        signal.asset,
        signal.side,
        json.dumps(signal.model_dump()),
    )


async def persist_outcome(conn, outcome: OutcomeEvent):
    await conn.execute(
        """
        INSERT INTO ticket_outcomes (ticket_id, closed_ts, result_r, closed_reason, notes)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (ticket_id) DO UPDATE SET
          closed_ts = EXCLUDED.closed_ts,
          result_r = EXCLUDED.result_r,
          closed_reason = EXCLUDED.closed_reason,
          notes = EXCLUDED.notes
        """,
        outcome.ticket_id,
        outcome.closed_ts,
        outcome.result_r,
        outcome.closed_reason,
        outcome.notes,
    )


async def calculate_pnl(signal: SignalEvent, entry_price: float, exit_price: float) -> float:
    """
    Calculate P&L as a fraction (R-multiple).
    For long: (exit - entry) / entry
    For short: (entry - exit) / entry
    """
    if entry_price <= 0 or exit_price <= 0:
        return 0.0

    if signal.side == "long":
        return (exit_price - entry_price) / entry_price
    else:  # short
        return (entry_price - exit_price) / entry_price


async def get_current_price(asset: str) -> float:
    """
    Fetch current price from the most recent fill for this asset.
    In a production system, this would query a price feed or market data API.
    """
    try:
        async with app.state.db.acquire() as conn:
            result = await conn.fetchrow(
                """
                SELECT payload->>'priceUsd' as price
                FROM hl_events
                WHERE type = 'trade' AND symbol = $1
                ORDER BY at DESC
                LIMIT 1
                """,
                asset
            )
            if result and result['price']:
                return float(result['price'])
    except Exception:
        pass
    return 0.0


async def emit_signal(address: str):
    score = scores.get(address)
    fill = fills.get(address)
    if not score or not fill:
        return
    with decision_latency.time():
        signal_ts = datetime.utcnow()
        ticket_id = str(uuid4())

        # Store entry price in payload for later P&L calculation
        entry_price = fill.price if hasattr(fill, 'price') and fill.price else 0.0

        signal = SignalEvent(
            ticket_id=ticket_id,
            address=address,
            asset=fill.asset,
            side=pick_side(score),
            confidence=min(max(abs(score.score), 0.1), 1.0),
            score_ts=score.ts,
            signal_ts=signal_ts,
            expires_at=signal_ts + timedelta(seconds=10),
            reason="consensus",
            payload={"fill_id": fill.fill_id, "weight": score.weight, "entry_price": entry_price},
        )
        await app.state.js.publish("d.signals.v1", signal.model_dump_json().encode("utf-8"))
        async with app.state.db.acquire() as conn:
            await persist_ticket(conn, ticket_id, signal)
        signal_counter.inc()

        # Track the outcome task to prevent duplicates and ensure cleanup
        if ticket_id not in pending_outcomes:
            task = asyncio.create_task(schedule_close(ticket_id, signal))
            pending_outcomes[ticket_id] = task


async def schedule_close(ticket_id: str, signal: SignalEvent):
    try:
        await asyncio.sleep(10)

        # Get entry price from signal payload
        entry_price = signal.payload.get('entry_price', 0.0) if isinstance(signal.payload, dict) else 0.0

        # Fetch current price
        exit_price = await get_current_price(signal.asset)

        # Calculate actual P&L
        result_r = await calculate_pnl(signal, entry_price, exit_price)

        outcome = OutcomeEvent(
            ticket_id=ticket_id,
            closed_ts=datetime.utcnow(),
            result_r=result_r,
            closed_reason="timebox",
            notes=f"Timeboxed exit: entry={entry_price:.2f}, exit={exit_price:.2f}, pnl_r={result_r:.4f}",
        )
        await app.state.js.publish("d.outcomes.v1", outcome.model_dump_json().encode("utf-8"))
        async with app.state.db.acquire() as conn:
            await persist_outcome(conn, outcome)
        outcome_counter.inc()
    finally:
        # Clean up the task from pending_outcomes to prevent memory leaks
        pending_outcomes.pop(ticket_id, None)


def enforce_limits():
    """Enforce memory limits on scores and fills using LRU eviction."""
    while len(scores) > MAX_SCORES:
        scores.popitem(last=False)  # Remove oldest
    while len(fills) > MAX_FILLS:
        fills.popitem(last=False)  # Remove oldest


async def handle_score(msg):
    data = ScoreEvent.model_validate_json(msg.data.decode())
    # Move to end (most recently used)
    if data.address in scores:
        scores.move_to_end(data.address)
    scores[data.address] = data
    enforce_limits()
    await emit_signal(data.address)


async def handle_fill(msg):
    data = FillEvent.model_validate_json(msg.data.decode())
    # Move to end (most recently used)
    if data.address in fills:
        fills.move_to_end(data.address)
    fills[data.address] = data
    enforce_limits()
    await emit_signal(data.address)


@app.on_event("startup")
async def startup():
    try:
        app.state.db = await asyncpg.create_pool(DB_URL)
        app.state.nc = await nats.connect(NATS_URL)
        app.state.js = app.state.nc.jetstream()
        await ensure_stream(app.state.js, "HL_D", ["d.signals.v1", "d.outcomes.v1"])
        await app.state.nc.subscribe("b.scores.v1", cb=handle_score)
        await app.state.nc.subscribe("c.fills.v1", cb=handle_fill)
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
    return {"status": "ok", "scores": len(scores), "fills": len(fills)}


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
