"""
Portfolio Management for SigmaPilot

Fetches account state and positions from Hyperliquid.
Phase 3e: Hyperliquid only. Multi-exchange support in Phase 4.

@module portfolio
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

import httpx
import asyncpg


# Hyperliquid API endpoint
HL_INFO_API = os.getenv("HL_INFO_API", "https://api.hyperliquid.xyz/info")


@dataclass
class Position:
    """A live position on an exchange."""
    exchange: str
    symbol: str
    side: str  # "long" or "short"
    size: float
    entry_price: float
    mark_price: float
    liquidation_price: Optional[float]
    unrealized_pnl: float
    margin_used: float
    leverage: int
    opened_at: Optional[datetime] = None


@dataclass
class PortfolioSummary:
    """Account summary for an exchange."""
    exchange: str
    account_value: float
    total_margin_used: float
    available_margin: float
    total_unrealized_pnl: float
    total_exposure_pct: float
    position_count: int
    positions: list[Position]


async def fetch_hyperliquid_state(address: str) -> Optional[dict]:
    """
    Fetch user state from Hyperliquid API.

    Args:
        address: The wallet address to fetch state for

    Returns:
        Raw API response or None if failed
    """
    if not address:
        return None

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            payload = {
                "type": "clearinghouseState",
                "user": address,
            }
            resp = await client.post(HL_INFO_API, json=payload)
            if resp.status_code == 200:
                return resp.json()
            else:
                print(f"[portfolio] HL API error: {resp.status_code}")
                return None
    except Exception as e:
        print(f"[portfolio] Failed to fetch HL state: {e}")
        return None


def parse_hyperliquid_state(data: dict) -> PortfolioSummary:
    """
    Parse Hyperliquid clearinghouse state into PortfolioSummary.

    Args:
        data: Raw API response from Hyperliquid

    Returns:
        Parsed PortfolioSummary
    """
    margin_summary = data.get("marginSummary", {})
    asset_positions = data.get("assetPositions", [])

    account_value = float(margin_summary.get("accountValue", 0))
    total_margin_used = float(margin_summary.get("totalMarginUsed", 0))
    total_unrealized_pnl = float(margin_summary.get("totalNtlPos", 0))  # Approximation

    # Calculate available margin
    available_margin = account_value - total_margin_used

    # Parse positions
    positions: list[Position] = []
    total_notional = 0.0

    for ap in asset_positions:
        pos = ap.get("position", {})
        coin = pos.get("coin", "")
        szi = float(pos.get("szi", 0))

        if abs(szi) < 1e-8:
            continue  # Skip zero positions

        entry_px = float(pos.get("entryPx", 0))
        mark_px = float(pos.get("positionValue", 0)) / abs(szi) if abs(szi) > 0 else entry_px
        liq_px = pos.get("liquidationPx")
        unrealized = float(pos.get("unrealizedPnl", 0))
        margin = float(pos.get("marginUsed", 0))
        leverage = int(pos.get("leverage", {}).get("value", 1)) if isinstance(pos.get("leverage"), dict) else 1

        side = "long" if szi > 0 else "short"
        notional = abs(szi) * entry_px
        total_notional += notional

        positions.append(Position(
            exchange="hyperliquid",
            symbol=coin,
            side=side,
            size=abs(szi),
            entry_price=entry_px,
            mark_price=mark_px,
            liquidation_price=float(liq_px) if liq_px else None,
            unrealized_pnl=unrealized,
            margin_used=margin,
            leverage=leverage,
        ))

    # Calculate exposure percentage
    exposure_pct = total_notional / account_value if account_value > 0 else 0

    # Recalculate total unrealized from positions
    total_unrealized_pnl = sum(p.unrealized_pnl for p in positions)

    return PortfolioSummary(
        exchange="hyperliquid",
        account_value=account_value,
        total_margin_used=total_margin_used,
        available_margin=available_margin,
        total_unrealized_pnl=total_unrealized_pnl,
        total_exposure_pct=exposure_pct,
        position_count=len(positions),
        positions=positions,
    )


async def get_portfolio_summary(db: asyncpg.Pool, address: Optional[str] = None) -> dict[str, Any]:
    """
    Get portfolio summary including account value and positions.

    Args:
        db: Database connection pool
        address: Optional Hyperliquid address (falls back to config)

    Returns:
        Portfolio summary dict
    """
    # Get address from config if not provided
    if not address:
        try:
            async with db.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT hl_address FROM execution_config WHERE id = 1"
                )
                if row and row["hl_address"]:
                    address = row["hl_address"]
        except Exception as e:
            print(f"[portfolio] Failed to get config: {e}")

    # If no address configured, return empty summary
    if not address:
        return {
            "configured": False,
            "message": "No Hyperliquid address configured",
            "exchanges": [],
            "total_equity": 0,
            "total_unrealized_pnl": 0,
            "total_positions": 0,
            "positions": [],
        }

    # Fetch from Hyperliquid
    hl_state = await fetch_hyperliquid_state(address)
    if not hl_state:
        return {
            "configured": True,
            "address": address,
            "error": "Failed to fetch Hyperliquid state",
            "exchanges": [],
            "total_equity": 0,
            "total_unrealized_pnl": 0,
            "total_positions": 0,
            "positions": [],
        }

    summary = parse_hyperliquid_state(hl_state)

    # Save snapshot to DB
    try:
        async with db.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO portfolio_snapshots
                (exchange, account_value, total_margin_used, available_margin,
                 total_unrealized_pnl, total_exposure_pct, position_count, raw_data)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                summary.exchange,
                summary.account_value,
                summary.total_margin_used,
                summary.available_margin,
                summary.total_unrealized_pnl,
                summary.total_exposure_pct,
                summary.position_count,
                json.dumps(hl_state),
            )

            # Update live positions
            for pos in summary.positions:
                await conn.execute(
                    """
                    INSERT INTO live_positions
                    (exchange, symbol, side, size, entry_price, mark_price,
                     liquidation_price, unrealized_pnl, margin_used, leverage, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NOW())
                    ON CONFLICT (exchange, symbol) DO UPDATE SET
                        side = EXCLUDED.side,
                        size = EXCLUDED.size,
                        entry_price = EXCLUDED.entry_price,
                        mark_price = EXCLUDED.mark_price,
                        liquidation_price = EXCLUDED.liquidation_price,
                        unrealized_pnl = EXCLUDED.unrealized_pnl,
                        margin_used = EXCLUDED.margin_used,
                        leverage = EXCLUDED.leverage,
                        updated_at = NOW()
                    """,
                    pos.exchange,
                    pos.symbol,
                    pos.side,
                    pos.size,
                    pos.entry_price,
                    pos.mark_price,
                    pos.liquidation_price,
                    pos.unrealized_pnl,
                    pos.margin_used,
                    pos.leverage,
                )

            # Remove positions that no longer exist
            current_symbols = [pos.symbol for pos in summary.positions]
            if current_symbols:
                await conn.execute(
                    """
                    DELETE FROM live_positions
                    WHERE exchange = $1 AND symbol != ALL($2)
                    """,
                    "hyperliquid",
                    current_symbols,
                )
            else:
                await conn.execute(
                    "DELETE FROM live_positions WHERE exchange = $1",
                    "hyperliquid",
                )
    except Exception as e:
        print(f"[portfolio] Failed to save snapshot: {e}")

    return {
        "configured": True,
        "address": address,
        "exchanges": [
            {
                "name": summary.exchange,
                "account_value": summary.account_value,
                "total_margin_used": summary.total_margin_used,
                "available_margin": summary.available_margin,
                "total_unrealized_pnl": summary.total_unrealized_pnl,
                "total_exposure_pct": round(summary.total_exposure_pct * 100, 2),
                "position_count": summary.position_count,
            }
        ],
        "total_equity": summary.account_value,
        "total_unrealized_pnl": summary.total_unrealized_pnl,
        "total_positions": summary.position_count,
        "positions": [
            {
                "exchange": pos.exchange,
                "symbol": pos.symbol,
                "side": pos.side,
                "size": pos.size,
                "entry_price": pos.entry_price,
                "mark_price": pos.mark_price,
                "liquidation_price": pos.liquidation_price,
                "unrealized_pnl": pos.unrealized_pnl,
                "margin_used": pos.margin_used,
                "leverage": pos.leverage,
            }
            for pos in summary.positions
        ],
    }


async def get_execution_config(db: asyncpg.Pool) -> dict[str, Any]:
    """
    Get current execution configuration.

    Args:
        db: Database connection pool

    Returns:
        Execution config dict
    """
    try:
        async with db.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT enabled, hl_enabled, hl_address, hl_max_leverage,
                       hl_max_position_pct, hl_max_exposure_pct,
                       max_daily_loss_pct, cooldown_after_loss_min, updated_at
                FROM execution_config
                WHERE id = 1
                """
            )

            if not row:
                return {"configured": False}

            return {
                "configured": True,
                "enabled": row["enabled"],
                "hyperliquid": {
                    "enabled": row["hl_enabled"],
                    "address": row["hl_address"],
                    "max_leverage": row["hl_max_leverage"],
                    "max_position_pct": float(row["hl_max_position_pct"] or 0) * 100,
                    "max_exposure_pct": float(row["hl_max_exposure_pct"] or 0) * 100,
                },
                "risk": {
                    "max_daily_loss_pct": float(row["max_daily_loss_pct"] or 0) * 100,
                    "cooldown_after_loss_min": row["cooldown_after_loss_min"],
                },
                "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
            }
    except Exception as e:
        print(f"[portfolio] Failed to get execution config: {e}")
        return {"configured": False, "error": str(e)}


async def update_execution_config(
    db: asyncpg.Pool,
    enabled: Optional[bool] = None,
    hl_enabled: Optional[bool] = None,
    hl_address: Optional[str] = None,
    hl_max_leverage: Optional[int] = None,
    hl_max_position_pct: Optional[float] = None,
    hl_max_exposure_pct: Optional[float] = None,
) -> dict[str, Any]:
    """
    Update execution configuration.

    Args:
        db: Database connection pool
        Various config parameters

    Returns:
        Updated config dict
    """
    updates = []
    params = []
    param_idx = 1

    if enabled is not None:
        updates.append(f"enabled = ${param_idx}")
        params.append(enabled)
        param_idx += 1

    if hl_enabled is not None:
        updates.append(f"hl_enabled = ${param_idx}")
        params.append(hl_enabled)
        param_idx += 1

    if hl_address is not None:
        updates.append(f"hl_address = ${param_idx}")
        params.append(hl_address)
        param_idx += 1

    if hl_max_leverage is not None:
        updates.append(f"hl_max_leverage = ${param_idx}")
        params.append(hl_max_leverage)
        param_idx += 1

    if hl_max_position_pct is not None:
        updates.append(f"hl_max_position_pct = ${param_idx}")
        params.append(hl_max_position_pct / 100)  # Convert from % to decimal
        param_idx += 1

    if hl_max_exposure_pct is not None:
        updates.append(f"hl_max_exposure_pct = ${param_idx}")
        params.append(hl_max_exposure_pct / 100)  # Convert from % to decimal
        param_idx += 1

    if not updates:
        return await get_execution_config(db)

    updates.append("updated_at = NOW()")

    try:
        async with db.acquire() as conn:
            await conn.execute(
                f"UPDATE execution_config SET {', '.join(updates)} WHERE id = 1",
                *params,
            )
        return await get_execution_config(db)
    except Exception as e:
        print(f"[portfolio] Failed to update config: {e}")
        return {"error": str(e)}


async def get_execution_logs(
    db: asyncpg.Pool,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """
    Get recent execution logs.

    Args:
        db: Database connection pool
        limit: Max results
        offset: Pagination offset

    Returns:
        Execution logs with pagination
    """
    try:
        async with db.acquire() as conn:
            total = await conn.fetchval("SELECT COUNT(*) FROM execution_logs")

            rows = await conn.fetch(
                """
                SELECT id, decision_id, created_at, exchange, symbol, side,
                       size, leverage, status, fill_price, fill_size,
                       error_message, account_value, position_pct,
                       exposure_before, exposure_after
                FROM execution_logs
                ORDER BY created_at DESC
                LIMIT $1 OFFSET $2
                """,
                limit,
                offset,
            )

            return {
                "total": total or 0,
                "limit": limit,
                "offset": offset,
                "items": [
                    {
                        "id": str(row["id"]),
                        "decision_id": str(row["decision_id"]) if row["decision_id"] else None,
                        "created_at": row["created_at"].isoformat(),
                        "exchange": row["exchange"],
                        "symbol": row["symbol"],
                        "side": row["side"],
                        "size": float(row["size"]),
                        "leverage": row["leverage"],
                        "status": row["status"],
                        "fill_price": float(row["fill_price"]) if row["fill_price"] else None,
                        "fill_size": float(row["fill_size"]) if row["fill_size"] else None,
                        "error_message": row["error_message"],
                        "account_value": float(row["account_value"]) if row["account_value"] else None,
                        "position_pct": float(row["position_pct"]) if row["position_pct"] else None,
                        "exposure_before": float(row["exposure_before"]) if row["exposure_before"] else None,
                        "exposure_after": float(row["exposure_after"]) if row["exposure_after"] else None,
                    }
                    for row in rows
                ],
            }
    except Exception as e:
        print(f"[portfolio] Failed to get execution logs: {e}")
        return {"total": 0, "limit": limit, "offset": offset, "items": [], "error": str(e)}
