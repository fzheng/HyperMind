"""
Decision Logging for Signal Auditability

Records every consensus evaluation with human-readable reasoning so we can:
- Understand why signals fired or were skipped
- Measure signal quality over time
- Debug gate failures
- Build trust through transparency

@module decision_logger
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional
from uuid import uuid4

import asyncpg


@dataclass
class GateResult:
    """Result of a single gate check."""
    name: str
    passed: bool
    value: float
    threshold: float
    detail: str = ""


@dataclass
class DecisionLog:
    """A logged decision from the consensus evaluation."""
    id: str
    created_at: datetime
    symbol: str
    direction: str
    decision_type: str  # "signal", "skip", "risk_reject"
    trader_count: int
    agreement_pct: float
    effective_k: float
    gates: list[GateResult]
    reasoning: str
    avg_confidence: Optional[float] = None
    ev_estimate: Optional[float] = None
    price_at_decision: Optional[float] = None
    risk_checks: Optional[list[dict]] = None
    # Outcome tracking (updated when position closes)
    outcome_pnl: Optional[float] = None
    outcome_r_multiple: Optional[float] = None
    outcome_closed_at: Optional[datetime] = None


def generate_reasoning(
    decision_type: str,
    symbol: str,
    direction: str,
    trader_count: int,
    agreement_pct: float,
    effective_k: float,
    gates: list[GateResult],
    risk_checks: Optional[list[dict]] = None,
) -> str:
    """
    Generate human-readable explanation of the decision.

    This creates a 2-4 sentence summary that explains:
    - What traders did
    - Whether gates passed and why/why not
    - The key metrics that drove the decision

    Args:
        decision_type: "signal", "skip", or "risk_reject"
        symbol: Asset symbol (BTC, ETH)
        direction: Trade direction (long, short)
        trader_count: Number of traders in the consensus window
        agreement_pct: Percentage agreeing on direction (0-1)
        effective_k: Correlation-adjusted effective trader count
        gates: List of gate results
        risk_checks: Optional list of risk check results

    Returns:
        Human-readable reasoning string
    """
    if decision_type == "signal":
        return (
            f"{trader_count} Alpha Pool traders opened {direction.upper()} {symbol}. "
            f"{agreement_pct:.0%} agreement with effK={effective_k:.1f}. "
            f"All consensus gates passed."
        )

    elif decision_type == "skip":
        failed = [g for g in gates if not g.passed]
        reasons = []
        for g in failed:
            if g.name == "supermajority":
                reasons.append(f"only {g.value:.0%} agreement (need {g.threshold:.0%})")
            elif g.name == "min_traders":
                reasons.append(f"only {int(g.value)} traders (need {int(g.threshold)})")
            elif g.name == "effective_k":
                reasons.append(f"effK={g.value:.1f} too low (need {g.threshold:.1f})")
            elif g.name == "freshness":
                reasons.append(f"signal {g.value:.0f}s stale (max {g.threshold:.0f}s)")
            elif g.name == "price_band":
                reasons.append(f"price drifted {g.value:.2f}R (max {g.threshold:.2f}R)")
            elif g.name == "ev_gate":
                reasons.append(f"EV={g.value:.2f}R below threshold ({g.threshold:.2f}R)")
            elif g.name == "atr_validity":
                reasons.append(f"ATR data invalid: {g.detail}")
            else:
                reasons.append(f"{g.name} failed ({g.value:.2f} vs {g.threshold:.2f})")

        if reasons:
            return (
                f"Skipped: {trader_count} traders detected but gates failed. "
                f"{'; '.join(reasons)}."
            )
        else:
            return f"Skipped: {trader_count} traders but no clear consensus."

    elif decision_type == "risk_reject":
        risk_reasons = []
        if risk_checks:
            for check in risk_checks:
                if not check.get("passed", True):
                    risk_reasons.append(check.get("reason", "unknown risk check failed"))

        reason_str = "; ".join(risk_reasons) if risk_reasons else "risk limits exceeded"
        return (
            f"Consensus detected but rejected by risk limits. "
            f"{trader_count} traders, {agreement_pct:.0%} agreement. "
            f"Reason: {reason_str}."
        )

    return "Decision recorded."


async def log_decision(
    db: asyncpg.Pool,
    symbol: str,
    direction: str,
    decision_type: str,
    trader_count: int,
    agreement_pct: float,
    effective_k: float,
    gates: list[GateResult],
    risk_checks: Optional[list[dict]] = None,
    price: Optional[float] = None,
    confidence: Optional[float] = None,
    ev: Optional[float] = None,
) -> str:
    """
    Log a decision and return its ID.

    Args:
        db: Database connection pool
        symbol: Asset symbol (BTC, ETH)
        direction: Trade direction (long, short, none)
        decision_type: "signal", "skip", or "risk_reject"
        trader_count: Number of traders in consensus window
        agreement_pct: Agreement percentage (0-1)
        effective_k: Correlation-adjusted effective K
        gates: List of gate check results
        risk_checks: Optional list of risk check results
        price: Current price at decision time
        confidence: Win probability estimate
        ev: Expected value estimate

    Returns:
        The decision log ID
    """
    decision_id = str(uuid4())
    now = datetime.now(timezone.utc)

    reasoning = generate_reasoning(
        decision_type, symbol, direction,
        trader_count, agreement_pct, effective_k,
        gates, risk_checks,
    )

    # Convert gates to JSON-serializable format
    gates_json = [
        {
            "name": g.name,
            "passed": g.passed,
            "value": g.value,
            "threshold": g.threshold,
            "detail": g.detail,
        }
        for g in gates
    ]

    try:
        async with db.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO decision_logs (
                    id, created_at, symbol, direction, decision_type,
                    trader_count, agreement_pct, effective_k,
                    avg_confidence, ev_estimate, price_at_decision,
                    gates, risk_checks, reasoning
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                """,
                decision_id,
                now,
                symbol,
                direction,
                decision_type,
                trader_count,
                agreement_pct,
                effective_k,
                confidence,
                ev,
                price,
                json.dumps(gates_json),
                json.dumps(risk_checks) if risk_checks else None,
                reasoning,
            )
    except Exception as e:
        print(f"[decision_logger] Failed to log decision: {e}")
        # Don't raise - logging should not break the signal flow

    return decision_id


async def update_decision_outcome(
    db: asyncpg.Pool,
    decision_id: str,
    pnl: float,
    r_multiple: float,
) -> None:
    """
    Update a decision with its outcome when the position closes.

    Args:
        db: Database connection pool
        decision_id: The decision log ID
        pnl: Realized P&L
        r_multiple: Result in R-multiples
    """
    try:
        async with db.acquire() as conn:
            await conn.execute(
                """
                UPDATE decision_logs
                SET outcome_pnl = $2,
                    outcome_r_multiple = $3,
                    outcome_closed_at = NOW()
                WHERE id = $1
                """,
                decision_id,
                pnl,
                r_multiple,
            )
    except Exception as e:
        print(f"[decision_logger] Failed to update outcome: {e}")


async def get_decisions(
    db: asyncpg.Pool,
    symbol: Optional[str] = None,
    decision_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """
    List decision logs with filters.

    Args:
        db: Database connection pool
        symbol: Filter by symbol (BTC, ETH)
        decision_type: Filter by type (signal, skip, risk_reject)
        limit: Maximum results to return
        offset: Pagination offset

    Returns:
        Dict with items and total count
    """
    conditions = []
    params = []
    param_idx = 1

    if symbol:
        conditions.append(f"symbol = ${param_idx}")
        params.append(symbol.upper())
        param_idx += 1

    if decision_type:
        conditions.append(f"decision_type = ${param_idx}")
        params.append(decision_type)
        param_idx += 1

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    params.extend([limit, offset])

    try:
        async with db.acquire() as conn:
            # Get total count
            count_query = f"SELECT COUNT(*) FROM decision_logs {where_clause}"
            total = await conn.fetchval(count_query, *params[:-2]) if params[:-2] else await conn.fetchval(count_query)

            # Get items
            query = f"""
                SELECT id, created_at, symbol, direction, decision_type,
                       trader_count, agreement_pct, effective_k,
                       avg_confidence, ev_estimate, price_at_decision,
                       gates, risk_checks, reasoning,
                       outcome_pnl, outcome_r_multiple, outcome_closed_at
                FROM decision_logs
                {where_clause}
                ORDER BY created_at DESC
                LIMIT ${param_idx} OFFSET ${param_idx + 1}
            """
            rows = await conn.fetch(query, *params)

            return {
                "total": total or 0,
                "limit": limit,
                "offset": offset,
                "items": [
                    {
                        "id": str(row["id"]),
                        "created_at": row["created_at"].isoformat(),
                        "symbol": row["symbol"],
                        "direction": row["direction"],
                        "decision_type": row["decision_type"],
                        "trader_count": row["trader_count"],
                        "agreement_pct": float(row["agreement_pct"]),
                        "effective_k": float(row["effective_k"]),
                        "avg_confidence": float(row["avg_confidence"]) if row["avg_confidence"] else None,
                        "ev_estimate": float(row["ev_estimate"]) if row["ev_estimate"] else None,
                        "price_at_decision": float(row["price_at_decision"]) if row["price_at_decision"] else None,
                        "gates": json.loads(row["gates"]) if row["gates"] else [],
                        "risk_checks": json.loads(row["risk_checks"]) if row["risk_checks"] else None,
                        "reasoning": row["reasoning"],
                        "outcome_pnl": float(row["outcome_pnl"]) if row["outcome_pnl"] else None,
                        "outcome_r_multiple": float(row["outcome_r_multiple"]) if row["outcome_r_multiple"] else None,
                        "outcome_closed_at": row["outcome_closed_at"].isoformat() if row["outcome_closed_at"] else None,
                    }
                    for row in rows
                ],
            }
    except Exception as e:
        print(f"[decision_logger] Failed to get decisions: {e}")
        return {"total": 0, "limit": limit, "offset": offset, "items": [], "error": str(e)}


async def get_decision(db: asyncpg.Pool, decision_id: str) -> Optional[dict[str, Any]]:
    """
    Get full details for a single decision.

    Args:
        db: Database connection pool
        decision_id: The decision log ID

    Returns:
        Decision details or None if not found
    """
    try:
        async with db.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, created_at, symbol, direction, decision_type,
                       trader_count, agreement_pct, effective_k,
                       avg_confidence, ev_estimate, price_at_decision,
                       gates, risk_checks, reasoning,
                       outcome_pnl, outcome_r_multiple, outcome_closed_at
                FROM decision_logs
                WHERE id = $1
                """,
                decision_id,
            )

            if not row:
                return None

            return {
                "id": str(row["id"]),
                "created_at": row["created_at"].isoformat(),
                "symbol": row["symbol"],
                "direction": row["direction"],
                "decision_type": row["decision_type"],
                "trader_count": row["trader_count"],
                "agreement_pct": float(row["agreement_pct"]),
                "effective_k": float(row["effective_k"]),
                "avg_confidence": float(row["avg_confidence"]) if row["avg_confidence"] else None,
                "ev_estimate": float(row["ev_estimate"]) if row["ev_estimate"] else None,
                "price_at_decision": float(row["price_at_decision"]) if row["price_at_decision"] else None,
                "gates": json.loads(row["gates"]) if row["gates"] else [],
                "risk_checks": json.loads(row["risk_checks"]) if row["risk_checks"] else None,
                "reasoning": row["reasoning"],
                "outcome_pnl": float(row["outcome_pnl"]) if row["outcome_pnl"] else None,
                "outcome_r_multiple": float(row["outcome_r_multiple"]) if row["outcome_r_multiple"] else None,
                "outcome_closed_at": row["outcome_closed_at"].isoformat() if row["outcome_closed_at"] else None,
            }
    except Exception as e:
        print(f"[decision_logger] Failed to get decision: {e}")
        return None


async def get_decision_stats(db: asyncpg.Pool, days: int = 7) -> dict[str, Any]:
    """
    Get aggregate statistics for decisions.

    Args:
        db: Database connection pool
        days: Number of days to look back

    Returns:
        Dict with aggregate statistics
    """
    try:
        async with db.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) as total_decisions,
                    COUNT(*) FILTER (WHERE decision_type = 'signal') as signals,
                    COUNT(*) FILTER (WHERE decision_type = 'skip') as skipped,
                    COUNT(*) FILTER (WHERE decision_type = 'risk_reject') as risk_rejected,
                    COUNT(*) FILTER (WHERE outcome_r_multiple IS NOT NULL) as closed,
                    COUNT(*) FILTER (WHERE outcome_r_multiple > 0) as wins,
                    COUNT(*) FILTER (WHERE outcome_r_multiple <= 0) as losses,
                    AVG(effective_k) FILTER (WHERE decision_type = 'signal') as avg_eff_k,
                    AVG(ev_estimate) FILTER (WHERE decision_type = 'signal') as avg_ev,
                    AVG(outcome_r_multiple) FILTER (WHERE outcome_r_multiple IS NOT NULL) as avg_result_r,
                    SUM(outcome_r_multiple) FILTER (WHERE outcome_r_multiple IS NOT NULL) as total_r
                FROM decision_logs
                WHERE created_at > NOW() - INTERVAL '%s days'
                """ % days,
            )

            total_signals = row["signals"] or 0
            closed = row["closed"] or 0
            wins = row["wins"] or 0

            return {
                "period_days": days,
                "total_decisions": row["total_decisions"] or 0,
                "signals": total_signals,
                "skipped": row["skipped"] or 0,
                "risk_rejected": row["risk_rejected"] or 0,
                "skip_rate": round((row["skipped"] or 0) / max(row["total_decisions"] or 1, 1) * 100, 1),
                "closed": closed,
                "wins": wins,
                "losses": row["losses"] or 0,
                "win_rate": round(wins / max(closed, 1) * 100, 1),
                "avg_eff_k": round(float(row["avg_eff_k"] or 0), 2),
                "avg_ev": round(float(row["avg_ev"] or 0), 3),
                "avg_result_r": round(float(row["avg_result_r"] or 0), 3),
                "total_r": round(float(row["total_r"] or 0), 2),
            }
    except Exception as e:
        print(f"[decision_logger] Failed to get stats: {e}")
        return {
            "period_days": days,
            "total_decisions": 0,
            "signals": 0,
            "skipped": 0,
            "risk_rejected": 0,
            "skip_rate": 0,
            "closed": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "avg_eff_k": 0,
            "avg_ev": 0,
            "avg_result_r": 0,
            "total_r": 0,
            "error": str(e),
        }
