# AI Decision Logging System Design

*Phase 3e Technical Specification - December 2025*

---

## 1. Overview

This document specifies a comprehensive decision logging system for SigmaPilot that captures, stores, and surfaces AI trading decisions with human-readable reasoning. The system enables:

- **Full transparency**: Every decision is recorded with inputs and reasoning
- **Queryable history**: Filter by symbol, result, time, exchange
- **Chain-of-thought**: Human-readable explanations for each decision
- **Execution tracking**: Link decisions to actual trades and outcomes
- **No secret leakage**: Anonymized/aggregated trader data only

---

## 2. Decision Log Schema

### Core Data Model (`services/hl-decide/app/models.py`)

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict
from enum import Enum

class DecisionType(Enum):
    SIGNAL = "signal"      # Consensus signal generated
    SKIP = "skip"          # Signal skipped (gate failed)
    RISK_REJECT = "risk_reject"  # Signal rejected by risk limits

class ExecutionStatus(Enum):
    PENDING = "pending"
    EXECUTED = "executed"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class GateResult:
    name: str
    passed: bool
    value: float
    threshold: float
    explanation: str

@dataclass
class TraderVote:
    """Anonymized trader contribution (no addresses)"""
    rank: int              # Position in Alpha Pool (1-50)
    weight: float          # Vote weight (0-1)
    direction: str         # "long" or "short"
    confidence: float      # Posterior-derived confidence
    correlation_factor: float  # How correlated with others

@dataclass
class DecisionInputs:
    """Aggregated inputs (anonymized)"""
    trader_count: int
    agreement_pct: float
    effective_k: float
    avg_confidence: float
    ev_estimate: float
    price_at_decision: float
    atr_value: float
    atr_source: str        # "marks_1m", "fallback", "hardcoded"
    votes: List[TraderVote]

@dataclass
class ExecutionResult:
    exchange: str
    order_id: str
    fill_price: float
    fill_size: float
    fill_time_ms: int
    fees: float
    slippage_bps: float

@dataclass
class DecisionLog:
    """Complete decision record"""
    id: str                # UUID
    timestamp: datetime
    symbol: str
    direction: str         # "long", "short", or "none"
    decision_type: DecisionType

    # Inputs
    inputs: DecisionInputs

    # Gate results (all 5 gates)
    gates: List[GateResult]

    # Risk check results
    risk_checks: List[GateResult]

    # Human-readable reasoning
    reasoning_summary: str

    # Execution (if applicable)
    execution_status: ExecutionStatus
    executions: List[ExecutionResult] = field(default_factory=list)

    # Outcome (filled in later)
    outcome_pnl: Optional[float] = None
    outcome_r_multiple: Optional[float] = None
    outcome_closed_at: Optional[datetime] = None

    def to_dict(self) -> Dict:
        """Serialize for API/storage"""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "direction": self.direction,
            "decision_type": self.decision_type.value,
            "inputs": {
                "trader_count": self.inputs.trader_count,
                "agreement_pct": self.inputs.agreement_pct,
                "effective_k": self.inputs.effective_k,
                "avg_confidence": self.inputs.avg_confidence,
                "ev_estimate": self.inputs.ev_estimate,
                "price_at_decision": self.inputs.price_at_decision,
                "atr_value": self.inputs.atr_value,
                "atr_source": self.inputs.atr_source,
                "votes": [
                    {
                        "rank": v.rank,
                        "weight": v.weight,
                        "direction": v.direction,
                        "confidence": v.confidence,
                        "correlation_factor": v.correlation_factor,
                    }
                    for v in self.inputs.votes
                ],
            },
            "gates": [
                {
                    "name": g.name,
                    "passed": g.passed,
                    "value": g.value,
                    "threshold": g.threshold,
                    "explanation": g.explanation,
                }
                for g in self.gates
            ],
            "risk_checks": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "value": r.value,
                    "threshold": r.threshold,
                    "explanation": r.explanation,
                }
                for r in self.risk_checks
            ],
            "reasoning_summary": self.reasoning_summary,
            "execution_status": self.execution_status.value,
            "executions": [
                {
                    "exchange": e.exchange,
                    "order_id": e.order_id,
                    "fill_price": e.fill_price,
                    "fill_size": e.fill_size,
                    "fill_time_ms": e.fill_time_ms,
                    "fees": e.fees,
                    "slippage_bps": e.slippage_bps,
                }
                for e in self.executions
            ],
            "outcome_pnl": self.outcome_pnl,
            "outcome_r_multiple": self.outcome_r_multiple,
            "outcome_closed_at": self.outcome_closed_at.isoformat() if self.outcome_closed_at else None,
        }
```

---

## 3. Reasoning Generator

### Chain-of-Thought Builder (`services/hl-decide/app/reasoning.py`)

```python
from typing import List
from .models import DecisionLog, DecisionType, GateResult

def generate_reasoning_summary(decision: DecisionLog) -> str:
    """
    Generate a human-readable explanation for a decision.

    This summary should:
    - Be 2-4 sentences
    - Explain the key factors
    - Not reveal sensitive data (addresses, exact positions)
    - Be understandable by a non-technical trader
    """
    symbol = decision.symbol
    inputs = decision.inputs

    if decision.decision_type == DecisionType.SIGNAL:
        return _generate_signal_reasoning(decision)
    elif decision.decision_type == DecisionType.SKIP:
        return _generate_skip_reasoning(decision)
    elif decision.decision_type == DecisionType.RISK_REJECT:
        return _generate_risk_reject_reasoning(decision)

    return "Decision recorded."


def _generate_signal_reasoning(decision: DecisionLog) -> str:
    """Generate reasoning for an executed signal."""
    inputs = decision.inputs
    direction = decision.direction.upper()

    # Count agreeing traders
    agreeing = [v for v in inputs.votes if v.direction == decision.direction]
    total = len(inputs.votes)

    # Find strongest factor
    strongest_gate = max(decision.gates, key=lambda g: g.value / g.threshold if g.threshold > 0 else 0)

    parts = [
        f"{len(agreeing)} of {total} Alpha Pool traders opened {direction} {decision.symbol} "
        f"positions within the consensus window.",
    ]

    # Add effective-K insight
    if inputs.effective_k >= 3.0:
        parts.append(
            f"Effective-K of {inputs.effective_k:.1f} indicates low correlation "
            f"(traders are making independent decisions)."
        )
    else:
        parts.append(
            f"Effective-K of {inputs.effective_k:.1f} shows moderate independence "
            f"among agreeing traders."
        )

    # Add EV insight
    parts.append(
        f"Expected value of +{inputs.ev_estimate:.2f}R after estimated fees and slippage. "
        f"All consensus gates and risk limits passed."
    )

    return " ".join(parts)


def _generate_skip_reasoning(decision: DecisionLog) -> str:
    """Generate reasoning for a skipped signal."""
    inputs = decision.inputs

    # Find which gates failed
    failed_gates = [g for g in decision.gates if not g.passed]

    if not failed_gates:
        return "Signal skipped: no consensus detected."

    # Build explanation from failed gates
    explanations = []

    for gate in failed_gates:
        if gate.name == "supermajority":
            explanations.append(
                f"Only {inputs.agreement_pct:.0%} agreement "
                f"({gate.threshold:.0%} required)"
            )
        elif gate.name == "effective_k":
            explanations.append(
                f"Effective-K of {gate.value:.1f} indicates high correlation "
                f"among traders (minimum {gate.threshold:.1f} required)"
            )
        elif gate.name == "freshness":
            explanations.append(
                f"Signal too stale ({gate.value:.0f}s old, max {gate.threshold:.0f}s)"
            )
        elif gate.name == "price_band":
            explanations.append(
                f"Price drifted {gate.value:.2f}R from consensus "
                f"(max {gate.threshold:.2f}R allowed)"
            )
        elif gate.name == "ev_gate":
            explanations.append(
                f"Expected value {gate.value:.2f}R below threshold "
                f"({gate.threshold:.2f}R minimum)"
            )

    failed_str = "; ".join(explanations)

    return (
        f"Signal skipped: {len(inputs.votes)} traders detected, but consensus gates failed. "
        f"{failed_str}."
    )


def _generate_risk_reject_reasoning(decision: DecisionLog) -> str:
    """Generate reasoning for a risk-rejected signal."""
    inputs = decision.inputs

    # Find which risk checks failed
    failed_checks = [r for r in decision.risk_checks if not r.passed]

    if not failed_checks:
        return "Signal rejected by risk limits."

    explanations = []

    for check in failed_checks:
        if check.name == "min_confidence":
            explanations.append(
                f"Confidence {check.value:.0%} below minimum {check.threshold:.0%}"
            )
        elif check.name == "min_ev":
            explanations.append(
                f"EV {check.value:.2f}R below minimum {check.threshold:.2f}R"
            )
        elif check.name == "max_position":
            explanations.append(
                f"Position would exceed {check.threshold:.0%} limit"
            )
        elif check.name == "max_exposure":
            explanations.append(
                f"Total exposure would exceed {check.threshold:.0%} limit"
            )
        elif check.name == "cooldown":
            explanations.append(
                f"Cooldown active ({check.value:.0f}s since last signal)"
            )
        elif check.name == "daily_loss":
            explanations.append(
                f"Daily loss limit reached ({check.value:.0%} of {check.threshold:.0%} max)"
            )

    failed_str = "; ".join(explanations)

    return (
        f"Consensus detected with {len(inputs.votes)} traders agreeing, "
        f"but signal rejected by risk limits: {failed_str}."
    )
```

---

## 4. Database Schema

### Migration (`db/migrations/055_decision_logs.sql`)

```sql
-- Decision log table
CREATE TABLE IF NOT EXISTS decision_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol VARCHAR(16) NOT NULL,
    direction VARCHAR(8) NOT NULL,  -- 'long', 'short', 'none'
    decision_type VARCHAR(16) NOT NULL,  -- 'signal', 'skip', 'risk_reject'

    -- Inputs (JSONB for flexibility)
    inputs JSONB NOT NULL,

    -- Gate results
    gates JSONB NOT NULL,

    -- Risk check results
    risk_checks JSONB NOT NULL,

    -- Human-readable reasoning
    reasoning_summary TEXT NOT NULL,

    -- Execution status
    execution_status VARCHAR(16) NOT NULL DEFAULT 'pending',

    -- Execution details (array of results)
    executions JSONB DEFAULT '[]',

    -- Outcome (updated when position closes)
    outcome_pnl DECIMAL(20, 8),
    outcome_r_multiple DECIMAL(10, 4),
    outcome_closed_at TIMESTAMPTZ
);

-- Indexes for efficient querying
CREATE INDEX idx_decision_logs_timestamp ON decision_logs(timestamp DESC);
CREATE INDEX idx_decision_logs_symbol ON decision_logs(symbol);
CREATE INDEX idx_decision_logs_type ON decision_logs(decision_type);
CREATE INDEX idx_decision_logs_status ON decision_logs(execution_status);

-- Composite index for common filters
CREATE INDEX idx_decision_logs_symbol_time ON decision_logs(symbol, timestamp DESC);

-- GIN index for JSONB queries
CREATE INDEX idx_decision_logs_inputs ON decision_logs USING GIN (inputs);

-- View for dashboard quick stats
CREATE OR REPLACE VIEW decision_stats AS
SELECT
    DATE_TRUNC('day', timestamp) AS date,
    symbol,
    COUNT(*) FILTER (WHERE decision_type = 'signal') AS signals,
    COUNT(*) FILTER (WHERE decision_type = 'skip') AS skips,
    COUNT(*) FILTER (WHERE decision_type = 'risk_reject') AS risk_rejects,
    COUNT(*) FILTER (WHERE execution_status = 'executed') AS executed,
    AVG(outcome_r_multiple) FILTER (WHERE outcome_r_multiple IS NOT NULL) AS avg_r,
    SUM(outcome_pnl) FILTER (WHERE outcome_pnl IS NOT NULL) AS total_pnl
FROM decision_logs
GROUP BY DATE_TRUNC('day', timestamp), symbol
ORDER BY date DESC, symbol;
```

---

## 5. API Endpoints

### Decision Log API (`services/hl-decide/app/api/decisions.py`)

```python
from fastapi import APIRouter, Query
from typing import Optional, List
from datetime import datetime, timedelta

router = APIRouter(prefix="/decisions", tags=["decisions"])

@router.get("")
async def list_decisions(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    decision_type: Optional[str] = Query(None, description="signal, skip, or risk_reject"),
    execution_status: Optional[str] = Query(None, description="executed, skipped, etc."),
    exchange: Optional[str] = Query(None, description="Filter by execution exchange"),
    start_date: Optional[datetime] = Query(None, description="Start of date range"),
    end_date: Optional[datetime] = Query(None, description="End of date range"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """
    List decision logs with filters.

    Returns paginated list of decisions with reasoning summaries.
    Use /decisions/{id} for full details including votes and gates.
    """
    # Build query
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

    if execution_status:
        conditions.append(f"execution_status = ${param_idx}")
        params.append(execution_status)
        param_idx += 1

    if exchange:
        conditions.append(f"executions @> $${param_idx}::jsonb")
        params.append(f'[{{"exchange": "{exchange}"}}]')
        param_idx += 1

    if start_date:
        conditions.append(f"timestamp >= ${param_idx}")
        params.append(start_date)
        param_idx += 1

    if end_date:
        conditions.append(f"timestamp <= ${param_idx}")
        params.append(end_date)
        param_idx += 1

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Execute query
    query = f"""
        SELECT id, timestamp, symbol, direction, decision_type,
               (inputs->>'trader_count')::int AS trader_count,
               (inputs->>'agreement_pct')::float AS agreement_pct,
               (inputs->>'ev_estimate')::float AS ev_estimate,
               reasoning_summary, execution_status,
               outcome_pnl, outcome_r_multiple
        FROM decision_logs
        WHERE {where_clause}
        ORDER BY timestamp DESC
        LIMIT ${param_idx} OFFSET ${param_idx + 1}
    """
    params.extend([limit, offset])

    rows = await db.fetch(query, *params)

    # Get total count
    count_query = f"SELECT COUNT(*) FROM decision_logs WHERE {where_clause}"
    total = await db.fetchval(count_query, *params[:-2])

    return {
        "items": [dict(row) for row in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/{decision_id}")
async def get_decision(decision_id: str):
    """
    Get full decision details including votes and gate results.
    """
    row = await db.fetchrow(
        """
        SELECT * FROM decision_logs WHERE id = $1
        """,
        decision_id,
    )

    if not row:
        raise HTTPException(status_code=404, detail="Decision not found")

    return {
        "id": row["id"],
        "timestamp": row["timestamp"].isoformat(),
        "symbol": row["symbol"],
        "direction": row["direction"],
        "decision_type": row["decision_type"],
        "inputs": row["inputs"],
        "gates": row["gates"],
        "risk_checks": row["risk_checks"],
        "reasoning_summary": row["reasoning_summary"],
        "execution_status": row["execution_status"],
        "executions": row["executions"],
        "outcome_pnl": row["outcome_pnl"],
        "outcome_r_multiple": row["outcome_r_multiple"],
        "outcome_closed_at": row["outcome_closed_at"].isoformat() if row["outcome_closed_at"] else None,
    }


@router.get("/stats/summary")
async def get_decision_stats(
    days: int = Query(7, ge=1, le=90),
):
    """
    Get aggregated decision statistics.
    """
    since = datetime.utcnow() - timedelta(days=days)

    stats = await db.fetchrow(
        """
        SELECT
            COUNT(*) AS total_decisions,
            COUNT(*) FILTER (WHERE decision_type = 'signal') AS signals,
            COUNT(*) FILTER (WHERE decision_type = 'skip') AS skips,
            COUNT(*) FILTER (WHERE decision_type = 'risk_reject') AS risk_rejects,
            COUNT(*) FILTER (WHERE execution_status = 'executed') AS executed,
            AVG(outcome_r_multiple) FILTER (WHERE outcome_r_multiple IS NOT NULL) AS avg_r,
            SUM(outcome_pnl) FILTER (WHERE outcome_pnl IS NOT NULL) AS total_pnl,
            COUNT(*) FILTER (WHERE outcome_r_multiple > 0) AS wins,
            COUNT(*) FILTER (WHERE outcome_r_multiple < 0) AS losses
        FROM decision_logs
        WHERE timestamp >= $1
        """,
        since,
    )

    total_closed = (stats["wins"] or 0) + (stats["losses"] or 0)
    win_rate = stats["wins"] / total_closed if total_closed > 0 else None

    return {
        "period_days": days,
        "total_decisions": stats["total_decisions"],
        "signals": stats["signals"],
        "skips": stats["skips"],
        "risk_rejects": stats["risk_rejects"],
        "executed": stats["executed"],
        "avg_r_multiple": float(stats["avg_r"]) if stats["avg_r"] else None,
        "total_pnl": float(stats["total_pnl"]) if stats["total_pnl"] else 0,
        "win_rate": win_rate,
        "wins": stats["wins"],
        "losses": stats["losses"],
    }
```

---

## 6. Decision Logger Service

### Integration Point (`services/hl-decide/app/decision_logger.py`)

```python
import uuid
from datetime import datetime
from typing import Optional, List
from .models import (
    DecisionLog, DecisionType, DecisionInputs,
    GateResult, TraderVote, ExecutionResult, ExecutionStatus
)
from .reasoning import generate_reasoning_summary

class DecisionLogger:
    """
    Service for recording and querying decision logs.

    Usage:
        logger = DecisionLogger(db_pool)

        # Log a signal
        decision = logger.create_signal(
            symbol="BTC",
            direction="long",
            inputs=inputs,
            gates=gates,
        )
        await logger.save(decision)

        # Later, update with execution result
        await logger.record_execution(decision.id, execution_result)

        # Even later, update with outcome
        await logger.record_outcome(decision.id, pnl=1234.56, r_multiple=1.25)
    """

    def __init__(self, db_pool):
        self.db = db_pool

    def create_signal(
        self,
        symbol: str,
        direction: str,
        inputs: DecisionInputs,
        gates: List[GateResult],
        risk_checks: Optional[List[GateResult]] = None,
    ) -> DecisionLog:
        """Create a new signal decision log."""
        decision = DecisionLog(
            id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            symbol=symbol,
            direction=direction,
            decision_type=DecisionType.SIGNAL,
            inputs=inputs,
            gates=gates,
            risk_checks=risk_checks or [],
            reasoning_summary="",  # Will be generated
            execution_status=ExecutionStatus.PENDING,
        )
        decision.reasoning_summary = generate_reasoning_summary(decision)
        return decision

    def create_skip(
        self,
        symbol: str,
        inputs: DecisionInputs,
        gates: List[GateResult],
    ) -> DecisionLog:
        """Create a skip decision log."""
        # Determine direction from majority vote
        long_votes = sum(1 for v in inputs.votes if v.direction == "long")
        short_votes = len(inputs.votes) - long_votes
        direction = "long" if long_votes > short_votes else "short" if short_votes > long_votes else "none"

        decision = DecisionLog(
            id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            symbol=symbol,
            direction=direction,
            decision_type=DecisionType.SKIP,
            inputs=inputs,
            gates=gates,
            risk_checks=[],
            reasoning_summary="",
            execution_status=ExecutionStatus.SKIPPED,
        )
        decision.reasoning_summary = generate_reasoning_summary(decision)
        return decision

    def create_risk_reject(
        self,
        symbol: str,
        direction: str,
        inputs: DecisionInputs,
        gates: List[GateResult],
        risk_checks: List[GateResult],
    ) -> DecisionLog:
        """Create a risk-rejected decision log."""
        decision = DecisionLog(
            id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            symbol=symbol,
            direction=direction,
            decision_type=DecisionType.RISK_REJECT,
            inputs=inputs,
            gates=gates,
            risk_checks=risk_checks,
            reasoning_summary="",
            execution_status=ExecutionStatus.SKIPPED,
        )
        decision.reasoning_summary = generate_reasoning_summary(decision)
        return decision

    async def save(self, decision: DecisionLog) -> None:
        """Persist decision to database."""
        await self.db.execute(
            """
            INSERT INTO decision_logs
            (id, timestamp, symbol, direction, decision_type,
             inputs, gates, risk_checks, reasoning_summary,
             execution_status, executions)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """,
            decision.id,
            decision.timestamp,
            decision.symbol,
            decision.direction,
            decision.decision_type.value,
            decision.to_dict()["inputs"],
            decision.to_dict()["gates"],
            decision.to_dict()["risk_checks"],
            decision.reasoning_summary,
            decision.execution_status.value,
            decision.to_dict()["executions"],
        )

    async def record_execution(
        self,
        decision_id: str,
        result: ExecutionResult,
    ) -> None:
        """Record an execution result for a decision."""
        await self.db.execute(
            """
            UPDATE decision_logs
            SET execution_status = 'executed',
                executions = executions || $2::jsonb
            WHERE id = $1
            """,
            decision_id,
            {
                "exchange": result.exchange,
                "order_id": result.order_id,
                "fill_price": result.fill_price,
                "fill_size": result.fill_size,
                "fill_time_ms": result.fill_time_ms,
                "fees": result.fees,
                "slippage_bps": result.slippage_bps,
            },
        )

    async def record_outcome(
        self,
        decision_id: str,
        pnl: float,
        r_multiple: float,
        closed_at: Optional[datetime] = None,
    ) -> None:
        """Record the final outcome when position closes."""
        await self.db.execute(
            """
            UPDATE decision_logs
            SET outcome_pnl = $2,
                outcome_r_multiple = $3,
                outcome_closed_at = $4
            WHERE id = $1
            """,
            decision_id,
            pnl,
            r_multiple,
            closed_at or datetime.utcnow(),
        )
```

---

## 7. WebSocket Events

### Real-Time Decision Stream

```typescript
// WebSocket event types for decisions
interface DecisionEvent {
  type: 'decision:new';
  payload: {
    id: string;
    timestamp: string;
    symbol: string;
    direction: string;
    decision_type: 'signal' | 'skip' | 'risk_reject';
    reasoning_summary: string;
    execution_status: string;
  };
}

interface ExecutionEvent {
  type: 'execution:complete';
  payload: {
    decision_id: string;
    exchange: string;
    fill_price: number;
    fill_size: number;
  };
}

interface OutcomeEvent {
  type: 'outcome:recorded';
  payload: {
    decision_id: string;
    pnl: number;
    r_multiple: number;
  };
}
```

---

## 8. Security Considerations

### No Secret Leakage

The decision log system is designed to **never** expose:

1. **Trader Addresses**: Only rank (1-50) is stored, not actual addresses
2. **API Keys**: No credentials in logs
3. **Exact Positions**: Only aggregated vote counts
4. **Internal IDs**: UUIDs are opaque

### Data Retention

```python
# Cleanup job (run daily)
async def cleanup_old_decisions():
    """Remove decision logs older than retention period."""
    retention_days = int(os.getenv("DECISION_LOG_RETENTION_DAYS", "90"))
    cutoff = datetime.utcnow() - timedelta(days=retention_days)

    deleted = await db.execute(
        "DELETE FROM decision_logs WHERE timestamp < $1",
        cutoff,
    )
    print(f"[hl-decide] Cleaned up {deleted} old decision logs")
```

---

## 9. Integration Points

### In Consensus Detection (`services/hl-decide/app/consensus.py`)

```python
# At the end of consensus detection:

async def process_consensus(
    symbol: str,
    votes: List[Vote],
    gate_results: List[GateResult],
    # ...
):
    # ... existing consensus logic ...

    # Create decision log
    inputs = DecisionInputs(
        trader_count=len(votes),
        agreement_pct=agreement / total,
        effective_k=eff_k,
        avg_confidence=avg_conf,
        ev_estimate=ev,
        price_at_decision=current_price,
        atr_value=atr,
        atr_source=atr_source,
        votes=[
            TraderVote(
                rank=pool_rank[v.address],
                weight=v.weight,
                direction=v.direction,
                confidence=v.confidence,
                correlation_factor=v.corr_factor,
            )
            for v in votes
        ],
    )

    if all(g.passed for g in gate_results):
        # All gates passed - check risk limits
        risk_results = check_risk_limits(symbol, ev, avg_conf)

        if all(r.passed for r in risk_results):
            # Signal!
            decision = logger.create_signal(
                symbol=symbol,
                direction=direction,
                inputs=inputs,
                gates=gate_results,
                risk_checks=risk_results,
            )
        else:
            # Risk rejected
            decision = logger.create_risk_reject(
                symbol=symbol,
                direction=direction,
                inputs=inputs,
                gates=gate_results,
                risk_checks=risk_results,
            )
    else:
        # Gates failed - skip
        decision = logger.create_skip(
            symbol=symbol,
            inputs=inputs,
            gates=gate_results,
        )

    await logger.save(decision)

    # Broadcast to WebSocket clients
    await broadcast_decision(decision)

    return decision
```

---

## 10. Example Decision Logs

### Executed Signal

```json
{
  "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "timestamp": "2025-12-10T14:32:15Z",
  "symbol": "BTC",
  "direction": "long",
  "decision_type": "signal",
  "inputs": {
    "trader_count": 10,
    "agreement_pct": 0.70,
    "effective_k": 3.2,
    "avg_confidence": 0.68,
    "ev_estimate": 0.38,
    "price_at_decision": 43150.50,
    "atr_value": 850.0,
    "atr_source": "marks_1m",
    "votes": [
      { "rank": 1, "weight": 0.15, "direction": "long", "confidence": 0.72, "correlation_factor": 0.85 },
      { "rank": 3, "weight": 0.12, "direction": "long", "confidence": 0.68, "correlation_factor": 0.90 }
    ]
  },
  "gates": [
    { "name": "supermajority", "passed": true, "value": 0.70, "threshold": 0.70, "explanation": "70% agreement meets 70% threshold" },
    { "name": "effective_k", "passed": true, "value": 3.2, "threshold": 2.0, "explanation": "3.2 independent traders exceeds minimum" },
    { "name": "freshness", "passed": true, "value": 12, "threshold": 300, "explanation": "Signal 12s old, within 300s window" },
    { "name": "price_band", "passed": true, "value": 0.12, "threshold": 0.25, "explanation": "0.12R drift within 0.25R band" },
    { "name": "ev_gate", "passed": true, "value": 0.38, "threshold": 0.20, "explanation": "0.38R EV exceeds 0.20R minimum" }
  ],
  "risk_checks": [
    { "name": "min_confidence", "passed": true, "value": 0.68, "threshold": 0.55, "explanation": "68% confidence above 55% minimum" },
    { "name": "min_ev", "passed": true, "value": 0.38, "threshold": 0.20, "explanation": "0.38R above 0.20R minimum" }
  ],
  "reasoning_summary": "7 of 10 Alpha Pool traders opened LONG BTC positions within the consensus window. Effective-K of 3.2 indicates low correlation (traders are making independent decisions). Expected value of +0.38R after estimated fees and slippage. All consensus gates and risk limits passed.",
  "execution_status": "executed",
  "executions": [
    {
      "exchange": "hyperliquid",
      "order_id": "0x7a8b...",
      "fill_price": 43155.00,
      "fill_size": 0.5,
      "fill_time_ms": 1200,
      "fees": 4.32,
      "slippage_bps": 1.04
    }
  ],
  "outcome_pnl": 350.25,
  "outcome_r_multiple": 1.25,
  "outcome_closed_at": "2025-12-10T16:45:30Z"
}
```

### Skipped Signal

```json
{
  "id": "b2c3d4e5-f6a7-8901-bcde-f23456789012",
  "timestamp": "2025-12-10T14:15:00Z",
  "symbol": "ETH",
  "direction": "short",
  "decision_type": "skip",
  "inputs": {
    "trader_count": 10,
    "agreement_pct": 0.40,
    "effective_k": 1.8,
    "avg_confidence": 0.55,
    "ev_estimate": 0.18
  },
  "gates": [
    { "name": "supermajority", "passed": false, "value": 0.40, "threshold": 0.70, "explanation": "Only 40% agreement" },
    { "name": "effective_k", "passed": false, "value": 1.8, "threshold": 2.0, "explanation": "High correlation among traders" }
  ],
  "reasoning_summary": "Signal skipped: 10 traders detected, but consensus gates failed. Only 40% agreement (70% required); Effective-K of 1.8 indicates high correlation among traders (minimum 2.0 required).",
  "execution_status": "skipped"
}
```

---

*This design provides full transparency into SigmaPilot's decision-making process.*
