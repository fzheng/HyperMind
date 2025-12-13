# Phase 3e Technical Plan

*Revised December 2025*

## Scope Clarification

SigmaPilot monitors traders **on Hyperliquid only**. Multi-exchange is only relevant for **trade execution** (Phase 4+), not for data collection.

**Phase 3e Focus:**
1. Decision Logging - Auditability for signal quality
2. Dashboard P&L - Show outcomes and performance
3. Trade Execution Foundation - Hyperliquid-first, extensible later

---

## 1. Decision Logging System

### Purpose
Record every consensus evaluation with reasoning so we can:
- Understand why signals fired or were skipped
- Measure signal quality over time
- Debug gate failures
- Build trust through transparency

### Database Schema

```sql
-- db/migrations/060_decision_logs.sql

CREATE TABLE decision_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Signal identification
    symbol VARCHAR(16) NOT NULL,           -- BTC, ETH
    direction VARCHAR(8) NOT NULL,         -- long, short, none
    decision_type VARCHAR(16) NOT NULL,    -- signal, skip, risk_reject

    -- Inputs (aggregated, no addresses)
    trader_count INT NOT NULL,
    agreement_pct DECIMAL(5,4) NOT NULL,
    effective_k DECIMAL(6,3) NOT NULL,
    avg_confidence DECIMAL(5,4),
    ev_estimate DECIMAL(8,4),
    price_at_decision DECIMAL(20,8),

    -- Gate results (JSONB for flexibility)
    gates JSONB NOT NULL,
    -- Example: [{"name": "supermajority", "passed": true, "value": 0.70, "threshold": 0.70}]

    -- Risk check results (if applicable)
    risk_checks JSONB,

    -- Human-readable summary (2-4 sentences)
    reasoning TEXT NOT NULL,

    -- Outcome tracking (updated when position closes)
    outcome_pnl DECIMAL(20,8),
    outcome_r_multiple DECIMAL(8,4),
    outcome_closed_at TIMESTAMPTZ
);

-- Indexes
CREATE INDEX idx_decision_logs_created ON decision_logs(created_at DESC);
CREATE INDEX idx_decision_logs_symbol ON decision_logs(symbol, created_at DESC);
CREATE INDEX idx_decision_logs_type ON decision_logs(decision_type);
```

### Integration Point (hl-decide)

Modify `services/hl-decide/app/consensus.py` to log decisions:

```python
# After consensus evaluation
async def log_decision(
    db,
    symbol: str,
    direction: str,
    decision_type: str,  # "signal", "skip", "risk_reject"
    trader_count: int,
    agreement_pct: float,
    effective_k: float,
    gates: list[dict],
    risk_checks: list[dict] | None = None,
    price: float | None = None,
    confidence: float | None = None,
    ev: float | None = None,
) -> str:
    """Log a decision and return its ID."""

    reasoning = generate_reasoning(
        decision_type, symbol, direction,
        trader_count, agreement_pct, effective_k, gates
    )

    row = await db.fetchrow("""
        INSERT INTO decision_logs
        (symbol, direction, decision_type, trader_count, agreement_pct,
         effective_k, avg_confidence, ev_estimate, price_at_decision,
         gates, risk_checks, reasoning)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        RETURNING id
    """, symbol, direction, decision_type, trader_count, agreement_pct,
        effective_k, confidence, ev, price,
        json.dumps(gates), json.dumps(risk_checks) if risk_checks else None,
        reasoning)

    return str(row["id"])
```

### Reasoning Generator

```python
def generate_reasoning(
    decision_type: str,
    symbol: str,
    direction: str,
    trader_count: int,
    agreement_pct: float,
    effective_k: float,
    gates: list[dict],
) -> str:
    """Generate human-readable explanation."""

    if decision_type == "signal":
        return (
            f"{trader_count} Alpha Pool traders opened {direction.upper()} {symbol}. "
            f"{agreement_pct:.0%} agreement with effK={effective_k:.1f}. "
            f"All consensus gates passed."
        )

    elif decision_type == "skip":
        failed = [g for g in gates if not g["passed"]]
        reasons = []
        for g in failed:
            if g["name"] == "supermajority":
                reasons.append(f"only {g['value']:.0%} agreement (need {g['threshold']:.0%})")
            elif g["name"] == "effective_k":
                reasons.append(f"effK={g['value']:.1f} too low (need {g['threshold']:.1f})")
            elif g["name"] == "freshness":
                reasons.append(f"signal {g['value']:.0f}s stale")
            elif g["name"] == "price_band":
                reasons.append(f"price drifted {g['value']:.2f}R")
            elif g["name"] == "ev_gate":
                reasons.append(f"EV={g['value']:.2f}R below threshold")

        return (
            f"Skipped: {trader_count} traders detected but gates failed. "
            f"{'; '.join(reasons)}."
        )

    elif decision_type == "risk_reject":
        return (
            f"Consensus detected but rejected by risk limits. "
            f"{trader_count} traders, {agreement_pct:.0%} agreement."
        )

    return "Decision recorded."
```

### API Endpoints (hl-decide)

```python
# GET /decisions?symbol=BTC&type=signal&limit=50
@app.get("/decisions")
async def list_decisions(
    symbol: str | None = None,
    decision_type: str | None = None,
    limit: int = 50,
    offset: int = 0,
):
    """List decision logs with filters."""
    # Query with filters...

# GET /decisions/{id}
@app.get("/decisions/{decision_id}")
async def get_decision(decision_id: str):
    """Get full decision details."""

# GET /decisions/stats
@app.get("/decisions/stats")
async def get_decision_stats(days: int = 7):
    """Aggregate stats: signal count, skip rate, outcomes."""
```

---

## 2. Dashboard P&L Display

### Purpose
Show signal outcomes and trader performance in the UI.

### New Dashboard Section: "Signal History"

Add to Alpha Pool tab:
- Recent signals with outcome (win/loss/pending)
- Aggregate stats: win rate, avg R-multiple, total signals

### API Changes (hl-stream proxy)

```typescript
// Proxy to hl-decide
app.get('/dashboard/api/decisions', async (req, res) => {
  const response = await fetch(`${DECIDE_URL}/decisions?${req.query}`);
  res.json(await response.json());
});

app.get('/dashboard/api/decisions/stats', async (req, res) => {
  const response = await fetch(`${DECIDE_URL}/decisions/stats`);
  res.json(await response.json());
});
```

### Frontend Changes

```javascript
// In dashboard.js - Add Signal History section
async function loadSignalHistory() {
  const response = await fetch('/dashboard/api/decisions?limit=20');
  const data = await response.json();

  renderSignalHistory(data.items);
  updateSignalStats(data.stats);
}

function renderSignalHistory(decisions) {
  // Render cards showing:
  // - Symbol, direction, timestamp
  // - Reasoning summary
  // - Outcome (if closed): P&L, R-multiple
  // - Gate results (expandable)
}
```

---

## 3. Trade Execution Foundation (Hyperliquid Only)

### Purpose
Enable optional auto-trading on Hyperliquid when signals fire.

### Scope for Phase 3e
- Hyperliquid execution adapter only
- Conservative defaults (disabled by default)
- No multi-exchange yet

### Database Schema

```sql
-- db/migrations/061_execution_config.sql

CREATE TABLE execution_config (
    id SERIAL PRIMARY KEY,
    enabled BOOLEAN DEFAULT false,

    -- Hyperliquid settings
    hl_enabled BOOLEAN DEFAULT false,
    hl_max_leverage INT DEFAULT 3,
    hl_max_position_pct DECIMAL(5,4) DEFAULT 0.02,  -- 2% max
    hl_max_exposure_pct DECIMAL(5,4) DEFAULT 0.10,  -- 10% total

    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Execution log
CREATE TABLE execution_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    decision_id UUID REFERENCES decision_logs(id),
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Order details
    symbol VARCHAR(16) NOT NULL,
    side VARCHAR(8) NOT NULL,        -- buy, sell
    size DECIMAL(20,8) NOT NULL,
    leverage INT NOT NULL,

    -- Result
    status VARCHAR(16) NOT NULL,     -- pending, filled, failed, rejected
    fill_price DECIMAL(20,8),
    fill_size DECIMAL(20,8),
    error_message TEXT,

    -- Risk context
    position_pct DECIMAL(5,4),       -- % of equity used
    exposure_before DECIMAL(5,4),    -- exposure before trade
    exposure_after DECIMAL(5,4)      -- exposure after trade
);

CREATE INDEX idx_execution_logs_decision ON execution_logs(decision_id);
CREATE INDEX idx_execution_logs_created ON execution_logs(created_at DESC);
```

### Hyperliquid Executor

```python
# services/hl-decide/app/executor.py

from hyperliquid.exchange import Exchange
from hyperliquid.info import Info

class HyperliquidExecutor:
    """Execute trades on Hyperliquid."""

    def __init__(self, private_key: str, address: str):
        self.info = Info()
        self.exchange = Exchange(private_key)
        self.address = address

    async def get_account_value(self) -> float:
        """Get current account value."""
        state = self.info.user_state(self.address)
        return float(state["marginSummary"]["accountValue"])

    async def get_current_exposure(self) -> float:
        """Get current total exposure as % of equity."""
        state = self.info.user_state(self.address)
        account_value = float(state["marginSummary"]["accountValue"])
        total_notional = sum(
            abs(float(p["position"]["szi"]) * float(p["position"]["entryPx"]))
            for p in state["assetPositions"]
        )
        return total_notional / account_value if account_value > 0 else 0

    async def execute_signal(
        self,
        symbol: str,
        direction: str,
        config: dict,
    ) -> dict:
        """
        Execute a signal with risk checks.

        Returns: {status, fill_price, fill_size, error_message}
        """
        # Get account state
        account_value = await self.get_account_value()
        current_exposure = await self.get_current_exposure()

        # Check exposure limit
        if current_exposure >= config["hl_max_exposure_pct"]:
            return {
                "status": "rejected",
                "error_message": f"Exposure {current_exposure:.1%} >= {config['hl_max_exposure_pct']:.1%} limit"
            }

        # Calculate position size
        max_size_usd = account_value * config["hl_max_position_pct"]

        # Get current price
        price = float(self.info.all_mids()[symbol])
        size_coin = max_size_usd / price

        # Execute
        try:
            is_buy = direction == "long"
            result = self.exchange.market_open(
                coin=symbol,
                is_buy=is_buy,
                sz=size_coin,
                slippage=0.01,  # 1% max slippage
            )

            return {
                "status": "filled",
                "fill_price": result["avgPx"],
                "fill_size": result["filledSz"],
            }
        except Exception as e:
            return {
                "status": "failed",
                "error_message": str(e),
            }
```

### Integration with Signal Generation

```python
# In consensus detection, after signal fires:

if decision_type == "signal" and config["enabled"] and config["hl_enabled"]:
    execution_result = await executor.execute_signal(
        symbol=symbol,
        direction=direction,
        config=config,
    )

    await db.execute("""
        INSERT INTO execution_logs
        (decision_id, symbol, side, size, leverage, status,
         fill_price, fill_size, error_message)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
    """, decision_id, symbol, "buy" if direction == "long" else "sell",
        execution_result.get("fill_size"), config["hl_max_leverage"],
        execution_result["status"], execution_result.get("fill_price"),
        execution_result.get("fill_size"), execution_result.get("error_message"))
```

---

## 4. Implementation Order

### Week 1: Decision Logging
1. Add `decision_logs` migration
2. Implement `log_decision()` in hl-decide
3. Integrate with consensus detection
4. Add `/decisions` API endpoints
5. Write tests

### Week 2: Dashboard Integration
1. Add proxy endpoints in hl-stream
2. Create Signal History UI section
3. Display decision reasoning
4. Show aggregate stats
5. E2E tests

### Week 3: Execution Foundation (Optional)
1. Add execution config/logs migrations
2. Implement HyperliquidExecutor
3. Add safety checks (exposure, position limits)
4. Integrate with signal flow (disabled by default)
5. Admin API to enable/configure

---

## 5. What's NOT in Phase 3e (Deferred to Phase 4+)

### Phase 4: Multi-Exchange Execution
- **Binance Futures execution adapter** - Similar to HyperliquidExecutor but for Binance
- **Bybit execution adapter** - Support Bybit perpetuals
- **OKX execution adapter** - Support OKX perpetuals
- **Exchange-specific rate limiting** - Per-exchange API limits
- **Unified order management** - Track orders across exchanges
- **Cross-exchange position reconciliation** - Sync positions with exchange state

### Phase 4: Advanced Position Sizing
- **Kelly criterion sizing** - Optimal position sizing based on edge
- **Fractional Kelly** - Conservative Kelly (e.g., half-Kelly)
- **Volatility-adjusted sizing** - Scale size by ATR/volatility
- **Correlation-adjusted portfolio** - Reduce correlated exposure

### Phase 4: Security & Infrastructure
- **Credential encryption/KMS** - Secure storage for exchange API keys
- **Hardware security module (HSM)** - For production key management
- **Audit logging** - All credential access logged
- **IP whitelisting** - Exchange-level IP restrictions

### Phase 5+: Future Enhancements
- React dashboard rewrite - Modern SPA architecture
- Mobile app - iOS/Android native apps
- Telegram/Discord bot - Trade notifications
- Webhook integrations - External alerting

---

### Per-Exchange Breakdown (Phase 4 Architecture)

When multi-exchange is implemented, the dashboard will show:

```
┌─────────────────────────────────────────────────────────┐
│ Overview                                                 │
├─────────────────────────────────────────────────────────┤
│ Total Equity: $125,432    Unrealized P&L: +$2,341       │
├─────────────────────────────────────────────────────────┤
│ Exchange      │ Equity    │ Positions │ Exposure │ P&L  │
├───────────────┼───────────┼───────────┼──────────┼──────┤
│ Hyperliquid   │ $50,000   │ 2         │ 15%      │ +$800│
│ Binance       │ $45,432   │ 1         │ 8%       │ +$541│
│ Bybit         │ $30,000   │ 0         │ 0%       │ $0   │
└─────────────────────────────────────────────────────────┘
```

### Per-Strategy Breakdown (Phase 4)

Multiple strategies can run concurrently:
- **Alpha Pool Consensus** - Current strategy (consensus signals)
- **Single Trader Copy** - Follow individual high-conviction traders
- **Mean Reversion** - Counter-trend on extreme deviations

Each strategy will have:
- Independent risk limits
- Separate P&L tracking
- Enable/disable toggle
- Performance metrics

---

## 6. Success Criteria

1. **Decision logs working**: Every consensus evaluation logged with reasoning
2. **Dashboard shows signals**: Recent signals visible with outcomes
3. **Stats queryable**: Win rate, R-multiple, skip rate available
4. **Execution ready** (optional): Can enable HL auto-trade with conservative limits

---

## 7. Files to Create/Modify

### New Files
- `db/migrations/060_decision_logs.sql`
- `db/migrations/061_execution_config.sql` (optional)
- `services/hl-decide/app/decision_logger.py`
- `services/hl-decide/app/executor.py` (optional)
- `tests/test_decision_logging.py`

### Modified Files
- `services/hl-decide/app/main.py` - Add decision logging endpoints
- `services/hl-decide/app/consensus.py` - Log decisions
- `services/hl-stream/src/index.ts` - Proxy decision endpoints
- `services/hl-stream/public/dashboard.js` - Signal history UI
- `services/hl-stream/public/dashboard.html` - Signal history section

---

*Ready for review. Focus is on decision logging + dashboard, execution is optional.*
