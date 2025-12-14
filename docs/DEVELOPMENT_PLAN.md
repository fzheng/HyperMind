# SigmaPilot Development Plan

## Vision

A collective intelligence trading system that learns from top Hyperliquid traders and generates consensus-based signals. Not blind copy-trading—intelligent filtering, Bayesian learning, and risk-controlled execution.

---

## Roadmap Overview

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Foundation (leaderboard, streaming, dashboard) | ✅ Complete |
| 2 | Trader Selection (position-based NIG bandit) | ✅ Complete |
| 2.5 | Algorithm Refinements (episode builder, consensus gates) | ✅ Complete |
| 3a | Alpha Pool UI & Runtime Wiring | ✅ Complete |
| 3b | Core Algorithm (Thompson Sampling, ATR, Correlation) | ✅ Complete |
| 3c | Observability & Hardening | ✅ Complete |
| 3d | Fill Sync & Auto-Refresh | ✅ Complete |
| 3e | Decision Logging & Execution Foundation | ✅ Complete |
| 3f | Selection Integrity (Shadow Ledger, FDR, Walk-Forward) | ✅ Complete |
| 4 | Risk Management (Kelly criterion, stops, circuit breakers) | ✅ Complete |
| 5 | Market Regime Detection | ✅ Complete |
| 4-5 Integration | Wire regime/risk/execution together | ✅ Complete |
| 6 | Multi-Exchange Integration | ✅ Complete |
| 6.1 | Multi-Exchange Refinements | ✅ Complete |

---

## Current State: Phase 6.1 Complete

### What's Working

**Core Algorithm Pipeline:**
```
Leaderboard → Quality Filter → Alpha Pool → Thompson Sampling → Consensus → Signal → Execution
   1000+          7 gates         50 traders      NIG posterior       5 gates    Kelly sized
```

**Multi-Exchange Support (Phase 6 Complete):**
```
                    ┌─────────────────────┐
                    │  ExchangeInterface  │
                    │    (Abstract ABC)   │
                    └──────────┬──────────┘
           ┌───────────────────┼───────────────────┐
           ▼                   ▼                   ▼
   ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
   │  Hyperliquid  │   │   Aster DEX   │   │    Bybit      │
   │    Adapter    │   │    Adapter    │   │   Adapter     │
   └───────────────┘   └───────────────┘   └───────────────┘
         DEX                 DEX                 CEX

                    ┌─────────────────────┐
                    │   ExchangeManager   │
                    │  (Singleton Router) │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        Route orders    Health checks    Per-exchange
        to target       w/ stagger       fee config
```

**Services:**
| Service | Port | Function |
|---------|------|----------|
| hl-scout | 4101 | Leaderboard scanning, candidate publishing |
| hl-stream | 4102 | Real-time feeds, dashboard, WebSocket |
| hl-sage | 4103 | NIG model, Thompson Sampling selection |
| hl-decide | 4104 | Consensus detection, episode tracking, multi-exchange execution |

**Test Coverage:**
- TypeScript: 1,035 unit tests (28 test suites)
- Python: 676 tests (hl-sage + hl-decide including Kelly, regime, exchange adapters, ATR/fee/funding/slippage/normalizer/executor/hold-time providers)
- E2E: 220 Playwright tests (6 spec files)

### Phase 3c Additions (December 2025)

**Rate Limiting:**
- All Hyperliquid SDK calls rate-limited (2/s default)
- Exponential backoff on 429 errors
- Configurable via `HL_SDK_CALLS_PER_SECOND`

**Observability Metrics:**
| Metric | Purpose |
|--------|---------|
| `decide_atr_stale_total` | ATR data freshness |
| `decide_atr_fallback_total` | Fallback usage tracking |
| `decide_correlation_stale` | Correlation data freshness |
| `decide_correlation_decay_factor` | Decay applied to correlations |
| `decide_effk_value` | Effective-K distribution |
| `decide_vote_weight_gini` | Weight concentration (0=equal, 1=concentrated) |
| `decide_signal_risk_rejected_total` | Risk limit rejections |
| `decide_signal_generated_total` | Signals passing all gates |

**Risk Fail-Safes:**
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `MAX_POSITION_SIZE_PCT` | 2% | Per-position limit |
| `MAX_TOTAL_EXPOSURE_PCT` | 10% | Total exposure cap |
| `MAX_DAILY_LOSS_PCT` | 5% | Drawdown halt |
| `MIN_SIGNAL_CONFIDENCE` | 55% | Minimum win probability |
| `MIN_SIGNAL_EV_R` | 0.20R | Minimum expected value |
| `MAX_LEVERAGE` | 1x | No leverage until Kelly |
| `SIGNAL_COOLDOWN_SECONDS` | 300s | Anti rapid-fire |

**Alpha Pool Fill Sync (Phase 3d):**
- Periodic fill sync every 5 minutes (`ALPHA_POOL_FILL_SYNC_INTERVAL`)
- New fills published to NATS for episode building
- Frontend polls every 30 seconds for real-time updates
- 21 new tests for fill sync and NATS publishing

**Alpha Pool Auto-Refresh (Phase 3d):**
- Automatic pool refresh every 24 hours (`ALPHA_POOL_REFRESH_HOURS`)
- Hourly check determines if refresh is due
- Consensus signals polled every 60 seconds in frontend
- Real API calls replace mock data in dashboard

**Alpha Pool Fill Backfill (Phase 3d):**
- New addresses automatically get historical fills backfilled on first add
- Manual backfill endpoint: `POST /alpha-pool/backfill/{address}`
- Backfill fetches fills from Hyperliquid API and stores in `hl_events`
- Only genuinely new addresses are backfilled (not re-activated addresses)

**Decision Logging & Dashboard (Phase 3e):**
- All consensus gate decisions logged to `decision_logs` table
- Signal history with outcome tracking in dashboard
- Stats bar showing win rate, EV, and signal counts
- Decision types: `signal`, `rejected`, `cooldown`, `risk_rejected`

**Portfolio & Execution Foundation (Phase 3e):**
- Portfolio API fetches real-time account state from Hyperliquid
- `portfolio_snapshots` table tracks equity over time
- `live_positions` table tracks current open positions
- `execution_config` table for auto-trading settings (disabled by default)
- `execution_logs` table tracks execution attempts and outcomes

**HyperliquidExecutor (Phase 3e):**
- Dry-run execution for Phase 3 (simulation mode)
- Validates risk limits before execution:
  - Max position size (default 2%)
  - Max total exposure (default 10%)
  - Account value verification
- Logs all execution attempts with full context
- Ready for real execution in Phase 4 (requires private key)

**Dashboard Overview Tab (Phase 3e):**
- Portfolio summary with account value and exposure
- Live positions display with P&L
- Execution logs viewer
- Execution config panel (view-only in Phase 3)

---

## E2E Testing Guide

### How to Test Phase 3

1. **Start Services:**
```bash
docker compose up -d
```

2. **Verify Alpha Pool:**
```bash
# Check pool has traders
curl http://localhost:4103/alpha-pool | jq '.count'

# Check fills are being synced
curl http://localhost:4102/dashboard/api/alpha-pool/fills?limit=3
```

3. **Monitor Consensus Detection:**
```bash
# Watch for consensus signals
docker compose logs -f hl-decide 2>&1 | grep -i "consensus\|signal"

# Check signal stats
curl http://localhost:4104/consensus/stats
```

4. **Verify Episode Building:**
```bash
# Episodes are built from fills
curl http://localhost:4104/episode/status
```

### Understanding Consensus Signals

**Real Signals vs Placeholder:**
- Real signals appear when ≥3 Alpha Pool traders agree on direction
- Dashboard shows "Waiting for consensus..." if no signals yet
- Check `consensus_signals` table for historical data

**When Signals Fire:**
1. Multiple traders open same-direction positions
2. All 5 consensus gates pass (dispersion, effK, freshness, price drift, EV)
3. Risk fail-safes pass (confidence, EV threshold)

**If No Signals Appear:**
- Check that traders are actively trading (fills appearing)
- Check ATR data freshness (stale ATR blocks signals)
- Check correlation data (default ρ=0.3 used if not computed)
- Verify consensus thresholds aren't too strict for current activity

### Test Commands

```bash
# Run all tests
npm test && cd services/hl-sage && python -m pytest && cd services/hl-decide && python -m pytest

# Run E2E tests
npm run test:e2e

# Check test coverage
npm run test:coverage
```

---

## Phase 3f: Selection Integrity (In Progress)

### Goal
Eliminate survivorship bias and ensure statistically valid trader selection through:
1. **Shadow Ledger**: Track all traders who ever appeared (including those who "blew up")
2. **Walk-Forward Validation**: Replay selection with as-of data, no look-ahead
3. **FDR Control**: Reduce false positive rate in trader selection
4. **Risk Governor**: Hard caps before any live trading

### Why This Matters
Current selection suffers from:
- **Survivorship bias**: Only see traders who haven't blown up yet
- **Look-ahead bias**: Selection may use future information
- **Multiple testing**: Testing many traders inflates false positives
- **No statistical significance**: No proof traders have real skill vs luck

### Shadow Ledger Schema

```sql
CREATE TABLE trader_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    snapshot_date DATE NOT NULL,
    address TEXT NOT NULL,

    -- Versioning
    selection_version TEXT NOT NULL,  -- e.g., "3f.1"

    -- Multi-universe membership (boolean flags)
    is_leaderboard_scanned BOOLEAN DEFAULT FALSE,
    is_candidate_filtered BOOLEAN DEFAULT FALSE,
    is_quality_qualified BOOLEAN DEFAULT FALSE,
    is_pool_selected BOOLEAN DEFAULT FALSE,
    is_pinned_custom BOOLEAN DEFAULT FALSE,

    -- As-of features
    account_value NUMERIC,
    pnl_30d NUMERIC,
    roi_30d NUMERIC,
    win_rate NUMERIC,
    episode_count INTEGER,
    week_volume NUMERIC,
    orders_per_day NUMERIC,

    -- R-multiple stats (gross vs net)
    avg_r_gross NUMERIC,
    avg_r_net NUMERIC,

    -- NIG posterior params
    nig_mu NUMERIC,
    nig_kappa NUMERIC,
    nig_alpha NUMERIC,
    nig_beta NUMERIC,

    -- Thompson sampling (stored for reproducibility)
    thompson_draw NUMERIC,
    thompson_seed BIGINT,
    selection_rank INTEGER,

    -- Lifecycle events
    event_type TEXT,  -- entered, active, promoted, demoted, death, censored
    death_type TEXT,  -- liquidation, drawdown_80, account_value_floor
    censor_type TEXT, -- inactive_30d, stopped_btc_eth, api_unavailable

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(snapshot_date, address, selection_version)
);
```

### Selection Definition (10 Gates)

| Gate | Criteria | Config |
|------|----------|--------|
| 1 | Min 30d PnL | `ALPHA_POOL_MIN_PNL=10000` |
| 2 | Min 30d ROI | `ALPHA_POOL_MIN_ROI=0.10` |
| 3 | Min Account Value | `ALPHA_POOL_MIN_ACCOUNT_VALUE=100000` |
| 4 | Weekly Activity | `ALPHA_POOL_MIN_WEEK_VLM=10000` |
| 5 | HFT Filter | `ALPHA_POOL_MAX_ORDERS_PER_DAY=100` |
| 6 | No Subaccounts | Checked during filtering |
| 7 | BTC/ETH History | Must have traded BTC or ETH |
| 8 | Min Episodes | `episode_count >= 30` |
| 9 | FDR-Qualified | Benjamini-Hochberg at α=0.10 |
| 10 | Effect Size | `avg_r_net >= 0.05` |

### Death vs Censor Events

**Death (Terminal)** - Trader permanently excluded:
| Type | Trigger |
|------|---------|
| `liquidation` | Account liquidated on Hyperliquid |
| `drawdown_80` | Current equity < 20% of peak |
| `account_value_floor` | Account < $10k |
| `negative_equity` | Account value <= 0 |

**Censored (Non-Terminal)** - Disappeared but not dead:
| Type | Trigger |
|------|---------|
| `inactive_30d` | No fills for 30 days |
| `stopped_btc_eth` | Only trading other assets |
| `api_unavailable` | HL API returns no data |

### Tasks

#### 3f.1 Shadow Ledger ✅
- [x] Design schema with Advisor A feedback
- [x] Create migration for `trader_snapshots` table
- [x] Implement daily snapshot job in hl-sage
- [x] Store Thompson draws for reproducibility
- [x] Track death/censor events
- [x] Add API endpoints (`/snapshots/*`)
- [x] Add unit tests (31 tests passing)

#### 3f.2 FDR Qualification ✅
- [x] Implement Benjamini-Hochberg procedure (correct k* finding)
- [x] Winsorize R-values before t-test
- [x] Store gross/net R-multiples separately
- [x] Effect size gate (avg_r_net >= 0.05)
- [x] Skill p-value computation via one-sided t-test

#### 3f.3 Walk-Forward Replay ✅
- [x] Snapshot-based universe freeze (no look-ahead)
- [x] Single-period replay skeleton
- [x] Cost estimation in replay (30bps round-trip)
- [x] Summary metrics output (Sharpe, win rate, survival)
- [x] API endpoints (`/replay/run`, `/replay/period`)
- [x] Unit tests (13 tests passing)

#### 3f.4 Risk Governor ✅
- [x] Liquidation distance guard (margin ratio < 1.5x)
- [x] Daily drawdown kill switch (5% threshold)
- [x] Equity floor check ($10k minimum)
- [x] Position size limits (10% max)
- [x] Total exposure limits (50% max)
- [x] Kill switch cooldown (24h)
- [x] Unit tests (27 tests passing)
- [x] Migration for state persistence

### Configurable Thresholds

All Phase 3f thresholds are configurable via environment variables in `docker-compose.yml`:

| Variable | Default | Production | Description |
|----------|---------|------------|-------------|
| `SNAPSHOT_MIN_EPISODES` | 5 | 30 | Min episodes for FDR qualification |
| `SNAPSHOT_FDR_ALPHA` | 0.10 | 0.10 | FDR control level (10%) |
| `SNAPSHOT_MIN_AVG_R_NET` | 0.0 | 0.05 | Min avg R-multiple after costs |
| `ROUND_TRIP_COST_BPS` | 30 | 30 | Cost estimate in basis points |
| `DEATH_DRAWDOWN_PCT` | 0.80 | 0.80 | Drawdown threshold for death (80%) |
| `DEATH_ACCOUNT_FLOOR` | 10000 | 10000 | Min account value ($10k) |
| `CENSOR_INACTIVE_DAYS` | 30 | 30 | Days without fills = censored |

**Testing Mode**: Default thresholds are lowered (5 episodes, 0 min R) to allow testing with limited data.

**Production Mode**: Set stricter thresholds for real deployment:
```bash
SNAPSHOT_MIN_EPISODES=30
SNAPSHOT_MIN_AVG_R_NET=0.05
```

### Fresh Install: Automatic Initialization

When you run `docker compose up -d` on a fresh database, **hl-sage automatically initializes the Alpha Pool**:

1. Detects empty Alpha Pool (fresh install)
2. Refreshes from Hyperliquid leaderboard (50 traders)
3. Backfills historical fills for all addresses
4. Creates initial snapshot for FDR qualification

**Watch initialization progress:**
```bash
docker compose logs -f hl-sage
```

#### Manual Initialization (Optional)

If automatic init is disabled or you want custom options:

```bash
# Option 1: Make command
make init

# Option 2: NPM script (cross-platform)
npm run init:alpha-pool

# Option 3: With custom options
node scripts/init-alpha-pool.mjs --limit 100 --delay 1000
```

#### Disable Automatic Initialization

Set in docker-compose.yml or .env:
```bash
ALPHA_POOL_AUTO_INIT=false
```

Verification steps are covered by the 71 unit tests in snapshot.py and walkforward.py test suites.

### Success Criteria

| Criteria | Pass Condition | Status |
|----------|----------------|--------|
| Shadow Ledger | Can compute survival curves by cohort | ✅ Death/censor tracking implemented |
| Walk-Forward | Out-of-sample results include costs | ✅ 30bps round-trip in net R |
| FDR Control | Pool reduced AND out-of-sample stability improves | ✅ BH procedure implemented |
| Risk Governor | DD kill switch triggers in tests | ✅ 27 tests verify behavior |

---

## Phase 4: Risk Management 🔶

### Goal
Implement Kelly criterion position sizing and real trade execution.

### Status: Components Ready, Integration Pending (December 2025)

**Note**: All components are implemented and tested individually. Integration wiring is in progress to connect:
- Regime detection → consensus thresholds, Kelly sizing, stop distances
- Risk governor → signal generation, execution validation
- Real execution path → hl_exchange with safety gates

### Foundation (Phase 3e)
- ✅ Risk limit validation (max position, max exposure)
- ✅ Real-time exposure tracking via Hyperliquid API
- ✅ Dry-run execution with full logging
- ✅ Signal cooldown (300s default)
- ✅ Execution config database table

### Phase 4.1: Kelly Calculator ✅

**Implemented** (December 2025):
- Kelly criterion formula: `f* = p - (1-p)/R`
- Fractional Kelly (25% default) for variance reduction
- Fallback to fixed sizing for insufficient data
- Integration with executor for position sizing
- 38 unit tests covering all edge cases

**Key Files:**
- `services/hl-decide/app/kelly.py` - Kelly calculator module
- `services/hl-decide/tests/test_kelly.py` - Unit tests
- `db/migrations/027_kelly_config.sql` - Config schema

**Configuration:**
```bash
KELLY_ENABLED=false          # Enable Kelly sizing
KELLY_FRACTION=0.25          # Fractional Kelly (quarter Kelly)
KELLY_MIN_EPISODES=30        # Minimum episodes for Kelly calc
KELLY_FALLBACK_PCT=0.01      # Fallback 1% if Kelly fails
```

### Phase 4.2: Real Execution Foundation ✅

**Implemented** (December 2025):
- Exchange API wrapper with safety gates
- Double-gated: env var + config flag required
- Market order with slippage tolerance
- Order tracking in execution_logs

**Key Files:**
- `services/hl-decide/app/hl_exchange.py` - Exchange wrapper
- `db/migrations/028_exchange_config.sql` - Config schema

**Completed** (December 2025):
- [x] Private key signing via hyperliquid-python-sdk (EIP-712)
- [x] Fill confirmation in order response parsing
- [x] 30 unit tests for exchange wrapper

### Phase 4.3: Position Management ✅

**Implemented** (December 2025):
- Local stop-loss monitoring with price polling
- Take-profit at configurable R:R ratio (2:1 default)
- Optional trailing stops
- Position timeout (7 days default)

**Key Files:**
- `services/hl-decide/app/stop_manager.py` - Stop manager
- `db/migrations/029_active_stops.sql` - Stop tracking table

**Configuration:**
```bash
STOP_POLL_INTERVAL_S=5       # Price check frequency
DEFAULT_RR_RATIO=2.0         # Take-profit at 2:1 R:R
MAX_POSITION_HOURS=168       # 7 day timeout
TRAILING_STOP_ENABLED=false  # Trail stops with price
```

### Phase 4.4: Risk Circuit Breakers ✅

**Implemented** (December 2025):
- Max concurrent positions limit (3 default)
- Per-symbol position limit (1 default)
- API error pause (5 min after 3 errors)
- Loss streak pause (1 hour after 5 losses)

**Key Files:**
- `services/hl-decide/app/risk_governor.py` - Extended with circuit breakers
- `db/migrations/030_circuit_breaker_state.sql` - State persistence

**Configuration:**
```bash
MAX_CONCURRENT_POSITIONS=3    # Total position limit
MAX_POSITION_PER_SYMBOL=1     # One per asset
API_ERROR_THRESHOLD=3         # Errors before pause
API_ERROR_PAUSE_SECONDS=300   # 5 minute pause
MAX_CONSECUTIVE_LOSSES=5      # Loss streak threshold
LOSS_STREAK_PAUSE_SECONDS=3600  # 1 hour pause
```

### Test Coverage (Phases 4-6)
- Kelly: 38 tests
- Exchange (hl_exchange): 30 tests
- Exchange Adapters (Phase 6): 34 tests
- Regime: 39 tests
- Risk Governor: 27 tests + circuit breaker extensions
- Integration: 7 fail-closed tests (retry, metrics)
- **Python total: 418 tests**
- **TypeScript total: 1,035 tests**

---

## Phase 5: Market Regime Detection 🔶

### Goal
Adapt strategy parameters based on market conditions.

### Status: Components Ready, Integration Pending (December 2025)

**Note**: Regime detection is implemented with API endpoints but adjustment functions are not yet wired into the live consensus/execution pipeline.

### Regime Types
| Regime | Detection | Response |
|--------|-----------|----------|
| TRENDING | MA20/MA50 spread > 2% | Wider stops (1.2x), full Kelly |
| RANGING | MAs converged, low vol | Tighter stops (0.8x), 75% Kelly |
| VOLATILE | ATR ratio > 1.5x average | Wide stops (1.5x), 50% Kelly |
| UNKNOWN | Insufficient data | Conservative defaults |

### Implementation

**Key Files:**
- `services/hl-decide/app/regime.py` - Regime detection engine
- `services/hl-decide/tests/test_regime.py` - 39 unit tests

**Detection Signals:**
1. **Moving Average Spread**: MA20 vs MA50 relationship
2. **Volatility Ratio**: Current ATR vs historical average
3. **Price Range**: Recent high-low range compression

**API Endpoints:**
- `GET /regime/{asset}` - Regime for single asset
- `GET /regime` - All regimes with summary
- `GET /regime/params` - Parameter presets per regime

**Dashboard Integration:**
- Market Regime card shows current BTC/ETH regime
- Displays MA spread, volatility ratio, ADR metrics
- Auto-refreshes every 60 seconds

**Configuration:**
```bash
REGIME_LOOKBACK_MINUTES=60       # Candle history for detection
REGIME_MA_SHORT=20               # Short MA period (minutes)
REGIME_MA_LONG=50                # Long MA period (minutes)
REGIME_TREND_THRESHOLD=0.02      # 2% MA spread = trending
REGIME_VOLATILITY_HIGH_MULT=1.5  # 1.5x avg = volatile
REGIME_CACHE_TTL_SECONDS=60      # Regime cache duration
```

### Completed Tasks
- [x] Implement regime classifier (MA spread, vol ratio, price range)
- [x] Add regime-specific parameter sets (stop, Kelly, confidence adjustments)
- [x] Add regime adjustment functions for Kelly, stops, confidence
- [x] Add API endpoints for regime data
- [x] Add dashboard regime card with live updates
- [x] 39 unit tests covering all regime types

---

## Phase 4-5 Integration ✅

### Goal
Wire together the Phase 4/5 components into the live signal/execution pipeline.

### Status: Complete (December 2025)

### Integration Tasks

#### 1. Regime → Consensus/Kelly/Stops ✅
- [x] Call `get_regime_adjusted_stop()` in ATR gate before consensus check
- [x] Call `get_regime_adjusted_kelly()` in executor before position sizing
- [x] Call `get_regime_adjusted_confidence()` in risk limit check
- [x] Add regime to decision_logs for auditability

#### 2. Risk Governor → Signal Generation & Execution ✅
- [x] Wire `check_risk_before_trade()` into executor validation
- [x] Block signal if risk checks fail (log as `risk_rejected`)
- [x] Add daily PnL tracking for drawdown kill switch
- [x] Circuit breaker checks before real execution

#### 3. Real Execution Path ✅
- [x] Add branch in `executor.execute_signal()` for `REAL_EXECUTION_ENABLED=true`
- [x] Call `hl_exchange.execute_market_order()` when enabled
- [x] Register stop with `StopManager` after execution
- [x] Circuit breaker gate before order submission

#### 4. Observability Metrics ✅
- [x] Call `update_weight_metrics()` from consensus pipeline
- [x] Increment `effk_default_fallback_counter` when default ρ used
- [x] Regime included in execution context

#### 5. Safety Hardening ✅
- [x] Fail-closed on account state unavailable (block trading, don't proceed)
- [x] Retry with exponential backoff for account state fetch (3 retries, 500ms base)
- [x] Safety block metrics (`decide_safety_block_total{guard=...}`)
  - `kill_switch` - Daily drawdown triggered
  - `account_state` - API unavailable after retries
  - `risk_governor` - Liquidation/exposure limits
  - `circuit_breaker` - Position/API/loss limits
- [x] All safety checks emit metrics for Grafana dashboards

### Success Criteria
| Criteria | Pass Condition |
|----------|----------------|
| Regime affects sizing | Kelly fraction varies by regime in execution logs |
| Risk governor blocks | Signals rejected when risk limits exceeded |
| Real execution works | Orders placed on testnet with `REAL_EXECUTION_ENABLED=true` |
| Metrics emitted | Grafana shows weight_gini, effK values |
| Safety fail-closed | API failures block trading, metrics show block reason |

---

## Phase 6: Multi-Exchange Integration ✅

### Goal
Expand beyond Hyperliquid to support additional exchanges with a unified interface.

### Status: Complete (December 2025)

**All Tasks Completed:**
- [x] Abstract exchange interface (`ExchangeInterface` ABC)
- [x] Hyperliquid adapter (wraps hyperliquid-python-sdk)
- [x] Aster DEX adapter (ECDSA signing, agent wallet support)
- [x] Bybit adapter (pybit SDK, USDT linear perpetuals)
- [x] Exchange factory for adapter creation
- [x] ExchangeManager singleton for order routing
- [x] Health check with rate-limiting stagger (configurable delay)
- [x] Per-exchange fee configuration (FeeConfig dataclass)
- [x] Per-exchange fees wired into Kelly sizing (executor)
- [x] Per-exchange fees wired into consensus EV gate
- [x] Target exchange configuration at startup
- [x] Multi-exchange position tracking via ExchangeManager
- [x] 65 unit tests for exchange module (adapters + manager)

### ExchangeManager (Singleton Router)

The `ExchangeManager` provides centralized exchange routing with these features:

| Feature | Description |
|---------|-------------|
| Singleton pattern | Global instance via `get_exchange_manager()` |
| Multi-exchange registration | Register adapters for HL, Aster, Bybit |
| Order routing | Route to target exchange or default |
| Position aggregation | Unified view across all exchanges |
| Health monitoring | Staggered health checks to avoid API limits |
| Fee configuration | Per-exchange fee lookup via `FeeConfig` |

**Configuration:**
```bash
# Target exchange for execution
EXECUTION_EXCHANGE=hyperliquid     # hyperliquid, aster, or bybit

# Health check rate limiting
EXCHANGE_HEALTH_STAGGER_DELAY_MS=500  # Delay between exchange health checks
```

### Per-Exchange Fee Configuration

Fees are configured per exchange for accurate Kelly sizing and EV calculation:

| Exchange | Maker | Taker | Round-Trip |
|----------|-------|-------|------------|
| Hyperliquid | 2 bps | 5 bps | 10 bps |
| Aster | 2 bps | 5 bps | 10 bps |
| Bybit (VIP0) | 2 bps | 6 bps | 12 bps |

**Fee Flow:**
1. Executor fetches `FeeConfig` for target exchange
2. Calculates `round_trip_fee_pct` from bps
3. Passes to Kelly sizing function
4. Consensus detector uses exchange fees in EV gate

### Exchange Interface Design

All adapters implement the `ExchangeInterface` ABC with these operations:

| Operation | Description |
|-----------|-------------|
| `connect()` / `disconnect()` | Connection lifecycle |
| `get_balance()` | Account equity, margin, P&L |
| `get_positions()` | Open positions with entry/mark prices |
| `open_position()` | Market/limit orders with stops |
| `close_position()` | Partial or full close |
| `set_leverage()` | Leverage configuration |
| `set_stop_loss()` / `set_take_profit()` | Position protection |
| `get_market_price()` / `get_market_data()` | Price and orderbook |
| `format_symbol()` / `format_quantity()` | Exchange-specific formatting |

### Supported Exchanges

| Exchange | Type | Adapter | Symbol Format |
|----------|------|---------|---------------|
| Hyperliquid | DEX | `HyperliquidAdapter` | `BTC`, `ETH` |
| Aster | DEX | `AsterAdapter` | `BTC-PERP`, `ETH-PERP` |
| Bybit | CEX | `BybitAdapter` | `BTCUSDT`, `ETHUSDT` |

### Key Files

| File | Description |
|------|-------------|
| `exchanges/__init__.py` | Module exports |
| `exchanges/interface.py` | Abstract interface & data classes |
| `exchanges/factory.py` | Adapter factory functions |
| `exchanges/manager.py` | ExchangeManager singleton router |
| `exchanges/hyperliquid_adapter.py` | Hyperliquid implementation |
| `exchanges/aster_adapter.py` | Aster DEX implementation |
| `exchanges/bybit_adapter.py` | Bybit implementation |
| `tests/test_exchanges.py` | Adapter unit tests (34 tests) |
| `tests/test_exchange_manager.py` | Manager unit tests (31 tests) |

### Configuration

Each exchange requires credentials via environment variables:

```bash
# Hyperliquid (DEX)
HL_PRIVATE_KEY=0x...
HL_ACCOUNT_ADDRESS=0x...

# Aster (DEX)
ASTER_PRIVATE_KEY=0x...
ASTER_ACCOUNT_ADDRESS=0x...

# Bybit (CEX)
BYBIT_API_KEY=your-api-key
BYBIT_API_SECRET=your-api-secret
```

### Usage Example

```python
from app.exchanges import (
    get_exchange,
    connect_exchange,
    ExchangeType,
    OrderParams,
    OrderSide,
)

# Create adapter with default env vars
exchange = get_exchange(ExchangeType.HYPERLIQUID, testnet=True)

# Or connect directly
exchange = await connect_exchange(ExchangeType.BYBIT, testnet=True)

# Get account state
balance = await exchange.get_balance()
positions = await exchange.get_positions()

# Place order
result = await exchange.open_position(
    OrderParams(
        symbol="BTC",
        side=OrderSide.BUY,
        size=0.01,
        stop_loss=49000.0,
        take_profit=52000.0,
    )
)

await exchange.disconnect()
```

### Success Criteria

| Criteria | Pass Condition | Status |
|----------|----------------|--------|
| Interface abstraction | All 3 adapters pass same test suite | ✅ 34 tests passing |
| Symbol formatting | Each exchange handles formats correctly | ✅ Tested |
| Credential loading | Secure env var loading, no hardcoded keys | ✅ Config pattern |
| Graceful failures | Not-connected returns None/empty, not exceptions | ✅ Tested |
| ExchangeManager routing | Orders route to correct exchange | ✅ 31 tests passing |
| Health check stagger | API rate limits respected | ✅ Configurable delay |
| Per-exchange fees | Kelly/EV use correct fee schedule | ✅ Wired in executor + consensus |

---

## Phase 6.1: Multi-Exchange Refinements 🔶

### Goal
Address remaining gaps in multi-exchange support for production-quality execution across venues.

### Status: Complete ✅ (December 2025)

### Remaining Gaps (from Quant Review)

| Gap | Risk | Priority |
|-----|------|----------|
| ATR/volatility HL-centric | Stop distances wrong for other venues | High |
| Fees static at startup | Miss VIP tier changes mid-session | Medium |
| No funding rate modeling | Holding costs differ 10x between venues | High |
| No slippage modeling | Bybit orderbooks may have less depth | Medium |
| Account state normalization | Bybit USDT vs HL USD equity confusion | Medium |
| No multi-exchange backtest | Unvalidated profitability assumptions | Low |

### Tasks

#### 6.1.1 Venue-Specific Volatility Data ✅
- [x] Abstract ATR provider interface (`ATRProviderInterface` ABC)
- [x] Hyperliquid ATR adapter (uses marks_1m table)
- [x] Bybit ATR adapter (via v5 market kline API)
- [x] ATR Manager for exchange-aware routing
- [x] Exchange-aware ATR lookup in consensus detector
- [x] 36 unit tests for multi-venue ATR

**Why**: Stop distances use ATR. If executing on Bybit but using HL's ATR, stops may be wrong by 20-50% during volatility divergence.

**Key Files:**
- `app/atr_provider/interface.py` - Abstract interface and core functions
- `app/atr_provider/hyperliquid.py` - Hyperliquid provider (DB-backed)
- `app/atr_provider/bybit.py` - Bybit provider (API-backed)
- `app/atr_provider/manager.py` - ATR routing manager
- `tests/test_atr_provider.py` - Unit tests

#### 6.1.2 Dynamic Fee Lookup ✅
- [x] FeeProvider with short-TTL caching (5 min default)
- [x] Static fallback when API unavailable
- [x] `get_exchange_fees_bps_dynamic()` async function in consensus
- [x] Initialization in main.py lifespan
- [x] 20 unit tests for fee provider

**Why**: VIP tiers change, promotions happen. Static fees can be 50% wrong.

**Key Files:**
- `app/fee_provider.py` - Dynamic fee provider with caching
- `tests/test_fee_provider.py` - Unit tests

#### 6.1.3 Funding Rate Modeling ✅
- [x] FundingProvider with short-TTL caching (5 min default)
- [x] Hyperliquid funding API integration (via /info meta endpoint)
- [x] Bybit funding API integration (via v5/market/tickers)
- [x] Static fallback rates when API unavailable
- [x] `get_funding_cost_bps_sync()` for use in consensus detection
- [x] Include funding in EV calculation (`calculate_ev` updated)
- [x] Pre-fetch funding rates on startup
- [x] **Direction-aware funding**: Long pays when rate > 0, short receives (and vice versa)
- [x] 26 unit tests for funding provider

**Why**: Funding rates differ 10x between venues. A +0.01% funding on HL vs -0.05% on Bybit = 0.06%/8h = ~6R/month drag.

**Funding Direction Logic:**
- Positive rate: longs pay shorts → long cost = +rate×intervals, short cost = -rate×intervals (rebate)
- Negative rate: shorts pay longs → long cost = -rate×intervals (rebate), short cost = +rate×intervals
- Consensus detector passes `majority_dir` for correct sign calculation

**Key Files:**
- `app/funding_provider.py` - Funding rate provider with caching
- `app/consensus.py` - `get_funding_cost_bps_sync()`, updated `calculate_ev()`
- `app/main.py` - `update_funding_for_consensus()` startup function
- `tests/test_funding_provider.py` - 26 unit tests

#### 6.1.4 Slippage Estimation ✅
- [x] SlippageProvider with short-TTL caching (1 min default)
- [x] Hyperliquid orderbook API integration (via l2Book endpoint)
- [x] Bybit orderbook API integration (via v5/market/orderbook)
- [x] Static fallback slippage estimates by order size
- [x] `get_slippage_estimate_bps_sync()` for use in consensus detection
- [x] Include slippage in EV calculation (`calculate_ev` updated)
- [x] Warning threshold for high slippage (10 bps default)
- [x] Pre-fetch orderbooks on startup
- [x] 30 unit tests for slippage provider

**Why**: Executing $100k on thin orderbook can add 20+ bps slippage.

**Key Files:**
- `app/slippage_provider.py` - Orderbook-based slippage estimation
- `app/consensus.py` - `get_slippage_estimate_bps_sync()`, updated EV calculation
- `app/main.py` - `update_orderbooks_for_slippage()` startup function
- `tests/test_slippage_provider.py` - 30 unit tests

#### 6.1.5 Account State Normalization ✅
- [x] Normalize all equity to USD (Bybit USDT → USD conversion)
- [x] USDT/USD rate fetching with caching (CoinGecko API, 60s TTL)
- [x] Static fallback rate (1.0) when API unavailable
- [x] Depeg warning detection (>0.5% deviation from $1.00)
- [x] `NormalizedBalance` dataclass for USD-normalized values
- [x] `NormalizedPosition` dataclass for USD-normalized notional
- [x] Sync and async normalization methods
- [x] Integration with main.py startup (pre-fetch USDT rate)
- [x] 33 unit tests for account normalizer

**Why**: Risk sizing and exposure caps can be wrong on non-HL venues until equity is normalized. Bybit reports USDT (stablecoin), HL reports USD (mark-to-market).

**Key Files:**
- `app/account_normalizer.py` - Account state normalizer with USDT/USD conversion
- `tests/test_account_normalizer.py` - 33 unit tests

**Usage:**
```python
from app.account_normalizer import get_account_normalizer

normalizer = get_account_normalizer()
bybit_balance = await bybit.get_balance()  # currency="USDT"
normalized = await normalizer.normalize_balance(bybit_balance)
print(f"Equity: ${normalized.total_equity_usd:.2f}")  # Now in USD
```

### Known Gaps (from Quant Review)

The following gaps have been identified through quant review and need to be addressed for production correctness:

#### Gap 1: Slippage Sizing ✅ (Resolved)
**Issue**: Slippage estimation uses vote notional ($100k reference), not actual Kelly-sized position.
**Impact**: Slippage could be underestimated by 2-5x for larger positions, overestimated for smaller.
**Fix Applied**: Two-stage slippage calculation:
1. Consensus detection uses `SLIPPAGE_REFERENCE_SIZE_USD` ($10k) for initial EV gating
2. Executor recalculates slippage with actual Kelly-sized position after sizing
3. EV is re-validated with actual slippage - signal rejected if EV drops below minimum
- Added `SLIPPAGE_REFERENCE_SIZE_USD` config (default $10k)
- Executor logs actual slippage and EV for audit trail
- 6 new tests for slippage recalculation

#### Gap 2: Hold-Time Assumption ✅ (Resolved)
**Issue**: Funding cost uses fixed 24-hour hold-time assumption (`DEFAULT_HOLD_HOURS=24`).
**Impact**: May over/underestimate funding by 2-3x if typical holds are 12h or 48h.
**Fix Applied**: Dynamic hold-time estimation from historical episode data:
- `HoldTimeEstimator` computes median hold time from `position_signals.hold_secs`
- Per-asset estimates (BTC and ETH tracked separately)
- Regime-adjusted multipliers (TRENDING +25%, VOLATILE -25%)
- 5-minute cache TTL with fallback to default 24h when insufficient data
- Wired into consensus detection and executor for accurate funding cost
- 26 new tests for hold time estimation

#### Gap 3: Market Data HL-Centric ✅ (Resolved)
**Issue**: ATR, regime, and correlation data still sourced from Hyperliquid only.
**Impact**: If executing on Bybit, ATR/stops based on HL data may be 20-50% wrong during volatility divergence.
**Fix Applied**: Multi-exchange regime detection implemented:
- `RegimeDetector` now accepts `exchange` parameter for venue-specific detection
- `_fetch_candles_multi_exchange()` routes through ATR provider infrastructure
- Cache keys include exchange for separate per-venue regime caching
- Fallback to Hyperliquid DB if target exchange unavailable
- `RegimeAnalysis` includes `exchange` field for audit trail
- 10 new tests for multi-exchange regime detection

**Key Files:**
- `app/regime.py` - Updated with multi-exchange support
- `tests/test_regime.py` - 49 tests (39 existing + 10 new)

#### Gap 4: Static Target Exchange ✅ (Resolved)
**Issue**: EV calculation uses single `self._target_exchange` for all signals.
**Impact**: Cannot compare profitability across venues for same signal.
**Fix Applied**: Per-venue EV calculation and comparison:
- `ConsensusDetector.calculate_ev_for_exchange()` - Calculate EV for specific exchange
- `ConsensusDetector.compare_ev_across_exchanges()` - Compare EV across venues and find best
- Returns detailed cost breakdown (fees, slippage, funding, hold_hours per exchange)
- Identifies best execution venue by net EV
- 17 new tests for per-venue EV calculation

**Key Files:**
- `app/consensus.py` - Added `calculate_ev_for_exchange()` and `compare_ev_across_exchanges()`
- `tests/test_ev_per_venue.py` - 17 new tests

#### Gap 5: Account Normalization Usage ✅ (Resolved)
**Issue**: Account normalizer exists but risk/exposure/sizing paths may not consume normalized equity.
**Fix Applied**: Executor `_to_hl_account_state()` now uses `get_account_normalizer().normalize_balance_sync()` to convert all values to USD before passing to risk governor. USDT → USD conversion happens automatically for Bybit positions.
- `accountValue`, `totalMarginUsed`, `totalNtlPos` are all USD-normalized
- `_normalization` metadata included for audit trail (original_currency, conversion_rate, is_depeg_warning)

### Pre-Live Validation Checklist

Before enabling live trading on any venue, validate:

1. **Multi-venue sim/backtest**: Run with venue-specific vol/funding/slippage and empirical hold-time distribution
2. ~~**Slippage sizing**: Confirm slippage uses actual Kelly-sized notional, not vote total~~ ✅ Fixed
3. ~~**Hold-time calibration**: Validate 24h assumption against actual episode durations~~ ✅ Fixed
4. ~~**Normalized exposure**: Verify risk paths consume USD-normalized equity~~ ✅ Fixed
5. ~~**Per-venue ATR/regime**: Confirm stops use target exchange volatility, not HL~~ ✅ Fixed
6. ~~**Per-venue EV comparison**: Confirm EV can be calculated per exchange~~ ✅ Fixed

### Success Criteria

| Criteria | Pass Condition | Status |
|----------|----------------|--------|
| Venue-specific ATR | Each exchange uses own volatility data | ✅ 36 tests |
| Dynamic fees | Fees refresh with TTL caching | ✅ 20 tests |
| Funding in EV | Hold cost affects signal selection | ✅ 26 tests |
| Slippage estimation | Large orders get slippage warning | ✅ 30 tests |
| Account normalization | USDT/USD equity consistent | ✅ 33 tests |
| Normalization wired | Risk paths use normalized equity | ✅ 22 tests |
| Slippage re-calc | Executor uses Kelly-sized position | ✅ 6 tests |
| Dynamic hold-time | Funding uses historical episode data | ✅ 26 tests |
| Multi-exchange regime | Regime detection per venue | ✅ 49 tests |
| Per-venue EV | EV calculation per exchange | ✅ 17 tests |
| Aggregated balance normalized | ExchangeManager uses USD normalization | ✅ Wired |

**Total Phase 6.1 Tests: 720+** (703 Python tests including all provider tests)

---

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  hl-scout   │────▶│  hl-sage    │────▶│  hl-stream  │────▶│  hl-decide  │
│   :4101     │     │   :4103     │     │   :4102     │     │   :4104     │
├─────────────┤     ├─────────────┤     ├─────────────┤     ├─────────────┤
│ Leaderboard │     │ NIG Model   │     │ Dashboard   │     │ Consensus   │
│ Scanning    │     │ Thompson    │     │ WebSocket   │     │ Detection   │
│ Filtering   │     │ Sampling    │     │ Fills       │     │ Episodes    │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │                   │
       └───────────────────┴───────────────────┴───────────────────┘
                                    │
                              ┌─────┴─────┐
                              │  NATS     │
                              │ PostgreSQL│
                              └───────────┘
```

### Key Files

| Component | File |
|-----------|------|
| NIG Model | `services/hl-sage/app/bandit.py` |
| Thompson Sampling | `services/hl-sage/app/main.py` |
| Shadow Ledger | `services/hl-sage/app/snapshot.py` |
| Walk-Forward Replay | `services/hl-sage/app/walkforward.py` |
| Consensus Detection | `services/hl-decide/app/consensus.py` |
| ATR Provider | `services/hl-decide/app/atr.py` |
| Correlation | `services/hl-decide/app/correlation.py` |
| Episode Tracker | `services/hl-decide/app/episode.py` |
| Decision Logger | `services/hl-decide/app/decision_logger.py` |
| Risk Governor | `services/hl-decide/app/risk_governor.py` |
| Portfolio Manager | `services/hl-decide/app/portfolio.py` |
| Trade Executor | `services/hl-decide/app/executor.py` |
| Kelly Calculator | `services/hl-decide/app/kelly.py` |
| Exchange Wrapper | `services/hl-decide/app/hl_exchange.py` |
| Regime Detector | `services/hl-decide/app/regime.py` |
| Exchange Interface | `services/hl-decide/app/exchanges/interface.py` |
| Exchange Factory | `services/hl-decide/app/exchanges/factory.py` |
| Hyperliquid Adapter | `services/hl-decide/app/exchanges/hyperliquid_adapter.py` |
| Aster Adapter | `services/hl-decide/app/exchanges/aster_adapter.py` |
| Bybit Adapter | `services/hl-decide/app/exchanges/bybit_adapter.py` |
| Exchange Manager | `services/hl-decide/app/exchanges/manager.py` |
| Dashboard | `services/hl-stream/public/dashboard.html` |
| Init Script | `scripts/init-alpha-pool.mjs` |

---

## Configuration Reference

### Core Settings
```bash
# Services
NATS_URL=nats://nats:4222
DATABASE_URL=postgresql://hlbot:hlbotpassword@postgres:5432/hlbot
OWNER_TOKEN=your-secret-token

# Consensus Gates
CONSENSUS_MIN_TRADERS=3
CONSENSUS_MIN_PCT=0.70
CONSENSUS_MIN_EFFECTIVE_K=2.0
CONSENSUS_EV_MIN_R=0.20
CONSENSUS_MAX_PRICE_DRIFT_R=0.25

# ATR & Volatility
ATR_MULTIPLIER_BTC=2.0
ATR_MULTIPLIER_ETH=1.5
ATR_STRICT_MODE=true
ATR_MAX_STALENESS_SECONDS=300

# Correlation
CORR_DECAY_HALFLIFE_DAYS=3.0
CORR_REFRESH_INTERVAL_HOURS=24
DEFAULT_CORRELATION=0.3

# Vote Weighting
VOTE_WEIGHT_MODE=log              # log, equity, or linear
VOTE_WEIGHT_LOG_BASE=10000.0
VOTE_WEIGHT_MAX=1.0

# Risk Limits (Phase 3c fail-safes)
MAX_POSITION_SIZE_PCT=2.0
MAX_TOTAL_EXPOSURE_PCT=10.0
MAX_DAILY_LOSS_PCT=5.0
MIN_SIGNAL_CONFIDENCE=0.55
MAX_LEVERAGE=1.0
SIGNAL_COOLDOWN_SECONDS=300

# Alpha Pool Quality Filters
ALPHA_POOL_MIN_PNL=10000
ALPHA_POOL_MIN_ROI=0.10
ALPHA_POOL_MIN_ACCOUNT_VALUE=100000
ALPHA_POOL_MAX_ORDERS_PER_DAY=100
```

---

## Quick Start

```bash
# Start services
docker compose up -d

# View dashboard
open http://localhost:4102/dashboard

# Run tests
make test            # All tests
make test-ts         # TypeScript only
make test-py         # Python only

# View logs
docker compose logs -f hl-decide
```

---

## Algorithm Summary

### The Problem
1. **Selection**: Which traders to follow? Leaderboards lag reality.
2. **Correlation**: "5 traders agree" may be 1 signal repeated 5 times.
3. **Measurement**: Win rate misleads; a 30% winner with +5R/-0.5R beats 80% winner with +0.2R/-0.5R.

### Our Solution

**Layer 1: Position Lifecycle**
- Track complete position lifecycles (open → close)
- Calculate R-multiples from realized P&L
- 1 position = 1 data point (not 1000s of fills)

**Layer 2: NIG Bayesian Model**
```
μ | σ² ~ N(m, σ²/κ)      # Mean R given variance
σ² ~ InverseGamma(α, β)  # Variance of R
```
- Proper uncertainty quantification
- New traders have wide posteriors (exploration)
- Proven traders have narrow posteriors (exploitation)

**Layer 3: Thompson Sampling**
- Sample μ from each trader's posterior
- Rank by sampled value, select top K
- Natural explore/exploit balance

**Layer 4: 5-Gate Consensus**
1. **Dispersion**: ≥70% agreement, ≥3 traders
2. **Effective-K**: Correlation-adjusted ≥2.0
3. **Freshness**: Oldest vote within window
4. **Price Drift**: <0.25R from median
5. **EV Gate**: Net EV ≥0.2R after costs

**Layer 5: Risk Fail-Safes**
- Min confidence (55%)
- Min EV (0.2R)
- Position/exposure limits
- Cooldowns

---

*Last updated: December 14, 2025 (Phase 6.1 complete: All multi-exchange gaps resolved - 703 Python tests; per-venue ATR/regime/EV; USD normalization in all risk paths; Kelly-sized slippage recalculation; dynamic hold-time estimation)*
