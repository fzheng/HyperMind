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
| 6 | Multi-Exchange Integration | 🔲 Planned |

---

## Current State: Phase 4-5 Complete

### What's Working

**Core Algorithm Pipeline:**
```
Leaderboard → Quality Filter → Alpha Pool → Thompson Sampling → Consensus → Signal → Execution
   1000+          7 gates         50 traders      NIG posterior       5 gates    Kelly sized
```

**Services:**
| Service | Port | Function |
|---------|------|----------|
| hl-scout | 4101 | Leaderboard scanning, candidate publishing |
| hl-stream | 4102 | Real-time feeds, dashboard, WebSocket |
| hl-sage | 4103 | NIG model, Thompson Sampling selection |
| hl-decide | 4104 | Consensus detection, episode tracking, execution |

**Test Coverage:**
- TypeScript: 1,035 unit tests (28 test suites)
- Python: 348 tests (hl-sage + hl-decide including Kelly, regime, exchange tests)
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

### Test Coverage (Phases 4-5)
- Kelly: 38 tests
- Exchange: 30 tests
- Regime: 39 tests
- Risk Governor: 27 tests + circuit breaker extensions
- Integration: 7 fail-closed tests (retry, metrics)
- **Python total: 384 tests**
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

## Phase 6: Multi-Exchange Integration

### Goal
Expand beyond Hyperliquid to support additional exchanges.

### Tasks
- [ ] Abstract exchange interface
- [ ] Add Binance/Bybit support
- [ ] Unified position tracking across exchanges
- [ ] Cross-exchange risk management

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

*Last updated: December 13, 2025 (Phase 4-5 Integration Complete + Safety Hardening)*
