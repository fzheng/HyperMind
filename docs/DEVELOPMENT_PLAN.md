# SigmaPilot Development Plan

## Business Goal

> A private, single-tenant platform that continuously tracks Hyperliquid wallets, identifies consistently strong performers, filters out low-value activity (losers, noise traders, HFT churn), and uses an online-learning engine to generate risk-controlled trade recommendations. It can connect to a Hyperliquid account (read-only first, then trading) to execute a rules-based "follow-the-leaders" strategy that self-adapts as leaders change in bull or bear markets.

### Core Principles

1. **Not blind copy-trading** - Intelligently filter who and when to follow
2. **Online learning** - Continuously adapt to changing market conditions and trader performance
3. **Risk-controlled** - Kelly criterion position sizing, drawdown limits, exposure management
4. **Self-adapting** - Automatically adjust to bull/bear markets and leader changes
5. **Private single-tenant** - Your own instance, your own data, your own edge

---

## Chief Scientist Overview: The Algorithm

### The Problem We're Solving

The naive approach to copy-trading fails for three reasons:

1. **Selection Problem**: Which traders should we follow? Past performance on leaderboards doesn't predict future returns.
2. **Correlation Problem**: Top traders often follow each other, so "5 traders agree" might really be "1 signal repeated 5 times."
3. **Measurement Problem**: Win rate is misleading. A 30% win-rate trader making +5R on wins and -0.5R on losses is far better than an 80% win-rate trader making +0.2R/-0.5R.

### Our Solution: Position-Based Bayesian Learning

We solve these problems with a three-layer approach:

#### Layer 1: Position Lifecycle Tracking

Instead of treating every fill as a signal (which led to 20,000+ spam "signals" in 5 hours), we track **complete position lifecycles**:

```
Position Open (Open Long/Short) → Track entry
Position Close (Close All) → Measure actual P&L
```

This means:
- 1 position = 1 data point
- We use Hyperliquid's `realized_pnl` for accuracy
- R-multiple = realized_pnl / (entry_notional × assumed_stop)

#### Layer 2: Normal-Inverse-Gamma (NIG) Posterior

For each trader, we maintain a Bayesian estimate of their expected R-multiple:

```
μ | σ² ~ N(m, σ²/κ)     # Mean R given variance
σ² ~ InverseGamma(α, β)  # Variance of R
```

**Why NIG instead of simple averages?**
- Proper uncertainty quantification (new traders have wide posteriors)
- Handles heavy-tailed R distributions (winsorize to ±2R)
- Naturally balances exploration (uncertain traders) vs exploitation (proven performers)
- Conjugate updates = fast, exact Bayesian inference

**The update formula** (on position close with R-multiple `r`):
```python
κ' = κ + 1
m' = (κ × m + r) / κ'
α' = α + 0.5
β' = β + 0.5 × κ × (r - m)² / κ'
```

#### Layer 3: Thompson Sampling for Selection

Each selection round:
1. Sample μ from each trader's posterior
2. Rank by sampled value
3. Select top K

This naturally handles the explore/exploit tradeoff:
- Uncertain traders (low κ) → wide samples → sometimes selected for exploration
- Proven winners (high κ, high m) → narrow samples around high mean → usually selected

#### Decay for Non-Stationarity

Markets change. Last year's best trader may be this year's worst. We apply exponential decay toward the prior:

```python
# 34-day half-life (δ ≈ 0.98)
κ' = 1 + (κ - 1) × δ
m' = 0 + (m - 0) × δ
α' = 3 + (α - 3) × δ
β' = 1 + (β - 1) × δ
```

After ~100 days of inactivity, a trader's posterior returns to the uninformed prior.

### What This Achieves

| Metric | Before | After |
|--------|--------|-------|
| Signal definition | Every fill | Position open only |
| Signals per day | 1000s (spam) | ~10-50 (real) |
| Performance metric | Win rate | Expected R-multiple |
| Uncertainty | Ignored | Quantified via posterior |
| Adaptation | None | Exponential decay |

### Future Layers (Phase 3+)

1. **Consensus Detection**: Only act when multiple *independent* traders agree
2. **Correlation Adjustment**: effK = K / (1 + (K-1)×ρ) accounts for correlated traders
3. **EV Gating**: Only fire signals with positive expected value after costs
4. **Kelly Sizing**: Position size based on confidence and bankroll

---

## Current State (Phase 2 Complete + Refinements)

### What's Built

#### Phase 1: Foundation (Complete)
- [x] **Leaderboard Scanner** (`hl-scout`): Scans top 1000 traders, composite scoring
- [x] **BTC/ETH Filtering**: Only tracks traders profitable on majors (≥10% contribution)
- [x] **Real-time Fill Tracking** (`hl-stream`): WebSocket feeds for top N traders
- [x] **Position Tracking**: Current positions with incremental updates
- [x] **Dashboard**: Live clock, BTC/ETH prices, top performers, live fills
- [x] **Streaming Aggregation**: Smart grouping of fills within time windows
- [x] **Custom Account Tracking**: Add custom addresses to monitor (pinned accounts)
- [x] **Historical Backfill**: Load more fills from Hyperliquid API

#### Phase 2: Trader Selection Engine (Complete)
- [x] **Thompson Sampling Bandit**: NIG posterior implementation
- [x] **Position Lifecycle Tracking**: Open → Close with R-multiple calculation
- [x] **Trader Performance Table**: NIG parameters (m, κ, α, β) per trader
- [x] **Exponential Decay**: 34-day half-life toward prior
- [x] **R-Winsorization**: Bounds to ±2R for heavy tail robustness
- [x] **Dashboard UI**: Shows trader posteriors, positions, R-multiples

#### Phase 2.5: Algorithm Refinements (Complete - December 2025)
- [x] **Episode Builder** (`ts-lib/episode.ts`): VWAP entry/exit, sign-aware segmentation
- [x] **Consensus Gates** (`ts-lib/consensus.ts`): 5-gate consensus detection
- [x] **effK with Shrinkage**: `ρ' = λ×ρ_measured + (1-λ)×ρ_base` for stability
- [x] **Price Drift in R-Units**: Drift gated as fraction of stop, not raw bps
- [x] **Correlation Matrix Table**: `trader_corr` for pairwise correlations
- [x] **Ticket Instrumentation**: Full audit trail for consensus decisions
- [x] **Comprehensive Tests**: 35 episode tests + 29 consensus gate tests

#### Quant Review Fixes (December 2025)
Per external quant review, applied the following corrections:
- [x] **NIG Update Atomicity**: SQL uses CTE with `FOR UPDATE` to read old params before update
- [x] **USD Audit Fields**: Added `entry_notional_usd`, `risk_usd`, `fees_usd`, `funding_usd` for reconciliation
- [x] **Hand-Verified Tests**: 18 quant acceptance tests with exact expected values
- [x] **Flip Atomics**: Verified single-fill direction reversal emits close+open
- [x] **effK Cluster Tests**: Two-cluster scenario with ρ_intra=0.8, ρ_inter=0 → effK≈2
- [x] **EV Unit Tests**: Specific bps→R conversions (12 bps / 40 bps stop = 0.3R)

#### Phase 3a: Alpha Pool & Runtime Integration (December 2025)
Wired the algorithm components into the runtime and created the Alpha Pool UI:
- [x] **Alpha Pool Tab**: New primary dashboard tab showing NIG-selected traders
- [x] **Legacy Leaderboard Tab**: Original PnL-curve view preserved as secondary tab
- [x] **Alpha Pool API** (`/alpha-pool`): Returns 50 traders ranked by NIG posterior mean
- [x] **NIG Score Emission**: `hl-sage` emits scores with NIG posterior mean (not Thompson Sampling yet)
- [x] **Consensus Runtime**: `hl-decide` processes fills through `ConsensusDetector` (with placeholder risk)
- [x] **Consensus Signals Table**: `consensus_signals` DB table with outcome tracking
- [x] **Consensus API** (`/consensus/signals`, `/consensus/stats`): Query signals and win rates
- [x] **Self Code Review**: Verified all NIG functions, consensus integration, and file staging
- [x] **Security**: OWNER_TOKEN fails-fast in production (NODE_ENV=production)
- [x] **Alpha Pool Decoupling**: Fully separated from legacy leaderboard system
- [x] **Dashboard UI Fixes**: Fixed tracked traders panel column width truncation (TRADES header)
- [x] **HFT Detection Improvement**: Replaced VLM/AV ratio with orders-per-day via fill history analysis
- [x] **PnL Curve Caching**: 24-hour cache for Alpha Pool PnL curves to reduce API calls
- [x] **Alpha Pool Activity**: Live fills filtered to pool traders only

**⚠️ Note**: Phase 3a provides the infrastructure but with placeholder implementations. See "Known Implementation Gaps" below for details on what needs to be completed in Phase 3b.

**Alpha Pool Architecture (Decoupled)**:
The Alpha Pool is now a completely independent system from the legacy leaderboard:

| Component | Alpha Pool (New) | Legacy Leaderboard |
|-----------|------------------|-------------------|
| **Address Source** | `alpha_pool_addresses` table | `hl_leaderboard_entries` table |
| **PnL Curves** | Direct Hyperliquid API fetch | `hl_leaderboard_pnl_points` table |
| **Nicknames** | `alpha_pool_addresses.nickname` | `hl_leaderboard_entries.remark` |
| **Refresh** | `POST /alpha-pool/refresh` | hl-scout daily cron |
| **Data Source** | `stats-data.hyperliquid.xyz` | hyperbot.network API |
| **Quality Filters** | 7 quality gates (see below) | None |

**Alpha Pool Quality Filters**:
The refresh process applies 7 quality gates to filter out noise:

| Filter | Threshold | Purpose |
|--------|-----------|---------|
| `ALPHA_POOL_MIN_PNL` | $10,000 | Positive 30d PnL (remove losers) |
| `ALPHA_POOL_MIN_ROI` | 10% | Positive 30d ROI (consistent returns) |
| `ALPHA_POOL_MIN_ACCOUNT_VALUE` | $100,000 | Minimum account size |
| `ALPHA_POOL_MIN_WEEK_VLM` | $10,000 | Weekly volume (remove inactive) |
| `ALPHA_POOL_MAX_ORDERS_PER_DAY` | 100 | Orders/day (remove HFT via fill history analysis) |
| Subaccount check | N/A | Remove users with subaccounts (untrackable) |
| BTC/ETH history | N/A | Must have traded BTC or ETH (we only track these) |

**Key APIs**:
- `POST /alpha-pool/refresh` - Populate Alpha Pool with quality filtering
- `GET /alpha-pool` - Get traders with NIG posteriors and PnL curves
- `GET /alpha-pool/addresses` - List addresses in the pool
- `GET /alpha-pool/status` - NIG model statistics

### Architecture
```
hl-scout (4101) → Leaderboard scanning, scoring, BTC/ETH filtering, candidate publishing
     ↓ a.candidates.v1
hl-sage (4103)  → Score computation, weight assignment, Thompson Sampling
     ↓ b.scores.v1
hl-stream (4102) → Real-time feeds, dashboard, WebSocket
     ↓ c.fills.v1
hl-decide (4104) → Position tracking, R-multiple calculation, NIG updates
```

### Key Files

| File | Purpose |
|------|---------|
| `services/hl-sage/app/bandit.py` | NIG posterior model, Thompson Sampling, Alpha Pool selection |
| `services/hl-sage/app/main.py` | Score emission with NIG params, `/alpha-pool` API, pool refresh |
| `services/hl-decide/app/main.py` | Position lifecycle, R calculation, consensus detection |
| `services/hl-decide/app/consensus.py` | ConsensusDetector class, 5-gate consensus logic |
| `packages/ts-lib/src/episode.ts` | Episode builder with VWAP, R-multiple calculation |
| `packages/ts-lib/src/consensus.ts` | TypeScript consensus gates, effK with shrinkage |
| `services/hl-stream/public/dashboard.html` | Dashboard with Alpha Pool + Legacy tabs |
| `services/hl-stream/public/dashboard.js` | Tab switching, Alpha Pool rendering |
| `db/migrations/014_position_signals.sql` | Position tracking schema |
| `db/migrations/019_consensus_signals.sql` | Consensus signals with outcome tracking |
| `db/migrations/020_alpha_pool.sql` | Alpha Pool addresses table (decoupled) |
| `db/migrations/021_alpha_pool_roi.sql` | ROI column for quality filtering |
| `tests/episode.test.ts` | Episode builder tests (35 tests) |
| `tests/consensus-gates.test.ts` | Consensus gate tests (29 tests) |
| `tests/quant-acceptance.test.ts` | Hand-verified quant acceptance tests (18 tests) |
| `docs/POSITION_TRACKING_DESIGN.md` | Algorithm design document |

---

## Phase 2: Position-Based Trader Selection (COMPLETE)

### Implementation Summary

#### 2.1 Position Lifecycle Tracking

**What triggers a signal?**
- `Open Long (Open New)` - Position goes from 0 → positive
- `Open Short (Open New)` - Position goes from 0 → negative

**What's ignored?**
- `Increase Long/Short` - Adding to existing position
- `Decrease Long/Short` - Partial close

**What closes a signal?**
- `Close Long (Close All)` - Position returns to 0
- `Close Short (Close All)` - Position returns to 0

#### 2.2 R-Multiple Calculation

```python
# When position closes
entry_notional = entry_price × entry_size
risk_amount = entry_notional × ASSUMED_STOP_FRACTION  # 1% default

# Use Hyperliquid's realized_pnl
result_r = realized_pnl / risk_amount

# Winsorize to ±2R
result_r = max(-2.0, min(2.0, result_r))
```

#### 2.3 NIG Posterior Update

```sql
-- On position close with R-multiple r
UPDATE trader_performance SET
    nig_kappa = nig_kappa + 1,
    nig_m = (nig_kappa * nig_m + r) / (nig_kappa + 1),
    nig_alpha = nig_alpha + 0.5,
    nig_beta = nig_beta + 0.5 * nig_kappa * (r - nig_m)^2 / (nig_kappa + 1),
    positions_closed = positions_closed + 1,
    positions_won = positions_won + CASE WHEN r > 0 THEN 1 ELSE 0 END
WHERE address = $1
```

#### 2.4 Database Schema

```sql
-- Position lifecycle tracking
CREATE TABLE position_signals (
    id UUID PRIMARY KEY,
    address TEXT NOT NULL,
    asset TEXT NOT NULL,
    direction TEXT NOT NULL,  -- 'long' or 'short'

    -- Entry
    entry_fill_id TEXT NOT NULL UNIQUE,
    entry_price DOUBLE PRECISION NOT NULL,
    entry_size DOUBLE PRECISION NOT NULL,
    entry_ts TIMESTAMPTZ NOT NULL,

    -- Exit (NULL until closed)
    exit_fill_id TEXT,
    exit_price DOUBLE PRECISION,
    exit_ts TIMESTAMPTZ,
    realized_pnl DOUBLE PRECISION,
    result_r DOUBLE PRECISION,

    status TEXT DEFAULT 'open',  -- 'open', 'closed', 'expired'
    closed_reason TEXT
);

-- Trader performance with NIG parameters
ALTER TABLE trader_performance ADD COLUMN
    nig_m DOUBLE PRECISION DEFAULT 0.0,
    nig_kappa DOUBLE PRECISION DEFAULT 1.0,
    nig_alpha DOUBLE PRECISION DEFAULT 3.0,
    nig_beta DOUBLE PRECISION DEFAULT 1.0,
    positions_opened INTEGER DEFAULT 0,
    positions_closed INTEGER DEFAULT 0,
    positions_won INTEGER DEFAULT 0;
```

#### 2.5 API Endpoints

- `GET /positions/open` - Currently tracked open positions
- `GET /positions/recent` - Recently closed positions with R-multiples
- `GET /healthz` - Includes open position count

---

## Phase 3: Consensus Signal Generation

### Goal
Generate trading signals when multiple selected traders take the same position. Filter noise by requiring consensus with **correlation adjustment** and **EV gating**.

### Key Innovation: Correlation-Adjusted Effective-K with Shrinkage

Raw trader counts are misleading when traders are correlated. The **effective-K** formula:

```
effK = (Σᵢ wᵢ)² / Σᵢ Σⱼ wᵢ wⱼ ρᵢⱼ
```

**Stability via Shrinkage**: To handle noisy correlation estimates:
```
ρ' = λ × ρ_measured + (1-λ) × ρ_base
```
Where `λ = 0.7` (70% measured, 30% prior) and `ρ_base = 0.3`.

**Example**: 5 traders with 80% measured correlation → shrunk ρ ≈ 0.65 → effK ≈ 1.4

### Implementation Status (ts-lib/consensus.ts)

#### ✅ Implemented Gates

| Gate | Description | Config |
|------|-------------|--------|
| Supermajority | ≥70% agreement, ≥3 traders | `minTraders=3, minPct=0.7` |
| Effective-K | Correlation-adjusted ≥ 2.0 | `minEffectiveK=2.0` |
| Freshness | Oldest vote < 1.25× window | `maxStalenessFactor=1.25` |
| Price Drift | Drift < 0.25R from median | `maxPriceDriftR=0.25` |
| EV Gate | Net EV ≥ 0.2R after costs | `evMinR=0.2` |

#### ✅ Key Functions

```typescript
// Full consensus check with all 5 gates
checkConsensus(votes, currentMidPrice, windowMs, stopBps, correlationMatrix, traderWinRates, config)

// Individual gates
checkSupermajority(votes, config)
calculateEffectiveK(weights, correlationMatrix, config)  // With shrinkage
checkFreshness(votes, windowMs, config)
checkPriceDrift(votes, currentMidPrice, stopBps, config)  // In R-units
calculateEV(pWin, stopBps, config)
estimateWinProbability(votes, traderWinRates)  // Shrunk toward 0.5

// Ticket instrumentation for audit
createTicketInstrumentation(result, windowMs, stopBps)
```

### Completed Integration Tasks

#### 3.1 Alpha Pool UI & API
- [x] Dashboard tabs: "Alpha Pool" (default) and "Legacy Leaderboard"
- [x] `/alpha-pool` API returns NIG-ranked traders with posterior params
- [x] Alpha Pool table shows μ, κ, σ, signals, avg_r, selection status
- [x] Alpha Pool Activity feed filters fills to pool traders only

#### 3.2 Runtime Wiring
- [x] `hl-sage/main.py`: Score emission includes NIG params in meta
- [x] `hl-decide/main.py`: Fills processed through `ConsensusDetector`
- [x] Consensus signals published to `d.signals.v1` NATS topic
- [x] Consensus signals persisted to `consensus_signals` table

#### 3.3 Consensus Signal API
- [x] `/consensus/signals` - Recent signals with metrics and outcomes
- [x] `/consensus/stats` - Aggregate win rate, EV statistics

### Known Implementation Gaps (Code Review - December 2025)

The following gaps exist between the documented design and current implementation:

#### Gap 1: Selection Uses Posterior Mean, Not Thompson Sampling
**Current**: `hl-sage/main.py` (lines 248-253) ranks traders by `nig_m * side` (posterior mean).
**Designed**: Thompson Sampling should sample from posterior and rank by sampled values.
**Impact**: No exploration of uncertain traders; system exploits only proven performers.
**Fix**: Invoke `thompson_sample_select_nig()` from `bandit.py` in score emission pipeline, not just admin endpoints.

#### Gap 2: Consensus Gates Use Placeholder Risk Inputs
**Current**: `consensus.py` (lines 244-250) uses hardcoded 1% stop distance and implicit ATR percentile of 0.5.
**Designed**: Stop distance should be ATR-based, adjusting to market volatility.
**Impact**: EV gate and price drift calculations are constant regardless of market regime.
**Fix**: Add ATR feed (from `marks_1m` price history or external API) and use for dynamic stop sizing.

#### Gap 3: Correlation Matrix Not Populated
**Current**: `ConsensusDetector.correlation_matrix` is initialized empty and never populated.
**Designed**: Daily job should compute pairwise correlations and store in `trader_corr` table.
**Impact**: `eff_k` always falls back to default `ρ_base=0.3` for all pairs; correlation gate is ineffective.
**Fix**: Implement daily correlation job (Phase 3b task 3.5) and hydrate detector on startup.

#### Gap 4: ScoreEvent Weight Uses Legacy Leaderboard Value
**Current**: `hl-sage/main.py` (line 263) emits `weight=state["weight"]` from leaderboard, even when score is NIG-based.
**Designed**: Weight should reflect NIG confidence (e.g., derived from posterior variance or κ).
**Impact**: Downstream consumers may interpret weight as confidence, but it's the legacy leaderboard weight.
**Fix**: Derive weight from posterior (e.g., `1/sqrt(variance)` or `κ/(κ+10)`) or document that it's legacy.

#### Gap 5: E2E Tests Fragile and Require Manual Setup
**Current**: `playwright.config.ts` has `webServer` commented out; specs use loose selectors and don't assert backend effects.
**Designed**: Tests should be self-contained with automatic app startup and verify state changes.
**Impact**: CI/CD may fail if dashboard not pre-started; tests may pass without verifying functionality.
**Fix**: Enable webServer config, add `data-testid` selectors, assert backend responses for pin/unpin operations.

#### Gap 6: CI Runs Unit Tests Only, Not E2E
**Current**: CI workflow runs `npm run test:coverage` (Jest unit tests). Quant tests ARE included.
**Note**: E2E tests (Playwright) are NOT run in CI - they require a running dashboard.
**Impact**: UI regressions won't be caught in CI; only unit-level algorithm tests run.
**Fix**: Add Docker-based E2E stage to CI or document that E2E is manual-only.

### Remaining Integration Tasks (Phase 3b)

#### 3.4 Thompson Sampling for Candidate Selection
- [ ] Replace posterior-mean ranking with actual Thompson Sampling
- [ ] Invoke `thompson_sample_select_nig()` in hl-sage score emission pipeline (not just admin endpoints)
- [ ] Configure exploration_ratio for new trader discovery
- [ ] Add uncertainty bonus for traders with low κ (few observations)
- [ ] Derive ScoreEvent.weight from posterior (e.g., `κ/(κ+10)`) instead of legacy leaderboard weight

#### 3.5 Daily Correlation Job
- [ ] Compute 5-minute bucket sign vectors per trader
- [ ] Calculate pairwise correlation from co-occurrence
- [ ] Store in `trader_corr` table (already migrated)
- [ ] Hydrate `ConsensusDetector.correlation_matrix` on startup
- [ ] Prune entries older than 30 days

#### 3.6 Dynamic Risk Inputs
- [ ] Add ATR calculation from `marks_1m` price history
- [ ] Replace hardcoded 1% stop with ATR-based stop distance
- [ ] Pass dynamic stop_bps to consensus gates
- [ ] Consider regime-specific ATR multipliers

#### 3.7 Episode-Based Votes
- [ ] hl-decide consumes episodes from ts-lib
- [ ] One vote per trader derived from episode state, not individual fills
- [ ] Weight votes by position conviction (size/equity)

### Environment Variables
```bash
CONSENSUS_MIN_TRADERS=3
CONSENSUS_MIN_PCT=0.70
CONSENSUS_MIN_EFFECTIVE_K=2.0
CONSENSUS_BASE_WINDOW_S=120
CONSENSUS_MAX_STALENESS_FACTOR=1.25
CONSENSUS_MAX_PRICE_DRIFT_R=0.25      # Now in R-units, not bps
CONSENSUS_EV_MIN_R=0.20
CONSENSUS_DEFAULT_CORRELATION=0.3     # ρ_base for shrinkage
CONSENSUS_CORRELATION_SHRINKAGE=0.7   # λ (70% measured, 30% prior)
```

---

## Phase 4: Risk Management & Position Sizing

### Goal
Implement Kelly criterion position sizing with practical guardrails.

### Tasks

#### 4.1 Kelly Calculator
```python
def kelly_fraction(win_rate, avg_win_r, avg_loss_r):
    R = avg_win_r / avg_loss_r
    kelly = win_rate - (1 - win_rate) / R
    return max(0, min(kelly, 1))

def position_size(kelly, fraction=0.25, account_value):
    return kelly * fraction * account_value
```

#### 4.2 Risk Limits
- Max position size: 5% of account
- Max total exposure: 20%
- Max concurrent signals: 3
- Daily drawdown limit: -10%

---

## Phase 5: Market Regime Detection

### Goal
Detect market regime (trending, ranging, volatile) and adjust strategy parameters.

### Simple Regime Detection
```python
def detect_regime(prices, atr):
    ma20, ma50 = np.mean(prices[-20:]), np.mean(prices[-50:])
    trend = (ma20 - ma50) / ma50
    volatility = atr / prices[-1]

    if abs(trend) > 0.02 and volatility < 0.03: return "trending"
    elif abs(trend) < 0.01 and volatility < 0.02: return "ranging"
    else: return "volatile"
```

### Regime-Specific Parameters
| Regime | Consensus Min | Kelly Fraction | Stop ATR Mult |
|--------|--------------|----------------|---------------|
| Trending | 2 | 0.35 | 2.0 |
| Ranging | 4 | 0.15 | 1.0 |
| Volatile | 5 | 0.10 | 1.5 |

---

## Phase 6-7: Hyperliquid Integration & Execution

(See original plan - unchanged)

---

## Configuration Reference

### Current Environment Variables
```bash
# Existing
OWNER_TOKEN=dev-owner
NATS_URL=nats://nats:4222
DATABASE_URL=postgresql://hlbot:hlbotpassword@postgres:5432/hlbot

# Phase 2: Position Tracking
ASSUMED_STOP_FRACTION=0.01           # 1% stop for R calculation

# Phase 2: Bandit (Thompson Sampling)
BANDIT_POOL_SIZE=50
BANDIT_SELECT_K=10
BANDIT_MIN_SAMPLES=30
BANDIT_DECAY_HALF_LIFE_DAYS=34       # δ ≈ 0.98

# NIG Prior
NIG_PRIOR_M=0.0
NIG_PRIOR_KAPPA=1.0
NIG_PRIOR_ALPHA=3.0
NIG_PRIOR_BETA=1.0
R_WINSORIZE_MIN=-2.0
R_WINSORIZE_MAX=2.0

# Phase 3: Consensus
CONSENSUS_MIN_TRADERS=3
CONSENSUS_MIN_PCT=0.70
CONSENSUS_MIN_EFFECTIVE_K=2.0
CONSENSUS_EV_MIN_R=0.20

# Alpha Pool Quality Filters
ALPHA_POOL_MIN_PNL=10000             # Min $10k 30d PnL
ALPHA_POOL_MIN_ROI=0.10              # Min 10% 30d ROI
ALPHA_POOL_MIN_ACCOUNT_VALUE=100000  # Min $100k account value
ALPHA_POOL_MIN_WEEK_VLM=10000        # Min $10k weekly volume (filter inactive)
ALPHA_POOL_MAX_ORDERS_PER_DAY=100    # Max 100 orders/day (filter HFT via fill history)
```

---

## Milestones

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Foundation (leaderboard, streaming, dashboard) | ✅ Complete |
| 2 | Trader Selection (position-based NIG bandit) | ✅ Complete |
| 2.5 | Algorithm Refinements (episode builder, consensus gates) | ✅ Complete |
| 3a | Alpha Pool UI & Runtime Wiring | ✅ Complete |
| 3b | Correlation Job & Episode Votes | 🔶 In Progress |
| 4 | Risk Management (Kelly criterion) | 🔲 Not started |
| 5 | Market Regime Detection | 🔲 Not started |
| 6 | Hyperliquid Read-Only Integration | 🔲 Not started |
| 7 | Automated Execution | 🔲 Not started |

---

## Validation Checklist

### Phase 2 Tests (Position-Based Tracking)
- [x] Position open creates signal with entry info
- [x] Position close calculates R-multiple from realized_pnl
- [x] NIG posterior updates correctly on close
- [x] Increase/Decrease fills are ignored
- [x] Duplicate fill_ids are rejected (unique constraint)
- [x] Data reset clears corrupted fill-level spam

### Phase 2.5 Tests (Algorithm Refinements)
- [x] Episode builder: VWAP calculation for entries/exits
- [x] Episode builder: Sign-aware position segmentation
- [x] Episode builder: Direction flip handling (close + reopen)
- [x] Episode builder: R-multiple with winsorization to ±2R
- [x] Episode builder: Partial close tracking
- [x] Consensus: Supermajority gate (≥70%, ≥3 traders)
- [x] Consensus: Effective-K with shrinkage stability
- [x] Consensus: Freshness gate (staleness factor)
- [x] Consensus: Price drift in R-units (not raw bps)
- [x] Consensus: EV gate with cost conversion to R
- [x] Consensus: Win probability estimation with shrinkage

### Quant Acceptance Tests (December 2025)
Per quant review, added 18 hand-verified acceptance tests in `tests/quant-acceptance.test.ts`:
- [x] R audit: +$1000 on $50k entry with 1% stop → R=2.0
- [x] R audit: -$2000 on $80k short → R=-2.0 (clamped)
- [x] R audit: Small winner with fees → R=0.47
- [x] R audit: VWAP across multiple entry fills
- [x] Flip atomics: Single fill reverses sign → close + open
- [x] Flip atomics: Short-to-long direction flip
- [x] Consensus dedupe: 100 micro-fills = 1 vote (effK=1)
- [x] effK extreme: 5 perfectly correlated traders → effK≈1
- [x] effK extreme: Two uncorrelated clusters → effK≈2
- [x] effK: 4 traders with ρ=0.5 → effK=1.6
- [x] EV units: 12 bps cost / 40 bps stop = 0.3R
- [x] EV units: 17 bps cost / 100 bps stop = 0.17R
- [x] EV calculation with specific p_win and costs
- [x] EV: Tighter stop doubles cost impact
- [x] EV gate rejects when net EV < threshold
- [x] Stop basis consistency between entry and exit
- [x] Stop basis recorded at entry for audit

### Upcoming Tests (Phase 3 Integration)
- [ ] One vote per trader derived from episode state
- [ ] Daily correlation job computes pairwise ρ
- [ ] End-to-end consensus → ticket → outcome flow

### Code Review Verification (December 2025)

Self code review verified the following runtime integrations:

| Component | File | Function/Class | Status |
|-----------|------|----------------|--------|
| NIG Posterior | `bandit.py:503` | `get_trader_posteriors_nig()` | ✅ Verified |
| NIG Selection | `bandit.py:556` | `thompson_sample_select_nig()` | ✅ Verified |
| NIG Status | `bandit.py:664` | `get_bandit_status_nig()` | ✅ Verified |
| NIG Update | `bandit.py:589` | `update_trader_nig()` | ✅ Verified |
| Consensus Detector | `consensus.py:124` | `ConsensusDetector` class | ✅ Verified |
| 5-Gate Check | `consensus.py:192` | `check_consensus()` | ✅ Verified |
| Score Emission | `main.py:221-250` | NIG params in `ScoreEvent.meta` | ✅ Verified |
| Fill Processing | `main.py:510-543` | `process_fill_for_consensus()` | ✅ Verified |

**Test Results**:
- Jest: 955 tests passing (includes 8 new HFT filter tests)
- Playwright E2E: 128 tests passing (6 skipped)

**Files Staged**: 14 new files + modified services ready for commit.

---

## How to Resume Development

1. **Check this document** for current phase
2. **Review phase tasks** - pick next uncompleted item
3. **Run services**: `docker compose up -d`
4. **Dashboard**: http://localhost:4102/dashboard (Alpha Pool tab is default)
5. **Logs**: `docker compose logs -f [service-name]`
6. **Run tests**: `npm test` (955 unit tests); E2E requires dashboard running first

### Key Files by Phase

**Phase 2 (Complete)**:
- `services/hl-sage/app/bandit.py` - NIG model
- `services/hl-decide/app/main.py` - Position tracking
- `db/migrations/014_position_signals.sql` - Schema

**Phase 2.5 (Complete)**:
- `packages/ts-lib/src/episode.ts` - Episode builder (VWAP, R-multiple)
- `packages/ts-lib/src/consensus.ts` - 5-gate consensus detection
- `services/hl-sage/app/bandit.py` - NIG update with CTE for atomicity
- `tests/episode.test.ts` - 35 episode tests
- `tests/consensus-gates.test.ts` - 29 consensus gate tests
- `tests/quant-acceptance.test.ts` - 18 hand-verified quant acceptance tests
- `db/migrations/015_episode_fields.sql` - VWAP/R audit columns
- `db/migrations/016_trader_correlation.sql` - Correlation matrix table
- `db/migrations/017_consensus_instrumentation.sql` - Ticket audit columns
- `db/migrations/018_episode_usd_fields.sql` - USD-denominated audit fields

**Phase 3a (Complete - Alpha Pool Runtime)**:
- `services/hl-sage/app/main.py` - `/alpha-pool` endpoint, NIG in score emission
- `services/hl-sage/app/bandit.py` - NIG functions: `get_trader_posteriors_nig`, `thompson_sample_select_nig`, `get_bandit_status_nig`
- `services/hl-decide/app/main.py` - Consensus detection via `process_fill_for_consensus()`
- `services/hl-decide/app/consensus.py` - ConsensusDetector class with 5-gate logic
- `services/hl-stream/public/dashboard.html` - Tabs (Alpha Pool + Legacy)
- `services/hl-stream/public/dashboard.js` - Tab logic, Alpha Pool rendering
- `services/hl-stream/src/index.ts` - Proxy routes for `/alpha-pool`, `/consensus`
- `db/migrations/019_consensus_signals.sql` - Consensus signals table

**Phase 3b (In Progress - Remaining)**:
- Replace posterior-mean selection with Thompson Sampling in score emission
- Derive ScoreEvent.weight from NIG posterior (not legacy leaderboard weight)
- Implement daily correlation computation job
- Hydrate ConsensusDetector.correlation_matrix from trader_corr table
- Replace hardcoded 1% stop with ATR-based dynamic stops
- Wire episode builder to hl-decide for episode-based votes
- Weight votes by position conviction
- Improve E2E test reliability (data-testid selectors, backend assertions)

---

*Last updated: December 2025*
