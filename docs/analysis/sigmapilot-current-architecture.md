# SigmaPilot Current Architecture Analysis

*Phase 3e Analysis - December 2025*

## Executive Summary

SigmaPilot is a collective intelligence trading system for Hyperliquid that aggregates wisdom from top traders to generate consensus-based signals. Unlike copy-trading platforms, it uses Bayesian learning (Thompson Sampling with NIG posteriors) to adaptively select traders and requires multi-party agreement before generating signals.

---

## 1. Service Architecture

### Overview

Four microservices communicate via NATS message bus:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  hl-scout   │────▶│  hl-sage    │────▶│  hl-stream  │────▶│  hl-decide  │
│   :4101     │     │   :4103     │     │   :4102     │     │   :4104     │
├─────────────┤     ├─────────────┤     ├─────────────┤     ├─────────────┤
│ Leaderboard │     │ NIG Model   │     │ Dashboard   │     │ Consensus   │
│ Scanning    │     │ Thompson    │     │ WebSocket   │     │ Detection   │
│ Filtering   │     │ Sampling    │     │ Fills       │     │ Episodes    │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

### Service Details

| Service | Language | Port | Primary Responsibilities |
|---------|----------|------|--------------------------|
| **hl-scout** | TypeScript | 4101 | Leaderboard ingestion, trader scoring, pinned accounts, migrations |
| **hl-stream** | TypeScript | 4102 | WebSocket streaming, dashboard UI, subscription management, real-time fills |
| **hl-sage** | Python/FastAPI | 4103 | Thompson Sampling, NIG posteriors, Alpha Pool management |
| **hl-decide** | Python/FastAPI | 4104 | Consensus detection, episode tracking, signal generation |

---

## 2. Front-End Stack

### Technology
- **Framework**: Vanilla JavaScript SPA (no React/Vue/Angular)
- **Location**: `services/hl-stream/public/`
- **Files**:
  - `dashboard.html` - HTML with semantic markup, `data-testid` attributes
  - `dashboard.js` - 3000+ lines event-driven JavaScript
  - `dashboard.css` - Mobile-first responsive CSS

### Real-Time Updates
- WebSocket connection to `/ws` for live fills
- Price ticker polling (5-sec interval)
- Theme persistence (dark/light/auto)
- Infinite scroll with historical backfill

### Current Dashboard Sections

**Alpha Pool Tab (Default):**
1. Consensus Signals - Real-time signals when ≥3 traders agree
2. Alpha Pool Traders - 50 NIG-selected traders with μ, κ, σ, avg_r
3. Alpha Pool Activity - Live fills filtered to pool addresses

**Legacy Tab:**
1. Leaderboard with BTC/ETH holdings
2. Pinned accounts (blue=leaderboard, gold=custom)
3. Historical fills with pagination

---

## 3. Hyperliquid Integration

### API Wrappers (`@hl/ts-lib/hyperliquid.ts`)

```typescript
// Read Operations
fetchUserFills(address)        // Historical fills (BTC/ETH)
fetchUserProfile(address)      // Account balance, leverage, holdings
fetchPerpPositions(address)    // Current open positions
fetchPerpMarkPrice(symbol)     // Current mark price

// Leaderboard
fetchLeaderboard(period)       // Top traders by period
fetchLeaderboardStats(address) // Detailed trader stats
```

### WebSocket Subscriptions (`realtime.ts`)

```typescript
// RealtimeTracker manages:
- user_fills endpoint       // Live trade fills
- user_positions endpoint   // Position changes
- Position priming          // Fetch current state on subscribe
- Auto-reconnect            // Exponential backoff
```

### Rate Limiting
- 2 calls/second default (`HL_SDK_CALLS_PER_SECOND`)
- Exponential backoff on 429 errors
- Configurable via environment

---

## 4. Message Bus (NATS with JetStream)

### Topic Contracts

| Topic | Schema | Flow |
|-------|--------|------|
| `a.candidates.v1` | CandidateEvent | scout → sage |
| `b.scores.v1` | ScoreEvent | sage → decide |
| `c.fills.v1` | FillEvent | stream → decide |
| `d.signals.v1` | SignalEvent | decide internal |
| `d.outcomes.v1` | OutcomeEvent | decide → persistence |

### Schema Location
- `contracts/jsonschema/*.json` - JSON Schema definitions
- Auto-generated TypeScript/Python models

---

## 5. Signal Generation Pipeline

### Flow

```
Hyperliquid Leaderboard
        │
        ▼
┌─────────────────────┐
│  7 Quality Filters  │  Min PnL $10k, ROI 10%, AV $100k,
│  (Alpha Pool Gates) │  Weekly volume, HFT detection,
│                     │  No subaccounts, BTC/ETH history
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Thompson Sampling  │  NIG posteriors: μ|σ² ~ N(m, σ²/κ)
│  (50 traders)       │  Sample from posterior for selection
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Position Tracking  │  Episode lifecycle: open → close
│  (R-multiple calc)  │  VWAP entry/exit prices
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  5 Consensus Gates  │  1. Supermajority (70%, 3+ traders)
│                     │  2. Effective-K ≥ 2.0
│                     │  3. Freshness (within window)
│                     │  4. Price band (<0.25R drift)
│                     │  5. EV gate (≥0.2R net)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Risk Fail-Safes    │  Max 2% position, 10% exposure,
│                     │  5% daily loss, 55% confidence,
│                     │  1x leverage, 5min cooldown
└─────────────────────┘
        │
        ▼
    SIGNAL OUTPUT
```

---

## 6. Current Logging & Observability

### Prometheus Metrics (exposed at `/metrics`)

**hl-decide metrics:**
- `decide_atr_stale_total` - ATR data freshness
- `decide_atr_fallback_total` - Fallback usage
- `decide_correlation_stale` - Correlation freshness
- `decide_effk_value` - Effective-K distribution
- `decide_vote_weight_gini` - Weight concentration
- `decide_signal_generated_total` - Signals passing gates
- `decide_signal_risk_rejected_total` - Risk rejections

### Console Logging

All services use `[service-name]` prefixed stdout logging:

```
[hl-decide] Position opened: 0xabc... long BTC @ 45000.50
[hl-decide] Consensus detected: 5 traders, effK=3.2, EV=0.45R
[hl-decide] CONSENSUS SIGNAL: BTC long, confidence=68%, EV=0.38R
[hl-sage] Alpha Pool refreshed: 50 traders, 42 fills synced
```

### Gaps
- No centralized log aggregation (ELK, Datadog)
- No distributed tracing
- Decision logs not persisted in queryable format
- No chain-of-thought reasoning capture

---

## 7. Database Schema

### Core Tables

| Table | Purpose |
|-------|---------|
| `addresses` | Tracked wallet addresses |
| `hl_events` | Trade history (deduped by hash) |
| `hl_current_positions` | Real-time position snapshots |
| `marks_1m` | 1-minute OHLC for ATR |
| `alpha_pool_addresses` | Decoupled Alpha Pool |
| `trader_performance` | NIG posteriors (μ, κ, σ, α, β) |
| `trader_correlation` | Pairwise correlations |
| `consensus_signals` | Signal records with gates |
| `tickets` / `ticket_outcomes` | Signal tickets and outcomes |

---

## 8. Shared Library (`@hl/ts-lib`)

Key modules:
- `nats.ts` - NATS connection helpers
- `postgres.ts` - Singleton pool management
- `hyperliquid.ts` - HL API wrappers
- `realtime.ts` - WebSocket tracker
- `subscription-manager.ts` - Slot allocation
- `consensus.ts` - 5-gate detection
- `episode.ts` - Position lifecycle
- `persist.ts` - Database operations
- `scoring.ts` - Performance metrics
- `metrics.ts` - Prometheus helpers

---

## 9. Current Limitations

1. **Single Exchange**: Hyperliquid only, no multi-exchange support
2. **No Execution**: Signal generation only, no auto-trading
3. **Limited Decision Logging**: Console logs only, not queryable
4. **No Chain-of-Thought**: No human-readable reasoning capture
5. **Basic Dashboard**: Vanilla JS, limited interactivity
6. **No P&L Tracking**: Per-position P&L not displayed in UI
7. **No Strategy Configuration**: Hardcoded parameters, no UI config

---

## 10. Strengths

1. **Robust Signal Pipeline**: 7 quality filters + 5 consensus gates
2. **Bayesian Learning**: Thompson Sampling adapts over time
3. **Risk-First Design**: Hard caps before any trading logic
4. **Event-Driven**: Clean separation via NATS
5. **Comprehensive Tests**: 1000+ unit tests, E2E coverage
6. **Real-Time Updates**: WebSocket for live fills
7. **Mobile-First UI**: Responsive design

---

*This document serves as the baseline for Phase 3e improvements.*
