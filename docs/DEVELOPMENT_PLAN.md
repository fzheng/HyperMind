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
| **4** | **Risk Management (Kelly criterion)** | 🔲 **Next** |
| 5 | Market Regime Detection | 🔲 Planned |
| 6 | Hyperliquid Read-Only Integration | 🔲 Planned |
| 7 | Automated Execution | 🔲 Planned |

---

## Current State: Phase 3 Complete

### What's Working

**Core Algorithm Pipeline:**
```
Leaderboard → Quality Filter → Alpha Pool → Thompson Sampling → Consensus → Signal
   1000+          7 gates         50 traders      NIG posterior       5 gates
```

**Services:**
| Service | Port | Function |
|---------|------|----------|
| hl-scout | 4101 | Leaderboard scanning, candidate publishing |
| hl-stream | 4102 | Real-time feeds, dashboard, WebSocket |
| hl-sage | 4103 | NIG model, Thompson Sampling selection |
| hl-decide | 4104 | Consensus detection, episode tracking |

**Test Coverage:**
- TypeScript: 973 unit tests
- Python: 164 tests (151 + 13 risk limits)
- E2E: 150 Playwright tests

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

---

## Phase 4: Risk Management (Next)

### Goal
Implement Kelly criterion position sizing with practical guardrails.

### Tasks

#### 4.1 Kelly Calculator
```python
def kelly_fraction(win_rate, avg_win_r, avg_loss_r):
    """Calculate optimal bet fraction using Kelly criterion."""
    R = avg_win_r / avg_loss_r
    kelly = win_rate - (1 - win_rate) / R
    return max(0, min(kelly, 1))

def position_size(kelly, account_value, fraction=0.25):
    """Apply fractional Kelly for conservative sizing."""
    return kelly * fraction * account_value
```

#### 4.2 Risk Limits (Enforce)
- [ ] Max position size: 5% of account
- [ ] Max total exposure: 20%
- [ ] Max concurrent positions: 3
- [ ] Daily drawdown limit: -10%
- [ ] Implement real-time exposure tracking

#### 4.3 Paper Trading Mode
- [ ] Simulation mode to validate signals
- [ ] Track simulated P&L without execution
- [ ] Compare signals vs actual market moves

#### 4.4 Signal Cooldown
- [ ] Implement per-symbol cooldown (already configured: 300s)
- [ ] Track last signal time per asset
- [ ] Reject rapid-fire signals

#### 4.5 E2E Test Hardening
- [ ] Add `data-testid` selectors to dashboard
- [ ] Enable `webServer` in Playwright config
- [ ] Assert backend state changes (not just UI)

---

## Phase 5: Market Regime Detection

### Goal
Adapt strategy parameters based on market conditions.

### Regime Types
| Regime | Detection | Response |
|--------|-----------|----------|
| Trending | MA20 > MA50 + 2%, low vol | Wider stops, higher Kelly |
| Ranging | MAs converged, low vol | Tighter stops, lower Kelly |
| Volatile | High ATR/price ratio | Conservative sizing |

### Tasks
- [ ] Implement regime classifier
- [ ] Add regime-specific parameter sets
- [ ] Auto-adjust consensus thresholds
- [ ] Backtest regime transitions

---

## Phase 6-7: Execution

### Phase 6: Read-Only Integration
- [ ] Connect to Hyperliquid API (read-only)
- [ ] Track account positions
- [ ] Monitor fills for paper trading

### Phase 7: Automated Execution
- [ ] Implement order placement
- [ ] Position management (stop-loss, take-profit)
- [ ] Risk circuit breakers

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
| Consensus Detection | `services/hl-decide/app/consensus.py` |
| ATR Provider | `services/hl-decide/app/atr.py` |
| Correlation | `services/hl-decide/app/correlation.py` |
| Episode Tracker | `services/hl-decide/app/episode.py` |
| Dashboard | `services/hl-stream/public/dashboard.html` |

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

*Last updated: December 2025*
