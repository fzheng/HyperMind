# SigmaPilot Operational Runbook

This document provides operational procedures for managing and monitoring the SigmaPilot trading system.

## System Overview

SigmaPilot is a consensus-based trading signal system that:
1. **hl-scout** - Ingests Hyperliquid leaderboard data, scores addresses
2. **hl-stream** - Subscribes to realtime fills, serves dashboard
3. **hl-sage** - Thompson Sampling trader selection, NIG posteriors
4. **hl-decide** - Consensus detection, episode tracking, signal generation

## Daily Operations

### Morning Health Check

```bash
# Check all services are running
docker compose ps

# Check recent consensus signals
curl -s http://localhost:4104/consensus/signals?limit=5 | jq .

# Check Alpha Pool status
curl -s http://localhost:4103/alpha-pool/status | jq .

# Verify NIG model statistics
curl -s http://localhost:4103/alpha-pool/status | jq '.nig_stats'
```

### Monitoring Endpoints

| Service | Port | Health | Metrics | Docs |
|---------|------|--------|---------|------|
| hl-scout | 4101 | `/healthz` | `/metrics` | `/docs` |
| hl-stream | 4102 | `/healthz` | `/metrics` | `/docs` |
| hl-sage | 4103 | `/healthz` | `/metrics` | `/docs` |
| hl-decide | 4104 | `/healthz` | `/metrics` | `/docs` |

### Key Metrics to Monitor

1. **Consensus Signals**
   - Signal rate (should see signals during market hours)
   - Win rate (target >50%)
   - Average R-multiple

2. **Alpha Pool**
   - Number of active traders
   - Average κ (confidence) - higher is better
   - Correlation distribution

3. **ATR/Volatility**
   - Current stop fractions (BTC, ETH)
   - ATR percentile

## Phase 3b Components

### Thompson Sampling (hl-sage)

**Purpose**: Balances exploration (trying uncertain traders) vs exploitation (using proven performers).

**How it works**:
- Samples from NIG posterior instead of using posterior mean
- Low-κ traders get wider samples (exploration)
- High-κ traders get tighter samples (exploitation)

**Weight Formula**: `weight = κ / (κ + 10)`
- κ=1: weight ~0.09 (new trader)
- κ=10: weight ~0.5 (moderate experience)
- κ=100: weight ~0.91 (proven performer)

**Monitoring**:
```bash
# Check recent scores and their sources
curl -s http://localhost:4103/scores | jq 'map({address, score, source: .meta.source})'
```

### ATR-Based Dynamic Stops (hl-decide)

**Purpose**: Adapts stop distances based on current market volatility.

**Configuration**:
```bash
# Environment variables
ATR_PERIOD=14           # Candles for ATR calculation
ATR_MULTIPLIER_BTC=2.0  # Stop = ATR × multiplier
ATR_MULTIPLIER_ETH=1.5
ATR_FALLBACK_PCT=1.0    # Used if ATR unavailable
```

**Stop Distance Bounds**:
- Minimum: 0.1% (prevents overly tight stops)
- Maximum: 10% (prevents excessively wide stops)

**Checking Current ATR**:
```bash
# Via database
docker compose exec postgres psql -U hlbot -d hlbot -c \
  "SELECT asset, atr14, mid, ts FROM marks_1m WHERE atr14 IS NOT NULL ORDER BY ts DESC LIMIT 2;"
```

### Correlation Job (hl-decide)

**Purpose**: Computes trader correlations for effective-K calculation.

**How it works**:
1. Builds 5-minute bucket sign vectors per trader
2. Computes phi correlation on co-occurring buckets
3. Clips negative correlation to 0 (treat anti-correlated as independent)
4. Stores in `trader_corr` table

**Effective-K Formula**:
```
effK = (Σwᵢ)² / ΣΣwᵢwⱼρᵢⱼ
```
- Independent traders (ρ=0): effK = n
- Perfectly correlated (ρ=1): effK = 1

**Manual Correlation Computation**:
```bash
# Trigger daily correlation job
curl -X POST http://localhost:4104/correlation/compute

# Check correlation status
curl -s http://localhost:4104/correlation/status | jq .
```

**Database Query**:
```sql
-- Latest correlations
SELECT addr_a, addr_b, rho, n_buckets
FROM trader_corr
WHERE as_of_date = (SELECT MAX(as_of_date) FROM trader_corr)
ORDER BY rho DESC LIMIT 10;
```

### Episode Tracking (hl-decide)

**Purpose**: Tracks position lifecycles and calculates R-multiples.

**R-Multiple Calculation**:
```
R = PnL / Risk
Risk = Entry Price × Stop Fraction × Size
```

**Episode States**:
- `open`: Position is active
- `closed`: Position fully closed
- `closed_reason`: `full_close`, `direction_flip`, `timeout`

**Database Query**:
```sql
-- Recent episodes with R-multiples
SELECT id, address, asset, direction,
       entry_vwap, exit_vwap, result_r, status
FROM position_signals
WHERE status = 'closed'
ORDER BY exit_ts DESC LIMIT 10;
```

## Troubleshooting

### No Consensus Signals

1. Check if Alpha Pool has traders:
   ```bash
   curl -s http://localhost:4103/alpha-pool | jq '.traders | length'
   ```

2. Verify fills are being received:
   ```bash
   curl -s http://localhost:4102/recent-fills | jq '. | length'
   ```

3. Check consensus detector state:
   ```bash
   curl -s http://localhost:4104/consensus/stats | jq .
   ```

### Low Win Rate

1. Check if correlation data is current:
   ```sql
   SELECT as_of_date, COUNT(*) as pairs
   FROM trader_corr
   GROUP BY as_of_date ORDER BY as_of_date DESC LIMIT 5;
   ```

2. Verify ATR is being calculated (not fallback):
   ```bash
   # Check marks_1m table for atr14 values
   docker compose exec postgres psql -U hlbot -d hlbot -c \
     "SELECT COUNT(*) as rows_with_atr FROM marks_1m WHERE atr14 IS NOT NULL;"
   ```

3. Review effK values in recent signals - if all ~1.0, correlation may not be working:
   ```sql
   SELECT id, eff_k, n_traders, n_agreeing
   FROM consensus_signals ORDER BY created_at DESC LIMIT 10;
   ```

### Service Recovery

```bash
# Restart a single service
docker compose restart hl-decide

# Full restart
docker compose down && docker compose up -d

# Check logs for errors
docker compose logs -f hl-decide --since=5m

# Force Alpha Pool refresh
curl -X POST http://localhost:4103/alpha-pool/refresh?limit=50
```

### Database Issues

```bash
# Check migration status
npm run migrate:status

# Manual migration if needed
npm run migrate

# Verify table exists
docker compose exec postgres psql -U hlbot -d hlbot -c "\dt trader_corr"
```

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CORR_BUCKET_MINUTES` | 5 | Time bucket size for correlation |
| `CORR_LOOKBACK_DAYS` | 30 | Days of history for correlation |
| `CORR_MIN_COMMON_BUCKETS` | 10 | Min overlapping buckets |
| `ATR_PERIOD` | 14 | ATR calculation period |
| `ATR_MULTIPLIER_BTC` | 2.0 | BTC stop multiplier |
| `ATR_MULTIPLIER_ETH` | 1.5 | ETH stop multiplier |
| `ATR_FALLBACK_PCT` | 1.0 | Fallback if no ATR data |
| `CONSENSUS_MIN_TRADERS` | 3 | Min traders for consensus |
| `CONSENSUS_MIN_PCT` | 0.67 | Supermajority threshold |

### Database Tables

| Table | Purpose |
|-------|---------|
| `trader_corr` | Pairwise trader correlations |
| `position_signals` | Episode lifecycle data |
| `consensus_signals` | Generated consensus signals |
| `trader_performance` | NIG posterior parameters |
| `marks_1m` | 1-minute candle data with ATR |
| `alpha_pool_addresses` | Active Alpha Pool traders |

## Scheduled Jobs

| Job | Frequency | Service | Trigger |
|-----|-----------|---------|---------|
| Leaderboard refresh | Daily 00:30 UTC | hl-scout | Cron |
| Correlation computation | Daily | hl-decide | Manual/API |
| ATR refresh | On new candle | hl-decide | Automatic |
| NIG update | On episode close | hl-decide | Automatic |

## Performance Expectations

### Normal Operation

- Alpha Pool: 30-50 active traders
- Correlation pairs: ~1000-2000 (n×(n-1)/2)
- effK range: 1.5-10 (depending on correlation)
- Win rate: 50-60%
- Average R: 0.3-0.8

### Warning Signs

- effK always ~1.0: Correlation not populated
- Win rate <40%: Check signal quality
- No signals for >4 hours: Check fills pipeline
- ATR source = "fallback": Missing candle data

---

## Phase 4/5 Components

### Kelly Criterion Position Sizing (hl-decide)

**Purpose**: Data-driven position sizing based on trader performance.

**Formula**: `f* = p - (1-p)/R` where p=win rate, R=avg_win/avg_loss

**Configuration**:
```bash
KELLY_ENABLED=false          # Enable Kelly sizing
KELLY_FRACTION=0.25          # Fractional Kelly (quarter Kelly)
KELLY_MIN_EPISODES=30        # Minimum episodes for Kelly calc
KELLY_FALLBACK_PCT=0.01      # Fallback 1% if Kelly fails
```

**Checking Kelly Status**:
```bash
# Check execution logs for Kelly sizing
docker compose exec postgres psql -U hlbot -d hlbot -c \
  "SELECT decision_id, kelly_method, kelly_position_pct FROM execution_logs ORDER BY created_at DESC LIMIT 5;"
```

### Market Regime Detection (hl-decide)

**Purpose**: Adapts strategy parameters based on market conditions.

**Regime Types**:
| Regime | Detection | Strategy |
|--------|-----------|----------|
| TRENDING | MA spread > 2% | Wider stops, full Kelly |
| RANGING | MAs converged | Tighter stops, 75% Kelly |
| VOLATILE | ATR > 1.5x avg | Wide stops, 50% Kelly |

**Checking Current Regime**:
```bash
# Via API
curl -s http://localhost:4104/regime | jq .

# Via dashboard
# Market Regime card shows BTC/ETH regime status
```

**Configuration**:
```bash
REGIME_LOOKBACK_MINUTES=60
REGIME_TREND_THRESHOLD=0.02
REGIME_VOLATILITY_HIGH_MULT=1.5
```

### Real Execution Safety (hl-decide)

**Purpose**: Guard rails for real trading (when enabled).

**Safety Gates**:
1. `REAL_EXECUTION_ENABLED=false` (env var)
2. `hl_enabled=false` in execution_config (database)
3. Max position limits (10% default)
4. Daily drawdown halt (5% default)
5. Kill switch cooldown (24h)

**Checking Execution Status**:
```bash
# Check if execution is enabled
docker compose exec postgres psql -U hlbot -d hlbot -c \
  "SELECT enabled, hyperliquid FROM execution_config ORDER BY updated_at DESC LIMIT 1;"

# Check recent execution attempts
curl -s http://localhost:4104/execution/logs?limit=5 | jq .
```
