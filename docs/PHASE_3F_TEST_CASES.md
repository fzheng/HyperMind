# Phase 3f: Selection Integrity - E2E Test Cases

This document describes end-to-end test scenarios for Phase 3f features. Use these to manually verify the system works correctly.

## Prerequisites

```bash
# Start all services
docker compose up -d

# Verify services are healthy
curl http://localhost:4103/healthz  # hl-sage
curl http://localhost:4104/healthz  # hl-decide

# Check database has migrations applied
docker compose exec postgres psql -U hlbot -d hlbot -c "\dt trader_snapshots"
docker compose exec postgres psql -U hlbot -d hlbot -c "\dt risk_governor_state"
```

---

## 0. Fresh Install: Initialize Alpha Pool

After a fresh `docker compose up -d`, the Alpha Pool will be automatically initialized.

### Automatic Initialization (Default)

When hl-sage starts and detects an empty Alpha Pool (fresh database), it automatically:
1. Refreshes the pool from Hyperliquid leaderboard (50 traders)
2. Backfills historical fills for all new addresses
3. Creates an initial snapshot for FDR qualification

**Watch the logs to see initialization progress:**
```bash
docker compose logs -f hl-sage
```

You'll see:
```
[hl-sage] Fresh install detected: Alpha Pool is empty
[hl-sage] Starting automatic initialization...
[hl-sage] [1/2] Refreshing Alpha Pool from leaderboard...
[hl-sage] [2/2] Creating initial snapshot for FDR qualification...
[hl-sage] Initial snapshot created: 3/53 traders FDR-qualified
[hl-sage] Automatic initialization complete!
```

### Manual Initialization (Optional)

If you prefer to run initialization manually or with custom options:

#### Option 1: NPM Script (Cross-platform)

```bash
npm run init:alpha-pool

# With custom options
node scripts/init-alpha-pool.mjs --limit 100 --delay 1000
```

#### Option 2: Bash Script (Linux/Mac/Git Bash)

```bash
./scripts/init-alpha-pool.sh
./scripts/init-alpha-pool.sh --limit 100    # More traders from leaderboard
./scripts/init-alpha-pool.sh --delay 1000   # Slower backfill (safer rate limit)
```

#### Option 3: Manual API Calls

```bash
# Step 1: Refresh pool from leaderboard (auto-backfills new addresses)
curl -X POST "http://localhost:4103/alpha-pool/refresh?limit=50"

# Step 2: Backfill all addresses (in case any were missed)
curl -X POST "http://localhost:4103/alpha-pool/backfill-all?delay_ms=500"

# Step 3: Create initial snapshot
curl -X POST "http://localhost:4103/snapshots/create"
```

### Verify Initialization

```bash
# Check pool status
curl -s "http://localhost:4103/alpha-pool/status"

# Check how many traders have enough episodes
docker compose exec postgres psql -U hlbot -d hlbot -c "
  SELECT
    CASE
      WHEN nig_kappa - 1 >= 30 THEN '30+ episodes'
      WHEN nig_kappa - 1 >= 10 THEN '10-29 episodes'
      WHEN nig_kappa - 1 >= 5 THEN '5-9 episodes'
      ELSE '<5 episodes'
    END as bucket,
    COUNT(*) as traders
  FROM trader_performance tp
  JOIN alpha_pool_addresses apa ON tp.address = apa.address
  GROUP BY bucket ORDER BY bucket DESC;
"

# Check FDR-qualified traders
docker compose exec postgres psql -U hlbot -d hlbot -c "
  SELECT address, episode_count, ROUND(avg_r_net::numeric, 3) as avg_r_net,
         ROUND(skill_p_value::numeric, 4) as p_value, selection_rank
  FROM trader_snapshots
  WHERE snapshot_date = CURRENT_DATE AND fdr_qualified = true
  ORDER BY selection_rank;
"
```

### Backfill API Reference

| Endpoint | Description |
|----------|-------------|
| `POST /alpha-pool/refresh?limit=N` | Refresh pool from leaderboard, auto-backfills new addresses |
| `POST /alpha-pool/backfill/{address}` | Backfill a single address |
| `POST /alpha-pool/backfill-all?delay_ms=N` | Backfill ALL addresses with rate limiting |
| `GET /alpha-pool/addresses` | List all addresses in the pool |
| `GET /alpha-pool/status` | Pool statistics including episode counts |

---

## 1. Shadow Ledger Tests

### 1.1 Create Daily Snapshot

**Purpose**: Verify snapshot creation captures all trader state.

```bash
# Create a snapshot for today
curl -X POST "http://localhost:4103/snapshots/create"

# Create a snapshot for a specific date
curl -X POST "http://localhost:4103/snapshots/create?snapshot_date=2025-12-10"
```

**Expected Response**:
```json
{
  "snapshot_date": "2025-12-12",
  "selection_version": "3f.1",
  "total_traders": 150,
  "by_universe": {
    "leaderboard_scanned": 150,
    "candidate_filtered": 80,
    "quality_qualified": 60,
    "pool_selected": 50
  },
  "fdr_qualified": 35,
  "deaths": 2,
  "censored": 5
}
```

**Verify in Database**:
```sql
SELECT
    snapshot_date,
    COUNT(*) as total,
    SUM(CASE WHEN is_pool_selected THEN 1 ELSE 0 END) as selected,
    SUM(CASE WHEN fdr_qualified THEN 1 ELSE 0 END) as fdr_qualified,
    SUM(CASE WHEN event_type = 'death' THEN 1 ELSE 0 END) as deaths
FROM trader_snapshots
WHERE snapshot_date = CURRENT_DATE
GROUP BY snapshot_date;
```

### 1.2 Get Snapshot Summary

**Purpose**: Verify summary statistics are computed correctly.

```bash
curl "http://localhost:4103/snapshots/summary"
curl "http://localhost:4103/snapshots/summary?snapshot_date=2025-12-10"
```

### 1.3 Load Universe at Date (Walk-Forward)

**Purpose**: Verify universe freeze prevents look-ahead bias.

```bash
# Get the universe as it existed on Dec 10
curl "http://localhost:4103/snapshots/universe?evaluation_date=2025-12-10"
```

**Expected Response**:
```json
{
  "evaluation_date": "2025-12-10",
  "version": "3f.1",
  "count": 50,
  "addresses": ["0x123...", "0x456...", ...]
}
```

**Key Verification**: Addresses returned should only be those who were `is_pool_selected=true` on Dec 10, NOT current pool members.

### 1.4 Trader Snapshot History

**Purpose**: Track individual trader evolution over time.

```bash
# Get 30-day history for a specific trader
curl "http://localhost:4103/snapshots/history?address=0x1234...&limit=30"
```

**Expected Response**:
```json
{
  "address": "0x1234...",
  "count": 30,
  "snapshots": [
    {
      "snapshot_date": "2025-12-12",
      "is_pool_selected": true,
      "thompson_draw": 0.35,
      "fdr_qualified": true,
      "event_type": "active"
    },
    ...
  ]
}
```

### 1.5 Death Events

**Purpose**: Verify death detection and tracking.

```bash
# Get recent death events
curl "http://localhost:4103/snapshots/deaths?days=30"

# Filter by death type
curl "http://localhost:4103/snapshots/deaths?days=30&death_type=drawdown_80"
```

**Expected Response**:
```json
{
  "period_days": 30,
  "total_deaths": 5,
  "by_type": {
    "drawdown_80": 3,
    "account_value_floor": 2
  },
  "events": [
    {
      "address": "0xdead...",
      "snapshot_date": "2025-12-08",
      "death_type": "drawdown_80",
      "account_value": 18000,
      "peak_account_value": 100000
    }
  ]
}
```

---

## 2. FDR Qualification Tests

### 2.1 Verify Benjamini-Hochberg Procedure

**Purpose**: Ensure FDR control is working correctly.

```bash
# Check snapshot config for FDR alpha
curl "http://localhost:4103/snapshots/config"
```

**Expected**: `fdr_alpha: 0.10`

**Database Verification**:
```sql
-- Count traders by FDR status
SELECT
    fdr_qualified,
    COUNT(*) as count,
    AVG(skill_p_value) as avg_p_value
FROM trader_snapshots
WHERE snapshot_date = CURRENT_DATE
GROUP BY fdr_qualified;

-- Verify BH procedure: p_i <= (i/n)*alpha for all selected
SELECT
    address,
    skill_p_value,
    selection_rank,
    (selection_rank::float / (SELECT COUNT(*) FROM trader_snapshots WHERE snapshot_date = CURRENT_DATE AND skill_p_value IS NOT NULL)) * 0.10 as bh_threshold
FROM trader_snapshots
WHERE snapshot_date = CURRENT_DATE
  AND fdr_qualified = true
ORDER BY skill_p_value;
```

### 2.2 Verify Gross vs Net R-multiples

**Purpose**: Ensure costs are properly deducted.

```sql
-- Check that avg_r_net < avg_r_gross for all traders
SELECT
    address,
    avg_r_gross,
    avg_r_net,
    avg_r_gross - avg_r_net as cost_deduction
FROM trader_snapshots
WHERE snapshot_date = CURRENT_DATE
  AND avg_r_gross IS NOT NULL
ORDER BY cost_deduction DESC
LIMIT 10;
```

**Expected**: All `avg_r_net < avg_r_gross` (costs reduce returns).

---

## 3. Walk-Forward Replay Tests

### 3.1 Single Period Replay

**Purpose**: Verify replay uses correct historical data.

```bash
# Replay selection from Dec 1
curl "http://localhost:4103/replay/period?selection_date=2025-12-01"
```

**Expected Response**:
```json
{
  "selection_date": "2025-12-01",
  "evaluation_start": "2025-12-01",
  "evaluation_end": "2025-12-08",
  "universe_size": 100,
  "selected_count": 50,
  "fdr_qualified_count": 40,
  "total_r_gross": 5.2,
  "total_r_net": 3.8,
  "avg_r_gross": 0.104,
  "avg_r_net": 0.076,
  "deaths": 1,
  "censored": 2,
  "traders": [
    {
      "address": "0x123...",
      "selection_rank": 1,
      "thompson_draw": 0.45,
      "episodes": 8,
      "r_gross": 0.6,
      "r_net": 0.45,
      "cost_r": 0.15
    }
  ]
}
```

### 3.2 Full Walk-Forward Replay

**Purpose**: Validate multi-period out-of-sample performance.

```bash
# Run 30-day replay
curl -X POST "http://localhost:4103/replay/run?start_date=2025-11-01&end_date=2025-12-01"
```

**Expected Response**:
```json
{
  "start_date": "2025-11-01",
  "end_date": "2025-12-01",
  "periods": 30,
  "performance": {
    "cumulative_r_gross": 15.5,
    "cumulative_r_net": 11.2,
    "avg_period_r_gross": 0.517,
    "avg_period_r_net": 0.373,
    "sharpe_gross": 1.8,
    "sharpe_net": 1.5
  },
  "win_rate": {
    "winning_periods": 20,
    "losing_periods": 10,
    "rate": 0.667
  },
  "survival": {
    "total_deaths": 5,
    "total_censored": 8
  }
}
```

**Key Metrics to Check**:
- `cumulative_r_net > 0` (positive out-of-sample returns)
- `sharpe_net > 0.5` (reasonable risk-adjusted returns)
- `win_rate.rate > 0.5` (winning more periods than losing)

---

## 4. Risk Governor Tests

### 4.1 Check Current Config

```bash
curl "http://localhost:4103/snapshots/config"
```

**Expected Thresholds**:
- `liquidation_distance_min: 1.5` (50% buffer)
- `daily_drawdown_kill_pct: 0.05` (5%)
- `min_equity_floor: 10000` ($10k)
- `max_position_size_pct: 0.10` (10%)
- `max_total_exposure_pct: 0.50` (50%)

### 4.2 Test Kill Switch via Daily Drawdown

**Purpose**: Verify kill switch triggers on large drawdown.

**Simulated Scenario** (modify test data):
1. Set `daily_starting_equity = 100000`
2. Submit trade with `daily_pnl = -6000` (6% loss)
3. Kill switch should trigger

**Database Verification**:
```sql
SELECT * FROM risk_governor_state;
-- kill_switch_active should be 'true'
```

### 4.3 Test Liquidation Distance Block

**Purpose**: Verify trades blocked when near liquidation.

**Simulated Scenario**:
- `account_value = 100000`
- `maintenance_margin = 80000`
- `margin_ratio = 1.25` (below 1.5 threshold)

**Expected**: Trade rejected with "too close to liquidation" message.

### 4.4 Test Position Size Limits

**Purpose**: Verify oversized positions are blocked.

**Simulated Scenario**:
- `account_value = 100000`
- `proposed_position = 15000` (15% of equity)
- Max allowed: 10%

**Expected**: Trade rejected with "Position size exceeds limit".

---

## 5. Integration Tests

### 5.1 Full Pipeline: Leaderboard → Signal

**Purpose**: Verify end-to-end signal generation with all Phase 3f checks.

```bash
# 1. Refresh Alpha Pool
curl -X POST "http://localhost:4103/alpha-pool/refresh?limit=50"

# 2. Wait for fill sync (~5 minutes)
sleep 300

# 3. Create snapshot (captures FDR qualification)
curl -X POST "http://localhost:4103/snapshots/create"

# 4. Check consensus signals
curl "http://localhost:4104/consensus/signals?limit=10"
```

### 5.2 Verify Thompson Sampling Reproducibility

**Purpose**: Ensure Thompson draws can be reproduced.

```sql
-- Get a trader's Thompson draw and seed
SELECT
    address,
    thompson_draw,
    thompson_seed,
    nig_mu,
    nig_kappa,
    nig_alpha,
    nig_beta
FROM trader_snapshots
WHERE snapshot_date = CURRENT_DATE
  AND thompson_draw IS NOT NULL
LIMIT 1;
```

**Verify in Python**:
```python
from services.hl_sage.app.snapshot import thompson_sample_nig

# Use params from query above
draw = thompson_sample_nig(
    m=<nig_mu>,
    kappa=<nig_kappa>,
    alpha=<nig_alpha>,
    beta=<nig_beta>,
    seed=<thompson_seed>
)
assert draw == <thompson_draw>  # Should match exactly
```

---

## 6. Failure Mode Tests

### 6.1 Empty Alpha Pool

```bash
# Clear pool
docker compose exec postgres psql -U hlbot -d hlbot -c "DELETE FROM alpha_pool_addresses"

# Try to create snapshot
curl -X POST "http://localhost:4103/snapshots/create"
```

**Expected**: Should handle gracefully with `total_traders: 0`.

### 6.2 Missing Historical Data

```bash
# Request replay for date with no snapshots
curl "http://localhost:4103/replay/period?selection_date=2020-01-01"
```

**Expected**: 404 with "No snapshot data for 2020-01-01".

### 6.3 Database Connection Failure

```bash
# Stop postgres
docker compose stop postgres

# Try API call
curl "http://localhost:4103/snapshots/summary"

# Should fail gracefully
```

---

## 7. Performance Tests

### 7.1 Large Snapshot Creation

```bash
# Time snapshot creation with many traders
time curl -X POST "http://localhost:4103/snapshots/create"
```

**Target**: < 30 seconds for 1000 traders.

### 7.2 Walk-Forward Replay Performance

```bash
# Time 90-day replay
time curl -X POST "http://localhost:4103/replay/run?start_date=2025-09-01&end_date=2025-12-01"
```

**Target**: < 60 seconds for 90 periods.

---

## Test Summary Checklist

| Test | Status | Notes |
|------|--------|-------|
| 1.1 Create Daily Snapshot | ⬜ | |
| 1.2 Get Snapshot Summary | ⬜ | |
| 1.3 Load Universe at Date | ⬜ | |
| 1.4 Trader Snapshot History | ⬜ | |
| 1.5 Death Events | ⬜ | |
| 2.1 BH Procedure | ⬜ | |
| 2.2 Gross vs Net R | ⬜ | |
| 3.1 Single Period Replay | ⬜ | |
| 3.2 Full Walk-Forward | ⬜ | |
| 4.1 Risk Config | ⬜ | |
| 4.2 Kill Switch | ⬜ | |
| 4.3 Liquidation Block | ⬜ | |
| 4.4 Position Size Limit | ⬜ | |
| 5.1 Full Pipeline | ⬜ | |
| 5.2 Thompson Reproducibility | ⬜ | |
| 6.1 Empty Pool | ⬜ | |
| 6.2 Missing Data | ⬜ | |
| 6.3 DB Failure | ⬜ | |
| 7.1 Snapshot Performance | ⬜ | |
| 7.2 Replay Performance | ⬜ | |

---

## Quick Verification Commands

```bash
# Service health
curl -s http://localhost:4103/healthz | jq .
curl -s http://localhost:4104/healthz | jq .

# Snapshot config
curl -s http://localhost:4103/snapshots/config | jq .

# Create and verify snapshot
curl -s -X POST "http://localhost:4103/snapshots/create" | jq .
curl -s "http://localhost:4103/snapshots/summary" | jq .

# Check deaths
curl -s "http://localhost:4103/snapshots/deaths?days=7" | jq .

# Run replay
curl -s -X POST "http://localhost:4103/replay/run?start_date=2025-12-01&end_date=2025-12-10" | jq .
```

---

*Last updated: December 12, 2025*
