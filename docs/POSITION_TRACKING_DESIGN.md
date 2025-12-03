# Position-Based Trader Performance Tracking

## Overview

This document describes the redesigned algorithm for tracking trader performance based on **complete position lifecycles** rather than individual fills.

## Key Concepts

### What is a "Signal"?

A **signal** is when a trader opens a new position (goes from flat to long/short). This represents a directional bet we could follow.

**Signal triggers:**
- `Open Long (Open New)` - Position goes from 0 → positive
- `Open Short (Open New)` - Position goes from 0 → negative
- Direction flip (long → short or short → long) counts as close + open

**NOT signals:**
- `Increase Long/Short` - Adding to existing position
- `Decrease Long/Short` - Partial close

### What is an "Outcome"?

An **outcome** is the realized P&L when a position fully closes.

**Outcome triggers:**
- `Close Long (Close All)` - Position returns to 0
- `Close Short (Close All)` - Position returns to 0
- Direction flip closes previous position

### R-Multiple Calculation

We calculate R-multiples using the **realized_pnl** from Hyperliquid:

```
R = realized_pnl / risk_amount
```

Where `risk_amount` is the notional value at entry × stop_loss_fraction.

For simplicity, we use a fixed assumed stop (e.g., 1%):
```
R = realized_pnl / (entry_notional × 0.01)
```

## New Data Model

### Table: `position_signals`

Tracks each position lifecycle from open to close.

```sql
CREATE TABLE position_signals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    address TEXT NOT NULL,
    asset TEXT NOT NULL,
    direction TEXT NOT NULL,  -- 'long' or 'short'

    -- Entry info
    entry_fill_id TEXT NOT NULL,
    entry_price DOUBLE PRECISION NOT NULL,
    entry_size DOUBLE PRECISION NOT NULL,
    entry_ts TIMESTAMPTZ NOT NULL,

    -- Exit info (NULL until closed)
    exit_fill_id TEXT,
    exit_price DOUBLE PRECISION,
    exit_ts TIMESTAMPTZ,

    -- P&L (NULL until closed)
    realized_pnl DOUBLE PRECISION,
    result_r DOUBLE PRECISION,

    -- Status
    status TEXT NOT NULL DEFAULT 'open',  -- 'open', 'closed'
    closed_reason TEXT,  -- 'full_close', 'direction_flip', 'timeout'

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Indexes
    UNIQUE (entry_fill_id)
);

CREATE INDEX idx_position_signals_address ON position_signals(address);
CREATE INDEX idx_position_signals_open ON position_signals(address, asset, status)
    WHERE status = 'open';
CREATE INDEX idx_position_signals_closed ON position_signals(address, updated_at DESC)
    WHERE status = 'closed';
```

### Updated: `trader_performance`

Now tracks **position-level** performance, not fill-level.

```sql
-- Add new columns for position-based tracking
ALTER TABLE trader_performance ADD COLUMN IF NOT EXISTS
    positions_opened INTEGER NOT NULL DEFAULT 0;
ALTER TABLE trader_performance ADD COLUMN IF NOT EXISTS
    positions_closed INTEGER NOT NULL DEFAULT 0;
ALTER TABLE trader_performance ADD COLUMN IF NOT EXISTS
    positions_won INTEGER NOT NULL DEFAULT 0;
```

## Algorithm Flow

### Step 1: On Fill Event

```python
async def handle_fill(fill: FillEvent):
    action = fill.meta.get('action', '')

    if 'Open' in action and 'Open New' in action:
        # New position opened - create signal
        await create_position_signal(fill)

    elif 'Close' in action and 'Close All' in action:
        # Position closed - find matching open signal, record outcome
        await close_position_signal(fill)

    # Ignore Increase/Decrease fills for performance tracking
```

### Step 2: On Position Close

```python
async def close_position_signal(fill: FillEvent):
    # Find the open position for this address+asset
    open_signal = await find_open_position(fill.address, fill.asset)

    if not open_signal:
        return  # No matching open position

    # Calculate R-multiple
    entry_notional = open_signal.entry_price * open_signal.entry_size
    assumed_stop = 0.01  # 1% stop loss assumption
    risk_amount = entry_notional * assumed_stop
    result_r = fill.realized_pnl / risk_amount if risk_amount > 0 else 0

    # Update the position signal
    await update_position_closed(
        signal_id=open_signal.id,
        exit_fill_id=fill.fill_id,
        exit_price=fill.price,
        exit_ts=fill.ts,
        realized_pnl=fill.realized_pnl,
        result_r=result_r
    )

    # Update trader's NIG posterior with the R-multiple
    await update_trader_nig(fill.address, result_r)
```

### Step 3: NIG Posterior Update

Same as before, but now with **position-level R-multiples**:

```python
# NIG conjugate update
kappa_new = kappa + 1
m_new = (kappa * m + r) / kappa_new
alpha_new = alpha + 0.5
beta_new = beta + 0.5 * kappa * (r - m)^2 / kappa_new
```

## Thompson Sampling with NIG

### Sampling for Selection

```python
def thompson_sample(trader: TraderPosteriorNIG) -> float:
    # 1. Sample variance from InverseGamma
    sigma2 = 1.0 / random.gammavariate(alpha, 1.0/beta)

    # 2. Sample mean from Normal
    mu = random.gauss(m, sqrt(sigma2 / kappa))

    return mu  # Expected R-multiple for this trader
```

### Selection Algorithm

```python
async def select_traders(pool_size: int = 50, k: int = 10) -> List[str]:
    # 1. Get all traders with sufficient history
    traders = await get_traders_with_min_positions(min_positions=5)

    # 2. Thompson sample from each trader's posterior
    samples = [(t, t.sample()) for t in traders]

    # 3. Select top K by sampled value
    samples.sort(key=lambda x: x[1], reverse=True)

    return [t.address for t, _ in samples[:k]]
```

## Why This Is Better

### Before (Fill-Based)

| Problem | Impact |
|---------|--------|
| 1000 fills = 1000 signals | Massive over-counting |
| 10-second timeout | Arbitrary, doesn't match reality |
| Win rate misleading | Loses magnitude information |
| No position tracking | Can't match entries to exits |

### After (Position-Based)

| Improvement | Benefit |
|-------------|---------|
| 1 position = 1 signal | Accurate counting |
| Actual close events | Matches trader's real P&L |
| R-multiple tracking | Captures profit magnitude |
| Full lifecycle | Entry → Exit tracking |

## Example

**Trader opens position:**
```
Fill: Open Long BTC @ $95,000, size 0.5
→ Create position_signal (status='open')
→ No posterior update yet
```

**Trader adds to position (ignored for tracking):**
```
Fill: Increase Long BTC @ $96,000, size 0.3
→ Ignored for performance tracking
```

**Trader closes position:**
```
Fill: Close Long BTC @ $98,000, realized_pnl = $1,500
→ Find open position
→ Calculate: entry_notional = 95000 × 0.5 = $47,500
→ Calculate: risk = $47,500 × 0.01 = $475
→ Calculate: R = $1,500 / $475 = 3.16R
→ Update NIG posterior with r = 3.16
→ Mark position_signal as closed
```

## Migration Plan

1. Create `position_signals` table
2. Reset `trader_performance` (current data is corrupted)
3. Update hl-decide to use position-based tracking
4. Let new data accumulate naturally

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ASSUMED_STOP_FRACTION` | 0.01 | Stop loss % for R calculation |
| `MIN_POSITIONS_RELIABLE` | 5 | Minimum closed positions for reliable estimate |
| `POSITION_TIMEOUT_HOURS` | 168 | Auto-close stale positions after 7 days |
