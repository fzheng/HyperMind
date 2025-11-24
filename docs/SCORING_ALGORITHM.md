# Leaderboard Scoring Algorithm

This document describes the scoring algorithm used to rank trading accounts on the Hyperliquid leaderboard.

## Overview

The scoring system evaluates trading accounts based on four key components, combined into a weighted composite score. The algorithm prioritizes **consistent, stable profit generation** over raw returns, filtering out suspicious patterns and inactive accounts.

## Algorithm Components

### 1. Stability Score (50% weight)

The most important component. Measures how smooth and controlled the profit generation is.

**Formula:**
```
stabilityScore = upFraction × exp(-maxDrawdown / D0) × exp(-ulcerIndex / D0) × exp(-σ_neg / S0)
```

**Parameters:**
- `D0` = 0.20 (drawdown tolerance, configurable via `SCORING_DRAWDOWN_TOLERANCE`)
- `S0` = 0.03 (downside volatility tolerance, configurable via `SCORING_DOWNSIDE_TOLERANCE`)

**Sub-components:**

| Component | Description | Calculation |
|-----------|-------------|-------------|
| `upFraction` | How often equity moves up | `count(delta > 0) / (n - 1)` where delta is step change in normalized equity |
| `maxDrawdown` | Maximum peak-to-trough decline | `max((peak - current) / peak)` on normalized equity [0, 1] |
| `ulcerIndex` | RMS of all drawdowns (persistence of losses) | `sqrt(sum(dd²) / n)` |
| `σ_neg` (downsideVolatility) | Volatility of negative moves only | `sqrt(mean(negativeDeltas²))` |

**Normalization Process:**
1. Extract PnL values from `pnlList` (supports multiple formats)
2. Subtract initial value to create net PnL series starting at 0
3. If final PnL ≤ 0 (not profitable), return `stabilityScore = 0`
4. Normalize to pseudo-equity in [0, 1]: `E[i] = (X[i] - min) / (max - min)`
5. Compute step changes (deltas) between consecutive points
6. Apply exponential penalties for drawdowns and volatility

**Why this matters:**
- Rewards accounts that consistently grow
- Penalizes volatile equity curves even if profitable
- Exponential penalties mean small drawdowns have minimal impact, but large ones are severely penalized

---

### 2. Win Rate Score (25% weight)

Measures trading accuracy with progressive penalties for low win rates.

**Formula:**
```typescript
function computeWinRateScore(winRate: number, threshold: number = 0.60): number {
  // Filter out 100% win rate as suspicious
  if (winRate >= 0.999) return 0;

  if (winRate >= threshold) return Math.min(1.0, winRate);

  // Progressive penalty below threshold
  const deficit = threshold - winRate;
  if (deficit <= 0.05) return winRate * 0.85;      // 55-60%
  else if (deficit <= 0.10) return winRate * 0.70; // 50-55%
  else if (deficit <= 0.15) return winRate * 0.50; // 45-50%
  else if (deficit <= 0.20) return winRate * 0.30; // 40-45%
  else if (deficit <= 0.25) return winRate * 0.15; // 35-40%
  else return winRate * 0.05;                       // <35%
}
```

**Penalty Table:**

| Win Rate Range | Multiplier | Rationale |
|----------------|------------|-----------|
| ≥99.9% | 0 (filtered) | Suspicious - likely wash trading or manipulation |
| 60-100% | 1.0× (no penalty) | Healthy win rate |
| 55-60% | 0.85× | Mild concern |
| 50-55% | 0.70× | Moderate concern |
| 45-50% | 0.50× | Severe penalty |
| 40-45% | 0.30× | Very severe penalty |
| 35-40% | 0.15× | Extreme penalty |
| <35% | 0.05× | Near-zero |

---

### 3. Trade Frequency Score (15% weight)

Balances between too few trades (insufficient data) and too many (scalping/noise).

**Formula:**
```typescript
function computeTradeFreqScore(
  numTrades: number,
  minTrades: number = 3,
  maxTrades: number = 200,
  penaltyThreshold: number = 100
): number {
  if (numTrades < minTrades) return 0;  // Hard filter
  if (numTrades > maxTrades) return 0;  // Hard filter (pre-filtered, won't reach here)
  if (numTrades <= penaltyThreshold) return 1.0;

  // Progressive penalty for trades > 100
  const excess = numTrades - penaltyThreshold;
  if (excess <= 25) return 0.85;      // 100-125 trades
  else if (excess <= 50) return 0.70; // 125-150 trades
  else if (excess <= 75) return 0.50; // 150-175 trades
  else return 0.30;                    // 175-200 trades
}
```

**Penalty Table:**

| Trade Count | Score | Rationale |
|-------------|-------|-----------|
| < 3 | 0 | Insufficient data for reliable scoring |
| 3-100 | 1.0 | Optimal range |
| 100-125 | 0.85 | Mild scalping concern |
| 125-150 | 0.70 | Moderate scalping |
| 150-175 | 0.50 | Heavy scalping |
| 175-200 | 0.30 | Severe scalping penalty |
| > 200 | N/A (removed) | Hard filter - removed before scoring |

---

### 4. Normalized PnL Score (10% weight)

Tiebreaker that gives modest weight to larger profitable accounts.

**Formula:**
```typescript
function computeNormalizedPnl(realizedPnl: number, reference: number = 100000): number {
  if (realizedPnl <= 0) return 0;
  const logScore = Math.log(1 + realizedPnl / reference) / Math.log(11);
  return Math.min(1, Math.max(0, logScore));
}
```

**Score Examples:**

| Realized PnL | Score | Notes |
|--------------|-------|-------|
| $0 or negative | 0 | Not profitable |
| $10,000 | ~0.04 | Small profit |
| $50,000 | ~0.16 | Moderate profit |
| $100,000 (reference) | ~0.30 | Reference point |
| $500,000 | ~0.75 | Large profit |
| $1,000,000+ | ~1.0 | Capped at 1.0 |

**Why log scale:**
- Prevents mega-whales from dominating
- Provides diminishing returns for larger profits
- Makes PnL a tiebreaker, not a primary factor

---

## Final Score Calculation

```
finalScore = 0.50 × stabilityScore
           + 0.25 × winRateScore
           + 0.15 × tradeFreqScore
           + 0.10 × normalizedPnl
```

**Maximum possible score:** 1.0 (all components at max)

**Typical top scores:** 0.45-0.70

---

## Hard Filters (Pre-Scoring Removal)

These filters completely remove accounts before scoring:

| Filter | Threshold | Configurable Via |
|--------|-----------|------------------|
| Excessive trades | > 200 | `SCORING_MAX_TRADES` |
| Inactivity | > 14 days since `lastOperationAt` | `SCORING_INACTIVITY_DAYS` |

**Post-Scoring Filters:**

| Filter | Condition | Effect |
|--------|-----------|--------|
| 100% win rate | `winRate >= 0.999` AND `trades >= 10` | Removed from results |
| Not profitable | `stabilityScore = 0` (final PnL ≤ 0) | Marked as filtered |

---

## Environment Variables

All scoring parameters can be configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SCORING_STABILITY_WEIGHT` | 0.50 | Weight for stability component |
| `SCORING_WIN_RATE_WEIGHT` | 0.25 | Weight for win rate component |
| `SCORING_TRADE_FREQ_WEIGHT` | 0.15 | Weight for trade frequency component |
| `SCORING_PNL_WEIGHT` | 0.10 | Weight for PnL component |
| `SCORING_PNL_REFERENCE` | 100000 | Reference PnL for log normalization |
| `SCORING_MIN_TRADES` | 3 | Minimum trades required |
| `SCORING_MAX_TRADES` | 200 | Maximum trades allowed (hard filter) |
| `SCORING_TRADE_COUNT_THRESHOLD` | 100 | Trades above this get progressive penalty |
| `SCORING_WIN_RATE_THRESHOLD` | 0.60 | Win rate below this gets progressive penalty |
| `SCORING_DRAWDOWN_TOLERANCE` | 0.20 | D0 parameter for stability score |
| `SCORING_DOWNSIDE_TOLERANCE` | 0.03 | S0 parameter for stability score |
| `SCORING_INACTIVITY_DAYS` | 14 | Days of inactivity before removal |

---

## Scoring Details Output

Each scored account includes detailed breakdown in `meta.scoringDetails`:

```typescript
{
  stabilityScore: number,      // [0, 1]
  maxDrawdown: number,         // [0, 1]
  ulcerIndex: number,          // [0, ~0.5]
  upFraction: number,          // [0, 1]
  downsideVolatility: number,  // [0, ~0.1]
  rawWinRate: number,          // [0, 1]
  winRateScore: number,        // [0, 1]
  tradeFreqScore: number,      // [0, 1]
  normalizedPnl: number,       // [0, 1]
  weightedComponents: {
    stability: number,         // stabilityWeight × stabilityScore
    winRate: number,           // winRateWeight × winRateScore
    tradeFreq: number,         // tradeFreqWeight × tradeFreqScore
    pnl: number                // pnlWeight × normalizedPnl
  }
}
```

---

## Example Scoring Scenarios

### Scenario 1: Ideal Trader
- Win rate: 75%
- Trades: 50
- Steady upward PnL curve (upFraction: 0.8, maxDrawdown: 5%)
- Realized PnL: $80,000
- Last active: 2 days ago

**Score breakdown:**
- Stability: 0.80 × exp(-0.05/0.20) × exp(-0.02/0.20) × exp(-0.01/0.03) ≈ 0.55
- Win rate: 0.75 (no penalty)
- Trade freq: 1.0 (in optimal range)
- Normalized PnL: ~0.25

**Final:** 0.50×0.55 + 0.25×0.75 + 0.15×1.0 + 0.10×0.25 = **0.59**

### Scenario 2: Volatile High Performer
- Win rate: 85%
- Trades: 40
- Volatile PnL (upFraction: 0.4, maxDrawdown: 40%)
- Realized PnL: $200,000
- Last active: 1 day ago

**Score breakdown:**
- Stability: 0.40 × exp(-0.40/0.20) × ... ≈ 0.05 (heavily penalized)
- Win rate: 0.85
- Trade freq: 1.0
- Normalized PnL: ~0.45

**Final:** 0.50×0.05 + 0.25×0.85 + 0.15×1.0 + 0.10×0.45 = **0.42**

### Scenario 3: Scalper (Filtered)
- Win rate: 65%
- Trades: 250
- Steady PnL
- Realized PnL: $50,000

**Result:** Removed by hard filter (>200 trades)

### Scenario 4: 100% Win Rate (Filtered)
- Win rate: 100%
- Trades: 30
- Realized PnL: $100,000

**Result:** winRateScore = 0, ranked very low

---

## Algorithm History

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | Nov 2025 | Initial smooth PnL score implementation |
| v2.0 | Nov 2025 | Replaced with stability score (upFraction × penalties) |
| v2.1 | Nov 2025 | Added hard filters: >200 trades, >14 days inactive |
| v2.1 | Nov 2025 | 100% win rate returns winRateScore = 0 |

---

## Future Optimization Ideas

1. **Sharpe-like ratio**: Incorporate risk-adjusted returns
2. **Position sizing analysis**: Penalize excessive leverage
3. **Time-weighted returns**: Account for compound growth
4. **Sector exposure**: Diversification bonus
5. **Drawdown recovery time**: How quickly they recover from losses
6. **Consecutive loss streaks**: Additional penalty for losing streaks
