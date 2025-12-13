# SigmaPilot Dashboard Redesign

*Phase 3e UX Specification - December 2025*

---

## 1. Executive Summary

This document specifies an **incremental dashboard enhancement** for SigmaPilot, adding decision transparency and signal performance tracking while keeping the existing vanilla JS architecture.

**Key additions:**
- **Decision Log section** - Why signals fired/skipped with human-readable reasoning
- **Signal Performance stats** - Win rate, avg R-multiple, signal count
- **Enhanced Consensus Signals** - Show gate results and reasoning inline

**Not changing:**
- Vanilla JS (no React rewrite)
- Existing layout structure
- Two-tab design (Alpha Pool / Legacy)

---

## 2. Current Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  HEADER: SigmaPilot | Clock | BTC/ETH Prices | WS Slots | Theme │
├─────────────────────────────────────────────────────────────────┤
│  TradingView Chart (BTC/ETH toggle, collapsible)                │
├─────────────────────────────────────────────────────────────────┤
│  [🎯 Alpha Pool] [📊 Legacy Leaderboard]  ← Tab Navigation      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ALPHA POOL TAB:                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ ⚡ Consensus Signals (table: time, symbol, action, EV...)  ││
│  └─────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ 🎯 Alpha Pool (50 traders: address, BTC, ETH, PnL curve)   ││
│  └─────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ 🔴 Alpha Pool Activity (live fills table)                   ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Proposed Changes

### 3.1 Enhanced Consensus Signals Card

**Current:** Simple table with Time, Symbol, Action, Entry, Stop, EV, Status

**New:** Add expandable reasoning and gate results

```
┌─────────────────────────────────────────────────────────────────┐
│ ⚡ Consensus Signals                           [Stats: 12 | 68%]│
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ 🟢 BTC LONG                                    2 min ago  │  │
│  │ Entry: $43,150  Stop: $42,300  EV: +0.38R                 │  │
│  │ ─────────────────────────────────────────────────────────  │  │
│  │ "7/10 traders opened long. EffK=3.2, low correlation."    │  │
│  │ Gates: ✅Maj ✅EffK ✅Fresh ✅Price ✅EV     [▼ Details]    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ ⚪ ETH SKIP                                   15 min ago   │  │
│  │ "4 traders agree but effK=1.8 (need 2.0). High corr."     │  │
│  │ Gates: ✅Maj ❌EffK ✅Fresh ✅Price ✅EV     [▼ Details]    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ 🟢 ETH SHORT (closed +$342)                   1 hour ago  │  │
│  │ Entry: $2,650  Exit: $2,615  R: +1.25                     │  │
│  │ "5/10 traders opened short. All gates passed."            │  │
│  │ Gates: ✅✅✅✅✅                              [▼ Details]    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Expanded Details (on click):**
```
┌───────────────────────────────────────────────────────────────┐
│ Gate Results:                                                  │
│  • Supermajority: 70% (≥70% required) ✅                      │
│  • Effective-K: 3.2 (≥2.0 required) ✅                        │
│  • Freshness: 12s (≤300s required) ✅                         │
│  • Price Band: 0.12R drift (≤0.25R required) ✅               │
│  • EV Gate: 0.38R (≥0.20R required) ✅                        │
│                                                                │
│ Traders: 7 agreed (ranks #1, #3, #5, #8, #12, #15, #22)       │
│ Avg confidence: 68%                                            │
└───────────────────────────────────────────────────────────────┘
```

### 3.2 Signal Performance Stats Bar

Add a stats bar at the top of Consensus Signals:

```
┌─────────────────────────────────────────────────────────────────┐
│ ⚡ Consensus Signals                                            │
│ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐    │
│ │ Signals │ │ Win Rate│ │ Avg R   │ │ Skipped │ │ Risk Rej│    │
│ │   12    │ │   68%   │ │ +0.45   │ │   23    │ │    3    │    │
│ │  7 days │ │ 8W / 4L │ │         │ │         │ │         │    │
│ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘    │
├─────────────────────────────────────────────────────────────────┤
```

### 3.3 New Section: Decision Log (below Alpha Pool Activity)

Add a new collapsible section for historical decisions:

```
┌─────────────────────────────────────────────────────────────────┐
│ 📋 Decision Log                                    [▼ Collapse] │
│ Filter: [All ▼] [BTC ▼] [7 days ▼]                  [Load More] │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Dec 11, 14:32 │ 🟢 BTC LONG │ +$342 (+1.25R)                   │
│  "7/10 traders opened long BTC. EffK=3.2. All gates passed."    │
│                                                                  │
│  Dec 11, 14:15 │ ⚪ ETH SKIP │ —                                 │
│  "4 traders agree but effK=1.8 (need 2.0). High correlation."   │
│                                                                  │
│  Dec 11, 13:45 │ 🔴 BTC SHORT │ -$156 (-0.5R)                   │
│  "6/10 traders opened short. Lost on reversal."                 │
│                                                                  │
│  Dec 11, 12:30 │ ⚫ ETH LONG │ Risk Rejected                     │
│  "Consensus detected but daily loss limit reached."             │
│                                                                  │
│                        [Load More]                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Requirements

### New API Endpoints

| Endpoint | Purpose | Update |
|----------|---------|--------|
| `GET /dashboard/api/decisions` | List decisions with filters | On demand |
| `GET /dashboard/api/decisions/stats` | Aggregate stats (7d) | 60s polling |

### Decision Object

```javascript
{
  id: "uuid",
  timestamp: "2025-12-11T14:32:00Z",
  symbol: "BTC",
  direction: "long",
  decision_type: "signal",  // signal, skip, risk_reject

  // Inputs
  trader_count: 10,
  agreement_pct: 0.70,
  effective_k: 3.2,
  ev_estimate: 0.38,

  // Gates
  gates: [
    { name: "supermajority", passed: true, value: 0.70, threshold: 0.70 },
    { name: "effective_k", passed: true, value: 3.2, threshold: 2.0 },
    ...
  ],

  // Reasoning
  reasoning: "7/10 traders opened long BTC. EffK=3.2, low correlation.",

  // Outcome (if closed)
  outcome_pnl: 342.50,
  outcome_r_multiple: 1.25,
  outcome_closed_at: "2025-12-11T16:45:00Z"
}
```

### Stats Object

```javascript
{
  period_days: 7,
  signals: 12,
  skips: 23,
  risk_rejects: 3,
  wins: 8,
  losses: 4,
  win_rate: 0.68,
  avg_r_multiple: 0.45,
  total_pnl: 1234.56
}
```

---

## 5. Implementation Details

### HTML Changes

Add to `dashboard.html` after Alpha Pool Activity section:

```html
<!-- Signal Performance Stats -->
<div class="signal-stats-bar" id="signal-stats-bar" data-testid="signal-stats-bar">
  <div class="stat-item">
    <span class="stat-value" id="stat-signals">—</span>
    <span class="stat-label">Signals (7d)</span>
  </div>
  <div class="stat-item">
    <span class="stat-value" id="stat-winrate">—</span>
    <span class="stat-label">Win Rate</span>
  </div>
  <div class="stat-item">
    <span class="stat-value" id="stat-avgr">—</span>
    <span class="stat-label">Avg R</span>
  </div>
  <div class="stat-item">
    <span class="stat-value" id="stat-skipped">—</span>
    <span class="stat-label">Skipped</span>
  </div>
</div>

<!-- Decision Log Section -->
<section class="decision-log-section" data-testid="decision-log-section">
  <div class="card decision-log-card" data-testid="decision-log-card">
    <div class="card-header">
      <h3>
        <span class="log-icon">📋</span>
        Decision Log
      </h3>
      <div class="card-actions">
        <select id="decision-filter-type" data-testid="decision-filter-type">
          <option value="all">All Types</option>
          <option value="signal">Signals Only</option>
          <option value="skip">Skipped</option>
          <option value="risk_reject">Risk Rejected</option>
        </select>
        <select id="decision-filter-symbol" data-testid="decision-filter-symbol">
          <option value="all">All Symbols</option>
          <option value="BTC">BTC</option>
          <option value="ETH">ETH</option>
        </select>
      </div>
    </div>
    <div class="decision-log-container" id="decision-log-container">
      <div id="decision-log-list" data-testid="decision-log-list"></div>
      <button id="decision-load-more" class="load-more-btn">Load More</button>
    </div>
  </div>
</section>
```

### CSS Changes

Add to `dashboard.css`:

```css
/* Signal Stats Bar */
.signal-stats-bar {
  display: flex;
  gap: 1rem;
  padding: 0.75rem 1rem;
  background: var(--card-bg);
  border-radius: 8px;
  margin-bottom: 1rem;
}

.stat-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 0.5rem 1rem;
}

.stat-value {
  font-size: 1.5rem;
  font-weight: 600;
  color: var(--text-primary);
}

.stat-label {
  font-size: 0.75rem;
  color: var(--text-secondary);
}

/* Decision Log */
.decision-log-card {
  margin-top: 1rem;
}

.decision-item {
  padding: 1rem;
  border-bottom: 1px solid var(--border-color);
}

.decision-item:last-child {
  border-bottom: none;
}

.decision-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}

.decision-type {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.decision-type.signal { color: var(--green); }
.decision-type.skip { color: var(--text-secondary); }
.decision-type.risk_reject { color: var(--text-secondary); }

.decision-reasoning {
  color: var(--text-secondary);
  font-size: 0.9rem;
  line-height: 1.4;
}

.decision-outcome {
  margin-top: 0.5rem;
  font-weight: 500;
}

.decision-outcome.win { color: var(--green); }
.decision-outcome.loss { color: var(--red); }

/* Gate indicators */
.gate-indicators {
  display: flex;
  gap: 0.25rem;
  margin-top: 0.5rem;
}

.gate-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.gate-dot.pass { background: var(--green); }
.gate-dot.fail { background: var(--red); }
```

### JavaScript Changes

Add to `dashboard.js`:

```javascript
// Decision Log State
let decisionLogState = {
  items: [],
  offset: 0,
  limit: 20,
  filters: { type: 'all', symbol: 'all' }
};

// Load decision stats
async function loadDecisionStats() {
  try {
    const response = await fetch('/dashboard/api/decisions/stats?days=7');
    const stats = await response.json();

    document.getElementById('stat-signals').textContent = stats.signals || '0';
    document.getElementById('stat-winrate').textContent =
      stats.win_rate ? `${(stats.win_rate * 100).toFixed(0)}%` : '—';
    document.getElementById('stat-avgr').textContent =
      stats.avg_r_multiple ? `${stats.avg_r_multiple > 0 ? '+' : ''}${stats.avg_r_multiple.toFixed(2)}` : '—';
    document.getElementById('stat-skipped').textContent = stats.skips || '0';
  } catch (err) {
    console.error('[dashboard] Failed to load decision stats:', err);
  }
}

// Load decision log
async function loadDecisionLog(append = false) {
  const { type, symbol } = decisionLogState.filters;
  const params = new URLSearchParams({
    limit: decisionLogState.limit,
    offset: append ? decisionLogState.offset : 0
  });

  if (type !== 'all') params.set('decision_type', type);
  if (symbol !== 'all') params.set('symbol', symbol);

  try {
    const response = await fetch(`/dashboard/api/decisions?${params}`);
    const data = await response.json();

    if (append) {
      decisionLogState.items.push(...data.items);
    } else {
      decisionLogState.items = data.items;
      decisionLogState.offset = 0;
    }

    decisionLogState.offset += data.items.length;
    renderDecisionLog();
  } catch (err) {
    console.error('[dashboard] Failed to load decisions:', err);
  }
}

// Render decision log
function renderDecisionLog() {
  const container = document.getElementById('decision-log-list');
  container.innerHTML = decisionLogState.items.map(d => `
    <div class="decision-item" data-testid="decision-item-${d.id}">
      <div class="decision-header">
        <div class="decision-type ${d.decision_type}">
          ${getDecisionIcon(d.decision_type)}
          <span>${d.symbol} ${d.direction?.toUpperCase() || ''}</span>
        </div>
        <span class="decision-time">${formatRelativeTime(d.timestamp)}</span>
      </div>
      <div class="decision-reasoning">"${d.reasoning}"</div>
      ${d.outcome_pnl !== null ? `
        <div class="decision-outcome ${d.outcome_r_multiple > 0 ? 'win' : 'loss'}">
          ${d.outcome_r_multiple > 0 ? '+' : ''}$${d.outcome_pnl.toFixed(2)}
          (${d.outcome_r_multiple > 0 ? '+' : ''}${d.outcome_r_multiple.toFixed(2)}R)
        </div>
      ` : ''}
      <div class="gate-indicators">
        ${d.gates.map(g => `
          <span class="gate-dot ${g.passed ? 'pass' : 'fail'}"
                title="${g.name}: ${g.value.toFixed(2)} ${g.passed ? '≥' : '<'} ${g.threshold}"></span>
        `).join('')}
      </div>
    </div>
  `).join('');
}

function getDecisionIcon(type) {
  switch (type) {
    case 'signal': return '🟢';
    case 'skip': return '⚪';
    case 'risk_reject': return '⚫';
    default: return '⚪';
  }
}

// Poll stats every 60s
setInterval(loadDecisionStats, 60000);

// Initialize
loadDecisionStats();
loadDecisionLog();
```

---

## 6. Mobile Considerations

The current dashboard is already mobile-first. The new sections follow the same pattern:

- Stats bar: horizontal scroll on mobile
- Decision log: full-width cards, stacked
- Filters: dropdowns work well on touch

---

## 7. Summary of Changes

| File | Change |
|------|--------|
| `dashboard.html` | Add stats bar, decision log section |
| `dashboard.css` | Add styles for new components |
| `dashboard.js` | Add loadDecisionStats, loadDecisionLog, renderDecisionLog |
| `hl-stream/src/index.ts` | Proxy `/dashboard/api/decisions*` to hl-decide |
| `hl-decide/app/main.py` | Add `/decisions` and `/decisions/stats` endpoints |

**Estimated effort:** 2-3 days for full implementation + tests
