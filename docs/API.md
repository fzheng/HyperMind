# API Reference

SigmaPilot exposes REST APIs and WebSocket endpoints across its services. Each service provides interactive Swagger documentation at `/docs`.

## Service URLs

| Service | Swagger UI | Base URL |
|---------|------------|----------|
| hl-scout | http://localhost:4101/docs | `http://localhost:4101` |
| hl-stream | http://localhost:4102/docs | `http://localhost:4102` |
| hl-sage | http://localhost:4103/docs | `http://localhost:4103` |
| hl-decide | http://localhost:4104/docs | `http://localhost:4104` |

## Authentication

Protected endpoints require the `x-owner-key` header:

```bash
curl -H "x-owner-key: YOUR_OWNER_TOKEN" http://localhost:4101/addresses
```

The token is set via `OWNER_TOKEN` environment variable.

---

## hl-scout API (Port 4101)

Leaderboard scanning and address management.

### Health & Metrics

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/healthz` | GET | No | Service health check |
| `/metrics` | GET | No | Prometheus metrics |

### Leaderboard

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/leaderboard` | GET | No | Get leaderboard entries |
| `/leaderboard/selected` | GET | Yes | Get top-ranked selected traders |
| `/leaderboard/refresh` | POST | Yes | Force leaderboard refresh |

**Query Parameters for `/leaderboard`:**
- `period` - Period in days (default: 30)
- `limit` - Max entries to return (default: 20)

### Pinned Accounts

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/pinned-accounts` | GET | No | List all pinned accounts |
| `/pinned-accounts/leaderboard` | POST | No | Pin account from leaderboard (unlimited) |
| `/pinned-accounts/custom` | POST | No | Add custom pinned account (max 3) |
| `/pinned-accounts/:address` | DELETE | No | Unpin account |

**POST `/pinned-accounts/leaderboard` Body:**
```json
{
  "address": "0x..."
}
```

**POST `/pinned-accounts/custom` Body:**
```json
{
  "address": "0x..."
}
```

**GET `/pinned-accounts` Response:**
```json
{
  "accounts": [
    { "id": 1, "address": "0x...", "isCustom": false, "pinnedAt": "..." },
    { "id": 2, "address": "0x...", "isCustom": true, "pinnedAt": "..." }
  ],
  "count": 2,
  "customCount": 1,
  "maxCustomAllowed": 3
}
```

### Custom Accounts (Legacy)

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/custom-accounts` | GET | No | List custom accounts (redirects to pinned) |
| `/custom-accounts` | POST | No | Add custom account (redirects to pinned) |
| `/custom-accounts/:address` | DELETE | No | Remove custom account (redirects to pinned) |

> **Note**: Legacy endpoints are kept for backward compatibility but use the new pinned accounts system internally.

### Addresses (Legacy)

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/addresses` | GET | No | List tracked addresses |
| `/addresses` | POST | Yes | Add address |
| `/addresses/:address` | DELETE | Yes | Remove address |
| `/addresses/seed` | POST | Yes | Seed multiple addresses |

### Data Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/fills` | GET | No | Recent trade fills |
| `/fills/:address` | GET | No | Fills for specific address |
| `/decisions` | GET | No | Recent AI decisions |

---

## hl-stream API (Port 4102)

Real-time feeds and dashboard.

### Health & Metrics

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/healthz` | GET | No | Service health check |
| `/metrics` | GET | No | Prometheus metrics |

### Dashboard

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/dashboard` | GET | No | Web dashboard UI |
| `/dashboard/static/*` | GET | No | Static assets |

### Dashboard API - Alpha Pool

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/dashboard/api/alpha-pool` | GET | No | Get Alpha Pool traders with NIG params (proxied to hl-sage) |
| `/dashboard/api/alpha-pool/fills` | GET | No | Get fills for Alpha Pool addresses only |
| `/dashboard/api/alpha-pool/last-activity` | GET | No | Get most recent fill timestamp per Alpha Pool trader |
| `/dashboard/api/alpha-pool/holdings` | GET | No | Get current positions for Alpha Pool traders |
| `/dashboard/api/alpha-pool/status` | GET | No | NIG model statistics (proxied to hl-sage) |
| `/dashboard/api/alpha-pool/refresh` | POST | No | Refresh pool from leaderboard (proxied to hl-sage) |
| `/dashboard/api/alpha-pool/refresh/status` | GET | No | Get refresh progress status |

**GET `/dashboard/api/alpha-pool/last-activity` Response:**
```json
{
  "lastActivity": {
    "0x1234...": "2025-12-08T14:26:00.000Z",
    "0x5678...": "2025-12-08T12:15:30.000Z"
  }
}
```

### Dashboard API - Legacy

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/dashboard/api/legacy/fills` | GET | No | Initial fills load for Legacy addresses |
| `/dashboard/api/legacy/fills/backfill` | GET | No | Paginated backfill for "Load More" |
| `/dashboard/api/legacy/fills/fetch-history` | POST | No | Fetch from Hyperliquid API |
| `/dashboard/api/legacy/fills/oldest` | GET | No | Get oldest fill timestamp |
| `/dashboard/api/legacy/fills/validate` | GET | No | Position chain validation |
| `/dashboard/api/legacy/fills/repair` | POST | No | Repair data for specific address |

### Dashboard API - Shared

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/dashboard/api/summary` | GET | No | Leaderboard summary |
| `/dashboard/api/prices` | GET | No | Real-time BTC/ETH prices |
| `/dashboard/api/consensus/signals` | GET | No | Consensus signals |
| `/dashboard/api/pinned-accounts` | GET | No | User's pinned accounts |

### Watchlist

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/watchlist` | GET | No | Current tracked addresses |
| `/watchlist/refresh` | POST | Yes | Refresh from hl-scout |

### Fills Data

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/fills` | GET | No | Recent fills with pagination |
| `/fills/backfill` | GET | No | Paginated historical fills |
| `/fills/validate` | GET | No | Validate position chain integrity |
| `/fills/repair` | POST | No | Repair data for a specific address |
| `/fills/repair-all` | POST | No | Auto-repair all invalid addresses |

**Query Parameters for `/fills`:**
- `limit` - Max fills (default: 40, max: 200)

**Query Parameters for `/fills/backfill`:**
- `before` - ISO timestamp for pagination
- `limit` - Max fills (default: 50, max: 100)

**Query Parameters for `/fills/validate`:**
- `symbol` - Asset symbol (`BTC` or `ETH`, default: `ETH`)

**POST `/fills/repair` Body:**
```json
{
  "address": "0x...",
  "symbol": "ETH"
}
```

**POST `/fills/repair-all` Body:**
```json
{
  "symbol": "ETH"
}
```

### Prices

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/prices` | GET | No | Current BTC/ETH prices |

### Positions

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/positions/status` | GET | No | Position tracking status |

### WebSocket

| Endpoint | Description |
|----------|-------------|
| `ws://localhost:4102/ws` | Real-time event stream |

**WebSocket Message Format:**
```json
{
  "events": [
    {
      "type": "trade",
      "seq": 123,
      "at": "2025-01-01T00:00:00Z",
      "address": "0x...",
      "symbol": "BTC",
      "side": "buy",
      "priceUsd": 95000,
      "size": 0.1
    }
  ]
}
```

---

## hl-sage API (Port 4103)

Score computation and Alpha Pool management (Python/FastAPI).

### Health & Metrics

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/healthz` | GET | No | Service health |
| `/metrics` | GET | No | Prometheus metrics |
| `/scores` | GET | No | Current computed scores |

### Alpha Pool (Decoupled System)

The Alpha Pool is **completely independent** from the legacy leaderboard:

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/alpha-pool` | GET | No | Get traders with NIG posteriors and PnL curves |
| `/alpha-pool/refresh` | POST | No | Populate pool from Hyperliquid leaderboard |
| `/alpha-pool/addresses` | GET | No | List addresses in the pool |
| `/alpha-pool/status` | GET | No | NIG model statistics |
| `/alpha-pool/sample` | POST | No | Demonstrate Thompson sampling |

**POST `/alpha-pool/refresh` Query Parameters:**
- `limit` - Number of traders to fetch (default: 50, max: 200)

**GET `/alpha-pool` Query Parameters:**
- `limit` - Max traders to return (default: 50)
- `min_signals` - Minimum signals required (default: 0)

**GET `/alpha-pool` Response:**
```json
{
  "count": 50,
  "pool_size": 50,
  "select_k": 10,
  "traders": [
    {
      "address": "0x...",
      "nickname": "Trader Name",
      "nig_m": 0.0,
      "nig_kappa": 1.0,
      "nig_alpha": 3.0,
      "nig_beta": 1.0,
      "posterior_std": 0.7071,
      "effective_samples": 0.0,
      "total_signals": 0,
      "total_pnl_r": 0.0,
      "avg_r": 0.0,
      "is_selected": true,
      "pnl_curve": [{"ts": 1762127280000, "value": "0.0"}, ...]
    }
  ]
}
```

### Thompson Sampling (Legacy)

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/bandit/status` | GET | No | Legacy bandit status |
| `/bandit/decay` | POST | No | Apply decay to posteriors |

---

## hl-decide API (Port 4104)

Signal generation and consensus detection (Python/FastAPI).

### Health & Metrics

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/healthz` | GET | No | Service health |
| `/metrics` | GET | No | Prometheus metrics |
| `/signals` | GET | No | Recent generated signals |
| `/outcomes` | GET | No | Signal outcomes |

### Consensus Signals

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/consensus/signals` | GET | No | Recent consensus signals with outcomes |
| `/consensus/stats` | GET | No | Aggregate win rate and EV statistics |

**GET `/consensus/signals` Query Parameters:**
- `limit` - Max signals to return (default: 20)

**GET `/consensus/signals` Response:**
```json
{
  "signals": [
    {
      "id": 1,
      "ts": "2025-12-01T00:00:00Z",
      "asset": "BTC",
      "direction": "long",
      "n_traders": 5,
      "n_agree": 4,
      "majority_pct": 0.8,
      "eff_k": 3.2,
      "p_win": 0.65,
      "ev_net_r": 0.12,
      "outcome": "win",
      "realized_r": 0.15
    }
  ],
  "count": 1
}
```

### Market Regime Detection (Phase 5)

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/regime` | GET | No | Get regime for all assets (BTC, ETH) |
| `/regime/{asset}` | GET | No | Get regime for specific asset |
| `/regime/params` | GET | No | Get regime parameter presets |

**GET `/regime` Response:**
```json
{
  "regimes": {
    "BTC": {
      "asset": "BTC",
      "regime": "trending",
      "confidence": 0.75,
      "params": {
        "stop_multiplier": 1.2,
        "kelly_multiplier": 1.0,
        "min_confidence_adjustment": 0.0,
        "max_position_fraction": 1.0
      },
      "signals": {
        "ma_spread_pct": 0.035,
        "volatility_ratio": 0.85,
        "price_range_pct": 0.025
      },
      "candles_used": 60
    }
  },
  "summary": {
    "trending": 1,
    "ranging": 1,
    "volatile": 0,
    "unknown": 0
  }
}
```

**Regime Types:**
| Regime | Detection | Strategy Adjustment |
|--------|-----------|---------------------|
| `trending` | MA spread > 2% | Wider stops (1.2x), full Kelly |
| `ranging` | MAs converged, low vol | Tighter stops (0.8x), 75% Kelly |
| `volatile` | ATR ratio > 1.5x | Wide stops (1.5x), 50% Kelly |
| `unknown` | Insufficient data | Conservative defaults |

### Multi-Exchange Module (Phase 6)

The exchange module provides a unified interface for trading across multiple exchanges. Currently not exposed as REST API - used internally by executor.

**Python Usage:**
```python
from app.exchanges import (
    get_exchange,
    connect_exchange,
    ExchangeType,
    OrderParams,
    OrderSide,
)

# Create and connect to exchange
exchange = await connect_exchange(ExchangeType.HYPERLIQUID, testnet=True)

# Get account state
balance = await exchange.get_balance()
positions = await exchange.get_positions()

# Place order with stops
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

**Supported Exchanges:**
| Exchange | Type | SDK | Symbol Format |
|----------|------|-----|---------------|
| Hyperliquid | DEX | hyperliquid-python-sdk | `BTC`, `ETH` |
| Aster | DEX | ECDSA/EIP-712 | `BTC-PERP`, `ETH-PERP` |
| Bybit | CEX | pybit | `BTCUSDT`, `ETHUSDT` |

**Interface Operations:**
| Operation | Description |
|-----------|-------------|
| `connect()` / `disconnect()` | Connection lifecycle |
| `get_balance()` | Account equity, margin, P&L |
| `get_positions()` | Open positions with entry/mark prices |
| `open_position()` | Market/limit orders with stops |
| `close_position()` | Partial or full close |
| `set_leverage()` | Leverage configuration |
| `set_stop_loss()` / `set_take_profit()` | Position protection |
| `get_market_price()` | Current mid price |

---

## NATS Message Topics

Internal pub/sub messaging between services:

| Topic | Publisher | Consumer | Description |
|-------|-----------|----------|-------------|
| `a.candidates.v1` | hl-scout | hl-sage | Candidate trader events |
| `b.scores.v1` | hl-sage | hl-decide | Computed scores |
| `c.fills.v1` | hl-stream | hl-decide | Real-time fill events |
| `d.signals.v1` | hl-decide | - | Generated signals |
| `d.outcomes.v1` | hl-decide | - | Signal outcomes |

---

## Error Responses

All APIs return consistent error format:

```json
{
  "error": "Error message description"
}
```

**HTTP Status Codes:**
- `200` - Success
- `400` - Bad request / validation error
- `403` - Forbidden (missing/invalid auth)
- `404` - Not found
- `500` - Internal server error
