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
