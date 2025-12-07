# Architecture

Technical documentation for developers working on SigmaPilot.

## Services

| Service | Port | Language | Description |
|---------|------|----------|-------------|
| hl-scout | 4101 | TypeScript | Leaderboard scanning, candidate publishing |
| hl-stream | 4102 | TypeScript | Real-time feeds, WebSocket, dashboard UI |
| hl-sage | 4103 | Python | Score computation, NIG Thompson Sampling |
| hl-decide | 4104 | Python | Consensus detection, signal generation |

**Dashboard**: http://localhost:4102/dashboard

## Message Flow

```
hl-scout → a.candidates.v1 → hl-sage → b.scores.v1 → hl-decide
                                                          ↓
hl-stream ← c.fills.v1 ←──────────────────────────────────┘
     ↓
d.signals.v1 → hl-decide → d.outcomes.v1
```

### NATS Topics

| Topic | Schema | Publisher | Consumer |
|-------|--------|-----------|----------|
| `a.candidates.v1` | CandidateEvent | hl-scout | hl-sage |
| `b.scores.v1` | ScoreEvent | hl-sage | hl-decide |
| `c.fills.v1` | FillEvent | hl-stream | hl-decide |
| `d.signals.v1` | SignalEvent | hl-decide | internal |
| `d.outcomes.v1` | OutcomeEvent | hl-decide | persistence |

Contracts defined in `contracts/jsonschema/*.json`.

## Alpha Pool Quality Filters

When refreshing the Alpha Pool, traders pass through 7 quality gates:

| Filter | Default | Environment Variable |
|--------|---------|---------------------|
| Min 30d PnL | $10,000 | `ALPHA_POOL_MIN_PNL` |
| Min 30d ROI | 10% | `ALPHA_POOL_MIN_ROI` |
| Min Account Value | $100,000 | `ALPHA_POOL_MIN_ACCOUNT_VALUE` |
| Min Weekly Volume | $10,000 | `ALPHA_POOL_MIN_WEEK_VLM` |
| Max Orders/Day | 100 | `ALPHA_POOL_MAX_ORDERS_PER_DAY` |
| Subaccounts | Excluded | (hardcoded) |
| BTC/ETH History | Required | (hardcoded) |

## Consensus Detection Gates

Signals must pass 5 gates before firing:

1. **Dispersion Gate** — Supermajority agreement (≥66% weight)
2. **Effective-K Gate** — Correlation-adjusted trader count (min 3 independent)
3. **Latency Gate** — Signal freshness (within window × factor)
4. **Price Band Gate** — Market hasn't drifted too far (ATR-based R-units)
5. **EV Gate** — Positive expected value after fees and slippage

## Database

PostgreSQL with automatic migrations on startup.

### Key Tables

| Table | Purpose |
|-------|---------|
| `addresses` | Tracked wallet addresses |
| `hl_events` | Trade history (deduped by hash) |
| `hl_current_positions` | Real-time position snapshots |
| `marks_1m` | 1-minute OHLC candles |
| `tickets` | Consensus signal tickets |
| `ticket_outcomes` | Realized trade outcomes |
| `trader_performance` | NIG posterior parameters |
| `alpha_pool_addresses` | Decoupled Alpha Pool |
| `hl_leaderboard_entries` | Legacy leaderboard data |

### Migrations

- Located in `db/migrations/*.sql`
- Run automatically on hl-scout startup
- Tracked in `schema_migrations` table

## Development Commands

```bash
make help            # Show all commands
make test            # Run all tests (TS + Python)
make up              # Start services
make down            # Stop services
make rebuild         # Rebuild and restart
make wipe            # Fresh start (deletes data!)
make logs            # Follow logs
```

## Shared Library (@hl/ts-lib)

Located in `packages/ts-lib/src/`:

| Module | Purpose |
|--------|---------|
| `nats.ts` | NATS connection/stream helpers |
| `postgres.ts` | Pool management (singleton) |
| `hyperliquid.ts` | Hyperliquid API helpers |
| `realtime.ts` | WebSocket tracker with position priming |
| `persist.ts` | Database operations |
| `address-store.ts` | Address list + Postgres sync |
| `validation.ts` | Input validation |
| `scoring.ts` | Performance scoring |
| `consensus.ts` | 5-gate consensus detection |
| `episode.ts` | Episode builder with VWAP |
| `metrics.ts` | Prometheus helpers |
| `migrate.ts` | Migration runner |

## API Endpoints

Each service exposes:
- `GET /healthz` — Health check
- `GET /metrics` — Prometheus metrics
- `GET /docs` — OpenAPI documentation

### Dashboard API (hl-stream)

**Alpha Pool**:
- `GET /dashboard/api/alpha-pool` — Pool traders with NIG params
- `GET /dashboard/api/alpha-pool/fills` — Fills for pool addresses
- `POST /dashboard/api/alpha-pool/refresh` — Refresh from leaderboard

**Legacy**:
- `GET /dashboard/api/legacy/fills` — Legacy watchlist fills
- `GET /dashboard/api/legacy/fills/backfill` — Paginated backfill

**Shared**:
- `GET /dashboard/api/prices` — Real-time BTC/ETH prices
- `GET /dashboard/api/consensus/signals` — Recent consensus signals
- `WebSocket /ws` — Live fill stream

## Environment Variables

Key configuration (see `.env.example`):

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string |
| `NATS_URL` | NATS server URL |
| `OWNER_TOKEN` | Admin authentication token |
| `LEADERBOARD_SELECT_COUNT` | Traders to track (default 10) |
| `ATR_STRICT_MODE` | Block signals on stale ATR (default true) |
| `VOTE_WEIGHT_MODE` | Weight calculation mode (log/equity/linear) |

## Testing

- **TypeScript**: Jest (977 tests)
- **Python**: pytest (144 tests)
- **E2E**: Playwright (150 tests)
- **Coverage**: ~76% overall, ~97% for ts-lib

```bash
make test            # All tests
make test-ts         # TypeScript only
make test-py         # Python only
make test-e2e        # Playwright (requires running dashboard)
make test-coverage   # With coverage report
```

## Docker

```bash
docker compose up -d      # Start all
docker compose logs -f    # Follow logs
docker compose ps         # Status
```

Services are networked internally. Don't set `DATABASE_URL` or `NATS_URL` in `.env` when using Docker—these are configured in `docker-compose.yml`.
