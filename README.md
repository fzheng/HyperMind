# hlbot

![Coverage](badges/coverage.svg) [![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/License-PolyForm%20Noncommercial%201.0.0-brightgreen.svg?style=flat-square)](https://polyformproject.org/licenses/noncommercial/1.0.0/)

Single-tenant Hyperliquid BTC Perp Tracker (MVP).

Quick start
- Prereqs: Node.js 18+ (global `fetch`), npm or pnpm.
- Install deps: `npm install`
- Dev run (memory storage): `npm run dev` then open http://localhost:3000

Local development (with Postgres)
- Start Postgres only via Compose: `docker compose up -d db`
- Set env for the app (PowerShell examples):
  - `$env:STORAGE_BACKEND = 'postgres'`
  - `$env:DATABASE_URL = 'postgresql://hlbot:hlbotpassword@localhost:5432/hlbot'`
- Apply migrations: `npm run migrate`
- Start dev server: `npm run dev` (http://localhost:3000)

Production (bare metal)
- Build: `npm run build`
- Set env (at minimum): `STORAGE_BACKEND=postgres`, `DATABASE_URL=postgresql://...`
- Migrate: `npm run migrate`
- Start: `npm start`

Docker
- Create `.env` from template: `cp .env.example .env` (or copy manually on Windows)
- Build and run all services: `docker compose up -d --build`
- Open the UI: http://localhost:3000 (or set `APP_PORT` in `.env`)
- View logs: `docker compose logs -f app`
- Stop containers (keep data): `docker compose down`
- Remove containers and data: `docker compose down -v`
- Notes:
  - Default backend is Postgres (`STORAGE_BACKEND=postgres`) with persistent volume `pgdata`.
  - Redis is included but optional. Switch by setting `STORAGE_BACKEND=redis` and ensuring `REDIS_URL` is set.
  - The app runs DB migrations automatically on container start.

Migrations
- All SQL migrations live in `migrations/` and are applied in lexicographic order (e.g., `001_init.sql`, `002_add_x.sql`).
- Applied versions are tracked in `schema_migrations`.
- Run locally: `npm run migrate` or check status with `npm run migrate:status`.
- In Docker: the app image runs migrations automatically on startup via `docker/entrypoint.sh` before launching the server.

Environment variables
- `PORT` (default `3000`) — Express server port.
- `POLL_INTERVAL_MS` (default `90000`) — background poll interval.
- `STORAGE_BACKEND` — `postgres` (recommended), `redis`, or `memory`.
- Postgres: use `DATABASE_URL` (e.g., `postgresql://user:pass@host:5432/db`).
- Redis: use `REDIS_URL` (e.g., `redis://localhost:6379`).
- See `.env.example` for a complete template used by docker-compose.

API endpoints
- `GET /api/addresses` — list tracked addresses.
- `POST /api/addresses` — add an address `{ address: string }`.
- `DELETE /api/addresses/:address` — remove an address.
- `GET /api/recommendations` — current recommendations.
- `POST /api/poll-now` — trigger an immediate background poll.
- `GET /api/positions/:address` — on-demand perp positions for an address.
- `GET /api/price` — current BTCUSD price (ws/http source info included).
- `GET /` — static UI.

Realtime changes (beta)
- The server streams BTC position changes and trade fills into an in-memory queue.
- Pull API: `GET /api/changes?since=<seq>&limit=<n>` returns ordered events and the next cursor.
- Event types:
  - `position`: `{ address, symbol: 'BTC', size (signed), side, entryPriceUsd, liquidationPriceUsd, leverage, pnlUsd, at, seq }`
  - `trade`: `{ address, symbol: 'BTC', side: 'buy'|'sell', direction: 'long'|'short', effect: 'open'|'close', priceUsd, size, realizedPnlUsd?, at, seq }`
- Notes: events are kept in memory (rolling buffer). Persisting or server-sent streaming can be added next.

Troubleshooting
- Port in use (3000/5432): change `APP_PORT`, `PG_PORT`, or `PORT` in `.env`.
- DB not ready: compose waits for Postgres health. Check `docker compose logs db`.
- Migrations failed: view `docker compose logs app` or run `npm run migrate` locally with correct `DATABASE_URL`.
- Reset data: `docker compose down -v` (removes Postgres and Redis volumes).

Features
- Add an address to track; de-duplicated and persisted via Redis or Postgres.
- Background poller (default 90s) fetches BTC price and best-effort BTC perp exposure.
- Recommendations computed server-side and polled by the UI every 10s.
- Minimal single-page UI served from `/`.

Config
- `PORT` env var to change port (default 3000).
- `POLL_INTERVAL_MS` to change poll frequency (default 90000).
- Storage backend (no local files):
  - Redis: set `STORAGE_BACKEND=redis` and `REDIS_URL=redis://localhost:6379`
  - Postgres: set `STORAGE_BACKEND=postgres` and either `PG_CONNECTION_STRING` or `DATABASE_URL`
  - If neither is set, an in-memory backend is used (dev/tests only).

Notes
- If Hyperliquid API parsing fails, exposure falls back to 0 (neutral rec). This keeps the server robust.

Realtime and trades
- Live WS subscriptions track BTC positions and user fills per address.
- Durable store in Postgres for trades (`hl_events`) and current positions (`hl_current_positions`).
- Cursor pagination for trades: `GET /api/latest-trades?limit=100&beforeId=<id>&address=<0x...>` → `{ trades, nextBeforeId }`.
- Backfill and cleanup:
  - `POST /api/backfill` with `{ address?: string|null, limit?: number }` imports recent fills for an address or for all when `address` is null.
  - `POST /api/cleanup-and-backfill` purges invalid trades and backfills up to 100 per tracked address.
  - New addresses are auto‑backfilled (latest 100).
  - Unique index on trade tx hash prevents duplicates during backfill.

Additional endpoints (summary)
- `GET /api/current-positions` — current BTC positions for all tracked addresses (with nicknames when set).
- `GET /api/latest-trades` — stable, paginated latest trades from DB.
- `GET /api/user-trades/:address` — recent BTC fills for a specific address (Info API).
- `POST /api/backfill` — backfill recent fills to DB (deduped by tx hash).
- `POST /api/cleanup-and-backfill` — cleanup invalid trades and seed latest fills for all addresses.
- `POST /api/addresses/:address/nickname` — set/clear nickname.

UI highlights
- Positions: nickname editing, integrated address actions, and updated time column with Absolute/Relative toggle.
- Trades: Address filter, Refresh, Load more (pagination), Backfill, Clean + Backfill actions.
- Sorting: click table headers (positions and trades).
