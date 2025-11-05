# hlbot

![Coverage](badges/coverage.svg) [![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/License-PolyForm%20Noncommercial%201.0.0-brightgreen.svg?style=flat-square)](https://polyformproject.org/licenses/noncommercial/1.0.0/)

Single-tenant Hyperliquid BTC Perp Tracker (MVP).

Quick start
- Prereqs: Node.js 18+ (global `fetch`), npm or pnpm.
- Install deps: `npm install`
- Dev run: `npm run dev` then open http://localhost:3000

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
