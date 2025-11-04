# hlbot

![Coverage](badges/coverage.svg)
License: PolyForm Noncommercial 1.0.0 — non-commercial use only. See LICENSE.

Single-tenant Hyperliquid BTC Perp Tracker (MVP).

Quick start
- Prereqs: Node.js 18+ (global `fetch`), npm or pnpm.
- Install deps: `npm install`
- Dev run: `npm run dev` then open http://localhost:3000

Features
- Add an address to track; de-duplicated and persisted to `data/tracked.json`.
- Background poller (default 90s) fetches BTC price and best-effort BTC perp exposure.
- Recommendations computed server-side and polled by the UI every 10s.
- Minimal single-page UI served from `/`.

Config
- `PORT` env var to change port (default 3000).
- `POLL_INTERVAL_MS` to change poll frequency (default 90000).

Notes
- If Hyperliquid API parsing fails, exposure falls back to 0 (neutral rec). This keeps the server robust.
