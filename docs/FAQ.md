# Frequently Asked Questions

## General

### What is HyperMind?

HyperMind is a collective intelligence trading system that monitors top traders on Hyperliquid and generates consensus-based trading signals. Instead of copy-trading a single wallet, it aggregates patterns from multiple top performers.

### Is this financial advice?

No. HyperMind is an experimental research tool. All signals are for informational purposes only. Always do your own research before making trading decisions.

### Which assets are supported?

Currently BTC and ETH perpetual contracts on Hyperliquid.

---

## Setup

### How do I start HyperMind?

```bash
npm install
cp .env.example .env
docker compose up --build
```

Then visit http://localhost:4102/dashboard

### What ports are used?

| Port | Service |
|------|---------|
| 4101 | hl-scout (API) |
| 4102 | hl-stream (Dashboard + API) |
| 4103 | hl-sage (Python) |
| 4104 | hl-decide (Python) |
| 5432 | PostgreSQL |
| 4222 | NATS |

### How do I configure the system?

Copy `.env.example` to `.env` and edit as needed. Key settings:

- `OWNER_TOKEN` - Auth token for admin endpoints
- `LEADERBOARD_TOP_N` - How many traders to scan (default: 1000)
- `LEADERBOARD_SELECT_COUNT` - How many to actively track (default: 12)

### The dashboard shows "Waiting for fills..."

This is normal on first startup. The system needs time to:
1. Scan the leaderboard
2. Connect to Hyperliquid WebSockets
3. Wait for tracked traders to make trades

You can also click "Load More" to fetch historical fills.

---

## Dashboard

### What do the panels show?

- **AI Signals** - Trading signals generated from tracked trader patterns (coming soon)
- **Tracked Traders** - Top performers being monitored, with stats
- **Trader Activity** - Real-time fills from tracked accounts

### What do the pin icons mean?

- **Blue pin** - Account pinned from the leaderboard (unlimited)
- **Gold pin** - Custom account you added manually (max 3)
- **Faded pin** - Unpinned account (system-selected from top performers)

Pinned accounts are always tracked, even when they drop out of the top rankings.

### How do I pin an account?

Two ways to pin accounts:

1. **From leaderboard**: Click the pin icon next to any trader in the list
2. **Custom address**: Enter an Ethereum address (0x...) in the "Add Custom" input box and click +

You can pin unlimited accounts from the leaderboard, but only add up to 3 custom addresses.

### Why are some fills grouped together?

Fills within 1 minute from the same trader and action are aggregated. Click the badge (e.g., "×3") to expand and see individual fills.

### What does "30-day" mean in rankings?

Traders are ranked based on their 30-day performance on Hyperliquid. This includes win rate, PnL, and consistency metrics.

### When does the leaderboard refresh?

The leaderboard refreshes daily at **00:30 UTC**. This timing avoids conflicts with other services that reset at UTC midnight. After refresh, the system:
1. Fetches the top 1000 traders from Hyperliquid
2. Scores and ranks them using the composite scoring algorithm
3. Enriches the top candidates with detailed stats
4. Filters by BTC/ETH trading performance
5. Publishes qualified candidates for tracking

---

## Technical

### How do I run tests?

```bash
npm test                # Run all tests
npm run test:coverage   # With coverage report
```

### How do I reset the database?

```bash
docker compose down -v
docker compose up --build
```

This removes all data and recreates the database from scratch.

### How do I view logs?

```bash
docker compose logs -f           # All services
docker compose logs -f hl-stream # Specific service
```

### Where is the data stored?

PostgreSQL stores all data in a Docker volume. Key tables:
- `hl_events` - Trade and position events
- `hl_leaderboard_entries` - Trader rankings
- `hl_pinned_accounts` - User's pinned accounts

### Why is leaderboard.ts coverage not 100%?

The leaderboard service (1600+ lines) makes many external API calls to Hyperliquid and Hyperbot. While external calls are mocked in tests, some internal methods are complex and integration-heavy. Coverage is ~49% with focus on scoring logic and API integration patterns. Full coverage would require extensive mocking of private methods.

---

## Troubleshooting

### "Connection refused" errors

Make sure all containers are running:
```bash
docker compose ps
```

If a container is down, check its logs:
```bash
docker compose logs hl-scout
```

### "Forbidden" on API calls

Protected endpoints require the `x-owner-key` header with your `OWNER_TOKEN`:
```bash
curl -H "x-owner-key: YOUR_TOKEN" http://localhost:4101/addresses
```

### Dashboard not loading

1. Check hl-stream is running: `docker compose ps`
2. Check browser console for errors
3. Try hard refresh (Ctrl+Shift+R)

### WebSocket disconnecting

The WebSocket auto-reconnects after 2 seconds. If it keeps disconnecting:
1. Check network connectivity
2. Check hl-stream logs for errors
3. Restart hl-stream: `docker compose restart hl-stream`

---

## Contributing

### How can I contribute?

See [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) for the roadmap. Key areas:
- Consensus signal engine (Phase 2)
- Performance feedback loop (Phase 3)
- AI learning layer (Phase 4)

### What's the code style?

- TypeScript with strict mode
- ESLint for linting
- Prettier for formatting
- Jest for testing
