# Sigma Pilot

![Coverage](badges/coverage.svg) [![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/License-PolyForm%20Noncommercial%201.0.0-brightgreen.svg?style=flat-square)](https://polyformproject.org/licenses/noncommercial/1.0.0/)

**Sigma Pilot** is a collective intelligence trading system that learns from the best traders on Hyperliquid. Instead of relying on traditional technical analysis or blindly copy-trading single wallets, SigmaPilot aggregates wisdom from top-performing traders and generates consensus-based trading signals.

> "Be as smart as the smartest traders by learning from their collective behavior"

## What It Does

- **Scans Top Traders**: Continuously monitors 1000+ traders on Hyperliquid leaderboard
- **Quality Filtering**: Removes losers, HFT bots, and inactive accounts with 7 quality gates
- **Alpha Pool**: Selects top 50 qualified traders ranked by NIG posterior mean
- **Real-time Tracking**: Monitors positions and trades of top performers live
- **Pin Favorites**: Pin accounts from leaderboard or add custom addresses to track
- **Consensus Signals**: Generates trading signals when multiple Alpha Pool traders agree
- **5-Gate Validation**: Supermajority, independence (effK), freshness, drift, and EV gates
- **Self-Learning**: Updates trader posteriors from realized R-multiples

> **Development Status**: Core infrastructure complete. Thompson Sampling exploration and dynamic risk inputs are planned for Phase 3b. See [Development Plan](docs/DEVELOPMENT_PLAN.md) for details.

### Alpha Pool Quality Filters

The Alpha Pool automatically filters out noise traders with 7 quality gates:

| Filter | Default | Description |
|--------|---------|-------------|
| Min 30d PnL | $10,000 | Only profitable traders |
| Min 30d ROI | 10% | Consistent positive returns |
| Min Account | $100,000 | Minimum account value |
| Min Week Vlm | $10,000 | Must be actively trading |
| Max Orders/Day | 100 | Filters out HFT bots via fill history |
| Subaccounts | Excluded | Filters subaccounts (address:X format) |
| BTC/ETH History | Required | Must have traded BTC or ETH |

## Quick Start

```bash
# Install dependencies
npm install

# Configure environment
cp .env.example .env

# Start all services
docker compose up --build
```

**Dashboard**: http://localhost:4102/dashboard

## Build

```bash
npm install          # Install dependencies
npm run build        # Build TypeScript
```

## Run

```bash
# Production (Docker)
docker compose up -d

# Development
npm run dev:scout    # hl-scout in watch mode
npm run dev:stream   # hl-stream in watch mode
```

## Test

```bash
npm run test:unit     # Run Jest unit tests (955 tests)
npm run test:e2e      # Run Playwright e2e tests (requires dashboard running)
npm test              # Run both Jest + Playwright
npm run test:coverage # Jest with coverage report
npm run e2e-smoke     # End-to-end smoke test
```

**E2E Prerequisites**: Playwright tests require a running dashboard:
```bash
docker compose up -d              # Start all services including dashboard
npx playwright install chromium   # First time only
npm run test:e2e                  # Run E2E tests
```
> **Note**: The `webServer` in `playwright.config.ts` is currently disabled. Tests expect the dashboard at `http://localhost:4102/dashboard`.

## Documentation

| Document | Description |
|----------|-------------|
| [API Reference](docs/API.md) | REST API endpoints and WebSocket |
| [FAQ](docs/FAQ.md) | Frequently asked questions |
| [Testing Guide](docs/TESTING.md) | Test suite documentation |
| [Development Plan](docs/DEVELOPMENT_PLAN.md) | Roadmap and phases |
| [Render Deployment](docs/RENDER_DEPLOYMENT.md) | Cloud deployment guide |

## Services

| Service | Port | Description |
|---------|------|-------------|
| Dashboard | [4102/dashboard](http://localhost:4102/dashboard) | Web UI (Alpha Pool + Legacy tabs) |
| hl-scout | 4101 | Leaderboard scanning, candidate publishing |
| hl-stream | 4102 | Real-time feeds, WebSocket, dashboard |
| hl-sage | 4103 | Score computation, NIG Thompson Sampling |
| hl-decide | 4104 | Consensus detection, signal generation |

## License

[PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/) – Free for personal and non-commercial use. For commercial licensing, please contact us.
