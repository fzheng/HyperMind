# HyperMind

![Coverage](badges/coverage.svg) [![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/License-PolyForm%20Noncommercial%201.0.0-brightgreen.svg?style=flat-square)](https://polyformproject.org/licenses/noncommercial/1.0.0/)

**HyperMind** is a collective intelligence trading system that learns from the best traders on Hyperliquid. Instead of relying on traditional technical analysis or blindly copy-trading single wallets, HyperMind aggregates wisdom from top-performing traders and generates consensus-based trading signals.

> "Be as smart as the smartest traders by learning from their collective behavior"

## What It Does

- **Scans Top Traders**: Continuously monitors 1000+ traders on Hyperliquid leaderboard
- **Smart Ranking**: Scores traders by win rate, PnL consistency, and risk management
- **Real-time Tracking**: Monitors positions and trades of top performers live
- **Pin Favorites**: Pin accounts from leaderboard or add custom addresses to track
- **AI Signals**: Generates trading signals when multiple top traders align (coming soon)
- **Self-Learning**: Improves by analyzing past signal performance (coming soon)

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
npm run test:unit     # Run Jest unit tests (830 tests)
npm run test:e2e      # Run Playwright e2e tests (requires dashboard running)
npm test              # Run both Jest + Playwright
npm run test:coverage # Jest with coverage report
npm run e2e-smoke     # End-to-end smoke test
```

**E2E Prerequisites**: Before running Playwright tests, start the dashboard:
```bash
docker compose up -d hl-stream
npx playwright install chromium  # First time only
```

## Documentation

| Document | Description |
|----------|-------------|
| [API Reference](docs/API.md) | REST API endpoints and WebSocket |
| [FAQ](docs/FAQ.md) | Frequently asked questions |
| [Testing Guide](docs/TESTING.md) | Test suite documentation |
| [Development Plan](docs/DEVELOPMENT_PLAN.md) | Roadmap and phases |

## Services

| Service | Port | Description |
|---------|------|-------------|
| Dashboard | [4102/dashboard](http://localhost:4102/dashboard) | Web UI |
| hl-scout | 4101 | Leaderboard scanning |
| hl-stream | 4102 | Real-time feeds |
| hl-sage | 4103 | Score computation |
| hl-decide | 4104 | Signal generation |

## License

[PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/) – Free for personal and non-commercial use. For commercial licensing, please contact us.
