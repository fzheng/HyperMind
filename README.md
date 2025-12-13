# Sigma Pilot

![Coverage](badges/coverage.svg) [![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/License-PolyForm%20Noncommercial%201.0.0-brightgreen.svg?style=flat-square)](https://polyformproject.org/licenses/noncommercial/1.0.0/)

**Website**: [sigmapilot.ai](https://sigmapilot.ai)

## Overview

**Sigma Pilot** is a collective intelligence trading system that learns from the best traders on [Hyperliquid](https://hyperliquid.xyz), the leading decentralized perpetual futures exchange.

Instead of relying on traditional technical analysis or blindly copy-trading individual wallets, SigmaPilot aggregates wisdom from top-performing traders and generates consensus-based trading signals—only acting when multiple successful traders agree.

> "Be as smart as the smartest traders by learning from their collective behavior"

## The Problem

Copy-trading a single wallet is risky:
- One trader can have a bad streak
- You're exposed to their mistakes and biases
- No way to know if their edge is real or luck

Traditional signals rely on lagging indicators that don't adapt to changing market conditions.

## Our Solution

SigmaPilot solves this by:

1. **Monitoring the Best** — Continuously tracks 1,000+ traders on Hyperliquid's leaderboard
2. **Filtering for Quality** — Removes noise traders, HFT bots, and inactive accounts
3. **Building a Smart Pool** — Selects top 50 qualified traders using Bayesian ranking
4. **Detecting Consensus** — Generates signals only when multiple independent traders agree
5. **Learning Continuously** — Updates trader rankings based on realized trade outcomes

## Key Features

- **Consensus-Based Signals** — Only trade when smart money agrees
- **Quality-Filtered Alpha Pool** — Automatically removes noise and bad actors
- **Real-Time Tracking** — Live monitoring of top trader positions
- **Self-Learning System** — Bayesian updates improve rankings over time
- **Multi-Gate Validation** — 5 independent checks before any signal fires
- **Risk-Aware** — Filters out low-confidence and high-cost opportunities

## How It Works

```
Leaderboard → Quality Filter → Alpha Pool → Consensus Detection → Signal
   1000+          7 gates         Top 50         5 gates          Trade
  traders                        traders                         signal
```

Signals only fire when:
- Multiple traders agree on direction
- Traders are statistically independent
- Signal is fresh (not stale)
- Market hasn't moved too far
- Expected value is positive after costs

## Getting Started

```bash
make install     # Install dependencies
cp .env.example .env
make up          # Start all services
make init        # Initialize Alpha Pool with historical data
```

**Dashboard**: http://localhost:4102/dashboard

Run `make help` for all available commands.

> **Windows Users**: Install make via `choco install make` or use npm scripts directly.
> Alternative: `npm install && docker compose up -d && npm run init:alpha-pool`

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/ARCHITECTURE.md) | Technical details, services, and APIs |
| [Development Plan](docs/DEVELOPMENT_PLAN.md) | Roadmap and implementation phases |
| [API Reference](docs/API.md) | REST API and WebSocket documentation |
| [Testing Guide](docs/TESTING.md) | Test suite and coverage |

## Technology

Built with TypeScript, Python, PostgreSQL, NATS, and Docker. Event-driven microservices architecture with real-time WebSocket feeds.

## Status

Core infrastructure is complete. Currently in Phase 3b development—adding Thompson Sampling exploration and dynamic risk inputs.

## License

[PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/) — Free for personal and non-commercial use.

For commercial licensing inquiries, please contact us at [sigmapilot.ai](https://sigmapilot.ai).
