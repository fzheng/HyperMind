# Testing Guide

## Quick Start

```bash
npm run test:unit     # Run Jest unit tests only (830 tests)
npm run test:e2e      # Run Playwright e2e tests only
npm test              # Run both Jest + Playwright
```

## Test Suites

### Unit Tests (Jest)

```bash
npm run test:unit           # Run all Jest tests
npm run test:coverage       # With coverage report
npm test -- --watch         # Watch mode for development
npx jest tests/validation   # Run specific test file
```

**Test Count**: 830 tests across 20 test files

### E2E Tests (Playwright)

```bash
npm run test:e2e            # Run Playwright headless
npm run test:e2e:headed     # Run with visible browser
npm run test:e2e:ui         # Open Playwright UI
npm run test:e2e:report     # View last test report
```

**Prerequisites**:
- Dashboard server running at `http://localhost:4102` (or set `DASHBOARD_URL`)
- Chromium browser installed: `npx playwright install chromium`

**Test Isolation**: E2E tests are designed to be **read-only** and do not modify server-side state. Tests that need to verify mutation behavior use API mocking (via `page.route()`) to intercept calls and prevent database changes.

## Test Coverage

**Overall: 76%** | **ts-lib: 95%**

| Module | Coverage | Description |
|--------|----------|-------------|
| env.ts | 100% | Environment variable utilities |
| validation.ts | 100% | Address and input validation |
| queue.ts | 100% | WebSocket event queue |
| utils.ts | 100% | Utility functions (retry, clamp, sleep) |
| pagination.ts | 100% | Trade pagination and deduplication |
| scoring.ts | 99% | Trader performance scoring |
| hyperliquid.ts | 98% | Hyperliquid API integration |
| persist.ts | 90% | Database operations |

## Test Files

### Jest Unit Tests (`tests/`)

| File | Tests | Description |
|------|-------|-------------|
| validation.test.ts | 44 | Input validation |
| scoring.test.ts | 89 | Performance scoring |
| persist.integration.test.ts | 150+ | Database operations, transactions |
| utils.test.ts | 93 | Utility functions |
| hyperliquid.integration.test.ts | 35 | External API calls |
| leaderboard.test.ts | 50+ | Leaderboard scoring, skip paths |
| leaderboard.integration.test.ts | 56 | Cache, rate limiter, API integration |
| pagination.test.ts | 12 | Trade deduplication |
| event-queue.test.ts | 15 | Event streaming |
| fill-aggregation.test.ts | 20 | Fill grouping |
| streaming-aggregation.test.ts | 25 | Real-time aggregation |
| position-chain.test.ts | 40+ | Position chain validation |
| dashboard.test.ts | 100+ | UI formatting, limit clamping, price API |
| pnl-calculations.test.ts | 50+ | PnL calculations, signal generation |
| backfill.test.ts | 30+ | Backfill validation, trade dedup |
| env.test.ts | 15 | Environment variable parsing |
| realtime-tracker.test.ts | 30+ | WebSocket tracking, custom accounts |

### Playwright E2E Tests (`e2e/`)

| File | Tests | Description |
|------|-------|-------------|
| dashboard.spec.ts | 76 | Core dashboard layout, theme, time toggle, responsive |
| pinned-accounts.spec.ts | 32 | Pin UI elements, styling, interactions (mocked) |
| resilience.spec.ts | 26 | Error handling, empty states, API failures |

**Total**: 134 tests across 3 files (67 chromium + 67 mobile-chrome)

## Writing Tests

### Jest Unit Test Structure

```typescript
describe('Feature', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should do something', () => {
    const result = functionUnderTest(input);
    expect(result).toBe(expected);
  });
});
```

### Mocking Database

```typescript
const mockQuery = jest.fn();
jest.mock('../packages/ts-lib/src/postgres', () => ({
  getPool: () => Promise.resolve({ query: mockQuery }),
}));
```

### Mocking External APIs

```typescript
const mockFetch = jest.fn();
global.fetch = mockFetch;

mockFetch.mockResolvedValueOnce({
  ok: true,
  json: () => Promise.resolve({ data: [] }),
});
```

### Playwright E2E Test Structure

```typescript
import { test, expect } from '@playwright/test';

test.describe('Feature', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard');
  });

  test('should display element', async ({ page }) => {
    await expect(page.locator('[data-testid="element"]')).toBeVisible();
  });
});
```

### Playwright API Mocking (Prevents DB Changes)

```typescript
test('should handle API interaction', async ({ page }) => {
  // Intercept API calls to prevent real database changes
  await page.route('**/admin/addresses/**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ success: true }),
    });
  });

  await page.goto('/dashboard');
  // Test interactions - API calls are mocked
});
```

## Running E2E Tests

### Prerequisites

1. **Install Chromium**:
   ```bash
   npx playwright install chromium
   ```

2. **Start Dashboard Server**:
   ```bash
   docker compose up -d hl-stream
   # Or
   npm run dev:stream
   ```

3. **Run Tests**:
   ```bash
   npm run test:e2e
   ```

### Configuration

Playwright config is in `playwright.config.ts`:
- **Base URL**: `DASHBOARD_URL` env var or `http://localhost:4102`
- **Browsers**: Chromium (desktop) + Mobile Chrome
- **Screenshots**: On failure only
- **Traces**: On first retry

### Viewing Reports

After running tests:
```bash
npm run test:e2e:report
```

This opens the HTML report in your browser at `playwright-report/`.

## Data-TestId Selectors

The dashboard uses `data-testid` attributes for stable test selectors:

```html
<!-- Examples -->
<header data-testid="header">
<div data-testid="leaderboard-table">
<button data-testid="add-custom-btn">
<input data-testid="custom-address-input">
```

Use these in Playwright tests:
```typescript
page.locator('[data-testid="header"]')
page.locator('[data-testid="leaderboard-table"]')
```

## Running in CI

Tests run automatically on:
- Pull requests
- Pre-commit hooks (if configured)

**CI Note**: For Playwright tests, ensure the dashboard service is running or use the `webServer` option in `playwright.config.ts` to start it automatically.

## Troubleshooting

### Jest tests timing out

Increase timeout in `jest.setup.ts`:
```typescript
jest.setTimeout(30000);
```

### Mock not resetting

Add to test file:
```typescript
beforeEach(() => {
  jest.clearAllMocks();
});
```

### Console noise in tests

Mock console.error:
```typescript
beforeAll(() => {
  console.error = jest.fn();
});
```

### Playwright can't connect to dashboard

1. Check server is running: `curl http://localhost:4102/healthz`
2. Set custom URL: `DASHBOARD_URL=http://localhost:3000 npm run test:e2e`

### E2E tests modifying real data

If you see database changes from e2e tests, check:
1. Tests should use `page.route()` to mock mutation APIs
2. Tests should NOT click buttons that modify state without mocking
3. Review tests in `pinned-accounts.spec.ts` for examples
