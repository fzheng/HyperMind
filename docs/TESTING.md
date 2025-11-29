# Testing Guide

## Quick Start

```bash
npm test             # Run all 623 tests
```

That's it! One command runs the entire test suite.

## Additional Commands

```bash
npm run test:coverage    # With coverage report
npm run e2e-smoke        # End-to-end smoke test
npm test -- --watch      # Watch mode for development
npm test -- validation   # Run specific test file
```

## Test Coverage

**Overall: 60%** | **ts-lib: 92%**

| Module | Coverage | Description |
|--------|----------|-------------|
| validation.ts | 100% | Address and input validation |
| queue.ts | 100% | WebSocket event queue |
| scoring.ts | 99% | Trader performance scoring |
| hyperliquid.ts | 98% | Hyperliquid API integration |
| persist.ts | 89% | Database operations |
| pagination.ts | 93% | Trade pagination |

## Test Files

| File | Tests | Description |
|------|-------|-------------|
| validation.test.ts | 44 | Input validation |
| scoring.test.ts | 89 | Performance scoring |
| persist.integration.test.ts | 70+ | Database operations |
| hyperliquid.integration.test.ts | 35 | External API calls |
| leaderboard.test.ts | 40 | Leaderboard scoring |
| leaderboard.integration.test.ts | 49 | Cache, rate limiter |
| pagination.test.ts | 30 | Trade deduplication |
| event-queue.test.ts | 15 | Event streaming |
| fill-aggregation.test.ts | 20 | Fill grouping |
| streaming-aggregation.test.ts | 25 | Real-time aggregation |
| position-chain.test.ts | 40+ | Position chain validation |
| dashboard.test.ts | 75 | UI formatting and aggregation |

## Writing Tests

### Basic Structure

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

## Running in CI

Tests run automatically on:
- Pull requests
- Pre-commit hooks (if configured)

## Troubleshooting

### Tests timing out

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
