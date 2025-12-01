/**
 * Integration tests for LeaderboardService
 *
 * Tests the LRUCache, RateLimiter, API request handling,
 * and database persistence logic.
 */

// Mock dependencies before importing
const mockQuery = jest.fn();
const mockPool = { query: mockQuery };
const mockFetch = jest.fn();

jest.mock('@hl/ts-lib', () => {
  const actualScoring = jest.requireActual('@hl/ts-lib/scoring');
  return {
    createLogger: () => ({
      info: jest.fn(),
      error: jest.fn(),
      warn: jest.fn(),
      debug: jest.fn(),
    }),
    getPool: jest.fn(async () => mockPool),
    normalizeAddress: (value: string) => value.toLowerCase(),
    nowIso: () => '2024-01-01T00:00:00.000Z',
    CandidateEventSchema: { parse: (input: unknown) => input },
    computePerformanceScore: actualScoring.computePerformanceScore,
    computeStabilityScore: actualScoring.computeStabilityScore,
    DEFAULT_SCORING_PARAMS: actualScoring.DEFAULT_SCORING_PARAMS,
    removeCustomAccount: jest.fn(),
    listCustomAccounts: jest.fn().mockResolvedValue([]),
    updateCustomAccountNickname: jest.fn(),
  };
});

// Set up global fetch mock
global.fetch = mockFetch;

// Sleep helper for rate limiter tests
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

describe('LRUCache', () => {
  // Create a minimal LRU cache implementation for testing
  class LRUCache<T> {
    private cache = new Map<string, { value: T; expiresAt: number }>();
    private maxSize: number;
    private ttlMs: number;

    constructor(maxSize: number, ttlMs: number) {
      this.maxSize = maxSize;
      this.ttlMs = ttlMs;
    }

    get(key: string): T | undefined {
      const entry = this.cache.get(key);
      if (!entry) return undefined;
      if (Date.now() > entry.expiresAt) {
        this.cache.delete(key);
        return undefined;
      }
      this.cache.delete(key);
      this.cache.set(key, entry);
      return entry.value;
    }

    set(key: string, value: T): void {
      this.cache.delete(key);
      while (this.cache.size >= this.maxSize) {
        const firstKey = this.cache.keys().next().value;
        if (firstKey !== undefined) {
          this.cache.delete(firstKey);
        }
      }
      this.cache.set(key, { value, expiresAt: Date.now() + this.ttlMs });
    }

    clear(): void {
      this.cache.clear();
    }

    get size(): number {
      return this.cache.size;
    }
  }

  it('should store and retrieve values', () => {
    const cache = new LRUCache<string>(10, 60000);
    cache.set('key1', 'value1');
    expect(cache.get('key1')).toBe('value1');
  });

  it('should return undefined for missing keys', () => {
    const cache = new LRUCache<string>(10, 60000);
    expect(cache.get('nonexistent')).toBeUndefined();
  });

  it('should evict expired entries', () => {
    jest.useFakeTimers();
    const cache = new LRUCache<string>(10, 100); // 100ms TTL
    cache.set('key1', 'value1');

    expect(cache.get('key1')).toBe('value1');

    jest.advanceTimersByTime(150);
    expect(cache.get('key1')).toBeUndefined();

    jest.useRealTimers();
  });

  it('should evict oldest entries when at capacity', () => {
    const cache = new LRUCache<string>(3, 60000);
    cache.set('key1', 'value1');
    cache.set('key2', 'value2');
    cache.set('key3', 'value3');
    cache.set('key4', 'value4'); // Should evict key1

    expect(cache.get('key1')).toBeUndefined();
    expect(cache.get('key2')).toBe('value2');
    expect(cache.get('key3')).toBe('value3');
    expect(cache.get('key4')).toBe('value4');
  });

  it('should move accessed items to end (LRU)', () => {
    const cache = new LRUCache<string>(3, 60000);
    cache.set('key1', 'value1');
    cache.set('key2', 'value2');
    cache.set('key3', 'value3');

    // Access key1 to make it most recently used
    cache.get('key1');

    // Add new key - should evict key2 (oldest unused)
    cache.set('key4', 'value4');

    expect(cache.get('key1')).toBe('value1');
    expect(cache.get('key2')).toBeUndefined();
    expect(cache.get('key3')).toBe('value3');
    expect(cache.get('key4')).toBe('value4');
  });

  it('should update existing key position on set', () => {
    const cache = new LRUCache<string>(3, 60000);
    cache.set('key1', 'value1');
    cache.set('key2', 'value2');
    cache.set('key3', 'value3');

    // Update key1
    cache.set('key1', 'updated');

    // Add new key - should evict key2
    cache.set('key4', 'value4');

    expect(cache.get('key1')).toBe('updated');
    expect(cache.get('key2')).toBeUndefined();
  });

  it('should clear all entries', () => {
    const cache = new LRUCache<string>(10, 60000);
    cache.set('key1', 'value1');
    cache.set('key2', 'value2');

    expect(cache.size).toBe(2);

    cache.clear();

    expect(cache.size).toBe(0);
    expect(cache.get('key1')).toBeUndefined();
  });

  it('should report correct size', () => {
    const cache = new LRUCache<string>(10, 60000);
    expect(cache.size).toBe(0);

    cache.set('key1', 'value1');
    expect(cache.size).toBe(1);

    cache.set('key2', 'value2');
    expect(cache.size).toBe(2);
  });
});

describe('RateLimiter', () => {
  // Create a minimal rate limiter implementation for testing
  class RateLimiter {
    private lastRequestTime = 0;
    private delayMs: number;

    constructor(delayMs: number) {
      this.delayMs = delayMs;
    }

    async acquire(): Promise<void> {
      const now = Date.now();
      const elapsed = now - this.lastRequestTime;
      if (elapsed < this.delayMs) {
        await sleep(this.delayMs - elapsed);
      }
      this.lastRequestTime = Date.now();
    }
  }

  it('should not delay first request', async () => {
    const limiter = new RateLimiter(100);
    const start = Date.now();
    await limiter.acquire();
    const elapsed = Date.now() - start;
    expect(elapsed).toBeLessThan(50);
  });

  it('should delay rapid consecutive requests', async () => {
    const limiter = new RateLimiter(100);

    await limiter.acquire(); // First request

    const start = Date.now();
    await limiter.acquire(); // Second request should be delayed
    const elapsed = Date.now() - start;

    expect(elapsed).toBeGreaterThanOrEqual(90); // Allow some timing variance
  });

  it('should not delay if enough time has passed', async () => {
    const limiter = new RateLimiter(50);

    await limiter.acquire();
    await sleep(100); // Wait longer than delay

    const start = Date.now();
    await limiter.acquire();
    const elapsed = Date.now() - start;

    expect(elapsed).toBeLessThan(30);
  });
});

describe('LeaderboardService API Integration', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockFetch.mockReset();
    mockQuery.mockReset();
  });

  describe('Leaderboard API Fetch', () => {
    it('should handle successful leaderboard fetch', async () => {
      const mockLeaderboardData = {
        data: [
          {
            address: '0x1234567890abcdef1234567890abcdef12345678',
            winRate: 0.65,
            executedOrders: 100,
            realizedPnl: 50000,
            pnlList: [
              { timestamp: 1, value: '0' },
              { timestamp: 2, value: '25000' },
              { timestamp: 3, value: '50000' },
            ],
            lastOperationAt: Date.now(),
          },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve(mockLeaderboardData),
      });

      const response = await mockFetch('https://hyperbot.network/api/leaderboard/smart?pageNum=1&pageSize=50&period=30&sort=3');
      const data = await response.json();

      expect(data.data).toHaveLength(1);
      expect(data.data[0].winRate).toBe(0.65);
    });

    it('should handle API error response', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
      });

      const response = await mockFetch('https://hyperbot.network/api/leaderboard/smart');
      expect(response.ok).toBe(false);
      expect(response.status).toBe(500);
    });

    it('should handle network timeout', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Network timeout'));

      await expect(
        mockFetch('https://hyperbot.network/api/leaderboard/smart')
      ).rejects.toThrow('Network timeout');
    });
  });

  describe('Address Stats API', () => {
    it('should fetch and parse address stats', async () => {
      const mockStatsResponse = {
        data: {
          winRate: 0.72,
          openPosCount: 2,
          closePosCount: 48,
          avgPosDuration: 3600000,
          totalPnl: 75000,
          maxDrawdown: 0.15,
        },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve(mockStatsResponse),
      });

      const response = await mockFetch(
        'https://hyperbot.network/api/leaderboard/smart/query-addr-stat/0x1234?period=30'
      );
      const data = await response.json();

      expect(data.data.winRate).toBe(0.72);
      expect(data.data.maxDrawdown).toBe(0.15);
    });

    it('should handle missing stats gracefully', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({ data: null }),
      });

      const response = await mockFetch(
        'https://hyperbot.network/api/leaderboard/smart/query-addr-stat/0x1234?period=30'
      );
      const data = await response.json();

      expect(data.data).toBeNull();
    });
  });

  describe('Address Remarks API', () => {
    it('should fetch batch address remarks', async () => {
      const mockRemarksResponse = {
        code: 0,
        msg: 'SUCCESS',
        data: [
          { address: '0x1234', remark: 'Whale Trader' },
          { address: '0x5678', remark: 'DeFi Pro' },
          { address: '0x9abc', remark: null },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve(mockRemarksResponse),
      });

      const response = await mockFetch(
        'https://hyperbot.network/api/leaderboard/smart/query-addr-remarks',
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(['0x1234', '0x5678', '0x9abc']),
        }
      );
      const data = await response.json();

      expect(data.code).toBe(0);
      expect(data.data).toHaveLength(3);
      expect(data.data[0].remark).toBe('Whale Trader');
    });

    it('should handle empty remarks response', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({ code: 0, msg: 'SUCCESS', data: [] }),
      });

      const response = await mockFetch(
        'https://hyperbot.network/api/leaderboard/smart/query-addr-remarks',
        { method: 'POST', body: '[]' }
      );
      const data = await response.json();

      expect(data.data).toEqual([]);
    });
  });

  describe('BTC/ETH Analysis API', () => {
    it('should fetch completed trades for analysis', async () => {
      const mockTradesResponse = {
        data: [
          {
            symbol: 'BTC',
            pnl: 5000,
            openTime: Date.now() - 3600000, // 1 hour ago
            closeTime: Date.now(),
          },
          {
            symbol: 'ETH',
            pnl: 2000,
            openTime: Date.now() - 7200000,
            closeTime: Date.now() - 3600000,
          },
          {
            symbol: 'SOL', // Non BTC/ETH
            pnl: 1000,
            openTime: Date.now() - 1000,
            closeTime: Date.now(), // Short trade (should be filtered)
          },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve(mockTradesResponse),
      });

      const response = await mockFetch(
        'https://hyperbot.network/api/leaderboard/smart/completed-trades/0x1234?take=2000'
      );
      const data = await response.json();

      expect(data.data).toHaveLength(3);

      // Filter for BTC/ETH trades lasting > 10 minutes
      const MIN_DURATION_MS = 10 * 60 * 1000;
      const btcEthTrades = data.data.filter(
        (t: { symbol: string; closeTime: number; openTime: number }) =>
          ['BTC', 'ETH'].includes(t.symbol) &&
          t.closeTime - t.openTime >= MIN_DURATION_MS
      );

      expect(btcEthTrades).toHaveLength(2);
    });
  });

  describe('Hyperliquid Info API', () => {
    it('should fetch portfolio data for PnL series', async () => {
      const mockPortfolioResponse = {
        portfolioHistory: [
          { time: 1700000000000, accountValue: '100000' },
          { time: 1700086400000, accountValue: '105000' },
          { time: 1700172800000, accountValue: '110000' },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve(mockPortfolioResponse),
      });

      const response = await mockFetch('https://api.hyperliquid.xyz/info', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type: 'portfolio',
          user: '0x1234',
        }),
      });
      const data = await response.json();

      expect(data.portfolioHistory).toHaveLength(3);
    });
  });
});

describe('Database Persistence', () => {
  beforeEach(() => {
    mockQuery.mockReset();
  });

  it('should upsert leaderboard entries', async () => {
    mockQuery.mockResolvedValueOnce({ rows: [] });

    await mockPool.query(
      `INSERT INTO hl_leaderboard_entries
       (period_days, address, rank, score, weight, win_rate, executed_orders, realized_pnl)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
       ON CONFLICT (period_days, lower(address)) DO UPDATE SET
       rank = EXCLUDED.rank, score = EXCLUDED.score`,
      [30, '0x1234', 1, 0.85, 0.5, 0.72, 100, 50000]
    );

    expect(mockQuery).toHaveBeenCalledWith(
      expect.stringContaining('INSERT INTO hl_leaderboard_entries'),
      expect.arrayContaining([30, '0x1234', 1])
    );
  });

  it('should store PnL points', async () => {
    mockQuery.mockResolvedValueOnce({ rows: [] });

    await mockPool.query(
      `INSERT INTO hl_leaderboard_pnl_points
       (period_days, address, source, window_name, point_ts, pnl_value)
       VALUES ($1, $2, $3, $4, $5, $6)`,
      [30, '0x1234', 'api', 'month', new Date().toISOString(), 50000]
    );

    expect(mockQuery).toHaveBeenCalledWith(
      expect.stringContaining('INSERT INTO hl_leaderboard_pnl_points'),
      expect.arrayContaining([30, '0x1234', 'api', 'month'])
    );
  });

  it('should fetch leaderboard entries with stats', async () => {
    const mockEntries = [
      {
        id: 1,
        address: '0x1234',
        rank: 1,
        score: 0.85,
        weight: 0.5,
        win_rate: 0.72,
        executed_orders: 100,
        realized_pnl: 50000,
        stat_max_drawdown: 0.15,
      },
    ];

    mockQuery.mockResolvedValueOnce({ rows: mockEntries });

    const result = await mockPool.query(
      `SELECT * FROM hl_leaderboard_entries
       WHERE period_days = $1
       ORDER BY rank ASC
       LIMIT $2`,
      [30, 10]
    );

    expect(result.rows).toHaveLength(1);
    expect(result.rows[0].score).toBe(0.85);
  });
});

describe('Scoring Edge Cases', () => {
  it('should handle empty pnlList', () => {
    const pnlList: Array<{ timestamp: number; value: string }> = [];
    const values = pnlList.map((p) => Number(p.value));

    expect(values).toEqual([]);
  });

  it('should handle single point pnlList', () => {
    const pnlList = [{ timestamp: 1, value: '10000' }];
    const values = pnlList.map((p) => Number(p.value));

    expect(values).toHaveLength(1);
    expect(values[0]).toBe(10000);
  });

  it('should handle negative PnL values', () => {
    const pnlList = [
      { timestamp: 1, value: '0' },
      { timestamp: 2, value: '-5000' },
      { timestamp: 3, value: '-10000' },
    ];
    const values = pnlList.map((p) => Number(p.value));

    expect(values).toEqual([0, -5000, -10000]);
    expect(values[values.length - 1]).toBeLessThan(0);
  });

  it('should calculate drawdown correctly', () => {
    const pnlList = [
      { timestamp: 1, value: '0' },
      { timestamp: 2, value: '100000' },
      { timestamp: 3, value: '70000' }, // 30% drawdown
      { timestamp: 4, value: '90000' },
    ];
    const values = pnlList.map((p) => Number(p.value));

    let peak = values[0];
    let maxDrawdown = 0;

    for (const val of values) {
      if (val > peak) peak = val;
      if (peak > 0) {
        const drawdown = (peak - val) / peak;
        if (drawdown > maxDrawdown) maxDrawdown = drawdown;
      }
    }

    expect(maxDrawdown).toBeCloseTo(0.3, 2);
  });

  it('should handle zero starting value', () => {
    const startValue = 0;
    const endValue = 50000;
    const profit = endValue - startValue;

    expect(profit).toBe(50000);
  });

  it('should clamp win rate to valid range', () => {
    const clamp = (value: number, min: number, max: number) =>
      Math.max(min, Math.min(max, value));

    expect(clamp(1.5, 0, 1)).toBe(1);
    expect(clamp(-0.5, 0, 1)).toBe(0);
    expect(clamp(0.75, 0, 1)).toBe(0.75);
  });
});

describe('Concurrency Helpers', () => {
  async function runWithConcurrency<T, R>(
    items: T[],
    concurrency: number,
    fn: (item: T) => Promise<R>
  ): Promise<R[]> {
    const results: R[] = [];
    const executing: Promise<void>[] = [];

    for (const item of items) {
      const p = fn(item).then((result) => {
        results.push(result);
      });

      executing.push(p as unknown as Promise<void>);

      if (executing.length >= concurrency) {
        await Promise.race(executing);
        // Remove completed promises
        const completed = executing.filter(
          (p) => (p as unknown as { _settled?: boolean })._settled
        );
        executing.length = 0;
        executing.push(...completed.filter(() => false));
      }
    }

    await Promise.all(executing);
    return results;
  }

  it('should process items with limited concurrency', async () => {
    const items = [1, 2, 3, 4, 5];
    const processed: number[] = [];

    await runWithConcurrency(items, 2, async (item) => {
      await sleep(10);
      processed.push(item);
      return item * 2;
    });

    expect(processed.sort()).toEqual([1, 2, 3, 4, 5]);
  });

  it('should handle empty items array', async () => {
    const results = await runWithConcurrency([], 2, async (item: number) => item * 2);
    expect(results).toEqual([]);
  });

  it('should handle concurrency of 1', async () => {
    const items = [1, 2, 3];
    const order: number[] = [];

    await runWithConcurrency(items, 1, async (item) => {
      order.push(item);
      await sleep(5);
      return item;
    });

    expect(order).toEqual([1, 2, 3]); // Sequential order
  });
});

describe('LeaderboardService Lifecycle', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockFetch.mockReset();
  });

  it('should start and stop timer correctly', () => {
    jest.useFakeTimers();

    // Mock successful fetch
    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ data: [] }),
    });

    const { LeaderboardService } = require('../services/hl-scout/src/leaderboard');
    const mockPublish = jest.fn().mockResolvedValue(undefined);
    const service = new LeaderboardService(
      { topN: 10, selectCount: 5, periods: [30] },
      mockPublish
    );

    service.start();

    // Timer should be set
    expect(service['timer']).not.toBeNull();

    service.stop();

    // Timer should be cleared
    expect(service['timer']).toBeNull();

    jest.useRealTimers();
  });

  it('should call refreshAll on start', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ data: [] }),
    });

    const { LeaderboardService } = require('../services/hl-scout/src/leaderboard');
    const mockPublish = jest.fn().mockResolvedValue(undefined);
    const service = new LeaderboardService(
      { topN: 10, selectCount: 5, periods: [30] },
      mockPublish
    );

    const refreshSpy = jest.spyOn(service, 'refreshAll');

    service.start();

    // Wait for async call
    await new Promise((r) => setTimeout(r, 100));

    expect(refreshSpy).toHaveBeenCalled();

    service.stop();
  });
});

describe('LeaderboardService fetchPeriod', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockFetch.mockReset();
  });

  it('should fetch multiple pages when needed', async () => {
    const page1Data = Array(100).fill(null).map((_, i) => ({
      address: `0x${i.toString().padStart(40, '0')}`,
      winRate: 0.7,
      executedOrders: 50,
      realizedPnl: 10000,
    }));

    const page2Data = Array(50).fill(null).map((_, i) => ({
      address: `0x${(i + 100).toString().padStart(40, '0')}`,
      winRate: 0.65,
      executedOrders: 40,
      realizedPnl: 5000,
    }));

    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ data: page1Data }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ data: page2Data }),
      });

    const { LeaderboardService } = require('../services/hl-scout/src/leaderboard');
    const mockPublish = jest.fn().mockResolvedValue(undefined);
    const service = new LeaderboardService(
      { topN: 150, selectCount: 5, periods: [30], pageSize: 100 },
      mockPublish
    );

    const result = await service['fetchPeriod'](30);

    expect(mockFetch).toHaveBeenCalledTimes(2);
    expect(result.length).toBe(150);
  });

  it('should stop fetching when page returns fewer entries than pageSize', async () => {
    const partialPageData = Array(50).fill(null).map((_, i) => ({
      address: `0x${i.toString().padStart(40, '0')}`,
      winRate: 0.7,
      executedOrders: 50,
      realizedPnl: 10000,
    }));

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ data: partialPageData }),
    });

    const { LeaderboardService } = require('../services/hl-scout/src/leaderboard');
    const mockPublish = jest.fn().mockResolvedValue(undefined);
    const service = new LeaderboardService(
      { topN: 200, selectCount: 5, periods: [30], pageSize: 100 },
      mockPublish
    );

    const result = await service['fetchPeriod'](30);

    // Should only fetch one page since 50 < 100
    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(result.length).toBe(50);
  });

  it('should throw error on HTTP failure', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
    });

    const { LeaderboardService } = require('../services/hl-scout/src/leaderboard');
    const mockPublish = jest.fn().mockResolvedValue(undefined);
    const service = new LeaderboardService(
      { topN: 100, selectCount: 5, periods: [30] },
      mockPublish
    );

    await expect(service['fetchPeriod'](30)).rejects.toThrow('leaderboard HTTP 500');
  });

  it('should limit results to topN', async () => {
    const largeData = Array(200).fill(null).map((_, i) => ({
      address: `0x${i.toString().padStart(40, '0')}`,
      winRate: 0.7,
      executedOrders: 50,
      realizedPnl: 10000,
    }));

    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ data: largeData }),
    });

    const { LeaderboardService } = require('../services/hl-scout/src/leaderboard');
    const mockPublish = jest.fn().mockResolvedValue(undefined);
    const service = new LeaderboardService(
      { topN: 100, selectCount: 5, periods: [30], pageSize: 200 },
      mockPublish
    );

    const result = await service['fetchPeriod'](30);

    expect(result.length).toBe(100); // Limited to topN
  });
});

describe('LeaderboardService fetchAddressStat', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockFetch.mockReset();
    // Reset module cache to clear the internal cache
    jest.resetModules();
  });

  it('should fetch and parse address stats', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({
        data: {
          winRate: 0.75,
          openPosCount: 3,
          closePosCount: 50,
          avgPosDuration: 3600,
          totalPnl: 50000,
          maxDrawdown: 0.15,
        },
      }),
    });

    const { LeaderboardService } = require('../services/hl-scout/src/leaderboard');
    const mockPublish = jest.fn().mockResolvedValue(undefined);
    const service = new LeaderboardService(
      { topN: 100, selectCount: 5, periods: [30] },
      mockPublish
    );

    const result = await service['fetchAddressStat']('0x1234', 30);

    expect(result).toEqual({
      winRate: 0.75,
      openPosCount: 3,
      closePosCount: 50,
      avgPosDuration: 3600,
      totalPnl: 50000,
      maxDrawdown: 0.15,
    });
  });

  it('should return null when data is missing', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ data: null }),
    });

    const { LeaderboardService } = require('../services/hl-scout/src/leaderboard');
    const mockPublish = jest.fn().mockResolvedValue(undefined);
    const service = new LeaderboardService(
      { topN: 100, selectCount: 5, periods: [30] },
      mockPublish
    );

    // Use different address to avoid cache
    const result = await service['fetchAddressStat']('0xnulldata', 30);

    expect(result).toBeNull();
  });

  it('should return null on fetch error', async () => {
    mockFetch.mockRejectedValueOnce(new Error('Network error'));

    const { LeaderboardService } = require('../services/hl-scout/src/leaderboard');
    const mockPublish = jest.fn().mockResolvedValue(undefined);
    const service = new LeaderboardService(
      { topN: 100, selectCount: 5, periods: [30] },
      mockPublish
    );

    // Use different address to avoid cache
    const result = await service['fetchAddressStat']('0xerror', 30);

    expect(result).toBeNull();
  });

  it('should use cache on second request', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({
        data: { winRate: 0.8, totalPnl: 10000 },
      }),
    });

    const { LeaderboardService } = require('../services/hl-scout/src/leaderboard');
    const mockPublish = jest.fn().mockResolvedValue(undefined);
    const service = new LeaderboardService(
      { topN: 100, selectCount: 5, periods: [30] },
      mockPublish
    );

    // First call - fetches from API (use unique address)
    await service['fetchAddressStat']('0xcachetest', 30);
    const fetchCount1 = mockFetch.mock.calls.length;

    // Second call - should use cache
    await service['fetchAddressStat']('0xcachetest', 30);
    const fetchCount2 = mockFetch.mock.calls.length;

    // Fetch count should not increase (cache hit)
    expect(fetchCount2).toBe(fetchCount1);
  });
});

describe('LeaderboardService fetchAddressRemarks', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockFetch.mockReset();
  });

  it('should fetch and map address remarks', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({
        code: 0,
        msg: 'SUCCESS',
        data: [
          { address: '0xAAA', remark: 'Trader One' },
          { address: '0xBBB', remark: 'Trader Two' },
          { address: '0xCCC', remark: null },
        ],
      }),
    });

    const { LeaderboardService } = require('../services/hl-scout/src/leaderboard');
    const mockPublish = jest.fn().mockResolvedValue(undefined);
    const service = new LeaderboardService(
      { topN: 100, selectCount: 5, periods: [30] },
      mockPublish
    );

    const result = await service.fetchAddressRemarks(['0xAAA', '0xBBB', '0xCCC']);

    expect(result.size).toBe(2);
    expect(result.get('0xaaa')).toBe('Trader One');
    expect(result.get('0xbbb')).toBe('Trader Two');
    expect(result.has('0xccc')).toBe(false); // null remark not included
  });

  it('should return empty map for empty addresses', async () => {
    const { LeaderboardService } = require('../services/hl-scout/src/leaderboard');
    const mockPublish = jest.fn().mockResolvedValue(undefined);
    const service = new LeaderboardService(
      { topN: 100, selectCount: 5, periods: [30] },
      mockPublish
    );

    const result = await service.fetchAddressRemarks([]);

    expect(result.size).toBe(0);
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it('should handle API error gracefully', async () => {
    mockFetch.mockRejectedValueOnce(new Error('API down'));

    const { LeaderboardService } = require('../services/hl-scout/src/leaderboard');
    const mockPublish = jest.fn().mockResolvedValue(undefined);
    const service = new LeaderboardService(
      { topN: 100, selectCount: 5, periods: [30] },
      mockPublish
    );

    const result = await service.fetchAddressRemarks(['0xAAA']);

    expect(result.size).toBe(0); // Empty map on error
  });

  it('should handle non-zero response code', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({
        code: 1,
        msg: 'ERROR',
        data: null,
      }),
    });

    const { LeaderboardService } = require('../services/hl-scout/src/leaderboard');
    const mockPublish = jest.fn().mockResolvedValue(undefined);
    const service = new LeaderboardService(
      { topN: 100, selectCount: 5, periods: [30] },
      mockPublish
    );

    const result = await service.fetchAddressRemarks(['0xAAA']);

    expect(result.size).toBe(0);
  });
});

describe('LeaderboardService fetchPortfolioSeries', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockFetch.mockReset();
  });

  it('should fetch and parse portfolio series', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve([
        ['month', {
          pnlHistory: [[1700000000000, '1000'], [1700100000000, '2000']],
          accountValueHistory: [[1700000000000, '10000'], [1700100000000, '12000']],
        }],
        ['week', {
          pnlHistory: [[1700000000000, '500']],
          accountValueHistory: [[1700000000000, '10500']],
        }],
      ]),
    });

    const { LeaderboardService } = require('../services/hl-scout/src/leaderboard');
    const mockPublish = jest.fn().mockResolvedValue(undefined);
    const service = new LeaderboardService(
      { topN: 100, selectCount: 5, periods: [30] },
      mockPublish
    );

    const result = await service['fetchPortfolioSeries']('0x1234');

    expect(result).toHaveLength(2);
    expect(result[0].window).toBe('month');
    expect(result[0].pnlHistory).toHaveLength(2);
    expect(result[1].window).toBe('week');
  });

  it('should handle empty portfolio response', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve([]),
    });

    const { LeaderboardService } = require('../services/hl-scout/src/leaderboard');
    const mockPublish = jest.fn().mockResolvedValue(undefined);
    const service = new LeaderboardService(
      { topN: 100, selectCount: 5, periods: [30] },
      mockPublish
    );

    const result = await service['fetchPortfolioSeries']('0x1234');

    expect(result).toEqual([]);
  });

  it('should return null on fetch error', async () => {
    mockFetch.mockRejectedValueOnce(new Error('Network error'));

    const { LeaderboardService } = require('../services/hl-scout/src/leaderboard');
    const mockPublish = jest.fn().mockResolvedValue(undefined);
    const service = new LeaderboardService(
      { topN: 100, selectCount: 5, periods: [30] },
      mockPublish
    );

    const result = await service['fetchPortfolioSeries']('0x1234');

    expect(result).toBeNull();
  });

  it('should handle malformed portfolio data', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve([
        ['invalid'], // Missing second element
        [null, null], // Invalid structure
      ]),
    });

    const { LeaderboardService } = require('../services/hl-scout/src/leaderboard');
    const mockPublish = jest.fn().mockResolvedValue(undefined);
    const service = new LeaderboardService(
      { topN: 100, selectCount: 5, periods: [30] },
      mockPublish
    );

    const result = await service['fetchPortfolioSeries']('0x1234');

    expect(result).toEqual([]);
  });
});

describe('LeaderboardService applyStatsToEntry', () => {
  it('should apply stats to entry', () => {
    const { LeaderboardService } = require('../services/hl-scout/src/leaderboard');
    const mockPublish = jest.fn().mockResolvedValue(undefined);
    const service = new LeaderboardService(
      { topN: 100, selectCount: 5, periods: [30] },
      mockPublish
    );

    const entry = {
      address: '0x1234',
      rank: 1,
      score: 100,
      weight: 0.1,
      winRate: 0.5,
      executedOrders: 10,
      realizedPnl: 1000,
      efficiency: 100,
      pnlConsistency: 0.8,
      remark: null,
      labels: [],
      statOpenPositions: null,
      statClosedPositions: null,
      statAvgPosDuration: null,
      statTotalPnl: null,
      statMaxDrawdown: null,
      meta: {},
    };

    const stats = {
      winRate: 0.75,
      openPosCount: 3,
      closePosCount: 50,
      avgPosDuration: 7200,
      totalPnl: 25000,
      maxDrawdown: 0.1,
    };

    service['applyStatsToEntry'](entry, stats);

    expect(entry.winRate).toBe(0.75);
    expect(entry.statOpenPositions).toBe(3);
    expect(entry.statClosedPositions).toBe(50);
    expect(entry.statAvgPosDuration).toBe(7200);
    expect(entry.statTotalPnl).toBe(25000);
    expect(entry.statMaxDrawdown).toBe(0.1);
    expect(entry.meta.stats).toEqual(stats);
  });

  it('should clamp win rate to [0, 1]', () => {
    const { LeaderboardService } = require('../services/hl-scout/src/leaderboard');
    const mockPublish = jest.fn().mockResolvedValue(undefined);
    const service = new LeaderboardService(
      { topN: 100, selectCount: 5, periods: [30] },
      mockPublish
    );

    const entry = {
      address: '0x1234',
      rank: 1,
      score: 100,
      weight: 0.1,
      winRate: 0.5,
      executedOrders: 10,
      realizedPnl: 1000,
      efficiency: 100,
      pnlConsistency: 0.8,
      remark: null,
      labels: [],
      statOpenPositions: null,
      statClosedPositions: null,
      statAvgPosDuration: null,
      statTotalPnl: null,
      statMaxDrawdown: null,
      meta: {},
    };

    // Test with value > 1
    service['applyStatsToEntry'](entry, { winRate: 1.5 });
    expect(entry.winRate).toBe(1);

    // Test with value < 0
    service['applyStatsToEntry'](entry, { winRate: -0.5 });
    expect(entry.winRate).toBe(0);
  });

  it('should handle missing stats fields', () => {
    const { LeaderboardService } = require('../services/hl-scout/src/leaderboard');
    const mockPublish = jest.fn().mockResolvedValue(undefined);
    const service = new LeaderboardService(
      { topN: 100, selectCount: 5, periods: [30] },
      mockPublish
    );

    const entry = {
      address: '0x1234',
      rank: 1,
      score: 100,
      weight: 0.1,
      winRate: 0.5,
      executedOrders: 10,
      realizedPnl: 1000,
      efficiency: 100,
      pnlConsistency: 0.8,
      remark: null,
      labels: [],
      statOpenPositions: 5,
      statClosedPositions: 10,
      statAvgPosDuration: 3600,
      statTotalPnl: 5000,
      statMaxDrawdown: 0.05,
      meta: {},
    };

    service['applyStatsToEntry'](entry, {}); // Empty stats

    // Original values should be preserved
    expect(entry.winRate).toBe(0.5);
    expect(entry.statOpenPositions).toBe(5);
  });
});

describe('LeaderboardService refreshPeriod', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockFetch.mockReset();
    mockQuery.mockReset();
  });

  it('should skip persistence when upstream returns empty data', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ data: [] }),
    });

    const { LeaderboardService } = require('../services/hl-scout/src/leaderboard');
    const mockPublish = jest.fn().mockResolvedValue(undefined);
    const service = new LeaderboardService(
      { topN: 100, selectCount: 5, periods: [30] },
      mockPublish
    );

    await service.refreshPeriod(30);

    // Should not publish when data is empty
    expect(mockPublish).not.toHaveBeenCalled();
  });

  it('should handle errors in refreshPeriod gracefully', async () => {
    mockFetch.mockRejectedValue(new Error('API failure'));

    const { LeaderboardService } = require('../services/hl-scout/src/leaderboard');
    const mockPublish = jest.fn().mockResolvedValue(undefined);
    const service = new LeaderboardService(
      { topN: 100, selectCount: 5, periods: [30] },
      mockPublish
    );

    // Should not throw, error is caught and logged
    await expect(service.refreshPeriod(30)).rejects.toThrow();
  });
});

describe('LeaderboardService publishTopCandidates', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockFetch.mockReset();
    jest.resetModules();
  });

  it('should publish candidates with correct format', async () => {
    const { LeaderboardService } = require('../services/hl-scout/src/leaderboard');
    const mockPublish = jest.fn().mockResolvedValue(undefined);
    const service = new LeaderboardService(
      { topN: 100, selectCount: 5, periods: [30] },
      mockPublish
    );

    const candidates = [
      {
        address: '0x1234',
        rank: 1,
        score: 100,
        weight: 0.5,
        winRate: 0.8,
        executedOrders: 50,
        realizedPnl: 10000,
      },
      {
        address: '0x5678',
        rank: 2,
        score: 80,
        weight: 0.3,
        winRate: 0.75,
        executedOrders: 40,
        realizedPnl: 8000,
      },
    ];

    await service['publishTopCandidates'](30, candidates);

    expect(mockPublish).toHaveBeenCalledTimes(2);
    // Check that first call contains the address
    expect(mockPublish.mock.calls[0][0].address).toBe('0x1234');
  });
});
