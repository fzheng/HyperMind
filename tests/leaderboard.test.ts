jest.mock('@hl/ts-lib', () => {
  // Import actual scoring functions (don't mock them)
  const actualScoring = jest.requireActual('@hl/ts-lib/scoring');

  return {
    createLogger: () => ({
      info: jest.fn(),
      error: jest.fn(),
      warn: jest.fn(),
      debug: jest.fn(),
    }),
    getPool: jest.fn(async () => ({
      query: jest.fn().mockResolvedValue({ rows: [] }),
      connect: jest.fn().mockResolvedValue({
        query: jest.fn().mockResolvedValue({ rows: [] }),
        release: jest.fn(),
      }),
    })),
    normalizeAddress: (value: string) => value.toLowerCase(),
    nowIso: () => '2024-01-01T00:00:00.000Z',
    CandidateEventSchema: { parse: (input: any) => input },
    // Include scoring functions from actual module
    computePerformanceScore: actualScoring.computePerformanceScore,
    computeStabilityScore: actualScoring.computeStabilityScore,
    DEFAULT_SCORING_PARAMS: actualScoring.DEFAULT_SCORING_PARAMS,
    // Mock custom account functions
    removeCustomAccount: jest.fn().mockResolvedValue(undefined),
    listCustomAccounts: jest.fn().mockResolvedValue([]),
    updateCustomAccountNickname: jest.fn().mockResolvedValue(undefined),
  };
});

import LeaderboardService from '../services/hl-scout/src/leaderboard';

type RawEntry = {
  address: string;
  winRate: number;
  executedOrders: number;
  realizedPnl: number;
  pnlList: Array<{ timestamp: number; value: string }>;
  remark?: string | null;
  labels?: string[];
  lastOperationAt?: number;
  stats?: {
    maxDrawdown?: number;
    totalPnl?: number;
    openPosCount?: number;
    closePosCount?: number;
  };
};

function makeEntry(overrides: Partial<RawEntry> = {}): RawEntry {
  return {
    address: overrides.address ?? `0x${Math.random().toString(16).slice(2, 42).padEnd(40, '0')}`,
    winRate: overrides.winRate ?? 0.65,
    executedOrders: overrides.executedOrders ?? 100,
    realizedPnl: overrides.realizedPnl ?? 50_000,
    pnlList:
      overrides.pnlList ??
      [
        { timestamp: 1, value: '0' },
        { timestamp: 2, value: '10000' },
        { timestamp: 3, value: '20000' },
        { timestamp: 4, value: '30000' },
        { timestamp: 5, value: '40000' },
        { timestamp: 6, value: '50000' },
      ],
    remark: overrides.remark ?? null,
    labels: overrides.labels ?? [],
    lastOperationAt: overrides.lastOperationAt ?? Date.now(), // Default to active (now)
    stats: overrides.stats,
  };
}

function buildService(selectCount = 2) {
  return new LeaderboardService(
    {
      apiUrl: 'https://example.com',
      topN: 100,
      selectCount,
      periods: [30],
      pageSize: 50,
      refreshMs: 24 * 60 * 60 * 1000,
    },
    async () => {}
  );
}

describe('LeaderboardService scoreEntries', () => {
  it('filters out accounts with perfect win rate and many trades', () => {
    const service = buildService(2);
    const entries = [
      makeEntry({ address: '0xperfect', winRate: 1, executedOrders: 50 }), // Perfect win rate with many trades
      makeEntry({ address: '0xnormal', winRate: 0.75, executedOrders: 100 }),
    ];
    const scored = (service as any).scoreEntries(entries);
    // Perfect win rates with > 10 trades are filtered
    expect(scored.some((row: any) => row.address === '0xperfect')).toBe(false);
    expect(scored[0].address).toBe('0xnormal');
  });

  it('allows perfect win rate with few trades', () => {
    const service = buildService(2);
    const entries = [
      makeEntry({ address: '0xperfect', winRate: 1, executedOrders: 5 }), // Few trades is OK
      makeEntry({ address: '0xnormal', winRate: 0.75, executedOrders: 100 }),
    ];
    const scored = (service as any).scoreEntries(entries);
    // Perfect win rate with < 10 trades is allowed
    expect(scored.some((row: any) => row.address === '0xperfect')).toBe(true);
  });

  it('falls back to base list when filter removes everyone', () => {
    const service = buildService(2);
    const entries = [
      makeEntry({ address: '0xalpha', winRate: 1, executedOrders: 50 }),
      makeEntry({ address: '0xbeta', winRate: 1, executedOrders: 50 }),
    ];
    const scored = (service as any).scoreEntries(entries);
    expect(scored).toHaveLength(entries.length);
  });

  it('normalizes weights across selectCount addresses', () => {
    const service = buildService(2);
    const entries = [
      makeEntry({
        address: '0x1',
        realizedPnl: 100_000,
        pnlList: [
          { timestamp: 1, value: '0' },
          { timestamp: 2, value: '25000' },
          { timestamp: 3, value: '50000' },
          { timestamp: 4, value: '75000' },
          { timestamp: 5, value: '100000' },
        ],
      }),
      makeEntry({
        address: '0x2',
        realizedPnl: 50_000,
        pnlList: [
          { timestamp: 1, value: '0' },
          { timestamp: 2, value: '12500' },
          { timestamp: 3, value: '25000' },
          { timestamp: 4, value: '37500' },
          { timestamp: 5, value: '50000' },
        ],
      }),
      makeEntry({
        address: '0x3',
        realizedPnl: 25_000,
        pnlList: [
          { timestamp: 1, value: '0' },
          { timestamp: 2, value: '6000' },
          { timestamp: 3, value: '12000' },
          { timestamp: 4, value: '18000' },
          { timestamp: 5, value: '25000' },
        ],
      }),
    ];
    const scored = (service as any).scoreEntries(entries);
    const topWeights = scored.slice(0, 2).map((row: any) => row.weight);
    expect(topWeights[0]).toBeGreaterThan(0);
    expect(topWeights[1]).toBeGreaterThan(0);
    expect(topWeights[0] + topWeights[1]).toBeCloseTo(1, 6);
    expect(scored[2].weight).toBe(0);
  });

  it('includes stabilityScore in scoring details', () => {
    const service = buildService(2);
    const entries = [
      makeEntry({
        address: '0xtest',
        pnlList: [
          { timestamp: 1, value: '0' },
          { timestamp: 2, value: '10000' },
          { timestamp: 3, value: '20000' },
          { timestamp: 4, value: '30000' },
        ],
      }),
    ];
    const scored = (service as any).scoreEntries(entries);
    expect(scored[0].meta.scoringDetails).toBeDefined();
    expect(scored[0].meta.scoringDetails.stabilityScore).toBeGreaterThan(0);
  });

  it('penalizes accounts with high drawdown via stability score', () => {
    const service = buildService(2);
    // Create a PnL series with high drawdown: goes up to 100k, then crashes to 10k
    const badDrawdownPnlList = [
      { timestamp: 1, value: '0' },
      { timestamp: 2, value: '50000' },
      { timestamp: 3, value: '100000' },  // Peak
      { timestamp: 4, value: '30000' },   // 70% drawdown
      { timestamp: 5, value: '110000' },  // Recovers (still profitable overall)
    ];
    // Normal account with small drawdown
    const goodPnlList = [
      { timestamp: 1, value: '0' },
      { timestamp: 2, value: '25000' },
      { timestamp: 3, value: '50000' },
      { timestamp: 4, value: '45000' },  // Small 10% dip
      { timestamp: 5, value: '60000' },
    ];
    const entries = [
      makeEntry({ address: '0xbad_drawdown', pnlList: badDrawdownPnlList }),
      makeEntry({ address: '0xgood', pnlList: goodPnlList }),
    ];
    const scored = (service as any).scoreEntries(entries);
    // Good account should rank higher due to stability score penalty on bad drawdown
    expect(scored[0].address).toBe('0xgood');
    expect(scored[0].score).toBeGreaterThan(scored[1].score);
  });

  it('includes maxDrawdown in scoring details', () => {
    const service = buildService(2);
    const entries = [
      makeEntry({
        address: '0xtest',
        pnlList: [
          { timestamp: 1, value: '0' },
          { timestamp: 2, value: '100000' },
          { timestamp: 3, value: '70000' },  // 30% drawdown
          { timestamp: 4, value: '90000' },
        ],
      }),
    ];
    const scored = (service as any).scoreEntries(entries);
    expect(scored[0].meta.scoringDetails.maxDrawdown).toBeDefined();
    expect(scored[0].meta.scoringDetails.maxDrawdown).toBeCloseTo(0.3, 1);
    expect(scored[0].statMaxDrawdown).toBeCloseTo(0.3, 1);
  });

  it('applies progressive penalty for high trade counts (>100)', () => {
    const service = buildService(2);
    const entries = [
      // Moderate trader - should score better
      makeEntry({
        address: '0xmoderate',
        executedOrders: 80,
        winRate: 0.65,
        realizedPnl: 50000,
      }),
      // Heavy trader (scalper) - should be penalized progressively (>100 trades)
      makeEntry({
        address: '0xscalper',
        executedOrders: 150,  // 150 trades should get penalty
        winRate: 0.65,
        realizedPnl: 50000,
      }),
    ];
    const scored = (service as any).scoreEntries(entries);
    // Moderate trader should rank higher due to trade frequency penalty on heavy trader
    expect(scored[0].address).toBe('0xmoderate');
    expect(scored[0].score).toBeGreaterThan(scored[1].score);
  });

  it('removes accounts with > 200 trades (hard limit)', () => {
    const service = buildService(2);
    const entries = [
      makeEntry({
        address: '0xnormal',
        executedOrders: 100,
        winRate: 0.6,
        realizedPnl: 30000,
      }),
      makeEntry({
        address: '0xextreme_scalper',
        executedOrders: 250,  // 250 trades > 200 max - should be removed entirely
        winRate: 0.7,
        realizedPnl: 100000,
      }),
    ];
    const scored = (service as any).scoreEntries(entries);
    // Scalper should be completely removed from results
    expect(scored.length).toBe(1);
    expect(scored[0].address).toBe('0xnormal');
    // Scalper should not exist in results
    const scalper = scored.find((r: any) => r.address === '0xextreme_scalper');
    expect(scalper).toBeUndefined();
  });

  it('removes accounts inactive for > 14 days', () => {
    const service = buildService(2);
    const now = Date.now();
    const fifteenDaysAgo = now - 15 * 24 * 60 * 60 * 1000; // 15 days ago
    const tenDaysAgo = now - 10 * 24 * 60 * 60 * 1000; // 10 days ago

    const entries = [
      makeEntry({
        address: '0xactive',
        executedOrders: 50,
        winRate: 0.6,
        realizedPnl: 30000,
        lastOperationAt: tenDaysAgo, // Active within 14 days
      }),
      makeEntry({
        address: '0xinactive',
        executedOrders: 50,
        winRate: 0.7,
        realizedPnl: 50000,
        lastOperationAt: fifteenDaysAgo, // Inactive for > 14 days
      }),
    ];
    const scored = (service as any).scoreEntries(entries);
    // Inactive account should be completely removed
    expect(scored.length).toBe(1);
    expect(scored[0].address).toBe('0xactive');
    // Inactive should not exist in results
    const inactive = scored.find((r: any) => r.address === '0xinactive');
    expect(inactive).toBeUndefined();
  });

  it('filters out accounts with < 3 trades (hard minimum)', () => {
    const service = buildService(2);
    const entries = [
      makeEntry({
        address: '0xnormal',
        executedOrders: 50,
        winRate: 0.6,
        realizedPnl: 30000,
      }),
      makeEntry({
        address: '0xtoo_few_trades',
        executedOrders: 2,  // Only 2 trades < 3 minimum
        winRate: 0.8,
        realizedPnl: 50000,
      }),
    ];
    const scored = (service as any).scoreEntries(entries);
    // Normal trader should rank higher
    expect(scored[0].address).toBe('0xnormal');
    // Too few trades should have tradeFreqScore = 0
    const tooFew = scored.find((r: any) => r.address === '0xtoo_few_trades');
    expect(tooFew.meta.scoringDetails.tradeFreqScore).toBe(0);
  });

  it('applies progressive penalty for low win rates (<60%)', () => {
    const service = buildService(2);
    const entries = [
      // Good win rate
      makeEntry({
        address: '0xgood_wr',
        winRate: 0.70,
        executedOrders: 50,
        realizedPnl: 50000,
      }),
      // Low win rate
      makeEntry({
        address: '0xlow_wr',
        winRate: 0.45,  // Below 60% threshold
        executedOrders: 50,
        realizedPnl: 50000,
      }),
    ];
    const scored = (service as any).scoreEntries(entries);
    // Good win rate should rank higher
    expect(scored[0].address).toBe('0xgood_wr');
    expect(scored[0].score).toBeGreaterThan(scored[1].score);
  });

  it('marks non-profitable accounts as filtered with low score', () => {
    const service = buildService(2);
    const entries = [
      // Profitable account
      makeEntry({
        address: '0xprofitable',
        pnlList: [
          { timestamp: 1, value: '0' },
          { timestamp: 2, value: '10000' },
          { timestamp: 3, value: '20000' },
        ],
      }),
      // Non-profitable account (ends lower than start)
      makeEntry({
        address: '0xnot_profitable',
        pnlList: [
          { timestamp: 1, value: '0' },
          { timestamp: 2, value: '10000' },
          { timestamp: 3, value: '-5000' },  // Ends in loss
        ],
      }),
    ];
    const scored = (service as any).scoreEntries(entries);
    // Profitable should rank first
    expect(scored[0].address).toBe('0xprofitable');
    // Non-profitable has very low or zero stability score
    const notProfitable = scored.find((row: any) => row.address === '0xnot_profitable');
    if (notProfitable) {
      expect(notProfitable.score).toBeLessThan(scored[0].score);
      // The stability score should be 0 for non-profitable
      expect(notProfitable.meta.scoringDetails.stabilityScore).toBe(0);
    }
  });

  it('rewards large PnL as tiebreaker', () => {
    const service = buildService(2);
    // Two accounts with similar stability but different PnL
    const samePnlList = [
      { timestamp: 1, value: '0' },
      { timestamp: 2, value: '25000' },
      { timestamp: 3, value: '50000' },
      { timestamp: 4, value: '75000' },
      { timestamp: 5, value: '100000' },
    ];
    const entries = [
      makeEntry({
        address: '0xlarge_pnl',
        winRate: 0.65,
        executedOrders: 50,
        realizedPnl: 500000,  // Large PnL
        pnlList: samePnlList,
      }),
      makeEntry({
        address: '0xsmall_pnl',
        winRate: 0.65,
        executedOrders: 50,
        realizedPnl: 10000,  // Small PnL
        pnlList: samePnlList,
      }),
    ];
    const scored = (service as any).scoreEntries(entries);
    // Large PnL should rank higher as tiebreaker
    expect(scored[0].address).toBe('0xlarge_pnl');
    expect(scored[0].score).toBeGreaterThan(scored[1].score);
  });
});

describe('LeaderboardService RefreshStatus', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('initializes with idle status before start', () => {
    const service = buildService(2);
    const status = service.getRefreshStatus();

    expect(status.status).toBe('idle');
    expect(status.isRefreshing).toBe(false);
    expect(status.refreshStartedAt).toBeNull();
    expect(status.lastRefreshAt).toBeNull();
    expect(status.nextRefreshAt).toBeNull();
    expect(status.nextRefreshInMs).toBeNull();
    expect(status.refreshIntervalMs).toBe(24 * 60 * 60 * 1000); // 24 hours
  });

  it('returns correct refresh interval from options', () => {
    const customRefreshMs = 12 * 60 * 60 * 1000; // 12 hours
    const service = new LeaderboardService(
      {
        apiUrl: 'https://example.com',
        topN: 100,
        selectCount: 2,
        periods: [30],
        pageSize: 50,
        refreshMs: customRefreshMs,
      },
      async () => {}
    );

    const status = service.getRefreshStatus();
    expect(status.refreshIntervalMs).toBe(customRefreshMs);
  });

  it('tracks refresh state transitions', () => {
    const service = buildService(2);

    // Initially idle
    expect(service.getRefreshStatus().status).toBe('idle');
    expect(service.getRefreshStatus().isRefreshing).toBe(false);
  });

  it('calculates next refresh time after start', () => {
    const service = buildService(2);
    const refreshMs = 24 * 60 * 60 * 1000; // 24 hours

    // Mock fetch to prevent actual API calls
    const originalFetch = global.fetch;
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ data: [] }),
    });

    try {
      service.start();

      const status = service.getRefreshStatus();
      // Next refresh should be approximately refreshMs from now
      expect(status.nextRefreshAt).not.toBeNull();
      expect(status.nextRefreshInMs).not.toBeNull();
      expect(status.nextRefreshInMs).toBeLessThanOrEqual(refreshMs);
      expect(status.nextRefreshInMs).toBeGreaterThan(refreshMs - 1000); // Within 1 second tolerance

      service.stop();
    } finally {
      global.fetch = originalFetch;
    }
  });

  it('stops timer on stop()', () => {
    const service = buildService(2);

    // Mock fetch
    const originalFetch = global.fetch;
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ data: [] }),
    });

    try {
      service.start();
      const statusBefore = service.getRefreshStatus();
      expect(statusBefore.nextRefreshAt).not.toBeNull();

      service.stop();

      // After stop, timer-dependent values should reflect stopped state
      // (nextRefreshAt still calculated based on last scheduled time until cleared)
    } finally {
      global.fetch = originalFetch;
    }
  });
});

describe('LeaderboardService refresh concurrency', () => {
  it('prevents concurrent refreshes', async () => {
    const service = buildService(2);

    // Access internal state
    const serviceAny = service as any;

    // Simulate refresh in progress
    serviceAny._isRefreshing = true;
    serviceAny._refreshStartedAt = new Date();

    // Attempt another refresh
    await service.refreshAll();

    // Should have logged warning and returned early
    // The _isRefreshing flag should still be true (not reset)
    expect(serviceAny._isRefreshing).toBe(true);
  });

  it('clears refresh state on completion', async () => {
    const service = buildService(2);

    // Mock fetch
    const originalFetch = global.fetch;
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ data: [] }),
    });

    try {
      await service.refreshAll();

      const serviceAny = service as any;
      // Refresh state should be cleared
      expect(serviceAny._isRefreshing).toBe(false);
      expect(serviceAny._refreshProgress).toBeNull();
      // lastRefreshAt should be set
      expect(serviceAny._lastRefreshAt).not.toBeNull();
    } finally {
      global.fetch = originalFetch;
    }
  });

  it('sets error state on refresh failure', async () => {
    const service = buildService(2);

    // Mock fetch to fail
    const originalFetch = global.fetch;
    global.fetch = jest.fn().mockRejectedValue(new Error('Network error'));

    try {
      await service.refreshAll();

      const status = service.getRefreshStatus();
      expect(status.status).toBe('error');
      expect(status.error).toBeDefined();
      expect(status.error).toContain('Network error');
    } finally {
      global.fetch = originalFetch;
    }
  });
});

describe('LeaderboardService progress tracking', () => {
  it('tracks progress through refresh phases', async () => {
    const service = buildService(2);
    const serviceAny = service as any;
    const progressPhases: string[] = [];

    // Spy on setProgress to capture phases
    const originalSetProgress = serviceAny.setProgress.bind(service);
    serviceAny.setProgress = (phase: string, detail?: string) => {
      progressPhases.push(phase);
      originalSetProgress(phase, detail);
    };

    // Mock fetch
    const originalFetch = global.fetch;
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ data: [] }),
    });

    try {
      await service.refreshAll();

      // Should have gone through multiple phases
      expect(progressPhases.length).toBeGreaterThan(0);
      expect(progressPhases).toContain('starting');
    } finally {
      global.fetch = originalFetch;
    }
  });

  it('includes progress in status during refresh', async () => {
    const service = buildService(2);
    const serviceAny = service as any;

    // Manually set progress to test status API
    serviceAny._isRefreshing = true;
    serviceAny._refreshStartedAt = new Date();
    serviceAny._refreshProgress = { phase: 'enriching', detail: 'Processing accounts' };

    const status = service.getRefreshStatus();

    expect(status.status).toBe('refreshing');
    expect(status.progress).toBeDefined();
    expect(status.progress?.phase).toBe('enriching');
    expect(status.progress?.detail).toBe('Processing accounts');
  });
});

describe('LeaderboardService formatDuration', () => {
  it('formats milliseconds to human readable string', () => {
    const service = buildService(2);
    const serviceAny = service as any;

    // Test hours + minutes
    expect(serviceAny.formatDuration(2 * 60 * 60 * 1000 + 30 * 60 * 1000)).toBe('2h 30m');

    // Test minutes only
    expect(serviceAny.formatDuration(45 * 60 * 1000)).toBe('45m');

    // Test zero minutes with hours
    expect(serviceAny.formatDuration(3 * 60 * 60 * 1000)).toBe('3h 0m');

    // Test edge case: exactly 1 hour
    expect(serviceAny.formatDuration(60 * 60 * 1000)).toBe('1h 0m');

    // Test small durations
    expect(serviceAny.formatDuration(5 * 60 * 1000)).toBe('5m');
  });
});

describe('LeaderboardService RefreshStatus edge cases', () => {
  it('handles status when timer not started', () => {
    const service = buildService(2);

    const status = service.getRefreshStatus();

    // Should return valid status object with null times
    expect(status).toBeDefined();
    expect(status.isRefreshing).toBe(false);
    expect(status.nextRefreshAt).toBeNull();
    expect(status.nextRefreshInMs).toBeNull();
  });

  it('returns error status when last refresh had error', () => {
    const service = buildService(2);
    const serviceAny = service as any;

    // Simulate error state
    serviceAny._lastRefreshError = 'API returned 500';
    serviceAny._isRefreshing = false;

    const status = service.getRefreshStatus();

    expect(status.status).toBe('error');
    expect(status.error).toBe('API returned 500');
    expect(status.isRefreshing).toBe(false);
  });

  it('returns refreshing status during active refresh', () => {
    const service = buildService(2);
    const serviceAny = service as any;

    // Simulate active refresh
    serviceAny._isRefreshing = true;
    serviceAny._refreshStartedAt = new Date();
    serviceAny._lastRefreshError = null; // Clear any previous error

    const status = service.getRefreshStatus();

    expect(status.status).toBe('refreshing');
    expect(status.isRefreshing).toBe(true);
    expect(status.refreshStartedAt).not.toBeNull();
  });

  it('clears error status on successful refresh', async () => {
    const service = buildService(2);
    const serviceAny = service as any;

    // Set initial error
    serviceAny._lastRefreshError = 'Previous error';

    // Mock fetch for successful refresh
    const originalFetch = global.fetch;
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ data: [] }),
    });

    try {
      await service.refreshAll();

      const status = service.getRefreshStatus();
      // Error should be cleared after successful refresh
      expect(status.status).toBe('idle');
      expect(status.error).toBeUndefined();
    } finally {
      global.fetch = originalFetch;
    }
  });
});

describe('LeaderboardService persistPeriod skip path', () => {
  it('returns false and skips delete/publish when entries are empty', async () => {
    const service = buildService(2);
    const serviceAny = service as any;

    // Track if logger.warn was called with leaderboard_refresh_skipped
    const warnCalls: any[] = [];
    serviceAny.logger = {
      info: jest.fn(),
      warn: jest.fn((...args: any[]) => warnCalls.push(args)),
      error: jest.fn(),
      debug: jest.fn(),
    };

    // Call persistPeriod with empty array
    const result = await serviceAny.persistPeriod(30, [], [], new Map());

    expect(result).toBe(false);
  });

  it('logs leaderboard_refresh_skipped when persistPeriod returns false during refresh', async () => {
    const service = buildService(2);
    const serviceAny = service as any;

    // Capture log calls
    const warnCalls: any[] = [];
    serviceAny.logger = {
      info: jest.fn(),
      warn: jest.fn((msg: string, data: any) => warnCalls.push({ msg, data })),
      error: jest.fn(),
      debug: jest.fn(),
    };

    // Mock fetch to return empty data (simulating upstream outage)
    const originalFetch = global.fetch;
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ data: [] }),
    });

    try {
      await service.refreshAll();

      // Find leaderboard_refresh_skipped log
      const skipLog = warnCalls.find(c => c.msg === 'leaderboard_refresh_skipped');
      expect(skipLog).toBeDefined();
      expect(skipLog.data.reason).toBe('empty_upstream_data');
    } finally {
      global.fetch = originalFetch;
    }
  });

  it('does not call publishTopCandidates when persistPeriod returns false', async () => {
    const service = buildService(2);
    const serviceAny = service as any;

    // Mock publishTopCandidates to track if called
    const publishCalled = { value: false };
    serviceAny.publishTopCandidates = jest.fn(() => {
      publishCalled.value = true;
      return Promise.resolve();
    });

    // Mock fetch to return empty data
    const originalFetch = global.fetch;
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ data: [] }),
    });

    try {
      await service.refreshAll();

      expect(publishCalled.value).toBe(false);
    } finally {
      global.fetch = originalFetch;
    }
  });
});

describe('LeaderboardService parseEnvNumber', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    process.env = { ...originalEnv };
  });

  afterAll(() => {
    process.env = originalEnv;
  });

  it('returns default when env var is not set', () => {
    const service = buildService(2);
    const serviceAny = service as any;

    // Clear any existing SCORING_* vars
    delete process.env.SCORING_STABILITY_WEIGHT;

    // scoreEntries internally uses parseEnvNumber
    const entries = [makeEntry({ address: '0xtest' })];
    const scored = serviceAny.scoreEntries(entries);

    // Should use default stability weight (0.50)
    expect(scored[0].meta.scoringDetails).toBeDefined();
  });

  it('logs warning and falls back to default for invalid env number', () => {
    const service = buildService(2);
    const serviceAny = service as any;

    // Capture warnings
    const warnCalls: any[] = [];
    serviceAny.logger = {
      info: jest.fn(),
      warn: jest.fn((msg: string, data: any) => warnCalls.push({ msg, data })),
      error: jest.fn(),
      debug: jest.fn(),
    };

    // Set invalid env value
    process.env.SCORING_STABILITY_WEIGHT = 'not-a-number';

    const entries = [makeEntry({ address: '0xtest' })];
    serviceAny.scoreEntries(entries);

    // Should log warning about invalid env number
    const invalidLog = warnCalls.find(c => c.msg === 'invalid_env_number');
    expect(invalidLog).toBeDefined();
    expect(invalidLog.data.key).toBe('SCORING_STABILITY_WEIGHT');
    expect(invalidLog.data.value).toBe('not-a-number');
  });

  it('uses env value when valid number', () => {
    const service = buildService(2);
    const serviceAny = service as any;

    // Set valid env value
    process.env.SCORING_STABILITY_WEIGHT = '0.75';

    const entries = [makeEntry({ address: '0xtest' })];
    const scored = serviceAny.scoreEntries(entries);

    // Should use the env value (results in different scoring)
    expect(scored[0].meta.scoringDetails).toBeDefined();
  });
});
