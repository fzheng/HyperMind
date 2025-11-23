jest.mock('@hl/ts-lib', () => {
  // Import actual scoring functions (don't mock them)
  const actualScoring = jest.requireActual('@hl/ts-lib/scoring');

  return {
    createLogger: () => ({
      info: jest.fn(),
      error: jest.fn(),
      warn: jest.fn(),
    }),
    getPool: jest.fn(async () => ({
      query: jest.fn().mockResolvedValue({ rows: [] }),
    })),
    normalizeAddress: (value: string) => value.toLowerCase(),
    nowIso: () => '2024-01-01T00:00:00.000Z',
    CandidateEventSchema: { parse: (input: any) => input },
    // Include scoring functions from actual module
    computePerformanceScore: actualScoring.computePerformanceScore,
    computeSmoothPnlScore: actualScoring.computeSmoothPnlScore,
    DEFAULT_SCORING_PARAMS: actualScoring.DEFAULT_SCORING_PARAMS,
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
};

function makeEntry(overrides: Partial<RawEntry> = {}): RawEntry {
  return {
    address: overrides.address ?? `0x${Math.random().toString(16).slice(2, 42).padEnd(40, '0')}`,
    winRate: overrides.winRate ?? 0.65,
    executedOrders: overrides.executedOrders ?? 100, // Higher trade count for better freq score
    realizedPnl: overrides.realizedPnl ?? 50_000, // More reasonable PnL
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

  it('includes smoothPnlScore in scoring details', () => {
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
    expect(scored[0].meta.scoringDetails.smoothPnlScore).toBeGreaterThan(0);
  });
});
