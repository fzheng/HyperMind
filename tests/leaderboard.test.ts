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
    executedOrders: overrides.executedOrders ?? 10,
    realizedPnl: overrides.realizedPnl ?? 1_000_000,
    pnlList:
      overrides.pnlList ??
      [
        { timestamp: 1, value: '0' },
        { timestamp: 2, value: '100000' },
        { timestamp: 3, value: '200000' },
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
  it('filters out accounts with perfect win rate', () => {
    const service = buildService(2);
    const entries = [
      makeEntry({ address: '0xperfect', winRate: 1 }),
      makeEntry({ address: '0xnormal', winRate: 0.75 }),
    ];
    const scored = (service as any).scoreEntries(entries);
    expect(scored.some((row: any) => row.address === '0xperfect')).toBe(false);
    expect(scored[0].address).toBe('0xnormal');
  });

  it('falls back to base list when filter removes everyone', () => {
    const service = buildService(2);
    const entries = [
      makeEntry({ address: '0xalpha', winRate: 1 }),
      makeEntry({ address: '0xbeta', winRate: 1 }),
    ];
    const scored = (service as any).scoreEntries(entries);
    expect(scored).toHaveLength(entries.length);
  });

  it('normalizes weights across selectCount addresses', () => {
    const service = buildService(2);
    const entries = [
      makeEntry({ address: '0x1', realizedPnl: 5_000_000 }),
      makeEntry({ address: '0x2', realizedPnl: 2_500_000 }),
      makeEntry({ address: '0x3', realizedPnl: 1_000_000 }),
    ];
    const scored = (service as any).scoreEntries(entries);
    const topWeights = scored.slice(0, 2).map((row: any) => row.weight);
    expect(topWeights[0]).toBeGreaterThan(0);
    expect(topWeights[1]).toBeGreaterThan(0);
    expect(topWeights[0] + topWeights[1]).toBeCloseTo(1, 6);
    expect(scored[2].weight).toBe(0);
  });
});
