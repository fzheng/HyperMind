/**
 * Tests for Consensus Gates (ts-lib/consensus.ts)
 *
 * Tests the full consensus detection flow with:
 * - Freshness gates (staleness, price drift)
 * - EV gate with R-unit cost conversion
 * - Integration with episode builder
 */

import {
  Vote,
  ConsensusConfig,
  DEFAULT_CONSENSUS_CONFIG,
  calculateEffectiveK,
  checkSupermajority,
  checkFreshness,
  checkPriceDrift,
  calculateEV,
  estimateWinProbability,
  checkConsensus,
  adaptiveWindowMs,
  createTicketInstrumentation,
  getMedianPrice,
} from '../packages/ts-lib/src/consensus';

describe('Consensus Gates (ts-lib)', () => {
  const config: ConsensusConfig = {
    ...DEFAULT_CONSENSUS_CONFIG,
    minTraders: 3,
    minPct: 0.7,
    minEffectiveK: 2.0,
    maxStalenessFactor: 1.25,
    maxPriceDriftR: 0.25, // Now in R-units (0.25R = 25 bps with 100 bps stop)
    evMinR: 0.2,
  };

  function createVote(
    address: string,
    direction: 'long' | 'short',
    price: number,
    weight: number = 1.0,
    ts: Date = new Date()
  ): Vote {
    return { address, direction, weight, price, ts };
  }

  describe('checkSupermajority', () => {
    it('should pass with 100% agreement and ≥3 traders', () => {
      const votes: Vote[] = [
        createVote('0xa', 'long', 50000),
        createVote('0xb', 'long', 50000),
        createVote('0xc', 'long', 50000),
      ];

      const result = checkSupermajority(votes, config);
      expect(result.passed).toBe(true);
      expect(result.direction).toBe('long');
      expect(result.pct).toBe(1.0);
    });

    it('should pass with 75% agreement (above 70%)', () => {
      const votes: Vote[] = [
        createVote('0xa', 'long', 50000),
        createVote('0xb', 'long', 50000),
        createVote('0xc', 'long', 50000),
        createVote('0xd', 'short', 50000),
      ];

      const result = checkSupermajority(votes, config);
      expect(result.passed).toBe(true);
      expect(result.pct).toBe(0.75);
    });

    it('should fail with 60% agreement (below 70%)', () => {
      const votes: Vote[] = [
        createVote('0xa', 'long', 50000),
        createVote('0xb', 'long', 50000),
        createVote('0xc', 'long', 50000),
        createVote('0xd', 'short', 50000),
        createVote('0xe', 'short', 50000),
      ];

      const result = checkSupermajority(votes, config);
      expect(result.passed).toBe(false);
      expect(result.pct).toBe(0.6);
    });

    it('should fail with only 2 agreeing traders', () => {
      const votes: Vote[] = [
        createVote('0xa', 'long', 50000),
        createVote('0xb', 'long', 50000),
      ];

      const result = checkSupermajority(votes, config);
      expect(result.passed).toBe(false);
    });
  });

  describe('calculateEffectiveK', () => {
    it('should return K for uncorrelated traders (ρ=0)', () => {
      const weights = new Map([
        ['trader1', 1.0],
        ['trader2', 1.0],
        ['trader3', 1.0],
      ]);

      const effK = calculateEffectiveK(weights, new Map(), { defaultCorrelation: 0.0 });
      expect(effK).toBeCloseTo(3.0, 5);
    });

    it('should return 1 for perfectly correlated traders (ρ=1)', () => {
      const weights = new Map([
        ['trader1', 1.0],
        ['trader2', 1.0],
        ['trader3', 1.0],
      ]);

      const correlations = new Map([
        ['trader1|trader2', 1.0],
        ['trader1|trader3', 1.0],
        ['trader2|trader3', 1.0],
      ]);

      // With full correlations provided, shrinkage won't matter as much
      const effK = calculateEffectiveK(weights, correlations, { defaultCorrelation: 1.0 });
      expect(effK).toBeCloseTo(1.0, 5);
    });

    it('should reduce K for partially correlated traders', () => {
      const weights = new Map([
        ['trader1', 1.0],
        ['trader2', 1.0],
        ['trader3', 1.0],
      ]);

      // With ρ=0.3: effK = 3 / (1 + 2×0.3) = 3 / 1.6 = 1.875
      const effK = calculateEffectiveK(weights, new Map(), { defaultCorrelation: 0.3 });
      expect(effK).toBeCloseTo(1.875, 2);
    });

    it('should handle 5 traders with 80% correlation', () => {
      const weights = new Map([
        ['t1', 1.0],
        ['t2', 1.0],
        ['t3', 1.0],
        ['t4', 1.0],
        ['t5', 1.0],
      ]);

      // effK = 5 / (1 + 4×0.8) = 5 / 4.2 ≈ 1.19
      const effK = calculateEffectiveK(weights, new Map(), { defaultCorrelation: 0.8 });
      expect(effK).toBeCloseTo(1.19, 1);
    });
  });

  describe('checkFreshness', () => {
    const windowMs = 120000; // 2 minutes

    it('should pass when all votes are fresh', () => {
      const now = Date.now();
      const votes: Vote[] = [
        createVote('0xa', 'long', 50000, 1.0, new Date(now - 30000)), // 30s ago
        createVote('0xb', 'long', 50000, 1.0, new Date(now - 60000)), // 60s ago
      ];

      const result = checkFreshness(votes, windowMs, config);
      expect(result.passed).toBe(true);
      expect(result.staleness).toBeLessThan(1.0);
    });

    it('should fail when oldest vote is too stale', () => {
      const now = Date.now();
      const votes: Vote[] = [
        createVote('0xa', 'long', 50000, 1.0, new Date(now - 30000)), // 30s ago
        createVote('0xb', 'long', 50000, 1.0, new Date(now - 180000)), // 3min ago (1.5x window)
      ];

      const result = checkFreshness(votes, windowMs, config);
      expect(result.passed).toBe(false);
      expect(result.staleness).toBeGreaterThan(config.maxStalenessFactor);
    });

    it('should pass at exactly max staleness', () => {
      const now = Date.now();
      const maxAge = windowMs * config.maxStalenessFactor;
      const votes: Vote[] = [
        createVote('0xa', 'long', 50000, 1.0, new Date(now - maxAge + 1000)),
      ];

      const result = checkFreshness(votes, windowMs, config);
      expect(result.passed).toBe(true);
    });
  });

  describe('checkPriceDrift', () => {
    const stopBps = 100; // 1% stop

    it('should pass when price has not drifted (within 0.25R)', () => {
      const votes: Vote[] = [
        createVote('0xa', 'long', 50000),
        createVote('0xb', 'long', 50010),
        createVote('0xc', 'long', 49990),
      ];

      // Median is 50000, current is 50005 → 0.01% = 1 bps → 0.01R
      const currentMid = 50005;
      const result = checkPriceDrift(votes, currentMid, stopBps, config);
      expect(result.passed).toBe(true);
      expect(result.driftR).toBeLessThan(config.maxPriceDriftR);
    });

    it('should fail when price has drifted too far (> 0.25R)', () => {
      const votes: Vote[] = [
        createVote('0xa', 'long', 50000),
        createVote('0xb', 'long', 50000),
      ];

      // 0.5% drift = 50 bps = 0.5R (above 0.25R threshold)
      const currentMid = 50250;
      const result = checkPriceDrift(votes, currentMid, stopBps, config);
      expect(result.passed).toBe(false);
      expect(result.driftR).toBeGreaterThan(config.maxPriceDriftR);
    });

    it('should calculate drift in R-units correctly', () => {
      const votes: Vote[] = [createVote('0xa', 'long', 50000)];

      // 0.5% drift = 50 bps = 0.5R with 100 bps stop
      const currentMid = 50250;
      const result = checkPriceDrift(votes, currentMid, stopBps, config);

      // driftR = 50 bps / 100 bps = 0.5 R
      expect(result.driftR).toBeCloseTo(0.5, 2);
    });
  });

  describe('calculateEV', () => {
    const stopBps = 100; // 1% stop

    it('should calculate positive EV for high win rate', () => {
      const pWin = 0.7;
      const ev = calculateEV(pWin, stopBps, config);

      // Gross: 0.7 × 0.5 - 0.3 × 0.3 = 0.35 - 0.09 = 0.26
      expect(ev.evGrossR).toBeCloseTo(0.26, 2);

      // Cost: 17 bps / 100 bps = 0.17 R
      expect(ev.evCostR).toBeCloseTo(0.17, 2);

      // Net: 0.26 - 0.17 = 0.09
      expect(ev.evNetR).toBeCloseTo(0.09, 2);
    });

    it('should calculate negative EV for low win rate', () => {
      const pWin = 0.3;
      const ev = calculateEV(pWin, stopBps, config);

      // Gross: 0.3 × 0.5 - 0.7 × 0.3 = 0.15 - 0.21 = -0.06
      expect(ev.evGrossR).toBeCloseTo(-0.06, 2);
      expect(ev.evNetR).toBeLessThan(0);
    });

    it('should have higher cost impact with tighter stop', () => {
      const pWin = 0.6;
      const tightStopBps = 50; // 0.5% stop
      const ev = calculateEV(pWin, tightStopBps, config);

      // Cost: 17 bps / 50 bps = 0.34 R
      expect(ev.evCostR).toBeCloseTo(0.34, 2);
    });
  });

  describe('estimateWinProbability', () => {
    it('should return 0.5 when no data', () => {
      const votes: Vote[] = [createVote('0xa', 'long', 50000)];
      const pWin = estimateWinProbability(votes, new Map());
      expect(pWin).toBe(0.5);
    });

    it('should weight by trader samples', () => {
      const votes: Vote[] = [
        createVote('0xa', 'long', 50000),
        createVote('0xb', 'long', 50000),
      ];

      const winRates = new Map([
        ['0xa', { winRate: 0.8, samples: 100 }],
        ['0xb', { winRate: 0.6, samples: 10 }],
      ]);

      const pWin = estimateWinProbability(votes, winRates);
      // With shrinkage toward 0.5 and sample weighting, should be between 0.5 and 0.75
      expect(pWin).toBeGreaterThan(0.5);
      expect(pWin).toBeLessThan(0.75);
    });
  });

  describe('getMedianPrice', () => {
    it('should return median for odd count', () => {
      const votes: Vote[] = [
        createVote('0xa', 'long', 50000),
        createVote('0xb', 'long', 51000),
        createVote('0xc', 'long', 52000),
      ];

      expect(getMedianPrice(votes)).toBe(51000);
    });

    it('should return average of middle two for even count', () => {
      const votes: Vote[] = [
        createVote('0xa', 'long', 50000),
        createVote('0xb', 'long', 51000),
        createVote('0xc', 'long', 52000),
        createVote('0xd', 'long', 53000),
      ];

      expect(getMedianPrice(votes)).toBe(51500);
    });
  });

  describe('adaptiveWindowMs', () => {
    const baseWindow = 120000;

    it('should use shorter window in low volatility', () => {
      const window = adaptiveWindowMs(0.1, baseWindow);
      expect(window).toBe(baseWindow * 0.5);
    });

    it('should use base window in normal volatility', () => {
      const window = adaptiveWindowMs(0.5, baseWindow);
      expect(window).toBe(baseWindow);
    });

    it('should use longer window in high volatility', () => {
      const window = adaptiveWindowMs(0.9, baseWindow);
      expect(window).toBe(baseWindow * 3.0);
    });
  });

  describe('checkConsensus (full flow)', () => {
    it('should pass when all gates pass', () => {
      const now = Date.now();
      const votes: Vote[] = [
        createVote('0xa', 'long', 50000, 1.0, new Date(now - 30000)),
        createVote('0xb', 'long', 50002, 1.0, new Date(now - 40000)),
        createVote('0xc', 'long', 50001, 1.0, new Date(now - 50000)),
        createVote('0xd', 'long', 50003, 1.0, new Date(now - 60000)),
      ];

      // High win rates with lots of samples to overcome shrinkage
      const traderWinRates = new Map([
        ['0xa', { winRate: 0.85, samples: 100 }],
        ['0xb', { winRate: 0.82, samples: 100 }],
        ['0xc', { winRate: 0.80, samples: 100 }],
        ['0xd', { winRate: 0.78, samples: 100 }],
      ]);

      // 4 uncorrelated traders with excellent win rates
      const result = checkConsensus(
        votes,
        50002, // currentMidPrice
        120000, // windowMs
        100, // stopBps (1%)
        new Map(), // No correlation info → use 0 for max effK
        traderWinRates,
        {
          ...config,
          defaultCorrelation: 0,
          evMinR: 0.0, // Relaxed EV threshold
          avgWinR: 0.5,
          avgLossR: 0.3,
        }
      );

      expect(result.gateResults.supermajority.passed).toBe(true);
      expect(result.gateResults.effectiveK.passed).toBe(true);
      expect(result.gateResults.freshness.passed).toBe(true);
      expect(result.gateResults.priceDrift.passed).toBe(true);
      expect(result.gateResults.ev.passed).toBe(true);

      expect(result.direction).toBe('long');
      expect(result.passes).toBe(true);
    });

    it('should fail when traders are highly correlated', () => {
      const now = Date.now();
      const votes: Vote[] = [
        createVote('0xa', 'long', 50000, 1.0, new Date(now)),
        createVote('0xb', 'long', 50000, 1.0, new Date(now)),
        createVote('0xc', 'long', 50000, 1.0, new Date(now)),
      ];

      const traderWinRates = new Map([
        ['0xa', { winRate: 0.7, samples: 50 }],
        ['0xb', { winRate: 0.7, samples: 50 }],
        ['0xc', { winRate: 0.7, samples: 50 }],
      ]);

      // All traders 90% correlated
      const correlations = new Map([
        ['0xa|0xb', 0.9],
        ['0xa|0xc', 0.9],
        ['0xb|0xc', 0.9],
      ]);

      const result = checkConsensus(
        votes,
        50000,
        120000,
        100,
        correlations,
        traderWinRates,
        config
      );

      // With shrinkage: ρ' = λ×0.9 + (1-λ)×ρ_base = 0.7×0.9 + 0.3×0.3 = 0.72
      // effK = 3 / (1 + 2×0.72) = 3 / 2.44 ≈ 1.23 < 2.0
      expect(result.passes).toBe(false);
      expect(result.gateResults.effectiveK.passed).toBe(false);
      expect(result.gateResults.effectiveK.value).toBeCloseTo(1.23, 1);
    });

    it('should fail when price has drifted', () => {
      const now = Date.now();
      const votes: Vote[] = [
        createVote('0xa', 'long', 50000, 1.0, new Date(now)),
        createVote('0xb', 'long', 50000, 1.0, new Date(now)),
        createVote('0xc', 'long', 50000, 1.0, new Date(now)),
        createVote('0xd', 'long', 50000, 1.0, new Date(now)),
      ];

      const result = checkConsensus(
        votes,
        50200, // Current mid 0.4% away = 40 bps drift = 0.4R (above 0.25R threshold)
        120000,
        100,
        new Map(),
        new Map(),
        { ...config, defaultCorrelation: 0 }
      );

      expect(result.passes).toBe(false);
      expect(result.gateResults.priceDrift.passed).toBe(false);
      // 40 bps drift / 100 bps stop = 0.4 R
      expect(result.gateResults.priceDrift.driftR).toBeCloseTo(0.4, 1);
    });
  });

  describe('createTicketInstrumentation', () => {
    it('should create instrumentation for passing consensus', () => {
      const now = Date.now();
      const votes: Vote[] = [
        createVote('0xa', 'long', 50000, 1.0, new Date(now)),
        createVote('0xb', 'long', 50000, 1.0, new Date(now)),
        createVote('0xc', 'long', 50000, 1.0, new Date(now)),
        createVote('0xd', 'long', 50000, 1.0, new Date(now)),
      ];

      // High win rates to ensure EV gate passes
      const traderWinRates = new Map([
        ['0xa', { winRate: 0.85, samples: 100 }],
        ['0xb', { winRate: 0.85, samples: 100 }],
        ['0xc', { winRate: 0.85, samples: 100 }],
        ['0xd', { winRate: 0.85, samples: 100 }],
      ]);

      const result = checkConsensus(
        votes,
        50000,
        120000,
        100,
        new Map(),
        traderWinRates,
        { ...config, defaultCorrelation: 0, evMinR: 0.0 }
      );

      expect(result.passes).toBe(true); // Verify consensus passes first

      const ticket = createTicketInstrumentation(result, 120000, 100);

      expect(ticket).not.toBeNull();
      expect(ticket!.nTraders).toBe(4);
      expect(ticket!.nAgree).toBe(4);
      expect(ticket!.effectiveK).toBe(4);
      expect(ticket!.direction).toBe('long');
      expect(ticket!.voterAddresses).toHaveLength(4);
    });

    it('should return null for failing consensus', () => {
      const votes: Vote[] = [
        createVote('0xa', 'long', 50000),
        createVote('0xb', 'short', 50000),
      ];

      const result = checkConsensus(
        votes,
        50000,
        120000,
        100,
        new Map(),
        new Map(),
        config
      );

      const ticket = createTicketInstrumentation(result, 120000, 100);
      expect(ticket).toBeNull();
    });
  });
});
