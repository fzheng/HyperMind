/**
 * Tests for Consensus Detection Module
 *
 * Tests cover:
 * 1. One vote per trader (collapse fills)
 * 2. Dispersion gate (supermajority)
 * 3. Effective-K calculation with correlation
 * 4. Latency and price band gates
 * 5. EV calculation with costs
 */

describe('Consensus Detection', () => {
  // Configuration constants (matching Python defaults)
  const CONSENSUS_MIN_TRADERS = 3;
  const CONSENSUS_MIN_AGREEING = 3;
  const CONSENSUS_MIN_PCT = 0.7;
  const CONSENSUS_MIN_EFFECTIVE_K = 2.0;
  const CONSENSUS_MAX_STALENESS_FACTOR = 1.25;
  const CONSENSUS_MAX_PRICE_BAND_BPS = 8.0;
  const CONSENSUS_EV_MIN_R = 0.2;

  const DEFAULT_AVG_WIN_R = 0.5;
  const DEFAULT_AVG_LOSS_R = 0.3;
  const DEFAULT_FEES_BPS = 7.0;
  const DEFAULT_SLIP_BPS = 10.0;
  const DEFAULT_CORRELATION = 0.3;

  interface Fill {
    fillId: string;
    address: string;
    asset: string;
    side: string;
    size: number;
    price: number;
    ts: Date;
  }

  interface Vote {
    address: string;
    direction: string;
    weight: number;
    price: number;
    ts: Date;
  }

  // Helper: collapse fills to votes
  function collapseToVotes(fills: Fill[], weightCap: number = 1.0): Vote[] {
    const byTrader: Map<string, Fill[]> = new Map();

    for (const f of fills) {
      const addr = f.address.toLowerCase();
      if (!byTrader.has(addr)) {
        byTrader.set(addr, []);
      }
      byTrader.get(addr)!.push(f);
    }

    const votes: Vote[] = [];
    for (const [addr, traderFills] of byTrader) {
      let netDelta = 0;
      for (const f of traderFills) {
        const signedSize =
          f.side.toLowerCase() === 'long' || f.side.toLowerCase() === 'buy'
            ? f.size
            : -f.size;
        netDelta += signedSize;
      }

      if (Math.abs(netDelta) < 1e-9) continue;

      const direction = netDelta > 0 ? 'long' : 'short';
      const weight = Math.min(Math.abs(netDelta) / weightCap, 1.0);

      // Weighted average price
      const totalSize = traderFills.reduce((sum, f) => sum + Math.abs(f.size), 0);
      const avgPrice =
        totalSize > 0
          ? traderFills.reduce(
              (sum, f) => sum + f.price * Math.abs(f.size),
              0
            ) / totalSize
          : traderFills[traderFills.length - 1].price;

      const latestTs = new Date(
        Math.max(...traderFills.map((f) => f.ts.getTime()))
      );

      votes.push({
        address: addr,
        direction,
        weight,
        price: avgPrice,
        ts: latestTs,
      });
    }

    return votes;
  }

  // Helper: check dispersion gate
  function passesConsensusGates(
    directions: string[],
    minAgreeing: number = CONSENSUS_MIN_AGREEING,
    minPct: number = CONSENSUS_MIN_PCT
  ): { passes: boolean; majorityDir: string } {
    if (directions.length === 0) {
      return { passes: false, majorityDir: '' };
    }

    const longCount = directions.filter((d) => d === 'long').length;
    const shortCount = directions.length - longCount;

    const majorityCount = Math.max(longCount, shortCount);
    const majorityDir = longCount >= shortCount ? 'long' : 'short';

    if (majorityCount < minAgreeing) {
      return { passes: false, majorityDir: '' };
    }

    if (majorityCount / directions.length < minPct) {
      return { passes: false, majorityDir: '' };
    }

    return { passes: true, majorityDir };
  }

  // Helper: effective K from correlation
  function effKFromCorr(
    weights: Map<string, number>,
    correlationMatrix: Map<string, number>,
    defaultCorr: number = DEFAULT_CORRELATION
  ): number {
    const addrs = Array.from(weights.keys());
    if (addrs.length <= 1) {
      return addrs.length;
    }

    const sumWeights = Array.from(weights.values()).reduce((a, b) => a + b, 0);
    const num = sumWeights ** 2;

    let den = 0;
    for (let i = 0; i < addrs.length; i++) {
      for (let j = 0; j < addrs.length; j++) {
        let rho: number;
        if (i === j) {
          rho = 1.0;
        } else {
          const key = [addrs[i], addrs[j]].sort().join('|');
          rho = correlationMatrix.get(key) ?? defaultCorr;
          rho = Math.max(0, Math.min(1, rho));
        }
        den += weights.get(addrs[i])! * weights.get(addrs[j])! * rho;
      }
    }

    return num / Math.max(den, 1e-9);
  }

  // Helper: bps to R conversion
  function bpsToR(entryPx: number, stopPx: number, bps: number): number {
    if (entryPx <= 0) return 0;
    const stopBps = (Math.abs(entryPx - stopPx) / entryPx) * 10000;
    return bps / Math.max(stopBps, 1);
  }

  // Helper: calculate EV
  function calculateEV(
    pWin: number,
    entryPx: number,
    stopPx: number,
    avgWinR: number = DEFAULT_AVG_WIN_R,
    avgLossR: number = DEFAULT_AVG_LOSS_R,
    feesBps: number = DEFAULT_FEES_BPS,
    slipBps: number = DEFAULT_SLIP_BPS
  ): { evGrossR: number; evCostR: number; evNetR: number } {
    const grossEv = pWin * avgWinR - (1 - pWin) * avgLossR;
    const totalBps = feesBps + slipBps;
    const costR = bpsToR(entryPx, stopPx, totalBps);
    const netEv = grossEv - costR;

    return {
      evGrossR: grossEv,
      evCostR: costR,
      evNetR: netEv,
    };
  }

  describe('one vote per trader', () => {
    it('should collapse multiple fills from same trader to one vote', () => {
      const fills: Fill[] = [
        {
          fillId: '1',
          address: '0xABC',
          asset: 'BTC',
          side: 'buy',
          size: 0.5,
          price: 50000,
          ts: new Date(),
        },
        {
          fillId: '2',
          address: '0xABC',
          asset: 'BTC',
          side: 'buy',
          size: 0.3,
          price: 50100,
          ts: new Date(),
        },
        {
          fillId: '3',
          address: '0xDEF',
          asset: 'BTC',
          side: 'buy',
          size: 1.0,
          price: 50050,
          ts: new Date(),
        },
      ];

      const votes = collapseToVotes(fills);

      // Should have 2 votes (one per trader)
      expect(votes.length).toBe(2);

      // Find ABC's vote
      const abcVote = votes.find((v) => v.address === '0xabc');
      expect(abcVote).toBeDefined();
      expect(abcVote!.direction).toBe('long');
      // Net size is 0.8, weighted avg price
    });

    it('should cancel out opposing fills from same trader', () => {
      const fills: Fill[] = [
        {
          fillId: '1',
          address: '0xABC',
          asset: 'BTC',
          side: 'buy',
          size: 1.0,
          price: 50000,
          ts: new Date(),
        },
        {
          fillId: '2',
          address: '0xABC',
          asset: 'BTC',
          side: 'sell',
          size: 1.0,
          price: 50100,
          ts: new Date(),
        },
      ];

      const votes = collapseToVotes(fills);

      // Should have 0 votes (net delta is 0)
      expect(votes.length).toBe(0);
    });

    it('should determine direction by net position', () => {
      const fills: Fill[] = [
        {
          fillId: '1',
          address: '0xABC',
          asset: 'BTC',
          side: 'buy',
          size: 1.0,
          price: 50000,
          ts: new Date(),
        },
        {
          fillId: '2',
          address: '0xABC',
          asset: 'BTC',
          side: 'sell',
          size: 0.3,
          price: 50100,
          ts: new Date(),
        },
      ];

      const votes = collapseToVotes(fills);

      expect(votes.length).toBe(1);
      expect(votes[0].direction).toBe('long'); // Net +0.7
    });
  });

  describe('dispersion gate', () => {
    it('should pass with 100% agreement', () => {
      const directions = ['long', 'long', 'long'];
      const result = passesConsensusGates(directions);

      expect(result.passes).toBe(true);
      expect(result.majorityDir).toBe('long');
    });

    it('should pass with 80% agreement (above 70% threshold)', () => {
      const directions = ['long', 'long', 'long', 'long', 'short'];
      const result = passesConsensusGates(directions);

      expect(result.passes).toBe(true);
      expect(result.majorityDir).toBe('long');
    });

    it('should fail with 60% agreement (below 70% threshold)', () => {
      const directions = ['long', 'long', 'long', 'short', 'short'];
      const result = passesConsensusGates(directions);

      expect(result.passes).toBe(false);
    });

    it('should fail with too few agreeing traders', () => {
      const directions = ['long', 'long']; // Only 2 agreeing
      const result = passesConsensusGates(directions);

      expect(result.passes).toBe(false);
    });

    it('should handle empty directions', () => {
      const directions: string[] = [];
      const result = passesConsensusGates(directions);

      expect(result.passes).toBe(false);
    });
  });

  describe('effective-K calculation', () => {
    it('should return K for uncorrelated traders', () => {
      const weights = new Map([
        ['trader1', 1.0],
        ['trader2', 1.0],
        ['trader3', 1.0],
      ]);
      const correlations = new Map<string, number>(); // No correlations = use default

      // With ρ=0.3, effK should be less than K
      const effK = effKFromCorr(weights, correlations, 0.0);

      // With ρ=0, effK should equal K
      expect(effK).toBeCloseTo(3.0, 5);
    });

    it('should return ~1 for perfectly correlated traders', () => {
      const weights = new Map([
        ['trader1', 1.0],
        ['trader2', 1.0],
        ['trader3', 1.0],
      ]);

      // All pairs perfectly correlated
      const correlations = new Map([
        ['trader1|trader2', 1.0],
        ['trader1|trader3', 1.0],
        ['trader2|trader3', 1.0],
      ]);

      const effK = effKFromCorr(weights, correlations);

      expect(effK).toBeCloseTo(1.0, 5);
    });

    it('should handle partial correlation', () => {
      const weights = new Map([
        ['trader1', 1.0],
        ['trader2', 1.0],
        ['trader3', 1.0],
        ['trader4', 1.0],
        ['trader5', 1.0],
      ]);

      // 80% correlation
      const correlations = new Map<string, number>();
      const addrs = ['trader1', 'trader2', 'trader3', 'trader4', 'trader5'];
      for (let i = 0; i < addrs.length; i++) {
        for (let j = i + 1; j < addrs.length; j++) {
          correlations.set(`${addrs[i]}|${addrs[j]}`, 0.8);
        }
      }

      const effK = effKFromCorr(weights, correlations);

      // 5 traders with 80% correlation → effK ≈ 1.19
      // Formula: K / (1 + (K-1) × ρ) = 5 / (1 + 4 × 0.8) = 5 / 4.2 ≈ 1.19
      expect(effK).toBeCloseTo(1.19, 1);
    });

    it('should return 1 for single trader', () => {
      const weights = new Map([['trader1', 1.0]]);
      const correlations = new Map<string, number>();

      const effK = effKFromCorr(weights, correlations);

      expect(effK).toBe(1);
    });
  });

  describe('bps to R conversion', () => {
    it('should convert correctly for 1% stop', () => {
      const entryPx = 50000;
      const stopPx = 49500; // 1% below entry = 100 bps
      const feesBps = 10;

      const costR = bpsToR(entryPx, stopPx, feesBps);

      // 10 bps / 100 bps = 0.1 R
      expect(costR).toBeCloseTo(0.1, 5);
    });

    it('should convert correctly for 0.5% stop', () => {
      const entryPx = 50000;
      const stopPx = 49750; // 0.5% below entry = 50 bps
      const feesBps = 17; // 17 bps total costs

      const costR = bpsToR(entryPx, stopPx, feesBps);

      // 17 bps / 50 bps = 0.34 R
      expect(costR).toBeCloseTo(0.34, 5);
    });

    it('should handle very tight stops', () => {
      const entryPx = 50000;
      const stopPx = 49995; // Very tight stop = 1 bps
      const feesBps = 10;

      const costR = bpsToR(entryPx, stopPx, feesBps);

      // 10 bps / 1 bps = 10 R (costs exceed potential profit!)
      expect(costR).toBeCloseTo(10, 0);
    });
  });

  describe('EV calculation', () => {
    it('should calculate positive EV for high win rate', () => {
      const pWin = 0.7;
      const entryPx = 50000;
      const stopPx = 49500; // 1% stop = 100 bps

      const ev = calculateEV(pWin, entryPx, stopPx);

      // Gross: 0.7 × 0.5 - 0.3 × 0.3 = 0.35 - 0.09 = 0.26
      expect(ev.evGrossR).toBeCloseTo(0.26, 2);

      // Cost: 17 bps / 100 bps = 0.17 R
      expect(ev.evCostR).toBeCloseTo(0.17, 2);

      // Net: 0.26 - 0.17 = 0.09
      expect(ev.evNetR).toBeCloseTo(0.09, 2);
    });

    it('should calculate negative EV for low win rate', () => {
      const pWin = 0.4;
      const entryPx = 50000;
      const stopPx = 49500;

      const ev = calculateEV(pWin, entryPx, stopPx);

      // Gross: 0.4 × 0.5 - 0.6 × 0.3 = 0.2 - 0.18 = 0.02
      expect(ev.evGrossR).toBeCloseTo(0.02, 2);

      // Net should be negative (costs > gross)
      expect(ev.evNetR).toBeLessThan(0);
    });

    it('should pass EV gate for good opportunities', () => {
      const pWin = 0.75;
      const entryPx = 50000;
      const stopPx = 49000; // 2% stop = 200 bps

      const ev = calculateEV(pWin, entryPx, stopPx);

      // Costs are lower relative to stop (17 bps / 200 bps = 0.085 R)
      // Net EV should be positive and above threshold
      expect(ev.evNetR).toBeGreaterThan(CONSENSUS_EV_MIN_R);
    });
  });

  describe('adaptive window', () => {
    function adaptiveWindowSeconds(atrPercentile: number): number {
      const base = 120;
      const lo = 60;
      const hi = 360;

      if (atrPercentile < 0.3) {
        return Math.max(lo, base);
      } else if (atrPercentile < 0.7) {
        return Math.min(hi, base * 2);
      } else {
        return Math.min(hi, base * 3);
      }
    }

    it('should use short window in low volatility', () => {
      const window = adaptiveWindowSeconds(0.1);
      expect(window).toBe(120); // base
    });

    it('should use medium window in normal volatility', () => {
      const window = adaptiveWindowSeconds(0.5);
      expect(window).toBe(240); // 2x base
    });

    it('should use long window in high volatility', () => {
      const window = adaptiveWindowSeconds(0.9);
      expect(window).toBe(360); // 3x base, capped at hi
    });
  });

  describe('integration: full consensus flow', () => {
    it('should detect consensus when all gates pass', () => {
      // 4 traders, 3 going long, 1 going short
      const fills: Fill[] = [
        {
          fillId: '1',
          address: '0xA',
          asset: 'BTC',
          side: 'buy',
          size: 1.0,
          price: 50000,
          ts: new Date(),
        },
        {
          fillId: '2',
          address: '0xB',
          asset: 'BTC',
          side: 'buy',
          size: 1.0,
          price: 50010,
          ts: new Date(),
        },
        {
          fillId: '3',
          address: '0xC',
          asset: 'BTC',
          side: 'buy',
          size: 1.0,
          price: 50020,
          ts: new Date(),
        },
        {
          fillId: '4',
          address: '0xD',
          asset: 'BTC',
          side: 'sell',
          size: 1.0,
          price: 50015,
          ts: new Date(),
        },
      ];

      // Step 1: Collapse to votes
      const votes = collapseToVotes(fills);
      expect(votes.length).toBe(4);

      // Step 2: Check dispersion
      const directions = votes.map((v) => v.direction);
      const dispersionResult = passesConsensusGates(directions);
      expect(dispersionResult.passes).toBe(true); // 75% > 70%
      expect(dispersionResult.majorityDir).toBe('long');

      // Step 3: Check effective-K (with default correlation)
      const agreeingVotes = votes.filter(
        (v) => v.direction === dispersionResult.majorityDir
      );
      const weights = new Map(agreeingVotes.map((v) => [v.address, v.weight]));

      // With 3 traders and ρ=0.3: effK = 3 / (1 + 2 × 0.3) = 3 / 1.6 = 1.875
      // This is below CONSENSUS_MIN_EFFECTIVE_K of 2.0
      // So with default correlation, 3 traders isn't quite enough
      const effK = effKFromCorr(weights, new Map(), 0.3);
      expect(effK).toBeCloseTo(1.875, 2);

      // With lower correlation (0.2), it would pass
      const effKLowCorr = effKFromCorr(weights, new Map(), 0.2);
      expect(effKLowCorr).toBeGreaterThanOrEqual(CONSENSUS_MIN_EFFECTIVE_K);

      // Step 4: EV check (with reasonable assumptions)
      const medianPrice =
        agreeingVotes.reduce((sum, v) => sum + v.price, 0) /
        agreeingVotes.length;
      const stopPrice = medianPrice * 0.99; // 1% stop
      const ev = calculateEV(0.6, medianPrice, stopPrice);

      // With 60% win rate and 1% stop, should be close to threshold
      // This might pass or fail depending on exact parameters
    });

    it('should reject consensus when traders are highly correlated', () => {
      const weights = new Map([
        ['trader1', 1.0],
        ['trader2', 1.0],
        ['trader3', 1.0],
      ]);

      // All pairs 90% correlated (likely copy-traders)
      const correlations = new Map([
        ['trader1|trader2', 0.9],
        ['trader1|trader3', 0.9],
        ['trader2|trader3', 0.9],
      ]);

      const effK = effKFromCorr(weights, correlations);

      // effK = 3 / (1 + 2 × 0.9) = 3 / 2.8 ≈ 1.07
      expect(effK).toBeLessThan(CONSENSUS_MIN_EFFECTIVE_K);
    });
  });
});
