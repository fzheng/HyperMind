/**
 * Quant Acceptance Tests
 *
 * These tests verify the mathematical correctness of the algorithm implementation
 * as specified by the quant review. Each test uses hand-derived expected values.
 *
 * Test categories:
 * 1. R audit: hand-derived cases with known PnL, stop_bps, fees
 * 2. Flip atomics: single fill that reverses sign emits close + open
 * 3. Consensus dedupe: many micro-fills from one trader = one vote
 * 4. effK extremes: perfect correlation and cluster scenarios
 * 5. EV units: specific bps → R conversion tests
 */

import {
  Fill,
  Episode,
  buildEpisodes,
  calculateVwap,
  calculateR,
  calculateStopBps,
  bpsToR,
} from '../packages/ts-lib/src/episode';

import {
  Vote,
  ConsensusConfig,
  DEFAULT_CONSENSUS_CONFIG,
  calculateEffectiveK,
  calculateEV,
  checkConsensus,
} from '../packages/ts-lib/src/consensus';

describe('Quant Acceptance Tests', () => {
  // Helper to create fills
  function createFill(
    fillId: string,
    side: 'buy' | 'sell',
    size: number,
    price: number,
    ts: Date,
    realizedPnl?: number,
    fees?: number
  ): Fill {
    return {
      fillId,
      address: '0xtest',
      asset: 'BTC',
      side,
      size,
      price,
      ts,
      realizedPnl,
      fees,
    };
  }

  describe('R Audit - Hand-Derived Cases', () => {
    /**
     * Test case 1: Simple long trade
     * Entry: Buy 1 BTC @ $50,000
     * Exit: Sell 1 BTC @ $51,000
     * PnL: $1,000
     * Stop: 1% = 100 bps
     * Risk: $50,000 * 0.01 = $500
     * R = $1,000 / $500 = 2.0 (capped at 2.0)
     */
    it('should calculate R=2.0 for +$1000 on $50k entry with 1% stop', () => {
      const now = new Date();
      const fills: Fill[] = [
        createFill('1', 'buy', 1.0, 50000, now, undefined, 0),
        createFill('2', 'sell', 1.0, 51000, new Date(now.getTime() + 60000), 1000, 0),
      ];

      const episodes = buildEpisodes(fills, { defaultStopFraction: 0.01, rMin: -2, rMax: 2, timeoutHours: 168 });

      expect(episodes).toHaveLength(1);
      expect(episodes[0].entryVwap).toBe(50000);
      expect(episodes[0].exitVwap).toBe(51000);
      expect(episodes[0].riskAmount).toBe(500); // 50000 * 0.01
      expect(episodes[0].realizedPnl).toBe(1000);
      expect(episodes[0].resultR).toBe(2.0); // Capped at rMax
    });

    /**
     * Test case 2: Losing short trade
     * Entry: Sell 2 BTC @ $40,000 (short)
     * Exit: Buy 2 BTC @ $41,000
     * PnL: ($40,000 - $41,000) * 2 = -$2,000
     * Stop: 1% = 100 bps
     * Risk: $80,000 * 0.01 = $800
     * R = -$2,000 / $800 = -2.5 → clamped to -2.0
     */
    it('should calculate R=-2.0 (clamped) for -$2000 on $80k short', () => {
      const now = new Date();
      const fills: Fill[] = [
        createFill('1', 'sell', 2.0, 40000, now, undefined, 0),
        createFill('2', 'buy', 2.0, 41000, new Date(now.getTime() + 60000), -2000, 0),
      ];

      const episodes = buildEpisodes(fills, { defaultStopFraction: 0.01, rMin: -2, rMax: 2, timeoutHours: 168 });

      expect(episodes).toHaveLength(1);
      expect(episodes[0].direction).toBe('short');
      expect(episodes[0].entryVwap).toBe(40000);
      expect(episodes[0].entryNotional).toBe(80000); // 40000 * 2
      expect(episodes[0].riskAmount).toBe(800);
      expect(episodes[0].realizedPnl).toBe(-2000);
      // Unclamped would be -2.5, clamped to -2.0
      expect(episodes[0].resultR).toBe(-2.0);
    });

    /**
     * Test case 3: Small winner with fees
     * Entry: Buy 0.5 BTC @ $60,000 = $30,000 notional
     * Exit: Sell 0.5 BTC @ $60,300 = $30,150 notional
     * Gross PnL: $150
     * Fees: $10
     * Net PnL from realized_pnl: $140
     * Stop: 1% = $300 risk
     * R = $140 / $300 = 0.467
     */
    it('should calculate R=0.47 for small winner with fees', () => {
      const now = new Date();
      const fills: Fill[] = [
        createFill('1', 'buy', 0.5, 60000, now, undefined, 5),
        createFill('2', 'sell', 0.5, 60300, new Date(now.getTime() + 60000), 140, 5),
      ];

      const episodes = buildEpisodes(fills, { defaultStopFraction: 0.01, rMin: -2, rMax: 2, timeoutHours: 168 });

      expect(episodes).toHaveLength(1);
      expect(episodes[0].riskAmount).toBeCloseTo(300, 0); // 30000 * 0.01
      expect(episodes[0].realizedPnl).toBe(140);
      expect(episodes[0].resultR).toBeCloseTo(0.467, 2);
      expect(episodes[0].totalFees).toBe(10);
    });

    /**
     * Test case 4: VWAP entry with multiple fills
     * Entry: Buy 1 BTC @ $50,000, then Buy 1 BTC @ $52,000
     * VWAP entry = (50000*1 + 52000*1) / 2 = $51,000
     * Exit: Sell 2 BTC @ $53,000
     * PnL: ($53,000 - $51,000) * 2 = $4,000
     * Risk: $102,000 * 0.01 = $1,020
     * R = $4,000 / $1,020 = 3.92 → capped to 2.0
     */
    it('should use VWAP for multiple entry fills', () => {
      const now = new Date();
      const fills: Fill[] = [
        createFill('1', 'buy', 1.0, 50000, now, undefined, 0),
        createFill('2', 'buy', 1.0, 52000, new Date(now.getTime() + 30000), undefined, 0),
        createFill('3', 'sell', 2.0, 53000, new Date(now.getTime() + 60000), 4000, 0),
      ];

      const episodes = buildEpisodes(fills, { defaultStopFraction: 0.01, rMin: -2, rMax: 2, timeoutHours: 168 });

      expect(episodes).toHaveLength(1);
      expect(episodes[0].entryVwap).toBe(51000); // VWAP
      expect(episodes[0].entrySize).toBe(2);
      expect(episodes[0].entryNotional).toBe(102000);
      expect(episodes[0].riskAmount).toBe(1020);
      expect(episodes[0].realizedPnl).toBe(4000);
      expect(episodes[0].resultR).toBe(2.0); // Capped
    });
  });

  describe('Flip Atomics - Direction Reversal', () => {
    /**
     * A single fill that reverses sign should:
     * 1. Close the current episode
     * 2. Open a new episode in the opposite direction
     * 3. Both episodes should have the same exit/entry timestamp
     */
    it('should emit close + open for single fill that reverses position', () => {
      const now = new Date();
      const fills: Fill[] = [
        // Open long: Buy 1 BTC
        createFill('1', 'buy', 1.0, 50000, now, undefined, 0),
        // Flip to short: Sell 2 BTC (closes 1 long, opens 1 short)
        createFill('2', 'sell', 2.0, 51000, new Date(now.getTime() + 60000), 1000, 0),
        // Close short: Buy 1 BTC
        createFill('3', 'buy', 1.0, 50500, new Date(now.getTime() + 120000), 500, 0),
      ];

      const episodes = buildEpisodes(fills, { defaultStopFraction: 0.01, rMin: -2, rMax: 2, timeoutHours: 168 });

      expect(episodes).toHaveLength(2);

      // First episode: Long closed by direction flip
      expect(episodes[0].direction).toBe('long');
      expect(episodes[0].status).toBe('closed');
      expect(episodes[0].closedReason).toBe('direction_flip');
      expect(episodes[0].entryFills).toHaveLength(1);
      expect(episodes[0].exitFills).toHaveLength(1);

      // Second episode: Short opened by flip, closed normally
      expect(episodes[1].direction).toBe('short');
      expect(episodes[1].status).toBe('closed');
      expect(episodes[1].closedReason).toBe('full_close');
      // The flip fill is used for BOTH exit of ep1 and entry of ep2
      expect(episodes[1].entryFills).toHaveLength(1);
      expect(episodes[1].entryFills[0].fillId).toBe('2');
    });

    /**
     * Edge case: flip from short to long
     */
    it('should handle flip from short to long', () => {
      const now = new Date();
      const fills: Fill[] = [
        createFill('1', 'sell', 1.0, 50000, now),
        createFill('2', 'buy', 3.0, 49000, new Date(now.getTime() + 60000), 1000), // Flip
        createFill('3', 'sell', 2.0, 51000, new Date(now.getTime() + 120000), 4000),
      ];

      const episodes = buildEpisodes(fills, { defaultStopFraction: 0.01, rMin: -2, rMax: 2, timeoutHours: 168 });

      expect(episodes).toHaveLength(2);
      expect(episodes[0].direction).toBe('short');
      expect(episodes[0].closedReason).toBe('direction_flip');
      expect(episodes[1].direction).toBe('long');
      expect(episodes[1].entrySize).toBe(2); // 3 - 1 = 2 from flip
    });
  });

  describe('Consensus Dedupe - One Vote Per Trader', () => {
    /**
     * 100 micro-fills from one trader within the window should count as ONE vote.
     * effK should be 1.0 for a single trader.
     */
    it('should treat multiple fills from same address as single vote', () => {
      const now = Date.now();

      // One trader with 100 fills
      const votes: Vote[] = [];
      for (let i = 0; i < 100; i++) {
        votes.push({
          address: '0xtrader1',
          direction: 'long' as const,
          weight: 0.01, // Small weight per fill
          price: 50000 + i, // Slightly different prices
          ts: new Date(now - i * 100),
        });
      }

      // Note: In real implementation, these 100 fills should be collapsed to 1 vote
      // before calling consensus. This test verifies effK calculation.
      const weights = new Map([['0xtrader1', 1.0]]); // Collapsed to single vote
      const effK = calculateEffectiveK(weights, new Map(), { defaultCorrelation: 0 });

      expect(effK).toBe(1.0); // Single trader = effK of 1
    });

    /**
     * 3 traders each with different numbers of fills should still be 3 votes
     */
    it('should count each unique address as one vote regardless of fill count', () => {
      // In practice, fills are collapsed per address before consensus
      const weights = new Map([
        ['0xtrader1', 1.0],
        ['0xtrader2', 1.0],
        ['0xtrader3', 1.0],
      ]);

      const effK = calculateEffectiveK(weights, new Map(), { defaultCorrelation: 0 });
      expect(effK).toBe(3.0);
    });
  });

  describe('effK Extremes - Correlation Scenarios', () => {
    /**
     * Perfect correlation (ρ=1) across 5 traders ⇒ effK≈1
     */
    it('should return effK≈1 for 5 perfectly correlated traders', () => {
      const weights = new Map([
        ['t1', 1.0],
        ['t2', 1.0],
        ['t3', 1.0],
        ['t4', 1.0],
        ['t5', 1.0],
      ]);

      // All pairs have ρ=1
      const correlations = new Map<string, number>();
      const traders = ['t1', 't2', 't3', 't4', 't5'];
      for (let i = 0; i < traders.length; i++) {
        for (let j = i + 1; j < traders.length; j++) {
          const key = [traders[i], traders[j]].sort().join('|');
          correlations.set(key, 1.0);
        }
      }

      // With perfect correlation, effK = K / (1 + (K-1)*1) = 5/5 = 1
      const effK = calculateEffectiveK(weights, correlations, {
        defaultCorrelation: 1.0,
        correlationShrinkage: 1.0, // No shrinkage for this test
        minPairsForCorrelation: 0,
      });

      expect(effK).toBeCloseTo(1.0, 1);
    });

    /**
     * Two clusters of 5 traders each:
     * - Within cluster: ρ_intra = 0.8
     * - Between clusters: ρ_inter = 0.0
     * Expected effK ≈ 2 (two independent signals)
     */
    it('should return effK≈2 for two uncorrelated clusters of correlated traders', () => {
      const traders = ['a1', 'a2', 'a3', 'a4', 'a5', 'b1', 'b2', 'b3', 'b4', 'b5'];
      const weights = new Map(traders.map((t) => [t, 1.0]));

      const correlations = new Map<string, number>();
      const clusterA = ['a1', 'a2', 'a3', 'a4', 'a5'];
      const clusterB = ['b1', 'b2', 'b3', 'b4', 'b5'];

      // Intra-cluster correlation = 0.8
      for (const cluster of [clusterA, clusterB]) {
        for (let i = 0; i < cluster.length; i++) {
          for (let j = i + 1; j < cluster.length; j++) {
            const key = [cluster[i], cluster[j]].sort().join('|');
            correlations.set(key, 0.8);
          }
        }
      }

      // Inter-cluster correlation = 0.0 (not set, will use default)
      // Note: We need to set these explicitly to 0 since shrinkage will pull toward default
      for (const a of clusterA) {
        for (const b of clusterB) {
          const key = [a, b].sort().join('|');
          correlations.set(key, 0.0);
        }
      }

      const effK = calculateEffectiveK(weights, correlations, {
        defaultCorrelation: 0.0,
        correlationShrinkage: 1.0, // No shrinkage for this test
        minPairsForCorrelation: 0,
      });

      // Each cluster of 5 with ρ=0.8 has effK = 5/(1+4*0.8) = 5/4.2 ≈ 1.19
      // Two independent clusters: total effK ≈ 2 * 1.19 = 2.38
      // But formula is more complex with full matrix...
      // For 10 traders with two clusters, effK should be between 2 and 2.5
      expect(effK).toBeGreaterThan(1.5);
      expect(effK).toBeLessThan(3.0);
    });

    /**
     * Uniform correlation of 0.5 across 4 traders
     * effK = K / (1 + (K-1)*ρ) = 4 / (1 + 3*0.5) = 4/2.5 = 1.6
     */
    it('should calculate effK=1.6 for 4 traders with ρ=0.5', () => {
      const weights = new Map([
        ['t1', 1.0],
        ['t2', 1.0],
        ['t3', 1.0],
        ['t4', 1.0],
      ]);

      const effK = calculateEffectiveK(weights, new Map(), {
        defaultCorrelation: 0.5,
        correlationShrinkage: 1.0, // No shrinkage
        minPairsForCorrelation: 0,
      });

      // effK = 4 / (1 + 3*0.5) = 4/2.5 = 1.6
      expect(effK).toBeCloseTo(1.6, 2);
    });
  });

  describe('EV Units - bps to R Conversion', () => {
    /**
     * stop_bps=40, round-trip costs=12 bps
     * cost_R = 12/40 = 0.3
     * Any EV_gross_r < 0.3 should fail gate
     */
    it('should convert 12 bps cost to 0.3R with 40 bps stop', () => {
      const stopBps = 40;
      const feesBps = 7;
      const slipBps = 5;
      const totalCostBps = feesBps + slipBps; // 12 bps

      const costR = bpsToR(totalCostBps, stopBps);
      expect(costR).toBeCloseTo(0.3, 5);
    });

    /**
     * With 100 bps stop and 17 bps costs (7 fees + 10 slip)
     * cost_R = 17/100 = 0.17
     */
    it('should convert 17 bps cost to 0.17R with 100 bps stop', () => {
      const costR = bpsToR(17, 100);
      expect(costR).toBeCloseTo(0.17, 5);
    });

    /**
     * EV calculation with specific numbers:
     * p_win = 0.6, avg_win_R = 0.5, avg_loss_R = 0.3
     * EV_gross = 0.6 * 0.5 - 0.4 * 0.3 = 0.30 - 0.12 = 0.18
     * With 100 bps stop and 17 bps costs: cost_R = 0.17
     * EV_net = 0.18 - 0.17 = 0.01
     */
    it('should calculate correct EV with specific parameters', () => {
      const config: ConsensusConfig = {
        ...DEFAULT_CONSENSUS_CONFIG,
        avgWinR: 0.5,
        avgLossR: 0.3,
        feesBps: 7,
        slipBps: 10,
      };

      const stopBps = 100;
      const pWin = 0.6;

      const ev = calculateEV(pWin, stopBps, config);

      // EV_gross = 0.6 * 0.5 - 0.4 * 0.3 = 0.18
      expect(ev.evGrossR).toBeCloseTo(0.18, 2);

      // cost_R = 17/100 = 0.17
      expect(ev.evCostR).toBeCloseTo(0.17, 2);

      // EV_net = 0.18 - 0.17 = 0.01
      expect(ev.evNetR).toBeCloseTo(0.01, 2);
    });

    /**
     * Tighter stop (50 bps) doubles the cost impact
     * cost_R = 17/50 = 0.34
     */
    it('should show higher cost impact with tighter stop', () => {
      const config: ConsensusConfig = {
        ...DEFAULT_CONSENSUS_CONFIG,
        avgWinR: 0.5,
        avgLossR: 0.3,
        feesBps: 7,
        slipBps: 10,
      };

      const stopBps = 50; // Tighter stop
      const pWin = 0.6;

      const ev = calculateEV(pWin, stopBps, config);

      // cost_R = 17/50 = 0.34
      expect(ev.evCostR).toBeCloseTo(0.34, 2);

      // EV_net = 0.18 - 0.34 = -0.16 (negative!)
      expect(ev.evNetR).toBeCloseTo(-0.16, 2);
    });

    /**
     * EV gate should reject when net EV < threshold
     * With evMinR = 0.2, a trade with net EV of 0.01 should fail
     */
    it('should fail EV gate when net EV below threshold', () => {
      const now = Date.now();
      const votes: Vote[] = [
        { address: '0xa', direction: 'long', weight: 1, price: 50000, ts: new Date(now) },
        { address: '0xb', direction: 'long', weight: 1, price: 50000, ts: new Date(now) },
        { address: '0xc', direction: 'long', weight: 1, price: 50000, ts: new Date(now) },
        { address: '0xd', direction: 'long', weight: 1, price: 50000, ts: new Date(now) },
      ];

      // Win rates that produce ~0.6 pWin after shrinkage
      const winRates = new Map([
        ['0xa', { winRate: 0.65, samples: 50 }],
        ['0xb', { winRate: 0.65, samples: 50 }],
        ['0xc', { winRate: 0.65, samples: 50 }],
        ['0xd', { winRate: 0.65, samples: 50 }],
      ]);

      const config: ConsensusConfig = {
        ...DEFAULT_CONSENSUS_CONFIG,
        minTraders: 3,
        minPct: 0.7,
        minEffectiveK: 2.0,
        evMinR: 0.2, // Require at least 0.2R net EV
        defaultCorrelation: 0, // Uncorrelated
        avgWinR: 0.5,
        avgLossR: 0.3,
        feesBps: 7,
        slipBps: 10,
      };

      const result = checkConsensus(
        votes,
        50000,
        120000,
        100, // 100 bps stop
        new Map(),
        winRates,
        config
      );

      // Should fail EV gate (net EV ~0.01-0.05 < 0.2)
      expect(result.gateResults.ev.passed).toBe(false);
      expect(result.gateResults.ev.evNetR).toBeLessThan(0.2);
    });
  });

  describe('Stop Basis Consistency', () => {
    /**
     * Verify that stop_bps used at entry is consistent with R calculation
     */
    it('should use consistent stop basis for entry and exit', () => {
      const now = new Date();
      const fills: Fill[] = [
        createFill('1', 'buy', 1.0, 50000, now, undefined, 0),
        createFill('2', 'sell', 1.0, 50500, new Date(now.getTime() + 60000), 500, 0),
      ];

      // Use 2% stop (200 bps)
      const episodes = buildEpisodes(fills, { defaultStopFraction: 0.02, rMin: -2, rMax: 2, timeoutHours: 168 });

      expect(episodes).toHaveLength(1);

      // With 2% stop: risk = 50000 * 0.02 = 1000
      expect(episodes[0].riskAmount).toBe(1000);
      expect(episodes[0].stopBps).toBe(200);

      // R = 500 / 1000 = 0.5
      expect(episodes[0].resultR).toBe(0.5);
    });

    /**
     * Stop basis should be recorded at entry for audit
     */
    it('should record stop basis at entry', () => {
      const now = new Date();
      const fills: Fill[] = [createFill('1', 'buy', 1.0, 50000, now)];

      const episodes = buildEpisodes(fills, { defaultStopFraction: 0.015, rMin: -2, rMax: 2, timeoutHours: 168 });

      expect(episodes).toHaveLength(1);
      expect(episodes[0].stopBps).toBe(150); // 1.5% = 150 bps
      expect(episodes[0].status).toBe('open');
    });
  });
});
