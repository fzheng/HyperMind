/**
 * Tests for Episode Builder
 *
 * Tests cover:
 * 1. Episode segmentation (sign changes)
 * 2. VWAP calculation for entry/exit
 * 3. R-multiple calculation with policy stop
 * 4. Direction flip handling
 * 5. Partial closes and adds
 * 6. Validation of episode integrity
 */

import {
  Fill,
  Episode,
  EpisodeBuilderConfig,
  buildEpisodes,
  calculateVwap,
  calculateR,
  calculateStopPrice,
  calculateStopBps,
  bpsToR,
  getSignedSize,
  validateEpisodes,
  getOpenEpisodes,
  getClosedEpisodes,
} from '../packages/ts-lib/src/episode';

describe('Episode Builder', () => {
  const defaultConfig: EpisodeBuilderConfig = {
    defaultStopFraction: 0.01, // 1% stop
    rMin: -2.0,
    rMax: 2.0,
    timeoutHours: 168,
  };

  // Helper to create fills
  function createFill(
    id: string,
    side: 'buy' | 'sell',
    size: number,
    price: number,
    ts: Date,
    realizedPnl?: number
  ): Fill {
    return {
      fillId: id,
      address: '0xtest',
      asset: 'BTC',
      side,
      size,
      price,
      ts,
      realizedPnl,
    };
  }

  describe('VWAP calculation', () => {
    it('should calculate VWAP correctly for single fill', () => {
      const fills: Fill[] = [createFill('1', 'buy', 1.0, 50000, new Date())];
      expect(calculateVwap(fills)).toBe(50000);
    });

    it('should calculate VWAP correctly for multiple fills', () => {
      const fills: Fill[] = [
        createFill('1', 'buy', 1.0, 50000, new Date()),
        createFill('2', 'buy', 2.0, 51000, new Date()),
      ];
      // VWAP = (1*50000 + 2*51000) / 3 = 152000 / 3 = 50666.67
      expect(calculateVwap(fills)).toBeCloseTo(50666.67, 2);
    });

    it('should return 0 for empty fills', () => {
      expect(calculateVwap([])).toBe(0);
    });
  });

  describe('R-multiple calculation', () => {
    it('should calculate R correctly for winning trade', () => {
      const pnl = 500; // $500 profit
      const riskAmount = 500; // $500 at risk (1% of $50k)
      expect(calculateR(pnl, riskAmount)).toBe(1.0);
    });

    it('should calculate R correctly for losing trade', () => {
      const pnl = -250;
      const riskAmount = 500;
      expect(calculateR(pnl, riskAmount)).toBe(-0.5);
    });

    it('should winsorize extreme positive R', () => {
      const pnl = 5000; // 10x the risk
      const riskAmount = 500;
      expect(calculateR(pnl, riskAmount)).toBe(2.0); // Capped at +2
    });

    it('should winsorize extreme negative R', () => {
      const pnl = -2500; // 5x the risk loss
      const riskAmount = 500;
      expect(calculateR(pnl, riskAmount)).toBe(-2.0); // Capped at -2
    });

    it('should handle zero risk amount', () => {
      expect(calculateR(100, 0)).toBe(0);
    });
  });

  describe('stop price calculation', () => {
    it('should calculate stop below entry for long', () => {
      const entry = 50000;
      const stopFraction = 0.01; // 1%
      const stop = calculateStopPrice(entry, 'long', stopFraction);
      expect(stop).toBe(49500); // 1% below
    });

    it('should calculate stop above entry for short', () => {
      const entry = 50000;
      const stopFraction = 0.01;
      const stop = calculateStopPrice(entry, 'short', stopFraction);
      expect(stop).toBe(50500); // 1% above
    });
  });

  describe('bps to R conversion', () => {
    it('should convert 10 bps cost with 100 bps stop to 0.1 R', () => {
      expect(bpsToR(10, 100)).toBe(0.1);
    });

    it('should convert 17 bps cost with 100 bps stop to 0.17 R', () => {
      expect(bpsToR(17, 100)).toBe(0.17);
    });

    it('should handle tight stop (high cost in R)', () => {
      // 17 bps cost with 20 bps stop = 0.85 R cost!
      expect(bpsToR(17, 20)).toBe(0.85);
    });

    it('should return 0 for zero stop', () => {
      expect(bpsToR(10, 0)).toBe(0);
    });
  });

  describe('simple episode building', () => {
    it('should build one episode from open + close', () => {
      const now = new Date();
      const fills: Fill[] = [
        createFill('1', 'buy', 1.0, 50000, now),
        createFill(
          '2',
          'sell',
          1.0,
          51000,
          new Date(now.getTime() + 3600000),
          1000
        ),
      ];

      const episodes = buildEpisodes(fills, defaultConfig);

      expect(episodes.length).toBe(1);
      expect(episodes[0].direction).toBe('long');
      expect(episodes[0].status).toBe('closed');
      expect(episodes[0].entryVwap).toBe(50000);
      expect(episodes[0].exitVwap).toBe(51000);
      expect(episodes[0].realizedPnl).toBe(1000);

      // R = 1000 / (50000 * 0.01) = 1000 / 500 = 2.0
      expect(episodes[0].resultR).toBe(2.0);
    });

    it('should build short episode correctly', () => {
      const now = new Date();
      const fills: Fill[] = [
        createFill('1', 'sell', 1.0, 50000, now),
        createFill(
          '2',
          'buy',
          1.0,
          49000,
          new Date(now.getTime() + 3600000),
          1000
        ),
      ];

      const episodes = buildEpisodes(fills, defaultConfig);

      expect(episodes.length).toBe(1);
      expect(episodes[0].direction).toBe('short');
      expect(episodes[0].status).toBe('closed');
      expect(episodes[0].realizedPnl).toBe(1000);
    });

    it('should keep episode open if not closed', () => {
      const fills: Fill[] = [createFill('1', 'buy', 1.0, 50000, new Date())];

      const episodes = buildEpisodes(fills, defaultConfig);

      expect(episodes.length).toBe(1);
      expect(episodes[0].status).toBe('open');
      expect(episodes[0].exitVwap).toBeNull();
      expect(episodes[0].resultR).toBeNull();
    });
  });

  describe('adding to positions', () => {
    it('should update VWAP when adding to long', () => {
      const now = new Date();
      const fills: Fill[] = [
        createFill('1', 'buy', 1.0, 50000, now),
        createFill('2', 'buy', 1.0, 52000, new Date(now.getTime() + 60000)),
      ];

      const episodes = buildEpisodes(fills, defaultConfig);

      expect(episodes.length).toBe(1);
      expect(episodes[0].status).toBe('open');
      expect(episodes[0].entrySize).toBe(2.0);
      // VWAP = (50000 + 52000) / 2 = 51000
      expect(episodes[0].entryVwap).toBe(51000);
      expect(episodes[0].entryFills.length).toBe(2);
    });

    it('should track multiple adds then close', () => {
      const now = new Date();
      const fills: Fill[] = [
        createFill('1', 'buy', 1.0, 50000, now),
        createFill('2', 'buy', 0.5, 51000, new Date(now.getTime() + 60000)),
        createFill('3', 'buy', 0.5, 52000, new Date(now.getTime() + 120000)),
        createFill(
          '4',
          'sell',
          2.0,
          53000,
          new Date(now.getTime() + 180000),
          4000
        ),
      ];

      const episodes = buildEpisodes(fills, defaultConfig);

      expect(episodes.length).toBe(1);
      expect(episodes[0].status).toBe('closed');
      expect(episodes[0].entrySize).toBe(2.0);
      // VWAP = (1*50000 + 0.5*51000 + 0.5*52000) / 2 = (50000 + 25500 + 26000) / 2 = 50750
      expect(episodes[0].entryVwap).toBeCloseTo(50750, 0);
      expect(episodes[0].exitVwap).toBe(53000);
    });
  });

  describe('partial closes', () => {
    it('should track partial close fills but not close episode', () => {
      const now = new Date();
      const fills: Fill[] = [
        createFill('1', 'buy', 2.0, 50000, now),
        createFill('2', 'sell', 1.0, 51000, new Date(now.getTime() + 60000)),
      ];

      const episodes = buildEpisodes(fills, defaultConfig);

      expect(episodes.length).toBe(1);
      expect(episodes[0].status).toBe('open');
      expect(episodes[0].exitFills.length).toBe(1);
    });

    it('should close episode after multiple partial closes', () => {
      const now = new Date();
      const fills: Fill[] = [
        createFill('1', 'buy', 2.0, 50000, now),
        createFill('2', 'sell', 0.5, 51000, new Date(now.getTime() + 60000)),
        createFill('3', 'sell', 0.5, 52000, new Date(now.getTime() + 120000)),
        createFill(
          '4',
          'sell',
          1.0,
          53000,
          new Date(now.getTime() + 180000),
          5000
        ),
      ];

      const episodes = buildEpisodes(fills, defaultConfig);

      expect(episodes.length).toBe(1);
      expect(episodes[0].status).toBe('closed');
      expect(episodes[0].exitFills.length).toBe(3);
      // Exit VWAP = (0.5*51000 + 0.5*52000 + 1.0*53000) / 2 = (25500 + 26000 + 53000) / 2 = 52250
      expect(episodes[0].exitVwap).toBeCloseTo(52250, 0);
    });
  });

  describe('direction flips', () => {
    it('should close long and open short on direction flip', () => {
      const now = new Date();
      const fills: Fill[] = [
        createFill('1', 'buy', 1.0, 50000, now),
        createFill(
          '2',
          'sell',
          2.0,
          49000,
          new Date(now.getTime() + 60000),
          -1000
        ),
      ];

      const episodes = buildEpisodes(fills, defaultConfig);

      expect(episodes.length).toBe(2);

      // First episode: long, closed due to flip
      expect(episodes[0].direction).toBe('long');
      expect(episodes[0].status).toBe('closed');
      expect(episodes[0].closedReason).toBe('direction_flip');
      expect(episodes[0].realizedPnl).toBe(-1000);

      // Second episode: short, still open
      expect(episodes[1].direction).toBe('short');
      expect(episodes[1].status).toBe('open');
      expect(episodes[1].entrySize).toBe(1.0); // Excess after closing long
    });

    it('should handle multiple flips', () => {
      const now = new Date();
      const fills: Fill[] = [
        createFill('1', 'buy', 1.0, 50000, now),
        createFill(
          '2',
          'sell',
          2.0,
          49000,
          new Date(now.getTime() + 60000),
          -1000
        ),
        createFill(
          '3',
          'buy',
          2.0,
          48000,
          new Date(now.getTime() + 120000),
          1000
        ),
      ];

      const episodes = buildEpisodes(fills, defaultConfig);

      expect(episodes.length).toBe(3);
      expect(episodes[0].direction).toBe('long');
      expect(episodes[1].direction).toBe('short');
      expect(episodes[2].direction).toBe('long');
    });
  });

  describe('multiple complete episodes', () => {
    it('should build two separate episodes', () => {
      const now = new Date();
      const fills: Fill[] = [
        // Episode 1: long
        createFill('1', 'buy', 1.0, 50000, now),
        createFill(
          '2',
          'sell',
          1.0,
          51000,
          new Date(now.getTime() + 60000),
          1000
        ),
        // Episode 2: short
        createFill('3', 'sell', 1.0, 51000, new Date(now.getTime() + 120000)),
        createFill(
          '4',
          'buy',
          1.0,
          50000,
          new Date(now.getTime() + 180000),
          1000
        ),
      ];

      const episodes = buildEpisodes(fills, defaultConfig);

      expect(episodes.length).toBe(2);

      expect(episodes[0].direction).toBe('long');
      expect(episodes[0].status).toBe('closed');
      expect(episodes[0].resultR).toBe(2.0);

      expect(episodes[1].direction).toBe('short');
      expect(episodes[1].status).toBe('closed');
      // R = 1000 / (51000 * 0.01) = 1000 / 510 ≈ 1.96
      expect(episodes[1].resultR).toBeCloseTo(1.96, 1);
    });
  });

  describe('validation', () => {
    it('should validate episodes with all fills assigned', () => {
      const now = new Date();
      const fills: Fill[] = [
        createFill('1', 'buy', 1.0, 50000, now),
        createFill(
          '2',
          'sell',
          1.0,
          51000,
          new Date(now.getTime() + 60000),
          1000
        ),
      ];

      const episodes = buildEpisodes(fills, defaultConfig);
      const result = validateEpisodes(episodes, fills);

      expect(result.valid).toBe(true);
      expect(result.errors.length).toBe(0);
      expect(result.fillCount).toBe(2);
      expect(result.episodeCount).toBe(1);
    });

    it('should detect missing fills', () => {
      const now = new Date();
      const fills: Fill[] = [
        createFill('1', 'buy', 1.0, 50000, now),
        createFill('2', 'sell', 1.0, 51000, new Date(now.getTime() + 60000)),
        createFill('3', 'buy', 1.0, 50000, new Date(now.getTime() + 120000)), // Not in episodes
      ];

      // Build episodes from only first two fills
      const episodes = buildEpisodes(fills.slice(0, 2), defaultConfig);
      const result = validateEpisodes(episodes, fills);

      expect(result.valid).toBe(false);
      expect(result.errors.length).toBeGreaterThan(0);
    });
  });

  describe('helper functions', () => {
    it('getSignedSize should return positive for buy', () => {
      const fill = createFill('1', 'buy', 1.5, 50000, new Date());
      expect(getSignedSize(fill)).toBe(1.5);
    });

    it('getSignedSize should return negative for sell', () => {
      const fill = createFill('1', 'sell', 1.5, 50000, new Date());
      expect(getSignedSize(fill)).toBe(-1.5);
    });

    it('getOpenEpisodes should filter correctly', () => {
      const now = new Date();
      const fills: Fill[] = [
        createFill('1', 'buy', 1.0, 50000, now),
        createFill('2', 'sell', 1.0, 51000, new Date(now.getTime() + 60000)),
        createFill('3', 'buy', 1.0, 50000, new Date(now.getTime() + 120000)),
      ];

      const episodes = buildEpisodes(fills, defaultConfig);
      const open = getOpenEpisodes(episodes);
      const closed = getClosedEpisodes(episodes);

      expect(open.length).toBe(1);
      expect(closed.length).toBe(1);
    });
  });

  describe('edge cases', () => {
    it('should handle empty fills', () => {
      const episodes = buildEpisodes([], defaultConfig);
      expect(episodes.length).toBe(0);
    });

    it('should handle single fill', () => {
      const fills: Fill[] = [createFill('1', 'buy', 1.0, 50000, new Date())];
      const episodes = buildEpisodes(fills, defaultConfig);

      expect(episodes.length).toBe(1);
      expect(episodes[0].status).toBe('open');
    });

    it('should handle fills with same timestamp', () => {
      const now = new Date();
      const fills: Fill[] = [
        createFill('1', 'buy', 0.5, 50000, now),
        createFill('2', 'buy', 0.5, 50100, now),
      ];

      const episodes = buildEpisodes(fills, defaultConfig);
      expect(episodes.length).toBe(1);
      expect(episodes[0].entrySize).toBe(1.0);
    });

    it('should calculate R with realized PnL from fill', () => {
      const now = new Date();
      const fills: Fill[] = [
        createFill('1', 'buy', 1.0, 50000, now),
        createFill(
          '2',
          'sell',
          1.0,
          52000,
          new Date(now.getTime() + 60000),
          2000
        ), // HL reports $2000 PnL
      ];

      const episodes = buildEpisodes(fills, defaultConfig);

      expect(episodes[0].realizedPnl).toBe(2000); // Uses HL's value
      // R = 2000 / (50000 * 0.01) = 2000 / 500 = 4.0 → capped at 2.0
      expect(episodes[0].resultR).toBe(2.0);
    });
  });

  describe('R calculation consistency', () => {
    it('should use policy stop not trader stop', () => {
      const now = new Date();
      // Trader might have used a 2% stop, but we use our policy 1% stop
      const fills: Fill[] = [
        createFill('1', 'buy', 1.0, 50000, now),
        createFill(
          '2',
          'sell',
          1.0,
          50500,
          new Date(now.getTime() + 60000),
          500
        ), // +1% move
      ];

      const episodes = buildEpisodes(fills, defaultConfig);

      // R = 500 / (50000 * 0.01) = 500 / 500 = 1.0
      // Even though price only moved 1%, we calculate R based on our policy stop
      expect(episodes[0].resultR).toBe(1.0);
      expect(episodes[0].stopBps).toBe(100); // 1% = 100 bps
    });

    it('should calculate stop correctly for different prices', () => {
      expect(calculateStopBps(50000, 49500)).toBe(100); // 1%
      expect(calculateStopBps(50000, 49000)).toBe(200); // 2%
      expect(calculateStopBps(50000, 49750)).toBe(50); // 0.5%
    });
  });
});
