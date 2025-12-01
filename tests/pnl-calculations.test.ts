/**
 * Tests for P&L calculation logic
 * Covers realized P&L, unrealized P&L, and position calculations
 */

describe('Position P&L Calculations', () => {
  // Calculate unrealized P&L for a position
  function calculateUnrealizedPnl(
    position: { size: number; entryPrice: number },
    currentPrice: number
  ): number {
    if (position.size === 0) return 0;
    const notional = Math.abs(position.size) * position.entryPrice;
    const currentNotional = Math.abs(position.size) * currentPrice;

    if (position.size > 0) {
      // Long position: profit when price goes up
      return currentNotional - notional;
    } else {
      // Short position: profit when price goes down
      return notional - currentNotional;
    }
  }

  describe('Long positions', () => {
    test('profit when price increases', () => {
      const position = { size: 1.0, entryPrice: 95000 };
      const pnl = calculateUnrealizedPnl(position, 96000);
      expect(pnl).toBe(1000); // 1 BTC * (96000 - 95000)
    });

    test('loss when price decreases', () => {
      const position = { size: 1.0, entryPrice: 95000 };
      const pnl = calculateUnrealizedPnl(position, 94000);
      expect(pnl).toBe(-1000); // 1 BTC * (94000 - 95000)
    });

    test('breakeven at entry price', () => {
      const position = { size: 0.5, entryPrice: 95000 };
      const pnl = calculateUnrealizedPnl(position, 95000);
      expect(pnl).toBe(0);
    });

    test('handles fractional positions', () => {
      const position = { size: 0.5, entryPrice: 95000 };
      const pnl = calculateUnrealizedPnl(position, 97000);
      expect(pnl).toBe(1000); // 0.5 BTC * (97000 - 95000)
    });
  });

  describe('Short positions', () => {
    test('profit when price decreases', () => {
      const position = { size: -1.0, entryPrice: 95000 };
      const pnl = calculateUnrealizedPnl(position, 94000);
      expect(pnl).toBe(1000); // 1 BTC * (95000 - 94000)
    });

    test('loss when price increases', () => {
      const position = { size: -1.0, entryPrice: 95000 };
      const pnl = calculateUnrealizedPnl(position, 96000);
      expect(pnl).toBe(-1000); // 1 BTC * (95000 - 96000)
    });

    test('breakeven at entry price', () => {
      const position = { size: -0.5, entryPrice: 95000 };
      const pnl = calculateUnrealizedPnl(position, 95000);
      expect(pnl).toBe(0);
    });

    test('handles fractional positions', () => {
      const position = { size: -0.5, entryPrice: 95000 };
      const pnl = calculateUnrealizedPnl(position, 93000);
      expect(pnl).toBe(1000); // 0.5 BTC * (95000 - 93000)
    });
  });

  describe('Flat positions', () => {
    test('returns 0 for zero size', () => {
      const position = { size: 0, entryPrice: 95000 };
      expect(calculateUnrealizedPnl(position, 100000)).toBe(0);
      expect(calculateUnrealizedPnl(position, 50000)).toBe(0);
    });
  });
});

describe('Trade P&L Calculations', () => {
  // Calculate realized P&L for closing a position
  function calculateRealizedPnl(
    entryPrice: number,
    exitPrice: number,
    size: number, // Positive for long, negative for short
    fees: number = 0
  ): number {
    if (size === 0) return 0;
    const grossPnl = size > 0
      ? (exitPrice - entryPrice) * Math.abs(size)
      : (entryPrice - exitPrice) * Math.abs(size);
    return grossPnl - fees;
  }

  describe('Closing long positions', () => {
    test('profit when closing above entry', () => {
      const pnl = calculateRealizedPnl(95000, 96000, 1.0);
      expect(pnl).toBe(1000);
    });

    test('loss when closing below entry', () => {
      const pnl = calculateRealizedPnl(95000, 94000, 1.0);
      expect(pnl).toBe(-1000);
    });

    test('breakeven at same price', () => {
      const pnl = calculateRealizedPnl(95000, 95000, 1.0);
      expect(pnl).toBe(0);
    });

    test('subtracts fees', () => {
      const pnl = calculateRealizedPnl(95000, 96000, 1.0, 50);
      expect(pnl).toBe(950); // 1000 gross - 50 fees
    });
  });

  describe('Closing short positions', () => {
    test('profit when closing below entry', () => {
      const pnl = calculateRealizedPnl(95000, 94000, -1.0);
      expect(pnl).toBe(1000);
    });

    test('loss when closing above entry', () => {
      const pnl = calculateRealizedPnl(95000, 96000, -1.0);
      expect(pnl).toBe(-1000);
    });

    test('breakeven at same price', () => {
      const pnl = calculateRealizedPnl(95000, 95000, -1.0);
      expect(pnl).toBe(0);
    });

    test('subtracts fees', () => {
      const pnl = calculateRealizedPnl(95000, 94000, -1.0, 50);
      expect(pnl).toBe(950); // 1000 gross - 50 fees
    });
  });
});

describe('Position Size Changes', () => {
  interface Position {
    size: number;
    avgEntryPrice: number;
    realizedPnl: number;
  }

  // Update position after a trade
  function applyTrade(
    position: Position,
    tradeSize: number, // Positive for buy, negative for sell
    tradePrice: number
  ): Position {
    const newPosition = { ...position };

    if (position.size === 0) {
      // Opening new position
      newPosition.size = tradeSize;
      newPosition.avgEntryPrice = tradePrice;
    } else if (Math.sign(position.size) === Math.sign(tradeSize)) {
      // Increasing position
      const totalNotional = Math.abs(position.size) * position.avgEntryPrice +
                           Math.abs(tradeSize) * tradePrice;
      const totalSize = Math.abs(position.size) + Math.abs(tradeSize);
      newPosition.avgEntryPrice = totalNotional / totalSize;
      newPosition.size = position.size + tradeSize;
    } else {
      // Decreasing/flipping position
      const closeSize = Math.min(Math.abs(position.size), Math.abs(tradeSize));
      const remainingTradeSize = Math.abs(tradeSize) - closeSize;

      // Calculate realized P&L on closed portion
      const pnl = position.size > 0
        ? (tradePrice - position.avgEntryPrice) * closeSize
        : (position.avgEntryPrice - tradePrice) * closeSize;
      newPosition.realizedPnl += pnl;

      if (remainingTradeSize === 0) {
        // Partial or full close, no flip
        newPosition.size = position.size + tradeSize;
      } else {
        // Flip to opposite side
        newPosition.size = tradeSize + (position.size > 0 ? closeSize : -closeSize);
        newPosition.avgEntryPrice = tradePrice;
      }

      if (newPosition.size === 0) {
        newPosition.avgEntryPrice = 0;
      }
    }

    return newPosition;
  }

  describe('Opening positions', () => {
    test('opens long from flat', () => {
      const position: Position = { size: 0, avgEntryPrice: 0, realizedPnl: 0 };
      const result = applyTrade(position, 1.0, 95000);

      expect(result.size).toBe(1.0);
      expect(result.avgEntryPrice).toBe(95000);
      expect(result.realizedPnl).toBe(0);
    });

    test('opens short from flat', () => {
      const position: Position = { size: 0, avgEntryPrice: 0, realizedPnl: 0 };
      const result = applyTrade(position, -1.0, 95000);

      expect(result.size).toBe(-1.0);
      expect(result.avgEntryPrice).toBe(95000);
      expect(result.realizedPnl).toBe(0);
    });
  });

  describe('Increasing positions', () => {
    test('adds to long position with averaged entry', () => {
      const position: Position = { size: 1.0, avgEntryPrice: 95000, realizedPnl: 0 };
      const result = applyTrade(position, 1.0, 97000);

      expect(result.size).toBe(2.0);
      expect(result.avgEntryPrice).toBe(96000); // (1*95000 + 1*97000) / 2
      expect(result.realizedPnl).toBe(0);
    });

    test('adds to short position with averaged entry', () => {
      const position: Position = { size: -1.0, avgEntryPrice: 95000, realizedPnl: 0 };
      const result = applyTrade(position, -1.0, 93000);

      expect(result.size).toBe(-2.0);
      expect(result.avgEntryPrice).toBe(94000); // (1*95000 + 1*93000) / 2
      expect(result.realizedPnl).toBe(0);
    });
  });

  describe('Closing positions', () => {
    test('partial close of long with profit', () => {
      const position: Position = { size: 2.0, avgEntryPrice: 95000, realizedPnl: 0 };
      const result = applyTrade(position, -1.0, 97000);

      expect(result.size).toBe(1.0);
      expect(result.avgEntryPrice).toBe(95000); // Entry doesn't change on partial close
      expect(result.realizedPnl).toBe(2000); // 1 * (97000 - 95000)
    });

    test('partial close of short with profit', () => {
      const position: Position = { size: -2.0, avgEntryPrice: 95000, realizedPnl: 0 };
      const result = applyTrade(position, 1.0, 93000);

      expect(result.size).toBe(-1.0);
      expect(result.avgEntryPrice).toBe(95000);
      expect(result.realizedPnl).toBe(2000); // 1 * (95000 - 93000)
    });

    test('full close of long with loss', () => {
      const position: Position = { size: 1.0, avgEntryPrice: 95000, realizedPnl: 0 };
      const result = applyTrade(position, -1.0, 93000);

      expect(result.size).toBe(0);
      expect(result.avgEntryPrice).toBe(0); // Reset on full close
      expect(result.realizedPnl).toBe(-2000); // 1 * (93000 - 95000)
    });

    test('full close of short with loss', () => {
      const position: Position = { size: -1.0, avgEntryPrice: 95000, realizedPnl: 0 };
      const result = applyTrade(position, 1.0, 97000);

      expect(result.size).toBe(0);
      expect(result.avgEntryPrice).toBe(0);
      expect(result.realizedPnl).toBe(-2000); // 1 * (95000 - 97000)
    });
  });

  describe('Flipping positions', () => {
    test('flips from long to short', () => {
      const position: Position = { size: 1.0, avgEntryPrice: 95000, realizedPnl: 0 };
      const result = applyTrade(position, -2.0, 97000);

      expect(result.size).toBe(-1.0);
      expect(result.avgEntryPrice).toBe(97000); // New entry for flipped portion
      expect(result.realizedPnl).toBe(2000); // Realized on closed long
    });

    test('flips from short to long', () => {
      const position: Position = { size: -1.0, avgEntryPrice: 95000, realizedPnl: 0 };
      const result = applyTrade(position, 2.0, 93000);

      expect(result.size).toBe(1.0);
      expect(result.avgEntryPrice).toBe(93000);
      expect(result.realizedPnl).toBe(2000); // Realized on closed short
    });
  });
});

describe('Liquidation Price Calculations', () => {
  // Calculate liquidation price for isolated margin
  function calculateLiquidationPrice(
    entryPrice: number,
    leverage: number,
    isLong: boolean,
    maintenanceMargin: number = 0.005 // 0.5% maintenance margin
  ): number {
    // For isolated margin:
    // Long: liq_price = entry * (1 - 1/leverage + mm)
    // Short: liq_price = entry * (1 + 1/leverage - mm)
    if (isLong) {
      return entryPrice * (1 - 1 / leverage + maintenanceMargin);
    } else {
      return entryPrice * (1 + 1 / leverage - maintenanceMargin);
    }
  }

  describe('Long positions', () => {
    test('calculates liquidation price with 10x leverage', () => {
      const liqPrice = calculateLiquidationPrice(100000, 10, true);
      // 100000 * (1 - 0.1 + 0.005) = 100000 * 0.905 = 90500
      expect(liqPrice).toBeCloseTo(90500, 0);
    });

    test('calculates liquidation price with 5x leverage', () => {
      const liqPrice = calculateLiquidationPrice(100000, 5, true);
      // 100000 * (1 - 0.2 + 0.005) = 100000 * 0.805 = 80500
      expect(liqPrice).toBeCloseTo(80500, 0);
    });

    test('calculates liquidation price with 20x leverage', () => {
      const liqPrice = calculateLiquidationPrice(100000, 20, true);
      // 100000 * (1 - 0.05 + 0.005) = 100000 * 0.955 = 95500
      expect(liqPrice).toBeCloseTo(95500, 0);
    });

    test('higher leverage means closer liquidation', () => {
      const liq10x = calculateLiquidationPrice(100000, 10, true);
      const liq20x = calculateLiquidationPrice(100000, 20, true);
      expect(liq20x).toBeGreaterThan(liq10x);
    });
  });

  describe('Short positions', () => {
    test('calculates liquidation price with 10x leverage', () => {
      const liqPrice = calculateLiquidationPrice(100000, 10, false);
      // 100000 * (1 + 0.1 - 0.005) = 100000 * 1.095 = 109500
      expect(liqPrice).toBeCloseTo(109500, 0);
    });

    test('calculates liquidation price with 5x leverage', () => {
      const liqPrice = calculateLiquidationPrice(100000, 5, false);
      // 100000 * (1 + 0.2 - 0.005) = 100000 * 1.195 = 119500
      expect(liqPrice).toBeCloseTo(119500, 0);
    });

    test('higher leverage means closer liquidation', () => {
      const liq10x = calculateLiquidationPrice(100000, 10, false);
      const liq20x = calculateLiquidationPrice(100000, 20, false);
      expect(liq20x).toBeLessThan(liq10x);
    });
  });
});

describe('ROI Calculations', () => {
  // Calculate ROI percentage
  function calculateRoi(pnl: number, initialMargin: number): number {
    if (initialMargin === 0) return 0;
    return (pnl / initialMargin) * 100;
  }

  // Calculate initial margin from position
  function calculateInitialMargin(
    positionSize: number,
    entryPrice: number,
    leverage: number
  ): number {
    return (Math.abs(positionSize) * entryPrice) / leverage;
  }

  test('calculates positive ROI', () => {
    const margin = calculateInitialMargin(1.0, 95000, 10); // 9500
    const roi = calculateRoi(1000, margin);
    expect(roi).toBeCloseTo(10.53, 1); // 1000 / 9500 * 100
  });

  test('calculates negative ROI', () => {
    const margin = calculateInitialMargin(1.0, 95000, 10);
    const roi = calculateRoi(-500, margin);
    expect(roi).toBeCloseTo(-5.26, 1);
  });

  test('handles zero margin', () => {
    expect(calculateRoi(1000, 0)).toBe(0);
  });

  test('higher leverage amplifies ROI', () => {
    const margin10x = calculateInitialMargin(1.0, 95000, 10);
    const margin20x = calculateInitialMargin(1.0, 95000, 20);

    const roi10x = calculateRoi(1000, margin10x);
    const roi20x = calculateRoi(1000, margin20x);

    expect(roi20x).toBeGreaterThan(roi10x);
    expect(roi20x).toBeCloseTo(roi10x * 2, 1);
  });
});

describe('Win Rate Calculations', () => {
  function calculateWinRate(wins: number, total: number): number {
    if (total === 0) return 0;
    return wins / total;
  }

  function isWinningTrade(pnl: number): boolean {
    return pnl > 0;
  }

  test('calculates win rate correctly', () => {
    expect(calculateWinRate(7, 10)).toBe(0.7);
    expect(calculateWinRate(5, 10)).toBe(0.5);
    expect(calculateWinRate(10, 10)).toBe(1.0);
    expect(calculateWinRate(0, 10)).toBe(0);
  });

  test('handles zero trades', () => {
    expect(calculateWinRate(0, 0)).toBe(0);
  });

  test('identifies winning trades', () => {
    expect(isWinningTrade(100)).toBe(true);
    expect(isWinningTrade(0.01)).toBe(true);
    expect(isWinningTrade(0)).toBe(false);
    expect(isWinningTrade(-100)).toBe(false);
  });
});

describe('Profit Factor Calculations', () => {
  function calculateProfitFactor(grossProfit: number, grossLoss: number): number {
    if (grossLoss === 0) {
      return grossProfit > 0 ? Infinity : 0;
    }
    return grossProfit / Math.abs(grossLoss);
  }

  test('calculates profit factor correctly', () => {
    expect(calculateProfitFactor(1000, 500)).toBe(2.0);
    expect(calculateProfitFactor(500, 500)).toBe(1.0);
    expect(calculateProfitFactor(500, 1000)).toBe(0.5);
  });

  test('handles no losses', () => {
    expect(calculateProfitFactor(1000, 0)).toBe(Infinity);
  });

  test('handles no profits', () => {
    expect(calculateProfitFactor(0, 500)).toBe(0);
  });

  test('handles no trades', () => {
    expect(calculateProfitFactor(0, 0)).toBe(0);
  });
});

describe('Drawdown Calculations', () => {
  function calculateMaxDrawdown(equityCurve: number[]): {
    maxDrawdown: number;
    maxDrawdownPercent: number;
  } {
    if (equityCurve.length < 2) {
      return { maxDrawdown: 0, maxDrawdownPercent: 0 };
    }

    let peak = equityCurve[0];
    let maxDrawdown = 0;
    let maxDrawdownPercent = 0;

    for (const equity of equityCurve) {
      if (equity > peak) {
        peak = equity;
      }
      const drawdown = peak - equity;
      if (drawdown > maxDrawdown) {
        maxDrawdown = drawdown;
        maxDrawdownPercent = peak > 0 ? drawdown / peak : 0;
      }
    }

    return { maxDrawdown, maxDrawdownPercent };
  }

  test('calculates max drawdown correctly', () => {
    const curve = [100, 120, 100, 150, 130, 160];
    const result = calculateMaxDrawdown(curve);
    // Peak at 120, drops to 100 = 20 drawdown (16.67% of 120)
    // This is larger than 150->130 drawdown (13.33% of 150)
    expect(result.maxDrawdown).toBe(20);
    expect(result.maxDrawdownPercent).toBeCloseTo(0.167, 2); // 20/120
  });

  test('returns 0 for monotonically increasing curve', () => {
    const curve = [100, 110, 120, 130, 140];
    const result = calculateMaxDrawdown(curve);
    expect(result.maxDrawdown).toBe(0);
    expect(result.maxDrawdownPercent).toBe(0);
  });

  test('handles single point', () => {
    const result = calculateMaxDrawdown([100]);
    expect(result.maxDrawdown).toBe(0);
  });

  test('handles empty curve', () => {
    const result = calculateMaxDrawdown([]);
    expect(result.maxDrawdown).toBe(0);
  });

  test('calculates severe drawdown', () => {
    const curve = [100, 200, 100]; // 50% drawdown
    const result = calculateMaxDrawdown(curve);
    expect(result.maxDrawdown).toBe(100);
    expect(result.maxDrawdownPercent).toBe(0.5);
  });
});

describe('HL-Decide Signal Generation', () => {
  describe('Signal emission requirements', () => {
    test('should require both score and fill for signal emission', () => {
      const scores: Record<string, unknown> = {
        '0x1234567890123456789012345678901234567890': { score: 0.75, weight: 1.0 },
      };
      const fills: Record<string, unknown> = {};

      const address = '0x1234567890123456789012345678901234567890';
      const canEmitSignal = scores[address] && fills[address];

      expect(canEmitSignal).toBeFalsy();
    });

    test('should emit signal when both score and fill present', () => {
      const scores: Record<string, unknown> = {
        '0x1234567890123456789012345678901234567890': { score: 0.75, weight: 1.0 },
      };
      const fills: Record<string, unknown> = {
        '0x1234567890123456789012345678901234567890': { fill_id: 'fill-1', asset: 'BTC' },
      };

      const address = '0x1234567890123456789012345678901234567890';
      const canEmitSignal = scores[address] && fills[address];

      expect(canEmitSignal).toBeTruthy();
    });
  });

  describe('Side determination', () => {
    const pickSide = (score: number) => (score >= 0 ? 'long' : 'short');

    test('should return long for positive score', () => {
      expect(pickSide(0.5)).toBe('long');
    });

    test('should return long for zero score', () => {
      expect(pickSide(0)).toBe('long');
    });

    test('should return short for negative score', () => {
      expect(pickSide(-0.5)).toBe('short');
      expect(pickSide(-1.0)).toBe('short');
    });
  });

  describe('Confidence clamping', () => {
    const clampConfidence = (score: number) =>
      Math.min(Math.max(Math.abs(score), 0.1), 1.0);

    test('should clamp to minimum 0.1 for zero', () => {
      expect(clampConfidence(0.0)).toBe(0.1);
    });

    test('should clamp to minimum 0.1 for small values', () => {
      expect(clampConfidence(0.05)).toBe(0.1);
    });

    test('should pass through values in range', () => {
      expect(clampConfidence(0.5)).toBe(0.5);
    });

    test('should clamp to maximum 1.0', () => {
      expect(clampConfidence(1.5)).toBe(1.0);
    });

    test('should convert negative to positive', () => {
      expect(clampConfidence(-0.8)).toBe(0.8);
    });
  });

  describe('Ticket payload creation', () => {
    test('should include entry price in payload', () => {
      const createTicketPayload = (fillId: string, weight: number, entryPrice: number) => ({
        fill_id: fillId,
        weight,
        entry_price: entryPrice,
      });

      const payload = createTicketPayload('fill-123', 1.0, 95000);

      expect(payload.fill_id).toBe('fill-123');
      expect(payload.weight).toBe(1.0);
      expect(payload.entry_price).toBe(95000);
    });
  });

  describe('Outcome notes formatting', () => {
    test('should include entry/exit prices and P&L', () => {
      const createOutcomeNotes = (entryPrice: number, exitPrice: number, pnlR: number) =>
        `Timeboxed exit: entry=${entryPrice.toFixed(2)}, exit=${exitPrice.toFixed(2)}, pnl_r=${pnlR.toFixed(4)}`;

      const notes = createOutcomeNotes(95000, 95500, 0.0053);

      expect(notes).toContain('entry=95000.00');
      expect(notes).toContain('exit=95500.00');
      expect(notes).toContain('pnl_r=0.0053');
    });
  });
});

describe('LRU Cache Eviction', () => {
  test('should enforce max scores limit', () => {
    const MAX_SCORES = 500;
    const scores: Map<string, unknown> = new Map();

    // Add MAX_SCORES + 50 items
    for (let i = 0; i < MAX_SCORES + 50; i++) {
      scores.set(`addr-${i}`, { score: i });
    }

    // Enforce limits (LRU eviction)
    while (scores.size > MAX_SCORES) {
      const firstKey = scores.keys().next().value;
      scores.delete(firstKey);
    }

    expect(scores.size).toBe(MAX_SCORES);
    // First 50 should be evicted
    expect(scores.has('addr-0')).toBe(false);
    expect(scores.has('addr-49')).toBe(false);
    // Recent ones should remain
    expect(scores.has('addr-549')).toBe(true);
  });

  test('should enforce max fills limit', () => {
    const MAX_FILLS = 500;
    const fills: Map<string, unknown> = new Map();

    for (let i = 0; i < MAX_FILLS + 100; i++) {
      fills.set(`addr-${i}`, { fill_id: `fill-${i}` });
    }

    while (fills.size > MAX_FILLS) {
      const firstKey = fills.keys().next().value;
      fills.delete(firstKey);
    }

    expect(fills.size).toBe(MAX_FILLS);
  });
});
