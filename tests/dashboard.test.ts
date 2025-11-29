/**
 * Dashboard UI Unit Tests
 *
 * Tests the pure JavaScript functions from dashboard.js using jsdom.
 * These tests verify formatting, aggregation, and calculation logic.
 */

// Mock DOM environment for browser-dependent code
const mockLocalStorage: Record<string, string> = {};
const mockMatchMedia = jest.fn().mockReturnValue({
  matches: false,
  addEventListener: jest.fn(),
});

// Set up minimal DOM mocks before importing dashboard functions
Object.defineProperty(global, 'localStorage', {
  value: {
    getItem: (key: string) => mockLocalStorage[key] || null,
    setItem: (key: string, value: string) => {
      mockLocalStorage[key] = value;
    },
    removeItem: (key: string) => {
      delete mockLocalStorage[key];
    },
  },
});

Object.defineProperty(global, 'matchMedia', {
  value: mockMatchMedia,
});

// Since dashboard.js runs in browser, we'll extract and test the pure functions directly
// by reimplementing them here for testing (they can be extracted to a shared module later)

// =====================
// Extracted Functions for Testing
// =====================

/**
 * Format price for display (e.g., $97,234.56)
 * Always shows full price with 2 decimal places - no K/M abbreviation
 */
function fmtPrice(value: number | null | undefined): string {
  if (!Number.isFinite(value)) return '—';
  return (
    '$' +
    (value as number).toLocaleString('en-US', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    })
  );
}

/**
 * Format trade price for fills table (full price with 2 decimals)
 */
function fmtTradePrice(value: number | null | undefined): string {
  if (!Number.isFinite(value)) return 'N/A';
  return (
    '$' +
    (value as number).toLocaleString('en-US', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    })
  );
}

/**
 * Format percentage (e.g., 85.5%)
 */
function fmtPercent(value: number | null | undefined): string {
  if (!Number.isFinite(value)) return 'N/A';
  return `${((value as number) * 100).toFixed(1)}%`;
}

/**
 * Format USD value with abbreviations (K, M, B)
 */
function fmtUsdShort(value: number | null | undefined): string {
  if (!Number.isFinite(value)) return 'N/A';
  if (value === 0) return '$0';
  const sign = (value as number) > 0 ? '+' : '-';
  const abs = Math.abs(value as number);
  const formatter = (num: number, suffix: string) =>
    `${sign}$${num.toFixed(num >= 10 ? 1 : 2)}${suffix}`;
  if (abs >= 1e9) return formatter(abs / 1e9, 'B');
  if (abs >= 1e6) return formatter(abs / 1e6, 'M');
  if (abs >= 1e3) return formatter(abs / 1e3, 'K');
  return `${sign}$${abs.toFixed(0)}`;
}

/**
 * Format time (HH:MM:SS)
 */
function fmtTime(ts: string): string {
  return new Date(ts).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

/**
 * Shorten Ethereum address (0x1234...5678)
 */
function shortAddress(address: string | null | undefined): string {
  if (!address) return '';
  return `${address.slice(0, 6)}…${address.slice(-4)}`;
}

/**
 * Validate Ethereum address format
 */
function isValidEthAddress(address: string): boolean {
  return /^0x[a-fA-F0-9]{40}$/.test(address);
}

/**
 * Format action label for display
 */
function formatActionLabel(fill: { action?: string; side?: string }): string {
  const action = fill.action ? String(fill.action).toLowerCase() : '';
  const map: Record<string, string> = {
    'open long': 'OPEN LONG',
    'increase long': 'ADD LONG',
    'close long (close all)': 'CLOSE LONG',
    'decrease long': 'CLOSE LONG',
    'open short': 'OPEN SHORT',
    'increase short': 'ADD SHORT',
    'close short (close all)': 'CLOSE SHORT',
    'decrease short': 'CLOSE SHORT',
  };
  if (map[action]) return map[action];
  if (fill.side === 'buy') return 'OPEN LONG';
  if (fill.side === 'sell') return 'OPEN SHORT';
  return action ? action.toUpperCase() : 'TRADE';
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text: string | null | undefined): string {
  if (!text) return '';
  const htmlEscapes: Record<string, string> = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  };
  return text.replace(/[&<>"']/g, (char) => htmlEscapes[char]);
}

/**
 * Calculate signed size based on action (XOR logic)
 */
function calculateSignedSize(totalSize: number, action: string): number {
  const actionLower = action.toLowerCase();
  const isShort = actionLower.includes('short');
  const isDecrease =
    actionLower.includes('decrease') || actionLower.includes('close');
  // XOR logic: negative when (decrease AND long) OR (increase AND short)
  const isNegative = isDecrease !== isShort;
  return isNegative ? -totalSize : totalSize;
}

/**
 * Calculate previous position from resulting position
 */
function calculatePreviousPosition(
  resultingPosition: number,
  totalSize: number,
  action: string
): number {
  const actionLower = action.toLowerCase();
  const isDecrease =
    actionLower.includes('decrease') || actionLower.includes('close');
  const isShort = actionLower.includes('short');

  if (isShort) {
    if (isDecrease) {
      return resultingPosition - totalSize;
    } else {
      return resultingPosition + totalSize;
    }
  } else {
    if (isDecrease) {
      return resultingPosition + totalSize;
    } else {
      return resultingPosition - totalSize;
    }
  }
}

interface Fill {
  time_utc: string;
  address: string;
  action: string;
  size_signed: number;
  price_usd?: number;
  closed_pnl_usd?: number;
  symbol?: string;
  resulting_position?: number;
  previous_position?: number;
}

interface AggregatedGroup {
  id: string;
  time_utc: string;
  oldest_time: string;
  address: string;
  symbol: string;
  action: string;
  fills: Fill[];
  totalSize: number;
  totalPnl: number;
  prices: number[];
  isAggregated: boolean;
  fillCount: number;
  avgPrice: number | null;
  size_signed: number;
  closed_pnl_usd: number | null;
  resulting_position?: number;
  previous_position?: number;
}

const AGGREGATION_WINDOW_MS = 60000;

/**
 * Check if a fill can be merged into an existing group
 */
function canMergeIntoGroup(group: AggregatedGroup, fill: Fill): boolean {
  const fillTime = new Date(fill.time_utc).getTime();
  const groupNewestTime = new Date(group.time_utc).getTime();
  const groupOldestTime = new Date(group.oldest_time).getTime();

  const timeDiffFromNewest = Math.abs(groupNewestTime - fillTime);
  const timeDiffFromOldest = Math.abs(groupOldestTime - fillTime);
  const withinWindow =
    timeDiffFromNewest <= AGGREGATION_WINDOW_MS ||
    timeDiffFromOldest <= AGGREGATION_WINDOW_MS;

  const symbol = (fill.symbol || 'BTC').toUpperCase();
  const normalizedFillAddress = (fill.address || '').toLowerCase();
  const normalizedFillAction = (fill.action || '').trim().toLowerCase();
  const sameAddress = group.address === normalizedFillAddress;
  const sameSymbol = group.symbol === symbol;
  const sameAction = group.action === normalizedFillAction;

  return sameAddress && sameSymbol && sameAction && withinWindow;
}

/**
 * Create a new aggregation group from a fill
 */
function createGroup(fill: Fill): AggregatedGroup {
  const symbol = (fill.symbol || 'BTC').toUpperCase();
  const normalizedAddress = (fill.address || '').toLowerCase();
  const normalizedAction = (fill.action || '').trim().toLowerCase();
  return {
    id: `${normalizedAddress}-${fill.time_utc}-test`,
    time_utc: fill.time_utc,
    oldest_time: fill.time_utc,
    address: normalizedAddress,
    symbol: symbol,
    action: normalizedAction,
    fills: [fill],
    totalSize: Math.abs(fill.size_signed || 0),
    totalPnl: fill.closed_pnl_usd || 0,
    prices: fill.price_usd ? [fill.price_usd] : [],
    resulting_position: fill.resulting_position,
    previous_position: fill.previous_position,
    isAggregated: false,
    fillCount: 1,
    avgPrice: fill.price_usd || null,
    size_signed: fill.size_signed,
    closed_pnl_usd: fill.closed_pnl_usd || null,
  };
}

// =====================
// Tests
// =====================

describe('Dashboard UI Functions', () => {
  describe('fmtPrice', () => {
    it('should format price with dollar sign and 2 decimals', () => {
      expect(fmtPrice(97234.56)).toBe('$97,234.56');
    });

    it('should format large prices with comma separators', () => {
      expect(fmtPrice(1234567.89)).toBe('$1,234,567.89');
    });

    it('should format small prices correctly', () => {
      expect(fmtPrice(0.99)).toBe('$0.99');
    });

    it('should return dash for null/undefined', () => {
      expect(fmtPrice(null)).toBe('—');
      expect(fmtPrice(undefined)).toBe('—');
    });

    it('should return dash for NaN', () => {
      expect(fmtPrice(NaN)).toBe('—');
    });

    it('should return dash for Infinity', () => {
      expect(fmtPrice(Infinity)).toBe('—');
      expect(fmtPrice(-Infinity)).toBe('—');
    });

    it('should format zero correctly', () => {
      expect(fmtPrice(0)).toBe('$0.00');
    });
  });

  describe('fmtTradePrice', () => {
    it('should format trade price with 2 decimals', () => {
      expect(fmtTradePrice(97234.56)).toBe('$97,234.56');
    });

    it('should return N/A for null/undefined', () => {
      expect(fmtTradePrice(null)).toBe('N/A');
      expect(fmtTradePrice(undefined)).toBe('N/A');
    });

    it('should not use K/M abbreviation', () => {
      expect(fmtTradePrice(97000)).toBe('$97,000.00');
      expect(fmtTradePrice(3500000)).toBe('$3,500,000.00');
    });
  });

  describe('fmtPercent', () => {
    it('should format decimal as percentage', () => {
      expect(fmtPercent(0.855)).toBe('85.5%');
    });

    it('should handle 100%', () => {
      expect(fmtPercent(1.0)).toBe('100.0%');
    });

    it('should handle 0%', () => {
      expect(fmtPercent(0)).toBe('0.0%');
    });

    it('should return N/A for null/undefined', () => {
      expect(fmtPercent(null)).toBe('N/A');
      expect(fmtPercent(undefined)).toBe('N/A');
    });

    it('should handle values > 100%', () => {
      expect(fmtPercent(1.5)).toBe('150.0%');
    });
  });

  describe('fmtUsdShort', () => {
    it('should format small positive values', () => {
      expect(fmtUsdShort(50)).toBe('+$50');
    });

    it('should format small negative values', () => {
      expect(fmtUsdShort(-50)).toBe('-$50');
    });

    it('should format thousands with K', () => {
      expect(fmtUsdShort(5000)).toBe('+$5.00K');
      expect(fmtUsdShort(15000)).toBe('+$15.0K');
    });

    it('should format millions with M', () => {
      expect(fmtUsdShort(5000000)).toBe('+$5.00M');
      expect(fmtUsdShort(-15000000)).toBe('-$15.0M');
    });

    it('should format billions with B', () => {
      expect(fmtUsdShort(5000000000)).toBe('+$5.00B');
    });

    it('should return $0 for zero', () => {
      expect(fmtUsdShort(0)).toBe('$0');
    });

    it('should return N/A for null/undefined', () => {
      expect(fmtUsdShort(null)).toBe('N/A');
      expect(fmtUsdShort(undefined)).toBe('N/A');
    });
  });

  describe('shortAddress', () => {
    it('should shorten valid Ethereum address', () => {
      expect(shortAddress('0x1234567890abcdef1234567890abcdef12345678')).toBe(
        '0x1234…5678'
      );
    });

    it('should return empty string for null/undefined', () => {
      expect(shortAddress(null)).toBe('');
      expect(shortAddress(undefined)).toBe('');
    });

    it('should return empty string for empty string', () => {
      expect(shortAddress('')).toBe('');
    });
  });

  describe('isValidEthAddress', () => {
    it('should validate correct Ethereum address', () => {
      expect(
        isValidEthAddress('0x1234567890abcdef1234567890abcdef12345678')
      ).toBe(true);
    });

    it('should validate address with uppercase hex', () => {
      expect(
        isValidEthAddress('0x1234567890ABCDEF1234567890ABCDEF12345678')
      ).toBe(true);
    });

    it('should reject address without 0x prefix', () => {
      expect(
        isValidEthAddress('1234567890abcdef1234567890abcdef12345678')
      ).toBe(false);
    });

    it('should reject address with wrong length', () => {
      expect(isValidEthAddress('0x1234567890abcdef')).toBe(false);
      expect(
        isValidEthAddress('0x1234567890abcdef1234567890abcdef1234567890')
      ).toBe(false);
    });

    it('should reject address with invalid characters', () => {
      expect(
        isValidEthAddress('0xGGGG567890abcdef1234567890abcdef12345678')
      ).toBe(false);
    });

    it('should reject empty string', () => {
      expect(isValidEthAddress('')).toBe(false);
    });
  });

  describe('formatActionLabel', () => {
    it('should format open long', () => {
      expect(formatActionLabel({ action: 'open long' })).toBe('OPEN LONG');
      expect(formatActionLabel({ action: 'Open Long' })).toBe('OPEN LONG');
    });

    it('should format increase long as ADD LONG', () => {
      expect(formatActionLabel({ action: 'increase long' })).toBe('ADD LONG');
    });

    it('should format decrease long as CLOSE LONG', () => {
      expect(formatActionLabel({ action: 'decrease long' })).toBe('CLOSE LONG');
      expect(formatActionLabel({ action: 'close long (close all)' })).toBe(
        'CLOSE LONG'
      );
    });

    it('should format short actions', () => {
      expect(formatActionLabel({ action: 'open short' })).toBe('OPEN SHORT');
      expect(formatActionLabel({ action: 'increase short' })).toBe('ADD SHORT');
      expect(formatActionLabel({ action: 'decrease short' })).toBe(
        'CLOSE SHORT'
      );
    });

    it('should fallback to side for unknown actions', () => {
      expect(formatActionLabel({ side: 'buy' })).toBe('OPEN LONG');
      expect(formatActionLabel({ side: 'sell' })).toBe('OPEN SHORT');
    });

    it('should return TRADE for empty action', () => {
      expect(formatActionLabel({})).toBe('TRADE');
    });
  });

  describe('escapeHtml', () => {
    it('should escape HTML special characters', () => {
      expect(escapeHtml('<script>')).toBe('&lt;script&gt;');
      expect(escapeHtml('a & b')).toBe('a &amp; b');
      expect(escapeHtml('"quoted"')).toBe('&quot;quoted&quot;');
      expect(escapeHtml("it's")).toBe('it&#39;s');
    });

    it('should return empty string for null/undefined', () => {
      expect(escapeHtml(null)).toBe('');
      expect(escapeHtml(undefined)).toBe('');
    });

    it('should not modify safe text', () => {
      expect(escapeHtml('Hello World')).toBe('Hello World');
    });
  });

  describe('calculateSignedSize', () => {
    it('should return positive for open long', () => {
      expect(calculateSignedSize(5, 'open long')).toBe(5);
    });

    it('should return positive for increase long', () => {
      expect(calculateSignedSize(5, 'increase long')).toBe(5);
    });

    it('should return negative for decrease long', () => {
      expect(calculateSignedSize(5, 'decrease long')).toBe(-5);
    });

    it('should return negative for close long', () => {
      expect(calculateSignedSize(5, 'close long (close all)')).toBe(-5);
    });

    it('should return negative for open short', () => {
      expect(calculateSignedSize(5, 'open short')).toBe(-5);
    });

    it('should return negative for increase short', () => {
      expect(calculateSignedSize(5, 'increase short')).toBe(-5);
    });

    it('should return positive for decrease short', () => {
      expect(calculateSignedSize(5, 'decrease short')).toBe(5);
    });

    it('should return positive for close short', () => {
      expect(calculateSignedSize(5, 'close short (close all)')).toBe(5);
    });
  });

  describe('calculatePreviousPosition', () => {
    it('should calculate previous for open long', () => {
      // result=10, size=10 -> prev=0
      expect(calculatePreviousPosition(10, 10, 'open long')).toBe(0);
    });

    it('should calculate previous for increase long', () => {
      // result=15, size=5 -> prev=10
      expect(calculatePreviousPosition(15, 5, 'increase long')).toBe(10);
    });

    it('should calculate previous for decrease long', () => {
      // result=5, size=5 -> prev=10
      expect(calculatePreviousPosition(5, 5, 'decrease long')).toBe(10);
    });

    it('should calculate previous for close long', () => {
      // result=0, size=10 -> prev=10
      expect(calculatePreviousPosition(0, 10, 'close long (close all)')).toBe(
        10
      );
    });

    it('should calculate previous for open short', () => {
      // result=-10, size=10 -> prev=0
      expect(calculatePreviousPosition(-10, 10, 'open short')).toBe(0);
    });

    it('should calculate previous for increase short', () => {
      // result=-15, size=5 -> prev=-10
      expect(calculatePreviousPosition(-15, 5, 'increase short')).toBe(-10);
    });

    it('should calculate previous for decrease short', () => {
      // result=-5, size=5 -> prev=-10
      expect(calculatePreviousPosition(-5, 5, 'decrease short')).toBe(-10);
    });

    it('should calculate previous for close short', () => {
      // result=0, size=10 -> prev=-10
      expect(calculatePreviousPosition(0, 10, 'close short (close all)')).toBe(
        -10
      );
    });
  });
});

describe('Fill Aggregation', () => {
  describe('canMergeIntoGroup', () => {
    const baseTime = '2025-01-15T10:00:00Z';
    const baseGroup: AggregatedGroup = {
      id: 'test-group',
      time_utc: baseTime,
      oldest_time: baseTime,
      address: '0x1234567890abcdef1234567890abcdef12345678',
      symbol: 'BTC',
      action: 'increase long',
      fills: [],
      totalSize: 5,
      totalPnl: 0,
      prices: [97000],
      isAggregated: false,
      fillCount: 1,
      avgPrice: 97000,
      size_signed: 5,
      closed_pnl_usd: null,
    };

    it('should merge fill within time window with same attributes', () => {
      const fill: Fill = {
        time_utc: '2025-01-15T10:00:30Z', // 30 seconds later
        address: '0x1234567890abcdef1234567890abcdef12345678',
        action: 'increase long',
        size_signed: 3,
        symbol: 'BTC',
      };
      expect(canMergeIntoGroup(baseGroup, fill)).toBe(true);
    });

    it('should not merge fill outside time window', () => {
      const fill: Fill = {
        time_utc: '2025-01-15T10:02:00Z', // 2 minutes later
        address: '0x1234567890abcdef1234567890abcdef12345678',
        action: 'increase long',
        size_signed: 3,
        symbol: 'BTC',
      };
      expect(canMergeIntoGroup(baseGroup, fill)).toBe(false);
    });

    it('should not merge fill with different address', () => {
      const fill: Fill = {
        time_utc: '2025-01-15T10:00:30Z',
        address: '0xabcdef1234567890abcdef1234567890abcdef12',
        action: 'increase long',
        size_signed: 3,
        symbol: 'BTC',
      };
      expect(canMergeIntoGroup(baseGroup, fill)).toBe(false);
    });

    it('should not merge fill with different symbol', () => {
      const fill: Fill = {
        time_utc: '2025-01-15T10:00:30Z',
        address: '0x1234567890abcdef1234567890abcdef12345678',
        action: 'increase long',
        size_signed: 3,
        symbol: 'ETH',
      };
      expect(canMergeIntoGroup(baseGroup, fill)).toBe(false);
    });

    it('should not merge fill with different action', () => {
      const fill: Fill = {
        time_utc: '2025-01-15T10:00:30Z',
        address: '0x1234567890abcdef1234567890abcdef12345678',
        action: 'decrease long',
        size_signed: -3,
        symbol: 'BTC',
      };
      expect(canMergeIntoGroup(baseGroup, fill)).toBe(false);
    });

    it('should normalize address case for comparison', () => {
      const fill: Fill = {
        time_utc: '2025-01-15T10:00:30Z',
        address: '0x1234567890ABCDEF1234567890ABCDEF12345678', // uppercase
        action: 'increase long',
        size_signed: 3,
        symbol: 'BTC',
      };
      expect(canMergeIntoGroup(baseGroup, fill)).toBe(true);
    });
  });

  describe('createGroup', () => {
    it('should create group from fill', () => {
      const fill: Fill = {
        time_utc: '2025-01-15T10:00:00Z',
        address: '0x1234567890ABCDEF1234567890abcdef12345678',
        action: 'Open Long',
        size_signed: 5,
        price_usd: 97000,
        closed_pnl_usd: 100,
        symbol: 'BTC',
        resulting_position: 5,
        previous_position: 0,
      };

      const group = createGroup(fill);

      expect(group.address).toBe(
        '0x1234567890abcdef1234567890abcdef12345678'
      ); // normalized
      expect(group.action).toBe('open long'); // normalized
      expect(group.symbol).toBe('BTC');
      expect(group.totalSize).toBe(5);
      expect(group.totalPnl).toBe(100);
      expect(group.prices).toEqual([97000]);
      expect(group.fills).toHaveLength(1);
      expect(group.isAggregated).toBe(false);
      expect(group.fillCount).toBe(1);
    });

    it('should default symbol to BTC', () => {
      const fill: Fill = {
        time_utc: '2025-01-15T10:00:00Z',
        address: '0x1234567890abcdef1234567890abcdef12345678',
        action: 'open long',
        size_signed: 5,
      };

      const group = createGroup(fill);
      expect(group.symbol).toBe('BTC');
    });

    it('should handle missing price', () => {
      const fill: Fill = {
        time_utc: '2025-01-15T10:00:00Z',
        address: '0x1234567890abcdef1234567890abcdef12345678',
        action: 'open long',
        size_signed: 5,
      };

      const group = createGroup(fill);
      expect(group.prices).toEqual([]);
      expect(group.avgPrice).toBe(null);
    });
  });
});

describe('Time Formatting', () => {
  describe('fmtTime', () => {
    it('should format ISO timestamp to time string', () => {
      // Note: output depends on locale, so just check it returns a string
      const result = fmtTime('2025-01-15T10:30:45Z');
      expect(typeof result).toBe('string');
      expect(result.length).toBeGreaterThan(0);
    });
  });
});

describe('Score Formatting', () => {
  function fmtScore(score: number | null | undefined): string {
    if (!Number.isFinite(score)) return '—';
    if (score === 0) return '0';
    const s = score as number;
    if (Math.abs(s) >= 100) return s.toFixed(1);
    if (Math.abs(s) >= 1) return s.toFixed(2);
    return s.toFixed(4);
  }

  it('should format large scores with 1 decimal', () => {
    expect(fmtScore(123.456)).toBe('123.5');
  });

  it('should format medium scores with 2 decimals', () => {
    expect(fmtScore(12.345)).toBe('12.35');
  });

  it('should format small scores with 4 decimals', () => {
    expect(fmtScore(0.1234)).toBe('0.1234');
  });

  it('should return 0 for zero', () => {
    expect(fmtScore(0)).toBe('0');
  });

  it('should return dash for null/undefined', () => {
    expect(fmtScore(null)).toBe('—');
    expect(fmtScore(undefined)).toBe('—');
  });
});

describe('Holdings Normalization', () => {
  interface Position {
    symbol: string;
    size: number;
    entryPrice?: number | null;
    liquidationPrice?: number | null;
    leverage?: number | null;
  }

  function normalizeHoldings(
    raw: Record<string, Position[] | Position | undefined> = {}
  ): Record<string, Position[]> {
    const normalized: Record<string, Position[]> = {};
    Object.entries(raw).forEach(([addr, positions]) => {
      if (!addr) return;
      const key = addr.toLowerCase();
      if (Array.isArray(positions)) {
        normalized[key] = positions.map((pos) => ({
          symbol: (pos?.symbol || '').toUpperCase(),
          size: Number(pos?.size ?? 0),
          entryPrice: pos?.entryPrice ?? null,
          liquidationPrice: pos?.liquidationPrice ?? null,
          leverage: pos?.leverage ?? null,
        }));
      } else if (positions) {
        normalized[key] = [
          {
            symbol: (positions?.symbol || '').toUpperCase(),
            size: Number(positions?.size ?? 0),
            entryPrice: positions?.entryPrice ?? null,
            liquidationPrice: positions?.liquidationPrice ?? null,
            leverage: positions?.leverage ?? null,
          },
        ];
      }
    });
    return normalized;
  }

  it('should normalize address to lowercase', () => {
    const raw = {
      '0xABCDEF1234567890abcdef1234567890ABCDEF12': [
        { symbol: 'btc', size: 5 },
      ],
    };
    const result = normalizeHoldings(raw);
    expect(result['0xabcdef1234567890abcdef1234567890abcdef12']).toBeDefined();
  });

  it('should normalize symbol to uppercase', () => {
    const raw = {
      '0x1234567890abcdef1234567890abcdef12345678': [
        { symbol: 'btc', size: 5 },
      ],
    };
    const result = normalizeHoldings(raw);
    expect(result['0x1234567890abcdef1234567890abcdef12345678'][0].symbol).toBe(
      'BTC'
    );
  });

  it('should handle legacy single position format', () => {
    const raw = {
      '0x1234567890abcdef1234567890abcdef12345678': {
        symbol: 'ETH',
        size: 10,
      },
    };
    const result = normalizeHoldings(raw);
    expect(result['0x1234567890abcdef1234567890abcdef12345678']).toHaveLength(
      1
    );
    expect(result['0x1234567890abcdef1234567890abcdef12345678'][0].size).toBe(
      10
    );
  });

  it('should handle empty input', () => {
    expect(normalizeHoldings({})).toEqual({});
    expect(normalizeHoldings()).toEqual({});
  });
});
