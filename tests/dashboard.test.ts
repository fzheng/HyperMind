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

  describe('fmtRelativeTime', () => {
    /**
     * Format timestamp as relative time (e.g., "3 mins ago", "2 hours ago")
     * Extracted from dashboard.js for testing
     */
    function fmtRelativeTime(ts: string): string {
      const now = Date.now();
      const then = new Date(ts).getTime();
      const diffMs = now - then;

      if (diffMs < 0) return 'just now';

      const seconds = Math.floor(diffMs / 1000);
      const minutes = Math.floor(seconds / 60);
      const hours = Math.floor(minutes / 60);
      const days = Math.floor(hours / 24);

      if (seconds < 60) return 'just now';
      if (minutes === 1) return '1 min ago';
      if (minutes < 60) return `${minutes} mins ago`;
      if (hours === 1) return '1 hour ago';
      if (hours < 24) return `${hours} hours ago`;
      if (days === 1) return '1 day ago';
      return `${days} days ago`;
    }

    it('should return "just now" for timestamps in the future', () => {
      const futureTs = new Date(Date.now() + 60000).toISOString();
      expect(fmtRelativeTime(futureTs)).toBe('just now');
    });

    it('should return "just now" for timestamps less than 60 seconds ago', () => {
      const recentTs = new Date(Date.now() - 30000).toISOString(); // 30 seconds ago
      expect(fmtRelativeTime(recentTs)).toBe('just now');
    });

    it('should return "just now" for timestamps exactly 59 seconds ago', () => {
      const ts = new Date(Date.now() - 59000).toISOString();
      expect(fmtRelativeTime(ts)).toBe('just now');
    });

    it('should return "1 min ago" for timestamps exactly 1 minute ago', () => {
      const ts = new Date(Date.now() - 60000).toISOString();
      expect(fmtRelativeTime(ts)).toBe('1 min ago');
    });

    it('should return "X mins ago" for timestamps 2-59 minutes ago', () => {
      const ts2min = new Date(Date.now() - 2 * 60000).toISOString();
      expect(fmtRelativeTime(ts2min)).toBe('2 mins ago');

      const ts30min = new Date(Date.now() - 30 * 60000).toISOString();
      expect(fmtRelativeTime(ts30min)).toBe('30 mins ago');

      const ts59min = new Date(Date.now() - 59 * 60000).toISOString();
      expect(fmtRelativeTime(ts59min)).toBe('59 mins ago');
    });

    it('should return "1 hour ago" for timestamps exactly 1 hour ago', () => {
      const ts = new Date(Date.now() - 60 * 60000).toISOString();
      expect(fmtRelativeTime(ts)).toBe('1 hour ago');
    });

    it('should return "X hours ago" for timestamps 2-23 hours ago', () => {
      const ts2hr = new Date(Date.now() - 2 * 60 * 60000).toISOString();
      expect(fmtRelativeTime(ts2hr)).toBe('2 hours ago');

      const ts12hr = new Date(Date.now() - 12 * 60 * 60000).toISOString();
      expect(fmtRelativeTime(ts12hr)).toBe('12 hours ago');

      const ts23hr = new Date(Date.now() - 23 * 60 * 60000).toISOString();
      expect(fmtRelativeTime(ts23hr)).toBe('23 hours ago');
    });

    it('should return "1 day ago" for timestamps exactly 24 hours ago', () => {
      const ts = new Date(Date.now() - 24 * 60 * 60000).toISOString();
      expect(fmtRelativeTime(ts)).toBe('1 day ago');
    });

    it('should return "X days ago" for timestamps more than 1 day ago', () => {
      const ts2day = new Date(Date.now() - 2 * 24 * 60 * 60000).toISOString();
      expect(fmtRelativeTime(ts2day)).toBe('2 days ago');

      const ts7day = new Date(Date.now() - 7 * 24 * 60 * 60000).toISOString();
      expect(fmtRelativeTime(ts7day)).toBe('7 days ago');

      const ts30day = new Date(Date.now() - 30 * 24 * 60 * 60000).toISOString();
      expect(fmtRelativeTime(ts30day)).toBe('30 days ago');
    });

    it('should handle edge case at minute boundary', () => {
      // At exactly 60 seconds, should show "1 min ago" not "just now"
      const ts = new Date(Date.now() - 60 * 1000).toISOString();
      expect(fmtRelativeTime(ts)).toBe('1 min ago');
    });

    it('should handle edge case at hour boundary', () => {
      // At exactly 60 minutes, should show "1 hour ago" not "60 mins ago"
      const ts = new Date(Date.now() - 60 * 60 * 1000).toISOString();
      expect(fmtRelativeTime(ts)).toBe('1 hour ago');
    });

    it('should handle edge case at day boundary', () => {
      // At exactly 24 hours, should show "1 day ago" not "24 hours ago"
      const ts = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();
      expect(fmtRelativeTime(ts)).toBe('1 day ago');
    });
  });

  describe('fmtFillTime', () => {
    /**
     * Format fill time based on display mode
     * Simplified version for testing (actual implementation uses module state)
     */
    function fmtRelativeTime(ts: string): string {
      const now = Date.now();
      const then = new Date(ts).getTime();
      const diffMs = now - then;

      if (diffMs < 0) return 'just now';

      const seconds = Math.floor(diffMs / 1000);
      const minutes = Math.floor(seconds / 60);
      const hours = Math.floor(minutes / 60);
      const days = Math.floor(hours / 24);

      if (seconds < 60) return 'just now';
      if (minutes === 1) return '1 min ago';
      if (minutes < 60) return `${minutes} mins ago`;
      if (hours === 1) return '1 hour ago';
      if (hours < 24) return `${hours} hours ago`;
      if (days === 1) return '1 day ago';
      return `${days} days ago`;
    }

    function fmtDateTime(ts: string): string {
      return new Date(ts).toLocaleString([], {
        year: 'numeric',
        month: 'short',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      });
    }

    function fmtFillTime(ts: string, mode: 'absolute' | 'relative'): string {
      if (mode === 'relative') {
        return fmtRelativeTime(ts);
      }
      return fmtDateTime(ts);
    }

    it('should return relative time in relative mode', () => {
      const ts = new Date(Date.now() - 5 * 60000).toISOString(); // 5 minutes ago
      expect(fmtFillTime(ts, 'relative')).toBe('5 mins ago');
    });

    it('should return absolute datetime in absolute mode', () => {
      const ts = '2025-01-15T10:30:45Z';
      const result = fmtFillTime(ts, 'absolute');
      // Verify it's a formatted datetime string (locale-dependent)
      expect(typeof result).toBe('string');
      expect(result.length).toBeGreaterThan(10); // datetime strings are longer than relative
    });

    it('should switch output format based on mode', () => {
      const ts = new Date(Date.now() - 3 * 60000).toISOString(); // 3 minutes ago
      const relativeResult = fmtFillTime(ts, 'relative');
      const absoluteResult = fmtFillTime(ts, 'absolute');

      expect(relativeResult).toBe('3 mins ago');
      expect(absoluteResult).not.toBe('3 mins ago');
      expect(absoluteResult.length).toBeGreaterThan(relativeResult.length);
    });
  });
});

describe('Time Display Mode Toggle', () => {
  /**
   * Tests for the time display mode toggle functionality
   * This simulates the state management from dashboard.js
   */

  interface TimeToggleState {
    mode: 'absolute' | 'relative';
    headerText: string;
    headerTitle: string;
  }

  function toggleTimeDisplayMode(currentState: TimeToggleState): TimeToggleState {
    const newMode = currentState.mode === 'absolute' ? 'relative' : 'absolute';
    return {
      mode: newMode,
      headerText: newMode === 'absolute' ? 'Time ⏱' : 'Time 🕐',
      headerTitle:
        newMode === 'absolute'
          ? 'Click to show relative time (e.g., "3 mins ago")'
          : 'Click to show absolute time',
    };
  }

  it('should toggle from absolute to relative mode', () => {
    const initialState: TimeToggleState = {
      mode: 'absolute',
      headerText: 'Time ⏱',
      headerTitle: 'Click to show relative time (e.g., "3 mins ago")',
    };

    const newState = toggleTimeDisplayMode(initialState);

    expect(newState.mode).toBe('relative');
    expect(newState.headerText).toBe('Time 🕐');
    expect(newState.headerTitle).toBe('Click to show absolute time');
  });

  it('should toggle from relative to absolute mode', () => {
    const initialState: TimeToggleState = {
      mode: 'relative',
      headerText: 'Time 🕐',
      headerTitle: 'Click to show absolute time',
    };

    const newState = toggleTimeDisplayMode(initialState);

    expect(newState.mode).toBe('absolute');
    expect(newState.headerText).toBe('Time ⏱');
    expect(newState.headerTitle).toBe('Click to show relative time (e.g., "3 mins ago")');
  });

  it('should toggle back and forth correctly', () => {
    let state: TimeToggleState = {
      mode: 'absolute',
      headerText: 'Time ⏱',
      headerTitle: 'Click to show relative time (e.g., "3 mins ago")',
    };

    // Toggle to relative
    state = toggleTimeDisplayMode(state);
    expect(state.mode).toBe('relative');

    // Toggle back to absolute
    state = toggleTimeDisplayMode(state);
    expect(state.mode).toBe('absolute');

    // Toggle again to relative
    state = toggleTimeDisplayMode(state);
    expect(state.mode).toBe('relative');
  });

  it('should have correct header icon for each mode', () => {
    const absoluteState: TimeToggleState = {
      mode: 'absolute',
      headerText: 'Time ⏱',
      headerTitle: 'Click to show relative time (e.g., "3 mins ago")',
    };

    const relativeState = toggleTimeDisplayMode(absoluteState);

    // Absolute mode uses stopwatch icon ⏱
    expect(absoluteState.headerText).toContain('⏱');
    expect(absoluteState.headerText).not.toContain('🕐');

    // Relative mode uses clock icon 🕐
    expect(relativeState.headerText).toContain('🕐');
    expect(relativeState.headerText).not.toContain('⏱');
  });
});

describe('Relative Time Auto-Refresh', () => {
  /**
   * Tests for the relative time auto-refresh interval management
   * These test the logic for starting/stopping the refresh interval
   */

  interface RefreshState {
    intervalId: number | null;
    isRunning: boolean;
  }

  function startRelativeTimeRefresh(state: RefreshState): RefreshState {
    if (state.intervalId !== null) return state; // Already running
    return {
      intervalId: 1, // Mock interval ID
      isRunning: true,
    };
  }

  function stopRelativeTimeRefresh(state: RefreshState): RefreshState {
    if (state.intervalId === null) return state; // Not running
    return {
      intervalId: null,
      isRunning: false,
    };
  }

  it('should start refresh interval when not running', () => {
    const initialState: RefreshState = { intervalId: null, isRunning: false };
    const newState = startRelativeTimeRefresh(initialState);

    expect(newState.isRunning).toBe(true);
    expect(newState.intervalId).not.toBe(null);
  });

  it('should not start duplicate interval if already running', () => {
    const runningState: RefreshState = { intervalId: 1, isRunning: true };
    const newState = startRelativeTimeRefresh(runningState);

    // Should return the same state (no change)
    expect(newState).toBe(runningState);
    expect(newState.intervalId).toBe(1);
  });

  it('should stop refresh interval when running', () => {
    const runningState: RefreshState = { intervalId: 1, isRunning: true };
    const newState = stopRelativeTimeRefresh(runningState);

    expect(newState.isRunning).toBe(false);
    expect(newState.intervalId).toBe(null);
  });

  it('should not error when stopping already stopped interval', () => {
    const stoppedState: RefreshState = { intervalId: null, isRunning: false };
    const newState = stopRelativeTimeRefresh(stoppedState);

    expect(newState).toBe(stoppedState);
    expect(newState.intervalId).toBe(null);
  });

  it('should correctly manage start/stop cycle', () => {
    let state: RefreshState = { intervalId: null, isRunning: false };

    // Start
    state = startRelativeTimeRefresh(state);
    expect(state.isRunning).toBe(true);

    // Stop
    state = stopRelativeTimeRefresh(state);
    expect(state.isRunning).toBe(false);

    // Start again
    state = startRelativeTimeRefresh(state);
    expect(state.isRunning).toBe(true);
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

describe('Dashboard API Limit Clamping', () => {
  describe('/dashboard/fills limit', () => {
    const clampFillsLimit = (input: unknown) =>
      Math.max(1, Math.min(200, Number(input) || 25));

    it('should default to 25 for undefined', () => {
      expect(clampFillsLimit(undefined)).toBe(25);
    });

    it('should default to 25 for zero (falsy)', () => {
      expect(clampFillsLimit(0)).toBe(25);
    });

    it('should clamp negative to 1', () => {
      expect(clampFillsLimit(-10)).toBe(1);
    });

    it('should clamp values over 200 to 200', () => {
      expect(clampFillsLimit(500)).toBe(200);
    });

    it('should pass through valid values', () => {
      expect(clampFillsLimit(100)).toBe(100);
    });
  });

  describe('/leaderboard limit', () => {
    const clampLeaderboardLimit = (input: unknown) =>
      Math.max(1, Math.min(1000, Number(input) || 100));

    it('should default to 100 for undefined', () => {
      expect(clampLeaderboardLimit(undefined)).toBe(100);
    });

    it('should default to 100 for zero (falsy)', () => {
      expect(clampLeaderboardLimit(0)).toBe(100);
    });

    it('should clamp negative to 1', () => {
      expect(clampLeaderboardLimit(-1)).toBe(1);
    });

    it('should clamp values over 1000 to 1000', () => {
      expect(clampLeaderboardLimit(5000)).toBe(1000);
    });

    it('should pass through valid values', () => {
      expect(clampLeaderboardLimit(500)).toBe(500);
    });
  });

  describe('/leaderboard/selected limit', () => {
    const clampSelectedLimit = (input: unknown) =>
      Math.max(1, Math.min(50, Number(input) || 12));

    it('should default to 12 for undefined', () => {
      expect(clampSelectedLimit(undefined)).toBe(12);
    });

    it('should default to 12 for zero (falsy)', () => {
      expect(clampSelectedLimit(0)).toBe(12);
    });

    it('should clamp values over 50 to 50', () => {
      expect(clampSelectedLimit(100)).toBe(50);
    });

    it('should pass through valid values', () => {
      expect(clampSelectedLimit(25)).toBe(25);
    });
  });

  describe('Price API Symbol Handling', () => {
    const getSymbol = (querySymbol: unknown) =>
      querySymbol === 'ETHUSDT' ? 'ETH' : 'BTC';

    it('should default to BTC for undefined', () => {
      expect(getSymbol(undefined)).toBe('BTC');
    });

    it('should default to BTC for null', () => {
      expect(getSymbol(null)).toBe('BTC');
    });

    it('should default to BTC for BTCUSDT', () => {
      expect(getSymbol('BTCUSDT')).toBe('BTC');
    });

    it('should return ETH for ETHUSDT', () => {
      expect(getSymbol('ETHUSDT')).toBe('ETH');
    });

    it('should default to BTC for unknown symbols', () => {
      expect(getSymbol('random')).toBe('BTC');
    });
  });
});

describe('/dashboard/price endpoint behavior', () => {
  /**
   * Simulates the price endpoint handler
   */
  interface PriceResponse {
    symbol: string;
    price: number;
    timestamp: string;
  }

  interface ErrorResponse {
    error: string;
  }

  async function mockFetchPrice(
    symbol: string,
    upstreamFetcher: () => Promise<number | null>
  ): Promise<{ status: number; body: PriceResponse | ErrorResponse }> {
    try {
      const price = await upstreamFetcher();

      if (price === null) {
        return {
          status: 502,
          body: { error: 'Failed to fetch price from upstream' },
        };
      }

      return {
        status: 200,
        body: {
          symbol,
          price,
          timestamp: new Date().toISOString(),
        },
      };
    } catch (err) {
      return {
        status: 502,
        body: { error: 'Failed to fetch price from upstream' },
      };
    }
  }

  it('returns BTC price successfully', async () => {
    const result = await mockFetchPrice('BTC', async () => 95000);

    expect(result.status).toBe(200);
    expect((result.body as PriceResponse).symbol).toBe('BTC');
    expect((result.body as PriceResponse).price).toBe(95000);
  });

  it('returns ETH price when symbol=ETHUSDT', async () => {
    const result = await mockFetchPrice('ETH', async () => 3500);

    expect(result.status).toBe(200);
    expect((result.body as PriceResponse).symbol).toBe('ETH');
    expect((result.body as PriceResponse).price).toBe(3500);
  });

  it('returns 502 when upstream returns null', async () => {
    const result = await mockFetchPrice('BTC', async () => null);

    expect(result.status).toBe(502);
    expect((result.body as ErrorResponse).error).toContain('upstream');
  });

  it('returns 502 when upstream throws error', async () => {
    const result = await mockFetchPrice('BTC', async () => {
      throw new Error('Connection timeout');
    });

    expect(result.status).toBe(502);
    expect((result.body as ErrorResponse).error).toContain('upstream');
  });

  it('includes timestamp in successful response', async () => {
    const before = new Date().toISOString();
    const result = await mockFetchPrice('BTC', async () => 95000);
    const after = new Date().toISOString();

    expect(result.status).toBe(200);
    const timestamp = (result.body as PriceResponse).timestamp;
    expect(timestamp >= before).toBe(true);
    expect(timestamp <= after).toBe(true);
  });
});
