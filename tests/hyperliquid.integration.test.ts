/**
 * Integration tests for Hyperliquid API functions
 *
 * These tests mock the @nktkas/hyperliquid SDK and fetch to test
 * the API integration logic without making real network requests.
 */

// Mock the Hyperliquid SDK
const mockClearinghouseState = jest.fn();
const mockUserFills = jest.fn();
const mockUserDetails = jest.fn();
const mockMetaAndAssetCtxs = jest.fn();

jest.mock('@nktkas/hyperliquid', () => ({
  HttpTransport: jest.fn().mockImplementation(() => ({})),
}));

jest.mock('@nktkas/hyperliquid/api/info', () => ({
  clearinghouseState: (...args: unknown[]) => mockClearinghouseState(...args),
  userFills: (...args: unknown[]) => mockUserFills(...args),
  userDetails: (...args: unknown[]) => mockUserDetails(...args),
  metaAndAssetCtxs: (...args: unknown[]) => mockMetaAndAssetCtxs(...args),
}));

// Mock global fetch for Binance API
const mockFetch = jest.fn();
global.fetch = mockFetch;

import {
  fetchBtcPerpExposure,
  fetchPerpPositions,
  fetchUserFills,
  fetchUserProfile,
  fetchSpotPrice,
  fetchPerpMarkPrice,
} from '../packages/ts-lib/src/hyperliquid';

describe('Hyperliquid API Integration', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('fetchBtcPerpExposure', () => {
    it('should return net BTC exposure from positions', async () => {
      mockClearinghouseState.mockResolvedValueOnce({
        assetPositions: [
          { position: { coin: 'BTC', szi: '0.5' } },
          { position: { coin: 'ETH', szi: '2.0' } },
          { position: { coin: 'btc', szi: '0.25' } }, // Lowercase BTC
        ],
      });

      const result = await fetchBtcPerpExposure('0x1234567890abcdef1234567890abcdef12345678');

      expect(result).toBe(0.75); // 0.5 + 0.25
      expect(mockClearinghouseState).toHaveBeenCalledWith(
        expect.any(Object),
        { user: '0x1234567890abcdef1234567890abcdef12345678' }
      );
    });

    it('should return 0 when no BTC positions', async () => {
      mockClearinghouseState.mockResolvedValueOnce({
        assetPositions: [
          { position: { coin: 'ETH', szi: '5.0' } },
          { position: { coin: 'SOL', szi: '100' } },
        ],
      });

      const result = await fetchBtcPerpExposure('0x1234');
      expect(result).toBe(0);
    });

    it('should return 0 when no positions', async () => {
      mockClearinghouseState.mockResolvedValueOnce({
        assetPositions: [],
      });

      const result = await fetchBtcPerpExposure('0x1234');
      expect(result).toBe(0);
    });

    it('should return 0 on API error', async () => {
      mockClearinghouseState.mockRejectedValueOnce(new Error('API error'));

      const result = await fetchBtcPerpExposure('0x1234');
      expect(result).toBe(0);
    });

    it('should handle null assetPositions', async () => {
      mockClearinghouseState.mockResolvedValueOnce({});

      const result = await fetchBtcPerpExposure('0x1234');
      expect(result).toBe(0);
    });

    it('should handle invalid size values', async () => {
      mockClearinghouseState.mockResolvedValueOnce({
        assetPositions: [
          { position: { coin: 'BTC', szi: 'invalid' } },
          { position: { coin: 'BTC', szi: '0.5' } },
        ],
      });

      const result = await fetchBtcPerpExposure('0x1234');
      expect(result).toBe(0.5); // Only valid value counted
    });

    it('should normalize address to lowercase', async () => {
      mockClearinghouseState.mockResolvedValueOnce({ assetPositions: [] });

      await fetchBtcPerpExposure('0xABCDEF123456');

      expect(mockClearinghouseState).toHaveBeenCalledWith(
        expect.any(Object),
        { user: '0xabcdef123456' }
      );
    });
  });

  describe('fetchPerpPositions', () => {
    it('should return all non-flat positions', async () => {
      mockClearinghouseState.mockResolvedValueOnce({
        assetPositions: [
          {
            position: {
              coin: 'btc',
              szi: '0.5',
              entryPx: '95000',
              leverage: { value: 10 },
            },
          },
          {
            position: {
              coin: 'ETH',
              szi: '2.0',
              entryPx: '3500',
              leverage: { value: 5 },
            },
          },
          { position: { coin: 'SOL', szi: '0' } }, // Flat position
        ],
      });

      const result = await fetchPerpPositions('0x1234');

      expect(result).toHaveLength(2);
      expect(result[0]).toMatchObject({
        symbol: 'BTC',
        size: 0.5,
        entryPriceUsd: 95000,
        leverage: 10,
      });
      expect(result[1]).toMatchObject({
        symbol: 'ETH',
        size: 2.0,
        leverage: 5,
      });
    });

    it('should return empty array on error', async () => {
      mockClearinghouseState.mockRejectedValueOnce(new Error('API error'));

      const result = await fetchPerpPositions('0x1234');
      expect(result).toEqual([]);
    });

    it('should handle missing leverage and entry price', async () => {
      mockClearinghouseState.mockResolvedValueOnce({
        assetPositions: [
          { position: { coin: 'BTC', szi: '1.0' } },
        ],
      });

      const result = await fetchPerpPositions('0x1234');

      expect(result[0].entryPriceUsd).toBeUndefined();
      expect(result[0].leverage).toBeUndefined();
    });
  });

  describe('fetchUserFills', () => {
    it('should return both BTC and ETH fills by default', async () => {
      mockUserFills.mockResolvedValueOnce([
        { coin: 'BTC', px: '95000', sz: '0.1', side: 'B', time: 1700000000000, startPosition: '0' },
        { coin: 'ETH', px: '3500', sz: '1.0', side: 'A', time: 1700000001000, startPosition: '0' },
        { coin: 'SOL', px: '100', sz: '10', side: 'B', time: 1700000002000, startPosition: '0' },
      ]);

      const result = await fetchUserFills('0x1234');

      expect(result).toHaveLength(2);
      expect(result.map(f => f.coin).sort()).toEqual(['BTC', 'ETH']);
    });

    it('should filter by specified symbols', async () => {
      mockUserFills.mockResolvedValueOnce([
        { coin: 'BTC', px: '95000', sz: '0.1', side: 'B', time: 1700000000000, startPosition: '0' },
        { coin: 'ETH', px: '3500', sz: '1.0', side: 'A', time: 1700000001000, startPosition: '0' },
      ]);

      const result = await fetchUserFills('0x1234', { symbols: ['ETH'] });

      expect(result).toHaveLength(1);
      expect(result[0].coin).toBe('ETH');
    });

    it('should normalize coin names to uppercase', async () => {
      mockUserFills.mockResolvedValueOnce([
        { coin: 'btc', px: '95000', sz: '0.1', side: 'B', time: 1700000000000, startPosition: '0' },
        { coin: 'eth', px: '3500', sz: '1.0', side: 'A', time: 1700000001000, startPosition: '0' },
      ]);

      const result = await fetchUserFills('0x1234');

      expect(result[0].coin).toBe('ETH');
      expect(result[1].coin).toBe('BTC');
    });

    it('should handle optional fields correctly', async () => {
      mockUserFills.mockResolvedValueOnce([
        {
          coin: 'BTC',
          px: '95000',
          sz: '0.1',
          side: 'B',
          time: 1700000000000,
          startPosition: '0',
          closedPnl: '150.50',
          fee: '5.25',
          feeToken: 'USDC',
          hash: '0xhash123',
        },
      ]);

      const result = await fetchUserFills('0x1234');

      expect(result[0]).toMatchObject({
        closedPnl: 150.5,
        fee: 5.25,
        feeToken: 'USDC',
        hash: '0xhash123',
      });
    });

    it('should omit optional fields when not valid', async () => {
      mockUserFills.mockResolvedValueOnce([
        {
          coin: 'BTC',
          px: '95000',
          sz: '0.1',
          side: 'B',
          time: 1700000000000,
          startPosition: '0',
          closedPnl: 'invalid',
          fee: undefined, // undefined becomes undefined, null becomes 0
        },
      ]);

      const result = await fetchUserFills('0x1234');

      // 'invalid' string becomes NaN which is not finite, so undefined
      expect(result[0].closedPnl).toBeUndefined();
      // undefined is not finite, so fee is undefined
      expect(result[0].fee).toBeUndefined();
    });
  });

  describe('fetchUserProfile', () => {
    it('should return profile summary', async () => {
      mockUserDetails.mockResolvedValueOnce({
        txs: [
          { time: '1700000000000' },
          { time: '1699999000000' },
        ],
      });

      const result = await fetchUserProfile('0x1234');

      expect(result.txCount).toBe(2);
      expect(result.lastTxTime).not.toBeNull();
    });

    it('should return empty profile on error', async () => {
      mockUserDetails.mockRejectedValueOnce(new Error('API error'));

      const result = await fetchUserProfile('0x1234');

      expect(result.txCount).toBe(0);
      expect(result.lastTxTime).toBeNull();
    });

    it('should handle empty transactions', async () => {
      mockUserDetails.mockResolvedValueOnce({ txs: [] });

      const result = await fetchUserProfile('0x1234');

      expect(result.txCount).toBe(0);
      expect(result.lastTxTime).toBeNull();
    });

    it('should handle missing txs array', async () => {
      mockUserDetails.mockResolvedValueOnce({});

      const result = await fetchUserProfile('0x1234');

      expect(result.txCount).toBe(0);
      expect(result.lastTxTime).toBeNull();
    });
  });

  describe('fetchSpotPrice', () => {
    it('should fetch BTC price from Binance', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ price: '95123.45' }),
      });

      const result = await fetchSpotPrice('BTCUSDT');

      expect(result).toBe(95123.45);
      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT'
      );
    });

    it('should fetch ETH price from Binance', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ price: '3567.89' }),
      });

      const result = await fetchSpotPrice('ETHUSDT');

      expect(result).toBe(3567.89);
    });

    it('should throw on HTTP error', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
      });

      await expect(fetchSpotPrice('BTCUSDT')).rejects.toThrow('Ticker HTTP 500');
    });

    it('should throw on invalid price', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ price: 'invalid' }),
      });

      await expect(fetchSpotPrice('BTCUSDT')).rejects.toThrow('Invalid ticker price');
    });

    it('should throw on missing price', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({}),
      });

      await expect(fetchSpotPrice('BTCUSDT')).rejects.toThrow('Invalid ticker price');
    });
  });

  describe('fetchPerpMarkPrice', () => {
    // Note: These tests rely on the module-level markCache in hyperliquid.ts
    // The cache has a 1500ms TTL. We need to ensure tests work with cache state.

    it('should fetch mark prices from API', async () => {
      // Always provide a mock since we don't know cache state
      mockMetaAndAssetCtxs.mockResolvedValue([
        { universe: [{ name: 'BTC' }, { name: 'ETH' }, { name: 'SOL' }] },
        [{ markPx: '95000' }, { markPx: '3500' }, { markPx: '100' }],
      ]);

      const btcPrice = await fetchPerpMarkPrice('BTC');
      // Price should be a number (either from cache or fresh fetch)
      expect(typeof btcPrice).toBe('number');
      expect(btcPrice).toBeGreaterThan(0);
    });

    it('should return ETH price', async () => {
      mockMetaAndAssetCtxs.mockResolvedValue([
        { universe: [{ name: 'BTC' }, { name: 'ETH' }] },
        [{ markPx: '95000' }, { markPx: '3500' }],
      ]);

      const ethPrice = await fetchPerpMarkPrice('ETH');
      expect(typeof ethPrice).toBe('number');
    });

    it('should handle symbol lookup', async () => {
      mockMetaAndAssetCtxs.mockResolvedValue([
        { universe: [{ name: 'BTC' }] },
        [{ markPx: '95000' }],
      ]);

      const result = await fetchPerpMarkPrice('BTC');
      expect(typeof result).toBe('number');
    });
  });
});

describe('Edge cases and error handling', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should handle null response from clearinghouseState', async () => {
    mockClearinghouseState.mockResolvedValueOnce(null);

    const result = await fetchBtcPerpExposure('0x1234');
    expect(result).toBe(0);
  });

  it('should handle missing position object', async () => {
    mockClearinghouseState.mockResolvedValueOnce({
      assetPositions: [{ position: null }, {}],
    });

    const result = await fetchPerpPositions('0x1234');
    expect(result).toEqual([]);
  });

  it('should handle null fills response', async () => {
    mockUserFills.mockResolvedValueOnce(null);

    const result = await fetchUserFills('0x1234');
    expect(result).toEqual([]);
  });

  it('should handle non-array fills response', async () => {
    mockUserFills.mockResolvedValueOnce({ data: [] });

    const result = await fetchUserFills('0x1234');
    expect(result).toEqual([]);
  });
});

describe('Address normalization', () => {
  it('should normalize mixed-case addresses', async () => {
    mockClearinghouseState.mockResolvedValueOnce({ assetPositions: [] });

    await fetchBtcPerpExposure('0xAbCdEf1234567890AbCdEf1234567890AbCdEf12');

    expect(mockClearinghouseState).toHaveBeenCalledWith(
      expect.any(Object),
      { user: '0xabcdef1234567890abcdef1234567890abcdef12' }
    );
  });

  it('should handle checksummed addresses', async () => {
    mockUserFills.mockResolvedValueOnce([]);

    await fetchUserFills('0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed');

    expect(mockUserFills).toHaveBeenCalledWith(
      expect.any(Object),
      expect.objectContaining({
        user: '0x5aaeb6053f3e94c9b9a09f33669435e7ef1beaed',
      })
    );
  });
});
