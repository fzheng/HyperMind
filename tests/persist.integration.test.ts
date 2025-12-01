/**
 * Integration tests for persist.ts database operations
 *
 * These tests use a mock pg Pool to test SQL query construction
 * and response handling without requiring a real database.
 */

// Mock the postgres module before any imports
const mockQuery = jest.fn();
const mockClientQuery = jest.fn();
const mockClient = {
  query: mockClientQuery,
  release: jest.fn(),
};
const mockPool = {
  query: mockQuery,
  connect: jest.fn(() => Promise.resolve(mockClient)),
};

jest.mock('../packages/ts-lib/src/postgres', () => ({
  getPool: jest.fn(() => Promise.resolve(mockPool)),
  closePool: jest.fn(() => Promise.resolve()),
}));

import {
  insertEvent,
  insertTradeIfNew,
  latestTrades,
  pageTrades,
  pageTradesByTime,
  deleteAllTrades,
  deleteTradesForAddress,
  countValidTradesForAddress,
  upsertCurrentPosition,
  clearPositionsForAddress,
  getAddressPerformance,
  listRecentFills,
  listLiveFills,
  fetchLatestFillForAddress,
  listCustomAccounts,
  listPinnedAccounts,
  addCustomPinnedAccount,
  pinLeaderboardAccount,
  unpinAccount,
  getPinnedAccountCount,
  isPinnedAccount,
  getLastRefreshTime,
  getBackfillFills,
  getOldestFillTime,
  listRecentDecisions,
  validatePositionChain,
  clearTradesForAddress,
  type InsertableEvent,
} from '../packages/ts-lib/src/persist';

// Mock console.error to prevent noise in test output from error handling paths
const originalConsoleError = console.error;
beforeAll(() => {
  console.error = jest.fn();
});
afterAll(() => {
  console.error = originalConsoleError;
});

describe('persist.ts database integration', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Reset the client mock for transactional tests
    mockClientQuery.mockReset();
  });

  describe('insertEvent', () => {
    it('should insert event and return id', async () => {
      mockQuery.mockResolvedValueOnce({ rows: [{ id: 123 }] });

      const event: InsertableEvent = {
        type: 'trade',
        at: '2025-01-01T00:00:00Z',
        address: '0x1234567890abcdef1234567890abcdef12345678',
        symbol: 'BTC',
        payload: { price: 95000 },
      };

      const result = await insertEvent(event);

      expect(result).toBe(123);
      expect(mockQuery).toHaveBeenCalledWith(
        'insert into hl_events (at, address, type, symbol, payload) values ($1,$2,$3,$4,$5) returning id',
        [event.at, event.address, event.type, event.symbol, event.payload]
      );
    });

    it('should return null on error', async () => {
      mockQuery.mockRejectedValueOnce(new Error('DB error'));

      const event: InsertableEvent = {
        type: 'position',
        at: '2025-01-01T00:00:00Z',
        address: '0x1234',
        symbol: 'ETH',
        payload: {},
      };

      const result = await insertEvent(event);
      expect(result).toBeNull();
    });

    it('should return null when rows is empty', async () => {
      mockQuery.mockResolvedValueOnce({ rows: [] });

      const event: InsertableEvent = {
        type: 'trade',
        at: '2025-01-01T00:00:00Z',
        address: '0x1234',
        symbol: 'BTC',
        payload: {},
      };

      const result = await insertEvent(event);
      expect(result).toBeNull();
    });
  });

  describe('insertTradeIfNew', () => {
    it('should insert new trade when hash does not exist', async () => {
      // First query: check for existing hash - returns empty
      mockQuery.mockResolvedValueOnce({ rows: [] });
      // Second query: insert new trade
      mockQuery.mockResolvedValueOnce({ rows: [{ id: 456 }] });

      const payload = {
        hash: '0xnewhash',
        at: '2025-01-01T00:00:00Z',
        symbol: 'BTC',
        price: 95000,
      };

      const result = await insertTradeIfNew('0xABCD', payload);

      expect(result.inserted).toBe(true);
      expect(result.id).toBe(456);
      expect(mockQuery).toHaveBeenCalledTimes(2);
    });

    it('should update existing trade when hash exists', async () => {
      // First query: check for existing hash - returns existing
      mockQuery.mockResolvedValueOnce({ rows: [{ id: 789 }] });
      // Second query: update existing trade
      mockQuery.mockResolvedValueOnce({});

      const payload = {
        hash: '0xexistinghash',
        at: '2025-01-01T00:00:00Z',
        symbol: 'ETH',
      };

      const result = await insertTradeIfNew('0xABCD', payload);

      expect(result.inserted).toBe(false);
      expect(result.id).toBe(789);
    });

    it('should normalize address to lowercase', async () => {
      mockQuery.mockResolvedValueOnce({ rows: [] });
      mockQuery.mockResolvedValueOnce({ rows: [{ id: 1 }] });

      await insertTradeIfNew('0xABCDEF', { hash: '0x123' });

      expect(mockQuery).toHaveBeenCalledWith(
        expect.stringContaining("payload->>'hash'"),
        ['0xabcdef', '0x123']
      );
    });

    it('should handle missing hash in payload', async () => {
      mockQuery.mockResolvedValueOnce({ rows: [{ id: 100 }] });

      const payload = { price: 95000, symbol: 'BTC' };
      const result = await insertTradeIfNew('0x1234', payload);

      expect(result.id).toBe(100);
      expect(result.inserted).toBe(true);
    });
  });

  describe('latestTrades', () => {
    it('should return latest trades with default limit', async () => {
      const mockTrades = [
        { payload: { action: 'Open Long', price: 95000 } },
        { payload: { action: 'Close Long', price: 96000 } },
      ];
      mockQuery.mockResolvedValueOnce({ rows: mockTrades });

      const result = await latestTrades();

      expect(result).toHaveLength(2);
      expect(mockQuery).toHaveBeenCalledWith(
        'select payload from hl_events where type = $1 order by id desc limit $2',
        ['trade', 50]
      );
    });

    it('should clamp limit between 1 and 200', async () => {
      mockQuery.mockResolvedValueOnce({ rows: [] });
      await latestTrades(500);
      expect(mockQuery).toHaveBeenCalledWith(
        expect.any(String),
        ['trade', 200]
      );

      mockQuery.mockResolvedValueOnce({ rows: [] });
      await latestTrades(0);
      expect(mockQuery).toHaveBeenCalledWith(
        expect.any(String),
        ['trade', 1]
      );
    });

    it('should return empty array on error', async () => {
      mockQuery.mockRejectedValueOnce(new Error('DB error'));
      const result = await latestTrades();
      expect(result).toEqual([]);
    });
  });

  describe('pageTrades', () => {
    it('should paginate trades by id', async () => {
      mockQuery.mockResolvedValueOnce({
        rows: [
          { id: 100, payload: { action: 'Open Long' } },
          { id: 99, payload: { action: 'Close Long' } },
        ],
      });

      const result = await pageTrades({ limit: 50, beforeId: 101 });

      expect(result).toHaveLength(2);
      expect(mockQuery).toHaveBeenCalledWith(
        expect.stringContaining('id < $'),
        [101, 50]
      );
    });

    it('should filter by address', async () => {
      mockQuery.mockResolvedValueOnce({ rows: [] });

      await pageTrades({ address: '0xABCD', limit: 10 });

      expect(mockQuery).toHaveBeenCalledWith(
        expect.stringContaining('address = $'),
        ['0xabcd', 10]
      );
    });

    it('should clamp limit between 1 and 200', async () => {
      mockQuery.mockResolvedValueOnce({ rows: [] });
      await pageTrades({ limit: 500 });
      expect(mockQuery).toHaveBeenCalledWith(
        expect.any(String),
        [200]
      );
    });
  });

  describe('pageTradesByTime', () => {
    it('should paginate by time cursor', async () => {
      mockQuery.mockResolvedValueOnce({
        rows: [
          { id: 1, address: '0x1', at: '2025-01-01T00:00:00Z', payload: {} },
        ],
      });

      await pageTradesByTime({ beforeAt: '2025-01-02T00:00:00Z', limit: 25 });

      expect(mockQuery).toHaveBeenCalledWith(
        expect.stringContaining('at < $'),
        ['2025-01-02T00:00:00Z', 25]
      );
    });

    it('should handle composite cursor (time + id)', async () => {
      mockQuery.mockResolvedValueOnce({ rows: [] });

      await pageTradesByTime({
        beforeAt: '2025-01-02T00:00:00Z',
        beforeId: 100,
        limit: 10,
      });

      expect(mockQuery).toHaveBeenCalledWith(
        expect.stringContaining('(at < $'),
        ['2025-01-02T00:00:00Z', 100, 10]
      );
    });

    it('should filter by address with time cursor', async () => {
      mockQuery.mockResolvedValueOnce({ rows: [] });

      await pageTradesByTime({
        address: '0xABCD',
        beforeAt: '2025-01-01T00:00:00Z',
      });

      expect(mockQuery).toHaveBeenCalledWith(
        expect.stringContaining('address = $'),
        expect.arrayContaining(['0xabcd'])
      );
    });
  });

  describe('deleteAllTrades', () => {
    it('should delete all trades and return count', async () => {
      mockQuery.mockResolvedValueOnce({ rowCount: 150 });

      const result = await deleteAllTrades();

      expect(result).toBe(150);
      expect(mockQuery).toHaveBeenCalledWith(
        "delete from hl_events where type = 'trade'"
      );
    });

    it('should return 0 on error', async () => {
      mockQuery.mockRejectedValueOnce(new Error('DB error'));
      const result = await deleteAllTrades();
      expect(result).toBe(0);
    });
  });

  describe('deleteTradesForAddress', () => {
    it('should delete trades for specific address', async () => {
      mockQuery.mockResolvedValueOnce({ rowCount: 25 });

      const result = await deleteTradesForAddress('0xABCD');

      expect(result).toBe(25);
      expect(mockQuery).toHaveBeenCalledWith(
        "delete from hl_events where type = 'trade' and address = $1",
        ['0xabcd']
      );
    });
  });

  describe('countValidTradesForAddress', () => {
    it('should count trades with startPosition', async () => {
      mockQuery.mockResolvedValueOnce({ rows: [{ c: 42 }] });

      const result = await countValidTradesForAddress('0x1234');

      expect(result).toBe(42);
      expect(mockQuery).toHaveBeenCalledWith(
        expect.stringContaining("payload ? 'startPosition'"),
        ['0x1234']
      );
    });

    it('should return 0 on error', async () => {
      mockQuery.mockRejectedValueOnce(new Error('DB error'));
      const result = await countValidTradesForAddress('0x1234');
      expect(result).toBe(0);
    });
  });

  describe('upsertCurrentPosition', () => {
    it('should upsert position when size > 0', async () => {
      mockQuery.mockResolvedValueOnce({});

      await upsertCurrentPosition({
        address: '0x1234',
        symbol: 'BTC',
        size: 0.5,
        entryPriceUsd: 95000,
        liquidationPriceUsd: 85000,
        leverage: 10,
        pnlUsd: 500,
      });

      expect(mockQuery).toHaveBeenCalledWith(
        expect.stringContaining('insert into hl_current_positions'),
        expect.arrayContaining(['0x1234', 'BTC', 0.5, 95000, 85000, 10, 500])
      );
    });

    it('should delete position when size = 0', async () => {
      mockQuery.mockResolvedValueOnce({});

      await upsertCurrentPosition({
        address: '0x1234',
        symbol: 'ETH',
        size: 0,
        entryPriceUsd: null,
        liquidationPriceUsd: null,
        leverage: null,
        pnlUsd: null,
      });

      expect(mockQuery).toHaveBeenCalledWith(
        'DELETE FROM hl_current_positions WHERE address = $1 AND symbol = $2',
        ['0x1234', 'ETH']
      );
    });
  });

  describe('clearPositionsForAddress', () => {
    it('should delete all positions for address', async () => {
      mockQuery.mockResolvedValueOnce({});

      await clearPositionsForAddress('0x1234');

      expect(mockQuery).toHaveBeenCalledWith(
        'DELETE FROM hl_current_positions WHERE address = $1',
        ['0x1234']
      );
    });
  });

  describe('getAddressPerformance', () => {
    it('should aggregate performance metrics', async () => {
      mockQuery.mockResolvedValueOnce({
        rows: [
          {
            address: '0x1234',
            trades: '50',
            wins: '35',
            pnl_total: '5000',
            pnl_7d: '1500',
            avg_size: '0.25',
          },
        ],
      });

      const result = await getAddressPerformance(7);

      expect(result).toHaveLength(1);
      expect(result[0]).toMatchObject({
        address: '0x1234',
        trades: 50,
        wins: 35,
        winRate: 0.7,
        pnl7d: 1500,
        avgSize: 0.25,
        efficiency: 100, // 5000 / 50
      });
    });

    it('should handle empty results', async () => {
      mockQuery.mockResolvedValueOnce({ rows: [] });
      const result = await getAddressPerformance();
      expect(result).toEqual([]);
    });

    it('should handle zero trades', async () => {
      mockQuery.mockResolvedValueOnce({
        rows: [
          {
            address: '0x1234',
            trades: 0,
            wins: 0,
            pnl_total: 0,
            pnl_7d: 0,
            avg_size: 0,
          },
        ],
      });

      const result = await getAddressPerformance();
      expect(result[0].winRate).toBe(0);
      expect(result[0].efficiency).toBe(0);
    });
  });

  describe('listRecentFills', () => {
    it('should return recent fills with proper transformation', async () => {
      mockQuery.mockResolvedValueOnce({
        rows: [
          {
            id: 1,
            address: '0x1234',
            at: '2025-01-01T00:00:00Z',
            payload: {
              side: 'sell',
              size: 0.5,
              priceUsd: 95000,
              realizedPnlUsd: 500,
              action: 'Close Long',
            },
          },
        ],
      });

      const result = await listRecentFills(10);

      expect(result).toHaveLength(1);
      expect(result[0]).toMatchObject({
        id: 1,
        address: '0x1234',
        side: 'sell',
        size: 0.5,
        priceUsd: 95000,
        realizedPnlUsd: 500,
        action: 'Close Long',
      });
    });

    it('should default side to buy', async () => {
      mockQuery.mockResolvedValueOnce({
        rows: [
          { id: 1, address: '0x1', at: '2025-01-01', payload: { side: 'buy' } },
        ],
      });

      const result = await listRecentFills();
      expect(result[0].side).toBe('buy');
    });
  });

  describe('listLiveFills', () => {
    it('should return live fills with signed size', async () => {
      mockQuery.mockResolvedValueOnce({
        rows: [
          {
            time_utc: '2025-01-01T00:00:00Z',
            address: '0x1234',
            action: 'Open Long',
            size_signed: 0.5,
            previous_position: 0,
            price_usd: 95000,
            closed_pnl_usd: null,
            tx_hash: '0xhash',
            symbol: 'BTC',
            fee: 5.25,
            fee_token: 'USDC',
          },
        ],
      });

      const result = await listLiveFills(25);

      expect(result).toHaveLength(1);
      expect(result[0]).toMatchObject({
        action: 'Open Long',
        size_signed: 0.5,
        previous_position: 0,
        symbol: 'BTC',
      });
    });

    it('should clamp limit between 1 and 200', async () => {
      mockQuery.mockResolvedValueOnce({ rows: [] });
      await listLiveFills(500);
      expect(mockQuery).toHaveBeenCalledWith(expect.any(String), [200]);
    });
  });

  describe('fetchLatestFillForAddress', () => {
    it('should return latest fill for address', async () => {
      mockQuery.mockResolvedValueOnce({
        rows: [
          {
            id: 100,
            address: '0x1234',
            at: '2025-01-01T00:00:00Z',
            payload: { side: 'buy', size: 0.1, priceUsd: 95000 },
          },
        ],
      });

      const result = await fetchLatestFillForAddress('0x1234');

      expect(result).not.toBeNull();
      expect(result!.address).toBe('0x1234');
      expect(mockQuery).toHaveBeenCalledWith(
        expect.stringContaining('address = $1'),
        ['0x1234']
      );
    });

    it('should return null when no fills exist', async () => {
      mockQuery.mockResolvedValueOnce({ rows: [] });
      const result = await fetchLatestFillForAddress('0x1234');
      expect(result).toBeNull();
    });
  });

  describe('Pinned Accounts', () => {
    describe('listCustomAccounts', () => {
      it('should return list of custom accounts', async () => {
        mockQuery.mockResolvedValueOnce({
          rows: [
            { id: 1, address: '0x1234', nickname: 'Whale', added_at: '2025-01-01' },
            { id: 2, address: '0x5678', nickname: null, added_at: '2025-01-02' },
          ],
        });

        const result = await listCustomAccounts();

        expect(result).toHaveLength(2);
        expect(result[0].nickname).toBe('Whale');
        expect(result[1].nickname).toBeNull();
      });
    });

    describe('addCustomPinnedAccount', () => {
      it('should add new custom pinned account', async () => {
        // Transaction flow: BEGIN, LOCK TABLE, COUNT, CHECK EXISTING, INSERT, COMMIT
        mockClientQuery
          .mockResolvedValueOnce({}) // BEGIN
          .mockResolvedValueOnce({}) // LOCK TABLE
          .mockResolvedValueOnce({ rows: [{ cnt: 1 }] }) // COUNT (1 existing custom)
          .mockResolvedValueOnce({ rows: [] }) // CHECK EXISTING (not already pinned)
          .mockResolvedValueOnce({
            rows: [{ id: 3, address: '0xnew', is_custom: true, pinned_at: '2025-01-01' }],
          }) // INSERT
          .mockResolvedValueOnce({}); // COMMIT

        const result = await addCustomPinnedAccount('0xNEW');

        expect(result.success).toBe(true);
        expect(result.account!.isCustom).toBe(true);
      });

      it('should reject when custom limit reached', async () => {
        // Transaction flow: BEGIN, LOCK TABLE, COUNT (3 = limit), ROLLBACK
        mockClientQuery
          .mockResolvedValueOnce({}) // BEGIN
          .mockResolvedValueOnce({}) // LOCK TABLE
          .mockResolvedValueOnce({ rows: [{ cnt: 3 }] }) // COUNT = 3 (limit reached)
          .mockResolvedValueOnce({}); // ROLLBACK

        const result = await addCustomPinnedAccount('0xnew');

        expect(result.success).toBe(false);
        expect(result.error).toContain('Maximum');
      });

      it('should handle duplicate address', async () => {
        // Transaction flow: BEGIN, LOCK TABLE, COUNT, CHECK EXISTING (found), ROLLBACK
        mockClientQuery
          .mockResolvedValueOnce({}) // BEGIN
          .mockResolvedValueOnce({}) // LOCK TABLE
          .mockResolvedValueOnce({ rows: [{ cnt: 0 }] }) // COUNT = 0 existing custom accounts
          .mockResolvedValueOnce({ rows: [{ is_custom: false }] }) // CHECK EXISTING (already pinned)
          .mockResolvedValueOnce({}); // ROLLBACK

        const result = await addCustomPinnedAccount('0xexisting');

        expect(result.success).toBe(false);
        expect(result.error).toContain('already pinned');
      });
    });

    describe('pinLeaderboardAccount', () => {
      it('should pin account from leaderboard', async () => {
        mockQuery.mockResolvedValueOnce({
          rows: [{ id: 1, address: '0x1234', is_custom: false, pinned_at: '2025-01-01' }],
        });

        const result = await pinLeaderboardAccount('0x1234');

        expect(result.success).toBe(true);
        expect(result.account!.isCustom).toBe(false);
      });

      it('should return error for duplicate', async () => {
        mockQuery.mockResolvedValueOnce({ rows: [] }); // ON CONFLICT DO NOTHING returns empty

        const result = await pinLeaderboardAccount('0xexisting');

        expect(result.success).toBe(false);
        expect(result.error).toContain('already pinned');
      });
    });

    describe('unpinAccount', () => {
      it('should unpin account and return true', async () => {
        mockQuery.mockResolvedValueOnce({ rowCount: 1 });
        const result = await unpinAccount('0x1234');
        expect(result).toBe(true);
      });

      it('should return false when account not found', async () => {
        mockQuery.mockResolvedValueOnce({ rowCount: 0 });
        const result = await unpinAccount('0xnotfound');
        expect(result).toBe(false);
      });
    });

    describe('getPinnedAccountCount', () => {
      it('should return count of all pinned accounts', async () => {
        mockQuery.mockResolvedValueOnce({ rows: [{ cnt: 5 }] });
        const result = await getPinnedAccountCount();
        expect(result).toBe(5);
      });

      it('should return count of custom pinned accounts when filtered', async () => {
        mockQuery.mockResolvedValueOnce({ rows: [{ cnt: 2 }] });
        const result = await getPinnedAccountCount(true);
        expect(result).toBe(2);
      });
    });

    describe('isPinnedAccount', () => {
      it('should return pinned status for custom account', async () => {
        mockQuery.mockResolvedValueOnce({ rows: [{ is_custom: true }] });
        const result = await isPinnedAccount('0x1234');
        expect(result).toEqual({ isPinned: true, isCustom: true });
      });

      it('should return pinned status for leaderboard-pinned account', async () => {
        mockQuery.mockResolvedValueOnce({ rows: [{ is_custom: false }] });
        const result = await isPinnedAccount('0x1234');
        expect(result).toEqual({ isPinned: true, isCustom: false });
      });

      it('should return null for non-pinned account', async () => {
        mockQuery.mockResolvedValueOnce({ rows: [] });
        const result = await isPinnedAccount('0xnotpinned');
        expect(result).toBeNull();
      });
    });

    describe('listPinnedAccounts', () => {
      it('should list all pinned accounts', async () => {
        mockQuery.mockResolvedValueOnce({
          rows: [
            { id: 1, address: '0x1111', is_custom: false, pinned_at: '2025-01-01' },
            { id: 2, address: '0x2222', is_custom: true, pinned_at: '2025-01-02' },
          ],
        });

        const result = await listPinnedAccounts();

        expect(result).toHaveLength(2);
        expect(result[0].isCustom).toBe(false);
        expect(result[1].isCustom).toBe(true);
      });

      it('should filter by isCustom when specified', async () => {
        mockQuery.mockResolvedValueOnce({
          rows: [{ id: 2, address: '0x2222', is_custom: true, pinned_at: '2025-01-02' }],
        });

        const result = await listPinnedAccounts(true);

        expect(result).toHaveLength(1);
        expect(result[0].isCustom).toBe(true);
      });

      it('should filter by isCustom=false for leaderboard-pinned', async () => {
        mockQuery.mockResolvedValueOnce({
          rows: [{ id: 1, address: '0x1111', is_custom: false, pinned_at: '2025-01-01' }],
        });

        const result = await listPinnedAccounts(false);

        expect(result).toHaveLength(1);
        expect(result[0].isCustom).toBe(false);
      });

      it('should return empty array when no pinned accounts', async () => {
        mockQuery.mockResolvedValueOnce({ rows: [] });

        const result = await listPinnedAccounts();

        expect(result).toEqual([]);
      });

      it('should handle database error', async () => {
        mockQuery.mockRejectedValueOnce(new Error('DB error'));

        const result = await listPinnedAccounts();

        expect(result).toEqual([]);
      });
    });

    describe('pinLeaderboardAccount edge cases', () => {
      it('should normalize address to lowercase', async () => {
        mockQuery.mockResolvedValueOnce({
          rows: [{ id: 1, address: '0xabcd', is_custom: false, pinned_at: '2025-01-01' }],
        });

        await pinLeaderboardAccount('0xABCD');

        expect(mockQuery).toHaveBeenCalledWith(
          expect.stringContaining('INSERT INTO hl_pinned_accounts'),
          ['0xabcd']
        );
      });

      it('should handle database error gracefully', async () => {
        mockQuery.mockRejectedValueOnce(new Error('DB connection failed'));

        const result = await pinLeaderboardAccount('0x1234');

        expect(result.success).toBe(false);
        expect(result.error).toContain('Failed');
      });
    });

    describe('addCustomPinnedAccount edge cases', () => {
      it('should normalize address to lowercase', async () => {
        mockClientQuery
          .mockResolvedValueOnce({}) // BEGIN
          .mockResolvedValueOnce({}) // LOCK
          .mockResolvedValueOnce({ rows: [{ cnt: 0 }] }) // COUNT
          .mockResolvedValueOnce({ rows: [] }) // CHECK EXISTING
          .mockResolvedValueOnce({
            rows: [{ id: 1, address: '0xabcd', is_custom: true, pinned_at: '2025-01-01' }],
          }) // INSERT
          .mockResolvedValueOnce({}); // COMMIT

        const result = await addCustomPinnedAccount('0xABCD');

        expect(result.success).toBe(true);
        // Verify the address was lowercased in the INSERT
        expect(mockClientQuery).toHaveBeenCalledWith(
          expect.stringContaining('INSERT INTO hl_pinned_accounts'),
          ['0xabcd']
        );
      });

      it('should handle transaction rollback on error', async () => {
        mockClientQuery
          .mockResolvedValueOnce({}) // BEGIN
          .mockResolvedValueOnce({}) // LOCK
          .mockRejectedValueOnce(new Error('DB error')) // COUNT fails
          .mockResolvedValueOnce({}); // ROLLBACK

        const result = await addCustomPinnedAccount('0x1234');

        expect(result.success).toBe(false);
        // ROLLBACK should have been called
        expect(mockClientQuery).toHaveBeenCalledWith('ROLLBACK');
      });
    });

    describe('unpinAccount edge cases', () => {
      it('should normalize address to lowercase', async () => {
        mockQuery.mockResolvedValueOnce({ rowCount: 1 });

        await unpinAccount('0xABCD');

        expect(mockQuery).toHaveBeenCalledWith(
          expect.stringContaining('DELETE FROM hl_pinned_accounts'),
          ['0xabcd']
        );
      });

      it('should handle database error', async () => {
        mockQuery.mockRejectedValueOnce(new Error('DB error'));

        const result = await unpinAccount('0x1234');

        expect(result).toBe(false);
      });
    });

    describe('getPinnedAccountCount edge cases', () => {
      it('should return 0 on database error', async () => {
        mockQuery.mockRejectedValueOnce(new Error('DB error'));

        const result = await getPinnedAccountCount();

        expect(result).toBe(0);
      });

      it('should return 0 when result is null', async () => {
        mockQuery.mockResolvedValueOnce({ rows: [{ cnt: null }] });

        const result = await getPinnedAccountCount();

        expect(result).toBe(0);
      });
    });

    describe('isPinnedAccount edge cases', () => {
      it('should normalize address to lowercase', async () => {
        mockQuery.mockResolvedValueOnce({ rows: [{ is_custom: true }] });

        await isPinnedAccount('0xABCD');

        expect(mockQuery).toHaveBeenCalledWith(
          expect.stringContaining('lower(address) = $1'),
          ['0xabcd']
        );
      });

      it('should handle database error', async () => {
        mockQuery.mockRejectedValueOnce(new Error('DB error'));

        const result = await isPinnedAccount('0x1234');

        expect(result).toBeNull();
      });
    });
  });

  describe('getLastRefreshTime', () => {
    it('should return last refresh timestamp', async () => {
      mockQuery.mockResolvedValueOnce({
        rows: [{ last_refresh: '2025-01-01T12:00:00Z' }],
      });

      const result = await getLastRefreshTime(30);

      expect(result).toBe('2025-01-01T12:00:00Z');
      expect(mockQuery).toHaveBeenCalledWith(
        expect.stringContaining('period_days = $1'),
        [30]
      );
    });

    it('should return null when no entries exist', async () => {
      mockQuery.mockResolvedValueOnce({ rows: [{ last_refresh: null }] });
      const result = await getLastRefreshTime(30);
      expect(result).toBeNull();
    });
  });

  describe('getBackfillFills', () => {
    it('should return fills with hasMore flag', async () => {
      const mockRows = Array(51).fill(null).map((_, i) => ({
        id: i,
        time_utc: `2025-01-0${(i % 9) + 1}T00:00:00Z`,
        address: '0x1234',
        symbol: 'BTC',
        action: 'Open Long',
        size_signed: 0.1,
        previous_position: 0,
        price_usd: 95000,
        closed_pnl_usd: null,
        tx_hash: `0xhash${i}`,
      }));

      mockQuery.mockResolvedValueOnce({ rows: mockRows });

      const result = await getBackfillFills({ limit: 50 });

      expect(result.fills).toHaveLength(50);
      expect(result.hasMore).toBe(true);
    });

    it('should filter by addresses', async () => {
      mockQuery.mockResolvedValueOnce({ rows: [] });

      await getBackfillFills({
        addresses: ['0xAAAA', '0xBBBB'],
        limit: 25,
      });

      expect(mockQuery).toHaveBeenCalledWith(
        expect.stringContaining('address = ANY($'),
        expect.arrayContaining([['0xaaaa', '0xbbbb']])
      );
    });

    it('should filter by beforeTime', async () => {
      mockQuery.mockResolvedValueOnce({ rows: [] });

      await getBackfillFills({
        beforeTime: '2025-01-15T00:00:00Z',
      });

      expect(mockQuery).toHaveBeenCalledWith(
        expect.stringContaining('< $'),
        expect.arrayContaining(['2025-01-15T00:00:00Z'])
      );
    });
  });

  describe('getOldestFillTime', () => {
    it('should return oldest fill time', async () => {
      mockQuery.mockResolvedValueOnce({
        rows: [{ oldest: '2024-01-01T00:00:00Z' }],
      });

      const result = await getOldestFillTime();

      expect(result).toBe('2024-01-01T00:00:00Z');
    });

    it('should filter by addresses', async () => {
      mockQuery.mockResolvedValueOnce({ rows: [{ oldest: '2024-06-01T00:00:00Z' }] });

      await getOldestFillTime(['0x1234', '0x5678']);

      expect(mockQuery).toHaveBeenCalledWith(
        expect.stringContaining('address = ANY($1)'),
        [['0x1234', '0x5678']]
      );
    });

    it('should return null when no fills exist', async () => {
      mockQuery.mockResolvedValueOnce({ rows: [{ oldest: null }] });
      const result = await getOldestFillTime();
      expect(result).toBeNull();
    });
  });

  describe('listRecentDecisions', () => {
    it('should return decisions with outcomes', async () => {
      mockQuery.mockResolvedValueOnce({
        rows: [
          {
            id: 'uuid-1',
            address: '0x1234',
            asset: 'BTC',
            side: 'long',
            ts: '2025-01-01T00:00:00Z',
            closed_reason: 'take_profit',
            result_r: 2.5,
            closed_ts: '2025-01-02T00:00:00Z',
          },
        ],
      });

      const result = await listRecentDecisions(20);

      expect(result).toHaveLength(1);
      expect(result[0]).toMatchObject({
        status: 'closed',
        closedReason: 'take_profit',
        result: 2.5,
      });
    });

    it('should mark open tickets correctly', async () => {
      mockQuery.mockResolvedValueOnce({
        rows: [
          {
            id: 'uuid-2',
            address: '0x5678',
            asset: 'ETH',
            side: 'short',
            ts: '2025-01-01T00:00:00Z',
            closed_reason: null,
            result_r: null,
          },
        ],
      });

      const result = await listRecentDecisions();

      expect(result[0].status).toBe('open');
      expect(result[0].closedReason).toBeNull();
    });
  });
});

describe('Error handling and edge cases', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should handle null rowCount in delete operations', async () => {
    mockQuery.mockResolvedValueOnce({ rowCount: null });
    const result = await deleteAllTrades();
    expect(result).toBe(0);
  });

  it('should handle upsertCurrentPosition error', async () => {
    mockQuery.mockRejectedValueOnce(new Error('DB error'));
    // Should not throw, just log error
    await expect(
      upsertCurrentPosition({
        address: '0x1234',
        symbol: 'BTC',
        size: 1,
        entryPriceUsd: 100,
        liquidationPriceUsd: 50,
        leverage: 10,
        pnlUsd: 5,
      })
    ).resolves.toBeUndefined();
  });

  it('should handle clearPositionsForAddress error', async () => {
    mockQuery.mockRejectedValueOnce(new Error('DB error'));
    // Should not throw, just log error
    await expect(clearPositionsForAddress('0x1234')).resolves.toBeUndefined();
  });

  it('should handle pageTrades error', async () => {
    mockQuery.mockRejectedValueOnce(new Error('DB error'));
    const result = await pageTrades({ limit: 10 });
    expect(result).toEqual([]);
  });

  it('should handle pageTradesByTime with beforeId only (no beforeAt)', async () => {
    mockQuery.mockResolvedValueOnce({
      rows: [{ id: 1, address: '0x1', at: '2025-01-01', payload: {} }],
    });

    const result = await pageTradesByTime({ beforeId: 100 });

    expect(result.length).toBe(1);
    expect(mockQuery).toHaveBeenCalledWith(
      expect.stringContaining('id < $'),
      expect.arrayContaining([100])
    );
  });

  it('should handle listLiveFills error', async () => {
    mockQuery.mockRejectedValueOnce(new Error('DB error'));
    const result = await listLiveFills();
    expect(result).toEqual([]);
  });

  it('should handle listRecentFills error', async () => {
    mockQuery.mockRejectedValueOnce(new Error('DB error'));
    const result = await listRecentFills();
    expect(result).toEqual([]);
  });

  it('should handle listRecentDecisions error', async () => {
    mockQuery.mockRejectedValueOnce(new Error('DB error'));
    const result = await listRecentDecisions();
    expect(result).toEqual([]);
  });

  it('should handle getAddressPerformance error', async () => {
    mockQuery.mockRejectedValueOnce(new Error('DB error'));
    const result = await getAddressPerformance();
    expect(result).toEqual([]);
  });

  it('should handle deleteTradesForAddress error', async () => {
    mockQuery.mockRejectedValueOnce(new Error('DB error'));
    const result = await deleteTradesForAddress('0x1234');
    expect(result).toBe(0);
  });

  it('should handle countValidTradesForAddress error', async () => {
    mockQuery.mockRejectedValueOnce(new Error('DB error'));
    const result = await countValidTradesForAddress('0x1234');
    expect(result).toBe(0);
  });

  it('should handle insertTradeIfNew error', async () => {
    mockQuery.mockRejectedValueOnce(new Error('DB error'));
    const result = await insertTradeIfNew('0x1234', { hash: 'abc' });
    expect(result).toEqual({ id: null, inserted: false });
  });

  it('should handle undefined in performance rows', async () => {
    mockQuery.mockResolvedValueOnce({
      rows: [
        {
          address: '0x1',
          trades: undefined,
          wins: undefined,
          pnl_total: undefined,
          pnl_7d: undefined,
          avg_size: undefined,
        },
      ],
    });

    const result = await getAddressPerformance();
    expect(result[0].trades).toBe(0);
    expect(result[0].wins).toBe(0);
  });

  it('should handle missing payload fields in listRecentFills', async () => {
    mockQuery.mockResolvedValueOnce({
      rows: [
        { id: 1, address: '0x1', at: '2025-01-01', payload: {} },
      ],
    });

    const result = await listRecentFills();
    expect(result[0].size).toBe(0);
    expect(result[0].priceUsd).toBe(0);
    expect(result[0].realizedPnlUsd).toBeNull();
  });

  it('should handle nested payload in listRecentFills', async () => {
    mockQuery.mockResolvedValueOnce({
      rows: [
        {
          id: 1,
          address: '0x1',
          at: '2025-01-01',
          payload: { payload: { size: 0.5 } },
        },
      ],
    });

    const result = await listRecentFills();
    expect(result[0].size).toBe(0.5);
  });
});

describe('Position chain validation and repair', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('clearTradesForAddress', () => {
    it('should delete trades for a specific address', async () => {
      mockQuery.mockResolvedValueOnce({ rowCount: 10 });

      const result = await clearTradesForAddress('0x1234567890abcdef1234567890abcdef12345678');

      expect(result).toBe(10);
      expect(mockQuery).toHaveBeenCalledWith(
        expect.stringContaining("DELETE FROM hl_events WHERE type = 'trade' AND LOWER(address) = LOWER($1)"),
        ['0x1234567890abcdef1234567890abcdef12345678']
      );
    });

    it('should filter by symbol when provided', async () => {
      mockQuery.mockResolvedValueOnce({ rowCount: 5 });

      const result = await clearTradesForAddress('0x1234', 'ETH');

      expect(result).toBe(5);
      expect(mockQuery).toHaveBeenCalledWith(
        expect.stringContaining("payload->>'symbol', 'BTC') = $2"),
        ['0x1234', 'ETH']
      );
    });

    it('should return 0 on error', async () => {
      mockQuery.mockRejectedValueOnce(new Error('DB error'));

      const result = await clearTradesForAddress('0x1234');

      expect(result).toBe(0);
    });

    it('should handle null rowCount', async () => {
      mockQuery.mockResolvedValueOnce({ rowCount: null });

      const result = await clearTradesForAddress('0x1234');

      expect(result).toBe(0);
    });
  });

  describe('validatePositionChain', () => {
    it('should return valid for continuous chain', async () => {
      // Fills sorted by time DESC (newest first)
      mockQuery.mockResolvedValueOnce({
        rows: [
          { time_utc: '2025-01-03T00:00:00Z', previous_position: 5, resulting_position: 0 },
          { time_utc: '2025-01-02T00:00:00Z', previous_position: 0, resulting_position: 5 },
        ],
      });

      const result = await validatePositionChain('0x1234', 'ETH');

      expect(result.valid).toBe(true);
      expect(result.gaps.length).toBe(0);
    });

    it('should detect gaps in position chain', async () => {
      // Gap: first fill starts at 5, but second fill results in 10 (missing fills)
      mockQuery.mockResolvedValueOnce({
        rows: [
          { time_utc: '2025-01-03T00:00:00Z', previous_position: 5, resulting_position: 0 },
          { time_utc: '2025-01-02T00:00:00Z', previous_position: 0, resulting_position: 10 },
        ],
      });

      const result = await validatePositionChain('0x1234', 'ETH');

      expect(result.valid).toBe(false);
      expect(result.gaps.length).toBe(1);
      expect(result.gaps[0].expected).toBe(10);
      expect(result.gaps[0].actual).toBe(5);
    });

    it('should allow small floating point differences', async () => {
      mockQuery.mockResolvedValueOnce({
        rows: [
          { time_utc: '2025-01-03T00:00:00Z', previous_position: 5.00001, resulting_position: 0 },
          { time_utc: '2025-01-02T00:00:00Z', previous_position: 0, resulting_position: 5 },
        ],
      });

      const result = await validatePositionChain('0x1234', 'ETH');

      expect(result.valid).toBe(true);
    });

    it('should return valid for empty result set', async () => {
      mockQuery.mockResolvedValueOnce({ rows: [] });

      const result = await validatePositionChain('0x1234', 'ETH');

      expect(result.valid).toBe(true);
      expect(result.gaps.length).toBe(0);
    });

    it('should return valid for single fill', async () => {
      mockQuery.mockResolvedValueOnce({
        rows: [
          { time_utc: '2025-01-01T00:00:00Z', previous_position: 0, resulting_position: 5 },
        ],
      });

      const result = await validatePositionChain('0x1234', 'ETH');

      expect(result.valid).toBe(true);
    });

    it('should handle database errors gracefully', async () => {
      mockQuery.mockRejectedValueOnce(new Error('DB error'));

      const result = await validatePositionChain('0x1234', 'ETH');

      expect(result.valid).toBe(false);
      expect(result.gaps.length).toBe(0);
    });

    it('should query with correct symbol filter', async () => {
      mockQuery.mockResolvedValueOnce({ rows: [] });

      await validatePositionChain('0x1234', 'BTC');

      expect(mockQuery).toHaveBeenCalledWith(
        expect.stringContaining("COALESCE(payload->>'symbol', 'BTC') = $2"),
        ['0x1234', 'BTC']
      );
    });

    it('should detect multiple gaps', async () => {
      mockQuery.mockResolvedValueOnce({
        rows: [
          { time_utc: '2025-01-04T00:00:00Z', previous_position: 0, resulting_position: 0 },
          { time_utc: '2025-01-03T00:00:00Z', previous_position: 5, resulting_position: 10 }, // gap: 10 != 0
          { time_utc: '2025-01-02T00:00:00Z', previous_position: 0, resulting_position: 2 },  // gap: 2 != 5
          { time_utc: '2025-01-01T00:00:00Z', previous_position: 0, resulting_position: 0 },
        ],
      });

      const result = await validatePositionChain('0x1234', 'ETH');

      expect(result.valid).toBe(false);
      expect(result.gaps.length).toBe(2);
    });
  });
});

describe('addCustomPinnedAccount transaction lock path', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockClientQuery.mockReset();
  });

  it('should execute SQL sequence: BEGIN → LOCK → COUNT → INSERT → COMMIT on success', async () => {
    // Setup successful transaction flow
    mockClientQuery
      .mockResolvedValueOnce({}) // BEGIN
      .mockResolvedValueOnce({}) // LOCK TABLE
      .mockResolvedValueOnce({ rows: [{ cnt: 0 }] }) // COUNT custom accounts
      .mockResolvedValueOnce({ rows: [] }) // CHECK EXISTING
      .mockResolvedValueOnce({
        rows: [{ id: 1, address: '0xnewaddr', is_custom: true, pinned_at: '2025-01-01' }],
      }) // INSERT
      .mockResolvedValueOnce({}); // COMMIT

    const result = await addCustomPinnedAccount('0xNEWADDR');

    expect(result.success).toBe(true);

    // Verify SQL sequence
    expect(mockClientQuery).toHaveBeenNthCalledWith(1, 'BEGIN');
    expect(mockClientQuery).toHaveBeenNthCalledWith(
      2,
      expect.stringContaining('LOCK TABLE hl_pinned_accounts')
    );
    expect(mockClientQuery).toHaveBeenNthCalledWith(
      3,
      expect.stringContaining('SELECT COUNT')
    );
    expect(mockClientQuery).toHaveBeenNthCalledWith(
      4,
      expect.stringContaining('SELECT 1 FROM hl_pinned_accounts'),
      ['0xnewaddr']
    );
    expect(mockClientQuery).toHaveBeenNthCalledWith(
      5,
      expect.stringContaining('INSERT INTO hl_pinned_accounts'),
      ['0xnewaddr']
    );
    expect(mockClientQuery).toHaveBeenNthCalledWith(6, 'COMMIT');
  });

  it('should execute SQL sequence: BEGIN → LOCK → COUNT → ROLLBACK on limit exceeded', async () => {
    mockClientQuery
      .mockResolvedValueOnce({}) // BEGIN
      .mockResolvedValueOnce({}) // LOCK TABLE
      .mockResolvedValueOnce({ rows: [{ cnt: 3 }] }) // COUNT = 3 (limit reached)
      .mockResolvedValueOnce({}); // ROLLBACK

    const result = await addCustomPinnedAccount('0xnewaddr');

    expect(result.success).toBe(false);
    expect(result.error).toContain('Maximum');

    // Verify SQL sequence - should rollback without insert
    expect(mockClientQuery).toHaveBeenNthCalledWith(1, 'BEGIN');
    expect(mockClientQuery).toHaveBeenNthCalledWith(
      2,
      expect.stringContaining('LOCK TABLE hl_pinned_accounts')
    );
    expect(mockClientQuery).toHaveBeenNthCalledWith(
      3,
      expect.stringContaining('SELECT COUNT')
    );
    expect(mockClientQuery).toHaveBeenNthCalledWith(4, 'ROLLBACK');
  });

  it('should execute SQL sequence: BEGIN → LOCK → COUNT → CHECK → ROLLBACK on duplicate', async () => {
    mockClientQuery
      .mockResolvedValueOnce({}) // BEGIN
      .mockResolvedValueOnce({}) // LOCK TABLE
      .mockResolvedValueOnce({ rows: [{ cnt: 1 }] }) // COUNT = 1 (under limit)
      .mockResolvedValueOnce({ rows: [{ is_custom: true }] }) // EXISTING found
      .mockResolvedValueOnce({}); // ROLLBACK

    const result = await addCustomPinnedAccount('0xexistingaddr');

    expect(result.success).toBe(false);
    expect(result.error).toContain('already pinned');

    // Verify rollback after finding existing
    expect(mockClientQuery).toHaveBeenCalledWith('ROLLBACK');
  });

  it('should rollback on COUNT query failure', async () => {
    mockClientQuery
      .mockResolvedValueOnce({}) // BEGIN
      .mockResolvedValueOnce({}) // LOCK TABLE
      .mockRejectedValueOnce(new Error('DB connection lost')) // COUNT fails
      .mockResolvedValueOnce({}); // ROLLBACK

    const result = await addCustomPinnedAccount('0xnewaddr');

    expect(result.success).toBe(false);
    expect(mockClientQuery).toHaveBeenCalledWith('ROLLBACK');
  });

  it('should rollback on INSERT failure', async () => {
    mockClientQuery
      .mockResolvedValueOnce({}) // BEGIN
      .mockResolvedValueOnce({}) // LOCK TABLE
      .mockResolvedValueOnce({ rows: [{ cnt: 0 }] }) // COUNT
      .mockResolvedValueOnce({ rows: [] }) // CHECK EXISTING
      .mockRejectedValueOnce(new Error('Constraint violation')) // INSERT fails
      .mockResolvedValueOnce({}); // ROLLBACK

    const result = await addCustomPinnedAccount('0xnewaddr');

    expect(result.success).toBe(false);
    expect(mockClientQuery).toHaveBeenCalledWith('ROLLBACK');
  });

  it('should always release client after transaction', async () => {
    // Even on failure, client should be released
    // Mock the ROLLBACK to also return a promise for .catch() chain
    mockClientQuery
      .mockResolvedValueOnce({}) // BEGIN
      .mockRejectedValueOnce(new Error('Lock timeout')) // LOCK fails
      .mockResolvedValueOnce({}); // ROLLBACK

    await addCustomPinnedAccount('0xnewaddr');

    expect(mockClient.release).toHaveBeenCalled();
  });
});
