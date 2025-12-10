/**
 * Unit tests for SubscriptionManager
 *
 * Tests the centralized address subscription management with:
 * - Registration and deduplication
 * - Source tracking
 * - Atomic replace operations
 * - Change notifications
 */

import { SubscriptionManager } from '../packages/ts-lib/src/subscription-manager';

describe('SubscriptionManager', () => {
  let manager: SubscriptionManager;

  beforeEach(() => {
    manager = new SubscriptionManager();
  });

  describe('register', () => {
    it('should register addresses from a source', async () => {
      const added = await manager.register('legacy', [
        '0x1234567890123456789012345678901234567890',
        '0xabcdefabcdefabcdefabcdefabcdefabcdefabcd',
      ]);

      expect(added).toHaveLength(2);
      expect(manager.getAllAddresses()).toHaveLength(2);
    });

    it('should normalize addresses to lowercase', async () => {
      await manager.register('legacy', [
        '0xABCDEFabcdefABCDEFabcdefABCDEFabcdefABCD',
      ]);

      const addresses = manager.getAllAddresses();
      expect(addresses[0]).toBe('0xabcdefabcdefabcdefabcdefabcdefabcdefabcd');
    });

    it('should deduplicate addresses from the same source', async () => {
      const addr = '0x1234567890123456789012345678901234567890';
      await manager.register('legacy', [addr, addr, addr]);

      expect(manager.getAllAddresses()).toHaveLength(1);
    });

    it('should deduplicate addresses across different sources', async () => {
      const addr = '0x1234567890123456789012345678901234567890';
      await manager.register('legacy', [addr]);
      await manager.register('alpha-pool', [addr]);

      // Same address, should only appear once in getAllAddresses
      expect(manager.getAllAddresses()).toHaveLength(1);

      // But should be tracked by both sources
      const sources = manager.getSourcesForAddress(addr);
      expect(sources).toContain('legacy');
      expect(sources).toContain('alpha-pool');
    });

    it('should return only newly added addresses', async () => {
      const addr1 = '0x1234567890123456789012345678901234567890';
      const addr2 = '0xabcdefabcdefabcdefabcdefabcdefabcdefabcd';

      const added1 = await manager.register('legacy', [addr1]);
      expect(added1).toHaveLength(1);

      // Registering same address from different source should not add new
      const added2 = await manager.register('alpha-pool', [addr1, addr2]);
      expect(added2).toHaveLength(1); // Only addr2 is new
      expect(added2[0]).toBe(addr2.toLowerCase());
    });

    it('should call onChanged callback when new addresses are added', async () => {
      const onChanged = jest.fn().mockResolvedValue(undefined);
      manager = new SubscriptionManager({ onChanged });

      await manager.register('legacy', [
        '0x1234567890123456789012345678901234567890',
      ]);

      expect(onChanged).toHaveBeenCalledTimes(1);
    });

    it('should not call onChanged if no new addresses added', async () => {
      const onChanged = jest.fn().mockResolvedValue(undefined);
      manager = new SubscriptionManager({ onChanged });

      const addr = '0x1234567890123456789012345678901234567890';
      await manager.register('legacy', [addr]);
      onChanged.mockClear();

      // Re-register same address from same source
      await manager.register('legacy', [addr]);
      expect(onChanged).not.toHaveBeenCalled();

      // Register same address from different source (no new unique addresses)
      await manager.register('alpha-pool', [addr]);
      expect(onChanged).not.toHaveBeenCalled();
    });
  });

  describe('unregister', () => {
    it('should remove addresses when last source unregisters', async () => {
      const addr = '0x1234567890123456789012345678901234567890';
      await manager.register('legacy', [addr]);
      expect(manager.getAllAddresses()).toHaveLength(1);

      const removed = await manager.unregister('legacy', [addr]);
      expect(removed).toHaveLength(1);
      expect(manager.getAllAddresses()).toHaveLength(0);
    });

    it('should keep address if other sources still registered', async () => {
      const addr = '0x1234567890123456789012345678901234567890';
      await manager.register('legacy', [addr]);
      await manager.register('alpha-pool', [addr]);

      const removed = await manager.unregister('legacy', [addr]);
      expect(removed).toHaveLength(0); // Not removed, still has alpha-pool

      expect(manager.getAllAddresses()).toHaveLength(1);
      expect(manager.getSourcesForAddress(addr)).toEqual(['alpha-pool']);
    });

    it('should call onChanged when addresses are removed', async () => {
      const onChanged = jest.fn().mockResolvedValue(undefined);
      manager = new SubscriptionManager({ onChanged });

      const addr = '0x1234567890123456789012345678901234567890';
      await manager.register('legacy', [addr]);
      onChanged.mockClear();

      await manager.unregister('legacy', [addr]);
      expect(onChanged).toHaveBeenCalledTimes(1);
    });
  });

  describe('replaceForSource', () => {
    it('should replace all addresses for a source atomically', async () => {
      const addr1 = '0x1111111111111111111111111111111111111111';
      const addr2 = '0x2222222222222222222222222222222222222222';
      const addr3 = '0x3333333333333333333333333333333333333333';

      await manager.register('legacy', [addr1, addr2]);
      expect(manager.getAddressesForSource('legacy')).toHaveLength(2);

      // Replace with new set
      await manager.replaceForSource('legacy', [addr2, addr3]);

      const legacyAddrs = manager.getAddressesForSource('legacy');
      expect(legacyAddrs).toHaveLength(2);
      expect(legacyAddrs).toContain(addr2.toLowerCase());
      expect(legacyAddrs).toContain(addr3.toLowerCase());
      expect(legacyAddrs).not.toContain(addr1.toLowerCase());
    });

    it('should not affect addresses from other sources', async () => {
      const addr1 = '0x1111111111111111111111111111111111111111';
      const addr2 = '0x2222222222222222222222222222222222222222';

      await manager.register('legacy', [addr1]);
      await manager.register('alpha-pool', [addr2]);

      // Replace legacy with empty set
      await manager.replaceForSource('legacy', []);

      // addr1 should be gone (only had legacy)
      expect(manager.getAllAddresses()).toHaveLength(1);
      expect(manager.getAllAddresses()[0]).toBe(addr2.toLowerCase());
    });

    it('should handle shared addresses correctly', async () => {
      const shared = '0x1111111111111111111111111111111111111111';
      const legacy = '0x2222222222222222222222222222222222222222';
      const alpha = '0x3333333333333333333333333333333333333333';

      await manager.register('legacy', [shared, legacy]);
      await manager.register('alpha-pool', [shared, alpha]);

      // Replace legacy, removing legacy-only address but keeping shared
      await manager.replaceForSource('legacy', []);

      expect(manager.getAllAddresses()).toHaveLength(2);
      expect(manager.getAllAddresses()).toContain(shared.toLowerCase());
      expect(manager.getAllAddresses()).toContain(alpha.toLowerCase());
      expect(manager.getAllAddresses()).not.toContain(legacy.toLowerCase());
    });
  });

  describe('getStatus', () => {
    it('should return correct counts', async () => {
      const addr1 = '0x1111111111111111111111111111111111111111';
      const addr2 = '0x2222222222222222222222222222222222222222';

      await manager.register('legacy', [addr1, addr2]);
      await manager.register('alpha-pool', [addr1]); // Duplicate

      const status = manager.getStatus();
      expect(status.totalAddresses).toBe(2);
      expect(status.addressesBySource).toEqual({
        legacy: 2,
        'alpha-pool': 1,
      });
      expect(status.addressesByMethod).toEqual({
        websocket: 2, // Both addresses fit in default 10 slots
        polling: 0,
        none: 0,
      });
      expect(status.duplicateCount).toBe(1); // addr1 is registered by both
      expect(status.sources).toEqual(['alpha-pool', 'legacy']); // sorted
      expect(status.maxWebSocketSlots).toBe(10); // default
    });

    it('should return empty status for fresh manager', () => {
      const status = manager.getStatus();
      expect(status.totalAddresses).toBe(0);
      expect(status.addressesBySource).toEqual({});
      expect(status.addressesByMethod).toEqual({
        websocket: 0,
        polling: 0,
        none: 0,
      });
      expect(status.duplicateCount).toBe(0);
      expect(status.sources).toEqual([]);
      expect(status.maxWebSocketSlots).toBe(10); // default
    });
  });

  describe('getAddressesForSource', () => {
    it('should return only addresses for the specified source', async () => {
      const addr1 = '0x1111111111111111111111111111111111111111';
      const addr2 = '0x2222222222222222222222222222222222222222';
      const addr3 = '0x3333333333333333333333333333333333333333';

      await manager.register('legacy', [addr1, addr2]);
      await manager.register('alpha-pool', [addr2, addr3]);

      const legacy = manager.getAddressesForSource('legacy');
      expect(legacy).toHaveLength(2);
      expect(legacy).toContain(addr1.toLowerCase());
      expect(legacy).toContain(addr2.toLowerCase());

      const alpha = manager.getAddressesForSource('alpha-pool');
      expect(alpha).toHaveLength(2);
      expect(alpha).toContain(addr2.toLowerCase());
      expect(alpha).toContain(addr3.toLowerCase());
    });

    it('should return empty array for unknown source', () => {
      expect(manager.getAddressesForSource('unknown')).toEqual([]);
    });
  });

  describe('getSourcesForAddress', () => {
    it('should return all sources for an address', async () => {
      const addr = '0x1234567890123456789012345678901234567890';
      await manager.register('legacy', [addr]);
      await manager.register('alpha-pool', [addr]);
      await manager.register('custom', [addr]);

      const sources = manager.getSourcesForAddress(addr);
      expect(sources).toHaveLength(3);
      expect(sources).toContain('legacy');
      expect(sources).toContain('alpha-pool');
      expect(sources).toContain('custom');
    });

    it('should return empty array for unknown address', () => {
      expect(manager.getSourcesForAddress('0x0000000000000000000000000000000000000000')).toEqual([]);
    });

    it('should handle case-insensitive lookup', async () => {
      const addr = '0xABCDEFabcdefABCDEFabcdefABCDEFabcdefABCD';
      await manager.register('legacy', [addr]);

      // Query with different case
      const sources = manager.getSourcesForAddress(addr.toLowerCase());
      expect(sources).toContain('legacy');
    });
  });

  describe('getAddressInfo', () => {
    it('should return detailed info about an address', async () => {
      const addr = '0x1234567890123456789012345678901234567890';
      await manager.register('legacy', [addr]);

      const info = manager.getAddressInfo(addr);
      expect(info).not.toBeNull();
      expect(info!.address).toBe(addr.toLowerCase());
      expect(info!.sources.has('legacy')).toBe(true);
    });

    it('should return null for unknown address', () => {
      expect(manager.getAddressInfo('0x0000000000000000000000000000000000000000')).toBeNull();
    });
  });

  describe('markSubscribed/markUnsubscribed', () => {
    it('should track subscription state', async () => {
      const addr = '0x1234567890123456789012345678901234567890';
      await manager.register('legacy', [addr]);

      let info = manager.getAddressInfo(addr);
      expect(info!.subscribedAt).toBeNull();

      manager.markSubscribed(addr);
      info = manager.getAddressInfo(addr);
      expect(info!.subscribedAt).toBeInstanceOf(Date);

      manager.markUnsubscribed(addr);
      info = manager.getAddressInfo(addr);
      expect(info!.subscribedAt).toBeNull();
    });
  });

  describe('clear', () => {
    it('should remove all registrations', async () => {
      await manager.register('legacy', [
        '0x1111111111111111111111111111111111111111',
        '0x2222222222222222222222222222222222222222',
      ]);
      await manager.register('alpha-pool', [
        '0x3333333333333333333333333333333333333333',
      ]);

      expect(manager.getAllAddresses()).toHaveLength(3);

      manager.clear();

      expect(manager.getAllAddresses()).toHaveLength(0);
      expect(manager.getStatus().totalAddresses).toBe(0);
    });
  });

  describe('WebSocket slot allocation', () => {
    it('should assign websocket method to addresses within slot limit', async () => {
      // Default is 10 slots
      await manager.register('legacy', [
        '0x1111111111111111111111111111111111111111',
        '0x2222222222222222222222222222222222222222',
      ]);

      expect(manager.getMethod('0x1111111111111111111111111111111111111111')).toBe('websocket');
      expect(manager.getMethod('0x2222222222222222222222222222222222222222')).toBe('websocket');
    });

    it('should assign polling method to addresses beyond slot limit', async () => {
      // Create manager with only 2 WebSocket slots
      const limitedManager = new SubscriptionManager({ maxWebSocketSlots: 2 });

      await limitedManager.register('legacy', [
        '0x1111111111111111111111111111111111111111',
        '0x2222222222222222222222222222222222222222',
        '0x3333333333333333333333333333333333333333',
      ]);

      const status = limitedManager.getStatus();
      expect(status.addressesByMethod.websocket).toBe(2);
      expect(status.addressesByMethod.polling).toBe(1);
    });

    it('should prioritize legacy addresses over alpha-pool for websocket slots', async () => {
      const limitedManager = new SubscriptionManager({ maxWebSocketSlots: 2 });

      // Register alpha-pool first
      await limitedManager.register('alpha-pool', [
        '0xaaaa111111111111111111111111111111111111',
        '0xaaaa222222222222222222222222222222222222',
      ]);

      // Then register legacy
      await limitedManager.register('legacy', [
        '0xbbbb111111111111111111111111111111111111',
      ]);

      // Legacy should get websocket priority
      expect(limitedManager.getMethod('0xbbbb111111111111111111111111111111111111')).toBe('websocket');

      // One of the alpha-pool addresses should be demoted to polling
      const alphaMethod1 = limitedManager.getMethod('0xaaaa111111111111111111111111111111111111');
      const alphaMethod2 = limitedManager.getMethod('0xaaaa222222222222222222222222222222222222');
      const alphaWebsockets = [alphaMethod1, alphaMethod2].filter(m => m === 'websocket').length;
      expect(alphaWebsockets).toBe(1); // Only 1 slot left for alpha-pool
    });

    it('should give shared addresses priority from highest-priority source', async () => {
      const limitedManager = new SubscriptionManager({ maxWebSocketSlots: 1 });

      // Shared address registered by both sources
      const shared = '0xaaaa111111111111111111111111111111111111';
      const alphaOnly = '0xaaaa222222222222222222222222222222222222';

      await limitedManager.register('alpha-pool', [shared, alphaOnly]);
      await limitedManager.register('legacy', [shared]); // Shared gets legacy priority

      // Shared address should have websocket (legacy priority)
      expect(limitedManager.getMethod(shared)).toBe('websocket');
      // Alpha-only should be polling
      expect(limitedManager.getMethod(alphaOnly)).toBe('polling');
    });

    it('should allow updating max websocket slots', async () => {
      await manager.register('legacy', [
        '0x1111111111111111111111111111111111111111',
        '0x2222222222222222222222222222222222222222',
        '0x3333333333333333333333333333333333333333',
      ]);

      // All should have websocket initially (default 10 slots)
      expect(manager.getStatus().addressesByMethod.websocket).toBe(3);

      // Reduce slots to 1
      manager.setMaxWebSocketSlots(1);

      expect(manager.getStatus().addressesByMethod.websocket).toBe(1);
      expect(manager.getStatus().addressesByMethod.polling).toBe(2);
    });

    it('should reassign methods when addresses are removed', async () => {
      const limitedManager = new SubscriptionManager({ maxWebSocketSlots: 1 });

      const addr1 = '0x1111111111111111111111111111111111111111';
      const addr2 = '0x2222222222222222222222222222222222222222';

      await limitedManager.register('legacy', [addr1, addr2]);

      // addr1 should have websocket, addr2 polling
      expect(limitedManager.getMethod(addr1)).toBe('websocket');
      expect(limitedManager.getMethod(addr2)).toBe('polling');

      // Remove addr1
      await limitedManager.unregister('legacy', [addr1]);

      // addr2 should now have websocket
      expect(limitedManager.getMethod(addr2)).toBe('websocket');
    });
  });

  describe('getAddressesByMethod', () => {
    it('should return addresses filtered by method', async () => {
      const limitedManager = new SubscriptionManager({ maxWebSocketSlots: 1 });

      await limitedManager.register('legacy', [
        '0x1111111111111111111111111111111111111111',
        '0x2222222222222222222222222222222222222222',
      ]);

      const wsAddresses = limitedManager.getAddressesByMethod('websocket');
      const pollingAddresses = limitedManager.getAddressesByMethod('polling');

      expect(wsAddresses).toHaveLength(1);
      expect(pollingAddresses).toHaveLength(1);
    });
  });

  describe('getAddressesByPriority', () => {
    it('should return addresses sorted by source priority', async () => {
      // Register alpha-pool first
      await manager.register('alpha-pool', ['0xaaaa111111111111111111111111111111111111']);
      // Then legacy
      await manager.register('legacy', ['0xbbbb111111111111111111111111111111111111']);

      const sorted = manager.getAddressesByPriority();

      // Legacy should come first (priority 1)
      expect(sorted[0]).toBe('0xbbbb111111111111111111111111111111111111');
      // Alpha-pool second (priority 2)
      expect(sorted[1]).toBe('0xaaaa111111111111111111111111111111111111');
    });

    it('should give pinned accounts highest priority', async () => {
      const limitedManager = new SubscriptionManager({ maxWebSocketSlots: 1 });

      // Register in reverse priority order
      await limitedManager.register('alpha-pool', ['0xaaaa111111111111111111111111111111111111']);
      await limitedManager.register('legacy', ['0xbbbb111111111111111111111111111111111111']);
      await limitedManager.register('pinned', ['0xcccc111111111111111111111111111111111111']);

      const sorted = limitedManager.getAddressesByPriority();

      // Pinned should come first (priority 0)
      expect(sorted[0]).toBe('0xcccc111111111111111111111111111111111111');
      // Legacy second (priority 1)
      expect(sorted[1]).toBe('0xbbbb111111111111111111111111111111111111');
      // Alpha-pool last (priority 2)
      expect(sorted[2]).toBe('0xaaaa111111111111111111111111111111111111');

      // Only pinned should get websocket (1 slot)
      expect(limitedManager.getMethod('0xcccc111111111111111111111111111111111111')).toBe('websocket');
      expect(limitedManager.getMethod('0xbbbb111111111111111111111111111111111111')).toBe('polling');
      expect(limitedManager.getMethod('0xaaaa111111111111111111111111111111111111')).toBe('polling');
    });
  });

  describe('Manual demote/promote', () => {
    it('should allow demoting unpinned WebSocket address to polling and reserve the slot', async () => {
      const limitedManager = new SubscriptionManager({ maxWebSocketSlots: 2 });

      const addr1 = '0x1111111111111111111111111111111111111111';
      const addr2 = '0x2222222222222222222222222222222222222222';
      const addr3 = '0x3333333333333333333333333333333333333333';

      await limitedManager.register('legacy', [addr1, addr2, addr3]);

      // addr1 and addr2 should have websocket (2 slots)
      expect(limitedManager.getMethod(addr1)).toBe('websocket');
      expect(limitedManager.getMethod(addr2)).toBe('websocket');
      expect(limitedManager.getMethod(addr3)).toBe('polling');

      // Demote addr1 to polling
      const success = limitedManager.demoteToPolling(addr1);
      expect(success).toBe(true);

      // addr1 should now be polling, slot is reserved (not auto-filled)
      // addr3 should NOT be auto-promoted - user must manually promote
      expect(limitedManager.getMethod(addr1)).toBe('polling');
      expect(limitedManager.getMethod(addr2)).toBe('websocket');
      expect(limitedManager.getMethod(addr3)).toBe('polling'); // NOT auto-promoted

      // Verify only 1 websocket slot is used (1 reserved for manual demotion)
      expect(limitedManager.getStatus().addressesByMethod.websocket).toBe(1);
    });

    it('should not allow demoting pinned address', async () => {
      await manager.register('pinned', ['0x1111111111111111111111111111111111111111']);

      const success = manager.demoteToPolling('0x1111111111111111111111111111111111111111');
      expect(success).toBe(false);
      expect(manager.getMethod('0x1111111111111111111111111111111111111111')).toBe('websocket');
    });

    it('should allow promoting polling address to WebSocket when slots available', async () => {
      const limitedManager = new SubscriptionManager({ maxWebSocketSlots: 2 });

      const addr1 = '0x1111111111111111111111111111111111111111';
      const addr2 = '0x2222222222222222222222222222222222222222';
      const addr3 = '0x3333333333333333333333333333333333333333';

      await limitedManager.register('legacy', [addr1, addr2, addr3]);

      // Demote addr1 to free a slot (slot is reserved, not auto-filled)
      limitedManager.demoteToPolling(addr1);
      expect(limitedManager.getMethod(addr3)).toBe('polling'); // NOT auto-promoted
      expect(limitedManager.getStatus().addressesByMethod.websocket).toBe(1); // addr2 only

      // Manually promote addr3 to use the freed slot
      const success = limitedManager.promoteToWebsocket(addr3);
      expect(success).toBe(true);
      expect(limitedManager.getMethod(addr3)).toBe('websocket');
      expect(limitedManager.getStatus().addressesByMethod.websocket).toBe(2); // addr2 + addr3
    });

    it('should not allow promoting when no slots available', async () => {
      const limitedManager = new SubscriptionManager({ maxWebSocketSlots: 1 });

      const addr1 = '0x1111111111111111111111111111111111111111';
      const addr2 = '0x2222222222222222222222222222222222222222';

      await limitedManager.register('legacy', [addr1, addr2]);

      // addr1 has websocket, addr2 has polling
      expect(limitedManager.getMethod(addr1)).toBe('websocket');
      expect(limitedManager.getMethod(addr2)).toBe('polling');

      // Try to promote addr2 - should fail (no slots)
      const success = limitedManager.promoteToWebsocket(addr2);
      expect(success).toBe(false);
      expect(limitedManager.getMethod(addr2)).toBe('polling');
    });

    it('should track manual override status', async () => {
      const limitedManager = new SubscriptionManager({ maxWebSocketSlots: 2 });

      const addr1 = '0x1111111111111111111111111111111111111111';
      await limitedManager.register('legacy', [addr1]);

      // Initially no override
      expect(limitedManager.getManualOverride(addr1)).toEqual({
        manualPolling: false,
        manualWebsocket: false
      });

      // Demote
      limitedManager.demoteToPolling(addr1);
      expect(limitedManager.getManualOverride(addr1)).toEqual({
        manualPolling: true,
        manualWebsocket: false
      });

      // Promote
      limitedManager.promoteToWebsocket(addr1);
      expect(limitedManager.getManualOverride(addr1)).toEqual({
        manualPolling: false,
        manualWebsocket: true
      });

      // Clear override
      limitedManager.clearManualOverride(addr1);
      expect(limitedManager.getManualOverride(addr1)).toEqual({
        manualPolling: false,
        manualWebsocket: false
      });
    });

    it('should respect manual demotion even after re-registration', async () => {
      const limitedManager = new SubscriptionManager({ maxWebSocketSlots: 10 });

      const addr1 = '0x1111111111111111111111111111111111111111';
      await limitedManager.register('legacy', [addr1]);

      // Demote
      limitedManager.demoteToPolling(addr1);
      expect(limitedManager.getMethod(addr1)).toBe('polling');

      // Re-register from another source - should still be polling due to manual flag
      await limitedManager.register('alpha-pool', [addr1]);
      expect(limitedManager.getMethod(addr1)).toBe('polling');
    });

    it('should correctly handle demote 2, promote 1 scenario (reserved slots)', async () => {
      const limitedManager = new SubscriptionManager({ maxWebSocketSlots: 10 });

      // Register 12 addresses (10 get websocket, 2 get polling)
      const addresses = Array.from({ length: 12 }, (_, i) =>
        `0x${(i + 1).toString().padStart(40, '0')}`
      );
      await limitedManager.register('legacy', addresses);

      // Initially 10 websocket, 2 polling
      expect(limitedManager.getStatus().addressesByMethod.websocket).toBe(10);
      expect(limitedManager.getStatus().addressesByMethod.polling).toBe(2);

      // Demote 2 websocket addresses
      limitedManager.demoteToPolling(addresses[0]);
      limitedManager.demoteToPolling(addresses[1]);

      // Should now be 8 websocket (10 - 2 reserved)
      expect(limitedManager.getStatus().addressesByMethod.websocket).toBe(8);
      expect(limitedManager.getStatus().addressesByMethod.polling).toBe(4);

      // Promote 1 polling address
      const success = limitedManager.promoteToWebsocket(addresses[10]); // Was originally polling
      expect(success).toBe(true);

      // Should be 9 websocket (8 auto + 1 manual, 1 reserved slot remaining)
      expect(limitedManager.getStatus().addressesByMethod.websocket).toBe(9);
      expect(limitedManager.getStatus().addressesByMethod.polling).toBe(3);
    });

    it('should allow promoting up to the number of demoted slots', async () => {
      const limitedManager = new SubscriptionManager({ maxWebSocketSlots: 3 });

      const addr1 = '0x1111111111111111111111111111111111111111';
      const addr2 = '0x2222222222222222222222222222222222222222';
      const addr3 = '0x3333333333333333333333333333333333333333';
      const addr4 = '0x4444444444444444444444444444444444444444';
      const addr5 = '0x5555555555555555555555555555555555555555';

      await limitedManager.register('legacy', [addr1, addr2, addr3, addr4, addr5]);

      // Initially: 3 websocket, 2 polling
      expect(limitedManager.getStatus().addressesByMethod.websocket).toBe(3);

      // Demote all 3 websocket addresses
      limitedManager.demoteToPolling(addr1);
      limitedManager.demoteToPolling(addr2);
      limitedManager.demoteToPolling(addr3);

      // All should be polling now (0 websocket, 3 reserved slots)
      expect(limitedManager.getStatus().addressesByMethod.websocket).toBe(0);
      expect(limitedManager.getStatus().addressesByMethod.polling).toBe(5);

      // Promote addr4 (1/3 reserved slots used)
      expect(limitedManager.promoteToWebsocket(addr4)).toBe(true);
      expect(limitedManager.getStatus().addressesByMethod.websocket).toBe(1);

      // Promote addr5 (2/3 reserved slots used)
      expect(limitedManager.promoteToWebsocket(addr5)).toBe(true);
      expect(limitedManager.getStatus().addressesByMethod.websocket).toBe(2);

      // Promote addr1 back (3/3 reserved slots used)
      expect(limitedManager.promoteToWebsocket(addr1)).toBe(true);
      expect(limitedManager.getStatus().addressesByMethod.websocket).toBe(3);

      // No more slots available
      expect(limitedManager.promoteToWebsocket(addr2)).toBe(false);
    });

    it('should handle pinned addresses correctly with demote/promote', async () => {
      const limitedManager = new SubscriptionManager({ maxWebSocketSlots: 3 });

      const pinned1 = '0x1111111111111111111111111111111111111111';
      const legacy1 = '0x2222222222222222222222222222222222222222';
      const legacy2 = '0x3333333333333333333333333333333333333333';
      const legacy3 = '0x4444444444444444444444444444444444444444';

      await limitedManager.register('pinned', [pinned1]);
      await limitedManager.register('legacy', [legacy1, legacy2, legacy3]);

      // Pinned gets 1 slot, legacy gets 2 slots
      expect(limitedManager.getMethod(pinned1)).toBe('websocket');
      expect(limitedManager.getStatus().addressesByMethod.websocket).toBe(3);

      // Cannot demote pinned
      expect(limitedManager.demoteToPolling(pinned1)).toBe(false);

      // Demote a legacy address
      limitedManager.demoteToPolling(legacy1);
      expect(limitedManager.getStatus().addressesByMethod.websocket).toBe(2);

      // Promote legacy3 (was polling)
      expect(limitedManager.promoteToWebsocket(legacy3)).toBe(true);
      expect(limitedManager.getStatus().addressesByMethod.websocket).toBe(3);
    });

    it('should not auto-fill reserved slots when new addresses are registered', async () => {
      const limitedManager = new SubscriptionManager({ maxWebSocketSlots: 2 });

      const addr1 = '0x1111111111111111111111111111111111111111';
      const addr2 = '0x2222222222222222222222222222222222222222';

      await limitedManager.register('legacy', [addr1, addr2]);

      // Both have websocket
      expect(limitedManager.getStatus().addressesByMethod.websocket).toBe(2);

      // Demote addr1
      limitedManager.demoteToPolling(addr1);
      expect(limitedManager.getStatus().addressesByMethod.websocket).toBe(1);

      // Register new address - should NOT auto-fill the reserved slot
      const addr3 = '0x3333333333333333333333333333333333333333';
      await limitedManager.register('legacy', [addr3]);

      // Still only 1 websocket (addr2), new address goes to polling
      expect(limitedManager.getStatus().addressesByMethod.websocket).toBe(1);
      expect(limitedManager.getMethod(addr3)).toBe('polling');
    });

    it('should clear reserved slot when demoted address is unregistered', async () => {
      const limitedManager = new SubscriptionManager({ maxWebSocketSlots: 2 });

      const addr1 = '0x1111111111111111111111111111111111111111';
      const addr2 = '0x2222222222222222222222222222222222222222';
      const addr3 = '0x3333333333333333333333333333333333333333';

      await limitedManager.register('legacy', [addr1, addr2, addr3]);

      // Demote addr1
      limitedManager.demoteToPolling(addr1);
      expect(limitedManager.getStatus().addressesByMethod.websocket).toBe(1);

      // Unregister addr1 entirely - should free the reserved slot
      await limitedManager.unregister('legacy', [addr1]);

      // Now addr3 should auto-get websocket since reservation is gone
      expect(limitedManager.getStatus().addressesByMethod.websocket).toBe(2);
      expect(limitedManager.getMethod(addr3)).toBe('websocket');
    });

    it('should return false when promoting non-existent address', () => {
      const success = manager.promoteToWebsocket('0x0000000000000000000000000000000000000000');
      expect(success).toBe(false);
    });

    it('should return false when demoting non-existent address', () => {
      const success = manager.demoteToPolling('0x0000000000000000000000000000000000000000');
      expect(success).toBe(false);
    });

    it('should handle clearManualOverride for non-existent address gracefully', () => {
      // Should not throw
      expect(() => manager.clearManualOverride('0x0000000000000000000000000000000000000000')).not.toThrow();
    });

    it('should return correct override status for non-existent address', () => {
      const override = manager.getManualOverride('0x0000000000000000000000000000000000000000');
      expect(override).toEqual({
        manualPolling: false,
        manualWebsocket: false
      });
    });
  });
});
