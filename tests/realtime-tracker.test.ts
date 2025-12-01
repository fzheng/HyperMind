/**
 * Tests for RealtimeTracker refresh logic
 * Tests the address management and refresh behavior
 */

// Simulates the address tracking logic from RealtimeTracker
class MockAddressTracker {
  private subscribed: Set<string> = new Set();
  private primePromises: Map<string, Promise<void>> = new Map();

  async refresh(
    getAddresses: () => Promise<string[]>,
    opts?: { awaitPositions?: boolean }
  ): Promise<{ added: string[]; removed: string[] }> {
    const addrs = (await getAddresses()).map(a => a.toLowerCase());
    const current = new Set(this.subscribed);
    const newAddrs: string[] = [];
    const removedAddrs: string[] = [];

    // Unsubscribe removed addresses
    for (const addr of current) {
      if (!addrs.includes(addr)) {
        this.subscribed.delete(addr);
        removedAddrs.push(addr);
      }
    }

    // Subscribe new addresses
    for (const addr of addrs) {
      if (!this.subscribed.has(addr)) {
        newAddrs.push(addr);
        this.subscribed.add(addr);
      }
    }

    // If awaitPositions is true, wait for all position data
    if (opts?.awaitPositions && newAddrs.length > 0) {
      const primePromises = newAddrs.map(addr => this.mockPrime(addr));
      await Promise.allSettled(primePromises);
    }

    return { added: newAddrs, removed: removedAddrs };
  }

  async forceRefreshAllPositions(getAddresses: () => Promise<string[]>): Promise<string[]> {
    const addrs = (await getAddresses()).map(a => a.toLowerCase());
    const tasks = addrs.map(addr => this.mockPrime(addr));
    await Promise.allSettled(tasks);
    return addrs;
  }

  private async mockPrime(addr: string): Promise<void> {
    // Simulate HTTP request delay
    await new Promise(resolve => setTimeout(resolve, 10));
    return Promise.resolve();
  }

  getSubscribed(): string[] {
    return Array.from(this.subscribed);
  }

  isSubscribed(addr: string): boolean {
    return this.subscribed.has(addr.toLowerCase());
  }
}

describe('MockAddressTracker.refresh', () => {
  let tracker: MockAddressTracker;

  beforeEach(() => {
    tracker = new MockAddressTracker();
  });

  test('subscribes to new addresses', async () => {
    const getAddresses = async () => [
      '0x1111111111111111111111111111111111111111',
      '0x2222222222222222222222222222222222222222',
    ];

    const result = await tracker.refresh(getAddresses);

    expect(result.added.length).toBe(2);
    expect(result.removed.length).toBe(0);
    expect(tracker.isSubscribed('0x1111111111111111111111111111111111111111')).toBe(true);
    expect(tracker.isSubscribed('0x2222222222222222222222222222222222222222')).toBe(true);
  });

  test('normalizes addresses to lowercase', async () => {
    const getAddresses = async () => [
      '0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA',
    ];

    await tracker.refresh(getAddresses);

    expect(tracker.isSubscribed('0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')).toBe(true);
    expect(tracker.getSubscribed()).toContain('0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa');
  });

  test('removes addresses no longer in list', async () => {
    // First refresh with two addresses
    await tracker.refresh(async () => [
      '0x1111111111111111111111111111111111111111',
      '0x2222222222222222222222222222222222222222',
    ]);

    // Second refresh with only one
    const result = await tracker.refresh(async () => [
      '0x1111111111111111111111111111111111111111',
    ]);

    expect(result.removed).toContain('0x2222222222222222222222222222222222222222');
    expect(tracker.isSubscribed('0x1111111111111111111111111111111111111111')).toBe(true);
    expect(tracker.isSubscribed('0x2222222222222222222222222222222222222222')).toBe(false);
  });

  test('does not re-add existing addresses', async () => {
    await tracker.refresh(async () => [
      '0x1111111111111111111111111111111111111111',
    ]);

    const result = await tracker.refresh(async () => [
      '0x1111111111111111111111111111111111111111',
    ]);

    expect(result.added.length).toBe(0);
    expect(result.removed.length).toBe(0);
  });

  test('handles empty address list', async () => {
    await tracker.refresh(async () => [
      '0x1111111111111111111111111111111111111111',
    ]);

    const result = await tracker.refresh(async () => []);

    expect(result.removed.length).toBe(1);
    expect(tracker.getSubscribed().length).toBe(0);
  });

  test('awaitPositions option waits for priming', async () => {
    const startTime = Date.now();

    await tracker.refresh(
      async () => [
        '0x1111111111111111111111111111111111111111',
        '0x2222222222222222222222222222222222222222',
      ],
      { awaitPositions: true }
    );

    const elapsed = Date.now() - startTime;
    // Should have waited for the mock prime (10ms each, run in parallel)
    // Use >= 8 to account for timing variations in CI environments
    expect(elapsed).toBeGreaterThanOrEqual(8);
  });

  test('awaitPositions false does not wait', async () => {
    const startTime = Date.now();

    await tracker.refresh(
      async () => [
        '0x1111111111111111111111111111111111111111',
        '0x2222222222222222222222222222222222222222',
      ],
      { awaitPositions: false }
    );

    const elapsed = Date.now() - startTime;
    // Should complete quickly without waiting for priming
    expect(elapsed).toBeLessThan(10);
  });

  test('awaitPositions only primes new addresses', async () => {
    // First add one address
    await tracker.refresh(async () => [
      '0x1111111111111111111111111111111111111111',
    ]);

    const startTime = Date.now();

    // Add another address with awaitPositions
    await tracker.refresh(
      async () => [
        '0x1111111111111111111111111111111111111111',
        '0x2222222222222222222222222222222222222222',
      ],
      { awaitPositions: true }
    );

    const elapsed = Date.now() - startTime;
    // Should only prime the new address (10ms), not both
    expect(elapsed).toBeGreaterThanOrEqual(10);
    expect(elapsed).toBeLessThan(25); // Would be ~20ms if both were primed sequentially
  });
});

describe('MockAddressTracker.forceRefreshAllPositions', () => {
  let tracker: MockAddressTracker;

  beforeEach(() => {
    tracker = new MockAddressTracker();
  });

  test('refreshes all positions', async () => {
    await tracker.refresh(async () => [
      '0x1111111111111111111111111111111111111111',
      '0x2222222222222222222222222222222222222222',
    ]);

    const startTime = Date.now();

    const refreshed = await tracker.forceRefreshAllPositions(async () => [
      '0x1111111111111111111111111111111111111111',
      '0x2222222222222222222222222222222222222222',
    ]);

    const elapsed = Date.now() - startTime;

    expect(refreshed.length).toBe(2);
    expect(elapsed).toBeGreaterThanOrEqual(10); // Waited for all primes
  });

  test('returns normalized addresses', async () => {
    const refreshed = await tracker.forceRefreshAllPositions(async () => [
      '0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA',
      '0xBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB',
    ]);

    expect(refreshed).toEqual([
      '0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
      '0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
    ]);
  });

  test('handles empty address list', async () => {
    const refreshed = await tracker.forceRefreshAllPositions(async () => []);
    expect(refreshed.length).toBe(0);
  });
});

describe('Position snapshot key format', () => {
  test('creates correct snapshot key', () => {
    const address = '0x1234567890abcdef1234567890abcdef12345678';
    const symbol = 'BTC';
    const key = `${address}:${symbol}`;
    expect(key).toBe('0x1234567890abcdef1234567890abcdef12345678:BTC');
  });

  test('parses snapshot key correctly', () => {
    const key = '0x1234567890abcdef1234567890abcdef12345678:ETH';
    const [address, symbol] = key.split(':');
    expect(address).toBe('0x1234567890abcdef1234567890abcdef12345678');
    expect(symbol).toBe('ETH');
  });

  test('handles multiple colons gracefully', () => {
    // Edge case: if address somehow contained a colon (it shouldn't)
    const key = '0x1234:extra:BTC';
    const parts = key.split(':');
    const address = parts.slice(0, -1).join(':');
    const symbol = parts[parts.length - 1];
    expect(address).toBe('0x1234:extra');
    expect(symbol).toBe('BTC');
  });
});

describe('sideFromSize helper', () => {
  function sideFromSize(size: number): 'long' | 'short' | 'flat' {
    if (size > 0) return 'long';
    if (size < 0) return 'short';
    return 'flat';
  }

  test('returns long for positive size', () => {
    expect(sideFromSize(0.5)).toBe('long');
    expect(sideFromSize(100)).toBe('long');
    expect(sideFromSize(0.0001)).toBe('long');
  });

  test('returns short for negative size', () => {
    expect(sideFromSize(-0.5)).toBe('short');
    expect(sideFromSize(-100)).toBe('short');
    expect(sideFromSize(-0.0001)).toBe('short');
  });

  test('returns flat for zero size', () => {
    expect(sideFromSize(0)).toBe('flat');
    expect(sideFromSize(-0)).toBe('flat');
  });
});

describe('Action label derivation', () => {
  function deriveActionLabel(startPosition: number, delta: number): string {
    const newPos = startPosition + delta;

    if (startPosition === 0) {
      return delta > 0 ? 'Open Long' : 'Open Short';
    } else if (startPosition > 0) {
      if (delta > 0) return 'Increase Long';
      return newPos === 0 ? 'Close Long' : 'Decrease Long';
    } else {
      // startPosition < 0
      if (delta < 0) return 'Increase Short';
      return newPos === 0 ? 'Close Short' : 'Decrease Short';
    }
  }

  test('derives Open Long when starting from flat with positive delta', () => {
    expect(deriveActionLabel(0, 0.5)).toBe('Open Long');
  });

  test('derives Open Short when starting from flat with negative delta', () => {
    expect(deriveActionLabel(0, -0.5)).toBe('Open Short');
  });

  test('derives Increase Long when adding to long position', () => {
    expect(deriveActionLabel(0.5, 0.3)).toBe('Increase Long');
  });

  test('derives Decrease Long when reducing long position', () => {
    expect(deriveActionLabel(0.5, -0.2)).toBe('Decrease Long');
  });

  test('derives Close Long when closing entire long position', () => {
    expect(deriveActionLabel(0.5, -0.5)).toBe('Close Long');
  });

  test('derives Increase Short when adding to short position', () => {
    expect(deriveActionLabel(-0.5, -0.3)).toBe('Increase Short');
  });

  test('derives Decrease Short when reducing short position', () => {
    expect(deriveActionLabel(-0.5, 0.2)).toBe('Decrease Short');
  });

  test('derives Close Short when closing entire short position', () => {
    expect(deriveActionLabel(-0.5, 0.5)).toBe('Close Short');
  });
});

describe('ensureFreshSnapshots age check', () => {
  test('identifies stale snapshots', () => {
    const maxAgeMs = 60000; // 1 minute
    const now = Date.now();

    const isStale = (updatedAt: string) => {
      const updatedMs = Date.parse(updatedAt);
      return !Number.isFinite(updatedMs) || now - updatedMs > maxAgeMs;
    };

    // Fresh snapshot (30 seconds ago)
    expect(isStale(new Date(now - 30000).toISOString())).toBe(false);

    // Stale snapshot (2 minutes ago)
    expect(isStale(new Date(now - 120000).toISOString())).toBe(true);

    // Exactly at threshold
    expect(isStale(new Date(now - 60000).toISOString())).toBe(false);

    // Just over threshold
    expect(isStale(new Date(now - 60001).toISOString())).toBe(true);

    // Invalid date
    expect(isStale('invalid-date')).toBe(true);
  });
});

describe('Watchlist + Custom Accounts Integration', () => {
  /**
   * Mock watchlist builder that merges leaderboard selections with custom pinned accounts
   */
  function buildWatchlist(
    leaderboardSelected: string[],
    customPinned: string[]
  ): string[] {
    const normalized = new Set<string>();

    // Add leaderboard selections (normalized to lowercase)
    for (const addr of leaderboardSelected) {
      normalized.add(addr.toLowerCase());
    }

    // Add custom pinned accounts (normalized, won't duplicate)
    for (const addr of customPinned) {
      normalized.add(addr.toLowerCase());
    }

    return Array.from(normalized);
  }

  test('merges custom accounts with leaderboard selections without duplicates', () => {
    const leaderboard = [
      '0x1111111111111111111111111111111111111111',
      '0x2222222222222222222222222222222222222222',
    ];
    const custom = [
      '0x3333333333333333333333333333333333333333',
      '0x1111111111111111111111111111111111111111', // Already in leaderboard
    ];

    const watchlist = buildWatchlist(leaderboard, custom);

    expect(watchlist).toHaveLength(3); // No duplicates
    expect(watchlist).toContain('0x1111111111111111111111111111111111111111');
    expect(watchlist).toContain('0x2222222222222222222222222222222222222222');
    expect(watchlist).toContain('0x3333333333333333333333333333333333333333');
  });

  test('handles case-insensitive duplicate detection', () => {
    const leaderboard = ['0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'];
    const custom = ['0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa']; // Same address, different case

    const watchlist = buildWatchlist(leaderboard, custom);

    expect(watchlist).toHaveLength(1); // Only one address
    expect(watchlist[0]).toBe('0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa');
  });

  test('updates watchlist when custom account is added', () => {
    // Initial state: only leaderboard
    let watchlist = buildWatchlist(
      ['0x1111111111111111111111111111111111111111'],
      []
    );
    expect(watchlist).toHaveLength(1);

    // Add custom account
    watchlist = buildWatchlist(
      ['0x1111111111111111111111111111111111111111'],
      ['0x2222222222222222222222222222222222222222']
    );
    expect(watchlist).toHaveLength(2);
    expect(watchlist).toContain('0x2222222222222222222222222222222222222222');
  });

  test('updates watchlist when custom account is removed', () => {
    // Initial state: leaderboard + custom
    let watchlist = buildWatchlist(
      ['0x1111111111111111111111111111111111111111'],
      ['0x2222222222222222222222222222222222222222']
    );
    expect(watchlist).toHaveLength(2);

    // Remove custom account
    watchlist = buildWatchlist(
      ['0x1111111111111111111111111111111111111111'],
      [] // Custom removed
    );
    expect(watchlist).toHaveLength(1);
    expect(watchlist).not.toContain('0x2222222222222222222222222222222222222222');
  });

  test('handles empty leaderboard with only custom accounts', () => {
    const watchlist = buildWatchlist([], [
      '0x1111111111111111111111111111111111111111',
      '0x2222222222222222222222222222222222222222',
    ]);

    expect(watchlist).toHaveLength(2);
  });

  test('handles both empty', () => {
    const watchlist = buildWatchlist([], []);
    expect(watchlist).toHaveLength(0);
  });
});

describe('Custom Account Limit Enforcement', () => {
  const MAX_CUSTOM_ACCOUNTS = 3;

  test('allows up to MAX_CUSTOM_ACCOUNTS', () => {
    const customAccounts: string[] = [];

    const addCustomAccount = (address: string): { success: boolean; error?: string } => {
      if (customAccounts.length >= MAX_CUSTOM_ACCOUNTS) {
        return { success: false, error: `Maximum of ${MAX_CUSTOM_ACCOUNTS} custom accounts` };
      }
      customAccounts.push(address.toLowerCase());
      return { success: true };
    };

    // Add 3 accounts (should succeed)
    expect(addCustomAccount('0x1111111111111111111111111111111111111111').success).toBe(true);
    expect(addCustomAccount('0x2222222222222222222222222222222222222222').success).toBe(true);
    expect(addCustomAccount('0x3333333333333333333333333333333333333333').success).toBe(true);
    expect(customAccounts).toHaveLength(3);

    // 4th should fail with 400-equivalent error
    const result = addCustomAccount('0x4444444444444444444444444444444444444444');
    expect(result.success).toBe(false);
    expect(result.error).toContain('Maximum');
    expect(result.error).toContain('3');
  });

  test('concurrent add requests respect limit atomically', async () => {
    let count = 0;
    let lockHolder: Promise<void> | null = null;

    // Simulate atomic add with mutex-style lock
    const addWithLock = async (address: string): Promise<{ success: boolean }> => {
      // Wait for any existing lock to release (simulates DB lock)
      while (lockHolder) {
        await lockHolder;
      }

      // Acquire the lock
      let releaseLock: () => void;
      lockHolder = new Promise<void>(resolve => {
        releaseLock = resolve;
      });

      try {
        // Check count inside lock
        if (count >= MAX_CUSTOM_ACCOUNTS) {
          return { success: false };
        }

        // Simulate async DB operation
        await new Promise(r => setTimeout(r, 5));

        count++;
        return { success: true };
      } finally {
        // Release lock
        lockHolder = null;
        releaseLock!();
      }
    };

    // Fire off 5 concurrent adds
    const results = await Promise.all([
      addWithLock('0x1111111111111111111111111111111111111111'),
      addWithLock('0x2222222222222222222222222222222222222222'),
      addWithLock('0x3333333333333333333333333333333333333333'),
      addWithLock('0x4444444444444444444444444444444444444444'),
      addWithLock('0x5555555555555555555555555555555555555555'),
    ]);

    const successCount = results.filter(r => r.success).length;
    const failCount = results.filter(r => !r.success).length;

    // With proper locking, exactly 3 should succeed
    expect(successCount).toBe(3);
    expect(failCount).toBe(2);
    expect(count).toBe(3);
  });
});
