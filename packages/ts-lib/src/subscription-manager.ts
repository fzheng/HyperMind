/**
 * Centralized WebSocket Subscription Manager
 *
 * Manages address subscriptions from multiple sources (Legacy, Alpha Pool, etc.)
 * with automatic deduplication. Any system can register addresses, and the manager
 * ensures each unique address is only subscribed once via WebSocket.
 *
 * @module subscription-manager
 */

import { normalizeAddress } from './utils';

/**
 * Metadata about a subscription source registration.
 */
export interface SourceRegistration {
  /** When this source registered the address */
  registeredAt: Date;
}

/**
 * Subscription method for an address.
 */
export type SubscriptionMethod = 'websocket' | 'polling' | 'none';

/**
 * A subscription entry for a single address.
 */
export interface SubscriptionEntry {
  /** Normalized address (lowercase) */
  address: string;
  /** Sources that registered this address (source name -> metadata) */
  sources: Map<string, SourceRegistration>;
  /** When this address was first subscribed (null if not yet subscribed) */
  subscribedAt: Date | null;
  /** Current subscription method */
  method: SubscriptionMethod;
  /** Manual override: force this address to polling (user demoted it) */
  manualPolling?: boolean;
  /** Manual override: force this address to websocket (user promoted it) */
  manualWebsocket?: boolean;
}

/**
 * Status snapshot for observability.
 */
export interface SubscriptionStatus {
  /** Total unique addresses (deduped) */
  totalAddresses: number;
  /** Count of addresses per source */
  addressesBySource: Record<string, number>;
  /** Count of addresses by subscription method */
  addressesByMethod: Record<SubscriptionMethod, number>;
  /** Number of duplicate registrations (same address from multiple sources) */
  duplicateCount: number;
  /** List of sources */
  sources: string[];
  /** Maximum WebSocket slots available */
  maxWebSocketSlots: number;
}

/**
 * Source priority for WebSocket allocation.
 * Lower number = higher priority.
 * Pinned accounts get highest priority since users explicitly chose to track them.
 */
export const SOURCE_PRIORITY: Record<string, number> = {
  pinned: 0, // User-pinned accounts have highest priority
  legacy: 1, // Legacy leaderboard auto-selected accounts
  'alpha-pool': 2, // NIG-selected Alpha Pool addresses
};

/**
 * Options for SubscriptionManager constructor.
 */
export interface SubscriptionManagerOptions {
  /**
   * Callback fired when the set of addresses changes.
   * Used to trigger RealtimeTracker refresh.
   */
  onChanged?: () => Promise<void>;
  /**
   * Maximum number of WebSocket slots available.
   * Addresses beyond this limit will use polling.
   */
  maxWebSocketSlots?: number;
}

/**
 * Centralized subscription manager for Hyperliquid address tracking.
 *
 * Features:
 * - Address-level deduplication (same address from multiple sources = 1 subscription)
 * - Source tracking for observability
 * - Atomic replace operation for source updates
 * - Callback notification when addresses change
 *
 * @example
 * ```typescript
 * const manager = new SubscriptionManager({
 *   onChanged: async () => {
 *     await tracker.refresh();
 *   }
 * });
 *
 * // Legacy system registers its addresses
 * await manager.replaceForSource('legacy', legacyAddresses);
 *
 * // Alpha Pool registers its selected addresses
 * await manager.replaceForSource('alpha-pool', selectedAddresses);
 *
 * // Get all unique addresses for WebSocket subscription
 * const allAddresses = manager.getAllAddresses();
 * ```
 */
export class SubscriptionManager {
  /** Map of normalized address -> subscription entry */
  private registrations: Map<string, SubscriptionEntry> = new Map();

  /** Callback when addresses change */
  private onChanged?: () => Promise<void>;

  /** Maximum WebSocket slots available */
  private maxWebSocketSlots: number;

  constructor(opts?: SubscriptionManagerOptions) {
    this.onChanged = opts?.onChanged;
    this.maxWebSocketSlots = opts?.maxWebSocketSlots ?? 10;
  }

  /**
   * Register addresses from a source.
   *
   * @param source - Source identifier (e.g., 'legacy', 'alpha-pool')
   * @param addresses - Array of Ethereum addresses to register
   * @returns Array of addresses that were newly added (not previously registered by any source)
   */
  async register(source: string, addresses: string[]): Promise<string[]> {
    const newAddresses: string[] = [];
    const now = new Date();

    for (const addr of addresses) {
      const normalized = normalizeAddress(addr);
      let entry = this.registrations.get(normalized);

      if (!entry) {
        // New address - create entry
        entry = {
          address: normalized,
          sources: new Map(),
          subscribedAt: null,
          method: 'none',
        };
        this.registrations.set(normalized, entry);
        newAddresses.push(normalized);
      }

      // Add/update source registration
      entry.sources.set(source, { registeredAt: now });
    }

    // Reassign subscription methods based on priority
    this.reassignMethods();

    // Notify if addresses were added
    if (newAddresses.length > 0 && this.onChanged) {
      await this.onChanged();
    }

    return newAddresses;
  }

  /**
   * Unregister addresses from a source.
   *
   * Only removes addresses that have no remaining sources after unregistration.
   *
   * @param source - Source identifier
   * @param addresses - Array of addresses to unregister
   * @returns Array of addresses that were completely removed (no sources left)
   */
  async unregister(source: string, addresses: string[]): Promise<string[]> {
    const removedAddresses: string[] = [];

    for (const addr of addresses) {
      const normalized = normalizeAddress(addr);
      const entry = this.registrations.get(normalized);

      if (entry) {
        // Remove this source's registration
        entry.sources.delete(source);

        // If no sources left, remove the address entirely
        if (entry.sources.size === 0) {
          this.registrations.delete(normalized);
          removedAddresses.push(normalized);
        }
      }
    }

    // Reassign subscription methods based on priority
    this.reassignMethods();

    // Notify if addresses were removed
    if (removedAddresses.length > 0 && this.onChanged) {
      await this.onChanged();
    }

    return removedAddresses;
  }

  /**
   * Replace all addresses for a source atomically.
   *
   * This is the preferred method for sources that maintain a fixed list of addresses.
   * It handles both adding new addresses and removing stale ones in a single operation.
   *
   * @param source - Source identifier
   * @param addresses - New complete list of addresses for this source
   */
  async replaceForSource(source: string, addresses: string[]): Promise<void> {
    // Get current addresses for this source
    const currentForSource: string[] = [];
    for (const [addr, entry] of this.registrations) {
      if (entry.sources.has(source)) {
        currentForSource.push(addr);
      }
    }

    // Compute diff
    const newSet = new Set(addresses.map((a) => normalizeAddress(a)));
    const currentSet = new Set(currentForSource);

    const toRemove = currentForSource.filter((a) => !newSet.has(a));
    const toAdd = addresses.filter((a) => !currentSet.has(normalizeAddress(a)));

    // Apply changes (unregister first to avoid unnecessary onChanged calls)
    let changed = false;

    // Remove stale addresses
    for (const addr of toRemove) {
      const entry = this.registrations.get(addr);
      if (entry) {
        entry.sources.delete(source);
        if (entry.sources.size === 0) {
          this.registrations.delete(addr);
          changed = true;
        }
      }
    }

    // Add new addresses
    const now = new Date();
    for (const addr of toAdd) {
      const normalized = normalizeAddress(addr);
      let entry = this.registrations.get(normalized);

      if (!entry) {
        entry = {
          address: normalized,
          sources: new Map(),
          subscribedAt: null,
          method: 'none',
        };
        this.registrations.set(normalized, entry);
        changed = true;
      }

      entry.sources.set(source, { registeredAt: now });
    }

    // Update existing addresses (re-register to update timestamp)
    for (const addr of addresses) {
      const normalized = normalizeAddress(addr);
      const entry = this.registrations.get(normalized);
      if (entry && !toAdd.includes(addr)) {
        entry.sources.set(source, { registeredAt: now });
      }
    }

    // Reassign subscription methods based on priority
    this.reassignMethods();

    // Notify if addresses changed
    if (changed && this.onChanged) {
      await this.onChanged();
    }
  }

  /**
   * Get all unique addresses (deduped across all sources).
   *
   * @returns Array of normalized addresses
   */
  getAllAddresses(): string[] {
    return Array.from(this.registrations.keys());
  }

  /**
   * Get addresses registered by a specific source.
   *
   * @param source - Source identifier
   * @returns Array of addresses for that source
   */
  getAddressesForSource(source: string): string[] {
    const result: string[] = [];
    for (const [addr, entry] of this.registrations) {
      if (entry.sources.has(source)) {
        result.push(addr);
      }
    }
    return result;
  }

  /**
   * Get all sources that registered a specific address.
   *
   * @param address - Address to look up
   * @returns Array of source identifiers
   */
  getSourcesForAddress(address: string): string[] {
    const entry = this.registrations.get(normalizeAddress(address));
    return entry ? Array.from(entry.sources.keys()) : [];
  }

  /**
   * Mark an address as subscribed (for tracking purposes).
   *
   * @param address - Address that was subscribed
   */
  markSubscribed(address: string): void {
    const entry = this.registrations.get(normalizeAddress(address));
    if (entry && !entry.subscribedAt) {
      entry.subscribedAt = new Date();
    }
  }

  /**
   * Mark an address as unsubscribed (for tracking purposes).
   *
   * @param address - Address that was unsubscribed
   */
  markUnsubscribed(address: string): void {
    const entry = this.registrations.get(normalizeAddress(address));
    if (entry) {
      entry.subscribedAt = null;
    }
  }

  /**
   * Get subscription status for observability.
   *
   * @returns Status snapshot with counts and metadata
   */
  getStatus(): SubscriptionStatus {
    const bySource: Record<string, number> = {};
    const byMethod: Record<SubscriptionMethod, number> = {
      websocket: 0,
      polling: 0,
      none: 0,
    };
    let totalRegistrations = 0;
    const sourcesSet = new Set<string>();

    for (const entry of this.registrations.values()) {
      byMethod[entry.method]++;
      for (const source of entry.sources.keys()) {
        bySource[source] = (bySource[source] || 0) + 1;
        totalRegistrations++;
        sourcesSet.add(source);
      }
    }

    return {
      totalAddresses: this.registrations.size,
      addressesBySource: bySource,
      addressesByMethod: byMethod,
      duplicateCount: totalRegistrations - this.registrations.size,
      sources: Array.from(sourcesSet).sort(),
      maxWebSocketSlots: this.maxWebSocketSlots,
    };
  }

  /**
   * Get detailed info about a specific address.
   *
   * @param address - Address to look up
   * @returns Entry details or null if not registered
   */
  getAddressInfo(address: string): SubscriptionEntry | null {
    return this.registrations.get(normalizeAddress(address)) || null;
  }

  /**
   * Get the subscription method for an address.
   *
   * @param address - Address to look up
   * @returns Subscription method or 'none' if not registered
   */
  getMethod(address: string): SubscriptionMethod {
    const entry = this.registrations.get(normalizeAddress(address));
    return entry?.method ?? 'none';
  }

  /**
   * Get all addresses using a specific subscription method.
   *
   * @param method - Subscription method to filter by
   * @returns Array of addresses using that method
   */
  getAddressesByMethod(method: SubscriptionMethod): string[] {
    const result: string[] = [];
    for (const [addr, entry] of this.registrations) {
      if (entry.method === method) {
        result.push(addr);
      }
    }
    return result;
  }

  /**
   * Get addresses sorted by source priority (for WebSocket allocation).
   * Legacy addresses come first, then alpha-pool, then others.
   *
   * @returns Array of addresses sorted by priority
   */
  getAddressesByPriority(): string[] {
    const entries = Array.from(this.registrations.values());

    // Sort by priority: lower number = higher priority
    entries.sort((a, b) => {
      const aPriority = this.getEntryPriority(a);
      const bPriority = this.getEntryPriority(b);
      return aPriority - bPriority;
    });

    return entries.map((e) => e.address);
  }

  /**
   * Get the priority for an entry based on its sources.
   * Uses the highest priority (lowest number) among all sources.
   */
  private getEntryPriority(entry: SubscriptionEntry): number {
    let minPriority = 999;
    for (const source of entry.sources.keys()) {
      const priority = SOURCE_PRIORITY[source] ?? 100;
      if (priority < minPriority) {
        minPriority = priority;
      }
    }
    return minPriority;
  }

  /**
   * Reassign subscription methods based on priority and available WebSocket slots.
   * Called automatically when addresses change.
   *
   * Priority order:
   * 1. Pinned addresses (always get WebSocket if slots available)
   * 2. Manually promoted addresses (user explicitly promoted)
   * 3. Auto-assigned by source priority (legacy > alpha-pool)
   *
   * Manually demoted addresses are forced to polling and their slots are reserved
   * (not auto-filled) so users can manually pick which address to promote.
   */
  private reassignMethods(): void {
    // Get all entries
    const entries = Array.from(this.registrations.values());

    // Separate into categories
    const pinnedEntries: SubscriptionEntry[] = [];
    const manualWsEntries: SubscriptionEntry[] = [];
    const manualPollingEntries: SubscriptionEntry[] = [];
    const autoEntries: SubscriptionEntry[] = [];

    for (const entry of entries) {
      const isPinned = entry.sources.has('pinned');
      if (isPinned) {
        // Pinned addresses: always highest priority, ignore manual flags
        pinnedEntries.push(entry);
      } else if (entry.manualPolling) {
        // User demoted: force to polling, reserve their slot
        manualPollingEntries.push(entry);
      } else if (entry.manualWebsocket) {
        // User promoted: try to give WebSocket
        manualWsEntries.push(entry);
      } else {
        // Auto-assign by source priority
        autoEntries.push(entry);
      }
    }

    // Sort auto entries by priority
    autoEntries.sort((a, b) => this.getEntryPriority(a) - this.getEntryPriority(b));

    // Calculate how many slots are reserved (not available for auto-assignment).
    // Each manually demoted address reserves 1 slot.
    // Each manually promoted address consumes 1 reserved slot.
    // Reserved = demoted - promoted (minimum 0)
    const reservedSlots = Math.max(0, manualPollingEntries.length - manualWsEntries.length);

    // Calculate slots available for auto-assignment:
    // Total slots minus reserved minus pinned minus manually promoted
    // This ensures demoted slots stay empty until manually filled
    const slotsForPinned = Math.min(pinnedEntries.length, this.maxWebSocketSlots);
    const slotsForManualWs = Math.min(manualWsEntries.length, this.maxWebSocketSlots - slotsForPinned);
    const slotsForAuto = Math.max(0, this.maxWebSocketSlots - slotsForPinned - slotsForManualWs - reservedSlots);

    // Assign methods:
    // 1. Pinned get WebSocket first
    // 2. Manual WebSocket get next available slots
    // 3. Auto-assigned fill remaining slots (limited to preserve reserved)
    // 4. Manual polling always get polling

    let wsCount = 0;

    // Pinned addresses get WebSocket (highest priority)
    for (const entry of pinnedEntries) {
      if (wsCount < this.maxWebSocketSlots) {
        entry.method = 'websocket';
        wsCount++;
      } else {
        entry.method = 'polling';
      }
    }

    // Manually promoted addresses get next available slots
    for (const entry of manualWsEntries) {
      if (wsCount < this.maxWebSocketSlots) {
        entry.method = 'websocket';
        wsCount++;
      } else {
        // No slots available - clear the manual flag and use polling
        entry.manualWebsocket = false;
        entry.method = 'polling';
      }
    }

    // Auto-assigned addresses fill remaining slots by priority
    // Limited to slotsForAuto to preserve reserved slots
    let autoAssigned = 0;
    for (const entry of autoEntries) {
      if (autoAssigned < slotsForAuto) {
        entry.method = 'websocket';
        autoAssigned++;
      } else {
        entry.method = 'polling';
      }
    }

    // Manually demoted addresses always get polling
    for (const entry of manualPollingEntries) {
      entry.method = 'polling';
    }
  }

  /**
   * Manually demote an address to polling (free a WebSocket slot).
   * Only works for unpinned addresses currently using WebSocket.
   *
   * @param address - Address to demote
   * @returns true if demoted, false if not possible (pinned or not found)
   */
  demoteToPolling(address: string): boolean {
    const normalized = normalizeAddress(address);
    const entry = this.registrations.get(normalized);

    if (!entry) return false;

    // Cannot demote pinned addresses
    if (entry.sources.has('pinned')) return false;

    // Set manual polling flag and clear any manual websocket flag
    entry.manualPolling = true;
    entry.manualWebsocket = false;

    // Reassign methods to apply the change
    this.reassignMethods();

    return true;
  }

  /**
   * Manually promote an address to WebSocket (use an available slot).
   * Only works if there's an available WebSocket slot.
   *
   * @param address - Address to promote
   * @returns true if promoted, false if no slots available or not found
   */
  promoteToWebsocket(address: string): boolean {
    const normalized = normalizeAddress(address);
    const entry = this.registrations.get(normalized);

    if (!entry) return false;

    // Check if slots are available
    const currentWsCount = this.getAddressesByMethod('websocket').length;
    if (currentWsCount >= this.maxWebSocketSlots) {
      return false;
    }

    // Set manual websocket flag and clear any manual polling flag
    entry.manualWebsocket = true;
    entry.manualPolling = false;

    // Reassign methods to apply the change
    this.reassignMethods();

    return true;
  }

  /**
   * Clear manual override for an address (return to auto-assignment).
   *
   * @param address - Address to clear override for
   */
  clearManualOverride(address: string): void {
    const normalized = normalizeAddress(address);
    const entry = this.registrations.get(normalized);

    if (entry) {
      entry.manualPolling = false;
      entry.manualWebsocket = false;
      this.reassignMethods();
    }
  }

  /**
   * Check if an address has a manual override.
   *
   * @param address - Address to check
   * @returns Object with override status
   */
  getManualOverride(address: string): { manualPolling: boolean; manualWebsocket: boolean } {
    const normalized = normalizeAddress(address);
    const entry = this.registrations.get(normalized);

    return {
      manualPolling: entry?.manualPolling ?? false,
      manualWebsocket: entry?.manualWebsocket ?? false,
    };
  }

  /**
   * Update the maximum WebSocket slots and reassign methods.
   *
   * @param maxSlots - New maximum WebSocket slots
   */
  setMaxWebSocketSlots(maxSlots: number): void {
    this.maxWebSocketSlots = maxSlots;
    this.reassignMethods();
  }

  /**
   * Get the current maximum WebSocket slots.
   */
  getMaxWebSocketSlots(): number {
    return this.maxWebSocketSlots;
  }

  /**
   * Clear all registrations (for testing).
   */
  clear(): void {
    this.registrations.clear();
  }
}
