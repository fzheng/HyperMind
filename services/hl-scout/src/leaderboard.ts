/**
 * Leaderboard Service Module
 *
 * Polls and scores the Hyperliquid leaderboard to identify top-performing traders.
 * Uses a composite scoring formula that prioritizes:
 * 1. Stability Score (50%) - smooth, controlled profit generation
 * 2. Win Rate (25%) - with penalty for rates below 60%
 * 3. Trade Frequency (15%) - with penalty for excessive trading
 * 4. Realized PnL (10%) - tiebreaker for large accounts
 *
 * Enriches top performers with detailed statistics from Hyperliquid's info API
 * and persists results to PostgreSQL for dashboard display.
 *
 * @module leaderboard
 */

import {
  createLogger,
  getPool,
  normalizeAddress,
  nowIso,
  CandidateEventSchema,
  computePerformanceScore,
  DEFAULT_SCORING_PARAMS,
  listPinnedAccounts,
  getPinnedAddressSet,
  unpinAccount,
  type ScoringParams
} from '@hl/ts-lib';
import type { CandidateEvent } from '@hl/ts-lib';

/** Base URL for Hyperbot leaderboard API */
const DEFAULT_API_URL = 'https://hyperbot.network/api/leaderboard/smart';
/** Hyperbot address remarks batch API endpoint */
const HYPERBOT_REMARKS_URL = 'https://hyperbot.network/api/leaderboard/smart/query-addr-remarks';
/** Hyperliquid info API endpoint */
const HYPERLIQUID_INFO_URL = 'https://api.hyperliquid.xyz/info';
/** Maps window names to period days */
const WINDOW_PERIOD_MAP: Record<string, number> = { day: 1, week: 7, month: 30 };
/** Concurrency limit for fetching address statistics (reduced from 4 to 2 for rate limit safety) */
const DEFAULT_STATS_CONCURRENCY = Number(process.env.LEADERBOARD_STATS_CONCURRENCY ?? 2);
/** Concurrency limit for fetching PnL series data */
const DEFAULT_SERIES_CONCURRENCY = Number(process.env.LEADERBOARD_SERIES_CONCURRENCY ?? 2);
/** Delay between paginated leaderboard requests in ms */
const PAGE_REQUEST_DELAY_MS = Number(process.env.LEADERBOARD_PAGE_DELAY_MS ?? 500);
/** Delay between individual API requests in ms (rate limiting) */
const REQUEST_DELAY_MS = Number(process.env.HYPERBOT_REQUEST_DELAY_MS ?? 300);
/** Cache TTL for address stats in ms (1 hour) */
const STATS_CACHE_TTL_MS = Number(process.env.HYPERBOT_STATS_CACHE_TTL_MS ?? 60 * 60 * 1000);
/** Cache TTL for BTC/ETH analysis in-memory cache in ms (1 hour) - DB cache is 30 days */
const BTC_ETH_CACHE_TTL_MS = Number(process.env.HYPERBOT_BTC_ETH_CACHE_TTL_MS ?? 60 * 60 * 1000);
/** Maximum cache size for LRU eviction */
const MAX_CACHE_SIZE = Number(process.env.HYPERBOT_MAX_CACHE_SIZE ?? 500);
/** Delay between BTC/ETH analysis requests in ms (separate from general rate limiter) */
const BTC_ETH_REQUEST_DELAY_MS = Number(process.env.BTC_ETH_REQUEST_DELAY_MS ?? 500);
/** Maximum retries for rate-limited requests */
const MAX_RATE_LIMIT_RETRIES = Number(process.env.MAX_RATE_LIMIT_RETRIES ?? 3);
/** Base delay for exponential backoff on rate limit (ms) */
const RATE_LIMIT_BACKOFF_BASE_MS = Number(process.env.RATE_LIMIT_BACKOFF_BASE_MS ?? 2000);
/** Database cache TTL for BTC/ETH analysis (30 days in ms) */
const BTC_ETH_DB_CACHE_TTL_DAYS = Number(process.env.BTC_ETH_DB_CACHE_TTL_DAYS ?? 30);

/**
 * Simple LRU cache with TTL support for Hyperbot API responses.
 * Prevents excessive API calls by caching frequently requested data.
 */
class LRUCache<T> {
  private cache = new Map<string, { value: T; expiresAt: number }>();
  private maxSize: number;
  private ttlMs: number;

  constructor(maxSize: number, ttlMs: number) {
    this.maxSize = maxSize;
    this.ttlMs = ttlMs;
  }

  get(key: string): T | undefined {
    const entry = this.cache.get(key);
    if (!entry) return undefined;
    if (Date.now() > entry.expiresAt) {
      this.cache.delete(key);
      return undefined;
    }
    // Move to end (most recently used)
    this.cache.delete(key);
    this.cache.set(key, entry);
    return entry.value;
  }

  set(key: string, value: T): void {
    // Delete if exists to update position
    this.cache.delete(key);
    // Evict oldest if at capacity
    while (this.cache.size >= this.maxSize) {
      const firstKey = this.cache.keys().next().value;
      if (firstKey !== undefined) {
        this.cache.delete(firstKey);
      }
    }
    this.cache.set(key, { value, expiresAt: Date.now() + this.ttlMs });
  }

  clear(): void {
    this.cache.clear();
  }

  get size(): number {
    return this.cache.size;
  }
}

/** Cache for address stats from Hyperbot */
const addressStatsCache = new LRUCache<AddressStats>(MAX_CACHE_SIZE, STATS_CACHE_TTL_MS);
/** Cache for BTC/ETH analysis results */
const btcEthAnalysisCache = new LRUCache<BtcEthAnalysis>(MAX_CACHE_SIZE, BTC_ETH_CACHE_TTL_MS);

/**
 * Rate limiter to prevent excessive API calls.
 * Uses token bucket algorithm with configurable delay between requests.
 */
class RateLimiter {
  private lastRequestTime = 0;
  private delayMs: number;

  constructor(delayMs: number) {
    this.delayMs = delayMs;
  }

  async acquire(): Promise<void> {
    const now = Date.now();
    const elapsed = now - this.lastRequestTime;
    if (elapsed < this.delayMs) {
      await sleep(this.delayMs - elapsed);
    }
    this.lastRequestTime = Date.now();
  }
}

/** Rate limiter for Hyperbot API requests */
const hyperbotRateLimiter = new RateLimiter(REQUEST_DELAY_MS);

/**
 * Hyperbot leaderboard sort options
 * Used in API query parameter: ?sort=<value>
 */
export enum LeaderboardSort {
  /** Sort by win rate (all accounts in hyperliquid) */
  WIN_RATE = 0,
  /** Sort by account total value */
  ACCOUNT_VALUE = 1,
  /** Sort by realized PnL */
  PNL = 3,
  /** Sort by total trades count */
  TRADES_COUNT = 4,
  /** Sort by profitable trades count */
  PROFITABLE_TRADES = 5,
  /** Sort by last operation time */
  LAST_OPERATION = 6,
  /** Sort by average holding period */
  AVG_HOLDING_PERIOD = 7,
  /** Sort by current positions */
  CURRENT_POSITIONS = 8,
}

const DEFAULT_LEADERBOARD_SORT = LeaderboardSort.PNL;

/**
 * Raw entry from the Hyperbot leaderboard API.
 * Contains basic trading statistics before scoring.
 */
type LeaderboardRawEntry = {
  /** Ethereum address */
  address: string;
  /** Win rate as decimal (0-1) */
  winRate?: number;
  /** Total number of executed orders */
  executedOrders?: number;
  /** Realized PnL in USD */
  realizedPnl?: number;
  /** User remark/bio */
  remark?: string | null;
  /** Account labels/tags */
  labels?: string[] | null;
  /** Historical PnL time series */
  pnlList?: Array<{ timestamp: number; value: string }>;
  /** Total account value in USD */
  accountValue?: number;
  /** Starting equity for period */
  startingEquity?: number;
  /** Maximum drawdown percentage */
  maxDrawdown?: number;
  /** Number of winning trades */
  numWins?: number;
  /** Number of losing trades */
  numLosses?: number;
  /** Timestamp (ms) of last operation - used for activity filtering */
  lastOperationAt?: number;
  /** Nested stats object from API (contains accurate maxDrawdown) */
  stats?: {
    maxDrawdown?: number;
    totalPnl?: number;
    openPosCount?: number;
    closePosCount?: number;
    avgPosDuration?: number;
    winRate?: number;
  };
};

/**
 * Scored and ranked leaderboard entry.
 * Contains composite score and all metrics for display.
 */
export type RankedEntry = {
  /** Ethereum address */
  address: string;
  /** Position in leaderboard (1 = best) */
  rank: number;
  /** Composite performance score (higher = better) */
  score: number;
  /** Weight for consensus calculations (score / sum of scores) */
  weight: number;
  /** Whether entry was filtered out */
  filtered?: boolean;
  /** Reason for filtering */
  filterReason?: 'not_profitable' | 'insufficient_data';
  /** Win rate as decimal (0-1) */
  winRate: number;
  /** Total executed orders */
  executedOrders: number;
  /** Realized PnL in USD */
  realizedPnl: number;
  /** PnL per trade efficiency */
  efficiency: number;
  /** Consistency of PnL over time */
  pnlConsistency: number;
  /** User remark/bio */
  remark: string | null;
  /** Account labels */
  labels: string[];
  /** Open position count from Hyperliquid stats */
  statOpenPositions: number | null;
  /** Closed position count from Hyperliquid stats */
  statClosedPositions: number | null;
  /** Average position duration in seconds */
  statAvgPosDuration: number | null;
  /** Total PnL from Hyperliquid stats */
  statTotalPnl: number | null;
  /** Max drawdown from Hyperliquid stats */
  statMaxDrawdown: number | null;
  /** Additional metadata */
  meta: any;
};

/** Stats from Hyperliquid leaderboard stats API */
type AddressStats = {
  winRate?: number;
  openPosCount?: number;
  closePosCount?: number;
  avgPosDuration?: number;
  totalPnl?: number;
  maxDrawdown?: number;
};

/** Portfolio PnL and equity history for a time window */
type PortfolioWindowSeries = {
  window: string;
  pnlHistory: Array<{ ts: number; value: number }>;
  equityHistory: Array<{ ts: number; value: number }>;
};

/** Completed trade record from Hyperliquid */
type CompletedTrade = {
  address: string;
  coin: string;
  marginMode: string;
  direction: string;
  size: number;
  entryPrice: number;
  startTime: string;
  endTime: string;
  closePrice: number;
  totalFee: number;
  pnl: number;
};

/** Analysis of BTC and ETH trading performance */
type BtcEthAnalysis = {
  btcPnl: number;
  ethPnl: number;
  totalPnl: number;
  btcEthPnl: number;
  btcEthRatio: number;
  qualified: boolean;
  reason?: string;
};

/**
 * Configuration options for the LeaderboardService.
 */
export interface LeaderboardOptions {
  /** Base URL for Hyperbot leaderboard API */
  apiUrl?: string;
  /** Maximum entries to fetch from API */
  topN?: number;
  /** Number of top entries to select for tracking */
  selectCount?: number;
  /** Time periods to fetch (in days) */
  periods?: number[];
  /** Page size for paginated API requests */
  pageSize?: number;
  /** Number of entries to enrich with detailed stats */
  enrichCount?: number;
  /** Sort order for leaderboard API (default: PNL) */
  sort?: LeaderboardSort;
}

/**
 * Service for polling and scoring the Hyperliquid leaderboard.
 *
 * Responsibilities:
 * - Periodically fetch leaderboard data from Hyperbot API
 * - Score entries using composite performance formula
 * - Enrich top performers with detailed Hyperliquid statistics
 * - Persist results to PostgreSQL
 * - Publish candidate events for downstream services
 *
 * @example
 * ```typescript
 * const service = new LeaderboardService({
 *   topN: 1000,
 *   selectCount: 12,
 *   periods: [30],
 * }, async (candidate) => {
 *   await publishJson(js, 'a.candidates.v1', candidate);
 * });
 * service.start(); // Refreshes daily at 00:30 UTC
 * ```
 */
/**
 * Refresh status information
 */
export interface RefreshStatus {
  /** Current status: 'idle', 'refreshing', 'error' */
  status: 'idle' | 'refreshing' | 'error';
  /** Whether a refresh is currently in progress */
  isRefreshing: boolean;
  /** Timestamp when current refresh started (if refreshing) */
  refreshStartedAt: string | null;
  /** Timestamp of last successful refresh */
  lastRefreshAt: string | null;
  /** Timestamp of next scheduled refresh */
  nextRefreshAt: string | null;
  /** Time until next refresh in milliseconds */
  nextRefreshInMs: number | null;
  /** Refresh interval in milliseconds */
  refreshIntervalMs: number;
  /** Current refresh progress (if refreshing) */
  progress?: {
    phase: string;
    detail?: string;
  };
  /** Error message if status is 'error' */
  error?: string;
}

export class LeaderboardService {
  private opts: Required<LeaderboardOptions>;
  private timer: NodeJS.Timeout | null = null;
  private logger = createLogger('leaderboard');
  private publishCandidate: (entry: CandidateEvent) => Promise<void>;
  private smartApiBase: string;
  private enrichCount: number;
  private statsConcurrency: number;
  private seriesConcurrency: number;

  // Refresh state tracking
  private _isRefreshing = false;
  private _refreshStartedAt: Date | null = null;
  private _lastRefreshAt: Date | null = null;
  private _nextScheduledRefreshAt: Date | null = null;
  private _refreshProgress: { phase: string; detail?: string } | null = null;
  private _lastRefreshError: string | null = null;

  /**
   * Creates a new LeaderboardService.
   *
   * @param opts - Configuration options
   * @param publishCandidate - Callback to publish candidate events to NATS
   */
  constructor(opts: LeaderboardOptions, publishCandidate: (entry: CandidateEvent) => Promise<void>) {
    this.opts = {
      apiUrl: opts.apiUrl || DEFAULT_API_URL,
      topN: opts.topN ?? 1000,
      selectCount: opts.selectCount ?? 12,
      periods: opts.periods?.length ? opts.periods : [30],
      pageSize: opts.pageSize ?? 100,
      enrichCount: opts.enrichCount ?? opts.selectCount ?? 12,
      sort: opts.sort ?? DEFAULT_LEADERBOARD_SORT,
    };
    this.publishCandidate = publishCandidate;
    this.smartApiBase = this.opts.apiUrl.endsWith('/') ? this.opts.apiUrl : `${this.opts.apiUrl}/`;
    this.enrichCount = Math.max(0, Math.min(this.opts.enrichCount, this.opts.topN));
    this.statsConcurrency = Math.max(1, DEFAULT_STATS_CONCURRENCY);
    this.seriesConcurrency = Math.max(1, DEFAULT_SERIES_CONCURRENCY);
  }

  /**
   * Calculate milliseconds until next daily refresh time (00:30:00 UTC)
   * Using 00:30 to avoid conflicts with other services that reset at UTC midnight
   */
  private getMillisUntilNextRefresh(): number {
    const now = new Date();
    // Target: 00:30:00 UTC
    let nextRefresh = new Date(Date.UTC(
      now.getUTCFullYear(),
      now.getUTCMonth(),
      now.getUTCDate(),
      0, 30, 0, 0 // 00:30:00.000 UTC
    ));

    // If we've already passed 00:30 UTC today, schedule for tomorrow
    if (now.getTime() >= nextRefresh.getTime()) {
      nextRefresh = new Date(Date.UTC(
        now.getUTCFullYear(),
        now.getUTCMonth(),
        now.getUTCDate() + 1,
        0, 30, 0, 0
      ));
    }

    return nextRefresh.getTime() - now.getTime();
  }

  /**
   * Get the next daily refresh time as a Date (00:30:00 UTC)
   */
  private getNextRefreshTime(): Date {
    const now = new Date();
    let nextRefresh = new Date(Date.UTC(
      now.getUTCFullYear(),
      now.getUTCMonth(),
      now.getUTCDate(),
      0, 30, 0, 0
    ));

    // If we've already passed 00:30 UTC today, schedule for tomorrow
    if (now.getTime() >= nextRefresh.getTime()) {
      nextRefresh = new Date(Date.UTC(
        now.getUTCFullYear(),
        now.getUTCMonth(),
        now.getUTCDate() + 1,
        0, 30, 0, 0
      ));
    }

    return nextRefresh;
  }

  /**
   * Schedule refresh at 00:30 UTC daily
   */
  private scheduleNextRefresh() {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }

    const msUntilRefresh = this.getMillisUntilNextRefresh();
    this._nextScheduledRefreshAt = this.getNextRefreshTime();

    this.logger.info('leaderboard_refresh_scheduled', {
      nextRefreshAt: this._nextScheduledRefreshAt.toISOString(),
      msUntilRefresh,
      humanUntilRefresh: this.formatDuration(msUntilRefresh),
    });

    this.timer = setTimeout(() => {
      this.refreshAll()
        .catch((err) => this.logger.error('leaderboard_refresh_failed', { err: err?.message }))
        .finally(() => this.scheduleNextRefresh()); // Schedule next day's refresh
    }, msUntilRefresh);
  }

  start() {
    this._nextScheduledRefreshAt = this.getNextRefreshTime();

    this.logger.info('leaderboard_service_started', {
      nextRefreshAt: this._nextScheduledRefreshAt.toISOString(),
      periods: this.opts.periods,
      topN: this.opts.topN,
      selectCount: this.opts.selectCount,
    });

    // Run initial refresh immediately on startup
    this.refreshAll().catch((err) => this.logger.error('leaderboard_refresh_failed', { err: err?.message }));

    // Schedule daily refresh at 00:30 UTC
    this.scheduleNextRefresh();
  }

  stop() {
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
    this._nextScheduledRefreshAt = null;
    this.logger.info('leaderboard_service_stopped');
  }

  /**
   * Get current refresh status for API/UI
   */
  getRefreshStatus(): RefreshStatus {
    const now = Date.now();
    let nextRefreshInMs: number | null = null;

    if (this._nextScheduledRefreshAt && this.timer) {
      nextRefreshInMs = Math.max(0, this._nextScheduledRefreshAt.getTime() - now);
    }

    return {
      status: this._isRefreshing ? 'refreshing' : (this._lastRefreshError ? 'error' : 'idle'),
      isRefreshing: this._isRefreshing,
      refreshStartedAt: this._refreshStartedAt?.toISOString() || null,
      lastRefreshAt: this._lastRefreshAt?.toISOString() || null,
      nextRefreshAt: this._nextScheduledRefreshAt?.toISOString() || null,
      nextRefreshInMs,
      refreshIntervalMs: 24 * 60 * 60 * 1000, // Daily at UTC midnight
      progress: this._refreshProgress || undefined,
      error: this._lastRefreshError || undefined,
    };
  }

  /**
   * Format milliseconds to human readable duration
   */
  private formatDuration(ms: number): string {
    const hours = Math.floor(ms / (60 * 60 * 1000));
    const minutes = Math.floor((ms % (60 * 60 * 1000)) / (60 * 1000));
    if (hours > 0) return `${hours}h ${minutes}m`;
    return `${minutes}m`;
  }

  /**
   * Set refresh progress (for status API)
   */
  private setProgress(phase: string, detail?: string) {
    this._refreshProgress = { phase, detail };
    this.logger.debug?.('refresh_progress', { phase, detail });
  }

  async refreshAll() {
    if (this._isRefreshing) {
      this.logger.warn('refresh_already_in_progress', {
        startedAt: this._refreshStartedAt?.toISOString(),
        elapsedMs: this._refreshStartedAt ? Date.now() - this._refreshStartedAt.getTime() : null,
      });
      return;
    }

    this._isRefreshing = true;
    this._refreshStartedAt = new Date();
    this._lastRefreshError = null;
    this.setProgress('starting', 'Initializing refresh cycle');

    this.logger.info('leaderboard_refresh_started', {
      periods: this.opts.periods,
      timestamp: this._refreshStartedAt.toISOString(),
    });

    const startTime = Date.now();

    try {
      for (const period of this.opts.periods) {
        try {
          this.setProgress('refreshing_period', `Processing ${period}-day period`);
          await this.refreshPeriod(period);
        } catch (err: any) {
          this.logger.error('leaderboard_period_failed', { period, err: err?.message });
          this._lastRefreshError = `Period ${period} failed: ${err?.message}`;
        }
      }

      this._lastRefreshAt = new Date();
      const elapsedMs = Date.now() - startTime;

      this.logger.info('leaderboard_refresh_completed', {
        elapsedMs,
        elapsedHuman: this.formatDuration(elapsedMs),
        timestamp: this._lastRefreshAt.toISOString(),
      });

      // Log cache statistics after refresh
      this.logger.info('hyperbot_cache_stats', {
        addressStatsSize: addressStatsCache.size,
        btcEthAnalysisSize: btcEthAnalysisCache.size,
      });
    } catch (err: any) {
      this._lastRefreshError = err?.message || 'Unknown error';
      this.logger.error('leaderboard_refresh_failed', {
        err: err?.message,
        elapsedMs: Date.now() - startTime,
      });
    } finally {
      this._isRefreshing = false;
      this._refreshProgress = null;
    }
  }

  async refreshPeriod(period: number) {
    const periodStartTime = Date.now();
    this.logger.info('refresh_period_started', { period });

    // Phase 1: Fetch leaderboard data
    this.setProgress('fetching', `Fetching ${period}-day leaderboard from API`);
    const raw = await this.fetchPeriod(period);
    let ranked = this.scoreEntries(raw);
    this.logger.debug?.('refresh_phase_complete', { phase: 'fetch', period, entries: raw.length });

    // Phase 2: Enrich top entries with detailed stats
    const enrichTarget = Math.min(ranked.length, Math.max(this.enrichCount, this.opts.selectCount * 2));
    const toEnrich = ranked.slice(0, enrichTarget);

    let hyperliquidSeries = new Map<string, PortfolioWindowSeries[]>();
    if (toEnrich.length) {
      this.setProgress('enriching', `Enriching top ${toEnrich.length} entries with stats`);
      await this.applyAddressStats(period, toEnrich);
      this.logger.debug?.('refresh_phase_complete', { phase: 'stats', period, enriched: toEnrich.length });

      this.setProgress('portfolio', `Fetching portfolio series for ${toEnrich.length} entries`);
      hyperliquidSeries = await this.fetchPortfolioSeriesBatch(period, toEnrich);
      this.logger.debug?.('refresh_phase_complete', { phase: 'portfolio', period, fetched: hyperliquidSeries.size });
    }

    const tracked = ranked.slice(0, this.enrichCount);

    // Phase 3: Apply BTC/ETH filtering
    this.setProgress('filtering', `Filtering for BTC/ETH qualified candidates`);
    const qualifiedCandidates = await this.filterBtcEthQualified(ranked, this.opts.selectCount);
    this.logger.debug?.('refresh_phase_complete', { phase: 'btc_eth_filter', period, qualified: qualifiedCandidates.length });

    // Auto-unpin custom accounts that are now displayed in the system rankings
    // Only unpin if the account is BTC/ETH qualified (actually shown in dashboard)
    // Pinned accounts from leaderboard remain pinned (they're excluded from top-10 selection)
    try {
      const pinnedAccounts = await listPinnedAccounts(true); // Only custom accounts
      const qualifiedAddresses = new Set(qualifiedCandidates.slice(0, this.opts.selectCount).map(e => normalizeAddress(e.address)));

      for (const pinned of pinnedAccounts) {
        const normalizedPinned = normalizeAddress(pinned.address);
        if (qualifiedAddresses.has(normalizedPinned)) {
          // This custom pinned account is now displayed in system rankings, unpin it
          await unpinAccount(pinned.address);
          this.logger.info('custom_account_auto_unpinned', {
            address: normalizedPinned,
            reason: 'Account is now in displayed system rankings (BTC/ETH qualified)',
          });
        }
      }
    } catch (err: any) {
      this.logger.warn('custom_account_auto_unpin_failed', { err: err?.message });
    }

    // Phase 4: Persist to database
    this.setProgress('persisting', `Saving ${ranked.length} entries to database`);
    const persisted = await this.persistPeriod(period, ranked, tracked, hyperliquidSeries);

    const periodElapsedMs = Date.now() - periodStartTime;

    if (persisted) {
      this.setProgress('publishing', `Publishing ${qualifiedCandidates.length} candidates`);
      await this.publishTopCandidates(period, qualifiedCandidates);
      this.logger.info('refresh_period_completed', {
        period,
        elapsedMs: periodElapsedMs,
        elapsedHuman: this.formatDuration(periodElapsedMs),
        totalEntries: ranked.length,
        qualifiedEntries: qualifiedCandidates.length,
        targetCount: this.opts.selectCount,
      });
    } else {
      // Log that refresh was skipped - operators can detect upstream outages
      this.logger.warn('leaderboard_refresh_skipped', {
        period,
        elapsedMs: periodElapsedMs,
        reason: 'empty_upstream_data',
        message: 'Upstream API returned empty data; serving stale snapshot',
      });
    }
  }

  private async fetchPeriod(period: number): Promise<LeaderboardRawEntry[]> {
    const results: LeaderboardRawEntry[] = [];
    const pagesNeeded = Math.ceil(this.opts.topN / this.opts.pageSize);
    for (let page = 1; page <= pagesNeeded; page += 1) {
      // Rate limit between paginated requests to avoid hitting API limits
      if (page > 1) {
        await sleep(PAGE_REQUEST_DELAY_MS);
      }
      const url = `${this.opts.apiUrl}?pageNum=${page}&pageSize=${this.opts.pageSize}&period=${period}&sort=${this.opts.sort}`;
      const res = await fetch(url);
      if (!res.ok) throw new Error(`leaderboard HTTP ${res.status}`);
      const payload = await res.json();
      const data: LeaderboardRawEntry[] = payload?.data || [];
      results.push(...data);
      if (data.length < this.opts.pageSize) break;
    }
    this.logger.info('leaderboard_fetched', {
      period,
      pages: Math.min(pagesNeeded, Math.ceil(results.length / this.opts.pageSize)),
      entries: results.length,
    });
    return results.slice(0, this.opts.topN);
  }

  /**
   * Scores entries using the new composite performance scoring formula.
   *
   * Formula components (weighted sum):
   * 1. Stability Score (50%) - smooth, controlled profit generation (most important)
   * 2. Win Rate (25%) - with progressive penalty for < 60%
   * 3. Trade Frequency (15%) - with progressive penalty for > 150 trades
   * 4. Realized PnL (10%) - tiebreaker for large accounts
   */
  private scoreEntries(entries: LeaderboardRawEntry[]): RankedEntry[] {
    // Helper to safely parse env numbers with fallback to default on NaN/invalid values
    const parseEnvNumber = (envKey: string, defaultValue: number): number => {
      const envValue = process.env[envKey];
      if (envValue === undefined) return defaultValue;
      const parsed = Number(envValue);
      if (!Number.isFinite(parsed)) {
        this.logger.warn('invalid_env_number', {
          key: envKey,
          value: envValue,
          usingDefault: defaultValue,
        });
        return defaultValue;
      }
      return parsed;
    };

    // Build scoring params from defaults with optional env overrides (validated)
    const scoringParams: ScoringParams = {
      stabilityWeight: parseEnvNumber('SCORING_STABILITY_WEIGHT', DEFAULT_SCORING_PARAMS.stabilityWeight),
      winRateWeight: parseEnvNumber('SCORING_WIN_RATE_WEIGHT', DEFAULT_SCORING_PARAMS.winRateWeight),
      tradeFreqWeight: parseEnvNumber('SCORING_TRADE_FREQ_WEIGHT', DEFAULT_SCORING_PARAMS.tradeFreqWeight),
      pnlWeight: parseEnvNumber('SCORING_PNL_WEIGHT', DEFAULT_SCORING_PARAMS.pnlWeight),
      pnlReference: parseEnvNumber('SCORING_PNL_REFERENCE', DEFAULT_SCORING_PARAMS.pnlReference),
      minTrades: parseEnvNumber('SCORING_MIN_TRADES', DEFAULT_SCORING_PARAMS.minTrades),
      maxTrades: parseEnvNumber('SCORING_MAX_TRADES', DEFAULT_SCORING_PARAMS.maxTrades),
      tradeCountThreshold: parseEnvNumber('SCORING_TRADE_COUNT_THRESHOLD', DEFAULT_SCORING_PARAMS.tradeCountThreshold),
      winRateThreshold: parseEnvNumber('SCORING_WIN_RATE_THRESHOLD', DEFAULT_SCORING_PARAMS.winRateThreshold),
      drawdownTolerance: parseEnvNumber('SCORING_DRAWDOWN_TOLERANCE', DEFAULT_SCORING_PARAMS.drawdownTolerance),
      downsideTolerance: parseEnvNumber('SCORING_DOWNSIDE_TOLERANCE', DEFAULT_SCORING_PARAMS.downsideTolerance),
    };

    // Hard filters applied before scoring
    const maxTrades = parseEnvNumber('SCORING_MAX_TRADES', scoringParams.maxTrades);
    const inactivityDays = parseEnvNumber('SCORING_INACTIVITY_DAYS', 14);
    const inactivityThresholdMs = inactivityDays * 24 * 60 * 60 * 1000;
    const now = Date.now();

    // Pre-filter entries before scoring
    const preFiltered = entries.filter((entry) => {
      const executed = Number(entry.executedOrders ?? 0);
      const lastOpAt = Number(entry.lastOperationAt ?? 0);

      // Hard filter: remove accounts with > maxTrades
      if (executed > maxTrades) {
        return false;
      }

      // Hard filter: remove accounts inactive for > inactivityDays
      if (lastOpAt > 0 && (now - lastOpAt) > inactivityThresholdMs) {
        return false;
      }

      return true;
    });

    this.logger.info('pre_filter_stats', {
      total: entries.length,
      afterFilter: preFiltered.length,
      removedHighTrades: entries.filter(e => Number(e.executedOrders ?? 0) > maxTrades).length,
      removedInactive: entries.filter(e => {
        const lastOpAt = Number(e.lastOperationAt ?? 0);
        return lastOpAt > 0 && (now - lastOpAt) > inactivityThresholdMs;
      }).length,
    });

    const base = preFiltered
      .map((entry) => {
        const address = normalizeAddress(entry.address);
        const winRate = clamp(Number(entry.winRate ?? 0), 0, 1);
        const executed = Number(entry.executedOrders ?? 0);
        const pnl = Number(entry.realizedPnl ?? 0);
        const efficiency = executed > 0 ? pnl / executed : pnl;

        // Estimate wins/losses from win rate and trade count
        const numTrades = executed;
        const numWins = Math.round(numTrades * winRate);
        const numLosses = numTrades - numWins;

        // Get pnlList for stability score calculation
        const pnlList = entry.pnlList || [];

        // Compute score using the new stability-based formula
        const scoringResult = computePerformanceScore({
          realizedPnl: pnl,
          numTrades,
          numWins,
          numLosses,
          pnlList,
        }, scoringParams);

        return {
          address,
          score: scoringResult.score,
          filtered: scoringResult.filtered,
          filterReason: scoringResult.filterReason,
          winRate,
          executedOrders: executed,
          realizedPnl: pnl,
          efficiency,
          pnlConsistency: scoringResult.details.stabilityScore, // Use stability score as consistency metric
          remark: entry.remark ?? null,
          labels: entry.labels || [],
          statOpenPositions: null,
          statClosedPositions: null,
          statAvgPosDuration: null,
          statTotalPnl: null,
          statMaxDrawdown: scoringResult.details.maxDrawdown,
          meta: {
            ...entry,
            scoringDetails: scoringResult.details,
            numWins,
            numLosses,
            filtered: scoringResult.filtered,
            filterReason: scoringResult.filterReason,
          },
        };
      })
      .filter((entry) => Number.isFinite(entry.score) || entry.filtered);

    // Filter out accounts that failed hard filters (MDD > 80% or scalping)
    let scored = base.filter((entry) => !entry.filtered);

    // Also filter out suspicious 100% win rates with many trades (already penalized in scoring, but extra filter)
    scored = scored.filter((entry) => entry.winRate < 0.999 || entry.executedOrders < 10);

    // Log filtering stats
    const filteredCount = base.filter(e => e.filtered).length;
    if (filteredCount > 0) {
      this.logger.info('entries_filtered', {
        total: base.length,
        filtered: filteredCount,
        notProfitable: base.filter(e => e.filterReason === 'not_profitable').length,
        insufficientData: base.filter(e => e.filterReason === 'insufficient_data').length,
      });
    }

    // Fallback to unfiltered list if everything got filtered
    if (!scored.length) {
      this.logger.warn('all_entries_filtered', { total: base.length, filteredCount });
      scored = base;
    }

    // Sort by score descending
    scored.sort((a, b) => b.score - a.score);

    // Normalize weights for top selectCount entries
    const totalScore = scored.slice(0, this.opts.selectCount).reduce((sum, e) => sum + Math.max(e.score, 0), 0) || 1;
    return scored.map((entry, idx) => ({
      ...entry,
      rank: idx + 1,
      weight: idx < this.opts.selectCount ? Math.max(entry.score, 0) / totalScore : 0,
    }));
  }

  private async applyAddressStats(period: number, entries: RankedEntry[]): Promise<void> {
    const statsMap = await this.fetchAddressStatsBatch(period, entries);
    for (const entry of entries) {
      const stat = statsMap.get(entry.address.toLowerCase());
      if (stat) {
        this.applyStatsToEntry(entry, stat);
      }
    }
  }

  private applyStatsToEntry(entry: RankedEntry, stats: AddressStats): void {
    if (typeof stats.winRate === 'number' && Number.isFinite(stats.winRate)) {
      entry.winRate = clamp(stats.winRate, 0, 1);
    }
    entry.statOpenPositions = Number.isFinite(Number(stats.openPosCount))
      ? Number(stats.openPosCount)
      : entry.statOpenPositions;
    entry.statClosedPositions = Number.isFinite(Number(stats.closePosCount))
      ? Number(stats.closePosCount)
      : entry.statClosedPositions;
    entry.statAvgPosDuration = Number.isFinite(Number(stats.avgPosDuration))
      ? Number(stats.avgPosDuration)
      : entry.statAvgPosDuration;
    entry.statTotalPnl = Number.isFinite(Number(stats.totalPnl))
      ? Number(stats.totalPnl)
      : entry.statTotalPnl;
    entry.statMaxDrawdown = Number.isFinite(Number(stats.maxDrawdown))
      ? Number(stats.maxDrawdown)
      : entry.statMaxDrawdown;
    entry.meta = {
      ...entry.meta,
      stats,
    };
  }

  private async fetchAddressStatsBatch(period: number, entries: RankedEntry[]): Promise<Map<string, AddressStats>> {
    const map = new Map<string, AddressStats>();
    const addresses = entries.map((entry) => entry.address);
    await runWithConcurrency(addresses, this.statsConcurrency, async (address) => {
      const stats = await this.fetchAddressStat(address, period);
      if (stats) map.set(address.toLowerCase(), stats);
    });
    return map;
  }

  private async fetchAddressStat(address: string, period: number): Promise<AddressStats | null> {
    const normalized = normalizeAddress(address);
    const cacheKey = `${normalized}:${period}`;

    // Check cache first
    const cached = addressStatsCache.get(cacheKey);
    if (cached) {
      this.logger.debug?.('addr_stat_cache_hit', { address: normalized, period });
      return cached;
    }

    // Rate limit before making request
    await hyperbotRateLimiter.acquire();

    const url = `${this.smartApiBase}query-addr-stat/${encodeURIComponent(normalized)}?period=${period}`;
    try {
      const payload = await this.requestJson<any>(url, { method: 'GET' }, 2, 6000);
      const data = payload?.data;
      if (!data) return null;
      const stats: AddressStats = {
        winRate: typeof data.winRate === 'number' ? data.winRate : undefined,
        openPosCount: Number.isFinite(Number(data.openPosCount)) ? Number(data.openPosCount) : undefined,
        closePosCount: Number.isFinite(Number(data.closePosCount)) ? Number(data.closePosCount) : undefined,
        avgPosDuration: Number.isFinite(Number(data.avgPosDuration)) ? Number(data.avgPosDuration) : undefined,
        totalPnl: Number.isFinite(Number(data.totalPnl)) ? Number(data.totalPnl) : undefined,
        maxDrawdown: Number.isFinite(Number(data.maxDrawdown)) ? Number(data.maxDrawdown) : undefined,
      };
      // Cache the result
      addressStatsCache.set(cacheKey, stats);
      return stats;
    } catch (err: any) {
      this.logger.warn('addr_stat_fetch_failed', { address: normalized, period, err: err?.message });
      return null;
    }
  }

  /**
   * Fetch address remarks (nicknames) from Hyperbot batch API.
   * Much faster than fetching the entire leaderboard.
   * This is already a batch API, so it's naturally efficient.
   *
   * API: POST https://hyperbot.network/api/leaderboard/smart/query-addr-remarks
   * Payload: ["0xaddress1", "0xaddress2", ...]
   * Response: { code: 0, msg: "SUCCESS", data: [{ address: "0x...", remark: "nickname" }, ...] }
   */
  async fetchAddressRemarks(addresses: string[]): Promise<Map<string, string>> {
    const result = new Map<string, string>();
    if (!addresses.length) return result;

    const normalizedAddresses = addresses.map(a => normalizeAddress(a));

    // Rate limit before making request
    await hyperbotRateLimiter.acquire();

    try {
      const payload = await this.requestJson<{
        code: number;
        msg: string;
        data?: Array<{ address: string; remark?: string | null }>;
      }>(
        HYPERBOT_REMARKS_URL,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(normalizedAddresses),
        },
        2,
        8000
      );

      if (payload?.code === 0 && Array.isArray(payload.data)) {
        for (const item of payload.data) {
          if (item.remark && typeof item.remark === 'string' && item.remark.trim()) {
            result.set(normalizeAddress(item.address), item.remark.trim());
          }
        }
      }

      this.logger.info('address_remarks_fetched', {
        requested: addresses.length,
        found: result.size,
      });
    } catch (err: any) {
      this.logger.warn('address_remarks_fetch_failed', { err: err?.message });
    }

    return result;
  }

  private async fetchPortfolioSeriesBatch(
    period: number,
    entries: RankedEntry[]
  ): Promise<Map<string, PortfolioWindowSeries[]>> {
    const map = new Map<string, PortfolioWindowSeries[]>();
    await runWithConcurrency(entries, this.seriesConcurrency, async (entry) => {
      const series = await this.fetchPortfolioSeries(entry.address);
      if (!series?.length) return;
      const relevant = series.filter((window) => WINDOW_PERIOD_MAP[window.window] === period);
      if (!relevant.length) return;
      map.set(entry.address.toLowerCase(), relevant);
    });
    return map;
  }

  private async fetchPortfolioSeries(address: string): Promise<PortfolioWindowSeries[] | null> {
    const normalized = normalizeAddress(address);
    try {
      const payload = await this.requestJson<any>(
        HYPERLIQUID_INFO_URL,
        {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ type: 'portfolio', user: normalized }),
        },
        1,
        8000
      );
      const rows: any[] = Array.isArray(payload) ? payload : Array.isArray(payload?.value) ? payload.value : [];
      const series: PortfolioWindowSeries[] = [];
      for (const row of rows) {
        if (!Array.isArray(row) || row.length < 2) continue;
        const windowName = String(row[0] ?? '');
        const data = row[1] || {};
        const pnlHistory = parseHistoryPoints(data.pnlHistory);
        const equityHistory = parseHistoryPoints(data.accountValueHistory);
        if (!pnlHistory.length && !equityHistory.length) continue;
        series.push({ window: windowName, pnlHistory, equityHistory });
      }
      return series;
    } catch (err: any) {
      this.logger.warn('portfolio_fetch_failed', { address: normalized, err: err?.message });
      return null;
    }
  }

  /**
   * Check database cache for BTC/ETH analysis (30-day TTL)
   */
  private async getBtcEthFromDbCache(address: string): Promise<BtcEthAnalysis | null> {
    try {
      const pool = await getPool();
      const { rows } = await pool.query(
        `SELECT btc_pnl, eth_pnl, btc_eth_pnl, total_pnl, btc_eth_ratio, qualified, reason
         FROM hl_btc_eth_analysis_cache
         WHERE address = $1 AND expires_at > NOW()`,
        [address]
      );
      if (rows.length > 0) {
        const row = rows[0];
        return {
          btcPnl: Number(row.btc_pnl),
          ethPnl: Number(row.eth_pnl),
          btcEthPnl: Number(row.btc_eth_pnl),
          totalPnl: Number(row.total_pnl),
          btcEthRatio: Number(row.btc_eth_ratio),
          qualified: Boolean(row.qualified),
          reason: row.reason || undefined,
        };
      }
    } catch (err: any) {
      this.logger.debug?.('btc_eth_db_cache_error', { address, err: err?.message });
    }
    return null;
  }

  /**
   * Store BTC/ETH analysis in database cache (30-day TTL)
   */
  private async saveBtcEthToDbCache(address: string, analysis: BtcEthAnalysis): Promise<void> {
    try {
      const pool = await getPool();
      await pool.query(
        `INSERT INTO hl_btc_eth_analysis_cache
         (address, btc_pnl, eth_pnl, btc_eth_pnl, total_pnl, btc_eth_ratio, qualified, reason, fetched_at, expires_at)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW(), NOW() + INTERVAL '${BTC_ETH_DB_CACHE_TTL_DAYS} days')
         ON CONFLICT (address) DO UPDATE SET
           btc_pnl = EXCLUDED.btc_pnl,
           eth_pnl = EXCLUDED.eth_pnl,
           btc_eth_pnl = EXCLUDED.btc_eth_pnl,
           total_pnl = EXCLUDED.total_pnl,
           btc_eth_ratio = EXCLUDED.btc_eth_ratio,
           qualified = EXCLUDED.qualified,
           reason = EXCLUDED.reason,
           fetched_at = NOW(),
           expires_at = NOW() + INTERVAL '${BTC_ETH_DB_CACHE_TTL_DAYS} days'`,
        [
          address,
          analysis.btcPnl,
          analysis.ethPnl,
          analysis.btcEthPnl,
          analysis.totalPnl,
          analysis.btcEthRatio,
          analysis.qualified,
          analysis.reason || null,
        ]
      );
    } catch (err: any) {
      this.logger.debug?.('btc_eth_db_cache_save_error', { address, err: err?.message });
    }
  }

  /**
   * Fetch BTC/ETH trades with retry logic for rate limiting (429 errors)
   */
  private async fetchBtcEthTradesWithRetry(
    url: string,
    address: string
  ): Promise<CompletedTrade[]> {
    let lastError: Error | null = null;

    for (let attempt = 0; attempt < MAX_RATE_LIMIT_RETRIES; attempt++) {
      // Apply rate limiting delay (increases with each retry)
      const delay = attempt === 0 ? BTC_ETH_REQUEST_DELAY_MS : RATE_LIMIT_BACKOFF_BASE_MS * Math.pow(2, attempt - 1);
      await sleep(delay);

      try {
        const payload = await this.requestJson<any>(url, { method: 'GET' }, 1, 10000);
        return Array.isArray(payload?.data) ? payload.data : [];
      } catch (err: any) {
        lastError = err;
        const isRateLimit = err?.message?.includes('429') || err?.message?.includes('rate');

        if (isRateLimit && attempt < MAX_RATE_LIMIT_RETRIES - 1) {
          const nextDelay = RATE_LIMIT_BACKOFF_BASE_MS * Math.pow(2, attempt);
          this.logger.warn('btc_eth_rate_limited', {
            address,
            attempt: attempt + 1,
            maxRetries: MAX_RATE_LIMIT_RETRIES,
            nextDelayMs: nextDelay,
            err: err?.message,
          });
          continue;
        }

        // Non-rate-limit error or exhausted retries
        break;
      }
    }

    // All retries exhausted
    throw lastError || new Error('Failed to fetch BTC/ETH trades');
  }

  /**
   * Fetch and analyze BTC/ETH trades for an address.
   * Returns analysis of BTC+ETH PnL contribution to total PnL.
   * Only considers trades lasting at least 10 minutes (excludes scalping).
   *
   * Caching strategy:
   * 1. In-memory LRU cache (1 hour TTL) for hot data
   * 2. Database cache (30 day TTL) for persistent storage
   * 3. Retry with exponential backoff for rate limiting (429 errors)
   *
   * API: GET https://hyperbot.network/api/leaderboard/smart/completed-trades/{address}?take=2000
   */
  private async analyzeBtcEthTrades(address: string, totalPnl: number): Promise<BtcEthAnalysis> {
    const normalized = normalizeAddress(address);

    // Helper to recalculate qualification with current totalPnl
    const recalculateQualification = (cached: BtcEthAnalysis): BtcEthAnalysis => {
      const btcEthRatio = totalPnl !== 0 ? Math.abs(cached.btcEthPnl / totalPnl) : 0;
      const qualified = cached.btcEthPnl >= 0 && btcEthRatio >= 0.10;
      return {
        ...cached,
        totalPnl,
        btcEthRatio,
        qualified,
        reason: !qualified ? (cached.btcEthPnl < 0 ? 'btc_eth_negative_pnl' : 'btc_eth_insufficient_contribution') : undefined,
      };
    };

    // 1. Check in-memory cache first (fastest)
    const memCached = btcEthAnalysisCache.get(normalized);
    if (memCached) {
      this.logger.debug?.('btc_eth_mem_cache_hit', { address: normalized });
      return recalculateQualification(memCached);
    }

    // 2. Check database cache (30-day TTL)
    const dbCached = await this.getBtcEthFromDbCache(normalized);
    if (dbCached) {
      this.logger.debug?.('btc_eth_db_cache_hit', { address: normalized });
      // Populate in-memory cache
      btcEthAnalysisCache.set(normalized, dbCached);
      return recalculateQualification(dbCached);
    }

    // 3. Fetch from API with retry logic
    const url = `${this.smartApiBase}completed-trades/${encodeURIComponent(normalized)}?take=2000`;
    const MIN_TRADE_DURATION_MS = 10 * 60 * 1000; // 10 minutes

    try {
      const trades = await this.fetchBtcEthTradesWithRetry(url, normalized);

      let btcPnl = 0;
      let ethPnl = 0;

      for (const trade of trades) {
        const coin = String(trade.coin || '').toUpperCase();
        const pnl = Number(trade.pnl || 0);

        // Skip trades that are not BTC or ETH
        if (coin !== 'BTC' && coin !== 'ETH') {
          continue;
        }

        // Calculate trade duration
        const startTime = new Date(trade.startTime).getTime();
        const endTime = new Date(trade.endTime).getTime();
        const durationMs = endTime - startTime;

        // Skip scalping trades (< 10 minutes)
        if (durationMs < MIN_TRADE_DURATION_MS) {
          continue;
        }

        if (coin === 'BTC') {
          btcPnl += pnl;
        } else if (coin === 'ETH') {
          ethPnl += pnl;
        }
      }

      const btcEthPnl = btcPnl + ethPnl;
      const btcEthRatio = totalPnl !== 0 ? Math.abs(btcEthPnl / totalPnl) : 0;

      // Qualification rules:
      // 1. If BTC+ETH combined PnL < 0, reject
      // 2. If BTC+ETH contribution < 10% of total PnL, reject
      let qualified = true;
      let reason: string | undefined;

      if (btcEthPnl < 0) {
        qualified = false;
        reason = 'btc_eth_negative_pnl';
      } else if (btcEthRatio < 0.10) {
        qualified = false;
        reason = 'btc_eth_insufficient_contribution';
      }

      const result: BtcEthAnalysis = {
        btcPnl,
        ethPnl,
        totalPnl,
        btcEthPnl,
        btcEthRatio,
        qualified,
        reason,
      };

      // Cache in memory
      btcEthAnalysisCache.set(normalized, result);

      // Cache in database (30-day TTL) - fire and forget
      this.saveBtcEthToDbCache(normalized, result).catch(() => {});

      this.logger.info('btc_eth_analysis_complete', {
        address: normalized,
        btcPnl: btcPnl.toFixed(2),
        ethPnl: ethPnl.toFixed(2),
        btcEthRatio: (btcEthRatio * 100).toFixed(2) + '%',
        qualified,
        reason,
      });

      return result;
    } catch (err: any) {
      const isRateLimit = err?.message?.includes('429') || err?.message?.includes('rate');

      if (isRateLimit) {
        this.logger.error('btc_eth_analysis_rate_limit_exhausted', {
          address: normalized,
          err: err?.message,
          suggestion: 'Consider increasing BTC_ETH_REQUEST_DELAY_MS or RATE_LIMIT_BACKOFF_BASE_MS',
        });
      } else {
        this.logger.warn('btc_eth_analysis_failed', { address: normalized, err: err?.message });
      }

      // On error, assume not qualified to be conservative
      return {
        btcPnl: 0,
        ethPnl: 0,
        totalPnl,
        btcEthPnl: 0,
        btcEthRatio: 0,
        qualified: false,
        reason: isRateLimit ? 'rate_limit_exhausted' : 'api_fetch_failed',
      };
    }
  }

  /**
   * Filter ranked entries to select top qualified candidates based on BTC/ETH trading.
   * Continues checking candidates until we have the target number of qualified accounts.
   *
   * Rules:
   * 1. If BTC+ETH combined PnL < 0, skip account
   * 2. If BTC+ETH contribution < 10% of total PnL, skip account
   * 3. Continue until we have targetCount qualified accounts
   */
  private async filterBtcEthQualified(ranked: RankedEntry[], targetCount: number): Promise<RankedEntry[]> {
    const qualified: RankedEntry[] = [];
    const maxCandidates = Math.min(ranked.length, targetCount * 5); // Check up to 5x target to find qualified

    for (let i = 0; i < maxCandidates && qualified.length < targetCount; i++) {
      const entry = ranked[i];

      // Skip custom accounts - they bypass this filter
      const isCustom = entry.meta?.custom === true || entry.meta?.custom === 'true';
      if (isCustom) {
        qualified.push(entry);
        continue;
      }

      // Use the entry's totalPnl from stats, or fall back to realizedPnl
      const totalPnl = entry.statTotalPnl ?? entry.realizedPnl;

      const analysis = await this.analyzeBtcEthTrades(entry.address, totalPnl);

      if (analysis.qualified) {
        // Add BTC/ETH analysis to entry metadata
        entry.meta = {
          ...entry.meta,
          btcEthAnalysis: {
            btcPnl: analysis.btcPnl,
            ethPnl: analysis.ethPnl,
            btcEthPnl: analysis.btcEthPnl,
            btcEthRatio: analysis.btcEthRatio,
            qualified: true,
          },
        };
        qualified.push(entry);
        this.logger.info('btc_eth_qualified', {
          address: entry.address,
          rank: entry.rank,
          btcEthRatio: analysis.btcEthRatio.toFixed(2),
          btcPnl: analysis.btcPnl.toFixed(2),
          ethPnl: analysis.ethPnl.toFixed(2),
        });
      } else {
        this.logger.info('btc_eth_filtered', {
          address: entry.address,
          rank: entry.rank,
          reason: analysis.reason,
          btcEthRatio: analysis.btcEthRatio.toFixed(2),
          btcPnl: analysis.btcPnl.toFixed(2),
          ethPnl: analysis.ethPnl.toFixed(2),
        });
      }
    }

    if (qualified.length < targetCount) {
      this.logger.warn('btc_eth_insufficient_qualified', {
        qualified: qualified.length,
        target: targetCount,
        checked: Math.min(maxCandidates, ranked.length),
      });
    }

    return qualified;
  }

  private async requestJson<T>(
    input: string,
    init: RequestInit,
    retries = 2,
    timeoutMs = 8000
  ): Promise<T> {
    let attempt = 0;
    let lastErr: any;
    while (attempt <= retries) {
      try {
        const res = await fetch(input, {
          ...init,
          signal: AbortSignal.timeout(timeoutMs),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return (await res.json()) as T;
      } catch (err) {
        lastErr = err;
        attempt += 1;
        if (attempt > retries) break;
        await sleep(200 * attempt);
      }
    }
    throw lastErr;
  }

  private async persistPeriod(
    period: number,
    entries: RankedEntry[],
    tracked: RankedEntry[],
    hyperliquidSeries: Map<string, PortfolioWindowSeries[]>
  ): Promise<boolean> {
    // Guard: Do NOT delete existing data if new fetch returned empty/error
    // This prevents wiping the leaderboard when upstream API fails
    if (!entries.length) {
      return false;
    }

    const pool = await getPool();
    const client = await pool.connect();
    try {
      await client.query('BEGIN');
      // Safe to delete now - we have valid new data to replace it with
      // Preserve pinned account entries by excluding addresses in hl_pinned_accounts
      await client.query(
        `DELETE FROM hl_leaderboard_entries
         WHERE period_days = $1
         AND lower(address) NOT IN (SELECT lower(address) FROM hl_pinned_accounts)`,
        [period]
      );
      await client.query(
        `DELETE FROM hl_leaderboard_pnl_points
         WHERE period_days = $1
         AND lower(address) NOT IN (SELECT lower(address) FROM hl_pinned_accounts)`,
        [period]
      );
    const chunkSize = 100;
    for (let i = 0; i < entries.length; i += chunkSize) {
      const chunk = entries.slice(i, i + chunkSize);
      const values = chunk.flatMap((entry) => [
        period,
        entry.address,
        entry.rank,
        entry.score,
        entry.weight,
        entry.winRate,
        entry.executedOrders,
        entry.realizedPnl,
        entry.pnlConsistency,
        entry.efficiency,
        entry.remark,
        JSON.stringify(entry.labels || []),
        JSON.stringify({
          raw: entry.meta,
          stats: {
            openPositions: entry.statOpenPositions,
            closedPositions: entry.statClosedPositions,
            avgPositionDuration: entry.statAvgPosDuration,
            totalPnl: entry.statTotalPnl,
            maxDrawdown: entry.statMaxDrawdown,
          },
        }),
        entry.statOpenPositions,
        entry.statClosedPositions,
        entry.statAvgPosDuration,
        entry.statTotalPnl,
        entry.statMaxDrawdown,
      ]);
      const placeholders = chunk
        .map(
          (_entry, idx) =>
            `($${idx * 18 + 1}, $${idx * 18 + 2}, $${idx * 18 + 3}, $${idx * 18 + 4}, $${idx * 18 + 5}, $${idx * 18 + 6}, $${idx * 18 + 7}, $${idx * 18 + 8}, $${idx * 18 + 9}, $${idx * 18 + 10}, $${idx * 18 + 11}, $${idx * 18 + 12}, $${idx * 18 + 13}, $${idx * 18 + 14}, $${idx * 18 + 15}, $${idx * 18 + 16}, $${idx * 18 + 17}, $${idx * 18 + 18})`
        )
        .join(',');
      await client.query(
        `
        INSERT INTO hl_leaderboard_entries (
          period_days, address, rank, score, weight, win_rate, executed_orders,
          realized_pnl, pnl_consistency, efficiency, remark, labels, metrics,
          stat_open_positions, stat_closed_positions, stat_avg_pos_duration, stat_total_pnl, stat_max_drawdown
        )
        VALUES ${placeholders}
        ON CONFLICT (period_days, lower(address)) DO UPDATE SET
          rank = EXCLUDED.rank,
          score = EXCLUDED.score,
          weight = EXCLUDED.weight,
          win_rate = EXCLUDED.win_rate,
          executed_orders = EXCLUDED.executed_orders,
          realized_pnl = EXCLUDED.realized_pnl,
          pnl_consistency = EXCLUDED.pnl_consistency,
          efficiency = EXCLUDED.efficiency,
          remark = EXCLUDED.remark,
          labels = EXCLUDED.labels,
          metrics = EXCLUDED.metrics,
          stat_open_positions = EXCLUDED.stat_open_positions,
          stat_closed_positions = EXCLUDED.stat_closed_positions,
          stat_avg_pos_duration = EXCLUDED.stat_avg_pos_duration,
          stat_total_pnl = EXCLUDED.stat_total_pnl,
          stat_max_drawdown = EXCLUDED.stat_max_drawdown,
          fetched_at = now()
      `,
        values
      );
    }
    if (tracked.length) {
      await this.insertPnlPointsInTransaction(client, period, tracked, hyperliquidSeries);
    }
      await client.query('COMMIT');
      return true;
    } catch (err) {
      await client.query('ROLLBACK');
      this.logger.error('leaderboard_persist_failed', { period, err: err instanceof Error ? err.message : String(err) });
      throw err;
    } finally {
      client.release();
    }
  }

  private async insertPnlPointsInTransaction(
    client: any,
    period: number,
    tracked: RankedEntry[],
    hyperliquidSeries: Map<string, PortfolioWindowSeries[]>
  ): Promise<void> {
    const points: Array<{
      address: string;
      source: string;
      window: string;
      ts: Date;
      pnlValue: number | null;
      equityValue: number | null;
    }> = [];

    for (const entry of tracked) {
      const pnlList = Array.isArray(entry.meta?.pnlList) ? entry.meta.pnlList : [];
      for (const point of pnlList) {
        const tsValue = Number(point?.timestamp);
        const pnlValue = Number(point?.value);
        if (!Number.isFinite(tsValue)) continue;
        const date = new Date(tsValue);
        if (Number.isNaN(date.getTime())) continue;
        points.push({
          address: entry.address.toLowerCase(),
          source: 'hyperbot',
          window: `period_${period}`,
          ts: date,
          pnlValue: Number.isFinite(pnlValue) ? pnlValue : null,
          equityValue: null,
        });
      }
    }

    for (const [address, seriesList] of hyperliquidSeries.entries()) {
      for (const series of seriesList) {
        if (WINDOW_PERIOD_MAP[series.window] !== period) continue;
        const merged = mergeHistories(series);
        for (const sample of merged) {
          const tsDate = new Date(sample.ts);
          if (Number.isNaN(tsDate.getTime())) continue;
          points.push({
            address,
            source: 'hyperliquid',
            window: series.window,
            ts: tsDate,
            pnlValue: sample.pnl ?? null,
            equityValue: sample.equity ?? null,
          });
        }
      }
    }

    if (!points.length) return;

    const chunkSize = 400;
    for (let i = 0; i < points.length; i += chunkSize) {
      const chunk = points.slice(i, i + chunkSize);
      const values = chunk.flatMap((point) => [
        period,
        point.address,
        point.source,
        point.window,
        point.ts.toISOString(),
        point.pnlValue,
        point.equityValue,
      ]);
      const placeholders = chunk
        .map(
          (_point, idx) =>
            `($${idx * 7 + 1}, $${idx * 7 + 2}, $${idx * 7 + 3}, $${idx * 7 + 4}, $${idx * 7 + 5}, $${idx * 7 + 6}, $${idx * 7 + 7})`
        )
        .join(',');
      await client.query(
        `
          INSERT INTO hl_leaderboard_pnl_points (
            period_days, address, source, window_name, point_ts, pnl_value, equity_value
          )
          VALUES ${placeholders}
        `,
        values
      );
    }
  }

  private async publishTopCandidates(period: number, entries: RankedEntry[]): Promise<void> {
    for (const entry of entries) {
      const candidate = CandidateEventSchema.parse({
        address: entry.address,
        source: 'daily',
        ts: nowIso(),
        tags: [`period:${period}`, 'leaderboard'],
        nickname: entry.remark || undefined,
        score_hint: entry.score,
        meta: {
          leaderboard: {
            period_days: period,
            rank: entry.rank,
            weight: entry.weight,
            score: entry.score,
            winRate: entry.winRate,
            executedOrders: entry.executedOrders,
            realizedPnl: entry.realizedPnl,
            pnlConsistency: entry.pnlConsistency,
            efficiency: entry.efficiency,
            labels: entry.labels,
          },
        },
      });
      await this.publishCandidate(candidate).catch((err) =>
        this.logger.error('publish_candidate_failed', { address: entry.address, err: err?.message })
      );
    }
  }

  async getEntries(period: number, limit = this.opts.selectCount): Promise<RankedEntry[]> {
    const pool = await getPool();
    const { rows } = await pool.query(
      `
        SELECT
          address,
          rank,
          score,
          weight,
          coalesce(win_rate,0) as win_rate,
          coalesce(executed_orders,0) as executed_orders,
          coalesce(realized_pnl,0) as realized_pnl,
          coalesce(pnl_consistency,0) as pnl_consistency,
          coalesce(efficiency,0) as efficiency,
          remark,
          labels,
          metrics,
          stat_open_positions,
          stat_closed_positions,
          stat_avg_pos_duration,
          stat_total_pnl,
          stat_max_drawdown
        FROM hl_leaderboard_entries
        WHERE period_days = $1
        ORDER BY rank ASC
        LIMIT $2
      `,
      [period, limit]
    );
    return rows.map((row: any) => ({
      address: row.address,
      rank: Number(row.rank),
      score: Number(row.score),
      weight: Number(row.weight),
      winRate: Number(row.win_rate),
      executedOrders: Number(row.executed_orders),
      realizedPnl: Number(row.realized_pnl),
      pnlConsistency: Number(row.pnl_consistency),
      efficiency: Number(row.efficiency),
      remark: row.remark,
      labels: Array.isArray(row.labels) ? row.labels : [],
      meta: row.metrics,
      statOpenPositions: row.stat_open_positions == null ? null : Number(row.stat_open_positions),
      statClosedPositions: row.stat_closed_positions == null ? null : Number(row.stat_closed_positions),
      statAvgPosDuration: row.stat_avg_pos_duration == null ? null : Number(row.stat_avg_pos_duration),
      statTotalPnl: row.stat_total_pnl == null ? null : Number(row.stat_total_pnl),
      statMaxDrawdown: row.stat_max_drawdown == null ? null : Number(row.stat_max_drawdown),
    }));
  }

  /**
   * Check if an address exists in system-ranked entries (non-custom)
   * Returns the entry if found, null otherwise
   */
  async isSystemRankedAccount(address: string, period?: number): Promise<RankedEntry | null> {
    const pool = await getPool();
    const targetPeriod = period ?? this.opts.periods[0] ?? 30;
    const normalized = normalizeAddress(address);

    // Only consider accounts that are BTC/ETH qualified (displayed in dashboard)
    // This matches the getSelected query criteria
    const { rows } = await pool.query(
      `
        SELECT
          address,
          rank,
          score,
          weight,
          coalesce(win_rate,0) as win_rate,
          coalesce(executed_orders,0) as executed_orders,
          coalesce(realized_pnl,0) as realized_pnl,
          coalesce(pnl_consistency,0) as pnl_consistency,
          coalesce(efficiency,0) as efficiency,
          remark,
          labels,
          metrics
        FROM hl_leaderboard_entries
        WHERE period_days = $1
          AND lower(address) = $2
          AND (metrics->'raw'->>'custom' IS NULL OR metrics->'raw'->>'custom' != 'true')
          AND (metrics->'raw'->'btcEthAnalysis'->>'qualified')::boolean IS TRUE
          AND score > 0
      `,
      [targetPeriod, normalized]
    );

    if (!rows.length) return null;

    const row = rows[0];
    return {
      address: row.address,
      rank: Number(row.rank),
      score: Number(row.score),
      weight: Number(row.weight),
      winRate: Number(row.win_rate),
      executedOrders: Number(row.executed_orders),
      realizedPnl: Number(row.realized_pnl),
      pnlConsistency: Number(row.pnl_consistency),
      efficiency: Number(row.efficiency),
      remark: row.remark,
      labels: Array.isArray(row.labels) ? row.labels : [],
      meta: row.metrics,
      statOpenPositions: null,
      statClosedPositions: null,
      statAvgPosDuration: null,
      statTotalPnl: null,
      statMaxDrawdown: null,
    };
  }

  async getSelected(period: number, limit = this.opts.selectCount): Promise<RankedEntry[]> {
    const pool = await getPool();
    // Get pinned addresses to exclude from selection
    const pinnedAddresses = await getPinnedAddressSet();
    const pinnedArray = Array.from(pinnedAddresses);

    const { rows } = await pool.query(
      `
        SELECT
          address,
          rank,
          score,
          weight,
          coalesce(win_rate,0) as win_rate,
          coalesce(executed_orders,0) as executed_orders,
          coalesce(realized_pnl,0) as realized_pnl,
          coalesce(pnl_consistency,0) as pnl_consistency,
          coalesce(efficiency,0) as efficiency,
          remark,
          labels,
          metrics,
          stat_open_positions,
          stat_closed_positions,
          stat_avg_pos_duration,
          stat_total_pnl,
          stat_max_drawdown
        FROM hl_leaderboard_entries
        WHERE period_days = $1
          AND (
            metrics->'raw'->>'custom' = 'true'
            OR (metrics->'raw'->'btcEthAnalysis'->>'qualified')::boolean IS TRUE
          )
          AND lower(address) NOT IN (SELECT unnest($3::text[]))
        ORDER BY weight DESC, rank ASC
        LIMIT $2
      `,
      [period, limit, pinnedArray]
    );
    return rows.map((row: any) => ({
      address: row.address,
      rank: Number(row.rank),
      score: Number(row.score),
      weight: Number(row.weight),
      winRate: Number(row.win_rate),
      executedOrders: Number(row.executed_orders),
      realizedPnl: Number(row.realized_pnl),
      pnlConsistency: Number(row.pnl_consistency),
      efficiency: Number(row.efficiency),
      remark: row.remark,
      labels: Array.isArray(row.labels) ? row.labels : [],
      meta: row.metrics,
      statOpenPositions: row.stat_open_positions == null ? null : Number(row.stat_open_positions),
      statClosedPositions: row.stat_closed_positions == null ? null : Number(row.stat_closed_positions),
      statAvgPosDuration: row.stat_avg_pos_duration == null ? null : Number(row.stat_avg_pos_duration),
      statTotalPnl: row.stat_total_pnl == null ? null : Number(row.stat_total_pnl),
      statMaxDrawdown: row.stat_max_drawdown == null ? null : Number(row.stat_max_drawdown),
    }));
  }

  async ensureSeeded(): Promise<void> {
    const pool = await getPool();
    for (const period of this.opts.periods) {
      const { rows } = await pool.query(
        'SELECT 1 FROM hl_leaderboard_entries WHERE period_days = $1 LIMIT 1',
        [period]
      );
      if (!rows.length) {
        this.logger.info('leaderboard_empty_seed', { period });
        await this.refreshPeriod(period);
      }
    }
  }

  /**
   * Fetch stats for a custom account and upsert into hl_leaderboard_entries.
   * This is called when a user adds a custom account so stats appear immediately.
   * Custom accounts are now properly scored using the same algorithm as system accounts.
   */
  async fetchAndStoreCustomAccountStats(address: string, nickname?: string | null): Promise<RankedEntry | null> {
    const normalized = normalizeAddress(address);
    const period = this.opts.periods[0] ?? 30; // Use first configured period (default 30d)

    this.logger.info('fetch_custom_account_stats', { address: normalized, period });

    // Fetch stats from Hyperbot API
    const stats = await this.fetchAddressStat(normalized, period);

    // Extract stats for scoring
    const rawWinRate = stats?.winRate ?? 0;
    const rawExecutedOrders = (stats?.openPosCount ?? 0) + (stats?.closePosCount ?? 0);
    const rawRealizedPnl = stats?.totalPnl ?? 0;

    // Fetch remark from Hyperbot batch API (much faster than fetching entire leaderboard)
    let apiRemark: string | null = null;
    try {
      const remarks = await this.fetchAddressRemarks([normalized]);
      apiRemark = remarks.get(normalized) ?? null;
      if (apiRemark) {
        this.logger.info('custom_account_remark_fetched', { address: normalized, apiRemark });
      }
    } catch (err: any) {
      this.logger.warn('custom_account_remark_fetch_failed', { address: normalized, err: err?.message });
    }

    // Fetch portfolio data for pnlList (PnL curve in dashboard)
    let pnlList: Array<{ timestamp: number; value: string }> = [];
    try {
      const portfolioSeries = await this.fetchPortfolioSeries(normalized);
      // Find the 30-day window (typically "month" or "30d")
      const monthSeries = portfolioSeries?.find(s => s.window === 'month' || s.window === '30d' || s.window === 'allTime');
      if (monthSeries?.pnlHistory?.length) {
        pnlList = monthSeries.pnlHistory.map(pt => ({
          timestamp: pt.ts,
          value: String(pt.value),
        }));
        this.logger.info('custom_account_pnl_history_fetched', { address: normalized, points: pnlList.length });
      }
    } catch (err: any) {
      this.logger.warn('custom_account_portfolio_fetch_failed', { address: normalized, err: err?.message });
    }

    // Compute proper score using the scoring algorithm (same as system accounts)
    // Use computeFullScore: true to get the full score even if the account would be filtered
    const numWins = Math.round(rawWinRate * rawExecutedOrders);
    const numLosses = rawExecutedOrders - numWins;

    const scoringResult = computePerformanceScore({
      realizedPnl: rawRealizedPnl,
      numTrades: rawExecutedOrders,
      numWins,
      numLosses,
      pnlList: pnlList.map(pt => ({ timestamp: pt.timestamp, value: pt.value })),
    }, DEFAULT_SCORING_PARAMS, { computeFullScore: true });

    // Build the entry with computed score
    const entry: RankedEntry = {
      address: normalized,
      rank: 9999, // Custom accounts get a high rank (sorted separately in UI)
      score: scoringResult.score,
      weight: 0, // Custom accounts don't contribute to weighted selection
      winRate: rawWinRate,
      executedOrders: rawExecutedOrders,
      realizedPnl: rawRealizedPnl,
      efficiency: rawRealizedPnl && rawExecutedOrders ? rawRealizedPnl / Math.max(1, rawExecutedOrders) : 0,
      pnlConsistency: scoringResult.details.stabilityScore,
      remark: nickname ?? apiRemark ?? null, // Use user nickname, fallback to API remark
      labels: ['custom'],
      statOpenPositions: stats?.openPosCount ?? null,
      statClosedPositions: stats?.closePosCount ?? null,
      statAvgPosDuration: stats?.avgPosDuration ?? null,
      statTotalPnl: stats?.totalPnl ?? null,
      statMaxDrawdown: scoringResult.details.maxDrawdown ?? stats?.maxDrawdown ?? null,
      meta: {
        custom: true,
        fetchedAt: nowIso(),
        pnlList, // Include pnlList for dashboard PnL curve
        scoringDetails: scoringResult.details,
        filtered: scoringResult.filtered,
        filterReason: scoringResult.filterReason,
      },
    };

    // Upsert into hl_leaderboard_entries
    const pool = await getPool();
    try {
      await pool.query(
        `
        INSERT INTO hl_leaderboard_entries (
          period_days, address, rank, score, weight, win_rate, executed_orders,
          realized_pnl, pnl_consistency, efficiency, remark, labels, metrics,
          stat_open_positions, stat_closed_positions, stat_avg_pos_duration, stat_total_pnl, stat_max_drawdown
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
        ON CONFLICT (period_days, lower(address)) DO UPDATE SET
          rank = EXCLUDED.rank,
          score = EXCLUDED.score,
          weight = EXCLUDED.weight,
          win_rate = EXCLUDED.win_rate,
          executed_orders = EXCLUDED.executed_orders,
          realized_pnl = EXCLUDED.realized_pnl,
          pnl_consistency = EXCLUDED.pnl_consistency,
          efficiency = EXCLUDED.efficiency,
          remark = COALESCE(EXCLUDED.remark, hl_leaderboard_entries.remark),
          labels = EXCLUDED.labels,
          metrics = EXCLUDED.metrics,
          stat_open_positions = EXCLUDED.stat_open_positions,
          stat_closed_positions = EXCLUDED.stat_closed_positions,
          stat_avg_pos_duration = EXCLUDED.stat_avg_pos_duration,
          stat_total_pnl = EXCLUDED.stat_total_pnl,
          stat_max_drawdown = EXCLUDED.stat_max_drawdown,
          fetched_at = now()
        `,
        [
          period,
          normalized,
          entry.rank,
          entry.score,
          entry.weight,
          entry.winRate,
          entry.executedOrders,
          entry.realizedPnl,
          entry.pnlConsistency,
          entry.efficiency,
          entry.remark,
          JSON.stringify(entry.labels),
          JSON.stringify({
            raw: entry.meta, // Match format of system entries for dashboard compatibility
            stats: {
              openPositions: entry.statOpenPositions,
              closedPositions: entry.statClosedPositions,
              avgPositionDuration: entry.statAvgPosDuration,
              totalPnl: entry.statTotalPnl,
              maxDrawdown: entry.statMaxDrawdown,
            },
          }),
          entry.statOpenPositions,
          entry.statClosedPositions,
          entry.statAvgPosDuration,
          entry.statTotalPnl,
          entry.statMaxDrawdown,
        ]
      );
      this.logger.info('custom_account_stats_stored', { address: normalized, period, stats });

      // Nickname is now stored in leaderboard_entries.remark (updated in the upsert above)
      if (!nickname && apiRemark) {
        this.logger.info('custom_account_nickname_auto_filled', { address: normalized, nickname: apiRemark });
      }

      return entry;
    } catch (err: any) {
      this.logger.error('custom_account_stats_store_failed', { address: normalized, err: err?.message });
      return null;
    }
  }
}

export default LeaderboardService;

async function runWithConcurrency<T>(
  items: T[],
  limit: number,
  worker: (item: T, index: number) => Promise<void>
): Promise<void> {
  if (!items.length) return;
  const poolSize = Math.max(1, Math.min(limit, items.length));
  let cursor = 0;
  await Promise.all(
    Array.from({ length: poolSize }).map(async () => {
      // eslint-disable-next-line no-constant-condition
      while (true) {
        const index = cursor;
        cursor += 1;
        if (index >= items.length) break;
        try {
          await worker(items[index], index);
        } catch {
          // Individual workers log their own failures; continue to next task.
        }
      }
    })
  );
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function parseHistoryPoints(list: any): Array<{ ts: number; value: number }> {
  if (!Array.isArray(list)) return [];
  const out: Array<{ ts: number; value: number }> = [];
  for (const item of list) {
    if (!Array.isArray(item) || item.length < 2) continue;
    const ts = Number(item[0]);
    const value = Number(item[1]);
    if (!Number.isFinite(ts)) continue;
    const numericValue = Number(value);
    out.push({ ts, value: Number.isFinite(numericValue) ? numericValue : 0 });
  }
  return out;
}

function mergeHistories(series: PortfolioWindowSeries): Array<{ ts: number; pnl?: number; equity?: number }> {
  const map = new Map<number, { pnl?: number; equity?: number }>();
  for (const point of series.pnlHistory) {
    const existing = map.get(point.ts) || {};
    existing.pnl = point.value;
    map.set(point.ts, existing);
  }
  for (const point of series.equityHistory) {
    const existing = map.get(point.ts) || {};
    existing.equity = point.value;
    map.set(point.ts, existing);
  }
  return Array.from(map.entries())
    .map(([ts, values]) => ({ ts, pnl: values.pnl, equity: values.equity }))
    .sort((a, b) => a.ts - b.ts);
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

// normalizeLog removed - no longer used with new scoring formula

function computeConsistency(list: Array<{ timestamp: number; value: string }>): number {
  if (!list?.length) return 0.5;
  const values = list.map((p) => Number(p.value)).filter((v) => Number.isFinite(v));
  if (values.length < 3) return 0.5;
  const first = values[0];
  const last = values[values.length - 1];
  const slope = (last - first) / Math.max(Math.abs(first) + 1, 1);
  const diffs: number[] = [];
  for (let i = 1; i < values.length; i += 1) {
    diffs.push(values[i] - values[i - 1]);
  }
  const variance = diffs.reduce((sum, d) => sum + d * d, 0) / diffs.length || 0;
  const std = Math.sqrt(variance) || 1;
  const ratio = slope / std;
  return clamp(0.5 + Math.atan(ratio) / Math.PI, 0, 1);
}
