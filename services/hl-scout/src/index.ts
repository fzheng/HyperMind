/**
 * HL-Scout Service
 *
 * Ingests and scores Hyperliquid leaderboard data to identify top-performing traders.
 * Publishes candidate events to NATS for downstream services and provides REST APIs
 * for address management, leaderboard access, and custom account tracking.
 *
 * Key responsibilities:
 * - Poll Hyperliquid leaderboard API and score traders
 * - Enrich top performers with detailed statistics
 * - Publish candidate events to `a.candidates.v1` topic
 * - Manage custom accounts for user-specified address tracking
 * - Expose REST APIs for dashboard data
 *
 * @module hl-scout
 */

import cors from 'cors';
import express, { Request, Response, NextFunction } from 'express';
import swaggerUi from 'swagger-ui-express';
import type { OpenAPIV3 } from 'openapi-types';
import {
  addAddress,
  removeAddress,
  CandidateEvent,
  CandidateEventSchema,
  createLogger,
  fetchUserFills,
  fetchUserProfile,
  fetchPerpMarkPrice,
  getOwnerToken,
  getPool,
  getPort,
  initMetrics,
  createHistogram,
  metricsHandler,
  insertTradeIfNew,
  listAddresses,
  publishJson,
  seedAddresses,
  ensureStream,
  connectNats,
  normalizeAddress,
  nowIso,
  listRecentFills,
  listRecentDecisions,
  fetchLatestFillForAddress,
  listLiveFills,
  validateEthereumAddress,
  validateAddressArray,
  sanitizeNickname,
  listPinnedAccounts,
  pinLeaderboardAccount,
  addCustomPinnedAccount,
  unpinAccount,
  getPinnedAccountCount,
  isPinnedAccount,
  getLastRefreshTime,
  runMigrations
} from '@hl/ts-lib';
import LeaderboardService, { LeaderboardSort } from './leaderboard';

const OWNER_TOKEN = getOwnerToken();
const logger = createLogger('hl-scout');
const metrics = initMetrics('hl_scout');
const candidateLatency = createHistogram(metrics, 'candidate_publish_seconds', 'Latency for publishing candidates', [0.01, 0.05, 0.1, 0.25, 0.5, 1]);
const DEFAULT_LEADERBOARD_PERIOD = Number(process.env.LEADERBOARD_DEFAULT_PERIOD ?? 30);
const LEADERBOARD_PERIODS = (process.env.LEADERBOARD_PERIODS ?? '30')
  .split(',')
  .map((v) => Number(v.trim()))
  .filter((v) => Number.isFinite(v) && v > 0);
let leaderboardService: LeaderboardService | null = null;

const swaggerDoc: OpenAPIV3.Document = {
  openapi: '3.0.0',
  info: {
    title: 'hl-scout API',
    version: '0.1.0',
    description: 'Address management, seeding, and candidate publishing service.'
  },
  servers: [{ url: '/' }],
  components: {
    securitySchemes: {
      ownerToken: {
        type: 'apiKey',
        in: 'header',
        name: 'x-owner-key'
      }
    },
    schemas: {
      AddressPayload: {
        type: 'object',
        required: ['address'],
        properties: {
          address: { type: 'string', example: '0xabc123...' },
          nickname: { type: 'string', nullable: true }
        }
      },
      SeedPayload: {
        type: 'object',
        required: ['addresses'],
        properties: {
          addresses: {
            type: 'array',
            items: { type: 'string' },
            minItems: 1
          }
        }
      },
      BackfillPayload: {
        type: 'object',
        properties: {
          limit: { type: 'integer', minimum: 1, maximum: 500, default: 50 }
        }
      }
    }
  },
  paths: {
    '/healthz': {
      get: {
        summary: 'Service health',
        responses: {
          200: { description: 'OK' }
        }
      }
    },
    '/metrics': {
      get: {
        summary: 'Prometheus metrics',
        responses: { 200: { description: 'Prometheus text payload' } }
      }
    },
    '/addresses': {
      get: {
        summary: 'List tracked addresses',
        responses: { 200: { description: 'Addresses with nicknames' } }
      },
      post: {
        summary: 'Add/update an address',
        security: [{ ownerToken: [] }],
        requestBody: {
          required: true,
          content: {
            'application/json': {
              schema: { $ref: '#/components/schemas/AddressPayload' }
            }
          }
        },
        responses: { 200: { description: 'Added' } }
      }
    },
    '/addresses/{address}': {
      delete: {
        summary: 'Remove an address',
        security: [{ ownerToken: [] }],
        parameters: [
          { name: 'address', in: 'path', required: true, schema: { type: 'string' } }
        ],
        responses: { 200: { description: 'Removed' } }
      }
    },
    '/admin/seed': {
      post: {
        summary: 'Seed multiple addresses',
        security: [{ ownerToken: [] }],
        requestBody: {
          required: true,
          content: {
            'application/json': {
              schema: { $ref: '#/components/schemas/SeedPayload' }
            }
          }
        },
        responses: { 200: { description: 'Seeded' } }
      }
    },
    '/admin/backfill/{address}': {
      post: {
        summary: 'Backfill recent fills for an address',
        security: [{ ownerToken: [] }],
        parameters: [
          { name: 'address', in: 'path', required: true, schema: { type: 'string' } }
        ],
        requestBody: {
          content: {
            'application/json': {
              schema: { $ref: '#/components/schemas/BackfillPayload' }
            }
          }
        },
        responses: { 200: { description: 'Backfill triggered' } }
      }
    },
    '/dashboard/summary': {
      get: {
        summary: 'Aggregated performance stats per address',
        responses: { 200: { description: 'Performance summary' } }
      }
    },
    '/dashboard/fills': {
      get: {
        summary: 'Recent fills feed',
        parameters: [
          { name: 'limit', in: 'query', schema: { type: 'integer', minimum: 1, maximum: 200 } }
        ],
        responses: { 200: { description: 'Recent fills' } }
      }
    },
    '/dashboard/decisions': {
      get: {
        summary: 'Recent decision tickets',
        parameters: [
          { name: 'limit', in: 'query', schema: { type: 'integer', minimum: 1, maximum: 100 } }
        ],
        responses: { 200: { description: 'Decision list' } }
      }
    },
    '/dashboard/price': {
      get: {
        summary: 'Latest price snapshot',
        parameters: [
          { name: 'symbol', in: 'query', schema: { type: 'string', enum: ['BTCUSDT', 'ETHUSDT'], default: 'BTCUSDT' } }
        ],
        responses: { 200: { description: 'Current price' } }
      }
    }
  }
};

/**
 * Parses and validates a leaderboard period value.
 * Falls back to the first configured period or default of 30 days.
 *
 * @param value - Period value from query parameter
 * @returns Valid period number in days
 */
function parsePeriod(value: any): number {
  const n = Number(value);
  if (Number.isFinite(n) && n > 0) return n;
  return LEADERBOARD_PERIODS[0] || DEFAULT_LEADERBOARD_PERIOD || 30;
}

/** Request body for adding/updating an address */
interface AddressPayload {
  /** Ethereum address to add */
  address: string;
  /** Optional display name */
  nickname?: string | null;
}

/**
 * Express middleware for owner-only endpoints.
 * Requires x-owner-key header to match OWNER_TOKEN environment variable.
 */
function ownerOnly(req: Request, res: Response, next: NextFunction) {
  const token = (req.headers['x-owner-key'] || req.headers['x-owner-token'] || '').toString();
  if (token !== OWNER_TOKEN) {
    return res.status(403).json({ error: 'forbidden' });
  }
  return next();
}

/** Default seed addresses when database is empty (from SCOUT_SEEDS env var) */
const DEFAULT_SEEDS = (process.env.SCOUT_SEEDS || '')
  .split(',')
  .map((addr) => addr.trim().toLowerCase())
  .filter((addr) => addr.startsWith('0x') && addr.length === 42);

/**
 * Publishes a candidate event to NATS JetStream.
 * Validates the candidate schema and records latency metrics.
 *
 * @param subject - NATS subject to publish to
 * @param js - JetStream client
 * @param candidate - Candidate event data
 */
async function publishCandidate(
  subject: string,
  js: Awaited<ReturnType<typeof connectNats>>['js'],
  candidate: CandidateEvent
) {
  const parsed = CandidateEventSchema.parse(candidate);
  const end = (candidateLatency as any).startTimer?.({ operation: 'publish' });
  await publishJson(js, subject, parsed);
  end?.();
  logger.info('candidate_published', { address: parsed.address, source: parsed.source });
}

/**
 * Backfills historical trades for an address from Hyperliquid API.
 * Fetches recent BTC and ETH fills and inserts them into the database.
 *
 * @param address - Ethereum address to backfill
 * @param limit - Maximum number of fills to fetch (default 50)
 * @returns Number of newly inserted trades
 */
async function backfillRecent(address: string, limit = 50): Promise<number> {
  // Fetch both BTC and ETH fills from Hyperliquid API
  const fills = await fetchUserFills(address, { aggregateByTime: true, symbols: ['BTC', 'ETH'] });
  let inserted = 0;
  for (const f of (fills || []).slice(0, limit)) {
    const delta = f.side === 'B' ? +f.sz : -f.sz;
    const newPos = f.startPosition + delta;
    let action = '';
    if (f.startPosition === 0) action = delta > 0 ? 'Open Long (Open New)' : 'Open Short (Open New)';
    else if (f.startPosition > 0) action = delta > 0 ? 'Increase Long' : (newPos === 0 ? 'Close Long (Close All)' : 'Decrease Long');
    else action = delta < 0 ? 'Increase Short' : (newPos === 0 ? 'Close Short (Close All)' : 'Decrease Short');
    const payload = {
      at: new Date(f.time).toISOString(),
      address,
      symbol: f.coin, // Use actual coin (BTC or ETH)
      action,
      size: Math.abs(f.sz),
      startPosition: f.startPosition,
      priceUsd: f.px,
      realizedPnlUsd: f.closedPnl ?? null,
      fee: (f as any).fee ?? null,
      feeToken: (f as any).feeToken ?? null,
      hash: (f as any).hash ?? null,
    };
    const result = await insertTradeIfNew(address, payload);
    if (result.inserted) inserted += 1;
  }
  return inserted;
}

/**
 * Bootstraps initial candidate events on service startup.
 * Seeds default addresses if database is empty, then publishes
 * the first 3 addresses as candidates.
 *
 * @param subject - NATS subject for candidate events
 * @param js - JetStream client
 */
async function bootstrapCandidates(subject: string, js: Awaited<ReturnType<typeof connectNats>>['js']) {
  const rows = await listAddresses();
  if (rows.length === 0 && DEFAULT_SEEDS.length) {
    await seedAddresses(DEFAULT_SEEDS);
    rows.push(...await listAddresses());
  }
  const seeds = rows.slice(0, 3);
  for (const entry of seeds) {
    await publishCandidate(subject, js, {
      address: normalizeAddress(entry.address),
      source: 'seed',
      ts: nowIso(),
      tags: ['phase1'],
      nickname: entry.nickname || null,
      meta: { seeded: true }
    });
  }
}

/**
 * Main entry point for hl-scout service.
 * Initializes database, NATS, leaderboard service, and Express server.
 */
async function main() {
  await getPool(); // ensure db connectivity
  await runMigrations(); // apply any pending migrations
  const natsUrl = process.env.NATS_URL || 'nats://localhost:4222';
  const topic = 'a.candidates.v1';
  const nats = await connectNats(natsUrl);
  await ensureStream(nats.jsm, 'HL_A', [topic]);
  await bootstrapCandidates(topic, nats.js);

  // Parse LEADERBOARD_SORT env var (default: 3 = PNL)
  const parsedSort = Number(process.env.LEADERBOARD_SORT ?? LeaderboardSort.PNL);
  const leaderboardSort = Object.values(LeaderboardSort).includes(parsedSort)
    ? (parsedSort as LeaderboardSort)
    : LeaderboardSort.PNL;

  const leaderboardOptions = {
    apiUrl: process.env.LEADERBOARD_API_URL,
    topN: Number(process.env.LEADERBOARD_TOP_N ?? 1000),
    selectCount: Number(process.env.LEADERBOARD_SELECT_COUNT ?? 12),
    periods: LEADERBOARD_PERIODS.length ? LEADERBOARD_PERIODS : [DEFAULT_LEADERBOARD_PERIOD || 30],
    pageSize: Number(process.env.LEADERBOARD_PAGE_SIZE ?? 100),
    enrichCount: Number(process.env.LEADERBOARD_ENRICH_COUNT ?? process.env.LEADERBOARD_SELECT_COUNT ?? 12),
    sort: leaderboardSort,
  };
  leaderboardService = new LeaderboardService(leaderboardOptions, async (candidate) => {
    await publishCandidate(topic, nats.js, candidate);
  });
  leaderboardService.start();
  await leaderboardService.ensureSeeded().catch((err) => {
    logger.error('leaderboard_seed_failed', { err: err?.message });
  });

  const app = express();
  app.use(cors());
  app.use(express.json());
  app.use('/docs', swaggerUi.serve, swaggerUi.setup(swaggerDoc));

  app.get('/healthz', (_req, res) => {
    res.json({ status: 'ok', service: 'hl-scout' });
  });

  app.get('/metrics', metricsHandler(metrics));

  app.get('/addresses', async (_req, res) => {
    const addresses = await listAddresses();
    res.json({ addresses });
  });

  app.post('/addresses', ownerOnly, async (req, res) => {
    try {
      const body = req.body as AddressPayload;
      if (!body?.address) return res.status(400).json({ error: 'address required' });
      const validAddress = validateEthereumAddress(body.address);
      const validNickname = sanitizeNickname(body.nickname);
      await addAddress(validAddress, validNickname);
      await publishCandidate(topic, nats.js, {
        address: validAddress,
        source: 'backfill',
        ts: nowIso(),
        nickname: validNickname,
        tags: ['api'],
        meta: { via: 'api' }
      });
      res.json({ ok: true });
    } catch (err: any) {
      logger.error('add_address_failed', { err: err?.message });
      res.status(400).json({ error: err?.message || 'validation failed' });
    }
  });

  app.delete('/addresses/:address', ownerOnly, async (req, res) => {
    await removeAddress(req.params.address);
    res.json({ ok: true });
  });

  app.post('/admin/seed', ownerOnly, async (req, res) => {
    try {
      const validAddresses = validateAddressArray(req.body?.addresses);
      await seedAddresses(validAddresses);
      for (const address of validAddresses) {
        await publishCandidate(topic, nats.js, {
          address,
          source: 'seed',
          ts: nowIso(),
          tags: ['admin-seed'],
          meta: { seeded: true }
        });
      }
      res.json({ ok: true, count: validAddresses.length });
    } catch (err: any) {
      logger.error('seed_addresses_failed', { err: err?.message });
      res.status(400).json({ error: err?.message || 'validation failed' });
    }
  });

  app.post('/admin/backfill/:address', ownerOnly, async (req, res) => {
    try {
      const address = validateEthereumAddress(req.params.address);
      const limit = Math.max(1, Math.min(500, Number(req.body?.limit) || 50));
      const inserted = await backfillRecent(address, limit);
      res.json({ ok: true, inserted });
    } catch (err: any) {
      logger.error('backfill_failed', { err: err?.message });
      res.status(400).json({ error: err?.message || 'validation failed' });
    }
  });

  app.post('/admin/leaderboard/refresh', ownerOnly, async (_req, res) => {
    if (!leaderboardService) return res.status(503).json({ error: 'leaderboard unavailable' });
    await leaderboardService.refreshAll().catch((err) => {
      logger.error('leaderboard_manual_refresh_failed', { err: err?.message });
      throw err;
    });
    res.json({ ok: true });
  });

  app.get('/leaderboard', async (req, res) => {
    if (!leaderboardService) return res.status(503).json({ error: 'leaderboard unavailable' });
    const period = parsePeriod(req.query.period);
    const limit = Number(req.query.limit ?? 100);
    const entries = await leaderboardService.getEntries(period, Math.max(1, Math.min(1000, limit)));
    res.json({ period, entries });
  });

  app.get('/leaderboard/selected', async (req, res) => {
    if (!leaderboardService) return res.status(503).json({ error: 'leaderboard unavailable' });
    const period = parsePeriod(req.query.period);
    const limit = Number(req.query.limit ?? process.env.LEADERBOARD_SELECT_COUNT ?? 12);
    const entries = await leaderboardService.getSelected(period, Math.max(1, Math.min(50, limit)));
    res.json({ period, entries });
  });

  app.get('/dashboard/summary', async (req, res) => {
    if (!leaderboardService) return res.status(503).json({ error: 'leaderboard unavailable' });
    const period = parsePeriod(req.query.period);
    const systemLimit = 10; // Always get top 10 system accounts
    const selected = await leaderboardService.getSelected(period, systemLimit);

    // Get pinned accounts (both leaderboard-pinned and custom)
    const pinnedAccounts = await listPinnedAccounts();

    // Mark system accounts (already filtered to exclude pinned in getSelected)
    const systemEntries = selected
      .slice(0, systemLimit)
      .map((entry) => ({
        ...entry,
        isPinned: false,
        isCustom: false,
      }));

    // For pinned accounts, try to get their leaderboard stats
    const pinnedEntries = await Promise.all(
      pinnedAccounts.map(async (pinned) => {
        // Try to find in leaderboard entries
        const pool = await getPool();
        const { rows } = await pool.query(
          `SELECT * FROM hl_leaderboard_entries
           WHERE period_days = $1 AND lower(address) = $2
           LIMIT 1`,
          [period, pinned.address.toLowerCase()]
        );
        if (rows.length) {
          const row = rows[0];
          return {
            address: pinned.address,
            rank: 0, // Will be re-ranked
            score: Number(row.score ?? 0),
            weight: 0,
            winRate: Number(row.win_rate ?? 0),
            executedOrders: Number(row.executed_orders ?? 0),
            realizedPnl: Number(row.realized_pnl ?? 0),
            efficiency: Number(row.efficiency ?? 0),
            pnlConsistency: Number(row.pnl_consistency ?? 0),
            remark: row.remark || null,
            labels: row.labels || [],
            statOpenPositions: row.stat_open_positions,
            statClosedPositions: row.stat_closed_positions,
            statAvgPosDuration: row.stat_avg_pos_duration,
            statTotalPnl: row.stat_total_pnl,
            statMaxDrawdown: row.stat_max_drawdown,
            meta: row.metrics || {},
            isPinned: true,
            isCustom: pinned.isCustom,
          };
        }
        // No leaderboard entry found - return minimal data
        return {
          address: pinned.address,
          rank: 0,
          score: 0,
          weight: 0,
          winRate: 0,
          executedOrders: 0,
          realizedPnl: 0,
          efficiency: 0,
          pnlConsistency: 0,
          remark: null,
          labels: [],
          statOpenPositions: null,
          statClosedPositions: null,
          statAvgPosDuration: null,
          statTotalPnl: null,
          statMaxDrawdown: null,
          meta: {},
          isPinned: true,
          isCustom: pinned.isCustom,
        };
      })
    );

    // Merge and re-rank all entries by score
    const allEntries = [...systemEntries, ...pinnedEntries];
    allEntries.sort((a, b) => b.score - a.score);
    allEntries.forEach((entry, idx) => {
      entry.rank = idx + 1;
    });

    const stats = allEntries;
    const top = allEntries[0] ?? null;
    const featured = top ? await fetchLatestFillForAddress(top.address) : null;

    // Get profiles for top entries
    const profileEntries = await Promise.all(
      stats.slice(0, 5).map(async (row) => ({
        address: row.address,
        profile: await fetchUserProfile(row.address),
      }))
    );
    const profiles: Record<string, unknown> = {};
    for (const entry of profileEntries) {
      profiles[entry.address] = entry.profile;
    }

    // Get holdings for all accounts (BTC and ETH positions) with entry and liquidation prices
    const holdings: Record<string, Array<{
      symbol: string;
      size: number;
      entryPrice: number | null;
      liquidationPrice: number | null;
      leverage: number | null;
    }>> = {};
    if (allEntries.length) {
      const pool = await getPool();
      const { rows } = await pool.query(
        `select address, symbol, size, entry_price, liquidation_price, leverage
         from hl_current_positions
         where lower(address) = any($1)
         and symbol in ('BTC', 'ETH')
         and abs(size) >= 0.0001
         order by address, symbol`,
        [allEntries.map((s) => s.address.toLowerCase())]
      );
      for (const row of rows) {
        const addr = String(row.address || '').toLowerCase();
        if (!addr) continue;
        if (!holdings[addr]) holdings[addr] = [];
        holdings[addr].push({
          symbol: String(row.symbol || 'BTC').toUpperCase(),
          size: Number(row.size || 0),
          entryPrice: row.entry_price != null ? Number(row.entry_price) : null,
          liquidationPrice: row.liquidation_price != null ? Number(row.liquidation_price) : null,
          leverage: row.leverage != null ? Number(row.leverage) : null,
        });
      }
    }

    // Get last refresh timestamp and live status
    const lastRefresh = await getLastRefreshTime(period);
    const refreshStatus = leaderboardService?.getRefreshStatus() || null;

    res.json({
      period,
      stats,
      selected: allEntries,
      featured,
      holdings,
      lastRefresh,
      lastRefreshFormatted: lastRefresh ? new Date(lastRefresh).toISOString() : null,
      // Include refresh status for dashboard UI
      refreshStatus: refreshStatus ? {
        status: refreshStatus.status,
        isRefreshing: refreshStatus.isRefreshing,
        nextRefreshAt: refreshStatus.nextRefreshAt,
        nextRefreshInMs: refreshStatus.nextRefreshInMs,
        refreshIntervalMs: refreshStatus.refreshIntervalMs,
        progress: refreshStatus.progress,
      } : null,
      pinnedAccountCount: pinnedAccounts.length,
      customPinnedCount: pinnedAccounts.filter(p => p.isCustom).length,
      maxCustomPinned: 3,
      recommendation: top
        ? {
            address: top.address,
            winRate: top.winRate,
            realizedPnl: top.realizedPnl,
            weight: top.weight,
            rank: top.rank,
            isCustom: top.isCustom,
            message: `Route new fills for ${top.address} (rank #${top.rank})`,
          }
        : null,
      profiles,
    });
  });

  app.get('/dashboard/fills', async (req, res) => {
    const limit = Math.max(1, Math.min(200, Number(req.query.limit) || 25));
    // Fetch one extra to check if there are more
    const fills = await listLiveFills(limit + 1);
    const hasMore = fills.length > limit;
    const trimmedFills = fills.slice(0, limit);
    res.json({ fills: trimmedFills, hasMore });
  });

  app.get('/dashboard/decisions', async (req, res) => {
    const limit = Number(req.query.limit) || 20;
    const decisions = await listRecentDecisions(limit);
    res.json({ decisions });
  });

  app.get('/dashboard/price', async (req, res) => {
    try {
      const symbol = (req.query.symbol === 'ETHUSDT' ? 'ETH' : 'BTC') as 'BTC' | 'ETH';
      const price = await fetchPerpMarkPrice(symbol);
      res.json({ symbol: `${symbol}USDT`, price, ts: new Date().toISOString() });
    } catch (err: any) {
      logger.error('price_fetch_failed', { err: err?.message });
      res.status(502).json({ error: 'price_fetch_failed' });
    }
  });

  // =====================
  // Pinned Accounts API
  // =====================

  // Get all pinned accounts
  app.get('/pinned-accounts', async (_req, res) => {
    try {
      const accounts = await listPinnedAccounts();
      const customCount = accounts.filter(a => a.isCustom).length;
      res.json({
        accounts,
        count: accounts.length,
        customCount,
        maxCustomAllowed: 3,
      });
    } catch (err: any) {
      logger.error('pinned_accounts_list_failed', { err: err?.message });
      res.status(500).json({ error: 'Failed to list pinned accounts' });
    }
  });

  // Pin an account from leaderboard (unlimited)
  app.post('/pinned-accounts/leaderboard', async (req, res) => {
    try {
      const { address } = req.body;
      if (!address || typeof address !== 'string') {
        res.status(400).json({ error: 'Address is required' });
        return;
      }

      // Validate Ethereum address
      try {
        validateEthereumAddress(address);
      } catch (e: any) {
        res.status(400).json({ error: e.message || 'Invalid Ethereum address' });
        return;
      }

      // Check if already pinned
      const existingPinned = await isPinnedAccount(address);
      if (existingPinned) {
        res.status(400).json({ error: 'Account is already pinned' });
        return;
      }

      const result = await pinLeaderboardAccount(address);

      if (!result.success) {
        res.status(400).json({ error: result.error });
        return;
      }

      logger.info('account_pinned_from_leaderboard', { address: result.account?.address });
      res.status(201).json({ success: true, account: result.account });
    } catch (err: any) {
      logger.error('pin_leaderboard_account_failed', { err: err?.message });
      res.status(500).json({ error: 'Failed to pin account' });
    }
  });

  // Add a custom pinned account (max 3)
  app.post('/pinned-accounts/custom', async (req, res) => {
    try {
      const { address } = req.body;
      if (!address || typeof address !== 'string') {
        res.status(400).json({ error: 'Address is required' });
        return;
      }

      // Validate Ethereum address
      try {
        validateEthereumAddress(address);
      } catch (e: any) {
        res.status(400).json({ error: e.message || 'Invalid Ethereum address' });
        return;
      }

      // Check if already pinned
      const existingPinned = await isPinnedAccount(address);
      if (existingPinned) {
        res.status(400).json({ error: 'Account is already pinned' });
        return;
      }

      // Check if account exists in current leaderboard (should use pin from leaderboard instead)
      if (leaderboardService) {
        const existingEntry = await leaderboardService.isSystemRankedAccount(address);
        if (existingEntry) {
          res.status(400).json({
            error: 'Account exists in leaderboard. Use the pin button on that row instead.',
            rank: existingEntry.rank,
          });
          return;
        }
      }

      const result = await addCustomPinnedAccount(address);

      if (!result.success) {
        res.status(400).json({ error: result.error });
        return;
      }

      logger.info('custom_pinned_account_added', { address: result.account?.address });

      // Return immediately - don't block on stats fetch
      res.status(201).json({ success: true, account: result.account });

      // Fetch stats in background (non-blocking) so UI doesn't timeout
      if (leaderboardService && result.account?.address) {
        leaderboardService.fetchAndStoreCustomAccountStats(result.account.address)
          .then(stats => {
            if (stats) {
              logger.info('custom_pinned_account_stats_fetched', { address: result.account?.address, remark: stats.remark });
            }
          })
          .catch((err: any) => {
            logger.error('custom_pinned_account_stats_fetch_failed', { address: result.account?.address, err: err?.message });
          });
      }
    } catch (err: any) {
      logger.error('add_custom_pinned_account_failed', { err: err?.message });
      res.status(500).json({ error: 'Failed to add custom pinned account' });
    }
  });

  // Unpin an account
  app.delete('/pinned-accounts/:address', async (req, res) => {
    try {
      const { address } = req.params;
      if (!address) {
        res.status(400).json({ error: 'Address is required' });
        return;
      }

      const removed = await unpinAccount(address);
      if (!removed) {
        res.status(404).json({ error: 'Account not found' });
        return;
      }

      logger.info('account_unpinned', { address });
      res.json({ success: true });
    } catch (err: any) {
      logger.error('unpin_account_failed', { err: err?.message });
      res.status(500).json({ error: 'Failed to unpin account' });
    }
  });

  // Legacy endpoints for backward compatibility
  // These redirect to the new pinned accounts endpoints
  app.get('/custom-accounts', async (_req, res) => {
    try {
      const accounts = await listPinnedAccounts(true); // Only custom accounts
      res.json({
        accounts: accounts.map(a => ({
          id: a.id,
          address: a.address,
          nickname: null,
          addedAt: a.pinnedAt,
        })),
        count: accounts.length,
        maxAllowed: 3,
      });
    } catch (err: any) {
      logger.error('custom_accounts_list_failed', { err: err?.message });
      res.status(500).json({ error: 'Failed to list custom accounts' });
    }
  });

  app.post('/custom-accounts', async (req, res) => {
    // Redirect to custom pinned account endpoint
    try {
      const { address } = req.body;
      if (!address || typeof address !== 'string') {
        res.status(400).json({ error: 'Address is required' });
        return;
      }

      validateEthereumAddress(address);

      const existingPinned = await isPinnedAccount(address);
      if (existingPinned) {
        res.status(400).json({ error: 'Account is already pinned' });
        return;
      }

      const result = await addCustomPinnedAccount(address);

      if (!result.success) {
        res.status(400).json({ error: result.error });
        return;
      }

      logger.info('custom_account_added_legacy', { address: result.account?.address });
      res.status(201).json({
        success: true,
        account: {
          id: result.account?.id,
          address: result.account?.address,
          nickname: null,
          addedAt: result.account?.pinnedAt,
        },
      });

      // Fetch stats in background
      if (leaderboardService && result.account?.address) {
        leaderboardService.fetchAndStoreCustomAccountStats(result.account.address)
          .catch((err: any) => {
            logger.error('custom_account_stats_fetch_failed', { address: result.account?.address, err: err?.message });
          });
      }
    } catch (err: any) {
      logger.error('custom_account_add_failed', { err: err?.message });
      res.status(500).json({ error: err?.message || 'Failed to add custom account' });
    }
  });

  app.delete('/custom-accounts/:address', async (req, res) => {
    try {
      const { address } = req.params;
      if (!address) {
        res.status(400).json({ error: 'Address is required' });
        return;
      }

      const removed = await unpinAccount(address);
      if (!removed) {
        res.status(404).json({ error: 'Account not found' });
        return;
      }

      logger.info('custom_account_removed_legacy', { address });
      res.json({ success: true });
    } catch (err: any) {
      logger.error('custom_account_remove_failed', { err: err?.message });
      res.status(500).json({ error: 'Failed to remove custom account' });
    }
  });

  // =====================
  // Refresh Status API (Public - read only)
  // =====================

  // Get refresh status - public endpoint for dashboard
  app.get('/leaderboard/refresh-status', async (req, res) => {
    try {
      const period = Number(req.query.period ?? DEFAULT_LEADERBOARD_PERIOD);
      const lastRefreshFromDb = await getLastRefreshTime(period);

      // Get live status from service if available
      const liveStatus = leaderboardService?.getRefreshStatus() || null;

      res.json({
        period,
        // Database-based last refresh time
        lastRefresh: lastRefreshFromDb,
        lastRefreshFormatted: lastRefreshFromDb ? new Date(lastRefreshFromDb).toISOString() : null,
        // Live status from service
        ...(liveStatus && {
          status: liveStatus.status,
          isRefreshing: liveStatus.isRefreshing,
          nextRefreshAt: liveStatus.nextRefreshAt,
          nextRefreshInMs: liveStatus.nextRefreshInMs,
          refreshIntervalMs: liveStatus.refreshIntervalMs,
          progress: liveStatus.progress,
        }),
      });
    } catch (err: any) {
      logger.error('refresh_status_failed', { err: err?.message });
      res.status(500).json({ error: 'Failed to get refresh status' });
    }
  });

  // =====================
  // Admin Refresh API (Protected - requires owner key)
  // =====================

  // Trigger a manual refresh - ADMIN ONLY
  // This is protected by ownerOnly middleware to prevent abuse
  app.post('/admin/leaderboard/trigger-refresh', ownerOnly, async (req, res) => {
    try {
      const period = Number(req.query.period ?? DEFAULT_LEADERBOARD_PERIOD);
      if (!leaderboardService) {
        res.status(503).json({ error: 'Leaderboard service not initialized' });
        return;
      }

      // Check if already refreshing
      const status = leaderboardService.getRefreshStatus();
      if (status.isRefreshing) {
        res.status(409).json({
          error: 'Refresh already in progress',
          startedAt: status.refreshStartedAt,
          progress: status.progress,
        });
        return;
      }

      logger.info('admin_refresh_triggered', {
        period,
        source: 'api',
        ip: req.ip,
      });

      // Run refresh in background, return immediately
      leaderboardService.refreshAll().catch((err) => {
        logger.error('admin_refresh_failed', { period, err: err?.message });
      });

      res.json({
        success: true,
        message: 'Refresh started in background',
        period,
        checkStatusAt: '/leaderboard/refresh-status',
      });
    } catch (err: any) {
      logger.error('admin_refresh_request_failed', { err: err?.message });
      res.status(500).json({ error: 'Failed to trigger refresh' });
    }
  });

  const port = getPort(8080);
  app.listen(port, () => {
    logger.info('hl-scout listening', { port });
  });
}

main().catch((err) => {
  console.error('[FATAL]', err);
  logger.error('fatal_startup', {
    err: err instanceof Error ? err.message : String(err),
    stack: err instanceof Error ? err.stack : undefined
  });
  process.exit(1);
});
