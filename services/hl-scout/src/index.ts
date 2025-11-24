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
  fetchUserBtcFills,
  fetchUserProfile,
  fetchPerpMarkPrice,
  getOwnerToken,
  getPool,
  getPort,
  initMetrics,
  createHistogram,
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
  listCustomAccounts,
  addCustomAccount,
  removeCustomAccount,
  getCustomAccountCount,
  getLastRefreshTime
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

function parsePeriod(value: any): number {
  const n = Number(value);
  if (Number.isFinite(n) && n > 0) return n;
  return LEADERBOARD_PERIODS[0] || DEFAULT_LEADERBOARD_PERIOD || 30;
}

interface AddressPayload {
  address: string;
  nickname?: string | null;
}

function ownerOnly(req: Request, res: Response, next: NextFunction) {
  const token = (req.headers['x-owner-key'] || req.headers['x-owner-token'] || '').toString();
  if (token !== OWNER_TOKEN) {
    return res.status(403).json({ error: 'forbidden' });
  }
  return next();
}

const DEFAULT_SEEDS = (process.env.SCOUT_SEEDS || '0x1111111111111111111111111111111111111111,0x2222222222222222222222222222222222222222,0x3333333333333333333333333333333333333333')
  .split(',')
  .map((addr) => addr.trim().toLowerCase())
  .filter((addr) => addr.startsWith('0x') && addr.length === 42);

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

async function backfillRecent(address: string, limit = 50): Promise<number> {
  const fills = await fetchUserBtcFills(address, { aggregateByTime: true });
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
      symbol: 'BTC',
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

async function main() {
  await getPool(); // ensure db connectivity
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
    refreshMs: Number(process.env.LEADERBOARD_REFRESH_MS ?? 24 * 60 * 60 * 1000),
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

    // Get custom accounts
    const customAccounts = await listCustomAccounts();
    const customAddressSet = new Set(customAccounts.map((a) => a.address.toLowerCase()));

    // Mark system accounts and filter out any that are also custom
    const systemEntries = selected
      .filter((entry) => !customAddressSet.has(entry.address.toLowerCase()))
      .slice(0, systemLimit)
      .map((entry) => ({
        ...entry,
        isCustom: false,
      }));

    // For custom accounts, try to get their leaderboard stats
    const customEntries = await Promise.all(
      customAccounts.map(async (custom) => {
        // Try to find in leaderboard entries
        const pool = await getPool();
        const { rows } = await pool.query(
          `SELECT * FROM hl_leaderboard_entries
           WHERE period_days = $1 AND lower(address) = $2
           LIMIT 1`,
          [period, custom.address.toLowerCase()]
        );
        if (rows.length) {
          const row = rows[0];
          return {
            address: custom.address,
            rank: 0, // Will be re-ranked
            score: Number(row.score ?? 0),
            weight: 0,
            winRate: Number(row.win_rate ?? 0),
            executedOrders: Number(row.executed_orders ?? 0),
            realizedPnl: Number(row.realized_pnl ?? 0),
            efficiency: Number(row.efficiency ?? 0),
            pnlConsistency: Number(row.pnl_consistency ?? 0),
            remark: custom.nickname || row.remark || null,
            labels: row.labels || [],
            statOpenPositions: row.stat_open_positions,
            statClosedPositions: row.stat_closed_positions,
            statAvgPosDuration: row.stat_avg_pos_duration,
            statTotalPnl: row.stat_total_pnl,
            statMaxDrawdown: row.stat_max_drawdown,
            meta: row.metrics || {},
            isCustom: true,
          };
        }
        // No leaderboard entry found - return minimal data
        return {
          address: custom.address,
          rank: 0,
          score: 0,
          weight: 0,
          winRate: 0,
          executedOrders: 0,
          realizedPnl: 0,
          efficiency: 0,
          pnlConsistency: 0,
          remark: custom.nickname || null,
          labels: [],
          statOpenPositions: null,
          statClosedPositions: null,
          statAvgPosDuration: null,
          statTotalPnl: null,
          statMaxDrawdown: null,
          meta: {},
          isCustom: true,
        };
      })
    );

    // Merge and re-rank all entries by score
    const allEntries = [...systemEntries, ...customEntries];
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

    // Get holdings for all accounts
    const holdings: Record<string, { symbol: string; size: number }> = {};
    if (allEntries.length) {
      const pool = await getPool();
      const { rows } = await pool.query(
        'select address, symbol, size from hl_current_positions where lower(address) = any($1)',
        [allEntries.map((s) => s.address.toLowerCase())]
      );
      for (const row of rows) {
        const addr = String(row.address || '').toLowerCase();
        if (!addr) continue;
        holdings[addr] = {
          symbol: String(row.symbol || 'BTC').toUpperCase(),
          size: Number(row.size || 0),
        };
      }
    }

    // Get last refresh timestamp
    const lastRefresh = await getLastRefreshTime(period);

    res.json({
      period,
      stats,
      selected: allEntries,
      featured,
      holdings,
      lastRefresh,
      lastRefreshFormatted: lastRefresh ? new Date(lastRefresh).toISOString() : null,
      customAccountCount: customAccounts.length,
      maxCustomAccounts: 3,
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
    const fills = await listLiveFills(limit);
    res.json({ fills });
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
  // Custom Accounts API
  // =====================

  // Get all custom accounts
  app.get('/custom-accounts', async (_req, res) => {
    try {
      const accounts = await listCustomAccounts();
      const count = accounts.length;
      res.json({ accounts, count, maxAllowed: 3 });
    } catch (err: any) {
      logger.error('custom_accounts_list_failed', { err: err?.message });
      res.status(500).json({ error: 'Failed to list custom accounts' });
    }
  });

  // Add a custom account
  app.post('/custom-accounts', async (req, res) => {
    try {
      const { address, nickname } = req.body;
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

      // Check if account already exists in top 10 system-ranked entries
      // Only block if it's already in the top performers shown in UI
      if (leaderboardService) {
        const existingEntry = await leaderboardService.isSystemRankedAccount(address);
        if (existingEntry && existingEntry.rank <= 10) {
          res.status(400).json({
            error: 'Account already exists in top 10 system rankings',
            rank: existingEntry.rank,
            score: existingEntry.score,
          });
          return;
        }
      }

      const sanitizedNickname = nickname ? sanitizeNickname(nickname) : null;
      const result = await addCustomAccount(address, sanitizedNickname);

      if (!result.success) {
        res.status(400).json({ error: result.error });
        return;
      }

      logger.info('custom_account_added', { address: result.account?.address });

      // Immediately fetch stats for the new custom account so it doesn't show all zeros
      // Custom accounts are now properly scored using the same algorithm as system accounts
      let stats = null;
      if (leaderboardService && result.account?.address) {
        try {
          stats = await leaderboardService.fetchAndStoreCustomAccountStats(result.account.address, sanitizedNickname);
        } catch (err: any) {
          logger.error('custom_account_stats_fetch_failed', { address: result.account?.address, err: err?.message });
        }
      }

      res.status(201).json({ success: true, account: result.account, stats });
    } catch (err: any) {
      logger.error('custom_account_add_failed', { err: err?.message });
      res.status(500).json({ error: 'Failed to add custom account' });
    }
  });

  // Remove a custom account
  app.delete('/custom-accounts/:address', async (req, res) => {
    try {
      const { address } = req.params;
      if (!address) {
        res.status(400).json({ error: 'Address is required' });
        return;
      }

      const removed = await removeCustomAccount(address);
      if (!removed) {
        res.status(404).json({ error: 'Account not found' });
        return;
      }

      logger.info('custom_account_removed', { address });
      res.json({ success: true });
    } catch (err: any) {
      logger.error('custom_account_remove_failed', { err: err?.message });
      res.status(500).json({ error: 'Failed to remove custom account' });
    }
  });

  // =====================
  // Manual Refresh API
  // =====================

  // Trigger a manual refresh of the leaderboard
  app.post('/leaderboard/refresh', ownerOnly, async (req, res) => {
    try {
      const period = Number(req.query.period ?? DEFAULT_LEADERBOARD_PERIOD);
      if (!leaderboardService) {
        res.status(503).json({ error: 'Leaderboard service not initialized' });
        return;
      }

      logger.info('manual_refresh_triggered', { period });

      // Run refresh in background, return immediately
      leaderboardService.refreshPeriod(period).catch((err) => {
        logger.error('manual_refresh_failed', { period, err: err?.message });
      });

      res.json({ success: true, message: 'Refresh started', period });
    } catch (err: any) {
      logger.error('manual_refresh_request_failed', { err: err?.message });
      res.status(500).json({ error: 'Failed to trigger refresh' });
    }
  });

  // Get last refresh timestamp
  app.get('/leaderboard/refresh-status', async (req, res) => {
    try {
      const period = Number(req.query.period ?? DEFAULT_LEADERBOARD_PERIOD);
      const lastRefresh = await getLastRefreshTime(period);
      res.json({
        period,
        lastRefresh,
        lastRefreshFormatted: lastRefresh ? new Date(lastRefresh).toISOString() : null
      });
    } catch (err: any) {
      logger.error('refresh_status_failed', { err: err?.message });
      res.status(500).json({ error: 'Failed to get refresh status' });
    }
  });

  const port = getPort(8080);
  app.listen(port, () => {
    logger.info('hl-scout listening', { port });
  });
}

function metricsHandler(ctx: ReturnType<typeof initMetrics>) {
  return async (_req: Request, res: Response) => {
    res.setHeader('Content-Type', ctx.registry.contentType);
    res.send(await ctx.registry.metrics());
  };
}

main().catch((err) => {
  console.error('[FATAL]', err);
  logger.error('fatal_startup', {
    err: err instanceof Error ? err.message : String(err),
    stack: err instanceof Error ? err.stack : undefined
  });
  process.exit(1);
});
