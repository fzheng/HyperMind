import cors from 'cors';
import express, { Request, Response, NextFunction } from 'express';
import http from 'http';
import { WebSocketServer, WebSocket } from 'ws';
import swaggerUi from 'swagger-ui-express';
import type { OpenAPIV3 } from 'openapi-types';
import path from 'path';
import fs from 'fs';
import { setMaxListeners } from 'events';
import {
  EventQueue,
  RealtimeTracker,
  createHistogram,
  createLogger,
  getOwnerToken,
  getPort,
  initMetrics,
  normalizeAddress,
  connectNats,
  ensureStream,
  publishJson,
  FillEventSchema
} from '@hl/ts-lib';

const OWNER_TOKEN = getOwnerToken();
const logger = createLogger('hl-stream');
const metrics = initMetrics('hl_stream');
const fillsHistogram = createHistogram(metrics, 'fills_publish_seconds', 'Latency for publishing fills', [0.005, 0.01, 0.05, 0.1]);
const WATCH_PERIOD = Number(process.env.LEADERBOARD_PERIOD ?? process.env.LEADERBOARD_WATCH_PERIOD ?? 30);
const WATCH_LIMIT = Number(process.env.LEADERBOARD_SELECT_COUNT ?? 12);
setMaxListeners(0);

const swaggerDoc: OpenAPIV3.Document = {
  openapi: '3.0.0',
  info: {
    title: 'hl-stream API',
    version: '0.1.0',
    description: 'Watchlist refresh, WebSocket feed access, and fake fill publisher.'
  },
  servers: [{ url: '/' }],
  components: {
    securitySchemes: {
      ownerToken: { type: 'apiKey', in: 'header', name: 'x-owner-key' }
    }
  },
  paths: {
    '/healthz': {
      get: { summary: 'Service health', responses: { 200: { description: 'OK' } } }
    },
    '/metrics': {
      get: { summary: 'Prometheus metrics', responses: { 200: { description: 'Prometheus text' } } }
    },
    '/watchlist': {
      get: { summary: 'Current watchlist', responses: { 200: { description: 'Addresses' } } }
    },
    '/watchlist/refresh': {
      post: {
        summary: 'Refresh watchlist from hl-scout',
        security: [{ ownerToken: [] }],
        responses: { 200: { description: 'Refreshed' } }
      }
    },
    '/ws': {
      get: {
        summary: 'WebSocket endpoint for realtime fills/positions',
        responses: { 101: { description: 'Switching to WebSocket' } }
      }
    }
  }
};

const queue = new EventQueue(5000);
const clients = new Set<{ ws: WebSocket; lastSeq: number; alive: boolean }>();
let watchlist: string[] = [];
const scoutUrl = process.env.SCOUT_URL || 'http://hl-scout:8080';
const dashboardDir = path.resolve(__dirname, '..', 'public');
const fillsSubject = 'c.fills.v1';
let tracker: RealtimeTracker | null = null;

function ownerOnly(req: Request, res: Response, next: NextFunction) {
  const token = (req.headers['x-owner-key'] || req.headers['x-owner-token'] || '').toString();
  if (token !== OWNER_TOKEN) return res.status(403).json({ error: 'forbidden' });
  return next();
}

async function fetchWatchlist(): Promise<string[]> {
  const selectedUrl = new URL('/leaderboard/selected', scoutUrl);
  selectedUrl.searchParams.set('period', String(WATCH_PERIOD));
  selectedUrl.searchParams.set('limit', String(WATCH_LIMIT));
  try {
    const res = await fetch(selectedUrl, { headers: { 'x-owner-key': OWNER_TOKEN } });
    if (res.ok) {
      const data = await res.json();
      if (Array.isArray(data?.entries) && data.entries.length) {
        return data.entries.map((entry: any) => normalizeAddress(entry.address));
      }
    }
  } catch (err) {
    logger.warn('selected_watchlist_failed', { err: err instanceof Error ? err.message : err });
  }
  try {
    const fallback = await fetch(`${scoutUrl}/addresses`, {
      headers: { 'x-owner-key': OWNER_TOKEN }
    });
    if (fallback.ok) {
      const fallbackData = await fallback.json();
      return (fallbackData?.addresses || []).map((entry: any) => normalizeAddress(entry.address));
    }
  } catch (err) {
    logger.warn('addresses_watchlist_failed', { err: err instanceof Error ? err.message : err });
  }
  if (watchlist.length) return watchlist;
  return [];
}

async function proxyScout(pathname: string, req: Request, res: Response) {
  try {
    const target = new URL(pathname, scoutUrl);
    const idx = req.originalUrl.indexOf('?');
    if (idx >= 0) {
      target.search = req.originalUrl.slice(idx);
    }
    const response = await fetch(target, { headers: { 'x-owner-key': OWNER_TOKEN } });
    const body = await response.text();
    const type = response.headers.get('content-type') || 'application/json';
    res.status(response.status).setHeader('Content-Type', type).send(body);
  } catch (err: any) {
    logger.error('dashboard_proxy_failed', { err: err?.message });
    res.status(502).json({ error: 'proxy_failed' });
  }
}

function toFillEventPayload(evt: any) {
  return FillEventSchema.parse({
    fill_id: evt.hash || `${evt.address}-${evt.at}`,
    source: 'hyperliquid',
    address: evt.address,
    asset: evt.symbol || 'BTC',
    side: evt.side === 'sell' ? 'sell' : 'buy',
    size: Number(evt.size ?? evt.payload?.size ?? 0),
    price: Number(evt.priceUsd ?? evt.payload?.priceUsd ?? 0),
    start_position: typeof evt.startPosition === 'number' ? evt.startPosition : null,
    realized_pnl: evt.realizedPnlUsd != null ? Number(evt.realizedPnlUsd) : null,
    ts: evt.at,
    meta: { action: evt.action ?? null }
  });
}

async function publishFillFromEvent(
  js: Awaited<ReturnType<typeof connectNats>>['js'],
  evt: any
) {
  try {
    const fill = toFillEventPayload(evt);
    const end = (fillsHistogram as any).startTimer?.({ operation: 'publish' });
    await publishJson(js, fillsSubject, fill);
    end?.();
  } catch (err: any) {
    logger.warn('fill_publish_failed', { err: err?.message });
  }
}

function configureWebSocket(server: http.Server) {
  const wss = new WebSocketServer({ server, path: '/ws' });
  let pingInterval: NodeJS.Timeout | null = null;
  let broadcastInterval: NodeJS.Timeout | null = null;

  wss.on('connection', (ws) => {
    const client = { ws, lastSeq: 0, alive: true };
    clients.add(client);
    ws.send(JSON.stringify({ type: 'hello', latestSeq: queue.latestSeq() }));
    ws.on('message', (raw) => {
      try {
        const msg = JSON.parse(String(raw));
        if (msg && typeof msg.since === 'number') {
          client.lastSeq = msg.since;
          const events = queue.listSince(msg.since, 500);
          if (events.length) {
            client.lastSeq = events[events.length - 1].seq;
            ws.send(JSON.stringify({ type: 'batch', events }));
          }
        }
      } catch {
        /* ignore */
      }
    });
    ws.on('pong', () => { client.alive = true; });
    ws.on('close', () => { clients.delete(client); });
    ws.on('error', () => { clients.delete(client); });
  });

  pingInterval = setInterval(() => {
    for (const client of clients) {
      if (!client.alive) {
        clients.delete(client);
        try { client.ws.terminate(); } catch (e) {
          logger.warn('ws_terminate_failed', { error: e });
        }
        continue;
      }
      client.alive = false;
      try { client.ws.ping(); } catch (e) {
        logger.warn('ws_ping_failed', { error: e });
        clients.delete(client);
      }
    }
  }, 30000);

  const broadcast = () => {
    if (!clients.size) return;
    for (const client of clients) {
      if (client.ws.readyState !== WebSocket.OPEN) {
        clients.delete(client);
        continue;
      }
      const events = queue.listSince(client.lastSeq, 200);
      if (events.length) {
        client.lastSeq = events[events.length - 1].seq;
        try {
          client.ws.send(JSON.stringify({ type: 'events', events }));
        } catch (e) {
          logger.warn('ws_send_failed', { error: e });
          clients.delete(client);
        }
      }
    }
  };

  broadcastInterval = setInterval(broadcast, 1000);

  // Cleanup on server shutdown
  process.on('SIGTERM', () => {
    if (pingInterval) clearInterval(pingInterval);
    if (broadcastInterval) clearInterval(broadcastInterval);
    wss.close();
  });
}

async function main() {
  const natsUrl = process.env.NATS_URL || 'nats://localhost:4222';
  const nats = await connectNats(natsUrl);
  await ensureStream(nats.jsm, 'HL_C', [fillsSubject]);

  watchlist = await fetchWatchlist();
  logger.info('watchlist_loaded', { count: watchlist.length });

  tracker = new RealtimeTracker(async () => watchlist, queue, {
    onTrade: ({ event }) => {
      publishFillFromEvent(nats.js, event).catch((err) =>
        logger.error('fill_publish_failed', { err: err?.message })
      );
    }
  });
  tracker.start().catch((err) => logger.error('realtime_start_failed', { err: err?.message }));

  const app = express();
  app.use(cors());
  app.use(express.json());
  app.use('/docs', swaggerUi.serve, swaggerUi.setup(swaggerDoc));
  if (fs.existsSync(dashboardDir)) {
    app.use('/dashboard/static', express.static(dashboardDir));
    app.get('/dashboard', (_req, res) => {
      res.sendFile(path.join(dashboardDir, 'dashboard.html'));
    });
  }

  app.get('/healthz', (_req, res) => res.json({ status: 'ok', watchlist: watchlist.length }));
  app.get('/metrics', metricsHandler(metrics));
  app.get('/watchlist', (_req, res) => res.json({ addresses: watchlist }));

  app.post('/watchlist/refresh', ownerOnly, async (_req, res) => {
    watchlist = await fetchWatchlist();
    await tracker?.refresh();
    logger.info('watchlist_refreshed', { count: watchlist.length });
    res.json({ ok: true, count: watchlist.length });
  });
  app.get('/dashboard/api/summary', (req, res) => proxyScout('/dashboard/summary', req, res));
  app.get('/dashboard/api/fills', (req, res) => proxyScout('/dashboard/fills', req, res));
  app.get('/dashboard/api/decisions', (req, res) => proxyScout('/dashboard/decisions', req, res));
  app.get('/dashboard/api/price', (req, res) => proxyScout('/dashboard/price', req, res));

  const server = http.createServer(app);
  configureWebSocket(server);

  setInterval(async () => {
    try {
      watchlist = await fetchWatchlist();
      await tracker?.refresh();
    } catch (err: any) {
      logger.warn('watchlist_poll_failed', { err: err instanceof Error ? err.message : String(err) });
    }
  }, 60000);
  setInterval(() => {
    void tracker?.ensureFreshSnapshots();
  }, 30000);

  const port = getPort(8080);
  server.listen(port, () => logger.info('hl-stream listening', { port }));
}

function metricsHandler(ctx: ReturnType<typeof initMetrics>) {
  return async (_req: Request, res: Response) => {
    res.setHeader('Content-Type', ctx.registry.contentType);
    res.send(await ctx.registry.metrics());
  };
}

main().catch((err) => {
  logger.error('fatal_startup', { err: err instanceof Error ? err.message : String(err) });
  process.exit(1);
});
