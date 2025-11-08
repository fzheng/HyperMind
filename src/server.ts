import express from 'express';
import cors from 'cors';
import path from 'path';
import { initStorage, listAddresses as storageList, addAddress as storageAdd, removeAddress as storageRemove, getNicknames as storageGetNicknames, setNickname as storageSetNickname } from './storage';
import { pageTradesByTime, insertTradeIfNew, countValidTradesForAddress, deleteTradesForAddress, deleteAllTrades } from './persist';
import type { Address } from './types';
import { fetchPerpPositions, fetchUserBtcFills } from './hyperliquid';
import { startPriceFeed, refreshPriceFeed, getCurrentBtcPrice } from './price';
import { EventQueue } from './queue';
import { RealtimeTracker } from './realtime';
import { WebSocketServer } from 'ws';

const app = express();
const PORT = process.env.PORT ? Number(process.env.PORT) : 3000;
const IPINFO_INTERVAL_MS = process.env.IPINFO_INTERVAL_MS ? Number(process.env.IPINFO_INTERVAL_MS) : 600_000; // 10 minutes

app.use(cors());
app.use(express.json());

// In-memory event queue for streaming incremental changes
const changes = new EventQueue(5000);

// Server-side IP/region tracking via ipinfo.io (VPN awareness)
type IpInfo = {
  ip: string | null;
  region: string | null;
  country?: string | null;
  city?: string | null;
  updatedAt: string | null;
};
let ipInfoState: IpInfo = { ip: null, region: null, country: null, city: null, updatedAt: null };

async function refreshIpInfo(): Promise<IpInfo> {
  try {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), 8000);
    const res = await fetch('https://ipinfo.io/json', {
      method: 'GET',
      headers: { 'Accept': 'application/json', 'User-Agent': 'hlbot/0.1 (+https://github.com/)' },
      signal: ctrl.signal,
    } as any);
    clearTimeout(t);
    if (!res.ok) throw new Error(`ipinfo HTTP ${res.status}`);
    const j: any = await res.json();
    ipInfoState = {
      ip: typeof j?.ip === 'string' ? j.ip : null,
      region: typeof j?.region === 'string' ? j.region : null,
      country: typeof j?.country === 'string' ? j.country : null,
      city: typeof j?.city === 'string' ? j.city : null,
      updatedAt: new Date().toISOString(),
    };
  } catch (e) {
    // Keep previous state on failure; stamp updatedAt to reflect attempt
    ipInfoState = { ...ipInfoState, updatedAt: new Date().toISOString() };
  }
  return ipInfoState;
}

async function getAddresses(): Promise<Address[]> {
  return await storageList();
}

// Helpers (local)
function deriveActionLabel(startPosition: number, sz: number, side: 'B' | 'A'): string {
  const delta = side === 'B' ? +sz : -sz;
  const newPos = startPosition + delta;
  if (startPosition === 0) return delta > 0 ? 'Open Long (Open New)' : 'Open Short (Open New)';
  if (startPosition > 0) return delta > 0 ? 'Increase Long' : (newPos === 0 ? 'Close Long (Close All)' : 'Decrease Long');
  return delta < 0 ? 'Increase Short' : (newPos === 0 ? 'Close Short (Close All)' : 'Decrease Short');
}

async function backfillRecentForAddress(addr: string, limit = 100) {
  const fills = await fetchUserBtcFills(addr, { aggregateByTime: true });
  let inserted = 0;
  for (const f of (fills || []).slice(0, limit)) {
    const action = deriveActionLabel(Number(f.startPosition), Number(f.sz), f.side);
    const payload = {
      at: new Date(f.time).toISOString(),
      address: addr,
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
    const result = await insertTradeIfNew(addr, payload);
    if (result.inserted) inserted += 1;
  }
  return inserted;
}

// API routes
app.get('/api/addresses', async (_req, res) => {
  const addrs = await getAddresses();
  const nicknames = await storageGetNicknames().catch(() => ({}));
  res.json({ addresses: addrs, nicknames });
});

app.post('/api/addresses', async (req, res) => {
  const address: unknown = req.body?.address;
  if (typeof address !== 'string' || address.trim().length === 0) {
    return res.status(400).json({ error: 'Invalid address' });
  }
  const addr = address.trim().toLowerCase();
  const existing = (await getAddresses()).map(a => a.toLowerCase());
  if (!existing.includes(addr)) {
    await storageAdd(addr);
    console.log(`[api] Added address ${addr}`);
    refreshPriceFeed().catch((e) => console.warn('[api] price feed refresh failed', e));
    realtime.refresh().catch((e) => console.warn('[api] realtime refresh failed', e));
    // Immediately prime snapshot via HTTP so UI shows latest without waiting for WS
    await realtime.primeFromHttp(addr).catch((e) => console.warn('[api] prime snapshot failed', e));
    // Backfill latest 100 trades for newly added address
    try {
      const inserted = await backfillRecentForAddress(addr, 100);
      if (inserted > 0) console.log(`[api] Backfilled ${inserted} trades for ${addr}`);
    } catch (e) {
      console.warn('[api] backfill on add failed', e);
    }
  }
  res.json({ addresses: await getAddresses() });
});

// Recommendations endpoint removed (feature deprecated)

app.get('/api/price', (_req, res) => {
  res.json(getCurrentBtcPrice());
});

// IP info endpoints (server-originated)
app.get('/api/ipinfo', (_req, res) => {
  res.json(ipInfoState);
});
app.post('/api/ipinfo/refresh', async (_req, res) => {
  try {
    const info = await refreshIpInfo();
    res.json(info);
  } catch (e) {
    res.status(500).json({ error: 'ipinfo refresh failed' });
  }
});

// Stream changes (poll-style API): /api/changes?since=0&limit=200
app.get('/api/changes', (req, res) => {
  const since = Number(req.query.since ?? 0) || 0;
  const limit = Math.min(1000, Math.max(1, Number(req.query.limit ?? 200)));
  const events = changes.listSince(since, limit);
  const next = events.length > 0 ? events[events.length - 1].seq : changes.latestSeq();
  res.json({ events, next });
});

// Current BTC positions across tracked addresses (from realtime snapshots)
app.get('/api/current-positions', async (_req, res) => {
  try {
    const base = realtime.getAllSnapshots();
    const nmap = await storageGetNicknames().catch(() => ({} as Record<string, string>));
    const enriched = base.map((p) => ({ ...p, nickname: (nmap as any)[p.address] || null }));
    res.json({ positions: enriched });
  } catch (e) {
    res.status(500).json({ error: 'failed to get positions' });
  }
});

// Latest trade fills (BTC only) across addresses, ordered strictly by timestamp desc then id desc
app.get('/api/latest-trades', async (req, res) => {
  const limit = Math.min(200, Math.max(1, Number(req.query.limit ?? 100)));
  const beforeAtRaw = typeof req.query.beforeAt === 'string' ? String(req.query.beforeAt) : null;
  const beforeIdRaw = typeof req.query.beforeId === 'string' ? Number(req.query.beforeId) : null;
  const beforeId = Number.isFinite(beforeIdRaw) ? Number(beforeIdRaw) : null;
  const address = typeof req.query.address === 'string' ? String(req.query.address).toLowerCase() : null;
  try {
    // Opportunistic light backfill when requesting first page for one address
    if (address && !beforeAtRaw && beforeId == null) {
      try { await backfillRecentForAddress(address, 100); } catch {}
    }
    const rows = await pageTradesByTime({ limit, beforeAt: beforeAtRaw, beforeId, address });
    const trades = rows.map((r) => {
      const p = r.payload || {};
      const closedPnl = p.realizedPnlUsd != null ? Number(p.realizedPnlUsd) : (p.closedPnl != null ? Number(p.closedPnl) : null);
      return {
        id: r.id,
        time: r.at || p.at || null,
        address: r.address || p.address || null,
        symbol: p.symbol || 'BTC',
        action: p.action || (p.side === 'buy' ? 'Buy' : 'Sell'),
        size: p.size != null ? Number(p.size) : (p.sz != null ? Math.abs(Number(p.sz)) : 0),
        startPosition: p.startPosition != null ? Number(p.startPosition) : null,
        price: p.priceUsd != null ? Number(p.priceUsd) : (p.px != null ? Number(p.px) : 0),
        closedPnl,
        fee: p.fee != null ? Number(p.fee) : null,
        feeToken: p.feeToken || null,
        tx: p.hash || null,
      };
    });
    const last = rows.length > 0 ? rows[rows.length - 1] : null;
    const nextCursor = last ? { beforeAt: last.at || null, beforeId: last.id } : null;
    return res.json({
      trades,
      nextCursor,
      nextBeforeAt: nextCursor?.beforeAt ?? null,
      nextBeforeId: nextCursor?.beforeId ?? null,
    });
  } catch (e) {
    return res.status(500).json({ error: 'failed to fetch trades' });
  }
});

// Manual backfill of recent BTC fills for all or one address
app.post('/api/backfill', async (req, res) => {
  const perAddrLimit = Math.min(1000, Math.max(1, Number(req.body?.limit ?? 300)));
  const onlyAddrRaw = typeof req.body?.address === 'string' ? String(req.body.address).trim().toLowerCase() : null;
  try {
    const addrs = onlyAddrRaw ? [onlyAddrRaw] : await getAddresses();
    let inserted = 0;
    for (const addr of addrs) {
      const fills = await fetchUserBtcFills(addr, { aggregateByTime: true });
      // Merge newest first, cap at perAddrLimit
      for (const f of fills.slice(0, perAddrLimit)) {
        const delta = f.side === 'B' ? +f.sz : -f.sz;
        const newPos = f.startPosition + delta;
        let action = '';
        if (f.startPosition === 0) action = delta > 0 ? 'Open Long (Open New)' : 'Open Short (Open New)';
        else if (f.startPosition > 0) action = delta > 0 ? 'Increase Long' : (newPos === 0 ? 'Close Long (Close All)' : 'Decrease Long');
        else action = delta < 0 ? 'Increase Short' : (newPos === 0 ? 'Close Short (Close All)' : 'Decrease Short');
        const payload = {
          at: new Date(f.time).toISOString(),
          address: addr,
          symbol: 'BTC',
          action,
          size: Math.abs(f.sz),
          startPosition: f.startPosition,
          priceUsd: f.px,
          realizedPnlUsd: f.closedPnl ?? null,
          fee: f.fee ?? null,
          feeToken: f.feeToken ?? null,
          hash: f.hash ?? null,
        };
        const result = await insertTradeIfNew(addr, payload);
        if (result.inserted) inserted += 1;
      }
    }
    res.json({ ok: true, inserted });
  } catch (e) {
    res.status(500).json({ ok: false, error: 'failed to backfill' });
  }
});

// Per-address recent BTC trades via Info API
app.get('/api/user-trades/:address', async (req, res) => {
  const addr = String(req.params.address || '').trim();
  if (!addr) return res.status(400).json({ error: 'Invalid address' });
  try {
    const fills = await fetchUserBtcFills(addr, { aggregateByTime: true });
    const out = fills.map((f) => ({
      time: new Date(f.time).toISOString(),
      address: addr,
      action: f.side === 'B' ? 'Buy' : 'Sell',
      size: Number(f.sz),
      startPosition: Number(f.startPosition),
      price: Number(f.px),
      closedPnl: f.closedPnl == null ? null : Number(f.closedPnl),
    }));
    res.json({ trades: out });
  } catch (e) {
    res.status(500).json({ error: 'failed to fetch user trades' });
  }
});

// Remove address
app.delete('/api/addresses/:address', async (req, res) => {
  const addrParam = String(req.params.address || '').trim().toLowerCase();
  if (!addrParam) return res.status(400).json({ error: 'Invalid address' });
  await storageRemove(addrParam);
  console.log(`[api] Removed address ${addrParam}`);
  await refreshPriceFeed().catch((e) => console.warn('[api] price feed refresh failed', e));
  await realtime.refresh().catch((e) => console.warn('[api] realtime refresh failed', e));
  res.json({ addresses: await getAddresses() });
});

// Set or clear nickname for an address
app.post('/api/addresses/:address/nickname', async (req, res) => {
  const addrParam = String(req.params.address || '').trim().toLowerCase();
  if (!addrParam) return res.status(400).json({ error: 'Invalid address' });
  const nicknameRaw = typeof req.body?.nickname === 'string' ? req.body.nickname : '';
  const nickname = nicknameRaw.trim().length ? nicknameRaw.trim() : null;
  try {
    await storageSetNickname(addrParam, nickname);
    const nicknames = await storageGetNicknames();
    res.json({ ok: true, nicknames });
  } catch (e) {
    res.status(500).json({ ok: false, error: 'failed to set nickname' });
  }
});

// Poller endpoint removed

// On-demand perp positions for an address
app.get('/api/positions/:address', async (req, res) => {
  const addr = String(req.params.address || '').trim();
  if (!addr) return res.status(400).json({ error: 'Invalid address' });
  try {
    const positions = await fetchPerpPositions(addr);
    res.json({ address: addr, positions });
  } catch (e) {
    res.status(500).json({ error: 'Failed to fetch positions' });
  }
});

// Static UI
const PUBLIC_DIR = path.resolve(process.cwd(), 'public');
app.use(express.static(PUBLIC_DIR));
app.get('/', (_req, res) => {
  res.sendFile(path.join(PUBLIC_DIR, 'index.html'));
});

// Realtime subscriptions for BTC positions and trades
const realtime = new RealtimeTracker(getAddresses, changes);
realtime.start().catch((e) => console.warn('[realtime] start failed', e));

initStorage()
  .then(async () => {
    await startPriceFeed(getAddresses);
    // Initial IP info fetch and periodic refresh
    await refreshIpInfo().catch(() => {});
    setInterval(() => { void refreshIpInfo(); }, IPINFO_INTERVAL_MS);
    const server = app.listen(PORT, () => {
      console.log(`hlbot server listening on http://localhost:${PORT}`);
    });

  // WebSocket: push real-time trade/position events
  const wss = new WebSocketServer({ server, path: '/ws' });
  interface WSClient { ws: any; lastSeq: number; alive: boolean; }
  const clients = new Set<WSClient>();

    wss.on('connection', (ws) => {
      const c: WSClient = { ws, lastSeq: 0, alive: true };
      clients.add(c);
      ws.send(JSON.stringify({ type: 'hello', latestSeq: changes.latestSeq() }));
      ws.on('message', (raw: any) => {
        try {
          const msg = JSON.parse(String(raw));
          if (msg && typeof msg.since === 'number') {
            c.lastSeq = msg.since;
            const events = changes.listSince(msg.since, 500);
            if (events.length) {
              c.lastSeq = events[events.length - 1].seq;
              ws.send(JSON.stringify({ type: 'batch', events }));
            }
          }
        } catch { /* ignore */ }
      });
      ws.on('pong', () => { c.alive = true; });
      ws.on('close', () => { clients.delete(c); });
    });

    // Adaptive broadcast + heartbeat
    let broadcastInterval = 1000; // default
    function scheduleBroadcast() {
      const count = clients.size;
      // Faster when many clients
      broadcastInterval = count > 25 ? 250 : count > 10 ? 500 : 1000;
      setTimeout(runBroadcast, broadcastInterval);
    }
    function runBroadcast() {
      if (clients.size) {
        for (const c of clients) {
          if (c.ws.readyState !== 1) continue; // OPEN
          const events = changes.listSince(c.lastSeq, 200);
          if (events.length) {
            c.lastSeq = events[events.length - 1].seq;
            c.ws.send(JSON.stringify({ type: 'events', events }));
          }
        }
      }
      scheduleBroadcast();
    }
    scheduleBroadcast();

    // Heartbeat: ping every 30s, drop unresponsive
    setInterval(() => {
      for (const c of clients) {
        if (!c.alive) {
          try { c.ws.terminate(); } catch {}
          clients.delete(c);
          continue;
        }
        c.alive = false;
        try { c.ws.ping(); } catch {}
      }
    }, 30000);

  // Optional startup cleanup + backfill (latest 100) when enabled
    if (String(process.env.BACKFILL_ON_START || '').toLowerCase() === 'true') {
      (async () => {
        try {
          const addrs = await getAddresses();
          for (const addr of addrs) {
            const valid = await countValidTradesForAddress(addr);
            if (valid > 0) continue;
            await deleteTradesForAddress(addr);
            const fills = await fetchUserBtcFills(addr, { aggregateByTime: true });
            let inserted = 0;
            for (const f of (fills || []).slice(0, 100)) {
              const delta = f.side === 'B' ? +f.sz : -f.sz;
              const newPos = f.startPosition + delta;
              let action = '';
              if (f.startPosition === 0) action = delta > 0 ? 'Open Long' : 'Open Short';
              else if (f.startPosition > 0) action = delta > 0 ? 'Increase Long' : (newPos === 0 ? 'Close Long' : 'Decrease Long');
              else action = delta < 0 ? 'Increase Short' : (newPos === 0 ? 'Close Short' : 'Decrease Short');
              const payload = {
                at: new Date(f.time).toISOString(),
                address: addr,
                symbol: 'BTC',
                action,
                size: Math.abs(f.sz),
                startPosition: f.startPosition,
                priceUsd: f.px,
                realizedPnlUsd: f.closedPnl ?? null,
                fee: f.fee ?? null,
                feeToken: f.feeToken ?? null,
              hash: f.hash ?? null,
            };
              const result = await insertTradeIfNew(addr, payload);
              if (result.inserted) inserted += 1;
            }
            if (inserted > 0) console.log(`[startup] Backfilled ${inserted} trade(s) for ${addr}`);
          }
        } catch (e) {
          console.warn('[startup] cleanup/backfill failed', e);
        }
      })();
    }

    // Periodic data integrity check (light backfill) for all addresses
    const integrityEnabled = String(process.env.INTEGRITY_CHECK_ENABLED || 'true').toLowerCase() === 'true';
    const integrityInterval = Math.max(60000, Number(process.env.INTEGRITY_CHECK_INTERVAL_MS || 900000));
    if (integrityEnabled) {
      setInterval(async () => {
        try {
          const addrs = await getAddresses();
          for (const addr of addrs) {
            await backfillRecentForAddress(addr, 100);
          }
        } catch (e) {
          console.warn('[integrity] check failed', e);
        }
      }, integrityInterval);
      console.log(`[integrity] enabled; interval ${integrityInterval}ms`);
    }
  })
  .catch((e) => {
    console.error('[server] failed to init storage', e);
    process.exit(1);
  });
// Clear ALL trades then backfill latest fills for every address (drives single Refresh All button in UI)
app.post('/api/clear-and-backfill-all', async (_req, res) => {
  try {
    const deleted = await deleteAllTrades();
    changes.reset();
    const addrs = await getAddresses();
    const perAddress: Record<string, { inserted: number }> = {};
    for (const addr of addrs) {
      const inserted = await backfillRecentForAddress(addr, 300);
      perAddress[addr] = { inserted };
    }
    res.json({ ok: true, deleted, perAddress });
  } catch (e) {
    res.status(500).json({ ok: false, error: 'clear-and-backfill failed' });
  }
});

// Clear ALL trades only (no backfill) for manual purge via UI button
app.post('/api/clear-all-trades', async (_req, res) => {
  try {
    const deleted = await deleteAllTrades();
    changes.reset();
    res.json({ ok: true, deleted });
  } catch (e) {
    res.status(500).json({ ok: false, error: 'clear-all-trades failed' });
  }
});
