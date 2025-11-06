import express from 'express';
import cors from 'cors';
import path from 'path';
import { initStorage, listAddresses as storageList, addAddress as storageAdd, removeAddress as storageRemove, getNicknames as storageGetNicknames, setNickname as storageSetNickname } from './storage';
import { latestTrades, pageTrades, insertTradeIfNew, countValidTradesForAddress, deleteTradesForAddress } from './persist';
import type { Address, Recommendation } from './types';
import { Poller } from './poller';
import { fetchPerpPositions, fetchUserBtcFills } from './hyperliquid';
import { startPriceFeed, refreshPriceFeed, getCurrentBtcPrice } from './price';
import { EventQueue } from './queue';
import { RealtimeTracker } from './realtime';

const app = express();
const PORT = process.env.PORT ? Number(process.env.PORT) : 3000;
const POLL_INTERVAL_MS = process.env.POLL_INTERVAL_MS ? Number(process.env.POLL_INTERVAL_MS) : 90_000;

app.use(cors());
app.use(express.json());

// In-memory state mirrors persisted addresses and latest recommendations
let recommendations: Recommendation[] = [];
const changes = new EventQueue(5000);

async function getAddresses(): Promise<Address[]> {
  return await storageList();
}
function setRecommendations(recs: Recommendation[]) {
  recommendations = recs;
}
function getRecommendations(): Recommendation[] {
  return recommendations;
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
    const ok = await insertTradeIfNew(addr, payload);
    if (ok) inserted += 1;
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
    poller.trigger().catch((e) => console.warn('[api] immediate poll failed', e));
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

app.get('/api/recommendations', (_req, res) => {
  res.json({ recommendations: getRecommendations() });
});

app.get('/api/price', (_req, res) => {
  res.json(getCurrentBtcPrice());
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

// Latest trade fills (BTC only) across addresses
app.get('/api/latest-trades', async (req, res) => {
  const limit = Math.min(200, Math.max(1, Number(req.query.limit ?? 100)));
  const beforeIdRaw = req.query.beforeId;
  const beforeId = beforeIdRaw != null ? Number(beforeIdRaw) : null;
  const address = typeof req.query.address === 'string' ? String(req.query.address).toLowerCase() : null;
  try {
    // If requesting trades for a specific address, proactively refresh from Info API
    if (address && !beforeId) {
      try { await backfillRecentForAddress(address, 100); } catch {}
    }
    const rows = await pageTrades({ limit, beforeId, address });
    const trades = rows.map((r) => {
      const p = r.payload || {};
      const closedPnl = p.realizedPnlUsd != null ? Number(p.realizedPnlUsd) : (p.closedPnl != null ? Number(p.closedPnl) : null);
      return {
        id: r.id,
        time: p.at || null,
        address: p.address || address || null,
        symbol: p.symbol || 'BTC',
        action: p.action || (p.side === 'buy' ? 'Buy' : 'Sell'),
        size: Number(p.size || 0),
        startPosition: p.startPosition != null ? Number(p.startPosition) : null,
        price: Number(p.priceUsd || p.px || 0),
        closedPnl,
        fee: p.fee != null ? Number(p.fee) : null,
        feeToken: p.feeToken || null,
        tx: p.hash || null,
      };
    });
    const nextBeforeId = trades.length > 0 ? Math.min(...rows.map(r => r.id)) : null;
    return res.json({ trades, nextBeforeId });
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
        const ok = await insertTradeIfNew(addr, payload);
        if (ok) inserted += 1;
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
  recommendations = recommendations.filter((r) => r.address.toLowerCase() !== addrParam);
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

// Trigger poll now
app.post('/api/poll-now', async (_req, res) => {
  try {
    await poller.trigger();
    res.json({ ok: true, at: new Date().toISOString() });
  } catch (e) {
    res.status(500).json({ ok: false, error: String(e) });
  }
});

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

// Poller
const poller = new Poller(getAddresses, setRecommendations, getRecommendations, {
  intervalMs: POLL_INTERVAL_MS
});
poller.start();

// Realtime subscriptions for BTC positions and trades
const realtime = new RealtimeTracker(getAddresses, changes);
realtime.start().catch((e) => console.warn('[realtime] start failed', e));

initStorage()
  .then(async () => {
    await startPriceFeed(getAddresses);
    app.listen(PORT, () => {
      console.log(`hlbot server listening on http://localhost:${PORT}`);
    });

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
              const ok = await insertTradeIfNew(addr, payload);
              if (ok) inserted += 1;
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
// Cleanup invalid trades and backfill up to 100 for each address without valid trades
app.post('/api/cleanup-and-backfill', async (_req, res) => {
  try {
    const addrs = await getAddresses();
    const results: Record<string, { deleted: number; inserted: number }> = {};
    for (const addr of addrs) {
      const valid = await countValidTradesForAddress(addr);
      if (valid > 0) { results[addr] = { deleted: 0, inserted: 0 }; continue; }
      const deleted = await deleteTradesForAddress(addr);
      const inserted = await backfillRecentForAddress(addr, 100);
      results[addr] = { deleted, inserted };
    }
    res.json({ ok: true, results });
  } catch (e) {
    res.status(500).json({ ok: false, error: 'cleanup failed' });
  }
});
