import express from 'express';
import cors from 'cors';
import path from 'path';
import { initStorage, listAddresses as storageList, addAddress as storageAdd, removeAddress as storageRemove } from './storage';
import type { Address, Recommendation } from './types';
import { Poller } from './poller';
import { fetchPerpPositions } from './hyperliquid';

const app = express();
const PORT = process.env.PORT ? Number(process.env.PORT) : 3000;
const POLL_INTERVAL_MS = process.env.POLL_INTERVAL_MS ? Number(process.env.POLL_INTERVAL_MS) : 90_000;

app.use(cors());
app.use(express.json());

// In-memory state mirrors persisted addresses and latest recommendations
let recommendations: Recommendation[] = [];

async function getAddresses(): Promise<Address[]> {
  return await storageList();
}
function setRecommendations(recs: Recommendation[]) {
  recommendations = recs;
}
function getRecommendations(): Recommendation[] {
  return recommendations;
}

// API routes
app.get('/api/addresses', async (_req, res) => {
  const addrs = await getAddresses();
  res.json({ addresses: addrs });
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
  }
  res.json({ addresses: await getAddresses() });
});

app.get('/api/recommendations', (_req, res) => {
  res.json({ recommendations: getRecommendations() });
});

// Remove address
app.delete('/api/addresses/:address', async (req, res) => {
  const addrParam = String(req.params.address || '').trim().toLowerCase();
  if (!addrParam) return res.status(400).json({ error: 'Invalid address' });
  await storageRemove(addrParam);
  console.log(`[api] Removed address ${addrParam}`);
  recommendations = recommendations.filter((r) => r.address.toLowerCase() !== addrParam);
  res.json({ addresses: await getAddresses() });
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

initStorage()
  .then(() => {
    app.listen(PORT, () => {
      console.log(`hlbot server listening on http://localhost:${PORT}`);
    });
  })
  .catch((e) => {
    console.error('[server] failed to init storage', e);
    process.exit(1);
  });
