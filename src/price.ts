import { fetchWithRetry } from './utils';
import type { PriceInfo } from './types';

let currentPrice: number | null = null;
let updatedAt: string | null = null;
let priceSource: 'ws' | 'http' | 'unknown' = 'unknown';

let wsTransport: any = null;
let wsSub: any | null = null;
let activeUser: string | null = null;
let provider: (() => Promise<string[]>) | null = null;

function setPrice(p: number, source: 'ws' | 'http') {
  if (Number.isFinite(p)) {
    currentPrice = p;
    updatedAt = new Date().toISOString();
    priceSource = source;
  }
}

type SubscribeFn = (user: string, listener: (evt: any) => void) => Promise<{ unsubscribe?: () => Promise<void> } | null>;

async function defaultSubscribeWebData2(user: string, listener: (evt: any) => void) {
  const hl = await import('@nktkas/hyperliquid');
  const { webData2 } = await import('@nktkas/hyperliquid/api/subscription');
  const WS = (await import('ws')).default as any;
  if (!wsTransport) {
    wsTransport = new (hl as any).WebSocketTransport({ reconnect: { WebSocket: WS } });
  }
  return (webData2 as any)({ transport: wsTransport }, { user: user as `0x${string}` }, listener);
}

let subscribeWebData2Impl: SubscribeFn = defaultSubscribeWebData2;

async function startWs(user: string) {
  if (wsSub) {
    try { await wsSub.unsubscribe?.(); } catch {}
    wsSub = null;
  }
  activeUser = user;
  wsSub = await subscribeWebData2Impl(user, (evt) => {
    try {
      const meta = evt.meta; // MetaResponse
      const ctxs = evt.assetCtxs; // contexts array
      const idx = meta.universe.findIndex((u: any) => String(u?.name).toUpperCase() === 'BTC');
      if (idx >= 0 && ctxs[idx]?.markPx) {
        const px = Number(ctxs[idx].markPx);
        if (Number.isFinite(px)) setPrice(px, 'ws');
      }
    } catch {}
  });
}

export async function startPriceFeed(getAddresses: () => Promise<string[]>) {
  provider = getAddresses;
  const addrs = await getAddresses();
  const first = addrs?.[0];
  if (first && first !== activeUser) {
    await startWs(first);
  }
}

export async function refreshPriceFeed() {
  if (!provider) return;
  const addrs = await provider();
  const first = addrs?.[0] ?? null;
  if (first && first !== activeUser) {
    await startWs(first);
  }
  if (!first && wsSub) {
    try { await wsSub.unsubscribe?.(); } catch {}
    wsSub = null;
    activeUser = null;
  }
}

export function getCurrentBtcPrice(): { symbol: string; price: number | null; updatedAt: string | null; source: string } {
  return { symbol: 'BTCUSD', price: currentPrice, updatedAt, source: priceSource };
}

// Test-only hooks (no side effects in production)
export function __setSubscribeWebData2ForTest(fn: SubscribeFn) {
  subscribeWebData2Impl = fn;
}
export async function __resetPriceForTest() {
  currentPrice = null;
  updatedAt = null;
  priceSource = 'unknown';
  activeUser = null;
  provider = null;
  if (wsSub) {
    try { await wsSub.unsubscribe?.(); } catch {}
  }
  wsSub = null;
  wsTransport = null;
}

async function binance(): Promise<number> {
  const data = await fetchWithRetry<{ symbol: string; price: string }>(
    'https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT',
    { method: 'GET' },
    { retries: 1, baseDelayMs: 500 }
  );
  const price = Number(data.price);
  if (!Number.isFinite(price)) throw new Error('Invalid price from Binance');
  return price;
}

async function coinbase(): Promise<number> {
  const data = await fetchWithRetry<any>(
    'https://api.exchange.coinbase.com/products/BTC-USD/ticker',
    { method: 'GET' },
    { retries: 1, baseDelayMs: 500 }
  );
  const price = Number(data.price ?? data.last);
  if (!Number.isFinite(price)) throw new Error('Invalid price from Coinbase');
  return price;
}

async function coingecko(): Promise<number> {
  const data = await fetchWithRetry<any>(
    'https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd',
    { method: 'GET' },
    { retries: 1, baseDelayMs: 500 }
  );
  const price = Number(data?.bitcoin?.usd);
  if (!Number.isFinite(price)) throw new Error('Invalid price from CoinGecko');
  return price;
}

async function bitstamp(): Promise<number> {
  const data = await fetchWithRetry<any>(
    'https://www.bitstamp.net/api/v2/ticker/btcusd/',
    { method: 'GET' },
    { retries: 1, baseDelayMs: 500 }
  );
  const price = Number(data?.last);
  if (!Number.isFinite(price)) throw new Error('Invalid price from Bitstamp');
  return price;
}

async function kraken(): Promise<number> {
  const data = await fetchWithRetry<any>(
    'https://api.kraken.com/0/public/Ticker?pair=XXBTZUSD',
    { method: 'GET' },
    { retries: 1, baseDelayMs: 500 }
  );
  const result = data?.result ?? {};
  const key = Object.keys(result)[0];
  const priceStr = result?.[key]?.c?.[0];
  const price = Number(priceStr);
  if (!Number.isFinite(price)) throw new Error('Invalid price from Kraken');
  return price;
}

export async function fetchBtcPriceUsd(): Promise<PriceInfo> {
  // Prefer live WebSocket price if available
  if (currentPrice && Number.isFinite(currentPrice)) {
    return { symbol: 'BTCUSD', price: currentPrice };
  }
  const sources: Array<() => Promise<number>> = [binance, coinbase, coingecko, bitstamp, kraken];
  const errors: string[] = [];
  for (const s of sources) {
    try {
      const price = await s();
      setPrice(price, 'http');
      return { symbol: 'BTCUSD', price };
    } catch (e: any) {
      errors.push(String(e?.message ?? e));
      continue;
    }
  }
  throw new Error('All price sources failed: ' + errors.join(' | '));
}
