import * as hl from '@nktkas/hyperliquid';
import { clearinghouseState, userFills, userDetails, metaAndAssetCtxs } from '@nktkas/hyperliquid/api/info';
import type { PositionInfo } from './types';

// Reuse a single HTTP transport for SDK calls
const transport = new hl.HttpTransport();

// Internal helpers
function isBtcCoin(coin: unknown): boolean {
  return typeof coin === 'string' && /^btc$/i.test(coin);
}

function isEthCoin(coin: unknown): boolean {
  return typeof coin === 'string' && /^eth$/i.test(coin);
}

function isBtcOrEthCoin(coin: unknown): boolean {
  return isBtcCoin(coin) || isEthCoin(coin);
}

function toUser(address: string): `0x${string}` {
  return address.toLowerCase() as `0x${string}`;
}

/**
 * Returns net BTC perp exposure (signed size in coin units).
 * Falls back to 0 on API errors to keep callers resilient.
 */
export async function fetchBtcPerpExposure(address: string): Promise<number> {
  try {
    const data = await clearinghouseState(
      { transport },
      { user: toUser(address) }
    );
    const positions = data?.assetPositions ?? [];
    let netBtc = 0;
    for (const assetPosition of positions) {
      const pos = assetPosition?.position;
      const coin = pos?.coin ?? '';
      const size = Number(pos?.szi ?? 0);
      if (isBtcCoin(coin) && Number.isFinite(size)) netBtc += size;
    }
    return netBtc;
  } catch {
    return 0;
  }
}

/**
 * Returns all non-flat perp positions for a user from the info API.
 * BTC is included here (for completeness) and symbol is uppercased.
 */
export async function fetchPerpPositions(address: string): Promise<PositionInfo[]> {
  try {
    const data = await clearinghouseState(
      { transport },
      { user: toUser(address) }
    );
    const out: PositionInfo[] = [];
    for (const assetPosition of data?.assetPositions ?? []) {
      const pos = assetPosition?.position;
      const coin = pos?.coin ?? '';
      const size = Number(pos?.szi ?? 0);
      if (!Number.isFinite(size) || size === 0) continue;
      const entry = Number(pos?.entryPx ?? NaN);
      const levValue = Number(pos?.leverage?.value ?? NaN);
      out.push({
        symbol: String(coin).toUpperCase(),
        size,
        entryPriceUsd: Number.isFinite(entry) ? entry : undefined,
        leverage: Number.isFinite(levValue) ? levValue : undefined,
      });
    }
    return out;
  } catch {
    return [];
  }
}

export interface UserFill {
  coin: string;
  px: number;
  sz: number;
  side: 'B' | 'A';
  time: number; // ms epoch
  startPosition: number;
  closedPnl?: number;
  fee?: number;
  feeToken?: string;
  hash?: string;
}

/**
 * Fetches recent user fills for BTC and ETH, newest first.
 * Values are normalized to numbers and optional fields omitted if invalid.
 */
export async function fetchUserFills(
  address: string,
  opts?: { aggregateByTime?: boolean; symbols?: ('BTC' | 'ETH')[] }
): Promise<UserFill[]> {
  const symbols = opts?.symbols ?? ['BTC', 'ETH'];
  try {
    const fills = await userFills(
      { transport },
      { user: toUser(address), aggregateByTime: opts?.aggregateByTime },
    );
    const out: UserFill[] = [];
    for (const f of fills || []) {
      const coin = (f as any)?.coin;
      if (!isBtcOrEthCoin(coin)) continue;
      const coinUpper = String(coin).toUpperCase();
      if (!symbols.includes(coinUpper as 'BTC' | 'ETH')) continue;

      const px = Number((f as any)?.px);
      const sz = Number((f as any)?.sz);
      const time = Number((f as any)?.time);
      const start = Number((f as any)?.startPosition);
      const closedRaw = (f as any)?.closedPnl;
      const feeRaw = (f as any)?.fee;
      const feeToken = typeof (f as any)?.feeToken === 'string' ? String((f as any).feeToken) : undefined;
      const hash = typeof (f as any)?.hash === 'string' ? String((f as any).hash) : undefined;
      const side = ((f as any)?.side === 'B' ? 'B' : 'A') as 'B' | 'A';

      if (!Number.isFinite(px) || !Number.isFinite(sz) || !Number.isFinite(time) || !Number.isFinite(start)) continue;

      const closed = Number.isFinite(Number(closedRaw)) ? Number(closedRaw) : undefined;
      const fee = Number.isFinite(Number(feeRaw)) ? Number(feeRaw) : undefined;

      out.push({
        coin: coinUpper,
        px,
        sz,
        side,
        time,
        startPosition: start,
        closedPnl: closed,
        fee,
        feeToken,
        hash,
      });
    }
    return out.sort((a, b) => b.time - a.time);
  } catch {
    return [];
  }
}

export interface UserProfileSummary {
  txCount: number;
  lastTxTime: string | null;
}

export async function fetchUserProfile(address: string): Promise<UserProfileSummary> {
  try {
    const data = await userDetails(
      { transport },
      { user: toUser(address) }
    );
    const txs = data?.txs ?? [];
    const last = txs[0];
    return {
      txCount: txs.length,
      lastTxTime: last?.time ? new Date(Number(last.time)).toISOString() : null
    };
  } catch {
    return { txCount: 0, lastTxTime: null };
  }
}

export async function fetchSpotPrice(symbol: 'BTCUSDT' | 'ETHUSDT'): Promise<number> {
  const url = `https://api.binance.com/api/v3/ticker/price?symbol=${symbol}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Ticker HTTP ${res.status}`);
  const data = (await res.json()) as { price?: string };
  const price = Number(data.price);
  if (!Number.isFinite(price)) throw new Error('Invalid ticker price');
  return price;
}

type MarkCache = {
  ts: number;
  prices: Record<string, number>;
};

let markCache: MarkCache | null = null;

async function loadMetaMarks(): Promise<MarkCache> {
  const [meta, ctxs] = await metaAndAssetCtxs({ transport });
  const prices: Record<string, number> = {};
  (meta?.universe || []).forEach((asset: any, idx: number) => {
    const name = String(asset?.name || '').toUpperCase();
    const ctx = ctxs?.[idx];
    const mark = Number(ctx?.markPx ?? ctx?.markPx ?? ctx?.midPx);
    if (Number.isFinite(mark)) prices[name] = mark;
  });
  const cache = { ts: Date.now(), prices };
  markCache = cache;
  return cache;
}

export async function fetchPerpMarkPrice(symbol: 'BTC' | 'ETH'): Promise<number> {
  const now = Date.now();
  if (!markCache || now - markCache.ts > 1500) {
    await loadMetaMarks();
  }
  const price = markCache?.prices?.[symbol.toUpperCase()];
  if (price == null) {
    await loadMetaMarks();
    return markCache?.prices?.[symbol.toUpperCase()] ?? 0;
  }
  return price;
}
