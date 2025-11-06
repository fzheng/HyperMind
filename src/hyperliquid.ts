import * as hl from '@nktkas/hyperliquid';
import { clearinghouseState, userFills } from '@nktkas/hyperliquid/api/info';
import type { PositionInfo } from './types';

// Reuse a single HTTP transport for SDK calls
const transport = new hl.HttpTransport();

export async function fetchBtcPerpExposure(address: string): Promise<number> {
  try {
    const data = await clearinghouseState(
      { transport },
      { user: address as `0x${string}` }
    );
    const positions = data.assetPositions || [];
    let netBtc = 0;
    for (const ap of positions) {
      const coin = ap?.position?.coin ?? '';
      const size = Number(ap?.position?.szi ?? 0);
      if (/^btc$/i.test(coin) && Number.isFinite(size)) {
        netBtc += size;
      }
    }
    return netBtc;
  } catch (e) {
    return 0;
  }
}

export async function fetchPerpPositions(address: string): Promise<PositionInfo[]> {
  try {
    const data = await clearinghouseState(
      { transport },
      { user: address as `0x${string}` }
    );
    const out: PositionInfo[] = [];
    for (const ap of data.assetPositions || []) {
      const coin = ap?.position?.coin ?? '';
      const size = Number(ap?.position?.szi ?? 0);
      if (!Number.isFinite(size) || size === 0) continue;
      const entry = Number(ap?.position?.entryPx ?? NaN);
      const levValue = Number(ap?.position?.leverage?.value ?? NaN);
      const symbol = coin; // e.g., BTC
      out.push({
        symbol,
        size,
        entryPriceUsd: Number.isFinite(entry) ? entry : undefined,
        leverage: Number.isFinite(levValue) ? levValue : undefined,
      });
    }
    return out;
  } catch (e) {
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

export async function fetchUserBtcFills(address: string, opts?: { aggregateByTime?: boolean }): Promise<UserFill[]> {
  try {
    const fills = await userFills(
      { transport },
      { user: address as `0x${string}`, aggregateByTime: opts?.aggregateByTime },
    );
    const out: UserFill[] = [];
    for (const f of fills || []) {
      if (!/^btc$/i.test(String(f?.coin))) continue;
      const px = Number(f?.px);
      const sz = Number(f?.sz);
      const time = Number(f?.time);
      const start = Number(f?.startPosition);
      const closed = f?.closedPnl != null ? Number(f?.closedPnl) : undefined;
      const fee = f?.fee != null ? Number(f?.fee) : undefined;
      const feeToken = typeof (f as any)?.feeToken === 'string' ? String((f as any).feeToken) : undefined;
      const hash = typeof (f as any)?.hash === 'string' ? String((f as any).hash) : undefined;
      const side = (f?.side === 'B' ? 'B' : 'A') as 'B' | 'A';
      if (!Number.isFinite(px) || !Number.isFinite(sz) || !Number.isFinite(time) || !Number.isFinite(start)) continue;
      out.push({ coin: 'BTC', px, sz, side, time, startPosition: start, closedPnl: Number.isFinite(closed!) ? closed : undefined, fee, feeToken, hash });
    }
    return out.sort((a, b) => b.time - a.time);
  } catch (e) {
    return [];
  }
}
