import * as hl from '@nktkas/hyperliquid';
import { clearinghouseState as infoClearinghouse } from '@nktkas/hyperliquid/api/info';
import { clearinghouseState as subClearinghouse, userEvents as subUserEvents } from '@nktkas/hyperliquid/api/subscription';
import { getCurrentBtcPrice } from './price';
import { EventQueue } from './queue';
import { insertEvent, upsertCurrentPosition, insertTradeIfNew } from './persist';

type Address = string;

export interface PositionSnapshot {
  size: number; // signed
  entryPriceUsd: number | null;
  liquidationPriceUsd: number | null;
  leverage: number | null;
}

function sideFromSize(size: number): 'long' | 'short' | 'flat' {
  if (size > 0) return 'long';
  if (size < 0) return 'short';
  return 'flat';
}

export class RealtimeTracker {
  private ws: any; // shared WebSocketTransport for clearinghouseState
  private http: any; // HttpTransport
  private wsImpl: any; // WS ctor from 'ws' (lazy)
  private subs: Map<Address, { ch?: any; ue?: any; ueTransport?: any }>; // subs + per-address UE transport
  private snapshots: Map<Address, { data: PositionSnapshot; updatedAt: string }>;
  private getAddresses: () => Promise<Address[]>;
  private q: EventQueue;
  private primeInflight: Map<Address, Promise<void>>;
  private lastPrimeAt: Map<Address, number>;

  constructor(getAddresses: () => Promise<Address[]>, queue: EventQueue) {
    this.getAddresses = getAddresses;
    this.q = queue;
    this.subs = new Map();
    this.snapshots = new Map();
    this.primeInflight = new Map();
    this.lastPrimeAt = new Map();
  }

  async start() {
    await this.ensureSharedTransports();
    await this.refresh();
  }

  private async ensureSharedTransports() {
    if (!this.wsImpl) {
      this.wsImpl = (await import('ws')).default as any;
    }
    if (!this.ws) {
      this.ws = new (hl as any).WebSocketTransport({ reconnect: { WebSocket: this.wsImpl } });
    }
    if (!this.http) {
      this.http = new (hl as any).HttpTransport();
    }
  }

  async stop() {
    for (const [, s] of this.subs) {
      try { await s.ch?.unsubscribe?.(); } catch {}
      try { await s.ue?.unsubscribe?.(); } catch {}
      try { await s.ueTransport?.close?.(); } catch {}
    }
    this.subs.clear();
  }

  async refresh() {
    const addrs = (await this.getAddresses()).map((a) => a.toLowerCase());
    const current = new Set(this.subs.keys());

    // Unsubscribe removed addresses
    for (const addr of current) {
      if (!addrs.includes(addr)) {
        const s = this.subs.get(addr);
        try { await s?.ch?.unsubscribe?.(); } catch {}
        try { await s?.ue?.unsubscribe?.(); } catch {}
        try { await s?.ueTransport?.close?.(); } catch {}
        this.subs.delete(addr);
        this.snapshots.delete(addr);
      }
    }

    // Subscribe new addresses
    for (const addr of addrs) {
      if (!this.subs.has(addr)) {
        await this.subscribeAddress(addr);
      }
    }
  }

  private async subscribeAddress(addr: Address) {
    await this.ensureSharedTransports();
    const user = addr as `0x${string}`;
    const subs: { ch?: any; ue?: any; ueTransport?: any } = {};

    // clearinghouseState: position snapshots and updates
    try {
      subs.ch = await subClearinghouse(
        { transport: this.ws },
        { user },
        (evt: any) => this.onClearinghouse(addr, evt)
      );
    } catch (e) {
      // eslint-disable-next-line no-console
      console.warn('[realtime] clearinghouse sub failed for', addr, e);
    }

    // userEvents: fills/trades
    try {
      const ueTransport = new (hl as any).WebSocketTransport({ reconnect: { WebSocket: this.wsImpl } });
      subs.ueTransport = ueTransport;
      subs.ue = await subUserEvents(
        { transport: ueTransport },
        { user },
        (evt: any) => this.onUserEvents(addr, evt)
      );
    } catch (e) {
      try { await subs.ueTransport?.close?.(); } catch {}
      // eslint-disable-next-line no-console
      console.warn('[realtime] userEvents sub failed for', addr, e);
    }

    this.subs.set(addr, subs);
    if (!this.snapshots.has(addr)) {
      void this.primeFromHttp(addr);
    }
  }

  private onClearinghouse(addr: Address, evt: any) {
    try {
      const positions = evt?.clearinghouseState?.assetPositions || [];
      let btc: any | null = null;
      for (const ap of positions as any[]) {
        const coin = (ap as any)?.position?.coin ?? '';
        if (typeof coin === 'string' && /^btc$/i.test(coin)) { btc = ap; break; }
      }
      const szi = Number(btc?.position?.szi ?? 0);
      const entry = Number(btc?.position?.entryPx ?? NaN);
      const levValue = Number(btc?.position?.leverage?.value ?? NaN);
      const liq = Number(btc?.position?.liquidationPx ?? NaN);

      const snapshot: PositionSnapshot = {
        size: Number.isFinite(szi) ? szi : 0,
        entryPriceUsd: Number.isFinite(entry) ? entry : null,
        liquidationPriceUsd: Number.isFinite(liq) ? liq : null,
        leverage: Number.isFinite(levValue) ? levValue : null,
      };

      const prev = this.snapshots.get(addr)?.data;
      const changed = !prev
        || prev.size !== snapshot.size
        || prev.entryPriceUsd !== snapshot.entryPriceUsd
        || prev.liquidationPriceUsd !== snapshot.liquidationPriceUsd
        || prev.leverage !== snapshot.leverage;

      if (changed) {
        const updatedAt = new Date().toISOString();
        this.snapshots.set(addr, { data: snapshot, updatedAt });
        const mark = (getCurrentBtcPrice().price ?? null) as number | null;
        const pnl = (snapshot.entryPriceUsd != null && mark != null)
          ? snapshot.size * (mark - snapshot.entryPriceUsd)
          : null;
        const evt = this.q.push({
          type: 'position',
          at: updatedAt,
          address: addr,
          symbol: 'BTC',
          size: snapshot.size,
          side: sideFromSize(snapshot.size),
          entryPriceUsd: snapshot.entryPriceUsd,
          liquidationPriceUsd: snapshot.liquidationPriceUsd,
          leverage: snapshot.leverage,
          pnlUsd: pnl,
        });
        void insertEvent({ type: 'position', at: evt.at, address: addr, symbol: 'BTC', payload: evt });
        void upsertCurrentPosition({
          address: addr,
          symbol: 'BTC',
          size: snapshot.size,
          entryPriceUsd: snapshot.entryPriceUsd,
          liquidationPriceUsd: snapshot.liquidationPriceUsd,
          leverage: snapshot.leverage,
          pnlUsd: pnl,
          updatedAt,
        });
      }
    } catch (e) {
      // eslint-disable-next-line no-console
      console.warn('[realtime] clearinghouse handler error', e);
    }
  }

  private async onUserEvents(addr: Address, evt: any) {
    try {
      // We care about FillEvent variant: { fills: [...] }
      if (!evt || !('fills' in evt)) return;
      const fills: any[] = Array.isArray(evt.fills) ? evt.fills : [];
      let touched = false;
      for (const f of fills) {
        const coin = f?.coin ?? '';
        if (!/^btc$/i.test(String(coin))) continue;
        const px = Number(f?.px ?? NaN);
        const sz = Number(f?.sz ?? NaN);
        const side = f?.side === 'B' ? 'buy' : 'sell';
        const startPosition = Number(f?.startPosition ?? NaN);
        const hash = typeof f?.hash === 'string' ? String(f.hash) : undefined;
        const fee = f?.fee != null ? Number(f.fee) : undefined;
        const feeToken = typeof f?.feeToken === 'string' ? String(f.feeToken) : undefined;
        if (!Number.isFinite(px) || !Number.isFinite(sz) || !Number.isFinite(startPosition)) continue;

        // Signed delta based on side
        const delta = side === 'buy' ? +sz : -sz;
        const newPos = startPosition + delta;
        const effect: 'open' | 'close' = Math.abs(newPos) > Math.abs(startPosition) ? 'open' : (newPos === 0 ? 'close' : 'close');
        const direction = sideFromSize(newPos) === 'flat' ? (delta > 0 ? 'long' : (delta < 0 ? 'short' : 'flat')) : sideFromSize(newPos);
        const realizedPnl = Number(f?.closedPnl ?? NaN);

        const at = new Date((Number(f?.time) || Date.now())).toISOString();
        // Derive action label
        let actionLabel = '';
        if (startPosition === 0) actionLabel = delta > 0 ? 'Open Long' : 'Open Short';
        else if (startPosition > 0) {
          if (delta > 0) actionLabel = 'Increase Long';
          else actionLabel = newPos === 0 ? 'Close Long' : 'Decrease Long';
        } else if (startPosition < 0) {
          if (delta < 0) actionLabel = 'Increase Short';
          else actionLabel = newPos === 0 ? 'Close Short' : 'Decrease Short';
        }

        const persistencePayload = {
          at,
          address: addr,
          symbol: 'BTC',
          action: actionLabel,
          size: Math.abs(sz),
          startPosition,
          priceUsd: px,
          realizedPnlUsd: Number.isFinite(realizedPnl) ? realizedPnl : null,
          fee,
          feeToken,
          hash,
        };
        const persistResult = await insertTradeIfNew(addr, persistencePayload);
        const evt = this.q.push({
          type: 'trade',
          at,
          address: addr,
          symbol: 'BTC',
          side,
          direction,
          effect,
          priceUsd: px,
          size: Math.abs(sz),
          realizedPnlUsd: Number.isFinite(realizedPnl) ? realizedPnl : undefined,
          startPosition,
          fee,
          feeToken,
          hash,
          action: actionLabel,
          dbId: persistResult.id ?? undefined,
        });
        touched = true;
      }
      if (touched) {
        void this.primeFromHttp(addr, { force: false, minIntervalMs: 2000 });
      }
    } catch (e) {
      // eslint-disable-next-line no-console
      console.warn('[realtime] userEvents handler error', e);
    }
  }

  getAllSnapshots(): Array<{
    address: string;
    symbol: 'BTC';
    size: number;
    side: 'long' | 'short' | 'flat';
    entryPriceUsd: number | null;
    liquidationPriceUsd: number | null;
    leverage: number | null;
    pnlUsd: number | null;
    updatedAt: string;
  }> {
    const mark = (getCurrentBtcPrice().price ?? null) as number | null;
    const out: Array<{
      address: string;
      symbol: 'BTC';
      size: number;
      side: 'long' | 'short' | 'flat';
      entryPriceUsd: number | null;
      liquidationPriceUsd: number | null;
      leverage: number | null;
      pnlUsd: number | null;
      updatedAt: string;
    }> = [];
    for (const [address, { data, updatedAt }] of this.snapshots.entries()) {
      const pnl = (data.entryPriceUsd != null && mark != null)
        ? data.size * (mark - data.entryPriceUsd)
        : null;
      out.push({
        address,
        symbol: 'BTC',
        size: data.size,
        side: sideFromSize(data.size),
        entryPriceUsd: data.entryPriceUsd,
        liquidationPriceUsd: data.liquidationPriceUsd,
        leverage: data.leverage,
        pnlUsd: pnl,
        updatedAt,
      });
    }
    // sort by address for stable output
    out.sort((a, b) => a.address.localeCompare(b.address));
    return out;
  }

  // Immediate prime via HTTP info API for newly added addresses
  async primeFromHttp(addr: Address, opts?: { force?: boolean; minIntervalMs?: number }): Promise<void> {
    const { force = true, minIntervalMs = 0 } = opts || {};
    const inflight = this.primeInflight.get(addr);
    if (inflight) return inflight;
    if (!force && minIntervalMs > 0) {
      const last = this.lastPrimeAt.get(addr) ?? 0;
      if (Date.now() - last < minIntervalMs) return Promise.resolve();
    }
    const task = this.performPrime(addr).finally(() => {
      this.lastPrimeAt.set(addr, Date.now());
      this.primeInflight.delete(addr);
    });
    this.primeInflight.set(addr, task);
    return task;
  }

  private async performPrime(addr: Address): Promise<void> {
    try {
      if (!this.http) {
        this.http = new (hl as any).HttpTransport();
      }
      const user = addr as `0x${string}`;
      const data = await infoClearinghouse(
        { transport: this.http },
        { user }
      );
      const positions = data.assetPositions || [];
      let btc: any | null = null;
      for (const ap of positions as any[]) {
        const coin = (ap as any)?.position?.coin ?? '';
        if (typeof coin === 'string' && /^btc$/i.test(coin)) { btc = ap; break; }
      }
      const szi = Number(btc?.position?.szi ?? 0);
      const entry = Number(btc?.position?.entryPx ?? NaN);
      const levValue = Number(btc?.position?.leverage?.value ?? NaN);
      const liq = Number(btc?.position?.liquidationPx ?? NaN);
      const snapshot: PositionSnapshot = {
        size: Number.isFinite(szi) ? szi : 0,
        entryPriceUsd: Number.isFinite(entry) ? entry : null,
        liquidationPriceUsd: Number.isFinite(liq) ? liq : null,
        leverage: Number.isFinite(levValue) ? levValue : null,
      };
      const updatedAt = new Date().toISOString();
      this.snapshots.set(addr, { data: snapshot, updatedAt });
      const mark = (getCurrentBtcPrice().price ?? null) as number | null;
      const pnl = (snapshot.entryPriceUsd != null && mark != null)
        ? snapshot.size * (mark - snapshot.entryPriceUsd)
        : null;
      const evt = this.q.push({
        type: 'position',
        at: updatedAt,
        address: addr,
        symbol: 'BTC',
        size: snapshot.size,
        side: sideFromSize(snapshot.size),
        entryPriceUsd: snapshot.entryPriceUsd,
        liquidationPriceUsd: snapshot.liquidationPriceUsd,
        leverage: snapshot.leverage,
        pnlUsd: pnl,
      });
      void insertEvent({ type: 'position', at: evt.at, address: addr, symbol: 'BTC', payload: evt });
      void upsertCurrentPosition({
        address: addr,
        symbol: 'BTC',
        size: snapshot.size,
        entryPriceUsd: snapshot.entryPriceUsd,
        liquidationPriceUsd: snapshot.liquidationPriceUsd,
        leverage: snapshot.leverage,
        pnlUsd: pnl,
        updatedAt,
      });
    } catch (_e) {
      // ignore
    }
  }

  async ensureFreshSnapshots(maxAgeMs = 60000): Promise<void> {
    try {
      const addrs = (await this.getAddresses()).map((a) => a.toLowerCase());
      const now = Date.now();
      const tasks: Promise<void>[] = [];
      for (const addr of addrs) {
        const snap = this.snapshots.get(addr);
        const updatedMs = snap?.updatedAt ? Date.parse(snap.updatedAt) : NaN;
        if (!snap || !Number.isFinite(updatedMs) || now - updatedMs > maxAgeMs) {
          tasks.push(this.primeFromHttp(addr, { force: true }));
        }
      }
      if (tasks.length) await Promise.allSettled(tasks);
    } catch {
      // best-effort safeguard; ignore failures
    }
  }
}
