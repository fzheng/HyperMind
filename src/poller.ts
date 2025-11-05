import { fetchBtcPriceUsd } from './price';
import { fetchBtcPerpExposure } from './hyperliquid';
import { computeRecommendation } from './recommend';
import type { Address, Recommendation } from './types';

export interface PollerOpts {
  intervalMs?: number;
}

export class Poller {
  private timer: NodeJS.Timeout | null = null;
  private intervalMs: number;
  private getAddresses: () => Promise<Address[]>;
  private setRecommendations: (recs: Recommendation[]) => void;
  private getRecommendations: () => Recommendation[];

  constructor(
    getAddresses: () => Promise<Address[]>,
    setRecommendations: (recs: Recommendation[]) => void,
    getRecommendations: () => Recommendation[],
    opts?: PollerOpts
  ) {
    this.intervalMs = opts?.intervalMs ?? 90_000;
    this.getAddresses = getAddresses;
    this.setRecommendations = setRecommendations;
    this.getRecommendations = getRecommendations;
  }

  start() {
    if (this.timer) return;
    console.log(`[poller] Starting with interval ${this.intervalMs}ms`);
    // Kick off immediately, then on interval
    this.runOnce().catch((e) => console.error('[poller] initial run error', e));
    this.timer = setInterval(() => {
      this.runOnce().catch((e) => console.error('[poller] run error', e));
    }, this.intervalMs);
  }

  stop() {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
  }

  async trigger() {
    await this.runOnce();
  }

  private async runOnce() {
    const addresses = await this.getAddresses();
    if (addresses.length === 0) return;

    console.log(`[poller] Polling ${addresses.length} address(es)`);

    let priceUsd: number | null = null;
    try {
      const px = await fetchBtcPriceUsd();
      priceUsd = px.price;
    } catch (e) {
      console.warn('[poller] price fetch failed:', e);
    }

    const prev = this.getRecommendations();
    const next: Recommendation[] = [];

    for (const addr of addresses) {
      try {
        const exposure = await fetchBtcPerpExposure(addr);
        const p = priceUsd ?? prev.find((r) => r.address === addr)?.priceUsd ?? 0;
        const rec = computeRecommendation(addr, exposure, p);
        next.push(rec);
      } catch (e) {
        console.warn(`[poller] address ${addr} failed`, e);
        const fallback = prev.find((r) => r.address === addr);
        if (fallback) next.push({ ...fallback, timestamp: new Date().toISOString() });
      }
    }

    this.setRecommendations(next);
  }
}
