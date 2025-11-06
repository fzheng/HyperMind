export type ChangeEvent =
  | {
      type: 'position';
      seq: number;
      at: string;
      address: string;
      symbol: 'BTC';
      size: number; // signed
      side: 'long' | 'short' | 'flat';
      entryPriceUsd: number | null;
      liquidationPriceUsd: number | null;
      pnlUsd: number | null;
      leverage: number | null;
    }
  | {
      type: 'trade';
      seq: number;
      at: string;
      address: string;
      symbol: 'BTC';
      side: 'buy' | 'sell';
      direction: 'long' | 'short' | 'flat';
      effect: 'open' | 'close';
      priceUsd: number;
      size: number; // absolute
      realizedPnlUsd?: number;
    };

export class EventQueue {
  private capacity: number;
  private buffer: ChangeEvent[] = [];
  private nextSeq = 1;

  constructor(capacity = 5000) {
    this.capacity = Math.max(100, capacity);
  }

  push<T extends Omit<ChangeEvent, 'seq'>>(evt: T): ChangeEvent {
    const withSeq = { ...(evt as any), seq: this.nextSeq++ } as ChangeEvent;
    this.buffer.push(withSeq);
    if (this.buffer.length > this.capacity) {
      this.buffer.splice(0, this.buffer.length - this.capacity);
    }
    return withSeq;
  }

  listSince(sinceSeq: number, limit = 200): ChangeEvent[] {
    const startIdx = this.buffer.findIndex((e) => e.seq > sinceSeq);
    if (startIdx === -1) return [];
    return this.buffer.slice(startIdx, startIdx + Math.max(1, Math.min(1000, limit)));
  }

  latestSeq(): number {
    return this.nextSeq - 1;
  }
}

