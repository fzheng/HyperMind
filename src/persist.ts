// Use dynamic import to avoid hard dependency in non-Postgres envs
let pool: any = null;

async function getPool(): Promise<any> {
  if (pool) return pool;
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const { Pool: PgPool } = require('pg');
  const connectionString = process.env.PG_CONNECTION_STRING || process.env.DATABASE_URL;
  pool = new PgPool({ connectionString });
  return pool;
}

export type InsertableEvent = {
  type: 'position' | 'trade';
  at: string; // ISO
  address: string;
  symbol: 'BTC';
  payload: any; // stored as JSON
};

export async function insertEvent(evt: InsertableEvent): Promise<number | null> {
  try {
    const p = await getPool();
    const { rows } = await p.query(
      'insert into hl_events (at, address, type, symbol, payload) values ($1,$2,$3,$4,$5) returning id',
      [evt.at, evt.address, evt.type, evt.symbol, evt.payload]
    );
    return rows?.[0]?.id ?? null;
  } catch (_e) {
    return null;
  }
}

export async function upsertCurrentPosition(args: {
  address: string;
  symbol: 'BTC';
  size: number;
  entryPriceUsd: number | null;
  liquidationPriceUsd: number | null;
  leverage: number | null;
  pnlUsd: number | null;
  updatedAt?: string; // ISO
}): Promise<void> {
  try {
    const p = await getPool();
    await p.query(
      `insert into hl_current_positions(address, symbol, size, entry_price, liquidation_price, leverage, pnl, updated_at)
       values ($1,$2,$3,$4,$5,$6,$7,$8)
       on conflict(address) do update set
         symbol=excluded.symbol,
         size=excluded.size,
         entry_price=excluded.entry_price,
         liquidation_price=excluded.liquidation_price,
         leverage=excluded.leverage,
         pnl=excluded.pnl,
         updated_at=excluded.updated_at` ,
      [
        args.address,
        args.symbol,
        args.size,
        args.entryPriceUsd,
        args.liquidationPriceUsd,
        args.leverage,
        args.pnlUsd,
        args.updatedAt || new Date().toISOString(),
      ]
    );
  } catch (_e) {
    // ignore
  }
}

export async function latestTrades(limit = 50): Promise<any[]> {
  try {
    const p = await getPool();
    const { rows } = await p.query(
      'select payload from hl_events where type = $1 order by id desc limit $2',
      ['trade', Math.max(1, Math.min(200, limit))]
    );
    return rows.map((r: any) => r.payload);
  } catch (_e) {
    return [];
  }
}

export async function insertTradeIfNew(address: string, payload: any): Promise<boolean> {
  try {
    const p = await getPool();
    const addr = address.toLowerCase();
    const hash = payload?.hash || payload?.tx || null;
    if (hash) {
      const { rows } = await p.query(
        "select id from hl_events where type = 'trade' and address = $1 and payload->>'hash' = $2 limit 1",
        [addr, String(hash)]
      );
      if (rows.length > 0) {
        // Update existing payload to the newer (e.g., aggregated) one
        await p.query('update hl_events set at = $1, payload = $2 where id = $3', [payload.at || new Date().toISOString(), payload, rows[0].id]);
        return true;
      }
    }
    await p.query(
      'insert into hl_events (at, address, type, symbol, payload) values ($1,$2,$3,$4,$5)',
      [payload.at || new Date().toISOString(), addr, 'trade', payload.symbol || 'BTC', payload]
    );
    return true;
  } catch (_e) {
    return false;
  }
}

export async function pageTrades(opts: { limit?: number; beforeId?: number | null; address?: string | null }): Promise<{ id: number; payload: any }[]> {
  const limit = Math.max(1, Math.min(200, opts.limit ?? 100));
  try {
    const p = await getPool();
    if (opts.address && opts.beforeId) {
      const { rows } = await p.query(
        'select id, payload from hl_events where type = $1 and address = $2 and id < $3 order by id desc limit $4',
        ['trade', opts.address.toLowerCase(), opts.beforeId, limit]
      );
      return rows as any[];
    } else if (opts.address) {
      const { rows } = await p.query(
        'select id, payload from hl_events where type = $1 and address = $2 order by id desc limit $3',
        ['trade', opts.address.toLowerCase(), limit]
      );
      return rows as any[];
    } else if (opts.beforeId) {
      const { rows } = await p.query(
        'select id, payload from hl_events where type = $1 and id < $2 order by id desc limit $3',
        ['trade', opts.beforeId, limit]
      );
      return rows as any[];
    } else {
      const { rows } = await p.query(
        'select id, payload from hl_events where type = $1 order by id desc limit $2',
        ['trade', limit]
      );
      return rows as any[];
    }
  } catch (_e) {
    return [];
  }
}

export async function countValidTradesForAddress(address: string): Promise<number> {
  try {
    const p = await getPool();
    const { rows } = await p.query(
      "select count(1)::int as c from hl_events where type = 'trade' and address = $1 and (payload ? 'startPosition') and (payload->>'startPosition') is not null",
      [address.toLowerCase()]
    );
    return Number(rows?.[0]?.c ?? 0);
  } catch (_e) {
    return 0;
  }
}

export async function deleteTradesForAddress(address: string): Promise<number> {
  try {
    const p = await getPool();
    const { rowCount } = await p.query(
      "delete from hl_events where type = 'trade' and address = $1",
      [address.toLowerCase()]
    );
    return rowCount ?? 0;
  } catch (_e) {
    return 0;
  }
}
