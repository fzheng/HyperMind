import type { Pool } from 'pg';
import { getPool as getSharedPool } from './postgres';

let poolPromise: Promise<Pool> | null = null;

async function getPool(): Promise<Pool> {
  if (!poolPromise) {
    poolPromise = getSharedPool();
  }
  return poolPromise;
}

export type InsertableEvent = {
  type: 'position' | 'trade';
  at: string; // ISO
  address: string;
  symbol: 'BTC';
  payload: Record<string, unknown>; // stored as JSON
};

export async function insertEvent(evt: InsertableEvent): Promise<number | null> {
  try {
    const p = await getPool();
    const { rows } = await p.query(
      'insert into hl_events (at, address, type, symbol, payload) values ($1,$2,$3,$4,$5) returning id',
      [evt.at, evt.address, evt.type, evt.symbol, evt.payload]
    );
    return rows?.[0]?.id ?? null;
  } catch (e) {
    console.error('[persist] insertEvent failed:', { type: evt.type, address: evt.address, error: e });
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
  } catch (e) {
    console.error('[persist] upsertCurrentPosition failed:', { address: args.address, error: e });
  }
}

export async function latestTrades(limit = 50): Promise<Record<string, unknown>[]> {
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

// Time-based pagination (preferred for chronological ordering). Optional beforeAt ISO cursor.
export async function pageTradesByTime(opts: { limit?: number; beforeAt?: string | null; beforeId?: number | null; address?: string | null }): Promise<{ id: number; address: string; at: string; payload: Record<string, unknown> }[]> {
  const limit = Math.max(1, Math.min(500, opts.limit ?? 100));
  try {
    const p = await getPool();
    const clauses: string[] = ["type = 'trade'"]; const params: any[] = []; let idx = 1;
    if (opts.address) { clauses.push(`address = $${idx++}`); params.push(String(opts.address).toLowerCase()); }
    if (opts.beforeAt && opts.beforeId != null) {
      clauses.push(`(at < $${idx} OR (at = $${idx} AND id < $${idx + 1}))`);
      params.push(opts.beforeAt, opts.beforeId);
      idx += 2;
    } else if (opts.beforeAt) {
      clauses.push(`at < $${idx++}`);
      params.push(opts.beforeAt);
    } else if (opts.beforeId != null) {
      clauses.push(`id < $${idx++}`);
      params.push(opts.beforeId);
    }
    const where = clauses.length ? 'where ' + clauses.join(' and ') : '';
    params.push(limit);
    const sql = `select id, address, at, payload from hl_events ${where} order by at desc, id desc limit $${idx}`;
    const { rows } = await p.query(sql, params);
    return rows as any[];
  } catch (e) {
    console.error('[persist] pageTradesByTime failed:', e);
    return [];
  }
}

export async function deleteAllTrades(): Promise<number> {
  try {
    const p = await getPool();
    const { rowCount } = await p.query("delete from hl_events where type = 'trade'");
    return rowCount ?? 0;
  } catch (_e) {
    return 0;
  }
}

export interface InsertTradeResult {
  id: number | null;
  inserted: boolean;
}

export async function insertTradeIfNew(address: string, payload: Record<string, unknown>): Promise<InsertTradeResult> {
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
        const targetId = Number(rows[0].id);
        await p.query('update hl_events set at = $1, payload = $2 where id = $3', [payload.at || new Date().toISOString(), payload, targetId]);
        return { id: targetId, inserted: false };
      }
    }
    const { rows } = await p.query(
      'insert into hl_events (at, address, type, symbol, payload) values ($1,$2,$3,$4,$5) returning id',
      [payload.at || new Date().toISOString(), addr, 'trade', payload.symbol || 'BTC', payload]
    );
    return { id: rows?.[0]?.id ?? null, inserted: true };
  } catch (_e) {
    return { id: null, inserted: false };
  }
}

export async function pageTrades(opts: { limit?: number; beforeId?: number | null; address?: string | null }): Promise<{ id: number; payload: Record<string, unknown> }[]> {
  const limit = Math.max(1, Math.min(200, opts.limit ?? 100));
  try {
    const p = await getPool();
    const clauses: string[] = ["type = 'trade'"];
    const params: any[] = [];
    let idx = 1;

    if (opts.address) {
      clauses.push(`address = $${idx++}`);
      params.push(opts.address.toLowerCase());
    }
    if (opts.beforeId != null) {
      clauses.push(`id < $${idx++}`);
      params.push(opts.beforeId);
    }

    params.push(limit);
    const where = clauses.join(' and ');
    const sql = `select id, payload from hl_events where ${where} order by id desc limit $${idx}`;
    const { rows } = await p.query(sql, params);
    return rows as any[];
  } catch (e) {
    console.error('[persist] pageTrades failed:', e);
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

export interface AddressPerformance {
  address: string;
  trades: number;
  wins: number;
  winRate: number;
  pnl7d: number;
  avgSize: number;
  efficiency: number;
}

export async function getAddressPerformance(days = 7): Promise<AddressPerformance[]> {
  try {
    const p = await getPool();
    const { rows } = await p.query(
      `
        select
          address,
          count(*) as trades,
          sum(case when (payload->>'realizedPnlUsd')::numeric > 0 then 1 else 0 end) as wins,
          coalesce(sum((payload->>'realizedPnlUsd')::numeric),0) as pnl_total,
          coalesce(sum(case when at >= now() - $1::interval then (payload->>'realizedPnlUsd')::numeric else 0 end),0) as pnl_7d,
          coalesce(avg((payload->>'size')::numeric),0) as avg_size
        from hl_events
        where type = 'trade'
        group by address
        order by pnl_7d desc, address asc
      `,
      [`${Math.max(1, days)} days`]
    );
    return rows.map((row: any) => {
      const trades = Number(row.trades) || 0;
      const wins = Number(row.wins) || 0;
      const pnlTotal = Number(row.pnl_total) || 0;
      const efficiency = trades > 0 ? pnlTotal / trades : pnlTotal;
      return {
        address: row.address,
        trades,
        wins,
        winRate: trades ? wins / trades : 0,
        pnl7d: Number(row.pnl_7d) || 0,
        avgSize: Number(row.avg_size) || 0,
        efficiency,
      };
    });
  } catch (_e) {
    return [];
  }
}

export interface RecentFill {
  id: number;
  address: string;
  at: string;
  side: 'buy' | 'sell';
  size: number;
  priceUsd: number;
  realizedPnlUsd: number | null;
  action?: string | null;
}

export async function listRecentFills(limit = 25): Promise<RecentFill[]> {
  try {
    const p = await getPool();
    const { rows } = await p.query(
      `
        select id, address, at, payload
        from hl_events
        where type = 'trade'
        order by at desc, id desc
        limit $1
      `,
      [Math.max(1, Math.min(200, limit))]
    );
    return rows.map((row: any) => ({
      id: Number(row.id),
      address: row.address,
      at: row.at,
      side: ((row.payload?.side === 'sell') ? 'sell' : 'buy') as 'buy' | 'sell',
      size: Number(row.payload?.size ?? row.payload?.payload?.size ?? 0),
      priceUsd: Number(row.payload?.priceUsd ?? row.payload?.price ?? 0),
      realizedPnlUsd: row.payload?.realizedPnlUsd != null ? Number(row.payload?.realizedPnlUsd) : null,
      action: row.payload?.action ?? null,
    }));
  } catch (_e) {
    return [];
  }
}

export interface LiveFill {
  time_utc: string;
  address: string;
  action: string;
  size_signed: number | null;
  previous_position: number | null;
  price_usd: number | null;
  closed_pnl_usd: number | null;
  tx_hash: string | null;
  symbol?: string | null;
  fee?: number | null;
  fee_token?: string | null;
}

export async function listLiveFills(limit = 25): Promise<LiveFill[]> {
  const safeLimit = Math.max(1, Math.min(200, limit));
  try {
    const p = await getPool();
    const { rows } = await p.query(
      `
        SELECT
          COALESCE((payload->>'at')::timestamptz, at) AS time_utc,
          address,
          payload->>'action' AS action,
          CASE payload->>'action'
            WHEN 'Increase Long'  THEN (payload->>'size')::numeric
            WHEN 'Decrease Short' THEN (payload->>'size')::numeric
            WHEN 'Close Short'    THEN (payload->>'size')::numeric
            WHEN 'Decrease Long'  THEN -(payload->>'size')::numeric
            WHEN 'Close Long'     THEN -(payload->>'size')::numeric
            WHEN 'Increase Short' THEN -(payload->>'size')::numeric
            ELSE (payload->>'size')::numeric
          END AS size_signed,
          (payload->>'startPosition')::numeric AS previous_position,
          (payload->>'priceUsd')::numeric      AS price_usd,
          (payload->>'realizedPnlUsd')::numeric AS closed_pnl_usd,
          payload->>'hash'                     AS tx_hash,
          payload->>'symbol'                   AS symbol,
          (payload->>'fee')::numeric           AS fee,
          payload->>'feeToken'                 AS fee_token
        FROM hl_events
        WHERE type = 'trade'
        ORDER BY time_utc DESC, id DESC
        LIMIT $1
      `,
      [safeLimit]
    );
    return rows.map((row: any) => ({
      time_utc: row.time_utc,
      address: row.address,
      action: row.action,
      size_signed: row.size_signed != null ? Number(row.size_signed) : null,
      previous_position: row.previous_position != null ? Number(row.previous_position) : null,
      price_usd: row.price_usd != null ? Number(row.price_usd) : null,
      closed_pnl_usd: row.closed_pnl_usd != null ? Number(row.closed_pnl_usd) : null,
      tx_hash: row.tx_hash || null,
      symbol: row.symbol || null,
      fee: row.fee != null ? Number(row.fee) : null,
      fee_token: row.fee_token || null,
    }));
  } catch (_e) {
    return [];
  }
}

export interface DecisionRecord {
  id: string;
  address: string;
  asset: string;
  side: string;
  ts: string;
  status: 'open' | 'closed';
  closedReason?: string | null;
  result?: number | null;
}

export async function listRecentDecisions(limit = 20): Promise<DecisionRecord[]> {
  try {
    const p = await getPool();
    const { rows } = await p.query(
      `
        select
          t.id,
          t.address,
          t.asset,
          t.side,
          t.ts,
          o.closed_reason,
          o.result_r,
          o.closed_ts
        from tickets t
        left join ticket_outcomes o on o.ticket_id = t.id
        order by t.ts desc
        limit $1
      `,
      [Math.max(1, Math.min(100, limit))]
    );
    return rows.map((row: any) => ({
      id: row.id,
      address: row.address,
      asset: row.asset,
      side: row.side,
      ts: row.ts,
      status: row.closed_reason ? 'closed' : 'open',
      closedReason: row.closed_reason,
      result: row.result_r != null ? Number(row.result_r) : null
    }));
  } catch (_e) {
    return [];
  }
}

export async function fetchLatestFillForAddress(address: string): Promise<RecentFill | null> {
  try {
    const p = await getPool();
    const { rows } = await p.query(
      `
        select id, address, at, payload
        from hl_events
        where type = 'trade'
          and address = $1
        order by at desc
        limit 1
      `,
      [address.toLowerCase()]
    );
    const row = rows[0];
    if (!row) return null;
    return {
      id: Number(row.id),
      address: row.address,
      at: row.at,
      side: ((row.payload?.side === 'sell') ? 'sell' : 'buy') as 'buy' | 'sell',
      size: Number(row.payload?.size ?? 0),
      priceUsd: Number(row.payload?.priceUsd ?? 0),
      realizedPnlUsd: row.payload?.realizedPnlUsd != null ? Number(row.payload?.realizedPnlUsd) : null,
      action: row.payload?.action ?? null,
    };
  } catch (_e) {
    return null;
  }
}
