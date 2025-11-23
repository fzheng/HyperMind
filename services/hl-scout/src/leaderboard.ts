import {
  createLogger,
  getPool,
  normalizeAddress,
  nowIso,
  CandidateEventSchema,
  computePerformanceScore,
  DEFAULT_SCORING_PARAMS,
  type ScoringParams
} from '@hl/ts-lib';
import type { CandidateEvent } from '@hl/ts-lib';

const DEFAULT_API_URL = 'https://hyperbot.network/api/leaderboard/smart';
const HYPERLIQUID_INFO_URL = 'https://api.hyperliquid.xyz/info';
const WINDOW_PERIOD_MAP: Record<string, number> = { day: 1, week: 7, month: 30 };
const DEFAULT_STATS_CONCURRENCY = Number(process.env.LEADERBOARD_STATS_CONCURRENCY ?? 4);
const DEFAULT_SERIES_CONCURRENCY = Number(process.env.LEADERBOARD_SERIES_CONCURRENCY ?? 2);

/**
 * Hyperbot leaderboard sort options
 * Used in API query parameter: ?sort=<value>
 */
export enum LeaderboardSort {
  /** Sort by win rate (all accounts in hyperliquid) */
  WIN_RATE = 0,
  /** Sort by account total value */
  ACCOUNT_VALUE = 1,
  /** Sort by realized PnL */
  PNL = 3,
  /** Sort by total trades count */
  TRADES_COUNT = 4,
  /** Sort by profitable trades count */
  PROFITABLE_TRADES = 5,
  /** Sort by last operation time */
  LAST_OPERATION = 6,
  /** Sort by average holding period */
  AVG_HOLDING_PERIOD = 7,
  /** Sort by current positions */
  CURRENT_POSITIONS = 8,
}

const DEFAULT_LEADERBOARD_SORT = LeaderboardSort.PNL;

type LeaderboardRawEntry = {
  address: string;
  winRate?: number;
  executedOrders?: number;
  realizedPnl?: number;
  remark?: string | null;
  labels?: string[] | null;
  pnlList?: Array<{ timestamp: number; value: string }>;
  // Additional fields for new scoring formula
  accountValue?: number;
  startingEquity?: number;
  maxDrawdown?: number;
  numWins?: number;
  numLosses?: number;
};

export type RankedEntry = {
  address: string;
  rank: number;
  score: number;
  weight: number;
  winRate: number;
  executedOrders: number;
  realizedPnl: number;
  efficiency: number;
  pnlConsistency: number;
  remark: string | null;
  labels: string[];
  statOpenPositions: number | null;
  statClosedPositions: number | null;
  statAvgPosDuration: number | null;
  statTotalPnl: number | null;
  statMaxDrawdown: number | null;
  meta: any;
};

type AddressStats = {
  winRate?: number;
  openPosCount?: number;
  closePosCount?: number;
  avgPosDuration?: number;
  totalPnl?: number;
  maxDrawdown?: number;
};

type PortfolioWindowSeries = {
  window: string;
  pnlHistory: Array<{ ts: number; value: number }>;
  equityHistory: Array<{ ts: number; value: number }>;
};

export interface LeaderboardOptions {
  apiUrl?: string;
  topN?: number;
  selectCount?: number;
  periods?: number[];
  pageSize?: number;
  refreshMs?: number;
  enrichCount?: number;
  /** Sort order for leaderboard API (default: PNL) */
  sort?: LeaderboardSort;
}

export class LeaderboardService {
  private opts: Required<LeaderboardOptions>;
  private timer: NodeJS.Timeout | null = null;
  private logger = createLogger('leaderboard');
  private publishCandidate: (entry: CandidateEvent) => Promise<void>;
  private smartApiBase: string;
  private enrichCount: number;
  private statsConcurrency: number;
  private seriesConcurrency: number;

  constructor(opts: LeaderboardOptions, publishCandidate: (entry: CandidateEvent) => Promise<void>) {
    this.opts = {
      apiUrl: opts.apiUrl || DEFAULT_API_URL,
      topN: opts.topN ?? 1000,
      selectCount: opts.selectCount ?? 12,
      periods: opts.periods?.length ? opts.periods : [30],
      pageSize: opts.pageSize ?? 100,
      refreshMs: opts.refreshMs ?? 24 * 60 * 60 * 1000,
      enrichCount: opts.enrichCount ?? opts.selectCount ?? 12,
      sort: opts.sort ?? DEFAULT_LEADERBOARD_SORT,
    };
    this.publishCandidate = publishCandidate;
    this.smartApiBase = this.opts.apiUrl.endsWith('/') ? this.opts.apiUrl : `${this.opts.apiUrl}/`;
    this.enrichCount = Math.max(0, Math.min(this.opts.enrichCount, this.opts.topN));
    this.statsConcurrency = Math.max(1, DEFAULT_STATS_CONCURRENCY);
    this.seriesConcurrency = Math.max(1, DEFAULT_SERIES_CONCURRENCY);
  }

  start() {
    this.refreshAll().catch((err) => this.logger.error('leaderboard_refresh_failed', { err: err?.message }));
    if (this.timer) clearInterval(this.timer);
    this.timer = setInterval(() => {
      this.refreshAll().catch((err) => this.logger.error('leaderboard_refresh_failed', { err: err?.message }));
    }, this.opts.refreshMs);
  }

  stop() {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
  }

  async refreshAll() {
    for (const period of this.opts.periods) {
      try {
        await this.refreshPeriod(period);
      } catch (err: any) {
        this.logger.error('leaderboard_period_failed', { period, err: err?.message });
      }
    }
  }

  async refreshPeriod(period: number) {
    const raw = await this.fetchPeriod(period);
    const ranked = this.scoreEntries(raw);
    const tracked = this.enrichCount > 0 ? ranked.slice(0, this.enrichCount) : [];
    let hyperliquidSeries = new Map<string, PortfolioWindowSeries[]>();
    if (tracked.length) {
      await this.applyAddressStats(period, tracked);
      hyperliquidSeries = await this.fetchPortfolioSeriesBatch(period, tracked);
    }
    await this.persistPeriod(period, ranked, tracked, hyperliquidSeries);
    await this.publishTopCandidates(period, ranked.slice(0, this.opts.selectCount));
    this.logger.info('leaderboard_updated', { period, count: ranked.length });
  }

  private async fetchPeriod(period: number): Promise<LeaderboardRawEntry[]> {
    const results: LeaderboardRawEntry[] = [];
    const pagesNeeded = Math.ceil(this.opts.topN / this.opts.pageSize);
    for (let page = 1; page <= pagesNeeded; page += 1) {
      const url = `${this.opts.apiUrl}?pageNum=${page}&pageSize=${this.opts.pageSize}&period=${period}&sort=${this.opts.sort}`;
      const res = await fetch(url);
      if (!res.ok) throw new Error(`leaderboard HTTP ${res.status}`);
      const payload = await res.json();
      const data: LeaderboardRawEntry[] = payload?.data || [];
      results.push(...data);
      if (data.length < this.opts.pageSize) break;
    }
    return results.slice(0, this.opts.topN);
  }

  /**
   * Scores entries using the new composite performance scoring formula.
   *
   * Formula components (weighted sum):
   * 1. Smooth PnL Score (45%) - performance over time (monotonicity, drawdowns)
   * 2. Adjusted Win Rate (30%) - with Laplace smoothing, penalize 100% win rates
   * 3. Normalized PnL (15%) - log-scaled realized PnL (modest weight for whales)
   * 4. Trade Frequency (10%) - prefer moderate activity, not too many trades
   */
  private scoreEntries(entries: LeaderboardRawEntry[]): RankedEntry[] {
    // Build scoring params from defaults with optional env overrides
    const scoringParams: ScoringParams = {
      smoothPnlWeight: Number(process.env.SCORING_SMOOTH_PNL_WEIGHT ?? DEFAULT_SCORING_PARAMS.smoothPnlWeight),
      winRateWeight: Number(process.env.SCORING_WIN_RATE_WEIGHT ?? DEFAULT_SCORING_PARAMS.winRateWeight),
      pnlWeight: Number(process.env.SCORING_PNL_WEIGHT ?? DEFAULT_SCORING_PARAMS.pnlWeight),
      tradeFreqWeight: Number(process.env.SCORING_TRADE_FREQ_WEIGHT ?? DEFAULT_SCORING_PARAMS.tradeFreqWeight),
      optimalTrades: Number(process.env.SCORING_OPTIMAL_TRADES ?? DEFAULT_SCORING_PARAMS.optimalTrades),
      tradeSigma: Number(process.env.SCORING_TRADE_SIGMA ?? DEFAULT_SCORING_PARAMS.tradeSigma),
      pnlReference: Number(process.env.SCORING_PNL_REFERENCE ?? DEFAULT_SCORING_PARAMS.pnlReference),
    };

    const base = entries
      .map((entry) => {
        const address = normalizeAddress(entry.address);
        const winRate = clamp(Number(entry.winRate ?? 0), 0, 1);
        const executed = Number(entry.executedOrders ?? 0);
        const pnl = Number(entry.realizedPnl ?? 0);
        const efficiency = executed > 0 ? pnl / executed : pnl;

        // Estimate wins/losses from win rate and trade count
        const numTrades = executed;
        const numWins = Math.round(numTrades * winRate);
        const numLosses = numTrades - numWins;

        // Get pnlList for smooth PnL score calculation
        const pnlList = entry.pnlList || [];

        // Compute score using the new formula with pnlList
        const scoringResult = computePerformanceScore({
          realizedPnl: pnl,
          numTrades,
          numWins,
          numLosses,
          pnlList,
        }, scoringParams);

        return {
          address,
          score: scoringResult.score,
          winRate,
          executedOrders: executed,
          realizedPnl: pnl,
          efficiency,
          pnlConsistency: scoringResult.details.smoothPnlScore, // Use smooth PnL as consistency metric
          remark: entry.remark ?? null,
          labels: entry.labels || [],
          statOpenPositions: null,
          statClosedPositions: null,
          statAvgPosDuration: null,
          statTotalPnl: null,
          statMaxDrawdown: null,
          meta: {
            ...entry,
            scoringDetails: scoringResult.details,
            numWins,
            numLosses,
          },
        };
      })
      .filter((entry) => Number.isFinite(entry.score));

    // Filter out suspicious 100% win rates with many trades (already penalized in scoring, but extra filter)
    let scored = base.filter((entry) => entry.winRate < 0.999 || entry.executedOrders < 10);
    if (!scored.length) scored = base;

    // Sort by score descending
    scored.sort((a, b) => b.score - a.score);

    // Normalize weights for top selectCount entries
    const totalScore = scored.slice(0, this.opts.selectCount).reduce((sum, e) => sum + Math.max(e.score, 0), 0) || 1;
    return scored.map((entry, idx) => ({
      ...entry,
      rank: idx + 1,
      weight: idx < this.opts.selectCount ? Math.max(entry.score, 0) / totalScore : 0,
    }));
  }

  private async applyAddressStats(period: number, entries: RankedEntry[]): Promise<void> {
    const statsMap = await this.fetchAddressStatsBatch(period, entries);
    for (const entry of entries) {
      const stat = statsMap.get(entry.address.toLowerCase());
      if (stat) {
        this.applyStatsToEntry(entry, stat);
      }
    }
  }

  private applyStatsToEntry(entry: RankedEntry, stats: AddressStats): void {
    if (typeof stats.winRate === 'number' && Number.isFinite(stats.winRate)) {
      entry.winRate = clamp(stats.winRate, 0, 1);
    }
    entry.statOpenPositions = Number.isFinite(Number(stats.openPosCount))
      ? Number(stats.openPosCount)
      : entry.statOpenPositions;
    entry.statClosedPositions = Number.isFinite(Number(stats.closePosCount))
      ? Number(stats.closePosCount)
      : entry.statClosedPositions;
    entry.statAvgPosDuration = Number.isFinite(Number(stats.avgPosDuration))
      ? Number(stats.avgPosDuration)
      : entry.statAvgPosDuration;
    entry.statTotalPnl = Number.isFinite(Number(stats.totalPnl))
      ? Number(stats.totalPnl)
      : entry.statTotalPnl;
    entry.statMaxDrawdown = Number.isFinite(Number(stats.maxDrawdown))
      ? Number(stats.maxDrawdown)
      : entry.statMaxDrawdown;
    entry.meta = {
      ...entry.meta,
      stats,
    };
  }

  private async fetchAddressStatsBatch(period: number, entries: RankedEntry[]): Promise<Map<string, AddressStats>> {
    const map = new Map<string, AddressStats>();
    const addresses = entries.map((entry) => entry.address);
    await runWithConcurrency(addresses, this.statsConcurrency, async (address) => {
      const stats = await this.fetchAddressStat(address, period);
      if (stats) map.set(address.toLowerCase(), stats);
    });
    return map;
  }

  private async fetchAddressStat(address: string, period: number): Promise<AddressStats | null> {
    const normalized = normalizeAddress(address);
    const url = `${this.smartApiBase}query-addr-stat/${encodeURIComponent(normalized)}?period=${period}`;
    try {
      const payload = await this.requestJson<any>(url, { method: 'GET' }, 2, 6000);
      const data = payload?.data;
      if (!data) return null;
      return {
        winRate: typeof data.winRate === 'number' ? data.winRate : undefined,
        openPosCount: Number.isFinite(Number(data.openPosCount)) ? Number(data.openPosCount) : undefined,
        closePosCount: Number.isFinite(Number(data.closePosCount)) ? Number(data.closePosCount) : undefined,
        avgPosDuration: Number.isFinite(Number(data.avgPosDuration)) ? Number(data.avgPosDuration) : undefined,
        totalPnl: Number.isFinite(Number(data.totalPnl)) ? Number(data.totalPnl) : undefined,
        maxDrawdown: Number.isFinite(Number(data.maxDrawdown)) ? Number(data.maxDrawdown) : undefined,
      };
    } catch (err: any) {
      this.logger.warn('addr_stat_fetch_failed', { address: normalized, period, err: err?.message });
      return null;
    }
  }

  private async fetchPortfolioSeriesBatch(
    period: number,
    entries: RankedEntry[]
  ): Promise<Map<string, PortfolioWindowSeries[]>> {
    const map = new Map<string, PortfolioWindowSeries[]>();
    await runWithConcurrency(entries, this.seriesConcurrency, async (entry) => {
      const series = await this.fetchPortfolioSeries(entry.address);
      if (!series?.length) return;
      const relevant = series.filter((window) => WINDOW_PERIOD_MAP[window.window] === period);
      if (!relevant.length) return;
      map.set(entry.address.toLowerCase(), relevant);
    });
    return map;
  }

  private async fetchPortfolioSeries(address: string): Promise<PortfolioWindowSeries[] | null> {
    const normalized = normalizeAddress(address);
    try {
      const payload = await this.requestJson<any>(
        HYPERLIQUID_INFO_URL,
        {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ type: 'portfolio', user: normalized }),
        },
        1,
        8000
      );
      const rows: any[] = Array.isArray(payload) ? payload : Array.isArray(payload?.value) ? payload.value : [];
      const series: PortfolioWindowSeries[] = [];
      for (const row of rows) {
        if (!Array.isArray(row) || row.length < 2) continue;
        const windowName = String(row[0] ?? '');
        const data = row[1] || {};
        const pnlHistory = parseHistoryPoints(data.pnlHistory);
        const equityHistory = parseHistoryPoints(data.accountValueHistory);
        if (!pnlHistory.length && !equityHistory.length) continue;
        series.push({ window: windowName, pnlHistory, equityHistory });
      }
      return series;
    } catch (err: any) {
      this.logger.warn('portfolio_fetch_failed', { address: normalized, err: err?.message });
      return null;
    }
  }

  private async requestJson<T>(
    input: string,
    init: RequestInit,
    retries = 2,
    timeoutMs = 8000
  ): Promise<T> {
    let attempt = 0;
    let lastErr: any;
    while (attempt <= retries) {
      try {
        const res = await fetch(input, {
          ...init,
          signal: AbortSignal.timeout(timeoutMs),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return (await res.json()) as T;
      } catch (err) {
        lastErr = err;
        attempt += 1;
        if (attempt > retries) break;
        await sleep(200 * attempt);
      }
    }
    throw lastErr;
  }

  private async persistPeriod(
    period: number,
    entries: RankedEntry[],
    tracked: RankedEntry[],
    hyperliquidSeries: Map<string, PortfolioWindowSeries[]>
  ): Promise<void> {
    const pool = await getPool();
    const client = await pool.connect();
    try {
      await client.query('BEGIN');
      await client.query('DELETE FROM hl_leaderboard_entries WHERE period_days = $1', [period]);
      await client.query('DELETE FROM hl_leaderboard_pnl_points WHERE period_days = $1', [period]);
      if (!entries.length) {
        await client.query('COMMIT');
        return;
      }
    const chunkSize = 100;
    for (let i = 0; i < entries.length; i += chunkSize) {
      const chunk = entries.slice(i, i + chunkSize);
      const values = chunk.flatMap((entry) => [
        period,
        entry.address,
        entry.rank,
        entry.score,
        entry.weight,
        entry.winRate,
        entry.executedOrders,
        entry.realizedPnl,
        entry.pnlConsistency,
        entry.efficiency,
        entry.remark,
        JSON.stringify(entry.labels || []),
        JSON.stringify({
          raw: entry.meta,
          stats: {
            openPositions: entry.statOpenPositions,
            closedPositions: entry.statClosedPositions,
            avgPositionDuration: entry.statAvgPosDuration,
            totalPnl: entry.statTotalPnl,
            maxDrawdown: entry.statMaxDrawdown,
          },
        }),
        entry.statOpenPositions,
        entry.statClosedPositions,
        entry.statAvgPosDuration,
        entry.statTotalPnl,
        entry.statMaxDrawdown,
      ]);
      const placeholders = chunk
        .map(
          (_entry, idx) =>
            `($${idx * 18 + 1}, $${idx * 18 + 2}, $${idx * 18 + 3}, $${idx * 18 + 4}, $${idx * 18 + 5}, $${idx * 18 + 6}, $${idx * 18 + 7}, $${idx * 18 + 8}, $${idx * 18 + 9}, $${idx * 18 + 10}, $${idx * 18 + 11}, $${idx * 18 + 12}, $${idx * 18 + 13}, $${idx * 18 + 14}, $${idx * 18 + 15}, $${idx * 18 + 16}, $${idx * 18 + 17}, $${idx * 18 + 18})`
        )
        .join(',');
      await client.query(
        `
        INSERT INTO hl_leaderboard_entries (
          period_days, address, rank, score, weight, win_rate, executed_orders,
          realized_pnl, pnl_consistency, efficiency, remark, labels, metrics,
          stat_open_positions, stat_closed_positions, stat_avg_pos_duration, stat_total_pnl, stat_max_drawdown
        )
        VALUES ${placeholders}
        ON CONFLICT (period_days, lower(address)) DO UPDATE SET
          rank = EXCLUDED.rank,
          score = EXCLUDED.score,
          weight = EXCLUDED.weight,
          win_rate = EXCLUDED.win_rate,
          executed_orders = EXCLUDED.executed_orders,
          realized_pnl = EXCLUDED.realized_pnl,
          pnl_consistency = EXCLUDED.pnl_consistency,
          efficiency = EXCLUDED.efficiency,
          remark = EXCLUDED.remark,
          labels = EXCLUDED.labels,
          metrics = EXCLUDED.metrics,
          stat_open_positions = EXCLUDED.stat_open_positions,
          stat_closed_positions = EXCLUDED.stat_closed_positions,
          stat_avg_pos_duration = EXCLUDED.stat_avg_pos_duration,
          stat_total_pnl = EXCLUDED.stat_total_pnl,
          stat_max_drawdown = EXCLUDED.stat_max_drawdown,
          fetched_at = now()
      `,
        values
      );
    }
    if (tracked.length) {
      await this.insertPnlPointsInTransaction(client, period, tracked, hyperliquidSeries);
    }
      await client.query('COMMIT');
    } catch (err) {
      await client.query('ROLLBACK');
      this.logger.error('leaderboard_persist_failed', { period, err: err instanceof Error ? err.message : String(err) });
      throw err;
    } finally {
      client.release();
    }
  }

  private async insertPnlPointsInTransaction(
    client: any,
    period: number,
    tracked: RankedEntry[],
    hyperliquidSeries: Map<string, PortfolioWindowSeries[]>
  ): Promise<void> {
    const points: Array<{
      address: string;
      source: string;
      window: string;
      ts: Date;
      pnlValue: number | null;
      equityValue: number | null;
    }> = [];

    for (const entry of tracked) {
      const pnlList = Array.isArray(entry.meta?.pnlList) ? entry.meta.pnlList : [];
      for (const point of pnlList) {
        const tsValue = Number(point?.timestamp);
        const pnlValue = Number(point?.value);
        if (!Number.isFinite(tsValue)) continue;
        const date = new Date(tsValue);
        if (Number.isNaN(date.getTime())) continue;
        points.push({
          address: entry.address.toLowerCase(),
          source: 'hyperbot',
          window: `period_${period}`,
          ts: date,
          pnlValue: Number.isFinite(pnlValue) ? pnlValue : null,
          equityValue: null,
        });
      }
    }

    for (const [address, seriesList] of hyperliquidSeries.entries()) {
      for (const series of seriesList) {
        if (WINDOW_PERIOD_MAP[series.window] !== period) continue;
        const merged = mergeHistories(series);
        for (const sample of merged) {
          const tsDate = new Date(sample.ts);
          if (Number.isNaN(tsDate.getTime())) continue;
          points.push({
            address,
            source: 'hyperliquid',
            window: series.window,
            ts: tsDate,
            pnlValue: sample.pnl ?? null,
            equityValue: sample.equity ?? null,
          });
        }
      }
    }

    if (!points.length) return;

    const chunkSize = 400;
    for (let i = 0; i < points.length; i += chunkSize) {
      const chunk = points.slice(i, i + chunkSize);
      const values = chunk.flatMap((point) => [
        period,
        point.address,
        point.source,
        point.window,
        point.ts.toISOString(),
        point.pnlValue,
        point.equityValue,
      ]);
      const placeholders = chunk
        .map(
          (_point, idx) =>
            `($${idx * 7 + 1}, $${idx * 7 + 2}, $${idx * 7 + 3}, $${idx * 7 + 4}, $${idx * 7 + 5}, $${idx * 7 + 6}, $${idx * 7 + 7})`
        )
        .join(',');
      await client.query(
        `
          INSERT INTO hl_leaderboard_pnl_points (
            period_days, address, source, window_name, point_ts, pnl_value, equity_value
          )
          VALUES ${placeholders}
        `,
        values
      );
    }
  }

  private async publishTopCandidates(period: number, entries: RankedEntry[]): Promise<void> {
    for (const entry of entries) {
      const candidate = CandidateEventSchema.parse({
        address: entry.address,
        source: 'daily',
        ts: nowIso(),
        tags: [`period:${period}`, 'leaderboard'],
        nickname: entry.remark || undefined,
        score_hint: entry.score,
        meta: {
          leaderboard: {
            period_days: period,
            rank: entry.rank,
            weight: entry.weight,
            score: entry.score,
            winRate: entry.winRate,
            executedOrders: entry.executedOrders,
            realizedPnl: entry.realizedPnl,
            pnlConsistency: entry.pnlConsistency,
            efficiency: entry.efficiency,
            labels: entry.labels,
          },
        },
      });
      await this.publishCandidate(candidate).catch((err) =>
        this.logger.error('publish_candidate_failed', { address: entry.address, err: err?.message })
      );
    }
  }

  async getEntries(period: number, limit = this.opts.selectCount): Promise<RankedEntry[]> {
    const pool = await getPool();
    const { rows } = await pool.query(
      `
        SELECT
          address,
          rank,
          score,
          weight,
          coalesce(win_rate,0) as win_rate,
          coalesce(executed_orders,0) as executed_orders,
          coalesce(realized_pnl,0) as realized_pnl,
          coalesce(pnl_consistency,0) as pnl_consistency,
          coalesce(efficiency,0) as efficiency,
          remark,
          labels,
          metrics,
          stat_open_positions,
          stat_closed_positions,
          stat_avg_pos_duration,
          stat_total_pnl,
          stat_max_drawdown
        FROM hl_leaderboard_entries
        WHERE period_days = $1
        ORDER BY rank ASC
        LIMIT $2
      `,
      [period, limit]
    );
    return rows.map((row: any) => ({
      address: row.address,
      rank: Number(row.rank),
      score: Number(row.score),
      weight: Number(row.weight),
      winRate: Number(row.win_rate),
      executedOrders: Number(row.executed_orders),
      realizedPnl: Number(row.realized_pnl),
      pnlConsistency: Number(row.pnl_consistency),
      efficiency: Number(row.efficiency),
      remark: row.remark,
      labels: Array.isArray(row.labels) ? row.labels : [],
      meta: row.metrics,
      statOpenPositions: row.stat_open_positions == null ? null : Number(row.stat_open_positions),
      statClosedPositions: row.stat_closed_positions == null ? null : Number(row.stat_closed_positions),
      statAvgPosDuration: row.stat_avg_pos_duration == null ? null : Number(row.stat_avg_pos_duration),
      statTotalPnl: row.stat_total_pnl == null ? null : Number(row.stat_total_pnl),
      statMaxDrawdown: row.stat_max_drawdown == null ? null : Number(row.stat_max_drawdown),
    }));
  }

  async getSelected(period: number, limit = this.opts.selectCount): Promise<RankedEntry[]> {
    const pool = await getPool();
    const { rows } = await pool.query(
      `
        SELECT
          address,
          rank,
          score,
          weight,
          coalesce(win_rate,0) as win_rate,
          coalesce(executed_orders,0) as executed_orders,
          coalesce(realized_pnl,0) as realized_pnl,
          coalesce(pnl_consistency,0) as pnl_consistency,
          coalesce(efficiency,0) as efficiency,
          remark,
          labels,
          metrics,
          stat_open_positions,
          stat_closed_positions,
          stat_avg_pos_duration,
          stat_total_pnl,
          stat_max_drawdown
        FROM hl_leaderboard_entries
        WHERE period_days = $1
        ORDER BY weight DESC, rank ASC
        LIMIT $2
      `,
      [period, limit]
    );
    return rows.map((row: any) => ({
      address: row.address,
      rank: Number(row.rank),
      score: Number(row.score),
      weight: Number(row.weight),
      winRate: Number(row.win_rate),
      executedOrders: Number(row.executed_orders),
      realizedPnl: Number(row.realized_pnl),
      pnlConsistency: Number(row.pnl_consistency),
      efficiency: Number(row.efficiency),
      remark: row.remark,
      labels: Array.isArray(row.labels) ? row.labels : [],
      meta: row.metrics,
      statOpenPositions: row.stat_open_positions == null ? null : Number(row.stat_open_positions),
      statClosedPositions: row.stat_closed_positions == null ? null : Number(row.stat_closed_positions),
      statAvgPosDuration: row.stat_avg_pos_duration == null ? null : Number(row.stat_avg_pos_duration),
      statTotalPnl: row.stat_total_pnl == null ? null : Number(row.stat_total_pnl),
      statMaxDrawdown: row.stat_max_drawdown == null ? null : Number(row.stat_max_drawdown),
    }));
  }

  async ensureSeeded(): Promise<void> {
    const pool = await getPool();
    for (const period of this.opts.periods) {
      const { rows } = await pool.query(
        'SELECT 1 FROM hl_leaderboard_entries WHERE period_days = $1 LIMIT 1',
        [period]
      );
      if (!rows.length) {
        this.logger.info('leaderboard_empty_seed', { period });
        await this.refreshPeriod(period);
      }
    }
  }

  /**
   * Fetch stats for a custom account and upsert into hl_leaderboard_entries.
   * This is called when a user adds a custom account so stats appear immediately.
   */
  async fetchAndStoreCustomAccountStats(address: string, nickname?: string | null): Promise<RankedEntry | null> {
    const normalized = normalizeAddress(address);
    const period = this.opts.periods[0] ?? 30; // Use first configured period (default 30d)

    this.logger.info('fetch_custom_account_stats', { address: normalized, period });

    // Fetch stats from Hyperbot API
    const stats = await this.fetchAddressStat(normalized, period);

    // Build a minimal RankedEntry with fetched stats
    const entry: RankedEntry = {
      address: normalized,
      rank: 9999, // Custom accounts get a high rank (sorted separately in UI)
      score: 0,
      weight: 0, // Custom accounts don't contribute to weighted selection
      winRate: stats?.winRate ?? 0,
      executedOrders: (stats?.openPosCount ?? 0) + (stats?.closePosCount ?? 0),
      realizedPnl: stats?.totalPnl ?? 0,
      efficiency: stats?.totalPnl && stats.closePosCount ? stats.totalPnl / Math.max(1, stats.closePosCount) : 0,
      pnlConsistency: 0,
      remark: nickname ?? null,
      labels: ['custom'],
      statOpenPositions: stats?.openPosCount ?? null,
      statClosedPositions: stats?.closePosCount ?? null,
      statAvgPosDuration: stats?.avgPosDuration ?? null,
      statTotalPnl: stats?.totalPnl ?? null,
      statMaxDrawdown: stats?.maxDrawdown ?? null,
      meta: { custom: true, fetchedAt: nowIso() },
    };

    // Upsert into hl_leaderboard_entries
    const pool = await getPool();
    try {
      await pool.query(
        `
        INSERT INTO hl_leaderboard_entries (
          period_days, address, rank, score, weight, win_rate, executed_orders,
          realized_pnl, pnl_consistency, efficiency, remark, labels, metrics,
          stat_open_positions, stat_closed_positions, stat_avg_pos_duration, stat_total_pnl, stat_max_drawdown
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
        ON CONFLICT (period_days, lower(address)) DO UPDATE SET
          win_rate = EXCLUDED.win_rate,
          executed_orders = EXCLUDED.executed_orders,
          realized_pnl = EXCLUDED.realized_pnl,
          efficiency = EXCLUDED.efficiency,
          remark = COALESCE(EXCLUDED.remark, hl_leaderboard_entries.remark),
          labels = EXCLUDED.labels,
          metrics = EXCLUDED.metrics,
          stat_open_positions = EXCLUDED.stat_open_positions,
          stat_closed_positions = EXCLUDED.stat_closed_positions,
          stat_avg_pos_duration = EXCLUDED.stat_avg_pos_duration,
          stat_total_pnl = EXCLUDED.stat_total_pnl,
          stat_max_drawdown = EXCLUDED.stat_max_drawdown,
          fetched_at = now()
        `,
        [
          period,
          normalized,
          entry.rank,
          entry.score,
          entry.weight,
          entry.winRate,
          entry.executedOrders,
          entry.realizedPnl,
          entry.pnlConsistency,
          entry.efficiency,
          entry.remark,
          JSON.stringify(entry.labels),
          JSON.stringify(entry.meta),
          entry.statOpenPositions,
          entry.statClosedPositions,
          entry.statAvgPosDuration,
          entry.statTotalPnl,
          entry.statMaxDrawdown,
        ]
      );
      this.logger.info('custom_account_stats_stored', { address: normalized, period, stats });
      return entry;
    } catch (err: any) {
      this.logger.error('custom_account_stats_store_failed', { address: normalized, err: err?.message });
      return null;
    }
  }
}

export default LeaderboardService;

async function runWithConcurrency<T>(
  items: T[],
  limit: number,
  worker: (item: T, index: number) => Promise<void>
): Promise<void> {
  if (!items.length) return;
  const poolSize = Math.max(1, Math.min(limit, items.length));
  let cursor = 0;
  await Promise.all(
    Array.from({ length: poolSize }).map(async () => {
      // eslint-disable-next-line no-constant-condition
      while (true) {
        const index = cursor;
        cursor += 1;
        if (index >= items.length) break;
        try {
          await worker(items[index], index);
        } catch {
          // Individual workers log their own failures; continue to next task.
        }
      }
    })
  );
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function parseHistoryPoints(list: any): Array<{ ts: number; value: number }> {
  if (!Array.isArray(list)) return [];
  const out: Array<{ ts: number; value: number }> = [];
  for (const item of list) {
    if (!Array.isArray(item) || item.length < 2) continue;
    const ts = Number(item[0]);
    const value = Number(item[1]);
    if (!Number.isFinite(ts)) continue;
    const numericValue = Number(value);
    out.push({ ts, value: Number.isFinite(numericValue) ? numericValue : 0 });
  }
  return out;
}

function mergeHistories(series: PortfolioWindowSeries): Array<{ ts: number; pnl?: number; equity?: number }> {
  const map = new Map<number, { pnl?: number; equity?: number }>();
  for (const point of series.pnlHistory) {
    const existing = map.get(point.ts) || {};
    existing.pnl = point.value;
    map.set(point.ts, existing);
  }
  for (const point of series.equityHistory) {
    const existing = map.get(point.ts) || {};
    existing.equity = point.value;
    map.set(point.ts, existing);
  }
  return Array.from(map.entries())
    .map(([ts, values]) => ({ ts, pnl: values.pnl, equity: values.equity }))
    .sort((a, b) => a.ts - b.ts);
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

// normalizeLog removed - no longer used with new scoring formula

function computeConsistency(list: Array<{ timestamp: number; value: string }>): number {
  if (!list?.length) return 0.5;
  const values = list.map((p) => Number(p.value)).filter((v) => Number.isFinite(v));
  if (values.length < 3) return 0.5;
  const first = values[0];
  const last = values[values.length - 1];
  const slope = (last - first) / Math.max(Math.abs(first) + 1, 1);
  const diffs: number[] = [];
  for (let i = 1; i < values.length; i += 1) {
    diffs.push(values[i] - values[i - 1]);
  }
  const variance = diffs.reduce((sum, d) => sum + d * d, 0) / diffs.length || 0;
  const std = Math.sqrt(variance) || 1;
  const ratio = slope / std;
  return clamp(0.5 + Math.atan(ratio) / Math.PI, 0, 1);
}
