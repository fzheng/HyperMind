/**
 * Performance Scoring Module
 *
 * Implements a composite performance score for ranking trading accounts.
 * The formula balances:
 * 1. Smooth PnL Score - performance over time (monotonicity, drawdowns)
 * 2. Win Rate - with Laplace smoothing (penalize 100% win rates)
 * 3. Realized PnL - modest weight (whales can make bad trades)
 * 4. Trade Count - prefer moderate activity (not too many)
 *
 * @module scoring
 */

/**
 * Hyperparameters for the scoring formula.
 */
export interface ScoringParams {
  /**
   * Weight for smooth PnL score component (0-1).
   * Default: 0.45
   */
  smoothPnlWeight: number;

  /**
   * Weight for adjusted win rate component (0-1).
   * Default: 0.30
   */
  winRateWeight: number;

  /**
   * Weight for normalized PnL component (0-1).
   * Default: 0.15
   */
  pnlWeight: number;

  /**
   * Weight for trade frequency component (0-1).
   * Default: 0.10
   */
  tradeFreqWeight: number;

  /**
   * Optimal number of trades (trades around this are preferred).
   * Default: 100
   */
  optimalTrades: number;

  /**
   * Spread for trade count preference (Gaussian-like decay).
   * Default: 150
   */
  tradeSigma: number;

  /**
   * Reference PnL for log normalization.
   * Default: 100000
   */
  pnlReference: number;
}

/**
 * Default scoring parameters
 */
export const DEFAULT_SCORING_PARAMS: ScoringParams = {
  smoothPnlWeight: 0.45,
  winRateWeight: 0.30,
  pnlWeight: 0.15,
  tradeFreqWeight: 0.10,
  optimalTrades: 100,
  tradeSigma: 150,
  pnlReference: 100000,
};

/**
 * PnL time series point - supports multiple formats
 */
export type PnlPoint =
  | number
  | [number, number]
  | [number, string]
  | { timestamp?: number; value?: number | string; pnl?: number | string };

/**
 * Account statistics required for performance scoring.
 */
export interface AccountStats {
  /** Realized PnL over the period (can be negative) */
  realizedPnl: number;

  /** Total number of closed trades */
  numTrades: number;

  /** Number of winning trades */
  numWins: number;

  /** Number of losing trades */
  numLosses: number;

  /** PnL time series for smooth PnL calculation */
  pnlList?: PnlPoint[];
}

/**
 * Result of computing a performance score for an account.
 */
export interface ScoringResult {
  /** Final composite performance score (higher = better) */
  score: number;

  /** Intermediate calculation values for debugging/display */
  details: {
    /** Smooth PnL score [0, 1] */
    smoothPnlScore: number;

    /** Raw win rate before adjustments */
    rawWinRate: number;

    /** Adjusted win rate with Laplace smoothing and 100% penalty */
    adjWinRate: number;

    /** Normalized PnL score [0, 1] */
    normalizedPnl: number;

    /** Trade frequency score [0, 1] */
    tradeFreqScore: number;

    /** Component scores weighted */
    weightedComponents: {
      smoothPnl: number;
      winRate: number;
      pnl: number;
      tradeFreq: number;
    };
  };
}

/**
 * Computes the Smooth PnL Score from a PnL time series.
 *
 * The metric rewards:
 * - Higher final PnL
 * - More monotonic up moves
 * - Smaller & shorter drawdowns
 *
 * @param pnlList - Array of PnL points in time order
 * @returns Non-negative float [0, ~0.5] (higher = better)
 */
export function computeSmoothPnlScore(pnlList: PnlPoint[]): number {
  // Extract numeric PnL values in time order
  const values: number[] = [];

  for (const pt of pnlList) {
    let v: number | string | undefined;

    if (Array.isArray(pt)) {
      // [timestamp, pnl] - take last element
      v = pt[pt.length - 1];
    } else if (typeof pt === 'object' && pt !== null) {
      // Try common keys
      v = pt.pnl ?? pt.value;
    } else {
      // Assume it's already a number/string
      v = pt;
    }

    const parsed = typeof v === 'string' ? parseFloat(v) : v;
    if (typeof parsed === 'number' && Number.isFinite(parsed)) {
      values.push(parsed);
    }
  }

  // Not enough data -> no meaningful score
  if (values.length < 2) {
    return 0;
  }

  // Normalize series to start at 0 (focus on shape & change)
  const base = values[0];
  const x = values.map((v) => v - base); // x[0] == 0

  const n = x.length;

  // Drawdown series, Max Drawdown (MDD), Ulcer Index
  let peak = x[0];
  let mdd = 0;
  let ddSqSum = 0;

  for (const xi of x) {
    if (xi > peak) {
      peak = xi;
    }

    let dd = 0;
    if (peak > 0) {
      dd = (peak - xi) / peak; // fractional drawdown from peak
      if (dd < 0) dd = 0;
    }

    mdd = Math.max(mdd, dd);
    ddSqSum += dd * dd;
  }

  const ulcer = Math.sqrt(ddSqSum / n); // Ulcer Index

  // Monotonicity: fraction of up steps
  let upCount = 0;
  for (let i = 1; i < n; i++) {
    if (x[i] > x[i - 1]) {
      upCount++;
    }
  }
  const upFrac = upCount / (n - 1);

  // Return component (scale-invariant, based on final vs path range)
  const last = x[n - 1];
  const maxAbs = Math.max(...x.map((v) => Math.abs(v)));

  let R = 0;
  if (last > 0 && maxAbs > 0) {
    // 0..1: how close final PnL is to the best level reached
    R = last / maxAbs;
  }

  // Final SmoothPnlScore
  // Denominator includes 1 to keep it stable even with tiny drawdowns
  const score = (Math.max(0, R) * upFrac) / (1 + mdd + ulcer);

  return Number.isFinite(score) ? score : 0;
}

/**
 * Computes adjusted win rate with Laplace smoothing and 100% penalty.
 *
 * @param numWins - Number of winning trades
 * @param numLosses - Number of losing trades
 * @returns Adjusted win rate [0, 1]
 */
export function computeAdjustedWinRate(numWins: number, numLosses: number): number {
  // Laplace (add-one) smoothing to prevent extreme values with few trades
  const baseWinRate = (numWins + 1) / (numWins + numLosses + 2);

  // Penalize suspicious 100% win rates (too good to be true)
  if (numLosses === 0 && numWins > 0) {
    return 0.7 * baseWinRate; // 30% penalty for zero losses
  }

  // Also penalize very high win rates with many trades (likely manipulation)
  if (baseWinRate > 0.95 && numWins + numLosses > 20) {
    return 0.8 * baseWinRate;
  }

  return baseWinRate;
}

/**
 * Computes normalized PnL score using log scaling.
 *
 * @param realizedPnl - Realized PnL (can be negative)
 * @param reference - Reference PnL for normalization
 * @returns Score [0, 1] where 1 = excellent, 0 = poor/negative
 */
export function computeNormalizedPnl(realizedPnl: number, reference: number): number {
  if (realizedPnl <= 0) {
    return 0;
  }

  // Log scale: log10(pnl) normalized by log10(reference)
  // This gives diminishing returns for extremely large PnLs
  const logPnl = Math.log10(realizedPnl + 1);
  const logRef = Math.log10(reference);

  // Clamp to [0, 1]
  return Math.min(1, Math.max(0, logPnl / logRef));
}

/**
 * Computes trade frequency score.
 * Prefers moderate activity - not too few (unreliable) or too many (overtrading).
 *
 * @param numTrades - Number of trades
 * @param optimal - Optimal number of trades
 * @param sigma - Spread for preference decay
 * @returns Score [0, 1] centered around optimal
 */
export function computeTradeFreqScore(
  numTrades: number,
  optimal: number,
  sigma: number
): number {
  if (numTrades <= 0) {
    return 0;
  }

  // Gaussian-like decay from optimal
  const diff = numTrades - optimal;
  const score = Math.exp(-(diff * diff) / (2 * sigma * sigma));

  // Bonus reduction for too many trades (overtrading)
  if (numTrades > optimal * 3) {
    return score * 0.5;
  }

  return score;
}

/**
 * Computes the composite performance score for a single account.
 *
 * Formula:
 * score = smoothPnlWeight * smoothPnlScore
 *       + winRateWeight * adjWinRate
 *       + pnlWeight * normalizedPnl
 *       + tradeFreqWeight * tradeFreqScore
 *
 * @param stats - Account statistics for the period
 * @param params - Scoring hyperparameters (optional, uses defaults)
 * @returns Scoring result with final score and intermediate details
 */
export function computePerformanceScore(
  stats: AccountStats,
  params: ScoringParams = DEFAULT_SCORING_PARAMS
): ScoringResult {
  const { realizedPnl, numTrades, numWins, numLosses, pnlList } = stats;

  // Validate inputs
  if (!Number.isFinite(numTrades) || numTrades < 0) {
    return createZeroResult();
  }
  if (!Number.isFinite(numWins) || numWins < 0) {
    return createZeroResult();
  }
  if (!Number.isFinite(numLosses) || numLosses < 0) {
    return createZeroResult();
  }

  // 1. Compute smooth PnL score from time series
  const smoothPnlScore = pnlList && pnlList.length >= 2
    ? computeSmoothPnlScore(pnlList)
    : 0;

  // 2. Compute adjusted win rate
  const rawWinRate = numWins + numLosses > 0
    ? numWins / (numWins + numLosses)
    : 0;
  const adjWinRate = computeAdjustedWinRate(numWins, numLosses);

  // 3. Compute normalized PnL
  const normalizedPnl = computeNormalizedPnl(
    realizedPnl ?? 0,
    params.pnlReference
  );

  // 4. Compute trade frequency score
  const tradeFreqScore = computeTradeFreqScore(
    numTrades,
    params.optimalTrades,
    params.tradeSigma
  );

  // 5. Compute weighted components
  const weightedSmooth = params.smoothPnlWeight * smoothPnlScore;
  const weightedWinRate = params.winRateWeight * adjWinRate;
  const weightedPnl = params.pnlWeight * normalizedPnl;
  const weightedFreq = params.tradeFreqWeight * tradeFreqScore;

  // 6. Final composite score
  const score = weightedSmooth + weightedWinRate + weightedPnl + weightedFreq;

  return {
    score: Number.isFinite(score) ? score : 0,
    details: {
      smoothPnlScore,
      rawWinRate,
      adjWinRate,
      normalizedPnl,
      tradeFreqScore,
      weightedComponents: {
        smoothPnl: weightedSmooth,
        winRate: weightedWinRate,
        pnl: weightedPnl,
        tradeFreq: weightedFreq,
      },
    },
  };
}

/**
 * Creates a zero-score result for invalid inputs
 */
function createZeroResult(): ScoringResult {
  return {
    score: 0,
    details: {
      smoothPnlScore: 0,
      rawWinRate: 0,
      adjWinRate: 0,
      normalizedPnl: 0,
      tradeFreqScore: 0,
      weightedComponents: {
        smoothPnl: 0,
        winRate: 0,
        pnl: 0,
        tradeFreq: 0,
      },
    },
  };
}

/**
 * Account with address and stats for ranking
 */
export interface RankableAccount {
  address: string;
  stats: AccountStats;
  /** Optional: indicates if this is a user-added custom account */
  isCustom?: boolean;
  /** Optional: any additional metadata to preserve */
  meta?: Record<string, unknown>;
}

/**
 * Ranked account with computed score
 */
export interface RankedAccount {
  address: string;
  rank: number;
  score: number;
  stats: AccountStats;
  details: ScoringResult['details'];
  isCustom: boolean;
  meta?: Record<string, unknown>;
}

/**
 * Computes scores for multiple accounts and returns them sorted by score descending.
 *
 * @param accounts - Array of accounts with their statistics
 * @param params - Scoring hyperparameters (optional)
 * @returns Array of ranked accounts sorted by score (highest first)
 */
export function rankAccounts(
  accounts: RankableAccount[],
  params: ScoringParams = DEFAULT_SCORING_PARAMS
): RankedAccount[] {
  // Compute scores for all accounts
  const scored = accounts.map((account) => {
    const result = computePerformanceScore(account.stats, params);
    return {
      address: account.address,
      score: result.score,
      stats: account.stats,
      details: result.details,
      isCustom: account.isCustom ?? false,
      meta: account.meta,
    };
  });

  // Sort by score descending
  scored.sort((a, b) => b.score - a.score);

  // Assign ranks (1-based)
  return scored.map((account, index) => ({
    ...account,
    rank: index + 1,
  }));
}

/**
 * Selects top N system accounts and merges with custom accounts.
 * Custom accounts are always included and ranked together with system accounts.
 *
 * @param systemAccounts - Array of system-selected accounts
 * @param customAccounts - Array of user-added custom accounts (max 3)
 * @param topN - Number of top system accounts to include (default: 10)
 * @param params - Scoring hyperparameters
 * @returns Combined and ranked array (10-13 accounts)
 */
export function selectAndRankAccounts(
  systemAccounts: RankableAccount[],
  customAccounts: RankableAccount[],
  topN: number = 10,
  params: ScoringParams = DEFAULT_SCORING_PARAMS
): RankedAccount[] {
  // Score all system accounts first
  const scoredSystem = systemAccounts.map((account) => {
    const result = computePerformanceScore(account.stats, params);
    return {
      ...account,
      score: result.score,
      details: result.details,
      isCustom: false,
    };
  });

  // Sort system accounts and take top N
  scoredSystem.sort((a, b) => b.score - a.score);
  const topSystem = scoredSystem.slice(0, topN);

  // Score custom accounts
  const scoredCustom = customAccounts.slice(0, 3).map((account) => {
    const result = computePerformanceScore(account.stats, params);
    return {
      ...account,
      score: result.score,
      details: result.details,
      isCustom: true,
    };
  });

  // Merge and re-rank all accounts together
  const allAccounts = [...topSystem, ...scoredCustom];

  // Remove duplicates (prefer custom if address appears in both)
  const customAddresses = new Set(scoredCustom.map((a) => a.address.toLowerCase()));
  const deduped = allAccounts.filter((account) => {
    if (account.isCustom) return true;
    return !customAddresses.has(account.address.toLowerCase());
  });

  // Sort by score and assign final ranks
  deduped.sort((a, b) => b.score - a.score);

  return deduped.map((account, index) => ({
    address: account.address,
    rank: index + 1,
    score: account.score,
    stats: account.stats,
    details: account.details,
    isCustom: account.isCustom,
    meta: account.meta,
  }));
}

/**
 * Maps raw leaderboard entry data to AccountStats format.
 * Handles field name differences and provides defaults for missing data.
 *
 * @param entry - Raw leaderboard entry from database or API
 * @returns AccountStats object ready for scoring
 */
export function mapToAccountStats(entry: {
  realizedPnl?: number;
  realized_pnl?: number;
  numTrades?: number;
  num_trades?: number;
  executedOrders?: number;
  executed_orders?: number;
  statClosedPositions?: number;
  stat_closed_positions?: number;
  numWins?: number;
  num_wins?: number;
  numLosses?: number;
  num_losses?: number;
  winRate?: number;
  win_rate?: number;
  pnlList?: PnlPoint[];
  pnl_list?: PnlPoint[];
}): AccountStats {
  // Get realized PnL
  const realizedPnl = entry.realizedPnl ?? entry.realized_pnl ?? 0;

  // Get number of trades
  const numTrades = entry.numTrades ?? entry.num_trades ??
    entry.executedOrders ?? entry.executed_orders ??
    entry.statClosedPositions ?? entry.stat_closed_positions ?? 0;

  // Get wins and losses
  let numWins = entry.numWins ?? entry.num_wins ?? 0;
  let numLosses = entry.numLosses ?? entry.num_losses ?? 0;

  // If wins/losses not available, estimate from win rate and numTrades
  if (numWins === 0 && numLosses === 0 && numTrades > 0) {
    const winRate = entry.winRate ?? entry.win_rate ?? 0.5;
    numWins = Math.round(numTrades * winRate);
    numLosses = numTrades - numWins;
  }

  // Get PnL list
  const pnlList = entry.pnlList ?? entry.pnl_list ?? [];

  return {
    realizedPnl,
    numTrades,
    numWins,
    numLosses,
    pnlList,
  };
}
