/**
 * Performance Scoring Module
 *
 * Implements a composite performance score for ranking trading accounts.
 * The formula balances return, win rate reliability, trade frequency, and risk.
 *
 * @module scoring
 */

/**
 * Hyperparameters for the scoring formula.
 * These can be adjusted to tune the ranking behavior.
 */
export interface ScoringParams {
  /**
   * N0: Number of trades needed for win rate to be considered reliable.
   * Higher values require more trades before trusting the win rate.
   * Default: 30
   */
  N0: number;

  /**
   * N1: Number of trades where "too high frequency" starts being penalized.
   * Earning the same return with fewer trades is considered better.
   * Default: 300
   */
  N1: number;

  /**
   * EPS: Small constant to avoid division by zero when maxDrawdown is 0.
   * Default: 0.01
   */
  EPS: number;
}

/**
 * Default scoring parameters
 */
export const DEFAULT_SCORING_PARAMS: ScoringParams = {
  N0: 30,   // trades needed for win rate reliability
  N1: 300,  // trades where high-frequency penalty starts
  EPS: 0.01 // epsilon for drawdown denominator
};

/**
 * Account statistics required for performance scoring.
 * All numeric fields should be validated before passing to scoring functions.
 */
export interface AccountStats {
  /** Realized PnL over the period (can be negative) */
  realizedPnl: number;

  /** Account equity at start of period (or average equity) */
  startingEquity: number;

  /** Total number of closed trades (numWins + numLosses) */
  numTrades: number;

  /** Number of winning trades */
  numWins: number;

  /** Number of losing trades */
  numLosses: number;

  /**
   * Maximum drawdown as a positive fraction.
   * Example: 0.25 represents a -25% drawdown.
   * Must be >= 0.
   */
  maxDrawdown: number;
}

/**
 * Result of computing a performance score for an account.
 */
export interface ScoringResult {
  /** Final composite performance score */
  score: number;

  /** Intermediate calculation values for debugging/display */
  details: {
    /** Normalized return: realizedPnl / startingEquity */
    normalizedReturn: number;

    /** Base win rate with Laplace smoothing */
    baseWinRate: number;

    /** Adjusted win rate (penalized if zero losses) */
    adjWinRate: number;

    /** Trade count reliability factor [0, 1) */
    reliability: number;

    /** High-frequency trading penalty factor (0, 1] */
    freqPenalty: number;

    /** Denominator: EPS + maxDrawdown */
    denominator: number;
  };
}

/**
 * Computes the performance score for a single account.
 *
 * Formula breakdown:
 * 1. r = realizedPnl / startingEquity (normalized return)
 * 2. baseWinRate = (numWins + 1) / (numWins + numLosses + 2) (Laplace smoothing)
 * 3. adjWinRate = 0.8 * baseWinRate if numLosses == 0, else baseWinRate
 * 4. reliability = numTrades / (numTrades + N0)
 * 5. freqPenalty = N1 / (numTrades + N1)
 * 6. denominator = EPS + maxDrawdown
 * 7. score = (r * adjWinRate * reliability * freqPenalty) / denominator
 *
 * @param stats - Account statistics for the period
 * @param params - Scoring hyperparameters (optional, uses defaults)
 * @returns Scoring result with final score and intermediate details
 */
export function computePerformanceScore(
  stats: AccountStats,
  params: ScoringParams = DEFAULT_SCORING_PARAMS
): ScoringResult {
  const { N0, N1, EPS } = params;
  const { realizedPnl, startingEquity, numTrades, numWins, numLosses, maxDrawdown } = stats;

  // Validate inputs to avoid NaN/Infinity
  if (!Number.isFinite(startingEquity) || startingEquity <= 0) {
    return createZeroResult('Invalid startingEquity');
  }
  if (!Number.isFinite(realizedPnl)) {
    return createZeroResult('Invalid realizedPnl');
  }
  if (!Number.isFinite(numTrades) || numTrades < 0) {
    return createZeroResult('Invalid numTrades');
  }
  if (!Number.isFinite(numWins) || numWins < 0) {
    return createZeroResult('Invalid numWins');
  }
  if (!Number.isFinite(numLosses) || numLosses < 0) {
    return createZeroResult('Invalid numLosses');
  }
  if (!Number.isFinite(maxDrawdown) || maxDrawdown < 0) {
    return createZeroResult('Invalid maxDrawdown');
  }

  // Step 1: Compute 30-day return normalized by account size
  const normalizedReturn = realizedPnl / startingEquity;

  // Step 2: Compute smoothed win rate using Laplace (add-one) smoothing
  // This prevents 100% win rate with few trades from dominating
  const baseWinRate = (numWins + 1.0) / (numWins + numLosses + 2.0);

  // Step 3: Apply extra penalty if zero losing trades (suspicious 100% win rate)
  const adjWinRate = numLosses === 0 ? 0.8 * baseWinRate : baseWinRate;

  // Step 4: Compute trade-count reliability factor
  // Makes statistics with very few trades less trusted
  // Approaches 1 as numTrades increases, starts at 0
  const reliability = numTrades / (numTrades + N0);

  // Step 5: Compute high-frequency penalty
  // Earning the same return with fewer trades is better
  // Approaches 1 when numTrades is small, shrinks as numTrades grows
  const freqPenalty = N1 / (numTrades + N1);

  // Step 6: Use max drawdown as penalty in denominator
  // Lower drawdown = higher score
  const denominator = EPS + maxDrawdown;

  // Step 7: Compute final composite performance score
  const score = (normalizedReturn * adjWinRate * reliability * freqPenalty) / denominator;

  return {
    score: Number.isFinite(score) ? score : 0,
    details: {
      normalizedReturn,
      baseWinRate,
      adjWinRate,
      reliability,
      freqPenalty,
      denominator
    }
  };
}

/**
 * Creates a zero-score result for invalid inputs
 */
function createZeroResult(_reason: string): ScoringResult {
  return {
    score: 0,
    details: {
      normalizedReturn: 0,
      baseWinRate: 0,
      adjWinRate: 0,
      reliability: 0,
      freqPenalty: 1,
      denominator: DEFAULT_SCORING_PARAMS.EPS
    }
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
      meta: account.meta
    };
  });

  // Sort by score descending
  scored.sort((a, b) => b.score - a.score);

  // Assign ranks (1-based)
  return scored.map((account, index) => ({
    ...account,
    rank: index + 1
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
      isCustom: false
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
      isCustom: true
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
    meta: account.meta
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
  startingEquity?: number;
  starting_equity?: number;
  accountValue?: number;
  account_value?: number;
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
  maxDrawdown?: number;
  max_drawdown?: number;
  statMaxDrawdown?: number;
  stat_max_drawdown?: number;
}): AccountStats {
  // Get realized PnL
  const realizedPnl = entry.realizedPnl ?? entry.realized_pnl ?? 0;

  // Get starting equity (use account value as fallback)
  const startingEquity = entry.startingEquity ?? entry.starting_equity ??
    entry.accountValue ?? entry.account_value ?? 1;

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

  // Get max drawdown (ensure it's positive)
  const rawDrawdown = entry.maxDrawdown ?? entry.max_drawdown ??
    entry.statMaxDrawdown ?? entry.stat_max_drawdown ?? 0;
  const maxDrawdown = Math.abs(rawDrawdown);

  return {
    realizedPnl,
    startingEquity: Math.max(startingEquity, 1), // Avoid division by zero
    numTrades,
    numWins,
    numLosses,
    maxDrawdown
  };
}
