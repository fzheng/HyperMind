/**
 * Consensus Detection Module
 *
 * Generates trading signals when multiple selected traders agree.
 * Implements correlation-adjusted effective-K and EV gating.
 *
 * Gates:
 * 1. Supermajority: ≥70% agreement, ≥3 traders
 * 2. Effective-K ≥ 2.0 (correlation adjusted)
 * 3. Freshness: staleness and price drift checks
 * 4. EV gate: positive expected value after costs
 *
 * @module consensus
 */

export interface Vote {
  address: string;
  direction: 'long' | 'short';
  weight: number; // |Δexposure|/equity, capped at 1.0
  price: number; // Entry price
  ts: Date; // Vote timestamp
  episodeId?: string; // Source episode
}

export interface ConsensusConfig {
  minTraders: number; // Minimum agreeing traders
  minPct: number; // Minimum agreement percentage (0-1)
  minEffectiveK: number; // Minimum correlation-adjusted K
  maxStalenessFactor: number; // Max (oldest_ts / window)
  maxPriceDriftR: number; // Max price drift from median in R-units
  evMinR: number; // Minimum net EV in R-units
  defaultCorrelation: number; // Default pairwise correlation (ρ_base)
  correlationShrinkage: number; // λ for shrinkage: ρ' = λρ + (1-λ)ρ_base
  minPairsForCorrelation: number; // Minimum pairs before using measured ρ
  maxWeightPerCluster: number; // Cap sum of weights per correlated cluster
  avgWinR: number; // Average winning R-multiple
  avgLossR: number; // Average losing R-multiple
  feesBps: number; // Round-trip fees in bps
  slipBps: number; // Expected slippage in bps
}

export const DEFAULT_CONSENSUS_CONFIG: ConsensusConfig = {
  minTraders: 3,
  minPct: 0.7,
  minEffectiveK: 2.0,
  maxStalenessFactor: 1.25,
  maxPriceDriftR: 0.25, // Max drift in R-units (was 8 bps)
  evMinR: 0.2,
  defaultCorrelation: 0.3, // ρ_base
  correlationShrinkage: 0.7, // 70% measured, 30% prior
  minPairsForCorrelation: 3, // Need at least 3 pairs to use measured ρ
  maxWeightPerCluster: 0.6, // No cluster can dominate >60%
  avgWinR: 0.5,
  avgLossR: 0.3,
  feesBps: 7.0,
  slipBps: 10.0,
};

export interface ConsensusResult {
  passes: boolean;
  direction?: 'long' | 'short';
  confidence?: number;
  effectiveK?: number;
  evNetR?: number;
  gateResults: GateResults;
  votes?: Vote[];
  medianPrice?: number;
}

export interface GateResults {
  supermajority: { passed: boolean; agreeing: number; total: number; pct: number };
  effectiveK: { passed: boolean; value: number; required: number };
  freshness: { passed: boolean; staleness: number; maxStaleness: number };
  priceDrift: { passed: boolean; driftR: number; maxR: number }; // Now in R-units
  ev: { passed: boolean; evGrossR: number; evCostR: number; evNetR: number };
}

/**
 * Apply shrinkage to correlation estimate
 *
 * ρ' = λ × ρ_measured + (1-λ) × ρ_base
 *
 * This stabilizes noisy estimates when we have few pairs.
 */
export function shrinkCorrelation(
  rhoMeasured: number | undefined,
  rhoBase: number,
  lambda: number
): number {
  if (rhoMeasured === undefined) {
    return rhoBase;
  }
  const rho = lambda * rhoMeasured + (1 - lambda) * rhoBase;
  return Math.max(0, Math.min(1, rho));
}

/**
 * Calculate effective K with correlation adjustment and shrinkage
 *
 * Formula: effK = (Σwᵢ)² / ΣᵢΣⱼ wᵢwⱼρᵢⱼ
 *
 * With uniform weights and common correlation:
 * effK = K / (1 + (K-1)ρ)
 *
 * Stability features:
 * 1. Shrinkage: ρ_ij ← λρ_ij + (1-λ)ρ_base
 * 2. Floor: if <minPairs pairs, use ρ_base for all
 * 3. Returns {effK, measuredPairs} for diagnostics
 */
export function calculateEffectiveK(
  weights: Map<string, number>,
  correlationMatrix: Map<string, number>,
  config: {
    defaultCorrelation?: number;
    correlationShrinkage?: number;
    minPairsForCorrelation?: number;
  } = {}
): number {
  const rhoBase = config.defaultCorrelation ?? DEFAULT_CONSENSUS_CONFIG.defaultCorrelation;
  const lambda = config.correlationShrinkage ?? DEFAULT_CONSENSUS_CONFIG.correlationShrinkage;
  const minPairs = config.minPairsForCorrelation ?? DEFAULT_CONSENSUS_CONFIG.minPairsForCorrelation;

  const addrs = Array.from(weights.keys());
  if (addrs.length <= 1) return addrs.length;

  // Count measured pairs
  let measuredPairs = 0;
  for (let i = 0; i < addrs.length; i++) {
    for (let j = i + 1; j < addrs.length; j++) {
      const key = [addrs[i], addrs[j]].sort().join('|');
      if (correlationMatrix.has(key)) {
        measuredPairs++;
      }
    }
  }

  // If too few measured pairs, use ρ_base for all
  const useMeasured = measuredPairs >= minPairs;

  const sumWeights = Array.from(weights.values()).reduce((a, b) => a + b, 0);
  const numerator = sumWeights ** 2;

  let denominator = 0;
  for (let i = 0; i < addrs.length; i++) {
    for (let j = 0; j < addrs.length; j++) {
      let rho: number;
      if (i === j) {
        rho = 1.0;
      } else if (!useMeasured) {
        // Not enough pairs - use base correlation
        rho = rhoBase;
      } else {
        // Use measured with shrinkage
        const key = [addrs[i], addrs[j]].sort().join('|');
        const rhoMeasured = correlationMatrix.get(key);
        rho = shrinkCorrelation(rhoMeasured, rhoBase, lambda);
      }
      denominator += weights.get(addrs[i])! * weights.get(addrs[j])! * rho;
    }
  }

  return numerator / Math.max(denominator, 1e-9);
}

/**
 * Calculate effective K with detailed diagnostics
 */
export function calculateEffectiveKWithDiagnostics(
  weights: Map<string, number>,
  correlationMatrix: Map<string, number>,
  config: {
    defaultCorrelation?: number;
    correlationShrinkage?: number;
    minPairsForCorrelation?: number;
  } = {}
): { effK: number; measuredPairs: number; totalPairs: number; usedMeasured: boolean } {
  const rhoBase = config.defaultCorrelation ?? DEFAULT_CONSENSUS_CONFIG.defaultCorrelation;
  const minPairs = config.minPairsForCorrelation ?? DEFAULT_CONSENSUS_CONFIG.minPairsForCorrelation;

  const addrs = Array.from(weights.keys());
  const totalPairs = (addrs.length * (addrs.length - 1)) / 2;

  let measuredPairs = 0;
  for (let i = 0; i < addrs.length; i++) {
    for (let j = i + 1; j < addrs.length; j++) {
      const key = [addrs[i], addrs[j]].sort().join('|');
      if (correlationMatrix.has(key)) {
        measuredPairs++;
      }
    }
  }

  const usedMeasured = measuredPairs >= minPairs;
  const effK = calculateEffectiveK(weights, correlationMatrix, config);

  return { effK, measuredPairs, totalPairs, usedMeasured };
}

/**
 * Check supermajority gate
 */
export function checkSupermajority(
  votes: Vote[],
  config: ConsensusConfig = DEFAULT_CONSENSUS_CONFIG
): { passed: boolean; direction: 'long' | 'short' | null; agreeing: number; pct: number } {
  if (votes.length === 0) {
    return { passed: false, direction: null, agreeing: 0, pct: 0 };
  }

  const longCount = votes.filter((v) => v.direction === 'long').length;
  const shortCount = votes.length - longCount;
  const majorityCount = Math.max(longCount, shortCount);
  const direction = longCount >= shortCount ? 'long' : 'short';
  const pct = majorityCount / votes.length;

  const passed = majorityCount >= config.minTraders && pct >= config.minPct;

  return { passed, direction: passed ? direction : null, agreeing: majorityCount, pct };
}

/**
 * Check freshness gate (staleness)
 *
 * Rejects if oldest vote is too stale relative to consensus window
 */
export function checkFreshness(
  votes: Vote[],
  windowMs: number,
  config: ConsensusConfig = DEFAULT_CONSENSUS_CONFIG
): { passed: boolean; staleness: number } {
  if (votes.length === 0) {
    return { passed: false, staleness: Infinity };
  }

  const now = Date.now();
  const oldestTs = Math.min(...votes.map((v) => v.ts.getTime()));
  const ageMs = now - oldestTs;
  const staleness = ageMs / windowMs;

  return {
    passed: staleness <= config.maxStalenessFactor,
    staleness,
  };
}

/**
 * Check price drift gate (in R-units)
 *
 * Rejects if current mid price has drifted too far from median voter entry.
 * Drift is expressed in R-units: driftR = driftBps / stopBps
 *
 * This ensures the gate scales properly with different stop sizes.
 */
export function checkPriceDrift(
  votes: Vote[],
  currentMidPrice: number,
  stopBps: number,
  config: ConsensusConfig = DEFAULT_CONSENSUS_CONFIG
): { passed: boolean; driftBps: number; driftR: number } {
  if (votes.length === 0 || currentMidPrice <= 0) {
    return { passed: false, driftBps: Infinity, driftR: Infinity };
  }

  // Calculate median entry price
  const prices = votes.map((v) => v.price).sort((a, b) => a - b);
  const mid = Math.floor(prices.length / 2);
  const medianPrice =
    prices.length % 2 === 0 ? (prices[mid - 1] + prices[mid]) / 2 : prices[mid];

  // Calculate drift in bps
  const driftBps = Math.abs((currentMidPrice - medianPrice) / medianPrice) * 10000;

  // Convert to R-units
  const driftR = stopBps > 0 ? driftBps / stopBps : Infinity;

  // Gate on R-units, not raw bps
  return {
    passed: driftR <= config.maxPriceDriftR,
    driftBps,
    driftR,
  };
}

/**
 * Calculate expected value in R-units
 */
export function calculateEV(
  pWin: number,
  stopBps: number,
  config: ConsensusConfig = DEFAULT_CONSENSUS_CONFIG
): { evGrossR: number; evCostR: number; evNetR: number } {
  const grossEv = pWin * config.avgWinR - (1 - pWin) * config.avgLossR;

  // Convert costs to R-units
  const totalCostBps = config.feesBps + config.slipBps;
  const costR = stopBps > 0 ? totalCostBps / stopBps : Infinity;

  return {
    evGrossR: grossEv,
    evCostR: costR,
    evNetR: grossEv - costR,
  };
}

/**
 * Get median price from votes
 */
export function getMedianPrice(votes: Vote[]): number {
  if (votes.length === 0) return 0;

  const prices = votes.map((v) => v.price).sort((a, b) => a - b);
  const mid = Math.floor(prices.length / 2);
  return prices.length % 2 === 0 ? (prices[mid - 1] + prices[mid]) / 2 : prices[mid];
}

/**
 * Calculate win probability from trader posteriors
 *
 * Uses weighted average of trader win rates with uncertainty adjustment
 */
export function estimateWinProbability(
  votes: Vote[],
  traderWinRates: Map<string, { winRate: number; samples: number }>
): number {
  if (votes.length === 0) return 0.5;

  let weightedSum = 0;
  let totalWeight = 0;

  for (const vote of votes) {
    const stats = traderWinRates.get(vote.address);
    if (stats && stats.samples >= 5) {
      // Weight by number of samples (more data = more reliable)
      const sampleWeight = Math.min(stats.samples / 30, 1.0);
      weightedSum += stats.winRate * vote.weight * sampleWeight;
      totalWeight += vote.weight * sampleWeight;
    }
  }

  if (totalWeight === 0) return 0.5; // Default when no data

  // Apply shrinkage toward 0.5 based on effective sample size
  const rawPWin = weightedSum / totalWeight;
  const shrinkage = Math.min(totalWeight / 10, 1.0);
  return 0.5 * (1 - shrinkage) + rawPWin * shrinkage;
}

/**
 * Full consensus check with all gates
 */
export function checkConsensus(
  votes: Vote[],
  currentMidPrice: number,
  windowMs: number,
  stopBps: number,
  correlationMatrix: Map<string, number> = new Map(),
  traderWinRates: Map<string, { winRate: number; samples: number }> = new Map(),
  config: ConsensusConfig = DEFAULT_CONSENSUS_CONFIG
): ConsensusResult {
  // Gate 1: Supermajority
  const supermajorityResult = checkSupermajority(votes, config);
  if (!supermajorityResult.passed) {
    return {
      passes: false,
      gateResults: {
        supermajority: {
          passed: false,
          agreeing: supermajorityResult.agreeing,
          total: votes.length,
          pct: supermajorityResult.pct,
        },
        effectiveK: { passed: false, value: 0, required: config.minEffectiveK },
        freshness: { passed: false, staleness: 0, maxStaleness: config.maxStalenessFactor },
        priceDrift: { passed: false, driftR: 0, maxR: config.maxPriceDriftR },
        ev: { passed: false, evGrossR: 0, evCostR: 0, evNetR: 0 },
      },
    };
  }

  const direction = supermajorityResult.direction!;
  const agreeingVotes = votes.filter((v) => v.direction === direction);

  // Gate 2: Effective-K (with shrinkage and stability)
  const weights = new Map(agreeingVotes.map((v) => [v.address, v.weight]));
  const effectiveK = calculateEffectiveK(weights, correlationMatrix, {
    defaultCorrelation: config.defaultCorrelation,
    correlationShrinkage: config.correlationShrinkage,
    minPairsForCorrelation: config.minPairsForCorrelation,
  });
  const effectiveKPassed = effectiveK >= config.minEffectiveK;

  // Gate 3: Freshness
  const freshnessResult = checkFreshness(agreeingVotes, windowMs, config);

  // Gate 4: Price drift (now in R-units)
  const driftResult = checkPriceDrift(agreeingVotes, currentMidPrice, stopBps, config);

  // Gate 5: EV
  const pWin = estimateWinProbability(agreeingVotes, traderWinRates);
  const evResult = calculateEV(pWin, stopBps, config);
  const evPassed = evResult.evNetR >= config.evMinR;

  const allPassed =
    supermajorityResult.passed &&
    effectiveKPassed &&
    freshnessResult.passed &&
    driftResult.passed &&
    evPassed;

  return {
    passes: allPassed,
    direction,
    confidence: pWin,
    effectiveK,
    evNetR: evResult.evNetR,
    votes: agreeingVotes,
    medianPrice: getMedianPrice(agreeingVotes),
    gateResults: {
      supermajority: {
        passed: supermajorityResult.passed,
        agreeing: supermajorityResult.agreeing,
        total: votes.length,
        pct: supermajorityResult.pct,
      },
      effectiveK: { passed: effectiveKPassed, value: effectiveK, required: config.minEffectiveK },
      freshness: {
        passed: freshnessResult.passed,
        staleness: freshnessResult.staleness,
        maxStaleness: config.maxStalenessFactor,
      },
      priceDrift: {
        passed: driftResult.passed,
        driftR: driftResult.driftR,
        maxR: config.maxPriceDriftR,
      },
      ev: {
        passed: evPassed,
        evGrossR: evResult.evGrossR,
        evCostR: evResult.evCostR,
        evNetR: evResult.evNetR,
      },
    },
  };
}

/**
 * Adaptive consensus window based on volatility
 *
 * Low volatility: short window (faster signals)
 * High volatility: long window (more confirmation needed)
 */
export function adaptiveWindowMs(
  atrPercentile: number,
  baseWindowMs: number = 120000,
  loMultiplier: number = 0.5,
  hiMultiplier: number = 3.0
): number {
  if (atrPercentile < 0.3) {
    return baseWindowMs * loMultiplier;
  } else if (atrPercentile < 0.7) {
    return baseWindowMs;
  } else {
    return baseWindowMs * hiMultiplier;
  }
}

/**
 * Create ticket instrumentation for a consensus signal
 */
export interface TicketInstrumentation {
  // Inputs
  nTraders: number;
  nAgree: number;
  effectiveK: number;
  dispersion: number; // 1 - majority_pct
  stalenessFactor: number;
  driftR: number;
  pWin: number;
  // Calculations
  evGrossR: number;
  evCostR: number;
  evNetR: number;
  // Meta
  direction: 'long' | 'short';
  medianPrice: number;
  windowMs: number;
  stopBps: number;
  voterAddresses: string[];
}

export function createTicketInstrumentation(
  result: ConsensusResult,
  windowMs: number,
  stopBps: number
): TicketInstrumentation | null {
  if (!result.passes || !result.direction || !result.votes) {
    return null;
  }

  return {
    nTraders: result.gateResults.supermajority.total,
    nAgree: result.gateResults.supermajority.agreeing,
    effectiveK: result.effectiveK ?? 0,
    dispersion: 1 - result.gateResults.supermajority.pct,
    stalenessFactor: result.gateResults.freshness.staleness,
    driftR: result.gateResults.priceDrift.driftR, // Already in R-units
    pWin: result.confidence ?? 0.5,
    evGrossR: result.gateResults.ev.evGrossR,
    evCostR: result.gateResults.ev.evCostR,
    evNetR: result.gateResults.ev.evNetR,
    direction: result.direction,
    medianPrice: result.medianPrice ?? 0,
    windowMs,
    stopBps,
    voterAddresses: result.votes.map((v) => v.address),
  };
}
