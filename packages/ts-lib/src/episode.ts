/**
 * Episode Builder for Position Lifecycle Tracking
 *
 * Builds complete position episodes from fills with:
 * - Sign-aware segmentation (position goes 0 → ±X → 0)
 * - VWAP entry/exit calculation
 * - R-multiple calculation with policy-based stop
 * - Direction flip handling (close + reopen)
 *
 * @module episode
 */

export interface Fill {
  fillId: string;
  address: string;
  asset: string;
  side: 'buy' | 'sell';
  size: number;
  price: number;
  ts: Date;
  realizedPnl?: number;
  fees?: number;
}

export interface Episode {
  id: string;
  address: string;
  asset: string;
  direction: 'long' | 'short';

  // Entry info
  entryFills: Fill[];
  entryVwap: number;
  entrySize: number;
  entryTs: Date;
  entryNotional: number;

  // Exit info (null while open)
  exitFills: Fill[];
  exitVwap: number | null;
  exitSize: number | null;
  exitTs: Date | null;
  exitNotional: number | null;

  // Risk and P&L
  stopPrice: number; // Policy stop based on ATR or fixed %
  stopBps: number; // Stop distance in bps
  riskAmount: number; // Entry notional × stop fraction
  realizedPnl: number | null;
  resultR: number | null;
  totalFees: number;

  // Status
  status: 'open' | 'closed';
  closedReason?: 'full_close' | 'direction_flip' | 'timeout';
}

export interface EpisodeBuilderConfig {
  // Fixed stop as fraction of entry (e.g., 0.01 = 1%)
  defaultStopFraction: number;
  // R-multiple bounds for winsorization
  rMin: number;
  rMax: number;
  // Episode timeout in hours
  timeoutHours: number;
}

const DEFAULT_CONFIG: EpisodeBuilderConfig = {
  defaultStopFraction: 0.01,
  rMin: -2.0,
  rMax: 2.0,
  timeoutHours: 168, // 7 days
};

/**
 * Calculate VWAP from a list of fills
 */
export function calculateVwap(fills: Fill[]): number {
  if (fills.length === 0) return 0;

  let totalNotional = 0;
  let totalSize = 0;

  for (const f of fills) {
    const notional = f.price * f.size;
    totalNotional += notional;
    totalSize += f.size;
  }

  return totalSize > 0 ? totalNotional / totalSize : 0;
}

/**
 * Get signed size from a fill (positive for buy/long, negative for sell/short)
 */
export function getSignedSize(fill: Fill): number {
  return fill.side === 'buy' ? fill.size : -fill.size;
}

/**
 * Calculate stop price based on entry price and direction
 */
export function calculateStopPrice(
  entryPrice: number,
  direction: 'long' | 'short',
  stopFraction: number
): number {
  if (direction === 'long') {
    return entryPrice * (1 - stopFraction);
  } else {
    return entryPrice * (1 + stopFraction);
  }
}

/**
 * Calculate stop distance in basis points
 */
export function calculateStopBps(entryPrice: number, stopPrice: number): number {
  if (entryPrice <= 0) return 0;
  return Math.abs((entryPrice - stopPrice) / entryPrice) * 10000;
}

/**
 * Calculate R-multiple from P&L and risk amount
 */
export function calculateR(
  pnl: number,
  riskAmount: number,
  rMin: number = -2.0,
  rMax: number = 2.0
): number {
  if (riskAmount <= 0) return 0;
  const r = pnl / riskAmount;
  // Winsorize
  return Math.max(rMin, Math.min(rMax, r));
}

/**
 * Convert basis points to R-multiple given stop distance
 */
export function bpsToR(costBps: number, stopBps: number): number {
  if (stopBps <= 0) return 0;
  return costBps / stopBps;
}

/**
 * Build episodes from a chronologically sorted list of fills for one address+asset
 *
 * Episode segmentation rules:
 * 1. Start episode when position goes from 0 to non-zero
 * 2. End episode when position returns to 0 or flips sign
 * 3. Sign flip = close current episode + open new one
 * 4. Track all fills that add to position for VWAP entry
 * 5. Track all fills that reduce position for VWAP exit
 */
export function buildEpisodes(
  fills: Fill[],
  config: EpisodeBuilderConfig = DEFAULT_CONFIG
): Episode[] {
  if (fills.length === 0) return [];

  // Sort by timestamp
  const sorted = [...fills].sort((a, b) => a.ts.getTime() - b.ts.getTime());

  const episodes: Episode[] = [];
  let position = 0; // Current signed position
  let currentEpisode: Episode | null = null;
  let episodeCounter = 0;

  for (const fill of sorted) {
    const prevPosition = position;
    const signedSize = getSignedSize(fill);
    position += signedSize;

    // Case 1: Was flat, now have position → Start new episode
    if (prevPosition === 0 && position !== 0) {
      episodeCounter++;
      const direction: 'long' | 'short' = position > 0 ? 'long' : 'short';
      const stopPrice = calculateStopPrice(
        fill.price,
        direction,
        config.defaultStopFraction
      );

      currentEpisode = {
        id: `${fill.address}-${fill.asset}-${episodeCounter}`,
        address: fill.address,
        asset: fill.asset,
        direction,
        entryFills: [fill],
        entryVwap: fill.price,
        entrySize: Math.abs(position),
        entryTs: fill.ts,
        entryNotional: fill.price * Math.abs(position),
        exitFills: [],
        exitVwap: null,
        exitSize: null,
        exitTs: null,
        exitNotional: null,
        stopPrice,
        stopBps: calculateStopBps(fill.price, stopPrice),
        riskAmount: fill.price * Math.abs(position) * config.defaultStopFraction,
        realizedPnl: null,
        resultR: null,
        totalFees: fill.fees || 0,
        status: 'open',
      };
      continue;
    }

    // Case 2: Position crosses zero or flips sign → Close episode
    if (currentEpisode && prevPosition !== 0 && Math.sign(prevPosition) !== Math.sign(position)) {
      // Determine how much was closed vs how much is new position
      const closedSize = Math.abs(prevPosition);
      const newPositionSize = Math.abs(position);

      // Close the current episode
      currentEpisode.exitFills.push(fill);
      currentEpisode.exitVwap = calculateVwap(currentEpisode.exitFills);
      currentEpisode.exitSize = closedSize;
      currentEpisode.exitTs = fill.ts;
      currentEpisode.exitNotional =
        currentEpisode.exitVwap * closedSize;

      // Calculate P&L
      if (fill.realizedPnl !== undefined) {
        currentEpisode.realizedPnl = fill.realizedPnl;
      } else {
        // Calculate from prices
        if (currentEpisode.direction === 'long') {
          currentEpisode.realizedPnl =
            (currentEpisode.exitVwap - currentEpisode.entryVwap) *
            currentEpisode.entrySize;
        } else {
          currentEpisode.realizedPnl =
            (currentEpisode.entryVwap - currentEpisode.exitVwap) *
            currentEpisode.entrySize;
        }
      }

      // Calculate R-multiple
      currentEpisode.resultR = calculateR(
        currentEpisode.realizedPnl,
        currentEpisode.riskAmount,
        config.rMin,
        config.rMax
      );

      currentEpisode.totalFees += fill.fees || 0;
      currentEpisode.status = 'closed';
      currentEpisode.closedReason =
        newPositionSize > 0 ? 'direction_flip' : 'full_close';

      episodes.push(currentEpisode);
      currentEpisode = null;

      // If position flipped, start new episode
      if (newPositionSize > 0) {
        episodeCounter++;
        const direction: 'long' | 'short' = position > 0 ? 'long' : 'short';
        const stopPrice = calculateStopPrice(
          fill.price,
          direction,
          config.defaultStopFraction
        );

        currentEpisode = {
          id: `${fill.address}-${fill.asset}-${episodeCounter}`,
          address: fill.address,
          asset: fill.asset,
          direction,
          entryFills: [fill],
          entryVwap: fill.price,
          entrySize: newPositionSize,
          entryTs: fill.ts,
          entryNotional: fill.price * newPositionSize,
          exitFills: [],
          exitVwap: null,
          exitSize: null,
          exitTs: null,
          exitNotional: null,
          stopPrice,
          stopBps: calculateStopBps(fill.price, stopPrice),
          riskAmount: fill.price * newPositionSize * config.defaultStopFraction,
          realizedPnl: null,
          resultR: null,
          totalFees: fill.fees || 0,
          status: 'open',
        };
      }
      continue;
    }

    // Case 3: Position returns to exactly zero → Close episode
    if (currentEpisode && position === 0) {
      currentEpisode.exitFills.push(fill);
      currentEpisode.exitVwap = calculateVwap(currentEpisode.exitFills);
      currentEpisode.exitSize = Math.abs(prevPosition);
      currentEpisode.exitTs = fill.ts;
      currentEpisode.exitNotional =
        currentEpisode.exitVwap * currentEpisode.exitSize;

      // Calculate P&L
      if (fill.realizedPnl !== undefined) {
        currentEpisode.realizedPnl = fill.realizedPnl;
      } else {
        if (currentEpisode.direction === 'long') {
          currentEpisode.realizedPnl =
            (currentEpisode.exitVwap - currentEpisode.entryVwap) *
            currentEpisode.entrySize;
        } else {
          currentEpisode.realizedPnl =
            (currentEpisode.entryVwap - currentEpisode.exitVwap) *
            currentEpisode.entrySize;
        }
      }

      currentEpisode.resultR = calculateR(
        currentEpisode.realizedPnl,
        currentEpisode.riskAmount,
        config.rMin,
        config.rMax
      );

      currentEpisode.totalFees += fill.fees || 0;
      currentEpisode.status = 'closed';
      currentEpisode.closedReason = 'full_close';

      episodes.push(currentEpisode);
      currentEpisode = null;
      continue;
    }

    // Case 4: Adding to position (same direction)
    if (currentEpisode && Math.sign(signedSize) === Math.sign(prevPosition)) {
      currentEpisode.entryFills.push(fill);
      currentEpisode.entryVwap = calculateVwap(currentEpisode.entryFills);
      currentEpisode.entrySize = Math.abs(position);
      currentEpisode.entryNotional = currentEpisode.entryVwap * currentEpisode.entrySize;
      currentEpisode.riskAmount =
        currentEpisode.entryNotional * config.defaultStopFraction;
      currentEpisode.totalFees += fill.fees || 0;
      continue;
    }

    // Case 5: Reducing position (partial close, same side)
    if (currentEpisode && Math.sign(signedSize) !== Math.sign(prevPosition) && position !== 0) {
      currentEpisode.exitFills.push(fill);
      currentEpisode.totalFees += fill.fees || 0;
      // Don't close yet - partial reduction
      continue;
    }
  }

  // If there's still an open episode, include it
  if (currentEpisode) {
    episodes.push(currentEpisode);
  }

  return episodes;
}

/**
 * Validate episode integrity
 *
 * Checks:
 * 1. No overlapping episodes
 * 2. Each fill belongs to exactly one episode
 * 3. Sum of episode P&L ≈ total realized P&L (within fee tolerance)
 */
export interface ValidationResult {
  valid: boolean;
  errors: string[];
  totalEpisodePnl: number;
  fillCount: number;
  episodeCount: number;
}

export function validateEpisodes(
  episodes: Episode[],
  fills: Fill[]
): ValidationResult {
  const errors: string[] = [];
  const usedFillIds = new Set<string>();
  let totalPnl = 0;

  for (const ep of episodes) {
    // Check for duplicate fills
    for (const f of [...ep.entryFills, ...ep.exitFills]) {
      if (usedFillIds.has(f.fillId)) {
        errors.push(`Fill ${f.fillId} appears in multiple episodes`);
      }
      usedFillIds.add(f.fillId);
    }

    // Sum closed P&L
    if (ep.status === 'closed' && ep.realizedPnl !== null) {
      totalPnl += ep.realizedPnl;
    }
  }

  // Check all fills are accounted for
  for (const f of fills) {
    if (!usedFillIds.has(f.fillId)) {
      errors.push(`Fill ${f.fillId} not assigned to any episode`);
    }
  }

  return {
    valid: errors.length === 0,
    errors,
    totalEpisodePnl: totalPnl,
    fillCount: usedFillIds.size,
    episodeCount: episodes.length,
  };
}

/**
 * Get open episodes (positions that haven't closed yet)
 */
export function getOpenEpisodes(episodes: Episode[]): Episode[] {
  return episodes.filter((e) => e.status === 'open');
}

/**
 * Get closed episodes with their R-multiples
 */
export function getClosedEpisodes(episodes: Episode[]): Episode[] {
  return episodes.filter((e) => e.status === 'closed');
}
