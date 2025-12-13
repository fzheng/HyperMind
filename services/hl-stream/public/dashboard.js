/**
 * SigmaPilot Dashboard JavaScript
 *
 * Real-time dashboard for monitoring top Hyperliquid traders.
 * Features:
 * - Live leaderboard with performance metrics
 * - Real-time trade fills via WebSocket
 * - TradingView charts for BTC and ETH
 * - Custom account tracking
 * - Infinite scroll for historical fills
 *
 * @module dashboard
 */

// =====================
// Theme Management
// =====================
const themeButtons = document.querySelectorAll('.theme-toggle button');
let currentTheme = localStorage.getItem('theme') || 'auto';
let currentSymbol = 'BTCUSDT'; // Track current chart symbol

/**
 * Detects system color scheme preference.
 * @returns {'dark'|'light'} System theme preference
 */
function getSystemTheme() {
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

/**
 * Applies theme to document and updates UI state.
 * @param {'auto'|'light'|'dark'} theme - Theme to apply
 * @param {boolean} reloadChart - Whether to reload TradingView chart
 */
function applyTheme(theme, reloadChart = false) {
  const effectiveTheme = theme === 'auto' ? getSystemTheme() : theme;
  document.documentElement.setAttribute('data-theme', effectiveTheme);

  // Update active button
  themeButtons.forEach(btn => {
    btn.classList.toggle('active', btn.getAttribute('data-theme') === theme);
  });

  currentTheme = theme;
  localStorage.setItem('theme', theme);

  // Reload chart with new theme if requested
  if (reloadChart && typeof renderChart === 'function') {
    renderChart(currentSymbol);
  }
}

// Initialize theme
applyTheme(currentTheme);

// Listen to system theme changes when in auto mode
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
  if (currentTheme === 'auto') {
    applyTheme('auto', true);
  }
});

// Theme toggle buttons
themeButtons.forEach(btn => {
  btn.addEventListener('click', () => {
    applyTheme(btn.getAttribute('data-theme'), true);
  });
});

// =====================
// DOM Element References
// =====================
const statusEl = null; // Status element removed - using live clock instead
const addressTable = document.getElementById('address-table');
const fillsTable = document.getElementById('fills-table');
const aiRecommendationsTable = document.getElementById('ai-recommendations-table');
const aiStatusEl = document.getElementById('ai-status');
const symbolButtons = document.querySelectorAll('.toggle-group button');
const lastRefreshEl = document.getElementById('last-refresh');
const nextRefreshEl = document.getElementById('next-refresh');
const refreshStatusEl = document.getElementById('refresh-status');
const customCountEl = document.getElementById('custom-count');
const customAddressInput = document.getElementById('custom-address-input');
const addCustomBtn = document.getElementById('add-custom-btn');
const customErrorEl = document.getElementById('custom-accounts-error');

// =====================
// Configuration
// =====================
const API_BASE = '/dashboard/api';
const SCOUT_API = '/api'; // hl-scout API base (proxied via hl-stream)
const TOP_TABLE_LIMIT = 20; // 10 system + unlimited pinned
const MAX_CUSTOM_PINNED = 3;

// =====================
// Application State
// =====================
let fillsCache = [];
let dashboardPeriod = 30;
let addressMeta = {};
let legacyAddresses = new Set(); // Addresses in Legacy Leaderboard (for WebSocket fill filtering)
let customPinnedCount = 0;
let positionsReady = false; // Track whether positions have been loaded

// =====================
// Price Ticker State
// =====================
let lastBtcPrice = null;
let lastEthPrice = null;
const btcPriceEl = document.getElementById('btc-price');
const ethPriceEl = document.getElementById('eth-price');
const btcPriceItem = document.getElementById('btc-price-item');
const ethPriceItem = document.getElementById('eth-price-item');

// =====================
// Live Clock
// =====================
const liveClockEl = document.getElementById('live-clock');

/**
 * Updates the live clock display with current time.
 * Uses blinking separators for visual effect.
 */
function updateLiveClock() {
  if (!liveClockEl) return;
  const now = new Date();
  const hours = String(now.getHours()).padStart(2, '0');
  const minutes = String(now.getMinutes()).padStart(2, '0');
  const seconds = String(now.getSeconds()).padStart(2, '0');
  // Use spans for separators to enable blinking animation
  liveClockEl.innerHTML = `${hours}<span class="clock-separator">:</span>${minutes}<span class="clock-separator">:</span>${seconds}`;
}

// Start clock immediately and update every second
updateLiveClock();
setInterval(updateLiveClock, 1000);

// =====================
// Fill Aggregation
// =====================
// Time range tracking for infinite scroll
let fillsOldestTime = null;
let fillsNewestTime = null;
let isLoadingMore = false;
let hasMoreFills = true;
let totalFillCount = 0;
let isInitialLoad = true; // Prevent flashing during initial load
let isFirstWsConnect = true; // Track first WebSocket connection

// Time display mode: 'absolute' or 'relative'
let fillsTimeDisplayMode = 'absolute';
let relativeTimeInterval = null; // Interval for auto-updating relative times

// Aggregation settings: groups fills within 1-minute windows
const AGGREGATION_WINDOW_MS = 60000; // 1 minute window
const MAX_AGGREGATED_GROUPS = 50; // Max groups to keep in memory

// Streaming aggregation state - stores pre-aggregated groups
let aggregatedGroups = [];

// Track expanded groups for collapsible UI
const expandedGroups = new Set();

/**
 * Creates a new aggregation group from a single fill.
 * Groups track multiple fills that can be merged together.
 *
 * @param {Object} fill - Fill event data
 * @returns {Object} New aggregation group
 */
function createGroup(fill) {
  const symbol = (fill.symbol || 'BTC').toUpperCase();
  const normalizedAddress = (fill.address || '').toLowerCase();
  const normalizedAction = (fill.action || '').trim().toLowerCase();
  return {
    id: `${normalizedAddress}-${fill.time_utc}-${Math.random().toString(36).slice(2, 8)}`,
    time_utc: fill.time_utc,
    oldest_time: fill.time_utc,
    address: normalizedAddress, // Normalized for grouping
    originalAddress: fill.address, // Original for display
    symbol: symbol,
    action: normalizedAction, // Normalized for grouping
    originalAction: fill.action, // Original for display
    fills: [fill],
    totalSize: Math.abs(fill.size_signed || 0),
    totalPnl: fill.closed_pnl_usd || 0,
    prices: fill.price_usd ? [fill.price_usd] : [],
    // Store the resulting position from the newest fill (for calculating previous_position later)
    resulting_position: fill.resulting_position,
    previous_position: fill.previous_position,
    isAggregated: false,
    fillCount: 1,
    avgPrice: fill.price_usd || null,
    size_signed: fill.size_signed,
    closed_pnl_usd: fill.closed_pnl_usd,
    price_usd: fill.price_usd,
  };
}

/**
 * Checks if a fill can be merged into an existing group.
 * Fills must match address, symbol, action, and be within time window.
 *
 * @param {Object} group - Existing aggregation group
 * @param {Object} fill - Fill to check for mergeability
 * @returns {boolean} True if fill can be merged
 */
function canMergeIntoGroup(group, fill) {
  const fillTime = new Date(fill.time_utc).getTime();
  const groupNewestTime = new Date(group.time_utc).getTime();
  const groupOldestTime = new Date(group.oldest_time).getTime();

  // Check if fill is within the aggregation window of the group
  const timeDiffFromNewest = Math.abs(groupNewestTime - fillTime);
  const timeDiffFromOldest = Math.abs(groupOldestTime - fillTime);
  const withinWindow = timeDiffFromNewest <= AGGREGATION_WINDOW_MS || timeDiffFromOldest <= AGGREGATION_WINDOW_MS;

  const symbol = (fill.symbol || 'BTC').toUpperCase();
  const normalizedFillAddress = (fill.address || '').toLowerCase();
  const normalizedFillAction = (fill.action || '').trim().toLowerCase();
  const sameAddress = group.address === normalizedFillAddress;
  const sameSymbol = group.symbol === symbol;
  const sameAction = group.action === normalizedFillAction;

  return sameAddress && sameSymbol && sameAction && withinWindow;
}

/**
 * Calculates the previous_position for a group based on the resulting_position and totalSize.
 * This is more reliable than tracking individual fill's previous_position because:
 * - We know the resulting_position from the newest fill
 * - We know the totalSize (sum of all fill sizes in the group)
 * - We know the action type (determines direction of position change)
 *
 * Position sign conventions:
 * - Long positions are POSITIVE (e.g., +5.0 BTC long)
 * - Short positions are NEGATIVE (e.g., -5.0 BTC short)
 *
 * Action effects:
 * - "Open Long" / "Increase Long": position becomes MORE positive
 * - "Close Long" / "Decrease Long": position becomes LESS positive (toward 0)
 * - "Open Short" / "Increase Short": position becomes MORE negative
 * - "Close Short" / "Decrease Short": position becomes LESS negative (toward 0)
 *
 * @param {Object} group - The aggregation group to calculate previous_position for
 */
function calculateGroupPreviousPosition(group) {
  const resultingPos = group.resulting_position;
  const totalSize = group.totalSize;

  // If we don't have a resulting position, we can't calculate
  if (resultingPos == null || totalSize == null) {
    return;
  }

  const action = (group.action || '').toLowerCase();
  const isDecrease = action.includes('decrease') || action.includes('close');
  const isShort = action.includes('short');

  // Calculate previous_position by reversing the trade effect
  if (isShort) {
    // SHORT positions are negative
    if (isDecrease) {
      // "Close Short" / "Decrease Short" means buying to cover
      // Position went from more negative to less negative (or 0)
      // prev = result - totalSize (e.g., result=0, size=5 -> prev=-5)
      group.previous_position = resultingPos - totalSize;
    } else {
      // "Open Short" / "Increase Short" means selling to go more negative
      // Position went from less negative to more negative
      // prev = result + totalSize (e.g., result=-10, size=5 -> prev=-5)
      group.previous_position = resultingPos + totalSize;
    }
  } else {
    // LONG positions are positive
    if (isDecrease) {
      // "Close Long" / "Decrease Long" means selling to reduce position
      // Position went from more positive to less positive (or 0)
      // prev = result + totalSize (e.g., result=0, size=5 -> prev=+5)
      group.previous_position = resultingPos + totalSize;
    } else {
      // "Open Long" / "Increase Long" means buying to increase position
      // Position went from less positive to more positive
      // prev = result - totalSize (e.g., result=10, size=5 -> prev=+5)
      group.previous_position = resultingPos - totalSize;
    }
  }
}

/**
 * Merges a fill into an existing aggregation group.
 * Updates totals, time range, and computed fields.
 *
 * @param {Object} group - Group to merge into
 * @param {Object} fill - Fill to merge
 */
function mergeIntoGroup(group, fill) {
  const fillTime = new Date(fill.time_utc).getTime();
  const groupNewestTime = new Date(group.time_utc).getTime();
  const groupOldestTime = new Date(group.oldest_time).getTime();

  group.fills.push(fill);
  group.totalSize += Math.abs(fill.size_signed || 0);
  group.totalPnl += fill.closed_pnl_usd || 0;
  if (fill.price_usd) {
    group.prices.push(fill.price_usd);
  }

  // Update time range and track resulting_position from the newest fill
  if (fillTime > groupNewestTime) {
    group.time_utc = fill.time_utc;
    // Update resulting_position from the newest fill
    if (fill.resulting_position != null) {
      group.resulting_position = fill.resulting_position;
    }
  }
  if (fillTime < groupOldestTime) {
    group.oldest_time = fill.time_utc;
  }

  // Update computed fields
  group.fillCount = group.fills.length;
  group.isAggregated = group.fills.length > 1;
  group.avgPrice = group.prices.length > 0
    ? group.prices.reduce((a, b) => a + b, 0) / group.prices.length
    : null;

  // Update signed size based on action
  // Increase Long → positive (buying)
  // Decrease Long → negative (selling)
  // Increase Short → negative (selling to go short)
  // Decrease Short → positive (buying to cover)
  const isShort = group.action.toLowerCase().includes('short');
  const isDecrease = group.action.toLowerCase().includes('decrease') || group.action.toLowerCase().includes('close');
  // XOR logic: negative when (decrease AND long) OR (increase AND short)
  const isNegative = isDecrease !== isShort; // XOR: true when exactly one is true
  group.size_signed = isNegative ? -group.totalSize : group.totalSize;
  group.closed_pnl_usd = group.totalPnl || null;
  group.price_usd = group.avgPrice;

  // Calculate previous_position from resulting_position and totalSize
  // This is more reliable than tracking individual fill's previous_position
  // because grouped fills may have incomplete previous_position values during backfill
  calculateGroupPreviousPosition(group);
}

/**
 * Adds a new fill to the streaming aggregation.
 * Attempts to merge with existing group or creates new one.
 * Only processes BTC and ETH fills.
 *
 * @param {Object} fill - Fill event to aggregate
 */
function addFillToAggregation(fill) {
  const symbol = (fill.symbol || 'BTC').toUpperCase();
  // Only process BTC and ETH
  if (symbol !== 'BTC' && symbol !== 'ETH') return;

  // Try to merge with existing groups (check recent groups within time window)
  let merged = false;
  for (let i = 0; i < aggregatedGroups.length; i++) {
    const group = aggregatedGroups[i];

    // Only check groups that could potentially match (within 2x window for safety)
    const groupTime = new Date(group.time_utc).getTime();
    const fillTime = new Date(fill.time_utc).getTime();
    if (Math.abs(groupTime - fillTime) > AGGREGATION_WINDOW_MS * 2) {
      // Groups are sorted by time, so if we're past the window, stop checking
      if (fillTime > groupTime) continue;
      break;
    }

    if (canMergeIntoGroup(group, fill)) {
      mergeIntoGroup(group, fill);
      merged = true;
      break;
    }
  }

  // If not merged, create a new group
  if (!merged) {
    const newGroup = createGroup(fill);
    aggregatedGroups.unshift(newGroup);
  }

  // Sort groups by newest time (descending)
  aggregatedGroups.sort((a, b) => new Date(b.time_utc) - new Date(a.time_utc));

  // Trim to max size
  if (aggregatedGroups.length > MAX_AGGREGATED_GROUPS) {
    aggregatedGroups = aggregatedGroups.slice(0, MAX_AGGREGATED_GROUPS);
  }

  // Update time range tracking
  updateAggregatedTimeRange();
}

// Initialize aggregation from a batch of fills (e.g., initial load)
function initializeAggregation(fills) {
  // Filter to BTC/ETH only
  const btcEthFills = fills.filter(fill => {
    const symbol = (fill.symbol || 'BTC').toUpperCase();
    return symbol === 'BTC' || symbol === 'ETH';
  });

  // Use the existing batch aggregation for initial load
  aggregatedGroups = aggregateFills(btcEthFills);

  // Trim to max size
  if (aggregatedGroups.length > MAX_AGGREGATED_GROUPS) {
    aggregatedGroups = aggregatedGroups.slice(0, MAX_AGGREGATED_GROUPS);
  }

  // Update time range
  updateAggregatedTimeRange();
}

// Update time range from aggregated groups
function updateAggregatedTimeRange() {
  if (aggregatedGroups.length === 0) {
    fillsNewestTime = null;
    fillsOldestTime = null;
  } else {
    // Newest is from first group's time_utc
    fillsNewestTime = aggregatedGroups[0].time_utc;
    // Oldest is from last group's oldest_time
    fillsOldestTime = aggregatedGroups[aggregatedGroups.length - 1].oldest_time;
  }
  updateTimeRangeDisplay();
}

function placeholder(text = 'No live data') {
  return `<span class="placeholder">${text}</span>`;
}

// Format price for display (e.g., $97,234.56)
// Always shows full price with 2 decimal places - no K/M abbreviation
function fmtPrice(value) {
  if (!Number.isFinite(value)) return '—';
  return '$' + value.toLocaleString('en-US', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  });
}

// Format trade price for fills table (full price with 2 decimals, no K/M abbreviation)
function fmtTradePrice(value) {
  if (!Number.isFinite(value)) return 'N/A';
  return '$' + value.toLocaleString('en-US', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  });
}

// Update price ticker display
function updatePriceTicker(btc, eth) {
  if (btcPriceEl && Number.isFinite(btc)) {
    const prevBtc = lastBtcPrice;
    btcPriceEl.textContent = fmtPrice(btc);

    // Flash animation on price change
    if (prevBtc !== null && btc !== prevBtc) {
      btcPriceItem?.classList.remove('flash-up', 'flash-down');
      void btcPriceItem?.offsetWidth; // Trigger reflow
      btcPriceItem?.classList.add(btc > prevBtc ? 'flash-up' : 'flash-down');
    }
    lastBtcPrice = btc;
  }

  if (ethPriceEl && Number.isFinite(eth)) {
    const prevEth = lastEthPrice;
    ethPriceEl.textContent = fmtPrice(eth);

    // Flash animation on price change
    if (prevEth !== null && eth !== prevEth) {
      ethPriceItem?.classList.remove('flash-up', 'flash-down');
      void ethPriceItem?.offsetWidth; // Trigger reflow
      ethPriceItem?.classList.add(eth > prevEth ? 'flash-up' : 'flash-down');
    }
    lastEthPrice = eth;
  }
}

// Fetch initial prices
async function fetchPrices() {
  try {
    const data = await fetchJson(`${API_BASE}/prices`);
    updatePriceTicker(data.btc, data.eth);
  } catch (err) {
    console.error('Failed to fetch prices:', err);
  }
}

function fmtPercent(value) {
  if (!Number.isFinite(value)) return 'N/A';
  return `${(value * 100).toFixed(1)}%`;
}

function fmtUsdShort(value) {
  if (!Number.isFinite(value)) return 'N/A';
  if (value === 0) return '$0';
  const sign = value > 0 ? '+' : '-';
  const abs = Math.abs(value);
  const formatter = (num, suffix) => `${sign}$${num.toFixed(num >= 10 ? 1 : 2)}${suffix}`;
  if (abs >= 1e9) return formatter(abs / 1e9, 'B');
  if (abs >= 1e6) return formatter(abs / 1e6, 'M');
  if (abs >= 1e3) return formatter(abs / 1e3, 'K');
  return `${sign}$${abs.toFixed(0)}`;
}

function fmtTime(ts) {
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function fmtDateTime(ts) {
  return new Date(ts).toLocaleString([], { year: 'numeric', month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

/**
 * Format timestamp as relative time (e.g., "3 mins ago", "2 hours ago")
 * @param {string} ts - ISO timestamp
 * @returns {string} Relative time string
 */
function fmtRelativeTime(ts) {
  const now = Date.now();
  const then = new Date(ts).getTime();
  const diffMs = now - then;

  if (diffMs < 0) return 'just now';

  const seconds = Math.floor(diffMs / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (seconds < 60) return 'just now';
  if (minutes === 1) return '1 min ago';
  if (minutes < 60) return `${minutes} mins ago`;
  if (hours === 1) return '1 hour ago';
  if (hours < 24) return `${hours} hours ago`;
  if (days === 1) return '1 day ago';
  return `${days} days ago`;
}

/**
 * Format fill time based on current display mode
 * @param {string} ts - ISO timestamp
 * @returns {string} Formatted time string
 */
function fmtFillTime(ts) {
  if (fillsTimeDisplayMode === 'relative') {
    return fmtRelativeTime(ts);
  }
  return fmtDateTime(ts);
}

/**
 * Update only the time cells without full re-render (for relative time auto-refresh)
 */
function updateFillsTimeCells() {
  if (fillsTimeDisplayMode !== 'relative') return;

  const timeCells = fillsTable.querySelectorAll('td[data-label="Time"]');
  timeCells.forEach((cell) => {
    const absoluteTime = cell.getAttribute('title');
    if (absoluteTime) {
      cell.textContent = fmtRelativeTime(absoluteTime);
    }
  });
}

/**
 * Start auto-refresh interval for relative times
 */
function startRelativeTimeRefresh() {
  if (relativeTimeInterval) return; // Already running
  relativeTimeInterval = setInterval(updateFillsTimeCells, 1000);
}

/**
 * Stop auto-refresh interval for relative times
 */
function stopRelativeTimeRefresh() {
  if (relativeTimeInterval) {
    clearInterval(relativeTimeInterval);
    relativeTimeInterval = null;
  }
}

/**
 * Toggle between absolute and relative time display
 */
function toggleTimeDisplayMode() {
  fillsTimeDisplayMode = fillsTimeDisplayMode === 'absolute' ? 'relative' : 'absolute';

  // Update header text to show current mode
  const header = document.getElementById('fills-time-header');
  if (header) {
    header.textContent = fillsTimeDisplayMode === 'absolute' ? 'Time ⏱' : 'Time 🕐';
    header.title = fillsTimeDisplayMode === 'absolute'
      ? 'Click to show relative time (e.g., "3 mins ago")'
      : 'Click to show absolute time';
  }

  // Manage auto-refresh interval
  if (fillsTimeDisplayMode === 'relative') {
    startRelativeTimeRefresh();
  } else {
    stopRelativeTimeRefresh();
  }

  // Re-render fills with new time format
  renderAggregatedFills();
}

function shortAddress(address) {
  if (!address) return '';
  return `${address.slice(0, 6)}…${address.slice(-4)}`;
}

function formatHolding(entry) {
  if (!entry || !Number.isFinite(entry.size) || Math.abs(entry.size) < 0.0001) {
    // This function is only called for actual position entries
    // Empty/invalid entries should show "No BTC/ETH position"
    return placeholder('No BTC/ETH position');
  }
  const size = Number(entry.size);
  const symbol = (entry.symbol || '').toUpperCase();
  const direction = size >= 0 ? 'holding-long' : 'holding-short';
  const magnitude = Math.abs(size);
  const precision = magnitude >= 1 ? 2 : 3;
  const signed = `${size >= 0 ? '+' : '-'}${magnitude.toFixed(precision)} ${symbol || ''}`.trim();

  // Build tooltip with entry and liquidation prices
  const tooltipParts = [];
  if (entry.entryPrice != null && Number.isFinite(entry.entryPrice)) {
    tooltipParts.push(`Entry: $${entry.entryPrice.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`);
  }
  if (entry.liquidationPrice != null && Number.isFinite(entry.liquidationPrice)) {
    tooltipParts.push(`Liq: $${entry.liquidationPrice.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`);
  }
  if (entry.leverage != null && Number.isFinite(entry.leverage)) {
    tooltipParts.push(`${entry.leverage}x`);
  }
  const tooltip = tooltipParts.length > 0 ? tooltipParts.join(' | ') : 'Live position';

  return `<span class="holding-chip ${direction}" title="${tooltip}">${signed}</span>`;
}

function normalizeHoldings(raw = {}) {
  const normalized = {};
  Object.entries(raw).forEach(([addr, positions]) => {
    if (!addr) return;
    const key = addr.toLowerCase();
    // Handle both old format (single position object) and new format (array of positions)
    if (Array.isArray(positions)) {
      normalized[key] = positions.map(pos => ({
        symbol: (pos?.symbol || '').toUpperCase(),
        size: Number(pos?.size ?? 0),
        entryPrice: pos?.entryPrice ?? null,
        liquidationPrice: pos?.liquidationPrice ?? null,
        leverage: pos?.leverage ?? null,
      }));
    } else {
      // Legacy single position format
      normalized[key] = [{
        symbol: (positions?.symbol || '').toUpperCase(),
        size: Number(positions?.size ?? 0),
        entryPrice: positions?.entryPrice ?? null,
        liquidationPrice: positions?.liquidationPrice ?? null,
        leverage: positions?.leverage ?? null,
      }];
    }
  });
  return normalized;
}

function formatActionLabel(fill) {
  const action = fill.action ? String(fill.action).toLowerCase() : '';
  const map = {
    'open long': 'OPEN LONG',
    'increase long': 'ADD LONG',
    'close long (close all)': 'CLOSE LONG',
    'decrease long': 'CLOSE LONG',
    'open short': 'OPEN SHORT',
    'increase short': 'ADD SHORT',
    'close short (close all)': 'CLOSE SHORT',
    'decrease short': 'CLOSE SHORT',
  };
  if (map[action]) return map[action];
  if (fill.side === 'buy') return 'OPEN LONG';
  if (fill.side === 'sell') return 'OPEN SHORT';
  return action ? action.toUpperCase() : 'TRADE';
}

// Aggregate fills within a time window (same address, symbol, side/action)
function aggregateFills(fills) {
  if (!fills.length) return [];

  const aggregated = [];
  let currentGroup = null;

  // Sort by time descending (newest first)
  const sorted = [...fills].sort((a, b) => new Date(b.time_utc) - new Date(a.time_utc));

  for (const fill of sorted) {
    const fillTime = new Date(fill.time_utc).getTime();
    const symbol = (fill.symbol || 'BTC').toUpperCase();
    const action = (fill.action || '').trim().toLowerCase(); // Normalize for comparison
    const address = (fill.address || '').toLowerCase(); // Normalize address

    // Check if this fill should be grouped with current group
    if (currentGroup) {
      const groupTime = new Date(currentGroup.time_utc).getTime();
      const timeDiff = Math.abs(groupTime - fillTime);
      const sameAddress = currentGroup.address === address;
      const sameSymbol = currentGroup.symbol === symbol;
      const sameAction = currentGroup.action === action;

      if (sameAddress && sameSymbol && sameAction && timeDiff <= AGGREGATION_WINDOW_MS) {
        // Add to current group
        currentGroup.fills.push(fill);
        currentGroup.totalSize += Math.abs(fill.size_signed || 0);
        currentGroup.totalPnl += fill.closed_pnl_usd || 0;
        if (fill.price_usd) {
          currentGroup.prices.push(fill.price_usd);
        }
        // Update oldest time
        if (fillTime < new Date(currentGroup.oldest_time).getTime()) {
          currentGroup.oldest_time = fill.time_utc;
        }
        // Note: We don't track previous_position here anymore - it will be calculated
        // from resulting_position in finalizeGroup
        continue;
      }
    }

    // Finalize current group and start new one
    if (currentGroup) {
      finalizeGroup(currentGroup);
      aggregated.push(currentGroup);
    }

    // Start new group - the first fill is the newest in this group
    // so its resulting_position is the final position after all fills
    currentGroup = {
      id: `${address}-${fill.time_utc}-${Math.random().toString(36).slice(2, 8)}`,
      time_utc: fill.time_utc,
      oldest_time: fill.time_utc,
      address: address, // Normalized (lowercase)
      originalAddress: fill.address, // Original for display
      symbol: symbol,
      action: action, // Normalized (lowercase) for grouping
      originalAction: fill.action, // Original for display
      fills: [fill],
      totalSize: Math.abs(fill.size_signed || 0),
      totalPnl: fill.closed_pnl_usd || 0,
      prices: fill.price_usd ? [fill.price_usd] : [],
      // Store resulting_position from newest fill for calculating previous_position later
      resulting_position: fill.resulting_position,
      previous_position: fill.previous_position, // Fallback for single fills
      isAggregated: false,
    };
  }

  // Don't forget the last group
  if (currentGroup) {
    finalizeGroup(currentGroup);
    aggregated.push(currentGroup);
  }

  return aggregated;
}

function finalizeGroup(group) {
  group.isAggregated = group.fills.length > 1;
  group.fillCount = group.fills.length;
  group.avgPrice = group.prices.length > 0
    ? group.prices.reduce((a, b) => a + b, 0) / group.prices.length
    : null;
  // Determine signed size based on action
  // Increase Long → positive (buying)
  // Decrease Long → negative (selling)
  // Increase Short → negative (selling to go short)
  // Decrease Short → positive (buying to cover)
  const isShort = group.action.toLowerCase().includes('short');
  const isDecrease = group.action.toLowerCase().includes('decrease') || group.action.toLowerCase().includes('close');
  // XOR logic: negative when (decrease AND long) OR (increase AND short)
  const isNegative = isDecrease !== isShort; // XOR: true when exactly one is true
  group.size_signed = isNegative ? -group.totalSize : group.totalSize;
  group.closed_pnl_usd = group.totalPnl || null;
  group.price_usd = group.avgPrice;

  // For aggregated groups, calculate previous_position from resulting_position
  // This is more reliable than the individual fill's previous_position values
  if (group.isAggregated && group.resulting_position != null) {
    calculateGroupPreviousPosition(group);
  }
}

// Update time range display
function updateTimeRangeDisplay() {
  const timeRangeEl = document.getElementById('fills-time-range');
  if (!timeRangeEl) return;

  if (!fillsNewestTime && !fillsOldestTime) {
    timeRangeEl.textContent = 'No fills yet';
    return;
  }

  const formatTimeShort = (ts) => {
    const d = new Date(ts);
    const now = new Date();
    const isToday = d.toDateString() === now.toDateString();
    if (isToday) {
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    return d.toLocaleString([], { month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit' });
  };

  const newestStr = fillsNewestTime ? formatTimeShort(fillsNewestTime) : 'now';
  const oldestStr = fillsOldestTime ? formatTimeShort(fillsOldestTime) : '—';

  timeRangeEl.textContent = `${oldestStr} → ${newestStr}`;
}

// Update fills count display
function updateFillsCount() {
  const countEl = document.getElementById('fills-count');
  if (!countEl) return;

  // Count total individual fills across all groups
  totalFillCount = aggregatedGroups.reduce((sum, g) => sum + (g.fillCount || 1), 0);
  const groupCount = aggregatedGroups.length;

  if (totalFillCount === 0) {
    countEl.textContent = '0 fills';
  } else if (totalFillCount === groupCount) {
    countEl.textContent = `${totalFillCount} fill${totalFillCount !== 1 ? 's' : ''}`;
  } else {
    countEl.textContent = `${totalFillCount} fills (${groupCount} groups)`;
  }
}

// Update fills status bar
function updateFillsStatus() {
  const statusBar = document.getElementById('fills-status-bar');
  const statusMessage = document.getElementById('fills-status-message');
  const loadBtn = document.getElementById('load-history-btn');

  if (!statusBar || !statusMessage) return;

  if (aggregatedGroups.length === 0) {
    statusMessage.textContent = 'Waiting for live fills...';
    if (loadBtn) {
      loadBtn.textContent = 'Fetch History';
      loadBtn.disabled = false;
      loadBtn.classList.remove('loading');
    }
    statusBar.classList.remove('all-loaded');
  } else if (!hasMoreFills) {
    statusMessage.textContent = `All ${totalFillCount} fills loaded`;
    statusBar.classList.add('all-loaded');
  } else {
    // Calculate time span
    if (fillsOldestTime) {
      const oldestDate = new Date(fillsOldestTime);
      const now = new Date();
      const hoursDiff = Math.round((now - oldestDate) / (1000 * 60 * 60));

      if (hoursDiff < 1) {
        statusMessage.textContent = `Showing last ${totalFillCount} fills (< 1 hour)`;
      } else if (hoursDiff < 24) {
        statusMessage.textContent = `Showing last ${totalFillCount} fills (~${hoursDiff}h)`;
      } else {
        const daysDiff = Math.round(hoursDiff / 24);
        statusMessage.textContent = `Showing last ${totalFillCount} fills (~${daysDiff}d)`;
      }
    } else {
      statusMessage.textContent = `Showing ${totalFillCount} fills`;
    }

    if (loadBtn) {
      loadBtn.textContent = 'Load More';
      loadBtn.disabled = false;
      loadBtn.classList.remove('loading');
    }
    statusBar.classList.remove('all-loaded');
  }
}

// Update all fills UI elements
function updateFillsUI() {
  updateFillsCount();
  updateTimeRangeDisplay();
  updateFillsStatus();
}

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Request failed: ${res.status}`);
  return res.json();
}

function fmtScore(score) {
  if (!Number.isFinite(score)) return '—';
  if (score === 0) return '0';
  // Display score with appropriate precision
  if (Math.abs(score) >= 100) return score.toFixed(1);
  if (Math.abs(score) >= 1) return score.toFixed(2);
  return score.toFixed(4);
}

// Generate SVG sparkline from pnlList data
function generateSparkline(pnlList, width = 80, height = 24) {
  if (!pnlList || !Array.isArray(pnlList) || pnlList.length < 2) {
    return '<span class="placeholder">—</span>';
  }

  // Extract values and normalize
  const values = pnlList.map(p => parseFloat(p.value) || 0);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;

  // Generate points for polyline
  const points = values.map((v, i) => {
    const x = (i / (values.length - 1)) * width;
    const y = height - ((v - min) / range) * (height - 4) - 2; // 2px padding
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');

  // Determine color based on trend (first vs last value)
  const startVal = values[0];
  const endVal = values[values.length - 1];
  const isPositive = endVal >= startVal;
  const strokeColor = isPositive ? 'var(--positive)' : 'var(--negative)';

  return `
    <svg class="sparkline" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
      <polyline
        points="${points}"
        fill="none"
        stroke="${strokeColor}"
        stroke-width="1.5"
        stroke-linecap="round"
        stroke-linejoin="round"
      />
    </svg>
  `;
}

function renderAddresses(stats = [], profiles = {}, holdings = {}) {
  const rows = (stats || []).slice(0, TOP_TABLE_LIMIT);
  if (!rows.length) {
    addressTable.innerHTML = `<tr><td colspan="6">${placeholder('No live leaderboard data')}</td></tr>`;
    return;
  }
  addressTable.innerHTML = rows
    .map((row) => {
      const scoreValue = typeof row.score === 'number' ? row.score.toFixed(4) : 'N/A';
      const winRateCell = typeof row.winRate === 'number' ? fmtPercent(row.winRate) : placeholder();
      const tradesValue =
        typeof row.statClosedPositions === 'number'
          ? row.statClosedPositions
          : typeof row.executedOrders === 'number'
            ? row.executedOrders
            : null;
      const tradesCell = tradesValue === null ? placeholder() : tradesValue;
      const holdingKey = row.address?.toLowerCase() || '';
      const holdingPositions = holdings[holdingKey] || [];
      const holdingCell = holdingPositions.length > 0
        ? holdingPositions.map(pos => formatHolding(pos)).join(' ')
        : placeholder('No BTC/ETH position');
      const pnlCell = typeof row.realizedPnl === 'number' ? fmtUsdShort(row.realizedPnl) : placeholder();

      // Generate sparkline from pnlList
      const pnlList = row.meta?.raw?.pnlList || [];
      const sparklineCell = generateSparkline(pnlList);

      const isPinned = row.isPinned === true;
      const isCustom = row.isCustom === true;
      const pinIconClass = isPinned ? (isCustom ? 'pinned-custom' : 'pinned-leaderboard') : 'unpinned';
      const pinTitle = isPinned
        ? (isCustom ? 'Custom pinned account (click to unpin)' : 'Pinned from leaderboard (click to unpin)')
        : 'Click to pin this account';
      const pinIcon = `
        <span class="pin-icon ${pinIconClass}" data-address="${row.address}" data-pinned="${isPinned}" data-custom="${isCustom}" title="${pinTitle}">
          <svg viewBox="0 0 24 24"><path d="M16 4l4 4-8.5 8.5-4-4L16 4zm-8 8l4 4-6 6v-4l2-6zM4 20h4v-4l-4 4z"/></svg>
        </span>`;
      const nicknameDisplay = row.remark
        ? `<span class="nickname-display" data-address="${row.address}" data-nickname="${escapeHtml(row.remark)}">${escapeHtml(row.remark)}</span>`
        : '';
      const addrLower = (row.address || '').toLowerCase();
      const rowClass = isPinned ? 'pinned-row' : '';

      // Subscription method indicator (websocket vs polling)
      const subInfo = subscriptionMethods[addrLower];
      const subMethod = subInfo?.method || 'none';
      const subIndicator = formatSubscriptionIndicator(subMethod, subInfo?.sources || [], addrLower);

      return `
        <tr class="${rowClass}" data-address="${addrLower}">
          <td data-label="Address" title="Score: ${scoreValue}">
            <div class="address-cell-with-pin">
              ${subIndicator}
              ${pinIcon}
              <a class="address-link" href="https://hypurrscan.io/address/${row.address}" target="_blank" rel="noopener noreferrer">
                ${shortAddress(row.address)}
              </a>
            </div>
            ${nicknameDisplay ? `<div class="addr-remark">${nicknameDisplay}</div>` : ''}
          </td>
          <td data-label="Win Rate">${winRateCell}</td>
          <td data-label="Trades">${tradesCell}</td>
          <td data-label="Holdings" class="holds-cell">
            ${holdingCell}
          </td>
          <td data-label="Realized PnL">${pnlCell}</td>
          <td data-label="PnL Curve" class="sparkline-cell">${sparklineCell}</td>
        </tr>
      `;
    })
    .join('');

  // Attach event listeners for pin icons
  document.querySelectorAll('.pin-icon').forEach((icon) => {
    icon.addEventListener('click', async (e) => {
      e.preventDefault();
      e.stopPropagation();
      const address = icon.dataset.address;
      const isPinned = icon.dataset.pinned === 'true';
      if (address) {
        if (isPinned) {
          await unpinAccount(address);
        } else {
          await pinLeaderboardAccount(address);
        }
      }
    });
  });
}

// =====================
// Bandit Status
// =====================
const banditTradersTable = document.getElementById('bandit-traders-table');
const banditConfigEl = document.getElementById('bandit-config');
const banditStatsEl = document.getElementById('bandit-stats');
const banditRefreshBtn = document.getElementById('bandit-refresh-btn');

/**
 * Fetches and renders bandit status from hl-sage
 */
async function refreshBanditStatus() {
  if (banditRefreshBtn) {
    banditRefreshBtn.disabled = true;
    banditRefreshBtn.textContent = '...';
  }

  try {
    const data = await fetchJson(`${API_BASE}/bandit/status`);
    renderBanditStatus(data);
  } catch (err) {
    console.error('Failed to fetch bandit status:', err);
    renderBanditEmpty('Unable to load bandit data');
  } finally {
    if (banditRefreshBtn) {
      banditRefreshBtn.disabled = false;
      banditRefreshBtn.textContent = '↻';
    }
  }
}

/**
 * Renders bandit configuration display
 */
function renderBanditConfig(config) {
  if (!banditConfigEl || !config) return;

  banditConfigEl.innerHTML = `
    <div class="bandit-config-item">
      <span class="bandit-config-label">Pool Size:</span>
      <span class="bandit-config-value">${config.pool_size || 50}</span>
    </div>
    <div class="bandit-config-item">
      <span class="bandit-config-label">Select K:</span>
      <span class="bandit-config-value">${config.select_k || 10}</span>
    </div>
    <div class="bandit-config-item">
      <span class="bandit-config-label">Min Samples:</span>
      <span class="bandit-config-value">${config.min_samples || 5}</span>
    </div>
    <div class="bandit-config-item">
      <span class="bandit-config-label">Decay Factor:</span>
      <span class="bandit-config-value">${config.decay_factor || 0.95}</span>
    </div>
  `;
}

/**
 * Renders bandit stats in header
 */
function renderBanditStats(stats) {
  if (!banditStatsEl || !stats) return;

  const total = stats.total_traders_with_signals || 0;
  const reliable = stats.reliable_traders || 0;

  banditStatsEl.innerHTML = `
    <span>${reliable} reliable</span>
    <span>/ ${total} total</span>
  `;
}

/**
 * Renders bandit traders table
 */
function renderBanditTraders(traders) {
  if (!banditTradersTable) return;

  if (!traders || traders.length === 0) {
    renderBanditEmpty('No traders with signal history yet');
    return;
  }

  const rows = traders.map(trader => {
    const winRate = trader.win_rate || 0;
    const winRatePercent = (winRate * 100).toFixed(1);
    const posteriorMean = trader.posterior_mean || 0.5;
    const posteriorPercent = (posteriorMean * 100).toFixed(1);
    const signals = trader.total_signals || 0;

    // Confidence based on number of signals (more signals = more confidence)
    const confidenceLevel = signals >= 20 ? 'high' : signals >= 10 ? 'medium' : 'low';
    const confidenceWidth = Math.min(100, signals * 5); // 20 signals = 100%

    return `
      <tr>
        <td data-label="Address">
          <a href="https://hypurrscan.io/address/${trader.address}" target="_blank" rel="noopener noreferrer">
            ${shortAddress(trader.address)}
          </a>
        </td>
        <td data-label="Win Rate">${winRatePercent}%</td>
        <td data-label="Signals">${signals}</td>
        <td data-label="Posterior Mean">
          <div class="posterior-mean">
            <div class="posterior-bar">
              <div class="posterior-bar-fill" style="width: ${posteriorPercent}%"></div>
            </div>
            <span class="posterior-value">${posteriorPercent}%</span>
          </div>
        </td>
        <td data-label="Confidence">
          <div class="confidence-bar">
            <div class="confidence-bar-track">
              <div class="confidence-bar-fill ${confidenceLevel}" style="width: ${confidenceWidth}%"></div>
            </div>
            <span class="confidence-value">${signals} sig</span>
          </div>
        </td>
      </tr>
    `;
  }).join('');

  banditTradersTable.innerHTML = rows;
}

/**
 * Renders empty state for bandit
 */
function renderBanditEmpty(message) {
  if (!banditTradersTable) return;

  banditTradersTable.innerHTML = `
    <tr>
      <td colspan="5">
        <div class="bandit-empty-state">
          <span class="empty-icon">🎰</span>
          <p>${message}</p>
          <span class="empty-hint">Trader performance will be tracked after signal outcomes</span>
        </div>
      </td>
    </tr>
  `;
}

/**
 * Renders the complete bandit status
 */
function renderBanditStatus(data) {
  if (!data) {
    renderBanditEmpty('No bandit data available');
    return;
  }

  renderBanditConfig(data.config);
  renderBanditStats(data.stats);
  renderBanditTraders(data.top_traders);
}

/**
 * Initialize bandit controls
 */
function initBanditControls() {
  if (banditRefreshBtn) {
    banditRefreshBtn.addEventListener('click', refreshBanditStatus);
  }
}

// Consensus signals cache (fetched from hl-decide via API)
let consensusSignals = [];
let consensusSignalsLoading = false;

/**
 * Fetch real consensus signals from hl-decide
 */
async function fetchConsensusSignals() {
  if (consensusSignalsLoading) return;
  consensusSignalsLoading = true;

  try {
    const res = await fetch(`${API_BASE}/consensus/signals?limit=20`);
    if (!res.ok) throw new Error('Failed to fetch consensus signals');
    const data = await res.json();
    consensusSignals = data.signals || [];
    renderAIRecommendations();
  } catch (err) {
    console.error('Failed to fetch consensus signals:', err);
  } finally {
    consensusSignalsLoading = false;
  }
}

function renderAIRecommendations() {
  if (!aiRecommendationsTable) return;

  // Show empty state if no real signals
  if (consensusSignals.length === 0) {
    aiRecommendationsTable.innerHTML = `
      <tr>
        <td colspan="7">
          <div class="waiting-state">
            <span class="waiting-icon">⏳</span>
            <span class="waiting-text">Waiting for consensus signals...</span>
            <span class="waiting-hint">Signals appear when ≥3 Alpha Pool traders agree on direction</span>
          </div>
        </td>
      </tr>
    `;
    if (aiStatusEl) {
      aiStatusEl.innerHTML = `
        <span class="ai-status-dot inactive"></span>
        No Active Signals
      `;
    }
    return;
  }

  const rows = consensusSignals.map(signal => {
    // Map API fields to display format
    const action = signal.direction?.toUpperCase() || (signal.side === 'buy' ? 'LONG' : 'SHORT');
    const actionClass = action === 'LONG' ? 'buy' : 'sell';

    // Determine status from signal data
    let statusClass = '';
    let statusText = '';
    const status = signal.status || signal.outcome || 'active';

    switch (status.toLowerCase()) {
      case 'active':
      case 'open':
        statusClass = 'status-active';
        statusText = 'Active';
        break;
      case 'tp_hit':
      case 'win':
        statusClass = 'status-tp';
        statusText = 'TP Hit';
        break;
      case 'sl_hit':
      case 'loss':
        statusClass = 'status-sl';
        statusText = 'SL Hit';
        break;
      case 'expired':
      case 'timeout':
        statusClass = 'status-expired';
        statusText = 'Expired';
        break;
      case 'closed':
        statusClass = 'status-closed';
        statusText = 'Closed';
        break;
      default:
        statusClass = '';
        statusText = status;
    }

    // Get prices from signal
    const entry = signal.entry_price || signal.median_price || 0;
    const stopLoss = signal.stop_price || signal.stop_loss || 0;
    const takeProfit = signal.take_profit || signal.ev_r || 0;
    const time = signal.created_at || signal.ts || signal.time;

    return `
      <tr>
        <td data-label="Time">${fmtTime(time)}</td>
        <td data-label="Symbol">${signal.asset || signal.symbol || 'BTC'}</td>
        <td data-label="Action"><span class="pill ${actionClass}">${action}</span></td>
        <td data-label="Entry">${fmtPrice(entry)}</td>
        <td data-label="SL">${fmtPrice(stopLoss)}</td>
        <td data-label="TP">${fmtPrice(takeProfit)}</td>
        <td data-label="Status"><span class="ai-status-badge ${statusClass}">${statusText}</span></td>
      </tr>
    `;
  }).join('');

  aiRecommendationsTable.innerHTML = rows;

  // Update AI status
  if (aiStatusEl) {
    const activeCount = consensusSignals.filter(s => {
      const status = (s.status || s.outcome || 'active').toLowerCase();
      return status === 'active' || status === 'open';
    }).length;
    aiStatusEl.innerHTML = `
      <span class="ai-status-dot${activeCount > 0 ? '' : ' inactive'}"></span>
      ${activeCount} Active Signal${activeCount !== 1 ? 's' : ''}
    `;
  }
}

// Toggle group expansion - exposed globally for onclick handlers
window.toggleGroupExpansion = function(groupId) {
  if (expandedGroups.has(groupId)) {
    expandedGroups.delete(groupId);
  } else {
    expandedGroups.add(groupId);
  }
  renderAggregatedFills();
};

// Render a single aggregated group as a table row
function renderGroupRow(group, isNew = false) {
  const symbol = (group.symbol || 'BTC').toUpperCase();
  const sizeVal = group.isAggregated ? group.totalSize : Math.abs(group.size_signed || 0);
  const sizeSign = group.size_signed >= 0 ? '+' : '-';
  const size = typeof sizeVal === 'number' ? `${sizeSign}${sizeVal.toFixed(5)} ${symbol}` : '—';
  const prev = typeof group.previous_position === 'number' ? `${group.previous_position.toFixed(5)} ${symbol}` : '—';
  const price = group.isAggregated && group.avgPrice
    ? `~${fmtTradePrice(group.avgPrice)}`
    : fmtTradePrice(group.price_usd ?? null);
  const pnl = fmtUsdShort(group.closed_pnl_usd ?? null);
  // Use originalAction for display, fallback to action (for backwards compatibility)
  const displayAction = group.originalAction || group.action || '—';
  const sideClass = displayAction.toLowerCase().includes('short') ? 'sell' : 'buy';
  // Use originalAddress for display/links, address for data attributes (normalized)
  const displayAddress = group.originalAddress || group.address;

  const isExpanded = expandedGroups.has(group.id);

  // Show aggregation badge with expand/collapse functionality
  let aggBadge = '';
  if (group.isAggregated) {
    const expandIcon = isExpanded ? '▼' : '▶';
    aggBadge = `<span class="agg-badge" onclick="toggleGroupExpansion('${group.id}')" title="Click to ${isExpanded ? 'collapse' : 'expand'} ${group.fillCount} fills">
      <span class="expand-icon">${expandIcon}</span>×${group.fillCount}
    </span>`;
  }

  // New fill animation class
  const newClass = isNew ? 'new-fill-row' : '';

  const addrLower = (group.address || '').toLowerCase();
  let html = `
    <tr class="${group.isAggregated ? 'aggregated-row' : ''} ${newClass}" data-group-id="${group.id || ''}" data-address="${addrLower}">
      <td data-label="Time" title="${fmtDateTime(group.time_utc)}">${fmtFillTime(group.time_utc)}</td>
      <td data-label="Address"><a href="https://hypurrscan.io/address/${displayAddress}" target="_blank" rel="noopener noreferrer">${shortAddress(displayAddress)}</a></td>
      <td data-label="Action"><span class="pill ${sideClass}">${displayAction}</span>${aggBadge}</td>
      <td data-label="Size">${size}</td>
      <td data-label="Previous Position">${prev}</td>
      <td data-label="Price">${price}</td>
      <td data-label="Closed PnL">${pnl}</td>
    </tr>
  `;

  // Add expandable details row for aggregated fills
  if (group.isAggregated && isExpanded) {
    const subFills = group.fills.map(fill => {
      const fillSize = Math.abs(fill.size_signed || 0);
      const fillPrice = fmtTradePrice(fill.price_usd ?? null);
      const fillPnl = fmtUsdShort(fill.closed_pnl_usd ?? null);
      const fillTime = fmtTime(fill.time_utc);
      return `<div class="sub-fill">
        <span>${fillTime}</span>
        <span>${fillSize.toFixed(5)} ${symbol}</span>
        <span>${fillPrice}</span>
        <span>${fillPnl}</span>
      </div>`;
    }).join('');

    html += `
      <tr class="group-details expanded" data-parent-id="${group.id}">
        <td colspan="7">
          <div class="group-fills-list">${subFills}</div>
        </td>
      </tr>
    `;
  }

  return html;
}

// Render all aggregated groups
function renderAggregatedFills() {
  if (aggregatedGroups.length === 0) {
    fillsTable.innerHTML = `<tr><td colspan="7">
      <div class="waiting-state">
        <span class="waiting-icon">📡</span>
        <span class="waiting-text">Waiting for live fills...</span>
        <span class="waiting-hint">BTC/ETH trades will appear in real-time as they happen</span>
      </div>
    </td></tr>`;
    updateFillsUI();
    return;
  }

  const rows = aggregatedGroups.map(group => renderGroupRow(group)).join('');
  fillsTable.innerHTML = rows;

  // Attach cross-highlight hover events
  attachFillsHoverEvents();

  // Update all UI elements
  updateFillsUI();
}

// Cross-table highlight: hovering fills highlights matching leaderboard row
function attachFillsHoverEvents() {
  const fillRows = fillsTable.querySelectorAll('tr[data-address]');

  fillRows.forEach(row => {
    row.addEventListener('mouseenter', () => {
      const addr = row.dataset.address;
      if (!addr) return;
      // Find matching row in leaderboard
      const leaderboardRow = addressTable.querySelector(`tr[data-address="${addr}"]`);
      if (leaderboardRow) {
        leaderboardRow.classList.add('highlight-match');
      }
    });

    row.addEventListener('mouseleave', () => {
      const addr = row.dataset.address;
      if (!addr) return;
      const leaderboardRow = addressTable.querySelector(`tr[data-address="${addr}"]`);
      if (leaderboardRow) {
        leaderboardRow.classList.remove('highlight-match');
      }
    });
  });
}

// Legacy function for initial load - initializes streaming aggregation
function renderFills(list) {
  // Initialize the streaming aggregation with the batch
  initializeAggregation(list);

  // Mark initial load as complete to prevent flashing
  isInitialLoad = false;

  // Render the aggregated groups
  renderAggregatedFills();
}


function updateLastRefreshDisplay(lastRefresh) {
  if (lastRefresh && lastRefreshEl) {
    const date = new Date(lastRefresh);
    // Always show full date + time for clarity
    const formatted = date.toLocaleString([], {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
    lastRefreshEl.textContent = `Updated: ${formatted}`;
  } else if (lastRefreshEl) {
    lastRefreshEl.textContent = '';
  }
}

/**
 * Update refresh status display (countdown timer and refreshing indicator)
 */
function updateRefreshStatusDisplay(refreshStatus) {
  if (!refreshStatus) {
    if (nextRefreshEl) nextRefreshEl.textContent = '';
    if (refreshStatusEl) refreshStatusEl.style.display = 'none';
    return;
  }

  // Show refreshing indicator if in progress
  if (refreshStatus.isRefreshing) {
    if (refreshStatusEl) refreshStatusEl.style.display = 'inline-flex';
    if (nextRefreshEl) nextRefreshEl.textContent = '';
  } else {
    if (refreshStatusEl) refreshStatusEl.style.display = 'none';

    // Show countdown to next refresh
    if (nextRefreshEl && refreshStatus.nextRefreshInMs != null) {
      nextRefreshEl.textContent = `Next: ${formatTimeUntil(refreshStatus.nextRefreshInMs)}`;
    } else if (nextRefreshEl) {
      nextRefreshEl.textContent = '';
    }
  }
}

/**
 * Format milliseconds into human readable time (e.g., "23h 45m" or "15m")
 */
function formatTimeUntil(ms) {
  if (ms <= 0) return 'soon';

  const hours = Math.floor(ms / (60 * 60 * 1000));
  const minutes = Math.floor((ms % (60 * 60 * 1000)) / (60 * 1000));

  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }
  return `${minutes}m`;
}

function updateCustomPinnedCount(count, max) {
  customPinnedCount = count;
  customCountEl.textContent = count;
  // Disable add button if at max
  addCustomBtn.disabled = count >= max;
}

// Check positions status and poll until ready
const MAX_POSITION_POLL_RETRIES = 30; // Max 30 retries (60 seconds total)
let positionPollRetries = 0;

async function checkPositionsStatus() {
  try {
    const data = await fetchJson(`${API_BASE}/positions-status`);
    positionsReady = data.positionsReady === true;
    return positionsReady;
  } catch (err) {
    console.error('Failed to check positions status:', err);
    return false;
  }
}

async function pollPositionsUntilReady() {
  const isReady = await checkPositionsStatus();
  if (!isReady && positionPollRetries < MAX_POSITION_POLL_RETRIES) {
    positionPollRetries++;
    // Poll every 2 seconds until positions ready or max retries reached
    setTimeout(async () => {
      await checkPositionsStatus();
      await refreshSummary();
      if (!positionsReady && positionPollRetries < MAX_POSITION_POLL_RETRIES) {
        pollPositionsUntilReady();
      } else if (positionPollRetries >= MAX_POSITION_POLL_RETRIES) {
        console.warn('Position polling max retries reached, continuing without positions');
      }
    }, 2000);
  }
}

async function refreshSummary() {
  try {
    // Fetch summary and subscription data in parallel
    const [summaryData] = await Promise.all([
      fetchJson(`${API_BASE}/summary?period=${dashboardPeriod}&limit=${TOP_TABLE_LIMIT}`),
      refreshSubscriptionStatus() // Shared function for subscription methods AND status
    ]);
    const data = summaryData;

    const rows = Array.isArray(data.stats)
      ? data.stats
      : Array.isArray(data.selected)
        ? data.selected
        : [];
    const holdings = normalizeHoldings(data.holdings || {});
    addressMeta = {};
    legacyAddresses = new Set(); // Reset legacy addresses
    rows.forEach((row) => {
      if (!row?.address) return;
      const lowerAddr = row.address.toLowerCase();
      addressMeta[lowerAddr] = { remark: row.remark || null };
      legacyAddresses.add(lowerAddr); // Track for WebSocket fill filtering
    });
    renderAddresses(rows, data.profiles || {}, holdings);

    // Update last refresh display
    updateLastRefreshDisplay(data.lastRefresh);

    // Update refresh status (countdown timer, refreshing indicator)
    updateRefreshStatusDisplay(data.refreshStatus);

    // Update custom pinned account count
    if (typeof data.customPinnedCount === 'number') {
      updateCustomPinnedCount(data.customPinnedCount, data.maxCustomPinned || MAX_CUSTOM_PINNED);
    }
  } catch (err) {
    console.error('Failed to load summary:', err);
  }
}

async function refreshFills() {
  try {
    // Fetch fills and subscription status in parallel
    const [fillsData] = await Promise.all([
      fetchJson(`${API_BASE}/legacy/fills?limit=40`),
      refreshSubscriptionStatus() // Shared function for subscription data
    ]);
    const newFills = fillsData.fills || [];

    if (fillsCache.length === 0) {
      // Initial load - just use the new fills
      fillsCache = newFills;
      hasMoreFills = fillsData.hasMore !== false;
    } else {
      // Incremental update - merge new fills at the front, keeping loaded history
      // Find fills that are newer than our current newest
      const newestTime = fillsCache.length > 0 ? new Date(fillsCache[0].time_utc).getTime() : 0;
      const trulyNewFills = newFills.filter(f => new Date(f.time_utc).getTime() > newestTime);

      if (trulyNewFills.length > 0) {
        // Add new fills to the front
        fillsCache = [...trulyNewFills, ...fillsCache];
        // Cap the cache to prevent unbounded growth
        if (fillsCache.length > 500) {
          fillsCache = fillsCache.slice(0, 500);
        }
      }
      // Don't update hasMoreFills here - keep user's "Load More" progress
    }

    // Set fillsOldestTime from the oldest fill in cache
    // This is needed for "Load More" pagination to work correctly
    if (fillsCache.length > 0) {
      fillsOldestTime = fillsCache[fillsCache.length - 1].time_utc;
    }

    renderFills(fillsCache);
  } catch (err) {
    console.error(err);
    // Still update UI even on error
    updateFillsUI();
  }
}


function pushFill(fill) {
  // Keep raw fills cache for potential re-aggregation
  fillsCache.unshift(fill);
  fillsCache = fillsCache.slice(0, 200); // Keep more raw fills for history

  // Use streaming aggregation - dynamically merge into existing groups
  addFillToAggregation(fill);

  // Re-render the aggregated view
  renderAggregatedFills();
}

function connectWs() {
  const wsUrl = (location.origin.startsWith('https') ? 'wss://' : 'ws://') + location.host + '/ws';
  const ws = new WebSocket(wsUrl);
  ws.addEventListener('message', (evt) => {
    try {
      const payload = JSON.parse(evt.data);

      // Handle price updates
      if (payload.type === 'price') {
        updatePriceTicker(payload.btc, payload.eth);
        return;
      }

      // Handle hello message with initial prices
      if (payload.type === 'hello' && payload.prices) {
        updatePriceTicker(payload.prices.btc, payload.prices.eth);
      }

      // Handle trade events
      const events = payload.events || payload.batch || payload;
      if (Array.isArray(events)) {
        events
          .filter((e) => e.type === 'trade')
          .forEach((e) => {
            const symbol = (e.symbol || 'BTC').toUpperCase();
            // Only process BTC and ETH fills
            if (symbol !== 'BTC' && symbol !== 'ETH') return;

            const sizeSigned = e.size ?? e.payload?.size ?? 0;
            const startPos = e.startPosition ?? e.payload?.startPosition ?? null;
            const sizeNum = Number(sizeSigned);
            const prevPos = startPos != null ? Number(startPos) : null;
            // Calculate resulting_position = previous_position + size_signed
            const resultingPos = prevPos != null ? prevPos + sizeNum : null;
            const row = {
              time_utc: e.at,
              address: e.address,
              action: e.action || e.payload?.action || '',
              size_signed: sizeNum,
              previous_position: prevPos,
              resulting_position: resultingPos,
              price_usd: e.priceUsd ?? e.payload?.priceUsd ?? null,
              closed_pnl_usd: e.realizedPnlUsd ?? e.payload?.realizedPnlUsd ?? null,
              symbol,
              hash: e.hash || e.payload?.hash,
              at: e.at,
              side: sizeNum >= 0 ? 'buy' : 'sell'
            };
            // Add to legacy fills cache ONLY if address is in legacy leaderboard
            const addrLower = (e.address || '').toLowerCase();
            if (legacyAddresses.has(addrLower)) {
              pushFill(row);
            }
            // Also add to Alpha Pool fills cache if address is in pool
            addAlphaFillFromWs(row);
          });
      }
    } catch (err) {
      console.error('ws parse', err);
    }
  });
  ws.addEventListener('open', () => {
    // Refresh fills when WebSocket reconnects to ensure data consistency
    // This prevents stale data showing after backend restart
    // Skip on first connect since refreshFills() was already called in init()
    if (isFirstWsConnect) {
      isFirstWsConnect = false;
    } else {
      refreshFills();
    }
  });
  ws.addEventListener('close', () => {
    // Clear stale fills data when connection is lost
    // This prevents showing outdated data when backend restarts
    fillsCache = [];
    aggregatedGroups = [];
    hasMoreFills = true; // Reset so "Load More" button is re-enabled after reconnect
    fillsOldestTime = null;
    renderAggregatedFills();
    setTimeout(connectWs, 2000);
  });
}

function renderChart(symbol) {
  const draw = () => {
    // Determine current theme
    const effectiveTheme = currentTheme === 'auto' ? getSystemTheme() : currentTheme;
    const tvTheme = effectiveTheme === 'light' ? 'light' : 'dark';

    document.getElementById('tradingview_chart').innerHTML = '';
    // eslint-disable-next-line no-undef
    new TradingView.widget({
      autosize: true,
      symbol: `BINANCE:${symbol}`,
      interval: '60',
      timezone: 'Etc/UTC',
      theme: tvTheme,
      style: '1',
      locale: 'en',
      container_id: 'tradingview_chart',
      withdateranges: true,
      hide_side_toolbar: false
    });
  };
  if (window.TradingView && window.TradingView.widget) {
    draw();
  } else {
    setTimeout(() => renderChart(symbol), 400);
  }
}

function initChartControls() {
  symbolButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      symbolButtons.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      currentSymbol = btn.dataset.symbol || 'BTCUSDT';
      renderChart(currentSymbol);
    });
  });

  // Chart collapse functionality
  const collapseBtn = document.getElementById('chart-collapse-btn');
  const chartCard = document.querySelector('.chart-card');
  if (collapseBtn && chartCard) {
    collapseBtn.addEventListener('click', () => {
      chartCard.classList.toggle('collapsed');
    });
  }
}

// Period controls removed - now fixed to 30 days

// Show error message in custom accounts section
function showCustomError(message) {
  customErrorEl.textContent = message;
  customErrorEl.classList.add('show');
  setTimeout(() => {
    customErrorEl.classList.remove('show');
  }, 5000);
}

// Clear error message
function clearCustomError() {
  customErrorEl.classList.remove('show');
  customErrorEl.textContent = '';
}

// Validate Ethereum address format
function isValidEthAddress(address) {
  return /^0x[a-fA-F0-9]{40}$/.test(address);
}

// Add a custom pinned account
async function addCustomPinnedAccount() {
  clearCustomError();
  const address = customAddressInput.value.trim();

  if (!address) {
    showCustomError('Please enter an Ethereum address');
    return;
  }

  if (!isValidEthAddress(address)) {
    showCustomError('Invalid Ethereum address format (must be 0x + 40 hex characters)');
    return;
  }

  if (customPinnedCount >= MAX_CUSTOM_PINNED) {
    showCustomError(`Maximum of ${MAX_CUSTOM_PINNED} custom accounts allowed`);
    return;
  }

  addCustomBtn.disabled = true;
  addCustomBtn.textContent = '...';

  try {
    const res = await fetch(`${API_BASE}/pinned-accounts/custom`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ address })
    });

    const data = await res.json();

    if (!res.ok) {
      showCustomError(data.error || 'Failed to add custom account');
      return;
    }

    // Clear input on success
    customAddressInput.value = '';

    // Refresh the summary to show the new account
    await refreshSummary();
  } catch (err) {
    console.error('Add custom pinned account error:', err);
    showCustomError('Failed to add custom account');
  } finally {
    addCustomBtn.disabled = customPinnedCount >= MAX_CUSTOM_PINNED;
    addCustomBtn.textContent = '+';
  }
}

// Pin an account from the leaderboard
async function pinLeaderboardAccount(address) {
  if (!address) return;

  try {
    const res = await fetch(`${API_BASE}/pinned-accounts/leaderboard`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ address })
    });

    if (!res.ok) {
      const data = await res.json();
      console.error('Pin account error:', data.error);
      showCustomError(data.error || 'Failed to pin account');
      return;
    }

    // Refresh the summary to update the table
    await refreshSummary();
  } catch (err) {
    console.error('Pin account error:', err);
    showCustomError('Failed to pin account');
  }
}

// Unpin an account
async function unpinAccount(address) {
  if (!address) return;

  try {
    const res = await fetch(`${API_BASE}/pinned-accounts/${encodeURIComponent(address)}`, {
      method: 'DELETE'
    });

    if (!res.ok) {
      const data = await res.json();
      console.error('Unpin account error:', data.error);
      return;
    }

    // Refresh the summary to update the table
    await refreshSummary();
  } catch (err) {
    console.error('Unpin account error:', err);
  }
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
  if (!text) return '';
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// Show inline nickname editor
function showNicknameEditor(el, address, currentNickname) {
  // Create inline editor
  const container = el.parentElement;
  const input = document.createElement('input');
  input.type = 'text';
  input.value = currentNickname;
  input.className = 'nickname-input';
  input.placeholder = 'Enter nickname';
  input.maxLength = 32;

  const saveBtn = document.createElement('button');
  saveBtn.className = 'nickname-save-btn';
  saveBtn.textContent = 'Save';

  const cancelBtn = document.createElement('button');
  cancelBtn.className = 'nickname-cancel-btn';
  cancelBtn.textContent = 'Cancel';

  const editorWrapper = document.createElement('div');
  editorWrapper.className = 'nickname-editor';
  editorWrapper.appendChild(input);
  editorWrapper.appendChild(saveBtn);
  editorWrapper.appendChild(cancelBtn);

  // Hide original display, show editor
  el.style.display = 'none';
  container.appendChild(editorWrapper);
  input.focus();
  input.select();

  // Save handler
  const save = async () => {
    const newNickname = input.value.trim();
    saveBtn.disabled = true;
    saveBtn.textContent = '...';

    const success = await updateNickname(address, newNickname);
    if (success) {
      editorWrapper.remove();
      await refreshSummary();
    } else {
      saveBtn.disabled = false;
      saveBtn.textContent = 'Save';
      input.focus();
    }
  };

  // Cancel handler
  const cancel = () => {
    editorWrapper.remove();
    el.style.display = '';
  };

  saveBtn.addEventListener('click', save);
  cancelBtn.addEventListener('click', cancel);

  // Enter to save, Escape to cancel
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      save();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      cancel();
    }
  });
}

// Update nickname via API
async function updateNickname(address, nickname) {
  try {
    const res = await fetch(`${API_BASE}/custom-accounts/${encodeURIComponent(address)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ nickname: nickname || null })
    });

    if (!res.ok) {
      const data = await res.json();
      console.error('Update nickname error:', data.error);
      showCustomError(data.error || 'Failed to update nickname');
      return false;
    }

    return true;
  } catch (err) {
    console.error('Update nickname error:', err);
    showCustomError('Failed to update nickname');
    return false;
  }
}

// Initialize pinned accounts controls
function initPinnedAccountsControls() {
  addCustomBtn.addEventListener('click', addCustomPinnedAccount);

  // Allow Enter key to submit
  customAddressInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') addCustomPinnedAccount();
  });
}

// Infinite scroll for fills
async function loadMoreFills() {
  if (isLoadingMore || !hasMoreFills) return;

  isLoadingMore = true;
  const loadMoreEl = document.getElementById('fills-load-more');
  const loadBtn = document.getElementById('load-history-btn');

  if (loadMoreEl) loadMoreEl.style.display = 'flex';
  if (loadBtn) {
    loadBtn.classList.add('loading');
    loadBtn.textContent = 'Loading';
  }

  try {
    const beforeTime = fillsOldestTime || new Date().toISOString();
    // Use Legacy-specific endpoint for Legacy tab fills
    const url = `${API_BASE}/legacy/fills/backfill?before=${encodeURIComponent(beforeTime)}&limit=30`;
    const data = await fetchJson(url);

    if (data.fills && data.fills.length > 0) {
      // Check for duplicates by comparing fill IDs
      const existingIds = new Set(fillsCache.map(f => f.id));
      const newFills = data.fills.filter(f => !existingIds.has(f.id));

      if (newFills.length === 0) {
        // All fills are duplicates - we've reached the end
        hasMoreFills = false;
      } else {
        // Append only new fills to cache
        fillsCache = [...fillsCache, ...newFills];
        hasMoreFills = data.hasMore;

        // Update oldest time from the actual oldest new fill
        const oldestNewFill = newFills[newFills.length - 1];
        if (oldestNewFill && oldestNewFill.time_utc) {
          fillsOldestTime = oldestNewFill.time_utc;
        }

        // Re-render with all fills (aggregation will be applied)
        renderFills(fillsCache);
      }
    } else {
      hasMoreFills = false;
    }

    // Update UI elements
    updateFillsUI();
  } catch (err) {
    console.error('Load more fills error:', err);
    hasMoreFills = false;
    updateFillsUI();
  } finally {
    isLoadingMore = false;
    if (loadMoreEl) loadMoreEl.style.display = 'none';
    if (loadBtn) {
      loadBtn.classList.remove('loading');
      // Reset button text based on state
      if (!hasMoreFills) {
        loadBtn.textContent = 'All loaded';
        loadBtn.disabled = true;
      } else {
        loadBtn.textContent = 'Load More';
      }
    }
  }
}

// Initialize infinite scroll
function initInfiniteScroll() {
  const container = document.getElementById('fills-scroll-container');
  if (!container) return;

  container.addEventListener('scroll', () => {
    // Check if we're near the bottom (within 50px)
    const { scrollTop, scrollHeight, clientHeight } = container;
    if (scrollHeight - scrollTop - clientHeight < 50) {
      loadMoreFills();
    }
  });
}

// Fetch historical fills from Hyperliquid API (when DB is empty)
async function fetchHistoryFromAPI() {
  const loadBtn = document.getElementById('load-history-btn');
  if (loadBtn) {
    loadBtn.classList.add('loading');
    loadBtn.textContent = 'Fetching...';
    loadBtn.disabled = true;
  }

  try {
    // Use Legacy-specific endpoint for Legacy tab historical fills
    const response = await fetch(`${API_BASE}/legacy/fills/fetch-history`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ limit: 50 })
    });
    const data = await response.json();

    if (data.inserted > 0) {
      // Refresh fills from database after fetching
      hasMoreFills = true; // Reset so we can load from DB
      await refreshFills();
    }
    return data.inserted || 0;
  } catch (err) {
    console.error('Fetch history error:', err);
    return 0;
  } finally {
    if (loadBtn) {
      loadBtn.classList.remove('loading');
      loadBtn.disabled = false;
      updateFillsStatus();
    }
  }
}

// Initialize load history button
function initLoadHistoryButton() {
  const btn = document.getElementById('load-history-btn');
  if (!btn) return;

  btn.addEventListener('click', async () => {
    // If we have no fills, fetch from Hyperliquid API first
    if (fillsCache.length === 0) {
      const inserted = await fetchHistoryFromAPI();
      if (inserted > 0) {
        return; // refreshFills already called
      }
    }
    // Otherwise load more from database
    await loadMoreFills();
  });
}

// Initialize time header toggle
function initTimeHeaderToggle() {
  const header = document.getElementById('fills-time-header');
  if (!header) return;

  header.addEventListener('click', toggleTimeDisplayMode);
}

async function init() {
  initChartControls();
  initPinnedAccountsControls();
  initInfiniteScroll();
  initLoadHistoryButton();
  initTimeHeaderToggle();
  initBanditControls();
  renderChart('BTCUSDT');

  // Initialize fills UI with initial state
  updateFillsUI();

  // Fetch initial prices
  fetchPrices();
  // Check positions status FIRST before loading data
  await checkPositionsStatus();
  refreshSummary();
  // Load Alpha Pool data first so alphaPoolAddresses is populated before fills
  await refreshAlphaPool();
  // Await initial fills load to prevent double-render flash
  await refreshFills();
  // Load Alpha Pool fills from dedicated endpoint (independent from legacy fills)
  await refreshAlphaFills();
  // Fetch real consensus signals from hl-decide
  await fetchConsensusSignals();
  refreshBanditStatus();
  connectWs();
  // Continue polling until positions are ready (if not already)
  if (!positionsReady) {
    pollPositionsUntilReady();
  }
  setInterval(refreshSummary, 30_000);
  setInterval(refreshFills, 20_000);
  setInterval(refreshAlphaFills, 30_000); // Refresh Alpha Pool fills every 30 seconds
  setInterval(fetchConsensusSignals, 60_000); // Refresh consensus signals every minute
  setInterval(refreshBanditStatus, 60_000); // Refresh bandit every minute
}

document.addEventListener('DOMContentLoaded', init);

// Click handler for activity-time toggle (event delegation)
document.addEventListener('click', (e) => {
  if (e.target.classList.contains('activity-time') && e.target.classList.contains('clickable')) {
    toggleAlphaPoolTimeMode();
  }
});

// =====================
// Tab Navigation
// =====================
const tabButtons = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

/**
 * Switch active tab
 * @param {string} tabId - The tab to activate
 */
function switchTab(tabId) {
  // Update buttons
  tabButtons.forEach(btn => {
    btn.classList.toggle('active', btn.getAttribute('data-tab') === tabId);
  });

  // Update content
  tabContents.forEach(content => {
    const isActive = content.id === `tab-${tabId}`;
    content.classList.toggle('active', isActive);
  });

  // Load data for the tab if needed
  if (tabId === 'alpha-pool') {
    refreshAlphaPool();
  } else if (tabId === 'legacy-leaderboard') {
    // Clear stale data and refresh Legacy fills when switching to this tab
    fillsCache = [];
    aggregatedGroups = [];
    refreshFills();
  }

  // Store preference
  localStorage.setItem('activeTab', tabId);
}

// Tab click handlers
tabButtons.forEach(btn => {
  btn.addEventListener('click', () => {
    switchTab(btn.getAttribute('data-tab'));
  });
});

// NOTE: Tab restoration moved to end of file after Alpha Pool is defined

// =====================
// Alpha Pool
// =====================
const alphaPoolTable = document.getElementById('alpha-pool-table');
const alphaPoolStats = document.getElementById('alpha-pool-stats');
const alphaPoolConfig = document.getElementById('alpha-pool-config');
const alphaFillsTable = document.getElementById('alpha-fills-table');
const alphaFillsCount = document.getElementById('alpha-fills-count');
const alphaPoolBadge = document.getElementById('alpha-pool-badge');
const alphaPoolRefreshStatus = document.getElementById('alpha-pool-refresh-status');

let alphaPoolData = [];
let alphaPoolAddresses = new Set();
let alphaPoolHoldings = {}; // { address: [{ symbol, size, entryPrice, liquidationPrice, leverage }] }
let alphaPoolLastActivity = {}; // { address: ISO timestamp }
let alphaPoolTimeMode = 'relative'; // 'relative' or 'absolute' - toggles on click
let alphaFillsCache = []; // Separate cache for Alpha Pool fills
let alphaFillsLoading = false;
let refreshStatusPolling = null; // { id: number, interval: number } or null
let lastRefreshCompleted = null;
let isRefreshRunning = false; // Track if a refresh is in progress (for empty state message)
let currentRefreshStatus = null; // Store full status for loading UI
let alphaFillsTimeDisplayMode = 'absolute'; // 'absolute' or 'relative' for Alpha Pool activity time
let subscriptionMethods = {}; // { address: { method: 'websocket'|'polling'|'none', sources: string[] } }
let subscriptionStatus = { maxWebSocketSlots: 10, addressesByMethod: { websocket: 0, polling: 0, none: 0 } }; // Subscription status for slot tracking
let activePopover = null; // Currently open subscription popover

/**
 * Fetch subscription status and methods (shared by all tabs)
 * This must be called before rendering any tab that shows subscription indicators
 */
async function refreshSubscriptionStatus() {
  try {
    const [subscriptionsRes, statusRes] = await Promise.all([
      fetch(`${API_BASE}/subscriptions/methods`),
      fetch(`${API_BASE}/subscriptions/status`)
    ]);

    // Process subscription methods (websocket vs polling)
    if (subscriptionsRes.ok) {
      subscriptionMethods = await subscriptionsRes.json();
    }

    // Process subscription status for slot indicator
    if (statusRes.ok) {
      subscriptionStatus = await statusRes.json();
      updateWebSocketSlotsIndicator();
    }
  } catch (err) {
    console.error('Failed to fetch subscription status:', err);
  }
}

/**
 * Fetch and render Alpha Pool data
 */
async function refreshAlphaPool() {
  // Show loading state if table is empty
  // But if refresh is running, let renderAlphaPoolTable() show the proper refresh progress
  if (alphaPoolTable && alphaPoolData.length === 0 && !isRefreshRunning) {
    alphaPoolTable.innerHTML = `
      <tr>
        <td colspan="8">
          <div class="alpha-pool-loading">
            <span class="loading-spinner"></span>
            <span class="loading-text">Loading Alpha Pool traders...</span>
            <span class="loading-hint">Fetching data from server</span>
          </div>
        </td>
      </tr>
    `;
  }

  try {
    // Fetch Alpha Pool data, holdings, last activity in parallel
    // Also refresh subscription status (shared with Legacy tab)
    const [poolRes, holdingsRes, lastActivityRes] = await Promise.all([
      fetch(`${API_BASE}/alpha-pool`),
      fetch(`${API_BASE}/alpha-pool/holdings`),
      fetch(`${API_BASE}/alpha-pool/last-activity`),
      refreshSubscriptionStatus() // Shared function for subscription data
    ]);

    if (!poolRes.ok) throw new Error('Failed to fetch alpha pool');
    const data = await poolRes.json();

    alphaPoolData = data.traders || [];
    alphaPoolAddresses = new Set(alphaPoolData.map(t => t.address.toLowerCase()));

    // Process holdings data
    if (holdingsRes.ok) {
      const holdingsData = await holdingsRes.json();
      alphaPoolHoldings = holdingsData.holdings || {};
    }

    // Process last activity per address (uses dedicated endpoint to avoid HFT domination)
    if (lastActivityRes.ok) {
      const lastActivityData = await lastActivityRes.json();
      alphaPoolLastActivity = lastActivityData.lastActivity || {};
    }

    // Update badge with selected count
    const selectedCount = alphaPoolData.filter(t => t.is_selected).length;
    if (alphaPoolBadge) {
      alphaPoolBadge.textContent = selectedCount > 0 ? selectedCount : '';
    }

    // Update stats
    if (alphaPoolStats) {
      if (alphaPoolData.length > 0) {
        alphaPoolStats.textContent = `${data.count} traders | ${selectedCount} selected`;
      } else {
        alphaPoolStats.textContent = 'No data';
      }
    }

    // Update config display with refresh timing
    if (alphaPoolConfig) {
      let refreshInfo = '';
      if (data.last_refreshed) {
        const lastRefreshed = new Date(data.last_refreshed);
        const nextRefresh = data.next_refresh ? new Date(data.next_refresh) : null;
        const now = new Date();

        // Format relative time for last refresh (uses existing fmtRelativeTime)
        const lastAgo = fmtRelativeTime(data.last_refreshed);

        // Format next refresh
        let nextInfo = '';
        if (nextRefresh) {
          if (nextRefresh <= now) {
            nextInfo = '<span class="refresh-overdue">overdue</span>';
          } else {
            // Calculate time until next refresh
            const diffMs = nextRefresh.getTime() - now.getTime();
            const hours = Math.floor(diffMs / (1000 * 60 * 60));
            const minutes = Math.floor((diffMs % (1000 * 60 * 60)) / (1000 * 60));
            if (hours > 0) {
              nextInfo = `in ${hours}h ${minutes}m`;
            } else {
              nextInfo = `in ${minutes}m`;
            }
          }
        }

        refreshInfo = `
          <span class="refresh-timing" title="Last refreshed: ${lastRefreshed.toLocaleString()}">
            🕐 ${lastAgo}
          </span>
          ${nextRefresh ? `<span class="refresh-timing next" title="Next refresh: ${nextRefresh.toLocaleString()}">→ Next ${nextInfo}</span>` : ''}
        `;
      }

      alphaPoolConfig.innerHTML = `
        <span>Pool: ${data.pool_size}</span>
        <span>K: ${data.select_k}</span>
        ${refreshInfo}
      `;
    }

    // Render table (handles empty state internally)
    renderAlphaPoolTable();
  } catch (err) {
    console.error('Alpha pool fetch error:', err);
    if (alphaPoolStats) {
      alphaPoolStats.textContent = 'Error';
    }
    // Show error state in table
    if (alphaPoolTable) {
      alphaPoolTable.innerHTML = `
        <tr>
          <td colspan="8">
            <div class="error-state">
              <span class="error-icon">⚠️</span>
              <span class="error-title">Failed to load Alpha Pool</span>
              <span class="error-message">Could not connect to server. Please check your connection and try again.</span>
              <button class="error-action" onclick="refreshAlphaPool()">Retry</button>
            </div>
          </td>
        </tr>
      `;
    }
  }
}

/**
 * Render Alpha Pool table rows
 */
function renderAlphaPoolTable() {
  if (!alphaPoolTable) return;

  // Handle empty state - no traders in pool
  if (alphaPoolData.length === 0) {
    // Show different message if refresh is running
    if (isRefreshRunning) {
      const stepText = currentRefreshStatus ? formatRefreshStep(currentRefreshStatus.current_step) : 'Starting...';
      const progressText = currentRefreshStatus && currentRefreshStatus.progress > 0 ? `${currentRefreshStatus.progress}%` : '';
      alphaPoolTable.innerHTML = `
        <tr>
          <td colspan="8">
            <div class="alpha-pool-loading">
              <span class="loading-spinner"></span>
              <span class="loading-text" id="loading-step-text">${stepText}</span>
              <span class="loading-progress" id="loading-progress-text">${progressText}</span>
              <span class="loading-hint">Fetching and filtering top traders from Hyperliquid. This may take a few minutes.</span>
            </div>
          </td>
        </tr>
      `;
    } else {
      alphaPoolTable.innerHTML = `
        <tr>
          <td colspan="8">
            <div class="empty-state">
              <span class="empty-icon">🎯</span>
              <span class="empty-title">No Alpha Pool Data</span>
              <span class="empty-message">The Alpha Pool hasn't been populated yet. Click the button below to fetch top traders from Hyperliquid.</span>
              <button class="empty-action" onclick="triggerAlphaPoolRefresh()" id="refresh-alpha-pool-btn">Refresh Alpha Pool</button>
            </div>
          </td>
        </tr>
      `;
    }
    return;
  }

  alphaPoolTable.innerHTML = alphaPoolData.map(trader => {
    const rowClass = trader.is_selected ? 'selected' : '';
    const shortAddr = `${trader.address.slice(0, 6)}...${trader.address.slice(-4)}`;
    const displayName = trader.nickname || shortAddr;
    const sparklineCell = generateSparkline(trader.pnl_curve || []);

    // Separate BTC and ETH holdings
    const addrLower = trader.address.toLowerCase();
    const holdings = alphaPoolHoldings[addrLower] || [];
    const btcHolding = getBtcHolding(holdings);
    const ethHolding = getEthHolding(holdings);
    const btcCell = formatSingleHolding(btcHolding);
    const ethCell = formatSingleHolding(ethHolding);

    // Calculate 30d PnL from curve - show negative for losses
    const pnl30d = calculate30dPnL(trader.pnl_curve || []);
    const pnl30dClass = pnl30d >= 0 ? 'mu-positive' : 'mu-negative';
    const pnl30dFormatted = pnl30d !== null
      ? `${pnl30d >= 0 ? '+' : '-'}$${Math.abs(pnl30d).toLocaleString(undefined, { maximumFractionDigits: 0 })}`
      : '—';

    // Signals with NIG stats tooltip
    const statsTooltip = `μ (E[R]): ${trader.nig_m >= 0 ? '+' : ''}${trader.nig_m.toFixed(3)}&#10;κ (conf): ${trader.effective_samples.toFixed(0)}&#10;Avg R: ${trader.avg_r >= 0 ? '+' : ''}${trader.avg_r.toFixed(3)}&#10;Total PnL (R): ${trader.total_pnl_r >= 0 ? '+' : ''}${trader.total_pnl_r.toFixed(2)}`;

    // Last activity
    const lastActivity = alphaPoolLastActivity[addrLower];
    const lastActivityCell = formatAlphaLastActivity(lastActivity);

    // Subscription method indicator (websocket vs polling)
    const subInfo = subscriptionMethods[addrLower];
    const subMethod = subInfo?.method || 'none';
    const subIndicator = formatSubscriptionIndicator(subMethod, subInfo?.sources || [], addrLower);

    return `
      <tr class="${rowClass}">
        <td>
          <span class="address-cell">
            ${subIndicator}
            <a href="https://hypurrscan.io/address/${trader.address}" target="_blank" rel="noopener" class="address-link" title="${trader.address}">
              ${displayName}
            </a>
          </span>
        </td>
        <td class="holds-cell">${btcCell}</td>
        <td class="holds-cell">${ethCell}</td>
        <td class="sparkline-cell">${sparklineCell}</td>
        <td class="${pnl30dClass}">${pnl30dFormatted}</td>
        <td class="signals-cell" title="${statsTooltip}">${trader.total_signals}</td>
        <td>${trader.is_selected ? '<span class="selected-badge" title="Selected for consensus"></span>' : ''}</td>
        <td class="last-activity-cell">${lastActivityCell}</td>
      </tr>
    `;
  }).join('');
}

/**
 * Format subscription method indicator (clickable for promote/demote)
 */
function formatSubscriptionIndicator(method, sources, address) {
  const addrLower = (address || '').toLowerCase();
  if (method === 'websocket') {
    const tooltip = `Real-time WebSocket\nSources: ${sources.join(', ') || 'unknown'}\nClick to manage`;
    return `<span class="sub-indicator sub-websocket clickable" data-testid="sub-method-${addrLower}" data-address="${addrLower}" data-method="websocket" title="${tooltip}" onclick="showSubscriptionPopover(event, '${addrLower}')">⚡</span>`;
  } else if (method === 'polling') {
    const tooltip = `Polling (5-min interval)\nSources: ${sources.join(', ') || 'unknown'}\nClick to promote`;
    return `<span class="sub-indicator sub-polling clickable" data-testid="sub-method-${addrLower}" data-address="${addrLower}" data-method="polling" title="${tooltip}" onclick="showSubscriptionPopover(event, '${addrLower}')">⏱️</span>`;
  }
  return '';
}

/**
 * Update WebSocket slots indicator in header
 */
function updateWebSocketSlotsIndicator() {
  const indicator = document.getElementById('ws-slots-indicator');
  const valueEl = document.getElementById('ws-slots-value');
  if (!indicator || !valueEl) return;

  const used = subscriptionStatus.addressesByMethod?.websocket || 0;
  const max = subscriptionStatus.maxWebSocketSlots || 10;
  const available = max - used;

  valueEl.textContent = `${used}/${max}`;
  indicator.title = `WebSocket slots: ${used} used, ${available} available`;

  // Update styling based on availability
  indicator.classList.remove('slots-available', 'slots-full');
  if (available > 0) {
    indicator.classList.add('slots-available');
  } else {
    indicator.classList.add('slots-full');
  }
}

/**
 * Show subscription popover for promote/demote
 */
function showSubscriptionPopover(event, address) {
  event.stopPropagation();

  // Close any existing popover
  closeSubscriptionPopover();

  const subInfo = subscriptionMethods[address];
  if (!subInfo) return;

  const method = subInfo.method;
  const sources = subInfo.sources || [];
  const isPinned = sources.includes('pinned');
  const used = subscriptionStatus.addressesByMethod?.websocket || 0;
  const max = subscriptionStatus.maxWebSocketSlots || 10;
  const available = max - used;

  // Create popover
  const popover = document.createElement('div');
  popover.className = 'sub-popover';
  popover.id = 'sub-popover';
  popover.setAttribute('data-testid', 'subscription-popover');

  const shortAddr = `${address.slice(0, 6)}...${address.slice(-4)}`;
  const methodIcon = method === 'websocket' ? '⚡' : '⏱️';
  const methodLabel = method === 'websocket' ? 'WebSocket (real-time)' : 'Polling (5-min)';

  let actionHtml = '';
  let warningHtml = '';

  if (method === 'websocket') {
    // WebSocket address
    if (isPinned) {
      // Case 1: Pinned + WebSocket
      // Show message that unpin is via pin icon
      warningHtml = `<div class="sub-popover-warning" data-testid="popover-pinned-message">Pinned addresses always use WebSocket.<br>Click the pin icon to unpin and free this slot.</div>`;
    } else {
      // Case 2: Unpinned + WebSocket (auto-assigned or manually promoted)
      // Can demote to polling to free a slot
      actionHtml = `<button class="sub-popover-action demote" data-testid="demote-btn" onclick="demoteAddress('${address}')">Demote to polling</button>`;
      warningHtml = `<div class="sub-popover-warning">Free this WebSocket slot for another address</div>`;
    }
  } else {
    // Polling address
    if (isPinned) {
      // Case: Pinned but on polling (shouldn't happen normally, but handle it)
      warningHtml = `<div class="sub-popover-warning">Pinned address using polling (all slots full)</div>`;
    } else if (available > 0) {
      // Case 4: Unpinned + Polling + slots available
      // Can promote to WebSocket
      actionHtml = `<button class="sub-popover-action promote" data-testid="promote-btn" onclick="promoteAddress('${address}')">Promote to WebSocket</button>`;
      warningHtml = `<div class="sub-popover-warning" data-testid="popover-slots-available">${available} slot${available !== 1 ? 's' : ''} available</div>`;
    } else {
      // Case 5: Unpinned + Polling + no slots
      // Cannot promote, must demote another first
      warningHtml = `<div class="sub-popover-warning" data-testid="popover-no-slots">All ${max} WebSocket slots in use.<br>Demote another address to free a slot.</div>`;
    }
  }

  popover.innerHTML = `
    <div class="sub-popover-header">${shortAddr}</div>
    <div class="sub-popover-method">${methodIcon} ${methodLabel}</div>
    <div class="sub-popover-sources">Sources: ${sources.join(', ') || 'none'}</div>
    ${actionHtml}
    ${warningHtml}
  `;

  // Position popover near the click
  const rect = event.target.getBoundingClientRect();
  popover.style.position = 'fixed';
  popover.style.top = `${rect.bottom + 5}px`;
  popover.style.left = `${rect.left}px`;

  // Ensure popover stays on screen
  document.body.appendChild(popover);
  const popoverRect = popover.getBoundingClientRect();
  if (popoverRect.right > window.innerWidth - 10) {
    popover.style.left = `${window.innerWidth - popoverRect.width - 10}px`;
  }
  if (popoverRect.bottom > window.innerHeight - 10) {
    popover.style.top = `${rect.top - popoverRect.height - 5}px`;
  }

  activePopover = popover;

  // Close on outside click
  setTimeout(() => {
    document.addEventListener('click', closeSubscriptionPopover, { once: true });
  }, 0);
}

/**
 * Close subscription popover
 */
function closeSubscriptionPopover() {
  if (activePopover) {
    activePopover.remove();
    activePopover = null;
  }
}

/**
 * Get priority number for sources (lower = higher priority)
 */
function getSourcePriority(sources) {
  const priorities = { pinned: 0, legacy: 1, 'alpha-pool': 2 };
  let minPriority = 100;
  for (const source of (sources || [])) {
    const p = priorities[source] ?? 100;
    if (p < minPriority) minPriority = p;
  }
  return minPriority;
}

/**
 * Promote address to WebSocket (manual promotion)
 */
async function promoteAddress(address) {
  closeSubscriptionPopover();

  try {
    // Use the new promote API endpoint
    const response = await fetch(`${API_BASE}/subscriptions/promote`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ address })
    });

    if (response.ok) {
      // Refresh data to see updated subscriptions
      await refreshSubscriptionStatus();
      await refreshSummary();
      await refreshAlphaPool();
    } else {
      const err = await response.json();
      console.error('Failed to promote address:', err);
      alert(`Failed to promote: ${err.error || 'Unknown error'}`);
    }
  } catch (err) {
    console.error('Error promoting address:', err);
    alert('Failed to promote address');
  }
}

/**
 * Demote address from WebSocket to polling (manual demotion)
 */
async function demoteAddress(address) {
  closeSubscriptionPopover();

  try {
    // Use the new demote API endpoint
    const response = await fetch(`${API_BASE}/subscriptions/demote`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ address })
    });

    if (response.ok) {
      // Refresh data to see updated subscriptions
      await refreshSubscriptionStatus();
      await refreshSummary();
      await refreshAlphaPool();
    } else {
      const err = await response.json();
      console.error('Failed to demote address:', err);
      alert(`Failed to demote: ${err.error || 'Unknown error'}`);
    }
  } catch (err) {
    console.error('Error demoting address:', err);
    alert('Failed to demote address');
  }
}

/**
 * Format a single holding position with tooltip
 */
function formatSingleHolding(pos) {
  if (!pos || !Number.isFinite(pos.size) || Math.abs(pos.size) < 0.0001) {
    return '<span class="no-position">—</span>';
  }
  const size = Number(pos.size);
  const symbol = (pos.symbol || '').toUpperCase();
  const direction = size >= 0 ? 'holding-long' : 'holding-short';
  const magnitude = Math.abs(size);
  const precision = magnitude >= 1 ? 2 : 4;
  const signed = `${size >= 0 ? '+' : ''}${size.toFixed(precision)}`;

  // Build tooltip with entry price, size, and leverage
  const tooltipParts = [];
  tooltipParts.push(`Size: ${signed} ${symbol}`);
  if (pos.entryPrice != null && Number.isFinite(pos.entryPrice)) {
    tooltipParts.push(`Entry: $${pos.entryPrice.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`);
  }
  if (pos.leverage != null && Number.isFinite(pos.leverage)) {
    tooltipParts.push(`Leverage: ${pos.leverage}x`);
  }
  const tooltip = tooltipParts.join('&#10;');

  return `<span class="holding-chip ${direction}" title="${tooltip}">${signed}</span>`;
}

/**
 * Get BTC position from holdings array
 */
function getBtcHolding(positions) {
  if (!positions || positions.length === 0) return null;
  return positions.find(p => (p.symbol || '').toUpperCase() === 'BTC');
}

/**
 * Get ETH position from holdings array
 */
function getEthHolding(positions) {
  if (!positions || positions.length === 0) return null;
  return positions.find(p => (p.symbol || '').toUpperCase() === 'ETH');
}

/**
 * Calculate 30-day PnL from curve data
 */
function calculate30dPnL(pnlCurve) {
  if (!pnlCurve || pnlCurve.length < 2) return null;
  // Curve is sorted by timestamp, last value minus first value
  const firstValue = parseFloat(pnlCurve[0]?.value ?? 0);
  const lastValue = parseFloat(pnlCurve[pnlCurve.length - 1]?.value ?? 0);
  return lastValue - firstValue;
}

/**
 * Format last activity time for Alpha Pool table.
 * Clickable to toggle between relative and absolute time display.
 */
function formatAlphaLastActivity(timestamp) {
  if (!timestamp) return '<span class="no-activity">—</span>';
  const date = new Date(timestamp);
  if (isNaN(date.getTime())) return '<span class="no-activity">—</span>';

  // Format absolute date as "Dec 8, 14:26"
  const absoluteStr = date.toLocaleDateString([], { month: 'short', day: 'numeric' }) + ', ' +
    date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  // Format relative time (e.g., "2h ago")
  const relativeStr = fmtRelativeTime(timestamp);

  // Display based on current mode
  const displayStr = alphaPoolTimeMode === 'relative' ? relativeStr : absoluteStr;
  const tooltipStr = alphaPoolTimeMode === 'relative' ? absoluteStr : relativeStr;

  return `<span class="activity-time clickable" data-ts="${timestamp}" data-absolute="${absoluteStr}" data-relative="${relativeStr}" title="Click to toggle (${tooltipStr})">${displayStr}</span>`;
}

/**
 * Toggle time display mode between relative and absolute
 */
function toggleAlphaPoolTimeMode() {
  alphaPoolTimeMode = alphaPoolTimeMode === 'relative' ? 'absolute' : 'relative';
  // Update all activity-time elements without full re-render
  document.querySelectorAll('#alpha-pool-table .activity-time.clickable').forEach(el => {
    const absolute = el.getAttribute('data-absolute');
    const relative = el.getAttribute('data-relative');
    if (absolute && relative) {
      const displayStr = alphaPoolTimeMode === 'relative' ? relative : absolute;
      const tooltipStr = alphaPoolTimeMode === 'relative' ? absolute : relative;
      el.textContent = displayStr;
      el.title = `Click to toggle (${tooltipStr})`;
    }
  });
}

/**
 * Trigger Alpha Pool refresh from Hyperliquid API
 */
async function triggerAlphaPoolRefresh() {
  const btn = document.getElementById('refresh-alpha-pool-btn');
  if (btn) {
    btn.disabled = true;
    btn.textContent = 'Refreshing...';
  }

  try {
    const res = await fetch(`${API_BASE}/alpha-pool/refresh?limit=50&background=true`, { method: 'POST' });

    // Handle 409 Conflict - refresh already in progress
    if (res.status === 409) {
      // Set running state and show loading UI
      isRefreshRunning = true;
      renderAlphaPoolTable();
      startRefreshStatusPolling();
      return;
    }

    if (!res.ok) throw new Error('Refresh failed');

    // Start polling for refresh status
    isRefreshRunning = true;
    renderAlphaPoolTable();
    startRefreshStatusPolling();
  } catch (err) {
    console.error('Alpha pool refresh error:', err);
    if (btn) {
      btn.disabled = false;
      btn.textContent = 'Refresh Alpha Pool';
    }
    alert('Failed to start refresh. Please try again.');
  }
}

/**
 * Fetch Alpha Pool fills from the dedicated endpoint
 */
async function refreshAlphaFills() {
  if (alphaFillsLoading) return;
  alphaFillsLoading = true;

  // Show loading state if cache is empty
  if (alphaFillsTable && alphaFillsCache.length === 0) {
    alphaFillsTable.innerHTML = `
      <tr>
        <td colspan="6">
          <div class="alpha-pool-loading">
            <span class="loading-spinner"></span>
            <span class="loading-text">Loading Alpha Pool activity...</span>
            <span class="loading-hint">Fetching recent trades</span>
          </div>
        </td>
      </tr>
    `;
  }

  try {
    const res = await fetch(`${API_BASE}/alpha-pool/fills?limit=50`);
    if (!res.ok) throw new Error('Failed to fetch alpha pool fills');
    const data = await res.json();

    // Merge with existing cache, avoiding duplicates
    const existingIds = new Set(alphaFillsCache.map(f => f.hash || f.fill_id || `${f.address}-${f.time_utc}`));
    const newFills = (data.fills || []).filter(f => {
      const id = f.hash || f.fill_id || `${f.address}-${f.time_utc}`;
      return !existingIds.has(id);
    });

    if (newFills.length > 0) {
      alphaFillsCache = [...alphaFillsCache, ...newFills];
      // Sort by time descending
      alphaFillsCache.sort((a, b) => {
        const timeA = new Date(a.time_utc || a.at || 0).getTime();
        const timeB = new Date(b.time_utc || b.at || 0).getTime();
        return timeB - timeA;
      });
      // Keep only most recent 200
      alphaFillsCache = alphaFillsCache.slice(0, 200);
    }

    renderAlphaFills();
  } catch (err) {
    console.error('Failed to fetch alpha pool fills:', err);
    if (alphaFillsCache.length === 0 && alphaFillsTable) {
      alphaFillsTable.innerHTML = `
        <tr>
          <td colspan="6">
            <div class="error-state">
              <span class="error-icon">⚠️</span>
              <span class="error-title">Failed to load activity</span>
              <span class="error-message">Could not fetch trades. Please try again.</span>
              <button class="error-action" onclick="refreshAlphaFills()">Retry</button>
            </div>
          </td>
        </tr>
      `;
    }
  } finally {
    alphaFillsLoading = false;
  }
}

/**
 * Add a WebSocket fill to Alpha Pool cache if it's from a pool address
 */
function addAlphaFillFromWs(fill) {
  if (!alphaPoolAddresses.has(fill.address?.toLowerCase())) return;

  // Check for duplicates
  const id = fill.hash || fill.fill_id || `${fill.address}-${fill.at}`;
  const exists = alphaFillsCache.some(f => {
    const existingId = f.hash || f.fill_id || `${f.address}-${f.time_utc || f.at}`;
    return existingId === id;
  });

  if (!exists) {
    alphaFillsCache.unshift(fill);
    // Keep only most recent 200
    if (alphaFillsCache.length > 200) {
      alphaFillsCache = alphaFillsCache.slice(0, 200);
    }
    renderAlphaFills();
  }
}

/**
 * Render Alpha Pool activity (fills from pool traders only)
 */
function renderAlphaFills() {
  if (!alphaFillsTable) return;

  // Use the dedicated Alpha Pool fills cache
  const alphaFills = alphaFillsCache;

  if (alphaFillsCount) {
    alphaFillsCount.textContent = `${alphaFills.length} fills`;
  }

  if (alphaFills.length === 0) {
    if (!alphaFillsLoading) {
      alphaFillsTable.innerHTML = `
        <tr><td colspan="6">
          <div class="waiting-state">
            <span class="waiting-icon">📡</span>
            <span class="waiting-text">Waiting for Alpha Pool activity...</span>
            <span class="waiting-hint">Trades from selected traders will appear in real-time</span>
          </div>
        </td></tr>`;
    }
    return;
  }

  alphaFillsTable.innerHTML = alphaFills.slice(0, 50).map(fill => {
    const shortAddr = fill.address ? `${fill.address.slice(0, 6)}...${fill.address.slice(-4)}` : '—';
    const action = fill.action || (fill.side === 'buy' ? 'Buy' : 'Sell');
    const actionClass = action.toLowerCase().includes('long') || action.toLowerCase().includes('buy') ? 'positive' : 'negative';
    const symbol = fill.symbol || fill.asset || 'BTC';
    // Handle both field naming conventions (time_utc from API, at from WS)
    const timeStr = fill.time_utc || fill.at;
    // Format time based on current display mode
    const timeCell = formatAlphaFillTime(timeStr);
    // Handle both field naming conventions (size_signed from API, size from WS)
    const size = Math.abs(fill.size_signed ?? fill.size ?? 0);
    // Handle both field naming conventions (price_usd from API, priceUsd/price from WS)
    const price = fill.price_usd ?? fill.priceUsd ?? fill.price ?? 0;
    // Handle both field naming conventions (closed_pnl_usd from API, realizedPnlUsd from WS)
    const pnlValue = fill.closed_pnl_usd ?? fill.realizedPnlUsd;
    const pnl = pnlValue != null ? (pnlValue >= 0 ? '+' : '') + pnlValue.toFixed(2) : '—';
    const pnlClass = pnlValue > 0 ? 'positive' : pnlValue < 0 ? 'negative' : '';

    return `
      <tr>
        <td class="alpha-time-cell clickable" data-ts="${timeStr || ''}">${timeCell}</td>
        <td>
          <a href="https://hypurrscan.io/address/${fill.address}" target="_blank" rel="noopener" class="address-link">
            ${shortAddr}
          </a>
        </td>
        <td class="${actionClass}">${action}</td>
        <td>${size.toFixed(4)} ${symbol}</td>
        <td>$${Number(price).toLocaleString()}</td>
        <td class="${pnlClass}">${pnl}</td>
      </tr>
    `;
  }).join('');

  // Add click handlers for time cells
  setupAlphaTimeClickHandlers();
}

/**
 * Format Alpha Pool fill time based on display mode
 */
function formatAlphaFillTime(timeStr) {
  if (!timeStr) return '—';
  const date = new Date(timeStr);
  if (isNaN(date.getTime())) return '—';

  if (alphaFillsTimeDisplayMode === 'relative') {
    return fmtRelativeTime(timeStr);
  } else {
    // Absolute: MM/DD HH:MM:SS
    const month = (date.getMonth() + 1).toString().padStart(2, '0');
    const day = date.getDate().toString().padStart(2, '0');
    const time = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    return `${month}/${day} ${time}`;
  }
}

/**
 * Setup click handlers for time cells to toggle display mode
 */
function setupAlphaTimeClickHandlers() {
  const timeCells = document.querySelectorAll('.alpha-time-cell.clickable');
  timeCells.forEach(cell => {
    cell.addEventListener('click', toggleAlphaFillsTimeMode);
  });

  // Also setup the header click
  const header = document.querySelector('[data-testid="alpha-fills-table"] thead th:first-child');
  if (header && !header.dataset.listenerAdded) {
    header.dataset.listenerAdded = 'true';
    header.addEventListener('click', toggleAlphaFillsTimeMode);
  }
}

/**
 * Toggle between absolute and relative time display for Alpha Pool fills
 */
function toggleAlphaFillsTimeMode() {
  alphaFillsTimeDisplayMode = alphaFillsTimeDisplayMode === 'absolute' ? 'relative' : 'absolute';

  // Update header text
  const header = document.querySelector('[data-testid="alpha-fills-table"] thead th:first-child');
  if (header) {
    header.textContent = alphaFillsTimeDisplayMode === 'absolute' ? 'Time ⏱' : 'Time 🕐';
    header.title = `Click to switch to ${alphaFillsTimeDisplayMode === 'absolute' ? 'relative' : 'absolute'} time`;
  }

  // Re-render fills
  renderAlphaFills();
}

/**
 * Update loading progress in the table (without re-rendering entire table)
 */
function updateLoadingProgress(status) {
  const stepEl = document.getElementById('loading-step-text');
  const progressEl = document.getElementById('loading-progress-text');

  if (stepEl) {
    stepEl.textContent = formatRefreshStep(status.current_step);
  }
  if (progressEl) {
    progressEl.textContent = status.progress > 0 ? `${status.progress}%` : '';
  }
}

function formatRefreshStep(step) {
  const stepNames = {
    'idle': 'Idle',
    'starting': 'Starting...',
    'fetching_leaderboard': 'Fetching leaderboard...',
    'filtering_candidates': 'Filtering candidates...',
    'analyzing_traders': 'Analyzing traders...',
    'saving_traders': 'Saving traders...',
    'backfilling_fills': 'Backfilling fills...',
    'reconciling': 'Reconciling...',
    'completed': 'Completed',
    'failed': 'Failed',
  };
  return stepNames[step] || step;
}

/**
 * Update the refresh status indicator in the UI
 */
function updateRefreshStatusUI(status) {
  // ALWAYS track running state globally, even if DOM element is missing
  // This is critical for showing correct empty state (loading vs button)
  const wasRunning = isRefreshRunning;
  isRefreshRunning = status.is_running;
  currentRefreshStatus = status;

  // Re-render table when refresh state changes AND pool is empty
  // This must happen regardless of header status element
  if (alphaPoolData.length === 0 && wasRunning !== isRefreshRunning) {
    renderAlphaPoolTable();
  }

  // Update loading UI with progress (if visible in table)
  if (alphaPoolData.length === 0 && isRefreshRunning) {
    updateLoadingProgress(status);
  }

  // Header status indicator is optional - return if not present
  if (!alphaPoolRefreshStatus) return;

  if (status.is_running) {
    alphaPoolRefreshStatus.style.display = 'inline-flex';
    alphaPoolRefreshStatus.className = 'alpha-pool-refresh-status';

    const textEl = alphaPoolRefreshStatus.querySelector('.refresh-text');
    const progressEl = alphaPoolRefreshStatus.querySelector('.refresh-progress');

    if (textEl) textEl.textContent = formatRefreshStep(status.current_step);
    if (progressEl) {
      if (status.progress > 0) {
        progressEl.textContent = `${status.progress}%`;
      } else {
        progressEl.textContent = '';
      }
    }
  } else if (status.current_step === 'completed') {
    // Show completed status briefly, then hide
    alphaPoolRefreshStatus.style.display = 'inline-flex';
    alphaPoolRefreshStatus.className = 'alpha-pool-refresh-status completed';

    const textEl = alphaPoolRefreshStatus.querySelector('.refresh-text');
    const progressEl = alphaPoolRefreshStatus.querySelector('.refresh-progress');

    if (textEl) textEl.textContent = '✓ Refreshed';
    if (progressEl && status.elapsed_seconds) {
      progressEl.textContent = `(${Math.round(status.elapsed_seconds)}s)`;
    }

    // Hide after 5 seconds and refresh data
    setTimeout(() => {
      alphaPoolRefreshStatus.style.display = 'none';
    }, 5000);

    // Refresh the Alpha Pool data to show updated traders
    if (lastRefreshCompleted !== status.completed_at) {
      lastRefreshCompleted = status.completed_at;
      refreshAlphaPool();
    }
  } else if (status.current_step === 'failed') {
    alphaPoolRefreshStatus.style.display = 'inline-flex';
    alphaPoolRefreshStatus.className = 'alpha-pool-refresh-status failed';

    const textEl = alphaPoolRefreshStatus.querySelector('.refresh-text');
    const progressEl = alphaPoolRefreshStatus.querySelector('.refresh-progress');

    if (textEl) textEl.textContent = '✗ Failed';
    if (progressEl) progressEl.textContent = '';

    // Hide after 10 seconds
    setTimeout(() => {
      alphaPoolRefreshStatus.style.display = 'none';
    }, 10000);
  } else {
    // Idle state - hide
    alphaPoolRefreshStatus.style.display = 'none';
  }
}

/**
 * Poll refresh status from the API
 */
async function pollRefreshStatus() {
  try {
    const res = await fetch(`${API_BASE}/alpha-pool/refresh/status`);
    if (!res.ok) return;

    const status = await res.json();
    updateRefreshStatusUI(status);

    // If running, poll more frequently; otherwise slow down
    if (status.is_running) {
      // Poll every 2 seconds while refresh is running
      if (!refreshStatusPolling || refreshStatusPolling.interval !== 2000) {
        stopRefreshStatusPolling();
        const id = setInterval(pollRefreshStatus, 2000);
        refreshStatusPolling = { id, interval: 2000 };
      }
    } else {
      // Poll every 30 seconds when idle (to catch external refreshes)
      if (!refreshStatusPolling || refreshStatusPolling.interval !== 30000) {
        stopRefreshStatusPolling();
        const id = setInterval(pollRefreshStatus, 30000);
        refreshStatusPolling = { id, interval: 30000 };
      }
    }
  } catch (err) {
    console.error('Failed to poll refresh status:', err);
  }
}

/**
 * Start polling for refresh status
 */
function startRefreshStatusPolling() {
  // Stop any existing polling first
  stopRefreshStatusPolling();
  // Start polling immediately
  pollRefreshStatus();
  // Then continue polling at normal interval
  const id = setInterval(pollRefreshStatus, 2000);
  refreshStatusPolling = { id, interval: 2000 };
}

/**
 * Stop polling for refresh status
 */
function stopRefreshStatusPolling() {
  if (refreshStatusPolling) {
    clearInterval(refreshStatusPolling.id);
    refreshStatusPolling = null;
  }
}

// Expose functions to window for onclick handlers in dynamically generated HTML
window.triggerAlphaPoolRefresh = triggerAlphaPoolRefresh;
window.refreshAlphaPool = refreshAlphaPool;
window.refreshAlphaFills = refreshAlphaFills;
window.showSubscriptionPopover = showSubscriptionPopover;
window.closeSubscriptionPopover = closeSubscriptionPopover;
window.promoteAddress = promoteAddress;
window.demoteAddress = demoteAddress;

// Initialize Alpha Pool on page load (if tab is active)
if (document.querySelector('.tab-btn.active[data-tab="alpha-pool"]')) {
  // Check refresh status first so we can show proper loading state
  pollRefreshStatus().then(() => {
    setTimeout(refreshAlphaPool, 100);
  });
  // Start polling for refresh status updates
  setTimeout(startRefreshStatusPolling, 2000);
}

// Refresh Alpha Pool periodically when tab is active
setInterval(() => {
  const isActive = document.querySelector('.tab-btn.active[data-tab="alpha-pool"]');
  if (isActive) {
    refreshAlphaPool();
  }
}, 60000); // Every minute

// Poll refresh status when Alpha Pool tab becomes active
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    if (btn.getAttribute('data-tab') === 'alpha-pool') {
      pollRefreshStatus();
    }
  });
});

// =====================
// Decision Logging & Stats
// =====================

// State
let decisionLogs = [];
let decisionLogsOffset = 0;
let decisionLogsLoading = false;
let decisionFilter = 'all'; // 'all', 'signal', 'skip'
const DECISIONS_LIMIT = 20;

// DOM Elements
const decisionListEl = document.getElementById('decision-list');
const decisionLoadMoreEl = document.getElementById('decision-load-more');
const decisionLoadMoreBtn = document.getElementById('decision-load-more-btn');
const decisionFilterBtns = document.querySelectorAll('.decision-filter .filter-btn');

// Stats elements
const statSignalsValue = document.getElementById('stat-signals-value');
const statWinRateValue = document.getElementById('stat-win-rate-value');
const statAvgRValue = document.getElementById('stat-avg-r-value');
const statTotalRValue = document.getElementById('stat-total-r-value');
const statSkippedValue = document.getElementById('stat-skipped-value');
const statSkipRateValue = document.getElementById('stat-skip-rate-value');

/**
 * Fetch decision stats from hl-decide
 */
async function fetchDecisionStats() {
  try {
    const res = await fetch(`${API_BASE}/decisions/stats?days=7`);
    if (!res.ok) throw new Error('Failed to fetch decision stats');
    const data = await res.json();
    renderDecisionStats(data);
  } catch (err) {
    console.error('Failed to fetch decision stats:', err);
  }
}

/**
 * Render decision stats to the stats bar
 */
function renderDecisionStats(stats) {
  if (statSignalsValue) {
    statSignalsValue.textContent = stats.signals || 0;
  }
  if (statWinRateValue) {
    const winRate = stats.win_rate || 0;
    statWinRateValue.textContent = `${winRate}%`;
    statWinRateValue.classList.toggle('positive', winRate >= 50);
    statWinRateValue.classList.toggle('negative', winRate < 50 && winRate > 0);
  }
  if (statAvgRValue) {
    const avgR = stats.avg_result_r || 0;
    statAvgRValue.textContent = avgR >= 0 ? `+${avgR.toFixed(2)}R` : `${avgR.toFixed(2)}R`;
    statAvgRValue.classList.toggle('positive', avgR > 0);
    statAvgRValue.classList.toggle('negative', avgR < 0);
  }
  if (statTotalRValue) {
    const totalR = stats.total_r || 0;
    statTotalRValue.textContent = totalR >= 0 ? `+${totalR.toFixed(1)}R` : `${totalR.toFixed(1)}R`;
    statTotalRValue.classList.toggle('positive', totalR > 0);
    statTotalRValue.classList.toggle('negative', totalR < 0);
  }
  if (statSkippedValue) {
    statSkippedValue.textContent = stats.skipped || 0;
  }
  if (statSkipRateValue) {
    statSkipRateValue.textContent = `${stats.skip_rate || 0}%`;
  }
}

/**
 * Fetch decision logs from hl-decide
 */
async function fetchDecisionLogs(reset = false) {
  if (decisionLogsLoading) return;
  decisionLogsLoading = true;

  if (reset) {
    decisionLogsOffset = 0;
    decisionLogs = [];
  }

  try {
    const typeParam = decisionFilter !== 'all' ? `&decision_type=${decisionFilter}` : '';
    const res = await fetch(`${API_BASE}/decisions?limit=${DECISIONS_LIMIT}&offset=${decisionLogsOffset}${typeParam}`);
    if (!res.ok) throw new Error('Failed to fetch decision logs');
    const data = await res.json();

    if (reset) {
      decisionLogs = data.items || [];
    } else {
      decisionLogs = [...decisionLogs, ...(data.items || [])];
    }
    decisionLogsOffset += data.items?.length || 0;

    renderDecisionLogs();

    // Show/hide load more button
    if (decisionLoadMoreEl) {
      decisionLoadMoreEl.style.display = (data.items?.length || 0) >= DECISIONS_LIMIT ? 'block' : 'none';
    }
  } catch (err) {
    console.error('Failed to fetch decision logs:', err);
  } finally {
    decisionLogsLoading = false;
  }
}

/**
 * Format a date for display
 */
function formatDecisionTime(isoString) {
  const date = new Date(isoString);
  const now = new Date();
  const diffMs = now - date;
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);

  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;

  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

/**
 * Render decision logs to the UI
 */
function renderDecisionLogs() {
  if (!decisionListEl) return;

  if (decisionLogs.length === 0) {
    decisionListEl.innerHTML = `
      <div class="decision-empty">
        <p>No decision logs yet.</p>
        <p>Decisions will appear when the consensus detector evaluates potential signals.</p>
      </div>
    `;
    return;
  }

  decisionListEl.innerHTML = decisionLogs.map(decision => {
    const typeClass = decision.decision_type || 'skip';
    const direction = decision.direction || 'none';
    const dirClass = direction === 'long' ? 'long' : direction === 'short' ? 'short' : '';

    // Format gate results for expandable section
    const gatesHtml = (decision.gates || []).map(gate => {
      const statusClass = gate.passed ? 'passed' : 'failed';
      const statusIcon = gate.passed ? '✓' : '✗';
      const valueStr = typeof gate.value === 'number' ?
        (gate.name === 'supermajority' ? `${(gate.value * 100).toFixed(0)}%` :
         gate.name === 'effective_k' ? gate.value.toFixed(1) :
         gate.name === 'ev_gate' ? `${gate.value.toFixed(2)}R` :
         gate.name === 'freshness' ? `${gate.value.toFixed(0)}s` :
         gate.name === 'price_band' ? `${gate.value.toFixed(2)}R` :
         gate.value.toFixed(2)) : String(gate.value);
      const threshStr = typeof gate.threshold === 'number' ?
        (gate.name === 'supermajority' ? `${(gate.threshold * 100).toFixed(0)}%` :
         gate.name === 'effective_k' ? gate.threshold.toFixed(1) :
         gate.name === 'ev_gate' ? `${gate.threshold.toFixed(2)}R` :
         gate.name === 'freshness' ? `${gate.threshold.toFixed(0)}s` :
         gate.name === 'price_band' ? `${gate.threshold.toFixed(2)}R` :
         gate.threshold.toFixed(2)) : String(gate.threshold);

      return `
        <div class="gate-item">
          <span class="gate-status ${statusClass}">${statusIcon}</span>
          <span class="gate-name">${gate.name}</span>
          <span class="gate-value">${valueStr} / ${threshStr}</span>
        </div>
      `;
    }).join('');

    return `
      <div class="decision-card type-${typeClass}" data-decision-id="${decision.id}">
        <div class="decision-header">
          <div class="decision-meta">
            <span class="decision-type-badge ${typeClass}">${typeClass.replace('_', ' ')}</span>
            <span class="decision-symbol">${decision.symbol}</span>
            <span class="decision-direction ${dirClass}">${direction.toUpperCase()}</span>
          </div>
          <span class="decision-time">${formatDecisionTime(decision.created_at)}</span>
        </div>
        <div class="decision-reasoning">${decision.reasoning || 'No reasoning available.'}</div>
        <div class="decision-metrics">
          <div class="decision-metric">
            <span class="metric-label">Traders:</span>
            <span class="metric-value">${decision.trader_count || 0}</span>
          </div>
          <div class="decision-metric">
            <span class="metric-label">Agreement:</span>
            <span class="metric-value">${decision.agreement_pct ? (decision.agreement_pct * 100).toFixed(0) + '%' : '—'}</span>
          </div>
          <div class="decision-metric">
            <span class="metric-label">EffK:</span>
            <span class="metric-value">${decision.effective_k ? decision.effective_k.toFixed(1) : '—'}</span>
          </div>
          ${decision.ev_estimate ? `
          <div class="decision-metric">
            <span class="metric-label">EV:</span>
            <span class="metric-value">${decision.ev_estimate.toFixed(2)}R</span>
          </div>
          ` : ''}
          ${decision.outcome_r_multiple !== null && decision.outcome_r_multiple !== undefined ? `
          <div class="decision-metric">
            <span class="metric-label">Result:</span>
            <span class="metric-value ${decision.outcome_r_multiple > 0 ? 'positive' : 'negative'}">${decision.outcome_r_multiple > 0 ? '+' : ''}${decision.outcome_r_multiple.toFixed(2)}R</span>
          </div>
          ` : ''}
        </div>
        <div class="decision-gates">
          <div class="gate-list">
            ${gatesHtml}
          </div>
        </div>
      </div>
    `;
  }).join('');

  // Add click handlers for expanding cards
  decisionListEl.querySelectorAll('.decision-card').forEach(card => {
    card.addEventListener('click', () => {
      card.classList.toggle('expanded');
    });
  });
}

// Event listeners for decision log
if (decisionFilterBtns) {
  decisionFilterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      decisionFilterBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      decisionFilter = btn.getAttribute('data-filter');
      fetchDecisionLogs(true);
    });
  });
}

if (decisionLoadMoreBtn) {
  decisionLoadMoreBtn.addEventListener('click', () => {
    fetchDecisionLogs(false);
  });
}

// Initial load
fetchDecisionStats();
fetchDecisionLogs(true);

// Refresh periodically
setInterval(fetchDecisionStats, 60000);
setInterval(() => fetchDecisionLogs(true), 60000);

// =====================
// Portfolio Overview & Auto-Trade
// =====================

// Portfolio state
let portfolioData = null;
let executionConfig = null;
let livePositions = [];

// DOM elements for overview
const equityValueEl = document.getElementById('equity-value');
const unrealizedPnlEl = document.getElementById('unrealized-pnl');
const positionCountEl = document.getElementById('position-count');
const exposurePctEl = document.getElementById('exposure-pct');
const portfolioStatusEl = document.getElementById('portfolio-status');
const positionsTbody = document.getElementById('positions-tbody');
const positionsCountBadge = document.getElementById('positions-count-badge');
const toggleStatusEl = document.getElementById('toggle-status');
const autoTradeToggleEl = document.getElementById('autotrade-toggle');
const maxLeverageEl = document.getElementById('max-leverage');
const maxPositionEl = document.getElementById('max-position');
const maxExposureEl = document.getElementById('max-exposure');

/**
 * Format currency value with appropriate precision
 */
function formatCurrency(value, showSign = false) {
  if (value == null || !Number.isFinite(value)) return '—';
  const sign = showSign && value >= 0 ? '+' : '';
  return sign + '$' + Math.abs(value).toLocaleString('en-US', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  });
}

/**
 * Format percentage value
 */
function formatPercent(value) {
  if (value == null || !Number.isFinite(value)) return '—';
  return (value * 100).toFixed(1) + '%';
}

/**
 * Fetch portfolio summary from API
 */
async function fetchPortfolio() {
  try {
    const res = await fetch(`${API_BASE}/portfolio`);
    if (!res.ok) {
      console.warn('Portfolio fetch failed:', res.status);
      return;
    }
    portfolioData = await res.json();
    renderPortfolioOverview();
  } catch (err) {
    console.error('Portfolio fetch error:', err);
  }
}

/**
 * Fetch execution config (auto-trade settings)
 */
async function fetchExecutionConfig() {
  try {
    const res = await fetch(`${API_BASE}/execution/config`);
    if (!res.ok) {
      console.warn('Execution config fetch failed:', res.status);
      return;
    }
    executionConfig = await res.json();
    renderAutoTradeStatus();
  } catch (err) {
    console.error('Execution config fetch error:', err);
  }
}

/**
 * Toggle auto-trade enabled state
 */
async function toggleAutoTrade() {
  if (!executionConfig) return;

  const newEnabled = !executionConfig.enabled;

  try {
    const res = await fetch(`${API_BASE}/execution/config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled: newEnabled })
    });

    if (res.ok) {
      executionConfig = await res.json();
      renderAutoTradeStatus();
    } else {
      alert('Failed to update auto-trade settings');
    }
  } catch (err) {
    console.error('Toggle auto-trade error:', err);
    alert('Failed to update auto-trade settings');
  }
}

/**
 * Render portfolio overview card
 */
function renderPortfolioOverview() {
  if (!portfolioData) return;

  const positions = portfolioData.positions || [];
  // Get exposure from first exchange (Hyperliquid only for now)
  const exchange = portfolioData.exchanges?.[0] || {};

  // Update equity value
  if (equityValueEl) {
    const equity = portfolioData.total_equity || 0;
    equityValueEl.textContent = formatCurrency(equity);
  }

  // Update unrealized P&L
  if (unrealizedPnlEl) {
    const pnl = portfolioData.total_unrealized_pnl || 0;
    unrealizedPnlEl.textContent = formatCurrency(pnl, true);
    unrealizedPnlEl.className = 'detail-value ' + (pnl >= 0 ? 'positive' : 'negative');
  }

  // Update position count
  if (positionCountEl) {
    positionCountEl.textContent = positions.length;
  }

  // Update exposure percentage
  if (exposurePctEl) {
    // API returns exposure as percentage (e.g., 25.5 for 25.5%)
    const exposurePct = exchange.total_exposure_pct || 0;
    exposurePctEl.textContent = exposurePct.toFixed(1) + '%';
    // Add warning class if exposure is high (>50%)
    exposurePctEl.className = 'detail-value' + (exposurePct > 50 ? ' warning' : '');
  }

  // Update portfolio status footer
  if (portfolioStatusEl) {
    if (!portfolioData.configured) {
      portfolioStatusEl.innerHTML = `<span class="status-text">${portfolioData.message || 'Not configured'}</span>`;
    } else if (portfolioData.error) {
      portfolioStatusEl.innerHTML = `<span class="status-text error">${portfolioData.error}</span>`;
    } else if (portfolioData.total_equity && portfolioData.total_equity > 0) {
      portfolioStatusEl.innerHTML = `<span class="status-text connected">Connected to Hyperliquid</span>`;
    } else {
      portfolioStatusEl.innerHTML = `<span class="status-text">Connected (no balance)</span>`;
    }
  }

  // Update positions count badge
  if (positionsCountBadge) {
    positionsCountBadge.textContent = positions.length;
  }

  // Render positions table
  renderLivePositions(positions);
}

/**
 * Render live positions table
 */
function renderLivePositions(positions) {
  if (!positionsTbody) return;

  if (!positions || positions.length === 0) {
    positionsTbody.innerHTML = `
      <tr class="empty-row">
        <td colspan="6">No open positions</td>
      </tr>
    `;
    return;
  }

  positionsTbody.innerHTML = positions.map(pos => {
    const symbol = pos.symbol || '—';
    const side = (pos.side || '').toLowerCase();
    const sideClass = side === 'long' ? 'positive' : side === 'short' ? 'negative' : '';
    const size = Math.abs(pos.size || 0);
    const entryPrice = pos.entry_price || 0;
    const markPrice = pos.mark_price || pos.entry_price || 0;
    const pnl = pos.unrealized_pnl || 0;
    const pnlClass = pnl >= 0 ? 'positive' : 'negative';
    const leverage = pos.leverage || 1;

    // Calculate P&L percentage if we have entry price
    let pnlPct = '';
    if (entryPrice > 0 && pos.unrealized_pnl != null) {
      const pctValue = (pnl / (size * entryPrice)) * 100 * leverage;
      pnlPct = ` (${pctValue >= 0 ? '+' : ''}${pctValue.toFixed(1)}%)`;
    }

    return `
      <tr>
        <td class="symbol-cell">${symbol}</td>
        <td class="side-cell ${sideClass}">${side.toUpperCase()}</td>
        <td class="size-cell">${size.toFixed(4)}</td>
        <td class="price-cell">$${Number(entryPrice).toLocaleString()}</td>
        <td class="mark-cell">$${Number(markPrice).toLocaleString()}</td>
        <td class="pnl-cell ${pnlClass}">${formatCurrency(pnl, true)}${pnlPct}</td>
      </tr>
    `;
  }).join('');
}

/**
 * Render auto-trade status card
 */
function renderAutoTradeStatus() {
  if (!executionConfig || !executionConfig.configured) {
    // Show placeholder state
    if (toggleStatusEl) toggleStatusEl.textContent = '—';
    return;
  }

  const enabled = executionConfig.enabled;
  const hl = executionConfig.hyperliquid || {};

  // Update main toggle status
  if (toggleStatusEl) {
    toggleStatusEl.textContent = enabled ? 'ON' : 'OFF';
    toggleStatusEl.className = 'toggle-status ' + (enabled ? 'on' : 'off');
  }

  // Update parent toggle element class for styling
  if (autoTradeToggleEl) {
    autoTradeToggleEl.className = 'autotrade-toggle ' + (enabled ? 'on' : 'off');
  }

  // Update max leverage (API returns raw value)
  if (maxLeverageEl) {
    const leverage = hl.max_leverage || 3;
    maxLeverageEl.textContent = `${leverage}x`;
  }

  // Update max position (API returns percentage value)
  if (maxPositionEl) {
    const maxPos = hl.max_position_pct || 2;
    maxPositionEl.textContent = `${maxPos.toFixed(0)}%`;
  }

  // Update max exposure (API returns percentage value)
  if (maxExposureEl) {
    const maxExp = hl.max_exposure_pct || 10;
    maxExposureEl.textContent = `${maxExp.toFixed(0)}%`;
  }
}

/**
 * Initialize overview section
 */
function initOverview() {
  // Set up toggle listener (the toggle is a clickable span, not an input)
  if (autoTradeToggleEl) {
    autoTradeToggleEl.addEventListener('click', toggleAutoTrade);
  }

  // Initial fetches
  fetchPortfolio();
  fetchExecutionConfig();

  // Periodic refresh
  setInterval(fetchPortfolio, 30000); // Every 30 seconds
  setInterval(fetchExecutionConfig, 60000); // Every minute
}

// Initialize on DOMContentLoaded (runs after main init)
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initOverview);
} else {
  initOverview();
}

// =====================
// Tab Restoration (must be after Alpha Pool is defined)
// =====================
const savedTab = localStorage.getItem('activeTab');
if (savedTab && document.getElementById(`tab-${savedTab}`)) {
  switchTab(savedTab);
}
