/**
 * HyperMind Dashboard JavaScript
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
const refreshBtn = document.getElementById('refresh-btn');
const customCountEl = document.getElementById('custom-count');
const customAddressInput = document.getElementById('custom-address-input');
const customNicknameInput = document.getElementById('custom-nickname-input');
const addCustomBtn = document.getElementById('add-custom-btn');
const customErrorEl = document.getElementById('custom-accounts-error');

// =====================
// Configuration
// =====================
const API_BASE = '/dashboard/api';
const SCOUT_API = '/api'; // hl-scout API base (proxied via hl-stream)
const TOP_TABLE_LIMIT = 13; // 10 system + up to 3 custom
const MAX_CUSTOM_ACCOUNTS = 3;

// =====================
// Application State
// =====================
let fillsCache = [];
let dashboardPeriod = 30;
let addressMeta = {};
let customAccountCount = 0;
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
  return {
    id: `${fill.address}-${fill.time_utc}-${Math.random().toString(36).slice(2, 8)}`,
    time_utc: fill.time_utc,
    oldest_time: fill.time_utc,
    address: fill.address,
    symbol: symbol,
    action: fill.action || '',
    fills: [fill],
    totalSize: Math.abs(fill.size_signed || 0),
    totalPnl: fill.closed_pnl_usd || 0,
    prices: fill.price_usd ? [fill.price_usd] : [],
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
  const sameAddress = group.address === fill.address;
  const sameSymbol = group.symbol === symbol;
  const sameAction = group.action === (fill.action || '');

  return sameAddress && sameSymbol && sameAction && withinWindow;
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

  group.fills.push(fill);
  group.totalSize += Math.abs(fill.size_signed || 0);
  group.totalPnl += fill.closed_pnl_usd || 0;
  if (fill.price_usd) {
    group.prices.push(fill.price_usd);
  }

  // Update time range
  if (fillTime > new Date(group.time_utc).getTime()) {
    group.time_utc = fill.time_utc;
  }
  if (fillTime < new Date(group.oldest_time).getTime()) {
    group.oldest_time = fill.time_utc;
  }

  // Update previous_position - use largest absolute value (true starting position)
  const fillPrev = fill.previous_position;
  const groupPrev = group.previous_position;
  if (fillPrev != null && (groupPrev == null || Math.abs(fillPrev) > Math.abs(groupPrev))) {
    group.previous_position = fillPrev;
  }

  // Update computed fields
  group.fillCount = group.fills.length;
  group.isAggregated = group.fills.length > 1;
  group.avgPrice = group.prices.length > 0
    ? group.prices.reduce((a, b) => a + b, 0) / group.prices.length
    : null;

  // Update signed size based on action
  const isShort = group.action.toLowerCase().includes('short');
  const isDecrease = group.action.toLowerCase().includes('decrease') || group.action.toLowerCase().includes('close');
  group.size_signed = isShort || isDecrease ? -group.totalSize : group.totalSize;
  group.closed_pnl_usd = group.totalPnl || null;
  group.price_usd = group.avgPrice;
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
function fmtPrice(value) {
  if (!Number.isFinite(value)) return '—';
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
    const action = fill.action || '';
    const address = fill.address;

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
        // For previous_position, we need the position BEFORE any fills in the group.
        // When fills are concurrent (same timestamp), they have different previous_position
        // values representing parallel fills against the order book. The "true" starting
        // position is the one with the largest absolute value (furthest from zero).
        const fillPrev = fill.previous_position;
        const groupPrev = currentGroup.previous_position;
        if (fillPrev != null && (groupPrev == null || Math.abs(fillPrev) > Math.abs(groupPrev))) {
          currentGroup.previous_position = fillPrev;
        }
        continue;
      }
    }

    // Finalize current group and start new one
    if (currentGroup) {
      finalizeGroup(currentGroup);
      aggregated.push(currentGroup);
    }

    // Start new group
    currentGroup = {
      id: `${address}-${fill.time_utc}-${Math.random().toString(36).slice(2, 8)}`,
      time_utc: fill.time_utc,
      oldest_time: fill.time_utc,
      address: address,
      symbol: symbol,
      action: action,
      fills: [fill],
      totalSize: Math.abs(fill.size_signed || 0),
      totalPnl: fill.closed_pnl_usd || 0,
      prices: fill.price_usd ? [fill.price_usd] : [],
      previous_position: fill.previous_position,
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
  const isShort = group.action.toLowerCase().includes('short');
  const isDecrease = group.action.toLowerCase().includes('decrease') || group.action.toLowerCase().includes('close');
  group.size_signed = isShort || isDecrease ? -group.totalSize : group.totalSize;
  group.closed_pnl_usd = group.totalPnl || null;
  group.price_usd = group.avgPrice;
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

      const isCustom = row.isCustom === true;
      const customIndicator = isCustom ? '<span class="custom-star" title="Custom tracked account">★</span>' : '';
      const removeBtn = isCustom ? `<button class="remove-custom-btn" data-address="${row.address}" title="Remove custom account">×</button>` : '';
      const nicknameDisplay = row.remark
        ? `<span class="nickname-display" data-address="${row.address}" data-nickname="${escapeHtml(row.remark)}" title="Click to edit nickname">${escapeHtml(row.remark)}</span>`
        : (isCustom ? `<span class="nickname-display nickname-empty" data-address="${row.address}" data-nickname="" title="Click to add nickname">+ Add nickname</span>` : '');
      const addrLower = (row.address || '').toLowerCase();
      return `
        <tr class="${isCustom ? 'custom-row' : ''}" data-address="${addrLower}">
          <td data-label="Address" title="Score: ${scoreValue}">
            <span class="custom-indicator">
              ${customIndicator}
              <a href="https://hypurrscan.io/address/${row.address}" target="_blank" rel="noopener noreferrer">
                ${shortAddress(row.address)}
              </a>
              ${removeBtn}
            </span>
            <div class="addr-remark">${nicknameDisplay}</div>
          </td>
          <td data-label="Win Rate">${winRateCell}</td>
          <td data-label="Trades">${tradesCell}</td>
          <td data-label="Holdings" class="holds-cell">
            ${holdingCell}
          </td>
          <td data-label="Realized PnL">${pnlCell}</td>
          <td data-label="30D PnL" class="sparkline-cell">${sparklineCell}</td>
        </tr>
      `;
    })
    .join('');

  // Attach event listeners for remove buttons
  document.querySelectorAll('.remove-custom-btn').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      const address = btn.dataset.address;
      if (address) removeCustomAccount(address);
    });
  });

  // Attach event listeners for nickname editing (custom accounts only)
  document.querySelectorAll('.nickname-display').forEach((el) => {
    el.addEventListener('click', (e) => {
      e.preventDefault();
      const address = el.dataset.address;
      const currentNickname = el.dataset.nickname || '';
      if (address) showNicknameEditor(el, address, currentNickname);
    });
  });
}

// Mock AI recommendations data
const mockAIRecommendations = [
  {
    time: new Date(Date.now() - 2 * 60000).toISOString(),
    symbol: 'BTC',
    action: 'LONG',
    entry: 97250,
    stopLoss: 96500,
    takeProfit: 99000,
    status: 'active'
  },
  {
    time: new Date(Date.now() - 15 * 60000).toISOString(),
    symbol: 'ETH',
    action: 'SHORT',
    entry: 3580,
    stopLoss: 3650,
    takeProfit: 3450,
    status: 'active'
  },
  {
    time: new Date(Date.now() - 45 * 60000).toISOString(),
    symbol: 'BTC',
    action: 'LONG',
    entry: 96800,
    stopLoss: 96200,
    takeProfit: 97500,
    status: 'tp_hit'
  },
  {
    time: new Date(Date.now() - 2 * 3600000).toISOString(),
    symbol: 'ETH',
    action: 'LONG',
    entry: 3520,
    stopLoss: 3480,
    takeProfit: 3600,
    status: 'tp_hit'
  },
  {
    time: new Date(Date.now() - 5 * 3600000).toISOString(),
    symbol: 'BTC',
    action: 'SHORT',
    entry: 97100,
    stopLoss: 97500,
    takeProfit: 96200,
    status: 'sl_hit'
  }
];

function renderAIRecommendations() {
  if (!aiRecommendationsTable) return;

  const rows = mockAIRecommendations.map(rec => {
    const actionClass = rec.action === 'LONG' ? 'buy' : 'sell';
    let statusClass = '';
    let statusText = '';

    switch (rec.status) {
      case 'active':
        statusClass = 'status-active';
        statusText = 'Active';
        break;
      case 'tp_hit':
        statusClass = 'status-tp';
        statusText = 'TP Hit';
        break;
      case 'sl_hit':
        statusClass = 'status-sl';
        statusText = 'SL Hit';
        break;
      case 'expired':
        statusClass = 'status-expired';
        statusText = 'Expired';
        break;
      default:
        statusClass = '';
        statusText = rec.status;
    }

    return `
      <tr>
        <td data-label="Time">${fmtTime(rec.time)}</td>
        <td data-label="Symbol">${rec.symbol}</td>
        <td data-label="Action"><span class="pill ${actionClass}">${rec.action}</span></td>
        <td data-label="Entry">${fmtPrice(rec.entry)}</td>
        <td data-label="SL">${fmtPrice(rec.stopLoss)}</td>
        <td data-label="TP">${fmtPrice(rec.takeProfit)}</td>
        <td data-label="Status"><span class="ai-status-badge ${statusClass}">${statusText}</span></td>
      </tr>
    `;
  }).join('');

  aiRecommendationsTable.innerHTML = rows;

  // Update AI status
  if (aiStatusEl) {
    const activeCount = mockAIRecommendations.filter(r => r.status === 'active').length;
    aiStatusEl.innerHTML = `
      <span class="ai-status-dot"></span>
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
    ? `~${fmtUsdShort(group.avgPrice)}`
    : fmtUsdShort(group.price_usd ?? null);
  const pnl = fmtUsdShort(group.closed_pnl_usd ?? null);
  const action = group.action || '—';
  const sideClass = action.toLowerCase().includes('short') ? 'sell' : 'buy';

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
      <td data-label="Time">${fmtDateTime(group.time_utc)}</td>
      <td data-label="Address"><a href="https://hypurrscan.io/address/${group.address}" target="_blank" rel="noopener noreferrer">${shortAddress(group.address)}</a></td>
      <td data-label="Action"><span class="pill ${sideClass}">${action}</span>${aggBadge}</td>
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
      const fillPrice = fmtUsdShort(fill.price_usd ?? null);
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
      <div class="fills-empty-state">
        <span class="empty-icon">📊</span>
        <p>No BTC/ETH fills yet</p>
        <span class="empty-hint">Fills will appear here as they happen</span>
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
  if (lastRefresh) {
    const date = new Date(lastRefresh);
    lastRefreshEl.textContent = `Last updated: ${date.toLocaleString([], { month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit' })}`;
  } else {
    lastRefreshEl.textContent = 'Last updated: —';
  }
}

function updateCustomAccountCount(count, max) {
  customAccountCount = count;
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
    const summaryUrl = `${API_BASE}/summary?period=${dashboardPeriod}&limit=${TOP_TABLE_LIMIT}`;
    const data = await fetchJson(summaryUrl);
    const rows = Array.isArray(data.stats)
      ? data.stats
      : Array.isArray(data.selected)
        ? data.selected
        : [];
    const holdings = normalizeHoldings(data.holdings || {});
    addressMeta = {};
    rows.forEach((row) => {
      if (!row?.address) return;
      addressMeta[row.address.toLowerCase()] = { remark: row.remark || null };
    });
    renderAddresses(rows, data.profiles || {}, holdings);

    // Update last refresh display
    updateLastRefreshDisplay(data.lastRefresh);

    // Update custom account count
    if (typeof data.customAccountCount === 'number') {
      updateCustomAccountCount(data.customAccountCount, data.maxCustomAccounts || MAX_CUSTOM_ACCOUNTS);
    }
  } catch (err) {
    console.error('Failed to load summary:', err);
  }
}

async function refreshFills() {
  try {
    const data = await fetchJson(`${API_BASE}/fills?limit=40`);
    fillsCache = data.fills || [];

    // Update hasMoreFills based on response
    if (data.hasMore !== undefined) {
      hasMoreFills = data.hasMore;
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
            const row = {
              time_utc: e.at,
              address: e.address,
              action: e.action || e.payload?.action || '',
              size_signed: Number(sizeSigned),
              previous_position: startPos != null ? Number(startPos) : null,
              price_usd: e.priceUsd ?? e.payload?.priceUsd ?? null,
              closed_pnl_usd: e.realizedPnlUsd ?? e.payload?.realizedPnlUsd ?? null,
              symbol
            };
            pushFill(row);
          });
      }
    } catch (err) {
      console.error('ws parse', err);
    }
  });
  ws.addEventListener('close', () => {
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

// Add a custom account
async function addCustomAccount() {
  clearCustomError();
  const address = customAddressInput.value.trim();
  const nickname = customNicknameInput.value.trim();

  if (!address) {
    showCustomError('Please enter an Ethereum address');
    return;
  }

  if (!isValidEthAddress(address)) {
    showCustomError('Invalid Ethereum address format (must be 0x + 40 hex characters)');
    return;
  }

  if (customAccountCount >= MAX_CUSTOM_ACCOUNTS) {
    showCustomError(`Maximum of ${MAX_CUSTOM_ACCOUNTS} custom accounts allowed`);
    return;
  }

  addCustomBtn.disabled = true;
  addCustomBtn.textContent = 'Adding...';

  try {
    const res = await fetch(`${API_BASE}/custom-accounts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ address, nickname: nickname || undefined })
    });

    const data = await res.json();

    if (!res.ok) {
      showCustomError(data.error || 'Failed to add custom account');
      return;
    }

    // Clear inputs on success
    customAddressInput.value = '';
    customNicknameInput.value = '';

    // Refresh the summary to show the new account
    await refreshSummary();
  } catch (err) {
    console.error('Add custom account error:', err);
    showCustomError('Failed to add custom account');
  } finally {
    addCustomBtn.disabled = customAccountCount >= MAX_CUSTOM_ACCOUNTS;
    addCustomBtn.textContent = 'Add';
  }
}

// Remove a custom account
async function removeCustomAccount(address) {
  if (!address) return;

  try {
    const res = await fetch(`${API_BASE}/custom-accounts/${encodeURIComponent(address)}`, {
      method: 'DELETE'
    });

    if (!res.ok) {
      const data = await res.json();
      console.error('Remove custom account error:', data.error);
      return;
    }

    // Refresh the summary to update the table
    await refreshSummary();
  } catch (err) {
    console.error('Remove custom account error:', err);
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

// Trigger manual leaderboard refresh
async function triggerLeaderboardRefresh() {
  refreshBtn.disabled = true;
  refreshBtn.classList.add('loading');

  try {
    const res = await fetch(`${API_BASE}/leaderboard/refresh`, {
      method: 'POST'
    });

    const data = await res.json();

    if (!res.ok) {
      console.error('Refresh error:', data.error);
      return;
    }

    // Poll for refresh completion
    pollRefreshStatus();
  } catch (err) {
    console.error('Refresh error:', err);
    refreshBtn.disabled = false;
    refreshBtn.classList.remove('loading');
  }
}

// Poll refresh status until complete
async function pollRefreshStatus() {
  try {
    const data = await fetchJson(`${API_BASE}/leaderboard/refresh-status`);

    if (data.status === 'refreshing') {
      setTimeout(pollRefreshStatus, 2000);
      return;
    }

    // Refresh complete
    refreshBtn.disabled = false;
    refreshBtn.classList.remove('loading');

    if (data.status === 'idle') {
      await refreshSummary();
    }
  } catch (err) {
    console.error('Poll refresh status error:', err);
    refreshBtn.disabled = false;
    refreshBtn.classList.remove('loading');
  }
}

// Initialize custom accounts controls
function initCustomAccountsControls() {
  addCustomBtn.addEventListener('click', addCustomAccount);

  // Allow Enter key to submit
  customAddressInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') addCustomAccount();
  });
  customNicknameInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') addCustomAccount();
  });
}

// Initialize refresh button
function initRefreshButton() {
  refreshBtn.addEventListener('click', triggerLeaderboardRefresh);
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
    const url = `${API_BASE}/fills/backfill?before=${encodeURIComponent(beforeTime)}&limit=30`;
    const data = await fetchJson(url);

    if (data.fills && data.fills.length > 0) {
      // Append to cache
      fillsCache = [...fillsCache, ...data.fills];
      hasMoreFills = data.hasMore;

      // Update oldest time
      if (data.oldestTime) {
        fillsOldestTime = data.oldestTime;
      }

      // Re-render with all fills (aggregation will be applied)
      renderFills(fillsCache);
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

// Fetch historical fills from Hyperliquid API
async function fetchHistoricalFills() {
  try {
    const response = await fetch(`${API_BASE}/fills/fetch-history`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ limit: 50 })
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (err) {
    console.error('Fetch historical fills error:', err);
    return null;
  }
}

// Initialize load history button
function initLoadHistoryButton() {
  const btn = document.getElementById('load-history-btn');
  if (!btn) return;

  btn.addEventListener('click', async () => {
    if (isLoadingMore) return;
    isLoadingMore = true;

    btn.disabled = true;
    btn.classList.add('loading');
    btn.textContent = 'Loading...';

    try {
      // Fetch historical fills from Hyperliquid API
      const result = await fetchHistoricalFills();

      if (result && result.fills && result.fills.length > 0) {
        // Update cache with fetched fills
        fillsCache = result.fills;
        hasMoreFills = result.hasMore !== false;
        fillsOldestTime = result.oldestTime;
        renderFills(fillsCache);

        // Show success message
        const newCount = result.inserted || 0;
        if (newCount > 0) {
          btn.textContent = `✓ ${newCount} new fills`;
        } else {
          btn.textContent = `✓ ${result.fills.length} fills loaded`;
        }
        btn.classList.remove('loading');

        // Reset button text after delay
        setTimeout(() => {
          updateFillsUI();
        }, 2000);
      } else if (result) {
        // API returned but no fills - maybe all caught up
        hasMoreFills = false;
        btn.textContent = 'No more fills';
        btn.classList.remove('loading');
        updateFillsUI();
      } else {
        // API call failed, try loading from local DB
        btn.textContent = 'Retrying...';
        btn.classList.remove('loading');
        await loadMoreFills();
      }
    } catch (err) {
      console.error('Load history button error:', err);
      btn.textContent = 'Error - retry';
      btn.classList.remove('loading');
    } finally {
      isLoadingMore = false;
      btn.disabled = false;
    }
  });
}

async function init() {
  initChartControls();
  initCustomAccountsControls();
  initRefreshButton();
  initInfiniteScroll();
  initLoadHistoryButton();
  renderChart('BTCUSDT');

  // Initialize fills UI with initial state
  updateFillsUI();

  // Fetch initial prices
  fetchPrices();
  // Check positions status FIRST before loading data
  await checkPositionsStatus();
  refreshSummary();
  refreshFills();
  renderAIRecommendations();
  connectWs();
  // Continue polling until positions are ready (if not already)
  if (!positionsReady) {
    pollPositionsUntilReady();
  }
  setInterval(refreshSummary, 30_000);
  setInterval(refreshFills, 20_000);
}

document.addEventListener('DOMContentLoaded', init);
