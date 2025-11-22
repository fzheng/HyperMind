const statusEl = document.getElementById('dashboard-status');
const addressTable = document.getElementById('address-table');
const fillsTable = document.getElementById('fills-table');
const decisionsList = document.getElementById('decisions-list');
const recommendationCard = document.getElementById('recommendation-card');
const symbolButtons = document.querySelectorAll('.toggle-group button');
const periodButtons = document.querySelectorAll('.period-toggle button');
const lastRefreshEl = document.getElementById('last-refresh');
const refreshBtn = document.getElementById('refresh-btn');
const customCountEl = document.getElementById('custom-count');
const customAddressInput = document.getElementById('custom-address-input');
const customNicknameInput = document.getElementById('custom-nickname-input');
const addCustomBtn = document.getElementById('add-custom-btn');
const customErrorEl = document.getElementById('custom-accounts-error');

const API_BASE = '/dashboard/api';
const SCOUT_API = '/api'; // hl-scout API base (proxied via hl-stream)
const TOP_TABLE_LIMIT = 13; // 10 system + up to 3 custom
let fillsCache = [];
let dashboardPeriod = 30;
let addressMeta = {};
let customAccountCount = 0;
const MAX_CUSTOM_ACCOUNTS = 3;

function placeholder(text = 'No live data') {
  return `<span class="placeholder">${text}</span>`;
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
    return placeholder('No live position');
  }
  const size = Number(entry.size);
  const symbol = (entry.symbol || '').toUpperCase();
  const direction = size >= 0 ? 'holding-long' : 'holding-short';
  const magnitude = Math.abs(size);
  const precision = magnitude >= 1 ? 2 : 3;
  const signed = `${size >= 0 ? '+' : '-'}${magnitude.toFixed(precision)} ${symbol || ''}`.trim();
  return `<span class="holding-chip ${direction}" title="Live Hyperliquid position">${signed}</span>`;
}

function normalizeHoldings(raw = {}) {
  const normalized = {};
  Object.entries(raw).forEach(([addr, pos]) => {
    if (!addr) return;
    const key = addr.toLowerCase();
    normalized[key] = {
      symbol: (pos?.symbol || '').toUpperCase(),
      size: Number(pos?.size ?? 0),
    };
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

function renderAddresses(stats = [], profiles = {}, holdings = {}) {
  const rows = (stats || []).slice(0, TOP_TABLE_LIMIT);
  if (!rows.length) {
    addressTable.innerHTML = `<tr><td colspan="7">${placeholder('No live leaderboard data')}</td></tr>`;
    return;
  }
  addressTable.innerHTML = rows
    .map((row) => {
      const txCount = profiles[row.address]?.txCount || 0;
      const winRateCell = typeof row.winRate === 'number' ? fmtPercent(row.winRate) : placeholder();
      const drawdownCell = typeof row.statMaxDrawdown === 'number' ? fmtPercent(row.statMaxDrawdown) : placeholder();
      const tradesValue =
        typeof row.statClosedPositions === 'number'
          ? row.statClosedPositions
          : typeof row.executedOrders === 'number'
            ? row.executedOrders
            : null;
      const tradesCell = tradesValue === null ? placeholder() : tradesValue;
      const holdingKey = row.address?.toLowerCase() || '';
      const holdingCell = holdings[holdingKey] ? formatHolding(holdings[holdingKey]) : placeholder('No live position');
      const pnlCell = typeof row.realizedPnl === 'number' ? fmtUsdShort(row.realizedPnl) : placeholder();
      const scoreCell = typeof row.score === 'number' ? fmtScore(row.score) : placeholder();
      const isCustom = row.isCustom === true;
      const customIndicator = isCustom ? '<span class="custom-star" title="Custom tracked account">★</span>' : '';
      const removeBtn = isCustom ? `<button class="remove-custom-btn" data-address="${row.address}" title="Remove custom account">×</button>` : '';
      return `
        <tr class="${isCustom ? 'custom-row' : ''}">
          <td data-label="Address" title="Hyperliquid tx count: ${txCount}">
            <span class="custom-indicator">
              ${customIndicator}
              <a href="https://hyperbot.network/trader/${row.address}" target="_blank" rel="noopener noreferrer">
                ${shortAddress(row.address)}
              </a>
              ${removeBtn}
            </span>
            ${row.remark ? `<div class="addr-remark">${row.remark}</div>` : ''}
          </td>
          <td data-label="Win Rate">${winRateCell}</td>
          <td data-label="Max Drawdown">${drawdownCell}</td>
          <td data-label="Trades">${tradesCell}</td>
          <td data-label="Holdings" class="holds-cell">
            ${holdingCell}
          </td>
          <td data-label="Realized PnL">${pnlCell}</td>
          <td data-label="Score" class="score-cell">${scoreCell}</td>
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
}

function renderRecommendation(summary) {
  if (!summary?.recommendation) {
    recommendationCard.innerHTML = '<p>No signal yet.</p>';
    return;
  }
  const rec = summary.recommendation;
  const featured = summary.featured;
  const profile = summary.profiles?.[rec.address];
  const remark = addressMeta[rec.address?.toLowerCase()]?.remark || '';
  const winRateText = typeof rec.winRate === 'number' ? fmtPercent(rec.winRate) : 'N/A (no live data)';
  const realizedText = typeof rec.realizedPnl === 'number' ? fmtUsdShort(rec.realizedPnl) : 'N/A (no live data)';
  const weightText = typeof rec.weight === 'number' ? `${(rec.weight * 100).toFixed(1)}%` : 'N/A';
  recommendationCard.innerHTML = `
    <span>Focus address</span>
    <strong>${remark ? `${remark} (${shortAddress(rec.address)})` : rec.address}</strong>
    <span>Win rate: ${winRateText} • Realized: ${realizedText}</span>
    ${
      featured
        ? `<span>Latest fill: ${featured.side.toUpperCase()} ${featured.size} @ ${featured.priceUsd}</span>`
        : ''
    }
    <span>Weight: ${weightText}</span>
    ${profile ? `<span>Total HL transactions: ${profile.txCount || 0}</span>` : ''}
    <em>${rec.message}</em>
  `;
}

function renderFills(list) {
  fillsTable.innerHTML = list
    .slice(0, 10)
    .map((fill) => {
      const symbol = (fill.symbol || 'BTC').toUpperCase();
      const size = typeof fill.size_signed === 'number' ? `${fill.size_signed >= 0 ? '+' : ''}${fill.size_signed.toFixed(5)} ${symbol}` : '—';
      const prev = typeof fill.previous_position === 'number' ? `${fill.previous_position.toFixed(5)} ${symbol}` : '—';
      const price = fmtUsdShort(fill.price_usd ?? null);
      const pnl = fmtUsdShort(fill.closed_pnl_usd ?? null);
      const action = fill.action || '—';
      const sideClass = action.toLowerCase().includes('short') ? 'sell' : 'buy';
      return `
        <tr>
          <td data-label="Time">${fmtDateTime(fill.time_utc)}</td>
          <td data-label="Address"><a href="https://hyperbot.network/trader/${fill.address}" target="_blank" rel="noopener noreferrer">${shortAddress(fill.address)}</a></td>
          <td data-label="Action"><span class="pill ${sideClass}">${action}</span></td>
          <td data-label="Size">${size}</td>
          <td data-label="Previous Position">${prev}</td>
          <td data-label="Price">${price}</td>
          <td data-label="Closed PnL">${pnl}</td>
        </tr>
      `;
    })
    .join('');
}

function renderDecisions(list) {
  decisionsList.innerHTML = list
    .map(
      (d) => `
        <li>
          <span class="decision-meta"><strong>${d.address}</strong> · ${d.asset} · ${fmtTime(d.ts)}</span>
          <span class="decision-status ${d.status === 'open' ? 'status-open' : 'status-closed'}">
            ${d.side.toUpperCase()} ${d.status === 'closed' && d.result != null ? `(${d.result.toFixed(2)})` : ''}
          </span>
        </li>
      `
    )
    .join('');
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
    renderRecommendation(data);
    statusEl.textContent = `Updated ${new Date().toLocaleTimeString()}`;

    // Update last refresh display
    updateLastRefreshDisplay(data.lastRefresh);

    // Update custom account count
    if (typeof data.customAccountCount === 'number') {
      updateCustomAccountCount(data.customAccountCount, data.maxCustomAccounts || MAX_CUSTOM_ACCOUNTS);
    }
  } catch (err) {
    statusEl.textContent = 'Failed to load summary';
    console.error(err);
  }
}

async function refreshFills() {
  try {
    const data = await fetchJson(`${API_BASE}/fills?limit=40`);
    fillsCache = data.fills || [];
    renderFills(fillsCache);
  } catch (err) {
    console.error(err);
  }
}

async function refreshDecisions() {
  try {
    const data = await fetchJson(`${API_BASE}/decisions?limit=20`);
    renderDecisions(data.decisions || []);
  } catch (err) {
    console.error(err);
  }
}

function pushFill(fill) {
  fillsCache.unshift(fill);
  fillsCache = fillsCache.slice(0, 40);
  renderFills(fillsCache);
}

function connectWs() {
  const wsUrl = (location.origin.startsWith('https') ? 'wss://' : 'ws://') + location.host + '/ws';
  const ws = new WebSocket(wsUrl);
  ws.addEventListener('message', (evt) => {
    try {
      const payload = JSON.parse(evt.data);
      const events = payload.events || payload.batch || payload;
      if (Array.isArray(events)) {
        events
          .filter((e) => e.type === 'trade')
          .forEach((e) => {
            const symbol = e.symbol || 'BTC';
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
    document.getElementById('tradingview_chart').innerHTML = '';
    // eslint-disable-next-line no-undef
    new TradingView.widget({
      autosize: true,
      symbol: `BINANCE:${symbol}`,
      interval: '60',
      timezone: 'Etc/UTC',
      theme: 'dark',
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
      renderChart(btn.dataset.symbol || 'BTCUSDT');
    });
  });
}

function initPeriodControls() {
  periodButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      periodButtons.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      dashboardPeriod = Number(btn.dataset.period || 30);
      refreshSummary();
    });
  });
}

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
      statusEl.textContent = 'Refresh failed';
      return;
    }

    // Poll for refresh completion
    statusEl.textContent = 'Refreshing leaderboard...';
    pollRefreshStatus();
  } catch (err) {
    console.error('Refresh error:', err);
    statusEl.textContent = 'Refresh failed';
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
      statusEl.textContent = 'Refresh complete';
      await refreshSummary();
    }
  } catch (err) {
    console.error('Poll refresh status error:', err);
    refreshBtn.disabled = false;
    refreshBtn.classList.remove('loading');
    statusEl.textContent = 'Refresh status unknown';
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

function init() {
  initChartControls();
  initPeriodControls();
  initCustomAccountsControls();
  initRefreshButton();
  renderChart('BTCUSDT');
  refreshSummary();
  refreshFills();
  refreshDecisions();
  connectWs();
  setInterval(refreshSummary, 30_000);
  setInterval(refreshFills, 20_000);
  setInterval(refreshDecisions, 45_000);
}

document.addEventListener('DOMContentLoaded', init);
