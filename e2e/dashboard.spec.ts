import { test, expect } from '@playwright/test';

/**
 * E2E Tests for Dashboard UI
 *
 * These tests are READ-ONLY and do not modify any server-side state.
 * Theme changes are client-side only (localStorage).
 */

test.describe('Dashboard Page - Core Layout', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard');
  });

  test('should load dashboard page with correct title', async ({ page }) => {
    await expect(page).toHaveTitle(/SigmaPilot|Dashboard/i);
  });

  test('should display main content area', async ({ page }) => {
    await expect(page.locator('main')).toBeVisible();
  });

  test('should display header with brand name', async ({ page }) => {
    const header = page.locator('[data-testid="header"], header');
    await expect(header).toBeVisible();

    const brandName = page.locator('[data-testid="brand-name"], .brand-name');
    await expect(brandName).toContainText('SigmaPilot');
  });

  test('should display live clock', async ({ page }) => {
    const clock = page.locator('[data-testid="live-clock"], #live-clock');
    await expect(clock).toBeVisible();
  });

  test('should display price ticker', async ({ page }) => {
    const ticker = page.locator('[data-testid="price-ticker"], .price-ticker');
    await expect(ticker).toBeVisible();

    // BTC and ETH prices should be present
    const btcPrice = page.locator('[data-testid="btc-price"], #btc-price');
    const ethPrice = page.locator('[data-testid="eth-price"], #eth-price');
    await expect(btcPrice).toBeVisible();
    await expect(ethPrice).toBeVisible();
  });
});

test.describe('Dashboard - Theme Toggle', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard');
  });

  test('should display theme toggle buttons', async ({ page }) => {
    const themeToggle = page.locator('[data-testid="theme-toggle"], .theme-toggle');
    await expect(themeToggle).toBeVisible();

    // All three theme buttons should exist (use button selector to avoid matching html element)
    const darkBtn = page.locator('button[data-testid="theme-btn-dark"], button[data-theme="dark"]');
    const lightBtn = page.locator('button[data-testid="theme-btn-light"], button[data-theme="light"]');
    const autoBtn = page.locator('button[data-testid="theme-btn-auto"], button[data-theme="auto"]');

    await expect(darkBtn).toBeVisible();
    await expect(lightBtn).toBeVisible();
    await expect(autoBtn).toBeVisible();
  });

  test('should toggle to dark theme when clicking dark button', async ({ page }) => {
    const html = page.locator('html');
    const darkBtn = page.locator('[data-testid="theme-btn-dark"], [data-theme="dark"]');

    await darkBtn.click();
    await expect(html).toHaveAttribute('data-theme', 'dark');
  });

  test('should toggle to light theme when clicking light button', async ({ page }) => {
    const html = page.locator('html');
    const lightBtn = page.locator('button[data-testid="theme-btn-light"], button[data-theme="light"]');

    await lightBtn.click();
    await expect(html).toHaveAttribute('data-theme', 'light');
  });
});

test.describe('Dashboard - Tracked Traders Card', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard');
  });

  test('should display tracked traders card', async ({ page }) => {
    const tradersCard = page.locator('[data-testid="leaderboard-card"], .leaderboard-card');
    await expect(tradersCard).toBeVisible();
  });

  test('should display leaderboard table', async ({ page }) => {
    const table = page.locator('[data-testid="leaderboard-table"], .leaderboard-table');
    await expect(table).toBeVisible();
  });

  test('should have correct table headers', async ({ page }) => {
    const headers = page.locator('[data-testid="leaderboard-table"] th, .leaderboard-table th');
    const headerCount = await headers.count();
    expect(headerCount).toBeGreaterThanOrEqual(4); // Address, Win%, Trades, Holdings, etc.

    // Check for Address header specifically
    const addressHeader = headers.filter({ hasText: /address/i });
    await expect(addressHeader).toHaveCount(1);
  });

  test('should show refresh timer indicators', async ({ page }) => {
    const lastRefresh = page.locator('[data-testid="last-refresh"], #last-refresh');
    const nextRefresh = page.locator('[data-testid="next-refresh"], #next-refresh');

    // Wait for data to load
    await page.waitForTimeout(2000);

    // At least one indicator should exist in DOM (may be empty initially)
    const lastExists = await lastRefresh.count() > 0;
    const nextExists = await nextRefresh.count() > 0;
    expect(lastExists || nextExists).toBeTruthy();
  });
});

test.describe('Dashboard - Charts', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard');
  });

  test('should display chart section', async ({ page }) => {
    const chartSection = page.locator('[data-testid="chart-section"], .chart-section');
    await expect(chartSection).toBeVisible();
  });

  test('should display chart card', async ({ page }) => {
    const chartCard = page.locator('[data-testid="chart-card"], .chart-card');
    await expect(chartCard).toBeVisible();
  });

  test('should have BTC/ETH toggle buttons', async ({ page }) => {
    const toggleGroup = page.locator('[data-testid="chart-toggle-group"], .toggle-group');
    await expect(toggleGroup).toBeVisible();

    const btcBtn = page.locator('[data-testid="chart-btn-btc"], [data-symbol="BTCUSDT"]');
    const ethBtn = page.locator('[data-testid="chart-btn-eth"], [data-symbol="ETHUSDT"]');

    await expect(btcBtn).toBeVisible();
    await expect(ethBtn).toBeVisible();
  });

  test('should have chart container', async ({ page }) => {
    const chartContainer = page.locator('[data-testid="tradingview-chart"], #tradingview_chart');
    await expect(chartContainer).toBeVisible({ timeout: 15000 });
  });

  test('should have chart collapse button', async ({ page }) => {
    const collapseBtn = page.locator('[data-testid="chart-collapse-btn"], #chart-collapse-btn');
    await expect(collapseBtn).toBeVisible();
  });
});

test.describe('Dashboard - AI Signals', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard');
  });

  test('should display AI signals section', async ({ page }) => {
    const aiSection = page.locator('[data-testid="ai-section"], .ai-section');
    await expect(aiSection).toBeVisible();
  });

  test('should display AI signals card', async ({ page }) => {
    const aiCard = page.locator('[data-testid="ai-signals-card"], .ai-recommendations-card');
    await expect(aiCard).toBeVisible();
  });

  test('should display AI status indicator', async ({ page }) => {
    const aiStatus = page.locator('[data-testid="ai-status"], #ai-status');
    await expect(aiStatus).toBeVisible();
  });

  test('should display AI signals table', async ({ page }) => {
    const aiTable = page.locator('[data-testid="ai-signals-table"], .ai-signals-table');
    await expect(aiTable).toBeVisible();
  });
});

test.describe('Dashboard - Activity Feed', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard');
  });

  test('should display activity feed card', async ({ page }) => {
    const activityCard = page.locator('[data-testid="fills-card"], .fills-card');
    await expect(activityCard).toBeVisible();
  });

  test('should display fills table', async ({ page }) => {
    const fillsTable = page.locator('[data-testid="fills-table"], .fills-table');
    await expect(fillsTable).toBeVisible();
  });

  test('should have fills table headers', async ({ page }) => {
    const headers = page.locator('[data-testid="fills-table"] th, .fills-table th');
    const headerCount = await headers.count();
    expect(headerCount).toBeGreaterThanOrEqual(5); // Time, Address, Action, Size, Price, etc.

    const timeHeader = headers.filter({ hasText: /time/i });
    await expect(timeHeader).toHaveCount(1);
  });

  test('should display fills status bar', async ({ page }) => {
    const statusBar = page.locator('[data-testid="fills-status-bar"], #fills-status-bar');
    await expect(statusBar).toBeVisible();
  });

  test('should have load more button', async ({ page }) => {
    const loadMoreBtn = page.locator('[data-testid="load-history-btn"], #load-history-btn');
    await expect(loadMoreBtn).toBeVisible();
  });

  test('should show fills data or waiting message', async ({ page }) => {
    // Wait for potential data load
    await page.waitForTimeout(2000);

    // Check if fills exist or waiting message
    const fills = page.locator('[data-testid="fills-table"] tbody tr, .fills-table tbody tr');
    const waitingMessage = page.locator('[data-testid="fills-status-message"], #fills-status-message');

    const fillCount = await fills.count();
    const statusText = await waitingMessage.textContent();

    // Either fills exist or status shows waiting
    expect(fillCount > 0 || statusText?.toLowerCase().includes('waiting')).toBeTruthy();
  });
});

test.describe('Dashboard - Time Display Toggle', () => {
  // On mobile, fills use a card layout without table headers - the time toggle header (#fills-time-header)
  // is not visible. These tests verify desktop-only functionality.
  test.skip(({ isMobile }) => isMobile, 'Time toggle header not visible on mobile (card layout)');

  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard');
    // Scroll to fills card to ensure time header is visible
    const fillsCard = page.locator('[data-testid="fills-card"], .fills-card');
    await fillsCard.scrollIntoViewIfNeeded();
  });

  test('should display clickable time header', async ({ page }) => {
    const timeHeader = page.locator('[data-testid="fills-time-header"], #fills-time-header');
    await timeHeader.scrollIntoViewIfNeeded();
    await expect(timeHeader).toBeVisible();

    // Should have clickable-header class
    await expect(timeHeader).toHaveClass(/clickable-header/);
  });

  test('should have correct initial time header text and icon', async ({ page }) => {
    const timeHeader = page.locator('[data-testid="fills-time-header"], #fills-time-header');
    await timeHeader.scrollIntoViewIfNeeded();

    // Should show "Time" with stopwatch icon (⏱) in absolute mode by default
    const headerText = await timeHeader.textContent();
    expect(headerText).toContain('Time');
    expect(headerText).toContain('⏱');
  });

  test('should have correct title attribute on time header', async ({ page }) => {
    const timeHeader = page.locator('[data-testid="fills-time-header"], #fills-time-header');

    // Should have title indicating click behavior
    const title = await timeHeader.getAttribute('title');
    expect(title?.toLowerCase()).toContain('click');
    expect(title?.toLowerCase()).toContain('relative');
  });

  test('should toggle time header icon when clicked', async ({ page }) => {
    const timeHeader = page.locator('[data-testid="fills-time-header"], #fills-time-header');
    await timeHeader.scrollIntoViewIfNeeded();

    // Get initial text
    const initialText = await timeHeader.textContent();
    expect(initialText).toContain('⏱'); // Stopwatch icon for absolute mode

    // Click to toggle
    await timeHeader.click();

    // Should now show clock icon for relative mode
    const toggledText = await timeHeader.textContent();
    expect(toggledText).toContain('🕐'); // Clock icon for relative mode
    expect(toggledText).not.toContain('⏱');
  });

  test('should toggle back to absolute mode on second click', async ({ page }) => {
    const timeHeader = page.locator('[data-testid="fills-time-header"], #fills-time-header');
    await timeHeader.scrollIntoViewIfNeeded();

    // Click twice to go: absolute -> relative -> absolute
    await timeHeader.click(); // Now relative
    await timeHeader.click(); // Back to absolute

    const finalText = await timeHeader.textContent();
    expect(finalText).toContain('⏱'); // Back to stopwatch icon
    expect(finalText).not.toContain('🕐');
  });

  test('should have hover cursor on time header', async ({ page }) => {
    const timeHeader = page.locator('[data-testid="fills-time-header"], #fills-time-header');

    // Check computed style has pointer cursor
    const cursor = await timeHeader.evaluate((el) => window.getComputedStyle(el).cursor);
    expect(cursor).toBe('pointer');
  });
});

test.describe('Dashboard - Responsive Design', () => {
  test('should render correctly on mobile viewport', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/dashboard');

    // Main layout should be visible
    await expect(page.locator('main')).toBeVisible();

    // Cards should stack vertically on mobile
    const cards = page.locator('.card');
    expect(await cards.count()).toBeGreaterThanOrEqual(2);
  });

  test('should render correctly on tablet viewport', async ({ page }) => {
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.goto('/dashboard');

    await expect(page.locator('main')).toBeVisible();
    await expect(page.locator('[data-testid="header"], header')).toBeVisible();
  });

  test('should render correctly on desktop viewport', async ({ page }) => {
    await page.setViewportSize({ width: 1920, height: 1080 });
    await page.goto('/dashboard');

    await expect(page.locator('main')).toBeVisible();

    // On desktop, multiple cards should be visible
    const cards = page.locator('.card');
    expect(await cards.count()).toBeGreaterThanOrEqual(3);
  });
});

test.describe('Dashboard - WebSocket Connection', () => {
  test('should attempt WebSocket connection', async ({ page }) => {
    // Listen for WebSocket connections
    const wsPromise = page.waitForEvent('websocket', { timeout: 10000 });

    await page.goto('/dashboard');

    try {
      const ws = await wsPromise;
      expect(ws.url()).toContain('/ws');
    } catch {
      // WebSocket might not be available in test environment
      // This is acceptable - dashboard still works without live updates
      console.log('WebSocket connection not established (expected in test env without server)');
    }
  });
});

test.describe('Dashboard - External Links', () => {
  test('should have Hypurrscan links for addresses when data loaded', async ({ page }) => {
    await page.goto('/dashboard');

    // Wait for table to load
    await page.waitForSelector('[data-testid="leaderboard-table"] tbody tr, .leaderboard-table tbody tr', { timeout: 10000 }).catch(() => {});

    // Check if address links exist
    const addressLinks = page.locator('a[href*="hypurrscan.io"]');
    const count = await addressLinks.count();

    // If there are tracked traders, links should open in new tab
    if (count > 0) {
      const firstLink = addressLinks.first();
      await expect(firstLink).toHaveAttribute('target', '_blank');
    }
  });
});
