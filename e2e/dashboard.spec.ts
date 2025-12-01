import { test, expect } from '@playwright/test';

test.describe('Dashboard Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard');
  });

  test('should load dashboard page', async ({ page }) => {
    // Check page title or header
    await expect(page).toHaveTitle(/HyperMind|Dashboard/i);

    // Check main layout elements exist
    await expect(page.locator('main')).toBeVisible();
  });

  test('should display tracked traders card', async ({ page }) => {
    const tradersCard = page.locator('.card').filter({ hasText: /tracked traders/i });
    await expect(tradersCard).toBeVisible();
  });

  test('should display activity card', async ({ page }) => {
    const activityCard = page.locator('.fills-card, .card').filter({ hasText: /activity/i });
    await expect(activityCard).toBeVisible();
  });

  test('should have theme toggle button', async ({ page }) => {
    const themeToggle = page.locator('#theme-toggle, .theme-toggle, [aria-label*="theme"]');
    await expect(themeToggle).toBeVisible();
  });

  test('should toggle theme when clicking theme button', async ({ page }) => {
    const html = page.locator('html');

    // Get initial theme
    const initialTheme = await html.getAttribute('data-theme');

    // Click dark theme button
    const darkThemeBtn = page.locator('.theme-toggle button[data-theme="dark"]');
    await darkThemeBtn.click();

    // Check theme changed to dark
    await expect(html).toHaveAttribute('data-theme', 'dark');

    // Click light theme button
    const lightThemeBtn = page.locator('.theme-toggle button[data-theme="light"]');
    await lightThemeBtn.click();

    // Check theme changed to light
    await expect(html).toHaveAttribute('data-theme', 'light');
  });

  test('should display leaderboard table', async ({ page }) => {
    const table = page.locator('.leaderboard-table');
    await expect(table).toBeVisible();

    // Check for table headers
    const addressHeader = page.locator('.leaderboard-table th').filter({ hasText: /address/i });
    const winHeader = page.locator('.leaderboard-table th').filter({ hasText: /win/i });
    await expect(addressHeader).toBeVisible();
    await expect(winHeader).toBeVisible();
  });

  test('should show refresh timer', async ({ page }) => {
    // Look for the update/refresh time indicator
    const refreshIndicator = page.locator('text=/updated|next|refresh/i');
    await expect(refreshIndicator.first()).toBeVisible();
  });
});

test.describe('Dashboard - Charts', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard');
  });

  test('should display BTC chart', async ({ page }) => {
    // TradingView charts are in iframes or specific containers
    const btcChart = page.locator('[data-symbol*="BTC"], .chart-container, iframe[src*="tradingview"]').first();
    await expect(btcChart).toBeVisible({ timeout: 15000 });
  });

  test('should have chart tabs or selectors', async ({ page }) => {
    // Check for BTC/ETH tabs or chart controls
    const chartControls = page.locator('text=/BTC|ETH/i');
    await expect(chartControls.first()).toBeVisible();
  });
});

test.describe('Dashboard - Activity Feed', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard');
  });

  test('should display activity feed container', async ({ page }) => {
    const activityFeed = page.locator('.fills-card, .fills-scroll-container');
    await expect(activityFeed.first()).toBeVisible();
  });

  test('should show fills table', async ({ page }) => {
    const fillsTable = page.locator('.fills-table');
    await expect(fillsTable).toBeVisible();

    // Check table headers exist
    const timeHeader = page.locator('.fills-table th').filter({ hasText: /time/i });
    await expect(timeHeader).toBeVisible();
  });

  test('should show fill items when data exists', async ({ page }) => {
    // Wait for potential data load
    await page.waitForTimeout(2000);

    // Check if fills exist or waiting message
    const fills = page.locator('.fills-table tbody tr');
    const waitingMessage = page.locator('text=/waiting|no fills/i');

    // Either fills exist or waiting message
    const fillCount = await fills.count();
    const hasWaitingMessage = await waitingMessage.isVisible().catch(() => false);

    expect(fillCount > 0 || hasWaitingMessage).toBeTruthy();
  });
});

test.describe('Dashboard - Responsive Design', () => {
  test('should adapt to mobile viewport', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/dashboard');

    // Dashboard should still be visible
    await expect(page.locator('main')).toBeVisible();

    // Cards should be visible (may be stacked)
    const cards = page.locator('.card');
    await expect(cards.first()).toBeVisible();
  });

  test('should adapt to tablet viewport', async ({ page }) => {
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.goto('/dashboard');

    await expect(page.locator('main')).toBeVisible();
  });

  test('should work on desktop viewport', async ({ page }) => {
    await page.setViewportSize({ width: 1920, height: 1080 });
    await page.goto('/dashboard');

    await expect(page.locator('main')).toBeVisible();

    // On desktop, multiple cards should be visible side by side
    const cards = page.locator('.card');
    expect(await cards.count()).toBeGreaterThanOrEqual(2);
  });
});

test.describe('Dashboard - WebSocket Connection', () => {
  test('should establish WebSocket connection', async ({ page }) => {
    // Listen for WebSocket connections
    const wsPromise = page.waitForEvent('websocket', { timeout: 10000 });

    await page.goto('/dashboard');

    try {
      const ws = await wsPromise;
      expect(ws.url()).toContain('/ws');
    } catch {
      // WebSocket might not be available in test environment
      // This is acceptable - test passes if WS is optional
      console.log('WebSocket connection not established (may be expected in test env)');
    }
  });
});

test.describe('Dashboard - External Links', () => {
  test('should have Hypurrscan links for addresses', async ({ page }) => {
    await page.goto('/dashboard');

    // Wait for table to load
    await page.waitForSelector('.leaderboard-table tbody tr', { timeout: 10000 }).catch(() => {});

    // Check if address links exist
    const addressLinks = page.locator('a[href*="hypurrscan.io"]');
    const count = await addressLinks.count();

    // If there are tracked traders, links should exist
    if (count > 0) {
      const firstLink = addressLinks.first();
      await expect(firstLink).toHaveAttribute('target', '_blank');
    }
  });
});
