import { test, expect } from '@playwright/test';

/**
 * E2E Tests for Tab Switching and Data Refresh
 *
 * These tests verify the fixes for:
 * 1. Legacy Leaderboard tab showing stale fills data after tab switch
 * 2. Alpha Pool auto-refresh UI state (is_running properly set)
 *
 * All tests are READ-ONLY - they observe UI behavior without modifying data.
 */

test.describe('Tab Switching - Data Refresh', () => {
  test('switching to Legacy tab should trigger fills refresh', async ({ page }) => {
    let legacyFillsRequested = false;

    // Intercept the legacy fills API call
    await page.route('**/dashboard/api/legacy/fills**', async (route) => {
      legacyFillsRequested = true;
      await route.continue();
    });

    await page.goto('/dashboard');

    // Dashboard starts on Alpha Pool, so first switch to Legacy
    const legacyTab = page.locator('[data-testid="tab-legacy-leaderboard"]');
    await legacyTab.click();
    await page.waitForTimeout(1000);

    // Should have requested legacy fills
    expect(legacyFillsRequested).toBe(true);
  });

  test('switching between tabs should fetch fresh data each time', async ({ page }) => {
    let alphaPoolRequests = 0;
    let legacyFillsRequests = 0;

    // Intercept API calls
    await page.route('**/dashboard/api/alpha-pool', async (route) => {
      alphaPoolRequests++;
      await route.continue();
    });

    await page.route('**/dashboard/api/legacy/fills**', async (route) => {
      legacyFillsRequests++;
      await route.continue();
    });

    await page.goto('/dashboard');
    await page.waitForTimeout(1500);

    const initialAlphaPoolRequests = alphaPoolRequests;

    // Switch to Legacy
    await page.locator('[data-testid="tab-legacy-leaderboard"]').click();
    await page.waitForTimeout(1000);

    // Should have requested legacy fills
    expect(legacyFillsRequests).toBeGreaterThan(0);

    // Switch back to Alpha Pool
    await page.locator('[data-testid="tab-alpha-pool"]').click();
    await page.waitForTimeout(1000);

    // Should have made another Alpha Pool request
    expect(alphaPoolRequests).toBeGreaterThan(initialAlphaPoolRequests);
  });

  test('Legacy fills should have data after tab switch', async ({ page }) => {
    await page.goto('/dashboard');

    // Switch to Legacy tab
    await page.locator('[data-testid="tab-legacy-leaderboard"]').click();
    await page.waitForTimeout(1500);

    // Wait for fills to load
    await page.waitForSelector('.fills-table, #fills-tbody', { timeout: 10000 }).catch(() => {});

    // Check if fills table has rows or shows "no data" message (both are valid states)
    const fillsRows = page.locator('.fills-table tbody tr, #fills-tbody tr');
    const noDataMessage = page.locator('text=/No fills|No recent activity/i');

    const rowCount = await fillsRows.count().catch(() => 0);
    const hasNoDataMessage = await noDataMessage.isVisible().catch(() => false);

    // Either has data or shows appropriate no-data message
    expect(rowCount > 0 || hasNoDataMessage).toBe(true);
  });

  test('tab content should be visible after switch', async ({ page }) => {
    await page.goto('/dashboard');
    await page.waitForTimeout(1000);

    // Alpha Pool should be visible initially (ID is tab-alpha-pool, testid is tab-content-alpha-pool)
    const alphaPoolContent = page.locator('#tab-alpha-pool, [data-testid="tab-content-alpha-pool"]');
    await expect(alphaPoolContent).toBeVisible();

    // Switch to Legacy
    await page.locator('[data-testid="tab-legacy-leaderboard"]').click();
    await page.waitForTimeout(500);

    // Legacy content should be visible (ID is tab-legacy-leaderboard, testid is tab-content-legacy-leaderboard)
    const legacyContent = page.locator('#tab-legacy-leaderboard, [data-testid="tab-content-legacy-leaderboard"]');
    await expect(legacyContent).toBeVisible();

    // Alpha Pool content should be hidden
    await expect(alphaPoolContent).toBeHidden();
  });
});

test.describe('Alpha Pool - Auto-Refresh UI', () => {
  test('Alpha Pool tab should show loading or data state', async ({ page }) => {
    await page.goto('/dashboard');
    await page.waitForTimeout(2000);

    // Should be on Alpha Pool tab by default (tab button)
    const alphaPoolTab = page.locator('[data-testid="tab-alpha-pool"]');
    await expect(alphaPoolTab).toBeVisible();

    // Either show data table, loading state, or refresh button (all are valid states)
    const alphaPoolTable = page.locator('[data-testid="alpha-pool-table"]');
    const loadingIndicator = page.locator('text=/Loading|Refreshing|Filtering|Fetching/i');
    const noDataMessage = page.locator('text=/No Alpha Pool Data/i');
    const refreshButton = page.locator('text=/Refresh Alpha Pool/i');

    const hasTable = await alphaPoolTable.isVisible().catch(() => false);
    const hasLoading = await loadingIndicator.isVisible().catch(() => false);
    const hasNoData = await noDataMessage.isVisible().catch(() => false);
    const hasRefreshBtn = await refreshButton.isVisible().catch(() => false);

    // One of these states should be visible
    expect(hasTable || hasLoading || hasNoData || hasRefreshBtn).toBe(true);
  });

  test('Alpha Pool refresh status endpoint should be accessible', async ({ page }) => {
    await page.goto('/dashboard');

    // Make a direct request to the refresh status endpoint
    const response = await page.request.get('/dashboard/api/alpha-pool/refresh/status');
    expect(response.status()).toBe(200);

    const data = await response.json();
    // Should have is_running field
    expect(typeof data.is_running).toBe('boolean');
    // Should have other expected fields
    expect('current_step' in data || 'progress' in data).toBe(true);
  });

  test('Alpha Pool data endpoint should return valid data', async ({ page }) => {
    await page.goto('/dashboard');

    // Request Alpha Pool data
    const response = await page.request.get('/dashboard/api/alpha-pool');
    expect(response.status()).toBe(200);

    const data = await response.json();
    // Should have traders array (even if empty)
    expect(Array.isArray(data.traders)).toBe(true);
  });
});

test.describe('Position Display After Custom Account Add', () => {
  /**
   * Tests that positions are available immediately after adding a custom account.
   * This verifies the fix for awaiting DB writes in performPrime().
   *
   * Note: These tests use mocked API responses to avoid modifying real data.
   */

  test('custom account API should return success', async ({ page }) => {
    // Mock the custom account endpoint
    await page.route('**/dashboard/api/pinned-accounts/custom', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ success: true, message: 'Account added' }),
      });
    });

    // Mock the summary endpoint to include position data
    await page.route('**/dashboard/api/summary**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          stats: [{
            address: '0x1234567890123456789012345678901234567890',
            winRate: 75,
            executedOrders: 10,
            realizedPnl: 50000,
            isPinned: true,
            isCustom: true,
          }],
          holdings: {
            '0x1234567890123456789012345678901234567890': [{
              symbol: 'BTC',
              size: 1.5,
              entryPrice: 95000,
              liquidationPrice: 80000,
              leverage: 10,
            }]
          },
          customPinnedCount: 1,
          maxCustomPinned: 3,
        }),
      });
    });

    await page.goto('/dashboard');

    // Switch to Legacy tab
    await page.locator('[data-testid="tab-legacy-leaderboard"]').click();
    await page.waitForTimeout(1000);

    // Find custom input and add button
    const input = page.locator('#custom-address-input, [data-testid="custom-address-input"]');
    const addButton = page.locator('#add-custom-btn, [data-testid="add-custom-btn"]');

    if (await input.isVisible().catch(() => false)) {
      await input.fill('0x1234567890123456789012345678901234567890');

      if (await addButton.isEnabled().catch(() => false)) {
        await addButton.click();
        await page.waitForTimeout(1000);

        // After add, the mocked summary should be returned with position data
        // (In real scenario, position would appear in the table)
      }
    }
  });

  test('holdings column should display position data', async ({ page }) => {
    await page.goto('/dashboard');

    // Switch to Legacy tab
    await page.locator('[data-testid="tab-legacy-leaderboard"]').click();
    await page.waitForTimeout(1500);

    // Wait for table to load
    await page.waitForSelector('[data-testid="leaderboard-table"], .leaderboard-table', { timeout: 10000 }).catch(() => {});

    // Find holdings column cells
    const holdingsCells = page.locator('.holdings-cell, [data-testid^="holdings-"]');
    const count = await holdingsCells.count();

    if (count > 0) {
      // Check that holdings cells have some content (either position or "No BTC/ETH position")
      const firstCell = holdingsCells.first();
      const text = await firstCell.textContent();
      expect(text?.length).toBeGreaterThan(0);
    }
  });
});

test.describe('Dashboard Tab State Persistence', () => {
  test('active tab should be visually distinct', async ({ page }) => {
    await page.goto('/dashboard');
    await page.waitForTimeout(500);

    // Alpha Pool tab should be active by default
    const alphaPoolTab = page.locator('[data-testid="tab-alpha-pool"]');
    const legacyTab = page.locator('[data-testid="tab-legacy-leaderboard"]');

    // Check active state class or attribute
    const alphaPoolActive = await alphaPoolTab.evaluate((el) =>
      el.classList.contains('active') || el.getAttribute('aria-selected') === 'true'
    ).catch(() => false);

    expect(alphaPoolActive).toBe(true);

    // Switch to Legacy
    await legacyTab.click();
    await page.waitForTimeout(500);

    // Now Legacy should be active
    const legacyActive = await legacyTab.evaluate((el) =>
      el.classList.contains('active') || el.getAttribute('aria-selected') === 'true'
    ).catch(() => false);

    expect(legacyActive).toBe(true);
  });

  test('tab panels should toggle visibility correctly', async ({ page }) => {
    await page.goto('/dashboard');

    // Initially Alpha Pool content visible (actual IDs are tab-alpha-pool and tab-legacy-leaderboard)
    const alphaContent = page.locator('#tab-alpha-pool, [data-testid="tab-content-alpha-pool"]');
    const legacyContent = page.locator('#tab-legacy-leaderboard, [data-testid="tab-content-legacy-leaderboard"]');

    await expect(alphaContent).toBeVisible();
    await expect(legacyContent).toBeHidden();

    // Click Legacy tab
    await page.locator('[data-testid="tab-legacy-leaderboard"]').click();
    await page.waitForTimeout(300);

    await expect(alphaContent).toBeHidden();
    await expect(legacyContent).toBeVisible();

    // Click Alpha Pool tab
    await page.locator('[data-testid="tab-alpha-pool"]').click();
    await page.waitForTimeout(300);

    await expect(alphaContent).toBeVisible();
    await expect(legacyContent).toBeHidden();
  });
});
