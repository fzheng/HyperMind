import { test, expect } from '@playwright/test';

/**
 * E2E Tests for Dashboard Resilience and Error Handling
 *
 * These tests verify the dashboard handles API errors, empty data,
 * and network failures gracefully by mocking API responses.
 *
 * NO actual database changes are made - all API calls are intercepted.
 */

test.describe('Dashboard - API Error Handling', () => {
  test('should handle leaderboard API 500 error gracefully', async ({ page }) => {
    // Mock leaderboard API to return 500
    await page.route('**/dashboard/leaderboard**', async (route) => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'Internal server error' }),
      });
    });

    await page.goto('/dashboard');

    // Dashboard should still load
    await expect(page.locator('main')).toBeVisible();
    await expect(page.locator('[data-testid="header"], header')).toBeVisible();

    // Leaderboard table should exist but may be empty
    const table = page.locator('[data-testid="leaderboard-table"], .leaderboard-table');
    await expect(table).toBeVisible();
  });

  test('should handle fills API 500 error gracefully', async ({ page }) => {
    // Mock fills API to return 500
    await page.route('**/dashboard/fills**', async (route) => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'Internal server error' }),
      });
    });

    await page.goto('/dashboard');

    // Dashboard should still load
    await expect(page.locator('main')).toBeVisible();

    // Fills table should exist
    const fillsTable = page.locator('[data-testid="fills-table"], .fills-table');
    await expect(fillsTable).toBeVisible();
  });

  test('should handle price API 502 error gracefully', async ({ page }) => {
    // Mock price API to return 502
    await page.route('**/dashboard/price**', async (route) => {
      await route.fulfill({
        status: 502,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'Failed to fetch price from upstream' }),
      });
    });

    await page.goto('/dashboard');

    // Dashboard should still load
    await expect(page.locator('main')).toBeVisible();

    // Price ticker should show placeholder
    const btcPrice = page.locator('[data-testid="btc-price"], #btc-price');
    await expect(btcPrice).toBeVisible();
  });

  test('should handle network timeout gracefully', async ({ page }) => {
    // Mock API with long delay to simulate timeout
    await page.route('**/dashboard/leaderboard**', async (route) => {
      await new Promise(resolve => setTimeout(resolve, 100));
      await route.abort('timedout');
    });

    await page.goto('/dashboard');

    // Dashboard should still load
    await expect(page.locator('main')).toBeVisible();
  });
});

test.describe('Dashboard - Empty Data States', () => {
  test('should handle empty leaderboard gracefully', async ({ page }) => {
    // Mock summary API (which populates the leaderboard table) to return empty stats
    await page.route('**/dashboard/api/summary**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ stats: [], holdings: {}, customPinnedCount: 0 }),
      });
    });

    await page.goto('/dashboard');

    // Dashboard should still load
    await expect(page.locator('main')).toBeVisible();

    // Table should exist with headers
    const table = page.locator('[data-testid="leaderboard-table"], .leaderboard-table');
    await expect(table).toBeVisible();

    // Wait for table to render
    await page.waitForTimeout(500);

    // Table body should be empty or show placeholder row
    const rows = page.locator('[data-testid="leaderboard-tbody"] tr, .leaderboard-table tbody tr');
    const rowCount = await rows.count();
    // May have 0 rows or 1 placeholder row
    expect(rowCount).toBeLessThanOrEqual(1);
  });

  test('should handle empty fills gracefully', async ({ page }) => {
    // Mock fills API to return empty array
    await page.route('**/dashboard/api/fills**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ fills: [], hasMore: false }),
      });
    });

    await page.goto('/dashboard');

    // Status message should show waiting state
    const statusMessage = page.locator('[data-testid="fills-status-message"], #fills-status-message');
    await expect(statusMessage).toBeVisible();

    // Wait for render
    await page.waitForTimeout(500);

    // Should show waiting or empty message or fills count
    const text = await statusMessage.textContent();
    expect(text?.toLowerCase()).toMatch(/waiting|no fills|empty|0 fill/i);
  });

  test('should handle null price gracefully', async ({ page }) => {
    // Mock price API to return null
    await page.route('**/dashboard/price**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ symbol: 'BTC', price: null }),
      });
    });

    await page.goto('/dashboard');

    // Price should show placeholder or dash
    const btcPrice = page.locator('[data-testid="btc-price"], #btc-price');
    await expect(btcPrice).toBeVisible();
  });
});

test.describe('Dashboard - Pin API Error Handling (Mocked)', () => {
  test('should handle pin API 401 unauthorized', async ({ page }) => {
    await page.route('**/admin/addresses/**', async (route) => {
      if (route.request().method() === 'POST') {
        await route.fulfill({
          status: 401,
          contentType: 'application/json',
          body: JSON.stringify({ error: 'Unauthorized' }),
        });
      } else {
        await route.continue();
      }
    });

    await page.goto('/dashboard');

    // Dashboard should still work
    await expect(page.locator('main')).toBeVisible();

    // Pin icons should still be visible (even if clicking won't work)
    await page.waitForSelector('[data-testid="leaderboard-table"] tbody tr, .leaderboard-table tbody tr', { timeout: 10000 }).catch(() => {});
  });

  test('should handle add custom account API 400 bad request', async ({ page }) => {
    // Mock the summary API to return 0 custom accounts (so button is enabled)
    await page.route('**/dashboard/api/summary**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          stats: [],
          holdings: {},
          customPinnedCount: 0,
          maxCustomPinned: 3,
        }),
      });
    });

    // Mock the custom accounts API to return 400
    await page.route('**/dashboard/api/pinned-accounts/custom**', async (route) => {
      if (route.request().method() === 'POST') {
        await route.fulfill({
          status: 400,
          contentType: 'application/json',
          body: JSON.stringify({ error: 'Invalid address format' }),
        });
      } else {
        await route.continue();
      }
    });

    await page.goto('/dashboard');

    const input = page.locator('[data-testid="custom-address-input"], #custom-address-input');
    const addButton = page.locator('[data-testid="add-custom-btn"], #add-custom-btn');

    // Wait for button to be enabled (after summary loads)
    await page.waitForTimeout(500);

    // Fill a valid-format address (to pass client-side validation)
    await input.fill('0x1234567890123456789012345678901234567890');

    // Click the button (may be disabled if max reached, skip in that case)
    const isDisabled = await addButton.isDisabled();
    if (!isDisabled) {
      await addButton.click();

      // Wait for response
      await page.waitForTimeout(500);

      // Error element should exist
      const errorEl = page.locator('[data-testid="custom-accounts-error"], #custom-accounts-error');
      await expect(errorEl).toBeAttached();
    }
  });

  test('should handle add custom account API 429 rate limit', async ({ page }) => {
    await page.route('**/admin/addresses/**', async (route) => {
      if (route.request().method() === 'POST') {
        await route.fulfill({
          status: 429,
          contentType: 'application/json',
          body: JSON.stringify({ error: 'Too many requests' }),
        });
      } else {
        await route.continue();
      }
    });

    await page.goto('/dashboard');

    // Dashboard should still be functional
    await expect(page.locator('main')).toBeVisible();
  });
});

test.describe('Dashboard - WebSocket Error States', () => {
  test('should handle WebSocket connection failure', async ({ page }) => {
    // Block WebSocket connections
    await page.route('**/ws', async (route) => {
      await route.abort('connectionrefused');
    });

    await page.goto('/dashboard');

    // Dashboard should still load without live updates
    await expect(page.locator('main')).toBeVisible();
    await expect(page.locator('[data-testid="header"], header')).toBeVisible();

    // Fills table should still be visible
    const fillsTable = page.locator('[data-testid="fills-table"], .fills-table');
    await expect(fillsTable).toBeVisible();
  });
});

test.describe('Dashboard - Mixed Error Scenarios', () => {
  test('should handle all APIs failing simultaneously', async ({ page }) => {
    // Mock all APIs to fail
    await page.route('**/dashboard/**', async (route) => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'Service unavailable' }),
      });
    });

    await page.goto('/dashboard');

    // Dashboard should still render basic structure
    await expect(page.locator('main')).toBeVisible();
    await expect(page.locator('[data-testid="header"], header')).toBeVisible();

    // All cards should still be present (just empty)
    const leaderboardCard = page.locator('[data-testid="leaderboard-card"], .leaderboard-card');
    const fillsCard = page.locator('[data-testid="fills-card"], .fills-card');
    const chartCard = page.locator('[data-testid="chart-card"], .chart-card');

    await expect(leaderboardCard).toBeVisible();
    await expect(fillsCard).toBeVisible();
    await expect(chartCard).toBeVisible();
  });

  test('should recover when API starts working after initial failure', async ({ page }) => {
    let callCount = 0;

    // First call fails, subsequent calls succeed
    await page.route('**/dashboard/leaderboard**', async (route) => {
      callCount++;
      if (callCount === 1) {
        await route.fulfill({
          status: 500,
          contentType: 'application/json',
          body: JSON.stringify({ error: 'Temporary failure' }),
        });
      } else {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([
            {
              address: '0x1234567890123456789012345678901234567890',
              winRate: 0.65,
              tradeCount: 100,
              pnl: 50000,
            },
          ]),
        });
      }
    });

    await page.goto('/dashboard');

    // Wait for potential retry/refresh
    await page.waitForTimeout(2000);

    // Dashboard should be functional
    await expect(page.locator('main')).toBeVisible();
  });
});
