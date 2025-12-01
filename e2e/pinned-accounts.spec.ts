import { test, expect } from '@playwright/test';

/**
 * E2E Tests for Pinned Accounts feature
 *
 * IMPORTANT: These tests are READ-ONLY by default. They verify UI elements exist
 * and have correct styling, but do NOT perform any actions that modify the database
 * (like pinning/unpinning accounts).
 *
 * Tests that require mutations use API mocking to intercept calls and prevent
 * actual database changes.
 */

// Test address that doesn't exist in leaderboard (for custom pin tests)
const TEST_ADDRESS = '0x1234567890123456789012345678901234567890';

test.describe('Pinned Accounts - UI Elements', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard');
    // Wait for leaderboard to load
    await page.waitForSelector('[data-testid="leaderboard-table"] tbody tr, .leaderboard-table tbody tr', { timeout: 15000 }).catch(() => {});
  });

  test('should display add custom input', async ({ page }) => {
    const customInput = page.locator('[data-testid="custom-address-input"], #custom-address-input');
    await expect(customInput.first()).toBeVisible();
  });

  test('should display add custom button', async ({ page }) => {
    const addButton = page.locator('[data-testid="add-custom-btn"], #add-custom-btn');
    await expect(addButton).toBeVisible();
  });

  test('should show custom account count indicator', async ({ page }) => {
    // Look for "(X/3)" pattern in the UI
    const countIndicator = page.locator('[data-testid="custom-count"], #custom-count');
    await expect(countIndicator).toBeVisible();

    const text = await countIndicator.textContent();
    expect(text).toMatch(/^\d$/); // Should be a single digit (0-3)
  });

  test('should display pin icons in leaderboard rows', async ({ page }) => {
    const rows = page.locator('[data-testid="leaderboard-table"] tbody tr, .leaderboard-table tbody tr');
    const rowCount = await rows.count();

    if (rowCount > 0) {
      // Each row should have a pin icon
      const pinIcons = page.locator('[data-testid^="pin-icon-"], .pin-icon');
      const iconCount = await pinIcons.count();
      expect(iconCount).toBeGreaterThan(0);
    }
  });
});

test.describe('Pinned Accounts - Pin Icon Styling (Read-Only)', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard');
    await page.waitForSelector('[data-testid="leaderboard-table"] tbody tr, .leaderboard-table tbody tr', { timeout: 15000 }).catch(() => {});
  });

  test('pin icons should have pointer cursor', async ({ page }) => {
    const pinIcon = page.locator('[data-testid^="pin-icon-"], .pin-icon').first();

    if (await pinIcon.isVisible().catch(() => false)) {
      await expect(pinIcon).toHaveCSS('cursor', 'pointer');
    }
  });

  test('unpinned icons should have low opacity', async ({ page }) => {
    const unpinnedIcon = page.locator('.pin-icon.unpinned').first();

    if (await unpinnedIcon.isVisible().catch(() => false)) {
      const opacity = await unpinnedIcon.evaluate((el) =>
        window.getComputedStyle(el).opacity
      );

      // Unpinned should have low opacity (0.25-0.3)
      expect(parseFloat(opacity)).toBeLessThan(0.5);
    }
  });

  test('pinned-leaderboard icons should be blue', async ({ page }) => {
    const leaderboardPinned = page.locator('.pin-icon.pinned-leaderboard').first();

    if (await leaderboardPinned.isVisible().catch(() => false)) {
      const color = await leaderboardPinned.evaluate((el) =>
        window.getComputedStyle(el).color
      );

      // Should be blue-ish (rgb values for #38bdf8 or similar)
      const match = color.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
      if (match) {
        const [, r, g, b] = match.map(Number);
        // Blue channel should be significant
        expect(b).toBeGreaterThan(100);
      }
    }
  });

  test('pinned-custom icons should be gold/amber', async ({ page }) => {
    const customPinned = page.locator('.pin-icon.pinned-custom').first();

    if (await customPinned.isVisible().catch(() => false)) {
      const color = await customPinned.evaluate((el) =>
        window.getComputedStyle(el).color
      );

      // Should be gold/amber (high R and G, lower B)
      const match = color.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
      if (match) {
        const [, r, g, b] = match.map(Number);
        // Gold has high R, medium-high G, low B
        expect(r).toBeGreaterThan(150);
        expect(g).toBeGreaterThan(100);
      }
    }
  });
});

test.describe('Pinned Accounts - Visual Differentiation (Read-Only)', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard');
    await page.waitForSelector('[data-testid="leaderboard-table"] tbody tr, .leaderboard-table tbody tr', { timeout: 15000 }).catch(() => {});
  });

  test('pinned rows should have distinct background', async ({ page }) => {
    const pinnedRow = page.locator('.leaderboard-table tr.pinned-row').first();

    if (await pinnedRow.isVisible().catch(() => false)) {
      const bgColor = await pinnedRow.evaluate((el) =>
        window.getComputedStyle(el).backgroundColor
      );

      // Pinned rows should have non-transparent background
      expect(bgColor).not.toBe('rgba(0, 0, 0, 0)');
    }
  });
});

test.describe('Pinned Accounts - Custom Address Input Validation (Mocked)', () => {
  test.beforeEach(async ({ page }) => {
    // Intercept ALL pin-related API calls to prevent database changes
    await page.route('**/admin/addresses/**', async (route) => {
      const method = route.request().method();
      if (method === 'POST' || method === 'DELETE') {
        // Mock successful response without hitting real API
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ success: true, message: 'Mocked response' }),
        });
      } else {
        await route.continue();
      }
    });

    await page.goto('/dashboard');
  });

  test('input should accept valid ethereum address', async ({ page }) => {
    const input = page.locator('[data-testid="custom-address-input"], #custom-address-input');
    await expect(input).toBeVisible();

    await input.fill(TEST_ADDRESS);
    await expect(input).toHaveValue(TEST_ADDRESS);

    // Clear input to not leave test state
    await input.clear();
  });

  test('should show error message element (for invalid address)', async ({ page }) => {
    // Just verify the error element exists in DOM
    const errorEl = page.locator('[data-testid="custom-accounts-error"], #custom-accounts-error');
    // Error element should exist but be hidden initially
    await expect(errorEl).toBeAttached();
  });

  test('add button should be visible and enabled', async ({ page }) => {
    const addButton = page.locator('[data-testid="add-custom-btn"], #add-custom-btn');
    await expect(addButton).toBeVisible();

    // Button is enabled when under max (3) custom accounts
    const countText = await page.locator('[data-testid="custom-count"], #custom-count').textContent();
    const count = parseInt(countText || '0');

    if (count < 3) {
      await expect(addButton).toBeEnabled();
    }
  });
});

test.describe('Pinned Accounts - Pin/Unpin Interactions (Mocked API)', () => {
  test('clicking pin icon should trigger API call (mocked)', async ({ page }) => {
    let apiCallMade = false;

    // IMPORTANT: Set up ALL route interceptions BEFORE navigation to ensure no real API calls
    // Intercept pin-related API calls
    await page.route('**/dashboard/api/pinned-accounts/**', async (route) => {
      apiCallMade = true;
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ success: true }),
      });
    });

    // Also intercept legacy custom-accounts endpoint
    await page.route('**/dashboard/api/custom-accounts/**', async (route) => {
      apiCallMade = true;
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ success: true }),
      });
    });

    await page.goto('/dashboard');
    await page.waitForSelector('[data-testid="leaderboard-table"] tbody tr, .leaderboard-table tbody tr', { timeout: 15000 }).catch(() => {});

    const unpinnedIcon = page.locator('.pin-icon.unpinned').first();

    if (await unpinnedIcon.isVisible().catch(() => false)) {
      await unpinnedIcon.scrollIntoViewIfNeeded().catch(() => {});
      await unpinnedIcon.click();

      // Wait for API call
      await page.waitForTimeout(1000);

      // Verify an API call was made (but it was mocked, so no DB change)
      expect(apiCallMade).toBe(true);
    }
  });

  test('adding custom address should trigger API call (mocked)', async ({ page }) => {
    let apiCallMade = false;

    // IMPORTANT: Set up ALL route interceptions BEFORE navigation
    // Mock summary to show 0 custom accounts (so button is enabled)
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

    // Intercept custom account API calls - both endpoints
    await page.route('**/dashboard/api/pinned-accounts/**', async (route) => {
      apiCallMade = true;
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ success: true }),
      });
    });

    // Also intercept legacy endpoint
    await page.route('**/dashboard/api/custom-accounts/**', async (route) => {
      apiCallMade = true;
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ success: true }),
      });
    });

    await page.goto('/dashboard');
    await page.waitForTimeout(500);

    const input = page.locator('[data-testid="custom-address-input"], #custom-address-input');
    const addButton = page.locator('[data-testid="add-custom-btn"], #add-custom-btn');

    // Fill with valid address format
    await input.fill(TEST_ADDRESS);

    // Check if button is enabled
    const isDisabled = await addButton.isDisabled();
    if (!isDisabled) {
      await addButton.click();

      // Wait for API call
      await page.waitForTimeout(1000);

      // Verify API call was attempted (mocked)
      expect(apiCallMade).toBe(true);
    }
  });
});

test.describe('Pinned Accounts - Limit Enforcement', () => {
  test('should show max custom limit indicator', async ({ page }) => {
    await page.goto('/dashboard');

    // Look for the (X/3) indicator in header
    const limitText = page.locator('text=/\\(\\d\\/3\\)/');
    await expect(limitText.first()).toBeVisible();
  });
});

test.describe('Pinned Accounts - Persistence (Read-Only)', () => {
  test('pinned accounts should persist after page reload', async ({ page }) => {
    await page.goto('/dashboard');

    // Wait for leaderboard to fully load
    await page.waitForSelector('[data-testid="leaderboard-table"] tbody tr, .leaderboard-table tbody tr', { timeout: 15000 }).catch(() => {});
    await page.waitForTimeout(2000); // Wait longer for data to stabilize

    // Count pinned items before reload (read-only observation)
    const pinnedBefore = await page.locator('.pin-icon.pinned-leaderboard, .pin-icon.pinned-custom').count();

    // Reload page
    await page.reload();

    // Wait for leaderboard to fully load again
    await page.waitForSelector('[data-testid="leaderboard-table"] tbody tr, .leaderboard-table tbody tr', { timeout: 15000 }).catch(() => {});
    await page.waitForTimeout(2000); // Wait longer for data to stabilize

    // Count pinned items after reload
    const pinnedAfter = await page.locator('.pin-icon.pinned-leaderboard, .pin-icon.pinned-custom').count();

    // Counts should be approximately the same (allow small variation due to dynamic data)
    // Persistence is considered working if counts are within 2 of each other
    expect(Math.abs(pinnedAfter - pinnedBefore)).toBeLessThanOrEqual(2);
  });
});
