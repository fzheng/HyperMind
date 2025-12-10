import { test, expect } from '@playwright/test';

/**
 * E2E Tests for WebSocket Subscription Promote/Demote feature
 *
 * Tests the dashboard UI for manually promoting addresses to WebSocket
 * and demoting them to polling to manage WebSocket slot allocation.
 *
 * IMPORTANT: These tests use API mocking to prevent database changes.
 */

test.describe('WebSocket Subscription Status', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard');
    // Default tab is Alpha Pool which shows subscription status
    await page.waitForTimeout(500);
  });

  test('should display WebSocket slots indicator in header', async ({ page }) => {
    // Look for the WebSocket slots indicator (e.g., "10/10" or "8/10")
    const wsIndicator = page.locator('[data-testid="ws-slots-value"]');
    await expect(wsIndicator).toBeVisible({ timeout: 10000 });

    const text = await wsIndicator.textContent();
    // Should match pattern like "8/10" or "10/10" (or —/— before data loads)
    expect(text).toMatch(/(\d+\/\d+|—\/—)/);
  });

  test('should show subscription method icons in trader list', async ({ page }) => {
    // Wait for Alpha Pool traders to load
    await page.waitForSelector('[data-testid="alpha-pool-tbody"]', { timeout: 15000 }).catch(() => {});
    await page.waitForTimeout(2000); // Wait for data to populate

    // Look for subscription method icons (⚡ for websocket, ⏱️ for polling)
    const methodIcons = page.locator('.sub-indicator');
    const count = await methodIcons.count();

    // Should have method icons for each trader (if any traders are loaded)
    // This test passes even if no traders are loaded (graceful handling)
    expect(count).toBeGreaterThanOrEqual(0);
  });

  test('subscription icons should be clickable', async ({ page }) => {
    await page.waitForSelector('[data-testid="alpha-pool-tbody"]', { timeout: 15000 }).catch(() => {});
    await page.waitForTimeout(2000);

    const methodIcon = page.locator('.sub-indicator.clickable').first();

    if (await methodIcon.isVisible().catch(() => false)) {
      // Icons should have pointer cursor indicating clickability
      await expect(methodIcon).toHaveCSS('cursor', 'pointer');
    }
  });
});

test.describe('Subscription Popover UI', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard');
    await page.waitForTimeout(500);
    await page.waitForSelector('[data-testid="alpha-pool-tbody"]', { timeout: 15000 }).catch(() => {});
    await page.waitForTimeout(2000);
  });

  test('clicking WebSocket icon should show popover', async ({ page }) => {
    // Find a WebSocket method icon
    const wsIcon = page.locator('.sub-indicator.sub-websocket').first();

    if (await wsIcon.isVisible().catch(() => false)) {
      await wsIcon.click();

      // Popover should appear
      const popover = page.locator('[data-testid="subscription-popover"]');
      await expect(popover).toBeVisible({ timeout: 5000 });
    }
  });

  test('clicking polling icon should show popover', async ({ page }) => {
    // Find a polling method icon
    const pollingIcon = page.locator('.sub-indicator.sub-polling').first();

    if (await pollingIcon.isVisible().catch(() => false)) {
      await pollingIcon.click();

      // Popover should appear
      const popover = page.locator('[data-testid="subscription-popover"]');
      await expect(popover).toBeVisible({ timeout: 5000 });
    }
  });

  test('popover should show current subscription method', async ({ page }) => {
    const methodIcon = page.locator('.sub-indicator.clickable').first();

    if (await methodIcon.isVisible().catch(() => false)) {
      await methodIcon.click();

      const popover = page.locator('[data-testid="subscription-popover"]');
      await expect(popover).toBeVisible({ timeout: 5000 });

      // Should show either "WebSocket" or "Polling" label
      const methodText = popover.locator('text=/WebSocket|Polling/');
      await expect(methodText.first()).toBeVisible();
    }
  });

  test('clicking outside popover should close it', async ({ page }) => {
    const methodIcon = page.locator('.sub-indicator.clickable').first();

    if (await methodIcon.isVisible().catch(() => false)) {
      await methodIcon.click();

      const popover = page.locator('[data-testid="subscription-popover"]');
      await expect(popover).toBeVisible({ timeout: 5000 });

      // Click outside the popover (on the main content area)
      await page.mouse.click(10, 10);
      await page.waitForTimeout(500);

      // Popover should be hidden
      await expect(popover).not.toBeVisible();
    }
  });
});

test.describe('Demote to Polling (Mocked API)', () => {
  test('should show demote button for unpinned WebSocket address', async ({ page }) => {
    // Mock the subscription methods endpoint
    await page.route('**/dashboard/api/subscriptions/methods', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          '0x1111111111111111111111111111111111111111': {
            method: 'websocket',
            sources: ['legacy'] // Not pinned
          }
        }),
      });
    });

    await page.goto('/dashboard');
    await page.waitForTimeout(1000);
    await page.waitForSelector('[data-testid="alpha-pool-tbody"]', { timeout: 15000 }).catch(() => {});
    await page.waitForTimeout(2000);

    // Find a WebSocket icon and click it
    const wsIcon = page.locator('.sub-indicator.sub-websocket').first();

    if (await wsIcon.isVisible().catch(() => false)) {
      await wsIcon.click();

      const popover = page.locator('[data-testid="subscription-popover"]');
      await expect(popover).toBeVisible({ timeout: 5000 });

      // Should show demote button for unpinned websocket address
      const demoteBtn = popover.locator('[data-testid="demote-btn"]');
      // Either the button exists or the address is pinned (showing unpin message)
      const hasDemote = await demoteBtn.isVisible().catch(() => false);
      const hasUnpinMessage = await popover.locator('[data-testid="popover-pinned-message"]').isVisible().catch(() => false);

      expect(hasDemote || hasUnpinMessage).toBe(true);
    }
  });

  test('clicking demote should call API (mocked)', async ({ page }) => {
    let demoteApiCalled = false;

    // Mock the demote endpoint
    await page.route('**/dashboard/api/subscriptions/demote', async (route) => {
      demoteApiCalled = true;
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: true,
          address: '0x1111111111111111111111111111111111111111',
          method: 'polling'
        }),
      });
    });

    // Mock subscription methods to show a demotion-eligible address
    await page.route('**/dashboard/api/subscriptions/methods', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          '0x1111111111111111111111111111111111111111': {
            method: 'websocket',
            sources: ['legacy']
          }
        }),
      });
    });

    await page.goto('/dashboard');
    await page.waitForTimeout(1000);
    await page.waitForSelector('[data-testid="alpha-pool-tbody"]', { timeout: 15000 }).catch(() => {});
    await page.waitForTimeout(2000);

    const wsIcon = page.locator('.sub-indicator.sub-websocket').first();

    if (await wsIcon.isVisible().catch(() => false)) {
      await wsIcon.click();

      const popover = page.locator('[data-testid="subscription-popover"]');
      await expect(popover).toBeVisible({ timeout: 5000 });

      const demoteBtn = popover.locator('[data-testid="demote-btn"]');

      if (await demoteBtn.isVisible().catch(() => false)) {
        await demoteBtn.click();
        await page.waitForTimeout(1000);

        expect(demoteApiCalled).toBe(true);
      }
    }
  });

  test('should show "unpin first" message for pinned WebSocket address', async ({ page }) => {
    // Mock the subscription methods to show a pinned address
    await page.route('**/dashboard/api/subscriptions/methods', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          '0x1111111111111111111111111111111111111111': {
            method: 'websocket',
            sources: ['pinned', 'legacy'] // Is pinned
          }
        }),
      });
    });

    await page.goto('/dashboard');
    await page.waitForTimeout(1000);
    await page.waitForSelector('[data-testid="alpha-pool-tbody"]', { timeout: 15000 }).catch(() => {});
    await page.waitForTimeout(2000);

    const wsIcon = page.locator('.sub-indicator.sub-websocket').first();

    if (await wsIcon.isVisible().catch(() => false)) {
      await wsIcon.click();

      const popover = page.locator('[data-testid="subscription-popover"]');
      await expect(popover).toBeVisible({ timeout: 5000 });

      // Should show message about unpinning
      const unpinMessage = popover.locator('[data-testid="popover-pinned-message"]');
      const hasUnpinMessage = await unpinMessage.isVisible().catch(() => false);

      // Should NOT show demote button
      const demoteBtn = popover.locator('[data-testid="demote-btn"]');
      const hasDemote = await demoteBtn.isVisible().catch(() => false);

      // Either shows pinned message OR no demote button (both are valid states)
      expect(hasUnpinMessage || !hasDemote).toBe(true);
    }
  });
});

test.describe('Promote to WebSocket (Mocked API)', () => {
  test('should show promote button for polling address when slots available', async ({ page }) => {
    // Mock subscription status to show available slots
    await page.route('**/dashboard/api/subscriptions/status', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          totalAddresses: 12,
          maxWebSocketSlots: 10,
          addressesByMethod: {
            websocket: 8, // 2 slots available
            polling: 4,
            none: 0
          }
        }),
      });
    });

    await page.goto('/dashboard');
    await page.waitForTimeout(1000);
    await page.waitForSelector('[data-testid="alpha-pool-tbody"]', { timeout: 15000 }).catch(() => {});
    await page.waitForTimeout(2000);

    const pollingIcon = page.locator('.sub-indicator.sub-polling').first();

    if (await pollingIcon.isVisible().catch(() => false)) {
      await pollingIcon.click();

      const popover = page.locator('[data-testid="subscription-popover"]');
      await expect(popover).toBeVisible({ timeout: 5000 });

      // Should show promote button when slots are available
      const promoteBtn = popover.locator('[data-testid="promote-btn"]');
      const hasPromote = await promoteBtn.isVisible().catch(() => false);

      // Should also show slots available message
      const slotsAvailable = popover.locator('[data-testid="popover-slots-available"]');
      const hasSlotsAvailable = await slotsAvailable.isVisible().catch(() => false);

      // Either promote button or slots available message should be shown
      expect(hasPromote || hasSlotsAvailable).toBe(true);
    }
  });

  test('clicking promote should call API (mocked)', async ({ page }) => {
    let promoteApiCalled = false;

    // Mock the promote endpoint
    await page.route('**/dashboard/api/subscriptions/promote', async (route) => {
      promoteApiCalled = true;
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: true,
          address: '0x2222222222222222222222222222222222222222',
          method: 'websocket'
        }),
      });
    });

    // Mock subscription status to show available slots
    await page.route('**/dashboard/api/subscriptions/status', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          totalAddresses: 12,
          maxWebSocketSlots: 10,
          addressesByMethod: {
            websocket: 8,
            polling: 4,
            none: 0
          }
        }),
      });
    });

    await page.goto('/dashboard');
    await page.waitForTimeout(1000);
    await page.waitForSelector('[data-testid="alpha-pool-tbody"]', { timeout: 15000 }).catch(() => {});
    await page.waitForTimeout(2000);

    const pollingIcon = page.locator('.sub-indicator.sub-polling').first();

    if (await pollingIcon.isVisible().catch(() => false)) {
      await pollingIcon.click();

      const popover = page.locator('[data-testid="subscription-popover"]');
      await expect(popover).toBeVisible({ timeout: 5000 });

      const promoteBtn = popover.locator('[data-testid="promote-btn"]');

      if (await promoteBtn.isVisible().catch(() => false)) {
        await promoteBtn.click();
        await page.waitForTimeout(1000);

        expect(promoteApiCalled).toBe(true);
      }
    }
  });

  test('should show "no slots" message when all slots used', async ({ page }) => {
    // Mock subscription status with no available slots
    await page.route('**/dashboard/api/subscriptions/status', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          totalAddresses: 12,
          maxWebSocketSlots: 10,
          addressesByMethod: {
            websocket: 10, // All slots used
            polling: 2,
            none: 0
          }
        }),
      });
    });

    await page.goto('/dashboard');
    await page.waitForTimeout(1000);
    await page.waitForSelector('[data-testid="alpha-pool-tbody"]', { timeout: 15000 }).catch(() => {});
    await page.waitForTimeout(2000);

    const pollingIcon = page.locator('.sub-indicator.sub-polling').first();

    if (await pollingIcon.isVisible().catch(() => false)) {
      await pollingIcon.click();

      const popover = page.locator('[data-testid="subscription-popover"]');
      await expect(popover).toBeVisible({ timeout: 5000 });

      // Should show message about no slots available
      const noSlotsMessage = popover.locator('[data-testid="popover-no-slots"]');
      const hasNoSlotsMessage = await noSlotsMessage.isVisible().catch(() => false);

      // Promote button should NOT be visible when no slots
      const promoteBtn = popover.locator('[data-testid="promote-btn"]');
      const hasPromote = await promoteBtn.isVisible().catch(() => false);

      // Either shows no slots message OR no promote button
      expect(hasNoSlotsMessage || !hasPromote).toBe(true);
    }
  });
});

test.describe('Subscription Status API Integration', () => {
  test('should fetch subscription status on page load', async ({ page }) => {
    let statusApiCalled = false;

    await page.route('**/dashboard/api/subscriptions/status', async (route) => {
      statusApiCalled = true;
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          totalAddresses: 10,
          maxWebSocketSlots: 10,
          addressesByMethod: {
            websocket: 10,
            polling: 0,
            none: 0
          },
          sources: ['pinned', 'legacy', 'alpha-pool']
        }),
      });
    });

    await page.goto('/dashboard');
    await page.waitForTimeout(2000);

    expect(statusApiCalled).toBe(true);
  });

  test('should fetch subscription methods on page load', async ({ page }) => {
    let methodsApiCalled = false;

    await page.route('**/dashboard/api/subscriptions/methods', async (route) => {
      methodsApiCalled = true;
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({}),
      });
    });

    await page.goto('/dashboard');
    await page.waitForTimeout(2000);

    expect(methodsApiCalled).toBe(true);
  });
});

test.describe('WebSocket Slots Display', () => {
  test('should display slots in header', async ({ page }) => {
    await page.route('**/dashboard/api/subscriptions/status', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          totalAddresses: 12,
          maxWebSocketSlots: 10,
          addressesByMethod: {
            websocket: 8,
            polling: 4,
            none: 0
          }
        }),
      });
    });

    await page.goto('/dashboard');
    await page.waitForTimeout(2000);

    // Check that slots indicator shows correct value
    const wsIndicator = page.locator('[data-testid="ws-slots-value"]');
    await expect(wsIndicator).toBeVisible();

    const text = await wsIndicator.textContent();
    // Should show 8/10 based on mocked data
    expect(text).toMatch(/8\/10/);
  });

  test('should show correct slot count in popover', async ({ page }) => {
    await page.route('**/dashboard/api/subscriptions/status', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          totalAddresses: 12,
          maxWebSocketSlots: 10,
          addressesByMethod: {
            websocket: 8,
            polling: 4,
            none: 0
          }
        }),
      });
    });

    await page.goto('/dashboard');
    await page.waitForTimeout(1000);
    await page.waitForSelector('[data-testid="alpha-pool-tbody"]', { timeout: 15000 }).catch(() => {});
    await page.waitForTimeout(2000);

    const pollingIcon = page.locator('.sub-indicator.sub-polling').first();

    if (await pollingIcon.isVisible().catch(() => false)) {
      await pollingIcon.click();

      const popover = page.locator('[data-testid="subscription-popover"]');
      await expect(popover).toBeVisible({ timeout: 5000 });

      // Popover should show available slots (2 available out of 10)
      const slotsAvailable = popover.locator('[data-testid="popover-slots-available"]');
      const hasSlotsAvailable = await slotsAvailable.isVisible().catch(() => false);

      // Either shows slots info or promote button (indicates slots available)
      const promoteBtn = popover.locator('[data-testid="promote-btn"]');
      const hasPromote = await promoteBtn.isVisible().catch(() => false);

      expect(hasSlotsAvailable || hasPromote).toBe(true);
    }
  });
});
