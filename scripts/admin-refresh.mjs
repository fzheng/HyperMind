#!/usr/bin/env node

/**
 * Admin CLI script to trigger a leaderboard refresh
 *
 * Usage:
 *   node scripts/admin-refresh.mjs                 # Uses default SCOUT_URL
 *   node scripts/admin-refresh.mjs --url http://0.0.0.0:4101
 *   OWNER_TOKEN=mytoken node scripts/admin-refresh.mjs
 *
 * Environment variables:
 *   OWNER_TOKEN - Required for authentication
 *   SCOUT_URL   - Base URL for hl-scout service (default: http://0.0.0.0:4101)
 */

import { parseArgs } from 'node:util';

const DEFAULT_SCOUT_URL = process.env.SCOUT_URL || 'http://0.0.0.0:4101';

async function main() {
  // Parse command line arguments
  const { values } = parseArgs({
    options: {
      url: { type: 'string', short: 'u', default: DEFAULT_SCOUT_URL },
      status: { type: 'boolean', short: 's', default: false },
      help: { type: 'boolean', short: 'h', default: false },
    },
  });

  if (values.help) {
    console.log(`
Admin Leaderboard Refresh Tool

Usage:
  node scripts/admin-refresh.mjs [options]

Options:
  -u, --url <url>    Scout service URL (default: ${DEFAULT_SCOUT_URL})
  -s, --status       Only show current refresh status, don't trigger refresh
  -h, --help         Show this help message

Environment Variables:
  OWNER_TOKEN        Required for authentication (must match service config)
  SCOUT_URL          Alternative way to set service URL

Examples:
  # Trigger a refresh
  OWNER_TOKEN=mytoken node scripts/admin-refresh.mjs

  # Check current status
  OWNER_TOKEN=mytoken node scripts/admin-refresh.mjs --status

  # Use custom URL
  OWNER_TOKEN=mytoken node scripts/admin-refresh.mjs --url http://scout.example.com:4101
`);
    process.exit(0);
  }

  const ownerToken = process.env.OWNER_TOKEN;
  if (!ownerToken) {
    console.error('Error: OWNER_TOKEN environment variable is required');
    console.error('Set it with: OWNER_TOKEN=your-token node scripts/admin-refresh.mjs');
    process.exit(1);
  }

  const baseUrl = values.url.replace(/\/$/, ''); // Remove trailing slash

  // If --status flag, just show status
  if (values.status) {
    await showStatus(baseUrl, ownerToken);
    return;
  }

  // Otherwise, trigger refresh and poll status
  await triggerRefresh(baseUrl, ownerToken);
}

async function showStatus(baseUrl, ownerToken) {
  const url = `${baseUrl}/leaderboard/refresh-status`;

  try {
    const res = await fetch(url, {
      headers: { 'x-owner-key': ownerToken },
    });

    if (!res.ok) {
      console.error(`Error: HTTP ${res.status} - ${res.statusText}`);
      process.exit(1);
    }

    const data = await res.json();
    console.log('\nLeaderboard Refresh Status');
    console.log('==========================');
    console.log(`Status:           ${data.status}`);
    console.log(`Is Refreshing:    ${data.isRefreshing ? 'Yes' : 'No'}`);
    console.log(`Last Refresh:     ${data.lastRefreshAt || 'Never'}`);
    console.log(`DB Last Refresh:  ${data.dbLastRefresh || 'Unknown'}`);
    console.log(`Next Refresh:     ${data.nextRefreshAt || 'Unknown'}`);

    if (data.nextRefreshInMs !== null) {
      const mins = Math.floor(data.nextRefreshInMs / 60000);
      const hours = Math.floor(mins / 60);
      const remainingMins = mins % 60;
      console.log(`Time Until Next:  ${hours}h ${remainingMins}m`);
    }

    if (data.progress) {
      console.log(`Progress:         ${data.progress.phase}${data.progress.detail ? ` - ${data.progress.detail}` : ''}`);
    }

    if (data.error) {
      console.log(`Last Error:       ${data.error}`);
    }

    console.log(`Refresh Interval: ${formatMs(data.refreshIntervalMs)}`);
    console.log('');
  } catch (err) {
    console.error(`Error fetching status: ${err.message}`);
    process.exit(1);
  }
}

async function triggerRefresh(baseUrl, ownerToken) {
  const triggerUrl = `${baseUrl}/admin/leaderboard/trigger-refresh`;
  const statusUrl = `${baseUrl}/leaderboard/refresh-status`;

  console.log(`Triggering leaderboard refresh on ${baseUrl}...`);

  try {
    const res = await fetch(triggerUrl, {
      method: 'POST',
      headers: { 'x-owner-key': ownerToken },
    });

    if (!res.ok) {
      const text = await res.text();
      console.error(`Error: HTTP ${res.status} - ${text}`);
      process.exit(1);
    }

    const data = await res.json();

    if (data.alreadyRefreshing) {
      console.log('A refresh is already in progress.');
      console.log(`Started at: ${data.refreshStartedAt}`);
    } else {
      console.log('Refresh triggered successfully!');
    }

    console.log(`Status: ${data.status}`);
    if (data.progress) {
      console.log(`Phase: ${data.progress.phase}`);
    }

    // Poll status until refresh completes
    console.log('\nMonitoring refresh progress...');
    console.log('(Press Ctrl+C to stop monitoring)\n');

    let lastPhase = '';
    let refreshComplete = false;
    const startTime = Date.now();

    while (!refreshComplete) {
      await sleep(2000); // Poll every 2 seconds

      try {
        const statusRes = await fetch(statusUrl, {
          headers: { 'x-owner-key': ownerToken },
        });

        if (!statusRes.ok) continue;

        const status = await statusRes.json();

        if (status.progress && status.progress.phase !== lastPhase) {
          const elapsed = formatMs(Date.now() - startTime);
          console.log(`[${elapsed}] ${status.progress.phase}${status.progress.detail ? ` - ${status.progress.detail}` : ''}`);
          lastPhase = status.progress.phase;
        }

        if (!status.isRefreshing) {
          refreshComplete = true;
          const elapsed = formatMs(Date.now() - startTime);

          if (status.error) {
            console.log(`\n[${elapsed}] Refresh failed: ${status.error}`);
          } else {
            console.log(`\n[${elapsed}] Refresh completed successfully!`);
            console.log(`Last refresh: ${status.lastRefreshAt}`);
          }
        }
      } catch (err) {
        // Ignore polling errors, continue
      }
    }

  } catch (err) {
    console.error(`Error triggering refresh: ${err.message}`);
    process.exit(1);
  }
}

function formatMs(ms) {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  const mins = Math.floor(ms / 60000);
  const secs = Math.floor((ms % 60000) / 1000);
  if (mins < 60) return `${mins}m ${secs}s`;
  const hours = Math.floor(mins / 60);
  const remainingMins = mins % 60;
  return `${hours}h ${remainingMins}m`;
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

main().catch(err => {
  console.error('Fatal error:', err.message);
  process.exit(1);
});
