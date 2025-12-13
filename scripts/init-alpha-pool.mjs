#!/usr/bin/env node
/**
 * Alpha Pool Initialization Script (Cross-platform Node.js version)
 *
 * Run this after a fresh docker compose up to populate the Alpha Pool
 * with historical data for Phase 3f FDR qualification.
 *
 * Usage:
 *   npm run init:alpha-pool
 *   node scripts/init-alpha-pool.mjs
 *   node scripts/init-alpha-pool.mjs --limit 100 --delay 1000
 */

const SAGE_URL = process.env.SAGE_URL || 'http://localhost:4103';
const DEFAULT_LIMIT = 50;
const DEFAULT_DELAY_MS = 500;

// Parse command line args
function parseArgs() {
  const args = process.argv.slice(2);
  const options = {
    limit: DEFAULT_LIMIT,
    delayMs: DEFAULT_DELAY_MS,
    sageUrl: SAGE_URL,
  };

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--limit':
        options.limit = parseInt(args[++i], 10);
        break;
      case '--delay':
        options.delayMs = parseInt(args[++i], 10);
        break;
      case '--url':
        options.sageUrl = args[++i];
        break;
      case '-h':
      case '--help':
        console.log('Usage: node scripts/init-alpha-pool.mjs [options]');
        console.log('');
        console.log('Options:');
        console.log('  --limit N    Number of traders to fetch from leaderboard (default: 50)');
        console.log('  --delay MS   Delay between backfill requests in ms (default: 500)');
        console.log('  --url URL    hl-sage URL (default: http://localhost:4103)');
        process.exit(0);
    }
  }

  return options;
}

async function fetchJson(url, method = 'GET') {
  const response = await fetch(url, { method });
  const text = await response.text();
  try {
    return JSON.parse(text);
  } catch {
    return { raw: text };
  }
}

async function waitForHealth(url, maxRetries = 30) {
  for (let i = 1; i <= maxRetries; i++) {
    try {
      const response = await fetch(`${url}/healthz`);
      if (response.ok) {
        return true;
      }
    } catch {
      // Ignore connection errors during startup
    }
    console.log(`      Waiting... (${i}/${maxRetries})`);
    await new Promise(r => setTimeout(r, 1000));
  }
  return false;
}

async function main() {
  const options = parseArgs();

  console.log('==============================================');
  console.log('Alpha Pool Initialization');
  console.log('==============================================');
  console.log(`Sage URL: ${options.sageUrl}`);
  console.log(`Trader limit: ${options.limit}`);
  console.log(`Backfill delay: ${options.delayMs}ms`);
  console.log('');

  // Step 1: Wait for services to be healthy
  console.log('[1/5] Waiting for hl-sage to be healthy...');
  const healthy = await waitForHealth(options.sageUrl);
  if (!healthy) {
    console.error('ERROR: hl-sage not responding after 30 seconds');
    process.exit(1);
  }
  console.log('      hl-sage is healthy!');

  // Step 2: Check current pool status
  console.log('');
  console.log('[2/5] Checking current Alpha Pool status...');
  try {
    const status = await fetchJson(`${options.sageUrl}/alpha-pool/status`);
    console.log('      ' + JSON.stringify(status));
  } catch (e) {
    console.log('      Failed to get status:', e.message);
  }

  // Step 3: Refresh Alpha Pool from leaderboard
  console.log('');
  console.log(`[3/5] Refreshing Alpha Pool from leaderboard (limit=${options.limit})...`);
  try {
    const refreshResult = await fetchJson(
      `${options.sageUrl}/alpha-pool/refresh?limit=${options.limit}`,
      'POST'
    );
    console.log('      ' + JSON.stringify(refreshResult));
  } catch (e) {
    console.log('      Refresh failed:', e.message);
  }

  // Step 4: Backfill historical fills for all addresses
  console.log('');
  console.log('[4/5] Backfilling historical fills for all addresses...');
  console.log('      This may take several minutes depending on the number of addresses.');
  console.log('      Progress will be shown in docker logs (docker compose logs -f hl-sage)');
  console.log('');
  try {
    const backfillResult = await fetchJson(
      `${options.sageUrl}/alpha-pool/backfill-all?delay_ms=${options.delayMs}`,
      'POST'
    );
    console.log('      Result: ' + JSON.stringify(backfillResult));
  } catch (e) {
    console.log('      Backfill failed:', e.message);
  }

  // Step 5: Create initial snapshot
  console.log('');
  console.log('[5/5] Creating initial snapshot...');
  try {
    const snapshotResult = await fetchJson(`${options.sageUrl}/snapshots/create`, 'POST');
    console.log('      ' + JSON.stringify(snapshotResult));
  } catch (e) {
    console.log('      Snapshot failed:', e.message);
  }

  // Summary
  console.log('');
  console.log('==============================================');
  console.log('Initialization Complete!');
  console.log('==============================================');
  console.log('');
  console.log('Verify with:');
  console.log(`  curl ${options.sageUrl}/alpha-pool/status`);
  console.log(`  curl ${options.sageUrl}/snapshots/config`);
  console.log(`  curl '${options.sageUrl}/snapshots/summary'`);
  console.log('');
  console.log('Check FDR-qualified traders:');
  console.log('  docker compose exec postgres psql -U hlbot -d hlbot -c "');
  console.log('    SELECT address, episode_count, avg_r_net, skill_p_value, fdr_qualified');
  console.log('    FROM trader_snapshots WHERE snapshot_date = CURRENT_DATE AND fdr_qualified = true"');
  console.log('');
}

main().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});
