#!/bin/bash
# =============================================================================
# Alpha Pool Initialization Script
# =============================================================================
# Run this after a fresh docker compose up to populate the Alpha Pool
# with historical data for Phase 3f FDR qualification.
#
# Usage:
#   ./scripts/init-alpha-pool.sh
#   ./scripts/init-alpha-pool.sh --limit 100    # More traders
#   ./scripts/init-alpha-pool.sh --delay 1000   # Slower (safer rate limit)
# =============================================================================

set -e

# Default values
LIMIT=${LIMIT:-50}
DELAY_MS=${DELAY_MS:-500}
SAGE_URL=${SAGE_URL:-http://localhost:4103}

# Parse command line args
while [[ $# -gt 0 ]]; do
    case $1 in
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --delay)
            DELAY_MS="$2"
            shift 2
            ;;
        --url)
            SAGE_URL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--limit N] [--delay MS] [--url URL]"
            echo ""
            echo "Options:"
            echo "  --limit N    Number of traders to fetch from leaderboard (default: 50)"
            echo "  --delay MS   Delay between backfill requests in ms (default: 500)"
            echo "  --url URL    hl-sage URL (default: http://localhost:4103)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "Alpha Pool Initialization"
echo "=============================================="
echo "Sage URL: $SAGE_URL"
echo "Trader limit: $LIMIT"
echo "Backfill delay: ${DELAY_MS}ms"
echo ""

# Step 1: Wait for services to be healthy
echo "[1/5] Waiting for hl-sage to be healthy..."
for i in {1..30}; do
    if curl -sf "$SAGE_URL/healthz" > /dev/null 2>&1; then
        echo "      hl-sage is healthy!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "ERROR: hl-sage not responding after 30 seconds"
        exit 1
    fi
    echo "      Waiting... ($i/30)"
    sleep 1
done

# Step 2: Check current pool status
echo ""
echo "[2/5] Checking current Alpha Pool status..."
STATUS=$(curl -sf "$SAGE_URL/alpha-pool/status" 2>/dev/null || echo '{"error": "failed"}')
echo "      $STATUS"

# Step 3: Refresh Alpha Pool from leaderboard
echo ""
echo "[3/5] Refreshing Alpha Pool from leaderboard (limit=$LIMIT)..."
REFRESH_RESULT=$(curl -sf -X POST "$SAGE_URL/alpha-pool/refresh?limit=$LIMIT" 2>/dev/null || echo '{"error": "refresh failed"}')
echo "      $REFRESH_RESULT"

# Step 4: Backfill historical fills for all addresses
echo ""
echo "[4/5] Backfilling historical fills for all addresses..."
echo "      This may take several minutes depending on the number of addresses."
echo "      Progress will be shown in docker logs (docker compose logs -f hl-sage)"
echo ""

BACKFILL_RESULT=$(curl -sf -X POST "$SAGE_URL/alpha-pool/backfill-all?delay_ms=$DELAY_MS" 2>/dev/null || echo '{"error": "backfill failed"}')
echo "      Result: $BACKFILL_RESULT"

# Step 5: Create initial snapshot
echo ""
echo "[5/5] Creating initial snapshot..."
SNAPSHOT_RESULT=$(curl -sf -X POST "$SAGE_URL/snapshots/create" 2>/dev/null || echo '{"error": "snapshot failed"}')
echo "      $SNAPSHOT_RESULT"

# Summary
echo ""
echo "=============================================="
echo "Initialization Complete!"
echo "=============================================="
echo ""
echo "Verify with:"
echo "  curl $SAGE_URL/alpha-pool/status"
echo "  curl $SAGE_URL/snapshots/config"
echo "  curl '$SAGE_URL/snapshots/summary'"
echo ""
echo "Check FDR-qualified traders:"
echo "  docker compose exec postgres psql -U hlbot -d hlbot -c \\"
echo "    \"SELECT address, episode_count, avg_r_net, skill_p_value, fdr_qualified"
echo "     FROM trader_snapshots WHERE snapshot_date = CURRENT_DATE AND fdr_qualified = true\""
echo ""
