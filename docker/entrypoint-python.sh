#!/bin/sh
set -e

# Wait for PostgreSQL to be ready AND migrations to complete
# This is necessary for cloud deployments where services start in parallel

echo "[entrypoint] waiting for database schema..."
MAX_RETRIES=60
RETRY_COUNT=0

# Check for a table that's created by migrations (position_signals from 014)
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
  if python3 -c "
import asyncio
import asyncpg
import os

async def check():
    try:
        conn = await asyncpg.connect(os.environ.get('DATABASE_URL', 'postgresql://hlbot:hlbotpassword@postgres:5432/hlbot'))
        # Check if position_signals table exists (created in migration 014)
        result = await conn.fetchval(\"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'position_signals')\")
        await conn.close()
        return result
    except Exception as e:
        return False

result = asyncio.run(check())
exit(0 if result else 1)
" 2>/dev/null; then
    echo "[entrypoint] database schema is ready"
    break
  fi

  RETRY_COUNT=$((RETRY_COUNT + 1))
  echo "[entrypoint] schema not ready, retry $RETRY_COUNT/$MAX_RETRIES..."
  sleep 3
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
  echo "[entrypoint] WARNING: database schema check timed out, proceeding anyway..."
fi

echo "[entrypoint] starting app..."
exec "$@"
