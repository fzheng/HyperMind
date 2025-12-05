#!/bin/sh
set -e

# Wait for PostgreSQL to be ready (for Railway/cloud deployments where depends_on doesn't work)
echo "[entrypoint] waiting for PostgreSQL..."
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
  if node -e "
    const { Pool } = require('pg');
    const pool = new Pool({ connectionString: process.env.DATABASE_URL || 'postgresql://hlbot:hlbotpassword@postgres:5432/hlbot' });
    pool.query('SELECT 1').then(() => { pool.end(); process.exit(0); }).catch(() => process.exit(1));
  " 2>/dev/null; then
    echo "[entrypoint] PostgreSQL is ready"
    break
  fi

  RETRY_COUNT=$((RETRY_COUNT + 1))
  echo "[entrypoint] PostgreSQL not ready, retry $RETRY_COUNT/$MAX_RETRIES..."
  sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
  echo "[entrypoint] ERROR: PostgreSQL not available after $MAX_RETRIES retries"
  exit 1
fi

echo "[entrypoint] running migrations..."
node /app/scripts/migrate.js up

echo "[entrypoint] starting app..."
exec "$@"
