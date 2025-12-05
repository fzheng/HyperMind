/**
 * PostgreSQL Connection Pool Module
 *
 * Provides a singleton connection pool for PostgreSQL database access.
 * Automatically constructs connection strings from environment variables.
 *
 * @module postgres
 */

import { Pool, type PoolConfig } from 'pg';

/** Singleton pool instance */
let pool: Pool | null = null;
/** Connection string used for the current pool */
let poolConnectionString: string | undefined = undefined;

/**
 * Gets or creates the PostgreSQL connection pool.
 * Uses a singleton pattern to reuse connections across the application.
 *
 * Connection string resolution order:
 * 1. config.connectionString (if provided)
 * 2. PG_CONNECTION_STRING environment variable
 * 3. DATABASE_URL environment variable
 * 4. Constructed from POSTGRES_USER, POSTGRES_PASSWORD, PGHOST, PGPORT, PGDATABASE
 *
 * @param config - Optional pool configuration to override defaults
 * @returns Promise resolving to the connection pool
 *
 * @example
 * ```typescript
 * const pool = await getPool();
 * const { rows } = await pool.query('SELECT * FROM users');
 * ```
 */
export async function getPool(config?: PoolConfig): Promise<Pool> {
  const connectionString =
    config?.connectionString ||
    process.env.PG_CONNECTION_STRING ||
    process.env.DATABASE_URL ||
    `postgresql://${process.env.POSTGRES_USER || 'postgres'}:${process.env.POSTGRES_PASSWORD || ''}@${process.env.PGHOST || '0.0.0.0'}:${process.env.PGPORT || '5432'}/${process.env.PGDATABASE || 'postgres'}`;

  if (pool) {
    // Warn if trying to use different connection string
    if (poolConnectionString && poolConnectionString !== connectionString) {
      console.warn('[postgres] Pool already initialized with different connection string. Returning existing pool.');
    }
    return pool;
  }

  poolConnectionString = connectionString;
  pool = new Pool({ connectionString, ...config });
  return pool;
}

/**
 * Closes the connection pool and releases all connections.
 * Should be called during graceful shutdown.
 *
 * @example
 * ```typescript
 * process.on('SIGTERM', async () => {
 *   await closePool();
 *   process.exit(0);
 * });
 * ```
 */
export async function closePool(): Promise<void> {
  if (pool) {
    await pool.end();
    pool = null;
    poolConnectionString = undefined;
  }
}
