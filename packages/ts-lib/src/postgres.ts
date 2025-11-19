import { Pool, type PoolConfig } from 'pg';

let pool: Pool | null = null;
let poolConnectionString: string | undefined = undefined;

export async function getPool(config?: PoolConfig): Promise<Pool> {
  const connectionString =
    config?.connectionString ||
    process.env.PG_CONNECTION_STRING ||
    process.env.DATABASE_URL ||
    `postgresql://${process.env.POSTGRES_USER || 'postgres'}:${process.env.POSTGRES_PASSWORD || ''}@${process.env.PGHOST || 'localhost'}:${process.env.PGPORT || '5432'}/${process.env.PGDATABASE || 'postgres'}`;

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

export async function closePool(): Promise<void> {
  if (pool) {
    await pool.end();
    pool = null;
    poolConnectionString = undefined;
  }
}
