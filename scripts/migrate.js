/* Simple Postgres migration runner for hlbot */
const fs = require('fs');
const path = require('path');
const { Client } = require('pg');

function getDatabaseUrl() {
  if (process.env.DATABASE_URL) return process.env.DATABASE_URL;
  const host = process.env.PGHOST || 'localhost';
  const port = process.env.PGPORT || '5432';
  const user = process.env.PGUSER || process.env.POSTGRES_USER || 'postgres';
  const password = process.env.PGPASSWORD || process.env.POSTGRES_PASSWORD || '';
  const database = process.env.PGDATABASE || process.env.POSTGRES_DB || 'postgres';
  let auth = user;
  if (password) auth += `:${encodeURIComponent(password)}`;
  return `postgresql://${auth}@${host}:${port}/${database}`;
}

function getMigrationsDir() {
  // Default directory moved from root 'migrations' to 'scripts/migrations'
  // Set MIGRATIONS_DIR to override if needed.
  return process.env.MIGRATIONS_DIR
    || path.resolve(process.cwd(), 'scripts', 'migrations');
}

function getMigrationsList(dir) {
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir)
    .filter(f => f.toLowerCase().endsWith('.sql'))
    .sort(); // lexicographic order (001_..., 002_...)
}

async function ensureMigrationsTable(client) {
  await client.query(`
    CREATE TABLE IF NOT EXISTS schema_migrations (
      version TEXT PRIMARY KEY,
      applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
  `);
}

async function appliedVersions(client) {
  const { rows } = await client.query('SELECT version FROM schema_migrations ORDER BY version ASC');
  return new Set(rows.map(r => r.version));
}

async function applyMigration(client, version, sql) {
  await client.query('BEGIN');
  try {
    await client.query(sql);
    await client.query('INSERT INTO schema_migrations(version) VALUES($1) ON CONFLICT (version) DO NOTHING', [version]);
    await client.query('COMMIT');
    console.log(`[migrate] applied ${version}`);
  } catch (e) {
    await client.query('ROLLBACK');
    throw e;
  }
}

async function up() {
  const databaseUrl = getDatabaseUrl();
  const client = new Client({ connectionString: databaseUrl });
  await client.connect();
  try {
    await ensureMigrationsTable(client);
    const dir = getMigrationsDir();
    const files = getMigrationsList(dir);
    const applied = await appliedVersions(client);

    for (const f of files) {
      const version = f; // store full filename as version key
      if (applied.has(version)) continue;
      const sql = fs.readFileSync(path.join(dir, f), 'utf8');
      await applyMigration(client, version, sql);
    }

    console.log('[migrate] up to date');
  } finally {
    await client.end();
  }
}

async function status() {
  const databaseUrl = getDatabaseUrl();
  const client = new Client({ connectionString: databaseUrl });
  await client.connect();
  try {
    await ensureMigrationsTable(client);
    const dir = getMigrationsDir();
    const files = getMigrationsList(dir);
    const applied = await appliedVersions(client);
    const pending = files.filter(f => !applied.has(f));
    console.log(JSON.stringify({ applied: Array.from(applied), pending }, null, 2));
  } finally {
    await client.end();
  }
}

const cmd = process.argv[2] || 'up';
if (cmd === 'up') {
  up().catch((e) => { console.error('[migrate] failed', e); process.exit(1); });
} else if (cmd === 'status') {
  status().catch((e) => { console.error('[migrate] failed', e); process.exit(1); });
} else {
  console.error('Usage: node scripts/migrate.js [up|status]');
  process.exit(2);
}

