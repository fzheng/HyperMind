import type { Address } from './types';

type Backend = 'redis' | 'postgres' | 'memory';

interface Storage {
  init(): Promise<void>;
  listAddresses(): Promise<Address[]>;
  addAddress(address: Address): Promise<void>;
  removeAddress(address: Address): Promise<void>;
}

// Memory backend for dev/tests
class MemoryStorage implements Storage {
  private set = new Set<Address>();
  async init(): Promise<void> {/* noop */}
  async listAddresses(): Promise<Address[]> { return Array.from(this.set); }
  async addAddress(address: Address): Promise<void> { this.set.add(address.toLowerCase()); }
  async removeAddress(address: Address): Promise<void> { this.set.delete(address.toLowerCase()); }
}

// Redis backend
class RedisStorage implements Storage {
  private client: any;
  private key = 'hlbot:addresses';
  constructor(url?: string) {
    // Lazy import to avoid requiring package when unused
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { createClient } = require('redis');
    this.client = createClient({ url: url || process.env.REDIS_URL });
    this.client.on('error', (err: any) => console.error('[storage][redis] error', err));
  }
  async init(): Promise<void> { await this.client.connect(); }
  async listAddresses(): Promise<Address[]> { return (await this.client.sMembers(this.key)) as Address[]; }
  async addAddress(address: Address): Promise<void> { await this.client.sAdd(this.key, address.toLowerCase()); }
  async removeAddress(address: Address): Promise<void> { await this.client.sRem(this.key, address.toLowerCase()); }
}

// Postgres backend
class PostgresStorage implements Storage {
  private pool: any;
  private table = 'addresses';
  constructor(connString?: string) {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { Pool } = require('pg');
    const connectionString = connString || process.env.PG_CONNECTION_STRING || process.env.DATABASE_URL;
    this.pool = new Pool({ connectionString });
  }
  async init(): Promise<void> {
    await this.pool.query(`create table if not exists ${this.table} (
      address text primary key
    )`);
  }
  async listAddresses(): Promise<Address[]> {
    const { rows } = await this.pool.query(`select address from ${this.table}`);
    return rows.map((r: any) => r.address as Address);
  }
  async addAddress(address: Address): Promise<void> {
    await this.pool.query(`insert into ${this.table} (address) values ($1) on conflict (address) do nothing`, [address.toLowerCase()]);
  }
  async removeAddress(address: Address): Promise<void> {
    await this.pool.query(`delete from ${this.table} where address = $1`, [address.toLowerCase()]);
  }
}

let backend: Backend = (process.env.STORAGE_BACKEND as Backend) || 'memory';

function selectStorage(): Storage {
  // Auto-detect if not explicitly set
  if (!process.env.STORAGE_BACKEND) {
    if (process.env.REDIS_URL) backend = 'redis';
    else if (process.env.PG_CONNECTION_STRING || process.env.DATABASE_URL || process.env.PGHOST) backend = 'postgres';
  }
  if (backend === 'redis') return new RedisStorage(process.env.REDIS_URL);
  if (backend === 'postgres') return new PostgresStorage(process.env.PG_CONNECTION_STRING || process.env.DATABASE_URL);
  return new MemoryStorage();
}

const storage: Storage = selectStorage();

export async function initStorage() { await storage.init(); }
export async function listAddresses() { return storage.listAddresses(); }
export async function addAddress(address: Address) { return storage.addAddress(address); }
export async function removeAddress(address: Address) { return storage.removeAddress(address); }
