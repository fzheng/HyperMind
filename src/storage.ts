import type { Address } from './types';

type Backend = 'redis' | 'postgres' | 'memory';

interface Storage {
  init(): Promise<void>;
  listAddresses(): Promise<Address[]>;
  addAddress(address: Address): Promise<void>;
  removeAddress(address: Address): Promise<void>;
  getNicknames(): Promise<Record<Address, string>>;
  setNickname(address: Address, nickname: string | null): Promise<void>;
}

// Memory backend for dev/tests
class MemoryStorage implements Storage {
  private set = new Set<Address>();
  private names = new Map<Address, string>();
  async init(): Promise<void> {/* noop */}
  async listAddresses(): Promise<Address[]> { return Array.from(this.set); }
  async addAddress(address: Address): Promise<void> { this.set.add(address.toLowerCase()); }
  async removeAddress(address: Address): Promise<void> { this.set.delete(address.toLowerCase()); this.names.delete(address.toLowerCase()); }
  async getNicknames(): Promise<Record<Address, string>> {
    const out: Record<Address, string> = {} as any;
    for (const [addr, nick] of this.names.entries()) out[addr] = nick;
    return out;
  }
  async setNickname(address: Address, nickname: string | null): Promise<void> {
    const a = address.toLowerCase();
    if (!nickname || nickname.trim() === '') this.names.delete(a);
    else this.names.set(a, nickname.trim());
  }
}

// Redis backend
class RedisStorage implements Storage {
  private client: any;
  private key = 'hlbot:addresses';
  private nickKey = 'hlbot:nicknames';
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
  async removeAddress(address: Address): Promise<void> { const a = address.toLowerCase(); await this.client.sRem(this.key, a); await this.client.hDel(this.nickKey, a); }
  async getNicknames(): Promise<Record<Address, string>> { return (await this.client.hGetAll(this.nickKey)) as Record<Address, string>; }
  async setNickname(address: Address, nickname: string | null): Promise<void> {
    const a = address.toLowerCase();
    if (!nickname || nickname.trim() === '') await this.client.hDel(this.nickKey, a);
    else await this.client.hSet(this.nickKey, a, nickname.trim());
  }
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
      address text primary key,
      nickname text
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
  async getNicknames(): Promise<Record<Address, string>> {
    const { rows } = await this.pool.query(`select address, nickname from ${this.table}`);
    const out: Record<Address, string> = {} as any;
    for (const r of rows) { if (r.nickname) out[r.address] = r.nickname; }
    return out;
  }
  async setNickname(address: Address, nickname: string | null): Promise<void> {
    await this.pool.query(`update ${this.table} set nickname = $2 where address = $1`, [address.toLowerCase(), nickname && nickname.trim() ? nickname.trim() : null]);
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
export async function getNicknames() { return storage.getNicknames(); }
export async function setNickname(address: Address, nickname: string | null) { return storage.setNickname(address, nickname); }
