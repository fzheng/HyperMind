export function getEnv(name: string, fallback?: string): string | undefined {
  const value = process.env[name];
  if (value == null || value === '') return fallback;
  return value;
}

export function requireEnv(name: string, fallback?: string): string {
  const value = getEnv(name, fallback);
  if (!value) {
    throw new Error(`Missing required env var ${name}`);
  }
  return value;
}

export function getPort(defaultPort = 8080): number {
  const raw = Number(getEnv('PORT'));
  if (Number.isFinite(raw) && raw > 0) return raw;
  return defaultPort;
}

/** Cached flag to only warn once per process */
let ownerTokenWarned = false;

export function getOwnerToken(): string {
  const token = requireEnv('OWNER_TOKEN', 'dev-owner');
  if (token === 'dev-owner') {
    // Fail-fast in production - don't allow default token
    if (process.env.NODE_ENV === 'production') {
      throw new Error(
        'OWNER_TOKEN must be explicitly set in production. ' +
        'Default "dev-owner" token is not allowed.'
      );
    }
    // Warn in development (once per process)
    if (!ownerTokenWarned) {
      ownerTokenWarned = true;
      console.warn(
        '[SECURITY WARNING] Using default OWNER_TOKEN "dev-owner". ' +
        'All owner endpoints are publicly writable. ' +
        'Set OWNER_TOKEN environment variable in production.'
      );
    }
  }
  return token;
}
