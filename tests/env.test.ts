/**
 * Tests for environment variable utilities
 * Tests getOwnerToken, requireEnv, getEnv, getPort functions
 */

// We need to test the actual module, not a mock
// Clear any cached module state
beforeEach(() => {
  jest.resetModules();
});

describe('getEnv', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    process.env = { ...originalEnv };
  });

  afterAll(() => {
    process.env = originalEnv;
  });

  it('returns env value when set', () => {
    process.env.TEST_VAR = 'test-value';
    const { getEnv } = require('../packages/ts-lib/src/env');
    expect(getEnv('TEST_VAR')).toBe('test-value');
  });

  it('returns undefined when not set and no fallback', () => {
    delete process.env.UNSET_VAR;
    const { getEnv } = require('../packages/ts-lib/src/env');
    expect(getEnv('UNSET_VAR')).toBeUndefined();
  });

  it('returns fallback when not set', () => {
    delete process.env.UNSET_VAR;
    const { getEnv } = require('../packages/ts-lib/src/env');
    expect(getEnv('UNSET_VAR', 'fallback')).toBe('fallback');
  });

  it('returns fallback when empty string', () => {
    process.env.EMPTY_VAR = '';
    const { getEnv } = require('../packages/ts-lib/src/env');
    expect(getEnv('EMPTY_VAR', 'fallback')).toBe('fallback');
  });
});

describe('requireEnv', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    process.env = { ...originalEnv };
  });

  afterAll(() => {
    process.env = originalEnv;
  });

  it('returns env value when set', () => {
    process.env.REQUIRED_VAR = 'required-value';
    const { requireEnv } = require('../packages/ts-lib/src/env');
    expect(requireEnv('REQUIRED_VAR')).toBe('required-value');
  });

  it('throws when not set and no fallback', () => {
    delete process.env.UNSET_VAR;
    const { requireEnv } = require('../packages/ts-lib/src/env');
    expect(() => requireEnv('UNSET_VAR')).toThrow('Missing required env var UNSET_VAR');
  });

  it('uses fallback when not set', () => {
    delete process.env.UNSET_VAR;
    const { requireEnv } = require('../packages/ts-lib/src/env');
    expect(requireEnv('UNSET_VAR', 'fallback')).toBe('fallback');
  });
});

describe('getOwnerToken', () => {
  const originalEnv = process.env;
  const originalWarn = console.warn;

  beforeEach(() => {
    jest.resetModules();
    process.env = { ...originalEnv };
    console.warn = jest.fn();
  });

  afterAll(() => {
    process.env = originalEnv;
    console.warn = originalWarn;
  });

  it('returns custom token without warning', () => {
    process.env.OWNER_TOKEN = 'my-secure-token';
    const { getOwnerToken } = require('../packages/ts-lib/src/env');

    const token = getOwnerToken();

    expect(token).toBe('my-secure-token');
    expect(console.warn).not.toHaveBeenCalled();
  });

  it('warns once when using default dev-owner token', () => {
    delete process.env.OWNER_TOKEN;
    const { getOwnerToken } = require('../packages/ts-lib/src/env');

    // First call should warn
    const token1 = getOwnerToken();
    expect(token1).toBe('dev-owner');
    expect(console.warn).toHaveBeenCalledTimes(1);
    expect(console.warn).toHaveBeenCalledWith(expect.stringContaining('SECURITY WARNING'));

    // Second call should NOT warn again
    const token2 = getOwnerToken();
    expect(token2).toBe('dev-owner');
    expect(console.warn).toHaveBeenCalledTimes(1); // Still 1, not 2
  });

  it('does not repeat warning on subsequent calls', () => {
    delete process.env.OWNER_TOKEN;
    const { getOwnerToken } = require('../packages/ts-lib/src/env');

    // Call multiple times
    getOwnerToken();
    getOwnerToken();
    getOwnerToken();

    // Should only warn once
    expect(console.warn).toHaveBeenCalledTimes(1);
  });

  it('throws in production when using default token', () => {
    delete process.env.OWNER_TOKEN;
    process.env.NODE_ENV = 'production';
    const { getOwnerToken } = require('../packages/ts-lib/src/env');

    expect(() => getOwnerToken()).toThrow('OWNER_TOKEN must be explicitly set in production');
  });

  it('returns custom token in production without error', () => {
    process.env.OWNER_TOKEN = 'prod-secure-token';
    process.env.NODE_ENV = 'production';
    const { getOwnerToken } = require('../packages/ts-lib/src/env');

    const token = getOwnerToken();

    expect(token).toBe('prod-secure-token');
    expect(console.warn).not.toHaveBeenCalled();
  });
});

describe('getPort', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    process.env = { ...originalEnv };
  });

  afterAll(() => {
    process.env = originalEnv;
  });

  it('returns default port when PORT not set', () => {
    delete process.env.PORT;
    const { getPort } = require('../packages/ts-lib/src/env');
    expect(getPort()).toBe(8080);
    expect(getPort(3000)).toBe(3000);
  });

  it('returns PORT env value when set', () => {
    process.env.PORT = '4000';
    const { getPort } = require('../packages/ts-lib/src/env');
    expect(getPort()).toBe(4000);
  });

  it('returns default when PORT is invalid', () => {
    process.env.PORT = 'not-a-number';
    const { getPort } = require('../packages/ts-lib/src/env');
    expect(getPort(5000)).toBe(5000);
  });

  it('returns default when PORT is 0 or negative', () => {
    process.env.PORT = '0';
    const { getPort } = require('../packages/ts-lib/src/env');
    expect(getPort(5000)).toBe(5000);

    process.env.PORT = '-1';
    expect(getPort(5000)).toBe(5000);
  });
});
