import { fetchWithRetry, clamp } from '../src/utils';

describe('utils', () => {
  const originalFetch = global.fetch as any;

  afterEach(() => {
    global.fetch = originalFetch;
    jest.clearAllMocks();
  });

  test('clamp limits values correctly', () => {
    expect(clamp(5, 0, 10)).toBe(5);
    expect(clamp(-1, 0, 10)).toBe(0);
    expect(clamp(42, 0, 10)).toBe(10);
  });

  test('fetchWithRetry retries and succeeds', async () => {
    let calls = 0;
    global.fetch = jest.fn(async () => {
      calls += 1;
      if (calls === 1) {
        throw new Error('network fail');
      }
      return { ok: true, json: async () => ({ ok: true }) } as any;
    }) as any;

    const start = Date.now();
    const res = await fetchWithRetry<any>('http://example.com', { method: 'GET' }, { retries: 1, baseDelayMs: 10 });
    const elapsed = Date.now() - start;
    expect(res.ok).toBe(true);
    expect(calls).toBe(2);
    expect(elapsed).toBeGreaterThanOrEqual(10);
  });
});

