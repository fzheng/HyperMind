// Increase default test timeout for any async tests if needed
jest.setTimeout(20000);
process.env.STORAGE_BACKEND = process.env.STORAGE_BACKEND || 'memory';
