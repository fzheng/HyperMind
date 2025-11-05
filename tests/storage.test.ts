describe('storage abstraction (memory backend for tests)', () => {
  test('init, add/list/remove addresses', async () => {
    process.env.STORAGE_BACKEND = 'memory';
    const storage = await import('../src/storage');
    await storage.initStorage();
    expect(await storage.listAddresses()).toEqual([]);
    await storage.addAddress('0x123');
    await storage.addAddress('0x456');
    const all = await storage.listAddresses();
    expect(all.sort()).toEqual(['0x123', '0x456']);
    await storage.removeAddress('0x123');
    expect(await storage.listAddresses()).toEqual(['0x456']);
  });
});
