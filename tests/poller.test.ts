import { Poller } from '../src/poller';

jest.mock('../src/price', () => ({
  fetchBtcPriceUsd: jest.fn().mockResolvedValue({ symbol: 'BTCUSD', price: 50000 }),
}));

jest.mock('../src/hyperliquid', () => ({
  fetchBtcPerpExposure: jest.fn().mockResolvedValue(2),
}));

describe('Poller', () => {
  test('trigger computes recommendations for addresses', async () => {
    const addresses = ['0xaaa', '0xbbb'];
    let recs: any[] = [];
    const poller = new Poller(
      async () => addresses,
      (r) => {
        recs = r;
      },
      () => recs,
      { intervalMs: 60_000 }
    );

    await poller.trigger();
    expect(recs.length).toBe(2);
    expect(recs[0].address).toBe('0xaaa');
    expect(typeof recs[0].score).toBe('number');
    expect(recs[0].priceUsd).toBe(50000);
  });
});
