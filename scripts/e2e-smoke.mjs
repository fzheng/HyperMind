import { connect, JSONCodec } from 'nats';

const OWNER_TOKEN = process.env.OWNER_TOKEN || 'dev-owner';
const BASE_URL = process.env.SCOUT_URL || 'http://0.0.0.0:4101';
const NATS_URL = process.env.NATS_URL || 'nats://0.0.0.0:4222';

async function seed() {
  const addresses = [
    `0x${Math.random().toString(16).slice(2, 42).padEnd(40, '1')}`,
    `0x${Math.random().toString(16).slice(2, 42).padEnd(40, '2')}`,
    `0x${Math.random().toString(16).slice(2, 42).padEnd(40, '3')}`,
  ];
  const res = await fetch(`${BASE_URL}/admin/seed`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-owner-key': OWNER_TOKEN,
    },
    body: JSON.stringify({ addresses }),
  });
  if (!res.ok) {
    throw new Error(`Seed failed ${res.status}`);
  }
  return addresses;
}

async function main() {
  console.log('[smoke] seeding candidates');
  await seed();
  const nc = await connect({ servers: NATS_URL });
  const jc = JSONCodec();
  const order = ['a.candidates.v1', 'b.scores.v1', 'c.fills.v1', 'd.signals.v1', 'd.outcomes.v1'];
  const seen = new Set();

  for (const subject of order) {
    const sub = nc.subscribe(subject);
    (async () => {
      for await (const msg of sub) {
        if (!seen.has(subject)) {
          seen.add(subject);
          try {
            const data = jc.decode(msg.data);
            console.log(`[smoke] ${subject} ->`, data);
          } catch {
            console.log(`[smoke] ${subject} received`);
          }
        }
        if (subject === 'd.outcomes.v1' && seen.size === order.length) {
          console.log('[smoke] flow complete');
          await nc.drain();
          process.exit(0);
        }
      }
    })();
  }

  // timeout
  setTimeout(async () => {
    console.error('[smoke] timed out waiting for outcomes');
    await nc.drain();
    process.exit(1);
  }, Number(process.env.SMOKE_TIMEOUT || 60000));
}

main().catch((err) => {
  console.error('[smoke] failed', err);
  process.exit(1);
});
