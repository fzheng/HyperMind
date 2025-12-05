/**
 * NATS Messaging Module
 *
 * Provides helpers for connecting to NATS JetStream and publishing/subscribing
 * to JSON-encoded messages. Used for inter-service communication in the
 * event-driven architecture.
 *
 * @module nats
 */

import { RetentionPolicy, connect, JSONCodec, type JetStreamClient, type JetStreamManager, type NatsConnection, type Subscription } from 'nats';

/** Shared JSON codec for encoding/decoding message payloads */
const jsonCodec = JSONCodec();

/**
 * Bundle of NATS connection objects returned by connectNats.
 */
export interface NatsBundle {
  /** Core NATS connection */
  nc: NatsConnection;
  /** JetStream client for durable messaging */
  js: JetStreamClient;
  /** JetStream manager for stream administration */
  jsm: JetStreamManager;
}

/**
 * Establishes a connection to NATS server with JetStream support.
 *
 * @param url - NATS server URL (e.g., 'nats://0.0.0.0:4222')
 * @returns Bundle containing connection, JetStream client, and manager
 *
 * @example
 * ```typescript
 * const { nc, js, jsm } = await connectNats('nats://0.0.0.0:4222');
 * ```
 */
export async function connectNats(url: string): Promise<NatsBundle> {
  const nc = await connect({ servers: url });
  const js = nc.jetstream();
  const jsm = await nc.jetstreamManager();
  return { nc, js, jsm };
}

/**
 * Ensures a JetStream stream exists, creating it if necessary.
 * Idempotent: does nothing if the stream already exists.
 *
 * @param jsm - JetStream manager
 * @param name - Stream name
 * @param subjects - Array of subject patterns to capture (e.g., ['a.candidates.v1'])
 *
 * @example
 * ```typescript
 * await ensureStream(jsm, 'CANDIDATES', ['a.candidates.v1']);
 * ```
 */
export async function ensureStream(
  jsm: JetStreamManager,
  name: string,
  subjects: string[]
): Promise<void> {
  try {
    await jsm.streams.info(name);
  } catch {
    await jsm.streams.add({ name, subjects, retention: RetentionPolicy.Limits, max_age: 0 });
  }
}

/**
 * Publishes a JSON-encoded message to a JetStream subject.
 *
 * @param js - JetStream client
 * @param subject - Subject to publish to (e.g., 'a.candidates.v1')
 * @param payload - Object to JSON-encode and publish
 *
 * @example
 * ```typescript
 * await publishJson(js, 'a.candidates.v1', { address: '0x...', score: 0.95 });
 * ```
 */
export async function publishJson(
  js: JetStreamClient,
  subject: string,
  payload: unknown
): Promise<void> {
  await js.publish(subject, jsonCodec.encode(payload));
}

/**
 * Subscribes to a subject and invokes a handler for each JSON message.
 * Automatically acknowledges messages on success, naks on handler failure.
 *
 * @param nc - NATS connection
 * @param subject - Subject pattern to subscribe to
 * @param handler - Async function called for each message
 * @returns Subscription object (can be drained/unsubscribed)
 *
 * @example
 * ```typescript
 * const sub = subscribeJson<CandidateEvent>(nc, 'a.candidates.v1', async (data) => {
 *   console.log('Received candidate:', data.address);
 * });
 *
 * // Later: await sub.drain();
 * ```
 */
export function subscribeJson<T>(
  nc: NatsConnection,
  subject: string,
  handler: (data: T) => Promise<void> | void
): Subscription {
  const sub = nc.subscribe(subject);
  (async () => {
    for await (const msg of sub) {
      try {
        const data = jsonCodec.decode(msg.data) as T;
        await handler(data);
        if (typeof (msg as any).ack === 'function') {
          (msg as any).ack();
        }
      } catch (err) {
        // eslint-disable-next-line no-console
        console.error('[nats] handler failed', err);
        if (typeof (msg as any).nak === 'function') {
          (msg as any).nak();
        }
      }
    }
  })().catch((err) => console.error('[nats] subscription failed', err));
  return sub;
}
