-- Durable event log for realtime changes
CREATE TABLE IF NOT EXISTS hl_events (
  id BIGSERIAL PRIMARY KEY,
  at TIMESTAMPTZ NOT NULL DEFAULT now(),
  address TEXT NOT NULL,
  type TEXT NOT NULL, -- 'position' | 'trade'
  symbol TEXT NOT NULL,
  payload JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS hl_events_at_idx ON hl_events (at DESC);
CREATE INDEX IF NOT EXISTS hl_events_type_at_idx ON hl_events (type, at DESC);
CREATE INDEX IF NOT EXISTS hl_events_addr_at_idx ON hl_events (address, at DESC);
