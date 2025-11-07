-- 006_add_hl_events_indexes.sql
-- Add composite indexes to accelerate time-based trade queries.
-- Global chronological trade index and per-address chronological index.

CREATE INDEX IF NOT EXISTS hl_events_trade_at_desc_idx
ON hl_events (at DESC) WHERE type = 'trade';

CREATE INDEX IF NOT EXISTS hl_events_trade_address_at_desc_idx
ON hl_events (address, at DESC) WHERE type = 'trade';
