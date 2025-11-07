-- Speed up de-duplication by trade tx hash
CREATE UNIQUE INDEX IF NOT EXISTS hl_events_trade_hash_uq
ON hl_events ((payload->>'hash'))
WHERE type = 'trade';
