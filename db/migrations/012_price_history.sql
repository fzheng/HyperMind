-- Migration 012: Price history for regime detection
-- Adds indexes and columns to marks_1m for regime detection
-- Table created in 000_base_schema.sql

-- Index for efficient lookups by asset and time range
CREATE INDEX IF NOT EXISTS marks_1m_asset_ts_desc_idx
  ON marks_1m (asset, ts DESC);

-- Index for finding recent prices
CREATE INDEX IF NOT EXISTS marks_1m_ts_idx
  ON marks_1m (ts DESC);

-- Add high/low columns if not exists (needed for ATR calculation)
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'marks_1m' AND column_name = 'high'
  ) THEN
    ALTER TABLE marks_1m ADD COLUMN high NUMERIC;
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'marks_1m' AND column_name = 'low'
  ) THEN
    ALTER TABLE marks_1m ADD COLUMN low NUMERIC;
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'marks_1m' AND column_name = 'close'
  ) THEN
    ALTER TABLE marks_1m ADD COLUMN close NUMERIC;
  END IF;
END $$;

-- Comments
COMMENT ON TABLE marks_1m IS 'Price history for market regime detection (BTC/ETH 1-minute candles)';
COMMENT ON COLUMN marks_1m.asset IS 'Asset symbol (BTC, ETH)';
COMMENT ON COLUMN marks_1m.ts IS 'Candle timestamp (start of minute)';
COMMENT ON COLUMN marks_1m.mid IS 'Mid price at candle open';
COMMENT ON COLUMN marks_1m.high IS 'High price during candle';
COMMENT ON COLUMN marks_1m.low IS 'Low price during candle';
COMMENT ON COLUMN marks_1m.close IS 'Close price at candle end';
COMMENT ON COLUMN marks_1m.atr14 IS '14-period Average True Range';
