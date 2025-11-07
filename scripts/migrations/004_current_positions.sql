-- Latest BTC position snapshots per address
CREATE TABLE IF NOT EXISTS hl_current_positions (
  address TEXT PRIMARY KEY,
  symbol TEXT NOT NULL,
  size DOUBLE PRECISION NOT NULL,
  entry_price NUMERIC,
  liquidation_price NUMERIC,
  leverage DOUBLE PRECISION,
  pnl NUMERIC,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS hl_current_positions_symbol_idx ON hl_current_positions (symbol);
