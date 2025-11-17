-- Base schema for hl-platform services

CREATE TABLE IF NOT EXISTS addresses (
  address TEXT PRIMARY KEY,
  nickname TEXT
);

CREATE TABLE IF NOT EXISTS hl_events (
  id BIGSERIAL PRIMARY KEY,
  at TIMESTAMPTZ NOT NULL DEFAULT now(),
  address TEXT NOT NULL,
  type TEXT NOT NULL,
  symbol TEXT NOT NULL,
  payload JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS hl_events_at_idx ON hl_events (at DESC);
CREATE INDEX IF NOT EXISTS hl_events_type_at_idx ON hl_events (type, at DESC);
CREATE INDEX IF NOT EXISTS hl_events_addr_at_idx ON hl_events (address, at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS hl_events_trade_hash_uq
  ON hl_events ((payload->>'hash'))
  WHERE type = 'trade';
CREATE INDEX IF NOT EXISTS hl_events_trade_at_desc_idx
  ON hl_events (at DESC) WHERE type = 'trade';
CREATE INDEX IF NOT EXISTS hl_events_trade_address_at_desc_idx
  ON hl_events (address, at DESC) WHERE type = 'trade';

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

CREATE TABLE IF NOT EXISTS marks_1m (
  asset TEXT NOT NULL,
  ts TIMESTAMPTZ NOT NULL,
  mid NUMERIC NOT NULL,
  atr14 NUMERIC,
  PRIMARY KEY(asset, ts)
);

CREATE TABLE IF NOT EXISTS tickets (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  ts TIMESTAMPTZ NOT NULL DEFAULT now(),
  address TEXT NOT NULL,
  asset TEXT NOT NULL,
  side TEXT NOT NULL,
  payload JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS tickets_ts_idx ON tickets (ts DESC);
CREATE INDEX IF NOT EXISTS tickets_asset_idx ON tickets (asset);

CREATE TABLE IF NOT EXISTS ticket_outcomes (
  ticket_id UUID PRIMARY KEY REFERENCES tickets(id) ON DELETE CASCADE,
  closed_ts TIMESTAMPTZ NOT NULL DEFAULT now(),
  result_r DOUBLE PRECISION,
  closed_reason TEXT NOT NULL,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS hl_leaderboard_entries (
  id BIGSERIAL PRIMARY KEY,
  period_days INT NOT NULL,
  address TEXT NOT NULL,
  rank INT NOT NULL,
  score DOUBLE PRECISION NOT NULL,
  weight DOUBLE PRECISION NOT NULL,
  win_rate DOUBLE PRECISION,
  executed_orders INT,
  realized_pnl DOUBLE PRECISION,
  pnl_consistency DOUBLE PRECISION,
  efficiency DOUBLE PRECISION,
  remark TEXT,
  labels JSONB,
  metrics JSONB,
  stat_open_positions INT,
  stat_closed_positions INT,
  stat_avg_pos_duration DOUBLE PRECISION,
  stat_total_pnl DOUBLE PRECISION,
  stat_max_drawdown DOUBLE PRECISION,
  fetched_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS hl_leaderboard_entries_period_address_idx
  ON hl_leaderboard_entries (period_days, lower(address));
CREATE INDEX IF NOT EXISTS hl_leaderboard_entries_period_rank_idx
  ON hl_leaderboard_entries (period_days, rank);

CREATE TABLE IF NOT EXISTS hl_leaderboard_pnl_points (
  id BIGSERIAL PRIMARY KEY,
  period_days INT NOT NULL,
  address TEXT NOT NULL,
  source TEXT NOT NULL,
  window_name TEXT NOT NULL,
  point_ts TIMESTAMPTZ NOT NULL,
  pnl_value DOUBLE PRECISION,
  equity_value DOUBLE PRECISION,
  fetched_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS hl_leaderboard_pnl_points_period_addr_idx
  ON hl_leaderboard_pnl_points (period_days, lower(address), window_name);
