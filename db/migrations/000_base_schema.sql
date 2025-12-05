-- Migration 000: Base schema
-- This migration creates all base tables needed by the application.
-- It consolidates all tables from docker/postgres-init/ for non-Docker deployments.

-- ==========================================
-- Core Tables (from 001_base.sql)
-- ==========================================

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
  address TEXT NOT NULL,
  symbol TEXT NOT NULL,
  size DOUBLE PRECISION NOT NULL,
  entry_price NUMERIC,
  liquidation_price NUMERIC,
  leverage DOUBLE PRECISION,
  pnl NUMERIC,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (address, symbol)
);
CREATE INDEX IF NOT EXISTS hl_current_positions_symbol_idx ON hl_current_positions (symbol);
CREATE INDEX IF NOT EXISTS hl_current_positions_address_idx ON hl_current_positions (address);

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
CREATE INDEX IF NOT EXISTS ticket_outcomes_closed_ts_idx ON ticket_outcomes (closed_ts DESC);

-- ==========================================
-- Leaderboard Tables (from 001_base.sql, 002_leaderboard_stats.sql)
-- ==========================================

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
  last_refresh_at TIMESTAMPTZ DEFAULT now(),
  fetched_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS hl_leaderboard_entries_period_address_idx
  ON hl_leaderboard_entries (period_days, lower(address));
CREATE INDEX IF NOT EXISTS hl_leaderboard_entries_period_rank_idx
  ON hl_leaderboard_entries (period_days, rank);
CREATE INDEX IF NOT EXISTS hl_leaderboard_entries_period_weight_idx
  ON hl_leaderboard_entries (period_days, weight DESC);
CREATE INDEX IF NOT EXISTS hl_leaderboard_entries_refresh_idx
  ON hl_leaderboard_entries (period_days, last_refresh_at DESC);

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
CREATE INDEX IF NOT EXISTS hl_leaderboard_pnl_points_ts_idx
  ON hl_leaderboard_pnl_points (period_days, address, point_ts DESC);

-- ==========================================
-- Performance Indexes (from 003_performance_indexes.sql)
-- ==========================================

CREATE INDEX IF NOT EXISTS hl_events_type_addr_id_desc_idx
  ON hl_events (type, address, id DESC)
  WHERE type = 'trade';

-- ==========================================
-- Custom Accounts (from 004_custom_accounts.sql)
-- ==========================================

CREATE TABLE IF NOT EXISTS hl_custom_accounts (
  id BIGSERIAL PRIMARY KEY,
  address TEXT NOT NULL,
  nickname TEXT,
  added_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS hl_custom_accounts_address_unique ON hl_custom_accounts (lower(address));

COMMENT ON TABLE hl_custom_accounts IS 'User-added custom accounts for tracking (max 3)';

-- ==========================================
-- Sage State (from 006_sage_state.sql)
-- ==========================================

CREATE TABLE IF NOT EXISTS sage_tracked_addresses (
  address TEXT PRIMARY KEY,
  weight DOUBLE PRECISION NOT NULL,
  rank INT NOT NULL,
  period INT NOT NULL,
  position DOUBLE PRECISION NOT NULL DEFAULT 0.0,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS sage_tracked_addresses_updated_idx
  ON sage_tracked_addresses (updated_at DESC);

COMMENT ON TABLE sage_tracked_addresses IS 'hl-sage tracked address state for recovery on restart';

-- ==========================================
-- Decide State (from 007_decide_state.sql)
-- ==========================================

CREATE TABLE IF NOT EXISTS decide_scores (
  address TEXT PRIMARY KEY,
  score DOUBLE PRECISION NOT NULL,
  weight DOUBLE PRECISION NOT NULL,
  rank INT NOT NULL,
  window_s INT NOT NULL,
  ts TIMESTAMPTZ NOT NULL,
  meta JSONB,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS decide_scores_updated_idx ON decide_scores (updated_at DESC);

CREATE TABLE IF NOT EXISTS decide_fills (
  address TEXT PRIMARY KEY,
  fill_id TEXT NOT NULL,
  asset TEXT NOT NULL,
  side TEXT NOT NULL,
  size DOUBLE PRECISION NOT NULL,
  price DOUBLE PRECISION,
  ts TIMESTAMPTZ NOT NULL,
  meta JSONB,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS decide_fills_updated_idx ON decide_fills (updated_at DESC);

COMMENT ON TABLE decide_scores IS 'hl-decide score state for recovery on restart';
COMMENT ON TABLE decide_fills IS 'hl-decide fill state for recovery on restart';

-- ==========================================
-- BTC/ETH Analysis Cache (from 009_btc_eth_analysis_cache.sql)
-- ==========================================

CREATE TABLE IF NOT EXISTS hl_btc_eth_analysis_cache (
  address TEXT PRIMARY KEY,
  btc_pnl DOUBLE PRECISION NOT NULL DEFAULT 0,
  eth_pnl DOUBLE PRECISION NOT NULL DEFAULT 0,
  btc_eth_pnl DOUBLE PRECISION NOT NULL DEFAULT 0,
  total_pnl DOUBLE PRECISION NOT NULL DEFAULT 0,
  btc_eth_ratio DOUBLE PRECISION NOT NULL DEFAULT 0,
  qualified BOOLEAN NOT NULL DEFAULT FALSE,
  reason TEXT,
  fetched_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
  expires_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT (NOW() + INTERVAL '30 days')
);
CREATE INDEX IF NOT EXISTS hl_btc_eth_analysis_cache_expires_idx
  ON hl_btc_eth_analysis_cache(expires_at);

COMMENT ON TABLE hl_btc_eth_analysis_cache IS 'Cache for BTC/ETH trading analysis, expires after 30 days';
