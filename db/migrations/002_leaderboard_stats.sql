-- Adds stat columns + pnl points table for leaderboard enrichment rollout.

ALTER TABLE hl_leaderboard_entries
  ADD COLUMN IF NOT EXISTS stat_open_positions INT,
  ADD COLUMN IF NOT EXISTS stat_closed_positions INT,
  ADD COLUMN IF NOT EXISTS stat_avg_pos_duration DOUBLE PRECISION,
  ADD COLUMN IF NOT EXISTS stat_total_pnl DOUBLE PRECISION,
  ADD COLUMN IF NOT EXISTS stat_max_drawdown DOUBLE PRECISION;

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
