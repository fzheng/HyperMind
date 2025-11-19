-- Performance optimization indexes
-- Migration 003: Add missing indexes for query optimization

-- Composite index for pageTrades() queries with address filter
CREATE INDEX IF NOT EXISTS hl_events_type_addr_id_desc_idx
  ON hl_events (type, address, id DESC)
  WHERE type = 'trade';

-- Index for ticket lookups with outcomes
CREATE INDEX IF NOT EXISTS ticket_outcomes_closed_ts_idx
  ON ticket_outcomes (closed_ts DESC);

-- Index for leaderboard queries by weight
CREATE INDEX IF NOT EXISTS hl_leaderboard_entries_period_weight_idx
  ON hl_leaderboard_entries (period_days, weight DESC);

-- Index for PnL point queries
CREATE INDEX IF NOT EXISTS hl_leaderboard_pnl_points_ts_idx
  ON hl_leaderboard_pnl_points (period_days, address, point_ts DESC);

-- Note: Cannot create a partial index for open tickets using NOT EXISTS
-- PostgreSQL doesn't support subqueries in partial index predicates
-- This would require application-level filtering or a materialized view
