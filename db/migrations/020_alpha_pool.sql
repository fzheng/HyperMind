-- Migration: Alpha Pool tables
--
-- Creates standalone tables for Alpha Pool, fully decoupled from legacy leaderboard.
-- Addresses are fetched directly from Hyperliquid API.

-- Alpha Pool tracked addresses (replaces dependency on hl_leaderboard_entries)
CREATE TABLE IF NOT EXISTS alpha_pool_addresses (
    address TEXT PRIMARY KEY,
    nickname TEXT,
    account_value NUMERIC,
    pnl_30d NUMERIC,
    win_rate NUMERIC,
    last_refreshed TIMESTAMPTZ DEFAULT NOW(),
    source TEXT DEFAULT 'leaderboard',  -- 'leaderboard', 'manual', 'discovered'
    is_active BOOLEAN DEFAULT true
);

-- Index for active addresses
CREATE INDEX IF NOT EXISTS idx_alpha_pool_active
    ON alpha_pool_addresses (is_active, last_refreshed DESC);

-- Comments
COMMENT ON TABLE alpha_pool_addresses IS 'Alpha Pool candidate addresses, decoupled from legacy leaderboard';
COMMENT ON COLUMN alpha_pool_addresses.source IS 'How address was added: leaderboard (from HL leaderboard), manual (user added), discovered (from tracking)';
COMMENT ON COLUMN alpha_pool_addresses.is_active IS 'Whether address is currently in the active pool';
