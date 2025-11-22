-- Migration 004: Custom accounts for user tracking
-- Allows users to add up to 3 custom accounts to track alongside system-selected accounts

CREATE TABLE IF NOT EXISTS hl_custom_accounts (
  id BIGSERIAL PRIMARY KEY,
  address TEXT NOT NULL,
  nickname TEXT,
  added_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Unique index on lowercase address for consistent lookups (prevents duplicates)
CREATE UNIQUE INDEX IF NOT EXISTS hl_custom_accounts_address_unique ON hl_custom_accounts (lower(address));

-- Add column to track last refresh timestamp for leaderboard
ALTER TABLE hl_leaderboard_entries
ADD COLUMN IF NOT EXISTS last_refresh_at TIMESTAMPTZ DEFAULT now();

-- Create index for refresh timestamp queries
CREATE INDEX IF NOT EXISTS hl_leaderboard_entries_refresh_idx
ON hl_leaderboard_entries (period_days, last_refresh_at DESC);

-- Comment for documentation
COMMENT ON TABLE hl_custom_accounts IS 'User-added custom accounts for tracking (max 3)';
COMMENT ON COLUMN hl_custom_accounts.address IS 'Ethereum address (stored lowercase)';
COMMENT ON COLUMN hl_custom_accounts.nickname IS 'Optional user-provided nickname';
