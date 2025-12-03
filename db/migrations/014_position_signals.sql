-- Migration: Position-based signal tracking
--
-- This migration introduces position lifecycle tracking for accurate
-- trader performance measurement. Instead of counting every fill as a signal,
-- we now track complete position lifecycles (open → close).

-- Create position_signals table to track position lifecycles
CREATE TABLE IF NOT EXISTS position_signals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    address TEXT NOT NULL,
    asset TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('long', 'short')),

    -- Entry info
    entry_fill_id TEXT NOT NULL,
    entry_price DOUBLE PRECISION NOT NULL,
    entry_size DOUBLE PRECISION NOT NULL,
    entry_ts TIMESTAMPTZ NOT NULL,

    -- Exit info (NULL until closed)
    exit_fill_id TEXT,
    exit_price DOUBLE PRECISION,
    exit_ts TIMESTAMPTZ,

    -- P&L (NULL until closed)
    realized_pnl DOUBLE PRECISION,
    result_r DOUBLE PRECISION,

    -- Status tracking
    status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'closed', 'expired')),
    closed_reason TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Ensure unique entry fill
    CONSTRAINT position_signals_entry_fill_unique UNIQUE (entry_fill_id)
);

-- Index for finding open positions by address+asset
CREATE INDEX IF NOT EXISTS idx_position_signals_open
    ON position_signals(address, asset)
    WHERE status = 'open';

-- Index for querying closed positions by address
CREATE INDEX IF NOT EXISTS idx_position_signals_closed
    ON position_signals(address, updated_at DESC)
    WHERE status = 'closed';

-- Index for recent positions
CREATE INDEX IF NOT EXISTS idx_position_signals_recent
    ON position_signals(updated_at DESC);

-- Add position tracking columns to trader_performance
ALTER TABLE trader_performance
    ADD COLUMN IF NOT EXISTS positions_opened INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS positions_closed INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS positions_won INTEGER NOT NULL DEFAULT 0;

-- Reset corrupted data from fill-level tracking
-- The previous data counted fills as signals, which is incorrect
UPDATE trader_performance SET
    total_signals = 0,
    winning_signals = 0,
    total_pnl_r = 0,
    alpha = 1,
    beta = 1,
    nig_m = 0.0,
    nig_kappa = 1.0,
    nig_alpha = 3.0,
    nig_beta = 1.0,
    avg_r = NULL,
    positions_opened = 0,
    positions_closed = 0,
    positions_won = 0,
    updated_at = NOW();

-- Clean up the corrupted tickets table
-- These were created from fill-level spam, not real position signals
TRUNCATE TABLE ticket_outcomes CASCADE;
TRUNCATE TABLE tickets CASCADE;

-- Comment explaining the reset
COMMENT ON TABLE position_signals IS
    'Tracks complete position lifecycles (open → close) for accurate trader performance measurement.
    Each row represents one position from entry to exit, with the final R-multiple used for NIG posterior updates.';
