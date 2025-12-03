-- Migration: Add episode tracking fields for R calculation audit
--
-- This migration adds fields to position_signals for:
-- 1. VWAP entry/exit prices
-- 2. Stop parameters used for R calculation
-- 3. Cost breakdown (fees, funding)
-- 4. Unclamped R for analysis
-- 5. Hold time for regime analysis

-- Add VWAP and R calculation fields
ALTER TABLE position_signals
    ADD COLUMN IF NOT EXISTS entry_px_vwap DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS exit_px_vwap DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS pnl_bps_incl_costs DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS atr_bps_entry DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS stop_bps_used DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS r_unclamped DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS r_clamped DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS hold_secs INTEGER,
    ADD COLUMN IF NOT EXISTS fees_bps DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS funding_bps DOUBLE PRECISION;

-- Update existing rows to have default stop_bps_used (1% = 100 bps)
UPDATE position_signals
SET stop_bps_used = 100,
    entry_px_vwap = entry_price,
    exit_px_vwap = exit_price,
    r_clamped = result_r,
    r_unclamped = result_r
WHERE stop_bps_used IS NULL;

-- Index for analysis queries
CREATE INDEX IF NOT EXISTS idx_position_signals_analysis
    ON position_signals (address, asset, entry_ts)
    WHERE status = 'closed';

-- Comment
COMMENT ON COLUMN position_signals.entry_px_vwap IS 'VWAP of all fills that opened the position';
COMMENT ON COLUMN position_signals.exit_px_vwap IS 'VWAP of all fills that closed the position';
COMMENT ON COLUMN position_signals.stop_bps_used IS 'Policy stop distance in bps used for R calculation';
COMMENT ON COLUMN position_signals.r_unclamped IS 'Raw R-multiple before winsorization (for analysis)';
COMMENT ON COLUMN position_signals.r_clamped IS 'R-multiple after winsorization to [-2, +2]';
COMMENT ON COLUMN position_signals.hold_secs IS 'Duration from entry to exit in seconds';
COMMENT ON COLUMN position_signals.fees_bps IS 'Total fees paid in bps of notional';
COMMENT ON COLUMN position_signals.funding_bps IS 'Net funding paid/received in bps';
