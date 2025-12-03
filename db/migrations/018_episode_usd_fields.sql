-- Migration: Add USD-denominated fields for R calculation audit
--
-- Per quant review: "ensure realized_pnl and entry_notional are in the same
-- quote currency. Persist fees_usd, funding_usd separately so realized_r
-- and EV_cost_R reconcile later."
--
-- This migration adds absolute USD amounts alongside the bps values
-- for complete audit trail and reconciliation.

-- Add USD-denominated fields for audit
ALTER TABLE position_signals
    ADD COLUMN IF NOT EXISTS entry_notional_usd DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS risk_usd DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS fees_usd DOUBLE PRECISION DEFAULT 0,
    ADD COLUMN IF NOT EXISTS funding_usd DOUBLE PRECISION DEFAULT 0,
    ADD COLUMN IF NOT EXISTS realized_pnl_usd DOUBLE PRECISION;

-- Backfill entry_notional_usd from existing data
UPDATE position_signals
SET entry_notional_usd = entry_price * entry_size,
    risk_usd = entry_price * entry_size * COALESCE(stop_bps_used, 100) / 10000,
    realized_pnl_usd = realized_pnl
WHERE entry_notional_usd IS NULL AND entry_price IS NOT NULL;

-- Comments for documentation
COMMENT ON COLUMN position_signals.entry_notional_usd IS 'Entry notional in USD (entry_price * entry_size)';
COMMENT ON COLUMN position_signals.risk_usd IS 'Risk amount in USD (entry_notional * stop_fraction)';
COMMENT ON COLUMN position_signals.fees_usd IS 'Total fees paid in USD';
COMMENT ON COLUMN position_signals.funding_usd IS 'Net funding paid/received in USD';
COMMENT ON COLUMN position_signals.realized_pnl_usd IS 'Realized P&L in USD (from Hyperliquid or calculated)';
