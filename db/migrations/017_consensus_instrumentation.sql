-- Migration: Consensus ticket instrumentation
--
-- Adds columns to tickets table for full consensus gate logging.
-- Essential for debugging, analysis, and learning EV parameters.

-- Add consensus instrumentation columns
ALTER TABLE tickets
    ADD COLUMN IF NOT EXISTS n_traders INTEGER,
    ADD COLUMN IF NOT EXISTS n_agree INTEGER,
    ADD COLUMN IF NOT EXISTS eff_k DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS dispersion DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS staleness_secs INTEGER,
    ADD COLUMN IF NOT EXISTS drift_r DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS p_win DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS ev_gross_r DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS ev_cost_r DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS ev_net_r DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS contributors_json JSONB;

-- Index for analysis queries
CREATE INDEX IF NOT EXISTS idx_tickets_consensus_analysis
    ON tickets (ts DESC, asset)
    WHERE n_traders IS NOT NULL;

-- Comments
COMMENT ON COLUMN tickets.n_traders IS 'Total number of traders with votes in window';
COMMENT ON COLUMN tickets.n_agree IS 'Number of traders agreeing with majority direction';
COMMENT ON COLUMN tickets.eff_k IS 'Correlation-adjusted effective number of independent signals';
COMMENT ON COLUMN tickets.dispersion IS '1 - majority_pct; higher = more disagreement';
COMMENT ON COLUMN tickets.staleness_secs IS 'Age of oldest agreeing vote in seconds';
COMMENT ON COLUMN tickets.drift_r IS 'Price drift from median voter entry in R-units';
COMMENT ON COLUMN tickets.p_win IS 'Estimated win probability from trader posteriors';
COMMENT ON COLUMN tickets.ev_gross_r IS 'Expected value before costs in R-units';
COMMENT ON COLUMN tickets.ev_cost_r IS 'Expected costs in R-units';
COMMENT ON COLUMN tickets.ev_net_r IS 'Net expected value after costs in R-units';
COMMENT ON COLUMN tickets.contributors_json IS 'JSON array of contributing trader addresses with their votes';
