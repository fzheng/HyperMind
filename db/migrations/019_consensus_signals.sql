-- Consensus signals table for storing Alpha Pool signal decisions
-- Created: 2025-12

CREATE TABLE IF NOT EXISTS consensus_signals (
    id UUID PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    direction VARCHAR(10) NOT NULL, -- 'long' or 'short'
    entry_price DOUBLE PRECISION NOT NULL,
    stop_price DOUBLE PRECISION,

    -- Consensus metrics
    n_traders INTEGER NOT NULL,
    n_agreeing INTEGER NOT NULL,
    eff_k DOUBLE PRECISION NOT NULL,
    dispersion DOUBLE PRECISION,

    -- Confidence & EV
    p_win DOUBLE PRECISION NOT NULL,
    ev_gross_r DOUBLE PRECISION NOT NULL,
    ev_cost_r DOUBLE PRECISION NOT NULL,
    ev_net_r DOUBLE PRECISION NOT NULL,

    -- Timing
    latency_ms INTEGER NOT NULL,
    median_voter_price DOUBLE PRECISION,
    mid_delta_bps DOUBLE PRECISION,

    -- Metadata
    trigger_addresses TEXT[] NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Outcome tracking (updated when position closes)
    outcome VARCHAR(20), -- 'win', 'loss', 'breakeven', 'expired'
    exit_price DOUBLE PRECISION,
    result_r DOUBLE PRECISION,
    closed_at TIMESTAMPTZ,

    -- Audit
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for time-based queries
CREATE INDEX IF NOT EXISTS idx_consensus_signals_created_at
    ON consensus_signals(created_at DESC);

-- Index for symbol filtering
CREATE INDEX IF NOT EXISTS idx_consensus_signals_symbol
    ON consensus_signals(symbol);

-- Index for outcome analysis
CREATE INDEX IF NOT EXISTS idx_consensus_signals_outcome
    ON consensus_signals(outcome) WHERE outcome IS NOT NULL;
