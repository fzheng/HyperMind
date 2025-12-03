-- Migration: Trader correlation matrix for consensus detection
--
-- Stores pairwise correlations between traders based on position posture.
-- Used for calculating effective-K (correlation-adjusted signal strength).
-- Updated daily by batch job.

CREATE TABLE IF NOT EXISTS trader_corr (
    as_of_date DATE NOT NULL,
    asset TEXT NOT NULL CHECK (asset IN ('BTC', 'ETH')),
    addr_a TEXT NOT NULL,
    addr_b TEXT NOT NULL,
    rho DOUBLE PRECISION NOT NULL CHECK (rho >= 0 AND rho <= 1),
    n_buckets INTEGER NOT NULL DEFAULT 0,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (as_of_date, asset, addr_a, addr_b)
);

-- Index for lookup by date and asset
CREATE INDEX IF NOT EXISTS idx_trader_corr_lookup
    ON trader_corr (as_of_date DESC, asset);

-- Index for finding pairs involving a specific trader
CREATE INDEX IF NOT EXISTS idx_trader_corr_by_trader
    ON trader_corr (addr_a, as_of_date DESC);

-- Comment
COMMENT ON TABLE trader_corr IS
    'Daily pairwise correlation between traders based on position posture sign series.
    Used for effective-K calculation in consensus detection.
    rho is clipped to [0,1] and computed from phi/Kendall correlation on 5-min buckets.';

COMMENT ON COLUMN trader_corr.n_buckets IS 'Number of 5-min buckets used for correlation calculation';
COMMENT ON COLUMN trader_corr.rho IS 'Pairwise correlation clipped to [0,1]';
