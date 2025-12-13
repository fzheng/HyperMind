-- Migration: 025_trader_snapshots
-- Phase 3f: Shadow Ledger for survivorship-bias-free selection
-- Tracks all traders who ever appeared, including those who "blew up"

CREATE TABLE IF NOT EXISTS trader_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    snapshot_date DATE NOT NULL,
    address TEXT NOT NULL,

    -- Versioning: tracks filter/scoring changes
    selection_version TEXT NOT NULL DEFAULT '3f.1',

    -- Multi-universe membership (boolean flags for full path tracking)
    is_leaderboard_scanned BOOLEAN NOT NULL DEFAULT FALSE,
    is_candidate_filtered BOOLEAN NOT NULL DEFAULT FALSE,
    is_quality_qualified BOOLEAN NOT NULL DEFAULT FALSE,
    is_pool_selected BOOLEAN NOT NULL DEFAULT FALSE,
    is_pinned_custom BOOLEAN NOT NULL DEFAULT FALSE,

    -- As-of features (snapshot at this date)
    account_value NUMERIC,
    pnl_30d NUMERIC,
    roi_30d NUMERIC,
    win_rate NUMERIC,
    episode_count INTEGER,
    week_volume NUMERIC,
    orders_per_day NUMERIC,

    -- R-multiple stats (gross vs net for proper gating)
    avg_r_gross NUMERIC,
    avg_r_net NUMERIC,

    -- Peak account value for drawdown calculation
    peak_account_value NUMERIC,

    -- NIG posterior params (if computed)
    nig_mu NUMERIC,
    nig_kappa NUMERIC,
    nig_alpha NUMERIC,
    nig_beta NUMERIC,

    -- Thompson sampling (stored for reproducibility)
    thompson_draw NUMERIC,
    thompson_seed BIGINT,
    selection_rank INTEGER,

    -- Lifecycle events
    event_type TEXT CHECK (event_type IN (
        'entered',    -- First appeared in this universe
        'active',     -- Continuing in universe
        'promoted',   -- Moved to higher universe
        'demoted',    -- Moved to lower universe
        'death',      -- Terminal event
        'censored'    -- Non-terminal disappearance
    )),

    -- Death types (terminal - trader permanently excluded)
    death_type TEXT CHECK (death_type IS NULL OR death_type IN (
        'liquidation',           -- Account liquidated on Hyperliquid
        'drawdown_80',           -- Current equity < 20% of peak
        'account_value_floor',   -- Account dropped below $10k
        'negative_equity'        -- Account value <= 0
    )),

    -- Censor types (non-terminal - disappeared but not dead)
    censor_type TEXT CHECK (censor_type IS NULL OR censor_type IN (
        'inactive_30d',          -- No fills for 30 days
        'stopped_btc_eth',       -- Only trading other assets
        'api_unavailable',       -- HL API returns no data
        'manual_removal'         -- Manually removed from tracking
    )),

    -- FDR qualification
    skill_p_value NUMERIC,       -- P-value from t-test on R-multiples
    fdr_qualified BOOLEAN,       -- Passed Benjamini-Hochberg at alpha=0.10

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- One row per trader per day per version
    UNIQUE(snapshot_date, address, selection_version)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_snapshots_date ON trader_snapshots(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_snapshots_address ON trader_snapshots(address);
CREATE INDEX IF NOT EXISTS idx_snapshots_version ON trader_snapshots(selection_version);
CREATE INDEX IF NOT EXISTS idx_snapshots_selected ON trader_snapshots(is_pool_selected) WHERE is_pool_selected = TRUE;
CREATE INDEX IF NOT EXISTS idx_snapshots_death ON trader_snapshots(event_type) WHERE event_type = 'death';
CREATE INDEX IF NOT EXISTS idx_snapshots_date_version ON trader_snapshots(snapshot_date, selection_version);

-- Comments for documentation
COMMENT ON TABLE trader_snapshots IS 'Shadow Ledger: Daily snapshots of all traders for survivorship-bias-free analysis';
COMMENT ON COLUMN trader_snapshots.selection_version IS 'Version of selection criteria (bump when filters change)';
COMMENT ON COLUMN trader_snapshots.thompson_draw IS 'Sampled mu from NIG posterior for Thompson selection';
COMMENT ON COLUMN trader_snapshots.thompson_seed IS 'RNG seed for reproducible Thompson sampling';
COMMENT ON COLUMN trader_snapshots.avg_r_gross IS 'Mean R-multiple before costs';
COMMENT ON COLUMN trader_snapshots.avg_r_net IS 'Mean R-multiple after conservative cost estimate';
COMMENT ON COLUMN trader_snapshots.death_type IS 'Terminal event type (trader permanently excluded)';
COMMENT ON COLUMN trader_snapshots.censor_type IS 'Non-terminal disappearance (for survival analysis)';
