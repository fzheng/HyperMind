-- Migration: 026_risk_governor_state
-- Phase 3f: Risk Governor state persistence
-- Stores kill switch state and daily tracking across restarts

CREATE TABLE IF NOT EXISTS risk_governor_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Initial state values
INSERT INTO risk_governor_state (key, value) VALUES
    ('kill_switch_active', 'false'),
    ('kill_switch_triggered_at', ''),
    ('daily_starting_equity', '0'),
    ('daily_start_date', '')
ON CONFLICT (key) DO NOTHING;

-- Daily PnL tracking for kill switch
CREATE TABLE IF NOT EXISTS risk_daily_pnl (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    date DATE NOT NULL,
    starting_equity NUMERIC NOT NULL,
    current_equity NUMERIC,
    realized_pnl NUMERIC DEFAULT 0,
    unrealized_pnl NUMERIC DEFAULT 0,
    daily_drawdown_pct NUMERIC DEFAULT 0,
    kill_switch_triggered BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(date)
);

-- Index for quick lookups
CREATE INDEX IF NOT EXISTS idx_risk_daily_pnl_date ON risk_daily_pnl(date DESC);

-- Comments
COMMENT ON TABLE risk_governor_state IS 'Risk Governor persistent state (kill switch, daily tracking)';
COMMENT ON TABLE risk_daily_pnl IS 'Daily PnL tracking for drawdown kill switch';
COMMENT ON COLUMN risk_daily_pnl.kill_switch_triggered IS 'Whether kill switch triggered on this day';
