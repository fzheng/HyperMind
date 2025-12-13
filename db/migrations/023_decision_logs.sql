-- Decision logging for signal auditability
-- Records every consensus evaluation with reasoning

CREATE TABLE IF NOT EXISTS decision_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Signal identification
    symbol VARCHAR(16) NOT NULL,           -- BTC, ETH
    direction VARCHAR(8) NOT NULL,         -- long, short, none
    decision_type VARCHAR(16) NOT NULL,    -- signal, skip, risk_reject

    -- Inputs (aggregated, no addresses)
    trader_count INT NOT NULL,
    agreement_pct DECIMAL(5,4) NOT NULL,
    effective_k DECIMAL(6,3) NOT NULL,
    avg_confidence DECIMAL(5,4),
    ev_estimate DECIMAL(8,4),
    price_at_decision DECIMAL(20,8),

    -- Gate results (JSONB for flexibility)
    -- Example: [{"name": "supermajority", "passed": true, "value": 0.70, "threshold": 0.70}]
    gates JSONB NOT NULL,

    -- Risk check results (if applicable)
    risk_checks JSONB,

    -- Human-readable summary (2-4 sentences)
    reasoning TEXT NOT NULL,

    -- Outcome tracking (updated when position closes)
    outcome_pnl DECIMAL(20,8),
    outcome_r_multiple DECIMAL(8,4),
    outcome_closed_at TIMESTAMPTZ
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_decision_logs_created ON decision_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_decision_logs_symbol ON decision_logs(symbol, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_decision_logs_type ON decision_logs(decision_type);
CREATE INDEX IF NOT EXISTS idx_decision_logs_symbol_type ON decision_logs(symbol, decision_type, created_at DESC);
