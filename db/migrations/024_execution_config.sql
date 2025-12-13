-- Execution configuration and portfolio tracking
-- Phase 3e: Hyperliquid-only, extensible for multi-exchange in Phase 4

-- Execution configuration (singleton row)
CREATE TABLE IF NOT EXISTS execution_config (
    id SERIAL PRIMARY KEY,
    enabled BOOLEAN DEFAULT false,

    -- Hyperliquid settings
    hl_enabled BOOLEAN DEFAULT false,
    hl_address VARCHAR(42),  -- Trading wallet address
    hl_max_leverage INT DEFAULT 3,
    hl_max_position_pct DECIMAL(5,4) DEFAULT 0.02,  -- 2% max per position
    hl_max_exposure_pct DECIMAL(5,4) DEFAULT 0.10,  -- 10% total exposure

    -- Risk settings
    max_daily_loss_pct DECIMAL(5,4) DEFAULT 0.05,   -- 5% max daily loss
    cooldown_after_loss_min INT DEFAULT 60,          -- 60 min cooldown after stop

    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Insert default config if not exists
INSERT INTO execution_config (id, enabled, hl_enabled)
VALUES (1, false, false)
ON CONFLICT (id) DO NOTHING;

-- Execution log (tracks all execution attempts)
CREATE TABLE IF NOT EXISTS execution_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    decision_id UUID REFERENCES decision_logs(id),
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Exchange info (Hyperliquid for now, extensible for Phase 4)
    exchange VARCHAR(32) NOT NULL DEFAULT 'hyperliquid',

    -- Order details
    symbol VARCHAR(16) NOT NULL,
    side VARCHAR(8) NOT NULL,        -- buy, sell
    size DECIMAL(20,8) NOT NULL,
    leverage INT NOT NULL,

    -- Result
    status VARCHAR(16) NOT NULL,     -- pending, filled, partial, failed, rejected
    fill_price DECIMAL(20,8),
    fill_size DECIMAL(20,8),
    error_message TEXT,

    -- Risk context at execution time
    account_value DECIMAL(20,8),
    position_pct DECIMAL(5,4),       -- % of equity used
    exposure_before DECIMAL(5,4),    -- exposure before trade
    exposure_after DECIMAL(5,4)      -- exposure after trade
);

CREATE INDEX IF NOT EXISTS idx_execution_logs_decision ON execution_logs(decision_id);
CREATE INDEX IF NOT EXISTS idx_execution_logs_created ON execution_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_execution_logs_exchange ON execution_logs(exchange, created_at DESC);

-- Portfolio snapshots (periodic equity tracking)
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Exchange info
    exchange VARCHAR(32) NOT NULL DEFAULT 'hyperliquid',

    -- Account summary
    account_value DECIMAL(20,8) NOT NULL,
    total_margin_used DECIMAL(20,8),
    available_margin DECIMAL(20,8),
    total_unrealized_pnl DECIMAL(20,8),
    total_realized_pnl DECIMAL(20,8),

    -- Risk metrics
    total_exposure_pct DECIMAL(5,4),
    position_count INT DEFAULT 0,

    -- Raw data for debugging
    raw_data JSONB
);

CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_created ON portfolio_snapshots(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_exchange ON portfolio_snapshots(exchange, created_at DESC);

-- Live positions (updated in real-time)
CREATE TABLE IF NOT EXISTS live_positions (
    id SERIAL PRIMARY KEY,
    exchange VARCHAR(32) NOT NULL DEFAULT 'hyperliquid',
    symbol VARCHAR(16) NOT NULL,

    -- Position details
    side VARCHAR(8) NOT NULL,        -- long, short
    size DECIMAL(20,8) NOT NULL,
    entry_price DECIMAL(20,8) NOT NULL,
    mark_price DECIMAL(20,8),
    liquidation_price DECIMAL(20,8),

    -- P&L
    unrealized_pnl DECIMAL(20,8),
    margin_used DECIMAL(20,8),
    leverage INT,

    -- Tracking
    opened_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Link to decision that opened this position
    decision_id UUID REFERENCES decision_logs(id),

    UNIQUE(exchange, symbol)
);

CREATE INDEX IF NOT EXISTS idx_live_positions_exchange ON live_positions(exchange);
CREATE INDEX IF NOT EXISTS idx_live_positions_updated ON live_positions(updated_at DESC);
