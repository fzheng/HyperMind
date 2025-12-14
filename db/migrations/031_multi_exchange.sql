-- Multi-Exchange Support
-- Phase 6: Multi-Exchange Integration
--
-- Adds exchange field to track which exchange was used for each trade/stop.
-- Supports: hyperliquid, aster, bybit

-- Add exchange column to execution_config for default exchange selection
ALTER TABLE execution_config
    ADD COLUMN IF NOT EXISTS default_exchange VARCHAR(16) DEFAULT 'hyperliquid';

-- Add exchange column to execution_logs
ALTER TABLE execution_logs
    ADD COLUMN IF NOT EXISTS exchange_type VARCHAR(16);

-- Update existing rows to have hyperliquid as exchange (backward compatibility)
UPDATE execution_logs
SET exchange_type = COALESCE(exchange, 'hyperliquid')
WHERE exchange_type IS NULL;

-- Add exchange column to active_stops
ALTER TABLE active_stops
    ADD COLUMN IF NOT EXISTS exchange VARCHAR(16) DEFAULT 'hyperliquid';

-- Create exchange_connections table to track connection status
CREATE TABLE IF NOT EXISTS exchange_connections (
    id SERIAL PRIMARY KEY,
    exchange_type VARCHAR(16) NOT NULL,
    testnet BOOLEAN DEFAULT true,
    is_connected BOOLEAN DEFAULT false,
    last_connected_at TIMESTAMPTZ,
    last_error TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(exchange_type, testnet)
);

-- Create exchange_balances table for tracking multi-exchange portfolio
CREATE TABLE IF NOT EXISTS exchange_balances (
    id SERIAL PRIMARY KEY,
    exchange_type VARCHAR(16) NOT NULL,
    total_equity DECIMAL(20,4),
    available_balance DECIMAL(20,4),
    margin_used DECIMAL(20,4),
    unrealized_pnl DECIMAL(20,4),
    timestamp TIMESTAMPTZ DEFAULT NOW(),

    -- Keep latest balance per exchange
    UNIQUE(exchange_type)
);

-- Index for efficient exchange lookups
CREATE INDEX IF NOT EXISTS idx_execution_logs_exchange
    ON execution_logs(exchange_type);

CREATE INDEX IF NOT EXISTS idx_active_stops_exchange
    ON active_stops(exchange);

-- Comments
COMMENT ON COLUMN execution_config.default_exchange IS 'Default exchange for execution: hyperliquid, aster, bybit';
COMMENT ON COLUMN execution_logs.exchange_type IS 'Exchange used for this trade';
COMMENT ON COLUMN active_stops.exchange IS 'Exchange where position is held';
COMMENT ON TABLE exchange_connections IS 'Tracks connection status for each exchange';
COMMENT ON TABLE exchange_balances IS 'Latest balance snapshot per exchange for portfolio view';
