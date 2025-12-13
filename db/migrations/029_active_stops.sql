-- Active Stops Table
-- Phase 4.3: Position Management
--
-- Tracks stop-loss, take-profit, and timeout settings for open positions.
-- Used by StopManager for local stop monitoring.

-- Create active_stops table
CREATE TABLE IF NOT EXISTS active_stops (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    decision_id UUID NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Position info
    symbol VARCHAR(16) NOT NULL,
    direction VARCHAR(8) NOT NULL,  -- 'long' or 'short'
    entry_price DECIMAL(20,8) NOT NULL,
    entry_size DECIMAL(20,8) NOT NULL,

    -- Stop levels
    stop_price DECIMAL(20,8) NOT NULL,
    take_profit_price DECIMAL(20,8),

    -- Trailing stop
    trailing_enabled BOOLEAN DEFAULT false,
    trail_distance_pct DECIMAL(5,4),

    -- Timeout
    timeout_at TIMESTAMPTZ,

    -- Status tracking
    status VARCHAR(16) DEFAULT 'active',  -- active, triggered, expired, cancelled
    triggered_at TIMESTAMPTZ,
    triggered_price DECIMAL(20,8),
    triggered_reason VARCHAR(32),  -- stop_loss, take_profit, timeout, manual

    -- Unique constraint to prevent duplicate stops per position
    UNIQUE(symbol, decision_id)
);

-- Index for quick lookup of active stops
CREATE INDEX IF NOT EXISTS idx_active_stops_status
    ON active_stops(status)
    WHERE status = 'active';

-- Index for cleanup of old stops
CREATE INDEX IF NOT EXISTS idx_active_stops_triggered_at
    ON active_stops(triggered_at)
    WHERE status IN ('triggered', 'expired', 'cancelled');

-- Comments
COMMENT ON TABLE active_stops IS 'Tracks stop-loss and take-profit levels for open positions';
COMMENT ON COLUMN active_stops.decision_id IS 'Reference to decision_logs entry that opened position';
COMMENT ON COLUMN active_stops.stop_price IS 'Stop-loss price level';
COMMENT ON COLUMN active_stops.take_profit_price IS 'Take-profit price level (optional)';
COMMENT ON COLUMN active_stops.trailing_enabled IS 'Whether stop trails with favorable price movement';
COMMENT ON COLUMN active_stops.trail_distance_pct IS 'Distance to maintain from price for trailing stop';
COMMENT ON COLUMN active_stops.timeout_at IS 'Auto-close position at this time';
COMMENT ON COLUMN active_stops.triggered_reason IS 'Why stop was triggered: stop_loss, take_profit, timeout, manual';
