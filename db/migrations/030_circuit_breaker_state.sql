-- Circuit Breaker State Persistence
-- Phase 4.4: Risk Circuit Breakers
--
-- Extends risk_governor_state to persist circuit breaker state
-- across service restarts.

-- Extend risk_governor_state for circuit breaker tracking
ALTER TABLE risk_governor_state
    ADD COLUMN IF NOT EXISTS consecutive_api_errors INT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS api_pause_until TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS last_api_error_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS consecutive_losses INT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS loss_streak_pause_until TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS last_loss_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS current_position_count INT DEFAULT 0;

-- Track position counts by symbol
CREATE TABLE IF NOT EXISTS risk_position_tracking (
    symbol VARCHAR(16) PRIMARY KEY,
    position_count INT DEFAULT 0,
    last_updated TIMESTAMPTZ DEFAULT NOW()
);

-- Comments
COMMENT ON COLUMN risk_governor_state.consecutive_api_errors IS 'Count of consecutive API errors';
COMMENT ON COLUMN risk_governor_state.api_pause_until IS 'Pause trading until this time due to API errors';
COMMENT ON COLUMN risk_governor_state.consecutive_losses IS 'Count of consecutive losing trades';
COMMENT ON COLUMN risk_governor_state.loss_streak_pause_until IS 'Pause trading until this time due to loss streak';
COMMENT ON COLUMN risk_governor_state.current_position_count IS 'Current number of open positions';

COMMENT ON TABLE risk_position_tracking IS 'Tracks position count per symbol for circuit breaker';
