-- Kelly Criterion Position Sizing Configuration
-- Phase 4.1: Risk Management
--
-- Adds Kelly-related configuration to execution_config and
-- Kelly sizing details to execution_logs for audit trail.

-- Add Kelly configuration columns to execution_config
ALTER TABLE execution_config
    ADD COLUMN IF NOT EXISTS kelly_enabled BOOLEAN DEFAULT false,
    ADD COLUMN IF NOT EXISTS kelly_fraction DECIMAL(4,3) DEFAULT 0.25,
    ADD COLUMN IF NOT EXISTS kelly_min_episodes INT DEFAULT 30,
    ADD COLUMN IF NOT EXISTS kelly_fallback_pct DECIMAL(6,5) DEFAULT 0.01;

-- Add Kelly details to execution_logs for audit trail
ALTER TABLE execution_logs
    ADD COLUMN IF NOT EXISTS kelly_full DECIMAL(6,4),
    ADD COLUMN IF NOT EXISTS kelly_fraction_used DECIMAL(6,4),
    ADD COLUMN IF NOT EXISTS kelly_position_pct DECIMAL(6,4),
    ADD COLUMN IF NOT EXISTS kelly_method VARCHAR(32),
    ADD COLUMN IF NOT EXISTS kelly_reasoning TEXT,
    ADD COLUMN IF NOT EXISTS kelly_capped BOOLEAN DEFAULT false;

-- Add trader performance columns for Kelly calculation if missing
-- These should already exist from Phase 2/3, but ensure they're present
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'trader_performance' AND column_name = 'avg_win_r'
    ) THEN
        ALTER TABLE trader_performance
            ADD COLUMN avg_win_r DECIMAL(8,4),
            ADD COLUMN avg_loss_r DECIMAL(8,4);
    END IF;
END $$;

-- Comment on new columns
COMMENT ON COLUMN execution_config.kelly_enabled IS 'Enable Kelly criterion position sizing';
COMMENT ON COLUMN execution_config.kelly_fraction IS 'Fractional Kelly multiplier (0.25 = quarter Kelly)';
COMMENT ON COLUMN execution_config.kelly_min_episodes IS 'Minimum episodes required for Kelly calculation';
COMMENT ON COLUMN execution_config.kelly_fallback_pct IS 'Fallback position size as fraction of equity';

COMMENT ON COLUMN execution_logs.kelly_full IS 'Full Kelly fraction before scaling';
COMMENT ON COLUMN execution_logs.kelly_fraction_used IS 'Fractional Kelly after scaling';
COMMENT ON COLUMN execution_logs.kelly_position_pct IS 'Final position size as fraction of equity';
COMMENT ON COLUMN execution_logs.kelly_method IS 'Method used: kelly, fallback_insufficient_data, fallback_negative_ev';
COMMENT ON COLUMN execution_logs.kelly_reasoning IS 'Human-readable explanation of Kelly calculation';
COMMENT ON COLUMN execution_logs.kelly_capped IS 'Whether position was capped by hard limits';
