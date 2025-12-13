-- Exchange API Configuration
-- Phase 4.2: Real Trade Execution
--
-- Adds exchange-specific configuration for real order placement
-- and order tracking fields to execution_logs.

-- Add exchange configuration columns to execution_config
ALTER TABLE execution_config
    ADD COLUMN IF NOT EXISTS hl_private_key_env VARCHAR(64) DEFAULT 'HL_PRIVATE_KEY',
    ADD COLUMN IF NOT EXISTS hl_subaccount INT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS hl_slippage_pct DECIMAL(5,3) DEFAULT 0.50,
    ADD COLUMN IF NOT EXISTS real_execution_enabled BOOLEAN DEFAULT false;

-- Add order tracking columns to execution_logs
ALTER TABLE execution_logs
    ADD COLUMN IF NOT EXISTS order_id VARCHAR(128),
    ADD COLUMN IF NOT EXISTS order_type VARCHAR(16) DEFAULT 'market',
    ADD COLUMN IF NOT EXISTS slippage_actual DECIMAL(6,4),
    ADD COLUMN IF NOT EXISTS limit_price DECIMAL(20,8),
    ADD COLUMN IF NOT EXISTS is_reduce_only BOOLEAN DEFAULT false;

-- Comment on new columns
COMMENT ON COLUMN execution_config.hl_private_key_env IS 'Environment variable name containing private key';
COMMENT ON COLUMN execution_config.hl_subaccount IS 'Hyperliquid subaccount index (0 = main)';
COMMENT ON COLUMN execution_config.hl_slippage_pct IS 'Slippage tolerance for market orders (%)';
COMMENT ON COLUMN execution_config.real_execution_enabled IS 'Master switch for real execution (requires env var too)';

COMMENT ON COLUMN execution_logs.order_id IS 'Hyperliquid order ID for tracking';
COMMENT ON COLUMN execution_logs.order_type IS 'Order type: market, limit';
COMMENT ON COLUMN execution_logs.slippage_actual IS 'Actual slippage from mid price';
COMMENT ON COLUMN execution_logs.limit_price IS 'Limit price used for market order';
COMMENT ON COLUMN execution_logs.is_reduce_only IS 'Whether order was reduce-only (position close)';
