-- Migration: Add ROI column to Alpha Pool
--
-- Adds roi_30d column for quality filtering (min 1% ROI requirement)

ALTER TABLE alpha_pool_addresses
    ADD COLUMN IF NOT EXISTS roi_30d NUMERIC;

COMMENT ON COLUMN alpha_pool_addresses.roi_30d IS '30-day ROI as decimal (0.01 = 1%)';
