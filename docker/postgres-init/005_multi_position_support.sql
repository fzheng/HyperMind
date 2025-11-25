-- Migration: Support multiple positions per address
-- Allow traders to hold BTC, ETH, and other positions simultaneously

-- Drop the old primary key constraint
ALTER TABLE hl_current_positions DROP CONSTRAINT IF EXISTS hl_current_positions_pkey;

-- Create composite primary key on (address, symbol)
ALTER TABLE hl_current_positions ADD PRIMARY KEY (address, symbol);

-- Add index for faster lookups by address
CREATE INDEX IF NOT EXISTS hl_current_positions_address_idx ON hl_current_positions (address);
