-- Add nickname column to addresses
ALTER TABLE IF EXISTS addresses
ADD COLUMN IF NOT EXISTS nickname TEXT;
