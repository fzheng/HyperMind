-- Migration 013: Upgrade trader performance to Normal-Inverse-Gamma (NIG) posterior
--
-- Background:
-- The Beta distribution is conjugate to Bernoulli (binary outcomes), but we track
-- continuous R-multiples. Using R-weighted fractional updates loses Bayesian semantics.
--
-- NIG is the proper conjugate prior for unknown mean with unknown variance:
-- - μ | σ² ~ N(m, σ²/κ)
-- - σ² ~ InverseGamma(α, β)
--
-- Prior: NIG(m=0, κ=1, α=3, β=1) - weakly informative, finite moments
-- (α ≥ 3 required for finite variance of the posterior predictive)

-- Add NIG parameters to existing table (preserves existing data)
ALTER TABLE trader_performance
  ADD COLUMN IF NOT EXISTS nig_m DOUBLE PRECISION NOT NULL DEFAULT 0.0,
  ADD COLUMN IF NOT EXISTS nig_kappa DOUBLE PRECISION NOT NULL DEFAULT 1.0,
  ADD COLUMN IF NOT EXISTS nig_alpha DOUBLE PRECISION NOT NULL DEFAULT 3.0,
  ADD COLUMN IF NOT EXISTS nig_beta DOUBLE PRECISION NOT NULL DEFAULT 1.0;

-- Add columns for tracking R statistics
ALTER TABLE trader_performance
  ADD COLUMN IF NOT EXISTS avg_r DOUBLE PRECISION DEFAULT 0.0,
  ADD COLUMN IF NOT EXISTS r_variance DOUBLE PRECISION DEFAULT 0.0,
  ADD COLUMN IF NOT EXISTS correlation_group INT DEFAULT NULL;

-- Index for NIG-based ranking (posterior mean)
CREATE INDEX IF NOT EXISTS trader_performance_nig_mean_idx
  ON trader_performance (nig_m DESC)
  WHERE total_signals >= 30;

-- Comments
COMMENT ON COLUMN trader_performance.nig_m IS 'NIG posterior mean of μ (expected R)';
COMMENT ON COLUMN trader_performance.nig_kappa IS 'NIG precision scaling (effective sample size)';
COMMENT ON COLUMN trader_performance.nig_alpha IS 'NIG shape parameter for variance (≥3 for finite moments)';
COMMENT ON COLUMN trader_performance.nig_beta IS 'NIG rate parameter for variance';
COMMENT ON COLUMN trader_performance.avg_r IS 'Rolling average R-multiple';
COMMENT ON COLUMN trader_performance.r_variance IS 'Rolling variance of R-multiples';
COMMENT ON COLUMN trader_performance.correlation_group IS 'Cluster ID for correlated traders';
