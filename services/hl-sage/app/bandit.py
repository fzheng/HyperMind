"""
Thompson Sampling Bandit for Trader Selection

Implements a multi-armed bandit algorithm to adaptively select which traders
to follow based on their historical signal performance.

Supports two posterior models:
1. Beta-Bernoulli (v1): Binary win/loss, simple but loses R-magnitude info
2. Normal-Inverse-Gamma (v2): Continuous R-multiples, proper Bayesian semantics

Key concepts:
- Thompson Sampling: sample from posterior, select highest sampled values
- Online learning: posteriors update after each signal outcome
- Exponential decay: handle non-stationarity (traders change over time)

NIG Model (v2):
- μ | σ² ~ N(m, σ²/κ)  - mean R-multiple given variance
- σ² ~ InverseGamma(α, β) - variance of R-multiples
- Prior: NIG(m=0, κ=1, α=3, β=1) - weakly informative, finite moments

## Alpha Pool Integration

The NIG model is used by the Alpha Pool (decoupled from legacy leaderboard).
get_trader_posteriors_nig() pulls addresses from alpha_pool_addresses table,
NOT from hl_leaderboard_entries. This allows the Alpha Pool to operate
independently from hl-scout's daily leaderboard sync.

@module bandit
"""

import math
import os
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import asyncpg

# Configuration
BANDIT_POOL_SIZE = int(os.getenv("BANDIT_POOL_SIZE", "50"))
BANDIT_SELECT_K = int(os.getenv("BANDIT_SELECT_K", "10"))
BANDIT_MIN_SAMPLES = int(os.getenv("BANDIT_MIN_SAMPLES", "30"))  # Increased from 5

# Decay configuration: convert half-life to decay factor
# δ = 0.5^(1/half_life_days)
BANDIT_DECAY_HALF_LIFE_DAYS = float(os.getenv("BANDIT_DECAY_HALF_LIFE_DAYS", "34"))
BANDIT_DECAY_FACTOR = 0.5 ** (1.0 / BANDIT_DECAY_HALF_LIFE_DAYS)  # ~0.98 for 34 days

# NIG Prior parameters (weakly informative, finite moments)
NIG_PRIOR_M = float(os.getenv("NIG_PRIOR_M", "0.0"))
NIG_PRIOR_KAPPA = float(os.getenv("NIG_PRIOR_KAPPA", "1.0"))
NIG_PRIOR_ALPHA = float(os.getenv("NIG_PRIOR_ALPHA", "3.0"))  # ≥3 for finite variance
NIG_PRIOR_BETA = float(os.getenv("NIG_PRIOR_BETA", "1.0"))

# R-multiple winsorization bounds (tame heavy tails)
R_WINSORIZE_MIN = float(os.getenv("R_WINSORIZE_MIN", "-2.0"))
R_WINSORIZE_MAX = float(os.getenv("R_WINSORIZE_MAX", "2.0"))


@dataclass
class TraderPosterior:
    """
    Beta-Bernoulli posterior for Thompson Sampling (v1 - legacy).

    Simple binary win/loss model. Kept for backwards compatibility.
    For new deployments, prefer TraderPosteriorNIG.
    """
    address: str
    alpha: float  # successes + 1
    beta: float   # failures + 1
    total_signals: int
    winning_signals: int
    total_pnl_r: float

    @property
    def posterior_mean(self) -> float:
        """Expected win rate from Beta distribution."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def posterior_variance(self) -> float:
        """Uncertainty in the estimate."""
        ab = self.alpha + self.beta
        return (self.alpha * self.beta) / (ab * ab * (ab + 1))

    def sample(self) -> float:
        """
        Draw a sample from the Beta posterior.
        This is the core of Thompson Sampling - we select traders
        based on random draws, not just expected values.
        """
        # Use Python's random.betavariate for sampling
        # This naturally balances exploration (high variance) vs exploitation (high mean)
        return random.betavariate(self.alpha, self.beta)


@dataclass
class TraderPosteriorNIG:
    """
    Normal-Inverse-Gamma posterior for Thompson Sampling (v2 - recommended).

    Proper conjugate prior for continuous R-multiples with unknown mean and variance.

    Model:
    - μ | σ² ~ N(m, σ²/κ)
    - σ² ~ InverseGamma(α, β)

    Prior: NIG(m=0, κ=1, α=3, β=1)
    - m=0: no prior belief about mean R
    - κ=1: equivalent to 1 pseudo-observation
    - α=3: ensures finite variance of posterior predictive
    - β=1: reasonable scale for R-multiples
    """
    address: str
    m: float = NIG_PRIOR_M          # posterior mean of μ
    kappa: float = NIG_PRIOR_KAPPA  # precision scaling (effective n)
    alpha: float = NIG_PRIOR_ALPHA  # shape for variance
    beta: float = NIG_PRIOR_BETA    # rate for variance
    total_signals: int = 0
    total_pnl_r: float = 0.0

    @property
    def posterior_mean(self) -> float:
        """Expected mean R from posterior."""
        return self.m

    @property
    def posterior_variance(self) -> float:
        """
        Variance of the posterior mean estimate.
        Var(μ) = β / (κ × (α - 1)) for α > 1
        """
        if self.alpha <= 1:
            return float('inf')
        return self.beta / (self.kappa * (self.alpha - 1))

    @property
    def effective_samples(self) -> float:
        """Effective sample size (how much data we have)."""
        return self.kappa - NIG_PRIOR_KAPPA

    def update(self, r: float) -> "TraderPosteriorNIG":
        """
        Conjugate update with a new R observation.

        Args:
            r: R-multiple (will be winsorized to bounds)

        Returns:
            Self (mutated) for chaining
        """
        # Winsorize to tame heavy tails
        r = max(R_WINSORIZE_MIN, min(R_WINSORIZE_MAX, r))

        # Conjugate NIG update formulas
        kappa_new = self.kappa + 1
        m_new = (self.kappa * self.m + r) / kappa_new
        alpha_new = self.alpha + 0.5
        beta_new = self.beta + 0.5 * self.kappa * ((r - self.m) ** 2) / kappa_new

        self.m = m_new
        self.kappa = kappa_new
        self.alpha = alpha_new
        self.beta = beta_new
        self.total_signals += 1
        self.total_pnl_r += r

        return self

    def sample(self) -> float:
        """
        Thompson sample from NIG posterior.

        1. Sample σ² from InverseGamma(α, β)
        2. Sample μ from N(m, σ²/κ)
        3. Return μ (the sampled expected R)

        Returns:
            Sampled mean R-multiple
        """
        # Sample from Inverse-Gamma by inverting Gamma sample
        # If X ~ Gamma(α, β), then 1/X ~ InverseGamma(α, β)
        gamma_sample = random.gammavariate(self.alpha, 1.0 / self.beta)
        sigma2 = 1.0 / gamma_sample if gamma_sample > 0 else 1.0

        # Sample mean from Normal(m, σ²/κ)
        std = math.sqrt(sigma2 / self.kappa) if self.kappa > 0 else 1.0
        mu = random.gauss(self.m, std)

        return mu

    def sample_sharpe(self) -> float:
        """
        Thompson sample returning risk-adjusted μ/σ (Sharpe-like).

        Use this for risk-adjusted selection instead of raw μ.

        Returns:
            Sampled Sharpe-like ratio
        """
        gamma_sample = random.gammavariate(self.alpha, 1.0 / self.beta)
        sigma2 = 1.0 / gamma_sample if gamma_sample > 0 else 1.0
        sigma = math.sqrt(sigma2)

        std = math.sqrt(sigma2 / self.kappa) if self.kappa > 0 else 1.0
        mu = random.gauss(self.m, std)

        return mu / sigma if sigma > 0 else 0.0

    def decay_toward_prior(self, decay_factor: float = BANDIT_DECAY_FACTOR) -> "TraderPosteriorNIG":
        """
        Apply exponential decay toward prior.

        Shrinks all parameters toward their prior values, making
        the posterior "forget" old observations over time.

        Args:
            decay_factor: δ in (0,1), higher = slower decay

        Returns:
            Self (mutated) for chaining
        """
        self.kappa = NIG_PRIOR_KAPPA + (self.kappa - NIG_PRIOR_KAPPA) * decay_factor
        self.m = NIG_PRIOR_M + (self.m - NIG_PRIOR_M) * decay_factor
        self.alpha = NIG_PRIOR_ALPHA + (self.alpha - NIG_PRIOR_ALPHA) * decay_factor
        self.beta = NIG_PRIOR_BETA + (self.beta - NIG_PRIOR_BETA) * decay_factor

        return self


def winsorize_r(r: float) -> float:
    """Winsorize R-multiple to configured bounds."""
    return max(R_WINSORIZE_MIN, min(R_WINSORIZE_MAX, r))


async def get_trader_posteriors(
    pool: asyncpg.Pool,
    limit: int = BANDIT_POOL_SIZE,
    min_signals: int = 0
) -> List[TraderPosterior]:
    """
    Fetch trader posteriors from the database.

    Args:
        pool: Database connection pool
        limit: Maximum number of traders to fetch
        min_signals: Minimum signals required to include trader

    Returns:
        List of TraderPosterior objects
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT address, alpha, beta, total_signals, winning_signals, total_pnl_r
            FROM trader_performance
            WHERE total_signals >= $1
            ORDER BY alpha / (alpha + beta) DESC
            LIMIT $2
            """,
            min_signals,
            limit,
        )

        return [
            TraderPosterior(
                address=row["address"],
                alpha=float(row["alpha"]),
                beta=float(row["beta"]),
                total_signals=int(row["total_signals"]),
                winning_signals=int(row["winning_signals"]),
                total_pnl_r=float(row["total_pnl_r"]),
            )
            for row in rows
        ]


async def get_all_candidate_addresses(
    pool: asyncpg.Pool,
    limit: int = BANDIT_POOL_SIZE
) -> List[str]:
    """
    Get candidate addresses from leaderboard that may not have signal history yet.
    These are potential new traders to explore.

    Args:
        pool: Database connection pool
        limit: Maximum candidates to return

    Returns:
        List of addresses from recent leaderboard entries
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT address
            FROM sage_tracked_addresses
            WHERE updated_at > NOW() - INTERVAL '24 hours'
            ORDER BY updated_at DESC
            LIMIT $1
            """,
            limit,
        )
        return [row["address"].lower() for row in rows]


def thompson_sample_select(
    posteriors: List[TraderPosterior],
    k: int = BANDIT_SELECT_K,
) -> List[TraderPosterior]:
    """
    Select top K traders using Thompson Sampling.

    For each trader, we sample from their Beta posterior and select
    the K traders with highest sampled values. This naturally balances:
    - Exploitation: traders with high posterior mean get selected often
    - Exploration: traders with high uncertainty sometimes get selected

    Args:
        posteriors: List of trader posteriors to sample from
        k: Number of traders to select

    Returns:
        Selected traders (sorted by sampled value, descending)
    """
    if not posteriors:
        return []

    # Sample from each trader's posterior
    samples: List[Tuple[TraderPosterior, float]] = [
        (p, p.sample()) for p in posteriors
    ]

    # Sort by sampled value (descending) and take top k
    samples.sort(key=lambda x: x[1], reverse=True)

    return [p for p, _ in samples[:k]]


async def select_traders_with_exploration(
    pool: asyncpg.Pool,
    k: int = BANDIT_SELECT_K,
    exploration_ratio: float = 0.2,
) -> List[str]:
    """
    Select traders using Thompson Sampling with forced exploration.

    Allocates a portion of selections to new/unexplored traders to ensure
    we don't get stuck in a local optimum.

    Args:
        pool: Database connection pool
        k: Total number of traders to select
        exploration_ratio: Fraction of slots for exploration (0.0-1.0)

    Returns:
        List of selected trader addresses
    """
    # Get traders with signal history
    posteriors = await get_trader_posteriors(pool, limit=BANDIT_POOL_SIZE)
    known_addresses = {p.address for p in posteriors}

    # Get all candidate addresses (including new ones)
    all_candidates = await get_all_candidate_addresses(pool, limit=BANDIT_POOL_SIZE * 2)

    # Identify unexplored traders (in candidates but no signal history)
    unexplored = [addr for addr in all_candidates if addr not in known_addresses]

    # Calculate exploration vs exploitation slots
    explore_slots = max(1, int(k * exploration_ratio)) if unexplored else 0
    exploit_slots = k - explore_slots

    selected: List[str] = []

    # Exploitation: Thompson Sampling from known traders
    if posteriors and exploit_slots > 0:
        exploited = thompson_sample_select(posteriors, k=exploit_slots)
        selected.extend([p.address for p in exploited])

    # Exploration: random selection from unexplored traders
    if unexplored and explore_slots > 0:
        explored = random.sample(unexplored, min(explore_slots, len(unexplored)))
        selected.extend(explored)

    # If we don't have enough, fill with any remaining candidates
    if len(selected) < k:
        remaining = [addr for addr in all_candidates if addr not in set(selected)]
        needed = k - len(selected)
        selected.extend(remaining[:needed])

    return selected[:k]


async def get_bandit_status(pool: asyncpg.Pool) -> dict:
    """
    Get current status of the bandit algorithm for monitoring.

    Returns:
        Dict with bandit statistics
    """
    async with pool.acquire() as conn:
        # Get total traders with any signals
        total_row = await conn.fetchrow(
            "SELECT COUNT(*) as count FROM trader_performance WHERE total_signals > 0"
        )
        total_with_signals = int(total_row["count"]) if total_row else 0

        # Get traders with enough signals for reliable estimates
        reliable_row = await conn.fetchrow(
            "SELECT COUNT(*) as count FROM trader_performance WHERE total_signals >= $1",
            BANDIT_MIN_SAMPLES,
        )
        reliable_count = int(reliable_row["count"]) if reliable_row else 0

        # Get top performers by posterior mean
        top_rows = await conn.fetch(
            """
            SELECT address, alpha, beta, total_signals, winning_signals,
                   alpha / (alpha + beta) as posterior_mean
            FROM trader_performance
            WHERE total_signals >= $1
            ORDER BY posterior_mean DESC
            LIMIT 10
            """,
            BANDIT_MIN_SAMPLES,
        )

        top_traders = [
            {
                "address": row["address"],
                "posterior_mean": float(row["posterior_mean"]),
                "total_signals": int(row["total_signals"]),
                "winning_signals": int(row["winning_signals"]),
                "win_rate": int(row["winning_signals"]) / int(row["total_signals"])
                    if row["total_signals"] > 0 else 0,
            }
            for row in top_rows
        ]

        return {
            "config": {
                "pool_size": BANDIT_POOL_SIZE,
                "select_k": BANDIT_SELECT_K,
                "min_samples": BANDIT_MIN_SAMPLES,
                "decay_factor": BANDIT_DECAY_FACTOR,
            },
            "stats": {
                "total_traders_with_signals": total_with_signals,
                "reliable_traders": reliable_count,
            },
            "top_traders": top_traders,
        }


async def apply_decay(pool: asyncpg.Pool, decay_factor: float = BANDIT_DECAY_FACTOR) -> int:
    """
    Apply exponential decay to all posteriors (both Beta and NIG).

    This makes recent performance more important than old performance,
    helping the bandit adapt to changing trader behavior.

    Beta decay formula:
        alpha_new = 1 + (alpha - 1) * decay_factor
        beta_new = 1 + (beta - 1) * decay_factor

    NIG decay formula:
        kappa_new = κ₀ + (kappa - κ₀) * decay_factor
        m_new = m₀ + (m - m₀) * decay_factor
        alpha_new = α₀ + (nig_alpha - α₀) * decay_factor
        beta_new = β₀ + (nig_beta - β₀) * decay_factor

    Args:
        pool: Database connection pool
        decay_factor: Decay multiplier (0.0-1.0), higher = slower decay

    Returns:
        Number of traders updated
    """
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE trader_performance SET
                -- Beta decay (toward Beta(1,1))
                alpha = 1 + (alpha - 1) * $1,
                beta = 1 + (beta - 1) * $1,
                -- NIG decay (toward prior)
                nig_kappa = $2 + (COALESCE(nig_kappa, $2) - $2) * $1,
                nig_m = $3 + (COALESCE(nig_m, $3) - $3) * $1,
                nig_alpha = $4 + (COALESCE(nig_alpha, $4) - $4) * $1,
                nig_beta = $5 + (COALESCE(nig_beta, $5) - $5) * $1,
                updated_at = NOW()
            WHERE total_signals > 0
            """,
            decay_factor,
            NIG_PRIOR_KAPPA,
            NIG_PRIOR_M,
            NIG_PRIOR_ALPHA,
            NIG_PRIOR_BETA,
        )
        # Extract count from "UPDATE N" result
        count = int(result.split()[-1]) if result else 0
        return count


# ============================================================================
# NIG-specific functions (v2 model) - Used by Alpha Pool
# ============================================================================
#
# These functions support the decoupled Alpha Pool system.
# Key difference from legacy: uses alpha_pool_addresses table, not
# hl_leaderboard_entries. This means:
#   1. Alpha Pool addresses come from /alpha-pool/refresh API
#   2. No dependency on hl-scout's daily leaderboard sync
#   3. Both systems can run side-by-side for comparison
# ============================================================================


async def get_trader_posteriors_nig(
    pool: asyncpg.Pool,
    limit: int = BANDIT_POOL_SIZE,
    min_signals: int = 0
) -> List[TraderPosteriorNIG]:
    """
    Fetch NIG trader posteriors from the database.

    Combines:
    1. Traders with performance history (trader_performance table)
    2. Alpha Pool addresses without performance history yet

    Uses alpha_pool_addresses table only (fully decoupled from legacy).
    New traders get default NIG priors, allowing them to appear in the
    Alpha Pool before they have any closed positions.

    Note: Requires `/alpha-pool/refresh` to be called first to populate
    alpha_pool_addresses. Returns empty list if no addresses in pool.

    Args:
        pool: Database connection pool
        limit: Maximum number of traders to fetch
        min_signals: Minimum signals required to include trader

    Returns:
        List of TraderPosteriorNIG objects
    """
    async with pool.acquire() as conn:
        # Use alpha_pool_addresses only (fully decoupled from legacy leaderboard)
        # Traders with performance are ranked by nig_m, new traders by default prior
        rows = await conn.fetch(
            """
            WITH pool_addresses AS (
                SELECT LOWER(address) as address
                FROM alpha_pool_addresses
                WHERE is_active = true
            ),
            performance_traders AS (
                SELECT tp.address,
                       COALESCE(tp.nig_m, $3) as nig_m,
                       COALESCE(tp.nig_kappa, $4) as nig_kappa,
                       COALESCE(tp.nig_alpha, $5) as nig_alpha,
                       COALESCE(tp.nig_beta, $6) as nig_beta,
                       tp.total_signals,
                       tp.total_pnl_r
                FROM trader_performance tp
                WHERE tp.total_signals >= $1
                  AND tp.address IN (SELECT address FROM pool_addresses)
            ),
            new_pool_traders AS (
                -- Pool addresses without performance history yet
                SELECT pa.address,
                       $3 as nig_m,
                       $4 as nig_kappa,
                       $5 as nig_alpha,
                       $6 as nig_beta,
                       0 as total_signals,
                       0.0 as total_pnl_r
                FROM pool_addresses pa
                WHERE pa.address NOT IN (SELECT address FROM performance_traders)
            )
            SELECT * FROM performance_traders
            UNION ALL
            SELECT * FROM new_pool_traders
            ORDER BY nig_m DESC, total_signals DESC
            LIMIT $2
            """,
            min_signals,
            limit,
            NIG_PRIOR_M,
            NIG_PRIOR_KAPPA,
            NIG_PRIOR_ALPHA,
            NIG_PRIOR_BETA,
        )

        return [
            TraderPosteriorNIG(
                address=row["address"],
                m=float(row["nig_m"]),
                kappa=float(row["nig_kappa"]),
                alpha=float(row["nig_alpha"]),
                beta=float(row["nig_beta"]),
                total_signals=int(row["total_signals"]),
                total_pnl_r=float(row["total_pnl_r"]),
            )
            for row in rows
        ]


def thompson_sample_select_nig(
    posteriors: List[TraderPosteriorNIG],
    k: int = BANDIT_SELECT_K,
    use_sharpe: bool = False,
) -> List[TraderPosteriorNIG]:
    """
    Select top K traders using Thompson Sampling with NIG posteriors.

    Args:
        posteriors: List of NIG trader posteriors
        k: Number of traders to select
        use_sharpe: If True, sample μ/σ instead of μ for risk-adjusted selection

    Returns:
        Selected traders (sorted by sampled value, descending)
    """
    if not posteriors:
        return []

    # Sample from each trader's posterior
    if use_sharpe:
        samples: List[Tuple[TraderPosteriorNIG, float]] = [
            (p, p.sample_sharpe()) for p in posteriors
        ]
    else:
        samples = [(p, p.sample()) for p in posteriors]

    # Sort by sampled value (descending) and take top k
    samples.sort(key=lambda x: x[1], reverse=True)

    return [p for p, _ in samples[:k]]


async def update_trader_nig(
    pool: asyncpg.Pool,
    address: str,
    pnl_r: float,
) -> None:
    """
    Update trader's NIG posterior with a new R observation.

    Performs conjugate NIG update in the database.

    Args:
        pool: Database connection pool
        address: Trader's Ethereum address
        pnl_r: P&L in R-multiples (will be winsorized)
    """
    # Winsorize the R-multiple
    r = winsorize_r(pnl_r)

    async with pool.acquire() as conn:
        # Ensure trader exists with default prior
        await conn.execute(
            """
            INSERT INTO trader_performance (address, nig_m, nig_kappa, nig_alpha, nig_beta)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (address) DO NOTHING
            """,
            address.lower(),
            NIG_PRIOR_M,
            NIG_PRIOR_KAPPA,
            NIG_PRIOR_ALPHA,
            NIG_PRIOR_BETA,
        )

        # Perform conjugate NIG update using CTE for atomic parameter reads
        # This ensures we read the OLD values before any updates
        # κ' = κ + 1
        # m' = (κ × m + r) / κ'
        # α' = α + 0.5
        # β' = β + 0.5 × κ × (r - m)² / κ'
        await conn.execute(
            """
            WITH cur AS (
                SELECT
                    COALESCE(nig_m, $3) AS m0,
                    COALESCE(nig_kappa, $2) AS k0,
                    COALESCE(nig_alpha, $5) AS a0,
                    COALESCE(nig_beta, $6) AS b0,
                    total_signals AS sig0,
                    COALESCE(avg_r, 0) AS avg0
                FROM trader_performance
                WHERE address = $1
                FOR UPDATE
            )
            UPDATE trader_performance t SET
                nig_kappa = cur.k0 + 1,
                nig_m = (cur.k0 * cur.m0 + $4) / (cur.k0 + 1),
                nig_alpha = cur.a0 + 0.5,
                nig_beta = cur.b0 + 0.5 * cur.k0 * POWER($4 - cur.m0, 2) / (cur.k0 + 1),
                total_signals = cur.sig0 + 1,
                total_pnl_r = total_pnl_r + $4,
                avg_r = (cur.avg0 * cur.sig0 + $4) / GREATEST(cur.sig0 + 1, 1),
                last_signal_at = NOW(),
                updated_at = NOW()
            FROM cur
            WHERE t.address = $1
            """,
            address.lower(),
            NIG_PRIOR_KAPPA,
            NIG_PRIOR_M,
            r,
            NIG_PRIOR_ALPHA,
            NIG_PRIOR_BETA,
        )


async def get_bandit_status_nig(pool: asyncpg.Pool) -> dict:
    """
    Get current status of the NIG bandit algorithm for monitoring.

    Returns:
        Dict with NIG bandit statistics
    """
    async with pool.acquire() as conn:
        # Get total traders with any signals
        total_row = await conn.fetchrow(
            "SELECT COUNT(*) as count FROM trader_performance WHERE total_signals > 0"
        )
        total_with_signals = int(total_row["count"]) if total_row else 0

        # Get traders with enough signals for reliable estimates
        reliable_row = await conn.fetchrow(
            "SELECT COUNT(*) as count FROM trader_performance WHERE total_signals >= $1",
            BANDIT_MIN_SAMPLES,
        )
        reliable_count = int(reliable_row["count"]) if reliable_row else 0

        # Get top performers by NIG posterior mean
        top_rows = await conn.fetch(
            """
            SELECT address,
                   COALESCE(nig_m, 0) as nig_m,
                   COALESCE(nig_kappa, 1) as nig_kappa,
                   COALESCE(nig_alpha, 3) as nig_alpha,
                   COALESCE(nig_beta, 1) as nig_beta,
                   total_signals,
                   total_pnl_r,
                   COALESCE(avg_r, 0) as avg_r
            FROM trader_performance
            WHERE total_signals >= $1
            ORDER BY COALESCE(nig_m, 0) DESC
            LIMIT 10
            """,
            BANDIT_MIN_SAMPLES,
        )

        top_traders = [
            {
                "address": row["address"],
                "nig_m": float(row["nig_m"]),
                "nig_kappa": float(row["nig_kappa"]),
                "effective_samples": float(row["nig_kappa"]) - NIG_PRIOR_KAPPA,
                "total_signals": int(row["total_signals"]),
                "total_pnl_r": float(row["total_pnl_r"]),
                "avg_r": float(row["avg_r"]),
            }
            for row in top_rows
        ]

        return {
            "model": "NIG",
            "config": {
                "pool_size": BANDIT_POOL_SIZE,
                "select_k": BANDIT_SELECT_K,
                "min_samples": BANDIT_MIN_SAMPLES,
                "decay_factor": BANDIT_DECAY_FACTOR,
                "decay_half_life_days": BANDIT_DECAY_HALF_LIFE_DAYS,
                "prior": {
                    "m": NIG_PRIOR_M,
                    "kappa": NIG_PRIOR_KAPPA,
                    "alpha": NIG_PRIOR_ALPHA,
                    "beta": NIG_PRIOR_BETA,
                },
                "r_winsorize_bounds": [R_WINSORIZE_MIN, R_WINSORIZE_MAX],
            },
            "stats": {
                "total_traders_with_signals": total_with_signals,
                "reliable_traders": reliable_count,
            },
            "top_traders": top_traders,
        }
