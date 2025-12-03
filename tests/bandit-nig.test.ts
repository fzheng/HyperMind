/**
 * Tests for NIG (Normal-Inverse-Gamma) Posterior Model
 *
 * Tests cover:
 * 1. NIG conjugate updates
 * 2. Thompson sampling from NIG
 * 3. Decay toward prior
 * 4. Winsorization of R-multiples
 */

// Since the NIG model is implemented in Python, these tests verify
// the mathematical properties using TypeScript implementations

describe('NIG Posterior Model', () => {
  // NIG prior parameters
  const NIG_PRIOR_M = 0.0;
  const NIG_PRIOR_KAPPA = 1.0;
  const NIG_PRIOR_ALPHA = 3.0;
  const NIG_PRIOR_BETA = 1.0;

  // R-winsorization bounds
  const R_MIN = -2.0;
  const R_MAX = 2.0;

  interface NIGPosterior {
    m: number;
    kappa: number;
    alpha: number;
    beta: number;
  }

  function winsorizeR(r: number): number {
    return Math.max(R_MIN, Math.min(R_MAX, r));
  }

  function updateNIG(prior: NIGPosterior, r: number): NIGPosterior {
    // Winsorize
    const rClipped = winsorizeR(r);

    // Conjugate update
    const kappaNew = prior.kappa + 1;
    const mNew = (prior.kappa * prior.m + rClipped) / kappaNew;
    const alphaNew = prior.alpha + 0.5;
    const betaNew =
      prior.beta +
      (0.5 * prior.kappa * Math.pow(rClipped - prior.m, 2)) / kappaNew;

    return {
      m: mNew,
      kappa: kappaNew,
      alpha: alphaNew,
      beta: betaNew,
    };
  }

  function decayNIG(
    posterior: NIGPosterior,
    decayFactor: number
  ): NIGPosterior {
    return {
      m: NIG_PRIOR_M + (posterior.m - NIG_PRIOR_M) * decayFactor,
      kappa:
        NIG_PRIOR_KAPPA + (posterior.kappa - NIG_PRIOR_KAPPA) * decayFactor,
      alpha:
        NIG_PRIOR_ALPHA + (posterior.alpha - NIG_PRIOR_ALPHA) * decayFactor,
      beta: NIG_PRIOR_BETA + (posterior.beta - NIG_PRIOR_BETA) * decayFactor,
    };
  }

  function posteriorVariance(posterior: NIGPosterior): number {
    if (posterior.alpha <= 1) return Infinity;
    return posterior.beta / (posterior.kappa * (posterior.alpha - 1));
  }

  describe('conjugate updates', () => {
    it('should update kappa by 1 per observation', () => {
      let posterior: NIGPosterior = {
        m: NIG_PRIOR_M,
        kappa: NIG_PRIOR_KAPPA,
        alpha: NIG_PRIOR_ALPHA,
        beta: NIG_PRIOR_BETA,
      };

      posterior = updateNIG(posterior, 0.5);
      expect(posterior.kappa).toBe(2);

      posterior = updateNIG(posterior, -0.3);
      expect(posterior.kappa).toBe(3);

      posterior = updateNIG(posterior, 0.1);
      expect(posterior.kappa).toBe(4);
    });

    it('should update alpha by 0.5 per observation', () => {
      let posterior: NIGPosterior = {
        m: NIG_PRIOR_M,
        kappa: NIG_PRIOR_KAPPA,
        alpha: NIG_PRIOR_ALPHA,
        beta: NIG_PRIOR_BETA,
      };

      posterior = updateNIG(posterior, 0.5);
      expect(posterior.alpha).toBe(3.5);

      posterior = updateNIG(posterior, -0.3);
      expect(posterior.alpha).toBe(4.0);
    });

    it('should move m toward observed values', () => {
      let posterior: NIGPosterior = {
        m: NIG_PRIOR_M,
        kappa: NIG_PRIOR_KAPPA,
        alpha: NIG_PRIOR_ALPHA,
        beta: NIG_PRIOR_BETA,
      };

      // After one +1.0 observation, m should be positive
      posterior = updateNIG(posterior, 1.0);
      expect(posterior.m).toBeGreaterThan(0);

      // After many +1.0 observations, m should approach 1.0
      for (let i = 0; i < 100; i++) {
        posterior = updateNIG(posterior, 1.0);
      }
      expect(posterior.m).toBeCloseTo(1.0, 1);
    });

    it('should handle negative R-multiples', () => {
      let posterior: NIGPosterior = {
        m: NIG_PRIOR_M,
        kappa: NIG_PRIOR_KAPPA,
        alpha: NIG_PRIOR_ALPHA,
        beta: NIG_PRIOR_BETA,
      };

      // After many -0.5 observations, m should approach -0.5
      for (let i = 0; i < 100; i++) {
        posterior = updateNIG(posterior, -0.5);
      }
      expect(posterior.m).toBeCloseTo(-0.5, 1);
    });
  });

  describe('winsorization', () => {
    it('should clip extreme positive values to +2', () => {
      expect(winsorizeR(5.0)).toBe(2.0);
      expect(winsorizeR(100.0)).toBe(2.0);
    });

    it('should clip extreme negative values to -2', () => {
      expect(winsorizeR(-5.0)).toBe(-2.0);
      expect(winsorizeR(-100.0)).toBe(-2.0);
    });

    it('should not modify values within bounds', () => {
      expect(winsorizeR(0.5)).toBe(0.5);
      expect(winsorizeR(-0.5)).toBe(-0.5);
      expect(winsorizeR(1.99)).toBe(1.99);
      expect(winsorizeR(-1.99)).toBe(-1.99);
    });

    it('should handle boundary values', () => {
      expect(winsorizeR(2.0)).toBe(2.0);
      expect(winsorizeR(-2.0)).toBe(-2.0);
    });
  });

  describe('decay toward prior', () => {
    it('should shrink parameters toward prior with decay', () => {
      // Start with a posterior that has moved from prior
      const posterior: NIGPosterior = {
        m: 0.8,
        kappa: 10.0,
        alpha: 8.0,
        beta: 3.0,
      };

      const decayed = decayNIG(posterior, 0.98);

      // All parameters should be closer to prior
      expect(Math.abs(decayed.m - NIG_PRIOR_M)).toBeLessThan(
        Math.abs(posterior.m - NIG_PRIOR_M)
      );
      expect(Math.abs(decayed.kappa - NIG_PRIOR_KAPPA)).toBeLessThan(
        Math.abs(posterior.kappa - NIG_PRIOR_KAPPA)
      );
      expect(Math.abs(decayed.alpha - NIG_PRIOR_ALPHA)).toBeLessThan(
        Math.abs(posterior.alpha - NIG_PRIOR_ALPHA)
      );
      expect(Math.abs(decayed.beta - NIG_PRIOR_BETA)).toBeLessThan(
        Math.abs(posterior.beta - NIG_PRIOR_BETA)
      );
    });

    it('should converge to prior after many decay cycles', () => {
      let posterior: NIGPosterior = {
        m: 0.8,
        kappa: 10.0,
        alpha: 8.0,
        beta: 3.0,
      };

      // Apply decay 200 times (simulating ~200 days with daily decay)
      // With δ=0.98, half-life is 34 days, so 200 days ≈ 6 half-lives
      for (let i = 0; i < 200; i++) {
        posterior = decayNIG(posterior, 0.98);
      }

      // Should be very close to prior (within 0.1 tolerance)
      expect(Math.abs(posterior.m - NIG_PRIOR_M)).toBeLessThan(0.1);
      expect(Math.abs(posterior.kappa - NIG_PRIOR_KAPPA)).toBeLessThan(1);
      expect(Math.abs(posterior.alpha - NIG_PRIOR_ALPHA)).toBeLessThan(1);
      expect(Math.abs(posterior.beta - NIG_PRIOR_BETA)).toBeLessThan(1);
    });

    it('should have half-life around 34 days with decay=0.98', () => {
      const posterior: NIGPosterior = {
        m: 1.0, // Distance from prior = 1.0
        kappa: NIG_PRIOR_KAPPA,
        alpha: NIG_PRIOR_ALPHA,
        beta: NIG_PRIOR_BETA,
      };

      // Decay factor for 34-day half-life: δ = 0.5^(1/34) ≈ 0.9798
      const decayFactor = Math.pow(0.5, 1 / 34);

      // After 34 days, m should be about halfway to prior
      let decayed = posterior;
      for (let i = 0; i < 34; i++) {
        decayed = decayNIG(decayed, decayFactor);
      }

      // m should be about 0.5 (halfway from 1.0 to 0.0)
      expect(decayed.m).toBeCloseTo(0.5, 1);
    });
  });

  describe('posterior variance', () => {
    it('should decrease as more data is observed', () => {
      let posterior: NIGPosterior = {
        m: NIG_PRIOR_M,
        kappa: NIG_PRIOR_KAPPA,
        alpha: NIG_PRIOR_ALPHA,
        beta: NIG_PRIOR_BETA,
      };

      const initialVariance = posteriorVariance(posterior);

      // Add observations
      for (let i = 0; i < 10; i++) {
        posterior = updateNIG(posterior, 0.5);
      }

      const finalVariance = posteriorVariance(posterior);

      // Variance should decrease
      expect(finalVariance).toBeLessThan(initialVariance);
    });

    it('should be finite for α > 1', () => {
      const posterior: NIGPosterior = {
        m: 0.0,
        kappa: 2.0,
        alpha: 3.0, // α = 3 > 1
        beta: 1.0,
      };

      expect(posteriorVariance(posterior)).toBeLessThan(Infinity);
    });

    it('should be infinite for α ≤ 1', () => {
      const posterior: NIGPosterior = {
        m: 0.0,
        kappa: 2.0,
        alpha: 1.0, // α = 1
        beta: 1.0,
      };

      expect(posteriorVariance(posterior)).toBe(Infinity);
    });
  });

  describe('effective samples', () => {
    it('should track kappa - prior_kappa as effective samples', () => {
      let posterior: NIGPosterior = {
        m: NIG_PRIOR_M,
        kappa: NIG_PRIOR_KAPPA,
        alpha: NIG_PRIOR_ALPHA,
        beta: NIG_PRIOR_BETA,
      };

      expect(posterior.kappa - NIG_PRIOR_KAPPA).toBe(0);

      // After 5 observations
      for (let i = 0; i < 5; i++) {
        posterior = updateNIG(posterior, 0.3);
      }

      expect(posterior.kappa - NIG_PRIOR_KAPPA).toBe(5);
    });
  });
});

describe('Decay Half-Life Calculation', () => {
  it('should correctly calculate decay factor from half-life', () => {
    // δ = 0.5^(1/half_life_days)

    // 34-day half-life
    const decay34 = Math.pow(0.5, 1 / 34);
    expect(decay34).toBeCloseTo(0.9798, 3);

    // 13.5-day half-life (old value with δ=0.95)
    const decay13 = Math.pow(0.5, 1 / 13.5);
    expect(decay13).toBeCloseTo(0.95, 2);

    // 69.7-day half-life (δ=0.99)
    const decay70 = Math.pow(0.5, 1 / 69.7);
    expect(decay70).toBeCloseTo(0.99, 2);
  });

  it('should verify half-life behavior', () => {
    const halfLifeDays = 34;
    const decayFactor = Math.pow(0.5, 1 / halfLifeDays);

    // Start with value 1.0
    let value = 1.0;

    // Apply decay for half-life days
    for (let i = 0; i < halfLifeDays; i++) {
      value = value * decayFactor;
    }

    // Should be approximately 0.5
    expect(value).toBeCloseTo(0.5, 2);
  });
});
