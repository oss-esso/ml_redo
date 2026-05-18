"""
Biologically calibrated prior distributions for Simeoni PKPD parameters.

LaTeX contract:
    log theta_i ~ N(mu, Sigma)

Literature calibration:
    - lam0 ~ 0.005-0.05 /day  (Claret 2009, Simeoni 2004)
    - lam1 ~ 2000-20000 mm^3/day
    - psi_g ~ 0.5-2.0 (shape, often fixed to 1 = Gompertz limit)
    - k_kill ~ 0.001-0.02 /day/conc (drug-dependent)
    - k_tr ~ 0.03-0.15 /day (transit half-life ~ 5-25 days)
    - v0 ~ 20-200 mm^3 (baseline SLD-equivalent)

All priors are log-normal (parameters must be positive).
We define them on log scale for SBI: theta_log ~ Uniform(low, high).
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .simeoni import PARAM_NAMES, N_PARAMS, SimeoniParams


@dataclass(frozen=True)
class PriorBounds:
    """Bounds on log-scale parameters for SBI BoxUniform prior."""
    log_low: np.ndarray
    log_high: np.ndarray

    @property
    def n_params(self) -> int:
        return len(self.log_low)

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample n parameter vectors on natural scale."""
        if rng is None:
            rng = np.random.default_rng()
        log_samples = rng.uniform(self.log_low, self.log_high, size=(n, self.n_params))
        return np.exp(log_samples)

    def sample_one(self, rng: np.random.Generator) -> SimeoniParams:
        """Sample a single patient's parameters."""
        arr = self.sample(1, rng)[0]
        return SimeoniParams.from_array(arr)

    def to_sbi_prior(self):
        """Convert to sbi-compatible BoxUniform prior on log scale."""
        import torch
        from sbi.utils import BoxUniform
        return BoxUniform(
            low=torch.tensor(self.log_low, dtype=torch.float32),
            high=torch.tensor(self.log_high, dtype=torch.float32),
        )


# Default prior: calibrated from Simeoni 2004 and Claret 2009 ranges
DEFAULT_PRIOR = PriorBounds(
    # On log scale: [log(lower_bound), log(upper_bound)]
    log_low=np.array([
        np.log(0.003),   # lam0: slow-growing
        np.log(1000.0),  # lam1: low saturation
        np.log(0.5),     # psi_g: flatter growth
        np.log(0.0005),  # k_kill: weak drug effect
        np.log(0.02),    # k_tr: slow transit
        np.log(15.0),    # v0: small baseline
    ]),
    log_high=np.array([
        np.log(0.08),     # lam0: fast-growing
        np.log(30000.0),  # lam1: high saturation
        np.log(2.0),      # psi_g: sharper saturation
        np.log(0.05),     # k_kill: strong drug effect
        np.log(0.20),     # k_tr: fast transit
        np.log(300.0),    # v0: large baseline
    ]),
)


def sample_population_params(
    n_patients: int,
    rng: Optional[np.random.Generator] = None,
    prior: PriorBounds = DEFAULT_PRIOR,
) -> np.ndarray:
    """
    Sample a cohort of patient parameter vectors.

    Returns:
        Array of shape (n_patients, N_PARAMS) on natural scale.
    """
    return prior.sample(n_patients, rng)
