"""
Simeoni-style tumor growth inhibition ODE model.

LaTeX contracts implemented:
    dx1/dt = g(x1; lam0, lam1, psi_g) - k_kill * C(t) * x1
    dx2/dt = k_kill * C(t) * x1 - k_tr * x2
    dx3/dt = k_tr * (x2 - x3)
    dx4/dt = k_tr * (x3 - x4)
    V(t)   = x1 + x2 + x3 + x4

    g(x1) = lam0 * x1 / (1 + (lam0/lam1 * x1)^psi_g)^(1/psi_g)

Reference: Simeoni et al., Cancer Research, 64(3), 1094-1101, 2004.
"""

from dataclasses import dataclass, asdict
import numpy as np
import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5, Kvaerno5
from typing import Optional

import torch
import torch.distributions as D

from .dosing import DoseSchedule


@dataclass(frozen=True)
class SimeoniParams:
    """Patient-specific PKPD parameters."""
    lam0: float     # exponential growth rate (1/day)
    lam1: float     # linear growth rate (mm^3/day) — saturation term
    psi_g: float    # growth function shape parameter
    k_kill: float   # drug kill rate constant (1/(day * concentration))
    k_tr: float     # transit rate through damage compartments (1/day)
    v0: float       # baseline tumor volume (mm^3 or SLD-equivalent)

    def to_array(self) -> np.ndarray:
        """Flat parameter vector for SBI."""
        return np.array([self.lam0, self.lam1, self.psi_g, self.k_kill, self.k_tr, self.v0])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "SimeoniParams":
        """Reconstruct from flat vector."""
        return cls(lam0=arr[0], lam1=arr[1], psi_g=arr[2], k_kill=arr[3], k_tr=arr[4], v0=arr[5])

    def to_dict(self) -> dict:
        return asdict(self)
    

class SimeoniPrior(D.Distribution):
    def __init__(self, log_low, log_hi):
        self._base = D.Independent(
            D.Uniform(log_low, log_hi),
            reinterpreted_batch_ndims=1
        )
        super().__init__(batch_shape=self._base._batch_shape,
                         event_shape=self._base.event_shape)
    
    def sample(self, sample_shape=torch.Size()):
        return self._base.sample(sample_shape)
    
    def log_prob(self, value):
        return self._base.log_prob(value)

# --- Parameter names and bounds for SBI ---
PARAM_NAMES = ["lam0", "lam1", "psi_g", "k_kill", "k_tr", "v0"]
N_PARAMS = len(PARAM_NAMES)


def growth_term(x1: float, lam0: float, lam1: float, psi_g: float) -> float:
    """
    Saturating growth function.

    LaTeX: g(x1) = lam0 * x1 / (1 + (lam0/lam1 * x1)^psi_g)^(1/psi_g)

    For small x1: approximately exponential (lam0 * x1).
    For large x1: approximately linear (lam1).
    """
    x1 = jnp.maximum(x1, 1e-8)
    ratio = (lam0 / lam1) * x1
    denom = (1.0 + ratio ** psi_g) ** (1.0 / psi_g)
    return lam0 * x1 / denom


def simeoni_rhs(
    t: float,
    x: jnp.ndarray,
    params: SimeoniParams,
    schedule: DoseSchedule,
) -> jnp.ndarray:
    """
    Right-hand side of the Simeoni ODE system.

    x = [x1, x2, x3, x4] where:
        x1 = proliferating tumor
        x2, x3, x4 = damage transit compartments
    """
    x1, x2, x3, x4 = jnp.maximum(x, 0.0)

    c_t = schedule.concentration(t)
    grow = growth_term(x1, params.lam0, params.lam1, params.psi_g)
    kill = params.k_kill * c_t * x1
    ktr = params.k_tr

    return jnp.array([
        grow - kill,               # dx1/dt
        kill - ktr * x2,           # dx2/dt
        ktr * (x2 - x3),          # dx3/dt
        ktr * (x3 - x4),          # dx4/dt
    ])


def simulate_tumor_jax(
    params: SimeoniParams,
    schedule: DoseSchedule,
    observation_times: jnp.ndarray,
    method: str = "LSODA",
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> jnp.ndarray:
    """
    Simulate tumor volume trajectory for one patient.

    Args:
        params: Patient-specific Simeoni parameters.
        schedule: Dosing schedule.
        observation_times: Times at which to record tumor volume (days).

    Returns:
        total_volume: Array of tumor volumes at observation_times.

    Raises:
        RuntimeError: If ODE solver fails.
    """
    observation_times = jnp.asarray(observation_times, dtype=float)
    t0 = float(jnp.min(observation_times))
    t1 = float(jnp.max(observation_times))

    # Initial condition: all tumor in proliferating compartment
    x0 = jnp.array([params.v0, 0.0, 0.0, 0.0], dtype=float)

    sol = diffeqsolve(
        solver=Tsit5(),  # Explicit Runge-Kutta method
        terms=ODETerm(lambda t, x, args: simeoni_rhs(t, x, params, schedule)),
        t0=t0,
        t1=t1,
        dt0=0.1,
        y0=x0,
        saveat=SaveAt(ts=observation_times)
    )


    # Total tumor volume = sum of all compartments, clamp to non-negative
    states = jnp.maximum(sol.ys, 0.0)  # shape (n_times, 4)
    total_volume = states.sum(axis=1)   # shape (n_times,)
    return total_volume


def simulate_patient_relative_vector(
    theta: jnp.ndarray,
    schedule: DoseSchedule,
    observation_times: jnp.ndarray,
    sigma_obs: float = 0.10,
    rng: Optional[np.random.Generator] = None,
) -> jnp.ndarray:
    """
    Full forward model returning log-RELATIVE tumor volume: log(V(t)/V(0)) + noise.

    This is dimensionless and directly comparable to log(SLD_t/SLD_baseline)
    from real clinical data, enabling cross-dataset validation.

    Args:
        theta: [lam0, lam1, psi_g, k_kill, k_tr, v0].
        schedule: Dosing schedule.
        observation_times: Scan times (first entry = baseline).
        sigma_obs: Log-normal noise standard deviation.
        rng: Random number generator.

    Returns:
        log_relative_volumes: (n_times,) — NaN vector if ODE fails.
    """
    if rng is None:
        rng = jnp.random.default_rng()

    params = SimeoniParams.from_array(theta)

    try:
        true_volume = simulate_tumor_jax(params, schedule, observation_times)
    except RuntimeError:
        return jnp.full(len(observation_times), jnp.nan)

    v0_sim = jnp.maximum(float(true_volume[0]), 1e-8)
    relative = jnp.maximum(true_volume, 1e-8) / v0_sim
    noise = rng.normal(0.0, sigma_obs, size=relative.shape)
    return jnp.log(relative) + noise


def simulate_patient_vector(
    theta: jnp.ndarray,
    schedule: DoseSchedule,
    observation_times: jnp.ndarray,
    sigma_obs: float = 0.10,
    rng: Optional[np.random.Generator] = None,
) -> jnp.ndarray:
    """
    Full forward model: theta -> noisy observed tumor volumes.

    This is the SBI simulator function: sample theta from prior,
    call this, get the observation vector x.

    Args:
        theta: Parameter vector [lam0, lam1, psi_g, k_kill, k_tr, v0].
        schedule: Dosing schedule.
        observation_times: Scan times.
        sigma_obs: Log-normal noise standard deviation.
        rng: Random number generator.

    Returns:
        log_observed_volumes: Log of noisy tumor volumes at scan times.
    """
    if rng is None:
        rng = np.random.default_rng()

    params = SimeoniParams.from_array(theta)

    try:
        true_volume = simulate_tumor_jax(params, schedule, observation_times)
    except RuntimeError:
        # Return NaN vector if solver fails — SBI will reject this sample
        return jnp.full(len(observation_times), jnp.nan)

    # Log-normal observation model: log y = log V + epsilon
    safe_volume = jnp.maximum(true_volume, 1e-8)
    noise = rng.normal(0.0, sigma_obs, size=safe_volume.shape)
    log_observed = jnp.log(safe_volume) + noise

    return log_observed
