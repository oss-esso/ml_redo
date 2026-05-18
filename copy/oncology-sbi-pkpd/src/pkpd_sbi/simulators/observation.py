"""
Observation model for tumor volume measurements.

LaTeX contract:
    log y_{ij} = log V_i(t_{ij}; theta_i, u_i) + epsilon_{ij}
    epsilon_{ij} ~ N(0, sigma_obs^2)

Clinical contract:
    Tumor measurements are CT-derived longest diameter sums (SLD).
    Measurement noise is approximately log-normal with CV ~ 10-20%.
"""

import numpy as np


def observe_lognormal(
    true_volume: np.ndarray,
    sigma_obs: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Log-normal observation model.

    Args:
        true_volume: True tumor volumes from simulator.
        sigma_obs: Standard deviation of log-normal noise.
        rng: Random number generator.

    Returns:
        Noisy observed tumor volumes.
    """
    safe = np.maximum(np.asarray(true_volume, dtype=float), 1e-8)
    noise = rng.normal(0.0, sigma_obs, size=safe.shape)
    return np.exp(np.log(safe) + noise)


def log_observe(
    true_volume: np.ndarray,
    sigma_obs: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return log-scale noisy observations directly."""
    safe = np.maximum(np.asarray(true_volume, dtype=float), 1e-8)
    noise = rng.normal(0.0, sigma_obs, size=safe.shape)
    return np.log(safe) + noise


def relative_to_baseline(values: np.ndarray) -> np.ndarray:
    """Normalize tumor volumes to baseline (first measurement)."""
    values = np.asarray(values, dtype=float)
    baseline = max(values[0], 1e-8)
    return values / baseline
