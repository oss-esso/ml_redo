"""
Simulation-Based Calibration (SBC).

LaTeX contract (mandatory validation):
    1. Sample theta^(m) ~ p(theta)
    2. Simulate x^(m) ~ p(x | theta^(m))
    3. Sample theta_tilde^(m,1:K) ~ q_phi(theta | x^(m))
    4. Compute rank of true parameter among posterior samples
    5. Check whether ranks are uniform

Reference: Talts et al., "Validating Bayesian inference algorithms
with simulation-based calibration," arXiv:1804.06788, 2018.

If ranks are not uniform, the posterior is miscalibrated.
The failure mode should be reported, not hidden.
"""

import numpy as np
from typing import Optional
from pathlib import Path


def compute_sbc_ranks(
    true_theta: np.ndarray,
    posterior_samples_list: list[np.ndarray],
) -> np.ndarray:
    """
    Compute SBC ranks for each parameter dimension.

    Args:
        true_theta: (n_sbc, d_theta) true parameter values used to generate data.
        posterior_samples_list: List of n_sbc arrays, each (K, d_theta) posterior samples.

    Returns:
        ranks: (n_sbc, d_theta) rank of true value among posterior samples.
    """
    n_sbc = len(posterior_samples_list)
    d_theta = true_theta.shape[1]
    ranks = np.zeros((n_sbc, d_theta), dtype=int)

    for i in range(n_sbc):
        samples = posterior_samples_list[i]  # (K, d_theta)
        for d in range(d_theta):
            ranks[i, d] = int(np.sum(samples[:, d] < true_theta[i, d]))

    return ranks


def check_uniformity(
    ranks: np.ndarray,
    n_posterior_samples: int,
    alpha: float = 0.01,
) -> dict:
    """
    Check whether SBC ranks are consistent with uniformity.

    Uses the Kolmogorov-Smirnov test on each parameter dimension.

    Args:
        ranks: (n_sbc, d_theta) rank array.
        n_posterior_samples: K (number of posterior samples per SBC trial).
        alpha: Significance level.

    Returns:
        Dict with per-parameter results.
    """
    from scipy.stats import kstest

    n_sbc, d_theta = ranks.shape
    results = {}

    for d in range(d_theta):
        # Normalize ranks to [0, 1]
        normalized = ranks[:, d] / (n_posterior_samples + 1)
        stat, pvalue = kstest(normalized, "uniform")
        results[d] = {
            "ks_statistic": float(stat),
            "p_value": float(pvalue),
            "passes": pvalue > alpha,
            "mean_rank": float(np.mean(ranks[:, d])),
            "expected_mean_rank": n_posterior_samples / 2.0,
        }

    return results


def run_sbc(
    posterior,
    prior,
    simulator_fn,
    n_sbc: int = 500,
    n_posterior_samples: int = 1000,
    seed: int = 99,
    param_names: Optional[list[str]] = None,
) -> dict:
    """
    Full SBC pipeline.

    Args:
        posterior: Trained sbi posterior.
        prior: sbi prior (for sampling true theta).
        simulator_fn: Callable(theta_log) -> x observation vector.
        n_sbc: Number of SBC trials.
        n_posterior_samples: Posterior samples per trial.
        seed: Random seed.
        param_names: Optional parameter names for reporting.

    Returns:
        Dict with ranks, uniformity results, and diagnostics.
    """
    import torch

    rng = np.random.default_rng(seed)

    # Sample true parameters from prior
    true_theta_tensor = prior.sample((n_sbc,))
    true_theta = true_theta_tensor.numpy()

    posterior_samples_list = []
    valid_mask = np.ones(n_sbc, dtype=bool)

    for i in range(n_sbc):
        theta_log_i = true_theta[i]

        # Simulate observation
        x_i = simulator_fn(theta_log_i)

        if np.any(np.isnan(x_i)):
            valid_mask[i] = False
            posterior_samples_list.append(np.zeros((n_posterior_samples, true_theta.shape[1])))
            continue

        # Sample from posterior
        x_tensor = torch.tensor(x_i, dtype=torch.float32)
        try:
            samples = posterior.sample((n_posterior_samples,), x=x_tensor).numpy()
        except Exception:
            valid_mask[i] = False
            posterior_samples_list.append(np.zeros((n_posterior_samples, true_theta.shape[1])))
            continue

        posterior_samples_list.append(samples)

        if (i + 1) % 100 == 0:
            print(f"  SBC: {i+1}/{n_sbc}")

    # Filter valid
    true_theta_valid = true_theta[valid_mask]
    samples_valid = [s for s, m in zip(posterior_samples_list, valid_mask) if m]

    print(f"SBC completed: {len(samples_valid)}/{n_sbc} valid trials")

    ranks = compute_sbc_ranks(true_theta_valid, samples_valid)
    uniformity = check_uniformity(ranks, n_posterior_samples)

    # Add param names
    if param_names:
        uniformity_named = {param_names[d]: v for d, v in uniformity.items()}
    else:
        uniformity_named = uniformity

    return {
        "ranks": ranks,
        "true_theta": true_theta_valid,
        "n_valid": len(samples_valid),
        "n_attempted": n_sbc,
        "uniformity": uniformity_named,
        "all_pass": all(v["passes"] for v in uniformity.values()),
    }
