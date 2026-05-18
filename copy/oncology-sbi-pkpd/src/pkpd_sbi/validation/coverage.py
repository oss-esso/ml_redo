"""
Posterior predictive checks and coverage analysis.

LaTeX contract:
    theta^(k) ~ q_phi(theta | x_obs)
    x_tilde^(k) ~ p(x | theta^(k))

    Metrics:
    - Coverage of observed points by posterior predictive intervals
    - RMSE/MAE of posterior predictive median
    - Calibration of 50%, 80%, 95% intervals
"""

import numpy as np
from typing import Optional


def compute_coverage(
    true_values: np.ndarray,
    posterior_samples: np.ndarray,
    levels: list[float] = [0.50, 0.80, 0.90, 0.95],
) -> dict:
    """
    Compute credible interval coverage across parameters.

    Args:
        true_values: (n_test, d_theta) true parameter values.
        posterior_samples: (n_test, K, d_theta) posterior samples per test case.
        levels: Credible levels to check.

    Returns:
        Dict mapping level -> per-parameter coverage fractions.
    """
    n_test, K, d_theta = posterior_samples.shape
    results = {}

    for level in levels:
        alpha = (1 - level) / 2
        lower_q = alpha
        upper_q = 1 - alpha

        coverages = np.zeros(d_theta)
        for d in range(d_theta):
            lower = np.quantile(posterior_samples[:, :, d], lower_q, axis=1)
            upper = np.quantile(posterior_samples[:, :, d], upper_q, axis=1)
            covered = (true_values[:, d] >= lower) & (true_values[:, d] <= upper)
            coverages[d] = np.mean(covered)

        results[level] = coverages

    return results


def posterior_predictive_metrics(
    observed_trajectory: np.ndarray,
    predictive_trajectories: list[np.ndarray],
    quantiles: list[float] = [0.025, 0.25, 0.50, 0.75, 0.975],
) -> dict:
    """
    Compute posterior predictive check metrics for one patient.

    Args:
        observed_trajectory: (n_times,) observed log tumor volumes.
        predictive_trajectories: List of (n_times,) simulated trajectories.
        quantiles: Quantiles to compute.

    Returns:
        Dict with median prediction, RMSE, MAE, quantile bands, coverage.
    """
    if len(predictive_trajectories) == 0:
        return {"error": "No valid predictive trajectories"}

    pred_matrix = np.array(predictive_trajectories)  # (n_traj, n_times)
    n_times = len(observed_trajectory)

    pred_median = np.median(pred_matrix, axis=0)
    pred_mean = np.mean(pred_matrix, axis=0)

    residuals = observed_trajectory - pred_median
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))

    # Quantile bands
    bands = {}
    for q in quantiles:
        bands[q] = np.quantile(pred_matrix, q, axis=0)

    # Point-wise coverage: is observed point within 95% band?
    in_95 = (observed_trajectory >= bands[0.025]) & (observed_trajectory <= bands[0.975])
    coverage_95 = float(np.mean(in_95))

    in_50 = (observed_trajectory >= bands[0.25]) & (observed_trajectory <= bands[0.75])
    coverage_50 = float(np.mean(in_50))

    return {
        "pred_median": pred_median,
        "pred_mean": pred_mean,
        "rmse": rmse,
        "mae": mae,
        "bands": bands,
        "coverage_95": coverage_95,
        "coverage_50": coverage_50,
        "n_predictive_trajectories": len(predictive_trajectories),
    }


def parameter_recovery_metrics(
    true_theta: np.ndarray,
    posterior_median: np.ndarray,
    param_names: Optional[list[str]] = None,
) -> dict:
    """
    Compute parameter recovery accuracy.

    Args:
        true_theta: (n_test, d_theta) true values.
        posterior_median: (n_test, d_theta) posterior median estimates.

    Returns:
        Per-parameter RMSE, MAE, correlation, and bias.
    """
    d_theta = true_theta.shape[1]
    results = {}

    for d in range(d_theta):
        name = param_names[d] if param_names else f"param_{d}"
        residuals = posterior_median[:, d] - true_theta[:, d]
        results[name] = {
            "rmse": float(np.sqrt(np.mean(residuals ** 2))),
            "mae": float(np.mean(np.abs(residuals))),
            "bias": float(np.mean(residuals)),
            "correlation": float(np.corrcoef(true_theta[:, d], posterior_median[:, d])[0, 1]),
        }

    return results
