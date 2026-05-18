"""
Plotting utilities for paper figures.

Main figures from LaTeX doc:
    1. Schematic of simulator and amortized inference pipeline
    2. Posterior recovery on synthetic patients
    3. SBC rank histograms for individual NPE
    4. Comparison against MCMC/NLME baseline
    5. Hierarchical vs individual NPE under sparse observations
    6. Posterior predictive checks on real or example data
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional


def plot_synthetic_trajectories(
    obs_df,
    n_patients: int = 20,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot observed tumor burden trajectories relative to baseline."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for pid, group in obs_df.groupby("patient_id"):
        if pid >= n_patients:
            break
        ax.plot(
            group["time_days"],
            group["relative_volume"],
            marker="o", markersize=3, alpha=0.6, linewidth=1,
        )

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="Baseline")
    ax.axhline(0.7, color="green", linestyle=":", linewidth=0.8, alpha=0.5, label="PR threshold (−30%)")
    ax.axhline(1.2, color="red", linestyle=":", linewidth=0.8, alpha=0.5, label="PD threshold (+20%)")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Tumor burden / baseline")
    ax.set_title(f"Synthetic tumor trajectories (n={n_patients})")
    ax.legend(fontsize=8)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_sbc_ranks(
    ranks: np.ndarray,
    n_posterior_samples: int,
    param_names: Optional[list[str]] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot SBC rank histograms.

    Uniform histograms = calibrated posterior.
    U-shaped = overconfident. Inverted-U = underconfident.
    Skewed = biased.
    """
    n_sbc, d_theta = ranks.shape
    n_bins = min(20, n_posterior_samples // 10)

    n_cols = min(3, d_theta)
    n_rows = (d_theta + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if d_theta == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    expected_count = n_sbc / n_bins

    for d in range(d_theta):
        ax = axes[d]
        name = param_names[d] if param_names else f"θ_{d}"
        ax.hist(ranks[:, d], bins=n_bins, density=False, alpha=0.7, color="steelblue", edgecolor="white")
        ax.axhline(expected_count, color="red", linestyle="--", linewidth=1, label="Expected (uniform)")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Rank")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)

    for d in range(d_theta, len(axes)):
        axes[d].set_visible(False)

    fig.suptitle("SBC Rank Histograms", fontsize=12, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_parameter_recovery(
    true_theta: np.ndarray,
    posterior_median: np.ndarray,
    param_names: Optional[list[str]] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Scatter plot of true vs estimated parameters (log scale)."""
    d_theta = true_theta.shape[1]
    n_cols = min(3, d_theta)
    n_rows = (d_theta + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if d_theta == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for d in range(d_theta):
        ax = axes[d]
        name = param_names[d] if param_names else f"θ_{d}"
        ax.scatter(true_theta[:, d], posterior_median[:, d], alpha=0.3, s=10)
        lims = [
            min(true_theta[:, d].min(), posterior_median[:, d].min()),
            max(true_theta[:, d].max(), posterior_median[:, d].max()),
        ]
        ax.plot(lims, lims, "r--", linewidth=1, label="y=x")
        r = np.corrcoef(true_theta[:, d], posterior_median[:, d])[0, 1]
        ax.set_title(f"{name}  (r={r:.3f})", fontsize=10)
        ax.set_xlabel("True (log)")
        ax.set_ylabel("Estimated (log)")
        ax.legend(fontsize=7)

    for d in range(d_theta, len(axes)):
        axes[d].set_visible(False)

    fig.suptitle("Parameter Recovery", fontsize=12, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_posterior_predictive(
    obs_times: np.ndarray,
    observed: np.ndarray,
    bands: dict,
    pred_median: np.ndarray,
    patient_id: int = 0,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot posterior predictive bands vs observed data for one patient."""
    fig, ax = plt.subplots(figsize=(7, 4))

    # 95% band
    ax.fill_between(obs_times, bands[0.025], bands[0.975],
                     alpha=0.15, color="steelblue", label="95% PI")
    # 50% band
    ax.fill_between(obs_times, bands[0.25], bands[0.75],
                     alpha=0.3, color="steelblue", label="50% PI")
    # Median prediction
    ax.plot(obs_times, pred_median, color="steelblue", linewidth=1.5, label="Median prediction")
    # Observed
    ax.scatter(obs_times, observed, color="black", zorder=5, s=30, label="Observed")

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Log tumor volume")
    ax.set_title(f"Posterior Predictive Check — Patient {patient_id}")
    ax.legend(fontsize=8)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig
