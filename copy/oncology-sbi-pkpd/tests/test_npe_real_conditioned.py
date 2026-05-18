"""
Smoke tests for train_npe_real_conditioned.py — no training, no SAS files.

Tests cover:
    - _generate_flat_training_data produces correct shapes
    - _simulate_rel_sld runs without error and returns rel_sld[0] ≈ 1.0
    - observation_tensor_for_npe integrates with fake patient obs
    - plot helpers produce figures without error
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")

from pkpd_sbi.experiments.train_npe_real_conditioned import (
    _generate_flat_training_data,
    _simulate_rel_sld,
    plot_real_ppc,
)
from pkpd_sbi.inference.summaries import MAX_TIMEPOINTS, observation_tensor_for_npe


# ---------------------------------------------------------------------------
# _generate_flat_training_data
# ---------------------------------------------------------------------------

def test_generate_flat_training_data_shapes() -> None:
    theta, x = _generate_flat_training_data(n_simulations=20, seed=0)
    assert theta.ndim == 2
    assert x.ndim == 2
    assert theta.shape[1] == 6                  # N_PARAMS
    assert x.shape[1] == 3 * MAX_TIMEPOINTS     # 72
    assert theta.shape[0] == x.shape[0]
    assert theta.shape[0] >= 15                 # at least 75% success expected


def test_generate_flat_training_data_finite() -> None:
    theta, x = _generate_flat_training_data(n_simulations=10, seed=1)
    assert np.isfinite(theta).all(), "NaN or Inf in theta"
    assert np.isfinite(x).all(), "NaN or Inf in x"


def test_generate_flat_training_data_x_range() -> None:
    _, x = _generate_flat_training_data(n_simulations=20, seed=2)
    # First MAX_TIMEPOINTS features are times in [0, 1]
    times = x[:, :MAX_TIMEPOINTS]
    # Mask is last MAX_TIMEPOINTS features, values 0 or 1
    mask = x[:, 2 * MAX_TIMEPOINTS:]
    assert (mask >= 0).all() and (mask <= 1).all()
    # Valid times (where mask == 1) must be in [0, 1]
    valid = mask > 0.5
    assert (times[valid] >= 0).all() and (times[valid] <= 1).all()


# ---------------------------------------------------------------------------
# _simulate_rel_sld
# ---------------------------------------------------------------------------

def test_simulate_rel_sld_baseline_is_one() -> None:
    from pkpd_sbi.simulators.priors import DEFAULT_PRIOR
    rng = np.random.default_rng(42)
    log_theta = np.log(DEFAULT_PRIOR.sample(1, rng)[0])
    times = np.array([0.0, 56.0, 112.0, 168.0])
    result = _simulate_rel_sld(log_theta, times, duration_days=200.0)
    assert result is not None
    assert result[0] == pytest.approx(1.0, abs=1e-6)


def test_simulate_rel_sld_nonnegative() -> None:
    from pkpd_sbi.simulators.priors import DEFAULT_PRIOR
    rng = np.random.default_rng(7)
    log_theta = np.log(DEFAULT_PRIOR.sample(1, rng)[0])
    times = np.arange(0.0, 365.0, 56.0)
    result = _simulate_rel_sld(log_theta, times, duration_days=400.0)
    assert result is not None
    assert (result >= 0).all()


# ---------------------------------------------------------------------------
# Observation tensor integration
# ---------------------------------------------------------------------------

def test_flat_tensor_from_real_patient_obs() -> None:
    obs = pd.DataFrame({
        "time_days": [0.0, 56.0, 112.0, 168.0, 224.0],
        "rel_sld":   [1.0, 0.85, 0.75, 0.80, 0.90],
    })
    x = observation_tensor_for_npe(obs, mode="flat")
    assert x.shape == (3 * MAX_TIMEPOINTS,)
    # Mask region: first 5 entries should be 1, rest 0
    mask_region = x[2 * MAX_TIMEPOINTS:]
    assert mask_region[:5].sum().item() == pytest.approx(5.0)
    assert mask_region[5:].sum().item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# plot_real_ppc — check it runs without error
# ---------------------------------------------------------------------------

def test_plot_real_ppc_runs(tmp_path) -> None:
    import matplotlib.pyplot as plt
    n = 5
    obs_times = np.arange(0.0, n * 56.0, 56.0)
    obs_rel = np.array([1.0, 0.9, 0.8, 0.85, 0.95])
    pred_times = np.linspace(0.0, 250.0, 50)

    # Synthetic PPC bands
    pred_mat = np.random.default_rng(0).uniform(0.5, 1.5, size=(50, len(pred_times)))
    bands = {
        0.025: np.quantile(pred_mat, 0.025, axis=0),
        0.25:  np.quantile(pred_mat, 0.25, axis=0),
        0.75:  np.quantile(pred_mat, 0.75, axis=0),
        0.975: np.quantile(pred_mat, 0.975, axis=0),
    }

    save_path = tmp_path / "ppc_test.png"
    plot_real_ppc(
        obs_times=obs_times,
        obs_rel_sld=obs_rel,
        pred_times=pred_times,
        ppc={
            "bands": bands,
            "pred_median": np.median(pred_mat, axis=0),
            "rmse": 0.12,
            "coverage_95": 0.90,
        },
        patient_id="TEST001",
        save_path=save_path,
    )
    assert save_path.exists()
    assert save_path.stat().st_size > 0
