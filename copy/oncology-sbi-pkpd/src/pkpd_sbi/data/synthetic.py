"""
Synthetic patient cohort generator.

LaTeX contract:
    phi ~ p(phi)                    [population hyperprior]
    theta_i ~ p(theta_i | phi)      [patient-level prior]
    x_i ~ p(x_i | theta_i)         [simulator]
    y_i ~ p(y_i | x_i)             [observation noise]

Clinical contract:
    - Patients are measured every 8 weeks (56 days) via CT scan
    - Treatment is q2w FOLFOX-like
    - Observation noise is log-normal with sigma ~ 0.10
    - Some patients may have missing scans (dropout / death)
"""

import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import asdict

from ..simulators.dosing import make_q2w_schedule, DoseSchedule, DEFAULT_SCAN_TIMES_DAYS
from ..simulators.simeoni import (
    SimeoniParams, simulate_tumor, simulate_patient_vector,
    simulate_patient_relative_vector, N_PARAMS, PARAM_NAMES,
)
from ..simulators.observation import observe_lognormal, relative_to_baseline
from ..simulators.priors import DEFAULT_PRIOR, PriorBounds


def generate_synthetic_cohort(
    n_patients: int = 100,
    duration_days: float = 365.0,
    scan_interval_days: float = 56.0,
    sigma_obs: float = 0.10,
    seed: int = 123,
    prior: PriorBounds = DEFAULT_PRIOR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate a synthetic clinical trial cohort.

    Returns:
        obs_df: Long-format DataFrame with columns:
            patient_id, time_days, true_volume, observed_volume,
            log_observed_volume, relative_volume, treatment_arm
        params_df: DataFrame with one row per patient, columns = parameter names.
    """
    rng = np.random.default_rng(seed)
    schedule = make_q2w_schedule(duration_days=duration_days)
    obs_times = np.arange(0.0, duration_days + 1e-9, scan_interval_days)

    obs_rows = []
    param_rows = []
    n_failed = 0

    for patient_id in range(n_patients):
        params = prior.sample_one(rng)

        try:
            true_volume = simulate_tumor(params, schedule, obs_times)
        except RuntimeError:
            n_failed += 1
            continue

        observed_volume = observe_lognormal(true_volume, sigma_obs, rng)
        rel_volume = relative_to_baseline(observed_volume)

        param_rows.append({"patient_id": patient_id, **params.to_dict()})

        for j, (t, v_true, v_obs, v_rel) in enumerate(
            zip(obs_times, true_volume, observed_volume, rel_volume)
        ):
            obs_rows.append({
                "patient_id": patient_id,
                "time_days": t,
                "scan_index": j,
                "true_volume": v_true,
                "observed_volume": v_obs,
                "log_observed_volume": np.log(max(v_obs, 1e-8)),
                "relative_volume": v_rel,
                "treatment_arm": "synthetic_q2w",
            })

    if n_failed > 0:
        print(f"Warning: {n_failed}/{n_patients} simulations failed (ODE solver).")

    return pd.DataFrame(obs_rows), pd.DataFrame(param_rows)


def generate_sbi_training_data(
    n_simulations: int = 50_000,
    duration_days: float = 365.0,
    scan_interval_days: float = 56.0,
    sigma_obs: float = 0.10,
    seed: int = 42,
    prior: PriorBounds = DEFAULT_PRIOR,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate (theta, x) pairs for SBI training.

    This is the core data generation loop for NPE:
        theta ~ prior
        x = simulator(theta) + noise

    Returns:
        theta_array: shape (n_valid, N_PARAMS) — log-scale parameters
        x_array: shape (n_valid, n_obs_times) — log observed tumor volumes
    """
    rng = np.random.default_rng(seed)
    schedule = make_q2w_schedule(duration_days=duration_days)
    obs_times = np.arange(0.0, duration_days + 1e-9, scan_interval_days)
    n_times = len(obs_times)

    theta_list = []
    x_list = []

    for i in range(n_simulations):
        # Sample on natural scale, store on log scale for SBI
        natural_params = prior.sample(1, rng)[0]
        log_params = np.log(natural_params)

        x = simulate_patient_vector(
            theta=natural_params,
            schedule=schedule,
            observation_times=obs_times,
            sigma_obs=sigma_obs,
            rng=rng,
        )

        # Skip failed simulations
        if np.any(np.isnan(x)):
            continue

        theta_list.append(log_params)
        x_list.append(x)

        if (i + 1) % 10_000 == 0:
            print(f"  Generated {i+1}/{n_simulations} simulations "
                  f"({len(theta_list)} valid)")

    theta_array = np.array(theta_list)
    x_array = np.array(x_list)

    print(f"SBI training data: {theta_array.shape[0]} valid / {n_simulations} attempted")
    return theta_array, x_array


def generate_sbi_training_data_relative(
    n_simulations: int = 50_000,
    duration_days: float = 365.0,
    scan_interval_days: float = 56.0,
    sigma_obs: float = 0.10,
    seed: int = 42,
    prior: PriorBounds = DEFAULT_PRIOR,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate (theta, x) pairs where x = log(V(t)/V(0)) + noise.

    Relative trajectories are dimensionless and directly comparable to
    log(SLD_t / SLD_baseline) from real clinical data. Enables cross-scale
    validation without unit conversion.

    Returns:
        theta_array: (n_valid, N_PARAMS) log-scale parameters
        x_array: (n_valid, n_obs_times) log-relative tumor volumes (start ≈ 0)
    """
    rng = np.random.default_rng(seed)
    schedule = make_q2w_schedule(duration_days=duration_days)
    obs_times = np.arange(0.0, duration_days + 1e-9, scan_interval_days)

    theta_list = []
    x_list = []

    for i in range(n_simulations):
        natural_params = prior.sample(1, rng)[0]
        log_params = np.log(natural_params)

        x = simulate_patient_relative_vector(
            theta=natural_params,
            schedule=schedule,
            observation_times=obs_times,
            sigma_obs=sigma_obs,
            rng=rng,
        )

        if np.any(np.isnan(x)):
            continue

        theta_list.append(log_params)
        x_list.append(x)

        if (i + 1) % 10_000 == 0:
            print(f"  Generated {i+1}/{n_simulations} relative simulations "
                  f"({len(theta_list)} valid)")

    theta_array = np.array(theta_list)
    x_array = np.array(x_list)
    print(f"Relative SBI training data: {theta_array.shape[0]} valid / {n_simulations} attempted")
    return theta_array, x_array
