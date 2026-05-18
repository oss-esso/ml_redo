"""A deliberately simple, standalone first script for learning oncology SBI.

What this script does:
1. Defines a dose schedule: repeated drug doses every 14 days.
2. Defines a simple Simeoni-style tumor growth inhibition ODE.
3. Simulates one or more synthetic patients.
4. Adds log-normal measurement noise, like noisy tumor scans.
5. Converts tumor sizes to relative-to-baseline trajectories.
6. Saves a plot and prints the simulated data.

What this script does NOT do yet:
- It does not train SBI / NPE.
- It does not use PyTorch, JAX, sbi, or real clinical data.
- It does not infer parameters from observations.

Run:
    python learn_oncology_sbi_step0.py

Dependencies:
    pip install numpy scipy matplotlib pandas
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


#----------------
# Parameters

@dataclass(frozen=True)
class SimeoniParams:
    lam0: float
    lam1: float
    psi_g: float
    k_kill: float
    k_tr: float
    v0: float

    def as_array(self):
        return np.array([self.lam0, self.lam1, self.psi_g, self.k_kill, self.k_tr, self.v0])
    
@dataclass(frozen=True)
class DoseSchedule:
    dose_times : np.ndarray
    dose_amounts : np.ndarray
    kel: float
    
    def concentration(self, t: float):
        dt = t - self.dose_times
        active = dt >= 0.0

        return float(np.sum(np.where(active, self.dose_amounts * np.exp(-self.kel * dt), 0.0)))
    

def make_q2w_schedule(
        duration_days: float = 365.0,
        dose_amount: float = 1.0,
        kel_per_day: float = 0.35
) -> DoseSchedule:
    
    dose_times = np.arange(0.0, duration_days + 1e-9, 14.0) # dose every two weeks
    dose_amounts = np.full_like(dose_times, dose_amount, dtype=float)

    return DoseSchedule(
        dose_times,
        dose_amounts,
        kel_per_day
    )


#----------
# Tumor model


def growth_term(x1: float, p: SimeoniParams):
    x1 = max(float(x1), 1e-12)
    ratio = (p.lam0 / p.lam1) * x1
    denom = (1.0 + ratio ** p.psi_g) ** (p.psi_g ** (-1))

    return p.lam0 * x1 / denom

def rhs(t: float, x: np.ndarray, p: SimeoniParams, schedule: DoseSchedule):

    x1, x2, x3, x4 = np.maximum(x, 0.0)

    c_t = schedule.concentration(t)
    grow = growth_term(x1, p)
    kill = p.k_kill * c_t * x1

    dx1 = grow - kill
    dx2 = kill - p.k_tr * x2
    dx3 = p.k_tr * (x2 - x3)
    dx4 = p.k_tr * (x3 - x4)


    return np.array([dx1, dx2, dx3, dx4], dtype=float)

def simulate_tumor(
        p: SimeoniParams,
        schedule: DoseSchedule,
        observation_times: np.ndarray
):
    observation_times = np.asarray(observation_times, dtype=float)
    x0 = np.array([p.v0, 0.0, 0.0, 0.0], dtype=float)

    sol = solve_ivp(
        fun=lambda t, x: rhs(t, x, p, schedule),
        t_span=(float(observation_times[0]), float(observation_times[-1])),
        y0=x0,
        t_eval=observation_times,
        method="LSODA",
        rtol=1e-6,
        atol=1e-8,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    states = np.maximum(sol.y.T, 0.0)  # shape: (n_times, 4)
    return states.sum(axis=1)



def observe_longnormal(true_volume: np.ndarray, sigma_obs: float, rng: np.random.Generator):
    safe = np.maximum(np.asarray(true_volume, dtype=float), 1e-8)
    noise = rng.normal(0.0, sigma_obs, size=safe.shape)
    return np.exp(np.log(safe) + noise)

def relative_to_baseline(values:np.ndarray):

    values = np.asarray(values, dtype=float)
    return values / np.maximum(values[0], 1e-8)


#----------
# Prior Sampling

PARAM_NAMES = ["lam0", "lam1", "psi_g", "k_kill", "k_tr", "v0"]

LOG_LOW = np.log(np.array([0.003, 1000.0, 0.5, 0.0005, 0.02, 15.0]))
LOG_HIGH = np.log(np.array([0.08, 30000.0, 2.0, 0.05, 0.20, 250.0]))

def sample_params(rng: np.random.Generator):
    log_theta = rng.uniform(LOG_LOW, LOG_HIGH)
    theta = np.exp(log_theta)
    return SimeoniParams(*theta)


def simulate_patient(
        patient_id: int,
        rng: np.random.Generator,
        observation_times: np.ndarray,
        schedule: DoseSchedule,
        sigma_obs: float = 10.0,
):
    p = sample_params(rng)
    true_volume = simulate_tumor(p, schedule, observation_times)
    observed_volume = observe_longnormal(true_volume, sigma_obs, rng)

    relative_volume = relative_to_baseline(observed_volume)
    log_relative_volume = np.log(np.maximum(relative_volume, 1e-8))

    obs = pd.DataFrame(
        {
            "patient_id": patient_id,
            "time_days": observation_times,
            "true_volume": true_volume,
            "observed_volume": observed_volume,
            "relative_volume": relative_volume,
            "log_relative_volume": log_relative_volume,
        }
    )

    params = {'patient_id': patient_id}
    params.update(dict(zip(PARAM_NAMES, p.as_array())))

    return obs, params

def simulate_cohort(
    n_patients: int = 20,
    seed: int = 7,
    sigma_obs: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate a small synthetic cohort."""
    rng = np.random.default_rng(seed)
    observation_times = np.arange(0.0, 365.0, 56.0)  # q8w scans: 0, 56, ..., 336
    schedule = make_q2w_schedule(duration_days=float(observation_times[-1]))

    obs_rows = []
    param_rows = []

    for patient_id in range(n_patients):
        obs, params = simulate_patient(
            patient_id=patient_id,
            rng=rng,
            observation_times=observation_times,
            schedule=schedule,
            sigma_obs=sigma_obs,
        )
        obs_rows.append(obs)
        param_rows.append(params)

    return pd.concat(obs_rows, ignore_index=True), pd.DataFrame(param_rows)


def plot_trajectories(obs_df: pd.DataFrame, out_path: Path, n_to_plot: int = 20) -> None:
    """Plot relative tumor burden for the first n patients."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for patient_id, group in obs_df.groupby("patient_id"):
        if patient_id >= n_to_plot:
            break
        ax.plot(
            group["time_days"],
            group["relative_volume"],
            marker="o",
            linewidth=1.0,
            alpha=0.7,
        )

    ax.axhline(1.0, linestyle="--", linewidth=0.8, label="baseline")
    ax.axhline(0.7, linestyle=":", linewidth=0.8, label="-30% response threshold")
    ax.axhline(1.2, linestyle=":", linewidth=0.8, label="+20% progression threshold")
    ax.set_xlabel("Time from treatment start [days]")
    ax.set_ylabel("Observed tumor burden / baseline")
    ax.set_title("Synthetic tumor trajectories")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    out_dir = Path("oncology-sbi-pkpd/notebooks/outputs/step0_simulator")
    out_dir.mkdir(parents=True, exist_ok=True)

    obs_df, params_df = simulate_cohort(n_patients=20, seed=7, sigma_obs=0.10)

    obs_path = out_dir / "synthetic_observations.csv"
    params_path = out_dir / "synthetic_parameters.csv"
    plot_path = out_dir / "trajectories.png"

    obs_df.to_csv(obs_path, index=False)
    params_df.to_csv(params_path, index=False)
    plot_trajectories(obs_df, plot_path)

    print("Generated synthetic oncology trajectories")
    print(f"  Patients: {obs_df['patient_id'].nunique()}")
    print(f"  Observations: {len(obs_df)}")
    print(f"  Observation times: {sorted(obs_df['time_days'].unique())}")
    print(f"  Observed volume range: {obs_df['observed_volume'].min():.2f} to {obs_df['observed_volume'].max():.2f}")
    print()
    print("First patient observations:")
    print(obs_df[obs_df["patient_id"] == 0].round(4).to_string(index=False))
    print()
    print("First patient true parameters:")
    print(params_df[params_df["patient_id"] == 0].round(6).to_string(index=False))
    print()
    print("Saved:")
    print(f"  {obs_path}")
    print(f"  {params_path}")
    print(f"  {plot_path}")


if __name__ == "__main__":
    main()