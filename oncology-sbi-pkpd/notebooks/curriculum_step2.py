"""
learn_oncology_sbi_step2_fit_one_patient.py

Step 2 in the learning path: connect the simple simulator to one real patient.

What this script does:
1. Reads the canonical clinical CSVs produced by Step 1.
2. Selects one PK/PD-eligible patient.
3. Uses the patient's actual scan times and relative SLD observations.
4. Samples many random Simeoni parameter vectors from the prior.
5. Simulates a tumor trajectory for each sampled parameter vector.
6. Keeps the parameter vectors whose simulated relative trajectory best matches the real one.
7. Saves a plot and a ranked CSV of candidate parameters.

This is NOT SBI yet.
It is deliberately a simple "random-search inverse problem" so you can see
why posterior inference is useful later.

Run Step 1 first:
    python learn_oncology_sbi_step1_load_real_data.py --data-root oncology-sbi-pkpd/data

Then run this:
    python learn_oncology_sbi_step2_fit_one_patient.py

Useful options:
    python learn_oncology_sbi_step2_fit_one_patient.py --study-id EFC10262 --n-samples 2000
    python learn_oncology_sbi_step2_fit_one_patient.py --patient-id SOME_PATIENT_ID --n-samples 5000

Dependencies:
    pip install numpy pandas scipy matplotlib
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


# -----------------------------------------------------------------------------
# 1. Minimal simulator copied from Step 0
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SimeoniParams:
    """Patient-specific tumor and drug-response parameters."""

    lam0: float      # early exponential growth rate, 1/day
    lam1: float      # late linear growth rate, SLD-equivalent/day
    psi_g: float     # transition shape between exponential and linear growth
    k_kill: float    # drug killing strength
    k_tr: float      # damaged-cell transit rate, 1/day
    v0: float        # initial tumor burden

    def as_array(self) -> np.ndarray:
        return np.array([self.lam0, self.lam1, self.psi_g, self.k_kill, self.k_tr, self.v0])


@dataclass(frozen=True)
class DoseSchedule:
    """Repeated bolus doses with first-order exponential elimination."""

    dose_times: np.ndarray
    dose_amounts: np.ndarray
    kel: float

    def concentration(self, t: float) -> float:
        dt = t - self.dose_times
        active = dt >= 0.0
        return float(np.sum(np.where(active, self.dose_amounts * np.exp(-self.kel * dt), 0.0)))


def make_q2w_schedule(
    duration_days: float,
    dose_amount: float = 1.0,
    kel_per_day: float = 0.35,
) -> DoseSchedule:
    """Simplified q2w dosing schedule: one normalized dose every 14 days."""
    dose_times = np.arange(0.0, duration_days + 14.0, 14.0)
    dose_amounts = np.full_like(dose_times, dose_amount, dtype=float)
    return DoseSchedule(dose_times=dose_times, dose_amounts=dose_amounts, kel=kel_per_day)


def growth_term(x1: float, p: SimeoniParams) -> float:
    """Saturating tumor growth: exponential when small, approximately linear when large."""
    x1 = max(float(x1), 1e-12)
    ratio = (p.lam0 / p.lam1) * x1
    denom = (1.0 + ratio ** p.psi_g) ** (1.0 / p.psi_g)
    return p.lam0 * x1 / denom


def rhs(t: float, x: np.ndarray, p: SimeoniParams, schedule: DoseSchedule) -> np.ndarray:
    """Four-compartment Simeoni-style tumor growth inhibition ODE."""
    x1, x2, x3, x4 = np.maximum(x, 0.0)

    c_t = schedule.concentration(t)
    grow = growth_term(x1, p)
    kill = p.k_kill * c_t * x1

    return np.array(
        [
            grow - kill,
            kill - p.k_tr * x2,
            p.k_tr * (x2 - x3),
            p.k_tr * (x3 - x4),
        ],
        dtype=float,
    )


def simulate_relative_trajectory(
    p: SimeoniParams,
    observation_times: np.ndarray,
    schedule: DoseSchedule,
) -> np.ndarray:
    """
    Simulate relative tumor burden at the requested times.

    We shift the real patient's first scan to t=0 before calling this function.
    """
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

    states = np.maximum(sol.y.T, 0.0)
    volume = states.sum(axis=1)
    return volume / max(float(volume[0]), 1e-8)


# -----------------------------------------------------------------------------
# 2. Prior sampling
# -----------------------------------------------------------------------------


PARAM_NAMES = ["lam0", "lam1", "psi_g", "k_kill", "k_tr", "v0"]

# Same broad log-uniform ranges used in the simple learning scripts.
LOG_LOW = np.log(np.array([0.003, 1000.0, 0.5, 0.0005, 0.02, 15.0]))
LOG_HIGH = np.log(np.array([0.08, 30000.0, 2.0, 0.05, 0.20, 250.0]))


def sample_params(rng: np.random.Generator) -> SimeoniParams:
    log_theta = rng.uniform(LOG_LOW, LOG_HIGH)
    theta = np.exp(log_theta)
    return SimeoniParams(*theta)


# -----------------------------------------------------------------------------
# 3. Clinical-data helpers
# -----------------------------------------------------------------------------


def find_eligible_files(audit_dir: Path) -> list[Path]:
    """Find Step-1 outputs like EFC10262_observations_pkpd_eligible.csv."""
    return sorted(audit_dir.glob("*_observations_pkpd_eligible.csv"))


def load_eligible_observations(audit_dir: Path, study_id: str | None = None) -> pd.DataFrame:
    files = find_eligible_files(audit_dir)
    if not files:
        raise FileNotFoundError(
            f"No *_observations_pkpd_eligible.csv files found in {audit_dir}. "
            "Run Step 1 first."
        )

    dfs = []
    for path in files:
        if study_id is not None and not path.name.startswith(study_id):
            continue
        df = pd.read_csv(path)
        df["source_file"] = path.name
        dfs.append(df)

    if not dfs:
        available = ", ".join(p.name.replace("_observations_pkpd_eligible.csv", "") for p in files)
        raise ValueError(f"No eligible file found for study_id={study_id!r}. Available: {available}")

    return pd.concat(dfs, ignore_index=True)


def choose_patient(obs: pd.DataFrame, patient_id: str | None = None) -> tuple[str, pd.DataFrame]:
    """Choose a patient. If none is provided, choose one with many observations."""
    if patient_id is not None:
        patient_id = str(patient_id)
        p_obs = obs[obs["patient_id"].astype(str) == patient_id].copy()
        if p_obs.empty:
            raise ValueError(f"Patient {patient_id!r} not found in eligible observations.")
        return patient_id, p_obs.sort_values("time_days")

    counts = obs.groupby("patient_id")["time_days"].count().sort_values(ascending=False)
    selected = str(counts.index[0])
    p_obs = obs[obs["patient_id"].astype(str) == selected].copy()
    return selected, p_obs.sort_values("time_days")


def prepare_patient_trajectory(p_obs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    df = p_obs.copy().sort_values("time_days")
    df = df[df["rel_sld"].notna() & (df["rel_sld"] > 0)]

    if len(df) < 3:
        raise ValueError("Need at least 3 positive relative-SLD observations.")

    before = len(df)
    df = (
        df.groupby("time_days", as_index=False)
        .agg(
            rel_sld=("rel_sld", "median"),
            sld_mm=("sld_mm", "median"),
            n_rows_collapsed=("rel_sld", "size"),
        )
        .sort_values("time_days")
    )
    after = len(df)

    if after < before:
        print(f"  collapsed duplicate scan days: {before} rows -> {after} unique timepoints")

    raw_times = df["time_days"].to_numpy(dtype=float)
    times = raw_times - raw_times[0]

    rel_sld = df["rel_sld"].to_numpy(dtype=float)
    rel_sld = rel_sld / max(float(rel_sld[0]), 1e-8)

    if np.any(np.diff(times) <= 0):
        raise ValueError(f"Observation times are not strictly increasing after cleanup: {times}")

    return times, rel_sld


# -----------------------------------------------------------------------------
# 4. Random-search fitting
# -----------------------------------------------------------------------------


def score_trajectory(sim_rel: np.ndarray, obs_rel: np.ndarray) -> float:
    """
    RMSE in log-relative space.

    Log-relative space treats 0.5 -> 1.0 and 1.0 -> 2.0 as similarly large
    multiplicative errors.
    """
    sim_log = np.log(np.maximum(sim_rel, 1e-8))
    obs_log = np.log(np.maximum(obs_rel, 1e-8))
    return float(np.sqrt(np.mean((sim_log - obs_log) ** 2)))


def random_search_fit(
    times: np.ndarray,
    obs_rel: np.ndarray,
    n_samples: int,
    seed: int,
    keep_top: int = 20,
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """Sample prior parameters, simulate, score, and keep the best candidates."""
    rng = np.random.default_rng(seed)
    schedule = make_q2w_schedule(duration_days=float(times[-1]) + 56.0)

    rows: list[dict] = []
    top_trajectories: list[np.ndarray] = []

    for i in range(n_samples):
        p = sample_params(rng)
        try:
            sim_rel = simulate_relative_trajectory(p, times, schedule)
            if not np.all(np.isfinite(sim_rel)):
                continue
            rmse_log = score_trajectory(sim_rel, obs_rel)
        except Exception:
            continue

        row = {"sample_idx": i, "rmse_log_rel_sld": rmse_log}
        row.update(dict(zip(PARAM_NAMES, p.as_array())))
        row["final_sim_rel_sld"] = float(sim_rel[-1])
        rows.append(row)

        if (i + 1) % max(1, n_samples // 10) == 0:
            print(f"  tried {i + 1}/{n_samples} parameter samples")

    if not rows:
        raise RuntimeError("All simulations failed. Try changing prior bounds or patient selection.")

    ranked = pd.DataFrame(rows).sort_values("rmse_log_rel_sld").reset_index(drop=True)

    # Re-simulate top candidates for plotting, in the same order as ranked.
    for _, row in ranked.head(keep_top).iterrows():
        p = SimeoniParams(*(float(row[name]) for name in PARAM_NAMES))
        top_trajectories.append(simulate_relative_trajectory(p, times, schedule))

    return ranked, top_trajectories


# -----------------------------------------------------------------------------
# 5. Plotting
# -----------------------------------------------------------------------------


def plot_patient_fit(
    times: np.ndarray,
    obs_rel: np.ndarray,
    ranked: pd.DataFrame,
    top_trajectories: list[np.ndarray],
    patient_id: str,
    study_id: str,
    save_path: Path,
) -> None:
    """Plot real patient trajectory against best random-search simulations."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Show top candidates as faint model-compatible explanations.
    for j, sim_rel in enumerate(top_trajectories):
        alpha = 0.15 if j > 0 else 0.95
        lw = 1.0 if j > 0 else 2.2
        label = "best simulated fit" if j == 0 else None
        ax.plot(times, sim_rel, linewidth=lw, alpha=alpha, label=label)

    # Show observed patient data clearly.
    ax.plot(
        times,
        obs_rel,
        marker="o",
        markersize=6,
        linewidth=2.0,
        color="black",
        label="real patient observations",
    )

    ax.axhline(1.0, linestyle="--", linewidth=0.8, label="baseline")
    ax.axhline(0.7, linestyle=":", linewidth=0.8, label="-30% response threshold")
    ax.axhline(1.2, linestyle=":", linewidth=0.8, label="+20% progression threshold")

    best_rmse = float(ranked.iloc[0]["rmse_log_rel_sld"])
    ax.set_title(f"{study_id} patient {patient_id}: simple random-search fit\nBest log-RMSE = {best_rmse:.3f}")
    ax.set_xlabel("Days since first selected tumor scan")
    ax.set_ylabel("Relative SLD / baseline")
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# 6. Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path("outputs/step1_clinical_audit"),
        help="Output directory produced by Step 1.",
    )
    parser.add_argument("--study-id", type=str, default=None, help="Optional study ID, e.g. EFC10262.")
    parser.add_argument("--patient-id", type=str, default=None, help="Optional patient_id to fit.")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of prior parameter samples to try.")
    parser.add_argument("--seed", type=int, default=11, help="Random seed.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/step2_one_patient_fit"),
        help="Output directory.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    obs = load_eligible_observations(args.audit_dir, study_id=args.study_id)
    patient_id, p_obs = choose_patient(obs, patient_id=args.patient_id)
    times, obs_rel = prepare_patient_trajectory(p_obs)
    study_id = str(p_obs["study_id"].iloc[0])

    print("Selected patient")
    print(f"  study_id:   {study_id}")
    print(f"  patient_id: {patient_id}")
    print(f"  n_scans:    {len(times)}")
    print(f"  times:      {np.round(times, 1).tolist()}")
    print(f"  rel_sld:    {np.round(obs_rel, 3).tolist()}")
    print()
    print(f"Running random-search fit with n_samples={args.n_samples}...")

    ranked, top_trajectories = random_search_fit(
        times=times,
        obs_rel=obs_rel,
        n_samples=args.n_samples,
        seed=args.seed,
        keep_top=25,
    )

    safe_patient = str(patient_id).replace("/", "_").replace("\\", "_").replace(" ", "_")
    ranked_path = args.out_dir / f"{study_id}_{safe_patient}_ranked_parameter_samples.csv"
    plot_path = args.out_dir / f"{study_id}_{safe_patient}_fit.png"

    ranked.to_csv(ranked_path, index=False)
    plot_patient_fit(
        times=times,
        obs_rel=obs_rel,
        ranked=ranked,
        top_trajectories=top_trajectories,
        patient_id=patient_id,
        study_id=study_id,
        save_path=plot_path,
    )

    print("\nBest candidate parameters:")
    cols = ["rmse_log_rel_sld", *PARAM_NAMES, "final_sim_rel_sld"]
    print(ranked.head(10)[cols].round(6).to_string(index=False))

    print("\nSaved:")
    print(f"  {ranked_path}")
    print(f"  {plot_path}")

    print("\nInterpretation:")
    print("  This is a crude inverse problem solver. It shows whether the simple")
    print("  Simeoni model can produce shapes resembling one real patient's trajectory.")
    print("  If many different parameter vectors fit similarly well, that is exactly")
    print("  why we later need a posterior distribution instead of one best fit.")


if __name__ == "__main__":
    main()
