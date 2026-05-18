"""
Experiment: Smoke test — generate synthetic cohort and visualize.

This is the MVP milestone from the document:
    Generate 100 synthetic patients
    → simulate tumor burden at q8-week scan times
    → save long-format CSV
    → plot a few trajectories
    → verify simulator is numerically sane

Run: python -m src.pkpd_sbi.experiments.smoke_simulate
"""

import sys
from pathlib import Path

from pkpd_sbi.data.synthetic import generate_synthetic_cohort
from pkpd_sbi.validation.plots import plot_synthetic_trajectories


def main() -> None:
    out_dir = Path("outputs/smoke_simulate")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic cohort (n=100)...")
    obs_df, params_df = generate_synthetic_cohort(n_patients=100, seed=7)

    # Save CSVs
    obs_df.to_csv(out_dir / "synthetic_observations.csv", index=False)
    params_df.to_csv(out_dir / "synthetic_parameters.csv", index=False)

    # Basic sanity checks
    n_patients = obs_df["patient_id"].nunique()
    n_obs = len(obs_df)
    print(f"  Patients: {n_patients}")
    print(f"  Total observations: {n_obs}")
    print(f"  Obs per patient: {n_obs / n_patients:.1f}")
    print(f"  Volume range: [{obs_df['observed_volume'].min():.1f}, {obs_df['observed_volume'].max():.1f}]")
    print(f"  All volumes positive: {(obs_df['observed_volume'] > 0).all()}")
    print(f"  Any NaN: {obs_df['observed_volume'].isna().any()}")

    # Plot
    fig = plot_synthetic_trajectories(obs_df, n_patients=20, save_path=out_dir / "trajectories.png")
    print(f"\nOutputs written to {out_dir}/")


if __name__ == "__main__":
    main()
