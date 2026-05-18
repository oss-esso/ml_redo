"""
Experiment: Train NPE and run full validation pipeline.

This is the Phase 1 deliverable from the document:
    1. Generate SBI training data (theta, x) pairs
    2. Train NPE using sbi package
    3. Run SBC to check calibration
    4. Evaluate parameter recovery on held-out test set
    5. Generate posterior predictive checks
    6. Save all figures and metrics

Run: python -m src.pkpd_sbi.experiments.train_and_evaluate
"""

import sys
import json
import time
import numpy as np
import torch
from pathlib import Path

from pkpd_sbi.data.synthetic import generate_sbi_training_data
from pkpd_sbi.simulators.priors import DEFAULT_PRIOR
from pkpd_sbi.simulators.simeoni import PARAM_NAMES, simulate_patient_vector
from pkpd_sbi.simulators.dosing import make_q2w_schedule, DEFAULT_SCAN_TIMES_DAYS
from pkpd_sbi.inference.npe import build_npe, sample_posterior, save_posterior
from pkpd_sbi.validation.sbc import run_sbc
from pkpd_sbi.validation.coverage import (
    compute_coverage, parameter_recovery_metrics, posterior_predictive_metrics,
)
from pkpd_sbi.validation.plots import (
    plot_sbc_ranks, plot_parameter_recovery, plot_posterior_predictive,
)


def make_simulator_fn(sigma_obs: float = 0.10, seed_offset: int = 0):
    """Create a simulator function for SBC that maps log_theta -> x."""
    schedule = make_q2w_schedule()
    obs_times = DEFAULT_SCAN_TIMES_DAYS
    rng = np.random.default_rng(seed_offset)

    def simulator_fn(log_theta: np.ndarray) -> np.ndarray:
        theta_natural = np.exp(log_theta)
        return simulate_patient_vector(
            theta=theta_natural,
            schedule=schedule,
            observation_times=obs_times,
            sigma_obs=sigma_obs,
            rng=rng,
        )
    return simulator_fn


def main(
    n_train: int = 50_000,
    n_test: int = 200,
    n_sbc: int = 300,
    device: str = "cpu",
) -> None:
    out_dir = Path("outputs/phase1_npe")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if CUDA is available
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    # ---- Step 1: Generate training data ----
    print("=" * 60)
    print("STEP 1: Generating SBI training data")
    print("=" * 60)
    t0 = time.time()
    theta_train, x_train = generate_sbi_training_data(
        n_simulations=n_train, seed=42,
    )
    dt = time.time() - t0
    print(f"  Training data: {theta_train.shape} theta, {x_train.shape} x")
    print(f"  Generation time: {dt:.1f}s")

    # Split off test set
    theta_test = theta_train[-n_test:]
    x_test = x_train[-n_test:]
    theta_train = theta_train[:-n_test]
    x_train = x_train[:-n_test]

    # ---- Step 2: Train NPE ----
    print("\n" + "=" * 60)
    print("STEP 2: Training NPE")
    print("=" * 60)
    prior = DEFAULT_PRIOR.to_sbi_prior()

    t0 = time.time()
    posterior, inference, summary = build_npe(
        theta=theta_train,
        x=x_train,
        prior=prior,
        device=device,
    )
    dt = time.time() - t0
    print(f"  Training time: {dt:.1f}s")
    print(f"  Epochs: {summary['epochs_trained']}")
    print(f"  Best val log prob: {summary['best_validation_log_prob']:.4f}")

    save_posterior(posterior, out_dir / "posterior.pkl")
    with open(out_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Step 3: Parameter recovery on test set ----
    print("\n" + "=" * 60)
    print("STEP 3: Parameter recovery")
    print("=" * 60)
    posterior_medians = np.zeros_like(theta_test)
    all_samples = []

    for i in range(n_test):
        samples = sample_posterior(posterior, x_test[i], n_samples=2000)
        posterior_medians[i] = np.median(samples, axis=0)
        all_samples.append(samples)
        if (i + 1) % 50 == 0:
            print(f"  Recovered {i+1}/{n_test} patients")

    recovery = parameter_recovery_metrics(theta_test, posterior_medians, PARAM_NAMES)
    print("\n  Parameter recovery (log scale):")
    for name, m in recovery.items():
        print(f"    {name:8s}: RMSE={m['rmse']:.4f}  r={m['correlation']:.3f}  bias={m['bias']:.4f}")

    with open(out_dir / "recovery_metrics.json", "w") as f:
        json.dump(recovery, f, indent=2)

    fig = plot_parameter_recovery(theta_test, posterior_medians, PARAM_NAMES,
                                  save_path=out_dir / "parameter_recovery.png")

    # Coverage
    samples_3d = np.array(all_samples)  # (n_test, K, d)
    coverage = compute_coverage(theta_test, samples_3d)
    print("\n  Coverage:")
    for level, covs in coverage.items():
        mean_cov = np.mean(covs)
        print(f"    {level*100:.0f}% CI: mean coverage = {mean_cov:.3f} (target: {level:.2f})")

    # ---- Step 4: SBC ----
    print("\n" + "=" * 60)
    print("STEP 4: Simulation-Based Calibration")
    print("=" * 60)
    simulator_fn = make_simulator_fn(seed_offset=9999)
    sbc_results = run_sbc(
        posterior=posterior,
        prior=prior,
        simulator_fn=simulator_fn,
        n_sbc=n_sbc,
        n_posterior_samples=500,
        param_names=PARAM_NAMES,
    )

    print(f"\n  SBC: {sbc_results['n_valid']}/{sbc_results['n_attempted']} valid")
    print(f"  All pass: {sbc_results['all_pass']}")
    for name, result in sbc_results["uniformity"].items():
        status = "PASS" if result["passes"] else "FAIL"
        print(f"    {name:8s}: KS={result['ks_statistic']:.4f}  p={result['p_value']:.4f}  [{status}]")

    fig = plot_sbc_ranks(sbc_results["ranks"], 500, PARAM_NAMES,
                          save_path=out_dir / "sbc_ranks.png")

    # ---- Step 5: Posterior predictive for a few patients ----
    print("\n" + "=" * 60)
    print("STEP 5: Posterior predictive checks")
    print("=" * 60)
    obs_times = DEFAULT_SCAN_TIMES_DAYS
    schedule = make_q2w_schedule()

    for patient_idx in range(min(5, n_test)):
        x_obs = x_test[patient_idx]
        samples = sample_posterior(posterior, x_obs, n_samples=200)

        pred_trajectories = []
        rng_pred = np.random.default_rng(patient_idx)
        for s in samples:
            theta_nat = np.exp(s)
            x_pred = simulate_patient_vector(theta_nat, schedule, obs_times, 0.10, rng_pred)
            if not np.any(np.isnan(x_pred)):
                pred_trajectories.append(x_pred)

        if len(pred_trajectories) > 10:
            ppc = posterior_predictive_metrics(x_obs, pred_trajectories)
            fig = plot_posterior_predictive(
                obs_times, x_obs, ppc["bands"], ppc["pred_median"],
                patient_id=patient_idx,
                save_path=out_dir / f"ppc_patient_{patient_idx}.png",
            )
            print(f"  Patient {patient_idx}: RMSE={ppc['rmse']:.4f}  "
                  f"95% coverage={ppc['coverage_95']:.2f}")

    print(f"\nAll outputs saved to {out_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=50_000)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--n-sbc", type=int, default=300)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    main(args.n_train, args.n_test, args.n_sbc, args.device)
