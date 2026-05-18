"""
M3: Train NPE with flat encoding, evaluate posterior predictive on real EFC10262 patients.

Key differences from train_and_evaluate.py:
    - Observation x: flat [times_norm | log_rel_sld | mask] shape (72,) via
      synthetic_obs_to_dataframe + observation_tensor_for_npe
    - Same Simeoni prior and q2w dosing schedule for training
    - After training: load EFC10262, run posterior inference on real patients
    - PPC comparison on rel_sld scale (dimensionless, same for synthetic and real)
    - Dose schedule approximated as standard q2w (consistent with training)

Run from repo root:
    python -m pkpd_sbi.experiments.train_npe_real_conditioned \\
        --data-dir /path/to/oncology-sbi-pkpd/data
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from pkpd_sbi.data.clinical_loaders import (
    filter_patients_for_pkpd,
    load_efc10262,
)
from pkpd_sbi.inference.npe import (
    build_npe,
    load_posterior,
    sample_posterior,
    save_posterior,
)
from pkpd_sbi.inference.summaries import (
    MAX_TIMEPOINTS,
    observation_tensor_for_npe,
    synthetic_obs_to_dataframe,
)
from pkpd_sbi.simulators.dosing import make_q2w_schedule, DEFAULT_SCAN_TIMES_DAYS
from pkpd_sbi.simulators.priors import DEFAULT_PRIOR
from pkpd_sbi.simulators.simeoni import (
    PARAM_NAMES,
    SimeoniParams,
    simulate_tumor,
)
from pkpd_sbi.validation.coverage import posterior_predictive_metrics


# ---------------------------------------------------------------------------
# Step 1 — Synthetic training data with flat encoding
# ---------------------------------------------------------------------------

def _sample_diverse_obs_times(
    rng: np.random.Generator,
    min_scans: int = 2,
    max_scans: int = 10,
    base_interval_days: float = 56.0,
    interval_jitter_days: float = 14.0,
    time_jitter_days: float = 7.0,
    max_duration_days: float = 700.0,
) -> np.ndarray:
    """
    Sample a realistic irregular scan schedule.

    Matches EFC10262 profile: median 4, max 24 scans, ~q8w spacing with jitter.
    Training on diverse schedules prevents constant-feature collapse where fixed-
    grid times are z-scored to zero, killing NPE generalization to real data.
    """
    n = int(rng.integers(min_scans, max_scans + 1))
    interval = base_interval_days + rng.uniform(-interval_jitter_days, interval_jitter_days)
    times = np.zeros(n, dtype=float)
    for i in range(1, n):
        jitter = rng.uniform(-time_jitter_days, time_jitter_days)
        times[i] = times[i - 1] + interval + jitter
    times = np.clip(times, 0.0, max_duration_days)
    return np.unique(times)


def _generate_flat_training_data(
    n_simulations: int,
    min_scans: int = 2,
    max_scans: int = 10,
    sigma_obs: float = 0.10,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate (theta, x) training pairs using 72-dim flat encoding with diverse scan schedules.

    Clinical contract:
        - x = [times_norm | log_rel_sld | mask], shape (3 * MAX_TIMEPOINTS,)
        - theta = log-scale Simeoni parameters, shape (6,)
        - Dosing: standard q2w normalized schedule
        - Scans: diverse random schedules (min_scans to max_scans per patient)
          → avoids constant-feature collapse that kills NPE generalization to real data

    Returns:
        theta_array: (N, 6) float32
        x_array:     (N, 72) float32
    """
    rng = np.random.default_rng(seed)

    theta_list: list[np.ndarray] = []
    x_list: list[np.ndarray] = []
    n_failed = 0

    for i in range(n_simulations):
        obs_times = _sample_diverse_obs_times(rng, min_scans=min_scans, max_scans=max_scans)
        duration = float(obs_times.max()) + 56.0
        schedule = make_q2w_schedule(duration_days=duration)

        natural_params = DEFAULT_PRIOR.sample(1, rng)[0]
        log_params = np.log(natural_params)
        params = SimeoniParams.from_array(natural_params)

        try:
            true_volume = simulate_tumor(params, schedule, obs_times)
        except RuntimeError:
            n_failed += 1
            continue

        safe_volume = np.maximum(true_volume, 1e-8)
        noise = rng.normal(0.0, sigma_obs, size=safe_volume.shape)
        log_observed = np.log(safe_volume) + noise

        if np.any(np.isnan(log_observed)):
            n_failed += 1
            continue

        df = synthetic_obs_to_dataframe(log_observed, obs_times)
        x_tensor = observation_tensor_for_npe(df, mode="flat")

        theta_list.append(log_params)
        x_list.append(x_tensor.numpy())

        if (i + 1) % 10_000 == 0:
            print(f"  Generated {i + 1}/{n_simulations}  ({len(theta_list)} valid)")

    if n_failed > 0:
        print(f"  Warning: {n_failed}/{n_simulations} simulations failed")

    theta_array = np.array(theta_list, dtype=np.float32)
    x_array = np.array(x_list, dtype=np.float32)
    print(f"Flat training data: {len(theta_list)} valid / {n_simulations} attempted")
    return theta_array, x_array


# ---------------------------------------------------------------------------
# Step 4 — Per-patient PPC helpers
# ---------------------------------------------------------------------------

def _simulate_rel_sld(
    log_theta: np.ndarray,
    query_times: np.ndarray,
    duration_days: float,
) -> Optional[np.ndarray]:
    """
    Simulate relative SLD trajectory.

    rel_V(t) = V(t) / V(0) — dimensionless, comparable to clinical rel_sld.
    """
    theta_nat = np.exp(log_theta)
    params = SimeoniParams.from_array(theta_nat)
    schedule = make_q2w_schedule(duration_days=duration_days)
    try:
        volumes = simulate_tumor(params, schedule, query_times)
    except RuntimeError:
        return None
    v0 = max(float(volumes[0]), 1e-8)
    return volumes / v0


def plot_real_ppc(
    obs_times: np.ndarray,
    obs_rel_sld: np.ndarray,
    pred_times: np.ndarray,
    ppc: dict,
    patient_id: str,
    save_path: Path,
) -> None:
    """Plot PPC bands vs real patient observations on rel_sld scale."""
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.fill_between(pred_times, ppc["bands"][0.025], ppc["bands"][0.975],
                    alpha=0.12, color="steelblue", label="95% PI")
    ax.fill_between(pred_times, ppc["bands"][0.25], ppc["bands"][0.75],
                    alpha=0.28, color="steelblue", label="50% PI")
    ax.plot(pred_times, ppc["pred_median"], color="steelblue", lw=1.5, label="Median")
    ax.scatter(obs_times, obs_rel_sld, color="black", zorder=5, s=30, label="Observed")

    ax.axhline(1.0, color="gray", linestyle="--", lw=0.8, alpha=0.5)
    ax.axhline(0.7, color="green", linestyle=":", lw=0.8, alpha=0.4, label="PR (−30%)")
    ax.axhline(1.2, color="red", linestyle=":", lw=0.8, alpha=0.4, label="PD (+20%)")

    cov95 = ppc.get("coverage_95", float("nan"))
    rmse = ppc.get("rmse", float("nan"))
    ax.set_title(
        f"PPC — Patient {patient_id}  "
        f"(RMSE={rmse:.3f}, 95% cov={cov95:.2f})",
        fontsize=10,
    )
    ax.set_xlabel("Days from treatment start")
    ax.set_ylabel("SLD / baseline SLD")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(
    data_dir: str,
    n_train: int = 10_000,
    n_ppc_patients: int = 30,
    n_posterior_samples: int = 500,
    n_ppc_trajectories: int = 200,
    device: str = "cpu",
    posterior_cache: Optional[str] = None,
) -> None:
    data_dir = Path(data_dir)
    out_dir = Path("outputs/real_data_validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    ppc_dir = out_dir / "ppc_plots"
    ppc_dir.mkdir(exist_ok=True)

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    # ---- Step 1: Synthetic training data ----
    print("=" * 60)
    print(f"STEP 1: Generating flat-encoded synthetic training data  (n={n_train})")
    print(f"        x_dim = {3 * MAX_TIMEPOINTS}")
    print("=" * 60)

    theta_cache = out_dir / "theta_train_flat.npy"
    x_cache = out_dir / "x_train_flat.npy"

    if theta_cache.exists() and x_cache.exists():
        print(f"  Loading cached training data from {out_dir}")
        theta_train = np.load(theta_cache)
        x_train = np.load(x_cache)
    else:
        t0 = time.time()
        theta_train, x_train = _generate_flat_training_data(
            n_simulations=n_train, seed=42,
        )
        np.save(theta_cache, theta_train)
        np.save(x_cache, x_train)
        print(f"  Cached to {out_dir}  ({time.time() - t0:.1f}s)")

    print(f"  theta={theta_train.shape}  x={x_train.shape}")

    # ---- Step 2: Train NPE (or load cache) ----
    print("\n" + "=" * 60)
    print("STEP 2: Training NPE")
    print("=" * 60)

    cache_path = Path(posterior_cache) if posterior_cache else out_dir / "posterior_flat.pkl"

    if cache_path.exists():
        print(f"  Loading cached posterior from {cache_path}")
        posterior = load_posterior(cache_path)
        summary: dict = {"cached": True}
    else:
        prior = DEFAULT_PRIOR.to_sbi_prior()
        t0 = time.time()
        posterior, _, summary = build_npe(
            theta=theta_train,
            x=x_train,
            prior=prior,
            device=device,
        )
        summary["training_time_s"] = time.time() - t0
        print(f"  Epochs: {summary['epochs_trained']}  "
              f"val log-prob={summary['best_validation_log_prob']:.4f}  "
              f"({summary['training_time_s']:.1f}s)")
        save_posterior(posterior, cache_path)
        with open(out_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    # ---- Step 3: Load and filter real data ----
    print("\n" + "=" * 60)
    print("STEP 3: Loading EFC10262 real data")
    print("=" * 60)

    efc_root = (
        data_dir
        / "AllProvidedFiles_131"
        / "Sanofi_AVE0005_EFC10262_data_files_and_discriptors"
    )

    if not efc_root.exists():
        print(f"  EFC10262 not found at {efc_root}")
        print("  Pass --data-dir pointing to the AllProvidedFiles_* parent.")
        return

    ds = load_efc10262(efc_root, evaluator="INVESTIGATOR")
    obs_filtered = filter_patients_for_pkpd(
        ds.observations,
        min_timepoints=3,
        require_baseline_near_zero=True,
        baseline_window_days=(-60.0, 14.0),
    )
    patient_ids = obs_filtered["patient_id"].unique()
    print(f"  Patients after filtering: {len(patient_ids)} "
          f"(from {ds.observations['patient_id'].nunique()} total)")

    rng_sel = np.random.default_rng(1)
    selected = rng_sel.choice(
        patient_ids,
        size=min(n_ppc_patients, len(patient_ids)),
        replace=False,
    )

    # ---- Step 4: Per-patient PPC ----
    print("\n" + "=" * 60)
    print(f"STEP 4: Posterior predictive checks on {len(selected)} patients")
    print("=" * 60)

    rows: list[dict] = []

    for patient_id in selected:
        p_obs = obs_filtered[
            obs_filtered["patient_id"] == patient_id
        ].sort_values("time_days").copy()

        # Encode patient observations → 72-dim flat vector
        x_obs = observation_tensor_for_npe(p_obs, mode="flat")
        n_valid = int((x_obs[2 * MAX_TIMEPOINTS:] > 0.5).sum().item())

        # Sample posterior. reject_outside_prior=False: real patients may have
        # parameters outside the xenograft-calibrated training prior. Samples
        # outside the prior are still used — this is a known limitation and
        # should be reported (see prior_outside_frac in output row).
        import warnings
        with warnings.catch_warnings(record=True) as caught_warns:
            warnings.simplefilter("always")
            samples_log = sample_posterior(
                posterior,
                x_obs.numpy(),
                n_samples=n_posterior_samples,
                reject_outside_prior=False,
            )
        prior_outside_frac = 1.0  # default: assume all outside
        for w in caught_warns:
            msg = str(w.message)
            if "outside the prior support" in msg:
                # parse "X% of samples ... outside the prior support"
                try:
                    prior_outside_frac = float(msg.split("%")[0].strip()) / 100.0
                except ValueError:
                    pass

        # Simulate predictive trajectories at patient's actual scan times
        obs_times = p_obs["time_days"].values.astype(float)
        obs_rel = p_obs["rel_sld"].values.astype(float)
        duration = float(obs_times.max()) + 100.0

        pred_trajs_at_obs: list[np.ndarray] = []
        for log_th in samples_log[:n_ppc_trajectories]:
            r = _simulate_rel_sld(log_th, obs_times, duration_days=duration)
            if r is not None and not np.any(np.isnan(r)):
                pred_trajs_at_obs.append(r)

        if len(pred_trajs_at_obs) < 10:
            print(f"  Patient {patient_id}: too few valid trajectories — skip")
            continue

        # Dense time grid for smooth plot curves
        pred_times = np.linspace(0.0, float(obs_times.max()) * 1.05, 200)
        pred_trajs_plot: list[np.ndarray] = []
        for log_th in samples_log[:n_ppc_trajectories]:
            r = _simulate_rel_sld(log_th, pred_times, duration_days=duration)
            if r is not None and not np.any(np.isnan(r)):
                pred_trajs_plot.append(r)

        # Metrics in log-rel space (standard for PPC error reporting)
        ppc_obs = posterior_predictive_metrics(
            np.log(np.maximum(obs_rel, 1e-8)),
            [np.log(np.maximum(t, 1e-8)) for t in pred_trajs_at_obs],
        )
        # Plot bands in natural rel_sld space (y-axis = SLD / baseline)
        ppc_plot = posterior_predictive_metrics(
            np.ones(len(pred_times)),   # dummy; only bands used for plot
            pred_trajs_plot,            # natural rel_sld scale
        )

        # Posterior summary on natural scale
        theta_nat_med = np.exp(np.median(samples_log, axis=0))
        theta_nat_lo = np.exp(np.percentile(samples_log, 5, axis=0))
        theta_nat_hi = np.exp(np.percentile(samples_log, 95, axis=0))

        row: dict = {
            "patient_id":          patient_id,
            "n_real_scans":        len(p_obs),
            "n_grid_matched":      n_valid,
            "n_pred_trajs":        len(pred_trajs_at_obs),
            "rmse_log_rel":        ppc_obs["rmse"],
            "coverage_50":         ppc_obs["coverage_50"],
            "coverage_95":         ppc_obs["coverage_95"],
            "prior_outside_frac":  prior_outside_frac,
        }
        for k, name in enumerate(PARAM_NAMES):
            row[f"{name}_median"] = float(theta_nat_med[k])
            row[f"{name}_lo90"]   = float(theta_nat_lo[k])
            row[f"{name}_hi90"]   = float(theta_nat_hi[k])
        rows.append(row)

        plot_real_ppc(
            obs_times=obs_times,
            obs_rel_sld=obs_rel,
            pred_times=pred_times,
            ppc={
                "bands": ppc_plot["bands"],
                "pred_median": ppc_plot["pred_median"],
                "rmse": ppc_obs["rmse"],
                "coverage_95": ppc_obs["coverage_95"],
            },
            patient_id=str(patient_id),
            save_path=ppc_dir / f"ppc_{patient_id}.png",
        )

        print(
            f"  {patient_id}: scans={len(p_obs)}  matched={n_valid}  "
            f"RMSE={ppc_obs['rmse']:.3f}  95%cov={ppc_obs['coverage_95']:.2f}  "
            f"outside_prior={prior_outside_frac:.0%}"
        )

    # ---- Step 5: Aggregate metrics ----
    print("\n" + "=" * 60)
    print("STEP 5: Aggregate PPC metrics")
    print("=" * 60)

    if rows:
        results_df = pd.DataFrame(rows)
        results_df.to_csv(out_dir / "real_data_ppc_summary.csv", index=False)

        print(f"  Patients evaluated: {len(results_df)}")
        print(f"  Mean RMSE (log-rel): {results_df['rmse_log_rel'].mean():.4f}")
        print(f"  Mean 95% coverage:  {results_df['coverage_95'].mean():.3f}  (target 0.95)")
        print(f"  Mean 50% coverage:  {results_df['coverage_50'].mean():.3f}  (target 0.50)")

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(results_df["rmse_log_rel"], bins=15, color="steelblue", edgecolor="white")
        axes[0].set_xlabel("RMSE (log-relative SLD)")
        axes[0].set_ylabel("Patients")
        axes[0].set_title("PPC RMSE distribution")
        axes[1].hist(results_df["coverage_95"], bins=np.linspace(0, 1, 11),
                     color="steelblue", edgecolor="white")
        axes[1].axvline(0.95, color="red", linestyle="--", label="Target 0.95")
        axes[1].set_xlabel("95% PI coverage")
        axes[1].set_ylabel("Patients")
        axes[1].set_title("95% PI Coverage Distribution")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(out_dir / "coverage_summary.png", dpi=150)
        plt.close(fig)

        print(f"\n  CSV:        {out_dir}/real_data_ppc_summary.csv")
        print(f"  PPC plots:  {ppc_dir}/")
        print(f"  Summary fig: {out_dir}/coverage_summary.png")
    else:
        print("  No patients passed all filters — check data path and filter settings.")

    print(f"\nAll outputs in: {out_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="M3: Train NPE with flat encoding, validate on EFC10262"
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Root data directory containing AllProvidedFiles_* subdirs",
    )
    parser.add_argument("--n-train", type=int, default=10_000)
    parser.add_argument("--n-ppc-patients", type=int, default=30)
    parser.add_argument("--n-posterior-samples", type=int, default=500)
    parser.add_argument("--n-ppc-trajectories", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--posterior-cache", default=None,
        help="Path to cached posterior.pkl to skip retraining",
    )
    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        n_train=args.n_train,
        n_ppc_patients=args.n_ppc_patients,
        n_posterior_samples=args.n_posterior_samples,
        n_ppc_trajectories=args.n_ppc_trajectories,
        device=args.device,
        posterior_cache=args.posterior_cache,
    )
