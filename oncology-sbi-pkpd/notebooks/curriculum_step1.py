"""
learn_oncology_sbi_step1_load_real_data.py

Step 1 in the learning path: load the real clinical oncology datasets and
convert them into the common "canonical" format used by the SBI pipeline.

This script does NOT train SBI yet.
Its goal is to answer:
    - Which datasets are present under my data directory?
    - How many patients / tumor observations / dose rows do I have?
    - Do patients have enough longitudinal tumor measurements for PK/PD fitting?
    - What do the relative tumor-size trajectories look like?

Expected data layout:
    oncology-sbi-pkpd/data/AllProvidedFiles_*/...

The number in AllProvidedFiles_* may change. This script searches recursively
for dataset directories by looking for the SAS files expected by clinical_loaders.py.

Run from the directory containing this file and clinical_loaders.py:
    python learn_oncology_sbi_step1_load_real_data.py --data-root oncology-sbi-pkpd/data

On Windows PowerShell, for example:
    python .\learn_oncology_sbi_step1_load_real_data.py --data-root .\oncology-sbi-pkpd\data

Dependencies:
    pip install pandas numpy matplotlib pyreadstat

Note:
    pandas.read_sas may need pyreadstat installed for SAS support depending on your setup.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# 0. Import the colleague's clinical_loaders.py
# -----------------------------------------------------------------------------

try:
    # Works if you run this from inside the installed repo/package.
    from pkpd_sbi.data.clinical_loaders import (  # type: ignore
        ClinicalDataset,
        concatenate_clinical_datasets,
        filter_patients_for_pkpd,
        load_a6181122,
        load_efc10262,
        load_efc4972,
        load_efc5505,
        summarize_observations,
    )
except Exception:
    # Works if clinical_loaders.py is copied next to this script.
    HERE = Path(__file__).resolve().parent
    sys.path.insert(0, str(HERE))
    from clinical_loaders import (  # type: ignore
        ClinicalDataset,
        concatenate_clinical_datasets,
        filter_patients_for_pkpd,
        load_a6181122,
        load_efc10262,
        load_efc4972,
        load_efc5505,
        summarize_observations,
    )


# -----------------------------------------------------------------------------
# 1. Dataset discovery
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetCandidate:
    """A discovered dataset directory and the loader that should read it."""

    study_id: str
    root: Path
    loader: Callable[[Path], ClinicalDataset]


def _has_files(folder: Path, required_names: set[str]) -> bool:
    """True if folder contains all required file names, case-insensitive."""
    if not folder.exists() or not folder.is_dir():
        return False
    names = {p.name.lower() for p in folder.iterdir() if p.is_file()}
    return {x.lower() for x in required_names}.issubset(names)


def discover_datasets(data_root: Path) -> list[DatasetCandidate]:
    """
    Search recursively under data_root for known clinical dataset structures.

    We avoid depending on the AllProvidedFiles_* number. Instead, we look for
    the actual SAS files that each loader expects.
    """
    data_root = data_root.resolve()
    candidates: list[DatasetCandidate] = []
    seen: set[tuple[str, Path]] = set()

    # Search all directories under data_root, including data_root itself.
    folders = [data_root] + [p for p in data_root.rglob("*") if p.is_dir()]

    for folder in folders:
        # Pfizer A6181122: files are usually directly inside AllProvidedFiles_*/
        if _has_files(folder, {"tmm_p.sas7bdat", "testdrug.sas7bdat", "demog.sas7bdat"}):
            key = ("A6181122", folder)
            if key not in seen:
                candidates.append(DatasetCandidate("A6181122", folder, load_a6181122))
                seen.add(key)

        # Sanofi EFC10262: SDTM-like files.
        if _has_files(folder, {"ls.sas7bdat", "ex.sas7bdat", "dm.sas7bdat"}):
            key = ("EFC10262", folder)
            if key not in seen:
                # load_efc10262 has an optional evaluator argument, so wrap it.
                candidates.append(
                    DatasetCandidate(
                        "EFC10262",
                        folder,
                        lambda root, _loader=load_efc10262: _loader(root, evaluator="INVESTIGATOR"),
                    )
                )
                seen.add(key)

        # Sanofi EFC5505 / EFC4972: shared file names.
        if _has_files(folder, {"measur.sas7bdat", "dose.sas7bdat", "demog.sas7bdat"}):
            folder_text = str(folder).lower()
            if "5505" in folder_text:
                study_id = "EFC5505"
                loader = load_efc5505
            elif "4972" in folder_text:
                study_id = "EFC4972"
                loader = load_efc4972
            else:
                # The same loader structure works, but we do not know the study ID.
                # Prefer explicit names in real analyses.
                study_id = "EFC55XX_UNKNOWN"
                loader = load_efc5505

            key = (study_id, folder)
            if key not in seen:
                candidates.append(DatasetCandidate(study_id, folder, loader))
                seen.add(key)

    return candidates


# -----------------------------------------------------------------------------
# 2. Summaries and sanity checks
# -----------------------------------------------------------------------------


def dataset_summary_row(ds: ClinicalDataset, source_root: Path) -> dict:
    """Compute one compact summary row for a loaded dataset."""
    obs = ds.observations
    doses = ds.doses
    cov = ds.covariates
    per_patient = summarize_observations(obs)

    filtered = filter_patients_for_pkpd(
        obs,
        min_timepoints=3,
        require_baseline_near_zero=True,
        baseline_window_days=(-60.0, 14.0),
    )

    return {
        "study_id": ds.study_id,
        "source_root": str(source_root),
        "n_patients_covariates": int(cov["patient_id"].nunique()),
        "n_patients_observed": int(obs["patient_id"].nunique()),
        "n_patients_pkpd_eligible": int(filtered["patient_id"].nunique()),
        "n_observation_rows": int(len(obs)),
        "n_dose_rows": int(len(doses)),
        "timepoints_min": int(per_patient["n_timepoints"].min()) if len(per_patient) else 0,
        "timepoints_median": float(per_patient["n_timepoints"].median()) if len(per_patient) else np.nan,
        "timepoints_max": int(per_patient["n_timepoints"].max()) if len(per_patient) else 0,
        "time_days_min": float(obs["time_days"].min()) if len(obs) else np.nan,
        "time_days_max": float(obs["time_days"].max()) if len(obs) else np.nan,
        "baseline_sld_median_mm": float(per_patient["baseline_sld_mm"].median()) if len(per_patient) else np.nan,
        "rel_sld_min": float(obs["rel_sld"].min()) if len(obs) else np.nan,
        "rel_sld_max": float(obs["rel_sld"].max()) if len(obs) else np.nan,
        "arms": " | ".join(sorted(str(x) for x in obs["arm"].dropna().unique())[:10]),
        "drugs": " | ".join(sorted(str(x) for x in doses["drug"].dropna().unique())[:15]),
    }


def print_dataset_summary(row: dict) -> None:
    """Human-readable summary for the terminal."""
    print(f"\n=== {row['study_id']} ===")
    print(f"source: {row['source_root']}")
    print(f"patients with observations: {row['n_patients_observed']}")
    print(f"patients in covariates:    {row['n_patients_covariates']}")
    print(f"PK/PD eligible patients:   {row['n_patients_pkpd_eligible']}  (>=3 scans, baseline near day 0)")
    print(f"observation rows:          {row['n_observation_rows']}")
    print(f"dose rows:                 {row['n_dose_rows']}")
    print(
        "timepoints/patient:      "
        f"min={row['timepoints_min']}, "
        f"median={row['timepoints_median']:.1f}, "
        f"max={row['timepoints_max']}"
    )
    print(f"time_days range:           {row['time_days_min']:.1f} to {row['time_days_max']:.1f}")
    print(f"baseline SLD median [mm]:  {row['baseline_sld_median_mm']:.2f}")
    print(f"relative SLD range:        {row['rel_sld_min']:.3f} to {row['rel_sld_max']:.3f}")
    print(f"arms:                      {row['arms']}")
    print(f"drugs:                     {row['drugs']}")


# -----------------------------------------------------------------------------
# 3. Plots
# -----------------------------------------------------------------------------


def plot_relative_sld_trajectories(
    obs: pd.DataFrame,
    study_id: str,
    save_path: Path,
    n_patients: int = 40,
    seed: int = 1,
) -> None:
    """Plot real relative SLD trajectories for a random subset of patients."""
    rng = np.random.default_rng(seed)
    patient_ids = np.array(sorted(obs["patient_id"].dropna().unique()))

    if len(patient_ids) == 0:
        return

    selected = rng.choice(patient_ids, size=min(n_patients, len(patient_ids)), replace=False)

    fig, ax = plt.subplots(figsize=(8, 5))

    for pid in selected:
        g = obs[obs["patient_id"] == pid].sort_values("time_days")
        ax.plot(g["time_days"], g["rel_sld"], marker="o", markersize=3, linewidth=1.0, alpha=0.55)

    ax.axhline(1.0, linestyle="--", linewidth=0.8, label="baseline")
    ax.axhline(0.7, linestyle=":", linewidth=0.8, label="RECIST-like PR threshold: -30%")
    ax.axhline(1.2, linestyle=":", linewidth=0.8, label="RECIST-like PD threshold: +20%")
    ax.set_xlabel("Days from treatment start")
    ax.set_ylabel("SLD / baseline SLD")
    ax.set_title(f"{study_id}: real relative tumor-size trajectories")
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_timepoint_histogram(obs: pd.DataFrame, study_id: str, save_path: Path) -> None:
    """Plot the distribution of number of scans per patient."""
    per_patient = summarize_observations(obs)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(per_patient["n_timepoints"], bins=range(1, int(per_patient["n_timepoints"].max()) + 2), alpha=0.8)
    ax.set_xlabel("Number of tumor measurements per patient")
    ax.set_ylabel("Number of patients")
    ax.set_title(f"{study_id}: scan count distribution")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# 4. Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root directory containing AllProvidedFiles_* folders, e.g. oncology-sbi-pkpd/data",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/step1_clinical_audit"),
        help="Output directory for canonical CSVs, summaries, and plots.",
    )
    parser.add_argument(
        "--no-concat",
        action="store_true",
        help="Do not write concatenated all-study CSV files.",
    )
    args = parser.parse_args()

    data_root = args.data_root
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    print(f"Searching for datasets under: {data_root.resolve()}")
    candidates = discover_datasets(data_root)

    if not candidates:
        print("\nNo known dataset structures found.")
        print("Expected one of these file combinations somewhere under --data-root:")
        print("  A6181122:  tmm_p.sas7bdat + testdrug.sas7bdat + demog.sas7bdat")
        print("  EFC10262:  ls.sas7bdat + ex.sas7bdat + dm.sas7bdat")
        print("  EFC55xx:   measur.sas7bdat + dose.sas7bdat + demog.sas7bdat")
        return

    print(f"Found {len(candidates)} candidate dataset(s):")
    for c in candidates:
        print(f"  - {c.study_id}: {c.root}")

    loaded: list[ClinicalDataset] = []
    summary_rows: list[dict] = []

    for c in candidates:
        print(f"\nLoading {c.study_id} from {c.root} ...")
        try:
            ds = c.loader(c.root)
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            continue

        loaded.append(ds)
        row = dataset_summary_row(ds, c.root)
        summary_rows.append(row)
        print_dataset_summary(row)

        # Save per-study canonical tables.
        safe_study = ds.study_id.replace("+", "_")
        ds.observations.to_csv(out_dir / f"{safe_study}_observations.csv", index=False)
        ds.doses.to_csv(out_dir / f"{safe_study}_doses.csv", index=False)
        ds.covariates.to_csv(out_dir / f"{safe_study}_covariates.csv", index=False)

        # Save filtered observations used later for PK/PD/SBI experiments.
        filtered = filter_patients_for_pkpd(
            ds.observations,
            min_timepoints=3,
            require_baseline_near_zero=True,
            baseline_window_days=(-60.0, 14.0),
        )
        filtered.to_csv(out_dir / f"{safe_study}_observations_pkpd_eligible.csv", index=False)

        # Plots.
        plot_relative_sld_trajectories(
            ds.observations,
            ds.study_id,
            save_path=plot_dir / f"{safe_study}_relative_sld_trajectories.png",
        )
        plot_timepoint_histogram(
            ds.observations,
            ds.study_id,
            save_path=plot_dir / f"{safe_study}_scan_count_histogram.png",
        )

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary_path = out_dir / "dataset_summary.csv"
        summary.to_csv(summary_path, index=False)

        print("\n=== Combined summary ===")
        print(
            summary[
                [
                    "study_id",
                    "n_patients_observed",
                    "n_patients_pkpd_eligible",
                    "n_observation_rows",
                    "n_dose_rows",
                    "timepoints_median",
                    "timepoints_max",
                ]
            ].to_string(index=False)
        )
        print(f"\nWrote summary: {summary_path}")

    if loaded and not args.no_concat:
        combined = concatenate_clinical_datasets(loaded)
        combined.observations.to_csv(out_dir / "ALL_observations.csv", index=False)
        combined.doses.to_csv(out_dir / "ALL_doses.csv", index=False)
        combined.covariates.to_csv(out_dir / "ALL_covariates.csv", index=False)
        print("\nWrote concatenated canonical tables:")
        print(f"  {out_dir / 'ALL_observations.csv'}")
        print(f"  {out_dir / 'ALL_doses.csv'}")
        print(f"  {out_dir / 'ALL_covariates.csv'}")

    print("\nNext learning step:")
    print("  Pick one eligible patient from *_observations_pkpd_eligible.csv")
    print("  and compare their real rel_sld trajectory against synthetic simulator outputs.")


if __name__ == "__main__":
    main()
