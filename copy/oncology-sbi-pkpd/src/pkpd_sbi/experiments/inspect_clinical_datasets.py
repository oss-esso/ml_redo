from __future__ import annotations

import argparse
from pathlib import Path

from pkpd_sbi.data.clinical_loaders import (
    load_a6181122,
    load_efc10262,
    load_efc5505,
    load_efc4972,
    summarize_observations,
)


def print_summary(name: str, ds) -> None:
    obs = ds.observations
    doses = ds.doses
    cov = ds.covariates
    per_patient = summarize_observations(obs)

    print(f"\n=== {name} ===")
    print(f"patients in covariates: {cov['patient_id'].nunique()}")
    print(f"patients with observations: {obs['patient_id'].nunique()}")
    print(f"observation rows: {len(obs)}")
    print(f"dose rows: {len(doses)}")
    print(
        "timepoints/patient:",
        f"min={per_patient['n_timepoints'].min()}",
        f"median={per_patient['n_timepoints'].median()}",
        f"max={per_patient['n_timepoints'].max()}",
    )
    print(
        "time_days range:",
        f"{obs['time_days'].min():.1f}",
        "to",
        f"{obs['time_days'].max():.1f}",
    )
    print("arms:", sorted([str(x) for x in obs["arm"].dropna().unique()])[:8])
    print("drugs:", sorted([str(x) for x in doses["drug"].dropna().unique()])[:12])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Root directory containing AllProvidedFiles_* subdirs (default: data/)",
    )
    args = parser.parse_args()
    base = Path(args.data_dir)

    datasets = []

    p114 = base / "AllProvidedFiles_114"
    if p114.exists():
        datasets.append(("A6181122", load_a6181122(p114)))

    p131 = base / "AllProvidedFiles_131" / "Sanofi_AVE0005_EFC10262_data_files_and_discriptors"
    if p131.exists():
        datasets.append(("EFC10262", load_efc10262(p131, evaluator="INVESTIGATOR")))

    p136 = base / "AllProvidedFiles_136" / "sanofi_sr57746A_efc5505_datasets_and readme"
    if p136.exists():
        datasets.append(("EFC5505", load_efc5505(p136)))

    p137 = base / "AllProvidedFiles_137" / "sanofi_sr57746A_efc4972_datasets_and readme"
    if p137.exists():
        datasets.append(("EFC4972", load_efc4972(p137)))

    if not datasets:
        print("No dataset directories found under data/. Run from repo root.")
        return

    out_dir = Path("outputs/clinical_loaders")
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, ds in datasets:
        print_summary(name, ds)
        ds.observations.to_csv(out_dir / f"{name}_observations.csv", index=False)
        ds.doses.to_csv(out_dir / f"{name}_doses.csv", index=False)
        ds.covariates.to_csv(out_dir / f"{name}_covariates.csv", index=False)

    print(f"\nWrote canonical CSVs to: {out_dir}")


if __name__ == "__main__":
    main()
