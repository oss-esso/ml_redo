"""
Tests for clinical_loaders.py.

Skips any dataset whose directory is not present so the suite can run
in CI without the raw SAS files mounted. All assertions encode the
clinical contract, not the SAS format.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pkpd_sbi.data.clinical_loaders import (
    ClinicalDataset,
    filter_patients_for_pkpd,
    load_a6181122,
    load_efc10262,
    load_efc4972,
    load_efc5505,
    summarize_observations,
)

# ---------------------------------------------------------------------------
# Data root — override with DATA_ROOT env var if needed
# ---------------------------------------------------------------------------
_DEFAULT_DATA_ROOT = Path(__file__).parents[3] / "oncology-sbi-pkpd" / "data"


def _data_root() -> Path:
    import os
    return Path(os.environ.get("DATA_ROOT", _DEFAULT_DATA_ROOT))


def _path(subdir: str) -> Path:
    return _data_root() / subdir


# ---------------------------------------------------------------------------
# Shared contract assertions
# ---------------------------------------------------------------------------

OBS_REQUIRED_COLS = {
    "study_id", "patient_id", "raw_day", "time_days",
    "sld_mm", "rel_sld", "baseline_sld_mm", "n_lesions", "arm",
}
DOSE_REQUIRED_COLS = {
    "study_id", "patient_id", "raw_day", "time_days",
    "drug", "dose_amount", "dose_unit", "arm",
}
COV_REQUIRED_COLS = {"study_id", "patient_id", "age", "sex", "race", "arm"}


def _assert_contract(ds: ClinicalDataset, expected_min_patients: int) -> None:
    obs = ds.observations
    doses = ds.doses
    cov = ds.covariates

    # Schema
    assert OBS_REQUIRED_COLS.issubset(obs.columns), (
        f"Missing obs cols: {OBS_REQUIRED_COLS - set(obs.columns)}"
    )
    assert DOSE_REQUIRED_COLS.issubset(doses.columns), (
        f"Missing dose cols: {DOSE_REQUIRED_COLS - set(doses.columns)}"
    )
    assert COV_REQUIRED_COLS.issubset(cov.columns), (
        f"Missing cov cols: {COV_REQUIRED_COLS - set(cov.columns)}"
    )

    # Non-empty
    assert len(obs) > 0, "observations is empty"
    assert len(doses) > 0, "doses is empty"
    assert len(cov) > 0, "covariates is empty"

    # Patient counts
    n_pts = obs["patient_id"].nunique()
    assert n_pts >= expected_min_patients, (
        f"Expected >= {expected_min_patients} patients, got {n_pts}"
    )

    # sld_mm non-negative (0 is valid: complete response)
    assert (obs["sld_mm"] >= 0).all(), "sld_mm has negative values"

    # time_days sorted within each patient
    for pid, grp in obs.groupby("patient_id"):
        days = grp["time_days"].values
        assert (np.diff(days) >= 0).all(), (
            f"patient {pid}: time_days not sorted"
        )

    # rel_sld first observation near 1.0 per patient
    first_rel = (
        obs.sort_values(["patient_id", "time_days"])
        .groupby("patient_id")["rel_sld"]
        .first()
    )
    assert (first_rel > 0.99).all() and (first_rel < 1.01).all(), (
        "First rel_sld per patient not close to 1.0"
    )

    # patient_id has no surrounding whitespace
    assert (obs["patient_id"] == obs["patient_id"].str.strip()).all(), (
        "patient_id has leading/trailing whitespace"
    )

    # No duplicate (patient_id, time_days) pairs
    dup = obs.duplicated(subset=["patient_id", "time_days"])
    assert not dup.any(), (
        f"Duplicate (patient_id, time_days): {obs[dup][['patient_id','time_days']].head()}"
    )

    # baseline_sld_mm consistent with sld_mm at first timepoint
    first_sld = (
        obs.sort_values(["patient_id", "time_days"])
        .groupby("patient_id", as_index=False)
        .first()[["patient_id", "sld_mm"]]
        .rename(columns={"sld_mm": "expected_baseline"})
    )
    merged = obs.merge(first_sld, on="patient_id")
    mismatch = (merged["baseline_sld_mm"] - merged["expected_baseline"]).abs()
    assert (mismatch < 1e-6).all(), "baseline_sld_mm inconsistent with first sld_mm"


# ---------------------------------------------------------------------------
# Dataset-specific tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (_path("AllProvidedFiles_114")).exists(),
    reason="A6181122 data not present",
)
def test_load_a6181122() -> None:
    ds = load_a6181122(_path("AllProvidedFiles_114"))
    assert ds.study_id == "A6181122"
    _assert_contract(ds, expected_min_patients=370)


@pytest.mark.skipif(
    not (
        _path("AllProvidedFiles_131")
        / "Sanofi_AVE0005_EFC10262_data_files_and_discriptors"
    ).exists(),
    reason="EFC10262 data not present",
)
def test_load_efc10262() -> None:
    root = (
        _path("AllProvidedFiles_131")
        / "Sanofi_AVE0005_EFC10262_data_files_and_discriptors"
    )
    ds = load_efc10262(root, evaluator="INVESTIGATOR")
    assert ds.study_id == "EFC10262"
    _assert_contract(ds, expected_min_patients=580)


@pytest.mark.skipif(
    not (
        _path("AllProvidedFiles_136")
        / "sanofi_sr57746A_efc5505_datasets_and readme"
    ).exists(),
    reason="EFC5505 data not present",
)
def test_load_efc5505() -> None:
    root = (
        _path("AllProvidedFiles_136")
        / "sanofi_sr57746A_efc5505_datasets_and readme"
    )
    ds = load_efc5505(root)
    assert ds.study_id == "EFC5505"
    _assert_contract(ds, expected_min_patients=420)


@pytest.mark.skipif(
    not (
        _path("AllProvidedFiles_137")
        / "sanofi_sr57746A_efc4972_datasets_and readme"
    ).exists(),
    reason="EFC4972 data not present",
)
def test_load_efc4972() -> None:
    root = (
        _path("AllProvidedFiles_137")
        / "sanofi_sr57746A_efc4972_datasets_and readme"
    )
    ds = load_efc4972(root)
    assert ds.study_id == "EFC4972"
    _assert_contract(ds, expected_min_patients=310)


# ---------------------------------------------------------------------------
# filter_patients_for_pkpd
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (_path("AllProvidedFiles_114")).exists(),
    reason="A6181122 data not present",
)
def test_filter_patients_for_pkpd() -> None:
    ds = load_a6181122(_path("AllProvidedFiles_114"))
    filtered = filter_patients_for_pkpd(ds.observations, min_timepoints=3)

    assert len(filtered) > 0
    counts = filtered.groupby("patient_id")["time_days"].count()
    assert (counts >= 3).all(), "filter_patients_for_pkpd: patient with < 3 timepoints present"

    n_full = ds.observations["patient_id"].nunique()
    n_filtered = filtered["patient_id"].nunique()
    assert n_filtered <= n_full, "filter increased patient count"


# ---------------------------------------------------------------------------
# summarize_observations
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (_path("AllProvidedFiles_114")).exists(),
    reason="A6181122 data not present",
)
def test_summarize_observations() -> None:
    ds = load_a6181122(_path("AllProvidedFiles_114"))
    summary = summarize_observations(ds.observations)

    required = {"patient_id", "n_timepoints", "first_day", "last_day",
                "baseline_sld_mm", "min_rel_sld", "max_rel_sld"}
    assert required.issubset(summary.columns)
    assert (summary["n_timepoints"] >= 1).all()
    assert (summary["last_day"] >= summary["first_day"]).all()
