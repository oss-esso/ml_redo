from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClinicalDataset:
    study_id: str
    observations: pd.DataFrame
    doses: pd.DataFrame
    covariates: pd.DataFrame


def _read_sas(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_sas(path, format="sas7bdat", encoding="latin1")


def _clean_str_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _first_treatment_day(
    doses: pd.DataFrame,
    patient_col: str,
    day_col: str,
) -> pd.DataFrame:
    tmp = doses[[patient_col, day_col]].copy()
    tmp[day_col] = _safe_numeric(tmp[day_col])
    tmp = tmp.dropna(subset=[day_col])
    return (
        tmp.groupby(patient_col, as_index=False)[day_col]
        .min()
        .rename(columns={day_col: "treatment_start_day"})
    )


def _add_relative_sld(obs: pd.DataFrame) -> pd.DataFrame:
    obs = obs.sort_values(["study_id", "patient_id", "time_days"]).copy()
    baseline = (
        obs.groupby(["study_id", "patient_id"], as_index=False)
        .first()[["study_id", "patient_id", "sld_mm"]]
        .rename(columns={"sld_mm": "baseline_sld_mm"})
    )
    obs = obs.merge(baseline, on=["study_id", "patient_id"], how="left")
    obs["rel_sld"] = obs["sld_mm"] / obs["baseline_sld_mm"].clip(lower=1e-8)
    return obs


def _finalize_observations(
    obs: pd.DataFrame,
    study_id: str,
    patient_col: str,
    raw_day_col: str,
    value_col: str,
    arm_col: Optional[str] = None,
    treatment_start: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    obs = obs.copy()
    obs["study_id"] = study_id
    obs["patient_id"] = _clean_str_series(obs[patient_col])
    obs["raw_day"] = _safe_numeric(obs[raw_day_col])
    obs["sld_mm"] = _safe_numeric(obs[value_col])

    keep = ["study_id", "patient_id", "raw_day", "sld_mm"]
    if "n_lesions" in obs.columns:
        keep.append("n_lesions")
    else:
        obs["n_lesions"] = np.nan
        keep.append("n_lesions")

    if arm_col is not None and arm_col in obs.columns:
        obs["arm"] = _clean_str_series(obs[arm_col])
    else:
        obs["arm"] = np.nan
    keep.append("arm")

    obs = obs[keep].dropna(subset=["raw_day", "sld_mm"])

    if treatment_start is not None and len(treatment_start) > 0:
        obs = obs.merge(treatment_start, on="patient_id", how="left")
        obs["time_days"] = obs["raw_day"] - obs["treatment_start_day"].fillna(0.0)
        obs = obs.drop(columns=["treatment_start_day"])
    else:
        obs["time_days"] = obs["raw_day"]

    obs = obs.sort_values(["patient_id", "time_days"])
    obs = _add_relative_sld(obs)

    final_cols = [
        "study_id", "patient_id", "raw_day", "time_days",
        "sld_mm", "rel_sld", "baseline_sld_mm", "n_lesions", "arm",
    ]
    return obs[final_cols]


def _finalize_doses(
    doses: pd.DataFrame,
    study_id: str,
    patient_col: str,
    raw_day_col: str,
    drug_col: str,
    amount_col: str,
    unit_col: Optional[str] = None,
    arm_col: Optional[str] = None,
    treatment_start: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    doses = doses.copy()
    doses["study_id"] = study_id
    doses["patient_id"] = _clean_str_series(doses[patient_col])
    doses["raw_day"] = _safe_numeric(doses[raw_day_col])
    doses["drug"] = _clean_str_series(doses[drug_col])
    doses["dose_amount"] = _safe_numeric(doses[amount_col])

    if unit_col is not None and unit_col in doses.columns:
        doses["dose_unit"] = _clean_str_series(doses[unit_col])
    else:
        doses["dose_unit"] = np.nan

    if arm_col is not None and arm_col in doses.columns:
        doses["arm"] = _clean_str_series(doses[arm_col])
    else:
        doses["arm"] = np.nan

    doses = doses[
        ["study_id", "patient_id", "raw_day", "drug", "dose_amount", "dose_unit", "arm"]
    ].dropna(subset=["raw_day", "drug"])

    if treatment_start is not None and len(treatment_start) > 0:
        doses = doses.merge(treatment_start, on="patient_id", how="left")
        doses["time_days"] = doses["raw_day"] - doses["treatment_start_day"].fillna(0.0)
        doses = doses.drop(columns=["treatment_start_day"])
    else:
        doses["time_days"] = doses["raw_day"]

    final_cols = [
        "study_id", "patient_id", "raw_day", "time_days",
        "drug", "dose_amount", "dose_unit", "arm",
    ]
    return doses[final_cols].sort_values(["patient_id", "time_days", "drug"])


def summarize_observations(obs: pd.DataFrame) -> pd.DataFrame:
    return (
        obs.groupby("patient_id")
        .agg(
            n_timepoints=("time_days", "count"),
            first_day=("time_days", "min"),
            last_day=("time_days", "max"),
            baseline_sld_mm=("baseline_sld_mm", "first"),
            min_rel_sld=("rel_sld", "min"),
            max_rel_sld=("rel_sld", "max"),
        )
        .reset_index()
    )


def filter_patients_for_pkpd(
    observations: pd.DataFrame,
    min_timepoints: int = 3,
    require_baseline_near_zero: bool = True,
    baseline_window_days: tuple[float, float] = (-60.0, 14.0),
) -> pd.DataFrame:
    obs = observations.copy()

    counts = obs.groupby("patient_id")["time_days"].count()
    eligible = counts[counts >= min_timepoints].index
    obs = obs[obs["patient_id"].isin(eligible)].copy()

    if require_baseline_near_zero:
        first = obs.groupby("patient_id")["time_days"].min()
        ok = first[
            (first >= baseline_window_days[0])
            & (first <= baseline_window_days[1])
        ].index
        obs = obs[obs["patient_id"].isin(ok)].copy()

    return obs


def load_a6181122(root: str | Path) -> ClinicalDataset:
    """
    Pfizer A6181122.
    Tumor: tmm_p.sas7bdat, Target lesions, TMMDIA summed by PID_A/EFDAY.
    Dose: testdrug.sas7bdat, FROMDAY/TODAY, DOSTOT by DRGNAME.
    Covariates: demog.sas7bdat.
    """
    root = Path(root)

    tmm = _read_sas(root / "tmm_p.sas7bdat")
    testdrug = _read_sas(root / "testdrug.sas7bdat")
    demog = _read_sas(root / "demog.sas7bdat")

    target = tmm[
        (_clean_str_series(tmm["LESTYPE"]).str.upper() == "TARGET")
        & tmm["TMMDIA"].notna()
        & tmm["EFDAY"].notna()
    ].copy()

    sld = (
        target.groupby(["PID_A", "EFDAY"], as_index=False)
        .agg(sld_mm=("TMMDIA", "sum"), n_lesions=("TMMDIA", "count"))
    )

    tx_start = _first_treatment_day(testdrug, "PID_A", "FROMDAY").rename(
        columns={"PID_A": "patient_id"}
    )

    observations = _finalize_observations(
        sld,
        study_id="A6181122",
        patient_col="PID_A",
        raw_day_col="EFDAY",
        value_col="sld_mm",
        treatment_start=tx_start,
    )

    doses = _finalize_doses(
        testdrug,
        study_id="A6181122",
        patient_col="PID_A",
        raw_day_col="FROMDAY",
        drug_col="DRGNAME",
        amount_col="DOSTOT",
        unit_col="DOSTOTUF",
        treatment_start=tx_start,
    )

    covariates = demog.copy()
    covariates["study_id"] = "A6181122"
    covariates["patient_id"] = _clean_str_series(covariates["PID_A"])
    covariates = covariates.rename(
        columns={"AGE": "age", "SEXC": "sex", "RACESC": "race"}
    )
    covariates = covariates[["study_id", "patient_id", "age", "sex", "race"]]
    covariates["arm"] = np.nan

    return ClinicalDataset("A6181122", observations, doses, covariates)


def load_efc10262(
    root: str | Path,
    evaluator: str = "INVESTIGATOR",
) -> ClinicalDataset:
    """
    Sanofi EFC10262, SDTM-like format.
    Tumor: ls.sas7bdat, LSCAT == TARGET, LSSTRESN summed by RSUBJID/LSDY.
    Dose: ex.sas7bdat, EXSTDY, EXDOSE, EXTRT.
    Covariates: dm.sas7bdat.
    """
    root = Path(root)

    ls = _read_sas(root / "ls.sas7bdat")
    ex = _read_sas(root / "ex.sas7bdat")
    dm = _read_sas(root / "dm.sas7bdat")

    ls["LSEVAL_CLEAN"] = _clean_str_series(ls["LSEVAL"]).str.upper()

    target = ls[
        (_clean_str_series(ls["LSCAT"]).str.upper() == "TARGET")
        & ls["LSSTRESN"].notna()
        & ls["LSDY"].notna()
    ].copy()

    if evaluator is not None:
        evaluator_upper = evaluator.upper()
        target = target[target["LSEVAL_CLEAN"] == evaluator_upper].copy()

    sld = (
        target.groupby(["RSUBJID", "LSDY"], as_index=False)
        .agg(sld_mm=("LSSTRESN", "sum"), n_lesions=("LSSTRESN", "count"))
    )

    dm_small = dm[["RSUBJID", "ARM", "AGEC", "SEX", "RACE"]].copy()
    sld = sld.merge(dm_small[["RSUBJID", "ARM"]], on="RSUBJID", how="left")

    tx_start = _first_treatment_day(ex, "RSUBJID", "EXSTDY").rename(
        columns={"RSUBJID": "patient_id"}
    )

    observations = _finalize_observations(
        sld,
        study_id="EFC10262",
        patient_col="RSUBJID",
        raw_day_col="LSDY",
        value_col="sld_mm",
        arm_col="ARM",
        treatment_start=tx_start,
    )

    doses = _finalize_doses(
        ex,
        study_id="EFC10262",
        patient_col="RSUBJID",
        raw_day_col="EXSTDY",
        drug_col="EXTRT",
        amount_col="EXDOSE",
        unit_col="EXDOSU",
        arm_col=None,
        treatment_start=tx_start,
    )

    covariates = dm.copy()
    covariates["study_id"] = "EFC10262"
    covariates["patient_id"] = _clean_str_series(covariates["RSUBJID"])
    covariates = covariates.rename(
        columns={"AGEC": "age", "SEX": "sex", "RACE": "race", "ARM": "arm"}
    )
    covariates = covariates[["study_id", "patient_id", "age", "sex", "race", "arm"]]

    return ClinicalDataset("EFC10262", observations, doses, covariates)


def load_efc55xx(root: str | Path, study_id: str) -> ClinicalDataset:
    """
    Shared loader for Sanofi EFC5505 and EFC4972.
    Tumor: measur.sas7bdat, MSDIM1 summed by RSUBJID/MSDY.
    Dose: dose.sas7bdat, DOSTRDY, DOSACNB, DRUG.
    Covariates: demog.sas7bdat.
    """
    root = Path(root)

    measur = _read_sas(root / "measur.sas7bdat")
    dose = _read_sas(root / "dose.sas7bdat")
    demog = _read_sas(root / "demog.sas7bdat")

    m = measur[measur["MSDIM1"].notna() & measur["MSDY"].notna()].copy()

    sld = (
        m.groupby(["RSUBJID", "MSDY"], as_index=False)
        .agg(sld_mm=("MSDIM1", "sum"), n_lesions=("MSDIM1", "count"))
    )

    if "TRTNAME" in measur.columns:
        arm_map = (
            measur[["RSUBJID", "TRTNAME"]]
            .dropna()
            .drop_duplicates("RSUBJID")
            .rename(columns={"TRTNAME": "arm"})
        )
        sld = sld.merge(arm_map, on="RSUBJID", how="left")

    tx_start = _first_treatment_day(dose, "RSUBJID", "DOSTRDY").rename(
        columns={"RSUBJID": "patient_id"}
    )

    observations = _finalize_observations(
        sld,
        study_id=study_id,
        patient_col="RSUBJID",
        raw_day_col="MSDY",
        value_col="sld_mm",
        arm_col="arm",
        treatment_start=tx_start,
    )

    doses = _finalize_doses(
        dose,
        study_id=study_id,
        patient_col="RSUBJID",
        raw_day_col="DOSTRDY",
        drug_col="DRUG",
        amount_col="DOSACNB",
        unit_col="DOSACUN",
        arm_col="TRTNAME",
        treatment_start=tx_start,
    )

    covariates = demog.copy()
    covariates["study_id"] = study_id
    covariates["patient_id"] = _clean_str_series(covariates["RSUBJID"])
    covariates = covariates.rename(
        columns={"AGE": "age", "SEX": "sex", "RACE": "race", "TRTNAME": "arm"}
    )
    covariates = covariates[["study_id", "patient_id", "age", "sex", "race", "arm"]]

    return ClinicalDataset(study_id, observations, doses, covariates)


def load_efc5505(root: str | Path) -> ClinicalDataset:
    return load_efc55xx(root, study_id="EFC5505")


def load_efc4972(root: str | Path) -> ClinicalDataset:
    return load_efc55xx(root, study_id="EFC4972")


def concatenate_clinical_datasets(datasets: list[ClinicalDataset]) -> ClinicalDataset:
    study_id = "+".join(ds.study_id for ds in datasets)
    observations = pd.concat([ds.observations for ds in datasets], ignore_index=True)
    doses = pd.concat([ds.doses for ds in datasets], ignore_index=True)
    covariates = pd.concat([ds.covariates for ds in datasets], ignore_index=True)
    return ClinicalDataset(study_id, observations, doses, covariates)
