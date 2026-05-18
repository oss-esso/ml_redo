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


# =============================================================================
# Whole-tree discovery and best-effort loaders
# =============================================================================
#
# Existing exact loaders:
#   - A6181122
#   - EFC10262
#   - EFC5505
#   - EFC4972
#
# Additional best-effort loaders added here:
#   - AstraZeneca Cediranib / AllProvidedFiles_78
#   - Amgen legacy folders with lesion/exposure/demo
#   - Amgen ADaM/PDS folders with adrsp/adsl-style tables
#
# These extra loaders are intentionally schema-tolerant. They inspect common
# oncology column names and canonicalize what they can.
# =============================================================================

import re
from typing import Callable


@dataclass(frozen=True)
class ClinicalDatasetCandidate:
    study_id: str
    root: Path
    loader_name: str
    loader: Callable[[Path], ClinicalDataset]
    schema: str


@dataclass(frozen=True)
class ClinicalTreeLoadResult:
    datasets: list[ClinicalDataset]
    report: pd.DataFrame


OBS_CANONICAL_COLS = [
    "study_id", "patient_id", "raw_day", "time_days",
    "sld_mm", "rel_sld", "baseline_sld_mm", "n_lesions", "arm",
]

DOSE_CANONICAL_COLS = [
    "study_id", "patient_id", "raw_day", "time_days",
    "drug", "dose_amount", "dose_unit", "arm",
]

COV_CANONICAL_COLS = [
    "study_id", "patient_id", "age", "sex", "race", "arm",
]


def _empty_observations() -> pd.DataFrame:
    return pd.DataFrame(columns=OBS_CANONICAL_COLS)


def _empty_doses() -> pd.DataFrame:
    return pd.DataFrame(columns=DOSE_CANONICAL_COLS)


def _empty_covariates() -> pd.DataFrame:
    return pd.DataFrame(columns=COV_CANONICAL_COLS)


def _find_first_file(root: str | Path, filename: str) -> Optional[Path]:
    root = Path(root)
    filename_lower = filename.lower()
    for p in root.rglob("*"):
        if p.is_file() and p.name.lower() == filename_lower:
            return p
    return None


def _has_files(folder: Path, required_names: set[str]) -> bool:
    if not folder.exists() or not folder.is_dir():
        return False
    names = {p.name.lower() for p in folder.iterdir() if p.is_file()}
    return {x.lower() for x in required_names}.issubset(names)


def _first_existing_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def _first_col_containing(df: pd.DataFrame, tokens: list[str]) -> Optional[str]:
    tokens = [t.lower() for t in tokens]
    for c in df.columns:
        lc = c.lower()
        if all(t in lc for t in tokens):
            return c
    return None


def _infer_patient_col(df: pd.DataFrame) -> str:
    col = _first_existing_col(
        df,
        [
            "USUBJID", "SUBJID", "SUBJECT", "SUBJECTID", "SUBJECT_ID",
            "PATIENT", "PATIENTID", "PID", "PID_A", "RSUBJID",
            "SUBID", "SUBNUM", "ID",
        ],
    )
    if col is not None:
        return col

    col = _first_col_containing(df, ["subj"])
    if col is not None:
        return col

    raise ValueError(f"Could not infer patient column. Columns={list(df.columns)}")


def _infer_day_col(df: pd.DataFrame) -> str:
    col = _first_existing_col(
        df,
        [
            "ADY", "DY", "DAY", "STDY", "VISITDY", "VISITDAY",
            "LSDY", "MSDY", "EFDAY", "EXSTDY", "DOSTRDY",
            "RCDY", "RSDY", "TRDY", "TLDY", "ASTDY", "AENDY",
        ],
    )
    if col is not None:
        return col

    for tokens in [["visit", "day"], ["study", "day"], ["assess", "day"]]:
        col = _first_col_containing(df, tokens)
        if col is not None:
            return col

    raise ValueError(f"Could not infer day column. Columns={list(df.columns)}")


def _infer_tumor_value_col(df: pd.DataFrame) -> str:
    col = _first_existing_col(
        df,
        [
            "SLD", "SUMDIAM", "SUM_DIAM", "SUMDIAMETER", "SUM_DIAMETER",
            "TARGETSUM", "TARGET_SUM", "TRGTSLD", "TLDIAM", "TLDSUM",
            "LSSTRESN", "MSDIM1", "TMMDIA", "DIAMETER", "DIAM",
            "LDIAM", "AVAL",
        ],
    )
    if col is not None:
        return col

    for tokens in [["sum", "diam"], ["target", "diam"], ["lesion", "diam"]]:
        col = _first_col_containing(df, tokens)
        if col is not None:
            return col

    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]
    raise ValueError(
        "Could not infer tumor measurement column. "
        f"Numeric columns={numeric_cols}; all columns={list(df.columns)}"
    )


def _infer_drug_col(df: pd.DataFrame) -> Optional[str]:
    return _first_existing_col(
        df,
        [
            "TRT", "TRT01P", "TRT01A", "EXTRT", "DRUG", "DRGNAME",
            "CMTRT", "TREATMENT", "TREAT", "MEDICATION", "THERAPY",
        ],
    )


def _infer_dose_amount_col(df: pd.DataFrame) -> Optional[str]:
    return _first_existing_col(
        df,
        [
            "DOSE", "EXDOSE", "DOSACNB", "DOSTOT", "DOSE_AMOUNT",
            "AMT", "AMOUNT", "AVAL",
        ],
    )


def _infer_unit_col(df: pd.DataFrame) -> Optional[str]:
    return _first_existing_col(
        df,
        [
            "DOSU", "EXDOSU", "DOSACUN", "DOSTOTUF", "UNIT", "DOSE_UNIT",
            "AVALU",
        ],
    )


def _infer_arm_col(df: pd.DataFrame) -> Optional[str]:
    return _first_existing_col(
        df,
        ["ARM", "TRTNAME", "TRT01P", "TRT01A", "ACTARM", "TREATMENT_ARM"],
    )


def _filter_target_like_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Best-effort filtering for target-lesion/tumor-burden rows.

    If no recognizable lesion/category columns exist, returns df unchanged.
    """
    out = df.copy()

    for col in out.columns:
        u = out[col].astype(str).str.upper()

        col_u = col.upper()
        if col_u in {"LESTYPE", "LSCAT", "LSTYPE", "TUTYPE", "LESIONTYPE"}:
            mask = u.str.contains("TARGET", na=False)
            if mask.any():
                return out[mask].copy()

        if col_u in {"PARAM", "PARAMCD", "TEST", "TESTCD", "LBTEST", "AVALCAT"}:
            pattern = r"SLD|SUM.*DIAM|TARGET.*LESION|TUMOU?R.*BURDEN|TLD|TLDSUM"
            mask = u.str.contains(pattern, regex=True, na=False)
            if mask.any():
                return out[mask].copy()

    return out


def _generic_observations_from_measurements(
    meas: pd.DataFrame,
    study_id: str,
    treatment_start: Optional[pd.DataFrame] = None,
    arm_source: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    m = _filter_target_like_rows(meas)

    patient_col = _infer_patient_col(m)
    day_col = _infer_day_col(m)
    value_col = _infer_tumor_value_col(m)

    arm_col = _infer_arm_col(m)
    if arm_col is None and arm_source is not None:
        try:
            arm_pid = _infer_patient_col(arm_source)
            arm_name = _infer_arm_col(arm_source)
            if arm_name is not None:
                arm_map = (
                    arm_source[[arm_pid, arm_name]]
                    .dropna()
                    .drop_duplicates(arm_pid)
                    .rename(columns={arm_pid: "_pid", arm_name: "_arm"})
                )
                m = m.copy()
                m["_pid_tmp"] = _clean_str_series(m[patient_col])
                m = m.merge(arm_map, left_on="_pid_tmp", right_on="_pid", how="left")
                arm_col = "_arm"
        except Exception:
            arm_col = None

    tmp = m.copy()
    tmp["_pid"] = _clean_str_series(tmp[patient_col])
    tmp["_day"] = _safe_numeric(tmp[day_col])
    tmp["_value"] = _safe_numeric(tmp[value_col])

    if arm_col is not None and arm_col in tmp.columns:
        tmp["_arm"] = _clean_str_series(tmp[arm_col])
    else:
        tmp["_arm"] = np.nan

    tmp = tmp.dropna(subset=["_pid", "_day", "_value"])

    if tmp.empty:
        raise ValueError("Measurement table produced no usable observation rows.")

    sld = (
        tmp.groupby(["_pid", "_day"], as_index=False)
        .agg(
            sld_mm=("_value", "sum"),
            n_lesions=("_value", "count"),
            arm=("_arm", "first"),
        )
    )

    return _finalize_observations(
        sld,
        study_id=study_id,
        patient_col="_pid",
        raw_day_col="_day",
        value_col="sld_mm",
        arm_col="arm",
        treatment_start=treatment_start,
    )


def _generic_doses_from_table(
    dose: Optional[pd.DataFrame],
    study_id: str,
    treatment_start_out: bool = False,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    if dose is None or dose.empty:
        return _empty_doses(), None

    d = dose.copy()

    patient_col = _infer_patient_col(d)
    day_col = _infer_day_col(d)

    drug_col = _infer_drug_col(d)
    if drug_col is None:
        d["_drug"] = "UNKNOWN"
        drug_col = "_drug"

    amount_col = _infer_dose_amount_col(d)
    if amount_col is None:
        d["_amount"] = np.nan
        amount_col = "_amount"

    unit_col = _infer_unit_col(d)

    arm_col = _infer_arm_col(d)

    tx_start = _first_treatment_day(d, patient_col, day_col).rename(
        columns={patient_col: "patient_id"}
    )

    doses = _finalize_doses(
        d,
        study_id=study_id,
        patient_col=patient_col,
        raw_day_col=day_col,
        drug_col=drug_col,
        amount_col=amount_col,
        unit_col=unit_col,
        arm_col=arm_col,
        treatment_start=tx_start,
    )

    return doses, tx_start


def _generic_covariates_from_table(
    cov: Optional[pd.DataFrame],
    study_id: str,
) -> pd.DataFrame:
    if cov is None or cov.empty:
        return _empty_covariates()

    c = cov.copy()
    patient_col = _infer_patient_col(c)

    age_col = _first_existing_col(c, ["AGE", "AGEC", "AGE_Y", "AGE_YRS"])
    sex_col = _first_existing_col(c, ["SEX", "SEXC", "GENDER"])
    race_col = _first_existing_col(c, ["RACE", "RACESC", "ETHNIC", "ETHNICITY"])
    arm_col = _infer_arm_col(c)

    out = pd.DataFrame()
    out["study_id"] = study_id
    out["patient_id"] = _clean_str_series(c[patient_col])
    out["age"] = _safe_numeric(c[age_col]) if age_col else np.nan
    out["sex"] = _clean_str_series(c[sex_col]) if sex_col else np.nan
    out["race"] = _clean_str_series(c[race_col]) if race_col else np.nan
    out["arm"] = _clean_str_series(c[arm_col]) if arm_col else np.nan

    return out[COV_CANONICAL_COLS].drop_duplicates("patient_id")


def _study_id_from_path(root: Path, default: str) -> str:
    s = str(root)
    m = re.search(r"(20\d{6})", s)
    if m:
        return f"AMGEN_{m.group(1)}"

    m = re.search(r"AllProvidedFiles_(\d+)", s, flags=re.IGNORECASE)
    if m:
        return f"DATASET_{m.group(1)}"

    return default


def load_amgen_legacy(root: str | Path, study_id: Optional[str] = None) -> ClinicalDataset:
    """
    Best-effort loader for Amgen legacy folders such as:
        lesion.sas7bdat
        exposure.sas7bdat
        demo.sas7bdat

    Example folders:
        AllProvidedFiles_262/.../SAS dataset - 20040249
        AllProvidedFiles_263
        AllProvidedFiles_264
    """
    root = Path(root)
    study_id = study_id or _study_id_from_path(root, "AMGEN_LEGACY")

    lesion_path = _find_first_file(root, "lesion.sas7bdat")
    exposure_path = _find_first_file(root, "exposure.sas7bdat")
    demo_path = _find_first_file(root, "demo.sas7bdat")

    if lesion_path is None:
        raise FileNotFoundError(f"Could not find lesion.sas7bdat under {root}")

    lesion = _read_sas(lesion_path)
    exposure = _read_sas(exposure_path) if exposure_path else None
    demo = _read_sas(demo_path) if demo_path else None

    doses, tx_start = _generic_doses_from_table(exposure, study_id)
    covariates = _generic_covariates_from_table(demo, study_id)

    observations = _generic_observations_from_measurements(
        lesion,
        study_id=study_id,
        treatment_start=tx_start,
        arm_source=demo,
    )

    return ClinicalDataset(study_id, observations, doses, covariates)


def load_amgen_adam_pds(root: str | Path, study_id: Optional[str] = None) -> ClinicalDataset:
    """
    Best-effort loader for Amgen ADaM/PDS folders such as:
        adrsp_pds2019.sas7bdat
        adsl_pds2019.sas7bdat

    Example folders:
        AllProvidedFiles_309/PDS_DSA_20050203
        AllProvidedFiles_310/PDS_DSA_20020408
    """
    root = Path(root)
    study_id = study_id or _study_id_from_path(root, "AMGEN_ADAM_PDS")

    adrsp_path = _find_first_file(root, "adrsp_pds2019.sas7bdat")
    adsl_path = _find_first_file(root, "adsl_pds2019.sas7bdat")
    adexposure_path = (
        _find_first_file(root, "adex_pds2019.sas7bdat")
        or _find_first_file(root, "adex.sas7bdat")
        or _find_first_file(root, "adsl_pds2019.sas7bdat")
    )

    if adrsp_path is None:
        raise FileNotFoundError(f"Could not find adrsp_pds2019.sas7bdat under {root}")

    adrsp = _read_sas(adrsp_path)
    adsl = _read_sas(adsl_path) if adsl_path else None
    adex = _read_sas(adexposure_path) if adexposure_path else None

    # ADaM response tables often store tumor measurements as PARAM/PARAMCD + AVAL.
    # The generic observation function filters SLD / sum-diameter-like PARAM rows.
    doses, tx_start = _generic_doses_from_table(adex, study_id)
    covariates = _generic_covariates_from_table(adsl, study_id)

    observations = _generic_observations_from_measurements(
        adrsp,
        study_id=study_id,
        treatment_start=tx_start,
        arm_source=adsl,
    )

    return ClinicalDataset(study_id, observations, doses, covariates)


def load_astrazeneca_cediranib(root: str | Path) -> ClinicalDataset:
    """
    Best-effort loader for AllProvidedFiles_78 Cediranib Horizon III.

    Expected files somewhere under root:
        rptarget.sas7bdat   target lesion measurements
        rpdosad.sas7bdat    dose administration
        rpdem.sas7bdat      demographics
    """
    root = Path(root)
    study_id = "CEDIRANIB_HORIZONIII"

    target_path = _find_first_file(root, "rptarget.sas7bdat")
    dose_path = _find_first_file(root, "rpdosad.sas7bdat")
    dem_path = _find_first_file(root, "rpdem.sas7bdat")

    if target_path is None:
        raise FileNotFoundError(f"Could not find rptarget.sas7bdat under {root}")

    target = _read_sas(target_path)
    dose = _read_sas(dose_path) if dose_path else None
    dem = _read_sas(dem_path) if dem_path else None

    doses, tx_start = _generic_doses_from_table(dose, study_id)
    covariates = _generic_covariates_from_table(dem, study_id)

    observations = _generic_observations_from_measurements(
        target,
        study_id=study_id,
        treatment_start=tx_start,
        arm_source=dem,
    )

    return ClinicalDataset(study_id, observations, doses, covariates)


def discover_clinical_datasets(data_root: str | Path) -> list[ClinicalDatasetCandidate]:
    """
    Discover all known dataset schemas under the full data tree.
    """
    data_root = Path(data_root)
    folders = [data_root] + [p for p in data_root.rglob("*") if p.is_dir()]

    candidates: list[ClinicalDatasetCandidate] = []
    seen: set[tuple[str, Path, str]] = set()

    def add(study_id: str, root: Path, loader_name: str, loader, schema: str) -> None:
        key = (study_id, root.resolve(), loader_name)
        if key not in seen:
            candidates.append(
                ClinicalDatasetCandidate(
                    study_id=study_id,
                    root=root,
                    loader_name=loader_name,
                    loader=loader,
                    schema=schema,
                )
            )
            seen.add(key)

    for folder in folders:
        # Existing exact schemas.
        if _has_files(folder, {"tmm_p.sas7bdat", "testdrug.sas7bdat", "demog.sas7bdat"}):
            add("A6181122", folder, "load_a6181122", load_a6181122, "pfizer_a6181122")

        if _has_files(folder, {"ls.sas7bdat", "ex.sas7bdat", "dm.sas7bdat"}):
            add(
                "EFC10262",
                folder,
                "load_efc10262",
                lambda root: load_efc10262(root, evaluator="INVESTIGATOR"),
                "sanofi_sdtm_efc10262",
            )

        if _has_files(folder, {"measur.sas7bdat", "dose.sas7bdat", "demog.sas7bdat"}):
            text = str(folder).lower()
            if "5505" in text:
                add("EFC5505", folder, "load_efc5505", load_efc5505, "sanofi_efc55xx")
            elif "4972" in text:
                add("EFC4972", folder, "load_efc4972", load_efc4972, "sanofi_efc55xx")
            else:
                add("EFC55XX_UNKNOWN", folder, "load_efc5505", load_efc5505, "sanofi_efc55xx_unknown")

        # AstraZeneca Cediranib.
        if _has_files(folder, {"rptarget.sas7bdat", "rpdosad.sas7bdat", "rpdem.sas7bdat"}):
            add(
                "CEDIRANIB_HORIZONIII",
                folder,
                "load_astrazeneca_cediranib",
                load_astrazeneca_cediranib,
                "astrazeneca_cediranib_best_effort",
            )

        # Amgen legacy.
        if _has_files(folder, {"lesion.sas7bdat", "exposure.sas7bdat", "demo.sas7bdat"}):
            sid = _study_id_from_path(folder, "AMGEN_LEGACY")
            add(
                sid,
                folder,
                "load_amgen_legacy",
                lambda root, _sid=sid: load_amgen_legacy(root, study_id=_sid),
                "amgen_legacy_best_effort",
            )

        # Amgen ADaM/PDS.
        if _has_files(folder, {"adrsp_pds2019.sas7bdat", "adsl_pds2019.sas7bdat"}):
            sid = _study_id_from_path(folder, "AMGEN_ADAM_PDS")
            add(
                sid,
                folder,
                "load_amgen_adam_pds",
                lambda root, _sid=sid: load_amgen_adam_pds(root, study_id=_sid),
                "amgen_adam_pds_best_effort",
            )

    return candidates


def load_clinical_tree(
    data_root: str | Path,
    strict: bool = False,
) -> ClinicalTreeLoadResult:
    """
    Load every discoverable clinical dataset under data_root.

    If strict=False:
        Continue after failures and record them in the report.

    If strict=True:
        Raise immediately on the first failed loader.
    """
    candidates = discover_clinical_datasets(data_root)

    datasets: list[ClinicalDataset] = []
    rows: list[dict] = []

    for c in candidates:
        row = {
            "study_id": c.study_id,
            "root": str(c.root),
            "loader_name": c.loader_name,
            "schema": c.schema,
            "status": "not_run",
            "error": "",
            "n_observations": 0,
            "n_doses": 0,
            "n_covariates": 0,
            "n_patients_observed": 0,
        }

        try:
            ds = c.loader(c.root)
            datasets.append(ds)

            row.update(
                {
                    "status": "loaded",
                    "n_observations": int(len(ds.observations)),
                    "n_doses": int(len(ds.doses)),
                    "n_covariates": int(len(ds.covariates)),
                    "n_patients_observed": int(ds.observations["patient_id"].nunique())
                    if len(ds.observations)
                    else 0,
                }
            )

        except Exception as exc:
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
            if strict:
                raise

        rows.append(row)

    return ClinicalTreeLoadResult(
        datasets=datasets,
        report=pd.DataFrame(rows),
    )


# =============================================================================
# Schema-specific fixes for failed best-effort loaders
# =============================================================================
#
# Add this at the bottom of clinical_loaders.py.
#
# Fixes:
#   - Amgen legacy:
#       lesion.sas7bdat:   SUBJID, DOSREFDY, LSSLD
#       exposure.sas7bdat: SUBJID, DOSREFDY, ACTTRT, DOSE, DOSEUNIT
#
#   - Amgen ADaM/PDS:
#       adls_pds2019.sas7bdat: SUBJID, VISITDY, LSSLD
#       adsl_pds2019.sas7bdat: covariates
#
#   - AstraZeneca Cediranib:
#       rdprcist.sas7bdat: RANDCODE, ORDYTRT, STLDI
#       rpdosad.sas7bdat:  RANDCODE, SD_SDY, SDDOSE, ACTTRTXT
#       rpdem.sas7bdat:   demographics
# =============================================================================

from typing import Callable


def _ensure_cols(df: pd.DataFrame, required: list[str], table_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{table_name} missing required columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def _make_observations_from_sld(
    df: pd.DataFrame,
    study_id: str,
    patient_col: str,
    day_col: str,
    sld_col: str,
    arm_col: Optional[str] = None,
    treatment_start: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build canonical observations when the source table already contains SLD.

    Important:
        For some datasets, LSSLD/STLDI is already the sum of lesion diameters.
        Therefore we must NOT sum it again across lesion rows/readers.
        We collapse duplicate patient/day rows by median SLD.
    """
    _ensure_cols(df, [patient_col, day_col, sld_col], f"{study_id} observations source")

    tmp = df.copy()
    tmp["_pid"] = _clean_str_series(tmp[patient_col])
    tmp["_day"] = _safe_numeric(tmp[day_col])
    tmp["_sld"] = _safe_numeric(tmp[sld_col])

    if arm_col is not None and arm_col in tmp.columns:
        tmp["_arm"] = _clean_str_series(tmp[arm_col])
    else:
        tmp["_arm"] = np.nan

    tmp = tmp.dropna(subset=["_pid", "_day", "_sld"])
    tmp = tmp[tmp["_sld"] > 0].copy()

    if tmp.empty:
        raise ValueError(f"{study_id}: no usable positive SLD rows from {sld_col}")

    collapsed = (
        tmp.groupby(["_pid", "_day"], as_index=False)
        .agg(
            sld_mm=("_sld", "median"),
            n_lesions=("_sld", "count"),
            arm=("_arm", "first"),
        )
    )

    return _finalize_observations(
        collapsed,
        study_id=study_id,
        patient_col="_pid",
        raw_day_col="_day",
        value_col="sld_mm",
        arm_col="arm",
        treatment_start=treatment_start,
    )


def _make_covariates_from_known_cols(
    df: Optional[pd.DataFrame],
    study_id: str,
    patient_col: str,
    age_col: Optional[str] = None,
    sex_col: Optional[str] = None,
    race_col: Optional[str] = None,
    arm_col: Optional[str] = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_covariates()

    _ensure_cols(df, [patient_col], f"{study_id} covariates source")

    out = pd.DataFrame()
    out["study_id"] = study_id
    out["patient_id"] = _clean_str_series(df[patient_col])
    out["age"] = _safe_numeric(df[age_col]) if age_col and age_col in df.columns else np.nan
    out["sex"] = _clean_str_series(df[sex_col]) if sex_col and sex_col in df.columns else np.nan
    out["race"] = _clean_str_series(df[race_col]) if race_col and race_col in df.columns else np.nan
    out["arm"] = _clean_str_series(df[arm_col]) if arm_col and arm_col in df.columns else np.nan

    return out[COV_CANONICAL_COLS].drop_duplicates("patient_id")


def _make_doses_from_known_cols(
    df: Optional[pd.DataFrame],
    study_id: str,
    patient_col: str,
    day_col: str,
    drug_col: Optional[str] = None,
    amount_col: Optional[str] = None,
    unit_col: Optional[str] = None,
    arm_col: Optional[str] = None,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    if df is None or df.empty:
        return _empty_doses(), None

    _ensure_cols(df, [patient_col, day_col], f"{study_id} dose source")

    tmp = df.copy()
    tmp["_pid"] = _clean_str_series(tmp[patient_col])
    tmp["_day"] = _safe_numeric(tmp[day_col])

    if drug_col is not None and drug_col in tmp.columns:
        tmp["_drug"] = _clean_str_series(tmp[drug_col])
    else:
        tmp["_drug"] = "UNKNOWN"

    if amount_col is not None and amount_col in tmp.columns:
        tmp["_amount"] = _safe_numeric(tmp[amount_col])
    else:
        tmp["_amount"] = np.nan

    if unit_col is not None and unit_col in tmp.columns:
        tmp["_unit"] = _clean_str_series(tmp[unit_col])
    else:
        tmp["_unit"] = np.nan

    if arm_col is not None and arm_col in tmp.columns:
        tmp["_arm"] = _clean_str_series(tmp[arm_col])
    else:
        tmp["_arm"] = np.nan

    tmp = tmp.dropna(subset=["_pid", "_day", "_drug"])

    if tmp.empty:
        return _empty_doses(), None

    tx_start = _first_treatment_day(tmp, "_pid", "_day").rename(
        columns={"_pid": "patient_id"}
    )

    doses = _finalize_doses(
        tmp,
        study_id=study_id,
        patient_col="_pid",
        raw_day_col="_day",
        drug_col="_drug",
        amount_col="_amount",
        unit_col="_unit",
        arm_col="_arm",
        treatment_start=tx_start,
    )

    return doses, tx_start


def _find_first_file(root: str | Path, filename: str) -> Optional[Path]:
    root = Path(root)
    target = filename.lower()
    for p in root.rglob("*"):
        if p.is_file() and p.name.lower() == target:
            return p
    return None


def _study_id_from_path(root: Path, default: str) -> str:
    text = str(root)

    m = re.search(r"(20\d{6})", text)
    if m:
        return f"AMGEN_{m.group(1)}"

    m = re.search(r"AllProvidedFiles_(\d+)", text, flags=re.IGNORECASE)
    if m:
        return f"DATASET_{m.group(1)}"

    return default


def load_amgen_legacy(root: str | Path, study_id: Optional[str] = None) -> ClinicalDataset:
    """
    Loader for Amgen legacy studies.

    Observations:
        lesion.sas7bdat
        SUBJID   = patient id
        DOSREFDY = day relative to dose/treatment reference
        LSSLD    = sum of longest diameters

    Doses:
        exposure.sas7bdat
        DOSREFDY, ACTTRT, DOSE, DOSEUNIT

    Covariates:
        demo.sas7bdat
    """
    root = Path(root)
    study_id = study_id or _study_id_from_path(root, "AMGEN_LEGACY")

    lesion_path = _find_first_file(root, "lesion.sas7bdat")
    exposure_path = _find_first_file(root, "exposure.sas7bdat")
    demo_path = _find_first_file(root, "demo.sas7bdat")

    if lesion_path is None:
        raise FileNotFoundError(f"{study_id}: could not find lesion.sas7bdat under {root}")

    lesion = _read_sas(lesion_path)
    exposure = _read_sas(exposure_path) if exposure_path else None
    demo = _read_sas(demo_path) if demo_path else None

    doses, tx_start = _make_doses_from_known_cols(
        exposure,
        study_id=study_id,
        patient_col="SUBJID",
        day_col="DOSREFDY",
        drug_col="ACTTRT",
        amount_col="DOSE",
        unit_col="DOSEUNIT",
        arm_col="ATRT",
    )

    observations = _make_observations_from_sld(
        lesion,
        study_id=study_id,
        patient_col="SUBJID",
        day_col="DOSREFDY",
        sld_col="LSSLD",
        arm_col="ATRT",
        treatment_start=tx_start,
    )

    covariates = _make_covariates_from_known_cols(
        demo,
        study_id=study_id,
        patient_col="SUBJID",
        age_col="AGE",
        sex_col="SEX",
        race_col="RACCAT",
        arm_col="ATRT",
    )

    return ClinicalDataset(study_id, observations, doses, covariates)


def load_amgen_adam_pds(root: str | Path, study_id: Optional[str] = None) -> ClinicalDataset:
    """
    Loader for Amgen ADaM/PDS studies.

    Key discovery:
        adls_pds2019.sas7bdat is not just subject-level info here.
        It contains longitudinal lesion/SLD rows:

            SUBJID, VISITDY, VISIT, LSCAT, LSSLD, LSLD, ...

        adrsp_pds2019.sas7bdat contains response categories, not SLD values.
    """
    root = Path(root)
    study_id = study_id or _study_id_from_path(root, "AMGEN_ADAM_PDS")

    adls_lesion_path = _find_first_file(root, "adls_pds2019.sas7bdat")
    adsl_path = _find_first_file(root, "adsl_pds2019.sas7bdat")

    if adls_lesion_path is None:
        raise FileNotFoundError(f"{study_id}: could not find adls_pds2019.sas7bdat under {root}")

    adls_lesion = _read_sas(adls_lesion_path)
    adsl = _read_sas(adsl_path) if adsl_path else None

    # Keep target-lesion rows if the category exists.
    if "LSCAT" in adls_lesion.columns:
        mask = _clean_str_series(adls_lesion["LSCAT"]).str.upper().str.contains("TARGET", na=False)
        if mask.any():
            adls_lesion = adls_lesion[mask].copy()

    # Merge treatment arm into the lesion table.
    if adsl is not None and "SUBJID" in adsl.columns:
        arm_cols = [c for c in ["ATRT", "TRT"] if c in adsl.columns]
        if arm_cols:
            arm_col = arm_cols[0]
            arm_map = (
                adsl[["SUBJID", arm_col]]
                .dropna()
                .drop_duplicates("SUBJID")
                .rename(columns={arm_col: "_arm"})
            )
            adls_lesion = adls_lesion.merge(arm_map, on="SUBJID", how="left")
        else:
            adls_lesion["_arm"] = np.nan
    else:
        adls_lesion["_arm"] = np.nan

    observations = _make_observations_from_sld(
        adls_lesion,
        study_id=study_id,
        patient_col="SUBJID",
        day_col="VISITDY",
        sld_col="LSSLD",
        arm_col="_arm",
        treatment_start=None,
    )

    # No reliable longitudinal dose table in the profiled files.
    # Keep an empty dose table rather than failing.
    doses = _empty_doses()

    covariates = _make_covariates_from_known_cols(
        adsl,
        study_id=study_id,
        patient_col="SUBJID",
        age_col="AGE",
        sex_col="SEX",
        race_col="RACE",
        arm_col="ATRT" if adsl is not None and "ATRT" in adsl.columns else "TRT",
    )

    return ClinicalDataset(study_id, observations, doses, covariates)


def load_astrazeneca_cediranib(root: str | Path) -> ClinicalDataset:
    """
    Loader for AstraZeneca Cediranib Horizon III.

    Use patient-level RECIST summary table instead of lesion-level rptarget:

        rdprcist.sas7bdat:
            RANDCODE = patient id
            ORDYTRT  = day relative to treatment
            STLDI    = sum of target lesion diameters
            ACTTRTXT = actual treatment text

    Doses:
        rpdosad.sas7bdat:
            RANDCODE, SD_SDY, SDDOSE, SDDOSE_U, ACTTRTXT

    Covariates:
        rpdem.sas7bdat:
            RANDCODE, SEX, RACE, ACTTRTXT
    """
    root = Path(root)
    study_id = "CEDIRANIB_HORIZONIII"

    recist_path = _find_first_file(root, "rdprcist.sas7bdat")
    dose_path = _find_first_file(root, "rpdosad.sas7bdat")
    dem_path = _find_first_file(root, "rpdem.sas7bdat")

    if recist_path is None:
        raise FileNotFoundError(f"{study_id}: could not find rdprcist.sas7bdat under {root}")

    recist = _read_sas(recist_path)
    dose = _read_sas(dose_path) if dose_path else None
    dem = _read_sas(dem_path) if dem_path else None

    doses, tx_start = _make_doses_from_known_cols(
        dose,
        study_id=study_id,
        patient_col="RANDCODE",
        day_col="SD_SDY" if dose is not None and "SD_SDY" in dose.columns else "VIS_DY",
        drug_col="ACTTRTXT",
        amount_col="SDDOSE" if dose is not None and "SDDOSE" in dose.columns else None,
        unit_col="SDDOSE_U" if dose is not None and "SDDOSE_U" in dose.columns else None,
        arm_col="ACTTRTXT",
    )

    observations = _make_observations_from_sld(
        recist,
        study_id=study_id,
        patient_col="RANDCODE",
        day_col="ORDYTRT" if "ORDYTRT" in recist.columns else "VIS_DY",
        sld_col="STLDI",
        arm_col="ACTTRTXT" if "ACTTRTXT" in recist.columns else None,
        treatment_start=None,
    )

    covariates = _make_covariates_from_known_cols(
        dem,
        study_id=study_id,
        patient_col="RANDCODE",
        age_col=None,
        sex_col="SEX",
        race_col="RACE",
        arm_col="ACTTRTXT" if dem is not None and "ACTTRTXT" in dem.columns else None,
    )

    return ClinicalDataset(study_id, observations, doses, covariates)


def discover_clinical_datasets(data_root: str | Path) -> list[ClinicalDatasetCandidate]:
    """
    Discover all known dataset schemas under the full data tree.

    This overrides the earlier version with corrected mappings for:
        - Amgen legacy
        - Amgen ADaM/PDS
        - AstraZeneca Cediranib
    """
    data_root = Path(data_root)
    folders = [data_root] + [p for p in data_root.rglob("*") if p.is_dir()]

    candidates: list[ClinicalDatasetCandidate] = []
    seen: set[tuple[str, Path, str]] = set()

    def add(study_id: str, root: Path, loader_name: str, loader, schema: str) -> None:
        key = (study_id, root.resolve(), loader_name)
        if key not in seen:
            candidates.append(
                ClinicalDatasetCandidate(
                    study_id=study_id,
                    root=root,
                    loader_name=loader_name,
                    loader=loader,
                    schema=schema,
                )
            )
            seen.add(key)

    for folder in folders:
        # Existing exact schemas.
        if _has_files(folder, {"tmm_p.sas7bdat", "testdrug.sas7bdat", "demog.sas7bdat"}):
            add("A6181122", folder, "load_a6181122", load_a6181122, "pfizer_a6181122")

        if _has_files(folder, {"ls.sas7bdat", "ex.sas7bdat", "dm.sas7bdat"}):
            add(
                "EFC10262",
                folder,
                "load_efc10262",
                lambda root: load_efc10262(root, evaluator="INVESTIGATOR"),
                "sanofi_sdtm_efc10262",
            )

        if _has_files(folder, {"measur.sas7bdat", "dose.sas7bdat", "demog.sas7bdat"}):
            text = str(folder).lower()
            if "5505" in text:
                add("EFC5505", folder, "load_efc5505", load_efc5505, "sanofi_efc55xx")
            elif "4972" in text:
                add("EFC4972", folder, "load_efc4972", load_efc4972, "sanofi_efc55xx")

        # AstraZeneca Cediranib.
        if _has_files(folder, {"rdprcist.sas7bdat", "rpdosad.sas7bdat", "rpdem.sas7bdat"}):
            add(
                "CEDIRANIB_HORIZONIII",
                folder,
                "load_astrazeneca_cediranib",
                load_astrazeneca_cediranib,
                "astrazeneca_cediranib",
            )

        # Amgen legacy.
        if _has_files(folder, {"lesion.sas7bdat", "exposure.sas7bdat", "demo.sas7bdat"}):
            sid = _study_id_from_path(folder, "AMGEN_LEGACY")
            add(
                sid,
                folder,
                "load_amgen_legacy",
                lambda root, _sid=sid: load_amgen_legacy(root, study_id=_sid),
                "amgen_legacy",
            )

        # Amgen ADaM/PDS.
        if _has_files(folder, {"adls_pds2019.sas7bdat", "adsl_pds2019.sas7bdat"}):
            sid = _study_id_from_path(folder, "AMGEN_ADAM_PDS")
            add(
                sid,
                folder,
                "load_amgen_adam_pds",
                lambda root, _sid=sid: load_amgen_adam_pds(root, study_id=_sid),
                "amgen_adam_pds",
            )

    return candidates