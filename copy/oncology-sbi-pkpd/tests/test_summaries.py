"""
Tests for the clinical-data encoding functions in summaries.py.

All tests use in-memory DataFrames — no SAS files required.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from pkpd_sbi.inference.summaries import (
    LOG_SLD_CLIP,
    MAX_TIMEPOINTS,
    MAX_TIME_DAYS,
    encode_cohort,
    encode_patient_to_tensor,
    observation_tensor_for_npe,
    synthetic_obs_to_dataframe,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_patient_obs(
    time_days: list[float],
    rel_sld: list[float],
) -> pd.DataFrame:
    return pd.DataFrame({"time_days": time_days, "rel_sld": rel_sld})


def _make_cohort_obs(n_patients: int, n_obs_each: int) -> pd.DataFrame:
    rows = []
    for i in range(n_patients):
        for j in range(n_obs_each):
            rows.append({
                "patient_id": f"P{i:03d}",
                "time_days": float(j * 56),
                "rel_sld": max(0.01, 1.0 - j * 0.05 + i * 0.02),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. encode_patient_to_tensor — basic
# ---------------------------------------------------------------------------

def test_encode_patient_basic() -> None:
    obs = _make_patient_obs(
        time_days=[0, 56, 112, 168, 224],
        rel_sld=[1.0, 0.8, 0.7, 0.9, 1.1],
    )
    enc = encode_patient_to_tensor(obs)

    assert enc["times"].shape == (MAX_TIMEPOINTS,)
    assert enc["log_rel_sld"].shape == (MAX_TIMEPOINTS,)
    assert enc["mask"].shape == (MAX_TIMEPOINTS,)

    assert enc["mask"].sum().item() == 5
    assert enc["mask"][:5].all()
    assert not enc["mask"][5:].any()

    assert enc["times"][0].item() == pytest.approx(0.0)
    assert (enc["times"][enc["mask"]] >= 0).all()
    assert (enc["times"][enc["mask"]] <= 1.0).all()

    # rel_sld=0.8 → log(0.8) < 0
    assert enc["log_rel_sld"][1].item() < 0.0


# ---------------------------------------------------------------------------
# 2. encode_patient_to_tensor — truncation
# ---------------------------------------------------------------------------

def test_encode_patient_truncation() -> None:
    obs = _make_patient_obs(
        time_days=list(range(0, 30 * 14, 14)),  # 30 timepoints
        rel_sld=[1.0] * 30,
    )
    enc = encode_patient_to_tensor(obs)

    assert enc["times"].shape == (MAX_TIMEPOINTS,)
    assert enc["mask"].sum().item() == MAX_TIMEPOINTS  # capped at 24


# ---------------------------------------------------------------------------
# 3. encode_patient_to_tensor — clipping
# ---------------------------------------------------------------------------

def test_encode_patient_clipping() -> None:
    obs = _make_patient_obs(
        time_days=[0, 56, 112],
        rel_sld=[1.0, 100.0, 0.5],  # 100 → log ≈ 4.6 > clip_max=3.0
    )
    enc = encode_patient_to_tensor(obs)

    assert enc["log_rel_sld"][1].item() == pytest.approx(LOG_SLD_CLIP[1])
    assert enc["log_rel_sld"][2].item() < 0.0

    # Lower clip: rel_sld=0.001 → log ≈ -6.9 < clip_min=-3.0
    obs_low = _make_patient_obs(time_days=[0, 56], rel_sld=[1.0, 0.001])
    enc_low = encode_patient_to_tensor(obs_low)
    assert enc_low["log_rel_sld"][1].item() == pytest.approx(LOG_SLD_CLIP[0])


# ---------------------------------------------------------------------------
# 4. encode_cohort — shape
# ---------------------------------------------------------------------------

def test_encode_cohort_shape() -> None:
    obs = _make_cohort_obs(n_patients=3, n_obs_each=5)
    batch = encode_cohort(obs)

    assert batch["times"].shape == (3, MAX_TIMEPOINTS)
    assert batch["log_rel_sld"].shape == (3, MAX_TIMEPOINTS)
    assert batch["mask"].shape == (3, MAX_TIMEPOINTS)
    assert len(batch["patient_ids"]) == 3
    assert batch["n_obs"].shape == (3,)
    assert (batch["n_obs"] == 5).all()


# ---------------------------------------------------------------------------
# 4b. encode_cohort — skips patients with < min_obs valid observations
# ---------------------------------------------------------------------------

def test_encode_cohort_skips_sparse() -> None:
    obs = _make_cohort_obs(n_patients=4, n_obs_each=5)
    # Nullify rel_sld for patient P000 to leave only 1 valid obs
    obs.loc[(obs["patient_id"] == "P000") & (obs["time_days"] > 0), "rel_sld"] = np.nan
    batch = encode_cohort(obs, min_obs=2)
    assert "P000" not in batch["patient_ids"]
    assert len(batch["patient_ids"]) == 3


# ---------------------------------------------------------------------------
# 5. synthetic_obs_to_dataframe → encode_patient_to_tensor roundtrip
# ---------------------------------------------------------------------------

def test_synthetic_to_real_roundtrip() -> None:
    obs_times = np.arange(0.0, 7 * 56.0, 56.0)  # 7 timepoints
    # Simulate a response: tumor shrinks then grows
    log_vols = np.log([100, 90, 80, 75, 80, 90, 95], dtype=float)

    df = synthetic_obs_to_dataframe(log_vols, obs_times)

    assert "time_days" in df.columns
    assert "rel_sld" in df.columns
    assert "baseline_sld_mm" in df.columns
    assert len(df) == 7
    assert df["rel_sld"].iloc[0] == pytest.approx(1.0)

    enc = encode_patient_to_tensor(df)
    assert enc["mask"].sum().item() == 7
    assert not torch.isnan(enc["log_rel_sld"]).any()


# ---------------------------------------------------------------------------
# 6. observation_tensor_for_npe — shape and modes
# ---------------------------------------------------------------------------

def test_observation_tensor_for_npe_flat_shape() -> None:
    obs = _make_patient_obs(
        time_days=[0, 56, 112, 168, 224],
        rel_sld=[1.0, 0.8, 0.7, 0.9, 1.1],
    )
    x = observation_tensor_for_npe(obs, mode="flat")
    assert x.shape == (3 * MAX_TIMEPOINTS,)
    assert x.dtype == torch.float32


def test_observation_tensor_for_npe_interleaved_shape() -> None:
    obs = _make_patient_obs(
        time_days=[0, 56, 112],
        rel_sld=[1.0, 0.9, 0.8],
    )
    x = observation_tensor_for_npe(obs, mode="interleaved")
    assert x.shape == (2 * MAX_TIMEPOINTS,)


def test_observation_tensor_for_npe_unknown_mode() -> None:
    obs = _make_patient_obs(time_days=[0], rel_sld=[1.0])
    with pytest.raises(ValueError, match="Unknown mode"):
        observation_tensor_for_npe(obs, mode="bad_mode")


# ---------------------------------------------------------------------------
# 7. padding correctness — padded slots are exactly zero
# ---------------------------------------------------------------------------

def test_padded_slots_are_zero() -> None:
    obs = _make_patient_obs(time_days=[0, 56], rel_sld=[1.0, 0.9])
    enc = encode_patient_to_tensor(obs)

    assert (enc["times"][2:] == 0.0).all()
    assert (enc["log_rel_sld"][2:] == 0.0).all()
    assert not enc["mask"][2:].any()
