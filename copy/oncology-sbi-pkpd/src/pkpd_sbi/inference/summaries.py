"""
Summary statistics / trajectory encoders for SBI.

LaTeX contract:
    h_xi(x_{1:T}) -> z in R^d
    Maps variable-length, irregular-time tumor volume series to
    a fixed-dimensional embedding that conditions the normalizing flow.

Implementation strategy (from doc):
    1. Fixed-grid interpolation + mask (baseline, used in Phase 1)
    2. Set encoder with attention (Phase 2)
    3. Neural CDE (optional future extension)

Phase 1 uses the simplest approach: since our synthetic data has
regular scan times (q8w), we can use the raw log-volume vector directly.
For real data with irregular timing, we add time features.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional

# ---------------------------------------------------------------------------
# Encoding constants — derived from real-data audit (see data_audit.md)
# ---------------------------------------------------------------------------
MAX_TIMEPOINTS: int = 24       # max observed across all 4 datasets (EFC10262)
MAX_TIME_DAYS: float = 730.0   # normalization constant; 2-year horizon
LOG_SLD_CLIP: tuple[float, float] = (-3.0, 3.0)  # clip log_rel_sld outliers


class FixedGridEmbedding(nn.Module):
    """
    Simplest summary network: raw log-volumes on a fixed grid.

    Input: (batch, n_timepoints) log tumor volumes
    Output: (batch, d_out) embedding

    This is the Phase 1 encoder. It works when all patients share
    the same observation schedule (synthetic data, or after interpolation).
    """

    def __init__(self, n_timepoints: int, d_hidden: int = 64, d_out: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_timepoints, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TimeAwareSetEncoder(nn.Module):
    """
    Set encoder for irregular time series (Phase 2).

    Input: (batch, max_T, 2) where features are [time, log_volume]
    Output: (batch, d_out) embedding

    Uses self-attention over timepoints, permutation-invariant.
    Handles variable-length sequences via masking.
    """

    def __init__(self, d_embed: int = 64, n_heads: int = 4, d_out: int = 32):
        super().__init__()
        # Encode each (time, log_volume) pair
        self.point_encoder = nn.Sequential(
            nn.Linear(2, d_embed),
            nn.GELU(),
            nn.Linear(d_embed, d_embed),
            nn.GELU(),
        )
        # Self-attention over timepoints
        self.attn = nn.MultiheadAttention(
            d_embed, num_heads=n_heads, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_embed)
        # Aggregate and project
        self.output = nn.Sequential(
            nn.Linear(d_embed, d_embed),
            nn.GELU(),
            nn.Linear(d_embed, d_out),
        )

    def forward(
        self,
        times: torch.Tensor,
        log_volumes: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            times: (B, T) normalized observation times
            log_volumes: (B, T) log tumor volumes
            mask: (B, T) boolean, True = valid observation

        Returns:
            (B, d_out) embedding
        """
        x = torch.stack([times, log_volumes], dim=-1)  # (B, T, 2)
        h = self.point_encoder(x)                       # (B, T, d_embed)

        # Self-attention with optional key_padding_mask
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask  # True = ignore this position

        h_attn, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask)
        h = self.norm(h + h_attn)  # residual connection

        # Mean pooling over valid timepoints
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            h = (h * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            h = h.mean(dim=1)

        return self.output(h)  # (B, d_out)


def prepare_sbi_observation(
    log_volumes: np.ndarray,
    observation_times: Optional[np.ndarray] = None,
    normalize_time: bool = True,
    max_time: float = 365.0,
    patient_obs: Optional[pd.DataFrame] = None,
) -> torch.Tensor:
    """
    Convert a single patient's observations to a tensor for SBI posterior sampling.

    If patient_obs (canonical clinical DataFrame) is provided, delegates to
    observation_tensor_for_npe. Otherwise falls back to raw log_volumes tensor
    (Phase 1 fixed-grid path).

    Args:
        log_volumes: (n_obs,) log tumor volumes — used when patient_obs is None
        observation_times: (n_obs,) times in days — unused in flat mode
        normalize_time: kept for API compat
        max_time: kept for API compat
        patient_obs: canonical observations DataFrame for one patient

    Returns:
        Tensor suitable for posterior.sample(x=...).
    """
    if patient_obs is not None:
        return observation_tensor_for_npe(patient_obs, mode="flat")
    return torch.tensor(log_volumes, dtype=torch.float32)


# Default fixed grid from dosing.py (0 to 336 days, every 56 days = 7 timepoints)
_DEFAULT_GRID_DAYS = np.arange(0.0, 365.0, 56.0)


def prepare_clinical_observation(
    patient_obs: pd.DataFrame,
    grid_times: Optional[np.ndarray] = None,
    tolerance_days: float = 28.0,
    fill_missing: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map real patient observations to the NPE fixed-scan-time grid.

    Uses log-relative SLD (log(SLD_t / SLD_baseline)) which is dimensionless
    and directly comparable to the synthetic relative-trajectory training data.

    Args:
        patient_obs: Canonical observations DataFrame for one patient.
            Must have columns: time_days, rel_sld.
        grid_times: Fixed grid times in days. Defaults to synthetic training grid
            [0, 56, 112, 168, 224, 280, 336].
        tolerance_days: Max allowed distance from grid point to real scan.
            Scans outside this window are marked missing.
        fill_missing: Value to fill NaN entries (default 0.0 = no change).

    Returns:
        x: (n_grid,) log-relative SLD interpolated to grid (NaN filled).
        mask: (n_grid,) bool True where real scan was found within tolerance.
    """
    if grid_times is None:
        grid_times = _DEFAULT_GRID_DAYS

    obs_times = patient_obs["time_days"].values.astype(float)
    # rel_sld = SLD_t / SLD_baseline; clip to avoid log(0)
    log_rel = np.log(np.maximum(patient_obs["rel_sld"].values.astype(float), 1e-8))

    x = np.full(len(grid_times), np.nan)
    mask = np.zeros(len(grid_times), dtype=bool)

    for i, t_grid in enumerate(grid_times):
        diffs = np.abs(obs_times - t_grid)
        nearest_idx = int(np.argmin(diffs))
        if diffs[nearest_idx] <= tolerance_days:
            x[i] = log_rel[nearest_idx]
            mask[i] = True

    # Fill masked entries
    nan_pos = ~mask
    x[nan_pos] = fill_missing

    return x, mask


# ---------------------------------------------------------------------------
# Real-data encoding: canonical DataFrame → tensors
# ---------------------------------------------------------------------------

def encode_patient_to_tensor(
    patient_obs: pd.DataFrame,
    max_timepoints: int = MAX_TIMEPOINTS,
    max_time_days: float = MAX_TIME_DAYS,
) -> dict[str, torch.Tensor]:
    """
    Encode one patient's canonical observations into padded tensors for NPE.

    Clinical contract:
        - rel_sld == 1.0 at baseline (first observation)
        - time_days == 0 at first treatment day
        - SLD = 0 (complete response) is dropped — log(0) undefined

    Args:
        patient_obs: rows for one patient, columns include time_days, rel_sld
        max_timepoints: pad / truncate to this length
        max_time_days: time normalization constant

    Returns:
        dict with:
            "times"       (max_timepoints,) float32  — normalized time, 0-padded
            "log_rel_sld" (max_timepoints,) float32  — log(rel_sld) clipped, 0-padded
            "mask"        (max_timepoints,) bool      — True = real obs
            "n_obs"       scalar int
    """
    df = patient_obs.copy()
    df = df[df["rel_sld"].notna() & (df["rel_sld"] > 0)]
    df = df.sort_values("time_days").iloc[:max_timepoints]

    n = len(df)
    times_arr = (df["time_days"].values.astype(np.float32) / max_time_days)
    log_rel_arr = np.log(df["rel_sld"].values.astype(np.float32))
    log_rel_arr = np.clip(log_rel_arr, LOG_SLD_CLIP[0], LOG_SLD_CLIP[1])

    times_padded = np.zeros(max_timepoints, dtype=np.float32)
    log_padded = np.zeros(max_timepoints, dtype=np.float32)
    mask = np.zeros(max_timepoints, dtype=bool)

    times_padded[:n] = times_arr
    log_padded[:n] = log_rel_arr
    mask[:n] = True

    return {
        "times":       torch.tensor(times_padded, dtype=torch.float32),
        "log_rel_sld": torch.tensor(log_padded,   dtype=torch.float32),
        "mask":        torch.tensor(mask,          dtype=torch.bool),
        "n_obs":       torch.tensor(n,             dtype=torch.int32),
    }


def encode_cohort(
    observations: pd.DataFrame,
    max_timepoints: int = MAX_TIMEPOINTS,
    max_time_days: float = MAX_TIME_DAYS,
    min_obs: int = 2,
) -> dict[str, torch.Tensor | list[str]]:
    """
    Encode a full cohort of patients into batched tensors.

    Patients with fewer than min_obs valid observations are skipped.

    Args:
        observations: canonical observations DataFrame (all patients)
        max_timepoints: pad / truncate length
        max_time_days: time normalization constant
        min_obs: minimum valid observations required to include patient

    Returns:
        dict with:
            "times"       (N, max_timepoints) float32
            "log_rel_sld" (N, max_timepoints) float32
            "mask"        (N, max_timepoints) bool
            "patient_ids" list[str] length N
            "n_obs"       (N,) int32
    """
    times_list: list[torch.Tensor] = []
    log_list:   list[torch.Tensor] = []
    mask_list:  list[torch.Tensor] = []
    nobs_list:  list[torch.Tensor] = []
    pids:       list[str] = []

    for pid, grp in observations.groupby("patient_id"):
        valid = grp[grp["rel_sld"].notna() & (grp["rel_sld"] > 0)]
        if len(valid) < min_obs:
            continue
        enc = encode_patient_to_tensor(grp, max_timepoints, max_time_days)
        times_list.append(enc["times"])
        log_list.append(enc["log_rel_sld"])
        mask_list.append(enc["mask"])
        nobs_list.append(enc["n_obs"])
        pids.append(str(pid))

    return {
        "times":       torch.stack(times_list),
        "log_rel_sld": torch.stack(log_list),
        "mask":        torch.stack(mask_list),
        "patient_ids": pids,
        "n_obs":       torch.stack(nobs_list),
    }


def observation_tensor_for_npe(
    patient_obs: pd.DataFrame,
    mode: str = "flat",
    max_timepoints: int = MAX_TIMEPOINTS,
    max_time_days: float = MAX_TIME_DAYS,
) -> torch.Tensor:
    """
    Convert one patient's observations into the 1D tensor sbi expects.

    mode="flat":        [times_norm | log_rel_sld | mask.float()]
                        shape (3 * max_timepoints,)
    mode="interleaved": [t0, y0, t1, y1, ...]
                        shape (2 * max_timepoints,)

    Args:
        patient_obs: canonical observations for one patient
        mode: "flat" (default) or "interleaved"

    Returns:
        1D float32 tensor
    """
    enc = encode_patient_to_tensor(patient_obs, max_timepoints, max_time_days)
    if mode == "flat":
        return torch.cat([
            enc["times"],
            enc["log_rel_sld"],
            enc["mask"].float(),
        ])
    if mode == "interleaved":
        interleaved = torch.zeros(2 * max_timepoints, dtype=torch.float32)
        interleaved[0::2] = enc["times"]
        interleaved[1::2] = enc["log_rel_sld"]
        return interleaved
    raise ValueError(f"Unknown mode '{mode}'. Use 'flat' or 'interleaved'.")


def synthetic_obs_to_dataframe(
    log_volumes: np.ndarray,
    observation_times: np.ndarray,
) -> pd.DataFrame:
    """
    Convert synthetic simulator output to canonical clinical DataFrame format.

    This is the bridge that lets the same encode_patient_to_tensor function
    work on both synthetic training data and real clinical data.

    Args:
        log_volumes: (n_obs,) log tumor volumes from simulate_patient_vector
        observation_times: (n_obs,) scan times in days

    Returns:
        DataFrame with columns: time_days, sld_mm, rel_sld, baseline_sld_mm
    """
    sld = np.exp(np.asarray(log_volumes, dtype=np.float64))
    baseline = float(sld[0]) if len(sld) > 0 else 1.0
    baseline_safe = max(baseline, 1e-8)
    return pd.DataFrame({
        "time_days":        np.asarray(observation_times, dtype=np.float64),
        "sld_mm":           sld,
        "rel_sld":          sld / baseline_safe,
        "baseline_sld_mm":  np.full(len(sld), baseline_safe),
    })
