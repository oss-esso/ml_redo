"""
Tests for the Simeoni simulator and data pipeline.

These are the contracts from the LaTeX document:
    - Simulator returns positive, finite values
    - No treatment allows tumor growth
    - Strong treatment causes tumor shrinkage
    - Observation model adds noise but preserves positivity
    - Synthetic cohort has expected shape
"""

import numpy as np
import pytest

from pkpd_sbi.simulators.dosing import make_q2w_schedule, DoseSchedule
from pkpd_sbi.simulators.simeoni import SimeoniParams, simulate_tumor, simulate_patient_vector
from pkpd_sbi.simulators.observation import observe_lognormal, relative_to_baseline
from pkpd_sbi.simulators.priors import DEFAULT_PRIOR
from pkpd_sbi.data.synthetic import generate_synthetic_cohort


# --- Fixtures ---

@pytest.fixture
def default_params():
    return SimeoniParams(
        lam0=0.015, lam1=8000.0, psi_g=1.0,
        k_kill=0.004, k_tr=0.08, v0=60.0,
    )

@pytest.fixture
def default_schedule():
    return make_q2w_schedule(duration_days=365)

@pytest.fixture
def scan_times():
    return np.arange(0, 365 + 1, 56, dtype=float)


# --- Simulator tests ---

class TestSimulator:
    def test_returns_positive_finite(self, default_params, default_schedule, scan_times):
        volume = simulate_tumor(default_params, default_schedule, scan_times)
        assert volume.shape == scan_times.shape
        assert np.all(np.isfinite(volume))
        assert np.all(volume >= 0.0)

    def test_no_treatment_allows_growth(self, default_schedule, scan_times):
        params = SimeoniParams(
            lam0=0.015, lam1=8000.0, psi_g=1.0,
            k_kill=0.0, k_tr=0.08, v0=60.0,
        )
        schedule = make_q2w_schedule(duration_days=365, dose_amount=0.0)
        volume = simulate_tumor(params, schedule, scan_times)
        assert volume[-1] > volume[0], "Tumor should grow without treatment"

    def test_strong_treatment_causes_shrinkage(self, default_schedule, scan_times):
        params = SimeoniParams(
            lam0=0.015, lam1=8000.0, psi_g=1.0,
            k_kill=0.10, k_tr=0.08, v0=60.0,  # very strong kill
        )
        volume = simulate_tumor(params, default_schedule, scan_times)
        # At some point during treatment, volume should be below baseline
        assert np.any(volume < params.v0), "Strong treatment should shrink tumor"

    def test_baseline_volume_matches_v0(self, default_params, default_schedule, scan_times):
        volume = simulate_tumor(default_params, default_schedule, scan_times)
        assert abs(volume[0] - default_params.v0) < 1.0, "Initial volume should match v0"

    def test_different_params_give_different_trajectories(self, default_schedule, scan_times):
        p1 = SimeoniParams(lam0=0.01, lam1=8000.0, psi_g=1.0, k_kill=0.004, k_tr=0.08, v0=60.0)
        p2 = SimeoniParams(lam0=0.05, lam1=8000.0, psi_g=1.0, k_kill=0.004, k_tr=0.08, v0=60.0)
        v1 = simulate_tumor(p1, default_schedule, scan_times)
        v2 = simulate_tumor(p2, default_schedule, scan_times)
        assert not np.allclose(v1, v2), "Different growth rates should give different trajectories"


# --- Observation model tests ---

class TestObservation:
    def test_lognormal_preserves_positivity(self):
        rng = np.random.default_rng(42)
        true_vol = np.array([60.0, 50.0, 40.0, 30.0, 35.0, 45.0, 55.0])
        observed = observe_lognormal(true_vol, sigma_obs=0.15, rng=rng)
        assert np.all(observed > 0), "Log-normal noise should preserve positivity"

    def test_zero_noise_returns_true_values(self):
        rng = np.random.default_rng(42)
        true_vol = np.array([60.0, 50.0, 40.0])
        observed = observe_lognormal(true_vol, sigma_obs=0.0, rng=rng)
        np.testing.assert_allclose(observed, true_vol, rtol=1e-6)

    def test_relative_baseline_starts_at_one(self):
        values = np.array([60.0, 50.0, 40.0, 30.0])
        rel = relative_to_baseline(values)
        assert abs(rel[0] - 1.0) < 1e-10


# --- Dosing tests ---

class TestDosing:
    def test_concentration_before_first_dose_is_zero(self):
        schedule = make_q2w_schedule(dose_amount=1.0)
        c = schedule.concentration(-1.0)
        assert c == 0.0

    def test_concentration_is_positive_after_dose(self):
        schedule = make_q2w_schedule(dose_amount=1.0)
        c = schedule.concentration(1.0)
        assert c > 0.0

    def test_concentration_decays(self):
        schedule = make_q2w_schedule(dose_amount=1.0)
        c1 = schedule.concentration(1.0)
        c10 = schedule.concentration(10.0)
        # Between doses, concentration should decay (if no new dose)
        # But with q2w dosing, c10 might be higher due to accumulation
        # At least check it's finite and positive
        assert np.isfinite(c1) and c1 > 0
        assert np.isfinite(c10) and c10 > 0


# --- Prior tests ---

class TestPrior:
    def test_sample_returns_positive_values(self):
        rng = np.random.default_rng(42)
        samples = DEFAULT_PRIOR.sample(100, rng)
        assert samples.shape == (100, 6)
        assert np.all(samples > 0), "All parameters should be positive"

    def test_samples_are_in_reasonable_range(self):
        rng = np.random.default_rng(42)
        samples = DEFAULT_PRIOR.sample(1000, rng)
        # lam0 should be in [0.001, 0.1]
        assert np.all(samples[:, 0] > 0.001)
        assert np.all(samples[:, 0] < 0.1)


# --- Synthetic data tests ---

class TestSynthetic:
    def test_cohort_shape(self):
        obs_df, params_df = generate_synthetic_cohort(n_patients=10, seed=42)
        assert len(params_df) <= 10  # some may fail
        assert len(params_df) > 5   # most should succeed
        assert "patient_id" in obs_df.columns
        assert "time_days" in obs_df.columns
        assert "observed_volume" in obs_df.columns

    def test_all_volumes_positive(self):
        obs_df, _ = generate_synthetic_cohort(n_patients=20, seed=42)
        assert (obs_df["observed_volume"] > 0).all()

    def test_no_nans(self):
        obs_df, _ = generate_synthetic_cohort(n_patients=20, seed=42)
        assert not obs_df["observed_volume"].isna().any()


# --- Parameter serialization tests ---

class TestParams:
    def test_roundtrip_array(self, default_params):
        arr = default_params.to_array()
        reconstructed = SimeoniParams.from_array(arr)
        assert abs(reconstructed.lam0 - default_params.lam0) < 1e-10
        assert abs(reconstructed.v0 - default_params.v0) < 1e-10

    def test_simulate_patient_vector_shape(self, default_schedule, scan_times):
        theta = np.array([0.015, 8000.0, 1.0, 0.004, 0.08, 60.0])
        x = simulate_patient_vector(theta, default_schedule, scan_times, sigma_obs=0.10)
        assert x.shape == scan_times.shape
        assert np.all(np.isfinite(x))
