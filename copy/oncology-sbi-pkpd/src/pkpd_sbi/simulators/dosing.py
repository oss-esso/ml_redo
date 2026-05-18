"""
Dosing schedule for oncology PKPD simulation.

LaTeX contract:
    C(t) = sum of dose_amounts * exp(-kel * (t - dose_times)) for active doses

Clinical contract:
    FOLFOX4 is administered every 2 weeks (q2w = 14 days).
    Tumor scans occur every 8 weeks (q8w = 56 days).
"""

from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp



class DoseSchedule():
    """Immutable dosing schedule with simple one-compartment exposure decay."""
    def __init__(self, dose_times: jnp.ndarray, dose_amounts: jnp.ndarray, kel: float):
        assert len(dose_times) == len(dose_amounts), "dose_times and dose_amounts must have the same length"
        self.dose_times = dose_times
        self.dose_amounts = dose_amounts
        self.kel = kel

    def concentration(self, t: float) -> float:
        """Drug concentration at time t via superposition of bolus doses."""
        dt = t - self.dose_times
        active = dt >= 0.0
        return jnp.sum(jnp.where(active, self.dose_amounts * jnp.exp(-self.kel * dt), 0.0))


def make_q2w_schedule(
    duration_days: float = 365.0,
    dose_amount: float = 1.0,
    kel_per_day: float = 0.35,
) -> DoseSchedule:
    """
    Standard q2w (every 2 weeks) dosing schedule.

    Maps to: "FOLFOX4 repeated every two weeks" from EFC4972 protocol.
    """
    dose_times = jnp.arange(0.0, duration_days + 1e-9, 14.0)
    dose_amounts = jnp.full_like(dose_times, dose_amount, dtype=float)
    return DoseSchedule(dose_times=dose_times, dose_amounts=dose_amounts, kel=kel_per_day)


# Clinical constants
DEFAULT_SCAN_INTERVAL_DAYS = 56   # "Tumor evaluation every 8 weeks"
DEFAULT_TREATMENT_DURATION_DAYS = 365.0
DEFAULT_SCAN_TIMES_DAYS = jnp.arange(0, DEFAULT_TREATMENT_DURATION_DAYS + 1, DEFAULT_SCAN_INTERVAL_DAYS)
