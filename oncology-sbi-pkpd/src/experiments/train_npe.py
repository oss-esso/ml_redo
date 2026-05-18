from sbi.inference import SNPE
from sbi.utils import BoxUniform
import torch
from .src.pkpd.simeoni import observe_tumor_volume

def simulator(theta):
    """
    Convert parameter vector theta into a simulated observation vector .
    Includes :
    - patient-specific PKPD parameters
    - dose schedule
    - observation times
    - log-normal observation noise
    - missingness mask if required
    """
    x = simulate_patient_trajectory(theta)
    return torch.as_tensor(x, dtype=torch.float32)

def simulate_patient_trajectory(theta):
    """
    Simulate a patient trajectory given a parameter vector theta.
    This includes:
    - simulating the tumor growth and drug response using the Simeoni model
    - applying the dose schedule
    - adding observation noise
    - applying missingness if required
    """
    # Unpack parameters from theta
    lam0, lam1, kel, k_kill, k_tr, sigma_obs = theta

    # Define dose schedule (for simplicity, we use a fixed schedule here)
    dose_times = torch.tensor([0.0, 7.0, 14.0])  # Doses at day 0, 7, and 14
    dose_amounts = torch.tensor([100.0, 100.0, 100.0])  # Fixed dose amount

    # Simulate tumor trajectory using the Simeoni model
    t_eval = torch.linspace(0.0, 30.0, steps=31)  # Observation times from day 0 to 30
    states = simulate_simeoni(t_eval, lam0, lam1, kel, k_kill, k_tr, dose_times, dose_amounts)

    # Add observation noise
    key = torch.Generator().manual_seed(42)  # For reproducibility
    observed_volumes = observe_tumor_volume(states, sigma_obs, key)

    return observed_volumes

def simulate_simeoni(t_eval, lam0, lam1, kel, k_kill, k_tr, dose_times, dose_amounts):
    """
    Simulate the Simeoni model given parameters and dose schedule.
    This function should implement the ODE solver for the Simeoni model.
    For simplicity, we can use a placeholder here.
    """
    # Placeholder for actual ODE simulation
    # In practice, you would use an ODE solver like scipy.integrate.solve_ivp
    # to simulate the tumor trajectory based on the Simeoni model equations.
    return torch.zeros((len(t_eval), 4))  # Placeholder for state trajectories





prior = BoxUniform(
    low=torch.tensor([-6.0, -6.0, -8.0, -6.0, -6.0, -4.0]),
    high=torch.tensor([2.0, 4.0, 1.0, 2.0, 2.0, 1.0])
)

inference = SNPE(prior=prior)

theta = prior.sample((50_000, ))
x = torch.stack([simulator[th] for th in theta])
inference = inference.append_simulations(theta, x)

density_estimator = inference.train()
posterior = inference.build_posterior(density_estimator)