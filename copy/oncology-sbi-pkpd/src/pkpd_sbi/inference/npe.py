"""
Neural Posterior Estimation wrapper.

LaTeX contract:
    q_phi(theta | x) ≈ p(theta | x)
    L(phi) = -1/M sum_m log q_phi(theta^(m) | x^(m))

Implementation:
    Uses the `sbi` package SNPE_C (APT) algorithm.
    Phase 1: one-shot NPE from broad prior.
    Phase 2: sequential NPE with proposal correction.

The key insight: after training, inference for any new patient
is a single forward pass. This is amortization.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import pickle


def build_npe(
    theta: np.ndarray,
    x: np.ndarray,
    prior,
    n_components: int = 10,
    hidden_features: int = 64,
    num_transforms: int = 5,
    training_batch_size: int = 256,
    learning_rate: float = 5e-4,
    max_epochs: int = 200,
    stop_after_epochs: int = 20,
    device: str = "cpu",
) -> Tuple:
    """
    Train an amortized NPE model.

    Args:
        theta: (N, d_theta) parameter samples on log scale.
        x: (N, d_x) observation vectors (log tumor volumes).
        prior: sbi-compatible prior (BoxUniform on log scale).
        n_components: Number of mixture components in the flow.
        hidden_features: Hidden layer width.
        num_transforms: Number of flow transforms.
        training_batch_size: Batch size for training.
        learning_rate: Adam learning rate.
        max_epochs: Maximum training epochs.
        stop_after_epochs: Early stopping patience.
        device: "cpu" or "cuda".

    Returns:
        (posterior, inference, summary): Trained posterior object,
            inference object (for diagnostics), training summary dict.
    """
    from sbi.inference import SNPE
    from sbi.neural_nets import posterior_nn

    theta_tensor = torch.tensor(theta, dtype=torch.float32)
    x_tensor = torch.tensor(x, dtype=torch.float32)

    # Build the density estimator (neural spline flow)
    density_estimator = posterior_nn(
        model="nsf",  # Neural Spline Flow
        hidden_features=hidden_features,
        num_transforms=num_transforms,
    )

    inference = SNPE(
        prior=prior,
        density_estimator=density_estimator,
        device=device,
    )

    inference = inference.append_simulations(theta_tensor, x_tensor)

    density_estimator = inference.train(
        training_batch_size=training_batch_size,
        learning_rate=learning_rate,
        max_num_epochs=max_epochs,
        stop_after_epochs=stop_after_epochs,
        show_train_summary=True,
    )

    posterior = inference.build_posterior(density_estimator)

    # Extract training summary — key names changed between sbi versions:
    #   sbi <0.23: best_validation_log_prob (list), training_log_probs (list)
    #   sbi >=0.23: best_validation_loss (scalar), epochs_trained (scalar)
    s = inference.summary

    def _get(candidates, default=float("nan")):
        for k in candidates:
            if k in s:
                v = s[k]
                return float(v[-1]) if hasattr(v, "__len__") else float(v)
        return default

    summary = {
        "best_validation_log_prob": _get(
            ["best_validation_log_prob", "best_validation_log_probs",
             "best_validation_loss"]
        ),
        "epochs_trained": int(_get(
            ["epochs_trained", "training_log_probs"]
        )),
        "n_training_samples": len(theta),
        "n_params": theta.shape[1],
        "n_obs_features": x.shape[1],
    }

    return posterior, inference, summary


def sample_posterior(
    posterior,
    x_obs: np.ndarray,
    n_samples: int = 10_000,
    reject_outside_prior: bool = True,
) -> np.ndarray:
    """
    Sample from the amortized posterior for one patient.

    Args:
        posterior: Trained sbi posterior object.
        x_obs: (d_x,) observed log tumor volumes for this patient.
        n_samples: Number of posterior samples.
        reject_outside_prior: If True (default), reject samples outside the prior
            support. Set to False when the clinical population may have parameters
            outside the training prior — avoids infinite rejection loops on real data.

    Returns:
        samples: (n_samples, d_theta) on log scale.
    """
    x_tensor = torch.tensor(x_obs, dtype=torch.float32)
    samples = posterior.sample(
        (n_samples,),
        x=x_tensor,
        reject_outside_prior=reject_outside_prior,
    )
    return samples.detach().numpy()


def posterior_predictive_sample(
    posterior,
    x_obs: np.ndarray,
    simulator_fn,
    n_samples: int = 200,
) -> list[np.ndarray]:
    """
    Generate posterior predictive trajectories.

    For each posterior sample theta, simulate a new trajectory.
    This is for posterior predictive checks.

    Args:
        posterior: Trained posterior.
        x_obs: Observed data for conditioning.
        simulator_fn: Callable(theta) -> trajectory.
        n_samples: Number of predictive trajectories.

    Returns:
        List of simulated trajectories.
    """
    log_theta_samples = sample_posterior(posterior, x_obs, n_samples)
    trajectories = []

    for log_theta in log_theta_samples:
        theta = np.exp(log_theta)
        try:
            traj = simulator_fn(theta)
            if not np.any(np.isnan(traj)):
                trajectories.append(traj)
        except RuntimeError:
            continue

    return trajectories


def save_posterior(posterior, path: Path) -> None:
    """Save trained posterior to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(posterior, f)


def load_posterior(path: Path):
    """Load trained posterior from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)
