import jax
import jax.numpy as jnp

def growth_term(x1, lam0, lam1, psi_growth:float = 1.0):
    """
    Simeooni growth rate. For early stage exponential, for large tumors the burden slows down"""

    den = (1.0 + (lam0 / lam1 * jnp.maximum(x1, 1e-8)) ** psi_growth) ** (1.0 / psi_growth)
    return lam0 * x1 / den

def concentration_from_schedule(t, dose_times, dose_amounts, kel):
    """
    Simle one-compartment 
    exposure model. For prod we need piecewise integrals across dose events"""
    dt = t - dose_times
    active = dt >= 0.0
    return jnp.sum(jnp.where(active, dose_amounts * jnp.exp(-kel * dt), 0.0))

def simeoni_rhs(t, x, params, schedule):
    x1, x2, x3, x4 = x
    C_t = concentration_from_schedule(
        t,
        schedule["dose_times"],
        schedule["dose_amounts"],
        params["kel"]
    )

    grow = growth_term(x1, params["lam0"], params["lam1"], params["psi_growth"])
    kill = params["k_kill"] * C_t * x1
    ktr = params["k_tr"]

    dx1 = grow - kill
    dx2 = kill - ktr * x2
    dx3 = ktr * (x2 - x3)
    dx4 = ktr * (x4 - x4)

    return jnp.array([dx1, dx2, dx3, dx4])


def observe_tumor_volume(states, sigma_obs, key):
    """Lognormal observation model"""
    true_volume = jnp.sum(states, axis=-1)
    eps = sigma_obs * jax.random.normal(key, shape=true_volume.shape)
    return jnp.exp(jnp.log(jnp.maximum(true_volume, 1e-8)) + eps)
