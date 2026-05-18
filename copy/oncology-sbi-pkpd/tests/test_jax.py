import sched
from unittest.mock import DEFAULT
import jax
from jax import numpy as jnp
from jax import random
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5



from pkpd_sbi.simulators.simeoni import SimeoniParams, simulate_tumor_jax
from pkpd_sbi.simulators.dosing import make_q2w_schedule
from pkpd_sbi.simulators.priors import DEFAULT_PRIOR


schedule = make_q2w_schedule()
obs_times = jnp.arange(0, 365.0 + 1, 56.0)  # Every 8 weeks for 1 year

rng = random.PRNGKey(0)



theta = DEFAULT_PRIOR.sample(100, rng)
for i in range(100):
    vol_numpy = simulate_tumor_jax(SimeoniParams.from_array(theta[i]), schedule, obs_times)
    vol_jax = simulate_tumor_jax(SimeoniParams.from_array(theta[i]), schedule, obs_times)
    assert jnp.allclose(vol_numpy, vol_jax, rtol=1e-4), f"Mismatch at patient {i}"



import time
import numpy as np



schedule = make_q2w_schedule()
obs_times = jnp.arange(0, 365.0 + 1, 56.0)

PARAM_NAMES = ["lam0", "lam1", "psi_g", "k_kill", "k_tr", "v0"]

DEFAULT_PRIOR = SimeoniParams(
    lam0=0.01,
    lam1=5000.0,
    psi_g=1.0,
    k_kill=0.01,
    k_tr=0.05,
    v0=100.0,
)

theta_batch = jnp.asarray(DEFAULT_PRIOR.sample(1000), dtype=jnp.float32)
theta_single = theta_batch[0]


def simulate_tumor_jax_theta(theta):
    params = SimeoniParams(
        lam0=theta[0],
        lam1=theta[1],
        psi_g=theta[2],
        k_kill=theta[3],
        k_tr=theta[4],
        v0=theta[5],
    )
    return simulate_tumor_jax(params, schedule, obs_times)


# E5a: JAX vmap timing
batched_jax = jax.jit(jax.vmap(simulate_tumor_jax_theta))

# warm-up compile
_ = batched_jax(theta_batch).block_until_ready()

t0 = time.perf_counter()
vols_jax = batched_jax(theta_batch).block_until_ready()
t_jax = time.perf_counter() - t0


# E5b: Python loop timing
theta_np = np.asarray(theta_batch)

t0 = time.perf_counter()
vols_loop = []
for i in range(theta_np.shape[0]):
    params = SimeoniParams.from_array(theta_np[i])
    vols_loop.append(np.asarray(simulate_tumor_jax(params, schedule, obs_times)))
vols_loop = np.stack(vols_loop)
t_loop = time.perf_counter() - t0

speedup = t_loop / t_jax

print(f"JAX batched time:   {t_jax:.4f} s")
print(f"Python loop time:   {t_loop:.4f} s")
print(f"Speedup:            {speedup:.2f}x")


# E6: Jacobian / sensitivity
jac = jax.jacobian(simulate_tumor_jax_theta)(theta_single)
# shape should be (n_times, 6), i.e. (7, 6)

last_time_sens = jnp.abs(jac[-1])
max_idx = int(jnp.argmax(last_time_sens))

print("Jacobian shape:", jac.shape)
print("Last-time sensitivities:")
for name, value in zip(PARAM_NAMES, last_time_sens):
    print(f"  {name}: {float(value):.6e}")

print(f"Largest sensitivity at final timepoint: {PARAM_NAMES[max_idx]}")