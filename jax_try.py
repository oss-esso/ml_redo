from tracemalloc import start

import jax
import jax.numpy as jnp
import numpy as np
import time
import matplotlib.pyplot as plt


#def row_softmax_loop(array):
#    softmax = np.zeros_like(array, dtype=float)
#    for i in range(array.shape[0]):
#        exp_row = np.exp(array[i])
#        softmax[i] = exp_row / np.sum(exp_row)
#    return softmax
#
#def row_softmax_jax(array):
#    softmax  = lambda x: jnp.exp(x) / jnp.sum(jnp.exp(x))
#
#    rwsoftmax = jax.vmap(softmax)(array)
#
#    return rwsoftmax
#
#def softmax_right(array):
#    return jax.nn.softmax(array, axis=1)
#
#
#if __name__ == "__main__":
#    
#    array = np.vstack([np.ones((4,)), 2*np.ones((4,)), 3*np.ones((4,))])
#    print("Input array:")
#    print(array)
#    print("\nSoftmax using JAX's built-in function:")
#    print(softmax_right(array))
#    start_time = time.time()
#    softmax_loop = row_softmax_loop(array)
#    print("\nSoftmax using loop:")
#    print(softmax_loop)
#    print(f'\n loop solution correct: {np.allclose(softmax_loop, softmax_right(array))}')
#    start_time_jax = time.time()
#    softmax_jax = row_softmax_jax(array)
#    print("\nSoftmax using JAX:")
#    print(softmax_jax)
#    print(f'\n JAX solution correct: {np.allclose(softmax_jax, softmax_right(array))}')
#    end_time_jax = time.time()
#    print(f"\nTime taken by loop: {start_time_jax - start_time} seconds")
#    print(f"\nTime taken by JAX: {end_time_jax - start_time_jax} seconds")


from diffrax import diffeqsolve as dsolve
from diffrax import ODETerm, SaveAt, Tsit5

#def f(t, y, args):
#    return -0.5*y
#
#term = ODETerm(f)
#solver = Tsit5()
#y0 = 1.0
#t0 = 0.0
#t1 = 10.0
#dt0 = 0.1
#
#saveat = SaveAt(ts=jnp.linspace(t0, t1, int((t1 - t0) / dt0) + 1))
#sol = dsolve(term, solver, t0, t1, dt0, y0, saveat=saveat)
#print(sol.ts)
#
#error = jnp.abs(sol.ys - jnp.exp(-0.5*sol.ts))
#print("Max error:", jnp.max(error))
#
#plt.plot(sol.ts, sol.ys, label='Diffrax Solution')
#plt.plot(sol.ts, jnp.exp(-0.5*sol.ts), label='Analytical Solution', linestyle='dashed')
#plt.xlabel('Time')
#plt.ylabel('y(t)')
#plt.title('ODE Solution using Diffrax')
#plt.legend()
#plt.show()
#
#



def exp_decay_sum(n_decays, t):
    decay_rates = jnp.arange(1, n_decays + 1)
    decay_sum = jnp.sum(jnp.exp(-decay_rates[:, None]*t), axis=0)
    return decay_sum


if __name__ == "__main__":
    n_decays = 1000
    t = jnp.linspace(0, 10, 100)
    start_time = time.time()
    result = exp_decay_sum(n_decays, t)
    end_time = time.time()
    print(f"Result shape: {result.shape}")
    print(f"Time taken: {end_time - start_time} seconds")