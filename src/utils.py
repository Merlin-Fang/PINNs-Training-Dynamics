import os
import jax
import scipy.io
import jax.numpy as jnp

from jax.tree_util import tree_map
from flax.training import checkpoints

def load_dataset(data_dir: str):
    data_dir = os.path.join("/scratch/merlinf/repos/PINNs-Training-Dynamics/pdes", data_dir)
    data = scipy.io.loadmat(data_dir)
    u_ref = data["usol"]
    t = data["t"].flatten()
    x = data["x"].flatten()
    return u_ref, t, x

def save_checkpoint(state, step, ckpt_dir, keep = 100):
    if os.path.exists(ckpt_dir) == False:
        os.makedirs(ckpt_dir) 
    
    state = jax.device_get(tree_map(lambda x: x[0], state))
    step = int(jax.device_get(step))

    checkpoints.save_checkpoint(ckpt_dir, state, step, keep=keep)

def load_checkpoint(state, ckpt_dir, step = None):
    return checkpoints.restore_checkpoint(ckpt_dir, state, step=step)

############################
## below is for corr net ###
############################

def interp2d_grid(
    t_grid: jnp.ndarray,
    x_grid: jnp.ndarray,
    Z: jnp.ndarray,
    t_query: jnp.ndarray,
    x_query: jnp.ndarray,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """JAX-compatible bilinear interpolation on a (t,x) grid.

    Args:
        t_grid: (Nt,) monotonic increasing 1D grid.
        x_grid: (Nx,) monotonic increasing 1D grid.
        Z:      (Nt, Nx) values defined on the grid.
        t_query:(B,) query t values.
        x_query:(B,) query x values.
        eps:    small constant to avoid divide-by-zero.

    Returns:
        Zq: (B,) interpolated values.

    Notes:
        - Out-of-range queries are clipped to the grid bounds.
        - Uses only JAX ops (jit/pmap safe).
    """
    # Ensure rank-1 queries
    t_query = jnp.asarray(t_query).reshape(-1)
    x_query = jnp.asarray(x_query).reshape(-1)

    t_grid = jnp.asarray(t_grid)
    x_grid = jnp.asarray(x_grid)
    Z = jnp.asarray(Z)

    Nt = t_grid.shape[0]
    Nx = x_grid.shape[0]

    # Clip queries to valid domain so indices are safe.
    tq = jnp.clip(t_query, t_grid[0], t_grid[-1])
    xq = jnp.clip(x_query, x_grid[0], x_grid[-1])

    # Cell indices (lower corner). searchsorted gives insertion index.
    it = jnp.searchsorted(t_grid, tq, side="right") - 1
    ix = jnp.searchsorted(x_grid, xq, side="right") - 1
    it = jnp.clip(it, 0, Nt - 2)
    ix = jnp.clip(ix, 0, Nx - 2)

    # Grid coordinates
    t0 = t_grid[it]
    t1 = t_grid[it + 1]
    x0 = x_grid[ix]
    x1 = x_grid[ix + 1]

    # Normalized weights in [0,1]
    wt = (tq - t0) / (t1 - t0 + eps)
    wx = (xq - x0) / (x1 - x0 + eps)

    # Gather corner values
    z00 = Z[it, ix]
    z10 = Z[it + 1, ix]
    z01 = Z[it, ix + 1]
    z11 = Z[it + 1, ix + 1]

    # Bilinear interpolation
    z0 = z00 * (1.0 - wt) + z10 * wt
    z1 = z01 * (1.0 - wt) + z11 * wt
    z = z0 * (1.0 - wx) + z1 * wx
    return z