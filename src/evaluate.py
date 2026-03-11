import jax
from jax import tree_map, vmap
import jax.numpy as jnp
import os
import numpy as np

import matplotlib.pyplot as plt

from src.utils import load_dataset, load_checkpoint, interp2d_grid

def eval(config, ckptdir, model, step=None):
    data_dir = os.path.join(config.pde.name, 'data', f"{config.pde.name}.mat")
    u_ref, t, x = load_dataset(data_dir)
    state_host = jax.device_get(tree_map(lambda y: y[0], model.state))
    state = load_checkpoint(state_host, ckptdir, step=step)
    params = state.params

    u_pred = vmap(vmap(model.get_solution, in_axes=(None, 0, None)), in_axes=(None, None, 0))(params, t, x)
    u_pred = u_pred.T
    l2_error = jax.numpy.linalg.norm(u_pred - u_ref) / jax.numpy.linalg.norm(u_ref)
    #print("L2 error: {:.3e}".format(l2_error))

    t_grid, x_grid = jax.numpy.meshgrid(t, x, indexing='ij')

    fig = plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(t_grid, x_grid, u_ref, cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Reference")
    plt.tight_layout()

    plt.subplot(1, 3, 2)
    plt.pcolor(t_grid, x_grid, u_pred, cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Predicted")
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.pcolor(t_grid, x_grid, jnp.abs(u_ref - u_pred), cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Absolute error (L2 error={:.3e})".format(l2_error))
    plt.tight_layout()

    workdir = '/scratch/merlinf/repos/PINNs-Training-Dynamics/pdes'
    save_dir = os.path.join(workdir, config.pde.name, "figures", config.pde.experiment)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    step_int = int(state.step)
    fig_path = os.path.join(save_dir, f"{config.pde.experiment}, step={step_int}.png")
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)


def eval_corr(config, ckptdir, model, step=None):
    """
    Corr evaluation (drop-in replacement).

    Produces 3 panels:
      1) |u_ref - u_base|
      2) |w(t,x) * u_corr|
      3) ||u_ref - u_base| - |w(t,x) * u_corr||  (abs difference)

    L2 error shown is for the FULL corrected prediction:
        u_hat = u_base + w(t,x) * u_corr
        L2 = ||u_hat - u_ref|| / ||u_ref||
    """

    

    # -------------------------
    # Load reference dataset
    # -------------------------
    data_dir = os.path.join(config.pde.name, "data", f"{config.pde.name}.mat")
    u_ref, t, x = load_dataset(data_dir)  # u_ref is typically (Nx, Nt) or (Nt, Nx) depending on dataset

    # Ensure JAX arrays for compute
    t = jnp.asarray(t)
    x = jnp.asarray(x)
    u_ref = jnp.asarray(u_ref)

    # -------------------------
    # Load checkpointed state (host copy of replica 0)
    # -------------------------
    state_host = jax.device_get(tree_map(lambda y: y[0], model.state))
    state = load_checkpoint(state_host, ckptdir, step=step)

    params_corr = state.params

    # base_params are now stored on the state (CorrTrainState.base_params)
    if hasattr(state, "base_params") and state.base_params is not None:
        base_params = state.base_params
    else:
        # Backward compatibility (older codepaths)
        if hasattr(model, "base_params"):
            base_params = jax.device_get(tree_map(lambda y: y[0], model.base_params))
        else:
            raise AttributeError(
                "Could not find base params. Expected state.base_params (new) or model.base_params (old)."
            )

    # -------------------------
    # Build dataset mesh (t,x)
    # -------------------------
    # meshgrid gives arrays shaped (Nt, Nx) with indexing='ij'
    T, X = jnp.meshgrid(t, x, indexing="ij")  # (Nt, Nx), (Nt, Nx)
    tq = T.reshape(-1)
    xq = X.reshape(-1)

    # -------------------------
    # Evaluate base and corr nets on the dataset grid
    # -------------------------
    def base_sol(ti, xi):
        inp = jnp.stack([ti, xi])  # (2,)
        return model.base_model.apply({"params": base_params}, inp)[0]

    def corr_sol(ti, xi):
        inp = jnp.stack([ti, xi])
        u_corr, _g = model.corr_model.apply({"params": params_corr}, inp)
        return u_corr[0]

    # Vectorize over flattened grid
    u_base_flat = vmap(base_sol)(tq, xq)          # (Nt*Nx,)
    u_corr_flat = vmap(corr_sol)(tq, xq)          # (Nt*Nx,)

    u_base = u_base_flat.reshape(T.shape)         # (Nt, Nx)
    u_corr_grid = u_corr_flat.reshape(T.shape)    # (Nt, Nx)

    # -------------------------
    # Interpolate wconf onto dataset grid (robust even if grids differ)
    # -------------------------
    # Prefer model's in-memory corr assets (fast, no file IO)
    if hasattr(model, "t_grid") and hasattr(model, "x_grid") and hasattr(model, "wconf_map"):
        t_grid = model.t_grid
        x_grid = model.x_grid
        wconf_map = model.wconf_map
    else:
        # Fallback: load from disk if model doesn't carry assets
        assets = np.load(config.corr.assets_path, allow_pickle=True)
        t_grid = jnp.asarray(assets["t_grid"])
        x_grid = jnp.asarray(assets["x_grid"])
        wconf_map = jnp.asarray(assets["wconf_map"])

    wconf_flat = interp2d_grid(t_grid, x_grid, wconf_map, tq, xq)  # (Nt*Nx,)
    wconf = wconf_flat.reshape(T.shape)                            # (Nt, Nx)

    # -------------------------
    # Combine + metrics
    # -------------------------
    corr_term = wconf * u_corr_grid
    u_hat = u_base + corr_term

    # Make sure u_ref is (Nt, Nx) to match mesh; many PDE .mat files store usol as (Nx, Nt)
    # Your existing non-corr eval transposes u_pred; mirror that convention here.
    # If u_ref is (Nx, Nt), transpose it.
    if u_ref.shape == (x.shape[0], t.shape[0]):
        u_ref_plot = u_ref.T  # -> (Nt, Nx)
    else:
        u_ref_plot = u_ref

    l2_error = jnp.linalg.norm(u_hat - u_ref_plot) / (jnp.linalg.norm(u_ref_plot) + 1e-12)

    abs_err_base = jnp.abs(u_ref_plot - u_base)
    abs_corr = jnp.abs(corr_term)
    abs_diff = jnp.abs(abs_err_base - abs_corr)

    # Pull to host for plotting (avoid implicit device sync surprises)
    abs_err_base_np = np.array(jax.device_get(abs_err_base))
    abs_corr_np = np.array(jax.device_get(abs_corr))
    abs_diff_np = np.array(jax.device_get(abs_diff))
    t_grid_np = np.array(jax.device_get(T))
    x_grid_np = np.array(jax.device_get(X))
    l2_error_val = float(jax.device_get(l2_error))

    # -------------------------
    # Plot
    # -------------------------
    fig = plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.pcolor(t_grid_np, x_grid_np, abs_err_base_np, cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Base abs error |u_ref - u_base|")
    plt.tight_layout()

    plt.subplot(1, 3, 2)
    plt.pcolor(t_grid_np, x_grid_np, abs_corr_np, cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Abs correction |w * u_corr|")
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.pcolor(t_grid_np, x_grid_np, abs_diff_np, cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Abs diff (L2 error={:.3e})".format(l2_error_val))
    plt.tight_layout()

    # -------------------------
    # Save
    # -------------------------
    workdir = "/scratch/merlinf/repos/PINNs-Training-Dynamics/pdes"
    save_dir = os.path.join(workdir, config.pde.name, "figures", config.pde.experiment)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    step_int = int(state.step)
    fig_path = os.path.join(save_dir, f"{config.pde.experiment}, step={step_int}.png")
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close(fig)