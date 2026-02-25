import jax
from jax import tree_map, vmap
import jax.numpy as jnp
import os
import numpy as np

import matplotlib.pyplot as plt

from src.utils import load_dataset, load_checkpoint

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
    Corr evaluation:

      Panel 1: |u_ref - u_base|
      Panel 2: |w * u_corr|
      Panel 3: ||u_ref - u_base| - |w * u_corr||  (abs difference between first two panels)

    L2 error shown is for the FULL corrected prediction:
        u_hat = u_base + w * u_corr
        L2 = ||u_hat - u_ref|| / ||u_ref||
    """
    # ---- load dataset ----
    data_dir = os.path.join(config.pde.name, 'data', f"{config.pde.name}.mat")
    u_ref, t, x = load_dataset(data_dir)

    # ---- load corr checkpoint params ----
    state_host = jax.device_get(tree_map(lambda y: y[0], model.state))
    state = load_checkpoint(state_host, ckptdir, step=step)
    params_corr = state.params

    # ---- get base params from the corr model instance (already frozen) ----
    # (model.base_params is typically replicated; take device 0 copy)
    base_params = jax.device_get(tree_map(lambda y: y[0], model.base_params))

    # ---- load wconf from assets (grid matches t,x exactly, so no interpolation) ----
    assets = np.load(config.corr.assets_path, allow_pickle=True)
    wconf = jnp.asarray(assets["wconf_map"], dtype=jnp.float32)  # (Nt, Nx)

    # ---- compute u_base grid ----
    def base_sol(ti, xi):
        inp = jnp.stack([ti, xi])  # (2,)
        return model.base_model.apply({"params": base_params}, inp)[0]

    u_base = vmap(
        vmap(base_sol, in_axes=(None, 0)),
        in_axes=(0, None),
    )(t, x).T  # -> (Nt, Nx) to match u_ref

    # ---- compute u_corr grid ----
    def corr_u(ti, xi):
        inp = jnp.stack([ti, xi])
        u_corr, _g = model.corr_model.apply({"params": params_corr}, inp)
        return u_corr[0]

    u_corr_grid = vmap(
        vmap(corr_u, in_axes=(None, 0)),
        in_axes=(0, None),
    )(t, x).T  # (Nt, Nx)

    # ---- correction term and combined prediction ----
    corr_term = wconf * u_corr_grid             # signed
    u_hat = u_base + corr_term                  # signed

    # ---- L2 error for combined model ----
    l2_error = jnp.linalg.norm(u_hat - u_ref) / jnp.linalg.norm(u_ref)

    # ---- panels requested ----
    abs_err_base = jnp.abs(u_ref - u_base)
    abs_corr = jnp.abs(corr_term)
    abs_diff = jnp.abs(abs_err_base - abs_corr)

    t_grid, x_grid = jnp.meshgrid(t, x, indexing='ij')

    fig = plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.pcolor(t_grid, x_grid, abs_err_base, cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Base abs error |u_ref - u_base|")
    plt.tight_layout()

    plt.subplot(1, 3, 2)
    plt.pcolor(t_grid, x_grid, abs_corr, cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Abs correction |w * u_corr|")
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.pcolor(t_grid, x_grid, abs_diff, cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Abs diff (L2 error={:.3e})".format(l2_error))
    plt.tight_layout()

    workdir = '/scratch/merlinf/repos/PINNs-Training-Dynamics/pdes'
    save_dir = os.path.join(workdir, config.pde.name, "figures", config.pde.experiment)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    step_int = int(state.step)
    fig_path = os.path.join(save_dir, f"{config.pde.experiment}, step={step_int}.png")
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close(fig)