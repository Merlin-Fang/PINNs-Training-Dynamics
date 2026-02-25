# train_corr.py
import os
import time

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

import numpy as np
import ml_collections
from tqdm import trange
import wandb
from flax import jax_utils

from src.sampling import UniformSampler
from src.logging import Logger
from src.utils import load_dataset, load_checkpoint, save_checkpoint

# Base PDE models (for loading base checkpoint params/weights)
from pdes.burgers.model import Burgers
from pdes.allen_cahn.model import Allen_Cahn

# Corr PDE models (you added these classes)
from pdes.burgers.model import Burgers_Corr
from pdes.allen_cahn.model import Allen_Cahn_Corr


_PDE_MODELS_BASE = {
    "burgers": Burgers,
    "allen_cahn": Allen_Cahn,
}

_PDE_MODELS_CORR = {
    "burgers": Burgers_Corr,
    "allen_cahn": Allen_Cahn_Corr,
}


def _load_corr_assets_npz(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    # float32 is what you want for training
    corr_assets = {
        "t_grid": jnp.asarray(data["t_grid"], dtype=jnp.float32),
        "x_grid": jnp.asarray(data["x_grid"], dtype=jnp.float32),
        "teacher_map": jnp.asarray(data["teacher_map"], dtype=jnp.float32),
        "wconf_map": jnp.asarray(data["wconf_map"], dtype=jnp.float32),
    }
    return corr_assets


def train_corr(config: ml_collections.ConfigDict):
    """
    Corr training loop (uniform sampling only for now), structurally similar to your base train().

    Requires the config to include (suggested):
      config.corr.base_ckptdir : str
      config.corr.base_step    : int
      config.corr.assets_path  : str  (npz produced by precompute_corr_assets.py)
      config.corr.experiment   : str  (optional; default = config.pde.experiment + "_corr")
      config.corr.alpha_schedule : dict (optional; passed into CorrPINNs)
    """
    assert config.pde.name in _PDE_MODELS_BASE, f"Unknown PDE: {config.pde.name}"
    assert config.pde.name in _PDE_MODELS_CORR, f"Missing corr PDE model for: {config.pde.name}"

    # ---- Required corr config ----
    if not hasattr(config, "corr"):
        raise ValueError("config.corr is missing. Need base_ckptdir/base_step/assets_path at minimum.")
    if not hasattr(config.corr, "base_ckptdir"):
        raise ValueError("config.corr.base_ckptdir is missing")
    if not hasattr(config.corr, "base_step"):
        raise ValueError("config.corr.base_step is missing")
    if not hasattr(config.corr, "assets_path"):
        raise ValueError("config.corr.assets_path is missing (path to corr_assets_*.npz)")

    workdir = "/scratch/merlinf/repos/PINNs-Training-Dynamics"
    experiment_name = config.pde.experiment

    if config.wandb.use:
        wandb.init(project=config.wandb.project, name=experiment_name)

    logger = Logger(
        name=experiment_name,
        handler_type=config.logging.handler_type,
        log_info={
            "log_dir": config.logging.log_dir,
            "file_name": experiment_name,
        } if config.logging.handler_type == "file" else None,
    )

    # ---- Load dataset ----
    data_dir = os.path.join(config.pde.name, "data", f"{config.pde.name}.mat")
    u_ref, t, x = load_dataset(data_dir)

    IC = (u_ref[0, :], jnp.full_like(x, t[0]), x)

    # ---- Load base checkpoint (params + frozen loss weights) ----
    BaseModelClass = _PDE_MODELS_BASE[config.pde.name]
    base_model = BaseModelClass(config, IC=IC)

    # Build a host state template from the base model so load_checkpoint can restore into it.
    base_state_host = jax.device_get(tree_map(lambda y: y[0], base_model.state))
    base_state = load_checkpoint(
        base_state_host,
        config.corr.base_ckptdir,
        step=int(config.corr.base_step),
    )

    base_params = base_state.params
    if not hasattr(base_state, "loss_weights"):
        raise ValueError(
            "Loaded base checkpoint state does not have 'loss_weights'. "
            "Corr training expects frozen IC/res weights from base training."
        )
    frozen_loss_weights = base_state.loss_weights  # expects keys like {"ic":..., "res":...}

    # ---- Load corr assets ----
    corr_assets = _load_corr_assets_npz(config.corr.assets_path)

    # ---- Replicate frozen things for pmap ----
    # CorrPINNs uses these inside a pmapped train_step, so they must be device arrays.
    base_params_repl = jax_utils.replicate(base_params)
    corr_assets_repl = {k: jax_utils.replicate(v) for k, v in corr_assets.items()}

    # Frozen weights are scalars; CorrPINNs converts them to jnp inside __init__ anyway.
    # Keep them as plain floats here for simplicity.
    frozen_w = {
        "ic": float(frozen_loss_weights["ic"]),
        "res": float(frozen_loss_weights["res"]),
    }

    alpha_schedule = getattr(config.corr, "alpha_schedule", None)

    # ---- Construct corr model ----
    CorrModelClass = _PDE_MODELS_CORR[config.pde.name]
    model = CorrModelClass(
        config,
        IC,
        base_params=base_params_repl,
        corr_assets=corr_assets_repl,
        frozen_loss_weights=frozen_w,
        alpha_schedule=alpha_schedule,
    )

    # ---- Sampler (uniform only, same structure as base train) ----
    per_device_batch_size = (
        (config.training.global_batch_size // jax.local_device_count())
        if config.training.global_batch_size > 1000
        else config.training.batch_size_per_device
    )

    sampler = UniformSampler(
        jnp.array([[t[0], t[-1]], [x[0], x[-1]]]),
        per_device_batch_size,
        config.training.seed,
    )

    # ---- Output dir ----
    save_dir = os.path.join(workdir, "ckpts", config.pde.name, experiment_name)

    print("Waiting for jit...")

    start_time = time.time()
    pbar = trange(
        config.training.num_steps,
        desc="Corr Training",
        dynamic_ncols=True,
    )

    num_steps = int(config.training.num_steps)
    denom = max(num_steps - 1, 1)

    for step in pbar:
        batch = sampler[0]  # should be sharded: (n_devices, per_device_batch, 2)

        progress = float(step / denom)  # percentile in [0,1]
        model.state = model.train_step(model.state, batch, progress)

        if step % config.logging.freq == 0:
            state_host = jax.device_get(tree_map(lambda x: x[0], model.state))
            batch_host = jax.device_get(tree_map(lambda x: x[0], batch))

            log_dict = model.metrics_step(state_host, batch_host, u_ref, t, x, progress)

            if config.wandb.use:
                wandb.log(log_dict, step)

            # show something lightweight in tqdm
            pbar.set_postfix({"L2_error": float(log_dict.get("l2_error", 0.0))})

            end_time = time.time()
            logger.record(step, log_dict, start_time, end_time)
            start_time = end_time

            if config.training.save_freq is not None:
                if step % config.training.save_freq == 0 and step > 0:
                    if not os.path.isdir(save_dir):
                        os.makedirs(save_dir)
                    save_checkpoint(model.state, step, save_dir)

    # final save
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_checkpoint(model.state, step, save_dir)

    return model, save_dir