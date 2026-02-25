import os
import time

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

import ml_collections
from tqdm import trange
import wandb

from src.sampling import UniformSampler, ResSampler
from src.logging import Logger
from src.utils import load_dataset, save_checkpoint
from pdes.burgers.model import Burgers
from pdes.allen_cahn.model import Allen_Cahn

_PDE_MODELS = {
    "burgers": Burgers,
    "allen_cahn": Allen_Cahn,
}

def train(config: ml_collections.ConfigDict):
    assert config.pde.name in _PDE_MODELS, f"Unknown PDE: {config.pde.name}"
    workdir = '/scratch/merlinf/repos/PINNs-Training-Dynamics'

    if config.wandb.use:
        wandb.init(
            project=config.wandb.project,
            name=config.pde.experiment,
        )
    
    logger = Logger(
        name=config.pde.experiment,
        handler_type=config.logging.handler_type,
        log_info={
            'log_dir': config.logging.log_dir,
            'file_name': config.pde.experiment,
        } if config.logging.handler_type == 'file' else None,
    )

    data_dir = os.path.join(config.pde.name, 'data', f"{config.pde.name}.mat")
    u_ref, t, x = load_dataset(data_dir)

    ModelClass = _PDE_MODELS[config.pde.name]
    model = ModelClass(config, IC=(u_ref[0, :], jnp.full_like(x, t[0]), x))
    per_device_batch_size = (config.training.global_batch_size // jax.local_device_count()) if config.training.global_batch_size > 1000 else config.training.batch_size_per_device

    if config.sampling.method == 'uniform':
        sampler = UniformSampler(jnp.array([[t[0], t[-1]], [x[0], x[-1]]]), per_device_batch_size, config.training.seed)

    elif config.sampling.method == 'residual':
        p_sampler = 0.0
        sampler = ResSampler(
            dom = jnp.array([[t[0], t[-1]], [x[0], x[-1]]]),
            batch_size = per_device_batch_size,
            residual_fn = model.get_residual,
            rng_key = config.training.seed,
            p = p_sampler,
            hard_bank_mult=config.sampling.hard_bank_mult,
            cand_mult=config.sampling.cand_mult,
            top_frac=config.sampling.top_frac,
            hardness=config.sampling.hardness,
        )
    save_dir = os.path.join(workdir, 'ckpts', config.pde.name, config.pde.experiment)

    print("Waiting for jit...")

    start_time = time.time()
    pbar = trange(
        config.training.num_steps,
        desc="Training",
        dynamic_ncols=True,
    )

    for step in pbar:
        batch = sampler[0]
        model.state = model.train_step(model.state, batch)

        if config.weighing.scheme == 'ntk':
            if step % config.weighing.update_freq == 0 and step > 0:
                model.state = model.update_loss_weights(model.state, batch)

        if config.sampling.method == 'residual':
            if step % config.sampling.refresh_freq == 0 and step > 0.4 * config.training.num_steps:
                p_sampler = (step - 0.4 * config.training.num_steps) / (0.4 * config.training.num_steps) * 0.2 # Linearly increase p from 0 to 0.2 between 40% and 80% of training
                sampler.p = p_sampler
                sampler.refresh(model.state.params)
 
        if step % config.logging.freq == 0:
            state = jax.device_get(tree_map(lambda x: x[0], model.state))
            batch = jax.device_get(tree_map(lambda x: x[0], batch))
            log_dict = model.metrics_step(state, batch, u_ref, t, x)
            if config.wandb.use:
                wandb.log(log_dict, step)

            pbar.set_postfix({"L2_error": float(log_dict.get("l2_error", 0.0))})

            end_time = time.time()
            logger.record(step, log_dict, start_time, end_time)
            start_time = end_time

            if config.training.save_freq is not None:
                if step % config.training.save_freq == 0 and step > 0:
                    if not os.path.isdir(save_dir):
                        os.makedirs(save_dir)
                    save_checkpoint(model.state, step, save_dir)

    save_checkpoint(model.state, step, save_dir)
    
    return model, save_dir