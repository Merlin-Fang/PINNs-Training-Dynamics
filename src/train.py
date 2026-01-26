import os
import time

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

import ml_collections
import wandb

from src.sampling import Sampler
from src.logging import Logger
from src.utils import load_dataset, save_checkpoint
from pdes.burgers.model import Burgers

def train(config: ml_collections.ConfigDict):
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

    u_ref, t, x = load_dataset()

    model = Burgers(config, IC=(u_ref[0, :], jnp.full_like(x, t[0]), x))
    sampler = iter(Sampler(jnp.array([[t[0], t[-1]], [x[0], x[-1]]]), config.training.batch_size, config.training.seed))
    save_dir = os.path.join(workdir, 'ckpts', config.pde.name, config.pde.experiment)

    print("Waiting for jit...")

    start_time = time.time()
    for step in range(config.training.num_steps):
        batch = next(sampler)
        model.state = model.train_step(model.state, batch)

        if config.weighing.scheme == 'ntk':
            if step % config.weighing.update_freq == 0 and step > 0:
                model.state = model.update_loss_weights(model.state, batch)

        if step % config.logging.freq == 0:
            print(f"Logging at step {step}...")
            state = jax.device_get(tree_map(lambda x: x[0], model.state))
            batch = jax.device_get(tree_map(lambda x: x[0], batch))
            log_dict = model.metrics_step(state, batch, u_ref, t, x)
            if config.wandb.use:
                wandb.log(log_dict, step)

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