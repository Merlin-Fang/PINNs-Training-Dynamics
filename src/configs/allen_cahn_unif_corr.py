from ml_collections import ConfigDict
from jax import numpy as jnp

def get_config():
    config = ConfigDict()

    config.pde = ConfigDict()
    config.pde.name = 'allen_cahn'
    config.pde.run = 'corr_lambda300_normalizedTeacherLoss'
    config.pde.experiment = config.pde.name + '_' + config.pde.run

    config.corr = ConfigDict()
    config.corr.base_ckptdir = '/scratch/merlinf/repos/PINNs-Training-Dynamics/ckpts/allen_cahn/allen_cahn_uniform_sampling'
    config.corr.base_step = 199000
    config.corr.assets_path = '/scratch/merlinf/repos/PINNs-Training-Dynamics/pdes/allen_cahn/corr_assets/corr_assets_early101000_late199000.npz'
    config.corr.alpha_schedule = {
        "type": "sigmoid",  # "linear" | "cosine" | "sigmoid"
        "a0": 1.0,
        "a1": 0.2,
        "t0": 0.3,
        "t1": 0.6,
        "k": 10.0,          # only used by sigmoid; higher = sharper transition
    }

    config.model = ConfigDict()
    config.model.hidden_layers = 4
    config.model.hidden_size = 256
    config.model.output_size = 1
    config.model.activation = 'tanh'
    config.model.weight_fact = {'mean': 0.5, 'stddev': 0.1}
    config.model.periodic_embed = {'period': jnp.pi, 'axis': (1,)}
    config.model.fourier_embed = {'scale': 1.0, 'dim': 256}

    config.training = ConfigDict()
    config.training.seed = 42
    config.training.global_batch_size = 4096
    config.training.batch_size_per_device = 1024
    config.training.num_steps = 50000
    config.training.save_freq = None

    config.optim = ConfigDict()
    config.optim.grad_accum_steps = 0
    config.optim.optimizer = "Adam"
    config.optim.beta1 = 0.9
    config.optim.beta2 = 0.999
    config.optim.eps = 1e-8
    config.optim.learning_rate = 3e-4
    config.optim.decay_rate = 0.9
    config.optim.decay_steps = 2000

    config.wandb = ConfigDict()
    config.wandb.use = True
    config.wandb.project = 'PINNs-Correction-Net'

    config.logging = ConfigDict()
    config.logging.handler_type = 'file'
    config.logging.log_dir = (
        '/scratch/merlinf/repos/PINNs-Training-Dynamics/pdes/'
        + config.pde.name + '/logs/corr_net'
    )
    config.logging.freq = 100

    config.logging.log_alpha = True
    config.logging.log_g = True
    config.logging.log_teacher_loss = True
    config.logging.log_pinns_loss = True
    config.logging.log_IC_res_loss = True
    config.logging.log_L2error = True

    # ---- dummy variables (value does not matter, only for structure for base state) ----
    config.training.loss_weights = {"ic": 1.0, "res": 1.0}
    config.training.momentum = 0.9
    config.sampling = ConfigDict()
    config.sampling.method = 'uniform'
    config.sampling.hard_bank_mult = 50
    config.sampling.cand_mult = 50
    config.sampling.top_frac = 0.05
    config.sampling.hardness = "abs"
    config.sampling.refresh_freq = 200

    return config