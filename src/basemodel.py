from typing import Dict
from functools import partial
from abc import ABC, abstractmethod

import jax
import optax
from jax import lax, pmap, jit, grad, vmap
import jax.numpy as jnp
from flax import linen as nn
from flax import jax_utils
from flax.training import train_state
from jax.tree_util import tree_map, tree_leaves, tree_reduce
from matplotlib import pyplot as plt

from src.architectures.mlp import MLP

_ACTIVATIONS = {
    "tanh": nn.tanh,
    "relu": nn.relu,
    "gelu": nn.gelu,
    "silu": nn.silu,
    "swish": nn.swish,
}

def get_activation(name: str) -> callable:
    if name in _ACTIVATIONS:
        return _ACTIVATIONS[name]
    else:
        raise ValueError(f"Activation '{name}' not recognized. Available activations: {list(_ACTIVATIONS.keys())}")

class TrainState(train_state.TrainState):
    loss_weights: Dict
    momentum: float
    ntk: Dict

    def apply_loss_weights(self, new_loss_weights: Dict, new_ntk: Dict):
        if not self.loss_weights:
            return self.replace(loss_weights=new_loss_weights, ntk=new_ntk)

        update_weights = (lambda old_weights, new_weights: self.momentum * old_weights + (1 - self.momentum) * new_weights)
        new_weights = tree_map(update_weights, self.loss_weights, new_loss_weights)
        new_weights = lax.stop_gradient(new_weights)
        return self.replace(loss_weights=new_weights, ntk=new_ntk)
    

def create_model(config) -> nn.Module:
    model = MLP(
        hidden_layers=config.model.hidden_layers,
        hidden_size=config.model.hidden_size,
        output_size=config.model.output_size,
        activation=get_activation(config.model.activation),
        weight_fact=config.model.weight_fact
    )
    return model

def create_optimizer(config):
    if config.optim.optimizer == "Adam":
        lr = optax.exponential_decay(
            init_value=config.optim.learning_rate,
            transition_steps=config.optim.decay_steps,
            decay_rate=config.optim.decay_rate,
        )
        tx = optax.adam(
            learning_rate=lr, b1=config.optim.beta1, b2=config.optim.beta2, eps=config.optim.eps
        )

    else:
        raise NotImplementedError(f"Optimizer {config.optim.optimizer} not supported yet!")

    if config.optim.grad_accum_steps > 1:
        tx = optax.MultiSteps(tx, every_k_schedule=config.optim.grad_accum_steps)

    return tx

def create_train_state(config):
    model = create_model(config)
    dummy = jnp.ones((2,))
    params = model.init(jax.random.PRNGKey(0), dummy)["params"]
    tx = create_optimizer(config)

    init_loss_weights = {
        k: jnp.array(float(v)) for k, v in dict(config.training.loss_weights).items()
    }
    init_ntk = {k: jnp.array(0.0) for k in init_loss_weights.keys()}

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        loss_weights=init_loss_weights,
        momentum=float(config.training.momentum),
        ntk=init_ntk,
    )
    return jax_utils.replicate(state)


class PINNs:
    def __init__(self, config):
        self.config = config
        self.model = create_model(config)
        self.state = create_train_state(config)

    def get_solution(self, params, t, x):
        x = jnp.stack([t, x])
        u = self.model.apply({'params': params}, x)
        return u[0]

    @abstractmethod
    def get_residual(self, params, t, x):
        pass  # To be implemented in each pde subclass

    @abstractmethod
    def get_losses(self, params, batch):
        pass  # To be implemented in each pde subclass

    @abstractmethod
    def get_diag_ntk(self, params, batch):
        pass  # To be implemented in each pde subclass

    @abstractmethod
    def get_l2_error(self, params, t, x):
        pass  # To be implemented in each pde subclass
    
    @partial(jit, static_argnums=(0,))
    def get_total_loss(self, params, weight, batch):
        losses = self.get_losses(params, batch)
        weighted_losses = tree_map(lambda l, w: l * w, losses, weight)
        total_loss = tree_reduce(lambda a, b: a + b, weighted_losses, initializer=0.0)
        return total_loss

    @partial(jit, static_argnums=(0,))
    def get_loss_weights(self, params, batch):
        if self.config.weighing.scheme == 'ntk':
            diag_ntk = self.get_diag_ntk(params, batch)
            mean_ntk_per_loss = tree_map(lambda x: jnp.mean(x), diag_ntk)
            mean_ntk = jnp.mean(jnp.stack(tree_leaves(mean_ntk_per_loss)))
            loss_weights = tree_map(lambda x: mean_ntk / (x + 1e-8), mean_ntk_per_loss)
        else:
            diag_ntk = self.get_diag_ntk(params, batch)
            loss_weights = tree_map(lambda _: jnp.array(1.0), diag_ntk)

        mean_ntk = tree_map(lambda x: jnp.mean(x), diag_ntk)
        return loss_weights, mean_ntk

    @partial(pmap, axis_name='batch', static_broadcasted_argnums=(0,))
    def update_loss_weights(self, state, batch):
        new_weights, new_ntk = self.get_loss_weights(state.params, batch)
        new_weights = tree_map(lambda x: lax.pmean(x, 'batch'), new_weights)
        new_ntk = tree_map(lambda x: lax.pmean(x, 'batch'), new_ntk)
        updated_state = state.apply_loss_weights(new_weights, new_ntk)
        return updated_state

    @partial(pmap, axis_name='batch', static_broadcasted_argnums=(0,))
    def train_step(self, state, batch):
        grads = grad(self.get_total_loss)(state.params, state.loss_weights, batch)
        grads = lax.pmean(grads, axis_name='batch')
        new_state = state.apply_gradients(grads=grads)
        return new_state
    
    def metrics_step(self, state, batch, u_ref, t, x):
        """
        Output log_dict: {
            'ic_loss': float,
            'res_loss': float,
            'ic_weight': float,
            'res_weight': float,
            'ic_ntk': float,
            'res_ntk': float,
            'l2_error': float,
            'u_pred': fig,
        }
        """
        log_dict = {}
        params = state.params

        if self.config.logging.log_losses:
            losses = self.get_losses(params, batch)
            for key, value in losses.items():
                log_dict[f"{key}_loss"] = value
        
        if self.config.logging.log_loss_weights:
            for key, value in state.loss_weights.items():
                log_dict[f"{key}_weight"] = value

        if self.config.logging.log_ntk:
            for key, value in state.ntk.items():
                log_dict[f"{key}_ntk"] = value

        if self.config.logging.log_l2_error:
            u_pred = vmap(vmap(self.get_solution, (None, None, 0)), (None, 0, None))(params, t, x)
            l2_error = jnp.linalg.norm(u_pred - u_ref) / jnp.linalg.norm(u_ref)
            log_dict['l2_error'] = l2_error

            fig = plt.figure(figsize=(6, 5))
            u_pred_np = jax.device_get(u_pred)
            plt.imshow(u_pred_np, cmap="jet")
            log_dict["u_pred"] = fig
            plt.close(fig)

        return log_dict