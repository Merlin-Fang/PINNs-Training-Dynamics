from typing import Any, Callable, Sequence, Tuple, Optional, Union, Dict
from functools import partial
from abc import ABC, abstractmethod

import jax
from jax import lax, pmap, jit, grad
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from jax.tree_util import tree_map, tree_leaves, tree_reduce

from src.models.mlp import MLP

class TrainState(train_state.TrainState):
    loss_weights: Dict
    momentum: float

    def apply_loss_weights(self, new_loss_weights: Dict):
        if not self.loss_weights:
            return self.replace(loss_weights=new_loss_weights)

        update_weights = (lambda old_weights, new_weights: self.momentum * old_weights + (1 - self.momentum) * new_weights)
        new_weights = tree_map(update_weights, self.loss_weights, new_loss_weights)
        new_weights = lax.stop_gradient(new_weights)
        return self.replace(loss_weights=new_weights)
    

def create_model(config) -> nn.Module:
    model = MLP(
        hidden_sizes=config.model.hidden_sizes,
        output_size=config.model.output_size,
        activation=config.model.activation,
        weight_fact=config.model.weight_fact
    )
    return model

def create_train_state(config):
    model = create_model(config)
    params = model.init(jax.random.PRNGKey(0), jnp.ones([1, config.model.hidden_sizes[0]]))['params']
    tx = jax.experimental.optimizers.adam(config.training.learning_rate)
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        loss_weights=config.training.loss_weights,
        momentum=config.training.momentum
    )
    return state

class PINNs:
    def __init__(self, config):
        self.config = config
        self.model = create_model(config)
        self.init_state = create_train_state(config)

    @abstractmethod 
    def get_solution(self, params, t, x):
        pass  # To be implemented in each pde subclass

    @abstractmethod
    def get_residual(self, params, t, x):
        pass  # To be implemented in each pde subclass

    @abstractmethod
    def get_losses(self, params, batch):
        pass  # To be implemented in each pde subclass

    @abstractmethod
    def get_diag_ntk(self, params, batch):
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
            loss_weights = tree_map(lambda x: 1.0, diag_ntk)
        return loss_weights

    @partial(pmap, axis_name='batch', static_broadcasted_argnums=(0,))
    def update_loss_weights(self, params, state, batch):
        new_weights = self.get_loss_weights(params, batch)
        new_weights = lax.pmean(new_weights, axis_name='batch')
        updated_state = state.apply_loss_weights(new_weights)
        return updated_state

    @partial(pmap, axis_name='batch', static_broadcasted_argnums=(0,))
    def train_step(self, params, state, batch):
        grads = grad(self.get_total_loss)(params, state.loss_weights, batch)
        grads = lax.pmean(grads, axis_name='batch')
        new_state = state.apply_gradients(grads=grads)
        return new_state