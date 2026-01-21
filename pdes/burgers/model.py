from typing import Any, Callable, Sequence, Tuple, Optional, Union, Dict
from functools import partial
from abc import ABC, abstractmethod

import jax
from jax import lax, pmap, jit, grad
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from jax.tree_util import tree_map, tree_leaves, tree_reduce

from src.architectures.mlp import MLP
from basemodel import PINNs

class Burgers(PINNs):
    def __init__(self, config):
        super().__init__(config)
