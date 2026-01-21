from typing import Any, Callable, Sequence, Tuple, Optional, Union, Dict

import jax
import jax.numpy as jnp
from flax import linen as nn

def weight_fact_init(init_fn, mean, stddev):
    def init(key, shape, dtype=jnp.float32):
        key1, key2 = jax.random.split(key)
        w = init_fn(key1, shape, dtype)
        g = jnp.exp(jax.random.normal(key2, (shape[-1],)) * stddev + mean) # Magnitude of Weights
        v = w / g # Direction of Weights
        return v, g
    return init

class Dense(nn.Module):
    features: int
    kernel_init: callable = nn.initializers.kaiming_normal()
    bias_init: callable = nn.initializers.zeros
    weight_fact: Union[None, Dict] = None

    @nn.compact
    def __call__(self, inputs):
        if self.weight_fact is not None:
            mean = self.weight_fact.get('mean', 0.0)
            stddev = self.weight_fact.get('stddev', 0.1)
            v, g = self.param('kernel', weight_fact_init(self.kernel_init, mean, stddev), (inputs.shape[-1], self.features))
            w = v * g
            y = jnp.dot(inputs, w)
        else:
            kernel = self.param('kernel', self.kernel_init, (inputs.shape[-1], self.features))
            y = jnp.dot(inputs, kernel)
        bias = self.param('bias', self.bias_init, (self.features,))
        y = y + bias
        return y
    
class MLP(nn.Module):
    hidden_sizes: Sequence[int] = [256, 256, 256, 256]
    output_size: int = 1
    activation: Callable = nn.relu
    use_bias: bool = True
    weight_fact: Union[None, Dict] = None

    @nn.compact
    def __call__(self, x):
        for size in self.hidden_sizes:
            x = Dense(features=size, weight_fact=self.weight_fact)(x)
            x = self.activation(x)
        x = Dense(features=self.output_size, weight_fact=self.weight_fact)(x)
        return x