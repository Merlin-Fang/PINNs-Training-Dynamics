from functools import partial

import jax.numpy as jnp
from jax import random, pmap, local_device_count

from torch.utils.data import Dataset

class Sampler(Dataset):
    def __init__(self, dom, batch_size, rng_key=random.PRNGKey(42)):
        self.key = rng_key
        self.batch_size = batch_size
        self.num_devices = local_device_count()
        self.dom = dom
        self.dim = dom.shape[0]

    def data_generation(self, keys):
        batch = random.uniform(
            keys,
            shape=(self.batch_size, self.dim),
            minval=self.dom[:, 0],
            maxval=self.dom[:, 1],
        )
        return batch

    def __getitem__(self, index):
        self.key, key1 = random.split(self.key)
        keys = random.split(key1, self.num_devices)
        batch = self.data_generation(keys)
        return batch