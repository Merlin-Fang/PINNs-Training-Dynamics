from typing import Callable, Optional, Dict, Tuple

import jax.numpy as jnp
from flax import linen as nn

from src.architectures.mlp import MLP


class CorrMLP(nn.Module):
    """
    Correction-network wrapper around the existing MLP.

    - u_corr(t,x) is produced by an internal MLP (same architecture options as your base MLP).
    - gamma is a *scalar* trainable parameter.
    - g is a bounded gain used ONLY for the teacher loss:
        g = 0.5 + 1.5 * sigmoid(gamma)  => g in (0.5, 2.0)

    Forward returns:
        (u_corr, g)

    Notes:
    - This does NOT apply g to u_corr. Your training code decides where to use g (teacher loss only).
    - This does NOT include wconf. wconf is applied outside: u_hat = u_base + wconf * u_corr.
    """

    # --- MLP settings (mirrors src/architectures/mlp.py) ---
    hidden_layers: int = 4
    hidden_size: int = 256
    output_size: int = 1
    activation: Callable = nn.relu
    use_bias: bool = True
    weight_fact: Optional[Dict] = None
    periodic_embed: Optional[Dict] = None
    fourier_embed: Optional[Dict] = None

    # --- gain param settings ---
    gain_range: Tuple[float, float] = (0.5, 2.0)
    gamma_init: float = -0.693147 # corresponds to g_init = 0.5 + 1.5 * sigmoid(gamma_init) = 1.0

    @nn.compact
    def __call__(self, x):
        # 1) correction field
        u_corr = MLP(
            hidden_layers=self.hidden_layers,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            activation=self.activation,
            use_bias=self.use_bias,
            weight_fact=self.weight_fact,
            periodic_embed=self.periodic_embed,
            fourier_embed=self.fourier_embed,
        )(x)

        # 2) teacher-only gain scalar
        gamma = self.param("gamma", nn.initializers.constant(self.gamma_init), ())
        lo, hi = self.gain_range
        g = lo + (hi - lo) * nn.sigmoid(gamma)

        return u_corr, g