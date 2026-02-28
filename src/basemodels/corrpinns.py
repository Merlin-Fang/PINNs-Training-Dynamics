from typing import Dict, Optional
from functools import partial
from abc import abstractmethod

import jax
import optax
from jax import lax, pmap, jit, grad, vmap
import jax.numpy as jnp
from flax import linen as nn
from flax import jax_utils
from flax.training import train_state
from flax.core import freeze

from jax.tree_util import tree_map, tree_reduce

from src.architectures.mlp import MLP
from src.architectures.corr_mlp import CorrMLP
from src.utils import interp2d_grid


# ---------------------------------------------------------------------
# Activation registry (mirrors basemodel.py)
# ---------------------------------------------------------------------
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
    raise ValueError(
        f"Activation '{name}' not recognized. Available activations: {list(_ACTIVATIONS.keys())}"
    )

def _freeze_if_dict(x):
    if x is None:
        return None
    # Works for dict and ConfigDict-like
    return freeze(dict(x))


# ---------------------------------------------------------------------
# Model / optimizer creation
# ---------------------------------------------------------------------
def create_base_model(config) -> nn.Module:
    return MLP(
        hidden_layers=config.model.hidden_layers,
        hidden_size=config.model.hidden_size,
        output_size=config.model.output_size,
        activation=get_activation(config.model.activation),
        weight_fact=_freeze_if_dict(config.model.weight_fact),
        periodic_embed=_freeze_if_dict(config.model.periodic_embed),
        fourier_embed=_freeze_if_dict(config.model.fourier_embed),
    )

def create_corr_model(config) -> nn.Module:
    return CorrMLP(
        hidden_layers=config.model.hidden_layers,
        hidden_size=config.model.hidden_size,
        output_size=config.model.output_size,
        activation=get_activation(config.model.activation),
        weight_fact=_freeze_if_dict(config.model.weight_fact),
        periodic_embed=_freeze_if_dict(config.model.periodic_embed),
        fourier_embed=_freeze_if_dict(config.model.fourier_embed),
        gain_range=(0.5, 2.0),
    )

def create_optimizer(config):
    if config.optim.optimizer == "Adam":
        lr = optax.exponential_decay(
            init_value=config.optim.learning_rate,
            transition_steps=config.optim.decay_steps,
            decay_rate=config.optim.decay_rate,
        )
        tx = optax.adam(
            learning_rate=lr,
            b1=config.optim.beta1,
            b2=config.optim.beta2,
            eps=config.optim.eps,
        )
    else:
        raise NotImplementedError(f"Optimizer {config.optim.optimizer} not supported yet!")

    if config.optim.grad_accum_steps > 1:
        tx = optax.MultiSteps(tx, every_k_schedule=config.optim.grad_accum_steps)

    return tx


# ---------------------------------------------------------------------
# TrainState wrapper that carries frozen base params (dynamic arg, not on static self)
# ---------------------------------------------------------------------
class CorrTrainState(train_state.TrainState):
    base_params: Dict  # pytree of arrays (FrozenDict typically)


def create_corr_train_state(config, corr_model: nn.Module, base_params):
    """
    Corr TrainState:
      - params: CorrMLP params (trainable, includes gamma)
      - base_params: frozen base checkpoint params (constant)
    """
    dummy = jnp.ones((2,))
    params = corr_model.init(jax.random.PRNGKey(0), dummy)["params"]
    tx = create_optimizer(config)

    state = CorrTrainState.create(
        apply_fn=corr_model.apply,
        params=params,
        tx=tx,
        base_params=base_params,
    )
    return jax_utils.replicate(state)


# ---------------------------------------------------------------------
# CorrPINNs
# ---------------------------------------------------------------------
class CorrPINNs:
    def __init__(
        self,
        config,
        IC,
        *,
        base_params,
        corr_assets: Dict,
        frozen_loss_weights: Dict,
        alpha_schedule: Optional[Dict] = None,
    ):
        self.config = config

        # Modules
        self.base_model = create_base_model(config)
        self.corr_model = create_corr_model(config)

        # State contains BOTH corr params (trainable) and base params (frozen)
        self.state = create_corr_train_state(config, self.corr_model, base_params)

        # Initial condition: (u0, t0, x0)
        self.IC = IC

        # Corr assets (host-side JAX arrays, unreplicated)
        self.t_grid = corr_assets["t_grid"]
        self.x_grid = corr_assets["x_grid"]
        self.teacher_map = corr_assets["teacher_map"]
        self.wconf_map = corr_assets["wconf_map"]

        # Frozen physics weights (constants)
        self.frozen_loss_weights = {k: jnp.array(float(v)) for k, v in frozen_loss_weights.items()}

        # Alpha schedule (FrozenDict)
        if alpha_schedule is None:
            self.alpha_schedule = freeze({"type": "linear"})
        else:
            self.alpha_schedule = freeze(dict(alpha_schedule))

    # --------------------------
    # Alpha schedule
    # --------------------------
    def calc_alpha(self, progress: jnp.ndarray) -> jnp.ndarray:
        p = jnp.clip(progress, 0.0, 1.0)
        sch = self.alpha_schedule
        sch_type = sch.get("type", "linear")

        a0 = float(sch.get("a0", 1.0))
        a1 = float(sch.get("a1", 0.0))
        t0 = float(sch.get("t0", 0.1))
        t1 = float(sch.get("t1", 0.6))

        t0 = max(0.0, min(t0, 1.0))
        t1 = max(0.0, min(t1, 1.0))
        denom = max(t1 - t0, 1e-12)

        u = jnp.clip((p - t0) / denom, 0.0, 1.0)

        if sch_type == "linear":
            curve = u
        elif sch_type == "cosine":
            curve = 0.5 * (1.0 - jnp.cos(jnp.pi * u))
        elif sch_type == "sigmoid":
            k = float(sch.get("k", 10.0))
            def s(z):
                return 1.0 / (1.0 + jnp.exp(-k * (z - 0.5)))
            s0 = s(0.0)
            s1 = s(1.0)
            curve = (s(u) - s0) / (s1 - s0 + 1e-12)
        else:
            raise ValueError(f"alpha_schedule.type '{sch_type}' not recognized")

        a_mid = a0 + (a1 - a0) * curve
        alpha = jnp.where(p <= t0, a0, jnp.where(p >= t1, a1, a_mid))
        return jnp.clip(alpha, 0.0, 1.0)

    # --------------------------
    # Corrected forward
    # --------------------------
    def get_solution(self, params_corr, base_params, t, x):
        """
        Corrected solution:
          u_hat = u_base + wconf(t,x) * u_corr
        """
        inp = jnp.stack([t, x])  # (2,)

        u_base = self.base_model.apply({"params": base_params}, inp)[0]
        u_corr, _g = self.corr_model.apply({"params": params_corr}, inp)
        u_corr = u_corr[0]

        # scalar query -> interp returns (1,), index [0] ok
        w = interp2d_grid(self.t_grid, self.x_grid, self.wconf_map, t, x)[0]
        return u_base + w * u_corr

    @abstractmethod
    def get_residual(self, params_corr, base_params, t, x):
        """
        PDE subclass must implement residual using corrected solution.
        Should call:
          self.get_solution(params_corr, base_params, t, x)
        """
        raise NotImplementedError

    # --------------------------
    # Loss components
    # --------------------------
    def get_losses(self, params_corr, base_params, batch):
        """
        Physics losses (IC + residual) computed on corrected solution.
        """
        # IC loss: evaluate corrected solution on IC points
        u0_pred = vmap(self.get_solution, in_axes=(None, None, 0, 0))(
            params_corr, base_params, self.IC[1], self.IC[2]
        )
        ic_loss = jnp.mean((u0_pred - self.IC[0]) ** 2)

        # Residual loss on collocation batch points
        res = vmap(self.get_residual, in_axes=(None, None, 0, 0))(
            params_corr, base_params, batch[:, 0], batch[:, 1]
        )
        res_loss = jnp.mean(res ** 2)

        return {"ic": ic_loss, "res": res_loss}

    def get_pinns_loss(self, params_corr, base_params, batch):
        losses = self.get_losses(params_corr, base_params, batch)
        weighted_losses = tree_map(lambda l, w: l * w, losses, self.frozen_loss_weights)
        total_loss = tree_reduce(lambda a, b: a + b, weighted_losses, initializer=0.0)
        return total_loss, losses

    def get_teacher_loss(self, params_corr, batch):
        """
        Teacher loss:
          E[ wconf(t,x) * (g*u_corr - teacher)^2 ]
        """
        t = batch[:, 0]
        x = batch[:, 1]

        teacher = interp2d_grid(self.t_grid, self.x_grid, self.teacher_map, t, x)  # (B,)
        wconf   = interp2d_grid(self.t_grid, self.x_grid, self.wconf_map, t, x)    # (B,)

        def corr_forward(ti, xi):
            inp = jnp.stack([ti, xi])   # (2,)
            u_corr, g = self.corr_model.apply({"params": params_corr}, inp)
            return u_corr[0], g         # scalar, scalar

        u_corr, g = vmap(corr_forward, in_axes=(0, 0))(t, x)  # (B,), (B,)
        diff = (g * u_corr) - teacher
        return jnp.mean(wconf * (diff ** 2)), g[0]

    # --------------------------
    # Total loss (mixture)
    # --------------------------
    @partial(jit, static_argnums=(0,))
    def get_total_loss(self, params_corr, base_params, progress, batch):
        """
        Total corr loss:
          alpha(progress) * L_teacher + (1-alpha) * L_PINNs
        """
        # base params are constant; ensure no grads flow
        base_params = lax.stop_gradient(base_params)

        alpha = self.calc_alpha(progress)
        L_teacher, _g = self.get_teacher_loss(params_corr, batch)
        L_pinns, _losses = self.get_pinns_loss(params_corr, base_params, batch)
        return alpha * L_teacher + (1.0 - alpha) * L_pinns

    # --------------------------
    # Train step (pmapped)
    # --------------------------
    @partial(
        pmap,
        axis_name="batch",
        static_broadcasted_argnums=(0,),
        in_axes=(None, 0, 0, None),  # self static, state sharded, batch sharded, progress broadcast
    )
    def train_step(self, state: CorrTrainState, batch, progress):
        """
        One corr update step.
        state: per-device CorrTrainState (contains params + base_params)
        batch: per-device (local_B, 2)
        progress: scalar broadcast to all devices
        """
        def loss_fn(params_corr):
            return self.get_total_loss(params_corr, state.base_params, progress, batch)

        grads = grad(loss_fn)(state.params)
        grads = lax.pmean(grads, axis_name="batch")
        new_state = state.apply_gradients(grads=grads)
        return new_state
    
    def metrics_step(self, state, batch, u_ref, t, x, progress):
        """
        Conditionally logs corr-training metrics based on config.logging flags.

        Possible outputs (depending on flags):
        - 'alpha': float
        - 'g': float
        - 'teacher_loss': float
        - 'pinns_loss': float
        - 'ic_loss': float
        - 'res_loss': float
        - 'l2_error': float
        - 'corr_term': fig   # image of |wconf * u_corr| (NOT including base_u)
        """
        import matplotlib.pyplot as plt  # keep local if you prefer
        log_dict = {}

        params_corr = state.params
        base_params = state.base_params

        # Only compute what we need
        need_alpha = bool(getattr(self.config.logging, "log_alpha", False))
        need_g = bool(getattr(self.config.logging, "log_g", False))
        need_teacher = bool(getattr(self.config.logging, "log_teacher_loss", False))
        need_pinns = bool(getattr(self.config.logging, "log_pinns_loss", False))
        need_ic_res = bool(getattr(self.config.logging, "log_IC_res_loss", False))
        need_l2 = bool(getattr(self.config.logging, "log_L2error", False))

        # alpha (cheap)
        if need_alpha:
            alpha = self.calc_alpha(jnp.array(progress, dtype=jnp.float32))
            log_dict["alpha"] = alpha
        else:
            alpha = None

        # teacher loss + g share work
        if need_teacher or need_g:
            L_teacher, g = self.get_teacher_loss(params_corr, batch)
            if need_teacher:
                log_dict["teacher_loss"] = L_teacher
            if need_g:
                log_dict["g"] = g

        # pinns loss + ic/res share work
        if need_pinns or need_ic_res:
            L_pinns, losses = self.get_pinns_loss(params_corr, base_params, batch)
            if need_pinns:
                log_dict["pinns_loss"] = L_pinns
            if need_ic_res:
                log_dict["ic_loss"] = losses["ic"]
                log_dict["res_loss"] = losses["res"]

        # L2 error on FULL corrected prediction u_hat = u_base + w*u_corr
        if need_l2:
            # u_hat over grid (t,x). Your original code assumes t and x are 1D grids.
            u_hat = vmap(
                vmap(self.get_solution, in_axes=(None, None, None, 0)),  # inner: x varies
                in_axes=(None, None, 0, None),                           # outer: t varies
            )(params_corr, base_params, t, x)

            l2_error = jnp.linalg.norm(u_hat - u_ref) / (jnp.linalg.norm(u_ref) + 1e-12)
            log_dict["l2_error"] = l2_error

        # Figure: ONLY |wconf * u_corr|
        def corr_term_single(ti, xi):
            inp = jnp.stack([ti, xi])
            u_corr, _g2 = self.corr_model.apply({"params": params_corr}, inp)
            u_corr = u_corr[0]
            w = interp2d_grid(self.t_grid, self.x_grid, self.wconf_map, ti, xi)[0]
            return jnp.abs(w * u_corr)

        corr_term = vmap(
            vmap(corr_term_single, in_axes=(0, None)),
            in_axes=(None, 0),
        )(t, x)

        fig = plt.figure(figsize=(6, 5))
        corr_term_np = jax.device_get(corr_term)
        plt.imshow(corr_term_np, cmap="jet")
        log_dict["corr_term"] = fig
        plt.close(fig)

        return log_dict