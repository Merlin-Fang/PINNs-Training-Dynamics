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
from jax.tree_util import tree_map, tree_reduce, tree_leaves
import matplotlib.pyplot as plt

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


# ---------------------------------------------------------------------
# Model / optimizer creation (mirrors basemodel.py)
# ---------------------------------------------------------------------
def create_base_model(config) -> nn.Module:
    """Base model architecture (used with frozen base params)."""
    return MLP(
        hidden_layers=config.model.hidden_layers,
        hidden_size=config.model.hidden_size,
        output_size=config.model.output_size,
        activation=get_activation(config.model.activation),
        weight_fact=config.model.weight_fact,
        periodic_embed=config.model.periodic_embed,
        fourier_embed=config.model.fourier_embed,
    )

def create_corr_model(config) -> nn.Module:
    """Correction model = CorrMLP wrapper around MLP + scalar gamma."""
    return CorrMLP(
        hidden_layers=config.model.hidden_layers,
        hidden_size=config.model.hidden_size,
        output_size=config.model.output_size,
        activation=get_activation(config.model.activation),
        weight_fact=config.model.weight_fact,
        periodic_embed=config.model.periodic_embed,
        fourier_embed=config.model.fourier_embed,
        # gain params (can be moved to config later)
        gain_range=(0.5, 2.0),
    )

def create_optimizer(config):
    """Same optimizer policy as basemodel.py (Adam with exp decay + optional grad accum)."""
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

def create_corr_train_state(config, corr_model: nn.Module):
    """Corr TrainState: params=CorrMLP params (includes gamma), tx=optimizer. No loss-weight logic."""
    dummy = jnp.ones((2,))
    params = corr_model.init(jax.random.PRNGKey(0), dummy)["params"]
    tx = create_optimizer(config)
    state = train_state.TrainState.create(
        apply_fn=corr_model.apply,
        params=params,
        tx=tx,
    )
    return jax_utils.replicate(state)


# ---------------------------------------------------------------------
# CorrPINNs: parallel to PINNs, but:
#  - uses frozen base params in get_solution
#  - interpolates teacher/wconf from grids
#  - total loss is alpha(progress)*teacher + (1-alpha)*physics
# ---------------------------------------------------------------------
class CorrPINNs:
    """
    Corr training base class.

    Intended usage (Option 2):
      - PDE class (e.g. Allen_Cahn) already implements get_residual(...) using self.get_solution(...)
      - Add class Allen_Cahn_Corr(CorrPINNs, Allen_Cahn): pass
        so residual is reused, but solution/loss/step are from CorrPINNs.

    Notes:
      - 'progress' is a scalar in [0,1] provided by the training loop (percentile through training).
      - alpha schedule is computed from progress (can be changed later).
      - Base loss weights are frozen (ic/res) and passed in.
    """

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
        """
        Args:
            config: your existing config object (same as base training).
            IC: tuple (u0, t0, x0) as in PINNs.
            base_params: frozen base checkpoint params (pytree).
            corr_assets: dict containing:
                - 't_grid' (Nt,)
                - 'x_grid' (Nx,)
                - 'teacher_map' (Nt, Nx)
                - 'wconf_map' (Nt, Nx)
            frozen_loss_weights: dict like {'ic': <float>, 'res': <float>} (static reuse)
            alpha_schedule: optional dict controlling alpha(progress). If None, defaults to linear.
        """
        self.config = config

        # Base (frozen) and Corr (trainable) modules
        self.base_model = create_base_model(config)
        self.corr_model = create_corr_model(config)

        # Corr train state (replicated)
        self.state = create_corr_train_state(config, self.corr_model)

        # Frozen base params (replicated by training script OR left host-side if not pmapped)
        self.base_params = base_params

        # Initial condition
        self.IC = IC

        # Corr assets (should be jnp arrays; typically replicated across devices)
        self.t_grid = corr_assets["t_grid"]
        self.x_grid = corr_assets["x_grid"]
        self.teacher_map = corr_assets["teacher_map"]
        self.wconf_map = corr_assets["wconf_map"]

        # Frozen physics weights
        self.frozen_loss_weights = {
            k: jnp.array(float(v)) for k, v in frozen_loss_weights.items()
        }

        # Alpha schedule config
        self.alpha_schedule = alpha_schedule or {"type": "linear"}

    # --------------------------
    # Alpha schedule (mixture coefficient for teacher vs physics loss)
    # --------------------------
    def calc_alpha(self, progress: jnp.ndarray) -> jnp.ndarray:
        """
        Compute alpha from training progress p in [0,1], with decay window [t0, t1].

        Behavior:
        - p <= t0  -> alpha = a0
        - p >= t1  -> alpha = a1
        - t0 < p < t1 -> interpolate from a0 -> a1 using schedule type

        alpha_schedule fields:
        type: "linear" | "cosine" | "sigmoid"
        a0: initial alpha (default 1.0)
        a1: final alpha   (default 0.0)
        t0: start of decay window (default 0.1)
        t1: end of decay window   (default 0.6)

        sigmoid-only:
        k: steepness (default 10.0)
        """

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

        # normalized window coordinate
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

        alpha = jnp.where(
            p <= t0,
            a0,
            jnp.where(p >= t1, a1, a_mid),
        )

        return jnp.clip(alpha, 0.0, 1.0)

    # --------------------------
    # Corrected forward (used by physics)
    # --------------------------
    def get_solution(self, params_corr, t, x):
        """
        Corrected solution:
            u_hat = u_base + wconf(t,x) * u_corr
        """
        inp = jnp.stack([t, x])  # (2,)

        u_base = self.base_model.apply({"params": self.base_params}, inp)[0]
        u_corr, _g = self.corr_model.apply({"params": params_corr}, inp)
        u_corr = u_corr[0]

        w = interp2d_grid(self.t_grid, self.x_grid, self.wconf_map, t, x)[0]
        return u_base + w * u_corr

    @abstractmethod
    def get_residual(self, params, t, x):
        """
        Must be provided by PDE subclass (Allen_Cahn, Burgers, ...).
        Crucially: should call self.get_solution(...) internally as before.
        """
        raise NotImplementedError

    # --------------------------
    # Loss components
    # --------------------------
    def get_losses(self, params_corr, batch):
        """
        Physics losses (IC + residual) computed on corrected solution.
        Mirrors PINNs.get_losses in basemodel.py.
        """
        u0_pred = vmap(self.get_solution, in_axes=(None, 0, 0))(params_corr, self.IC[1], self.IC[2])
        ic_loss = jnp.mean((u0_pred - self.IC[0]) ** 2)

        res = vmap(self.get_residual, in_axes=(None, 0, 0))(params_corr, batch[:, 0], batch[:, 1])
        res_loss = jnp.mean(res ** 2)

        return {"ic": ic_loss, "res": res_loss}

    def get_pinns_loss(self, params_corr, batch):
        """
        Weighted physics loss with *frozen* weights.
        """
        losses = self.get_losses(params_corr, batch)
        weighted = tree_map(lambda l, w: l * w, losses, self.frozen_loss_weights)
        return tree_reduce(lambda a, b: a + b, weighted, initializer=0.0), losses

    def get_teacher_loss(self, params_corr, batch):
        """
        Teacher loss:
            E[ wconf(t,x) * (g*u_corr - teacher)^2 ]
        """
        t = batch[:, 0]
        x = batch[:, 1]

        # interpolate targets/gates at batch points
        teacher = interp2d_grid(self.t_grid, self.x_grid, self.teacher_map, t, x)  # (B,)
        wconf = interp2d_grid(self.t_grid, self.x_grid, self.wconf_map, t, x)      # (B,)

        # corr outputs at batch points
        def corr_forward(ti, xi):
            inp = jnp.stack([ti, xi])
            u_corr, g = self.corr_model.apply({"params": params_corr}, inp)
            return u_corr[0], g

        u_corr, g = vmap(corr_forward, in_axes=(0, 0))(t, x)  # u_corr:(B,), g:(B,)

        # g is scalar, but vmap returns (B,) copies; we can safely take g[0] for logging
        diff = (g * u_corr) - teacher
        return jnp.mean(wconf * (diff ** 2)), g[0]

    # --------------------------
    # Total loss (mixture)
    # --------------------------
    @partial(jit, static_argnums=(0,))
    def get_total_loss(self, params_corr, progress, batch):
        """
        Total corr loss:
            alpha(progress) * L_teacher + (1-alpha) * L_PINNs
        """
        alpha = self.calc_alpha(progress)
        L_teacher, _g = self.get_teacher_loss(params_corr, batch)
        L_pinns, _losses = self.get_pinns_loss(params_corr, batch)
        return alpha * L_teacher + (1.0 - alpha) * L_pinns

    # --------------------------
    # Train step (pmapped)
    # --------------------------
    @partial(
        pmap,
        axis_name="batch",
        static_broadcasted_argnums=(0,),
        in_axes=(0, 0, None),  # state sharded, batch sharded, progress broadcast
    )
    def train_step(self, state, batch, progress):
        """
        One corr update step.
        Args:
            state: replicated/sharded TrainState (corr params only)
            batch: (local_B, 2) per-device batch
            progress: scalar in [0,1] (broadcast to all devices)
        """
        grads = grad(self.get_total_loss)(state.params, progress, batch)
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
    log_dict = {}
    params = state.params

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
        alpha = None  # keep local var defined

    # teacher loss + g share work
    if need_teacher or need_g:
        L_teacher, g = self.get_teacher_loss(params, batch)
        if need_teacher:
            log_dict["teacher_loss"] = L_teacher
        if need_g:
            log_dict["g"] = g

    # pinns loss + ic/res share work
    if need_pinns or need_ic_res:
        L_pinns, losses = self.get_pinns_loss(params, batch)
        if need_pinns:
            log_dict["pinns_loss"] = L_pinns
        if need_ic_res:
            log_dict["ic_loss"] = losses["ic"]
            log_dict["res_loss"] = losses["res"]

    # L2 error on FULL corrected prediction u_hat = u_base + w*u_corr
    if need_l2:
        u_hat = vmap(
            vmap(self.get_solution, in_axes=(None, 0, None)),
            in_axes=(None, None, 0),
        )(params, t, x)

        l2_error = jnp.linalg.norm(u_hat - u_ref) / jnp.linalg.norm(u_ref)
        log_dict["l2_error"] = l2_error

    # Figure: ONLY |wconf * u_corr|
    def corr_term_single(ti, xi):
        inp = jnp.stack([ti, xi])
        u_corr, _g2 = self.corr_model.apply({"params": params}, inp)
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