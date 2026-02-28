from jax import lax, pmap, jit, grad, vmap
import jax.numpy as jnp

from src.basemodels.corrpinns import CorrPINNs
from src.basemodels.pinns import PINNs

class Burgers(PINNs):
    """
    Burgers Equation: u_t + u * u_x - v * u_xx = 0
    v = 0.01 / π
    Initial condition: u(0, x) = -sin(π * x)
    """
    def __init__(self, config, IC):
        super().__init__(config, IC)
        self.v = 0.01 / jnp.pi

    def get_residual(self, params, t, x):
        u = self.get_solution(params, t, x)
        u_t = grad(self.get_solution, argnums=1)(params, t, x)
        u_x = grad(self.get_solution, argnums=2)(params, t, x)
        u_xx = grad(grad(self.get_solution, argnums=2), argnums=2)(params, t, x)
        residual = u_t + u * u_x - self.v * u_xx
        return residual
    
class Burgers_Corr(CorrPINNs, Burgers):
    """
    Corr-net trainer for Burgers equation.

    Reuses Burgers.get_residual(...) (which calls self.get_solution).
    CorrPINNs supplies self.get_solution (u_base + wconf*u_corr) and corr losses/step.
    """

    def __init__(
        self,
        config,
        IC,
        *,
        base_params,
        corr_assets,
        frozen_loss_weights,
        alpha_schedule=None,
    ):
        # initialize CorrPINNs (sets up base_model, corr_model, state, assets, weights, schedule)
        CorrPINNs.__init__(
            self,
            config,
            IC,
            base_params=base_params,
            corr_assets=corr_assets,
            frozen_loss_weights=frozen_loss_weights,
            alpha_schedule=alpha_schedule,
        )

        self.v = 0.01 / jnp.pi