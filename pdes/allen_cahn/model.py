from jax import grad
import jax.numpy as jnp

from basemodels.pinns import PINNs
from basemodels.corrpinns import CorrPINNs

class Allen_Cahn(PINNs):
    """
    Allen–Cahn Equation: u_t = a*u + v*u_xx - a*u^3
    a = 5
    v = 0.0001
    """
    def __init__(self, config, IC):
        super().__init__(config, IC)
        self.a = 5.0
        self.v = 1e-4

    def get_residual(self, params, t, x):
        u = self.get_solution(params, t, x)
        u_t = grad(self.get_solution, argnums=1)(params, t, x)
        u_xx = grad(grad(self.get_solution, argnums=2), argnums=2)(params, t, x)

        residual = u_t - self.a * u - self.v * u_xx + self.a * (u ** 3)
        return residual

class Allen_Cahn_Corr(CorrPINNs, Allen_Cahn):
    """
    Corr-net trainer for Allen–Cahn.

    Reuses Allen_Cahn.get_residual(...) (which calls self.get_solution).
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

        # Allen–Cahn constants (same as Allen_Cahn.__init__)
        self.a = 5.0
        self.v = 1e-4