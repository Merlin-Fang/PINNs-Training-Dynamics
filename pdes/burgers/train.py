from jax import lax, pmap, jit, grad, vmap
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from src.architectures.mlp import MLP
from basemodel import PINNs

def ntk(fn, params, *args):
    J = grad(fn, argnums=0)(params, *args)
    J = ravel_pytree(J)[0]
    return jnp.dot(J, J)

class Burgers(PINNs):
    """
    Burgers Equation: u_t + u * u_x - v * u_xx = 0
    v = 0.01 / π
    Initial condition: u(0, x) = -sin(π * x)
    """
    def __init__(self, config, IC):
        super().__init__(config)
        self.v = 0.01 / jnp.pi
        self.IC = IC # Initial condition, IC[0]: u values, IC[1]: t values, IC[2]: x values

    def get_residual(self, params, t, x):
        u = self.get_solution(params, t, x)
        u_t = grad(self.get_solution, argnums=1)(params, t, x)
        u_x = grad(self.get_solution, argnums=2)(params, t, x)
        u_xx = grad(grad(self.get_solution, argnums=2), argnums=2)(params, t, x)
        residual = u_t + u * u_x - self.v * u_xx
        return residual

    def get_losses(self, params, batch):
        u0_pred = self.get_solution(params, self.IC[1], self.IC[2])
        ic_loss = jnp.mean((u0_pred - self.IC[0])**2)

        res = vmap(self.get_residual, in_axes=(None, 0, 0))(params, batch[:, 0], batch[:, 1])
        res_loss = jnp.mean(res**2)

        losses = {'ic': ic_loss, 'res': res_loss}
        return losses

    def get_diag_ntk(self, params, batch):
        ic_ntk = vmap(ntk, in_axes=(None, None, 0, 0))(self.get_solution, params, self.IC[1], self.IC[2])
        res_ntk = vmap(ntk, in_axes=(None, None, 0, 0))(self.get_residual, params, batch[:, 0], batch[:, 1])

        diag_ntk = {'ic': ic_ntk, 'res': res_ntk}
        return diag_ntk
    
