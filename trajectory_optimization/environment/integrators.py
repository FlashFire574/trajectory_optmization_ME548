import jax
from typing import Callable, NamedTuple

class EulerIntegrator(NamedTuple):
    """Euler (1st-order) discrete-time integrator."""
    ode: Callable
    dt:  float

    @jax.jit
    def __call__(self, x, u, k):
        return x + self.dt * self.ode(x, u)


class RK4Integrator(NamedTuple):
    """4th-order Runge-Kutta discrete-time integrator."""
    ode: Callable
    dt:  float

    @jax.jit
    def __call__(self, x, u, k):
        k1 = self.dt * self.ode(x, u)
        k2 = self.dt * self.ode(x + k1 / 2, u)
        k3 = self.dt * self.ode(x + k2 / 2, u)
        k4 = self.dt * self.ode(x + k3, u)
        return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6