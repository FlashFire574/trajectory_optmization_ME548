import jax.numpy as jnp
from typing import NamedTuple

class ContinuousTimeSpaceshipDynamics(NamedTuple):
    """
    5-state spaceship: (x, y, θ, v_x, v_y)
    Controls: (r=turn rate, a=thrust magnitude)
    """
    def __call__(self, state, control):
        x, y, q, v_x, v_y = state
        r, a = control
        return jnp.array([
            v_x,
            v_y,
            r,
            a * jnp.cos(q),
            a * jnp.sin(q),
        ])


class ContinuousTimeRocketCarDynamics(NamedTuple):
    """
    4-state rocket car: (x, y, θ, v)
    Controls: (r=turn rate, a=acceleration)
    """
    def __call__(self, state, control):
        x, y, q, v = state
        r, a = control
        return jnp.array([
            v * jnp.cos(q),
            v * jnp.sin(q),
            r,
            a,
        ])