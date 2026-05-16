# =============================================================================
# controllers/base.py
# Abstract base class for all controllers.
# Every controller must implement __call__ with this exact signature so the
# evaluation harness (utils/metrics.py) can treat all controllers identically.
# =============================================================================

from abc import ABC, abstractmethod
import jax.numpy as jnp


class BaseController(ABC):
    """
    Shared interface for all trajectory-tracking controllers.

    All controllers receive:
        state   : current 5-state (x, y, θ, v_x, v_y)
        k       : current time step index
        xs_nom  : full nominal trajectory, shape (T+1, 5)
        us_nom  : full nominal controls,   shape (T,   2)

    All controllers return:
        control : jnp.ndarray of shape (2,) — (r=turn rate, a=thrust)
    """

    @abstractmethod
    def __call__(
        self,
        state:  jnp.ndarray,
        k:      int,
        xs_nom: jnp.ndarray,
        us_nom: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute control input for the current state and time step."""
        ...

    @abstractmethod
    def reset(self):
        """Reset any internal state (integrators, histories) between trials."""
        ...