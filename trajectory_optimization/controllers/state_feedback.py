# controllers/state_feedback.py
# State Feedback Controller using time-varying LQR gains.
#
# The nominal trajectory (xs_nom, us_nom) is computed offline by iLQR.
# At each step k, we linearize the dynamics around (xs_nom[k], us_nom[k])
# and run a backward Riccati recursion to get time-varying gains K_k.
#
# Control law:  u_k = us_nom[k] + K_k @ (x - xs_nom[k])
#
# Two classes:
#   StateFeedbackController   — time-varying LQR (recomputed each trial)
#   InfiniteHorizonLQRController — fixed gains from linearization at midpoint

import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional

from .base import BaseController


# Helper: backward Riccati recursion

def _compute_lqr_gains(
    dynamics,
    xs_nom: jnp.ndarray,
    us_nom: jnp.ndarray,
    Q:      jnp.ndarray,
    R:      jnp.ndarray,
    Q_terminal: Optional[jnp.ndarray] = None,
):
    """
    Compute time-varying LQR feedback gains via backward Riccati recursion.

    Args:
        dynamics  : discrete-time dynamics callable (x, u, k) -> x_next
        xs_nom    : nominal states   (T+1, n)
        us_nom    : nominal controls (T,   m)
        Q         : state cost matrix         (n x n)
        R         : control cost matrix       (m x m)
        Q_terminal: terminal state cost matrix (n x n), defaults to 10*Q

    Returns:
        Ks : list of T feedback gain matrices, each (m x n)
             Control law: u_k = us_nom[k] + Ks[k] @ (x - xs_nom[k])
    """
    T = us_nom.shape[0]
    n = xs_nom.shape[1]
    m = us_nom.shape[1]

    if Q_terminal is None:
        Q_terminal = 10.0 * Q

    step_range = jnp.arange(T)

    # Linearize dynamics along nominal trajectory
    # f_x[k] = A_k, f_u[k] = B_k
    f_x, f_u = jax.vmap(
        jax.jacobian(dynamics, (0, 1))
    )(xs_nom[:-1], us_nom, step_range)

    # Backward Riccati recursion
    P = Q_terminal  # terminal cost-to-go
    Ks = []

    for k in reversed(range(T)):
        A = np.array(f_x[k])  # (n x n)
        B = np.array(f_u[k])  # (n x m)
        P = np.array(P)

        # Standard discrete-time Riccati step
        # K = (R + B^T P B)^{-1} B^T P A
        S   = R + B.T @ P @ B
        K   = np.linalg.solve(S, B.T @ P @ A)   # (m x n)
        # P update: P = Q + A^T P A - A^T P B K
        P   = Q + A.T @ P @ A - A.T @ P @ B @ K

        Ks.insert(0, jnp.array(K))

    return Ks


# 1. Time-Varying State Feedback (LQR)

class StateFeedbackController(BaseController):
    """
    Time-varying LQR state feedback controller.

    Gains are computed once via backward Riccati recursion along the full
    nominal trajectory before the rollout begins. At each step k:

        u_k = us_nom[k] + K_k @ (x - xs_nom[k])

    Args:
        dynamics    : RK4Integrator (or any discrete dynamics callable)
        Q           : state deviation cost  (5x5)
        R           : control effort cost   (2x2)
        Q_terminal  : terminal state cost   (5x5), default 10*Q
        r_limit     : max |turn rate|
        a_limit     : max |thrust|
    """

    def __init__(
        self,
        dynamics,
        Q:          jnp.ndarray,
        R:          jnp.ndarray,
        Q_terminal: Optional[jnp.ndarray] = None,
        r_limit:    float = 1.0,
        a_limit:    float = 2.0,
    ):
        self.dynamics    = dynamics
        self.Q           = Q
        self.R           = R
        self.Q_terminal  = Q_terminal
        self.r_limit     = r_limit
        self.a_limit     = a_limit

        # Gains are computed in prepare() before rollout
        self._Ks: Optional[list] = None

    def prepare(self, xs_nom: jnp.ndarray, us_nom: jnp.ndarray):
        """
        Compute LQR gains along the nominal trajectory.
        Must be called once before starting a rollout.

        Args:
            xs_nom : nominal trajectory (T+1, 5)
            us_nom : nominal controls   (T,   2)
        """
        print("[StateFeedback] Computing LQR gains...", end=" ")
        self._Ks = _compute_lqr_gains(
            self.dynamics, xs_nom, us_nom,
            self.Q, self.R, self.Q_terminal
        )
        print("done.")

    def reset(self):
        """
        Reset between trials.
        Gains stay valid as long as xs_nom/us_nom don't change.
        Call prepare() again if the nominal trajectory changes.
        """
        pass  # No integrator state to reset for pure state feedback

    def __call__(
        self,
        state:  jnp.ndarray,
        k:      int,
        xs_nom: jnp.ndarray,
        us_nom: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Args:
            state   : current (x, y, θ, v_x, v_y)
            k       : current time step
            xs_nom  : nominal trajectory (T+1, 5)
            us_nom  : nominal controls   (T,   2)
        Returns:
            u = (r, a)
        """
        if self._Ks is None:
            raise RuntimeError(
                "StateFeedbackController: call prepare(xs_nom, us_nom) before rollout."
            )

        # State deviation from nominal
        delta_x = state - xs_nom[k]

        # Feedback correction + nominal feedforward
        K   = self._Ks[k]
        u   = us_nom[k] - K @ delta_x   # note: gain convention u = u_nom + K(-δx)

        # Clip to actuator limits
        r = jnp.clip(u[0], -self.r_limit, self.r_limit)
        a = jnp.clip(u[1], -self.a_limit, self.a_limit)

        return jnp.array([r, a])


# 2. Infinite-Horizon LQR (fixed gains, time-invariant)

class InfiniteHorizonLQRController(BaseController):
    """
    Infinite-horizon LQR with fixed gains computed by solving the discrete
    algebraic Riccati equation (DARE) at a single linearization point
    (midpoint of the nominal trajectory).

    Simpler and faster than time-varying LQR but less accurate far from
    the linearization point. Good baseline comparison.

        u_k = us_nom[k] + K @ (x - xs_nom[k])

    Args:
        dynamics  : RK4Integrator
        Q         : state cost  (5x5)
        R         : control cost (2x2)
        r_limit   : max |turn rate|
        a_limit   : max |thrust|
        maxiter   : max DARE iterations
        tol       : convergence tolerance
    """

    def __init__(
        self,
        dynamics,
        Q:        jnp.ndarray,
        R:        jnp.ndarray,
        r_limit:  float = 1.0,
        a_limit:  float = 2.0,
        maxiter:  int   = 1000,
        tol:      float = 1e-8,
    ):
        self.dynamics = dynamics
        self.Q        = Q
        self.R        = R
        self.r_limit  = r_limit
        self.a_limit  = a_limit
        self.maxiter  = maxiter
        self.tol      = tol
        self._K: Optional[jnp.ndarray] = None

    def _solve_dare(self, A, B):
        """Solve discrete algebraic Riccati equation by value iteration."""
        P = np.array(self.Q)
        for _ in range(self.maxiter):
            S     = self.R + B.T @ P @ B
            K     = np.linalg.solve(S, B.T @ P @ A)
            P_new = self.Q + A.T @ P @ A - A.T @ P @ B @ K
            if np.max(np.abs(P_new - P)) < self.tol:
                break
            P = P_new
        K = np.linalg.solve(self.R + B.T @ P @ B, B.T @ P @ A)
        return jnp.array(K)

    def prepare(self, xs_nom: jnp.ndarray, us_nom: jnp.ndarray):
        """
        Compute fixed LQR gain K at the midpoint of the nominal trajectory.
        Must be called once before rollout.
        """
        T    = us_nom.shape[0]
        mid  = T // 2
        step = jnp.array([mid])

        print("[InfiniteHorizonLQR] Solving DARE at midpoint...", end=" ")
        A, B = jax.vmap(jax.jacobian(self.dynamics, (0, 1)))(
            xs_nom[mid:mid+1], us_nom[mid:mid+1], step
        )
        A, B = np.array(A[0]), np.array(B[0])
        self._K = self._solve_dare(A, B)
        print("done.")

    def reset(self):
        pass  # No integrator state

    def __call__(
        self,
        state:  jnp.ndarray,
        k:      int,
        xs_nom: jnp.ndarray,
        us_nom: jnp.ndarray,
    ) -> jnp.ndarray:
        if self._K is None:
            raise RuntimeError(
                "InfiniteHorizonLQRController: call prepare(xs_nom, us_nom) before rollout."
            )

        delta_x = state - xs_nom[k]
        u       = us_nom[k] - self._K @ delta_x

        r = jnp.clip(u[0], -self.r_limit, self.r_limit)
        a = jnp.clip(u[1], -self.a_limit, self.a_limit)

        return jnp.array([r, a])