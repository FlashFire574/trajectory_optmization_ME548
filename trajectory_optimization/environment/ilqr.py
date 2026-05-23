# `NamedTuple`s are used (more accurately, abused) in this code to minimize dependencies;
# much better JAX-compatible choices to fit the archetype of "parameterized function"
# would be `flax.struct.dataclass` or `equinox.Module`.
from typing import Callable, NamedTuple
import functools
import jax
import jax.numpy as jnp
import numpy as np; np.seterr(invalid="ignore")

from environment.asteroid_env import Environment


class LinearDynamics(NamedTuple):
    f_x: jnp.array  # A
    f_u: jnp.array  # B

    def __call__(self, x, u, k=None):
        f_x, f_u = self
        return f_x @ x + f_u @ u if k is None else self[k](x, u)

    def __getitem__(self, key):
        return jax.tree_util.tree_map(lambda x: x[key], self)


class AffinePolicy(NamedTuple):
    l: jnp.array  # l
    l_x: jnp.array  # L

    def __call__(self, x, k=None):
        l, l_x = self
        return l + l_x @ x if k is None else self[k](x)

    def __getitem__(self, key):
        return jax.tree_util.tree_map(lambda x: x[key], self)


class QuadraticCost(NamedTuple):
    c: jnp.array  # c
    c_x: jnp.array  # q
    c_u: jnp.array  # r
    c_xx: jnp.array  # Q
    c_uu: jnp.array  # R
    c_ux: jnp.array  # H.T

    @classmethod
    def from_pure_quadratic(cls, c_xx, c_uu, c_ux):
        return cls(
            jnp.zeros((c_xx.shape[:-2])),
            jnp.zeros(c_xx.shape[:-1]),
            jnp.zeros(c_uu.shape[:-1]),
            c_xx,
            c_uu,
            c_ux,
        )

    def __call__(self, x, u, k=None):
        c, c_x, c_u, c_xx, c_uu, c_ux = self
        return c + c_x @ x + c_u @ u + x @ c_xx @ x / 2 + u @ c_uu @ u / 2 + u @ c_ux @ x if k is None else self[k](x)

    def __getitem__(self, key):
        return jax.tree_util.tree_map(lambda x: x[key], self)


class QuadraticStateCost(NamedTuple):
    v: jnp.array  # p (scalar)
    v_x: jnp.array  # p (vector)
    v_xx: jnp.array  # P

    @classmethod
    def from_pure_quadratic(cls, v_xx):
        return cls(
            jnp.zeros(v_xx.shape[:-2]),
            jnp.zeros(v_xx.shape[:-1]),
            v_xx,
        )

    def __call__(self, x, k=None):
        v, v_x, v_xx = self
        return v + v_x @ x + x @ v_xx @ x / 2 if k is None else self[k](x)

    def __getitem__(self, key):
        return jax.tree_util.tree_map(lambda x: x[key], self)


def rollout_state_feedback_policy(dynamics, policy, x0, step_range, x_nom=None, u_nom=None):

    def scan_fn(x, k):
        u = policy(x, k) if x_nom is None else u_nom[k] + policy(x - x_nom[k], k)
        x1 = dynamics(x, u, k)
        return (x1, (x1, u))

    xs, us = jax.lax.scan(scan_fn, x0, step_range)[1]
    return jnp.concatenate([x0[None], xs]), us


def riccati_step(
    current_step_dynamics: LinearDynamics,
    current_step_cost: QuadraticCost,
    next_state_value: QuadraticStateCost,
):
    f_x, f_u = current_step_dynamics
    c, c_x, c_u, c_xx, c_uu, c_ux = current_step_cost
    v, v_x, v_xx = next_state_value

    q = c + v
    q_x = c_x + f_x.T @ v_x
    q_u = c_u + f_u.T @ v_x
    q_xx = c_xx + f_x.T @ v_xx @ f_x
    q_uu = c_uu + f_u.T @ v_xx @ f_u
    q_ux = c_ux + f_u.T @ v_xx @ f_x

    l = -jnp.linalg.solve(q_uu, q_u)
    l_x = -jnp.linalg.solve(q_uu, q_ux)

    current_state_value = QuadraticStateCost(
        q - l.T @ q_uu @ l / 2,
        q_x - l_x.T @ q_uu @ l,
        q_xx - l_x.T @ q_uu @ l_x,
    )
    current_step_optimal_policy = AffinePolicy(l, l_x)
    return current_state_value, current_step_optimal_policy


def ensure_positive_definite(a, eps=1e-3):
    w, v = jnp.linalg.eigh(a)
    return (v * jnp.maximum(w, eps)) @ v.T


class TotalCost(NamedTuple):
    running_cost: Callable
    terminal_cost: Callable

    def __call__(self, xs, us):
        step_range = jnp.arange(us.shape[0])
        return jnp.sum(jax.vmap(self.running_cost)(xs[:-1], us, step_range)) + self.terminal_cost(xs[-1])

jax.tree_util.register_pytree_node(
    TotalCost,
    lambda tc: ((tc.running_cost, tc.terminal_cost), None),
    lambda aux, children: TotalCost(*children),
)


class EulerIntegrator(NamedTuple):
    """Discrete time dynamics from time-invariant continuous time dynamics using the Euler method."""
    ode: Callable
    dt: float

    @jax.jit
    def __call__(self, x, u, k):
        return x + self.dt * self.ode(x, u)


class RK4Integrator(NamedTuple):
    """Discrete time dynamics from time-invariant continuous time dynamics using a 4th order Runge-Kutta method."""
    ode: Callable
    dt: float

    @jax.jit
    def __call__(self, x, u, k):
        k1 = self.dt * self.ode(x, u)
        k2 = self.dt * self.ode(x + k1 / 2, u)
        k3 = self.dt * self.ode(x + k2 / 2, u)
        k4 = self.dt * self.ode(x + k3, u)
        return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6


@functools.partial(jax.jit, static_argnames=["dynamics", "maxiter"])
def iterative_linear_quadratic_regulator(dynamics, total_cost, x0, u_guess, maxiter=100, atol=1e-3):
    running_cost, terminal_cost = total_cost
    n, (N, m) = x0.shape[-1], u_guess.shape
    step_range = jnp.arange(N)

    xs, us = rollout_state_feedback_policy(dynamics, lambda x, k: u_guess[k], x0, step_range)
    j = total_cost(xs, us)

    def continuation_criterion(loop_vars):
        i, _, _, j_curr, j_prev = loop_vars
        return (j_curr < j_prev - atol) & (i < maxiter)

    def ilqr_iteration(loop_vars):
        i, xs, us, j_curr, j_prev = loop_vars

        f_x, f_u = jax.vmap(jax.jacobian(dynamics, (0, 1)))(xs[:-1], us, step_range)
        c = jax.vmap(running_cost)(xs[:-1], us, step_range)
        c_x, c_u = jax.vmap(jax.grad(running_cost, (0, 1)))(xs[:-1], us, step_range)
        (c_xx, c_xu), (c_ux, c_uu) = jax.vmap(jax.hessian(running_cost, (0, 1)))(xs[:-1], us, step_range)
        v, v_x, v_xx = terminal_cost(xs[-1]), jax.grad(terminal_cost)(xs[-1]), jax.hessian(terminal_cost)(xs[-1])

        # Ensure quadratic cost terms are positive definite.
        c_zz = jnp.block([[c_xx, c_xu], [c_ux, c_uu]])
        c_zz = jax.vmap(ensure_positive_definite)(c_zz)
        c_xx, c_uu, c_ux = c_zz[:, :n, :n], c_zz[:, -m:, -m:], c_zz[:, -m:, :n]
        v_xx = ensure_positive_definite(v_xx)

        linearized_dynamics = LinearDynamics(f_x, f_u)
        quadratized_running_cost = QuadraticCost(c, c_x, c_u, c_xx, c_uu, c_ux)
        quadratized_terminal_cost = QuadraticStateCost(v, v_x, v_xx)

        def scan_fn(next_state_value, current_step_dynamics_cost):
            current_step_dynamics, current_step_cost = current_step_dynamics_cost
            current_state_value, current_step_policy = riccati_step(
                current_step_dynamics,
                current_step_cost,
                next_state_value,
            )
            return current_state_value, current_step_policy

        policy = jax.lax.scan(scan_fn,
                              quadratized_terminal_cost, (linearized_dynamics, quadratized_running_cost),
                              reverse=True)[1]

        def rollout_linesearch_policy(alpha):
            # Note that we roll out the true `dynamics`, not the `linearized_dynamics`!
            l, l_x = policy
            return rollout_state_feedback_policy(dynamics, AffinePolicy(alpha * l, l_x), x0, step_range, xs, us)

        # Backtracking line search (step sizes evaluated in parallel).
        all_xs, all_us = jax.vmap(rollout_linesearch_policy)(0.5**jnp.arange(16))
        js = jax.vmap(total_cost)(all_xs, all_us)
        a = jnp.argmin(js)
        j = js[a]
        xs = jnp.where(j < j_curr, all_xs[a], xs)
        us = jnp.where(j < j_curr, all_us[a], us)
        return i + 1, xs, us, jnp.minimum(j, j_curr), j_curr

    i, xs, us, j, _ = jax.lax.while_loop(continuation_criterion, ilqr_iteration, (0, xs, us, j, jnp.inf))

    return {
        "optimal_trajectory": (xs, us),
        "optimal_cost": j,
        "num_iterations": i,
    }

class RunningCost(NamedTuple):
    env: Environment
    dt:  jnp.array

    def __call__(self, state, control, step):
        asteroids = self.env.asteroids.at_time(step * self.dt)
        separation_distance = jnp.where(
            jnp.isnan(asteroids.radius), np.inf,
            jnp.linalg.norm(
                self.env.wrap_vector(state[:2] - asteroids.center), axis=-1
            ) - asteroids.radius - self.env.ship_radius
        )
        collision_avoidance_penalty = jnp.sum(
            jnp.where(separation_distance > 0.3, 0,
                      1e4 * (0.3 - separation_distance) ** 2)
        )
        r, a = control
        yaw_rate_penalty     = 1e4 * jnp.maximum(jnp.abs(r) - np.pi / 2, 0) ** 2
        acceleration_penalty = 1e4 * jnp.maximum(jnp.abs(a) - 4, 0) ** 2
        return collision_avoidance_penalty + yaw_rate_penalty + acceleration_penalty + r**2 + a**2


class FullHorizonTerminalCost(NamedTuple):
    env:           Environment
    goal_position: jnp.array

    @classmethod
    def create_ignoring_extra_args(cls, env, goal_position, *args, **kwargs):
        return cls(env, goal_position)

    def __call__(self, state):
        return 1000.0 * (
            jnp.sum(jnp.square(state[:2] - self.goal_position))
            + state[3] ** 2
            + state[4] ** 2
        )


def compute_nominal_trajectory(dynamics, env, start_state, goal_position, T, verbose=True):
    u_init    = np.zeros((T, 2))
    empty_env = Environment.create(0)

    if verbose:
        print("Stage 1: empty environment...", end=" ", flush=True)
    stage1 = iterative_linear_quadratic_regulator(
        dynamics,
        TotalCost(RunningCost(empty_env, dynamics.dt),
                  FullHorizonTerminalCost(empty_env, jnp.array(goal_position))),
        jnp.array(start_state), jnp.array(u_init),
    )
    if verbose:
        print(f"done ({stage1['num_iterations']} iters)")

    if verbose:
        print("Stage 2: full asteroid environment...", end=" ", flush=True)
    stage2 = iterative_linear_quadratic_regulator(
        dynamics,
        TotalCost(RunningCost(env, dynamics.dt),
                  FullHorizonTerminalCost(env, jnp.array(goal_position))),
        jnp.array(start_state), stage1["optimal_trajectory"][1],
    )
    if verbose:
        print(f"done ({stage2['num_iterations']} iters, cost={stage2['optimal_cost']:.2f})")

    return jax.tree_util.tree_map(np.array, stage2)