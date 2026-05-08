# =============================================================================
# trajectory_optimization.py
# Trajectory Optimization in Uncertain Environments
# Author: Aditya Joshi  |  Student ID: 2530019
# =============================================================================
# Controllers implemented / scaffolded:
#   1. Nominal iLQR       — baseline optimal trajectory (no obstacles)
#   2. PID                — TODO
#   3. State Feedback     — TODO
#   4. CBF-CLF-QP         — TODO (paste your code in Section 5)
#   [5. LQR extension     — TODO]
#   [6. MPPI extension    — TODO]
# =============================================================================

# ── Imports ──────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.collections
import matplotlib.transforms
import jax
import jax.numpy as jnp
from typing import Callable, NamedTuple

# Optional: suppress JAX GPU/TPU warnings if running on CPU
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"


# =============================================================================
# SECTION 1 — DYNAMICS
# =============================================================================

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
    Kept for reference / comparison — project uses Spaceship dynamics.
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


# =============================================================================
# SECTION 2 — ENVIRONMENT  (asteroids + sensing + rendering)
# =============================================================================

class Asteroid(NamedTuple):
    center:   jnp.array
    radius:   jnp.array
    velocity: jnp.array = 0

    def at_time(self, time):
        return self._replace(center=self.center + self.velocity * time)


class Environment(NamedTuple):
    asteroids:      Asteroid
    ship_radius:    jnp.array
    sensing_radius: jnp.array
    bounds:         jnp.array

    @classmethod
    def create(cls, num_asteroids, ship_radius=1.0, sensing_radius=5, bounds=(50, 40)):
        bounds = np.array(bounds)
        return cls(
            Asteroid(
                np.random.rand(num_asteroids, 2) * bounds,
                1 + 2 * np.random.rand(num_asteroids),
                np.random.randn(num_asteroids, 2),
            ),
            ship_radius,
            sensing_radius,
            bounds,
        )

    def at_time(self, time):
        return self._replace(asteroids=self.asteroids.at_time(time))

    def wrap_vector(self, vector):
        return (vector + self.bounds / 2) % self.bounds - self.bounds / 2

    def sense(self, position):
        """Return environment with unseen asteroid radii set to NaN."""
        deltas = self.wrap_vector(position - self.asteroids.center)
        return self._replace(
            asteroids=self.asteroids._replace(
                radius=jnp.where(
                    jnp.linalg.norm(deltas, axis=-1) - self.asteroids.radius
                    < self.sensing_radius,
                    self.asteroids.radius,
                    np.nan,
                )
            )
        )

    def plot(self, state=None, plan=None, history=None,
             sensor=False, scaled_thrust=None, ax=None):
        pose         = np.full(3, np.nan) if state is None else state[:3]
        plan         = np.full((0, 2), np.nan) if plan is None else plan[:, :2]
        history      = np.full((0, 2), np.nan) if history is None else history[:, :2]
        scaled_thrust = np.full((), np.nan) if scaled_thrust is None else scaled_thrust

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.set_xlim(0, self.bounds[0])
            ax.set_ylim(0, self.bounds[1])
            ax.set_aspect(1)
            asteroids = ax.add_collection(
                matplotlib.collections.PatchCollection(
                    [plt.Circle(np.zeros(2), r) for r in self.asteroids.radius] * 4,
                    offsets=np.zeros(2),
                    transOffset=matplotlib.transforms.AffineDeltaTransform(ax.transData),
                    color="black",
                ))
            ship = ax.add_collection(
                matplotlib.collections.PatchCollection(
                    [plt.Polygon(self.ship_radius * np.array([[-.2, -.4], [1., 0], [-.2, .4]]))] * 4,
                    offsets=np.zeros(2),
                    transOffset=matplotlib.transforms.AffineDeltaTransform(ax.transData),
                    color="orange",
                    zorder=10,
                ))
            circle = ax.add_collection(
                matplotlib.collections.PatchCollection(
                    [plt.Circle(np.zeros(2), self.sensing_radius)] * 4,
                    offsets=np.zeros(2),
                    transOffset=matplotlib.transforms.AffineDeltaTransform(ax.transData),
                    facecolor=(0, 0, 0, 0),
                    edgecolor="black",
                    linestyle="--",
                    zorder=10,
                ))
            thruster = ax.add_collection(
                matplotlib.collections.PatchCollection(
                    [plt.Polygon(self.ship_radius * np.array([[-1., 0.], [0., -.25], [0., .25]]))] * 4,
                    offsets=np.zeros(2),
                    transOffset=matplotlib.transforms.AffineDeltaTransform(ax.transData),
                    color="red",
                    zorder=5,
                ))
            plan_line    = ax.plot(plan[:, 0],    plan[:, 1],    color="green")[0]
            history_line = ax.plot(history[:, 0], history[:, 1], color="blue")[0]
        else:
            fig = ax.figure
            asteroids, ship, circle, thruster = ax.collections
            plan_line, history_line = ax.lines

        screen_offsets = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        asteroids.set_offsets(
            (self.wrap_vector(self.asteroids.center)
             + self.bounds * screen_offsets[:, None, :]).reshape(-1, 2)
        )
        if sensor:
            asteroids.set_alpha(
                np.where(np.isnan(self.sense(pose[:2]).asteroids.radius), 0.1, 1.0)
            )
        ship.set_offsets(self.wrap_vector(pose[:2]) + self.bounds * screen_offsets)
        ship.set_transform(
            matplotlib.transforms.Affine2D().rotate(pose[2]) + ax.transData
        )
        circle.set_offsets(
            self.wrap_vector(pose[:2] if sensor else np.full(2, np.nan))
            + self.bounds * screen_offsets
        )
        thruster.set_offsets(self.wrap_vector(pose[:2]) + self.bounds * screen_offsets)
        thruster.set_transform(
            matplotlib.transforms.Affine2D()
            .scale(0.2 + 0.8 * scaled_thrust, 1)
            .rotate(pose[2])
            + ax.transData
        )

        def tile_line(line):
            if line.shape[0] == 0:
                return line
            irange, jrange = [
                range(int(x[0]), int(x[1] + 1))
                for x in zip(
                    np.min(line, 0) // self.bounds,
                    np.max(line, 0) // self.bounds,
                )
            ]
            return np.concatenate([
                np.pad(
                    line - np.array([i, j]) * self.bounds,
                    ((0, 1), (0, 0)),
                    constant_values=np.nan,
                )
                for i in irange
                for j in jrange
            ], 0)

        plan_line.set_data(*tile_line(plan).T)
        history_line.set_data(*tile_line(history).T)
        return fig, ax


# =============================================================================
# SECTION 3 — LQR / iLQR INFRASTRUCTURE
# =============================================================================

class LinearDynamics(NamedTuple):
    f_x: jnp.array  # A matrix
    f_u: jnp.array  # B matrix

    def __call__(self, x, u, k=None):
        f_x, f_u = self
        return f_x @ x + f_u @ u if k is None else self[k](x, u)

    def __getitem__(self, key):
        return jax.tree_util.tree_map(lambda x: x[key], self)


class AffinePolicy(NamedTuple):
    l:   jnp.array  # feedforward
    l_x: jnp.array  # feedback gain L

    def __call__(self, x, k=None):
        l, l_x = self
        return l + l_x @ x if k is None else self[k](x)

    def __getitem__(self, key):
        return jax.tree_util.tree_map(lambda x: x[key], self)


class QuadraticCost(NamedTuple):
    c:    jnp.array  # scalar offset
    c_x:  jnp.array  # q
    c_u:  jnp.array  # r
    c_xx: jnp.array  # Q
    c_uu: jnp.array  # R
    c_ux: jnp.array  # H^T

    @classmethod
    def from_pure_quadratic(cls, c_xx, c_uu, c_ux):
        return cls(
            jnp.zeros(c_xx.shape[:-2]),
            jnp.zeros(c_xx.shape[:-1]),
            jnp.zeros(c_uu.shape[:-1]),
            c_xx, c_uu, c_ux,
        )

    def __call__(self, x, u, k=None):
        c, c_x, c_u, c_xx, c_uu, c_ux = self
        return (c + c_x @ x + c_u @ u
                + x @ c_xx @ x / 2
                + u @ c_uu @ u / 2
                + u @ c_ux @ x) if k is None else self[k](x)

    def __getitem__(self, key):
        return jax.tree_util.tree_map(lambda x: x[key], self)


class QuadraticStateCost(NamedTuple):
    v:    jnp.array  # scalar
    v_x:  jnp.array  # gradient
    v_xx: jnp.array  # Hessian P

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


class TotalCost(NamedTuple):
    running_cost:  Callable
    terminal_cost: Callable

    def __call__(self, xs, us):
        step_range = jnp.arange(us.shape[0])
        return (
            jnp.sum(jax.vmap(self.running_cost)(xs[:-1], us, step_range))
            + self.terminal_cost(xs[-1])
        )


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


def ensure_positive_definite(a, eps=1e-3):
    w, v = jnp.linalg.eigh(a)
    return (v * jnp.maximum(w, eps)) @ v.T


def rollout_state_feedback_policy(dynamics, policy, x0, step_range,
                                   x_nom=None, u_nom=None):
    def scan_fn(x, k):
        u = (policy(x, k) if x_nom is None
             else u_nom[k] + policy(x - x_nom[k], k))
        x1 = dynamics(x, u, k)
        return x1, (x1, u)

    xs, us = jax.lax.scan(scan_fn, x0, step_range)[1]
    return jnp.concatenate([x0[None], xs]), us


def riccati_step(current_step_dynamics: LinearDynamics,
                 current_step_cost: QuadraticCost,
                 next_state_value: QuadraticStateCost):
    f_x, f_u = current_step_dynamics
    c, c_x, c_u, c_xx, c_uu, c_ux = current_step_cost
    v, v_x, v_xx = next_state_value

    q    = c + v
    q_x  = c_x  + f_x.T @ v_x
    q_u  = c_u  + f_u.T @ v_x
    q_xx = c_xx + f_x.T @ v_xx @ f_x
    q_uu = c_uu + f_u.T @ v_xx @ f_u
    q_ux = c_ux + f_u.T @ v_xx @ f_x

    l   = -jnp.linalg.solve(q_uu, q_u)
    l_x = -jnp.linalg.solve(q_uu, q_ux)

    current_state_value = QuadraticStateCost(
        q - l.T @ q_uu @ l / 2,
        q_x - l_x.T @ q_uu @ l,
        q_xx - l_x.T @ q_uu @ l_x,
    )
    return current_state_value, AffinePolicy(l, l_x)


@jax.jit
def iterative_linear_quadratic_regulator(dynamics, total_cost, x0,
                                          u_guess, maxiter=100, atol=1e-3):
    running_cost, terminal_cost = total_cost
    n, (N, m) = x0.shape[-1], u_guess.shape
    step_range = jnp.arange(N)

    xs, us = rollout_state_feedback_policy(
        dynamics, lambda x, k: u_guess[k], x0, step_range
    )
    j = total_cost(xs, us)

    def continuation_criterion(loop_vars):
        i, _, _, j_curr, j_prev = loop_vars
        return (j_curr < j_prev - atol) & (i < maxiter)

    def ilqr_iteration(loop_vars):
        i, xs, us, j_curr, j_prev = loop_vars

        f_x, f_u = jax.vmap(jax.jacobian(dynamics, (0, 1)))(xs[:-1], us, step_range)
        c               = jax.vmap(running_cost)(xs[:-1], us, step_range)
        c_x, c_u        = jax.vmap(jax.grad(running_cost, (0, 1)))(xs[:-1], us, step_range)
        (c_xx, c_xu), (c_ux, c_uu) = jax.vmap(
            jax.hessian(running_cost, (0, 1))
        )(xs[:-1], us, step_range)
        v    = terminal_cost(xs[-1])
        v_x  = jax.grad(terminal_cost)(xs[-1])
        v_xx = jax.hessian(terminal_cost)(xs[-1])

        c_zz = jnp.block([[c_xx, c_xu], [c_ux, c_uu]])
        c_zz = jax.vmap(ensure_positive_definite)(c_zz)
        c_xx, c_uu, c_ux = c_zz[:, :n, :n], c_zz[:, -m:, -m:], c_zz[:, -m:, :n]
        v_xx = ensure_positive_definite(v_xx)

        lin_dyn  = LinearDynamics(f_x, f_u)
        quad_run = QuadraticCost(c, c_x, c_u, c_xx, c_uu, c_ux)
        quad_ter = QuadraticStateCost(v, v_x, v_xx)

        def scan_fn(next_val, step_dc):
            cur_val, cur_policy = riccati_step(*step_dc, next_val)
            return cur_val, cur_policy

        policy = jax.lax.scan(scan_fn, quad_ter,
                               (lin_dyn, quad_run), reverse=True)[1]

        def rollout_linesearch(alpha):
            l, l_x = policy
            return rollout_state_feedback_policy(
                dynamics, AffinePolicy(alpha * l, l_x), x0, step_range, xs, us
            )

        all_xs, all_us = jax.vmap(rollout_linesearch)(0.5 ** jnp.arange(16))
        js = jax.vmap(total_cost)(all_xs, all_us)
        a  = jnp.argmin(js)
        j  = js[a]
        xs = jnp.where(j < j_curr, all_xs[a], xs)
        us = jnp.where(j < j_curr, all_us[a], us)
        return i + 1, xs, us, jnp.minimum(j, j_curr), j_curr

    i, xs, us, j, _ = jax.lax.while_loop(
        continuation_criterion, ilqr_iteration, (0, xs, us, j, jnp.inf)
    )
    return {"optimal_trajectory": (xs, us), "optimal_cost": j, "num_iterations": i}


# =============================================================================
# SECTION 4 — PROBLEM SETUP  (shared by all controllers)
# =============================================================================

# Time horizon & integrator
T  = 125          # number of time steps
dt = 0.1          # seconds per step

# Use Spaceship dynamics (5-state) with RK4
dynamics = RK4Integrator(ContinuousTimeSpaceshipDynamics(), dt)

# Spaceship state: (x, y, θ, v_x, v_y)
start_state   = np.array([5.,  5.,  0., 0., 0.])   # added v_y=0 for 5-state
goal_position = np.array([45., 35.])

# Random environment
np.random.seed(2)
env = Environment.create(20)

# ── Cost function weights ──────────────────────────────────────────────────
# Objective:  Σ [ α‖x_t - x_nom‖² + β‖u_t‖² ]  +  terminal penalty
alpha = 1.0    # state deviation weight
beta  = 0.1    # control effort weight

STATE_DIM   = 5   # (x, y, θ, v_x, v_y)
CONTROL_DIM = 2   # (r, a)

# Q and R matrices (used by iLQR nominal and State Feedback)
Q = alpha * jnp.eye(STATE_DIM)
R = beta  * jnp.eye(CONTROL_DIM)

# ── Nominal terminal cost (reach goal_position) ───────────────────────────
def terminal_cost(x):
    """Penalize distance from goal position (x,y only)."""
    return 1000.0 * jnp.sum((x[:2] - goal_position) ** 2)

# ── Nominal running cost (minimize control effort + stay near nominal) ────
def running_cost(x, u, k):
    """Pure control effort for nominal planning (no obstacles)."""
    return beta * jnp.sum(u ** 2)

nominal_total_cost = TotalCost(running_cost, terminal_cost)
u_guess = jnp.zeros((T, CONTROL_DIM))


# =============================================================================
# SECTION 5 — CONTROLLER 1: NOMINAL iLQR  (fuel-optimal, no obstacles)
# =============================================================================

def run_nominal_ilqr():
    """
    Compute the nominal fuel-optimal trajectory from start to goal
    with no obstacle awareness. This is the reference trajectory
    all other controllers try to track.
    """
    result = iterative_linear_quadratic_regulator(
        dynamics, nominal_total_cost, start_state, u_guess
    )
    xs_nom, us_nom = result["optimal_trajectory"]
    print(f"[iLQR Nominal] cost={result['optimal_cost']:.4f},"
          f" iters={result['num_iterations']}")
    return xs_nom, us_nom


# =============================================================================
# SECTION 6 — CONTROLLER 2: PID
# =============================================================================

class PIDController:
    """
    Proportional-Integral-Derivative controller for trajectory tracking.

    Tracks the nominal trajectory (xs_nom) by computing errors in
    position (x, y) and feeding them through PID gains to produce
    turn-rate (r) and thrust (a) commands.
    """

    def __init__(self, Kp_pos, Ki_pos, Kd_pos,
                 Kp_ang=1.0, dt=0.1):
        # TODO: store gains and initialize integrator / derivative state
        self.Kp_pos = Kp_pos
        self.Ki_pos = Ki_pos
        self.Kd_pos = Kd_pos
        self.Kp_ang = Kp_ang
        self.dt     = dt

        # TODO: initialize integral and previous-error accumulators
        self.integral_error = np.zeros(2)
        self.prev_error     = np.zeros(2)

    def reset(self):
        # TODO: reset integrator and derivative state between trials
        self.integral_error = np.zeros(2)
        self.prev_error     = np.zeros(2)

    def __call__(self, state, k, xs_nom):
        """
        Args:
            state  : current 5-state (x, y, θ, v_x, v_y)
            k      : current time step index
            xs_nom : nominal trajectory array, shape (T+1, 5)
        Returns:
            control : (r, a) — turn rate and thrust
        """
        # TODO: compute position error relative to nominal waypoint xs_nom[k]
        # TODO: update integral term (clamp to avoid windup)
        # TODO: compute derivative term
        # TODO: map position error → desired heading → heading error → r
        # TODO: map speed error → a
        # HINT: desired heading = atan2(error_y, error_x)
        raise NotImplementedError("PID controller not yet implemented.")


def run_pid(xs_nom, us_nom, env):
    """
    Roll out the PID controller in the live environment with obstacle sensing.

    Args:
        xs_nom : nominal trajectory (T+1, 5)
        us_nom : nominal controls  (T,   2)
        env    : Environment at t=0
    Returns:
        history : array of visited states (T+1, 5)
        controls : array of applied controls (T, 2)
        success  : bool — reached goal without collision
    """
    # TODO: implement rollout loop
    #   for k in range(T):
    #       env_k  = env.at_time(k * dt)           # move asteroids
    #       u_k    = pid(state, k, xs_nom)          # compute control
    #       state  = dynamics(state, u_k, k)        # step dynamics
    #       check collision with env_k.asteroids    # success/fail flag
    raise NotImplementedError("PID rollout not yet implemented.")


# =============================================================================
# SECTION 7 — CONTROLLER 3: STATE FEEDBACK (LQR gains)
# =============================================================================

def compute_lqr_gains(xs_nom, us_nom):
    """
    Linearize dynamics along nominal trajectory and compute time-varying
    LQR feedback gains via backward Riccati recursion.

    Returns:
        gains : list of (K_k) feedback matrices, length T
                u = us_nom[k] + K_k @ (x - xs_nom[k])
    """
    # TODO: linearize dynamics at each (xs_nom[k], us_nom[k]) using jax.jacobian
    # TODO: run riccati_step backward from terminal cost
    # TODO: extract l_x (K_k) from AffinePolicy at each step
    raise NotImplementedError("LQR gain computation not yet implemented.")


def run_state_feedback(xs_nom, us_nom, env):
    """
    Roll out the state-feedback (LQR) controller in the live environment.

    Args:
        xs_nom : nominal trajectory (T+1, 5)
        us_nom : nominal controls  (T,   2)
        env    : Environment at t=0
    Returns:
        history  : array of visited states (T+1, 5)
        controls : array of applied controls (T, 2)
        success  : bool
    """
    # TODO: call compute_lqr_gains(xs_nom, us_nom)
    # TODO: rollout loop analogous to run_pid but using u = us_nom[k] + K_k @ δx
    raise NotImplementedError("State feedback rollout not yet implemented.")


# =============================================================================
# SECTION 8 — CONTROLLER 4: CBF-CLF-QP  (safety-critical)
# =============================================================================
#
# Paste your CBF-CLF-QP implementation here.
#
# Expected interface:
#   cbf_clf_qp(state, k, xs_nom, us_nom, env_k) -> control (r, a)
#
# The QP should solve:
#   min_{u, δ}   ‖u - u_nom‖² + λ δ²
#   s.t.  CLF:   V̇(x,u) ≤ -γ V(x) + δ        (relaxed Lyapunov decrease)
#         CBF:   ḣ_i(x,u) ≥ -α h_i(x)         for each visible asteroid i
#         bounds: u_min ≤ u ≤ u_max
#
# Barrier function suggestion:
#   h_i(x) = ‖x[:2] - c_i‖² - (r_i + ship_radius)²
#
# Lyapunov function suggestion:
#   V(x) = ‖x[:2] - x_nom[:2]‖² + ‖x[3:] - x_nom[3:]‖²

class CBFCLFQPController:
    """
    Control Barrier Function + Control Lyapunov Function QP controller.
    Guarantees obstacle avoidance (CBF) while driving toward the nominal
    trajectory (CLF), solved as a quadratic program at each step.
    """

    def __init__(self, alpha_cbf=1.0, gamma_clf=1.0, lam=10.0,
                 u_min=(-1.0, -2.0), u_max=(1.0, 2.0)):
        # TODO: store CBF/CLF parameters
        # TODO: initialize CVXPY problem (warm-start across steps)
        self.alpha_cbf = alpha_cbf
        self.gamma_clf = gamma_clf
        self.lam       = lam
        self.u_min     = np.array(u_min)
        self.u_max     = np.array(u_max)

    def __call__(self, state, k, xs_nom, us_nom, env_k):
        """
        Args:
            state  : current 5-state
            k      : time step
            xs_nom : nominal trajectory
            us_nom : nominal controls
            env_k  : Environment at current time (for obstacle positions)
        Returns:
            u : safe control (r, a)
        """
        # TODO: sense nearby asteroids from env_k
        # TODO: build CBF constraints for each visible asteroid
        # TODO: build CLF constraint relative to nominal state xs_nom[k]
        # TODO: set up and solve CVXPY QP
        # TODO: return optimal u; fall back to u_nom[k] if QP is infeasible
        raise NotImplementedError("CBF-CLF-QP controller not yet implemented.")


def run_cbf_clf_qp(xs_nom, us_nom, env):
    """
    Roll out the CBF-CLF-QP controller in the live environment.

    Returns:
        history  : (T+1, 5)
        controls : (T, 2)
        success  : bool
    """
    # TODO: analogous rollout loop to run_pid / run_state_feedback
    raise NotImplementedError("CBF-CLF-QP rollout not yet implemented.")


# =============================================================================
# SECTION 9 — EXTENSIONS (time permitting)
# =============================================================================

# ── LQR (infinite-horizon, time-invariant gains) ──────────────────────────
def run_lqr_extension(xs_nom, us_nom, env):
    """
    TODO: compute infinite-horizon LQR gains around the midpoint of the
          nominal trajectory, then roll out with fixed K.
    """
    raise NotImplementedError("LQR extension not yet implemented.")


# ── MPPI (Model Predictive Path Integral) ─────────────────────────────────
def run_mppi_extension(xs_nom, env):
    """
    TODO: implement MPPI sampling-based MPC.
    Reference: Williams et al. (2017); Wang et al. (2025) MPPI-DB.
    Key hyperparameters: num_samples, horizon, temperature λ, noise σ.
    """
    raise NotImplementedError("MPPI extension not yet implemented.")


# =============================================================================
# SECTION 10 — EVALUATION & BENCHMARKING
# =============================================================================

def evaluate_controller(rollout_fn, xs_nom, us_nom,
                         num_trials=20, num_asteroids=20,
                         density_label="medium"):
    """
    Run `rollout_fn` over `num_trials` randomized environments and
    collect performance metrics.

    Args:
        rollout_fn  : callable(xs_nom, us_nom, env) -> (history, controls, success)
        xs_nom      : nominal trajectory
        us_nom      : nominal controls
        num_trials  : number of random seeds to evaluate
        num_asteroids: asteroid count (obstacle density)
        density_label: string tag for reporting

    Returns:
        metrics dict with keys:
            tracking_error  — mean L2 deviation from nominal (x,y)
            control_effort  — mean ‖u‖² (proxy for fuel)
            traversal_time  — steps to reach goal (or T if failed)
            success_rate    — fraction of collision-free trials
    """
    # TODO: loop over seeds, instantiate Environment.create(num_asteroids),
    #       call rollout_fn, compute per-trial metrics, aggregate stats.
    raise NotImplementedError("Evaluation harness not yet implemented.")


# =============================================================================
# SECTION 11 — MAIN
# =============================================================================

if __name__ == "__main__":

    # ── Step 1: compute nominal trajectory ──────────────────────────────────
    print("=" * 60)
    print("Step 1: Computing nominal iLQR trajectory...")
    xs_nom, us_nom = run_nominal_ilqr()
    print(f"  Start : {xs_nom[0]}")
    print(f"  Goal  : {xs_nom[-1, :2]}  (target: {goal_position})")

    # ── Step 2: visualize nominal trajectory ────────────────────────────────
    print("\nStep 2: Plotting nominal trajectory on environment...")
    fig, ax = env.plot(state=start_state, plan=np.array(xs_nom))
    plt.title("Nominal iLQR Trajectory (no obstacles avoided)")
    plt.savefig("nominal_trajectory.png", dpi=150, bbox_inches="tight")
    print("  Saved → nominal_trajectory.png")
    plt.show()

    # ── Step 3: run controllers (uncomment as you implement each) ───────────
    print("\nStep 3: Running controllers...")

    # history_pid, ctrls_pid, ok_pid = run_pid(xs_nom, us_nom, env)
    # history_sf,  ctrls_sf,  ok_sf  = run_state_feedback(xs_nom, us_nom, env)
    # history_cbf, ctrls_cbf, ok_cbf = run_cbf_clf_qp(xs_nom, us_nom, env)

    # ── Step 4: benchmark (uncomment after controllers are ready) ───────────
    # for label, fn in [("PID",          run_pid),
    #                   ("StateFeedback", run_state_feedback),
    #                   ("CBF-CLF-QP",   run_cbf_clf_qp)]:
    #     metrics = evaluate_controller(fn, xs_nom, us_nom, num_trials=20)
    #     print(f"\n[{label}] {metrics}")

    print("\nDone. Implement the TODO sections and re-run.")