
# utils/sensing.py
# Receding-horizon MPC simulation with limited sensing.
#
# How it works:
#   At each time step t:
#     1. env.sense(state[:2])  filters out asteroids beyond sensing_radius
#        (sets their radius to NaN → RunningCost ignores them automatically)
#     2. iLQR replans a short horizon of N steps using only visible asteroids
#        (two-stage warm-start: empty env first, then sensed env)
#     3. Only the FIRST control of the plan is applied to the real dynamics
#     4. The spacecraft moves, and the process repeats
#
# This gives any controller awareness of nearby obstacles without needing
# to know the full asteroid field in advance.
#
# Two modes:
#   simulate_mpc_open   — pure iLQR MPC, no external controller
#   simulate_with_sensing — wraps any BaseController with sensing-aware
#                           replanning so PID / State Feedback / CBF also
#                           benefit from local obstacle informatio

import functools
from typing import Optional, Type

import jax
import jax.numpy as jnp
import numpy as np

from environment.asteroid_env import Environment
from environment.ilqr import (
    TotalCost,
    RunningCost,
    FullHorizonTerminalCost,
    iterative_linear_quadratic_regulator,
)

# SECTION A — Core MPC policy (single step

@functools.partial(jax.jit, static_argnames=["running_cost_type",
                                              "terminal_cost_type",
                                              "limited_sensing", "N"])
def mpc_policy_step(
    state,
    env,
    dynamics,
    goal_position,
    running_cost_type  = RunningCost,
    terminal_cost_type = FullHorizonTerminalCost,
    limited_sensing:   bool = True,
    N:                 int  = 20,
):
    """
    Compute one MPC control action via two-stage iLQR replanning.

    Args:
        state              : current 5-state (x, y, θ, v_x, v_y)
        env                : Environment at current time t
        dynamics           : RK4Integrator
        goal_position      : target (x, y)
        running_cost_type  : cost class with signature (env, dt) -> callable
        terminal_cost_type : cost class with create_ignoring_extra_args()
        limited_sensing    : if True, filter env to sensing radius first
        N                  : MPC planning horizon (steps)

    Returns:
        control : (r, a) — first control of the replanned sequence
        plan    : (states, controls) — full N-step plan for visualization
    """
    #  Step 1: apply sensing filter
    sensed_env = env.sense(state[:2]) if limited_sensing else env
    empty_env  = Environment.create(0)

    #  Step 2: Stage 1 warm-start on empty environment 
    stage1 = iterative_linear_quadratic_regulator(
        dynamics,
        TotalCost(
            running_cost_type(empty_env, dynamics.dt),
            terminal_cost_type.create_ignoring_extra_args(
                empty_env, goal_position,
                state[:2], empty_env.sensing_radius,
            ),
        ),
        state,
        jnp.zeros((N, 2)),
    )

    #  Step 3: Stage 2 replan on sensed environment
    stage2 = iterative_linear_quadratic_regulator(
        dynamics,
        TotalCost(
            running_cost_type(sensed_env, dynamics.dt),
            terminal_cost_type.create_ignoring_extra_args(
                sensed_env, goal_position,
                state[:2], sensed_env.sensing_radius,
            ),
        ),
        state,
        stage1["optimal_trajectory"][1],   # warm-start
    )

    plan_states, plan_controls = stage2["optimal_trajectory"]

    # Return only the first control action (receding horizon)
    return plan_controls[0], (plan_states, plan_controls)

# Pure iLQR MPC simulation

def simulate_mpc(
    start_state,
    env,
    dynamics,
    goal_position,
    running_cost_type  = RunningCost,
    terminal_cost_type = FullHorizonTerminalCost,
    limited_sensing:   bool = True,
    N:                 int  = 20,
    T:                 int  = 125,
    goal_threshold:    float = 2.0,
    verbose:           bool  = True,
):
    """
    Simulate the spacecraft using pure iLQR MPC with optional limited sensing.

    At each step, replans a horizon of N steps and applies only the first
    control. Asteroids outside sensing_radius are invisible to the planner.

    Args:
        start_state        : initial 5-state
        env                : Environment at t=0
        dynamics           : RK4Integrator
        goal_position      : target (x, y)
        running_cost_type  : RunningCost or subclass
        terminal_cost_type : FullHorizonTerminalCost or subclass
        limited_sensing    : enable sensing radius filter
        N                  : MPC planning horizon
        T                  : total simulation steps
        goal_threshold     : distance to goal counted as arrival (m)
        verbose            : print step-by-step progress

    Returns:
        dict with keys:
            history   : (T+1, 5) visited states
            controls  : (T,   2) applied controls
            plans     : (T, N+1, 5) per-step MPC plans (for animation)
            success   : bool
            collision : bool
    """
    states   = [np.array(start_state)]
    controls = []
    plans    = []

    collision      = False
    collision_step = T
    success        = False

    for t in range(T):
        # Move asteroids to current time
        env_t = env.at_time(t * dynamics.dt)

        # Replan and get first control
        control, (mpc_states, mpc_controls) = mpc_policy_step(
            jnp.array(states[-1]),
            env_t,
            dynamics,
            jnp.array(goal_position),
            running_cost_type  = running_cost_type,
            terminal_cost_type = terminal_cost_type,
            limited_sensing    = limited_sensing,
            N                  = N,
        )

        # Apply control — use mpc_states[1] (the predicted next state)
        # This matches the original simulate_mpc convention exactly
        next_state = np.array(mpc_states[1])
        states.append(next_state)
        controls.append(np.array(control))
        plans.append(np.array(mpc_states))

        # Collision check
        if not collision:
            from utils.metrics import check_collision
            if check_collision(next_state[:2], env_t, env.ship_radius):
                collision      = True
                collision_step = t + 1
                if verbose:
                    print(f"  [MPC] Collision at step {t+1}")

        # Goal check
        dist = np.linalg.norm(next_state[:2] - np.array(goal_position))
        if dist < goal_threshold and not success:
            success = not collision
            if verbose:
                print(f"  [MPC] Goal reached at step {t+1}  (dist={dist:.2f}m)")

        if verbose and t % 25 == 0:
            print(f"  [MPC] step={t:3d}/{T}  pos=({next_state[0]:.1f},{next_state[1]:.1f})"
                  f"  dist_to_goal={dist:.1f}")

    return {
        "history":         np.array(states),    # (T+1, 5)
        "controls":        np.array(controls),  # (T,   2)
        "plans":           np.array(plans),     # (T, N+1, 5)
        "success":         success,
        "collision":       collision,
        "collision_step":  collision_step,
    }

# SECTION C — Sensing wrapper for external controllers
#             (PID, State Feedback, CBF-CLF-QP

def simulate_with_sensing(
    controller,
    start_state,
    env,
    dynamics,
    xs_nom,
    us_nom,
    goal_position,
    T:              int   = 125,
    dt:             float = 0.1,
    limited_sensing: bool = True,
    goal_threshold: float = 2.0,
    verbose:        bool  = False,
):
    """
    Roll out any BaseController with sensing-aware obstacle information.

    Unlike simulate_mpc (which replans via iLQR), this wraps classical
    controllers (PID, State Feedback, CBF) so they receive a sensing-filtered
    environment at each step. The controller still tracks xs_nom/us_nom,
    but the environment it "sees" is limited to its sensing radius.

    This is the right way to evaluate PID / State Feedback / CBF fairly —
    they are not replanning, but they at least know which asteroids are nearby.

    Args:
        controller      : BaseController instance (call prepare() first if needed)
        start_state     : initial 5-state
        env             : Environment at t=0
        dynamics        : RK4Integrator
        xs_nom          : nominal trajectory (T+1, 5)
        us_nom          : nominal controls   (T,   2)
        goal_position   : target (x, y)
        T               : total steps
        dt              : time step
        limited_sensing : filter env to sensing radius each step
        goal_threshold  : goal arrival distance (m)
        verbose         : print progress

    Returns:
        dict with keys:
            history          : (T+1, 5)
            controls         : (T,   2)
            sensed_asteroids : list of per-step visible asteroid counts
            tracking_errors  : (T,) L2 distance from nominal (x, y)
            control_efforts  : (T,) ||u||²
            traversal_time   : steps to goal (T if not reached)
            success          : bool
            collision        : bool
            collision_step   : int
    """
    controller.reset()

    state   = jnp.array(start_state)
    history = [np.array(state)]
    controls         = []
    sensed_counts    = []
    tracking_errors  = []
    control_efforts  = []

    collision      = False
    collision_step = T
    success        = False
    goal_reached   = False
    traversal_time = T

    for k in range(T):
        #  Move asteroids─
        env_k = env.at_time(k * dt)

        #  Apply sensing filter
        # env_sensed has NaN radii for asteroids beyond sensing_radius.
        # Controllers that use env_k for CBF constraints will naturally
        # ignore unseen asteroids because NaN distances are filtered.
        env_sensed = env_k.sense(np.array(state[:2])) if limited_sensing else env_k

        # Count visible asteroids (non-NaN radii)
        visible = int(np.sum(~np.isnan(np.array(env_sensed.asteroids.radius))))
        sensed_counts.append(visible)

        #  Compute control
        # CBF-CLF-QP controllers may accept env_sensed as extra arg;
        # PID and State Feedback ignore it (they track xs_nom directly).
        try:
            # Try calling with env_sensed (for CBF-CLF-QP)
            u = controller(state, k, xs_nom, us_nom, env_sensed)
        except TypeError:
            # Fall back to standard interface (PID, State Feedback)
            u = controller(state, k, xs_nom, us_nom)

        u = jnp.array(u)

        #  Step dynamics
        state = dynamics(state, u, k)

        history.append(np.array(state))
        controls.append(np.array(u))

        #  Tracking error─
        track_err = float(np.linalg.norm(
            np.array(state[:2]) - np.array(xs_nom[k, :2])
        ))
        tracking_errors.append(track_err)
        control_efforts.append(float(np.sum(np.array(u) ** 2)))

        #  Collision check
        from utils.metrics import check_collision
        if not collision and check_collision(np.array(state[:2]), env_k, env.ship_radius):
            collision      = True
            collision_step = k + 1
            if verbose:
                print(f"  Collision at step {k+1}")

        #  Goal check
        dist = np.linalg.norm(np.array(state[:2]) - np.array(goal_position))
        if dist < goal_threshold and not goal_reached:
            traversal_time = k + 1
            goal_reached = True
            # Lock success at the moment of arrival (consistent with simulate_mpc).
            # A later drift-collision after the goal must not retroactively fail
            # a trial that cleanly reached the goal.
            success = not collision
            if verbose:
                print(f"  Goal reached at step {k+1}  dist={dist:.2f}m")

        if verbose and k % 25 == 0:
            print(f"  step={k:3d}/{T}  pos=({float(state[0]):.1f},{float(state[1]):.1f})"
                  f"  visible_asteroids={visible}  track_err={track_err:.2f}")

    return {
        "history":          np.array(history),           # (T+1, 5)
        "controls":         np.array(controls),          # (T,   2)
        "sensed_asteroids": sensed_counts,               # list of T ints
        "tracking_errors":  np.array(tracking_errors),  # (T,)
        "control_efforts":  np.array(control_efforts),  # (T,)
        "traversal_time":   traversal_time,
        "goal_reached": goal_reached,
        "success":      success,
        "collision":        collision,
        "collision_step":   collision_step,
    }
