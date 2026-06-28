# utils/sensing.py
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
    sensed_env = env.sense(state[:2]) if limited_sensing else env
    empty_env  = Environment.create(0)

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
        stage1["optimal_trajectory"][1],
    )

    plan_states, plan_controls = stage2["optimal_trajectory"]
    return plan_controls[0], (plan_states, plan_controls)


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
    states   = [np.array(start_state)]
    controls = []
    plans    = []

    collision      = False
    collision_step = T
    success        = False

    for t in range(T):
        env_t = env.at_time(t * dynamics.dt)

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

        next_state = np.array(mpc_states[1])
        states.append(next_state)
        controls.append(np.array(control))
        plans.append(np.array(mpc_states))

        if not collision:
            from utils.metrics import check_collision
            if check_collision(next_state[:2], env_t, env.ship_radius):
                collision      = True
                collision_step = t + 1
                if verbose:
                    print(f"  [MPC] Collision at step {t+1}")

        dist = np.linalg.norm(next_state[:2] - np.array(goal_position))
        if dist < goal_threshold and not success:
            success = not collision
            if verbose:
                print(f"  [MPC] Goal reached at step {t+1}  (dist={dist:.2f}m)")

        if verbose and t % 25 == 0:
            print(f"  [MPC] step={t:3d}/{T}  pos=({next_state[0]:.1f},{next_state[1]:.1f})"
                  f"  dist_to_goal={dist:.1f}")

    return {
        "history":         np.array(states),
        "controls":        np.array(controls),
        "plans":           np.array(plans),
        "success":         success,
        "collision":       collision,
        "collision_step":  collision_step,
    }


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
    controller.reset()

    # Force concrete numpy array to break JAX trace cache between trials
    state   = jnp.array(np.array(start_state))
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
        env_k = env.at_time(k * dt)

        # Force state to concrete numpy before sensing to break JAX cache
        state_np   = np.array(state)
        env_sensed = env_k.sense(state_np[:2]) if limited_sensing else env_k

        visible = int(np.sum(~np.isnan(np.array(env_sensed.asteroids.radius))))
        sensed_counts.append(visible)

        # Force concrete state for controller input
        state_in = jnp.array(state_np)

        try:
            u = controller(state_in, k, xs_nom, us_nom, env_sensed)
        except TypeError:
            u = controller(state_in, k, xs_nom, us_nom)

        u = jnp.array(u)

        # Step dynamics and immediately materialize to numpy
        state = jnp.array(np.array(dynamics(state_in, u, k)))

        history.append(np.array(state))
        controls.append(np.array(u))

        track_err = float(np.linalg.norm(
            np.array(state[:2]) - np.array(xs_nom[k, :2])
        ))
        tracking_errors.append(track_err)
        control_efforts.append(float(np.sum(np.array(u) ** 2)))

        from utils.metrics import check_collision
        if not collision and check_collision(np.array(state[:2]), env_k, env.ship_radius):
            collision      = True
            collision_step = k + 1
            if verbose:
                print(f"  Collision at step {k+1}")

        dist = np.linalg.norm(np.array(state[:2]) - np.array(goal_position))
        if dist < goal_threshold and not goal_reached:
            traversal_time = k + 1
            goal_reached   = True
            success        = not collision
            if verbose:
                print(f"  Goal reached at step {k+1}  dist={dist:.2f}m")

        if verbose and k % 25 == 0:
            print(f"  step={k:3d}/{T}  pos=({float(state[0]):.1f},{float(state[1]):.1f})"
                  f"  visible_asteroids={visible}  track_err={track_err:.2f}")

    return {
        "history":          np.array(history),
        "controls":         np.array(controls),
        "sensed_asteroids": sensed_counts,
        "tracking_errors":  np.array(tracking_errors),
        "control_efforts":  np.array(control_efforts),
        "traversal_time":   traversal_time,
        "goal_reached":     goal_reached,
        "success":          success,
        "collision":        collision,
        "collision_step":   collision_step,
    }