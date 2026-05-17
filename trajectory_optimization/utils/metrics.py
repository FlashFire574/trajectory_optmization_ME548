
# utils/metrics.py
# Evaluation harness and performance metrics.
#
# Metrics collected per trial:
#   tracking_error  — mean L2 distance from nominal (x, y) over trajectory
#   control_effort  — mean ||u||² (proxy for fuel consumption)
#   traversal_time  — steps taken to reach within goal_threshold of goal
#                     (= T if goal never reached)
#   success         — bool: reached goal AND no collision
#
# Main entry point:
#   evaluate_controller(controller, dynamics, xs_nom, us_nom, env_config, ...)

import numpy as np
import jax.numpy as jnp
from typing import Callable, Dict, List, Optional, Tuple


# Collision detection

def check_collision(position: np.ndarray, env, ship_radius: float) -> bool:
    """
    Returns True if `position` collides with any asteroid in `env`.

    Args:
        position   : (x, y) array
        env        : Environment NamedTuple (with asteroids.center, asteroids.radius)
        ship_radius: radius of the spacecraft
    """
    centers = np.array(env.asteroids.center)
    radii   = np.array(env.asteroids.radius)

    # Handle wrapped (toroidal) positions
    bounds  = np.array(env.bounds)
    deltas  = (position - centers + bounds / 2) % bounds - bounds / 2
    dists   = np.linalg.norm(deltas, axis=-1)

    # Collision if distance < asteroid_radius + ship_radius
    return bool(np.any(dists < radii + ship_radius))


# Single trial rollout

def rollout(
    controller,
    dynamics,
    xs_nom:          jnp.ndarray,
    us_nom:          jnp.ndarray,
    env,
    start_state:     np.ndarray,
    goal_position:   np.ndarray,
    T:               int,
    dt:              float,
    goal_threshold:  float = 2.0,
    use_env_at_time: bool  = True,
) -> Dict:
    """
    Roll out a controller for one trial and collect metrics.

    Args:
        controller     : BaseController instance (already prepared if needed)
        dynamics       : RK4Integrator
        xs_nom         : nominal trajectory (T+1, 5)
        us_nom         : nominal controls   (T,   2)
        env            : Environment at t=0
        start_state    : initial state (5,)
        goal_position  : target (x, y)
        T              : number of steps
        dt             : time step
        goal_threshold : distance to goal counted as success (m)
        use_env_at_time: if True, asteroids move each step

    Returns:
        dict with keys: history, controls, tracking_errors, control_efforts,
                        traversal_time, success, collision_step
    """
    controller.reset()

    state    = jnp.array(start_state)
    history  = [np.array(state)]
    controls = []

    success        = False
    collision      = False
    collision_step = T
    traversal_time = T

    for k in range(T):
        # Move asteroids to current time
        env_k = env.at_time(k * dt) if use_env_at_time else env

        # Compute control
        u = controller(state, k, xs_nom, us_nom)
        u = jnp.array(u)

        # Step dynamics
        state = dynamics(state, u, k)

        history.append(np.array(state))
        controls.append(np.array(u))

        # Check collision
        if not collision and check_collision(np.array(state[:2]), env_k, env.ship_radius):
            collision      = True
            collision_step = k + 1

        # Check goal reached
        dist_to_goal = np.linalg.norm(np.array(state[:2]) - goal_position)
        if dist_to_goal < goal_threshold and not success:
            traversal_time = k + 1
            success        = not collision

    history  = np.array(history)   # (T+1, 5)
    controls = np.array(controls)  # (T,   2)

    #  Tracking error: mean L2 distance in (x, y) from nominal 
    tracking_errors = np.linalg.norm(
        history[:-1, :2] - np.array(xs_nom[:-1, :2]), axis=-1
    )

    #  Control effort: mean ||u||² 
    control_efforts = np.sum(controls ** 2, axis=-1)

    return {
        "history":         history,
        "controls":        controls,
        "tracking_errors": tracking_errors,
        "control_efforts": control_efforts,
        "traversal_time":  traversal_time,
        "success":         success and not collision,
        "collision":       collision,
        "collision_step":  collision_step,
    }


# Multi-trial evaluation

def evaluate_controller(
    controller,
    dynamics,
    xs_nom:         jnp.ndarray,
    us_nom:         jnp.ndarray,
    start_state:    np.ndarray,
    goal_position:  np.ndarray,
    T:              int,
    dt:             float,
    num_asteroids:  int   = 20,
    num_trials:     int   = 20,
    ship_radius:    float = 1.0,
    sensing_radius: float = 5.0,
    bounds:         tuple = (50, 40),
    goal_threshold: float = 2.0,
    seed_start:     int   = 0,
    density_label:  str   = "medium",
    verbose:        bool  = True,
) -> Dict:
    """
    Evaluate a controller over multiple randomized environments.

    Args:
        controller    : BaseController (call prepare() before passing in if needed)
        dynamics      : RK4Integrator
        xs_nom        : nominal trajectory
        us_nom        : nominal controls
        start_state   : initial state
        goal_position : target (x, y)
        T, dt         : time horizon and step size
        num_asteroids : obstacle density
        num_trials    : number of random seeds
        seed_start    : first random seed
        density_label : string tag for printing

    Returns:
        dict with aggregate metrics (mean ± std) and per-trial results
    """
    # Import here to avoid circular imports
    from environment.asteroid_env import Environment

    all_tracking  = []
    all_effort    = []
    all_time      = []
    all_success   = []
    all_trials    = []

    for trial in range(num_trials):
        seed = seed_start + trial
        np.random.seed(seed)
        env = Environment.create(
            num_asteroids,
            ship_radius=ship_radius,
            sensing_radius=sensing_radius,
            bounds=bounds,
        )

        result = rollout(
            controller, dynamics,
            xs_nom, us_nom, env,
            start_state, goal_position,
            T, dt, goal_threshold,
        )

        all_tracking.append(np.mean(result["tracking_errors"]))
        all_effort.append(np.mean(result["control_efforts"]))
        all_time.append(result["traversal_time"])
        all_success.append(float(result["success"]))
        all_trials.append(result)

        if verbose:
            status = "✓" if result["success"] else "✗"
            print(
                f"  [{density_label}] Trial {trial+1:02d}/{num_trials} {status} "
                f"| track={all_tracking[-1]:.2f} "
                f"| effort={all_effort[-1]:.2f} "
                f"| time={all_time[-1]}"
            )

    metrics = {
        "density":              density_label,
        "num_trials":           num_trials,
        # Mean metrics
        "tracking_error_mean":  float(np.mean(all_tracking)),
        "tracking_error_std":   float(np.std(all_tracking)),
        "control_effort_mean":  float(np.mean(all_effort)),
        "control_effort_std":   float(np.std(all_effort)),
        "traversal_time_mean":  float(np.mean(all_time)),
        "traversal_time_std":   float(np.std(all_time)),
        "success_rate":         float(np.mean(all_success)),
        # Raw per-trial data
        "per_trial":            all_trials,
    }

    if verbose:
        print(
            f"\n  [{density_label}] Summary — "
            f"success={metrics['success_rate']:.0%} | "
            f"track={metrics['tracking_error_mean']:.2f}±{metrics['tracking_error_std']:.2f} | "
            f"effort={metrics['control_effort_mean']:.2f}±{metrics['control_effort_std']:.2f} | "
            f"time={metrics['traversal_time_mean']:.1f}±{metrics['traversal_time_std']:.1f}"
        )

    return metrics