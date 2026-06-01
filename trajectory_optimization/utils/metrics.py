
# utils/metrics.py
# Evaluation harness and performance metrics.
#
# Metrics per trial:
#   success          — bool: reached goal AND no collision
#   fuel_burn        — sum of ||u||² over full trajectory
#   tracking_error   — mean L2 distance from nominal (x, y)
#   traversal_time   — steps to reach goal (T if not reached)
#   collision        — bool
#   progress_pct     — % of start→goal distance covered at collision
#                      (only meaningful when collision=True

import numpy as np
import jax.numpy as jnp
from typing import Dict, Optional

# Collision detectio

def check_collision(position: np.ndarray, env, ship_radius: float) -> bool:
    """Returns True if position collides with any asteroid in env."""
    centers = np.array(env.asteroids.center)
    radii   = np.array(env.asteroids.radius)
    bounds  = np.array(env.bounds)

    # Ignore NaN radii (asteroids outside sensing radius)
    valid = ~np.isnan(radii)
    if not np.any(valid):
        return False

    deltas = (position - centers[valid] + bounds / 2) % bounds - bounds / 2
    dists  = np.linalg.norm(deltas, axis=-1)
    return bool(np.any(dists < radii[valid] + ship_radius))

# Progress metric — how far along start→goal at time of collisio

def compute_progress(
    position:       np.ndarray,
    start_position: np.ndarray,
    goal_position:  np.ndarray,
) -> float:
    """
    Returns % of start→goal distance covered when reaching `position`.

    Projects position onto the start→goal line, clamped to [0, 100].
    This gives a meaningful progress metric even when the spacecraft
    drifts off the straight-line path.
    """
    total_vec  = goal_position - start_position
    total_dist = np.linalg.norm(total_vec)
    if total_dist < 1e-6:
        return 100.0

    traveled_vec = position - start_position
    # Scalar projection onto start→goal direction
    progress = np.dot(traveled_vec, total_vec) / total_dist
    return float(np.clip(progress / total_dist * 100.0, 0.0, 100.0))

# Multi-trial evaluation — uses simulate_with_sensin

def evaluate_controller(
    controller,
    dynamics,
    xs_nom:         np.ndarray,
    us_nom:         np.ndarray,
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
    limited_sensing: bool = True,
    seed_start:     int   = 0,
    density_label:  str   = "medium",
    verbose:        bool  = True,
) -> Dict:
    """
    Evaluate a controller over num_trials randomized environments.
    Uses simulate_with_sensing so evaluation matches the notebook visualization.

    Returns dict with aggregate metrics and per-trial results.
    """
    from environment.asteroid_env import Environment
    from utils.sensing import simulate_with_sensing

    all_success      = []
    all_fuel         = []
    all_tracking     = []
    all_time         = []
    all_progress     = []   # progress % at collision (nan if no collision)
    all_trials       = []
    all_goal_reached   = []
    all_collision_free = []

    for trial in range(num_trials):
        seed = seed_start + trial
        np.random.seed(seed)
        env = Environment.create(
            num_asteroids,
            ship_radius    = ship_radius,
            sensing_radius = sensing_radius,
            bounds         = bounds,
        )

        # Re-prepare LQR-based controllers for each trial
        if hasattr(controller, 'prepare'):
            controller.prepare(xs_nom, us_nom)

        # Run with sensing — same as notebook visualization
        res = simulate_with_sensing(
            controller, start_state, env, dynamics,
            xs_nom, us_nom, goal_position,
            T              = T,
            dt             = dt,
            limited_sensing = limited_sensing,
            goal_threshold  = goal_threshold,
            verbose         = False,
        )

        # Fuel burn: sum of ||u||²
        fuel = float(np.sum(res['control_efforts']))

        # Tracking error: mean L2 from nominal (x,y)
        track = float(np.mean(res['tracking_errors']))

        # Progress at collision
        if res['collision']:
            col_step = res['collision_step']
            col_pos  = res['history'][col_step, :2]
            progress = compute_progress(col_pos, start_state[:2], goal_position)
        else:
            progress = 100.0 if res['success'] else compute_progress(
                res['history'][-1, :2], start_state[:2], goal_position
            )

        all_success.append(float(res['success']))
        all_fuel.append(fuel)
        all_tracking.append(track)
        all_time.append(res['traversal_time'])
        all_progress.append(progress)
        all_trials.append(res)
        all_goal_reached.append(float(res['goal_reached']))
        all_collision_free.append(float(not res['collision']))

        if verbose:
            status = "✓" if res['success'] else ("✗ collision" if res['collision'] else "✗ timeout")
            print(
                f"  [{density_label}] Trial {trial+1:02d}/{num_trials}  {status:12s} | "
                f"fuel={fuel:7.2f}  track={track:.2f}m  "
                f"time={res['traversal_time']:3d}  progress={progress:.0f}%"
            )

    metrics = {
        "density":             density_label,
        "num_trials":          num_trials,
        # Success
        "success_rate":        float(np.mean(all_success)),
        "goal_reached_rate":   float(np.mean(all_goal_reached)),
        "collision_free_rate": float(np.mean(all_collision_free)),
        # Fuel burn
        "fuel_burn_mean":      float(np.mean(all_fuel)),
        "fuel_burn_std":       float(np.std(all_fuel)),
        "fuel_burn_total":     float(np.sum(all_fuel)),
        # Tracking error
        "tracking_error_mean": float(np.mean(all_tracking)),
        "tracking_error_std":  float(np.std(all_tracking)),
        # Traversal time
        "traversal_time_mean": float(np.mean(all_time)),
        "traversal_time_std":  float(np.std(all_time)),
        # Collision progress
        "progress_mean":       float(np.mean(all_progress)),
        "progress_std":        float(np.std(all_progress)),
        # Raw per-trial data
        "per_trial":           all_trials,
    }

    if verbose:
        print(
            f"\n  [{density_label}] Summary\n"
            f"  Success rate : {metrics['success_rate']:.0%}\n"
            f"  Fuel burn    : {metrics['fuel_burn_mean']:.2f} ± {metrics['fuel_burn_std']:.2f}\n"
            f"  Track error  : {metrics['tracking_error_mean']:.2f} ± {metrics['tracking_error_std']:.2f} m\n"
            f"  Traversal    : {metrics['traversal_time_mean']:.1f} ± {metrics['traversal_time_std']:.1f} steps\n"
            f"  Progress     : {metrics['progress_mean']:.1f} ± {metrics['progress_std']:.1f} %\n"
            f"  Goal reached : {metrics['goal_reached_rate']:.0%}\n"
            f"  Collision free: {metrics['collision_free_rate']:.0%}\n"
        )

    return metrics