
# controllers/pid.py
# Two PID controller classes for spaceship trajectory tracking.
#
#   PositionPIDController   — tracks (x, y) position only
#                             maps position error → desired heading → (r, a)
#
#   FullStatePIDController  — tracks full state (x, y, θ, v_x, v_y)
#                             separate PD channels per state component,
#                             adapted from the rocket-landing PD structure
#
# Control outputs: u = (r, a)
#   r  : turn rate  (rad/s)
#   a  : thrust magnitude (m/s²)

import jax.numpy as jnp
import numpy as np

from .base import BaseController


# 1. Position-Only PID Controller
#    Tracks the (x, y) waypoint of the nominal trajectory.
#    Heading and speed are derived from the position error.

class PositionPIDController(BaseController):
    """
    PID controller that tracks position (x, y) waypoints along the nominal
    trajectory. The desired heading is computed from the position error vector,
    and the heading error drives the turn rate r. Speed error drives thrust a.

    Gains:
        Kp_pos, Ki_pos, Kd_pos : position PID gains (scalar, applied to both x and y)
        Kp_heading              : proportional gain on heading error → turn rate r
        Kp_speed, Kd_speed      : PD gains on speed error → thrust a

    Control limits:
        r_limit : max |turn rate|  (rad/s)
        a_limit : max |thrust|     (m/s²)
    """

    def __init__(
        self,
        Kp_pos:     float = 1.0,
        Ki_pos:     float = 0.01,
        Kd_pos:     float = 0.5,
        Kp_heading: float = 2.0,
        Kp_speed:   float = 1.0,
        Kd_speed:   float = 0.3,
        r_limit:    float = 1.0,
        a_limit:    float = 2.0,
        dt:         float = 0.1,
    ):
        self.Kp_pos     = Kp_pos
        self.Ki_pos     = Ki_pos
        self.Kd_pos     = Kd_pos
        self.Kp_heading = Kp_heading
        self.Kp_speed   = Kp_speed
        self.Kd_speed   = Kd_speed
        self.r_limit    = r_limit
        self.a_limit    = a_limit
        self.dt         = dt

        # Internal integrator and derivative state
        self._integral  = np.zeros(2)
        self._prev_error = np.zeros(2)
        self._prev_speed_error = 0.0

    def reset(self):
        """Reset integrators and derivative memory between trials."""
        self._integral         = np.zeros(2)
        self._prev_error       = np.zeros(2)
        self._prev_speed_error = 0.0

    def __call__(
        self,
        state:  jnp.ndarray,
        k:      int,
        xs_nom: jnp.ndarray,
        us_nom: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Args:
            state   : (x, y, θ, v_x, v_y)
            k       : current time step
            xs_nom  : nominal trajectory (T+1, 5)
            us_nom  : nominal controls   (T,   2)
        Returns:
            u = (r, a)
        """
        x, y, theta, v_x, v_y = np.array(state)

        # Target waypoint from nominal trajectory
        target     = np.array(xs_nom[k])
        x_t, y_t   = target[0], target[1]

        # Position error
        pos_error  = np.array([x_t - x, y_t - y])

        # PID on position
        self._integral  += pos_error * self.dt
        # Anti-windup: clamp integral
        self._integral   = np.clip(self._integral, -5.0, 5.0)
        derivative       = (pos_error - self._prev_error) / self.dt
        self._prev_error = pos_error.copy()

        pid_output = (
            self.Kp_pos * pos_error
            + self.Ki_pos * self._integral
            + self.Kd_pos * derivative
        )

        # Desired heading from position error
        desired_heading = np.arctan2(pid_output[1], pid_output[0])
        heading_error   = desired_heading - theta
        # Wrap heading error to [-π, π]
        heading_error   = (heading_error + np.pi) % (2 * np.pi) - np.pi

        # Turn rate r from heading error
        r = self.Kp_heading * heading_error
        r = np.clip(r, -self.r_limit, self.r_limit)

        # Thrust a from speed error (PD control)
        current_speed      = np.sqrt(v_x**2 + v_y**2)
        desired_speed      = np.linalg.norm(pid_output)
        speed_error        = desired_speed - current_speed
        speed_deriv        = (speed_error - self._prev_speed_error) / self.dt
        self._prev_speed_error = speed_error

        a = self.Kp_speed * speed_error + self.Kd_speed * speed_deriv
        a = np.clip(a, -self.a_limit, self.a_limit)

        return jnp.array([r, a])


# 2. Full-State PID Controller
#    Tracks the full nominal state (x, y, θ, v_x, v_y) waypoint-by-waypoint.
#    Adapted from the rocket-landing PD structure — separate gain channels
#    per state component, mapped to spaceship controls (r, a).

class FullStatePIDController(BaseController):
    """
    Full-state PID controller. Maintains separate PID channels for each state
    component and maps them to the two spaceship controls (r, a).

    State:   (x, y, θ, v_x, v_y)
    Control: (r=turn rate, a=thrust)

    Gain structure (mirroring the rocket PD pattern):
        Kp_x,  Kd_x   : position x  error → contributes to thrust direction
        Kp_y,  Kd_y   : position y  error → contributes to thrust direction
        Kp_th, Kd_th  : heading θ   error → turn rate r
        Kp_vx, Kd_vx  : x-velocity  error → thrust a (x-component)
        Kp_vy, Kd_vy  : y-velocity  error → thrust a (y-component)

    Control limits:
        r_limit : max |r|
        a_limit : max |a|
    """

    def __init__(
        self,
        Kp_x:    float = 1.0,
        Kd_x:    float = 0.5,
        Ki_x:    float = 0.01,
        Kp_y:    float = 1.0,
        Kd_y:    float = 0.5,
        Ki_y:    float = 0.01,
        Kp_th:   float = 2.0,
        Kd_th:   float = 0.3,
        Ki_th:   float = 0.0,
        Kp_vx:   float = 0.5,
        Kd_vx:   float = 0.1,
        Kp_vy:   float = 0.5,
        Kd_vy:   float = 0.1,
        r_limit: float = 1.0,
        a_limit: float = 2.0,
        dt:      float = 0.1,
    ):
        # Store all gains
        self.Kp_x  = Kp_x;  self.Kd_x  = Kd_x;  self.Ki_x  = Ki_x
        self.Kp_y  = Kp_y;  self.Kd_y  = Kd_y;  self.Ki_y  = Ki_y
        self.Kp_th = Kp_th; self.Kd_th = Kd_th; self.Ki_th = Ki_th
        self.Kp_vx = Kp_vx; self.Kd_vx = Kd_vx
        self.Kp_vy = Kp_vy; self.Kd_vy = Kd_vy
        self.r_limit = r_limit
        self.a_limit = a_limit
        self.dt      = dt

        # Internal state: integrals and previous errors for each channel
        self._int_x  = 0.0; self._prev_ex  = 0.0
        self._int_y  = 0.0; self._prev_ey  = 0.0
        self._int_th = 0.0; self._prev_eth = 0.0
        self._prev_evx = 0.0
        self._prev_evy = 0.0

    def reset(self):
        """Reset all integrators and derivative memory between trials."""
        self._int_x  = 0.0; self._prev_ex  = 0.0
        self._int_y  = 0.0; self._prev_ey  = 0.0
        self._int_th = 0.0; self._prev_eth = 0.0
        self._prev_evx = 0.0
        self._prev_evy = 0.0

    def __call__(
        self,
        state:  jnp.ndarray,
        k:      int,
        xs_nom: jnp.ndarray,
        us_nom: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Args:
            state   : (x, y, θ, v_x, v_y)
            k       : current time step
            xs_nom  : nominal trajectory (T+1, 5)
            us_nom  : nominal controls   (T,   2)
        Returns:
            u = (r, a)
        """
        x, y, theta, v_x, v_y = np.array(state)

        # Target from nominal trajectory
        x_t, y_t, th_t, vx_t, vy_t = np.array(xs_nom[k])

        # Per-channel errors
        e_x  = x_t  - x
        e_y  = y_t  - y
        e_th = th_t - theta
        e_vx = vx_t - v_x
        e_vy = vy_t - v_y

        # Wrap heading error to [-π, π]
        e_th = (e_th + np.pi) % (2 * np.pi) - np.pi

        # Integrals (with anti-windup clamp)─
        self._int_x  = np.clip(self._int_x  + e_x  * self.dt, -5.0, 5.0)
        self._int_y  = np.clip(self._int_y  + e_y  * self.dt, -5.0, 5.0)
        self._int_th = np.clip(self._int_th + e_th * self.dt, -2.0, 2.0)

        # Derivatives─
        d_x  = (e_x  - self._prev_ex)  / self.dt
        d_y  = (e_y  - self._prev_ey)  / self.dt
        d_th = (e_th - self._prev_eth) / self.dt
        d_vx = (e_vx - self._prev_evx) / self.dt
        d_vy = (e_vy - self._prev_evy) / self.dt

        # Store previous errors─
        self._prev_ex  = e_x
        self._prev_ey  = e_y
        self._prev_eth = e_th
        self._prev_evx = e_vx
        self._prev_evy = e_vy

        # Map to controls─
        # Turn rate r: driven by heading error (PID on θ)
        r = (
            self.Kp_th * e_th
            + self.Ki_th * self._int_th
            + self.Kd_th * d_th
        )

        # Thrust a: combine position corrections and velocity corrections
        # x-channel contribution
        ax = (
            self.Kp_x  * e_x  + self.Ki_x * self._int_x  + self.Kd_x  * d_x
            + self.Kp_vx * e_vx + self.Kd_vx * d_vx
        )
        # y-channel contribution
        ay = (
            self.Kp_y  * e_y  + self.Ki_y * self._int_y  + self.Kd_y  * d_y
            + self.Kp_vy * e_vy + self.Kd_vy * d_vy
        )
        # Project onto heading direction to get scalar thrust
        a = ax * np.cos(theta) + ay * np.sin(theta)

        # Clip to control limits
        r = np.clip(r, -self.r_limit, self.r_limit)
        a = np.clip(a, -self.a_limit, self.a_limit)

        return jnp.array([r, a])