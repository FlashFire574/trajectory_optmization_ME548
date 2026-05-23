
# Single source of truth for all shared parameters.

import numpy as np
import jax.numpy as jnp

# Time horizon
T  = 125
dt = 0.1

# Dimensions
STATE_DIM   = 5   # (x, y, θ, v_x, v_y)
CONTROL_DIM = 2   # (r, a)

# Boundary conditions 
start_state   = np.array([5.,  5.,  0., 0., 0.])
goal_position = np.array([45., 35.])

# Environment
NUM_ASTEROIDS  = 20
SHIP_RADIUS    = 1.0
SENSING_RADIUS = 5.0
BOUNDS         = (50, 40)
RANDOM_SEED    = 2

np.random.seed(RANDOM_SEED)

# LQR cost matrices 
# Q: state deviation cost — higher = track nominal more tightly
#    (x, y) weighted 10x higher than heading and velocities
Q = jnp.diag(jnp.array([
    10.0,   # x position
    10.0,   # y position
    1.0,    # heading θ
    1.0,    # v_x
    1.0,    # v_y
]))

# R: control effort cost — keep small relative to Q
#    higher = more fuel-efficient but sluggish response
R = jnp.diag(jnp.array([
    0.1,    # turn rate r
    0.1,    # thrust a
]))

# Terminal cost for LQR (Q_terminal = 10 * Q)
Q_terminal = 10.0 * Q

# Controller limits
R_LIMIT = np.pi / 2   # max |turn rate|  (rad/s)
A_LIMIT = 4.0         # max |thrust|     (m/s²)

# MPC
MPC_HORIZON     = 20
LIMITED_SENSING = True

# Evaluation
NUM_TRIALS     = 20
GOAL_THRESHOLD = 2.0
SEED_START     = 0
