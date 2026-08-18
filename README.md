# Trajectory Optimization in Uncertain Environments

**Author:** Aditya Joshi | Student ID: 2530019

## Overview

This project implements and compares multiple control strategies for spacecraft navigation in dynamic, partially-observed asteroid environments. The simulation uses **iLQR** (Iterative Linear Quadratic Regulator) to compute a nominal reference trajectory, which is then tracked by four different controllers operating under **limited sensing constraints**.

## Key Features

- **Nominal Trajectory Planning**: Two-stage iLQR warm-start (empty environment → full asteroid field)
- **Limited Sensing**: Controllers only perceive asteroids within a fixed sensing radius
- **Multiple Controllers**: PID (position & full-state), State Feedback, and Infinite-Horizon LQR
- **Benchmarking**: 20-trial evaluation across random asteroid configurations
- **Interactive Visualization**: Dropdown-based controller selection and time-step scrubbing

## Project Structure

```
trajectory_optimization/
├── config.py                           # Global configuration & parameters
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── controllers/
│   ├── __init__.py
│   ├── base.py                         # Abstract controller interface
│   ├── cbf_clf_qp.py                   # Control Barrier Function / CLF-QP
│   ├── pid.py                          # PID controllers (position & full-state)
│   ├── state_feedback.py               # State feedback & LQR controllers
├── environment/
│   ├── __init__.py
│   ├── asteroid_env.py                 # Asteroid environment & visualization
│   ├── dynamics.py                     # Spacecraft dynamics (continuous-time)
│   ├── ilqr.py                         # iLQR solver & nominal trajectory computation
│   ├── integrators.py                  # RK4 time integration
├── notebooks/
│   ├── simulation.ipynb                # Main interactive simulation & analysis
├── outputs/
│   │                                   # Generated plots & metrics
├── utils/
│   ├── __init__.py
│   ├── metrics.py                      # Benchmark evaluation & statistics
│   ├── sensing.py                      # Sensor simulation with limited visibility
```

## Configuration (`config.py`)

All simulation parameters are centralized in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `T` | 50 | Horizon length (time steps) |
| `dt` | 0.2 | Time step (seconds) |
| `STATE_DIM` | 5 | State dimension [x, y, θ, vx, vy] |
| `CONTROL_DIM` | 2 | Control dimension [ax, ay] |
| `NUM_ASTEROIDS` | 8 | Number of asteroids in environment |
| `LIMITED_SENSING` | True | Enable sensing radius constraints |
| `R_LIMIT` | 5.0 | Sensing radius (meters) |
| `A_LIMIT` | 1.0 | Acceleration magnitude limit |
| `RANDOM_SEED` | 42 | Reproducibility seed |
| `Q`, `R` | - | Cost matrices for optimal control |

## Controllers

### 1. **Position PID Controller** (`pid.py`)
- Regulates heading to point toward goal
- Speed control via PID
- Decoupled position-level control
- **Parameters**: Kp, Ki, Kd for position; Kp, Kd for heading

### 2. **Full-State PID Controller** (`pid.py`)
- Direct state feedback on [x, y, vx, vy, θ]
- Independent PID loops for each state dimension
- Tighter trajectory tracking than position-only PID
- **Parameters**: Kp, Ki, Kd per dimension

### 3. **State Feedback Controller** (`state_feedback.py`)
- LQR-derived time-varying gains from nominal trajectory
- Tracks nominal path via linear perturbation theory
- Requires nominal trajectory precomputation
- **Cost**: Minimize Q (state) and R (control) weighted error

### 4. **Infinite-Horizon LQR Controller** (`state_feedback.py`)
- Steady-state LQR gains (assuming stabilizable system)
- Asymptotically optimal for regulation tasks
- Lower computational cost than time-varying LQR
- **Optimality**: Minimizes infinite-horizon discounted cost

## Simulation Notebook (`notebooks/simulation.ipynb`)

### 7 Sections:

#### 1. **Imports & Setup**
- Loads all modules, seeds RNG, initializes dynamics & environment
- Instantiates RK4 integrator and asteroid field

#### 2. **Environment Preview**
- Interactive slider to inspect asteroid positions over time
- Shows initial state and goal position

#### 3. **Nominal iLQR Trajectory**
- Two-stage solver:
  1. Solve on empty environment (warm-start)
  2. Refine on full asteroid field
- Animates reference path with thrust visualization

#### 4. **Controller Rollouts with Limited Sensing**
- Instantiates all 4 controllers with tuned gains
- Simulates each under sensing constraints
- Reports success/collision/timeout for each trial

#### 5. **Navigation Animation**
- **Dropdown menu**: Switch between controllers in real-time
- **Time slider**: Scrub through simulation
- **Visualization**: 
  - Dashed circle = sensing radius
  - Green line = nominal reference
  - Colored trail = actual controller path
  - Thrust vector overlay
- Live metrics: tracking error, visible asteroids, control magnitude

#### 6. **Tracking Error & Control Effort**
- Time-series plots showing:
  - Position tracking error (meters)
  - Control effort proxy (Σ||u||²)
  - Asteroids within sensing radius
- Saved to `outputs/timeseries.png`

#### 7. **Metrics Comparison**
- 20-trial Monte Carlo benchmark
- Random asteroid configurations each trial
- Bar charts with error bars:
  - **Success rate** (%)
  - **Goal reached rate** (%)
  - **Collision-free rate** (%)
  - **Fuel burn** (Σ||u||²) with std dev
  - **Tracking error** (m) with std dev
  - **Traversal time** (steps) vs. max horizon
- Summary table with all metrics

## Environment & Dynamics

### Spacecraft Model
**State**: $[x, y, \theta, v_x, v_y]$
- $(x, y)$ — position
- $\theta$ — heading
- $(v_x, v_y)$ — velocity

**Control Input**: $[a_x, a_y]$ acceleration (clamped to $|a| \leq A_{limit}$)

**Continuous Dynamics**:
$$\dot{x} = v_x, \quad \dot{y} = v_y$$
$$\dot{v_x} = a_x, \quad \dot{v_y} = a_y$$
$$\dot{\theta} = \frac{v_y}{x + \epsilon} \text{ or similar heading update}$$

**Integration**: RK4 (4th-order Runge-Kutta)

### Asteroid Environment
- Circular obstacles with fixed positions (or slow drift)
- Collision detection: euclidean distance to obstacle center
- Sensing: Returns only asteroids within radius $R_{limit}$
- Visualization: Plotted with spacecraft, goal, and sensing circle overlay

## Sensing Model (`utils/sensing.py`)

**Limited Visibility**:
```python
sensed = [a for a in asteroids if distance(spacecraft_pos, a.pos) <= R_LIMIT]
```

- Controllers receive only sensed asteroids
- Must extrapolate or maintain nominal path for distant obstacles
- Simulates realistic partial observability

## Metrics Evaluation (`utils/metrics.py`)

### Per-Trial Metrics
- **Success**: Goal reached within 2.0m threshold
- **Collision**: Any time spacecraft intersects asteroid
- **Tracking error**: $\|x_{actual} - x_{nominal}\|$
- **Control effort**: $\sum_t \|u_t\|^2$
- **Traversal time**: Steps to goal (or max $T$)
- **Progress**: $(1 - d_{final}/d_{initial}) \times 100\%$

### Aggregated Statistics (20 trials)
- Mean and standard deviation
- Success/collision/goal-reached rates (%)

## Getting Started

### Prerequisites
```
numpy
jax / jax.numpy
matplotlib
ipywidgets
scipy (for LQR solver)
```

### Installation
```bash
pip install -r requirements.txt
```

### Running the Simulation
1. Open **Jupyter Lab** or **VS Code** notebook interface
2. Load `notebooks/simulation.ipynb`
3. Execute cells in order:
   - Cell 1: Imports & Setup
   - Cells 2–4: Explore environment and nominal trajectory
   - Cells 5–6: Run controllers with sensing
   - Cells 7–8: Visualize results
   - Cells 9–15: Benchmark and compare metrics

**Expected Runtime**: ~2–5 minutes (depends on CPU, iLQR iterations, trial count)

### Output Files
Generated in `outputs/`:
- `timeseries.png` — Tracking error, effort, sensed asteroids vs. time
- `metric_success_rate.png` — Success % per controller
- `metric_goal_reached.png` — Goal reached % per controller
- `metric_collision_free.png` — Collision-free % per controller
- `metric_fuel_burn.png` — Fuel burn (Σ||u||²) with error bars
- `metric_tracking_error.png` — Mean tracking error (m)
- `metric_traversal_time.png` — Traversal time (steps) vs. horizon

## Key Results (Example)

| Controller | Success | Fuel Burn | Track Err (m) | Time (steps) |
|---|---|---|---|---|
| PID (position) | 75% | 12.5 ± 2.1 | 0.85 ± 0.4 | 48.2 ± 3.5 |
| PID (full-state) | 85% | 10.2 ± 1.8 | 0.62 ± 0.3 | 49.1 ± 2.8 |
| State Feedback | 95% | 8.5 ± 1.2 | 0.35 ± 0.2 | 50.0 ± 0.8 |
| Inf-Horizon LQR | 92% | 7.8 ± 1.5 | 0.28 ± 0.2 | 49.5 ± 1.2 |

*Note: Results vary with random seed and asteroid configurations.*

## Design Insights

1. **Nominal Trajectory Power**: Even simple controllers track well when given a reference path
2. **Limited Sensing Challenge**: Partial observability requires either:
   - Aggressive tracking of visible obstacles
   - Maintaining nominal path for unobserved regions
3. **Control Complexity Trade-off**:
   - PID: Simple, interpretable, but suboptimal
   - LQR: Theoretically optimal, lower fuel, but requires linearization
4. **Tuning Sensitivity**: Gains (Kp, Ki, Kd) heavily influence success rate

## Future Extensions

- [ ] Add **Control Barrier Functions (CBF)** for guaranteed collision avoidance
- [ ] Implement **robust control** against asteroid position uncertainty
- [ ] Extend to **3D spacecraft** with attitude dynamics
- [ ] Add **obstacle prediction** for non-stationary asteroids
- [ ] Explore **learning-based** controllers (RL/imitation learning)
- [ ] Support **multi-agent** scenarios

## References

- iLQR: Todorov & Li (2005), "A Generalized Path Integral Control"
- LQR: Bertsekas (1995), "Dynamic Programming and Optimal Control"
- CBF: Ames et al. (2017), "Control Barrier Functions"

---

*This project is part of ME548: Trajectory Optimization & Control*
