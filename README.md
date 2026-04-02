# A* + MPC Planner for Go2 Quadruped

Local navigation stack for the Unitree Go2 quadruped robot.  
Two ROS 2 nodes work in sequence: **A\* path planning** produces a collision-free waypoint path; **MPC tracking** follows that path while performing real-time obstacle avoidance.

---

## Architecture

```
LiDAR (/lidar/points_filtered)
        │
        ▼
┌─────────────────────────────┐
│  FixedGaussianGridMap        │  10 m × 10 m grid centred on robot
│  (rebuilt every replan tick) │  P(obstacle) = 1 − Φ(d_min / σ)
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  AStarPlanner                │  8-connected grid search
│  Rolling-horizon local goal  │  Soft + hard obstacle cost
└─────────────┬───────────────┘
              │  /a_star/path  (nav_msgs/Path)
              ▼
┌─────────────────────────────┐
│  MPCTracker                  │  N=30, dt=0.1 s  →  3 s horizon
│  2-D kinematic model         │  CasADi/IPOPT solver
│  Logistic obstacle barrier   │
└─────────────┬───────────────┘
              │  /mpc/next_setpoint  (geometry_msgs/PoseStamped)
              ▼
         Go2 controller
```

---

## How It Works

### Stage 1 — Gaussian Occupancy Grid

`FixedGaussianGridMap` builds a square grid (default 10 m × 10 m, 0.25 m/cell) centred on the robot every replanning tick.  
For each cell, the occupancy probability is:

```
P(cell) = 1 − Φ(d_min / σ)
```

where `d_min` is the distance to the nearest LiDAR point inside the grid and `σ` is the Gaussian spread (default 0.4 m).  
The map is rebuilt from scratch on every tick — no accumulation across steps.

### Stage 2 — Rolling-Horizon A*

`AStarPlanner` runs a standard A\* on the occupancy grid with an 8-connected neighbourhood.

**Local goal selection:**
- If the global goal lies *inside* the current grid, A\* targets it directly.
- If the global goal is *outside* the grid, the planner intersects the ray (robot → goal) with the grid boundary and targets that boundary cell.  
  This lets the robot advance toward a distant goal one grid-width at a time.

**Cost function:**
```
g(n→n') = move_cost × reso × (1 + w_obs × P(n'))
```
Cells above `obstacle_threshold` (default 0.5) are treated as hard obstacles (infinite cost).  
Cells below the threshold incur a soft cost proportional to their occupancy probability, pushing the path away from obstacles.

### Stage 3 — 2-D Kinematic MPC

`MPCTracker` solves a nonlinear program (NLP) over a 3-second horizon using CasADi/IPOPT.

#### Dynamics model

The Go2 is treated as a **holonomic kinematic robot** — it can be commanded with body-frame velocities in any direction (forward, lateral, yaw):

```
State:   x = [px, py, yaw]       (3-D)
Control: u = [vx, vy, ω]         (body-frame velocities)
```

World-frame propagation via rotation matrix R(yaw):

```
px_{k+1}  = px_k + (vx_k · cos(yaw_k) − vy_k · sin(yaw_k)) · dt
py_{k+1}  = py_k + (vx_k · sin(yaw_k) + vy_k · cos(yaw_k)) · dt
yaw_{k+1} = yaw_k + ω_k · dt
```

Forward Euler integration (dt = 0.1 s).

#### Objective function

```
J = Σ_{k=0}^{N-1} [ eₖᵀ Q eₖ  +  uₖᵀ R uₖ  +  R_jerk ‖Δuₖ‖²  +  J_obs(xₖ) ]
    + e_N^T Q_T e_N  +  J_obs(x_N)
```

| Term | Weight | Purpose |
|---|---|---|
| `eₖᵀ Q eₖ` | Q_xy=20, Q_yaw=0.5 | Track A\* reference path |
| `uₖᵀ R uₖ` | R_vel=1.0, R_ω=0.5 | Penalise control effort |
| `R_jerk ‖Δuₖ‖²` | 0.2 | Smooth velocity transitions |
| `J_obs` | 500 per point | Repel from obstacles |
| Terminal `Q_T` | 50 × Q | Drive to end of horizon |

#### Obstacle avoidance barrier

For each LiDAR point `p_j` and predicted state `x_k`:

```
J_obs(x_k, p_j) = W / (1 + exp(α · (dist(x_k, p_j) − r)))
```

- `W = 500`, `α = 8` (steepness), `r = 0.8 m` (safety radius)  
- Non-zero gradient everywhere — IPOPT always has a slope to climb away from obstacles even before the safety boundary is reached  
- Bounded — no infeasibility if the robot starts inside `r`

Up to 15 nearest LiDAR points within 3 m are used per solve; the rest are replaced with far sentinels so the NLP structure never changes between calls.

#### Reference trajectory

A reference `x_ref[k]` is built by advancing along the A\* path at `v_ref = 0.5 m/s` from the closest waypoint to the robot.  
At each step `k`, the reference position is linearly interpolated along the path arc, and the reference yaw is the tangent direction of the segment.

#### Setpoint selection

After solving, the node walks the predicted trajectory `x_pred[0..N]` and picks the first predicted state that is at least `lookahead_dist = 0.5 m` away from the robot.  
If the entire horizon stays closer (near-goal case), it steers toward the last A\* waypoint directly.

---

## ROS 2 Interface

### `a_star_node`

| Direction | Topic | Type | Description |
|---|---|---|---|
| Sub | `/go2/pose` | `PoseStamped` | Robot pose |
| Sub | `/lidar/points_filtered` | `PointCloud2` | LiDAR hits (world frame) |
| Sub | `/global_goal` | `PoseStamped` | Runtime goal override |
| Pub | `/a_star/path` | `Path` | Local A\* waypoint path |
| Pub | `/a_star/local_goal` | `PoseStamped` | Current local grid target |
| Pub | `/a_star/occupancy_grid` | `OccupancyGrid` | Gaussian map (Foxglove) |
| Pub | `/a_star/grid_raw` | `Float32MultiArray` | Raw grid + metadata |

### `mpc_node`

| Direction | Topic | Type | Description |
|---|---|---|---|
| Sub | `/go2/pose` | `PoseStamped` | Robot pose |
| Sub | `/lidar/points_filtered` | `PointCloud2` | LiDAR hits (world frame) |
| Sub | `/a_star/path` | `Path` | A\* path |
| Pub | `/mpc/next_setpoint` | `PoseStamped` | Lookahead setpoint for controller |
| Pub | `/mpc/predicted_path` | `Path` | Full N-step predicted trajectory |
| Pub | `/mpc/diagnostics` | `Float64MultiArray` | `[success, cost, solve_ms, avg_ms, fails]` |

---

## Parameters (`config/planner_params.yaml`)

### A* node

| Parameter | Default | Description |
|---|---|---|
| `grid_reso` | 0.25 m | Cell size |
| `grid_half_width` | 5.0 m | Half-extent of local grid |
| `grid_std` | 0.4 m | Gaussian spread σ |
| `obstacle_threshold` | 0.5 | Hard obstacle cutoff |
| `obstacle_cost_weight` | 10.0 | Soft cost multiplier |
| `replan_rate_hz` | 2.0 | Replanning frequency |
| `goal_reached_radius` | 0.3 m | Stop replanning within this distance |
| `max_lidar_range` | 6.0 m | LiDAR range filter (from robot) |

### MPC node

| Parameter | Default | Description |
|---|---|---|
| `mpc_N` | 30 | Prediction horizon steps |
| `mpc_dt` | 0.1 s | Step duration (horizon = 3 s) |
| `mpc_rate_hz` | 2.0 | Solve frequency |
| `mpc_vx_max` | 1.0 m/s | Max forward velocity |
| `mpc_vy_max` | 0.5 m/s | Max lateral velocity |
| `mpc_omega_max` | 1.5 rad/s | Max yaw rate |
| `mpc_v_ref` | 0.5 m/s | Reference cruise speed |
| `mpc_Q_xy` | 20.0 | Position tracking weight |
| `mpc_Q_yaw` | 0.5 | Yaw tracking weight |
| `mpc_Q_terminal` | 50.0 | Terminal cost multiplier |
| `mpc_W_obs_sigmoid` | 500.0 | Obstacle barrier weight |
| `mpc_obs_alpha` | 8.0 | Barrier steepness [1/m] |
| `mpc_obs_r` | 0.8 m | Safety radius |
| `mpc_lookahead_dist` | 0.5 m | Setpoint lookahead distance |

---

## Dependencies

- ROS 2 (Humble or Foxy)
- `casadi` — symbolic NLP formulation
- `ipopt` — interior-point NLP solver
- `numpy`, `scipy`
- `sensor_msgs_py`

---

## Build & Run

```bash
# Build
cd ~/go2/anubi
colcon build --packages-select a_star_mpc_planner
source install/setup.bash

# Run A* node
ros2 run a_star_mpc_planner a_star_node \
  --ros-args -p goal_x:=10.0 -p goal_y:=5.0

# Run MPC node
ros2 run a_star_mpc_planner mpc_node

# Override goal at runtime
ros2 topic pub /global_goal geometry_msgs/PoseStamped \
  "{pose: {position: {x: 10.0, y: 5.0, z: 0.0}}}"
```

---

## File Overview

```
a_star_mpc_planner/
├── a_star_node.py        — ROS 2 node: LiDAR → grid → A* → /a_star/path
├── mpc_node.py           — ROS 2 node: path + LiDAR → MPC → setpoint
├── a_star_planner.py     — Pure A* algorithm on occupancy grid
├── mpc_tracker.py        — CasADi/IPOPT MPC, dynamics, NLP build
├── gaussian_grid_map.py  — Fixed Gaussian occupancy grid map
└── config/
    └── planner_params.yaml
```
