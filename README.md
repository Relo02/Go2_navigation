# A* + MPC Planner for Go2 Quadruped

Local navigation stack for the Unitree Go2 quadruped robot.  
Four ROS 2 nodes work in sequence: an **odometry bridge** converts EKF output to the shared pose topic; an **A\* planner** builds a rolling Gaussian occupancy map and produces a collision-free waypoint path; an **MPC tracker** follows that path while performing real-time obstacle avoidance; a **setpoint controller** converts the MPC lookahead pose into body-frame `/cmd_vel` commands.

---

## System Architecture

```mermaid
flowchart TD
    ODOM["/odom/raw\n(Odometry)"]
    LIDAR["/lidar/points_filtered\n(PointCloud2)"]
    GOAL["/global_goal\n(PoseStamped)"]

    BRIDGE["odom_to_pose_node\nBridge: Odometry → PoseStamped"]
    ASTAR["a_star_node\nGaussian grid map + A* search"]
    MPC["mpc_node\n2-D kinematic MPC  (CasADi/IPOPT)"]
    CMD["setpoint_to_cmd_vel_node\nP-controller  →  /cmd_vel"]

    ODOM --> BRIDGE
    BRIDGE -- "/go2/pose" --> ASTAR
    BRIDGE -- "/go2/pose" --> MPC
    BRIDGE -- "/go2/pose" --> CMD
    LIDAR --> ASTAR
    LIDAR --> MPC
    GOAL --> ASTAR

    ASTAR -- "/a_star/path" --> MPC
    ASTAR -- "/a_star/occupancy_grid" --> VIZ["Foxglove / RViz2"]
    MPC -- "/mpc/next_setpoint" --> CMD
    MPC -- "/mpc/predicted_path" --> VIZ
    CMD -- "/cmd_vel" --> GO2["Go2 Controller\n(CHAMP / spot_ros2)"]
```

---

## Mathematical Description

### Stage 1 — Gaussian Occupancy Grid (`FixedGaussianGridMap`)

A square local grid of side $2 \cdot h_w$ (default $10\,\text{m}$) centred on the robot is rebuilt from scratch at every replanning tick.  
Each grid cell stores the probability that it is occupied, derived from the minimum distance $d_{\min}$ to any LiDAR return projected into the cell:

$$P(\text{cell}) = 1 - \Phi\!\left(\frac{d_{\min}}{\sigma}\right)$$

where $\Phi$ is the standard normal CDF and $\sigma$ is the Gaussian spread parameter (default $\sigma = 0.05\,\text{m}$, yielding sharp, narrow inflated obstacles).  
The grid has no memory — it is always built from the latest scan alone.

**Grid parameters**

| Symbol | Parameter | Default |
|---|---|---|
| $h_w$ | `grid_half_width` | $5.0\,\text{m}$ |
| $\Delta$ | `grid_reso` | $0.25\,\text{m/cell}$ |
| $\sigma$ | `grid_std` | $0.05\,\text{m}$ |
| $P_{\text{thresh}}$ | `obstacle_threshold` | $0.45$ |

---

### Stage 2 — Rolling-Horizon A\* (`AStarPlanner`)

A\* runs on the 8-connected occupancy grid. The grid is always centred on the robot, so A\* is replanned from scratch every tick at `replan_rate_hz` (default $10\,\text{Hz}$).

#### Local goal selection

$$\text{local goal} = \begin{cases} \text{global goal cell} & \text{if global goal} \in \text{grid} \\ \text{ray}(\text{robot} \to \text{goal}) \cap \partial\text{grid} & \text{otherwise} \end{cases}$$

The ray is clipped to the grid boundary using parametric intersection, allowing the robot to advance toward a distant goal one grid-width at a time.  
If the target cell is occupied, a BFS finds the nearest free cell.

#### Edge traversal cost

$$g(n \to n') = c_{\text{move}} \cdot \Delta \cdot \bigl(1 + w_{\text{obs}} \cdot P(n')\bigr)$$

where $c_{\text{move}} = 1$ for cardinal moves and $\sqrt{2}$ for diagonal moves.  
Cells with $P \ge P_{\text{thresh}}$ are treated as hard obstacles (infinite cost); cells below the threshold incur a soft penalty proportional to their occupancy probability, steering the path away from nearby obstacles without requiring them to be impassable.

#### Heuristic

$$h(n) = \sqrt{(i_x - g_x)^2 + (i_y - g_y)^2} \cdot \Delta$$

Euclidean distance in world units — admissible and consistent on the uniform grid.

---

### Stage 3 — 2-D Kinematic MPC (`MPCTracker`)

The MPC tracker solves a Nonlinear Program (NLP) symbolically compiled by **CasADi** and solved by **IPOPT** at every control tick (`mpc_rate_hz = 10\,\text{Hz}`).  
The NLP structure is built **once** at startup; only the numeric parameter values (initial state, reference trajectory, obstacle positions) are updated each call.

#### State and control

$$\mathbf{x}_k = \begin{bmatrix} p_{x,k} \\ p_{y,k} \\ \psi_k \end{bmatrix}, \qquad \mathbf{u}_k = \begin{bmatrix} v_{x,k} \\ v_{y,k} \\ \omega_k \end{bmatrix}$$

The Go2 is modelled as a **holonomic kinematic robot** — it can be commanded with independent forward, lateral and yaw-rate velocities simultaneously.

#### Discrete-time dynamics (Forward Euler, $\Delta t = 0.1\,\text{s}$)

$$\mathbf{x}_{k+1} = f(\mathbf{x}_k, \mathbf{u}_k) = \begin{bmatrix} p_{x,k} + (v_{x,k}\cos\psi_k - v_{y,k}\sin\psi_k)\,\Delta t \\ p_{y,k} + (v_{x,k}\sin\psi_k + v_{y,k}\cos\psi_k)\,\Delta t \\ \psi_k + \omega_k\,\Delta t \end{bmatrix}$$

#### Objective function

$$\min_{\mathbf{u}_{0\ldots N-1}} \; J = \underbrace{\mathbf{e}_N^\top \mathbf{Q}_T \mathbf{e}_N + J_{\text{obs}}(\mathbf{x}_N)}_{\text{terminal}} + \sum_{k=0}^{N-1} \Bigl[ \underbrace{\mathbf{e}_k^\top \mathbf{Q} \mathbf{e}_k}_{\text{tracking}} + \underbrace{\mathbf{u}_k^\top \mathbf{R} \mathbf{u}_k}_{\text{effort}} + \underbrace{R_{\text{jerk}}\|\Delta\mathbf{u}_k\|^2}_{\text{smoothness}} + \underbrace{J_{\text{obs}}(\mathbf{x}_k)}_{\text{obstacles}} \Bigr]$$

where $\mathbf{e}_k = \mathbf{x}_k - \mathbf{x}_{\text{ref},k}$ is the tracking error and

$$\mathbf{Q} = \text{diag}(Q_{xy},\; Q_{xy},\; Q_\psi), \qquad \mathbf{Q}_T = Q_T \cdot \mathbf{Q}, \qquad \mathbf{R} = \text{diag}(R_v,\; R_v,\; R_\omega)$$

| Term | Weight | Config key | Tuned value |
|---|---|---|---|
| $Q_{xy}$ | Position tracking | `mpc_Q_xy` | $200$ |
| $Q_\psi$ | Yaw tracking | `mpc_Q_yaw` | $1$ |
| $Q_T$ | Terminal multiplier | `mpc_Q_terminal` | $100$ |
| $R_v$ | Linear velocity effort | `mpc_R_vel` | $1$ |
| $R_\omega$ | Angular velocity effort | `mpc_R_omega` | $0.5$ |
| $R_{\text{jerk}}$ | Smoothness | `mpc_R_jerk` | $0.5$ |

#### Obstacle barrier

For each of the $M$ selected LiDAR points $\mathbf{p}_j$ and each predicted state $\mathbf{x}_k$, a logistic sigmoid barrier is added to the cost:

$$J_{\text{obs}}(\mathbf{x}_k) = \sum_{j=1}^{M} \frac{W}{1 + e^{\alpha \left(d(\mathbf{x}_k, \mathbf{p}_j) - r\right)}}$$

where $d(\mathbf{x}_k, \mathbf{p}_j) = \sqrt{(p_{x,k} - p_{j,x})^2 + (p_{y,k} - p_{j,y})^2 + \epsilon}$.

Numerically, the implementation uses the stable form $\tfrac{W}{2}(1 - \tanh(\tfrac{\alpha}{2}(d - r)))$, which avoids $e^{\infty}$ overflow.

| Symbol | Config key | Value | Role |
|---|---|---|---|
| $W$ | `mpc_W_obs_sigmoid` | $200$ | Barrier height |
| $\alpha$ | `mpc_obs_alpha` | $4\,\text{m}^{-1}$ | Steepness |
| $r$ | `mpc_obs_r` | $0.55\,\text{m}$ | Safety radius |
| $M$ | `mpc_max_obs_constraints` | $12$ | Points per solve |

Only the $M$ nearest LiDAR returns within `mpc_obs_check_radius` = $3\,\text{m}$ are used; the remainder are replaced with far sentinels at $(10^3, 10^3)$ so the NLP sparsity pattern never changes between solves.

#### Box constraints

$$-v_{x,\max} \le v_{x,k} \le v_{x,\max}, \quad -v_{y,\max} \le v_{y,k} \le v_{y,\max}, \quad -\omega_{\max} \le \omega_k \le \omega_{\max}$$

with $v_{x,\max} = 1.0\,\text{m/s}$, $v_{y,\max} = 0.5\,\text{m/s}$, $\omega_{\max} = 1.5\,\text{rad/s}$.

#### Reference trajectory construction

The reference $\{\mathbf{x}_{\text{ref},k}\}_{k=0}^{N}$ is built by advancing along the A\* path at cruise speed $v_{\text{ref}}$ from the closest waypoint to the robot:

$$s_k = \min\!\left(s_0 + v_{\text{ref}} \cdot k \cdot \Delta t,\; L_{\text{path}}\right)$$

where $s_0$ is the arc-length coordinate of the closest waypoint and $L_{\text{path}}$ is the total path arc length.  
Position $\mathbf{p}_{\text{ref},k}$ is linearly interpolated along the segment containing $s_k$; reference yaw is the tangent angle $\psi_{\text{ref},k} = \text{atan2}(\Delta y_{\text{seg}}, \Delta x_{\text{seg}})$.

Before entering the MPC, the raw A\* path is **resampled** at $\Delta s = 0.20\,\text{m}$ and **smoothed** with a moving-average kernel of width $5$ to remove cell-to-cell zig-zag jitter while preserving endpoints.

#### Warm starting

The previous solution is shifted by one step and used as the initial guess for the next solve, significantly reducing IPOPT iterations.  
On a solver failure the warm-start cache is cleared to avoid poisoning subsequent solves.

#### Setpoint extraction

After solving, the node walks the predicted trajectory $\mathbf{x}_{0\ldots N}$ and selects the first predicted state at least `mpc_lookahead_dist` = $2.0\,\text{m}$ ahead of the robot as the setpoint.  
If the entire horizon stays closer (near-goal), the last A\* waypoint is used directly.

A low-pass filter with $\alpha = 0.35$ and a maximum jump clamp of $0.30\,\text{m}$ smooths the published setpoint stream to prevent command jitter from path flicker.

---

### Stage 4 — Setpoint Controller (`setpoint_to_cmd_vel_node`)

A proportional controller running at $20\,\text{Hz}$ converts the MPC setpoint into body-frame `/cmd_vel`:

$$\begin{bmatrix} e_x \\ e_y \end{bmatrix}_{\text{body}} = \mathbf{R}(\psi)^\top \begin{bmatrix} s_x - p_x \\ s_y - p_y \end{bmatrix}$$

$$v_x^{\text{cmd}} = \text{clip}(k_{p,xy}\, e_x,\; \pm v_{x,\max}^{\text{cmd}}), \qquad v_y^{\text{cmd}} = \text{clip}(k_{p,xy}\, e_y,\; \pm v_{y,\max}^{\text{cmd}})$$

Optionally (when `enable_yaw_control: true`), the yaw error to the MPC-predicted heading is also fed through a P-controller:

$$\omega^{\text{cmd}} = \text{clip}(k_{p,\psi}\,\text{wrap}(\psi_{\text{sp}} - \psi),\; \pm\omega_{\max}^{\text{cmd}})$$

A safety timeout (`setpoint_timeout_sec = 2.0\,\text{s}`) zeroes `/cmd_vel` if no fresh setpoint arrives, preventing runaway motion.

---

## ROS 2 Interface

### `odom_to_pose_node`

| Direction | Topic | Type | Description |
|---|---|---|---|
| Sub | `/odom/raw` | `Odometry` | EKF pose from CHAMP/robot_localization |
| Pub | `/go2/pose` | `PoseStamped` | Republished pose for A\*, MPC, controller |

### `a_star_node`

| Direction | Topic | Type | Description |
|---|---|---|---|
| Sub | `/go2/pose` | `PoseStamped` | Robot pose |
| Sub | `/lidar/points_filtered` | `PointCloud2` | LiDAR hits (world frame) |
| Sub | `/global_goal` | `PoseStamped` | Runtime goal override |
| Pub | `/a_star/path` | `Path` | Local A\* waypoint path |
| Pub | `/a_star/local_goal` | `PoseStamped` | Current local grid target |
| Pub | `/a_star/occupancy_grid` | `OccupancyGrid` | Gaussian map (Foxglove/RViz2) |
| Pub | `/a_star/grid_raw` | `Float32MultiArray` | Raw grid + metadata |

### `mpc_node`

| Direction | Topic | Type | Description |
|---|---|---|---|
| Sub | `/go2/pose` | `PoseStamped` | Robot pose |
| Sub | `/lidar/points_filtered` | `PointCloud2` | LiDAR hits (world frame) |
| Sub | `/a_star/path` | `Path` | A\* path |
| Pub | `/mpc/next_setpoint` | `PoseStamped` | Lookahead setpoint |
| Pub | `/mpc/predicted_path` | `Path` | Full $N$-step predicted trajectory |
| Pub | `/mpc/diagnostics` | `Float64MultiArray` | `[success, cost, solve_ms, avg_ms, fails]` |

### `setpoint_to_cmd_vel_node`

| Direction | Topic | Type | Description |
|---|---|---|---|
| Sub | `/go2/pose` | `PoseStamped` | Robot pose |
| Sub | `/mpc/next_setpoint` | `PoseStamped` | MPC lookahead setpoint |
| Pub | `/cmd_vel` | `Twist` | Body-frame velocity commands |

---

## Parameters (`config/planner_params.yaml`)

All four nodes share a single parameter file loaded with `--params-file`.

### A\* node

| Parameter | Default | Description |
|---|---|---|
| `wait_for_goal` | `true` | Wait for goal on `/global_goal` before planning |
| `goal_x`, `goal_y` | `5.0, 5.0` | Static goal (used only when `wait_for_goal: false`) |
| `grid_reso` | `0.25 m` | Cell size |
| `grid_half_width` | `5.0 m` | Half-extent of local grid |
| `grid_std` | `0.05 m` | Gaussian spread $\sigma$ |
| `obstacle_threshold` | `0.45` | Hard obstacle cutoff |
| `obstacle_cost_weight` | `8.0` | Soft cost multiplier $w_{\text{obs}}$ |
| `replan_rate_hz` | `10.0 Hz` | Replanning frequency |
| `goal_reached_radius` | `0.30 m` | Stop replanning within this distance |
| `max_lidar_range` | `6.0 m` | LiDAR range filter (from robot) |

### MPC node

| Parameter | Default | Description |
|---|---|---|
| `mpc_N` | `30` | Prediction horizon steps |
| `mpc_dt` | `0.10 s` | Step duration (horizon = 3 s) |
| `mpc_rate_hz` | `10.0 Hz` | Solve frequency |
| `mpc_vx_max` | `1.0 m/s` | Max forward velocity |
| `mpc_vy_max` | `0.5 m/s` | Max lateral velocity |
| `mpc_omega_max` | `1.5 rad/s` | Max yaw rate |
| `mpc_v_ref` | `0.5 m/s` | Reference cruise speed $v_{\text{ref}}$ |
| `mpc_Q_xy` | `200.0` | Position tracking weight $Q_{xy}$ |
| `mpc_Q_yaw` | `1.0` | Yaw tracking weight $Q_\psi$ |
| `mpc_Q_terminal` | `100.0` | Terminal cost multiplier $Q_T$ |
| `mpc_R_vel` | `1.0` | Linear velocity effort $R_v$ |
| `mpc_R_omega` | `0.5` | Angular velocity effort $R_\omega$ |
| `mpc_R_jerk` | `0.5` | Smoothness weight $R_{\text{jerk}}$ |
| `mpc_W_obs_sigmoid` | `200.0` | Obstacle barrier weight $W$ |
| `mpc_obs_alpha` | `4.0 m⁻¹` | Barrier steepness $\alpha$ |
| `mpc_obs_r` | `0.55 m` | Safety radius $r$ |
| `mpc_max_obs_constraints` | `12` | LiDAR points per solve $M$ |
| `mpc_obs_check_radius` | `3.0 m` | Obstacle search radius |
| `mpc_lookahead_dist` | `2.0 m` | Setpoint lookahead distance |
| `mpc_warm_start` | `true` | Warm-start IPOPT from previous solution |
| `mpc_max_iter` | `100` | Max IPOPT iterations |

### Setpoint controller node

| Parameter | Default | Description |
|---|---|---|
| `cmd_rate_hz` | `20.0 Hz` | `/cmd_vel` publish frequency |
| `cmd_kp_xy` | `2.0` | Body-frame XY proportional gain $k_{p,xy}$ |
| `cmd_kp_yaw` | `1.2` | Yaw proportional gain $k_{p,\psi}$ |
| `cmd_max_vx` | `0.8 m/s` | Forward speed clamp |
| `cmd_max_vy` | `0.25 m/s` | Lateral speed clamp |
| `cmd_max_omega` | `1.0 rad/s` | Yaw-rate clamp |
| `cmd_stop_radius` | `0.2 m` | Zero `/cmd_vel` within this distance of setpoint |
| `setpoint_timeout_sec` | `2.0 s` | Safety timeout before zeroing commands |
| `enable_yaw_control` | `true` | Enable path-aligned heading control |

---

## Build

```bash
cd ~/Go2_navigation
colcon build
source install/setup.bash
```

---

## Running the Stack

All four nodes must run simultaneously. Open four terminals (all sourced with `source ~/go2/anubi/install/setup.bash`).

```bash
ros2 launch robot_sim sim_a_star_mpc.launch.py
```

For now the odometry bridge is done through Gazebo plugins, so the above command runs the full stack in simulation. In the future, the `odom_to_pose_node` will require to subscribe to the real robot's EKF output instead.

---

## Diagnostics

Monitor the MPC solver health:

```bash
ros2 topic echo /mpc/diagnostics
# data: [success(0/1), cost, solve_time_ms, avg_solve_ms, total_failures]
```

Watch the predicted horizon and occupancy grid in Foxglove Studio or RViz2:

| Topic | Type | Visualisation |
|---|---|---|
| `/a_star/occupancy_grid` | `OccupancyGrid` | Gaussian obstacle map |
| `/a_star/path` | `Path` | A\* waypoint path |
| `/mpc/predicted_path` | `Path` | MPC predicted trajectory |
| `/mpc/next_setpoint` | `PoseStamped` | Current setpoint arrow |

---

## Dependencies

- ROS 2 Humble or Foxy
- `casadi` — symbolic NLP formulation and code generation
- `ipopt` — interior-point NLP solver (via CasADi)
- `numpy`, `scipy`
- `sensor_msgs_py`

---

## File Overview

```
a_star_mpc_planner/
├── a_star_node.py              — ROS 2 node: LiDAR → Gaussian grid → A* → /a_star/path
├── a_star_planner.py           — Pure A* algorithm on occupancy grid
├── gaussian_grid_map.py        — Fixed Gaussian occupancy grid map
├── mpc_node.py                 — ROS 2 node: path + LiDAR → MPC → /mpc/next_setpoint
├── mpc_tracker.py              — CasADi/IPOPT MPC, dynamics, NLP build
├── setpoint_to_cmd_vel_node.py — P-controller: setpoint + pose → /cmd_vel
├── odom_to_pose_node.py        — Bridge: /odom/raw (Odometry) → /go2/pose (PoseStamped)
└── config/
    └── planner_params.yaml     — Shared parameters for all four nodes
```
