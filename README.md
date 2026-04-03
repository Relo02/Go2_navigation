# Anubi — Go2 Nav2 Autonomy Stack

Minimal, clean, production-oriented ROS 2 Humble autonomy stack for the **Unitree Go2**
quadruped robot.  Navigation uses **Nav2**; localisation uses **slam_toolbox** (2D SLAM).
Everything is self-contained: no upstream workspace needs to be sourced at runtime.

---

## Repository Layout

```
src/
├── unitree_api/              # Unitree Go2 API request/response messages
├── unitree_go/               # Go2 hardware messages (SportModeState, LowState, …)
├── go2_description/          # URDF/xacro: body + sensor frames
├── robot_common_interfaces/  # Shared message types and robot adapter contract
├── go2_bringup/              # Go2 hardware adapter (go2_hw_bridge) + launch files
├── robot_nav/                # Robot-agnostic Nav2 + slam_toolbox config/launch
├── robot_safety/             # Safety-stop placeholder
├── sensor_models/            # Reusable Gazebo sensor xacro macros (L1 LiDAR)
├── go2_sim/                  # Go2 Gazebo simulation URDF (planar_move + L1 LiDAR + IMU)
├── sim_worlds/               # World files + SDF obstacle models
├── sim_scenarios/            # Runtime obstacle spawn/remove CLI + scenario manager
├── d1_sim/                   # AgiBot D1 Ultra simulation stub (TODO(agibot))
└── robot_sim/                # Simulation bringup launch (sim_bringup.launch.py)
```

### TF tree (full chain)

```
map  ──(slam_toolbox)──►  odom  ──(go2_hw_bridge)──►  base_footprint
                                                             └─(URDF)─►  base_link
                                                                              ├──►  lidar_link
                                                                              └──►  imu_link
```

### Data flow

```
Go2 hardware
  │
  ├── /sportmodestate  [SportModeState, ~50 Hz]
  │       └──►  go2_hw_bridge  ──►  /odom + TF(odom→base_footprint)
  │
  └── /unilidar/cloud  [PointCloud2]
          ├──►  pointcloud_to_laserscan  ──►  /scan
          │         └──►  slam_toolbox  ──►  /map + TF(map→odom)
          └──►  Nav2 costmaps (obstacle layer, direct PointCloud2)

Nav2  ──►  /cmd_vel  [Twist]
               └──►  go2_hw_bridge  ──►  /api/sport/request  ──►  Go2 hardware
```

---

## Dependencies

All packages are in this workspace — no upstream workspace needed at runtime.

| Dependency | Install |
|---|---|
| ROS 2 Humble base | `apt install ros-humble-ros-base` |
| Nav2 | `apt install ros-humble-navigation2 ros-humble-nav2-bringup` |
| slam_toolbox (LGPL-2.1) | `apt install ros-humble-slam-toolbox` |
| pointcloud_to_laserscan (BSD-3-Clause) | `apt install ros-humble-pointcloud-to-laserscan` |
| robot_state_publisher, xacro | `apt install ros-humble-robot-state-publisher ros-humble-xacro` |
| nlohmann/json (MIT) | `apt install nlohmann-json3-dev` |
| rviz2 | `apt install ros-humble-rviz2` |

> **Assumption A1:** The Go2 robot's internal ROS2 bridge is running and publishing:
> - `/sportmodestate` — `unitree_go/SportModeState` (proprioceptive state, ~50 Hz)
> - `/lowstate` — `unitree_go/LowState` (low-level state; used for battery data)
> - `/unilidar/cloud` — `sensor_msgs/PointCloud2` (3D LiDAR)
>
> `/lowstate` is used only for battery reporting in `RobotStatus`.  If the bridge
> does not publish it, navigation still works but `battery_soc` and `battery_voltage`
> in `/robot_status` will read 0.
>
> **Assumption A2:** The Go2 bridge subscribes to `/api/sport/request` for motion commands.
>
> **Assumption A3:** `slam_toolbox` is LGPL-2.1.  Under dynamic linking (colcon default),
> this is compatible with proprietary application code.

### SLAM Choice Rationale

**slam_toolbox** (LGPL-2.1) was selected over alternatives because:
- Native integration with Nav2 lifecycle and map server
- Supports online mapping **and** map-serialise-then-localise workflow
- Async mode decouples scan processing from the Nav2 control loop
- Loop closure for long-session reliability in industrial environments
- Actively maintained as part of the Nav2 ecosystem
- LGPL-2.1 is compatible with industrial deployment via dynamic linking

---

## How to Build

```bash
# Install system dependencies
sudo apt install \
    ros-humble-navigation2 ros-humble-nav2-bringup \
    ros-humble-slam-toolbox \
    ros-humble-pointcloud-to-laserscan \
    ros-humble-robot-state-publisher ros-humble-xacro \
    ros-humble-rviz2 \
    nlohmann-json3-dev

# Build
source /opt/ros/humble/setup.bash
cd /ws/anubi
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
source install/setup.bash
```

---

## How to Run on a Real Go2

### Step 1 — Start the Go2 internal bridge on the robot

The Unitree Go2 internal software must publish:
- `/sportmodestate`  (`unitree_go/SportModeState`)
- `/unilidar/cloud`  (`sensor_msgs/PointCloud2`)

And subscribe to:
- `/api/sport/request`  (`unitree_api/Request`)

Refer to the Unitree Go2 ROS 2 SDK documentation for activating the bridge.

### Step 2 — Map the environment (first time only)

```bash
source /opt/ros/humble/setup.bash
source /ws/anubi/install/setup.bash

# Launch in mapping mode (default)
ros2 launch go2_bringup go2_bringup.launch.py slam_mode:=mapping

# Drive the robot around manually to build the map, then save it:
ros2 service call /slam_toolbox/serialize_map \
    slam_toolbox/srv/SerializePoseGraph \
    "{filename: '/ws/anubi/src/robot_nav/maps/my_map'}"
```

### Step 3 — Navigate with a saved map

```bash
ros2 launch go2_bringup go2_bringup.launch.py \
    slam_mode:=localization \
    map:=/ws/anubi/src/robot_nav/maps/my_map
```

### Step 4 — Send a goal in RViz

1. RViz opens automatically (set `rviz:=false` to disable).
2. Click **"2D Goal Pose"** in the toolbar.
3. Click + drag on the map to set position and orientation.
4. The robot plans a path and drives autonomously.

### Optional — Mapping mode without goal (SLAM only, no navigation)

```bash
ros2 launch go2_bringup go2_bringup.launch.py slam_mode:=mapping rviz:=true
```

---

## How to Port to AgiBot D1 Ultra

1. Create `agibot_d1_bringup` package, mirroring `go2_bringup`.
2. Implement `d1_hw_bridge.cpp`:
   - Subscribe to the D1 Ultra state topic → publish `/odom` + TF `odom → base_footprint`
   - Subscribe to `/cmd_vel` → convert to D1 Ultra motion API
3. Create `agibot_d1_description` with the same frame names: `base_footprint`, `base_link`, `lidar_link`, `imu_link`.
4. Create `agibot_d1_bringup/config/d1_params.yaml` with D1-specific topic names.
5. Use `go2_bringup/launch/go2_bringup.launch.py` as a template.
6. **Zero changes** required in `robot_nav`, `slam_toolbox_params.yaml`, or `robot_common_interfaces`.

---

## How to Run in Gazebo Simulation

The simulation layer is fully implemented. **No hardware required.**
The Gazebo `libgazebo_ros_planar_move` plugin replaces `go2_hw_bridge`,
providing the identical `/cmd_vel` + `/odom` + TF interface to Nav2.

```bash
source /opt/ros/humble/setup.bash
source /ws/anubi/install/setup.bash

# Full simulation — mapping mode, with GUI and RViz
ros2 launch robot_sim sim_bringup.launch.py

# Warehouse world, localization with a saved map, no GUI
ros2 launch robot_sim sim_bringup.launch.py \
    world:=$(ros2 pkg prefix sim_worlds)/share/sim_worlds/worlds/warehouse.world \
    slam_mode:=localization \
    map:=/ws/anubi/src/robot_nav/maps/my_map \
    gui:=false
```

### Runtime obstacle management

```bash
# Spawn a crate at (2, 1)
ros2 run sim_scenarios spawn_obstacle \
    --name box_1 --model obstacle_box --x 2.0 --y 1.0 --z 0.5

# Delete it
ros2 run sim_scenarios spawn_obstacle --delete box_1

# Load a pre-defined scenario (crowded_room or narrow_corridor)
ros2 run sim_scenarios scenario_manager \
    --ros-args -p scenario_file:=crowded_room

# Restart Nav2 without restarting Gazebo
ros2 launch robot_sim sim_nav.launch.py
```

### Known simulation limits

- Gazebo Classic 11 (EOL Jan 2025) — functional on Ubuntu 22.04 / Humble, no new patches.
  Port to Gazebo Fortress (`ros_gz`) when long-term maintenance is required.
- Robot locomotion uses `libgazebo_ros_planar_move` (holonomic approximation, no leg dynamics).
- AgiBot D1 Ultra sim (`d1_sim/`) is a stub — chassis dimensions are placeholders.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| `go2_hw_bridge` single node for both odom and cmd_vel | Minimal Go2-specific surface area |
| `SportModeState` → `odom/TF` (not Point-LIO) | Self-contained; Go2's internal state estimator is accurate short-term; slam_toolbox corrects long-term drift |
| `pointcloud_to_laserscan` → slam_toolbox | slam_toolbox needs 2D scan; Nav2 costmaps still use full 3D cloud |
| `odom → base_footprint` (not base_link) | Standard Nav2 2D convention; slam_toolbox tracks `base_footprint` |
| cmd_vel watchdog (0.5 s) | Stops robot if Nav2 crashes; prevents runaway |
| nlohmann/json (system package, MIT) | No vendoring; standard Go2 SDK serialisation format |
| slam_toolbox LGPL-2.1 | Standard Nav2 SLAM; LGPL OK under dynamic linking |

---

## TODO Markers in Code

- `TODO(agibot)` — changes for AgiBot D1 Ultra porting (implement `d1_sim/urdf/d1_sim.urdf.xacro` and `agibot_d1_bringup`)
- `TODO(sim)` — Gazebo plugin stubs remaining in `go2_description/urdf/go2.urdf.xacro` (hardware URDF); the simulation itself is fully working via `go2_sim/`
- `TODO(safety)` — `tilt_monitor` node (IMU-based fall detection); `velocity_limiter` is already implemented
- `TODO(localization)` — where to wire in map-based localization improvements
