# robot_sim

Gazebo Classic (Gazebo 11) simulation bringup for the Go2 autonomy stack.

---

## Overview

`sim_bringup.launch.py` starts a complete simulation:

```
Gazebo Classic  +  robot_state_publisher  +  spawn_entity
    +  slam.launch.py      (pointcloud_to_laserscan + slam_toolbox, use_sim_time=true)
    +  nav2.launch.py      (Nav2 navigation stack, use_sim_time=true)
    +  RViz2               (optional, reuses robot_nav/rviz/nav2.rviz)
```

`go2_hw_bridge` is **not** started.  Gazebo's `libgazebo_ros_planar_move` plugin
(inside `go2_sim/urdf/go2_sim.urdf.xacro`) provides an identical
`/cmd_vel` + `/odom` + TF(`odom → base_footprint`) interface to Nav2.

TF chain in simulation (identical to real robot):

```
map ─(slam_toolbox)─► odom ─(planar_move)─► base_footprint
                                                  └─(URDF)─► base_link
                                                                  ├──► lidar_link
                                                                  └──► imu_link
```

All Nav2 configuration is shared with the real robot (`robot_nav/`).

---

## Quick Start

```bash
source /opt/ros/humble/setup.bash
source /ws/anubi/install/setup.bash

# Full sim — navigation_empty world, mapping mode, Gazebo GUI + RViz
ros2 launch robot_sim sim_bringup.launch.py

# Warehouse world, localization with a saved map, no GUI
ros2 launch robot_sim sim_bringup.launch.py \
    world:=$(ros2 pkg prefix sim_worlds)/share/sim_worlds/worlds/warehouse.world \
    slam_mode:=localization \
    map:=/ws/anubi/src/robot_nav/maps/my_map \
    gui:=false

# Restart Nav2 without restarting Gazebo or SLAM
ros2 launch robot_sim sim_nav.launch.py
```

---

## Launch Arguments

### `sim_bringup.launch.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `world` | `navigation_empty.world` | Path to Gazebo world file |
| `robot` | `go2` | Robot model: `go2` or `d1` |
| `slam_mode` | `mapping` | `mapping` or `localization` |
| `map` | `''` | Path to `.posegraph` file (localization mode only) |
| `use_rviz` | `true` | Launch RViz2 |
| `slam_params` | robot_nav default | slam_toolbox params YAML override |
| `nav_params` | robot_nav default | Nav2 params YAML override |
| `spawn_z` | `0.05` m | Robot Z spawn height (slight ground clearance) |
| `gui` | `true` | Show Gazebo gzclient GUI |

### `sim_nav.launch.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `use_sim_time` | `true` | Use simulation clock |
| `params_file` | robot_nav default | Nav2 params YAML override |
| `use_rviz` | `true` | Launch RViz2 |

---

## Obstacle Management

```bash
# Spawn a crate at (2, 1)
ros2 run sim_scenarios spawn_obstacle \
    --name box_1 --model obstacle_box --x 2.0 --y 1.0 --z 0.5

# Delete it
ros2 run sim_scenarios spawn_obstacle --delete box_1

# Load a pre-defined scenario (crowded_room or narrow_corridor)
ros2 run sim_scenarios scenario_manager \
    --ros-args -p scenario_file:=crowded_room
```

---

## Known Limitations

- Gazebo Classic 11 (EOL Jan 2025) — functional on Ubuntu 22.04 / Humble.
  Port to Gazebo Fortress (`ros_gz`) when long-term maintenance is required.
- Robot locomotion uses `libgazebo_ros_planar_move` (holonomic approximation; no leg dynamics).
- AgiBot D1 Ultra model (`d1_sim/`) is a stub — chassis dimensions are placeholders.
  Launch with `robot:=d1` is supported once `d1_sim/urdf/d1_sim.urdf.xacro` is completed.
- `config/sim_params.yaml` is intentionally empty — `use_sim_time=true` is set via
  launch argument, and no per-node simulation overrides are needed.
