# robot_nav

Robot-agnostic Nav2 configuration and launch files.
**No robot-specific code is in this package.**

---

## Architecture

```
/lidar/points (PointCloud2)
    ├──► pointcloud_to_laserscan ──► /scan (LaserScan)
    │          └──► slam_toolbox ──► /map + TF(map→odom)
    └──► Nav2 local/global costmaps (obstacle detection)

/odom + TF(odom→base_footprint) ──► slam_toolbox + Nav2
Nav2 /cmd_vel ──► [go2_hw_bridge] ──► Go2 hardware
```

## Robot Physical Dimensions (Unitree Go2)

| Dimension | Trunk only | Full robot (with legs) |
|---|---|---|
| Length (X) | 0.440 m | **0.700 m** |
| Width (Y) | 0.120 m | **0.310 m** |
| Height (Z) | 0.100 m | 0.400 m (standing) |
| Standing height (ground to body centre) | — | **0.280 m** |
| LiDAR height above ground | — | **0.400 m** |
| Circumscribed radius | — | **0.383 m** |

The URDF `base_link` collision box models the trunk only (440 × 120 × 100 mm).
The Nav2 footprint parameter uses the FULL robot dimensions including legs.

## Packages Included

| Package | Licence | Role |
|---------|---------|------|
| slam_toolbox | LGPL-2.1 | 2D SLAM: map + map→odom TF |
| pointcloud_to_laserscan | BSD-3-Clause | UniLidar PointCloud2 → /scan |
| nav2_bringup + components | Apache-2.0 | Navigation stack |

**slam_toolbox LGPL-2.1** — compatible with industrial/commercial projects under
dynamic linking (the default in ROS 2 / colcon workspaces).

---

## Launch Files

| File | Purpose |
|------|---------|
| `slam.launch.py` | pointcloud_to_laserscan + slam_toolbox |
| `nav2.launch.py` | Nav2 navigation nodes |
| `ekf.launch.py` | Optional robot_localization EKF (not enabled by default) |

## Config Files

| File | Purpose |
|------|---------|
| `config/slam_toolbox_params.yaml` | SLAM algorithm tuning |
| `config/scan_from_cloud.yaml` | PointCloud2 → LaserScan conversion params |
| `config/nav2_params.yaml` | DWB controller, NavFn planner, costmaps |
| `config/ekf_fusion_params.yaml` | Optional EKF template for `/odom_raw` + IMU + lidar odom fusion |

---

## SLAM Modes

### Mapping (default)
```bash
ros2 launch go2_bringup go2_bringup.launch.py slam_mode:=mapping
# Drive around to build map, then save:
ros2 service call /slam_toolbox/serialize_map \
    slam_toolbox/srv/SerializePoseGraph \
    "{filename: '/path/to/map'}"
```

### Localization (after mapping)
```bash
ros2 launch go2_bringup go2_bringup.launch.py \
    slam_mode:=localization \
    map:=/path/to/map
```

---

## Costmap Configuration

- **Local costmap**: PointCloud2 from `/lidar/points`, rolling window 10×10 m
- **Global costmap**: PointCloud2 from `/lidar/points` + static map layer
- Obstacle height filter: 0.08 m – 1.8 m (ignores ground, clears ceiling)
- Footprint: polygon 0.70×0.31 m (full standing footprint including legs)
- Inflation radius: 0.58 m (local), 0.68 m (global)

## Key Tuning Points

| Parameter | File | Value | When to change |
|-----------|------|-------|----------------|
| `max_laser_range` | `slam_toolbox_params.yaml` | 15.0 m | Large spaces → increase |
| `resolution` | `slam_toolbox_params.yaml` | 0.05 m | Precision vs memory trade-off |
| `max_vel_x` | `nav2_params.yaml` | 0.5 m/s | Increase when space allows |
| `inflation_radius` | `nav2_params.yaml` | 0.55 m | Reduce if paths too narrow |
| `min_height` / `max_height` | `scan_from_cloud.yaml` | −0.40 m / +0.40 m | Adjust for LiDAR tilt or mounting height |
| `min_obstacle_height` / `max_obstacle_height` | `nav2_params.yaml` | 0.08 m / 1.8 m | Adjust costmap height filter |

## DWB Local Planner — Tuning Notes

The DWB critic weights are set for a slow indoor quadruped.  The values below
were chosen to prevent a failure mode where the robot hangs near its goal:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `trans_stopped_velocity` | 0.05 m/s | Threshold below which robot is considered "stopped translating". Must be well below `min_vel_x` to avoid premature RotateToGoal activation. |
| `RotateToGoal.scale` | 16.0 | Half the weight of PathDist/PathAlign (32.0). Prevents heading-alignment from dominating and deadlocking trajectories near the goal. |
| `BaseObstacle.scale` | 0.1 | Enough weight to keep robot out of occupied cells without overwhelming directional critics. |
| `failure_tolerance` | 0.5 s | 10 control cycles (at 20 Hz) before recovery is triggered. Provides margin for momentary plan failures without premature recovery cascades. |

**Recovery hang budget** (using the default Nav2 BT): if failure_tolerance fires,
the BT runs up to 6 retries of: ClearCostmap + Spin(1.57 rad) + **Wait(5 s)** +
BackUp(0.3 m). Worst case: ~90 s before navigation aborts.  The fixes above
reduce the frequency of recovery triggering near the goal.

---

## Adding a New Robot

To keep this package robot-agnostic, new robots must preserve the nav-stack
contract below. If these frame/topic names are kept stable, the shared files
`config/nav2_params.yaml`, `config/slam_toolbox_params.yaml`, and
`config/scan_from_cloud.yaml` do not need changes.

### Required Frames

- `base_footprint` — ground projection, `z=0` at contact
- `base_link` — torso center at standing height
- `lidar_link` — LiDAR optical center, horizontal mount (`rpy="0 0 0"`)
- `imu_link` — IMU frame
- `odom` — odometry origin (published by robot bridge)
- `map` — SLAM map origin (published by `slam_toolbox`)

### Required Topics

- `/lidar/points` — LiDAR `sensor_msgs/PointCloud2` (or remapped to this)
- `/scan` — 2D `sensor_msgs/LaserScan` from `pointcloud_to_laserscan`
- `/odom` — `nav_msgs/Odometry`
- `/cmd_vel` — `geometry_msgs/Twist` command input

### D1 Ultra Integration Pattern

1. Create `src/d1_description/urdf/d1.urdf.xacro` with the required frame names.
2. Create `src/d1_bringup/` with `robot_interface.launch.py` that remaps native
   odometry/velocity topics to `/odom` and `/cmd_vel`.
3. If the LiDAR topic differs, remap it to `/lidar/points` in launch, or add
   a `d1_scan_from_cloud.yaml` and keep outward nav topics unchanged.
4. Keep shared nav files unchanged:
   `config/nav2_params.yaml`, `config/slam_toolbox_params.yaml`,
   `config/scan_from_cloud.yaml`.
