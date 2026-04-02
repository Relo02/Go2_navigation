# robot_common_interfaces

Shared ROS 2 interface definitions and the **robot adapter contract**.

All hardware adapter packages (`go2_bringup`, future `agibot_d1_bringup`) must
satisfy this contract.  Navigation and application code depends only on this
contract, never on hardware-specific packages.

---

## Frame Contract

Every hardware adapter MUST maintain this TF tree:

```
map  ──►  odom  ──►  base_footprint  ──►  base_link  ──►  lidar_link
                                                     └──►  imu_link
```

| Frame | Published by | Notes |
|-------|-------------|-------|
| `map → odom` | slam_toolbox | Async 2D SLAM; mapping or localization mode |
| `odom → base_footprint` | go2_hw_bridge (real) / planar_move plugin (sim) | Dynamic odometry |
| `base_footprint → base_link` | robot_state_publisher (from URDF) | Fixed, z = standing height |
| `base_link → lidar_link` | robot_state_publisher (from URDF) | Static |
| `base_link → imu_link` | robot_state_publisher (from URDF) | Static |

---

## Topic Contract

| Topic | Type | Direction | Notes |
|-------|------|-----------|-------|
| `/unilidar/cloud` | `sensor_msgs/PointCloud2` | Hardware → Nav2 | LiDAR data in `lidar_link` frame |
| `/unilidar/imu` | `sensor_msgs/Imu` | Hardware → Nav2 | IMU data |
| `/odom` | `nav_msgs/Odometry` | go2_hw_bridge → Nav2 | Odometry with `odom`→`base_footprint` frames |
| `/cmd_vel` | `geometry_msgs/Twist` | Nav2 → go2_hw_bridge | Velocity command |
| `/robot_status` | `robot_common_interfaces/RobotStatus` | Hardware → all | Health status at 1 Hz |
| `/goal_pose` | `geometry_msgs/PoseStamped` | RViz → bt_navigator | Navigation goal |

---

## Messages

### RobotStatus.msg

Published by each hardware adapter at 1 Hz on `/robot_status`.
Contains: battery state, estop flag, motion mode, hw_ok flag.

---

## Adding a New Robot Adapter

1. Create `<robot>_bringup` package.
2. Implement a node that subscribes to the robot's proprioceptive state and publishes
   `/odom` + TF `odom → base_footprint`.
3. Implement a `/cmd_vel` subscriber that forwards velocity commands to the robot's
   motion API (clamped to hardware limits).
4. Create `<robot>_description` with the same frame names: `base_footprint`, `base_link`,
   `lidar_link`, `imu_link`.
5. Publish `RobotStatus` on `/robot_status`.
6. No changes required in `robot_nav` or `robot_common_interfaces`.

See `go2_bringup/src/go2_hw_bridge.cpp` and the `TODO(agibot)` markers for the
exact locations that change per robot.
