# go2_description

URDF/xacro robot description for the Unitree Go2 quadruped.

## Frame Tree

```
base_footprint   (z=0, ground projection — Nav2 footprint reference)
  └── base_link  (robot body centre at nominal standing height ~0.335 m)
        ├── imu_link    (co-located with body centre; Go2 internal IMU)
        └── lidar_link  (Unitree L1 LiDAR, forward-top mount)
```

## Files

| File | Purpose |
|------|---------|
| `urdf/go2.urdf.xacro` | Main robot description; accepts `use_gazebo` xacro arg |
| `rviz/` | Reserved for RViz configs specific to the description |
| `config/` | Reserved for joint state publisher config if legs are added |

## Assumptions

| ID | Assumption | Action if wrong |
|----|-----------|----------------|
| A-URDF-1 | Nominal standing height = 0.335 m | Measure and update `standing_height` in xacro |
| A-URDF-2 | LiDAR position = [0.2834, 0.0, 0.1625] relative to `base_link` | Calibrate and update `lidar_x/y/z` in xacro |
| A-URDF-3 | Leg links omitted | Add legs when joint visualisation is needed |

## Porting to AgiBot D1 Ultra

Create `agibot_d1_description` with the same frame names:
- `base_footprint`, `base_link`, `lidar_link`, `imu_link`
- Same joint naming convention
All robot_nav and robot_common_interfaces code will work without changes.
