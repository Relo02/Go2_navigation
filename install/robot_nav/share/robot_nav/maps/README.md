# Maps

Store pre-built slam_toolbox pose-graph files here.

## Creating a map

1. Launch in mapping mode (default):
   ```bash
   ros2 launch go2_bringup go2_bringup.launch.py slam_mode:=mapping
   ```

2. Drive the robot around the environment.
   slam_toolbox builds the map in memory as it processes `/scan`.

3. Save the pose-graph:
   ```bash
   ros2 service call /slam_toolbox/serialize_map \
       slam_toolbox/srv/SerializePoseGraph \
       "{filename: '/ws/anubi/src/robot_nav/maps/my_map'}"
   ```
   This creates `my_map.data` and `my_map.posegraph` in this directory.

## Using a saved map

```bash
ros2 launch go2_bringup go2_bringup.launch.py \
    slam_mode:=localization \
    map:=/ws/anubi/src/robot_nav/maps/my_map
```

## Map file format

slam_toolbox serialises to:
- `.posegraph` — graph nodes and constraints (text)
- `.data`       — binary scan data
