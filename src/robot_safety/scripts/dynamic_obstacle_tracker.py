#!/usr/bin/env python3
"""
dynamic_obstacle_tracker.py
PLACEHOLDER - not active.

Future implementation: subscribe to /lidar/points, cluster moving points
between consecutive frames, estimate each cluster's velocity vector, and
publish a smeared PointCloud2 on /dynamic_obstacle_markers that marks the
projected future position of each moving obstacle.
This additional cloud should be added as a second observation_source in the
local and global VoxelLayer configs (source name: dynamic_projection).

Architecture contract (for implementer):
  Input:  /lidar/points  (sensor_msgs/PointCloud2, frame: lidar_link)
  Output: /dynamic_obstacle_markers  (sensor_msgs/PointCloud2, frame: odom)
  Node name: dynamic_obstacle_tracker
  Package: robot_safety

The smearing direction is: obstacle_centroid + velocity_vector * lookahead_time
The lookahead_time should be a ROS parameter (default: 1.5 s).
Preferred avoidance side: the planner will naturally route away from the
smeared projection, which will be on the side the obstacle is moving toward,
leaving the opposite side as the lower-cost path.

HOW TO ACTIVATE WHEN READY:
  1. Implement this node.
  2. Add to local_costmap voxel_layer observation_sources:
       observation_sources: cloud dynamic_projection
       dynamic_projection:
         topic: /dynamic_obstacle_markers
         data_type: "PointCloud2"
         marking: true
         clearing: false
         min_obstacle_height: 0.05
         max_obstacle_height: 2.0
         obstacle_max_range: 5.0
         raytrace_max_range: 0.0
         expected_update_rate: 10.0
  3. Launch this node from robot_safety/launch/ alongside the nav stack.
  4. Remove this PLACEHOLDER comment once active.
"""
# TODO: implement
raise NotImplementedError("dynamic_obstacle_tracker is a placeholder.")
