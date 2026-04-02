"""
slam.launch.py
2D SLAM pipeline using slam_toolbox (LGPL-2.1).

Starts:
  1. pointcloud_to_laserscan  — converts /lidar/points (PointCloud2) to /scan (LaserScan)
  2. slam_toolbox             — async 2D SLAM; mapping or localisation mode
  3. map_saver + lifecycle_manager_localization
                               keeps Nav2's localization lifecycle panel active
                               and provides map saving services

Arguments
---------
use_sim_time : bool   (default: true)
slam_params  : str    Path to slam_toolbox YAML params file
slam_mode    : str    'mapping' (default) | 'localization'
map          : str    For localization mode: path to .posegraph file
autostart    : bool   Autostart localization lifecycle manager (default: true)

SLAM choice rationale
---------------------
slam_toolbox (licence: LGPL-2.1) was chosen for:
  - Native Nav2 integration (directly maintained alongside Nav2)
  - Supports both online mapping and localization-only modes
  - Async mode handles variable scan rates without blocking Nav2
  - Loop closure for long-session reliability
  - Map serialization (save/load .posegraph files)
  - LGPL-2.1 is compatible with industrial deployments using dynamic linking
    (the standard in ROS 2 / colcon workspaces)

Alternative evaluated:
  - cartographer (Apache-2.0): more permissive licence but less actively maintained
    for ROS 2 Humble; complex configuration; no localization-only mode without cartographer_grpc
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def _validate_localization_inputs(context):
    """Fail fast if localization mode is requested without a valid map file."""
    mode = LaunchConfiguration("slam_mode").perform(context).strip()
    map_file = LaunchConfiguration("map").perform(context).strip()

    if mode == "localization":
        if not map_file:
            raise RuntimeError(
                "slam_mode:=localization requires map:=<path/to/map.posegraph>."
            )
        if not os.path.isfile(map_file):
            raise RuntimeError(
                f"Localization map file not found: '{map_file}'. "
                "Provide a valid slam_toolbox .posegraph file path via map:=..."
            )

    return []


def generate_launch_description():

    robot_nav_share = get_package_share_directory("robot_nav")

    default_slam_params  = os.path.join(robot_nav_share, "config", "slam_toolbox_params.yaml")
    default_scan_params  = os.path.join(robot_nav_share, "config", "scan_from_cloud.yaml")

    args = [
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("slam_params",  default_value=default_slam_params),
        DeclareLaunchArgument("slam_mode",    default_value="mapping",
                              description="'mapping' or 'localization'"),
        DeclareLaunchArgument("map",          default_value="",
                              description="Path to .posegraph file for localization mode"),
        DeclareLaunchArgument("autostart",    default_value="true"),
    ]

    use_sim_time = LaunchConfiguration("use_sim_time")
    slam_params  = LaunchConfiguration("slam_params")
    slam_mode    = LaunchConfiguration("slam_mode")
    map_path     = LaunchConfiguration("map")
    autostart    = LaunchConfiguration("autostart")

    is_mapping       = PythonExpression(["'", slam_mode, "' == 'mapping'"])
    is_localization  = PythonExpression(["'", slam_mode, "' == 'localization'"])

    # ── 1. pointcloud_to_laserscan ────────────────────────────────────────
    # Converts the 3D UniLidar PointCloud2 to a 2D LaserScan required by
    # slam_toolbox and by Nav2 costmaps (configured on /scan).
    # A horizontal slice is extracted around z=0 relative to lidar_link.
    pointcloud_to_laserscan = Node(
        package="pointcloud_to_laserscan",
        executable="pointcloud_to_laserscan_node",
        name="pointcloud_to_laserscan",
        output="screen",
        parameters=[
            default_scan_params,
            {"use_sim_time": use_sim_time},
        ],
        remappings=[
            ("cloud_in", "/lidar/points"),
            ("scan",     "/scan"),
        ],
    )

    # ── 2a. slam_toolbox — async mapping mode ─────────────────────────────
    slam_mapping = Node(
        package="slam_toolbox",
        executable="async_slam_toolbox_node",
        name="slam_toolbox",
        output="screen",
        parameters=[
            slam_params,
            {"use_sim_time": use_sim_time},
        ],
        condition=IfCondition(is_mapping),
    )

    # ── 2b. slam_toolbox — localisation mode (no mapping) ─────────────────
    # Used after a map has been saved with:
    #   ros2 service call /slam_toolbox/serialize_map
    #   slam_toolbox/srv/SerializePoseGraph "{filename: '/path/to/map'}"
    slam_localization = Node(
        package="slam_toolbox",
        executable="localization_slam_toolbox_node",
        name="slam_toolbox",
        output="screen",
        parameters=[
            slam_params,
            {
                "use_sim_time": use_sim_time,
                "map_file_name": map_path,
            },
        ],
        condition=IfCondition(is_localization),
    )

    # ── 3. map_saver + localization lifecycle manager ─────────────────────
    # Nav2 RViz plugin expects a localization lifecycle manager. Even when
    # SLAM provides map->odom (without AMCL), running map_saver under
    # lifecycle_manager_localization keeps the localization stack state active.
    map_saver = Node(
        package="nav2_map_server",
        executable="map_saver_server",
        name="map_saver",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "save_map_timeout": 5.0,
            "free_thresh_default": 0.25,
            "occupied_thresh_default": 0.65,
            "map_subscribe_transient_local": True,
        }],
    )

    lifecycle_manager_localization = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_localization",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "autostart": autostart,
            "node_names": ["map_saver"],
        }],
    )

    return LaunchDescription(args + [
        OpaqueFunction(function=_validate_localization_inputs),
        pointcloud_to_laserscan,
        slam_mapping,
        slam_localization,
        map_saver,
        lifecycle_manager_localization,
    ])
