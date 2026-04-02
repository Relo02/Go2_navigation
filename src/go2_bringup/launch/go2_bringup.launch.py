"""
go2_bringup.launch.py
Master entry point for the Go2 autonomy stack.

Starts (in order):
  1. robot_state_publisher  (URDF + static TF from joints)
  2. go2_hw_bridge          (SportModeState → odom/TF,  cmd_vel → Sport API)
  3. velocity_limiter       (optional safety gate; enabled by use_safety:=true)
  4. slam.launch.py         (pointcloud_to_laserscan + slam_toolbox)
  5. nav2.launch.py         (full Nav2 navigation stack)
  6. RViz2                  (optional)

Arguments
---------
use_sim_time    : bool   (default: false)
map             : str    Path to slam_toolbox serialised map (.posegraph) for
                         localisation-only mode. Empty = mapping mode (default).
robot_name      : str    Robot namespace prefix  (default: '')
rviz            : bool   Launch RViz2 (default: true)
params_file     : str    Override Nav2 params file
slam_params     : str    Override slam_toolbox params file
slam_mode       : str    'mapping' or 'localization' (default: 'mapping')
use_safety      : bool   Enable velocity_limiter safety gate (default: false)
                         When true: go2_hw_bridge reads /cmd_vel_safe instead of /cmd_vel.
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():

    go2_bringup_share     = get_package_share_directory("go2_bringup")
    go2_description_share = get_package_share_directory("go2_description")
    robot_nav_share       = get_package_share_directory("robot_nav")
    robot_safety_share    = get_package_share_directory("robot_safety")

    go2_params      = os.path.join(go2_bringup_share,    "config", "go2_params.yaml")
    default_nav     = os.path.join(robot_nav_share,      "config", "nav2_params.yaml")
    default_slam    = os.path.join(robot_nav_share,      "config", "slam_toolbox_params.yaml")
    urdf_file       = os.path.join(go2_description_share,"urdf",   "go2.urdf.xacro")
    rviz_config     = os.path.join(robot_nav_share,      "rviz",   "nav2.rviz")

    # ── Arguments ─────────────────────────────────────────────────────────
    args = [
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("map",          default_value="",
                              description="slam_toolbox map file (.posegraph) for"
                                          " localisation mode; empty = mapping mode"),
        DeclareLaunchArgument("robot_name",   default_value=""),
        DeclareLaunchArgument("rviz",         default_value="true"),
        DeclareLaunchArgument("params_file",  default_value=default_nav),
        DeclareLaunchArgument("slam_params",  default_value=default_slam),
        DeclareLaunchArgument("slam_mode",    default_value="mapping",
                              description="'mapping' or 'localization'"),
        DeclareLaunchArgument("use_safety",   default_value="false",
                              description="Enable velocity_limiter safety gate; "
                                          "go2_hw_bridge will read /cmd_vel_safe"),
    ]

    use_sim_time = LaunchConfiguration("use_sim_time")
    map_path     = LaunchConfiguration("map")
    robot_name   = LaunchConfiguration("robot_name")
    rviz         = LaunchConfiguration("rviz")
    params_file  = LaunchConfiguration("params_file")
    slam_params  = LaunchConfiguration("slam_params")
    slam_mode    = LaunchConfiguration("slam_mode")
    use_safety   = LaunchConfiguration("use_safety")

    # When use_safety=true, go2_hw_bridge reads /cmd_vel_safe (velocity_limiter output).
    # Otherwise it reads /cmd_vel directly from Nav2.
    hw_bridge_cmd_topic = PythonExpression(
        ["'/cmd_vel_safe' if '", use_safety, "' == 'true' else '/cmd_vel'"]
    )

    # ── 1. robot_state_publisher ──────────────────────────────────────────
    robot_description = Command(
        ["xacro ", urdf_file, " use_gazebo:=", use_sim_time]
    )
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            go2_params,
            {"robot_description": robot_description, "use_sim_time": use_sim_time},
        ],
    )

    # ── 2. go2_hw_bridge ──────────────────────────────────────────────────
    # Converts SportModeState → odom + TF(odom→base_footprint)
    # Converts cmd_vel → /api/sport/request
    go2_hw_bridge = Node(
        package="go2_bringup",
        executable="go2_hw_bridge",
        name="go2_hw_bridge",
        output="screen",
        parameters=[go2_params, {
            "use_sim_time":  use_sim_time,
            "cmd_vel_topic": hw_bridge_cmd_topic,
        }],
    )

    # ── 3. Safety gate (optional) ─────────────────────────────────────────
    # velocity_limiter sits between Nav2 /cmd_vel and go2_hw_bridge.
    # Only launched when use_safety:=true.
    safety = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(robot_safety_share, "launch", "safety.launch.py")
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
        condition=IfCondition(use_safety),
    )

    # ── 4. SLAM (slam_toolbox + pointcloud_to_laserscan) ──────────────────
    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(robot_nav_share, "launch", "slam.launch.py")
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "slam_params":  slam_params,
            "slam_mode":    slam_mode,
            "map":          map_path,
        }.items(),
    )

    # ── 5. Nav2 ───────────────────────────────────────────────────────────
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(robot_nav_share, "launch", "nav2.launch.py")
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "params_file":  params_file,
            "namespace":    robot_name,
        }.items(),
    )

    # ── 6. RViz2 ─────────────────────────────────────────────────────────
    rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen",
        condition=IfCondition(rviz),
    )

    return LaunchDescription(args + [
        robot_state_publisher,
        go2_hw_bridge,
        safety,
        slam,
        nav2,
        rviz2,
    ])
