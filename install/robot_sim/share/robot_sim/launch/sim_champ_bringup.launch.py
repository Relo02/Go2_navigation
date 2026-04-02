"""
sim_champ_bringup.launch.py
Full Gazebo Fortress + CHAMP + Nav2 simulation bringup for Go2.

What this launch file starts
-----------------------------
  1. go2_sim/sim_champ.launch.py
       - Ignition Gazebo (Fortress)
       - ros_ign bridges
       - robot_state_publisher
       - CHAMP controller + state estimator
       - CHAMP EKF chain (odom -> base_footprint)
  2. cloud_self_filter.py (optional)
       - /lidar/points -> /lidar/points_filtered for optional voxel-layer preprocessing
  3. robot_nav/slam.launch.py (optional via start_nav:=true)
  4. robot_nav/nav2.launch.py (optional via start_nav:=true)
  5. RViz2 (optional)

Key integration rule
--------------------
sim_champ.launch.py publishes a fallback static map->odom transform by default.
For SLAM/Nav2 bringup we disable that fallback so slam_toolbox is the sole
publisher of map->odom.
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    go2_sim_share = get_package_share_directory("go2_sim")
    robot_nav_share = get_package_share_directory("robot_nav")
    a_star_mpc_share = get_package_share_directory("a_star_mpc_planner")

    default_world = os.path.join(go2_sim_share, "worlds", "default.sdf")
    default_ros_control = os.path.join(go2_sim_share, "config", "ros_control.yaml")
    default_slam_params = os.path.join(robot_nav_share, "config", "slam_toolbox_params.yaml")
    default_nav_params = os.path.join(robot_nav_share, "config", "nav2_params.yaml")
    default_rviz_config = os.path.join(robot_nav_share, "rviz", "nav2.rviz")
    default_a_star_params = os.path.join(a_star_mpc_share, "config", "planner_params.yaml")

    args = [
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("world", default_value=default_world),
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument("robot_name", default_value="go2"),
        DeclareLaunchArgument("ros_control_file", default_value=default_ros_control),
        DeclareLaunchArgument("world_init_x", default_value="0.0"),
        DeclareLaunchArgument("world_init_y", default_value="0.0"),
        DeclareLaunchArgument("world_init_z", default_value="0.375"),
        DeclareLaunchArgument("world_init_heading", default_value="0.0"),
        DeclareLaunchArgument("publish_map_to_odom_tf", default_value="false"),
        DeclareLaunchArgument("start_nav", default_value="true"),
        DeclareLaunchArgument("nav_delay_sec", default_value="8.0"),
        DeclareLaunchArgument("slam_mode", default_value="mapping"),
        DeclareLaunchArgument("map", default_value=""),
        DeclareLaunchArgument("slam_params", default_value=default_slam_params),
        DeclareLaunchArgument("nav_params", default_value=default_nav_params),
        DeclareLaunchArgument("launch_cloud_filter", default_value="true"),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument("rviz_config", default_value=default_rviz_config),
        DeclareLaunchArgument("start_a_star_planner", default_value="false"),
        DeclareLaunchArgument("a_star_planner_params", default_value=default_a_star_params),
    ]

    start_nav = LaunchConfiguration("start_nav")
    use_sim_time = LaunchConfiguration("use_sim_time")

    champ_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(go2_sim_share, "launch", "sim_champ.launch.py")
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "rviz": "false",
            "robot_name": LaunchConfiguration("robot_name"),
            "ros_control_file": LaunchConfiguration("ros_control_file"),
            "world": LaunchConfiguration("world"),
            "gui": LaunchConfiguration("gui"),
            "world_init_x": LaunchConfiguration("world_init_x"),
            "world_init_y": LaunchConfiguration("world_init_y"),
            "world_init_z": LaunchConfiguration("world_init_z"),
            "world_init_heading": LaunchConfiguration("world_init_heading"),
            "publish_map_to_odom_tf": LaunchConfiguration("publish_map_to_odom_tf"),
        }.items(),
    )

    cloud_self_filter = Node(
        package="robot_sim",
        executable="cloud_self_filter.py",
        name="cloud_self_filter",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "input_topic": "/lidar/points",
            "output_topic": "/lidar/points_filtered",
            "lidar_x": 0.28945,
            "lidar_y": 0.0,
            "lidar_z": -0.046825,
            "lidar_roll": 0.0,
            "lidar_pitch": 0.0,
            "lidar_yaw": 0.0,
            "min_ray_elevation_deg": 6.0,
            "min_world_z": 0.10,
        }],
        condition=IfCondition(LaunchConfiguration("launch_cloud_filter")),
    )

    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(robot_nav_share, "launch", "slam.launch.py")
        ),
        condition=IfCondition(start_nav),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "slam_mode": LaunchConfiguration("slam_mode"),
            "slam_params": LaunchConfiguration("slam_params"),
            "map": LaunchConfiguration("map"),
        }.items(),
    )

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(robot_nav_share, "launch", "nav2.launch.py")
        ),
        condition=IfCondition(start_nav),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "params_file": LaunchConfiguration("nav_params"),
        }.items(),
    )

    start_nav_stack = TimerAction(
        period=LaunchConfiguration("nav_delay_sec"),
        actions=[
            slam,
            # Give slam_toolbox time to receive scans and publish map->odom before Nav2 activates.
            TimerAction(period=6.0, actions=[nav2]),
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_rviz")),
    )

    # ---- A* MPC Planner ----
    a_star_node = Node(
        package="a_star_mpc_planner",
        executable="a_star_node",
        name="a_star_node",
        output="screen",
        parameters=[
            LaunchConfiguration("a_star_planner_params"),
            {"use_sim_time": use_sim_time},
        ],
        condition=IfCondition(LaunchConfiguration("start_a_star_planner")),
    )

    mpc_node = Node(
        package="a_star_mpc_planner",
        executable="mpc_node",
        name="mpc_node",
        output="screen",
        parameters=[
            LaunchConfiguration("a_star_planner_params"),
            {"use_sim_time": use_sim_time},
        ],
        condition=IfCondition(LaunchConfiguration("start_a_star_planner")),
    )

    return LaunchDescription(args + [
        champ_sim,
        cloud_self_filter,
        start_nav_stack,
        a_star_node,
        mpc_node,
        rviz,
    ])
