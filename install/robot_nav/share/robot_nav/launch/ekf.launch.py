"""
ekf.launch.py
Optional robot_localization EKF launch for odometry fusion.

This launch is not included by default in go2_bringup/robot_sim bringups.
Use it explicitly when raw odometry, IMU, and optional lidar odometry topics
are available and TF ownership is configured to avoid duplicates.
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    robot_nav_share = get_package_share_directory("robot_nav")
    default_params = os.path.join(robot_nav_share, "config", "ekf_fusion_params.yaml")

    args = [
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("params_file", default_value=default_params),
        DeclareLaunchArgument("namespace", default_value=""),
        DeclareLaunchArgument(
            "output_odom_topic",
            default_value="/odometry/filtered",
            description="EKF odometry output topic (remap to /odom for Nav2 consumption)",
        ),
    ]

    ekf = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        namespace=LaunchConfiguration("namespace"),
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
        remappings=[
            ("odometry/filtered", LaunchConfiguration("output_odom_topic")),
        ],
    )

    return LaunchDescription(args + [ekf])
