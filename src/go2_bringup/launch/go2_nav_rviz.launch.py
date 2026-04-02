"""
go2_nav_rviz.launch.py
Convenience launch: Nav2 + RViz only.
Assumes robot_state_publisher, go2_hw_bridge, and SLAM are already running.
Useful for restarting the navigation stack without restarting hardware.

Arguments
---------
use_sim_time  bool   (default: false)
params_file   str    Nav2 params override  (default: robot_nav default)
namespace     str    Robot namespace       (default: '')
rviz          bool   Launch RViz2         (default: true)
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    robot_nav_share = get_package_share_directory("robot_nav")
    default_params  = os.path.join(robot_nav_share, "config", "nav2_params.yaml")
    rviz_config     = os.path.join(robot_nav_share, "rviz", "nav2.rviz")

    args = [
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("params_file",  default_value=default_params),
        DeclareLaunchArgument("namespace",    default_value=""),
        DeclareLaunchArgument("rviz",         default_value="true"),
    ]

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(robot_nav_share, "launch", "nav2.launch.py")
        ),
        launch_arguments={
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "params_file":  LaunchConfiguration("params_file"),
            "namespace":    LaunchConfiguration("namespace"),
        }.items(),
    )

    rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
        output="screen",
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    return LaunchDescription(args + [nav2, rviz2])
