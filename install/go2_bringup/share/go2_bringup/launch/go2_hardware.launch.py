"""
go2_hardware.launch.py
Hardware-only launch: robot description + hw bridge.
No SLAM, no Nav2, no RViz.
Useful for testing the hardware interface in isolation.

NOTE: The Go2 robot's internal ROS2 bridge must be running and publishing:
  /sportmodestate  [unitree_go/SportModeState]
before this launch is started.
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    go2_bringup_share     = get_package_share_directory("go2_bringup")
    go2_description_share = get_package_share_directory("go2_description")

    go2_params = os.path.join(go2_bringup_share,    "config", "go2_params.yaml")
    urdf_file  = os.path.join(go2_description_share,"urdf",   "go2.urdf.xacro")

    args = [
        DeclareLaunchArgument("use_sim_time", default_value="false"),
    ]

    use_sim_time = LaunchConfiguration("use_sim_time")

    robot_description = Command(
        ["xacro ", urdf_file, " use_gazebo:=", use_sim_time]
    )

    nodes = [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[
                go2_params,
                {"robot_description": robot_description, "use_sim_time": use_sim_time},
            ],
        ),
        Node(
            package="go2_bringup",
            executable="go2_hw_bridge",
            name="go2_hw_bridge",
            output="screen",
            parameters=[go2_params, {"use_sim_time": use_sim_time}],
        ),
    ]

    return LaunchDescription(args + nodes)
