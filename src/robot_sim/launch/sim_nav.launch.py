"""
sim_nav.launch.py
Convenience launch: Nav2 + RViz for an already-running simulation.

Use this when you want to restart Nav2 or RViz without restarting
Gazebo, robot_state_publisher, or SLAM.

Prerequisites (must already be running):
  ros2 launch robot_sim sim_bringup.launch.py use_rviz:=false

Arguments
---------
use_sim_time  : bool  (default: true  — always true in simulation)
params_file   : str   Nav2 params file (default: robot_nav default)
use_rviz      : bool  (default: true)
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

    robot_nav_share = get_package_share_directory('robot_nav')

    default_params = os.path.join(
        robot_nav_share,
        'config',
        'nav2_params.yaml',
    )
    rviz_config    = os.path.join(robot_nav_share, 'rviz', 'nav2.rviz')

    args = [
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='Use simulation clock (always true in sim)'),
        DeclareLaunchArgument('params_file',  default_value=default_params,
                              description='Nav2 params YAML override'),
        DeclareLaunchArgument('use_rviz',     default_value='true'),
    ]

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(robot_nav_share, 'launch', 'nav2.launch.py')
        ),
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'params_file':  LaunchConfiguration('params_file'),
        }.items(),
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_rviz')),
    )

    return LaunchDescription(args + [nav2, rviz])
