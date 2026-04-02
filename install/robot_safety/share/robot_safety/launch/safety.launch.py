"""
safety.launch.py
Launches the velocity_limiter safety gating node.

Intended to be included from go2_bringup.launch.py with use_safety:=true.
When active, go2_hw_bridge receives /cmd_vel_safe instead of /cmd_vel.

Arguments
---------
use_sim_time   bool   (default: false)
cmd_timeout    float  Watchdog timeout in seconds  (default: 0.5)
publish_rate   float  Output publish rate in Hz    (default: 20.0)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    args = [
        DeclareLaunchArgument('use_sim_time',  default_value='false'),
        DeclareLaunchArgument('cmd_timeout',   default_value='0.5',
                              description='Watchdog: zero velocity after this silence (s)'),
        DeclareLaunchArgument('publish_rate',  default_value='20.0',
                              description='Output /cmd_vel_safe publish rate (Hz)'),
    ]

    velocity_limiter = Node(
        package='robot_safety',
        executable='velocity_limiter',
        name='velocity_limiter',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'cmd_timeout':  LaunchConfiguration('cmd_timeout'),
            'publish_rate': LaunchConfiguration('publish_rate'),
            'cmd_vel_in':   '/cmd_vel',
            'cmd_vel_out':  '/cmd_vel_safe',
            'status_topic': '/robot_status',
        }],
    )

    return LaunchDescription(args + [velocity_limiter])
