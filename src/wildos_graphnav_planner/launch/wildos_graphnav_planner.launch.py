"""Launch the WildOS-style graph planner and remap its path to /wildos/path."""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    planner = Node(
        package='wildos_graphnav_planner',
        executable='planner_node',
        name='wildos_graphnav_planner',
        output='screen',
        remappings=[
            ('~/path', '/wildos/path'),
        ],
    )

    return LaunchDescription([planner])
