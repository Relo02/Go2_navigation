"""
graphnav_mpc_planner.launch.py

Launches the three-node graphnav MPC planner stack:

  planner_node          -- Dijkstra on navigation graph -> nav_msgs/Path
  mpc_node              -- CasADi/IPOPT MPC tracker -> /mpc/next_setpoint
  setpoint_to_cmd_vel   -- Setpoint -> /cmd_vel

Topic wiring (matches nebula2-wildos convention):
  Input  nav_graph  : graphnav_mpc_planner/nav_graph  <- scored_nav_graph
  Input  goal_pose  : graphnav_mpc_planner/goal_pose  <- /goal_pose or rviz
  Input  odom       : graphnav_mpc_planner/odom       <- /odom
  Output path       : graphnav_mpc_planner/path       (→ consumed by mpc_node)
  Output setpoint   : /mpc/next_setpoint
  Output cmd_vel    : /cmd_vel

The navigation graph (scored_nav_graph) must be published by an external
graph building + scoring system (e.g., the wildos visual_navigation stack).
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('graphnav_mpc_planner')
    default_params = os.path.join(pkg_share, 'config', 'graphnav_mpc_params.yaml')
    default_rviz   = os.path.join(pkg_share, 'rviz', 'graphnav_mpc.rviz')

    args = [
        DeclareLaunchArgument('use_sim_time',    default_value='false'),
        DeclareLaunchArgument('params_file',     default_value=default_params),
        DeclareLaunchArgument('nav_graph_topic', default_value='scored_nav_graph',
                              description='Topic publishing graphnav_msgs/NavigationGraph'),
        DeclareLaunchArgument('odom_topic',      default_value='odom',
                              description='nav_msgs/Odometry topic for robot pose'),
        DeclareLaunchArgument('goal_pose_topic', default_value='goal_pose',
                              description='PoseStamped goal topic'),
        DeclareLaunchArgument('ns',              default_value='graphnav_mpc',
                              description='Node namespace'),
        DeclareLaunchArgument('use_rviz',        default_value='false'),
        DeclareLaunchArgument('rviz_config',     default_value=default_rviz),
    ]

    use_sim_time    = LaunchConfiguration('use_sim_time')
    params_file     = LaunchConfiguration('params_file')
    nav_graph_topic = LaunchConfiguration('nav_graph_topic')
    odom_topic      = LaunchConfiguration('odom_topic')
    goal_pose_topic = LaunchConfiguration('goal_pose_topic')
    ns              = LaunchConfiguration('ns')

    planner_node = Node(
        package='graphnav_mpc_planner',
        executable='planner_node',
        name='planner_node',
        namespace=ns,
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        remappings=[
            ('~/nav_graph',  nav_graph_topic),
            ('~/odom',       odom_topic),
            ('~/goal_pose',  goal_pose_topic),
        ],
    )

    mpc_node = Node(
        package='graphnav_mpc_planner',
        executable='mpc_node',
        name='mpc_node',
        namespace=ns,
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        remappings=[
            # path comes from the sibling planner_node in the same namespace
            ('~/path',  'planner_node/path'),
            ('~/odom',  odom_topic),
        ],
    )

    setpoint_to_cmd_vel = Node(
        package='graphnav_mpc_planner',
        executable='setpoint_to_cmd_vel_node',
        name='setpoint_to_cmd_vel_node',
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        # /go2/pose and /mpc/next_setpoint are absolute — no remapping needed
        # unless the robot pose comes from a different topic.
    )

    from launch.conditions import IfCondition
    from launch_ros.actions import Node as RosNode

    rviz = RosNode(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=['-d', LaunchConfiguration('rviz_config')],
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_rviz')),
    )

    return LaunchDescription(args + [
        planner_node,
        mpc_node,
        setpoint_to_cmd_vel,
        rviz,
    ])
