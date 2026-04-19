"""
sim_graphnav_mpc.launch.py

Gazebo Fortress + CHAMP simulation with the graphnav MPC planner stack.

What this launches
------------------
  1. go2_sim/sim_champ.launch.py
       - Ignition Gazebo (Fortress), ros_ign bridges, robot_state_publisher
       - CHAMP controller + state estimator + EKF (odom -> base_footprint)
  2. cloud_self_filter (optional, start_cloud_filter:=true)
       - /lidar/points -> /lidar/points_filtered
  3. robot_nav/slam.launch.py (optional, start_slam:=true)
       - slam_toolbox in mapping mode for map->odom TF + occupancy grid
  4. graphnav_mpc_planner stack (start_graphnav_mpc:=true)
       - planner_node   : Dijkstra on graphnav NavigationGraph -> nav_msgs/Path
       - mpc_node       : CasADi/IPOPT MPC tracker -> /mpc/next_setpoint
       - setpoint_to_cmd_vel : setpoint -> /cmd_vel
  5. RViz2 (optional)

Key differences from sim_champ_bringup.launch.py
-------------------------------------------------
- Nav2 is NOT started: cmd_vel is produced directly by the MPC setpoint controller.
- The A* local planner is NOT started.
- graphnav_mpc_planner replaces both Nav2 and the A*+MPC stack.

Prerequisites
-------------
The navigation graph (graphnav_msgs/NavigationGraph on topic scored_nav_graph)
must be published by an external system — e.g. the nebula2-wildos
visual_navigation + graphnav stack sourced from a second workspace.
Without an incoming graph the planner silently waits; the robot will not move.

Usage
-----
  ros2 launch robot_sim sim_graphnav_mpc.launch.py
  ros2 launch robot_sim sim_graphnav_mpc.launch.py gui:=false start_slam:=false
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    go2_sim_share          = get_package_share_directory('go2_sim')
    robot_nav_share        = get_package_share_directory('robot_nav')
    graphnav_mpc_share     = get_package_share_directory('graphnav_mpc_planner')

    default_world          = os.path.join(go2_sim_share, 'worlds', 'default.sdf')
    default_ros_control    = os.path.join(go2_sim_share, 'config', 'ros_control.yaml')
    default_slam_params    = os.path.join(robot_nav_share, 'config', 'slam_toolbox_params.yaml')
    default_mpc_params     = os.path.join(graphnav_mpc_share, 'config', 'graphnav_mpc_params.yaml')
    default_rviz_config    = os.path.join(graphnav_mpc_share, 'rviz', 'graphnav_mpc.rviz')

    # ── Launch arguments ──────────────────────────────────────────────
    args = [
        DeclareLaunchArgument('use_sim_time',          default_value='true'),
        DeclareLaunchArgument('world',                 default_value=default_world),
        DeclareLaunchArgument('gui',                   default_value='true'),
        DeclareLaunchArgument('robot_name',            default_value='go2'),
        DeclareLaunchArgument('ros_control_file',      default_value=default_ros_control),
        DeclareLaunchArgument('world_init_x',          default_value='0.0'),
        DeclareLaunchArgument('world_init_y',          default_value='0.0'),
        DeclareLaunchArgument('world_init_z',          default_value='0.375'),
        DeclareLaunchArgument('world_init_heading',    default_value='0.0'),
        DeclareLaunchArgument('publish_map_to_odom_tf',default_value='false'),

        # SLAM (optional — needed for map->odom TF if using a mapped environment)
        DeclareLaunchArgument('start_slam',            default_value='false'),
        DeclareLaunchArgument('slam_delay_sec',        default_value='8.0'),
        DeclareLaunchArgument('slam_mode',             default_value='mapping'),
        DeclareLaunchArgument('slam_params',           default_value=default_slam_params),

        # Cloud filter
        DeclareLaunchArgument('start_cloud_filter',    default_value='true'),

        # GraphNav MPC planner
        DeclareLaunchArgument('start_graphnav_mpc',    default_value='true'),
        DeclareLaunchArgument('graphnav_mpc_delay_sec',default_value='30.0',
                              description='Delay before starting the planner (lets the sim settle)'),
        DeclareLaunchArgument('graphnav_mpc_params',   default_value=default_mpc_params),
        DeclareLaunchArgument('nav_graph_topic',       default_value='scored_nav_graph',
                              description='NavigationGraph topic from external wildos stack'),
        DeclareLaunchArgument('odom_topic',            default_value='odom'),
        DeclareLaunchArgument('goal_pose_topic',       default_value='goal_pose'),

        # RViz
        DeclareLaunchArgument('use_rviz',              default_value='true'),
        DeclareLaunchArgument('rviz_config',           default_value=default_rviz_config),
    ]

    use_sim_time     = LaunchConfiguration('use_sim_time')
    start_slam       = LaunchConfiguration('start_slam')
    start_graphnav   = LaunchConfiguration('start_graphnav_mpc')

    # ── 1. Gazebo + CHAMP ─────────────────────────────────────────────
    champ_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(go2_sim_share, 'launch', 'sim_champ.launch.py')
        ),
        launch_arguments={
            'use_sim_time':          use_sim_time,
            'rviz':                  'false',
            'robot_name':            LaunchConfiguration('robot_name'),
            'ros_control_file':      LaunchConfiguration('ros_control_file'),
            'world':                 LaunchConfiguration('world'),
            'gui':                   LaunchConfiguration('gui'),
            'world_init_x':          LaunchConfiguration('world_init_x'),
            'world_init_y':          LaunchConfiguration('world_init_y'),
            'world_init_z':          LaunchConfiguration('world_init_z'),
            'world_init_heading':    LaunchConfiguration('world_init_heading'),
            'publish_map_to_odom_tf':LaunchConfiguration('publish_map_to_odom_tf'),
        }.items(),
    )

    # ── 2. Cloud self-filter ──────────────────────────────────────────
    cloud_self_filter = Node(
        package='robot_sim',
        executable='cloud_self_filter.py',
        name='cloud_self_filter',
        output='screen',
        parameters=[{
            'use_sim_time':        use_sim_time,
            'input_topic':         '/lidar/points',
            'output_topic':        '/lidar/points_filtered',
            'lidar_x':             0.28945,
            'lidar_y':             0.0,
            'lidar_z':             -0.046825,
            'lidar_roll':          0.0,
            'lidar_pitch':         0.0,
            'lidar_yaw':           0.0,
            'min_ray_elevation_deg': -45.0,  # Accept points at any elevation (relaxed)
            'min_world_z':         -1.0,     # Accept points below ground (relaxed)
            'min_radius':          0.05,     # Reject only very close points
            'publish_in_world_frame': False,  # Keep points in sensor frame (easier to debug)
            'stereo_cloud_topic':  '/stereo/points',
            'stereo_min_radius':   0.30,
            'stereo_max_range':    8.0,
            'stereo_to_lidar_dx':  0.02445,
            'stereo_to_lidar_dy': -0.06,
            'stereo_to_lidar_dz': -0.16683,
        }],
    )

    # ── 3. SLAM (optional) ────────────────────────────────────────────
    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(robot_nav_share, 'launch', 'slam.launch.py')
        ),
        condition=IfCondition(start_slam),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'slam_mode':    LaunchConfiguration('slam_mode'),
            'slam_params':  LaunchConfiguration('slam_params'),
            'map':          '',
        }.items(),
    )

    start_slam_delayed = TimerAction(
        period=LaunchConfiguration('slam_delay_sec'),
        actions=[slam],
    )

    # ── 4. Odometry -> PoseStamped bridge ─────────────────────────────
    # This must start early (before cloud_self_filter) to provide /go2/pose
    odom_to_pose = Node(
        package='graphnav_mpc_planner',
        executable='odom_to_pose_node',
        name='odom_to_pose_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        remappings=[('/odom', LaunchConfiguration('odom_topic'))],
        condition=IfCondition(LaunchConfiguration('start_cloud_filter')),
    )

    # Delay odom_to_pose start to allow Gazebo to begin publishing /odom
    # Start early (3 seconds) so /go2/pose is ready before cloud_self_filter at 4 seconds
    odom_to_pose_delayed = TimerAction(
        period='3.0',
        actions=[odom_to_pose],
    )

    # ── 5. Cloud filter (depends on /go2/pose from odom_to_pose) ───────
    # Delay cloud_self_filter to ensure odom_to_pose has started (4 seconds)
    cloud_self_filter_delayed = TimerAction(
        period='4.0',
        actions=[cloud_self_filter],
    )

    # ── 6. Mock navigation graph publisher (for testing) ──────────────
    # In production, this would be replaced by an external visual_navigation
    # stack that publishes the scored_nav_graph. This mock version creates
    # a simple grid of waypoints for development.
    mock_nav_graph = Node(
        package='graphnav_mpc_planner',
        executable='mock_nav_graph_publisher',
        name='mock_nav_graph_publisher',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(
            LaunchConfiguration('start_graphnav_mpc')
        ),
    )

    # ── 6. GraphNav MPC planner stack ────────────────────────────────
    planner_node = Node(
        package='graphnav_mpc_planner',
        executable='planner_node',
        name='planner_node',
        namespace='graphnav_mpc',
        output='screen',
        parameters=[
            LaunchConfiguration('graphnav_mpc_params'),
            {'use_sim_time': use_sim_time},
        ],
        remappings=[
            ('~/nav_graph',  LaunchConfiguration('nav_graph_topic')),
            ('~/odom',       LaunchConfiguration('odom_topic')),
            ('~/goal_pose',  LaunchConfiguration('goal_pose_topic')),
        ],
        condition=IfCondition(start_graphnav),
    )

    mpc_node = Node(
        package='graphnav_mpc_planner',
        executable='mpc_node',
        name='mpc_node',
        namespace='graphnav_mpc',
        output='screen',
        parameters=[
            LaunchConfiguration('graphnav_mpc_params'),
            {'use_sim_time': use_sim_time},
        ],
        remappings=[
            ('~/path',  'planner_node/path'),
            ('~/odom',  LaunchConfiguration('odom_topic')),
        ],
        condition=IfCondition(start_graphnav),
    )

    setpoint_to_cmd_vel = Node(
        package='graphnav_mpc_planner',
        executable='setpoint_to_cmd_vel_node',
        name='setpoint_to_cmd_vel_node',
        output='screen',
        parameters=[
            LaunchConfiguration('graphnav_mpc_params'),
            {'use_sim_time': use_sim_time},
        ],
        condition=IfCondition(start_graphnav),
    )

    start_graphnav_delayed = TimerAction(
        period=LaunchConfiguration('graphnav_mpc_delay_sec'),
        actions=[
            mock_nav_graph,
            planner_node,
            mpc_node,
            setpoint_to_cmd_vel,
        ],
    )

    # ── 7. RViz ───────────────────────────────────────────────────────
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=['-d', LaunchConfiguration('rviz_config')],
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_rviz')),
    )

    return LaunchDescription(args + [
        champ_sim,
        odom_to_pose_delayed,
        cloud_self_filter_delayed,
        start_slam_delayed,
        start_graphnav_delayed,
        rviz,
    ])
