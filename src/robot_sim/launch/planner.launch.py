"""
planner.launch.py
Gazebo Fortress + CHAMP + WildOS graph planner + path follower bringup for Go2.
Nav2 and SLAM are intentionally NOT launched.

What this starts
----------------
    1. go2_sim/sim_champ.launch.py
             - Ignition Gazebo (Fortress)
             - ros_ign bridges  (/lidar/points, /odom/raw, /clock, /imu/data, /tf ...)
             - robot_state_publisher
             - CHAMP quadruped controller + state estimator
             - EKF chain (odom/raw + imu -> odom)
             - Static map -> odom identity transform
    2. cloud_self_filter          /lidar/points -> /lidar/points_filtered
    3. odom_to_pose_node          /odom/raw -> /go2/pose
    4. mock_nav_graph_publisher   fallback /nav_graph source for simulation
    5. wildos_graphnav_planner    graph navigation path planning -> /wildos/path
    6. path_follower_node         tracks /wildos/path and publishes /path_follower/next_setpoint
    7. setpoint_to_cmd_vel_node   converts setpoint into /cmd_vel for CHAMP
    8. RViz2 (optional)

Key topic wiring
----------------
    /lidar/points            (Gazebo bridge) -> cloud_self_filter
    /odom/raw                (Gazebo bridge) -> odom_to_pose_node, wildos_graphnav_planner
    /go2/pose                                -> (available for downstream controllers)
    /nav_graph                               -> wildos_graphnav_planner (from mock_nav_graph_publisher)
    /goal_pose                               -> wildos_graphnav_planner (RViz goal input)
    /wildos/path            (wildos planner) -> path_follower_node
    /path_follower/next_setpoint             -> setpoint_to_cmd_vel_node
    /cmd_vel                                 -> CHAMP/go2_sim locomotion pipeline

Goal source
-----------
    Goal is RViz-driven (2D Goal Pose tool) through topic /goal_pose.
    The WildOS planner consumes that goal and publishes the resulting path on /wildos/path.
    No hard-coded launch goal is required.

Usage
-----
    # Default run (Gazebo GUI on, RViz off):
    ros2 launch robot_sim planner.launch.py

    # Headless, RViz on:
    ros2 launch robot_sim planner.launch.py gui:=false use_rviz:=true

        # Override path follower lookahead distance:
        ros2 launch robot_sim planner.launch.py \
            wp_lookahead_dist:=2.5
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

    go2_sim_share = get_package_share_directory("go2_sim")
    robot_sim_share = get_package_share_directory("robot_sim")
    default_world = os.path.join(go2_sim_share, "worlds", "default.sdf")
    default_ros_control = os.path.join(go2_sim_share, "config", "ros_control.yaml")

    # -- Launch arguments ----------------------------------------------------

    args = [
        # Simulation
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("world", default_value=default_world),
        DeclareLaunchArgument("gui", default_value="true", description="Launch Gazebo GUI"),
        DeclareLaunchArgument("robot_name", default_value="go2"),
        DeclareLaunchArgument("ros_control_file", default_value=default_ros_control),
        DeclareLaunchArgument("world_init_x", default_value="0.0"),
        DeclareLaunchArgument("world_init_y", default_value="0.0"),
        DeclareLaunchArgument("world_init_z", default_value="0.375"),
        DeclareLaunchArgument("world_init_heading", default_value="0.0"),

        DeclareLaunchArgument(
            "wp_lookahead_dist",
            default_value="2.0",
            description="Path follower lookahead distance in meters",
        ),
        DeclareLaunchArgument(
            "path_timeout",
            default_value="10.0",
            description="Drop stale paths older than this many seconds",
        ),
        DeclareLaunchArgument(
            "cloud_min_ray_elevation_deg",
            default_value="3.0",
            description=(
                "Reject lidar returns below this downward elevation angle (deg). "
                "Lower values keep more lower obstacle points."
            ),
        ),
        DeclareLaunchArgument(
            "cloud_min_world_z",
            default_value="0.12",
            description=(
                "Drop points below this world Z threshold (m). "
                "Lower values keep obstacle bases while still removing floor."
            ),
        ),

        # Bringup timing
        DeclareLaunchArgument(
            "planner_delay_sec",
            default_value="15.0",
            description="Seconds to wait before starting planner nodes",
        ),
        DeclareLaunchArgument(
            "motion_delay_sec",
            default_value="32.0",
            description="Seconds to wait before starting motion command nodes",
        ),
        DeclareLaunchArgument(
            "use_mock_nav_graph",
            default_value="true",
            description="Publish fallback NavigationGraph for simulation",
        ),
        DeclareLaunchArgument(
            "use_wildos_planner",
            default_value="true",
            description="Enable WildOS graph planner",
        ),
        DeclareLaunchArgument(
            "use_mpc_controller",
            default_value="false",
            description="Use mpc_node for trajectory tracking and cmd_vel output",
        ),
        DeclareLaunchArgument(
            "use_path_follower",
            default_value="true",
            description="Enable path follower tracking node",
        ),
        DeclareLaunchArgument(
            "use_mpc_path_optimizer",
            default_value="false",
            description="Enable MPC path optimization from /wildos/path to /wildos/path_safe",
        ),
        DeclareLaunchArgument(
            "path_follower_path_topic",
            default_value="/wildos/path",
            description="Path topic consumed by controllers",
        ),
        DeclareLaunchArgument(
            "mpc_path_horizon_points",
            default_value="25",
            description="MPC path optimizer horizon length in points",
        ),
        DeclareLaunchArgument(
            "mpc_path_safety_clearance",
            default_value="0.45",
            description="Obstacle clearance target (m) for MPC path optimization",
        ),
        DeclareLaunchArgument(
            "mpc_path_curvature_weight",
            default_value="30.0",
            description="Curvature suppression weight for MPC path optimization",
        ),
        DeclareLaunchArgument(
            "mpc_path_obstacle_weight",
            default_value="180.0",
            description="Obstacle barrier weight for MPC path optimization",
        ),
        DeclareLaunchArgument(
            "cmd_max_vx",
            default_value="0.3",
            description="Max forward speed sent to /cmd_vel (m/s)",
        ),
        DeclareLaunchArgument(
            "cmd_max_vy",
            default_value="0.12",
            description="Max lateral speed sent to /cmd_vel (m/s)",
        ),

        # Visualization
        DeclareLaunchArgument("use_rviz", default_value="false"),
        DeclareLaunchArgument(
            "rviz_delay_sec",
            default_value="6.0",
            description="Seconds to wait before starting RViz",
        ),
        DeclareLaunchArgument(
            "rviz_config",
            default_value=os.path.join(robot_sim_share, "rviz", "a_star_mpc.rviz"),
            description="RViz2 config file",
        ),
    ]

    use_sim_time = LaunchConfiguration("use_sim_time")
    planner_delay = LaunchConfiguration("planner_delay_sec")
    motion_delay = LaunchConfiguration("motion_delay_sec")
    rviz_delay = LaunchConfiguration("rviz_delay_sec")

    # -- 1. Base simulation (Gazebo + CHAMP + EKF) --------------------------
    champ_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(go2_sim_share, "launch", "sim_champ.launch.py")),
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
            "publish_map_to_odom_tf": "true",
        }.items(),
    )

    # -- 2. Odometry -> PoseStamped bridge ----------------------------------
    odom_to_pose = Node(
        package="mpc",
        executable="odom_to_pose_node",
        name="odom_to_pose_node",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "cmd_max_vx": 1.5,
            "cmd_max_vy": 1.0,
        }],
    )

    # -- 3. Point-cloud self-filter -----------------------------------------
    cloud_self_filter = Node(
        package="robot_sim",
        executable="cloud_self_filter.py",
        name="cloud_self_filter",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "input_topic": "/lidar/points",
            "output_topic": "/lidar/points_filtered",
            "world_frame_id": "odom",
            "lidar_x": 0.28945,
            "lidar_y": 0.0,
            "lidar_z": -0.046825,
            "lidar_roll": 0.0,
            "lidar_pitch": 0.0,
            "lidar_yaw": 0.0,
            "min_ray_elevation_deg": LaunchConfiguration("cloud_min_ray_elevation_deg"),
            "min_world_z": LaunchConfiguration("cloud_min_world_z"),
        }],
    )

    # -- 4. Optional fallback nav-graph publisher ---------------------------
    mock_nav_graph_publisher = Node(
        package="robot_sim",
        executable="mock_nav_graph_publisher.py",
        name="mock_nav_graph_publisher",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "graph_frame_id": "odom",
            "far_publish_period_sec": 1.6,
            "near_publish_period_sec": 0.45,
            "near_obstacle_dist_m": 1.1,
            "far_motion_threshold_m": 0.20,
            "obstacle_cost_scale": 20.0,
        }],
        condition=IfCondition(LaunchConfiguration("use_mock_nav_graph")),
    )

    # -- 5. WildOS graph planner --------------------------------------------
    wildos_graphnav_planner = Node(
        package="wildos_graphnav_planner",
        executable="planner_node",
        name="wildos_graphnav_planner",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "path_smoothness_period": 20.0,
            "local_frontier_radius": 9.0,
        }],
        condition=IfCondition(LaunchConfiguration("use_wildos_planner")),
        remappings=[
            ("~/nav_graph", "/nav_graph"),
            ("~/goal_pose", "/goal_pose"),
            ("~/odom", "/odom/raw"),
            ("~/path", "/wildos/path"),
        ],
    )

    # -- 6. Controllers ------------------------------------------------------
    path_follower_node = Node(
        package="path_follower",
        executable="path_follower_node",
        name="path_follower_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_path_follower")),
        parameters=[{
            "use_sim_time": use_sim_time,
            "wp_lookahead_dist": LaunchConfiguration("wp_lookahead_dist"),
            "path_timeout": LaunchConfiguration("path_timeout"),
        }],
        remappings=[
            ("~/path", LaunchConfiguration("path_follower_path_topic")),
            ("~/odom", "/odom/raw"),
            ("~/goal_pose", "/path_follower/next_setpoint"),
        ],
    )

    # mpc_path_optimizer_node = Node(
    #     package="mpc",
    #     executable="mpc_path_optimizer_node",
    #     name="mpc_path_optimizer_node",
    #     output="screen",
    #     condition=IfCondition(LaunchConfiguration("use_mpc_path_optimizer")),
    #     parameters=[{
    #         "input_path_topic": "~/path_in",
    #         "output_path_topic": "~/path_out",
    #         "obstacle_cloud_topic": "~/obstacles",
    #         "horizon_points": LaunchConfiguration("mpc_path_horizon_points"),
    #         "obs_clearance": LaunchConfiguration("mpc_path_safety_clearance"),
    #         "w_curvature": LaunchConfiguration("mpc_path_curvature_weight"),
    #         "w_obs": LaunchConfiguration("mpc_path_obstacle_weight"),
    #     }],
    #     remappings=[
    #         ("~/path_in", "/wildos/path"),
    #         ("~/path_out", "/wildos/path_safe"),
    #         ("~/obstacles", "/lidar/points_filtered"),
    #     ],
    # )

    # mpc_controller_node = Node(
    #     package="mpc",
    #     executable="mpc_node",
    #     name="mpc_node",
    #     output="screen",
    #     condition=IfCondition(LaunchConfiguration("use_mpc_controller")),
    #     parameters=[{
    #         "use_sim_time": use_sim_time,
    #         "cmd_max_vx": LaunchConfiguration("cmd_max_vx"),
    #         "cmd_max_vy": LaunchConfiguration("cmd_max_vy"),
    #     }],
    #     remappings=[
    #         ("/wildos/path", LaunchConfiguration("path_follower_path_topic")),
    #         ("/lidar/points_filtered", "/lidar/points_filtered"),
    #         ("/go2/pose", "/go2/pose"),
    #     ],
    # )

    setpoint_to_cmd_vel = Node(
        package="mpc",
        executable="setpoint_to_cmd_vel_node",
        name="setpoint_to_cmd_vel_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_path_follower")),
        parameters=[{
            "use_sim_time": use_sim_time,
            "cmd_max_vx": LaunchConfiguration("cmd_max_vx"),
            "cmd_max_vy": LaunchConfiguration("cmd_max_vy"),
        }],
        remappings=[
            ("/mpc/next_setpoint", "/path_follower/next_setpoint"),
        ],
    )

    planner_bringup = TimerAction(
        period=planner_delay,
        actions=[
            odom_to_pose,
            cloud_self_filter,
            mock_nav_graph_publisher,
            wildos_graphnav_planner,
            # mpc_path_optimizer_node,
        ],
    )

    motion_bringup = TimerAction(
        period=motion_delay,
        actions=[
            # mpc_controller_node,
            path_follower_node,
            setpoint_to_cmd_vel,
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
        arguments=["-d", LaunchConfiguration("rviz_config")],
        condition=IfCondition(LaunchConfiguration("use_rviz")),
    )

    rviz_bringup = TimerAction(
        period=rviz_delay,
        actions=[rviz],
    )

    return LaunchDescription(args + [
        champ_sim,
        planner_bringup,
        motion_bringup,
        rviz_bringup,
    ])
