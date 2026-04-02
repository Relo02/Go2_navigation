"""
sim_a_star_mpc.launch.py
Gazebo Fortress + CHAMP + A* MPC planner bringup for Go2.
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
  3. odom_to_pose_node          /odom -> /go2/pose
  4. a_star_node                occupancy grid + A* path planning
  5. mpc_node                   MPC trajectory tracking
  6. setpoint_to_cmd_vel_node   /mpc/next_setpoint -> /cmd_vel
  7. RViz2 (optional)

Key topic wiring
----------------
  /lidar/points            (Gazebo bridge) -> cloud_self_filter
  /lidar/points_filtered                      -> a_star_node, mpc_node
  /go2/pose                                    -> a_star_node, mpc_node, setpoint_to_cmd_vel_node
  /a_star/path             (a_star_node)       -> mpc_node
  /mpc/next_setpoint       (mpc_node)          -> setpoint_to_cmd_vel_node
  /cmd_vel                 (setpoint_to_cmd_vel_node) -> CHAMP (via safety gate)

Goal source
-----------
  Goal is RViz-driven (2D Goal Pose tool) through topic /goal_pose.
  The launch remaps /global_goal (A* subscription) to /goal_pose.
  No hard-coded launch goal is required.

Usage
-----
  # Default run (Gazebo GUI on, RViz off):
  ros2 launch robot_sim sim_a_star_mpc.launch.py

  # Headless, RViz on:
  ros2 launch robot_sim sim_a_star_mpc.launch.py gui:=false use_rviz:=true

  # Override A* + MPC parameters file:
  ros2 launch robot_sim sim_a_star_mpc.launch.py \
      planner_params:=/path/to/my_params.yaml

  # Manual startup goal (no RViz goal required):
  ros2 launch robot_sim sim_a_star_mpc.launch.py \
      wait_for_goal:=false goal_x:=2.0 goal_y:=-0.75 goal_z:=0.0
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
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():

    go2_sim_share = get_package_share_directory("go2_sim")
    robot_sim_share = get_package_share_directory("robot_sim")
    a_star_mpc_share = get_package_share_directory("a_star_mpc_planner")

    default_world = os.path.join(go2_sim_share, "worlds", "default.sdf")
    default_ros_control = os.path.join(go2_sim_share, "config", "ros_control.yaml")
    default_planner_params = os.path.join(a_star_mpc_share, "config", "planner_params.yaml")

    # ── Launch arguments ────────────────────────────────────────────────────

    args = [
        # Simulation
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("world", default_value=default_world),
        DeclareLaunchArgument("gui", default_value="true",
                              description="Launch Gazebo GUI"),
        DeclareLaunchArgument("robot_name", default_value="go2"),
        DeclareLaunchArgument("ros_control_file", default_value=default_ros_control),
        DeclareLaunchArgument("world_init_x", default_value="0.0"),
        DeclareLaunchArgument("world_init_y", default_value="0.0"),
        DeclareLaunchArgument("world_init_z", default_value="0.375"),
        DeclareLaunchArgument("world_init_heading", default_value="0.0"),

        # Planner parameters file
        DeclareLaunchArgument("planner_params", default_value=default_planner_params,
                              description="Path to A* + MPC YAML parameters file"),
        DeclareLaunchArgument(
            "wait_for_goal",
            default_value="true",
            description=(
                "If true, wait for runtime goal on /goal_pose (RViz). "
                "If false, use goal_x/goal_y/goal_z launch args."
            ),
        ),
        DeclareLaunchArgument("goal_x", default_value="5.0"),
        DeclareLaunchArgument("goal_y", default_value="5.0"),
        DeclareLaunchArgument("goal_z", default_value="0.0"),

        # Cloud filter delay — give Gazebo time to publish the first scan
        DeclareLaunchArgument("planner_delay_sec", default_value="15.0",
                              description="Seconds to wait before starting planner nodes"),

        # Visualisation
        DeclareLaunchArgument("use_rviz", default_value="false"),
        DeclareLaunchArgument("rviz_config",
                              default_value=os.path.join(
                                  robot_sim_share, "rviz", "a_star_mpc.rviz"),
                              description="RViz2 config file"),
    ]

    use_sim_time = LaunchConfiguration("use_sim_time")
    planner_params = LaunchConfiguration("planner_params")
    planner_delay = LaunchConfiguration("planner_delay_sec")
    wait_for_goal = ParameterValue(LaunchConfiguration("wait_for_goal"), value_type=bool)
    goal_x = ParameterValue(LaunchConfiguration("goal_x"), value_type=float)
    goal_y = ParameterValue(LaunchConfiguration("goal_y"), value_type=float)
    goal_z = ParameterValue(LaunchConfiguration("goal_z"), value_type=float)

    # ── 1. Base simulation (Gazebo + CHAMP + EKF) ───────────────────────────
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
            "publish_map_to_odom_tf": "true",
        }.items(),
    )

    # ── 2. Odometry -> PoseStamped bridge ───────────────────────────────────
    odom_to_pose = Node(
        package="a_star_mpc_planner",
        executable="odom_to_pose_node",
        name="odom_to_pose_node",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    # ── 3. Point-cloud self-filter ──────────────────────────────────────────
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
            "min_ray_elevation_deg": 10.0,
            "min_world_z": 0.20,
        }],
    )

    # ── 4. A* path planner ──────────────────────────────────────────────────
    a_star_node = Node(
        package="a_star_mpc_planner",
        executable="a_star_node",
        name="a_star_node",
        output="screen",
        parameters=[
            planner_params,
            {
                "use_sim_time": use_sim_time,
                "wait_for_goal": wait_for_goal,
                "goal_x": goal_x,
                "goal_y": goal_y,
                "goal_z": goal_z,
            },
        ],
        remappings=[
            # RViz "2D Goal Pose" tool publishes PoseStamped on /goal_pose.
            ("/global_goal", "/goal_pose"),
        ],
    )

    # ── 5. MPC tracker ──────────────────────────────────────────────────────
    mpc_node = Node(
        package="a_star_mpc_planner",
        executable="mpc_node",
        name="mpc_node",
        output="screen",
        parameters=[
            planner_params,
            {"use_sim_time": use_sim_time},
        ],
        remappings=[
            ("/go2/pose", "/go2/pose"),
            ("/lidar/points_filtered", "/lidar/points_filtered"),
        ],
    )

    # ── 6. Setpoint -> cmd_vel bridge ───────────────────────────────────────
    setpoint_to_cmd_vel = Node(
        package="a_star_mpc_planner",
        executable="setpoint_to_cmd_vel_node",
        name="setpoint_to_cmd_vel_node",
        output="screen",
        parameters=[
            planner_params,
            {"use_sim_time": use_sim_time},
        ],
    )

    # Delay planner startup so Gazebo is fully initialised and publishing.
    planner_bringup = TimerAction(
        period=planner_delay,
        actions=[
            odom_to_pose,
            cloud_self_filter,
            a_star_node,
            mpc_node,
            setpoint_to_cmd_vel,
        ],
    )

    # ── 7. RViz2 (optional) ─────────────────────────────────────────────────
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
        arguments=["-d", LaunchConfiguration("rviz_config")],
        condition=IfCondition(LaunchConfiguration("use_rviz")),
    )

    # ── Launch description ──────────────────────────────────────────────────
    return LaunchDescription(args + [
        champ_sim,
        planner_bringup,
        rviz,
    ])
