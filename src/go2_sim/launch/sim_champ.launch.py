"""
sim_champ.launch.py
Fortress (Ignition Gazebo 6) + CHAMP simulation launch file for the Unitree Go2.

Source: khaledgabr77/unitree_go2_ros2 / unitree_go2_sim/launch/unitree_go2_launch.py
Port changes:
  Fortress packages only (ros_ign_gazebo, ros_ign_bridge, ign_ros2_control)
  Ignition message namespace only (ignition.msgs.*)

Topic bridge (all required by nav stack):
  /clock              ignition → ROS   rosgraph_msgs/Clock
  /imu/data           ignition → ROS   sensor_msgs/Imu
  /joint_states       ignition → ROS   sensor_msgs/JointState
  /lidar/points             ignition(/unilidar/cloud/points) → ROS   sensor_msgs/PointCloud2
  /odom/raw           ignition → ROS   nav_msgs/Odometry
  /cmd_vel_safe       ROS → ignition   geometry_msgs/Twist
  /joint_group_effort_controller/joint_trajectory
                      ROS → ignition   trajectory_msgs/JointTrajectory

Usage:
  ros2 launch go2_sim sim_champ.launch.py
  ros2 launch go2_sim sim_champ.launch.py gui:=false rviz:=false
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
    SetEnvironmentVariable,
    AppendEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")

    go2_sim_dir = get_package_share_directory("go2_sim")

    # ── Resource path setup ────────────────────────────────────────────────
    # Add the package's share directory to the Ignition Gazebo resource path.
    # This allows Gazebo to find meshes via 'model://go2_sim/...' or 'package://go2_sim/...'.
    # We use both IGN_GAZEBO_RESOURCE_PATH (Fortress) and GZ_SIM_RESOURCE_PATH (Harmonic).
    ign_resource_path = AppendEnvironmentVariable(
        name="IGN_GAZEBO_RESOURCE_PATH",
        value=[os.path.join(go2_sim_dir, "..")]
    )
    gz_resource_path = AppendEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=[os.path.join(go2_sim_dir, "..")]
    )

    # ── Config file paths ──────────────────────────────────────────────────
    joints_config    = os.path.join(go2_sim_dir, "config", "joints.yaml")
    links_config     = os.path.join(go2_sim_dir, "config", "links.yaml")
    gait_config      = os.path.join(go2_sim_dir, "config", "gait.yaml")
    ros_control_file = os.path.join(go2_sim_dir, "config", "ros_control.yaml")
    urdf_path        = os.path.join(go2_sim_dir, "urdf", "go2_sim.urdf.xacro")
    world_path       = os.path.join(go2_sim_dir, "worlds", "default.sdf")

    # ── Launch arguments ────────────────────────────────────────────────────
    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time", default_value="true",
        description="Use simulation (Gazebo) clock if true",
    )
    declare_rviz = DeclareLaunchArgument(
        "rviz", default_value="false",
        description="Launch RViz2 (no config bundled; add -d <config> manually)",
    )
    declare_robot_name = DeclareLaunchArgument(
        "robot_name", default_value="go2", description="Robot name in Gazebo"
    )
    declare_ros_control_file = DeclareLaunchArgument(
        "ros_control_file", default_value=ros_control_file,
        description="Path to ros_control YAML",
    )
    declare_world = DeclareLaunchArgument(
        "world", default_value=world_path,
        description="Path to Gazebo world SDF",
    )
    declare_gui = DeclareLaunchArgument(
        "gui", default_value="true", description="Launch Gazebo GUI"
    )
    declare_world_init_x = DeclareLaunchArgument("world_init_x", default_value="0.0")
    declare_world_init_y = DeclareLaunchArgument("world_init_y", default_value="0.0")
    declare_world_init_z = DeclareLaunchArgument(
        "world_init_z", default_value="0.375",
        description="Spawn height (0.375 m = nominal standing height + clearance)",
    )
    declare_world_init_heading = DeclareLaunchArgument(
        "world_init_heading", default_value="0.0"
    )
    declare_publish_map_to_odom_tf = DeclareLaunchArgument(
        "publish_map_to_odom_tf",
        default_value="true",
        description=(
            "Publish static map->odom identity transform. "
            "Set false when SLAM/localization owns map->odom."
        ),
    )
    declare_enable_base_to_footprint_ekf = DeclareLaunchArgument(
        "enable_base_to_footprint_ekf",
        default_value="true",
        description=(
            "Enable CHAMP base_to_footprint EKF. Disable for pure Gazebo odom."
        ),
    )
    declare_footprint_base_frame = DeclareLaunchArgument(
        "footprint_base_frame",
        default_value="base_footprint",
        description=(
            "Base frame used by footprint_to_odom_ekf (base_footprint or base_link)."
        ),
    )
    declare_enable_footprint_to_odom_ekf = DeclareLaunchArgument(
        "enable_footprint_to_odom_ekf",
        default_value="true",
        description=(
            "Enable CHAMP footprint_to_odom EKF. Disable to use raw Gazebo odometry only."
        ),
    )
    declare_gazebo_odom_topic = DeclareLaunchArgument(
        "gazebo_odom_topic",
        default_value="/odom/raw",
        description="ROS topic receiving Gazebo odometry bridge output.",
    )

    # ── Robot description (processed at launch time by xacro) ──────────────
    robot_description_content = ParameterValue(
        Command([
            "xacro ", urdf_path,
            " robot_controllers:=", LaunchConfiguration("ros_control_file"),
        ]),
        value_type=str,
    )
    robot_description = {"robot_description": robot_description_content}

    # ── Nodes ──────────────────────────────────────────────────────────────

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description, {"use_sim_time": use_sim_time}, {"ignore_timestamp": True}],
    )

    # CHAMP quadruped locomotion controller
    quadruped_controller_node = Node(
        package="champ_base",
        executable="quadruped_controller_node",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            {"gazebo": True},
            {"publish_joint_states": True},
            {"publish_joint_control": True},
            {"publish_foot_contacts": True},
            {"joint_controller_topic": "joint_group_effort_controller/joint_trajectory"},
            {"urdf": robot_description_content},
            {"hardware_connected": False},
            {"close_loop_odom": True},
            joints_config,
            links_config,
            gait_config,
        ],
        remappings=[("/cmd_vel/smooth", "/cmd_vel_safe")],
    )

    # CHAMP state estimator (orientation from IMU)
    # NOTE: remapped odom/raw -> odom/champ_raw to avoid conflicting with the
    # Gazebo OdometryPublisher plugin which also writes to /odom/raw.
    # The Gazebo ground-truth /odom/raw is used by odom_to_pose_node.
    state_estimator_node = Node(
        package="champ_base",
        executable="state_estimation_node",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            {"orientation_from_imu": True},
            {"urdf": robot_description_content},
            joints_config,
            links_config,
            gait_config,
        ],
        remappings=[
            ("odom/raw", "odom/champ_raw"),
        ],
    )

    # EKF: base_link → base_footprint (local odometry frame)
    base_to_footprint_ekf = Node(
        package="robot_localization",
        executable="ekf_node",
        name="base_to_footprint_ekf",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_base_to_footprint_ekf")),
        parameters=[
            {"base_link_frame": "base_link"},
            {"use_sim_time": use_sim_time},
            os.path.join(
                get_package_share_directory("champ_base"),
                "config", "ekf", "base_to_footprint.yaml",
            ),
        ],
        remappings=[("odometry/filtered", "odom/local")],
    )

    # EKF: base_footprint → odom (global odometry, publishes /odom + TF)
    footprint_to_odom_ekf = Node(
        package="robot_localization",
        executable="ekf_node",
        name="footprint_to_odom_ekf",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_footprint_to_odom_ekf")),
        parameters=[
            {"use_sim_time": use_sim_time},
            {"base_link_frame": LaunchConfiguration("footprint_base_frame")},
            {"odom_frame": "odom"},
            {"world_frame": "odom"},
            {"publish_tf": True},
            {"frequency": 20.0},
            {"two_d_mode": True},
            {"odom0": LaunchConfiguration("gazebo_odom_topic")},
            {"odom0_config": [True, True, False,
                              False, False, True,
                              True, True, False,
                              False, False, True,
                              False, False, False]},
            {"imu0": "imu/data"},
            {"imu0_config": [False, False, False,
                             False, False, True,
                             False, False, False,
                             False, False, True,
                             False, False, False]},
        ],
        remappings=[("odometry/filtered", "odom")],
    )

    # Static transform: map → odom (identity; nav stack expects this until SLAM publishes)
    map_to_odom_tf = Node(
        package="tf2_ros",
        name="map_to_odom_tf",
        executable="static_transform_publisher",
        condition=IfCondition(LaunchConfiguration("publish_map_to_odom_tf")),
        parameters=[{"use_sim_time": use_sim_time}],
        arguments=[
            "--x", "0", "--y", "0", "--z", "0",
            "--roll", "0", "--pitch", "0", "--yaw", "0",
            "--frame-id", "map", "--child-frame-id", "odom",
        ],
    )

    # RViz2 (optional; no bundled config — pass -d <config> via extra_args if needed)
    rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        condition=IfCondition(LaunchConfiguration("rviz")),
        parameters=[{"use_sim_time": use_sim_time}],
    )

    # ── Gazebo Fortress (ign gazebo) ────────────────────────────────────────
    pkg_ros_ign_gazebo = get_package_share_directory("ros_ign_gazebo")

    ign_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_ign_gazebo, "launch", "ign_gazebo.launch.py")
        ),
        launch_arguments={
            "gz_args": [
                LaunchConfiguration("world"),
                " -r",  # start unpaused
            ],
        }.items(),
    )

    # Spawn robot in Ignition Gazebo
    gazebo_spawn_robot = Node(
        package="ros_ign_gazebo",
        executable="create",
        output="screen",
        arguments=[
            "-name", LaunchConfiguration("robot_name"),
            "-topic", "robot_description",
            "-x", LaunchConfiguration("world_init_x"),
            "-y", LaunchConfiguration("world_init_y"),
            "-z", LaunchConfiguration("world_init_z"),
            "-Y", LaunchConfiguration("world_init_heading"),
        ],
    )

    # ── ros_ign_bridge: topic bridges between Ignition and ROS 2 ────────────
    #
    # Bridge direction syntax (Fortress):
    #   topic@ros_type[ignition_type   — ignition → ROS  (unidirectional)
    #   topic@ros_type]ignition_type   — ROS → ignition  (unidirectional)
    #   topic@ros_type@ignition_type   — bidirectional
    #
    # Message namespace: ignition.msgs.*  (NOT gz.msgs.* which is Harmonic)
    #
    # Required topics for nav stack:
    #   /joint_states   → robot_state_publisher, CHAMP
    #   /imu/data       → CHAMP state estimator, EKF
    #   /lidar/points         ← ignition /unilidar/cloud/points
    #                         → voxel_layer, pointcloud_to_laserscan
    #   /odom/raw       → EKF footprint_to_odom
    #   /clock          → use_sim_time
    #   TF is published by robot_state_publisher + EKF (not bridged from Ignition)
    #   /cmd_vel_safe   → CHAMP locomotion controller (through safety gate)
    #   /joint_group_effort_controller/joint_trajectory → effort controller
    gazebo_bridge = Node(
        package="ros_ign_bridge",
        executable="parameter_bridge",
        name="gazebo_bridge",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
        arguments=[
            # ── Ignition → ROS ──────────────────────────────────────────
            "/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock",
            "/imu/data@sensor_msgs/msg/Imu[ignition.msgs.IMU",
            "/world/default/model/go2/joint_state@sensor_msgs/msg/JointState[ignition.msgs.Model",
            "/unilidar/cloud/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked",
            "/odom@nav_msgs/msg/Odometry[ignition.msgs.Odometry",
            # ── ROS → Ignition ──────────────────────────────────────────
            "/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist",
            "/joint_group_effort_controller/joint_trajectory"
            "@trajectory_msgs/msg/JointTrajectory]ignition.msgs.JointTrajectory",
        ],
        remappings=[
            ("/world/default/model/go2/joint_state", "/joint_states"),
            ("/unilidar/cloud/points", "/lidar/points"),
            ("/odom", LaunchConfiguration("gazebo_odom_topic")),
            ("/cmd_vel", "/cmd_vel_safe"),
        ],
    )

    # Safety gate in simulation: Nav2 /cmd_vel -> /cmd_vel_safe
    velocity_limiter = Node(
        package="robot_safety",
        executable="velocity_limiter",
        name="velocity_limiter",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "cmd_timeout": 1.5,
            "publish_rate": 20.0,
            "cmd_vel_in": "/cmd_vel",
            "cmd_vel_out": "/cmd_vel_safe",
            "status_topic": "/robot_status",
        }],
    )

    # ── Controller spawners (delayed to allow Gazebo + ign_ros2_control init) ──
    # controller_manager is brought up by ign_ros2_control-system plugin in the URDF.
    controller_spawner_js = TimerAction(
        period=20.0,
        actions=[
            Node(
                package="controller_manager",
                executable="spawner",
                output="screen",
                arguments=[
                    "--controller-manager-timeout", "120",
                    "joint_states_controller",
                ],
                parameters=[{"use_sim_time": use_sim_time}],
            )
        ],
    )

    controller_spawner_effort = TimerAction(
        period=30.0,
        actions=[
            Node(
                package="controller_manager",
                executable="spawner",
                output="screen",
                arguments=[
                    "--controller-manager-timeout", "120",
                    "joint_group_effort_controller",
                ],
                parameters=[{"use_sim_time": use_sim_time}],
            )
        ],
    )

    # ── Launch description ─────────────────────────────────────────────────
    return LaunchDescription([
        # Resource path environment variables
        ign_resource_path,
        gz_resource_path,

        # Arguments
        declare_use_sim_time,
        declare_rviz,
        declare_robot_name,
        declare_ros_control_file,
        declare_world,
        declare_gui,
        declare_world_init_x,
        declare_world_init_y,
        declare_world_init_z,
        declare_world_init_heading,
        declare_publish_map_to_odom_tf,
        declare_enable_base_to_footprint_ekf,
        declare_footprint_base_frame,
        declare_enable_footprint_to_odom_ekf,
        declare_gazebo_odom_topic,

        # Gazebo + robot spawn
        ign_gazebo,
        robot_state_publisher,
        gazebo_spawn_robot,
        gazebo_bridge,
        velocity_limiter,

        # CHAMP locomotion stack
        quadruped_controller_node,
        state_estimator_node,

        # Odometry / TF
        base_to_footprint_ekf,
        footprint_to_odom_ekf,
        map_to_odom_tf,

        # Controller lifecycle
        controller_spawner_js,
        controller_spawner_effort,

        # Visualisation
        rviz2,
    ])
