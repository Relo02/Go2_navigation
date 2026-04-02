"""
sim_bringup.launch.py
Full Gazebo Classic (Gazebo 11) simulation bringup for the Go2 autonomy stack.

What this launch file starts
-----------------------------
  1. Gazebo Classic (gzserver + gzclient) with the selected world
  2. robot_state_publisher — broadcasts static TF from URDF
                             (base_footprint→base_link→lidar_link, imu_link)
  3. spawn_entity           — places the robot in Gazebo at (0, 0, spawn_z)
  4. slam.launch.py         — pointcloud_to_laserscan + slam_toolbox (use_sim_time=true)
  5. nav2.launch.py         — full Nav2 stack (use_sim_time=true)
  6. RViz2                  — optional, reuses robot_nav/rviz/nav2.rviz

What is NOT started (differs from go2_bringup.launch.py)
---------------------------------------------------------
  - go2_hw_bridge: replaced by Gazebo's libgazebo_ros_planar_move plugin
    (the plugin subscribes /cmd_vel and publishes /odom + TF odom→base_footprint,
     providing the identical Nav2 interface without touching any hardware)

TF chain in simulation (identical to real robot)
------------------------------------------------
  map ─(slam_toolbox)─► odom ─(planar_move)─► base_footprint
                                                    └─(URDF)─► base_link
                                                                    ├──► lidar_link
                                                                    └──► imu_link

Launch arguments
----------------
  world       : path to .world file   (default: navigation_empty.world)
  robot       : 'go2' | 'd1'          (default: go2)
  slam_mode   : 'mapping' | 'localization'  (default: mapping)
  map         : path to .posegraph file (for localization mode)
  use_rviz    : 'true' | 'false'      (default: true)
  start_nav   : 'true' | 'false'      (default: true)
  slam_params : path to slam_toolbox YAML overrides
  nav_params  : path to Nav2 YAML overrides
  spawn_z     : robot spawn height in m (default: 0.05 — just above ground)
  gui         : 'true' | 'false' — show Gazebo GUI  (default: true)

Portability
-----------
  To add AgiBot D1 Ultra: implement d1_sim/urdf/d1_sim.urdf.xacro, then launch with robot:=d1.
  No other changes required.
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _urdf_path(robot: str) -> str:
    """Resolve the simulation URDF path for the given robot name."""
    pkg_map = {
        'go2': ('go2_sim', 'urdf/go2_sim.urdf.xacro'),
        'd1':  ('d1_sim',  'urdf/d1_sim.urdf.xacro'),
    }
    if robot not in pkg_map:
        raise ValueError(
            f"Unknown robot '{robot}'. Valid choices: {list(pkg_map.keys())}"
        )
    pkg, rel = pkg_map[robot]
    return os.path.join(get_package_share_directory(pkg), rel)


def generate_launch_description():

    # ── Package share directories ──────────────────────────────────────
    robot_nav_share  = get_package_share_directory('robot_nav')
    robot_sim_share  = get_package_share_directory('robot_sim')
    gazebo_ros_share = get_package_share_directory('gazebo_ros')
    sim_worlds_share = get_package_share_directory('sim_worlds')

    default_world = os.path.join(
        sim_worlds_share, 'worlds', 'navigation_empty.world'
    )
    default_slam_params = os.path.join(
        robot_nav_share, 'config', 'slam_toolbox_params.yaml'
    )
    default_nav_params = os.path.join(
        robot_sim_share, 'config', 'nav2_sim_params.yaml'
    )
    default_rviz_config = os.path.join(
        robot_nav_share, 'rviz', 'nav2.rviz'
    )

    # ── Declare launch arguments ───────────────────────────────────────
    args = [
        DeclareLaunchArgument(
            'world', default_value=default_world,
            description='Path to Gazebo world file (.world)',
        ),
        DeclareLaunchArgument(
            'robot', default_value='go2',
            description="Robot model: 'go2' or 'd1'",
        ),
        DeclareLaunchArgument(
            'slam_mode', default_value='mapping',
            description="SLAM mode: 'mapping' or 'localization'",
        ),
        DeclareLaunchArgument(
            'map', default_value='',
            description='Path to .posegraph file (for localization mode)',
        ),
        DeclareLaunchArgument(
            'use_rviz', default_value='true',
            description='Launch RViz2',
        ),
        DeclareLaunchArgument(
            'start_nav', default_value='true',
            description='Start SLAM + Nav2 after spawning the robot',
        ),
        DeclareLaunchArgument(
            'slam_params', default_value=default_slam_params,
            description='Path to slam_toolbox params YAML',
        ),
        DeclareLaunchArgument(
            'nav_params', default_value=default_nav_params,
            description='Path to Nav2 params YAML',
        ),
        DeclareLaunchArgument(
            'spawn_z', default_value='0.05',
            description='Robot Z spawn height (m); 0.05 gives slight ground clearance',
        ),
        DeclareLaunchArgument(
            'gui', default_value='true',
            description='Show Gazebo GUI (gzclient)',
        ),
    ]

    # ── LaunchConfiguration references ────────────────────────────────
    world       = LaunchConfiguration('world')
    robot       = LaunchConfiguration('robot')
    slam_mode   = LaunchConfiguration('slam_mode')
    map_path    = LaunchConfiguration('map')
    use_rviz    = LaunchConfiguration('use_rviz')
    start_nav   = LaunchConfiguration('start_nav')
    slam_params = LaunchConfiguration('slam_params')
    nav_params  = LaunchConfiguration('nav_params')
    spawn_z     = LaunchConfiguration('spawn_z')
    gui         = LaunchConfiguration('gui')

    # ── Environment: make sim_worlds models findable by Gazebo ────────
    gazebo_model_path = SetEnvironmentVariable(
        'GAZEBO_MODEL_PATH',
        [
            os.path.join(sim_worlds_share, 'models'),
            ':',
            os.path.join(get_package_share_directory('go2_sim'), '..'),
            ':',
            # Preserve any pre-existing GAZEBO_MODEL_PATH
            os.environ.get('GAZEBO_MODEL_PATH', ''),
        ],
    )

    # ── Compute URDF path and robot_description parameter ─────────────
    # We use PythonExpression to select the URDF at runtime based on robot arg.
    # xacro processes the file and returns the URDF XML string.
    go2_urdf_path = os.path.join(
        get_package_share_directory('go2_sim'), 'urdf', 'go2_sim.urdf.xacro'
    )
    d1_urdf_path = os.path.join(
        get_package_share_directory('d1_sim'), 'urdf', 'd1_sim.urdf.xacro'
    )

    # robot_description for go2 (default)
    robot_description_go2 = ParameterValue(
        Command(['xacro ', go2_urdf_path]),
        value_type=str,
    )
    robot_description_d1 = ParameterValue(
        Command(['xacro ', d1_urdf_path]),
        value_type=str,
    )

    is_go2 = PythonExpression(["'", robot, "' == 'go2'"])
    is_d1  = PythonExpression(["'", robot, "' == 'd1'"])

    # ── 1. Gazebo ──────────────────────────────────────────────────────
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_share, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'world':   world,
            'verbose': 'false',
            'pause':   'false',
            'gui':     gui,
        }.items(),
    )

    # ── 2. robot_state_publisher ───────────────────────────────────────
    # Publishes static TF from the URDF fixed joints.
    # use_sim_time=true so timestamps align with Gazebo clock.
    rsp_go2 = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_go2,
            'use_sim_time': True,
        }],
        condition=IfCondition(is_go2),
    )
    rsp_d1 = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_d1,
            'use_sim_time': True,
        }],
        condition=IfCondition(is_d1),
    )

    # ── 3. spawn_entity ────────────────────────────────────────────────
    # Spawns the robot URDF in Gazebo by reading robot_description from
    # the topic published by robot_state_publisher.
    spawn_go2 = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_robot',
        output='screen',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'go2',
            '-x', '0.0',
            '-y', '0.0',
            '-z', spawn_z,
            '-R', '0.0',
            '-P', '0.0',
            '-Y', '0.0',
        ],
        condition=IfCondition(is_go2),
    )
    spawn_d1 = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_robot',
        output='screen',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'd1',
            '-x', '0.0',
            '-y', '0.0',
            '-z', spawn_z,
        ],
        condition=IfCondition(is_d1),
    )

    cloud_self_filter = Node(
        package='robot_sim',
        executable='cloud_self_filter.py',
        name='cloud_self_filter',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'input_topic': '/lidar/points',
            'output_topic': '/lidar/points_filtered',
            'lidar_x': 0.28945,
            'lidar_y': 0.0,
            'lidar_z': -0.046825,
            'lidar_roll': 0.0,
            'lidar_pitch': 0.0,
            'lidar_yaw': 0.0,
            'min_ray_elevation_deg': 6.0,
            'min_world_z': 0.10,
        }],
        condition=IfCondition(is_go2),
    )

    # ── 4. SLAM (pointcloud_to_laserscan + slam_toolbox) ──────────────
    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(robot_nav_share, 'launch', 'slam.launch.py')
        ),
        condition=IfCondition(start_nav),
        launch_arguments={
            'use_sim_time': 'true',
            'slam_mode':    slam_mode,
            'slam_params':  slam_params,
            'map':          map_path,
        }.items(),
    )

    # ── 5. Nav2 ────────────────────────────────────────────────────────
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(robot_nav_share, 'launch', 'nav2.launch.py')
        ),
        condition=IfCondition(start_nav),
        launch_arguments={
            'use_sim_time': 'true',
            'params_file':  nav_params,
        }.items(),
    )

    # Start SLAM and Nav2 only after the robot has been spawned in Gazebo.
    # This avoids Nav2 activation before odom->base_footprint exists.
    start_nav_after_go2_spawn = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_go2,
            on_exit=[
                slam,
                TimerAction(period=2.0, actions=[nav2]),
            ],
        )
    )
    start_nav_after_d1_spawn = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_d1,
            on_exit=[
                slam,
                TimerAction(period=2.0, actions=[nav2]),
            ],
        )
    )

    # ── 6. RViz2 ──────────────────────────────────────────────────────
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', default_rviz_config],
        parameters=[{'use_sim_time': True}],
        output='screen',
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription([
        # Environment must be set before Gazebo starts
        gazebo_model_path,
        *args,
        gazebo,
        cloud_self_filter,
        rsp_go2,
        rsp_d1,
        spawn_go2,
        spawn_d1,
        start_nav_after_go2_spawn,
        start_nav_after_d1_spawn,
        rviz,
    ])
